#!/usr/bin/env python3
"""SpikeForest validation benchmark for Zerostone spike sorter.

Uses SpikeInterface's ``compare_sorter_to_ground_truth()`` for standardized
accuracy computation (Hungarian matching, TP/FN/FP per unit). Falls back to
SpikeInterface's ``generate_ground_truth_recording()`` when SpikeForest
paired Kampff data is not available.

Results are saved to ``benchmarks/results/`` as JSON.

Usage:
    python benchmarks/spikeforest_benchmark.py
    python benchmarks/spikeforest_benchmark.py --max-recordings 2
    python benchmarks/spikeforest_benchmark.py --fallback-synthetic
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import zpybci as zbci

try:
    import spikeinterface.core as si
    from spikeinterface.comparison import compare_sorter_to_ground_truth
    from spikeinterface.core import NumpySorting

    HAS_SI = True
except ImportError:
    HAS_SI = False

PAIRED_KAMPFF_URI = (
    "sha1://b8b571d001f9a531040e79165e8f492d758ec5e0"
    "?paired-kampff-spikeforest-recordings.json"
)

SUPPORTED_CHANNELS = [4, 8, 16, 32, 64, 128]
TOLERANCE_MS = 0.4  # SpikeInterface default: 0.4 ms
TOLERANCE_SAMPLES = 12  # 0.4 ms at 30 kHz

# Approximate published MountainSort5 numbers on paired Kampff
MS5_REFERENCE = {"average_accuracy": "~70-80%"}

RESULTS_DIR = Path(__file__).parent / "results"


def nearest_supported_channels(n):
    """Return the largest supported channel count <= n."""
    valid = [c for c in SUPPORTED_CHANNELS if c <= n]
    return max(valid) if valid else SUPPORTED_CHANNELS[0]


def _sort_and_build_numpy_sorting(traces, n_channels, fs, params):
    """Run Zerostone on raw traces and return NumpySorting + raw result.

    Parameters
    ----------
    traces : np.ndarray
        2D float64 array ``(n_samples, n_channels)``.
    n_channels : int
        Number of channels in the data.
    fs : float
        Sampling frequency in Hz.
    params : dict
        Sorter parameters passed to ``sort_multichannel``.

    Returns
    -------
    NumpySorting or None
        SpikeInterface sorting object (None if SI unavailable).
    dict
        Raw result from ``sort_multichannel``.
    float
        Elapsed sorting time in seconds.
    """
    probe = zbci.ProbeLayout.linear(n_channels, params.get("probe_pitch", 25.0))

    t0 = time.perf_counter()
    result = zbci.sort_multichannel(
        traces,
        probe,
        threshold=params.get("threshold", 5.0),
        refractory=params.get("refractory", 15),
        spatial_radius=params.get("spatial_radius", 75.0),
        temporal_radius=params.get("temporal_radius", 5),
        align_half_window=params.get("align_half_window", 5),
        pre_samples=params.get("pre_samples", 16),
        cluster_threshold=params.get("cluster_threshold", 5.0),
        cluster_max_count=params.get("cluster_max_count", 1000),
        whitening_epsilon=params.get("whitening_epsilon", 1e-6),
        detection_mode=params.get("detection_mode", "amplitude"),
        sneo_smooth_window=params.get("sneo_smooth_window", 3),
        ccg_merge=params.get("ccg_merge", False),
        matched_filter_detect=params.get("matched_filter_detect", True),
        matched_filter_threshold=params.get("matched_filter_threshold", 4.0),
        gmm_refine=params.get("gmm_refine", False),
        gmm_max_iter=params.get("gmm_max_iter", 10),
        bandpass_low=params.get("bandpass_low", 0.0),
        bandpass_high=params.get("bandpass_high", 0.0),
        sample_rate=params.get("sample_rate", fs),
        common_median_ref=params.get("common_median_ref", False),
    )
    elapsed = time.perf_counter() - t0

    n_spikes = result["n_spikes"]
    if n_spikes == 0 or not HAS_SI:
        return None, result, elapsed

    spike_times = np.asarray(result["spike_times"][:n_spikes], dtype=np.int64)
    labels = np.asarray(result["labels"][:n_spikes], dtype=np.int64)

    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[spike_times],
        labels_list=[labels],
        sampling_frequency=fs,
    )
    return sorting, result, elapsed


def _build_gt_numpy_sorting(gt_trains, fs):
    """Build a NumpySorting from ground-truth spike trains.

    Parameters
    ----------
    gt_trains : dict
        Mapping ``unit_id -> sorted array of spike sample indices``.
    fs : float
        Sampling frequency.

    Returns
    -------
    NumpySorting
        Ground-truth sorting object.
    """
    all_times = []
    all_labels = []
    for uid in sorted(gt_trains.keys()):
        t = gt_trains[uid]
        all_times.append(t)
        all_labels.append(np.full(len(t), uid, dtype=np.int64))

    if len(all_times) == 0:
        return NumpySorting.from_unit_dict({}, sampling_frequency=fs)

    times = np.concatenate(all_times)
    labels = np.concatenate(all_labels)
    idx = np.argsort(times)
    return NumpySorting.from_samples_and_labels(
        samples_list=[times[idx]],
        labels_list=[labels[idx]],
        sampling_frequency=fs,
    )


def _compare_si(gt_sorting, tested_sorting, delta_time=TOLERANCE_MS):
    """Run SpikeInterface comparison and return metrics dict.

    Parameters
    ----------
    gt_sorting : BaseSorting
        Ground-truth sorting.
    tested_sorting : BaseSorting
        Tested (Zerostone) sorting.
    delta_time : float
        Matching tolerance in ms.

    Returns
    -------
    dict
        Per-unit and overall metrics.
    """
    comp = compare_sorter_to_ground_truth(
        gt_sorting,
        tested_sorting,
        exhaustive_gt=True,
        delta_time=delta_time,
        match_mode="hungarian",
    )

    perf = comp.get_performance()
    avg = comp.get_performance(method="pooled_with_average")

    per_unit = []
    for uid in perf.index:
        row = perf.loc[uid]
        per_unit.append({
            "gt_unit": str(uid),
            "accuracy": float(row["accuracy"]),
            "recall": float(row["recall"]),
            "precision": float(row["precision"]),
            "miss_rate": float(row["miss_rate"]),
            "false_discovery_rate": float(row["false_discovery_rate"]),
        })

    overall = {
        "accuracy": float(avg["accuracy"]),
        "recall": float(avg["recall"]),
        "precision": float(avg["precision"]),
        "miss_rate": float(avg["miss_rate"]),
        "false_discovery_rate": float(avg["false_discovery_rate"]),
    }

    return {
        "per_unit": per_unit,
        "overall": overall,
        "well_detected": int(comp.count_well_detected_units(well_detected_score=0.8)),
        "n_gt_units": len(gt_sorting.get_unit_ids()),
        "n_tested_units": len(tested_sorting.get_unit_ids()),
    }


def process_recording(name, traces, fs, gt_trains, n_channels_orig, params):
    """Sort a single recording and return metrics."""
    n_ch = nearest_supported_channels(n_channels_orig)
    if n_ch != n_channels_orig:
        idx = np.linspace(0, n_channels_orig - 1, n_ch, dtype=int)
        traces = traces[:, idx]
        print(f"    Subsampled {n_channels_orig} -> {n_ch} channels")

    sorting, result, elapsed = _sort_and_build_numpy_sorting(traces, n_ch, fs, params)

    n_gt_total = sum(len(v) for v in gt_trains.values())

    if sorting is None or result["n_spikes"] == 0:
        print(f"    {name}: {n_ch}ch, {n_gt_total} GT spikes, "
              f"0 detected, 0 clusters, {elapsed:.1f}s")
        return {
            "name": name, "n_channels": n_ch, "elapsed": elapsed,
            "metrics": None, "n_gt": n_gt_total, "n_sorted": 0,
        }

    # Build GT NumpySorting and compare
    gt_sorting = _build_gt_numpy_sorting(gt_trains, fs)
    metrics = _compare_si(gt_sorting, sorting)

    overall = metrics["overall"]
    print(f"    {name}: {n_ch}ch, {n_gt_total} GT spikes, "
          f"{result['n_spikes']} detected, {result['n_clusters']} clusters, "
          f"acc={overall['accuracy']:.3f}, prec={overall['precision']:.3f}, "
          f"rec={overall['recall']:.3f}, well_detected={metrics['well_detected']}/{metrics['n_gt_units']}, "
          f"{elapsed:.1f}s")

    for pu in metrics["per_unit"]:
        print(f"      Unit {pu['gt_unit']:>3}  "
              f"acc={pu['accuracy']:.3f}  prec={pu['precision']:.3f}  "
              f"rec={pu['recall']:.3f}")

    return {
        "name": name, "n_channels": n_ch, "elapsed": elapsed,
        "metrics": metrics, "n_gt": n_gt_total, "n_sorted": result["n_spikes"],
    }


def run_si_synthetic(params):
    """Run benchmark using SpikeInterface's generate_ground_truth_recording.

    This uses SpikeInterface's built-in realistic recording generator which
    creates probe geometries, unit locations, and proper template injection.
    """
    print("Running SpikeInterface synthetic benchmarks (3 recordings)\n")

    configs = [
        {
            "label": "si-easy",
            "num_channels": 4, "num_units": 3, "duration": 30.0,
            "noise_levels": 3.0,
        },
        {
            "label": "si-medium",
            "num_channels": 16, "num_units": 8, "duration": 30.0,
            "noise_levels": 5.0,
        },
        {
            "label": "si-hard",
            "num_channels": 32, "num_units": 15, "duration": 30.0,
            "noise_levels": 8.0,
        },
    ]

    results = []
    for ci, cfg in enumerate(configs):
        name = cfg["label"]
        print(f"  [{ci+1}/{len(configs)}] {name}")
        try:
            rec, sorting_true = si.generate_ground_truth_recording(
                durations=[cfg["duration"]],
                sampling_frequency=30000.0,
                num_channels=cfg["num_channels"],
                num_units=cfg["num_units"],
                seed=42,
                noise_kwargs={
                    "noise_levels": cfg["noise_levels"],
                    "strategy": "on_the_fly",
                },
            )

            traces = rec.get_traces(return_in_uV=True).astype(np.float64)
            fs = rec.get_sampling_frequency()
            n_ch = rec.get_num_channels()

            gt_trains = {}
            for uid in sorting_true.get_unit_ids():
                train = sorting_true.get_unit_spike_train(uid)
                gt_trains[uid] = np.sort(train)

            res = process_recording(name, traces, fs, gt_trains, n_ch, params)
            results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def run_zpybci_synthetic(params):
    """Fall back to zpybci's own synthetic data generator."""
    from zpybci.synthetic import generate_recording

    print("Running zpybci synthetic fallback (3 recordings)\n")
    configs = [
        {"n_channels": 32, "n_units": 3, "duration_s": 30.0,
         "noise_std": 1.0, "label": "synth-easy"},
        {"n_channels": 32, "n_units": 8, "duration_s": 30.0,
         "noise_std": 1.5, "label": "synth-medium"},
        {"n_channels": 64, "n_units": 15, "duration_s": 30.0,
         "noise_std": 2.0, "label": "synth-hard"},
    ]
    results = []
    for ci, cfg in enumerate(configs):
        name = cfg["label"]
        print(f"  [{ci+1}/{len(configs)}] {name}")
        try:
            rec = generate_recording(
                n_channels=cfg["n_channels"], n_units=cfg["n_units"],
                duration_s=cfg["duration_s"], noise_std=cfg["noise_std"],
                sampling_rate=30000.0, firing_rate=5.0, seed=42,
            )
            gt_trains = {}
            for u in range(rec["n_units"]):
                mask = rec["spike_labels"] == u
                gt_trains[u] = np.sort(rec["all_spike_times"][mask])

            res = process_recording(
                name, rec["data"], 30000.0, gt_trains, cfg["n_channels"], params
            )
            results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    return results


def run_spikeforest(max_recordings, params):
    """Load and process SpikeForest paired Kampff recordings."""
    import spikeforest as sf

    print("Loading paired Kampff recordings from SpikeForest...")
    recordings = sf.load_spikeforest_recordings(PAIRED_KAMPFF_URI)
    n_total = len(recordings)
    n_run = min(max_recordings, n_total) if max_recordings else n_total
    print(f"Found {n_total} recordings, processing {n_run}\n")

    results = []
    for i in range(n_run):
        R = recordings[i]
        name = f"{R.study_name}/{R.recording_name}"
        print(f"  [{i+1}/{n_run}] {name}")
        try:
            recording = R.get_recording_extractor()
            sorting_true = R.get_sorting_true_extractor()

            traces = recording.get_traces().astype(np.float64)
            fs = recording.get_sampling_frequency()
            n_ch = recording.get_num_channels()

            gt_trains = {}
            for uid in sorting_true.get_unit_ids():
                train = sorting_true.get_unit_spike_train(uid)
                gt_trains[int(uid)] = np.sort(train)

            res = process_recording(name, traces, fs, gt_trains, n_ch, params)
            results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    return results


def print_summary(results, data_source):
    """Print summary table."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({data_source})")
    print(f"{'='*70}")
    print(f"  {'Recording':<30} {'Ch':>4} {'GT#':>7} {'Det#':>7} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Time':>6}")
    print("  " + "-" * 66)

    accs = []
    for r in results:
        m = r.get("metrics")
        if m is None:
            print(f"  {r['name']:<30} {r['n_channels']:>4} {r['n_gt']:>7} "
                  f"{r['n_sorted']:>7} {'---':>7} {'---':>7} {'---':>7} "
                  f"{r['elapsed']:>5.1f}s")
            continue
        o = m["overall"]
        accs.append(o["accuracy"])
        print(f"  {r['name']:<30} {r['n_channels']:>4} {r['n_gt']:>7} "
              f"{r['n_sorted']:>7} {o['accuracy']:>7.3f} "
              f"{o['precision']:>7.3f} {o['recall']:>7.3f} "
              f"{r['elapsed']:>5.1f}s")

    if accs:
        avg = np.mean(accs)
        print("  " + "-" * 66)
        print(f"  {'AVERAGE':<30} {'':>4} {'':>7} {'':>7} {avg:>7.3f}")

    if data_source == "spikeforest":
        print(f"\n  Reference: MountainSort5 on paired Kampff: "
              f"{MS5_REFERENCE['average_accuracy']} average accuracy")
    print()


def save_results(results, data_source):
    """Save results to JSON in benchmarks/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spikeforest_{data_source}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    output = {
        "timestamp": timestamp,
        "data_source": data_source,
        "zpybci_version": zbci.__version__,
        "spikeinterface_available": HAS_SI,
        "tolerance_ms": TOLERANCE_MS,
        "results": [],
    }

    for r in results:
        entry = {
            "name": r["name"],
            "n_channels": r["n_channels"],
            "elapsed_s": r["elapsed"],
            "n_gt_spikes": r["n_gt"],
            "n_sorted_spikes": r["n_sorted"],
        }
        if r.get("metrics") is not None:
            entry["metrics"] = r["metrics"]
        output["results"].append(entry)

    # Overall summary
    accs = [
        r["metrics"]["overall"]["accuracy"]
        for r in results
        if r.get("metrics") is not None
    ]
    if accs:
        output["summary"] = {
            "mean_accuracy": float(np.mean(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "n_recordings": len(accs),
        }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="SpikeForest validation benchmark for Zerostone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--max-recordings", type=int, default=None,
                        help="Limit number of recordings to process.")
    parser.add_argument("--fallback-synthetic", action="store_true",
                        help="Force synthetic data mode.")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Detection threshold (MAD units).")
    parser.add_argument("--matched-filter", action="store_true", default=True,
                        help="Enable matched filter second-pass (default: on).")
    parser.add_argument("--no-matched-filter", dest="matched_filter",
                        action="store_false",
                        help="Disable matched filter second-pass.")
    parser.add_argument("--tolerance-ms", type=float, default=TOLERANCE_MS,
                        help=f"Spike matching tolerance in ms (default: {TOLERANCE_MS}).")
    args = parser.parse_args()

    tolerance_ms = args.tolerance_ms

    params = {
        "threshold": args.threshold,
        "matched_filter_detect": args.matched_filter,
        "matched_filter_threshold": 4.0,
    }

    print("=" * 70)
    print("  Zerostone SpikeForest Validation Benchmark")
    print(f"  zpybci {zbci.__version__}")
    if HAS_SI:
        import spikeinterface
        print(f"  spikeinterface {spikeinterface.__version__}")
    print(f"  Comparison: SpikeInterface compare_sorter_to_ground_truth"
          f" (delta_time={tolerance_ms}ms, hungarian matching)")
    print("=" * 70)

    data_source = "synthetic"
    is_synthetic = args.fallback_synthetic

    if not is_synthetic:
        try:
            import spikeforest  # noqa: F401
            results = run_spikeforest(args.max_recordings, params)
            data_source = "spikeforest"
        except ImportError:
            print("spikeforest not available. Install with:")
            print("  pip install spikeforest kachery-cloud\n")
            is_synthetic = True

    if is_synthetic:
        if HAS_SI:
            results = run_si_synthetic(params)
            data_source = "si_synthetic"
        else:
            results = run_zpybci_synthetic(params)
            data_source = "zpybci_synthetic"

    if results:
        print_summary(results, data_source)
        save_results(results, data_source)
    else:
        print("\nNo recordings processed successfully.")


if __name__ == "__main__":
    main()
