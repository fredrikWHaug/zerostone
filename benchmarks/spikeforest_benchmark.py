#!/usr/bin/env python3
"""SpikeForest validation benchmark for Zerostone spike sorter.

Loads paired Kampff recordings (juxtacellular ground truth) from SpikeForest
and evaluates Zerostone's sort_multichannel against real-world ground truth.
Falls back to synthetic data if spikeforest/kachery-cloud are not installed.

Usage:
    python benchmarks/spikeforest_benchmark.py
    python benchmarks/spikeforest_benchmark.py --max-recordings 2
    python benchmarks/spikeforest_benchmark.py --fallback-synthetic
"""

import argparse
import time

import numpy as np

import zpybci as zbci

PAIRED_KAMPFF_URI = (
    "sha1://b8b571d001f9a531040e79165e8f492d758ec5e0"
    "?paired-kampff-spikeforest-recordings.json"
)

SUPPORTED_CHANNELS = [4, 8, 16, 32, 64, 128]
TOLERANCE = 20  # samples at 30kHz (~0.67ms)

# Approximate published MountainSort5 numbers on paired Kampff
MS5_REFERENCE = {"average_accuracy": "~70-80%"}


def nearest_supported_channels(n):
    """Return the largest supported channel count <= n."""
    valid = [c for c in SUPPORTED_CHANNELS if c <= n]
    return max(valid) if valid else SUPPORTED_CHANNELS[0]


def match_spikes(gt_times, sorted_times, tolerance=TOLERANCE):
    """Greedy nearest-neighbor spike matching within tolerance window."""
    if len(gt_times) == 0:
        return 0, 0, len(sorted_times)
    if len(sorted_times) == 0:
        return 0, len(gt_times), 0

    matched = set()
    tp = 0
    search_start = 0

    for gt_t in gt_times:
        best_idx = -1
        best_dist = tolerance + 1

        while search_start < len(sorted_times) and sorted_times[search_start] < gt_t - tolerance:
            search_start += 1

        for j in range(search_start, len(sorted_times)):
            st = sorted_times[j]
            if st > gt_t + tolerance:
                break
            dist = abs(int(st) - int(gt_t))
            if dist < best_dist and j not in matched:
                best_dist = dist
                best_idx = j

        if best_idx >= 0:
            tp += 1
            matched.add(best_idx)

    fn = len(gt_times) - tp
    fp = len(sorted_times) - tp
    return tp, fn, fp


def compute_per_unit_accuracy(gt_trains, sorted_times, sorted_labels, tolerance=TOLERANCE):
    """Compute per-GT-unit metrics with greedy best-match assignment.

    Parameters
    ----------
    gt_trains : dict
        Mapping unit_id -> sorted array of spike sample indices.
    sorted_times : np.ndarray
        Detected spike times.
    sorted_labels : np.ndarray
        Cluster labels for detected spikes.

    Returns
    -------
    list of dict, one per GT unit, plus overall metrics dict.
    """
    sorted_clusters = {}
    if len(sorted_labels) > 0:
        for cl in np.unique(sorted_labels):
            sorted_clusters[int(cl)] = np.sort(sorted_times[sorted_labels == cl])

    gt_ids = sorted(gt_trains.keys())
    cluster_ids = sorted(sorted_clusters.keys())

    # Build agreement matrix
    agreement = np.zeros((len(gt_ids), len(cluster_ids)), dtype=np.float64)
    details = {}
    for i, uid in enumerate(gt_ids):
        for j, cl in enumerate(cluster_ids):
            tp, fn, fp = match_spikes(gt_trains[uid], sorted_clusters[cl], tolerance)
            denom = tp + fn + fp
            agreement[i, j] = tp / denom if denom > 0 else 0.0
            details[(i, j)] = (tp, fn, fp)

    # Greedy assignment
    assigned_gt = set()
    assigned_cl = set()
    per_unit = []
    total_tp, total_fn, total_fp = 0, 0, 0

    for _ in range(min(len(gt_ids), len(cluster_ids))):
        best_acc = -1.0
        best_i, best_j = -1, -1
        for i in range(len(gt_ids)):
            if i in assigned_gt:
                continue
            for j in range(len(cluster_ids)):
                if j in assigned_cl:
                    continue
                if agreement[i, j] > best_acc:
                    best_acc = agreement[i, j]
                    best_i, best_j = i, j
        if best_i < 0 or best_acc <= 0.0:
            break

        tp, fn, fp = details[(best_i, best_j)]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        per_unit.append({
            "gt_unit": gt_ids[best_i], "cluster": cluster_ids[best_j],
            "accuracy": acc, "precision": prec, "recall": rec,
            "tp": tp, "fn": fn, "fp": fp,
            "n_gt": len(gt_trains[gt_ids[best_i]]),
        })
        assigned_gt.add(best_i)
        assigned_cl.add(best_j)
        total_tp += tp
        total_fn += fn
        total_fp += fp

    # Unmatched GT units
    for i, uid in enumerate(gt_ids):
        if i not in assigned_gt:
            n = len(gt_trains[uid])
            per_unit.append({
                "gt_unit": uid, "cluster": -1,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                "tp": 0, "fn": n, "fp": 0, "n_gt": n,
            })
            total_fn += n

    denom = total_tp + total_fn + total_fp
    overall = {
        "accuracy": total_tp / denom if denom > 0 else 0.0,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0,
    }
    per_unit.sort(key=lambda p: p["gt_unit"])
    return per_unit, overall


def process_recording(name, traces, fs, gt_trains, n_channels_orig):
    """Sort a single recording and return metrics."""
    n_ch = nearest_supported_channels(n_channels_orig)
    if n_ch != n_channels_orig:
        # Subsample channels evenly
        idx = np.linspace(0, n_channels_orig - 1, n_ch, dtype=int)
        traces = traces[:, idx]
        print(f"    Subsampled {n_channels_orig} -> {n_ch} channels")

    probe = zbci.ProbeLayout.linear(n_ch, 25.0)

    t0 = time.perf_counter()
    result = zbci.sort_multichannel(
        traces, probe,
        threshold=5.0,
        refractory=15,
        matched_filter_detect=True,
        matched_filter_threshold=4.0,
    )
    elapsed = time.perf_counter() - t0

    sorted_times = np.array(result["spike_times"][:result["n_spikes"]], dtype=np.int64)
    sorted_labels = np.array(result["labels"][:result["n_spikes"]], dtype=np.int64)

    per_unit, overall = compute_per_unit_accuracy(gt_trains, sorted_times, sorted_labels)

    n_gt_total = sum(len(v) for v in gt_trains.values())
    print(f"    {name}: {n_ch}ch, {n_gt_total} GT spikes, "
          f"{result['n_spikes']} detected, {result['n_clusters']} clusters, "
          f"acc={overall['accuracy']:.3f}, prec={overall['precision']:.3f}, "
          f"rec={overall['recall']:.3f}, {elapsed:.1f}s")

    # Per-unit detail
    for pu in per_unit:
        cl_str = str(pu["cluster"]) if pu["cluster"] >= 0 else "---"
        print(f"      Unit {pu['gt_unit']:>3} -> cl {cl_str:>4}  "
              f"acc={pu['accuracy']:.3f}  prec={pu['precision']:.3f}  "
              f"rec={pu['recall']:.3f}  ({pu['tp']}/{pu['n_gt']} matched)")

    return {
        "name": name, "n_channels": n_ch, "elapsed": elapsed,
        "per_unit": per_unit, "overall": overall,
        "n_gt": n_gt_total, "n_sorted": result["n_spikes"],
    }


def run_spikeforest(max_recordings):
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

            res = process_recording(name, traces, fs, gt_trains, n_ch)
            results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    return results


def run_synthetic():
    """Fall back to synthetic data."""
    from zpybci.synthetic import generate_recording

    print("Running synthetic fallback (3 recordings)\n")
    configs = [
        {"n_channels": 32, "n_units": 3, "duration_s": 30.0, "noise_std": 1.0, "label": "synth-easy"},
        {"n_channels": 32, "n_units": 8, "duration_s": 30.0, "noise_std": 1.5, "label": "synth-medium"},
        {"n_channels": 64, "n_units": 15, "duration_s": 30.0, "noise_std": 2.0, "label": "synth-hard"},
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

            res = process_recording(name, rec["data"], 30000.0, gt_trains, cfg["n_channels"])
            results.append(res)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    return results


def print_summary(results, is_synthetic):
    """Print summary table with reference comparison."""
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Recording':<30} {'Ch':>4} {'GT#':>7} {'Det#':>7} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Time':>6}")
    print("  " + "-" * 66)

    accs = []
    for r in results:
        o = r["overall"]
        accs.append(o["accuracy"])
        print(f"  {r['name']:<30} {r['n_channels']:>4} {r['n_gt']:>7} "
              f"{r['n_sorted']:>7} {o['accuracy']:>7.3f} "
              f"{o['precision']:>7.3f} {o['recall']:>7.3f} "
              f"{r['elapsed']:>5.1f}s")

    if accs:
        avg = np.mean(accs)
        print("  " + "-" * 66)
        print(f"  {'AVERAGE':<30} {'':>4} {'':>7} {'':>7} {avg:>7.3f}")

    if not is_synthetic:
        print(f"\n  Reference: MountainSort5 on paired Kampff: "
              f"{MS5_REFERENCE['average_accuracy']} average accuracy")
    print()


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
    args = parser.parse_args()

    print("=" * 70)
    print("  Zerostone SpikeForest Validation Benchmark")
    print("=" * 70)

    is_synthetic = args.fallback_synthetic
    if not is_synthetic:
        try:
            import spikeforest  # noqa: F401
            results = run_spikeforest(args.max_recordings)
        except ImportError:
            print("spikeforest not available. Install with:")
            print("  pip install spikeforest kachery-cloud\n")
            print("Falling back to synthetic data.\n")
            is_synthetic = True

    if is_synthetic:
        results = run_synthetic()

    if results:
        print_summary(results, is_synthetic)
    else:
        print("\nNo recordings processed successfully.")


if __name__ == "__main__":
    main()
