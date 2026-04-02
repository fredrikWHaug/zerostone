"""Parameter sweep benchmark for Zerostone spike sorter.

Systematically tests parameter configurations using SpikeInterface's
generate_ground_truth_recording() and compare_sorter_to_ground_truth()
for standardized accuracy computation (Hungarian matching).

The sweep is structured in phases:
  Phase 1: Bandpass filtering (the biggest expected win)
  Phase 2: Detection mode (amplitude vs NEO vs SNEO)
  Phase 3: Threshold sweep
  Phase 4: Feature toggles (ccg_merge, gmm_refine, common_median_ref)
  Phase 5: Combined best parameters

Usage:
    python benchmarks/param_sweep.py
    python benchmarks/param_sweep.py --quick   # medium only
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

RESULTS_DIR = Path(__file__).parent / "results"
TOLERANCE_MS = 0.4


# ---------------------------------------------------------------------------
# Recording generation (cached per difficulty)
# ---------------------------------------------------------------------------

RECORDING_CONFIGS = {
    "easy": {
        "num_channels": 4, "num_units": 3, "duration": 30.0,
        "noise_levels": 3.0,
    },
    "medium": {
        "num_channels": 16, "num_units": 8, "duration": 30.0,
        "noise_levels": 5.0,
    },
    "hard": {
        "num_channels": 32, "num_units": 15, "duration": 30.0,
        "noise_levels": 8.0,
    },
}

_recording_cache = {}


def get_recording(difficulty):
    """Generate and cache a synthetic recording."""
    if difficulty in _recording_cache:
        return _recording_cache[difficulty]

    cfg = RECORDING_CONFIGS[difficulty]
    print(f"  Generating {difficulty} recording ({cfg['num_channels']}ch, "
          f"{cfg['num_units']} units, noise={cfg['noise_levels']})...", end=" ", flush=True)
    t0 = time.perf_counter()

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

    gt_trains = {}
    for uid in sorting_true.get_unit_ids():
        train = sorting_true.get_unit_spike_train(uid)
        gt_trains[uid] = np.sort(train)

    gt_sorting = _build_gt_sorting(gt_trains, fs)

    cached = {
        "traces": traces,
        "fs": fs,
        "n_channels": rec.get_num_channels(),
        "gt_trains": gt_trains,
        "gt_sorting": gt_sorting,
        "n_gt_spikes": sum(len(v) for v in gt_trains.values()),
    }
    _recording_cache[difficulty] = cached
    print(f"done ({time.perf_counter() - t0:.1f}s, {cached['n_gt_spikes']} GT spikes)")
    return cached


def _build_gt_sorting(gt_trains, fs):
    """Build NumpySorting from ground-truth spike trains."""
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


# ---------------------------------------------------------------------------
# Sorting + evaluation
# ---------------------------------------------------------------------------

# Baseline params (current defaults -- NO bandpass)
BASELINE_PARAMS = {
    "threshold": 5.0,
    "refractory": 15,
    "spatial_radius": 75.0,
    "temporal_radius": 5,
    "align_half_window": 5,
    "pre_samples": 16,
    "cluster_threshold": 5.0,
    "cluster_max_count": 1000,
    "whitening_epsilon": 1e-6,
    "detection_mode": "amplitude",
    "sneo_smooth_window": 3,
    "ccg_merge": False,
    "matched_filter_detect": True,
    "matched_filter_threshold": 4.0,
    "gmm_refine": False,
    "gmm_max_iter": 10,
    "bandpass_low": 0.0,
    "bandpass_high": 0.0,
    "sample_rate": 30000.0,
    "common_median_ref": False,
}


def run_config(difficulty, params, label=""):
    """Run a single parameter configuration and return metrics."""
    rec = get_recording(difficulty)
    traces = rec["traces"]
    n_ch = rec["n_channels"]
    fs = rec["fs"]

    probe = zbci.ProbeLayout.linear(n_ch, params.get("probe_pitch", 25.0))

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

    if n_spikes == 0:
        return {
            "label": label, "difficulty": difficulty, "elapsed": elapsed,
            "n_spikes": 0, "n_clusters": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "well_detected": 0, "n_gt_units": len(rec["gt_trains"]),
            "per_unit": [],
        }

    spike_times = np.asarray(result["spike_times"][:n_spikes], dtype=np.int64)
    labels = np.asarray(result["labels"][:n_spikes], dtype=np.int64)

    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[spike_times],
        labels_list=[labels],
        sampling_frequency=fs,
    )

    comp = compare_sorter_to_ground_truth(
        rec["gt_sorting"],
        sorting,
        exhaustive_gt=True,
        delta_time=TOLERANCE_MS,
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
        })

    return {
        "label": label, "difficulty": difficulty, "elapsed": elapsed,
        "n_spikes": n_spikes, "n_clusters": result["n_clusters"],
        "accuracy": float(avg["accuracy"]),
        "precision": float(avg["precision"]),
        "recall": float(avg["recall"]),
        "well_detected": int(comp.count_well_detected_units(well_detected_score=0.8)),
        "n_gt_units": len(rec["gt_trains"]),
        "per_unit": per_unit,
    }


def print_result(r, indent=4):
    """Print a single result line."""
    prefix = " " * indent
    wd = f"{r['well_detected']}/{r['n_gt_units']}"
    print(f"{prefix}{r['label']:<45} "
          f"acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"rec={r['recall']:.3f}  wd={wd:<5}  "
          f"{r['n_spikes']:>5} spk  {r['n_clusters']:>3} cl  "
          f"{r['elapsed']:.1f}s")


# ---------------------------------------------------------------------------
# Sweep phases
# ---------------------------------------------------------------------------

def phase_1_bandpass(difficulty):
    """Test bandpass filtering."""
    print(f"\n  PHASE 1: Bandpass filtering ({difficulty})")
    print("  " + "-" * 70)
    results = []

    # No bandpass (baseline)
    p = dict(BASELINE_PARAMS)
    r = run_config(difficulty, p, "baseline (no bandpass)")
    print_result(r)
    results.append(r)

    # Standard spike band
    p = dict(BASELINE_PARAMS)
    p["bandpass_low"] = 300.0
    p["bandpass_high"] = 6000.0
    r = run_config(difficulty, p, "bandpass 300-6000")
    print_result(r)
    results.append(r)

    # Wider band
    p = dict(BASELINE_PARAMS)
    p["bandpass_low"] = 200.0
    p["bandpass_high"] = 8000.0
    r = run_config(difficulty, p, "bandpass 200-8000")
    print_result(r)
    results.append(r)

    # Narrower band
    p = dict(BASELINE_PARAMS)
    p["bandpass_low"] = 400.0
    p["bandpass_high"] = 5000.0
    r = run_config(difficulty, p, "bandpass 400-5000")
    print_result(r)
    results.append(r)

    return results


def phase_2_detection_mode(difficulty, best_bp):
    """Test detection modes with best bandpass setting."""
    print(f"\n  PHASE 2: Detection mode ({difficulty})")
    print("  " + "-" * 70)
    results = []

    for mode in ["amplitude", "neo", "sneo"]:
        p = dict(BASELINE_PARAMS)
        p.update(best_bp)
        p["detection_mode"] = mode
        label = f"det={mode}"
        r = run_config(difficulty, p, label)
        print_result(r)
        results.append(r)

    return results


def phase_3_threshold(difficulty, best_bp, best_det):
    """Sweep threshold values."""
    print(f"\n  PHASE 3: Threshold sweep ({difficulty})")
    print("  " + "-" * 70)
    results = []

    for thr in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        p = dict(BASELINE_PARAMS)
        p.update(best_bp)
        p.update(best_det)
        p["threshold"] = thr
        label = f"thr={thr:.1f}"
        r = run_config(difficulty, p, label)
        print_result(r)
        results.append(r)

    return results


def phase_4_features(difficulty, best_bp, best_det, best_thr):
    """Test feature toggles."""
    print(f"\n  PHASE 4: Feature toggles ({difficulty})")
    print("  " + "-" * 70)
    results = []

    base = dict(BASELINE_PARAMS)
    base.update(best_bp)
    base.update(best_det)
    base.update(best_thr)

    # Baseline with best so far
    r = run_config(difficulty, base, "best so far (features off)")
    print_result(r)
    results.append(r)

    # CCG merge
    p = dict(base)
    p["ccg_merge"] = True
    r = run_config(difficulty, p, "+ ccg_merge")
    print_result(r)
    results.append(r)

    # GMM refine
    p = dict(base)
    p["gmm_refine"] = True
    r = run_config(difficulty, p, "+ gmm_refine")
    print_result(r)
    results.append(r)

    # Common median ref
    p = dict(base)
    p["common_median_ref"] = True
    r = run_config(difficulty, p, "+ common_median_ref")
    print_result(r)
    results.append(r)

    # CCG + GMM
    p = dict(base)
    p["ccg_merge"] = True
    p["gmm_refine"] = True
    r = run_config(difficulty, p, "+ ccg_merge + gmm_refine")
    print_result(r)
    results.append(r)

    # All features
    p = dict(base)
    p["ccg_merge"] = True
    p["gmm_refine"] = True
    p["common_median_ref"] = True
    r = run_config(difficulty, p, "+ all features")
    print_result(r)
    results.append(r)

    # Matched filter threshold sweep
    for mft in [3.0, 3.5, 4.5, 5.0]:
        p = dict(base)
        p["matched_filter_threshold"] = mft
        r = run_config(difficulty, p, f"mf_thr={mft:.1f}")
        print_result(r)
        results.append(r)

    return results


def find_best(results):
    """Return the result with highest accuracy."""
    return max(results, key=lambda r: r["accuracy"])


def extract_params_diff(result_label, all_results):
    """Extract the parameter delta that produced the best result."""
    # This is just informational; actual best params are tracked explicitly
    pass


def run_sweep(difficulty):
    """Run full parameter sweep on one difficulty level."""
    print(f"\n{'='*76}")
    print(f"  SWEEP: {difficulty.upper()}")
    print(f"{'='*76}")

    # Phase 1: Bandpass
    bp_results = phase_1_bandpass(difficulty)
    best_bp_r = find_best(bp_results)
    # Extract bandpass params from the best
    best_bp = {}
    if "300-6000" in best_bp_r["label"]:
        best_bp = {"bandpass_low": 300.0, "bandpass_high": 6000.0}
    elif "200-8000" in best_bp_r["label"]:
        best_bp = {"bandpass_low": 200.0, "bandpass_high": 8000.0}
    elif "400-5000" in best_bp_r["label"]:
        best_bp = {"bandpass_low": 400.0, "bandpass_high": 5000.0}
    else:
        best_bp = {"bandpass_low": 0.0, "bandpass_high": 0.0}
    print(f"  >> Best bandpass: {best_bp_r['label']} (acc={best_bp_r['accuracy']:.3f})")

    # Phase 2: Detection mode
    det_results = phase_2_detection_mode(difficulty, best_bp)
    best_det_r = find_best(det_results)
    best_det = {"detection_mode": best_det_r["label"].split("=")[1]}
    print(f"  >> Best detection: {best_det_r['label']} (acc={best_det_r['accuracy']:.3f})")

    # Phase 3: Threshold
    thr_results = phase_3_threshold(difficulty, best_bp, best_det)
    best_thr_r = find_best(thr_results)
    best_thr = {"threshold": float(best_thr_r["label"].split("=")[1])}
    print(f"  >> Best threshold: {best_thr_r['label']} (acc={best_thr_r['accuracy']:.3f})")

    # Phase 4: Features
    feat_results = phase_4_features(difficulty, best_bp, best_det, best_thr)
    best_feat_r = find_best(feat_results)
    print(f"  >> Best features: {best_feat_r['label']} (acc={best_feat_r['accuracy']:.3f})")

    all_results = bp_results + det_results + thr_results + feat_results
    return all_results, best_bp, best_det, best_thr


def run_final_combined(difficulties, best_params_per_diff):
    """Run the combined best parameter set across all difficulties."""
    print(f"\n{'='*76}")
    print("  FINAL: Combined best parameters across all difficulties")
    print(f"{'='*76}")

    # Aggregate best params: pick the most common winner for each param
    # or just use the medium difficulty's best (most representative)
    if "medium" in best_params_per_diff:
        bp, det, thr = best_params_per_diff["medium"]
    else:
        # Use whatever we have
        key = list(best_params_per_diff.keys())[0]
        bp, det, thr = best_params_per_diff[key]

    # Build the combined config
    combined = dict(BASELINE_PARAMS)
    combined.update(bp)
    combined.update(det)
    combined.update(thr)

    # Also test with feature combinations
    configs = [
        ("combined-base", {}),
        ("combined+ccg", {"ccg_merge": True}),
        ("combined+gmm", {"gmm_refine": True}),
        ("combined+cmr", {"common_median_ref": True}),
        ("combined+ccg+gmm", {"ccg_merge": True, "gmm_refine": True}),
        ("combined+all", {"ccg_merge": True, "gmm_refine": True, "common_median_ref": True}),
    ]

    all_results = {}
    for diff in difficulties:
        print(f"\n  {diff.upper()}:")
        print("  " + "-" * 70)
        diff_results = []
        for label, overrides in configs:
            p = dict(combined)
            p.update(overrides)
            r = run_config(diff, p, label)
            print_result(r)
            diff_results.append(r)
        all_results[diff] = diff_results

    return all_results, combined


def print_summary_table(all_results_by_diff, baseline_results):
    """Print a cross-difficulty summary comparing baseline vs best."""
    print(f"\n{'='*76}")
    print("  SUMMARY: Baseline vs Best per difficulty")
    print(f"{'='*76}")
    print(f"  {'Difficulty':<10} {'Baseline Acc':>12} {'Best Config':>40} {'Best Acc':>10} {'Delta':>8}")
    print(f"  {'-'*82}")

    for diff in ["easy", "medium", "hard"]:
        if diff not in all_results_by_diff:
            continue
        bl = baseline_results.get(diff, 0.0)
        best = find_best(all_results_by_diff[diff])
        delta = best["accuracy"] - bl
        sign = "+" if delta >= 0 else ""
        print(f"  {diff:<10} {bl:>12.3f} {best['label']:>40} {best['accuracy']:>10.3f} {sign}{delta:>7.3f}")

    # Overall average
    bl_avg = np.mean([v for v in baseline_results.values() if v > 0]) if baseline_results else 0
    best_accs = []
    for diff in ["easy", "medium", "hard"]:
        if diff in all_results_by_diff:
            best_accs.append(find_best(all_results_by_diff[diff])["accuracy"])
    best_avg = np.mean(best_accs) if best_accs else 0
    delta = best_avg - bl_avg
    sign = "+" if delta >= 0 else ""
    print(f"  {'-'*82}")
    print(f"  {'AVERAGE':<10} {bl_avg:>12.3f} {'':>40} {best_avg:>10.3f} {sign}{delta:>7.3f}")
    print()


def save_sweep_results(all_results, baseline_accs, combined_params):
    """Save sweep results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"param_sweep_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "zpybci_version": zbci.__version__,
        "tolerance_ms": TOLERANCE_MS,
        "baseline_params": BASELINE_PARAMS,
        "baseline_accuracies": baseline_accs,
        "combined_best_params": combined_params,
        "results": {},
    }

    for diff, results in all_results.items():
        output["results"][diff] = []
        for r in results:
            output["results"][diff].append({
                "label": r["label"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "n_spikes": r["n_spikes"],
                "n_clusters": r["n_clusters"],
                "elapsed": r["elapsed"],
                "well_detected": r["well_detected"],
                "n_gt_units": r["n_gt_units"],
            })

    # Find overall best per difficulty
    summary = {}
    for diff, results in all_results.items():
        best = find_best(results)
        summary[diff] = {
            "best_label": best["label"],
            "best_accuracy": best["accuracy"],
            "baseline_accuracy": baseline_accs.get(diff, 0.0),
            "improvement": best["accuracy"] - baseline_accs.get(diff, 0.0),
        }
    output["summary"] = summary

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for Zerostone spike sorter.")
    parser.add_argument("--quick", action="store_true", help="Only run medium difficulty.")
    args = parser.parse_args()

    if not HAS_SI:
        print("ERROR: spikeinterface is required. Install with: pip install spikeinterface")
        return

    print("=" * 76)
    print("  Zerostone Parameter Sweep Benchmark")
    print(f"  zpybci {zbci.__version__}")
    import spikeinterface
    print(f"  spikeinterface {spikeinterface.__version__}")
    print(f"  Matching: Hungarian, delta_time={TOLERANCE_MS}ms")
    print("=" * 76)

    difficulties = ["medium"] if args.quick else ["easy", "medium", "hard"]

    # Collect baseline accuracies first
    print("\n  Computing baselines...")
    baseline_accs = {}
    for diff in difficulties:
        r = run_config(diff, BASELINE_PARAMS, f"baseline-{diff}")
        baseline_accs[diff] = r["accuracy"]
        print_result(r)

    # Run sweep per difficulty
    best_params_per_diff = {}
    all_sweep_results = {}

    for diff in difficulties:
        sweep_results, bp, det, thr = run_sweep(diff)
        best_params_per_diff[diff] = (bp, det, thr)
        all_sweep_results[diff] = sweep_results

    # Run final combined configs across all difficulties
    final_results, combined_params = run_final_combined(difficulties, best_params_per_diff)

    # Merge final results into all_sweep_results for summary
    for diff in difficulties:
        if diff in final_results:
            all_sweep_results[diff].extend(final_results[diff])

    # Print summary
    print_summary_table(all_sweep_results, baseline_accs)

    # Save
    save_sweep_results(all_sweep_results, baseline_accs, combined_params)

    # Print the recommended parameters
    print("\n  RECOMMENDED PARAMETERS:")
    print("  " + "-" * 40)
    for key in sorted(combined_params.keys()):
        if combined_params[key] != BASELINE_PARAMS.get(key):
            print(f"    {key}: {BASELINE_PARAMS.get(key)} -> {combined_params[key]}")


if __name__ == "__main__":
    main()
