"""Comprehensive parameter sweep benchmark for Zerostone spike sorting.

Sweeps detection threshold, detection mode, CCG merge, template subtraction
passes, and ISI split across easy/medium/hard presets.

Usage:
    python benchmarks/sweep_benchmark.py
    python benchmarks/sweep_benchmark.py --preset medium
    python benchmarks/sweep_benchmark.py --threshold-only
"""

import argparse
import itertools
import time

import numpy as np

from zpybci.zpybci import ProbeLayout, sort_multichannel
from zpybci.synthetic import generate_recording

PRESETS = {
    "easy": {
        "n_channels": 32,
        "n_units": 5,
        "noise_std": 1.0,
        "duration_s": 60.0,
        "firing_rate": 5.0,
    },
    "medium": {
        "n_channels": 32,
        "n_units": 10,
        "noise_std": 1.5,
        "duration_s": 60.0,
        "firing_rate": 8.0,
    },
    "hard": {
        "n_channels": 64,
        "n_units": 20,
        "noise_std": 2.0,
        "duration_s": 60.0,
        "firing_rate": 10.0,
    },
}

TOLERANCE = 20  # 0.67 ms at 30 kHz


def match_spikes(gt_times, sorted_times, tolerance=TOLERANCE):
    """Greedy nearest-neighbor spike matching."""
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


def compute_accuracy(rec, sort_result):
    """Compute overall accuracy from recording and sort result."""
    gt_times = rec["all_spike_times"]
    gt_labels = rec["spike_labels"]
    n_units = rec["n_units"]

    sorted_times = np.array(sort_result["spike_times"], dtype=np.int64)
    sorted_labels = np.array(sort_result["labels"], dtype=np.int64)

    if len(sorted_times) == 0:
        return 0.0, 0.0, 0.0, sort_result["n_clusters"]

    # Group by unit/cluster
    gt_by_unit = {}
    for u in range(n_units):
        mask = gt_labels == u
        gt_by_unit[u] = np.sort(gt_times[mask])

    sorted_by_cluster = {}
    for cl in np.unique(sorted_labels):
        mask = sorted_labels == cl
        sorted_by_cluster[int(cl)] = np.sort(sorted_times[mask])

    cluster_ids = sorted(sorted_by_cluster.keys())

    # Agreement matrix
    agreement = {}
    details = {}
    for u in range(n_units):
        for cl in cluster_ids:
            tp, fn, fp = match_spikes(gt_by_unit[u], sorted_by_cluster[cl])
            denom = tp + fn + fp
            agreement[(u, cl)] = tp / denom if denom > 0 else 0.0
            details[(u, cl)] = (tp, fn, fp)

    # Greedy assignment
    used_units = set()
    used_clusters = set()
    total_tp = total_fn = total_fp = 0

    for _ in range(n_units):
        best_acc = -1.0
        best_pair = None
        for u in range(n_units):
            if u in used_units:
                continue
            for cl in cluster_ids:
                if cl in used_clusters:
                    continue
                if agreement[(u, cl)] > best_acc:
                    best_acc = agreement[(u, cl)]
                    best_pair = (u, cl)

        if best_pair is None or best_acc <= 0:
            break

        u, cl = best_pair
        tp, fn, fp = details[(u, cl)]
        total_tp += tp
        total_fn += fn
        total_fp += fp
        used_units.add(u)
        used_clusters.add(cl)

    # Unmatched units
    for u in range(n_units):
        if u not in used_units:
            total_fn += len(gt_by_unit[u])

    denom = total_tp + total_fn + total_fp
    acc = total_tp / denom if denom > 0 else 0.0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return acc, prec, rec_val, sort_result["n_clusters"]


def generate_data(preset_name, seed=42):
    """Generate recording for a preset (cached per call)."""
    params = PRESETS[preset_name]
    np.random.seed(seed)
    rec = generate_recording(
        n_channels=params["n_channels"],
        duration_s=params["duration_s"],
        sampling_rate=30000.0,
        n_units=params["n_units"],
        firing_rate=params["firing_rate"],
        noise_std=params["noise_std"],
        seed=seed,
    )
    return rec, params


def run_single(rec, params, threshold=5.0, detection_mode="amplitude",
               ccg_merge=False, template_subtract_passes=2,
               isi_split_threshold=0.1, sneo_smooth_window=3,
               ccg_template_corr_threshold=0.5):
    """Run sort with given parameters and return metrics."""
    data = rec["data"]
    n_ch = params["n_channels"]
    probe = ProbeLayout.linear(n_ch, 25.0)

    t0 = time.perf_counter()
    result = sort_multichannel(
        data, probe,
        threshold=threshold,
        refractory=15,
        detection_mode=detection_mode,
        sneo_smooth_window=sneo_smooth_window,
        ccg_merge=ccg_merge,
        ccg_template_corr_threshold=ccg_template_corr_threshold,
        template_subtract_passes=template_subtract_passes,
        isi_split_threshold=isi_split_threshold,
    )
    elapsed = time.perf_counter() - t0

    acc, prec, recall, n_clusters = compute_accuracy(rec, result)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": recall,
        "n_clusters": n_clusters,
        "n_spikes": result["n_spikes"],
        "elapsed": elapsed,
    }


def sweep_threshold(preset_name, seed=42):
    """Sweep detection threshold from 3.0 to 7.0."""
    rec, params = generate_data(preset_name, seed)
    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

    print(f"\n{'='*70}")
    print(f"  Threshold Sweep: {preset_name}")
    print(f"{'='*70}")
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'Clust':>5}  {'Spikes':>7}  {'Time':>6}")
    print(f"  {'-'*55}")

    best_acc = 0
    best_thresh = 0
    for t in thresholds:
        m = run_single(rec, params, threshold=t)
        tag = ""
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_thresh = t
            tag = " <--"
        print(f"  {t:>6.1f}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['n_clusters']:>5}  {m['n_spikes']:>7}  "
              f"{m['elapsed']:>5.2f}s{tag}")

    print(f"\n  Best: threshold={best_thresh:.1f}, accuracy={best_acc:.3f}")
    return best_thresh, best_acc


def sweep_config(preset_name, seed=42):
    """Sweep detection mode x CCG x passes x ISI split."""
    rec, params = generate_data(preset_name, seed)

    configs = []
    # Core sweep: amplitude with varying settings
    for det in ["amplitude"]:
        for ccg in [False, True]:
            for passes in [1, 2, 3]:
                for isi in [0.0, 0.05, 0.1, 0.2]:
                    configs.append({
                        "detection_mode": det,
                        "ccg_merge": ccg,
                        "template_subtract_passes": passes,
                        "isi_split_threshold": isi,
                    })

    print(f"\n{'='*70}")
    print(f"  Config Sweep: {preset_name} ({len(configs)} configs)")
    print(f"{'='*70}")
    print(f"  {'Mode':>5} {'CCG':>4} {'Pass':>4} {'ISI':>5}  "
          f"{'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'Clust':>5}  {'Time':>6}")
    print(f"  {'-'*60}")

    best_acc = 0
    best_cfg = None
    for cfg in configs:
        # Use threshold 4.0 for medium, 5.0 for easy, 3.5 for hard
        threshold = {"easy": 5.0, "medium": 4.0, "hard": 3.5}.get(preset_name, 5.0)
        m = run_single(rec, params, threshold=threshold, **cfg)
        tag = ""
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_cfg = {**cfg, "threshold": threshold}
            tag = " <--"
        det_short = cfg["detection_mode"][:3]
        ccg_str = "Y" if cfg["ccg_merge"] else "N"
        print(f"  {det_short:>5} {ccg_str:>4} {cfg['template_subtract_passes']:>4} "
              f"{cfg['isi_split_threshold']:>5.2f}  "
              f"{m['accuracy']:>6.3f}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
              f"{m['n_clusters']:>5}  {m['elapsed']:>5.2f}s{tag}")

    print(f"\n  Best: {best_cfg}, accuracy={best_acc:.3f}")
    return best_cfg, best_acc


def sweep_ccg_threshold(preset_name, seed=42):
    """Sweep CCG template correlation threshold."""
    rec, params = generate_data(preset_name, seed)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"\n{'='*70}")
    print(f"  CCG Correlation Threshold Sweep: {preset_name}")
    print(f"{'='*70}")
    print(f"  {'CCG_T':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'Clust':>5}  {'Spikes':>7}  {'Time':>6}")
    print(f"  {'-'*55}")

    threshold = {"easy": 5.0, "medium": 4.0, "hard": 3.5}.get(preset_name, 5.0)
    best_acc = 0
    best_t = 0
    for t in thresholds:
        m = run_single(rec, params, threshold=threshold, ccg_merge=True,
                       ccg_template_corr_threshold=t)
        tag = ""
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_t = t
            tag = " <--"
        print(f"  {t:>6.2f}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['n_clusters']:>5}  {m['n_spikes']:>7}  "
              f"{m['elapsed']:>5.2f}s{tag}")

    print(f"\n  Best: ccg_threshold={best_t:.2f}, accuracy={best_acc:.3f}")
    return best_t, best_acc


def final_comparison(seed=42):
    """Run all presets with default vs best config."""
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON: Default vs Optimized")
    print(f"{'='*70}")

    for preset_name in ["easy", "medium", "hard"]:
        rec, params = generate_data(preset_name, seed)

        # Default config
        threshold = {"easy": 5.0, "medium": 4.0, "hard": 3.5}[preset_name]
        default = run_single(rec, params, threshold=threshold)

        # Best config from sweeps (will be determined by running)
        # For now, try the most promising combinations
        optimized = run_single(rec, params, threshold=threshold,
                               ccg_merge=True, ccg_template_corr_threshold=0.4,
                               template_subtract_passes=2,
                               isi_split_threshold=0.05)

        print(f"\n  {preset_name:>8}  Default: acc={default['accuracy']:.3f}  "
              f"prec={default['precision']:.3f}  rec={default['recall']:.3f}  "
              f"clust={default['n_clusters']}  {default['elapsed']:.2f}s")
        print(f"  {' ':>8}  Optimized: acc={optimized['accuracy']:.3f}  "
              f"prec={optimized['precision']:.3f}  rec={optimized['recall']:.3f}  "
              f"clust={optimized['n_clusters']}  {optimized['elapsed']:.2f}s")
        delta = optimized["accuracy"] - default["accuracy"]
        print(f"  {' ':>8}  Delta: {delta:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep benchmark")
    parser.add_argument("--preset", choices=["easy", "medium", "hard", "all"],
                        default="all")
    parser.add_argument("--threshold-only", action="store_true",
                        help="Only sweep detection threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    presets = ["easy", "medium", "hard"] if args.preset == "all" else [args.preset]

    print("Zerostone Parameter Sweep Benchmark")
    print("=" * 70)

    all_best = {}

    for preset in presets:
        best_thresh, _ = sweep_threshold(preset, args.seed)
        all_best[preset] = {"threshold": best_thresh}

        if not args.threshold_only:
            sweep_config(preset, args.seed)
            sweep_ccg_threshold(preset, args.seed)

    if not args.threshold_only and len(presets) == 3:
        final_comparison(args.seed)


if __name__ == "__main__":
    main()
