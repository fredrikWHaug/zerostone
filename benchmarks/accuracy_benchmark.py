"""Spike sorting accuracy benchmark for Zerostone.

Generates synthetic multi-channel recordings with known ground truth,
runs the Zerostone sorter, and computes accuracy metrics following the
SpikeInterface/SpikeForest methodology:

    accuracy  = TP / (TP + FN + FP)
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)

Ground-truth-to-sorted matching uses greedy best-match by accuracy,
with a configurable tolerance window (default 0.4 ms = 12 samples at
30 kHz), consistent with SpikeInterface defaults.

Spike times and cluster labels are extracted directly from the
sort_multichannel() output, which exposes them as "spike_times" and
"labels" in the result dict.

Usage:
    python benchmarks/accuracy_benchmark.py
    python benchmarks/accuracy_benchmark.py --difficulty easy
    python benchmarks/accuracy_benchmark.py --difficulty medium
    python benchmarks/accuracy_benchmark.py --difficulty hard
    python benchmarks/accuracy_benchmark.py --all
"""

import argparse
import time

import numpy as np

from zpybci.zpybci import (
    ProbeLayout,
    sort_multichannel,
)
from zpybci.synthetic import generate_recording

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
PRESETS = {
    "easy": {
        "n_channels": 32,
        "n_units": 5,
        "noise_std": 1.0,
        "duration_s": 60.0,
        "firing_rate": 5.0,
        "threshold": 5.0,
        "cluster_threshold": 8.0,
    },
    "medium": {
        "n_channels": 32,
        "n_units": 10,
        "noise_std": 1.5,
        "duration_s": 60.0,
        "firing_rate": 8.0,
        "threshold": 4.0,
        "cluster_threshold": 8.0,
        "min_cluster_snr": 1.5,
        "matched_filter_detect": True,
        "matched_filter_threshold": 4.2,
    },
    "hard": {
        "n_channels": 64,
        "n_units": 20,
        "noise_std": 2.0,
        "duration_s": 60.0,
        "firing_rate": 10.0,
        "threshold": 5.0,
        "matched_filter_detect": True,
        "matched_filter_threshold": 4.0,
    },
}

# Default tolerance: 0.67 ms at 30 kHz = 20 samples
# (accounts for whitening-induced peak shift between GT and detected times)
DEFAULT_TOLERANCE = 20


# ---------------------------------------------------------------------------
# Spike matching
# ---------------------------------------------------------------------------
def match_spikes(gt_times, sorted_times, tolerance=DEFAULT_TOLERANCE):
    """Match ground-truth spikes to sorted spikes within a tolerance window.

    Uses greedy nearest-neighbor matching: for each GT spike (in temporal
    order), find the nearest unmatched sorted spike within *tolerance*
    samples. This mirrors the SpikeInterface ``match_event_count`` logic.

    Parameters
    ----------
    gt_times : np.ndarray
        Sorted 1-D array of ground-truth spike sample indices.
    sorted_times : np.ndarray
        Sorted 1-D array of sorted spike sample indices.
    tolerance : int
        Maximum sample offset for a match (default 12, i.e. 0.4 ms at 30 kHz).

    Returns
    -------
    tp : int
        True positives (matched pairs).
    fn : int
        False negatives (GT spikes with no match).
    fp : int
        False positives (sorted spikes with no match).
    """
    if len(gt_times) == 0:
        return 0, 0, len(sorted_times)
    if len(sorted_times) == 0:
        return 0, len(gt_times), 0

    matched_sorted = set()
    tp = 0
    search_start = 0  # sliding window optimization

    for gt_t in gt_times:
        best_idx = -1
        best_dist = tolerance + 1

        # Advance search_start past already-impossible candidates
        while search_start < len(sorted_times) and sorted_times[search_start] < gt_t - tolerance:
            search_start += 1

        for j in range(search_start, len(sorted_times)):
            st = sorted_times[j]
            if st > gt_t + tolerance:
                break
            dist = abs(int(st) - int(gt_t))
            if dist < best_dist and j not in matched_sorted:
                best_dist = dist
                best_idx = j

        if best_idx >= 0:
            tp += 1
            matched_sorted.add(best_idx)

    fn = len(gt_times) - tp
    fp = len(sorted_times) - tp
    return tp, fn, fp


# ---------------------------------------------------------------------------
# Per-unit accuracy computation
# ---------------------------------------------------------------------------
def compute_accuracy(gt_spike_times, gt_labels, sorted_times, sorted_labels,
                     n_gt_units, tolerance=DEFAULT_TOLERANCE):
    """Compute per-unit and overall accuracy metrics.

    For each ground-truth unit, finds the best-matching sorted cluster
    using greedy assignment by accuracy (Hungarian-lite). Each sorted
    cluster may be assigned to at most one GT unit.

    Parameters
    ----------
    gt_spike_times : np.ndarray
        All GT spike sample indices, sorted.
    gt_labels : np.ndarray
        Cluster label for each GT spike.
    sorted_times : np.ndarray
        All detected spike sample indices from the sorter, sorted.
    sorted_labels : np.ndarray
        Cluster label for each detected spike.
    n_gt_units : int
        Number of ground-truth units.
    tolerance : int
        Sample tolerance for matching.

    Returns
    -------
    dict with keys:
        per_unit : list of dict
            Per-GT-unit metrics (accuracy, precision, recall, tp, fn, fp,
            matched_cluster).
        overall_accuracy : float
        overall_precision : float
        overall_recall : float
        n_gt_spikes : int
        n_sorted_spikes : int
    """
    # Group GT spikes by unit
    gt_by_unit = {}
    for u in range(n_gt_units):
        mask = gt_labels == u
        gt_by_unit[u] = np.sort(gt_spike_times[mask])

    # Group sorted spikes by cluster
    sorted_clusters = {}
    if len(sorted_labels) > 0:
        for cl in np.unique(sorted_labels):
            mask = sorted_labels == cl
            sorted_clusters[int(cl)] = np.sort(sorted_times[mask])

    # Build agreement matrix: rows = GT units, cols = sorted clusters
    cluster_ids = sorted(sorted_clusters.keys())
    agreement = np.zeros((n_gt_units, len(cluster_ids)), dtype=np.float64)
    match_details = {}

    for i, u in enumerate(range(n_gt_units)):
        for j, cl in enumerate(cluster_ids):
            tp, fn, fp = match_spikes(gt_by_unit[u], sorted_clusters[cl], tolerance)
            denom = tp + fn + fp
            acc = tp / denom if denom > 0 else 0.0
            agreement[i, j] = acc
            match_details[(i, j)] = (tp, fn, fp)

    # Greedy best-match assignment (sorted cluster used at most once)
    assigned_clusters = set()
    per_unit = []
    total_tp, total_fn, total_fp = 0, 0, 0

    for _ in range(n_gt_units):
        # Find (gt_unit, cluster) pair with highest unassigned accuracy
        best_acc = -1.0
        best_i, best_j = -1, -1
        for i in range(n_gt_units):
            if any(p["gt_unit"] == i for p in per_unit):
                continue
            for j in range(len(cluster_ids)):
                if j in assigned_clusters:
                    continue
                if agreement[i, j] > best_acc:
                    best_acc = agreement[i, j]
                    best_i, best_j = i, j

        if best_i < 0:
            break

        tp, fn, fp = match_details[(best_i, best_j)]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        per_unit.append({
            "gt_unit": best_i,
            "matched_cluster": cluster_ids[best_j] if best_j >= 0 else -1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "n_gt_spikes": len(gt_by_unit[best_i]),
        })
        assigned_clusters.add(best_j)
        total_tp += tp
        total_fn += fn
        total_fp += fp

    # Units with no match
    for i in range(n_gt_units):
        if not any(p["gt_unit"] == i for p in per_unit):
            n_gt = len(gt_by_unit[i])
            per_unit.append({
                "gt_unit": i,
                "matched_cluster": -1,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fn": n_gt,
                "fp": 0,
                "n_gt_spikes": n_gt,
            })
            total_fn += n_gt

    per_unit.sort(key=lambda p: p["gt_unit"])

    overall_denom = total_tp + total_fn + total_fp
    overall_acc = total_tp / overall_denom if overall_denom > 0 else 0.0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "per_unit": per_unit,
        "overall_accuracy": overall_acc,
        "overall_precision": overall_prec,
        "overall_recall": overall_rec,
        "n_gt_spikes": int(np.sum([len(gt_by_unit[u]) for u in range(n_gt_units)])),
        "n_sorted_spikes": len(sorted_times),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(preset_name, seed=42, tolerance=DEFAULT_TOLERANCE, verbose=True,
                  detection_mode="amplitude", sneo_smooth_window=3,
                  ccg_merge=False):
    """Run a single benchmark at the given difficulty level.

    Parameters
    ----------
    preset_name : str
        Key into PRESETS dict ("easy", "medium", "hard").
    seed : int
        Random seed for reproducibility.
    tolerance : int
        Sample tolerance for spike matching.
    verbose : bool
        Print progress messages.
    detection_mode : str
        Detection mode: "amplitude", "neo", or "sneo".
    sneo_smooth_window : int
        SNEO smoothing window size.
    ccg_merge : bool
        Enable CCG-based cluster merging.

    Returns
    -------
    dict with keys: preset, params, sort_result, metrics, elapsed_s
    """
    params = PRESETS[preset_name]
    n_ch = params["n_channels"]

    mode_str = detection_mode
    if ccg_merge:
        mode_str += "+ccg"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {preset_name} (detection={mode_str})")
        print(f"  {n_ch} channels, {params['n_units']} units, "
              f"noise_std={params['noise_std']}, "
              f"firing_rate={params['firing_rate']} Hz, "
              f"{params['duration_s']}s")
        print(f"{'='*60}")

    # 1. Generate recording
    if verbose:
        print("  Generating synthetic recording...", end=" ", flush=True)
    t0 = time.perf_counter()
    rec = generate_recording(
        n_channels=n_ch,
        duration_s=params["duration_s"],
        sampling_rate=30000.0,
        n_units=params["n_units"],
        firing_rate=params["firing_rate"],
        noise_std=params["noise_std"],
        seed=seed,
    )
    t_gen = time.perf_counter() - t0
    if verbose:
        n_gt = len(rec["all_spike_times"])
        print(f"done ({t_gen:.1f}s, {n_gt} GT spikes)")

    data = rec["data"]  # (n_samples, n_channels)

    # 2. Build probe
    probe = ProbeLayout.linear(n_ch, 25.0)

    # 3. Run sort_multichannel
    if verbose:
        print("  Running sort_multichannel...", end=" ", flush=True)
    t0 = time.perf_counter()
    sort_result = sort_multichannel(
        data, probe,
        threshold=params.get("threshold", 5.0),
        refractory=15,
        detection_mode=detection_mode,
        sneo_smooth_window=sneo_smooth_window,
        ccg_merge=ccg_merge,
        cluster_threshold=params.get("cluster_threshold", 5.0),
        min_cluster_snr=params.get("min_cluster_snr", 2.5),
        template_subtract_passes=1,
        gmm_refine=params.get("gmm_refine", False),
        gmm_max_iter=params.get("gmm_max_iter", 10),
        matched_filter_detect=params.get("matched_filter_detect", False),
        matched_filter_threshold=params.get("matched_filter_threshold", 4.0),
    )
    t_sort = time.perf_counter() - t0
    if verbose:
        print(f"done ({t_sort:.1f}s, {sort_result['n_spikes']} spikes, "
              f"{sort_result['n_clusters']} clusters)")

    # 4. Extract spike times from sorter output
    sorted_times = np.array(sort_result["spike_times"], dtype=np.int64)
    sort_labels = np.array(sort_result["labels"], dtype=np.int64)
    n_sort = sort_result["n_spikes"]

    if n_sort == 0:
        if verbose:
            print("  WARNING: No spikes detected. Skipping metrics.")
        return {
            "preset": preset_name,
            "params": params,
            "sort_result": sort_result,
            "metrics": None,
            "elapsed_s": t_sort,
        }

    # 5. Compute accuracy metrics
    if verbose:
        print("  Computing accuracy metrics...", end=" ", flush=True)
    metrics = compute_accuracy(
        rec["all_spike_times"],
        rec["spike_labels"],
        sorted_times,
        sort_labels,
        rec["n_units"],
        tolerance=tolerance,
    )
    if verbose:
        print("done")

    return {
        "preset": preset_name,
        "params": params,
        "sort_result": sort_result,
        "metrics": metrics,
        "elapsed_s": t_sort,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_results(result):
    """Print a formatted results table for a single benchmark run."""
    metrics = result["metrics"]
    if metrics is None:
        print("  No metrics available (no spikes detected).")
        return

    print(f"\n  GT spikes: {metrics['n_gt_spikes']}  |  "
          f"Sorted spikes: {metrics['n_sorted_spikes']}  |  "
          f"Sort time: {result['elapsed_s']:.2f}s")

    # Per-unit table
    print()
    header = (f"  {'Unit':>5} {'Cluster':>8} {'GT#':>6} "
              f"{'TP':>6} {'FN':>6} {'FP':>6} "
              f"{'Acc':>7} {'Prec':>7} {'Rec':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for pu in metrics["per_unit"]:
        cl_str = str(pu["matched_cluster"]) if pu["matched_cluster"] >= 0 else "---"
        print(f"  {pu['gt_unit']:>5} {cl_str:>8} {pu['n_gt_spikes']:>6} "
              f"{pu['tp']:>6} {pu['fn']:>6} {pu['fp']:>6} "
              f"{pu['accuracy']:>7.3f} {pu['precision']:>7.3f} {pu['recall']:>7.3f}")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':>5} {'':>8} {metrics['n_gt_spikes']:>6} "
          f"{'':>6} {'':>6} {'':>6} "
          f"{metrics['overall_accuracy']:>7.3f} "
          f"{metrics['overall_precision']:>7.3f} "
          f"{metrics['overall_recall']:>7.3f}")


def print_summary(results):
    """Print a compact summary table across all difficulty levels."""
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Preset':<10} {'Units':>5} {'GT#':>7} {'Sorted#':>8} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Time':>7}")
    print("  " + "-" * 56)

    for r in results:
        m = r["metrics"]
        if m is None:
            print(f"  {r['preset']:<10} {r['params']['n_units']:>5} "
                  f"{'---':>7} {'---':>8} {'---':>7} {'---':>7} {'---':>7} "
                  f"{r['elapsed_s']:>6.2f}s")
        else:
            print(f"  {r['preset']:<10} {r['params']['n_units']:>5} "
                  f"{m['n_gt_spikes']:>7} {m['n_sorted_spikes']:>8} "
                  f"{m['overall_accuracy']:>7.3f} "
                  f"{m['overall_precision']:>7.3f} "
                  f"{m['overall_recall']:>7.3f} "
                  f"{r['elapsed_s']:>6.2f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spike sorting accuracy benchmark for Zerostone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--difficulty",
        choices=list(PRESETS.keys()),
        default=None,
        help="Run a single difficulty preset (default: easy).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all difficulty presets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Spike matching tolerance in samples (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "--detection-mode",
        choices=["amplitude", "neo", "sneo"],
        default="amplitude",
        help="Detection mode (default: amplitude).",
    )
    parser.add_argument(
        "--sneo-smooth-window",
        type=int,
        default=3,
        help="SNEO smoothing window (default: 3).",
    )
    parser.add_argument(
        "--ccg-merge",
        action="store_true",
        help="Enable CCG-based cluster merging.",
    )
    args = parser.parse_args()

    if args.all:
        presets_to_run = list(PRESETS.keys())
    elif args.difficulty:
        presets_to_run = [args.difficulty]
    else:
        presets_to_run = ["easy"]

    results = []
    for preset in presets_to_run:
        result = run_benchmark(
            preset, seed=args.seed, tolerance=args.tolerance,
            detection_mode=args.detection_mode,
            sneo_smooth_window=args.sneo_smooth_window,
            ccg_merge=args.ccg_merge,
        )
        print_results(result)
        results.append(result)

    if len(results) > 1:
        print_summary(results)

    print()


if __name__ == "__main__":
    main()
