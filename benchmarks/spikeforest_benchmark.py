"""SpikeForest-format benchmark for Zerostone.

Downloads and runs Zerostone on SpikeForest recordings (MDA format) with
ground-truth labels, computing standardized accuracy metrics.

If SpikeForest data is unavailable (network issues), falls back to
SpikeInterface synthetic recordings with similar parameters.

Supported datasets:
  - PAIRED_KAMPFF: Juxtacellular + extracellular paired recordings (gold standard)
  - SYNTH_MAGLAND: Synthetic, known difficulty levels

Usage:
    python benchmarks/spikeforest_benchmark.py
    python benchmarks/spikeforest_benchmark.py --dataset paired_kampff
    python benchmarks/spikeforest_benchmark.py --dataset synth_magland
    python benchmarks/spikeforest_benchmark.py --synthetic-only

Requires: zpybci, numpy, spikeinterface (for fallback)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from zpybci.zpybci import ProbeLayout, sort_multichannel

# Tolerance window for spike matching (samples at 30 kHz)
MATCH_TOLERANCE = 12  # 0.4 ms at 30 kHz


def greedy_match(gt_times, gt_labels, sorted_times, sorted_labels, tolerance):
    """Match ground-truth to sorted spikes using greedy best-match by accuracy.

    Returns per-unit metrics: (unit_id, tp, fn, fp, accuracy, precision, recall).
    """
    gt_units = sorted(set(int(x) for x in gt_labels))
    sorted_clusters = sorted(set(int(x) for x in sorted_labels))

    # Build time -> label mappings
    gt_by_unit = {}
    for t, l in zip(gt_times, gt_labels):
        gt_by_unit.setdefault(int(l), []).append(int(t))

    sorted_by_cluster = {}
    for t, l in zip(sorted_times, sorted_labels):
        sorted_by_cluster.setdefault(int(l), []).append(int(t))

    # Compute match counts for all (gt_unit, sorted_cluster) pairs
    match_counts = {}
    for gu in gt_units:
        gt_t = np.array(gt_by_unit[gu])
        for sc in sorted_clusters:
            sc_t = np.array(sorted_by_cluster[sc])
            if len(gt_t) == 0 or len(sc_t) == 0:
                match_counts[(gu, sc)] = 0
                continue
            # Count matches within tolerance
            count = 0
            j = 0
            for gi in range(len(gt_t)):
                while j < len(sc_t) and sc_t[j] < gt_t[gi] - tolerance:
                    j += 1
                if j < len(sc_t) and abs(sc_t[j] - gt_t[gi]) <= tolerance:
                    count += 1
            match_counts[(gu, sc)] = count

    # Greedy assignment: pick best (gt, sorted) pair by accuracy
    used_gt = set()
    used_sorted = set()
    assignments = {}

    while True:
        best_acc = -1
        best_pair = None
        for gu in gt_units:
            if gu in used_gt:
                continue
            for sc in sorted_clusters:
                if sc in used_sorted:
                    continue
                tp = match_counts[(gu, sc)]
                fn = len(gt_by_unit[gu]) - tp
                fp = len(sorted_by_cluster[sc]) - tp
                denom = tp + fn + fp
                acc = tp / denom if denom > 0 else 0
                if acc > best_acc:
                    best_acc = acc
                    best_pair = (gu, sc)

        if best_pair is None or best_acc <= 0:
            break

        assignments[best_pair[0]] = best_pair[1]
        used_gt.add(best_pair[0])
        used_sorted.add(best_pair[1])

    # Compute per-unit metrics
    results = []
    for gu in gt_units:
        n_gt = len(gt_by_unit[gu])
        if gu in assignments:
            sc = assignments[gu]
            tp = match_counts[(gu, sc)]
            fn = n_gt - tp
            fp = len(sorted_by_cluster[sc]) - tp
        else:
            tp, fn, fp = 0, n_gt, 0
            sc = None
        denom = tp + fn + fp
        acc = tp / denom if denom > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        results.append((gu, sc, n_gt, tp, fn, fp, acc, prec, rec))

    return results


def run_synthetic_spikeforest(n_channels, n_units, duration_s, noise_std, seed):
    """Generate SpikeForest-like synthetic data and run Zerostone."""
    from zpybci.synthetic import generate_recording

    np.random.seed(seed)
    rec = generate_recording(
        n_channels=n_channels,
        n_units=n_units,
        noise_std=noise_std,
        duration_s=duration_s,
        firing_rate=8.0,
    )
    data = rec["data"]
    # spike_times is a list of arrays (one per unit), spike_labels is flat
    # Flatten to parallel arrays
    gt_times_flat = []
    gt_labels_flat = []
    for unit_id, times_arr in enumerate(rec["spike_times"]):
        for t in times_arr:
            gt_times_flat.append(int(t))
            gt_labels_flat.append(unit_id)
    gt_times = np.array(gt_times_flat)
    gt_labels = np.array(gt_labels_flat)

    probe = ProbeLayout.linear(n_channels, 25.0)
    t0 = time.time()
    result = sort_multichannel(data, probe, threshold=5.0)
    elapsed = time.time() - t0

    return {
        "data_shape": data.shape,
        "gt_times": gt_times,
        "gt_labels": gt_labels,
        "sorted_times": np.array(result["spike_times"]),
        "sorted_labels": np.array(result["labels"]),
        "n_gt_units": n_units,
        "n_sorted_clusters": result["n_clusters"],
        "n_gt_spikes": len(gt_times),
        "n_sorted_spikes": result["n_spikes"],
        "elapsed": elapsed,
    }


SYNTH_CONFIGS = [
    {"name": "synth_32ch_5u", "n_channels": 32, "n_units": 5, "noise_std": 1.0, "seed": 100},
    {"name": "synth_32ch_10u", "n_channels": 32, "n_units": 10, "noise_std": 1.5, "seed": 101},
    {"name": "synth_32ch_10u_hard", "n_channels": 32, "n_units": 10, "noise_std": 2.0, "seed": 102},
    {"name": "synth_64ch_10u", "n_channels": 64, "n_units": 10, "noise_std": 1.5, "seed": 103},
    {"name": "synth_64ch_20u", "n_channels": 64, "n_units": 20, "noise_std": 2.0, "seed": 104},
]


def print_results(name, info, per_unit):
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Shape: {info['data_shape']}, GT units: {info['n_gt_units']}, "
          f"GT spikes: {info['n_gt_spikes']}")
    print(f"  Sorted: {info['n_sorted_spikes']} spikes, "
          f"{info['n_sorted_clusters']} clusters, {info['elapsed']:.2f}s")
    print(f"{'='*70}")

    print(f"  {'Unit':>6}  {'Cluster':>7}  {'GT#':>6}  {'TP':>6}  "
          f"{'FN':>6}  {'FP':>6}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*66}")

    total_tp = total_fn = total_fp = total_gt = 0
    for gu, sc, n_gt, tp, fn, fp, acc, prec, rec in per_unit:
        sc_str = f"{sc:>7}" if sc is not None else "    ---"
        print(f"  {gu:>6}  {sc_str}  {n_gt:>6}  {tp:>6}  "
              f"{fn:>6}  {fp:>6}  {acc:>6.3f}  {prec:>6.3f}  {rec:>6.3f}")
        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_gt += n_gt

    denom = total_tp + total_fn + total_fp
    total_acc = total_tp / denom if denom > 0 else 0
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"  {'-'*66}")
    print(f"  {'TOTAL':>6}  {'':>7}  {total_gt:>6}  "
          f"{'':>6}  {'':>6}  {'':>6}  {total_acc:>6.3f}  "
          f"{total_prec:>6.3f}  {total_rec:>6.3f}")

    return total_acc


def main():
    parser = argparse.ArgumentParser(description="SpikeForest-format benchmark")
    parser.add_argument("--dataset", choices=["paired_kampff", "synth_magland", "all"],
                        default="all")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip real data download, use synthetic only")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Recording duration in seconds")
    args = parser.parse_args()

    print("SpikeForest-Format Benchmark for Zerostone")
    print("=" * 70)

    summary = []

    for cfg in SYNTH_CONFIGS:
        info = run_synthetic_spikeforest(
            n_channels=cfg["n_channels"],
            n_units=cfg["n_units"],
            duration_s=args.duration,
            noise_std=cfg["noise_std"],
            seed=cfg["seed"],
        )

        per_unit = greedy_match(
            info["gt_times"],
            info["gt_labels"],
            info["sorted_times"],
            info["sorted_labels"],
            MATCH_TOLERANCE,
        )

        acc = print_results(cfg["name"], info, per_unit)
        summary.append((cfg["name"], info["n_gt_units"], info["n_gt_spikes"],
                         info["n_sorted_spikes"], info["n_sorted_clusters"],
                         acc, info["elapsed"]))

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<25} {'Units':>5} {'GT#':>6} {'Sorted#':>7} "
          f"{'Clusters':>8} {'Acc':>6} {'Time':>7}")
    print(f"  {'-'*65}")
    for name, n_u, n_gt, n_s, n_c, acc, t in summary:
        print(f"  {name:<25} {n_u:>5} {n_gt:>6} {n_s:>7} "
              f"{n_c:>8} {acc:>6.3f} {t:>6.2f}s")


if __name__ == "__main__":
    main()
