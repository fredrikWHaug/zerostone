"""SpikeInterface hybrid validation benchmark for Zerostone.

Uses SpikeInterface's generate_ground_truth_recording() to create realistic
synthetic recordings with known ground truth, runs Zerostone, and computes
standardized accuracy metrics via SpikeInterface's comparison module.

This provides an independent validation path separate from our own synthetic
generator, using SpikeInterface's template library and noise models.

Usage:
    python benchmarks/hybrid_benchmark.py
    python benchmarks/hybrid_benchmark.py --num-channels 32 --num-units 5
    python benchmarks/hybrid_benchmark.py --num-channels 32 --num-units 10

Requires: spikeinterface >= 0.100, zpybci
"""

import argparse
import time

import numpy as np

try:
    import spikeinterface.core as sc
    import spikeinterface.comparison as scmp
    import spikeinterface.generation as sg
except ImportError:
    raise ImportError(
        "spikeinterface is required. Install with: pip install spikeinterface"
    )

from zpybci.spikeinterface import run_zerostone


def run_benchmark(num_channels, num_units, duration_s, seed, threshold):
    """Run a single benchmark configuration.

    Parameters
    ----------
    num_channels : int
        Number of recording channels.
    num_units : int
        Number of ground-truth units.
    duration_s : float
        Recording duration in seconds.
    seed : int
        Random seed.
    threshold : float
        Detection threshold in MAD units.

    Returns
    -------
    dict
        Benchmark results with accuracy metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"  SpikeInterface Hybrid Benchmark")
    print(f"  {num_channels} channels, {num_units} units, {duration_s}s, seed={seed}")
    print(f"{'=' * 60}")

    # Generate ground-truth recording
    print("  Generating recording via SpikeInterface...", end=" ", flush=True)
    t0 = time.time()
    recording, sorting_gt = sg.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=30000.0,
        num_channels=num_channels,
        num_units=num_units,
        generate_sorting_kwargs={
            "firing_rates": 5.0,
            "refractory_period_ms": 1.5,
        },
        noise_kwargs={
            "noise_levels": 5.0,
            "strategy": "on_the_fly",
        },
        generate_probe_kwargs={
            "num_columns": 1,
            "xpitch": 25,
            "ypitch": 25,
            "contact_shapes": "circle",
            "contact_shape_params": {"radius": 6},
        },
        seed=seed,
    )
    t_gen = time.time() - t0
    gt_units = sorting_gt.get_unit_ids()
    gt_total = sum(len(sorting_gt.get_unit_spike_train(u)) for u in gt_units)
    print(f"done ({t_gen:.1f}s, {gt_total} GT spikes, {len(gt_units)} units)")

    # Run Zerostone
    print("  Running Zerostone...", end=" ", flush=True)
    t0 = time.time()
    sorting_zs = run_zerostone(
        recording,
        threshold=threshold,
        align_half_window=15,
        pre_samples=20,
    )
    t_sort = time.time() - t0
    zs_units = sorting_zs.get_unit_ids()
    zs_total = sum(len(sorting_zs.get_unit_spike_train(u)) for u in zs_units)
    print(f"done ({t_sort:.1f}s, {zs_total} spikes, {len(zs_units)} clusters)")

    # Compare using SpikeInterface
    print("  Computing comparison metrics...", end=" ", flush=True)
    t0 = time.time()
    comparison = scmp.compare_sorter_to_ground_truth(
        sorting_gt,
        sorting_zs,
        exhaustive_gt=True,
        delta_time=0.4,  # 0.4 ms tolerance (SpikeInterface default at 30kHz = 12 samples)
    )
    t_cmp = time.time() - t0
    print(f"done ({t_cmp:.1f}s)")

    # Extract per-unit metrics
    perf = comparison.get_performance()
    print(f"\n  GT units: {len(gt_units)}  |  Sorted clusters: {len(zs_units)}  |  Sort time: {t_sort:.2f}s\n")

    print(f"  {'Unit':>6s}  {'Matched':>8s}  {'Accuracy':>8s}  {'Precision':>9s}  {'Recall':>8s}")
    print(f"  {'-' * 50}")

    accuracies = []
    precisions = []
    recalls = []

    # Get Hungarian matching from agreement scores
    agreement = comparison.get_ordered_agreement_scores()

    for gt_id in gt_units:
        gt_str = str(gt_id)
        acc = perf.loc[gt_id, "accuracy"] if gt_id in perf.index else 0.0
        prec = perf.loc[gt_id, "precision"] if gt_id in perf.index else 0.0
        rec = perf.loc[gt_id, "recall"] if gt_id in perf.index else 0.0

        # Find best matching sorted unit from agreement scores
        if gt_id in agreement.index and len(agreement.columns) > 0:
            best_col = agreement.loc[gt_id].idxmax()
            best_score = agreement.loc[gt_id, best_col]
            matched_str = str(best_col) if best_score > 0 else "---"
        else:
            matched_str = "---"

        print(f"  {gt_str:>6s}  {matched_str:>8s}  {acc:>8.3f}  {prec:>9.3f}  {rec:>8.3f}")
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)

    mean_acc = np.mean(accuracies) if accuracies else 0.0
    mean_prec = np.mean(precisions) if precisions else 0.0
    mean_rec = np.mean(recalls) if recalls else 0.0

    print(f"  {'-' * 50}")
    print(f"  {'MEAN':>6s}  {'':>8s}  {mean_acc:>8.3f}  {mean_prec:>9.3f}  {mean_rec:>8.3f}")

    return {
        "num_channels": num_channels,
        "num_units": num_units,
        "duration_s": duration_s,
        "gt_total": gt_total,
        "sorted_total": zs_total,
        "n_clusters": len(zs_units),
        "mean_accuracy": float(mean_acc),
        "mean_precision": float(mean_prec),
        "mean_recall": float(mean_rec),
        "sort_time": t_sort,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SpikeInterface hybrid benchmark for Zerostone"
    )
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--num-units", type=int, default=5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run easy (5 units), medium (10 units), hard (20 units)",
    )
    args = parser.parse_args()

    results = []

    if args.all:
        configs = [
            {"num_channels": 32, "num_units": 5, "threshold": 5.0},
            {"num_channels": 32, "num_units": 10, "threshold": 4.0},
        ]
        for cfg in configs:
            r = run_benchmark(
                num_channels=cfg["num_channels"],
                num_units=cfg["num_units"],
                duration_s=args.duration,
                seed=args.seed,
                threshold=cfg["threshold"],
            )
            results.append(r)

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY (SpikeInterface Ground Truth)")
        print(f"{'=' * 60}")
        print(f"  {'Config':>10s}  {'Units':>5s}  {'GT':>6s}  {'Sorted':>7s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'Time':>6s}")
        print(f"  {'-' * 60}")
        for r in results:
            label = f"{r['num_channels']}ch/{r['num_units']}u"
            print(
                f"  {label:>10s}  {r['num_units']:>5d}  {r['gt_total']:>6d}  "
                f"{r['sorted_total']:>7d}  {r['mean_accuracy']:>6.3f}  "
                f"{r['mean_precision']:>6.3f}  {r['mean_recall']:>6.3f}  "
                f"{r['sort_time']:>5.1f}s"
            )
    else:
        run_benchmark(
            num_channels=args.num_channels,
            num_units=args.num_units,
            duration_s=args.duration,
            seed=args.seed,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
