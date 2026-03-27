#!/usr/bin/env python3
"""Drift robustness benchmark for the streaming spike sorter.

Tests how well the StreamingSorter maintains unit identity as the probe
drifts relative to the tissue. Uses synthetic drifting recordings with
known ground truth.

Usage:
    python benchmarks/drift_benchmark.py [--drift-um DRIFT] [--duration SECS]
"""

import argparse
import time

import numpy as np

import zpybci as zbci
from zpybci.synthetic import generate_drifting_recording


def run_drift_benchmark(
    drift_um=50.0,
    duration_s=30.0,
    n_channels=32,
    n_units=5,
    segment_s=2.0,
    seed=42,
):
    """Run drift benchmark and return metrics.

    Parameters
    ----------
    drift_um : float
        Total probe drift in micrometers.
    duration_s : float
        Recording duration in seconds.
    n_channels : int
        Number of channels.
    n_units : int
        Number of ground-truth units.
    segment_s : float
        Segment length for streaming sorter.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Benchmark results including accuracy, drift tracking, etc.
    """
    print(f"Generating {duration_s}s {n_channels}ch recording with {drift_um}um drift...")
    rec = generate_drifting_recording(
        n_channels=n_channels,
        duration_s=duration_s,
        n_units=n_units,
        drift_um=drift_um,
        firing_rate=5.0,
        noise_std=1.0,
        seed=seed,
    )

    data = rec["data"]
    gt_times = rec["all_spike_times"]
    gt_labels = rec["spike_labels"]
    n_samples = data.shape[0]
    fs = rec["sampling_rate"]
    segment_samples = int(segment_s * fs)

    print(f"  {n_samples} samples, {len(gt_times)} ground-truth spikes")
    print(f"  Drift: 0 -> {drift_um}um over {duration_s}s")

    # Sort using the non-streaming pipeline first (baseline)
    probe = zbci.ProbeLayout.linear(n_channels, 25.0)
    t0 = time.time()
    result_static = zbci.sort_multichannel(
        data.copy(),
        probe,
        threshold=5.0,
        matched_filter_detect=True,
        matched_filter_threshold=4.0,
    )
    t_static = time.time() - t0

    # Sort using streaming sorter
    t0 = time.time()
    sorter = zbci.StreamingSorter(n_channels, decay=0.95)
    n_segments = (n_samples + segment_samples - 1) // segment_samples
    all_sorted_times = []
    all_sorted_labels = []

    for seg in range(n_segments):
        start = seg * segment_samples
        end = min(start + segment_samples, n_samples)
        segment_data = np.ascontiguousarray(data[start:end])

        result = sorter.feed(segment_data, probe)
        n_spikes = result["n_spikes"]
        if n_spikes > 0:
            times = np.array(result["spike_times"]) + start
            labels = np.array(result["labels"])
            all_sorted_times.append(times)
            all_sorted_labels.append(labels)

    t_stream = time.time() - t0

    if all_sorted_times:
        sorted_times = np.concatenate(all_sorted_times)
        sorted_labels = np.concatenate(all_sorted_labels)
    else:
        sorted_times = np.array([], dtype=np.int64)
        sorted_labels = np.array([], dtype=np.int64)

    # Match sorted spikes to ground truth (greedy nearest-neighbor)
    tolerance = 20  # samples (~0.67ms at 30kHz)

    def match_spikes(pred_times, pred_labels, gt_times, gt_labels, tol):
        """Greedy match predicted to ground-truth spikes."""
        if len(pred_times) == 0 or len(gt_times) == 0:
            return 0, len(gt_times), len(pred_times)

        # Build per-unit spike trains
        gt_units = np.unique(gt_labels)
        pred_units = np.unique(pred_labels)

        # For each predicted unit, find best matching GT unit
        best_accuracy = 0.0
        tp_total = 0
        fn_total = 0
        fp_total = 0

        # Hungarian matching: for each GT unit, find the predicted unit
        # that matches the most spikes
        for gu in gt_units:
            gt_mask = gt_labels == gu
            gt_t = gt_times[gt_mask]

            best_tp = 0
            best_pu = -1
            for pu in pred_units:
                pred_mask = pred_labels == pu
                pred_t = pred_times[pred_mask]

                # Count matches
                tp = 0
                pi = 0
                for gt in gt_t:
                    while pi < len(pred_t) and pred_t[pi] < gt - tol:
                        pi += 1
                    if pi < len(pred_t) and abs(pred_t[pi] - gt) <= tol:
                        tp += 1
                        pi += 1

                if tp > best_tp:
                    best_tp = tp
                    best_pu = pu

            tp_total += best_tp
            fn_total += len(gt_t) - best_tp

        fp_total = max(0, len(pred_times) - tp_total)
        accuracy = tp_total / max(1, tp_total + fn_total + fp_total)
        return tp_total, fn_total, fp_total, accuracy

    # Static pipeline metrics
    static_times = np.array(result_static["spike_times"][:result_static["n_spikes"]])
    static_labels = np.array(result_static["labels"][:result_static["n_spikes"]])
    s_tp, s_fn, s_fp, s_acc = match_spikes(static_times, static_labels, gt_times, gt_labels, tolerance)

    # Streaming pipeline metrics
    st_tp, st_fn, st_fp, st_acc = match_spikes(sorted_times, sorted_labels, gt_times, gt_labels, tolerance)

    # Drift tracking
    drift_rate = sorter.drift_rate
    drift_fitted = sorter.drift_fitted
    estimated_total_drift = drift_rate * n_samples if drift_fitted else 0.0

    print(f"\n  Static pipeline:    acc={s_acc:.3f}  spikes={len(static_times)}  "
          f"clusters={result_static['n_clusters']}  time={t_static:.2f}s")
    print(f"  Streaming pipeline: acc={st_acc:.3f}  spikes={len(sorted_times)}  "
          f"templates={sorter.n_templates}  time={t_stream:.2f}s")
    print(f"  Drift tracking:     fitted={drift_fitted}  "
          f"est_total={estimated_total_drift:.1f}um  actual={drift_um}um")
    if drift_fitted and drift_um > 0:
        tracking_error = abs(estimated_total_drift - drift_um)
        print(f"  Drift error:        {tracking_error:.1f}um")

    return {
        "drift_um": drift_um,
        "duration_s": duration_s,
        "n_units": n_units,
        "gt_spikes": len(gt_times),
        "static_accuracy": s_acc,
        "static_spikes": len(static_times),
        "static_clusters": result_static["n_clusters"],
        "static_time": t_static,
        "streaming_accuracy": st_acc,
        "streaming_spikes": len(sorted_times),
        "streaming_templates": sorter.n_templates,
        "streaming_time": t_stream,
        "drift_fitted": drift_fitted,
        "estimated_drift": estimated_total_drift,
        "drift_error": abs(estimated_total_drift - drift_um) if drift_fitted else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Drift robustness benchmark")
    parser.add_argument("--drift-um", type=float, default=50.0,
                        help="Total drift in micrometers (default: 50)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--channels", type=int, default=32,
                        help="Number of channels (default: 32)")
    parser.add_argument("--units", type=int, default=5,
                        help="Number of units (default: 5)")
    parser.add_argument("--segment", type=float, default=2.0,
                        help="Segment length in seconds (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Drift Robustness Benchmark")
    print("=" * 60)

    results = run_drift_benchmark(
        drift_um=args.drift_um,
        duration_s=args.duration,
        n_channels=args.channels,
        n_units=args.units,
        segment_s=args.segment,
        seed=args.seed,
    )

    # Also test with no drift (baseline)
    print("\n" + "-" * 60)
    print("No-drift baseline:")
    baseline = run_drift_benchmark(
        drift_um=0.0,
        duration_s=args.duration,
        n_channels=args.channels,
        n_units=args.units,
        segment_s=args.segment,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  No drift:   static={baseline['static_accuracy']:.3f}  "
          f"streaming={baseline['streaming_accuracy']:.3f}")
    print(f"  {args.drift_um}um drift: static={results['static_accuracy']:.3f}  "
          f"streaming={results['streaming_accuracy']:.3f}")
    if results["drift_fitted"]:
        print(f"  Drift tracking error: {results['drift_error']:.1f}um")


if __name__ == "__main__":
    main()
