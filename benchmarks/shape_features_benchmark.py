"""Benchmark: neighbor-channel MF scoring + spike half-width shape features.

Compares shape-feature defaults against the auto-CMR baseline on 3 difficulty levels.
Baseline (auto-CMR): 68.2% avg (easy=76.1%, medium=69.2%, hard=59.2%)
"""

import json
import time
from datetime import datetime

import numpy as np
import spikeinterface.core as si
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core import NumpySorting

import zpybci as zbci

TOLERANCE_MS = 0.4
_cache = {}


def gen(difficulty):
    if difficulty in _cache:
        return _cache[difficulty]
    cfgs = {
        "easy":   {"num_channels": 4,  "num_units": 3,  "duration": 30.0, "noise_levels": 3.0},
        "medium": {"num_channels": 16, "num_units": 8,  "duration": 30.0, "noise_levels": 5.0},
        "hard":   {"num_channels": 32, "num_units": 15, "duration": 30.0, "noise_levels": 8.0},
    }
    c = cfgs[difficulty]
    rec, gt = si.generate_ground_truth_recording(
        durations=[c["duration"]], sampling_frequency=30000.0,
        num_channels=c["num_channels"], num_units=c["num_units"],
        seed=42, noise_kwargs={"noise_levels": c["noise_levels"], "strategy": "on_the_fly"},
    )
    traces = rec.get_traces(return_in_uV=True).astype(np.float64)
    fs = rec.get_sampling_frequency()
    gt_trains = {}
    for uid in gt.get_unit_ids():
        gt_trains[uid] = np.sort(gt.get_unit_spike_train(uid))
    all_t = np.concatenate([gt_trains[u] for u in sorted(gt_trains)])
    all_l = np.concatenate([np.full(len(gt_trains[u]), u, dtype=np.int64)
                            for u in sorted(gt_trains)])
    idx = np.argsort(all_t)
    gt_sorting = NumpySorting.from_samples_and_labels(
        [all_t[idx]], [all_l[idx]], sampling_frequency=fs
    )
    _cache[difficulty] = (traces, fs, c["num_channels"], gt_trains, gt_sorting)
    return _cache[difficulty]


def run(difficulty, label, detail=False, **kwargs):
    traces, fs, n_ch, gt_trains, gt_sorting = gen(difficulty)
    data = traces.copy()
    probe = zbci.ProbeLayout.linear(n_ch, 25.0)
    defaults = dict(
        threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
        align_half_window=15, pre_samples=20, cluster_threshold=7.0,
        cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
        sneo_smooth_window=3, ccg_merge=False, matched_filter_detect=True,
        matched_filter_threshold=4.0, gmm_refine=False, gmm_max_iter=10,
        bandpass_low=0.0, bandpass_high=0.0, sample_rate=fs,
        common_median_ref=False, merge_dprime_threshold=2.0,
        refinement_iterations=0, use_amplitude_profile=False,
        amplitude_profile_neighbors=4,
    )
    defaults.update(kwargs)
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **defaults)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label:<60} acc=0.000  0 spk  {elapsed:.1f}s")
        return 0.0, {}
    spike_times = np.asarray(result["spike_times"][:n_spikes], dtype=np.int64)
    labels_out = np.asarray(result["labels"][:n_spikes], dtype=np.int64)
    sorting = NumpySorting.from_samples_and_labels(
        [spike_times], [labels_out], sampling_frequency=fs
    )
    comp = compare_sorter_to_ground_truth(
        gt_sorting, sorting, exhaustive_gt=True,
        delta_time=TOLERANCE_MS, match_mode="hungarian"
    )
    avg = comp.get_performance(method="pooled_with_average")
    acc = float(avg["accuracy"])
    prec = float(avg["precision"])
    rec_ = float(avg["recall"])
    wd = comp.count_well_detected_units(well_detected_score=0.8)
    print(f"  {label:<60} acc={acc:.3f}  prec={prec:.3f}  rec={rec_:.3f}  "
          f"wd={wd}/{n_gt}  {n_spikes:>5}spk  {n_cl:>2}cl  {elapsed:.1f}s")
    if detail:
        perf = comp.get_performance()
        for uid in perf.index:
            row = perf.loc[uid]
            gt_n = len(gt_trains.get(uid, []))
            print(f"    Unit {uid}: acc={float(row['accuracy']):.3f}  "
                  f"prec={float(row['precision']):.3f}  rec={float(row['recall']):.3f}  "
                  f"({gt_n} GT)")
    return acc, {"acc": acc, "prec": prec, "rec": rec_, "wd": wd, "n_gt": n_gt,
                 "n_spikes": n_spikes, "n_clusters": n_cl, "elapsed": elapsed}


def main():
    print("=" * 80)
    print("  Day 7 Benchmark: Neighbor-Channel MF Scoring + Spike Half-Width Features")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {}
    day6_baseline = {"easy": 0.761, "medium": 0.692, "hard": 0.592}

    for diff in ["easy", "medium", "hard"]:
        print(f"\n--- {diff.upper()} ---")
        gen(diff)

        # Day 6 baseline: no new features
        acc_d6, meta_d6 = run(
            diff, "Day6 baseline (no shape, no neighbor-MF)",
            use_shape_features=False,
            neighbor_mf_detect=False,
        )

        # Day 7 default: shape features only
        acc_shape, meta_shape = run(
            diff, "Day7 shape only (half-width feature)",
            use_shape_features=True,
            neighbor_mf_detect=False,
        )

        # Day 7 default: neighbor-MF only
        acc_nmf, meta_nmf = run(
            diff, "Day7 neighbor-MF only",
            use_shape_features=False,
            neighbor_mf_detect=True,
        )

        # Day 7 full default: both features
        acc_d7, meta_d7 = run(
            diff, "Day7 full default (shape + neighbor-MF)",
            detail=True,
            use_shape_features=True,
            neighbor_mf_detect=True,
        )

        results[diff] = {
            "day6_acc": acc_d6,
            "shape_only_acc": acc_shape,
            "nmf_only_acc": acc_nmf,
            "day7_acc": acc_d7,
            "day6_baseline_ref": day6_baseline[diff],
            "meta_d7": meta_d7,
        }
        delta = acc_d7 - acc_d6
        print(f"  >> {diff}: Day6={acc_d6:.3f}  Day7={acc_d7:.3f}  "
              f"delta={delta:+.3f}  (ref={day6_baseline[diff]:.3f})")

    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    avg_d6 = sum(results[d]["day6_acc"] for d in ["easy", "medium", "hard"]) / 3
    avg_d7 = sum(results[d]["day7_acc"] for d in ["easy", "medium", "hard"]) / 3
    print(f"  {'Preset':<10} {'Day6 Acc':>10} {'Day7 Acc':>10} {'Delta':>8} {'Ref Day6':>10}")
    print(f"  {'-'*50}")
    for diff in ["easy", "medium", "hard"]:
        r = results[diff]
        delta = r["day7_acc"] - r["day6_acc"]
        print(f"  {diff:<10} {r['day6_acc']:>10.3f} {r['day7_acc']:>10.3f} "
              f"{delta:>+8.3f} {r['day6_baseline_ref']:>10.3f}")
    print(f"  {'-'*50}")
    delta_avg = avg_d7 - avg_d6
    print(f"  {'AVERAGE':<10} {avg_d6:>10.3f} {avg_d7:>10.3f} {delta_avg:>+8.3f}")
    print(f"\n  Target: >= 0.690 avg. Got: {avg_d7:.3f}  {'PASS' if avg_d7 >= 0.690 else 'MISS'}")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"benchmarks/results/day7_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "date": ts,
            "day6_avg": avg_d6,
            "day7_avg": avg_d7,
            "target": 0.690,
            "results": {
                d: {k: v for k, v in r.items() if k != "meta_d7"}
                for d, r in results.items()
            },
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
