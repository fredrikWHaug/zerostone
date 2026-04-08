"""Benchmark: refinement_iterations=1 with auto_refine guard.

Day 10 changes:
- refinement_iterations: 0 → 1 (feature-space k-means reassignment after full pipeline)
- auto_refine: true (skips refinement for C<8 to prevent over-reassignment on small probes)

Isolation ablation findings:
- refine=1 alone: easy -5.1% (C=4, over-reassignment into noise clusters)
                  medium +0.4%, hard +1.3%
- auto_refine skips refinement for C<8, preserving easy accuracy
- amplitude_profile: medium -10.1% (replaces shape features, loses discriminability)
  → dropped; auto_amplitude_profile not implemented

Baseline: Day 8 defaults (72.5% avg: easy=82.7%, medium=73.3%, hard=61.4%)
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

BASELINE = {"easy": 0.8265, "medium": 0.7334, "hard": 0.6137}


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
    # Day 10 defaults
    defaults = dict(
        threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
        align_half_window=15, pre_samples=20, cluster_threshold=7.0,
        cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
        sneo_smooth_window=3, matched_filter_detect=True,
        matched_filter_threshold=3.5, gmm_max_iter=10,
        bandpass_low=0.0, bandpass_high=0.0, sample_rate=fs,
        common_median_ref=False, merge_dprime_threshold=2.0,
        use_amplitude_profile=False, amplitude_profile_neighbors=4,
        ccg_merge=True, auto_cluster_threshold=True,
        svd_init=False, gmm_refine=False, min_cluster_snr=2.5,
        # Day 10 new defaults:
        refinement_iterations=1, auto_refine=True,
    )
    defaults.update(kwargs)
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **defaults)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label:<65} acc=0.000  0 spk  {elapsed:.1f}s")
        return 0.0, {}
    spike_times = np.asarray(result["spike_times"][:n_spikes], dtype=np.int64)
    labels_out = np.asarray(result["labels"][:n_spikes], dtype=np.int64)
    sorting = NumpySorting.from_samples_and_labels(
        [spike_times], [labels_out], sampling_frequency=fs
    )
    cmp = compare_sorter_to_ground_truth(
        gt_sorting, sorting, exhaustive_gt=True,
        delta_time=TOLERANCE_MS, match_mode="hungarian"
    )
    avg = cmp.get_performance(method="pooled_with_average")
    acc = float(avg["accuracy"])
    prec = float(avg["precision"])
    rec_ = float(avg["recall"])
    wd = cmp.count_well_detected_units(well_detected_score=0.8)
    delta = acc - BASELINE[difficulty]
    print(f"  {label:<65} acc={acc:.3f}  ({delta:+.3f})  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    per_unit_df = cmp.get_performance()
    if detail:
        for uid in per_unit_df.index:
            row = per_unit_df.loc[uid]
            print(f"    unit {uid}: acc={float(row['accuracy']):.3f}")
    return acc, {str(uid): float(per_unit_df.loc[uid]["accuracy"])
                 for uid in per_unit_df.index}


def main():
    results = {}

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        detail = (difficulty == "hard")

        # Day 10 defaults
        acc_new, per_unit = run(difficulty, "Day 10 defaults (refine=1, auto_refine=T)", detail=detail)

        # Regression: Day 8 defaults (refinement_iterations=0)
        acc_day8, _ = run(difficulty, "Day 8 regression (refine=0, auto_refine=F)",
                          refinement_iterations=0, auto_refine=False)

        # Ablation: auto_refine=False (refinement always runs, including C=4)
        acc_no_guard, _ = run(difficulty, "ablation: auto_refine=False (no guard)",
                              auto_refine=False)

        delta = acc_new - BASELINE[difficulty]
        print(f"  Day8→Day10: {BASELINE[difficulty]:.3f} → {acc_new:.3f}  ({delta:+.3f})")
        results[difficulty] = {
            "baseline_acc": BASELINE[difficulty],
            "day10_acc": acc_new,
            "day8_regression_acc": acc_day8,
            "ablation_no_guard_acc": acc_no_guard,
            "delta": delta,
            "per_unit": {str(k): float(v) for k, v in per_unit.items()},
        }

    easy_acc = results["easy"]["day10_acc"]
    med_acc = results["medium"]["day10_acc"]
    hard_acc = results["hard"]["day10_acc"]
    avg = (easy_acc + med_acc + hard_acc) / 3.0
    base_avg = sum(BASELINE.values()) / 3.0

    print(f"\n{'='*70}")
    print(f"Day 8 avg:   {base_avg:.3f}  (easy={BASELINE['easy']:.3f}  medium={BASELINE['medium']:.3f}  hard={BASELINE['hard']:.3f})")
    print(f"Day 10 avg:  {avg:.3f}  (easy={easy_acc:.3f}  medium={med_acc:.3f}  hard={hard_acc:.3f})  ({avg - base_avg:+.3f})")
    print(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "date": ts,
        "day8_avg": base_avg,
        "day10_avg": avg,
        "target": 0.730,
        "results": results,
    }
    out_path = f"benchmarks/results/day10_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
