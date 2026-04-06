"""Benchmark: channel-adaptive cluster threshold + CCG merge default.

Features validated:
- auto_cluster_threshold=True: scales cluster threshold by sqrt(8/C) for C < 8
  channels, closing the easy-recording over-splitting gap
- ccg_merge=True: CCG-based merge now enabled by default

Baseline (shape features): 70.4% avg (easy=76.5%, medium=73.3%, hard=61.4%)
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
    # Day 8 defaults: auto_cluster_threshold=True, ccg_merge=True
    defaults = dict(
        threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
        align_half_window=15, pre_samples=20, cluster_threshold=7.0,
        cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
        sneo_smooth_window=3, matched_filter_detect=True,
        matched_filter_threshold=3.5, gmm_refine=False, gmm_max_iter=10,
        bandpass_low=0.0, bandpass_high=0.0, sample_rate=fs,
        common_median_ref=False, merge_dprime_threshold=2.0,
        refinement_iterations=0, use_amplitude_profile=False,
        amplitude_profile_neighbors=4,
        # Day 8 new defaults:
        ccg_merge=True,
        auto_cluster_threshold=True,
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
    cmp = compare_sorter_to_ground_truth(
        gt_sorting, sorting, exhaustive_gt=True,
        delta_time=TOLERANCE_MS, match_mode="hungarian"
    )
    avg = cmp.get_performance(method="pooled_with_average")
    acc = float(avg["accuracy"])
    prec = float(avg["precision"])
    rec_ = float(avg["recall"])
    wd = cmp.count_well_detected_units(well_detected_score=0.8)
    print(f"  {label:<60} acc={acc:.3f}  prec={prec:.3f}  rec={rec_:.3f}"
          f"  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    per_unit_df = cmp.get_performance()
    if detail:
        for uid in per_unit_df.index:
            row = per_unit_df.loc[uid]
            print(f"    unit {uid}: acc={float(row['accuracy']):.3f}")
    return acc, {str(uid): float(per_unit_df.loc[uid]["accuracy"])
                 for uid in per_unit_df.index}


# Baseline values from shape-features benchmark (shape_features_benchmark.py)
BASELINE = {"easy": 0.7650, "medium": 0.7334, "hard": 0.6137}


def main():
    results = {}

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        # New defaults: auto_cluster_threshold + ccg_merge
        acc_new, per_unit = run(difficulty, "adaptive_threshold + ccg_merge (new defaults)", detail=True)

        # Regression check: disable new features, should match baseline
        acc_base, _ = run(difficulty, "baseline (auto_thresh=F, ccg_merge=F)",
                          auto_cluster_threshold=False, ccg_merge=False)

        delta = acc_new - BASELINE[difficulty]
        print(f"  baseline→new: {BASELINE[difficulty]:.3f} → {acc_new:.3f}  ({delta:+.3f})")
        results[difficulty] = {
            "baseline_acc": BASELINE[difficulty],
            "new_acc": acc_new,
            "regression_acc": acc_base,
            "delta": delta,
            "per_unit": {str(k): float(v) for k, v in per_unit.items()},
        }

    easy_acc = results["easy"]["new_acc"]
    med_acc = results["medium"]["new_acc"]
    hard_acc = results["hard"]["new_acc"]
    avg = (easy_acc + med_acc + hard_acc) / 3.0
    base_avg = sum(BASELINE.values()) / 3.0

    print(f"\n{'='*60}")
    print(f"Baseline avg: {base_avg:.3f}")
    print(f"New avg:      {avg:.3f}  ({avg - base_avg:+.3f})")
    print(f"{'='*60}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "date": ts,
        "day7_avg": day7_avg,
        "day8_avg": avg,
        "target": 0.715,
        "results": results,
    }
    out_path = f"benchmarks/results/day8_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
