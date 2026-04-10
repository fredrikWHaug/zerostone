"""Benchmark: collapse-safe refinement + SVD init interaction test.

The svd+refine interaction (Day 9) collapsed medium unit 7 from 80.9% to 0%
(-13.4% average). Root cause: SVD-seeded centroids placed two cluster territories
close enough that a single refinement pass pulled all of one unit's spikes into
its neighbor. refine_collapse_guard performs a dry-run before committing each
pass; if any cluster with >= split_min_cluster_size spikes would be emptied,
the pass is skipped.

This benchmark answers: with the collapse guard in place, is svd_init now safe
to enable? If yes, what accuracy does svd+refine+guard achieve vs the current
default (refine-only)?

Baseline: current defaults (refine=1, auto_refine=T, collapse_guard=T) = 73.0% avg
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

BASELINE = {"easy": 0.8265, "medium": 0.7370, "hard": 0.6270}


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
    # Current defaults (post-Day 10)
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
        refinement_iterations=1, auto_refine=True, refine_collapse_guard=True,
    )
    defaults.update(kwargs)
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **defaults)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label:<70} acc=0.000  0 spk  {elapsed:.1f}s")
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
    wd = cmp.count_well_detected_units(well_detected_score=0.8)
    delta = acc - BASELINE[difficulty]
    print(f"  {label:<70} acc={acc:.3f}  ({delta:+.3f})  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    per_unit_df = cmp.get_performance()
    if detail:
        for uid in per_unit_df.index:
            print(f"    unit {uid}: acc={float(per_unit_df.loc[uid]['accuracy']):.3f}")
    return acc, {str(uid): float(per_unit_df.loc[uid]["accuracy"])
                 for uid in per_unit_df.index}


def main():
    results = {}

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        detail = (difficulty == "medium")  # unit 7 is the canary for medium

        # Current default (refine=1, guard=T, svd=F)
        acc_base, _ = run(difficulty, "current default (refine=1, guard=T, svd=F)")

        # svd + refine + guard — the previously-crashing combination
        acc_svd_guard, pu_svd = run(
            difficulty, "svd=T + refine=1 + guard=T  (target: no crash)",
            svd_init=True, detail=detail,
        )

        # svd + refine, NO guard — should reproduce Day 9 crash on medium
        acc_svd_no_guard, pu_svd_ng = run(
            difficulty, "svd=T + refine=1 + guard=F  (Day 9 crash reproduction)",
            svd_init=True, refine_collapse_guard=False,
        )

        # guard alone with no svd — verify guard doesn't hurt the baseline
        acc_guard_no_svd, _ = run(
            difficulty, "guard=T + svd=F             (guard cost on clean init)",
        )

        # svd + guard, no refinement — upper bound of svd-init benefit without refine
        acc_svd_no_refine, _ = run(
            difficulty, "svd=T + refine=0 + guard=T  (svd contribution isolated)",
            svd_init=True, refinement_iterations=0,
        )

        results[difficulty] = {
            "baseline": acc_base,
            "svd_refine_guard": acc_svd_guard,
            "svd_refine_no_guard": acc_svd_no_guard,
            "guard_no_svd": acc_guard_no_svd,
            "svd_no_refine": acc_svd_no_refine,
            "per_unit_svd_guard": {str(k): float(v) for k, v in pu_svd.items()},
            "per_unit_svd_no_guard": {str(k): float(v) for k, v in pu_svd_ng.items()},
        }

    print(f"\n{'='*80}")
    print(f"{'Config':<48} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"{'='*80}")
    configs = [
        ("current default (refine+guard, no svd)", "baseline"),
        ("svd + refine + guard",                   "svd_refine_guard"),
        ("svd + refine, NO guard (Day 9 crash)",   "svd_refine_no_guard"),
        ("guard only, no svd",                     "guard_no_svd"),
        ("svd + no refine",                        "svd_no_refine"),
    ]
    for label, key in configs:
        e = results["easy"][key]
        m = results["medium"][key]
        h = results["hard"][key]
        avg = (e + m + h) / 3.0
        print(f"  {label:<46} {e:>7.3f} {m:>7.3f} {h:>7.3f} {avg:>7.3f}")
    print(f"{'='*80}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "date": ts,
        "baseline_avg": sum(BASELINE.values()) / 3.0,
        "results": results,
    }
    out_path = f"benchmarks/results/collapse_guard_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
