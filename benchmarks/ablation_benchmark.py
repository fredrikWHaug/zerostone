"""Benchmark: single-feature isolation ablation from Day 9 findings.

Tests each disabled feature independently to identify safe wins:
- refinement_iterations=1 alone (confirmed +0.4% on medium; easy/hard unknown)
- use_amplitude_profile=True alone (never benchmarked)
- gmm_refine=True alone (known: +0.8% medium, -3.4% easy — informational)
- svd_init=True alone (known: -0.2% medium — informational)
- template_subtract_passes=3 (one extra subtraction pass)

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
    # Day 8 defaults (the stable baseline)
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
        svd_init=False, gmm_refine=False, refinement_iterations=0, min_cluster_snr=2.5,
        auto_refine=False,
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

        acc_base, _ = run(difficulty, "Day 8 baseline")

        acc_refine, pu_refine = run(
            difficulty, "refine=1 only",
            refinement_iterations=1, detail=detail,
        )
        acc_amp, pu_amp = run(
            difficulty, "amplitude_profile only",
            use_amplitude_profile=True, detail=detail,
        )
        acc_gmm, _ = run(
            difficulty, "gmm only  (informational)",
            gmm_refine=True,
        )
        acc_svd, _ = run(
            difficulty, "svd only  (informational)",
            svd_init=True,
        )
        acc_ts3, _ = run(
            difficulty, "template_subtract_passes=3",
            template_subtract_passes=3,
        )
        acc_refine_amp, pu_refine_amp = run(
            difficulty, "refine=1 + amplitude_profile",
            refinement_iterations=1, use_amplitude_profile=True, detail=detail,
        )

        results[difficulty] = {
            "baseline": acc_base,
            "refine_1": acc_refine,
            "amplitude_profile": acc_amp,
            "gmm_only": acc_gmm,
            "svd_only": acc_svd,
            "ts_passes_3": acc_ts3,
            "refine_1_plus_amp": acc_refine_amp,
            "per_unit_refine": {str(k): float(v) for k, v in pu_refine.items()},
            "per_unit_amp": {str(k): float(v) for k, v in pu_amp.items()},
            "per_unit_refine_amp": {str(k): float(v) for k, v in pu_refine_amp.items()},
        }

    print(f"\n{'='*75}")
    print(f"{'Config':<40} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"{'='*75}")
    configs = [
        ("Day 8 baseline",          "baseline"),
        ("refine=1 only",           "refine_1"),
        ("amplitude_profile only",  "amplitude_profile"),
        ("refine=1 + amp_profile",  "refine_1_plus_amp"),
        ("gmm only",                "gmm_only"),
        ("svd only",                "svd_only"),
        ("template_passes=3",       "ts_passes_3"),
    ]
    for label, key in configs:
        e = results["easy"][key]
        m = results["medium"][key]
        h = results["hard"][key]
        avg = (e + m + h) / 3.0
        print(f"  {label:<38} {e:>7.3f} {m:>7.3f} {h:>7.3f} {avg:>7.3f}")
    print(f"{'='*75}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "date": ts,
        "baseline_avg": sum(BASELINE.values()) / 3.0,
        "results": results,
    }
    out_path = f"benchmarks/results/ablation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
