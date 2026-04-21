"""Diagnostic: understand where the remaining accuracy loss comes from.

For each difficulty, examine:
1. Baseline per-unit accuracy breakdown
2. Per-unit recall vs precision to identify whether the loss is misses or false assigns
3. Effect of merge_dprime_threshold (are units being over-merged?)
4. Effect of split thresholds (are units being under-split?)
"""

import time
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


def run_and_analyze(difficulty, label, **kwargs):
    traces, fs, n_ch, gt_trains, gt_sorting = gen(difficulty)
    data = traces.copy()
    probe = zbci.ProbeLayout.linear(n_ch, 25.0)
    defaults = dict(
        threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
        align_half_window=15, pre_samples=20, cluster_threshold=7.0,
        cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
        matched_filter_detect=True, matched_filter_threshold=3.5,
        bandpass_low=0.0, bandpass_high=0.0, sample_rate=fs,
        common_median_ref=False, merge_dprime_threshold=2.0,
        use_amplitude_profile=False, amplitude_profile_neighbors=4,
        ccg_merge=True, auto_cluster_threshold=True,
        svd_init=False, auto_svd_init=True, gmm_refine=False,
        min_cluster_snr=2.5, refinement_iterations=1, auto_refine=True,
        refine_collapse_guard=True, refine_isi_guard=False,
        auto_threshold=True, auto_refine_iterations=3, ncc_threshold=0.70,
        template_subtract_passes=2, auto_amplitude_profile=False,
    )
    defaults.update(kwargs)
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **defaults)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label}: 0 spikes")
        return 0.0
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
    print(f"\n  {label}: acc={acc:.3f}  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")

    per_unit = cmp.get_performance()
    for uid in per_unit.index:
        r = float(per_unit.loc[uid]["recall"])
        p = float(per_unit.loc[uid]["precision"])
        a = float(per_unit.loc[uid]["accuracy"])
        gt_count = len(gt_trains.get(uid, []))
        bottleneck = "RECALL" if r < p else "PRECISION" if p < r else "EQUAL"
        if a < 0.01:
            bottleneck = "UNDETECTED"
        print(f"    unit {uid:>2}: acc={a:.3f}  recall={r:.3f}  prec={p:.3f}  gt={gt_count:>4}  [{bottleneck}]")

    return acc


def main():
    for difficulty in ["medium", "hard"]:
        print(f"\n{'='*80}")
        print(f"  {difficulty.upper()}")
        print(f"{'='*80}")

        # Baseline
        run_and_analyze(difficulty, "BASELINE (defaults)")

        # Looser merge (more merging)
        run_and_analyze(difficulty, "merge_dprime=1.5 (more merging)",
                        merge_dprime_threshold=1.5)

        # Tighter merge (less merging)
        run_and_analyze(difficulty, "merge_dprime=3.0 (less merging)",
                        merge_dprime_threshold=3.0)

        # Spatial merge tighter
        run_and_analyze(difficulty, "spatial_merge_dprime=2.5 (less spatial merge)",
                        spatial_merge_dprime=2.5)

        # No CCG merge
        run_and_analyze(difficulty, "ccg_merge=False",
                        ccg_merge=False)

        # Lower min_cluster_snr
        run_and_analyze(difficulty, "min_cluster_snr=1.5",
                        min_cluster_snr=1.5)

        # GMM refine
        run_and_analyze(difficulty, "gmm_refine=True",
                        gmm_refine=True)


if __name__ == "__main__":
    main()
