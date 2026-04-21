"""Benchmark: amplitude profile scale tuning.

The amplitude_profile feature uses profile_scale = cluster_threshold * 2.0 = 14.
This scale may be too large, dominating the PCA component in K-2 and causing
cluster merging errors. Test lower scales by varying cluster_threshold with
use_amplitude_profile=True directly.

Since profile_scale = cluster_threshold * 2.0, we can't decouple them directly.
Instead, test use_amplitude_profile at different cluster_threshold values and
compare to half-width (default) at the same threshold.
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


def run(difficulty, label, detail=False, **kwargs):
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
        refine_isi_tolerance=0.1, auto_threshold=True,
        auto_refine_iterations=3, ncc_threshold=0.70,
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
        print(f"  {label:<70} acc=0.000  0 spk  {elapsed:.1f}s")
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
    print(f"  {label:<70} acc={acc:.3f}  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    if detail:
        per_unit_df = cmp.get_performance()
        for uid in per_unit_df.index:
            print(f"    unit {uid}: acc={float(per_unit_df.loc[uid]['accuracy']):.3f}")
    return acc


def main():
    for difficulty in ["medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        detail = difficulty == "hard"

        # Baseline: half-width (no amplitude profile)
        run(difficulty, "baseline (half-width, ct=7.0)",
            use_amplitude_profile=False, detail=detail)

        # Amplitude profile with standard cluster_threshold=7.0
        run(difficulty, "amp_profile (ct=7.0, scale=14.0)",
            use_amplitude_profile=True, detail=detail)

        # Try lower cluster thresholds to reduce profile_scale
        for ct in [5.0, 4.0, 3.5]:
            ps = ct * 2.0
            run(difficulty, f"amp_profile (ct={ct}, scale={ps})",
                use_amplitude_profile=True, cluster_threshold=ct, detail=detail)

        # Try more neighbors
        run(difficulty, "amp_profile (ct=7.0, nbr=2)",
            use_amplitude_profile=True, amplitude_profile_neighbors=2, detail=detail)


if __name__ == "__main__":
    main()
