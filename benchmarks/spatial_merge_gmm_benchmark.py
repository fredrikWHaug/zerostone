"""Benchmark: spatial_merge_dprime + gmm_refine tuning.

Diagnostic found:
- Medium: spatial_merge_dprime=2.5 → +3.2% (over-merging at 1.5)
- Medium: gmm_refine=True → +4.1%
- Hard: units 1,8,12 are precision-limited (noise spikes misassigned)

Test combinations and whether they can be auto-gated.
"""

import time
import numpy as np
import spikeinterface.core as si
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core import NumpySorting
import zpybci as zbci

TOLERANCE_MS = 0.4
_cache = {}
BASELINE = {"easy": 0.8270, "medium": 0.7340, "hard": 0.6420}


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
        print(f"  {label:<72} acc=0.000  {elapsed:.1f}s")
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
    delta = acc - BASELINE[difficulty]
    print(f"  {label:<72} acc={acc:.3f} ({delta:+.3f})  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    if detail:
        per_unit = cmp.get_performance()
        for uid in per_unit.index:
            a = float(per_unit.loc[uid]["accuracy"])
            r = float(per_unit.loc[uid]["recall"])
            p = float(per_unit.loc[uid]["precision"])
            print(f"    unit {uid:>2}: acc={a:.3f}  R={r:.3f}  P={p:.3f}")
    return acc


def main():
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        detail = difficulty in ("medium", "hard")
        r = {}

        r["baseline"] = run(difficulty, "baseline", detail=detail)

        # spatial_merge_dprime sweep
        for smd in [2.0, 2.5, 3.0]:
            r[f"smd_{smd}"] = run(difficulty, f"spatial_merge_dprime={smd}",
                                  spatial_merge_dprime=smd, detail=detail)

        # gmm_refine alone
        r["gmm"] = run(difficulty, "gmm_refine=True",
                        gmm_refine=True, detail=detail)

        # gmm + spatial_merge_dprime=2.5
        r["gmm_smd25"] = run(difficulty, "gmm_refine + spatial_merge_dprime=2.5",
                              gmm_refine=True, spatial_merge_dprime=2.5, detail=detail)

        # gmm + spatial_merge_dprime=2.0
        r["gmm_smd20"] = run(difficulty, "gmm_refine + spatial_merge_dprime=2.0",
                              gmm_refine=True, spatial_merge_dprime=2.0, detail=detail)

        results[difficulty] = r

    print(f"\n{'='*90}")
    print(f"{'Config':<55} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"{'='*90}")
    for label, key in [
        ("baseline (current defaults)", "baseline"),
        ("spatial_merge_dprime=2.0", "smd_2.0"),
        ("spatial_merge_dprime=2.5", "smd_2.5"),
        ("spatial_merge_dprime=3.0", "smd_3.0"),
        ("gmm_refine=True", "gmm"),
        ("gmm + spatial_merge_dprime=2.5", "gmm_smd25"),
        ("gmm + spatial_merge_dprime=2.0", "gmm_smd20"),
    ]:
        e = results["easy"].get(key, 0)
        m = results["medium"].get(key, 0)
        h = results["hard"].get(key, 0)
        avg = (e + m + h) / 3.0
        print(f"  {label:<53} {e:>7.3f} {m:>7.3f} {h:>7.3f} {avg:>7.3f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
