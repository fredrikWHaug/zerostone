"""Day 5 benchmark: template reassignment + tuned configs."""

import time
import numpy as np
import zpybci as zbci

import spikeinterface.core as si
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core import NumpySorting

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
    all_l = np.concatenate([np.full(len(gt_trains[u]), u, dtype=np.int64) for u in sorted(gt_trains)])
    idx = np.argsort(all_t)
    gt_sorting = NumpySorting.from_samples_and_labels([all_t[idx]], [all_l[idx]], sampling_frequency=fs)
    _cache[difficulty] = (traces, fs, c["num_channels"], gt_trains, gt_sorting)
    return _cache[difficulty]


def run(difficulty, label, detail=False, **kwargs):
    traces, fs, n_ch, gt_trains, gt_sorting = gen(difficulty)
    data = traces.copy()
    probe = zbci.ProbeLayout.linear(n_ch, kwargs.pop("probe_pitch", 25.0))
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
        print(f"  {label:<55} acc=0.000  0 spk  {elapsed:.1f}s")
        return 0.0
    spike_times = np.asarray(result["spike_times"][:n_spikes], dtype=np.int64)
    labels_out = np.asarray(result["labels"][:n_spikes], dtype=np.int64)
    sorting = NumpySorting.from_samples_and_labels([spike_times], [labels_out], sampling_frequency=fs)
    comp = compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=True,
                                          delta_time=TOLERANCE_MS, match_mode="hungarian")
    perf = comp.get_performance()
    avg = comp.get_performance(method="pooled_with_average")
    acc = float(avg["accuracy"])
    prec = float(avg["precision"])
    rec_ = float(avg["recall"])
    wd = comp.count_well_detected_units(well_detected_score=0.8)
    print(f"  {label:<55} acc={acc:.3f}  prec={prec:.3f}  rec={rec_:.3f}  "
          f"wd={wd}/{n_gt}  {n_spikes:>5} spk  {n_cl:>3} cl  {elapsed:.1f}s")
    if detail:
        for uid in perf.index:
            row = perf.loc[uid]
            gt_n = len(gt_trains.get(uid, []))
            print(f"    Unit {uid}: acc={float(row['accuracy']):.3f}  "
                  f"prec={float(row['precision']):.3f}  rec={float(row['recall']):.3f}  "
                  f"({gt_n} GT)")
    return acc


def main():
    print("=" * 76)
    print("  Day 5 Benchmark: Template Reassignment")
    print("=" * 76)

    summary = {}
    for diff in ["easy", "medium", "hard"]:
        print(f"\n--- {diff.upper()} ---")
        gen(diff)

        # Day 5 default (template reassignment active)
        print("  [A] Day 5 default (template reassignment):")
        a = run(diff, "Day5 default", detail=True)

        # + CMR
        b = run(diff, "Day5 + CMR", common_median_ref=True)

        # + thr=4.5
        c = run(diff, "Day5 + thr=4.5", threshold=4.5)

        # + thr=4.5 + CMR
        d = run(diff, "Day5 + thr=4.5 + CMR", threshold=4.5, common_median_ref=True)

        # + ct=10
        e = run(diff, "Day5 + ct=10", cluster_threshold=10.0)

        # + ct=10 + CMR
        f = run(diff, "Day5 + ct=10 + CMR", cluster_threshold=10.0, common_median_ref=True)

        # + thr=4.5 + ct=10
        g = run(diff, "Day5 + thr=4.5 + ct=10", threshold=4.5, cluster_threshold=10.0)

        # + thr=4.5 + ct=10 + CMR
        h = run(diff, "Day5 + thr=4.5 + ct=10 + CMR",
                threshold=4.5, cluster_threshold=10.0, common_median_ref=True)

        # CCG merge combos
        run(diff, "Day5 + ccg", ccg_merge=True)
        run(diff, "Day5 + ccg + cmr", ccg_merge=True, common_median_ref=True)

        # GMM combos
        run(diff, "Day5 + gmm", gmm_refine=True)

        # Best from Day 4
        run(diff, "Day5 + thr=4.5 + cmr + ccg",
            threshold=4.5, common_median_ref=True, ccg_merge=True)

        best = max(a, b, c, d, e, f, g, h)
        summary[diff] = best
        print(f"  >> Best {diff}: {best:.3f}")

    print(f"\n{'='*76}")
    print("  SUMMARY")
    print(f"{'='*76}")
    total = 0
    for diff in ["easy", "medium", "hard"]:
        total += summary[diff]
        print(f"  {diff:<8} {summary[diff]:.3f}")
    print(f"  {'AVERAGE':<8} {total/3:.3f}")


if __name__ == "__main__":
    main()
