"""Day 4 final benchmark: new defaults vs old, per-unit analysis, before/after table."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zpybci as zbci

import spikeinterface.core as si
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core import NumpySorting

TOLERANCE_MS = 0.4
RESULTS_DIR = Path(__file__).parent / "results"
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
    print(f"  Generating {difficulty} recording ({c['num_channels']}ch, "
          f"{c['num_units']} units, noise={c['noise_levels']})...", end=" ", flush=True)
    t0 = time.perf_counter()
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
    n_gt = sum(len(v) for v in gt_trains.values())
    _cache[difficulty] = (traces, fs, c["num_channels"], gt_trains, gt_sorting)
    print(f"done ({time.perf_counter() - t0:.1f}s, {n_gt} GT spikes)")
    return _cache[difficulty]


def run(difficulty, label, detail=False, **kwargs):
    traces, fs, n_ch, gt_trains, gt_sorting = gen(difficulty)
    data = traces.copy()
    probe = zbci.ProbeLayout.linear(n_ch, kwargs.pop("probe_pitch", 25.0))
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **kwargs)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)

    if n_spikes == 0:
        print(f"  {label:<55} acc=0.000  0 spk  {elapsed:.1f}s")
        return {"label": label, "difficulty": difficulty, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "well_detected": 0, "n_gt": n_gt, "n_spikes": 0, "n_clusters": 0}

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
        print(f"    {'Unit':<6} {'GT spk':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7}")
        print(f"    {'-'*42}")
        for uid in perf.index:
            row = perf.loc[uid]
            gt_n = len(gt_trains.get(uid, []))
            print(f"    {str(uid):<6} {gt_n:>7} {float(row['accuracy']):>7.3f} "
                  f"{float(row['precision']):>7.3f} {float(row['recall']):>7.3f}")

    return {"label": label, "difficulty": difficulty, "accuracy": acc, "precision": prec,
            "recall": rec_, "well_detected": int(wd), "n_gt": n_gt, "n_spikes": n_spikes,
            "n_clusters": n_cl, "elapsed": elapsed}


# Old defaults (Day 3 era)
OLD_DEFAULTS = dict(
    threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
    align_half_window=15, pre_samples=20, cluster_threshold=5.0,
    cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
    sneo_smooth_window=3, ccg_merge=False, matched_filter_detect=False,
    matched_filter_threshold=4.0, gmm_refine=False, gmm_max_iter=10,
    bandpass_low=0.0, bandpass_high=0.0, sample_rate=30000.0,
    common_median_ref=False, merge_dprime_threshold=3.1,
)

# New defaults (Day 4: ct=7, dp=2.0, mf=true)
NEW_DEFAULTS = dict(
    threshold=5.0, refractory=15, spatial_radius=75.0, temporal_radius=5,
    align_half_window=15, pre_samples=20, cluster_threshold=7.0,
    cluster_max_count=1000, whitening_epsilon=1e-6, detection_mode="amplitude",
    sneo_smooth_window=3, ccg_merge=False, matched_filter_detect=True,
    matched_filter_threshold=4.0, gmm_refine=False, gmm_max_iter=10,
    bandpass_low=0.0, bandpass_high=0.0, sample_rate=30000.0,
    common_median_ref=False, merge_dprime_threshold=2.0,
)


def main():
    print("=" * 76)
    print("  Day 4 Final Benchmark: Before/After Comparison")
    print(f"  zpybci {zbci.__version__}")
    print(f"  Matching: Hungarian, delta_time={TOLERANCE_MS}ms")
    print("=" * 76)

    all_results = {}
    for diff in ["easy", "medium", "hard"]:
        print(f"\n{'='*76}")
        print(f"  {diff.upper()}")
        print(f"{'='*76}")
        gen(diff)

        results = []

        # Before (Day 3 defaults)
        print("\n  --- Day 3 Defaults (ct=5, dp=3.1, no MF) ---")
        r = run(diff, "Day3 defaults", detail=True, **OLD_DEFAULTS)
        results.append(r)

        # After (new defaults)
        print("\n  --- Day 4 New Defaults (ct=7, dp=2.0, MF=true) ---")
        r = run(diff, "Day4 new defaults", detail=True, **NEW_DEFAULTS)
        results.append(r)

        # Additional tuned configs
        print("\n  --- Tuned Configs ---")

        # CMR
        r = run(diff, "new + CMR", **{**NEW_DEFAULTS, "common_median_ref": True})
        results.append(r)

        # Threshold 4.5
        r = run(diff, "new + thr=4.5", **{**NEW_DEFAULTS, "threshold": 4.5})
        results.append(r)

        # ct=10
        r = run(diff, "new + ct=10", **{**NEW_DEFAULTS, "cluster_threshold": 10.0})
        results.append(r)

        # CCG merge
        r = run(diff, "new + ccg", **{**NEW_DEFAULTS, "ccg_merge": True})
        results.append(r)

        # GMM
        r = run(diff, "new + gmm", **{**NEW_DEFAULTS, "gmm_refine": True})
        results.append(r)

        # Best combo from earlier runs
        r = run(diff, "new + thr=4.5 + cmr",
                **{**NEW_DEFAULTS, "threshold": 4.5, "common_median_ref": True})
        results.append(r)

        r = run(diff, "new + thr=4.5 + ccg + cmr",
                **{**NEW_DEFAULTS, "threshold": 4.5, "ccg_merge": True, "common_median_ref": True})
        results.append(r)

        r = run(diff, "new + bp=300-6000",
                **{**NEW_DEFAULTS, "bandpass_low": 300.0, "bandpass_high": 6000.0})
        results.append(r)

        all_results[diff] = results

    # Summary table
    print(f"\n{'='*76}")
    print("  BEFORE/AFTER SUMMARY")
    print(f"{'='*76}")
    print(f"  {'Difficulty':<10} {'Day3 Acc':>10} {'Day4 Acc':>10} {'Best Acc':>10} {'Best Config':>35} {'Delta':>8}")
    print(f"  {'-'*78}")

    avg_day3 = 0
    avg_day4 = 0
    avg_best = 0
    for diff in ["easy", "medium", "hard"]:
        results = all_results[diff]
        day3 = results[0]["accuracy"]
        day4 = results[1]["accuracy"]
        best_r = max(results, key=lambda r: r["accuracy"])
        best = best_r["accuracy"]
        delta = best - day3
        sign = "+" if delta >= 0 else ""
        print(f"  {diff:<10} {day3:>10.3f} {day4:>10.3f} {best:>10.3f} {best_r['label']:>35} {sign}{delta:>7.3f}")
        avg_day3 += day3
        avg_day4 += day4
        avg_best += best

    print(f"  {'-'*78}")
    print(f"  {'AVERAGE':<10} {avg_day3/3:>10.3f} {avg_day4/3:>10.3f} {avg_best/3:>10.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"day4_final_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "zpybci_version": zbci.__version__,
        "tolerance_ms": TOLERANCE_MS,
        "old_defaults": OLD_DEFAULTS,
        "new_defaults": NEW_DEFAULTS,
        "results": {},
        "summary": {},
    }
    for diff in ["easy", "medium", "hard"]:
        output["results"][diff] = [
            {k: v for k, v in r.items() if k != "elapsed"}
            for r in all_results[diff]
        ]
        results = all_results[diff]
        best_r = max(results, key=lambda r: r["accuracy"])
        output["summary"][diff] = {
            "day3_accuracy": results[0]["accuracy"],
            "day4_accuracy": results[1]["accuracy"],
            "best_accuracy": best_r["accuracy"],
            "best_config": best_r["label"],
        }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
