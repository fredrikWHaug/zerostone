"""Benchmark: auto_gmm_refine — auto-gate GMM refinement for C>=8.

Diagnostic found gmm_refine=True gives Medium +4.1% but Easy -3.4%.
Easy has C=4, Medium C=16, Hard C=32. An auto-gate for C>=8 would
capture the medium improvement without harming easy.

Question: does GMM actually help medium with the new spatial_merge_dprime=2.0?
Previous test was with smd=1.5. Need to reconfirm.
"""

import time
import numpy as np
import spikeinterface.core as si
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core import NumpySorting
import zpybci as zbci

TOLERANCE_MS = 0.4
_cache = {}
BASELINE = {"easy": 0.8270, "medium": 0.7650, "hard": 0.6400}


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
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **kwargs)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label:<65} acc=0.000  {elapsed:.1f}s")
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
    print(f"  {label:<65} acc={acc:.3f} ({delta:+.3f})  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
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

        r["baseline"] = run(difficulty, "baseline (new defaults, smd=2.0)", detail=detail)

        r["gmm"] = run(difficulty, "gmm_refine=True",
                        gmm_refine=True, detail=detail)

        r["gmm_iter20"] = run(difficulty, "gmm_refine=True, gmm_max_iter=20",
                               gmm_refine=True, gmm_max_iter=20, detail=detail)

        r["gmm_iter5"] = run(difficulty, "gmm_refine=True, gmm_max_iter=5",
                              gmm_refine=True, gmm_max_iter=5, detail=detail)

        results[difficulty] = r

    print(f"\n{'='*85}")
    print(f"{'Config':<50} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"{'='*85}")
    for label, key in [
        ("baseline (smd=2.0 defaults)", "baseline"),
        ("gmm_refine=True", "gmm"),
        ("gmm_refine, max_iter=20", "gmm_iter20"),
        ("gmm_refine, max_iter=5", "gmm_iter5"),
    ]:
        e = results["easy"][key]
        m = results["medium"][key]
        h = results["hard"][key]
        avg = (e + m + h) / 3.0
        print(f"  {label:<48} {e:>7.3f} {m:>7.3f} {h:>7.3f} {avg:>7.3f}")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
