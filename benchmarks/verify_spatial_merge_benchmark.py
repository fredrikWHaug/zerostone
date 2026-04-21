"""Verify: spatial_merge_dprime=2.0 (new default) vs 1.5 (old default).

Expected: Medium +3.1%, Easy unchanged, Hard flat.
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


def run(difficulty, label, **kwargs):
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
        print(f"  {label:<50} acc=0.000")
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
    print(f"  {label:<50} acc={acc:.3f}  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
    return acc


def main():
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n=== {difficulty.upper()} ===")
        # New default (spatial_merge_dprime=2.0 is now the default)
        acc_new = run(difficulty, "NEW DEFAULT (spatial_merge_dprime=2.0)")
        # Old default
        acc_old = run(difficulty, "OLD DEFAULT (spatial_merge_dprime=1.5)",
                      spatial_merge_dprime=1.5)
        results[difficulty] = {"new": acc_new, "old": acc_old}

    print(f"\n{'='*70}")
    e_old, e_new = results["easy"]["old"], results["easy"]["new"]
    m_old, m_new = results["medium"]["old"], results["medium"]["new"]
    h_old, h_new = results["hard"]["old"], results["hard"]["new"]
    avg_old = (e_old + m_old + h_old) / 3
    avg_new = (e_new + m_new + h_new) / 3
    print(f"  OLD (smd=1.5): Easy {e_old:.3f}  Medium {m_old:.3f}  Hard {h_old:.3f}  Avg {avg_old:.3f}")
    print(f"  NEW (smd=2.0): Easy {e_new:.3f}  Medium {m_new:.3f}  Hard {h_new:.3f}  Avg {avg_new:.3f}")
    print(f"  Delta:         Easy {e_new-e_old:+.3f}  Medium {m_new-m_old:+.3f}  Hard {h_new-h_old:+.3f}  Avg {avg_new-avg_old:+.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
