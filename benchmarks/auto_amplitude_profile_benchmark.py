"""Benchmark: auto_amplitude_profile — amplitude-profile spatial features for large probes.

Baseline (auto_amplitude_profile=False): Easy 82.7%, Medium 73.4%, Hard 64.2%, Avg 73.4%

auto_amplitude_profile=True activates use_amplitude_profile for C>=8.
The amplitude profile encodes neighbor-channel energy bleed as a physics-based
spatial fingerprint in feature dimension K-2, replacing the weaker spike half-width.
On large probes, multiple units often share the same peak channel; their waveform
shapes are similar, but their spatial decay profiles differ. The ratio:
  neighbor_sum / (n_neighbors * peak_amp)
ranges [0.3, 2.0]: tightly-localized units ~0.5, spread units ~1.2.
profile_scale = cluster_threshold * 2.0 = 14 maps this to feature range [4.2, 28].
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
        sneo_smooth_window=3, matched_filter_detect=True,
        matched_filter_threshold=3.5, gmm_max_iter=10,
        bandpass_low=0.0, bandpass_high=0.0, sample_rate=fs,
        common_median_ref=False, merge_dprime_threshold=2.0,
        use_amplitude_profile=False, amplitude_profile_neighbors=4,
        ccg_merge=True, auto_cluster_threshold=True,
        svd_init=False, auto_svd_init=True, gmm_refine=False,
        min_cluster_snr=2.5, refinement_iterations=1, auto_refine=True,
        refine_collapse_guard=True, refine_isi_guard=False,
        refine_isi_tolerance=0.1, auto_threshold=True,
        auto_refine_iterations=3, ncc_threshold=0.70,
        template_subtract_passes=2, auto_amplitude_profile=True,
    )
    defaults.update(kwargs)
    t0 = time.perf_counter()
    result = zbci.sort_multichannel(data, probe, **defaults)
    elapsed = time.perf_counter() - t0
    n_spikes = result["n_spikes"]
    n_cl = result["n_clusters"]
    n_gt = len(gt_trains)
    if n_spikes == 0:
        print(f"  {label:<78} acc=0.000  0 spk  {elapsed:.1f}s")
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
    print(f"  {label:<78} acc={acc:.3f}  ({delta:+.3f})  wd={wd}/{n_gt}  {n_spikes}spk/{n_cl}cl  {elapsed:.1f}s")
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
        detail = difficulty == "hard"

        # Baseline: half-width feature (old behavior, auto_amplitude_profile=False)
        acc_off, _ = run(difficulty, "auto_amplitude_profile=False  (half-width fallback, baseline)",
                         auto_amplitude_profile=False)

        # New default: auto_amplitude_profile=True
        acc_on, pu_on = run(difficulty, "auto_amplitude_profile=True   (amplitude profile for C>=8)",
                            auto_amplitude_profile=True, detail=detail)

        # Forced on for all probes: use_amplitude_profile=True, auto=False
        acc_forced, _ = run(difficulty, "use_amplitude_profile=True  (forced, all probe sizes)",
                            use_amplitude_profile=True, auto_amplitude_profile=False)

        # Profile with more neighbors: 8 instead of 4
        acc_8nbr, _ = run(difficulty, "auto_amplitude_profile=True, neighbors=8",
                          auto_amplitude_profile=True, amplitude_profile_neighbors=8)

        results[difficulty] = {
            "off": acc_off,
            "auto_on": acc_on,
            "forced": acc_forced,
            "neighbors8": acc_8nbr,
            "per_unit_auto_on": {str(k): float(v) for k, v in pu_on.items()},
        }

    print(f"\n{'='*92}")
    print(f"{'Config':<62} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"{'='*92}")
    for label, key in [
        ("auto_amplitude_profile=False  (half-width, baseline)",  "off"),
        ("auto_amplitude_profile=True   (default, C>=8 gate)",    "auto_on"),
        ("use_amplitude_profile=True  (forced all sizes)",        "forced"),
        ("auto_amplitude_profile=True, neighbors=8",              "neighbors8"),
    ]:
        e = results["easy"][key]
        m = results["medium"][key]
        h = results["hard"][key]
        avg = (e + m + h) / 3.0
        print(f"  {label:<60} {e:>7.3f} {m:>7.3f} {h:>7.3f} {avg:>7.3f}")
    print(f"{'='*92}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"date": ts, "baseline_avg": sum(BASELINE.values()) / 3.0, "results": results}
    out_path = f"benchmarks/results/auto_amplitude_profile_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
