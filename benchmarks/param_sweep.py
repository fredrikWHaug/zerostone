"""Parameter sweep benchmark for Zerostone spike sorter.

Generates a single synthetic recording per difficulty level and sweeps
threshold, matched_filter_detect, svd_init, bandpass, and common_median_ref
to find optimal settings. Uses greedy spike matching (tolerance=20 samples).

Usage:
    python benchmarks/param_sweep.py
"""

import itertools
import time

import numpy as np

from zpybci.zpybci import ProbeLayout, sort_multichannel
from zpybci.synthetic import generate_recording

TOLERANCE = 20  # 0.67 ms at 30 kHz

RECORDING_PRESETS = {
    "easy": dict(n_channels=32, n_units=5, noise_std=1.0, duration_s=60.0, firing_rate=5.0),
    "medium": dict(n_channels=32, n_units=10, noise_std=1.5, duration_s=60.0, firing_rate=8.0),
    "hard": dict(n_channels=64, n_units=20, noise_std=2.0, duration_s=60.0, firing_rate=10.0),
}

SWEEP_PARAMS = {
    "threshold": [3.5, 4.0, 4.5, 5.0],
    "matched_filter_detect": [True, False],
    "svd_init": [True, False],
    "bandpass": [True, False],
    "common_median_ref": [True, False],
}


def match_spikes(gt_times, sorted_times, tolerance=TOLERANCE):
    if len(gt_times) == 0:
        return 0, 0, len(sorted_times)
    if len(sorted_times) == 0:
        return 0, len(gt_times), 0
    matched = set()
    tp = 0
    start = 0
    for gt_t in gt_times:
        best_idx, best_dist = -1, tolerance + 1
        while start < len(sorted_times) and sorted_times[start] < gt_t - tolerance:
            start += 1
        for j in range(start, len(sorted_times)):
            if sorted_times[j] > gt_t + tolerance:
                break
            dist = abs(int(sorted_times[j]) - int(gt_t))
            if dist < best_dist and j not in matched:
                best_dist = dist
                best_idx = j
        if best_idx >= 0:
            tp += 1
            matched.add(best_idx)
    return tp, len(gt_times) - tp, len(sorted_times) - tp


def compute_accuracy(rec, sorted_times, sorted_labels):
    gt_times, gt_labels, n_units = rec["all_spike_times"], rec["spike_labels"], rec["n_units"]
    gt_by_unit = {u: np.sort(gt_times[gt_labels == u]) for u in range(n_units)}
    sorted_clusters = {}
    if len(sorted_labels) > 0:
        for cl in np.unique(sorted_labels):
            sorted_clusters[int(cl)] = np.sort(sorted_times[sorted_labels == cl])
    cluster_ids = sorted(sorted_clusters.keys())
    # Build agreement matrix
    agreement = np.zeros((n_units, len(cluster_ids)))
    details = {}
    for i in range(n_units):
        for j, cl in enumerate(cluster_ids):
            tp, fn, fp = match_spikes(gt_by_unit[i], sorted_clusters[cl])
            d = tp + fn + fp
            agreement[i, j] = tp / d if d > 0 else 0.0
            details[(i, j)] = (tp, fn, fp)
    # Greedy assignment
    assigned_c, assigned_u = set(), set()
    total_tp = total_fn = total_fp = 0
    for _ in range(n_units):
        best_a, bi, bj = -1.0, -1, -1
        for i in range(n_units):
            if i in assigned_u:
                continue
            for j in range(len(cluster_ids)):
                if j in assigned_c:
                    continue
                if agreement[i, j] > best_a:
                    best_a, bi, bj = agreement[i, j], i, j
        if bi < 0:
            break
        tp, fn, fp = details[(bi, bj)]
        total_tp += tp; total_fn += fn; total_fp += fp
        assigned_u.add(bi); assigned_c.add(bj)
    for u in range(n_units):
        if u not in assigned_u:
            total_fn += len(gt_by_unit[u])
    d = total_tp + total_fn + total_fp
    return total_tp / d if d > 0 else 0.0


def generate_cached(preset_name, seed=42):
    p = RECORDING_PRESETS[preset_name]
    print(f"  Generating {preset_name} recording...", end=" ", flush=True)
    t0 = time.perf_counter()
    rec = generate_recording(
        n_channels=p["n_channels"], duration_s=p["duration_s"],
        sampling_rate=30000.0, n_units=p["n_units"],
        firing_rate=p["firing_rate"], noise_std=p["noise_std"], seed=seed,
    )
    print(f"done ({time.perf_counter() - t0:.1f}s, {len(rec['all_spike_times'])} GT spikes)")
    return rec


def run_combo(rec, n_channels, combo):
    thr, mf, svd, bp, cmr = combo
    probe = ProbeLayout.linear(n_channels, 25.0)
    result = sort_multichannel(
        rec["data"], probe,
        threshold=thr,
        refractory=15,
        matched_filter_detect=mf,
        matched_filter_threshold=4.0,
        svd_init=svd,
        bandpass_low=300.0 if bp else 0.0,
        bandpass_high=6000.0 if bp else 0.0,
        sample_rate=30000.0,
        common_median_ref=cmr,
        cluster_threshold=8.0,
        min_cluster_snr=1.5,
    )
    st = np.array(result["spike_times"], dtype=np.int64)
    sl = np.array(result["labels"], dtype=np.int64)
    acc = compute_accuracy(rec, st, sl) if result["n_spikes"] > 0 else 0.0
    return acc, result["n_spikes"], result["n_clusters"]


def sweep(rec, preset_name, combos):
    n_ch = RECORDING_PRESETS[preset_name]["n_channels"]
    results = []
    total = len(combos)
    for idx, combo in enumerate(combos, 1):
        thr, mf, svd, bp, cmr = combo
        t0 = time.perf_counter()
        acc, n_spk, n_cl = run_combo(rec, n_ch, combo)
        elapsed = time.perf_counter() - t0
        results.append((acc, thr, mf, svd, bp, cmr, n_spk, n_cl, elapsed))
        print(f"  [{idx:>3}/{total}] thr={thr:.1f} mf={mf!s:<5} svd={svd!s:<5} "
              f"bp={bp!s:<5} cmr={cmr!s:<5} -> acc={acc:.4f} "
              f"({n_spk} spk, {n_cl} cl, {elapsed:.1f}s)")
    results.sort(key=lambda r: -r[0])
    return results


def print_table(results, title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Rank':>4} {'Acc':>7} {'Thr':>5} {'MF':>5} {'SVD':>5} "
          f"{'BP':>5} {'CMR':>5} {'Spikes':>7} {'Clust':>5} {'Time':>6}")
    print(f"  {'-'*74}")
    for i, (acc, thr, mf, svd, bp, cmr, n_spk, n_cl, t) in enumerate(results, 1):
        print(f"  {i:>4} {acc:>7.4f} {thr:>5.1f} {str(mf):>5} {str(svd):>5} "
              f"{str(bp):>5} {str(cmr):>5} {n_spk:>7} {n_cl:>5} {t:>5.1f}s")


def main():
    all_combos = list(itertools.product(
        SWEEP_PARAMS["threshold"],
        SWEEP_PARAMS["matched_filter_detect"],
        SWEEP_PARAMS["svd_init"],
        SWEEP_PARAMS["bandpass"],
        SWEEP_PARAMS["common_median_ref"],
    ))
    print(f"Parameter sweep: {len(all_combos)} combinations\n")

    # --- Medium sweep (full) ---
    rec_med = generate_cached("medium")
    print(f"\n  Sweeping {len(all_combos)} combos on medium...")
    med_results = sweep(rec_med, "medium", all_combos)
    print_table(med_results, "MEDIUM -- All combos ranked by accuracy")

    # --- Top-3 on easy and hard ---
    top3_combos = [(r[1], r[2], r[3], r[4], r[5]) for r in med_results[:3]]
    print(f"\n  Top 3 combos from medium to test on easy/hard:")
    for i, (thr, mf, svd, bp, cmr) in enumerate(top3_combos, 1):
        print(f"    #{i}: thr={thr:.1f} mf={mf} svd={svd} bp={bp} cmr={cmr}")

    rec_easy = generate_cached("easy")
    print(f"\n  Running top-3 on easy...")
    easy_results = sweep(rec_easy, "easy", top3_combos)
    print_table(easy_results, "EASY -- Top 3 from medium")

    rec_hard = generate_cached("hard")
    print(f"\n  Running top-3 on hard...")
    hard_results = sweep(rec_hard, "hard", top3_combos)
    print_table(hard_results, "HARD -- Top 3 from medium")

    # Final summary
    print(f"\n{'='*80}")
    print("  CROSS-DIFFICULTY SUMMARY (top 3 combos)")
    print(f"{'='*80}")
    print(f"  {'Combo':>40} {'Easy':>7} {'Medium':>7} {'Hard':>7} {'Avg':>7}")
    print(f"  {'-'*72}")
    for i, (thr, mf, svd, bp, cmr) in enumerate(top3_combos):
        label = f"thr={thr:.1f} mf={mf!s:<5} svd={svd!s:<5} bp={bp!s:<5} cmr={cmr!s:<5}"
        e = easy_results[i][0] if i < len(easy_results) else 0.0
        m = med_results[i][0]
        h = hard_results[i][0] if i < len(hard_results) else 0.0
        # Find correct acc for this combo in easy/hard results
        for r in easy_results:
            if (r[1], r[2], r[3], r[4], r[5]) == (thr, mf, svd, bp, cmr):
                e = r[0]; break
        for r in hard_results:
            if (r[1], r[2], r[3], r[4], r[5]) == (thr, mf, svd, bp, cmr):
                h = r[0]; break
        avg = (e + m + h) / 3.0
        print(f"  {label:>40} {e:>7.4f} {m:>7.4f} {h:>7.4f} {avg:>7.4f}")
    print()


if __name__ == "__main__":
    main()
