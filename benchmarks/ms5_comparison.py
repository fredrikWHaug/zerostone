"""MountainSort5 vs Zerostone comparison benchmark.

Runs both sorters on the same synthetic datasets (easy, medium, hard)
using the SpikeInterface comparison framework, and reports accuracy,
precision, recall, and timing for each.

MountainSort5 uses sorting_scheme2 (sklearn-based clustering, does not
require isosplit6).

Usage:
    python benchmarks/ms5_comparison.py
    python benchmarks/ms5_comparison.py --difficulty easy
    python benchmarks/ms5_comparison.py --all
"""

import argparse
import sys
import time
import types

import numpy as np

# isosplit6 C++ extension may not compile on all platforms.
# Use isosplit5 (pure-Python compatible) as a drop-in replacement.
if "isosplit6" not in sys.modules:
    _stub = types.ModuleType("isosplit6")
    try:
        from isosplit5 import isosplit5 as _isosplit5_fn

        _stub.isosplit6 = _isosplit5_fn
    except ImportError:
        _stub.isosplit6 = None
    sys.modules["isosplit6"] = _stub

try:
    import spikeinterface.core as si
    from spikeinterface.comparison import compare_sorter_to_ground_truth
    from spikeinterface.core import NumpySorting

    HAS_SI = True
except ImportError:
    HAS_SI = False

try:
    import mountainsort5 as ms5

    HAS_MS5 = True
except ImportError:
    HAS_MS5 = False

try:
    import zpybci as zbci
    from zpybci.zpybci import ProbeLayout, sort_multichannel

    HAS_ZBCI = True
except ImportError:
    HAS_ZBCI = False

# Use our own synthetic generator (no SpikeInterface dependency needed)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "python"))
from zpybci.synthetic import generate_recording

# Reuse accuracy computation from our benchmark
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from accuracy_benchmark import PRESETS, DEFAULT_TOLERANCE, compute_accuracy

SAMPLING_RATE = 30000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_si_recording(data, sampling_rate=SAMPLING_RATE):
    """Wrap a numpy array as a SpikeInterface NumpyRecording with linear probe."""
    import probeinterface

    n_channels = data.shape[1]
    recording = si.NumpyRecording(
        traces_list=[data],
        sampling_frequency=sampling_rate,
    )
    # Attach a linear probe so MS5 has channel locations
    probe = probeinterface.generate_linear_probe(num_elec=n_channels, ypitch=25.0)
    probe.set_device_channel_indices(np.arange(n_channels))
    recording = recording.set_probe(probe)
    return recording


def make_si_gt_sorting(spike_times_per_unit, sampling_rate=SAMPLING_RATE):
    """Build a SpikeInterface NumpySorting from per-unit spike times."""
    all_times = []
    all_labels = []
    for u, times in enumerate(spike_times_per_unit):
        all_times.append(times)
        all_labels.append(np.full(len(times), u, dtype=np.int64))

    if len(all_times) > 0:
        all_times = np.concatenate(all_times)
        all_labels = np.concatenate(all_labels)
        order = np.argsort(all_times)
        all_times = all_times[order]
        all_labels = all_labels[order]
    else:
        all_times = np.array([], dtype=np.int64)
        all_labels = np.array([], dtype=np.int64)

    return NumpySorting.from_samples_and_labels(
        samples_list=[all_times],
        labels_list=[all_labels],
        sampling_frequency=sampling_rate,
    )


# ---------------------------------------------------------------------------
# Sorter runners
# ---------------------------------------------------------------------------
def run_ms5(data, sampling_rate=SAMPLING_RATE):
    """Run MountainSort5 scheme2 on numpy data array.

    Returns (sorting, elapsed_s).
    """
    recording = make_si_recording(data, sampling_rate)

    t0 = time.perf_counter()
    sorting = ms5.sorting_scheme2(
        recording=recording,
        sorting_parameters=ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=100,
            detect_channel_radius=100,
            phase1_detect_threshold=5.0,
            detect_threshold=5.0,
        ),
    )
    elapsed = time.perf_counter() - t0
    return sorting, elapsed


def run_zerostone(data, n_channels):
    """Run Zerostone sort_multichannel on numpy data array.

    Returns (sorting_si, elapsed_s) where sorting_si is a NumpySorting.
    """
    probe = ProbeLayout.linear(n_channels, 25.0)

    t0 = time.perf_counter()
    result = sort_multichannel(data, probe, threshold=5.0, refractory=15)
    elapsed = time.perf_counter() - t0

    n_spikes = result["n_spikes"]
    labels = np.array(result["labels"][:n_spikes], dtype=np.int64)

    # We need spike times -- run detection on raw data as approximation
    noise = np.zeros(n_channels, dtype=np.float64)
    for ch in range(n_channels):
        noise[ch] = zbci.estimate_noise_mad(np.ascontiguousarray(data[:, ch]))

    from zpybci.zpybci import detect_spikes_multichannel

    events = detect_spikes_multichannel(data, 5.0, noise, 15)
    det_times = np.array([e["sample"] for e in events], dtype=np.int64)

    n_use = min(n_spikes, len(det_times))
    spike_times = det_times[:n_use]
    spike_labels = labels[:n_use]

    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[spike_times],
        labels_list=[spike_labels],
        sampling_frequency=SAMPLING_RATE,
    )
    return sorting, elapsed


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_sorter(sorting, gt_sorting, sampling_rate=SAMPLING_RATE):
    """Compare a sorting result to ground truth using SpikeInterface."""
    comp = compare_sorter_to_ground_truth(
        gt_sorting=gt_sorting,
        tested_sorting=sorting,
        exhaustive_gt=True,
        delta_time=0.4 / 1000.0,  # 0.4 ms tolerance
    )
    perf = comp.get_performance()
    return perf


def run_comparison(preset_name, seed=42, verbose=True):
    """Run both sorters on the same synthetic dataset and compare."""
    params = PRESETS[preset_name]
    n_ch = params["n_channels"]

    if verbose:
        print(f"\n{'='*65}")
        print(f"  {preset_name.upper()} -- {n_ch}ch, {params['n_units']} units, "
              f"noise={params['noise_std']}, rate={params['firing_rate']} Hz")
        print(f"{'='*65}")

    # Generate recording
    if verbose:
        print("  Generating synthetic recording...", end=" ", flush=True)
    rec = generate_recording(
        n_channels=n_ch,
        duration_s=params["duration_s"],
        sampling_rate=SAMPLING_RATE,
        n_units=params["n_units"],
        firing_rate=params["firing_rate"],
        noise_std=params["noise_std"],
        seed=seed,
    )
    data = rec["data"]
    n_gt = len(rec["all_spike_times"])
    if verbose:
        print(f"done ({n_gt} GT spikes)")

    # Build ground truth sorting
    gt_sorting = make_si_gt_sorting(rec["spike_times"], SAMPLING_RATE)

    results = {}

    # --- MountainSort5 ---
    if HAS_MS5:
        if verbose:
            print("  Running MountainSort5 (scheme2)...", end=" ", flush=True)
        try:
            ms5_sorting, ms5_time = run_ms5(data, SAMPLING_RATE)
            ms5_units = len(ms5_sorting.get_unit_ids())
            if verbose:
                print(f"done ({ms5_time:.1f}s, {ms5_units} clusters)")

            if verbose:
                print("  Computing MS5 accuracy...", end=" ", flush=True)
            ms5_perf = compare_sorter(ms5_sorting, gt_sorting, SAMPLING_RATE)
            if verbose:
                print("done")
            results["ms5"] = {
                "time_s": ms5_time,
                "n_clusters": ms5_units,
                "performance": ms5_perf,
            }
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results["ms5"] = {"error": str(e)}
    else:
        if verbose:
            print("  MountainSort5 not available -- skipping")

    # --- Zerostone ---
    if HAS_ZBCI:
        if verbose:
            print("  Running Zerostone...", end=" ", flush=True)
        try:
            zs_sorting, zs_time = run_zerostone(data, n_ch)
            zs_units = len(zs_sorting.get_unit_ids())
            if verbose:
                print(f"done ({zs_time:.1f}s, {zs_units} clusters)")

            if verbose:
                print("  Computing Zerostone accuracy...", end=" ", flush=True)
            zs_perf = compare_sorter(zs_sorting, gt_sorting, SAMPLING_RATE)
            if verbose:
                print("done")
            results["zerostone"] = {
                "time_s": zs_time,
                "n_clusters": zs_units,
                "performance": zs_perf,
            }
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results["zerostone"] = {"error": str(e)}

    return {
        "preset": preset_name,
        "params": params,
        "n_gt_spikes": n_gt,
        "n_gt_units": params["n_units"],
        "results": results,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_comparison(comparison):
    """Print per-unit accuracy comparison between sorters."""
    preset = comparison["preset"]
    n_gt = comparison["n_gt_spikes"]
    n_units = comparison["n_gt_units"]

    for sorter_name, res in comparison["results"].items():
        if "error" in res:
            print(f"\n  {sorter_name}: ERROR -- {res['error']}")
            continue

        perf = res["performance"]
        print(f"\n  {sorter_name.upper()} -- {res['n_clusters']} clusters, "
              f"{res['time_s']:.2f}s")

        # Per-unit metrics
        print(f"  {'Unit':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
        print(f"  {'-'*32}")

        for idx in perf.index:
            acc = perf.loc[idx, "accuracy"]
            prec = perf.loc[idx, "precision"]
            rec = perf.loc[idx, "recall"]
            print(f"  {idx:>6} {acc:>8.3f} {prec:>8.3f} {rec:>8.3f}")

        # Mean metrics
        mean_acc = perf["accuracy"].mean()
        mean_prec = perf["precision"].mean()
        mean_rec = perf["recall"].mean()
        print(f"  {'-'*32}")
        print(f"  {'MEAN':>6} {mean_acc:>8.3f} {mean_prec:>8.3f} {mean_rec:>8.3f}")


def print_summary_table(comparisons):
    """Print compact comparison table across all difficulties."""
    print(f"\n{'='*65}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Preset':<8} {'Sorter':<14} {'Clusters':>8} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'Time':>8}")
    print(f"  {'-'*60}")

    for comp in comparisons:
        preset = comp["preset"]
        for sorter_name, res in comp["results"].items():
            if "error" in res:
                print(f"  {preset:<8} {sorter_name:<14} {'ERR':>8} "
                      f"{'---':>7} {'---':>7} {'---':>7} {'---':>8}")
                continue

            perf = res["performance"]
            mean_acc = perf["accuracy"].mean()
            mean_prec = perf["precision"].mean()
            mean_rec = perf["recall"].mean()
            print(f"  {preset:<8} {sorter_name:<14} {res['n_clusters']:>8} "
                  f"{mean_acc:>7.3f} {mean_prec:>7.3f} {mean_rec:>7.3f} "
                  f"{res['time_s']:>7.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MountainSort5 vs Zerostone comparison benchmark.",
    )
    parser.add_argument("--difficulty", choices=list(PRESETS.keys()), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_SI:
        print("ERROR: spikeinterface is required. pip install spikeinterface")
        sys.exit(1)

    if args.all:
        presets = list(PRESETS.keys())
    elif args.difficulty:
        presets = [args.difficulty]
    else:
        presets = ["easy"]

    comparisons = []
    for preset in presets:
        comp = run_comparison(preset, seed=args.seed)
        print_comparison(comp)
        comparisons.append(comp)

    if len(comparisons) > 1:
        print_summary_table(comparisons)

    print()


if __name__ == "__main__":
    main()
