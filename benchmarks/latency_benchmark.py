"""Latency benchmark for Zerostone spike sorting.

Measures per-spike classification time for both the batch pipeline
(sort_multichannel) and the online classifier (OnlineSorter).

Target: <100 microseconds per spike for the online path.

Usage:
    python benchmarks/latency_benchmark.py
"""

import time
import numpy as np

# Try importing zpybci
try:
    import zpybci as zbci
    from zpybci.zpybci import ProbeLayout, sort_multichannel, OnlineSorter

    HAS_ZBCI = True
except ImportError:
    HAS_ZBCI = False
    print("zpybci not installed. Install with: pip install zpybci")

try:
    from zpybci.synthetic import generate_recording

    HAS_SYNTHETIC = True
except ImportError:
    HAS_SYNTHETIC = False


def bench_online_sorter(n_templates, n_spikes, n_features=3):
    """Benchmark OnlineSorter.classify() latency."""
    sorter = OnlineSorter()

    # Add random templates
    rng = np.random.default_rng(42)
    for _ in range(n_templates):
        template = rng.standard_normal(n_features).tolist()
        sorter.add_template(template)

    # Generate random feature vectors
    features = rng.standard_normal((n_spikes, n_features))

    # Warm up
    for i in range(min(100, n_spikes)):
        sorter.classify(features[i].tolist())
    sorter.reset_counters()

    # Timed run
    start = time.perf_counter_ns()
    for i in range(n_spikes):
        sorter.classify(features[i].tolist())
    elapsed_ns = time.perf_counter_ns() - start

    per_spike_us = elapsed_ns / n_spikes / 1000.0
    total_ms = elapsed_ns / 1e6
    return per_spike_us, total_ms


def bench_batch_sorter(n_channels, duration_s, sampling_rate=30000.0):
    """Benchmark sort_multichannel() throughput."""
    if not HAS_SYNTHETIC:
        return None, None, None

    rec = generate_recording(
        n_channels=n_channels,
        n_units=5,
        duration_s=duration_s,
        sampling_rate=sampling_rate,
        noise_std=1.0,
        seed=42,
    )
    data = rec["data"]  # (n_samples, n_channels)
    probe = ProbeLayout.linear(n_channels, 25.0)

    # Warm up
    small = data[:1000]
    try:
        sort_multichannel(small, probe, threshold=5.0)
    except Exception:
        pass

    # Timed run
    start = time.perf_counter_ns()
    result = sort_multichannel(data, probe, threshold=5.0)
    elapsed_ns = time.perf_counter_ns() - start

    n_spikes = result["n_spikes"]
    total_ms = elapsed_ns / 1e6
    n_samples = data.shape[0]
    per_spike_us = (elapsed_ns / max(n_spikes, 1)) / 1000.0
    throughput_mhz = n_samples / (elapsed_ns / 1e9) / 1e6

    return total_ms, per_spike_us, throughput_mhz


def main():
    if not HAS_ZBCI:
        return

    print("=" * 60)
    print("Zerostone Latency Benchmark")
    print("=" * 60)

    # --- Online sorter latency ---
    print("\n--- Online Sorter (per-spike classify) ---")
    print(
        f"{'Templates':>10} {'Spikes':>10} {'Per-spike (us)':>15} {'Total (ms)':>12}"
    )
    print("-" * 50)

    for n_templates in [2, 5, 10, 16]:
        for n_spikes in [1000, 10000]:
            per_spike_us, total_ms = bench_online_sorter(n_templates, n_spikes)
            print(
                f"{n_templates:>10} {n_spikes:>10} "
                f"{per_spike_us:>15.2f} {total_ms:>12.1f}"
            )

    # --- Batch sorter throughput ---
    if HAS_SYNTHETIC:
        print("\n--- Batch Sorter (sort_multichannel) ---")
        print(
            f"{'Channels':>10} {'Duration':>10} {'Total (ms)':>12} "
            f"{'Per-spike (us)':>15} {'Throughput':>12}"
        )
        print("-" * 62)

        for n_ch, dur in [(8, 10.0), (16, 5.0), (32, 2.0)]:
            try:
                total_ms, per_spike_us, throughput = bench_batch_sorter(n_ch, dur)
            except Exception as e:
                print(f"{n_ch:>10} {dur:>8.1f}s {'ERROR':>12} -- {e}")
                continue
            if total_ms is not None:
                thr_str = f"{throughput:.2f} MHz" if throughput else "N/A"
                ps_str = f"{per_spike_us:.1f}" if per_spike_us else "N/A"
                print(
                    f"{n_ch:>10} {dur:>8.1f}s {total_ms:>12.1f} "
                    f"{ps_str:>15} {thr_str:>12}"
                )

    print("\n" + "=" * 60)
    print("Target: <100 us per spike (online path)")
    print("=" * 60)


if __name__ == "__main__":
    main()
