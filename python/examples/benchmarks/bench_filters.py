"""IIR filter benchmarks: zpybci vs scipy.

Compares butterworth bandpass and notch filters across signal lengths
and channel counts. Reports wall-clock times and speedup factors.

Usage:
    python examples/benchmarks/bench_filters.py
"""
import timeit

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt

import zpybci as zbci

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FS = 250.0
WARMUP = 3
REPEATS = 5
# Chunk sizes typical in real-time BCI (32-250 samples at 250 Hz)
BANDPASS_CHUNK_SIZES = [32, 64, 250, 2500]
NOTCH_CHANNELS = [1, 8, 32, 64]
NOTCH_CHUNK = 32  # samples -- typical real-time chunk


def median_time(stmt, number):
    """Run stmt REPEATS times, return median wall-clock seconds per call."""
    times = []
    for _ in range(REPEATS):
        t = timeit.timeit(stmt, number=number) / number
        times.append(t)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Bandpass benchmarks
# ---------------------------------------------------------------------------

def bench_bandpass():
    """Bandpass filter: zpybci IirFilter vs scipy sosfilt."""
    print("=== Bandpass Filter (4th-order Butterworth, 8-30 Hz) ===\n")
    print(f"{'Chunk':<12} {'Samples':<10} {'zpybci (us)':<14} {'scipy (us)':<14} {'Speedup':<10}")
    print("-" * 60)

    sos = butter(4, [8, 30], btype="bandpass", fs=FS, output="sos")

    for n_samples in BANDPASS_CHUNK_SIZES:
        signal = np.random.randn(n_samples).astype(np.float32)
        signal_f64 = signal.astype(np.float64)

        filt = zbci.IirFilter.butterworth_bandpass(FS, 8.0, 30.0)
        number = max(100, int(50000 / n_samples))

        # Warmup
        for _ in range(WARMUP):
            filt.reset()
            filt.process(signal)
            sosfilt(sos, signal_f64)

        def run_zpybci():
            filt.process(signal)

        def run_scipy():
            sosfilt(sos, signal_f64)

        t_z = median_time(run_zpybci, number)
        t_s = median_time(run_scipy, number)
        speedup = t_s / t_z if t_z > 0 else float("inf")

        label = f"{n_samples} samp"
        print(f"{label:<12} {n_samples:<10} {t_z*1e6:<14.1f} {t_s*1e6:<14.1f} {speedup:.1f}x")

    print()
    return True


# ---------------------------------------------------------------------------
# Notch benchmarks
# ---------------------------------------------------------------------------

def bench_notch():
    """Notch filter: zpybci NotchFilter vs scipy sosfilt."""
    print("=== Notch Filter (60 Hz + harmonics, real-time chunks) ===\n")
    print(f"{'Channels':<12} {'zpybci (us)':<14} {'scipy (us)':<14} {'Speedup':<10}")
    print("-" * 50)

    n_samples = NOTCH_CHUNK

    # scipy: 60 Hz fundamental + harmonic at 120 Hz (both below Nyquist)
    notch_freqs = [f for f in [60.0, 120.0] if f < FS / 2]
    sos_list = []
    for freq in notch_freqs:
        b, a = iirnotch(freq, 30.0, FS)
        from scipy.signal import tf2sos
        sos_list.append(tf2sos(b, a))

    for C in NOTCH_CHANNELS:
        data = np.random.randn(n_samples, C).astype(np.float32)
        data_f64 = data.astype(np.float64)

        notch = zbci.NotchFilter.powerline_60hz(FS, channels=C)
        number = 2000

        # Warmup
        for _ in range(WARMUP):
            notch.process(data)
            tmp = data_f64.copy()
            for sos in sos_list:
                sosfilt(sos, tmp, axis=0)

        def run_zpybci():
            notch.process(data)

        def run_scipy():
            for sos in sos_list:
                sosfilt(sos, data_f64, axis=0)

        t_z = median_time(run_zpybci, number)
        t_s = median_time(run_scipy, number)
        speedup = t_s / t_z if t_z > 0 else float("inf")

        print(f"{C:<12} {t_z*1e6:<14.1f} {t_s*1e6:<14.1f} {speedup:.1f}x")

    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("zpybci Filter Benchmarks")
    print(f"Sample rate: {FS} Hz, Warmup: {WARMUP}, Repeats: {REPEATS}")
    print()
    bench_bandpass()
    bench_notch()


if __name__ == "__main__":
    main()
