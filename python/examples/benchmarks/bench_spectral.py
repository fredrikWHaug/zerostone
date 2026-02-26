"""FFT and Welch PSD benchmarks: zpybci vs numpy/scipy.

Compares power spectrum and PSD estimation across FFT sizes.

Usage:
    python examples/benchmarks/bench_spectral.py
"""
import timeit

import numpy as np
from scipy.signal import welch as scipy_welch

import zpybci as zbci

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FS = 250.0
WARMUP = 3
REPEATS = 5
FFT_SIZES = [256, 512, 1024]


def median_time(stmt, number):
    times = []
    for _ in range(REPEATS):
        t = timeit.timeit(stmt, number=number) / number
        times.append(t)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# FFT power spectrum
# ---------------------------------------------------------------------------

def bench_fft():
    """FFT power spectrum: zpybci Fft vs numpy rfft."""
    print("=== FFT Power Spectrum ===\n")
    print(f"{'Size':<10} {'zpybci (us)':<14} {'numpy (us)':<14} {'Speedup':<10}")
    print("-" * 52)

    for N in FFT_SIZES:
        signal = np.random.randn(N).astype(np.float32)
        signal_f64 = signal.astype(np.float64)
        fft = zbci.Fft(size=N)
        number = 5000

        for _ in range(WARMUP):
            fft.power_spectrum(signal)
            np.abs(np.fft.rfft(signal_f64)) ** 2

        def run_zpybci():
            fft.power_spectrum(signal)

        def run_numpy():
            np.abs(np.fft.rfft(signal_f64)) ** 2

        t_z = median_time(run_zpybci, number)
        t_n = median_time(run_numpy, number)
        speedup = t_n / t_z if t_z > 0 else float("inf")

        print(f"{N:<10} {t_z*1e6:<14.1f} {t_n*1e6:<14.1f} {speedup:.1f}x")

    print()
    return True


# ---------------------------------------------------------------------------
# Welch PSD
# ---------------------------------------------------------------------------

def bench_welch():
    """Welch PSD: zpybci WelchPsd vs scipy welch."""
    print("=== Welch PSD Estimation ===\n")
    print(f"{'FFT size':<10} {'zpybci (us)':<14} {'scipy (us)':<14} {'Speedup':<10}")
    print("-" * 52)

    # Generate a long signal (4x largest FFT size for enough segments)
    max_size = max(FFT_SIZES)
    n_total = max_size * 8
    signal = np.random.randn(n_total).astype(np.float32)
    signal_f64 = signal.astype(np.float64)

    for N in FFT_SIZES:
        welch = zbci.WelchPsd(fft_size=N, window="hann", overlap=0.5)
        number = 1000

        for _ in range(WARMUP):
            welch.estimate(signal, FS)
            scipy_welch(signal_f64, fs=FS, nperseg=N, window="hann",
                        noverlap=N // 2)

        def run_zpybci():
            welch.estimate(signal, FS)

        def run_scipy():
            scipy_welch(signal_f64, fs=FS, nperseg=N, window="hann",
                        noverlap=N // 2)

        t_z = median_time(run_zpybci, number)
        t_s = median_time(run_scipy, number)
        speedup = t_s / t_z if t_z > 0 else float("inf")

        print(f"{N:<10} {t_z*1e6:<14.1f} {t_s*1e6:<14.1f} {speedup:<10.1f}x")

    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("zpybci Spectral Benchmarks")
    print(f"Sample rate: {FS} Hz, Warmup: {WARMUP}, Repeats: {REPEATS}")
    print()
    bench_fft()
    bench_welch()


if __name__ == "__main__":
    main()
