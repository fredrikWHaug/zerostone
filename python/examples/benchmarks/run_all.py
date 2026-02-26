"""Run all zpybci performance benchmarks and produce a combined summary.

Usage:
    python examples/benchmarks/run_all.py
"""
import sys
import os
import timeit

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from scipy.signal import welch as scipy_welch

import zpybci as zbci

try:
    from pyriemann.tangentspace import TangentSpace as PyRiemannTS
    from pyriemann.utils.mean import mean_riemann
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FS = 250.0
WARMUP = 3
REPEATS = 5


def median_time(stmt, number):
    times = []
    for _ in range(REPEATS):
        t = timeit.timeit(stmt, number=number) / number
        times.append(t)
    return float(np.median(times))


def make_spd_matrices(n, C):
    matrices = np.empty((n, C, C), dtype=np.float64)
    for i in range(n):
        A = np.random.randn(C, C)
        matrices[i] = A @ A.T + np.eye(C) * 0.1
    return matrices


# ---------------------------------------------------------------------------
# Individual benchmarks (return rows for the summary table)
# ---------------------------------------------------------------------------

def bench_bandpass_rows():
    sos = butter(4, [8, 30], btype="bandpass", fs=FS, output="sos")
    rows = []
    for n in [32, 64, 250, 2500]:
        sig = np.random.randn(n).astype(np.float32)
        sig64 = sig.astype(np.float64)
        filt = zbci.IirFilter.butterworth_bandpass(FS, 8.0, 30.0)
        number = max(100, int(50000 / n))
        for _ in range(WARMUP):
            filt.process(sig); sosfilt(sos, sig64)
        def rz():
            filt.process(sig)
        def rs():
            sosfilt(sos, sig64)
        tz = median_time(rz, number)
        ts = median_time(rs, number)
        rows.append((f"Bandpass {n}samp", tz, ts, "scipy"))
    return rows


def bench_notch_rows():
    notch_freqs = [f for f in [60.0, 120.0] if f < FS / 2]
    sos_list = [tf2sos(*iirnotch(f, 30.0, FS)) for f in notch_freqs]
    rows = []
    for C in [1, 8, 64]:
        data = np.random.randn(32, C).astype(np.float32)
        data64 = data.astype(np.float64)
        notch = zbci.NotchFilter.powerline_60hz(FS, channels=C)
        number = 2000
        for _ in range(WARMUP):
            notch.process(data)
            for s in sos_list: sosfilt(s, data64, axis=0)
        def rz():
            notch.process(data)
        def rs():
            for s in sos_list: sosfilt(s, data64, axis=0)
        tz = median_time(rz, number)
        ts = median_time(rs, number)
        rows.append((f"Notch {C}ch", tz, ts, "scipy"))
    return rows


def bench_fft_rows():
    rows = []
    for N in [256, 512, 1024]:
        sig = np.random.randn(N).astype(np.float32)
        sig64 = sig.astype(np.float64)
        fft = zbci.Fft(size=N)
        number = 5000
        for _ in range(WARMUP):
            fft.power_spectrum(sig); np.abs(np.fft.rfft(sig64))**2
        def rz():
            fft.power_spectrum(sig)
        def rn():
            np.abs(np.fft.rfft(sig64))**2
        tz = median_time(rz, number)
        tn = median_time(rn, number)
        rows.append((f"FFT {N}", tz, tn, "numpy"))
    return rows


def bench_welch_rows():
    rows = []
    for N in [256, 512, 1024]:
        sig = np.random.randn(N * 8).astype(np.float32)
        sig64 = sig.astype(np.float64)
        welch = zbci.WelchPsd(fft_size=N, window="hann", overlap=0.5)
        number = 1000
        for _ in range(WARMUP):
            welch.estimate(sig, FS)
            scipy_welch(sig64, fs=FS, nperseg=N, window="hann", noverlap=N//2)
        def rz():
            welch.estimate(sig, FS)
        def rs():
            scipy_welch(sig64, fs=FS, nperseg=N, window="hann", noverlap=N//2)
        tz = median_time(rz, number)
        ts = median_time(rs, number)
        rows.append((f"Welch {N}", tz, ts, "scipy"))
    return rows


def bench_riemannian_rows():
    rows = []
    for C in [4, 8, 16]:
        matrices = make_spd_matrices(20, C)
        number = 100
        for _ in range(WARMUP):
            zbci.frechet_mean(matrices)
        def rz():
            zbci.frechet_mean(matrices)
        tz = median_time(rz, number)

        if HAS_PYRIEMANN:
            for _ in range(WARMUP):
                mean_riemann(matrices)
            def rp():
                mean_riemann(matrices)
            tp = median_time(rp, number)
            rows.append((f"Frechet mean {C}ch", tz, tp, "pyriemann"))
        else:
            rows.append((f"Frechet mean {C}ch", tz, None, "pyriemann"))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("zpybci Performance Benchmark Suite")
    print("=" * 70)
    print()

    # Run individual benchmarks with full output
    sys.path.insert(0, os.path.dirname(__file__))
    from bench_filters import main as run_filters
    from bench_spectral import main as run_spectral
    from bench_covariance import main as run_cov
    from bench_riemannian import main as run_riem

    run_filters()
    run_spectral()
    run_cov()
    run_riem()

    # Combined summary table
    print("=" * 70)
    print("Combined Summary")
    print("=" * 70)
    print()

    all_rows = []
    all_rows.extend(bench_bandpass_rows())
    all_rows.extend(bench_notch_rows())
    all_rows.extend(bench_fft_rows())
    all_rows.extend(bench_welch_rows())
    all_rows.extend(bench_riemannian_rows())

    # Markdown table
    print("| Operation | zpybci (us) | Baseline (us) | Baseline | Speedup |")
    print("|-----------|-------------|---------------|----------|---------|")

    for name, tz, tb, baseline in all_rows:
        tz_us = f"{tz*1e6:.1f}"
        if tb is not None:
            tb_us = f"{tb*1e6:.1f}"
            speedup = tb / tz if tz > 0 else float("inf")
            sp = f"{speedup:.1f}x"
        else:
            tb_us = "N/A"
            sp = "N/A"
        print(f"| {name:<20} | {tz_us:>11} | {tb_us:>13} | {baseline:<8} | {sp:>7} |")

    print()


if __name__ == "__main__":
    main()
