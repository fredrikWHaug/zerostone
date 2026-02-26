"""Riemannian operation benchmarks: zpybci vs pyriemann.

Compares TangentSpace projection and Frechet mean across channel counts.
pyriemann benchmarks are skipped gracefully if the package is not installed.

Usage:
    python examples/benchmarks/bench_riemannian.py
"""
import timeit

import numpy as np

import zpybci as zbci

try:
    import pyriemann
    from pyriemann.tangentspace import TangentSpace as PyRiemannTS
    from pyriemann.utils.mean import mean_riemann
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 3
REPEATS = 5
CHANNEL_COUNTS = [4, 8, 16]
N_MATRICES = 20


def median_time(stmt, number):
    times = []
    for _ in range(REPEATS):
        t = timeit.timeit(stmt, number=number) / number
        times.append(t)
    return float(np.median(times))


def make_spd_matrices(n, C):
    """Generate n random SPD matrices of size CxC."""
    matrices = np.empty((n, C, C), dtype=np.float64)
    for i in range(n):
        A = np.random.randn(C, C)
        matrices[i] = A @ A.T + np.eye(C) * 0.1
    return matrices


# ---------------------------------------------------------------------------
# TangentSpace benchmark
# ---------------------------------------------------------------------------

def bench_tangent_space():
    """TangentSpace: zpybci vs pyriemann."""
    print("=== TangentSpace Projection ===\n")

    if HAS_PYRIEMANN:
        print(f"{'Channels':<12} {'zpybci (us)':<14} {'pyriemann (us)':<16} {'Speedup'}")
        print("-" * 56)
    else:
        print("pyriemann not installed -- zpybci-only timings")
        print(f"{'Channels':<12} {'zpybci (us)':<14}")
        print("-" * 28)

    for C in CHANNEL_COUNTS:
        matrices = make_spd_matrices(N_MATRICES, C)
        ref = np.mean(matrices, axis=0)
        number = 200

        # zpybci
        ts = zbci.TangentSpace(channels=C)
        ts.fit(ref)

        for _ in range(WARMUP):
            for i in range(N_MATRICES):
                ts.transform(matrices[i])

        def run_zpybci():
            for i in range(N_MATRICES):
                ts.transform(matrices[i])

        t_z = median_time(run_zpybci, number)

        if HAS_PYRIEMANN:
            pyts = PyRiemannTS(metric="riemann")
            pyts.fit(matrices)

            for _ in range(WARMUP):
                pyts.transform(matrices)

            def run_pyriemann():
                pyts.transform(matrices)

            t_p = median_time(run_pyriemann, number)
            speedup = t_p / t_z if t_z > 0 else float("inf")
            print(f"{C:<12} {t_z*1e6:<14.1f} {t_p*1e6:<16.1f} {speedup:.1f}x")
        else:
            print(f"{C:<12} {t_z*1e6:<14.1f}")

    print()
    return True


# ---------------------------------------------------------------------------
# Frechet mean benchmark
# ---------------------------------------------------------------------------

def bench_frechet_mean():
    """Frechet mean: zpybci vs pyriemann."""
    print("=== Frechet Mean (Riemannian geometric mean) ===\n")

    if HAS_PYRIEMANN:
        print(f"{'Channels':<12} {'zpybci (us)':<14} {'pyriemann (us)':<16} {'Speedup':<10} {'Max err'}")
        print("-" * 66)
    else:
        print("pyriemann not installed -- zpybci-only timings")
        print(f"{'Channels':<12} {'zpybci (us)':<14}")
        print("-" * 28)

    for C in CHANNEL_COUNTS:
        matrices = make_spd_matrices(N_MATRICES, C)
        number = 100

        for _ in range(WARMUP):
            zbci.frechet_mean(matrices)

        def run_zpybci():
            zbci.frechet_mean(matrices)

        t_z = median_time(run_zpybci, number)

        if HAS_PYRIEMANN:
            for _ in range(WARMUP):
                mean_riemann(matrices)

            def run_pyriemann():
                mean_riemann(matrices)

            t_p = median_time(run_pyriemann, number)
            speedup = t_p / t_z if t_z > 0 else float("inf")

            out_z = zbci.frechet_mean(matrices)
            out_p = mean_riemann(matrices)
            max_err = float(np.max(np.abs(out_z - out_p)))

            print(f"{C:<12} {t_z*1e6:<14.1f} {t_p*1e6:<16.1f} {speedup:<10.1f}x {max_err:.2e}")
        else:
            print(f"{C:<12} {t_z*1e6:<14.1f}")

    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("zpybci Riemannian Benchmarks")
    print(f"Matrices: {N_MATRICES}, Warmup: {WARMUP}, Repeats: {REPEATS}")
    if not HAS_PYRIEMANN:
        print("(pyriemann not installed -- comparative benchmarks skipped)")
    print()
    bench_tangent_space()
    bench_frechet_mean()


if __name__ == "__main__":
    main()
