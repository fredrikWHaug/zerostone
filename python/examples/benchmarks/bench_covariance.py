"""Online covariance benchmarks: zpybci vs numpy.

Compares streaming covariance estimation across channel counts.

Usage:
    python examples/benchmarks/bench_covariance.py
"""
import timeit

import numpy as np

import zpybci as zbci

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 3
REPEATS = 5
CHANNEL_COUNTS = [4, 8, 16, 32, 64]
N_SAMPLES = 1000


def median_time(stmt, number):
    times = []
    for _ in range(REPEATS):
        t = timeit.timeit(stmt, number=number) / number
        times.append(t)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Covariance benchmark
# ---------------------------------------------------------------------------

def bench_covariance():
    """Online covariance: zpybci OnlineCov vs numpy.cov.

    This is an apples-to-oranges comparison by design: OnlineCov is a
    streaming estimator that processes one sample at a time (O(1) memory),
    while numpy.cov requires the full data matrix. We report both to show
    the cost of streaming vs batch.
    """
    print("=== Covariance Estimation (streaming vs batch) ===\n")
    print(f"{'Channels':<12} {'zpybci (us)':<14} {'numpy (us)':<14} {'Ratio':<10} {'Max err'}")
    print("-" * 64)

    for C in CHANNEL_COUNTS:
        data = np.random.randn(N_SAMPLES, C)
        number = 50

        # Warmup
        for _ in range(WARMUP):
            cov = zbci.OnlineCov(channels=C)
            for i in range(N_SAMPLES):
                cov.update(data[i])
            np.cov(data, rowvar=False)

        def run_zpybci():
            cov_obj = zbci.OnlineCov(channels=C)
            for i in range(N_SAMPLES):
                cov_obj.update(data[i])
            return cov_obj.covariance

        def run_numpy():
            return np.cov(data, rowvar=False)

        t_z = median_time(run_zpybci, number)
        t_n = median_time(run_numpy, number)
        ratio = t_z / t_n if t_n > 0 else float("inf")

        # Numerical comparison
        out_z = run_zpybci()
        out_n = run_numpy()
        max_err = float(np.max(np.abs(out_z - out_n)))

        print(f"{C:<12} {t_z*1e6:<14.1f} {t_n*1e6:<14.1f} {ratio:.0f}x{'':<6} {max_err:.2e}")

    print()
    print("Note: zpybci OnlineCov processes samples one at a time (O(1) memory,")
    print("suitable for real-time streaming). numpy.cov requires the full data")
    print("matrix. The ratio reflects FFI overhead per sample, not algorithmic")
    print("inefficiency.")
    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("zpybci Covariance Benchmarks")
    print(f"Samples: {N_SAMPLES}, Warmup: {WARMUP}, Repeats: {REPEATS}")
    print()
    bench_covariance()


if __name__ == "__main__":
    main()
