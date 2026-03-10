"""Tests for entropy measures: sample, approximate, spectral, and multiscale entropy."""

import math

import numpy as np
import pytest

import zpybci as zbci


class TestSampleEntropy:
    """Tests for sample_entropy function."""

    def test_constant_signal(self):
        """Constant signal has zero complexity."""
        data = np.ones(100)
        se = zbci.sample_entropy(data)
        assert se < 1e-10, f"Constant signal SampEn should be 0, got {se}"

    def test_random_high_entropy(self):
        """Random data should have high SampEn."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        se = zbci.sample_entropy(data, m=2, r=0.2 * np.std(data))
        assert se > 0.5, f"Random data should have high SampEn, got {se}"

    def test_periodic_moderate_entropy(self):
        """Periodic signal should have moderate SampEn."""
        t = np.arange(200, dtype=np.float64)
        data = np.sin(t * 0.3)
        se = zbci.sample_entropy(data, m=2, r=0.2)
        assert se > 0.0, f"Periodic signal should have SampEn > 0, got {se}"

    def test_regularity_ordering(self):
        """Constant < periodic complexity."""
        constant = np.ones(100)
        t = np.arange(100, dtype=np.float64)
        periodic = np.sin(t * 0.5)
        se_const = zbci.sample_entropy(constant, m=2, r=0.2)
        se_periodic = zbci.sample_entropy(periodic, m=2, r=0.2)
        assert se_const < se_periodic, (
            f"Constant ({se_const}) should have lower SampEn than periodic ({se_periodic})"
        )

    def test_default_params(self):
        """Default m=2, r=0.2 should work."""
        data = np.ones(50)
        se = zbci.sample_entropy(data)
        assert se < 1e-10

    def test_returns_inf(self):
        """Monotonic data with tight tolerance -> inf (no matches)."""
        data = np.arange(50, dtype=np.float64)
        se = zbci.sample_entropy(data, m=2, r=0.001)
        assert math.isinf(se), f"Should return inf, got {se}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            zbci.sample_entropy(np.array([], dtype=np.float64))

    def test_m_zero_raises(self):
        with pytest.raises(ValueError, match="m must be >= 1"):
            zbci.sample_entropy(np.ones(10), m=0)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError, match="r must be > 0"):
            zbci.sample_entropy(np.ones(10), r=0.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="must be >"):
            zbci.sample_entropy(np.ones(3), m=2)


class TestApproximateEntropy:
    """Tests for approximate_entropy function."""

    def test_constant_near_zero(self):
        """Constant signal should have ApEn near 0."""
        data = np.ones(50)
        ae = zbci.approximate_entropy(data)
        assert ae < 0.01, f"Constant signal ApEn should be near 0, got {ae}"

    def test_always_finite(self):
        """ApEn should always be finite, even with tight tolerance."""
        data = np.array([1.0, -1.0] * 25)
        ae = zbci.approximate_entropy(data, m=2, r=0.001)
        assert math.isfinite(ae), f"ApEn should always be finite, got {ae}"

    def test_non_negative(self):
        """ApEn should be non-negative."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0])
        ae = zbci.approximate_entropy(data, m=2, r=0.5)
        assert ae >= 0.0, f"ApEn should be non-negative, got {ae}"

    def test_default_params(self):
        """Default m=2, r=0.2 should work."""
        data = np.ones(50)
        ae = zbci.approximate_entropy(data)
        assert ae < 0.01

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            zbci.approximate_entropy(np.array([], dtype=np.float64))

    def test_m_zero_raises(self):
        with pytest.raises(ValueError, match="m must be >= 1"):
            zbci.approximate_entropy(np.ones(10), m=0)


class TestSpectralEntropy:
    """Tests for spectral_entropy function."""

    def test_flat_normalized(self):
        """Flat spectrum normalized should be 1.0."""
        psd = np.ones(64)
        h = zbci.spectral_entropy(psd)
        assert abs(h - 1.0) < 1e-10, f"Flat PSD normalized should be 1.0, got {h}"

    def test_flat_unnormalized(self):
        """Flat spectrum unnormalized should be ln(N)."""
        n = 64
        psd = np.ones(n)
        h = zbci.spectral_entropy(psd, normalize=False)
        expected = math.log(n)
        assert abs(h - expected) < 1e-10, (
            f"Flat PSD unnormalized should be ln({n})={expected}, got {h}"
        )

    def test_single_peak(self):
        """Single peak -> near 0."""
        psd = np.zeros(64)
        psd[10] = 100.0
        h = zbci.spectral_entropy(psd)
        assert h < 0.01, f"Single peak should give near 0, got {h}"

    def test_two_equal_peaks(self):
        """Two equal peaks -> ln(2)/ln(N)."""
        n = 64
        psd = np.zeros(n)
        psd[10] = 1.0
        psd[30] = 1.0
        h = zbci.spectral_entropy(psd)
        expected = math.log(2) / math.log(n)
        assert abs(h - expected) < 1e-10, (
            f"Two equal peaks should give ln(2)/ln(N)={expected}, got {h}"
        )

    def test_normalized_range(self):
        """Normalized spectral entropy should be in [0, 1]."""
        psd = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        h = zbci.spectral_entropy(psd)
        assert 0.0 <= h <= 1.0, f"Normalized spectral entropy should be in [0,1], got {h}"

    def test_default_normalize_true(self):
        """Default normalize=True."""
        psd = np.ones(64)
        h = zbci.spectral_entropy(psd)
        assert abs(h - 1.0) < 1e-10

    def test_all_zero(self):
        """All-zero PSD -> 0."""
        psd = np.zeros(32)
        h = zbci.spectral_entropy(psd)
        assert abs(h) < 1e-10, f"All-zero PSD should give 0, got {h}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            zbci.spectral_entropy(np.array([], dtype=np.float64))

    def test_integration_welch_sinusoid(self):
        """Sinusoid through Welch PSD should have low spectral entropy."""
        # A pure sinusoid has most energy in one bin -> low entropy
        fs = 256.0
        t = np.arange(1024) / fs
        signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        welch = zbci.WelchPsd(fft_size=256)
        freqs, psd_f32 = welch.estimate(signal, fs)
        psd = psd_f32.astype(np.float64)
        h = zbci.spectral_entropy(psd)
        assert h < 0.5, f"Sinusoid should have low spectral entropy, got {h}"


class TestMultiscaleEntropy:
    """Tests for multiscale_entropy function."""

    def test_scale1_equals_sample_entropy(self):
        """Scale 1 should equal sample_entropy."""
        data = np.tile([1.0, 2.0], 20)
        mse1 = zbci.multiscale_entropy(data, scale=1)
        se = zbci.sample_entropy(data)
        assert abs(mse1 - se) < 1e-10, (
            f"MSE scale=1 ({mse1}) should equal SampEn ({se})"
        )

    def test_constant_zero(self):
        """Constant signal -> 0 at all scales."""
        data = np.ones(100)
        for scale in [1, 2, 4]:
            mse = zbci.multiscale_entropy(data, scale=scale)
            assert mse < 1e-10, (
                f"Constant signal MSE at scale={scale} should be 0, got {mse}"
            )

    def test_non_negative(self):
        """MSE should be non-negative."""
        t = np.arange(100, dtype=np.float64)
        data = np.sin(t * 0.3)
        mse = zbci.multiscale_entropy(data, scale=2, m=2, r=0.2)
        assert mse >= 0.0, f"MSE should be non-negative, got {mse}"

    def test_default_params(self):
        """Default scale=1, m=2, r=0.2 should work."""
        data = np.ones(50)
        mse = zbci.multiscale_entropy(data)
        assert mse < 1e-10

    def test_scale_zero_raises(self):
        with pytest.raises(ValueError, match="scale must be >= 1"):
            zbci.multiscale_entropy(np.ones(100), scale=0)

    def test_too_short_after_coarsening_raises(self):
        with pytest.raises(ValueError, match="too short|must be >"):
            zbci.multiscale_entropy(np.ones(10), scale=5, m=2)
