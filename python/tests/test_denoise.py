"""Tests for wavelet denoising functions."""

import numpy as np
import pytest
import zpybci as zbci


class TestDenoiseHaar:
    """Tests for denoise_haar()."""

    def test_basic_denoise(self):
        """Denoising should not crash on simple input."""
        signal = np.random.randn(64)
        result = zbci.denoise_haar(signal)
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))

    def test_zero_input(self):
        """Zero input should stay zero."""
        signal = np.zeros(32)
        result = zbci.denoise_haar(signal)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_reduces_noise(self):
        """Denoising should reduce variance of pure noise."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(256)
        denoised = zbci.denoise_haar(noise, levels=3, mode="soft")
        assert np.var(denoised) < np.var(noise)

    def test_preserves_large_signal(self):
        """Large-amplitude signal should be mostly preserved."""
        t = np.linspace(0, 1, 128)
        signal = 100.0 * np.sin(2 * np.pi * 5 * t)
        denoised = zbci.denoise_haar(signal, levels=3, mode="soft")
        snr = 10 * np.log10(np.sum(signal ** 2) / (np.sum((signal - denoised) ** 2) + 1e-30))
        assert snr > 10.0, f"SNR = {snr:.1f} dB"

    def test_spike_preserved(self):
        """A large spike-like transient should survive denoising."""
        signal = np.zeros(64)
        signal[30] = -4.0
        signal[31] = -8.0
        signal[32] = 3.0
        denoised = zbci.denoise_haar(signal, levels=3, mode="hard")
        assert denoised[31] < -1.0, f"Spike peak should be preserved, got {denoised[31]}"

    def test_soft_mode(self):
        """Soft thresholding mode."""
        signal = np.array([0.0, 5.0, -5.0, 0.1, -0.1, 3.0, -3.0, 0.0])
        result = zbci.denoise_haar(signal, levels=1, mode="soft")
        assert np.all(np.isfinite(result))

    def test_hard_mode(self):
        """Hard thresholding mode."""
        signal = np.array([0.0, 5.0, -5.0, 0.1, -0.1, 3.0, -3.0, 0.0])
        result = zbci.denoise_haar(signal, levels=1, mode="hard")
        assert np.all(np.isfinite(result))

    def test_multiple_levels(self):
        """Multiple decomposition levels should work."""
        signal = np.random.randn(128)
        for levels in [1, 2, 3, 4, 5]:
            result = zbci.denoise_haar(signal, levels=levels)
            assert result.shape == signal.shape
            assert np.all(np.isfinite(result))

    def test_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        signal = np.zeros(16)
        with pytest.raises(ValueError, match="mode must be"):
            zbci.denoise_haar(signal, mode="invalid")

    def test_too_short(self):
        """Signal too short should raise ValueError."""
        with pytest.raises(ValueError):
            zbci.denoise_haar(np.array([1.0]), levels=1)

    def test_zero_levels(self):
        """Zero levels should raise ValueError."""
        with pytest.raises(ValueError, match="levels must be"):
            zbci.denoise_haar(np.zeros(16), levels=0)


class TestSoftThreshold:
    """Tests for soft_threshold()."""

    def test_basic(self):
        """Basic soft thresholding."""
        signal = np.array([3.0, -3.0, 0.5, -0.5, 0.0])
        result = zbci.soft_threshold(signal, 1.0)
        expected = np.array([2.0, -2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_zero_lambda(self):
        """Lambda=0 should return input unchanged."""
        signal = np.array([1.0, -2.0, 0.5])
        result = zbci.soft_threshold(signal, 0.0)
        np.testing.assert_allclose(result, signal, atol=1e-12)

    def test_reduces_magnitude(self):
        """Soft threshold should never increase magnitude."""
        rng = np.random.default_rng(123)
        signal = rng.standard_normal(100)
        result = zbci.soft_threshold(signal, 0.5)
        assert np.all(np.abs(result) <= np.abs(signal) + 1e-12)


class TestHardThreshold:
    """Tests for hard_threshold()."""

    def test_basic(self):
        """Basic hard thresholding."""
        signal = np.array([3.0, -3.0, 0.5, -0.5, 1.0, -1.0])
        result = zbci.hard_threshold(signal, 1.0)
        expected = np.array([3.0, -3.0, 0.0, 0.0, 1.0, -1.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_preserves_large(self):
        """Hard threshold preserves amplitude of large values."""
        signal = np.array([10.0, -10.0])
        result = zbci.hard_threshold(signal, 1.0)
        np.testing.assert_allclose(result, signal, atol=1e-12)


class TestUniversalThreshold:
    """Tests for universal_threshold_py()."""

    def test_basic(self):
        """Universal threshold for sigma=1, n=100."""
        lam = zbci.universal_threshold_py(1.0, 100)
        assert abs(lam - np.sqrt(2 * np.log(100))) < 0.01

    def test_scales_with_sigma(self):
        """Should scale linearly with sigma."""
        l1 = zbci.universal_threshold_py(1.0, 100)
        l2 = zbci.universal_threshold_py(2.0, 100)
        assert abs(l2 - 2.0 * l1) < 1e-10

    def test_n_one(self):
        """n=1 should return 0."""
        assert zbci.universal_threshold_py(1.0, 1) == 0.0
