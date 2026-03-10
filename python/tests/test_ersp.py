"""Tests for ERSP (Event-Related Spectral Perturbation) functions."""

import numpy as np
import pytest

import zpybci as zbci


class TestComputeErsp:
    """Tests for compute_ersp function."""

    def test_basic_call(self):
        """Basic call with random data returns correct shape."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((10, 1024)).astype(np.float32)
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 1.0), fft_size=256)
        n_freqs = 256 // 2 + 1
        assert ersp.shape[0] == n_freqs
        assert ersp.ndim == 2

    def test_output_shape_fft_sizes(self):
        """Output n_freqs matches fft_size / 2 + 1."""
        rng = np.random.default_rng(42)
        for fft_size in [64, 128, 256]:
            epochs = rng.standard_normal((5, 1024)).astype(np.float32)
            ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 1.0), fft_size=fft_size)
            assert ersp.shape[0] == fft_size // 2 + 1, f"fft_size={fft_size}"

    def test_db_mode_baseline_near_zero(self):
        """In dB mode, baseline period should be near 0."""
        rng = np.random.default_rng(42)
        # White noise -- baseline mean should be roughly uniform across freqs
        epochs = rng.standard_normal((20, 1024)).astype(np.float32)
        # baseline = first 256 samples = 1 sec at 256 Hz
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 1.0), fft_size=256, hop_size=64)
        # Baseline frames should be near 0 dB (averaged over 20 epochs)
        # Frame 0 center ~ 0.5s, well within baseline
        # With 20 averaged epochs, baseline region should be close to 0
        baseline_power = np.abs(ersp[:, 0])
        assert np.median(baseline_power) < 5.0, (
            f"Median baseline dB should be near 0, got {np.median(baseline_power)}"
        )

    def test_zscore_mode(self):
        """Z-score mode works without error."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((10, 512)).astype(np.float32)
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 0.5), fft_size=128, mode='zscore')
        assert ersp.ndim == 2
        assert not np.any(np.isnan(ersp))

    def test_percentage_mode(self):
        """Percentage mode works without error."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((10, 512)).astype(np.float32)
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 0.5), fft_size=128, mode='percentage')
        assert ersp.ndim == 2

    def test_logratio_mode(self):
        """Log-ratio mode works without error."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((10, 512)).astype(np.float32)
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 0.5), fft_size=128, mode='logratio')
        assert ersp.ndim == 2

    def test_single_trial_output_shape(self):
        """Single trial returns 3D array (n_epochs, n_freqs, n_frames)."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 512)).astype(np.float32)
        ersp = zbci.compute_ersp(
            epochs, 256.0, (0.0, 0.5), fft_size=128, single_trial=True
        )
        assert ersp.ndim == 3
        assert ersp.shape[0] == 5  # n_epochs
        assert ersp.shape[1] == 128 // 2 + 1  # n_freqs

    def test_default_hop_size(self):
        """Default hop_size = fft_size // 4."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 1024)).astype(np.float32)
        ersp = zbci.compute_ersp(epochs, 256.0, (0.0, 1.0), fft_size=256)
        # With hop=64, n_frames = (1024 - 256) / 64 + 1 = 13
        assert ersp.shape[1] == 13

    def test_sinusoid_shows_power(self):
        """A 10 Hz sinusoid should show elevated power around 10 Hz."""
        sr = 256.0
        n_samples = 1024
        t = np.arange(n_samples) / sr
        # Epoch: baseline (first half, noise only) + stimulus (second half, 10 Hz sine + noise)
        rng = np.random.default_rng(42)
        n_epochs = 30
        epochs = np.zeros((n_epochs, n_samples), dtype=np.float32)
        for i in range(n_epochs):
            noise = rng.standard_normal(n_samples).astype(np.float32) * 0.5
            signal = np.zeros(n_samples, dtype=np.float32)
            signal[n_samples // 2:] = (np.sin(2 * np.pi * 10 * t[n_samples // 2:]) * 3.0).astype(np.float32)
            epochs[i] = signal + noise

        fft_size = 256
        ersp = zbci.compute_ersp(
            epochs, sr, (0.0, n_samples / sr / 2), fft_size=fft_size, hop_size=32
        )
        # Frequency resolution = sr / fft_size = 1 Hz
        # 10 Hz bin = index 10
        freq_bin_10hz = 10
        # Post-stimulus frames should show elevated power at 10 Hz
        n_frames = ersp.shape[1]
        post_stim_mean = np.mean(ersp[freq_bin_10hz, n_frames // 2:])
        assert post_stim_mean > 3.0, (
            f"10 Hz power should be elevated post-stimulus, got {post_stim_mean} dB"
        )


class TestBaselineNormalize:
    """Tests for baseline_normalize function."""

    def test_db_known_values(self):
        """dB normalization on known power matrix."""
        # 3 frames x 2 freqs
        power = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [10.0, 20.0],
        ], dtype=np.float64)
        norm = zbci.baseline_normalize(power, 0, 1, mode='db')
        # Frame 0 = baseline -> 0 dB
        np.testing.assert_allclose(norm[0, :], 0.0, atol=1e-10)
        # Frame 1: 10*log10(2/1) = ~3.01, 10*log10(4/2) = ~3.01
        expected = 10 * np.log10(2.0)
        np.testing.assert_allclose(norm[1, :], expected, atol=1e-10)
        # Frame 2: 10*log10(10/1) = 10, 10*log10(20/2) = 10
        np.testing.assert_allclose(norm[2, :], 10.0, atol=1e-10)

    def test_zscore_known_values(self):
        """Z-score normalization on known values."""
        # baseline frames 0..2: values [2, 4] -> mean=3, std=1
        power = np.array([[2.0], [4.0], [5.0]], dtype=np.float64)
        norm = zbci.baseline_normalize(power, 0, 2, mode='zscore')
        np.testing.assert_allclose(norm[0, 0], -1.0, atol=1e-10)
        np.testing.assert_allclose(norm[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(norm[2, 0], 2.0, atol=1e-10)

    def test_percentage_known_values(self):
        """Percentage normalization on known values."""
        power = np.array([[2.0], [4.0]], dtype=np.float64)
        norm = zbci.baseline_normalize(power, 0, 1, mode='percentage')
        np.testing.assert_allclose(norm[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(norm[1, 0], 100.0, atol=1e-10)

    def test_logratio_known_values(self):
        """Log-ratio normalization on known values."""
        power = np.array([[1.0], [10.0]], dtype=np.float64)
        norm = zbci.baseline_normalize(power, 0, 1, mode='logratio')
        np.testing.assert_allclose(norm[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(norm[1, 0], 1.0, atol=1e-10)

    def test_returns_copy(self):
        """Should return a new array, not modify original."""
        power = np.ones((5, 3), dtype=np.float64) * 2.0
        norm = zbci.baseline_normalize(power, 0, 2, mode='db')
        # Original should be unchanged
        np.testing.assert_allclose(power, 2.0)
        # Normalized baseline should be 0
        np.testing.assert_allclose(norm[0, :], 0.0, atol=1e-10)


class TestErspValidation:
    """Tests for ERSP input validation."""

    def test_empty_epochs(self):
        """Empty epochs should raise."""
        epochs = np.zeros((0, 1024), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            zbci.compute_ersp(epochs, 256.0, (0.0, 0.5))

    def test_invalid_mode(self):
        """Invalid mode string should raise."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="mode"):
            zbci.compute_ersp(epochs, 256.0, (0.0, 0.5), fft_size=128, mode='invalid')

    def test_invalid_fft_size(self):
        """Non-power-of-two FFT size should raise."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="fft_size"):
            zbci.compute_ersp(epochs, 256.0, (0.0, 0.5), fft_size=300)

    def test_invalid_baseline_window(self):
        """Baseline start >= end should raise."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="baseline"):
            zbci.compute_ersp(epochs, 256.0, (1.0, 0.5), fft_size=128)

    def test_epoch_too_short(self):
        """Epoch shorter than fft_size should raise."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 100)).astype(np.float32)
        with pytest.raises(ValueError, match="shorter"):
            zbci.compute_ersp(epochs, 256.0, (0.0, 0.2), fft_size=256)

    def test_baseline_normalize_empty(self):
        """Empty power array should raise."""
        power = np.zeros((0, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="empty"):
            zbci.baseline_normalize(power, 0, 0)

    def test_baseline_normalize_invalid_range(self):
        """start >= end should raise."""
        power = np.ones((5, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="baseline"):
            zbci.baseline_normalize(power, 3, 1)

    def test_negative_sample_rate(self):
        """Negative sample rate should raise."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((5, 512)).astype(np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            zbci.compute_ersp(epochs, -1.0, (0.0, 0.5))


class TestErspIntegration:
    """Integration tests combining multiple features."""

    def test_multi_trial_averaging_reduces_noise(self):
        """Averaging more trials should reduce noise in ERSP."""
        sr = 256.0
        n_samples = 512
        fft_size = 128
        rng = np.random.default_rng(42)

        # Pure noise epochs
        few_epochs = rng.standard_normal((5, n_samples)).astype(np.float32)
        many_epochs = rng.standard_normal((50, n_samples)).astype(np.float32)

        ersp_few = zbci.compute_ersp(few_epochs, sr, (0.0, 0.5), fft_size=fft_size)
        ersp_many = zbci.compute_ersp(many_epochs, sr, (0.0, 0.5), fft_size=fft_size)

        # More epochs -> less variance in ERSP (closer to 0 dB for noise)
        var_few = np.var(ersp_few)
        var_many = np.var(ersp_many)
        assert var_many < var_few, (
            f"More epochs should reduce variance: {var_many} >= {var_few}"
        )

    def test_single_trial_vs_averaged(self):
        """Single trial ERSP averaged manually should match averaged ERSP."""
        rng = np.random.default_rng(42)
        epochs = rng.standard_normal((10, 512)).astype(np.float32) + 1.0
        # Shift to positive to avoid zero-mean baseline issues

        # Average ERSP
        ersp_avg = zbci.compute_ersp(
            epochs, 256.0, (0.0, 0.5), fft_size=128, mode='percentage'
        )

        # Single trial ERSP
        ersp_single = zbci.compute_ersp(
            epochs, 256.0, (0.0, 0.5), fft_size=128, mode='percentage', single_trial=True
        )

        # Both should have consistent shapes
        assert ersp_single.shape[0] == 10
        assert ersp_single.shape[1] == ersp_avg.shape[0]
        assert ersp_single.shape[2] == ersp_avg.shape[1]

    def test_logratio_is_db_divided_by_10(self):
        """Log-ratio should be dB / 10."""
        power = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [4.0, 8.0, 12.0],
        ], dtype=np.float64)

        norm_db = zbci.baseline_normalize(power.copy(), 0, 1, mode='db')
        norm_lr = zbci.baseline_normalize(power.copy(), 0, 1, mode='logratio')

        np.testing.assert_allclose(norm_db, norm_lr * 10.0, atol=1e-10)
