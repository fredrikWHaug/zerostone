"""Tests for Continuous Wavelet Transform (CWT) bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestCwt:
    """Tests for Cwt."""

    def test_create_cwt(self):
        """Test creating a CWT processor."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        assert cwt.size == 256
        assert cwt.num_scales == 8
        assert cwt.sample_rate == 250.0
        assert cwt.min_freq == 5.0
        assert cwt.max_freq == 50.0
        assert abs(cwt.omega0 - 6.0) < 0.1  # Default omega0

    def test_create_with_custom_omega0(self):
        """Test creating CWT with custom omega0."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0,
            omega0=8.0
        )
        assert abs(cwt.omega0 - 8.0) < 0.1

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        # Invalid size
        with pytest.raises(ValueError):
            zbci.Cwt(size=100, num_scales=8, sample_rate=250.0, min_freq=5.0, max_freq=50.0)

        # Invalid num_scales
        with pytest.raises(ValueError):
            zbci.Cwt(size=256, num_scales=5, sample_rate=250.0, min_freq=5.0, max_freq=50.0)

        # min_freq >= max_freq
        with pytest.raises(ValueError):
            zbci.Cwt(size=256, num_scales=8, sample_rate=250.0, min_freq=50.0, max_freq=10.0)

        # max_freq > Nyquist
        with pytest.raises(ValueError):
            zbci.Cwt(size=256, num_scales=8, sample_rate=250.0, min_freq=5.0, max_freq=200.0)

        # min_freq <= 0
        with pytest.raises(ValueError):
            zbci.Cwt(size=256, num_scales=8, sample_rate=250.0, min_freq=0.0, max_freq=50.0)

    def test_supported_configurations(self):
        """Test all supported size/scale combinations."""
        sizes = [64, 128, 256, 512, 1024]
        scales = [4, 8, 16, 32]

        for size in sizes:
            for num_scales in scales:
                cwt = zbci.Cwt(
                    size=size,
                    num_scales=num_scales,
                    sample_rate=250.0,
                    min_freq=5.0,
                    max_freq=50.0
                )
                assert cwt.size == size
                assert cwt.num_scales == num_scales

    def test_power_shape(self):
        """Test power output shape."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        signal = np.random.randn(256).astype(np.float32)
        power = cwt.power(signal)

        assert power.shape == (8, 256)  # (num_scales, size)
        assert np.all(power >= 0)  # Power is non-negative

    def test_magnitude_shape(self):
        """Test magnitude output shape."""
        cwt = zbci.Cwt(
            size=128,
            num_scales=4,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=40.0
        )
        signal = np.random.randn(128).astype(np.float32)
        magnitude = cwt.magnitude(signal)

        assert magnitude.shape == (4, 128)
        assert np.all(magnitude >= 0)

    def test_frequencies_shape(self):
        """Test frequencies output."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=16,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        freqs = cwt.frequencies()

        assert freqs.shape == (16,)
        # Frequencies should be in specified range (approximately)
        assert np.min(freqs) >= 4.0
        assert np.max(freqs) <= 55.0

    def test_scales_shape(self):
        """Test scales output."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        scales = cwt.scales()
        assert scales.shape == (8,)
        assert np.all(scales > 0)

    def test_signal_size_mismatch(self):
        """Test that size mismatch raises error."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        signal = np.random.randn(128).astype(np.float32)

        with pytest.raises(ValueError, match="expected 256"):
            cwt.power(signal)

    def test_repr(self):
        """Test string representation."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=8,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )
        r = repr(cwt)
        assert "256" in r
        assert "8" in r


class TestCwtTimeFrequency:
    """Tests for CWT time-frequency analysis."""

    def test_detect_pure_tone(self):
        """Test that CWT detects pure tone at correct frequency."""
        cwt = zbci.Cwt(
            size=256,
            num_scales=16,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )

        # Generate 20 Hz sine wave
        t = np.arange(256) / 250.0
        signal = np.sin(2 * np.pi * 20.0 * t).astype(np.float32)
        power = cwt.power(signal)

        # Find scale with maximum average power
        avg_power = np.mean(power, axis=1)
        max_scale_idx = np.argmax(avg_power)

        freqs = cwt.frequencies()
        detected_freq = freqs[max_scale_idx]

        # Should be close to 20 Hz
        assert abs(detected_freq - 20.0) < 5.0

    def test_impulse_response(self):
        """Test CWT response to impulse."""
        cwt = zbci.Cwt(
            size=64,
            num_scales=4,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=30.0
        )

        # Impulse signal
        signal = np.zeros(64, dtype=np.float32)
        signal[32] = 1.0

        power = cwt.power(signal)

        # All scales should have some response
        for scale_power in power:
            assert np.sum(scale_power) > 0

        # Power should be centered around impulse
        for scale_power in power:
            assert scale_power[32] > scale_power[0]

    def test_multi_frequency_signal(self):
        """Test CWT with multiple frequency components."""
        cwt = zbci.Cwt(
            size=512,
            num_scales=32,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=50.0
        )

        # Signal with 10 Hz and 30 Hz components (equal amplitude for easier detection)
        t = np.arange(512) / 250.0
        signal = (np.sin(2 * np.pi * 10.0 * t) +
                  np.sin(2 * np.pi * 30.0 * t)).astype(np.float32)

        power = cwt.power(signal)
        freqs = cwt.frequencies()

        # Average power per scale
        avg_power = np.mean(power, axis=1)

        # Find the top 4 highest power scales
        sorted_idx = np.argsort(avg_power)[::-1]
        top_freqs = freqs[sorted_idx[:4]]

        # Should detect at least one component near 10 Hz
        assert any(abs(f - 10.0) < 10.0 for f in top_freqs), f"No frequency near 10 Hz in {top_freqs}"

    def test_power_vs_magnitude(self):
        """Test that power = magnitude^2."""
        cwt = zbci.Cwt(
            size=128,
            num_scales=4,
            sample_rate=250.0,
            min_freq=5.0,
            max_freq=30.0
        )

        signal = np.random.randn(128).astype(np.float32)
        power = cwt.power(signal)
        magnitude = cwt.magnitude(signal)

        # power should approximately equal magnitude^2
        np.testing.assert_allclose(power, magnitude ** 2, rtol=1e-5)
