"""Tests for Welch PSD estimation."""

import numpy as np
import pytest
import zpybci as zbci


class TestWelchPsdCreation:
    """Tests for WelchPsd construction."""

    def test_create_default(self):
        """Test creating with default parameters."""
        welch = zbci.WelchPsd(fft_size=256)
        assert welch.fft_size == 256
        assert welch.window == "hann"
        assert welch.overlap == 0.5

    def test_create_with_params(self):
        """Test creating with custom parameters."""
        welch = zbci.WelchPsd(fft_size=1024, window="hamming", overlap=0.75)
        assert welch.fft_size == 1024
        assert welch.window == "hamming"
        assert welch.overlap == 0.75

    def test_different_sizes(self):
        """Test all supported FFT sizes."""
        for size in [256, 512, 1024, 2048, 4096]:
            welch = zbci.WelchPsd(fft_size=size)
            assert welch.fft_size == size

    def test_invalid_size(self):
        """Test that invalid FFT sizes raise errors."""
        with pytest.raises(ValueError):
            zbci.WelchPsd(fft_size=128)
        with pytest.raises(ValueError):
            zbci.WelchPsd(fft_size=300)

    def test_invalid_overlap(self):
        """Test that invalid overlap values raise errors."""
        with pytest.raises(ValueError):
            zbci.WelchPsd(fft_size=256, overlap=1.0)
        with pytest.raises(ValueError):
            zbci.WelchPsd(fft_size=256, overlap=-0.1)

    def test_repr(self):
        """Test string representation."""
        welch = zbci.WelchPsd(fft_size=1024, window="hann", overlap=0.5)
        r = repr(welch)
        assert "WelchPsd" in r
        assert "1024" in r
        assert "hann" in r
        assert "0.5" in r


class TestWelchPsdEstimate:
    """Tests for WelchPsd.estimate()."""

    def test_sinusoid_peak_detection(self):
        """Test that a pure sinusoid produces a peak at the correct frequency."""
        welch = zbci.WelchPsd(fft_size=256)
        sample_rate = 256.0
        freq = 10.0

        t = np.arange(2048) / sample_rate
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

        freqs, psd = welch.estimate(signal, sample_rate=sample_rate)

        # Output shape
        assert freqs.shape == (129,)
        assert psd.shape == (129,)

        # Peak at 10 Hz
        peak_idx = np.argmax(psd[1:]) + 1  # skip DC
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - freq) < 2.0, f"Peak at {peak_freq} Hz, expected {freq} Hz"

    def test_normalization_parseval(self):
        """Test that PSD integrates to time-domain power (Parseval's theorem)."""
        welch = zbci.WelchPsd(fft_size=256, window="rectangular", overlap=0.5)
        sample_rate = 256.0

        t = np.arange(2048) / sample_rate
        signal = (np.sin(2 * np.pi * 20.0 * t) + 0.5 * np.cos(2 * np.pi * 50.0 * t)).astype(
            np.float32
        )

        time_power = np.mean(signal**2)

        freqs, psd = welch.estimate(signal, sample_rate=sample_rate)
        freq_res = freqs[1] - freqs[0]
        freq_power = np.sum(psd) * freq_res

        ratio = freq_power / time_power
        assert abs(ratio - 1.0) < 0.15, f"Parseval ratio: {ratio}"

    def test_white_noise_flatness(self):
        """Test that white noise produces a roughly flat PSD."""
        welch = zbci.WelchPsd(fft_size=256, overlap=0.5)
        sample_rate = 256.0

        rng = np.random.RandomState(42)
        signal = rng.randn(8192).astype(np.float32)

        freqs, psd = welch.estimate(signal, sample_rate=sample_rate)

        # Check flatness in middle range (avoid DC and Nyquist edge)
        mid = psd[5:120]
        max_val = np.max(mid)
        min_val = np.min(mid)
        flatness = max_val / min_val
        assert flatness < 3.0, f"White noise PSD max/min ratio: {flatness}"

    def test_overlap_parameter(self):
        """Test different overlap values produce valid results."""
        sample_rate = 256.0
        t = np.arange(2048) / sample_rate
        signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)

        for overlap in [0.0, 0.25, 0.5, 0.75]:
            welch = zbci.WelchPsd(fft_size=256, overlap=overlap)
            freqs, psd = welch.estimate(signal, sample_rate=sample_rate)

            assert psd.shape == (129,)
            assert np.all(psd >= 0), f"PSD should be non-negative (overlap={overlap})"

            # Should still find 10 Hz peak
            peak_idx = np.argmax(psd[1:]) + 1
            peak_freq = freqs[peak_idx]
            assert abs(peak_freq - 10.0) < 2.0

    def test_window_types(self):
        """Test all supported window types."""
        sample_rate = 256.0
        t = np.arange(2048) / sample_rate
        signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)

        for window in ["rectangular", "hann", "hamming", "blackman", "blackman_harris"]:
            welch = zbci.WelchPsd(fft_size=256, window=window)
            freqs, psd = welch.estimate(signal, sample_rate=sample_rate)
            assert psd.shape == (129,)

            peak_idx = np.argmax(psd[1:]) + 1
            peak_freq = freqs[peak_idx]
            assert abs(peak_freq - 10.0) < 2.0, f"Window '{window}' peak at {peak_freq}"

    def test_invalid_window(self):
        """Test that invalid window type raises error."""
        with pytest.raises(ValueError):
            zbci.WelchPsd(fft_size=256, window="kaiser")

    def test_short_signal_error(self):
        """Test that signal shorter than fft_size raises error."""
        welch = zbci.WelchPsd(fft_size=256)
        signal = np.zeros(100, dtype=np.float32)

        with pytest.raises(ValueError, match="Signal length"):
            welch.estimate(signal, sample_rate=256.0)

    def test_frequency_array(self):
        """Test that frequency array is correct."""
        welch = zbci.WelchPsd(fft_size=256)
        sample_rate = 500.0

        signal = np.zeros(256, dtype=np.float32)
        freqs, _ = welch.estimate(signal, sample_rate=sample_rate)

        assert abs(freqs[0]) < 1e-6, "First frequency should be 0 Hz"
        expected_nyquist = sample_rate / 2.0
        assert abs(freqs[-1] - expected_nyquist) < 1e-3, f"Last freq: {freqs[-1]}, expected {expected_nyquist}"

        # Check uniform spacing
        diffs = np.diff(freqs)
        expected_res = sample_rate / 256.0
        np.testing.assert_allclose(diffs, expected_res, rtol=1e-5)
