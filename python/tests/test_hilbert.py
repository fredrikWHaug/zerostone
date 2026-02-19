"""Tests for HilbertTransform bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestHilbertTransform:
    """Tests for HilbertTransform."""

    def test_create_hilbert(self):
        """Test creating a Hilbert transform."""
        for size in [64, 128, 256, 512, 1024, 2048]:
            h = zbci.HilbertTransform(size=size)
            assert h.size == size

    def test_invalid_size(self):
        """Test that invalid size raises error."""
        with pytest.raises(ValueError):
            zbci.HilbertTransform(size=100)  # Not a supported size

    def test_transform_shape(self):
        """Test transform output shape."""
        h = zbci.HilbertTransform(size=128)
        signal = np.random.randn(128).astype(np.float32)
        result = h.transform(signal)
        assert result.shape == (128,)

    def test_instantaneous_amplitude_shape(self):
        """Test amplitude output shape."""
        h = zbci.HilbertTransform(size=256)
        signal = np.random.randn(256).astype(np.float32)
        amplitude = h.instantaneous_amplitude(signal)
        assert amplitude.shape == (256,)
        assert np.all(amplitude >= 0)  # Amplitude is non-negative

    def test_instantaneous_phase_shape(self):
        """Test phase output shape."""
        h = zbci.HilbertTransform(size=256)
        signal = np.random.randn(256).astype(np.float32)
        phase = h.instantaneous_phase(signal)
        assert phase.shape == (256,)
        # Phase should be in [-pi, pi]
        assert np.all(phase >= -np.pi - 0.01)
        assert np.all(phase <= np.pi + 0.01)

    def test_instantaneous_frequency_shape(self):
        """Test frequency output shape."""
        h = zbci.HilbertTransform(size=128)
        signal = np.random.randn(128).astype(np.float32)
        freq = h.instantaneous_frequency(signal, sample_rate=250.0)
        assert freq.shape == (127,)  # N-1 elements

    def test_signal_size_mismatch(self):
        """Test that size mismatch raises error."""
        h = zbci.HilbertTransform(size=128)
        signal = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError, match="expected 128"):
            h.transform(signal)

    def test_hilbert_of_dc_is_zero(self):
        """Test that Hilbert transform of DC signal is ~zero."""
        h = zbci.HilbertTransform(size=64)
        signal = np.ones(64, dtype=np.float32) * 5.0
        result = h.transform(signal)
        assert np.max(np.abs(result)) < 0.1

    def test_hilbert_cosine_gives_sine(self):
        """Test that H[cos] = sin."""
        h = zbci.HilbertTransform(size=128)
        sample_rate = 256.0
        freq = 4.0  # Exactly periodic in window

        t = np.arange(128) / sample_rate
        signal = np.cos(2 * np.pi * freq * t).astype(np.float32)
        result = h.transform(signal)
        expected = np.sin(2 * np.pi * freq * t)

        # Skip edges due to FFT artifacts
        assert np.allclose(result[10:-10], expected[10:-10], atol=0.15)

    def test_amplitude_constant_for_pure_tone(self):
        """Test that amplitude envelope is constant for pure tone."""
        h = zbci.HilbertTransform(size=256)
        sample_rate = 256.0
        freq = 10.0

        t = np.arange(256) / sample_rate
        signal = (2.0 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        amplitude = h.instantaneous_amplitude(signal)

        # Amplitude should be ~2.0 (skip edges)
        assert np.allclose(amplitude[20:-20], 2.0, atol=0.3)

    def test_frequency_estimation(self):
        """Test that instantaneous frequency matches signal frequency."""
        h = zbci.HilbertTransform(size=256)
        sample_rate = 256.0
        freq = 15.0

        t = np.arange(256) / sample_rate
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
        inst_freq = h.instantaneous_frequency(signal, sample_rate)

        # Mean frequency (skip edges) should be close to 15 Hz
        mean_freq = np.mean(inst_freq[30:-30])
        assert abs(mean_freq - freq) < 2.0

    def test_repr(self):
        """Test string representation."""
        h = zbci.HilbertTransform(size=256)
        assert "256" in repr(h)


class TestHilbertApplications:
    """Application-oriented tests for Hilbert transform."""

    def test_amplitude_modulation_detection(self):
        """Test detecting amplitude modulation."""
        h = zbci.HilbertTransform(size=256)
        sample_rate = 256.0

        # AM signal: carrier with amplitude modulation
        t = np.arange(256) / sample_rate
        modulation = 1.0 + 0.5 * np.cos(2 * np.pi * 2.0 * t)
        carrier = np.sin(2 * np.pi * 20.0 * t)
        signal = (modulation * carrier).astype(np.float32)

        amplitude = h.instantaneous_amplitude(signal)

        # Amplitude should follow modulation envelope
        max_amp = np.max(amplitude)
        min_amp = np.min(amplitude[10:-10])  # Skip edges
        assert max_amp > 1.2
        assert min_amp < 0.8

    def test_phase_extraction(self):
        """Test phase extraction for phase-locking analysis."""
        h = zbci.HilbertTransform(size=256)
        sample_rate = 256.0
        freq = 10.0

        # Pure sine wave
        t = np.arange(256) / sample_rate
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
        phase = h.instantaneous_phase(signal)

        # Phase should increase monotonically (when unwrapped)
        # Check that phase differences are mostly positive (modulo 2pi wraps)
        dphase = np.diff(phase)
        # Most phase diffs should be positive or wrap around (-2pi)
        positive_or_wrap = (dphase > -0.1) | (dphase < -5.0)
        assert np.mean(positive_or_wrap) > 0.9
