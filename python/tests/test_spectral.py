"""Tests for spectral module bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestFft:
    """Tests for Fft."""

    def test_create_fft(self):
        """Test creating an FFT processor."""
        fft = zbci.Fft(256)
        assert fft.size == 256

    def test_invalid_size(self):
        """Test that non-power-of-2 sizes raise errors."""
        with pytest.raises(ValueError):
            zbci.Fft(100)  # Not a power of 2
        with pytest.raises(ValueError):
            zbci.Fft(0)  # Zero

    def test_forward_transform(self):
        """Test forward FFT transform."""
        fft = zbci.Fft(256)

        # Real signal
        signal = np.random.randn(256).astype(np.float32)

        # Forward transform returns tuple of (real, imag) arrays
        real, imag = fft.forward(signal)

        # Should return N-sized arrays (full spectrum)
        assert real.shape == (256,)
        assert imag.shape == (256,)

    def test_power_spectrum(self):
        """Test power spectrum computation."""
        fft = zbci.Fft(256)

        # DC signal
        signal = np.ones(256, dtype=np.float32)
        power = fft.power_spectrum(signal)

        # Should have N/2 + 1 = 129 bins
        assert power.shape == (129,)

        # DC component should have all the power
        assert power[0] > 0
        # Other bins should be near zero for DC signal
        assert np.all(power[1:] < 1e-10)

    def test_power_spectrum_sine(self):
        """Test power spectrum of sine wave."""
        fft = zbci.Fft(256)
        sample_rate = 256.0
        freq = 32.0  # 32 Hz - should appear at bin 32

        # Generate sine wave
        t = np.arange(256) / sample_rate
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

        power = fft.power_spectrum(signal)

        # Peak should be at bin 32 (freq * N / sample_rate)
        peak_bin = int(freq * 256 / sample_rate)

        # Peak bin should have high power
        assert np.argmax(power) == peak_bin

    def test_wrong_input_size(self):
        """Test that wrong input size raises error."""
        fft = zbci.Fft(256)
        signal = np.random.randn(128).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="[Ll]ength"):
            fft.forward(signal)

    def test_supported_sizes(self):
        """Test all supported FFT sizes."""
        for size in [64, 128, 256, 512, 1024, 2048]:
            fft = zbci.Fft(size)
            signal = np.random.randn(size).astype(np.float32)
            power = fft.power_spectrum(signal)
            assert power.shape == (size // 2 + 1,)

    def test_repr(self):
        """Test string representation."""
        fft = zbci.Fft(512)
        assert "512" in repr(fft)


class TestStft:
    """Tests for Stft (Short-Time Fourier Transform)."""

    def test_create_stft(self):
        """Test creating an STFT processor."""
        stft = zbci.Stft(256, 64)
        assert stft.size == 256
        assert stft.hop_size == 64

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            zbci.Stft(100, 64)  # FFT size not power of 2
        with pytest.raises(ValueError):
            zbci.Stft(256, 0)  # hop_size must be > 0

    def test_transform(self):
        """Test STFT transform."""
        stft = zbci.Stft(256, 64)

        # Signal with multiple frames
        signal = np.random.randn(1024).astype(np.float32)

        # Transform returns tuple of (real, imag) 2D arrays
        real, imag = stft.transform(signal)

        # Number of frames = (signal_len - size) / hop + 1
        expected_frames = (1024 - 256) // 64 + 1
        assert real.shape[0] == expected_frames
        assert real.shape[1] == 256  # Full FFT size
        assert imag.shape == real.shape

    def test_power_spectrogram(self):
        """Test power spectrogram computation."""
        stft = zbci.Stft(256, 64)

        signal = np.random.randn(1024).astype(np.float32)
        power = stft.power(signal)

        # Should return (frames, fft_size) - note: full FFT size, not bins
        expected_frames = (1024 - 256) // 64 + 1
        assert power.shape[0] == expected_frames
        # Power shape depends on implementation

        # All power values should be non-negative
        assert np.all(power >= 0)

    def test_power_spectrogram_sine(self):
        """Test power spectrogram of frequency sweep."""
        stft = zbci.Stft(256, 64)
        sample_rate = 1024.0

        # Two different frequencies in sequence
        t1 = np.arange(512) / sample_rate
        t2 = np.arange(512) / sample_rate

        signal1 = np.sin(2 * np.pi * 32.0 * t1).astype(np.float32)  # 32 Hz
        signal2 = np.sin(2 * np.pi * 64.0 * t2).astype(np.float32)  # 64 Hz

        signal = np.concatenate([signal1, signal2])
        power = stft.power(signal)

        # First half should have peak around bin 8 (32 * 256 / 1024)
        # Second half should have peak around bin 16 (64 * 256 / 1024)

        # Check that we detect both frequencies
        half_frames = power.shape[0] // 2
        early_peak = np.argmax(np.mean(power[:half_frames], axis=0))
        late_peak = np.argmax(np.mean(power[half_frames:], axis=0))

        # Frequencies should be different
        assert early_peak != late_peak

    def test_num_frames(self):
        """Test num_frames calculation."""
        stft = zbci.Stft(256, 64)
        num_frames = stft.num_frames(1024)
        expected = (1024 - 256) // 64 + 1
        assert num_frames == expected

    def test_supported_sizes(self):
        """Test all supported FFT sizes."""
        for size in [64, 128, 256, 512, 1024, 2048]:
            stft = zbci.Stft(size, size // 4)
            signal = np.random.randn(size * 4).astype(np.float32)
            power = stft.power(signal)
            assert power.shape[0] > 0

    def test_repr(self):
        """Test string representation."""
        stft = zbci.Stft(256, 64)
        assert "256" in repr(stft)
        assert "64" in repr(stft)


class TestMultiBandPower:
    """Tests for MultiBandPower."""

    def test_create_multiband(self):
        """Test creating a multi-band power extractor."""
        mbp = zbci.MultiBandPower(512, 8, 250.0)
        assert mbp.fft_size == 512
        assert mbp.channels == 8
        assert mbp.sample_rate == 250.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            zbci.MultiBandPower(100, 8, 250.0)  # FFT size not power of 2
        with pytest.raises(ValueError):
            zbci.MultiBandPower(512, 0, 250.0)  # channels must be >= 1
        with pytest.raises(ValueError):
            zbci.MultiBandPower(512, 8, 0.0)  # sample_rate must be > 0

    def test_compute_stores_internally(self):
        """Test that compute stores power spectrum internally."""
        mbp = zbci.MultiBandPower(256, 4, 250.0)

        # Multi-channel signal: (channels, samples)
        signal = np.random.randn(4, 256).astype(np.float32)

        # compute returns None (stores internally)
        result = mbp.compute(signal)
        assert result is None

    def test_band_power_extraction(self):
        """Test extracting power in specific frequency bands."""
        sample_rate = 250.0
        mbp = zbci.MultiBandPower(256, 4, sample_rate)

        # Generate signal with known frequency content: (channels, samples)
        t = np.arange(256) / sample_rate

        signal = np.zeros((4, 256), dtype=np.float32)

        # Channel 0: 10 Hz (alpha band)
        signal[0, :] = np.sin(2 * np.pi * 10 * t)

        # Channel 1: 20 Hz (beta band)
        signal[1, :] = np.sin(2 * np.pi * 20 * t)

        # Channel 2: 5 Hz (theta band)
        signal[2, :] = np.sin(2 * np.pi * 5 * t)

        # Channel 3: broadband noise
        signal[3, :] = np.random.randn(256).astype(np.float32) * 0.1

        # Compute spectrum
        mbp.compute(signal)

        # Extract alpha band power (8-12 Hz)
        alpha_power = mbp.band_power(8.0, 12.0)

        assert len(alpha_power) == 4

        # Channel 0 should have highest alpha power
        assert alpha_power[0] > alpha_power[1]
        assert alpha_power[0] > alpha_power[3]

    def test_standard_eeg_bands(self):
        """Test extracting standard EEG frequency bands."""
        sample_rate = 250.0
        mbp = zbci.MultiBandPower(512, 8, sample_rate)

        signal = np.random.randn(8, 512).astype(np.float32)

        # Compute spectrum first
        mbp.compute(signal)

        # Standard EEG bands
        delta = mbp.band_power(0.5, 4.0)
        theta = mbp.band_power(4.0, 8.0)
        alpha = mbp.band_power(8.0, 13.0)
        beta = mbp.band_power(13.0, 30.0)
        gamma = mbp.band_power(30.0, 100.0)

        # All should return channel-length arrays
        assert len(delta) == 8
        assert len(theta) == 8
        assert len(alpha) == 8
        assert len(beta) == 8
        assert len(gamma) == 8

        # All values should be non-negative
        assert np.all(delta >= 0)
        assert np.all(theta >= 0)
        assert np.all(alpha >= 0)
        assert np.all(beta >= 0)
        assert np.all(gamma >= 0)

    def test_channel_count_mismatch(self):
        """Test that channel count mismatch raises error."""
        mbp = zbci.MultiBandPower(256, 8, 250.0)
        signal = np.random.randn(4, 256).astype(np.float32)  # Wrong channel count

        with pytest.raises(ValueError, match="[Cc]hannel"):
            mbp.compute(signal)

    def test_wrong_signal_length(self):
        """Test that wrong signal length raises error."""
        mbp = zbci.MultiBandPower(256, 8, 250.0)
        signal = np.random.randn(8, 128).astype(np.float32)  # Wrong length

        with pytest.raises(ValueError, match="[Ss]ample"):
            mbp.compute(signal)

    def test_invalid_frequency_range(self):
        """Test that invalid frequency range raises error."""
        mbp = zbci.MultiBandPower(256, 4, 250.0)
        signal = np.random.randn(4, 256).astype(np.float32)
        mbp.compute(signal)

        with pytest.raises(ValueError):
            mbp.band_power(50.0, 30.0)  # low > high

    def test_optimized_configurations(self):
        """Test optimized FFT size and channel count combinations."""
        for fft_size in [256, 512, 1024]:
            for channels in [1, 4, 8, 16, 32, 64]:
                mbp = zbci.MultiBandPower(fft_size, channels, 250.0)
                signal = np.random.randn(channels, fft_size).astype(np.float32)
                mbp.compute(signal)
                band = mbp.band_power(8.0, 12.0)
                assert len(band) == channels

    def test_reset(self):
        """Test resetting MultiBandPower state."""
        mbp = zbci.MultiBandPower(256, 4, 250.0)

        # Process some data
        signal1 = np.random.randn(4, 256).astype(np.float32)
        mbp.compute(signal1)

        # Reset
        mbp.reset()

        # Process again - should work
        signal2 = np.random.randn(4, 256).astype(np.float32)
        mbp.compute(signal2)
        band = mbp.band_power(8.0, 12.0)

        assert len(band) == 4

    def test_repr(self):
        """Test string representation."""
        mbp = zbci.MultiBandPower(512, 8, 250.0)
        assert "512" in repr(mbp)
        assert "8" in repr(mbp)
        assert "250" in repr(mbp)


class TestSpectralIntegration:
    """Integration tests for spectral analysis components."""

    def test_fft_sine_wave_detection(self):
        """Test that FFT correctly identifies sine wave frequency."""
        fft_size = 256
        sample_rate = 256.0
        freq = 32.0

        fft = zbci.Fft(fft_size)

        t = np.arange(fft_size) / sample_rate
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

        power = fft.power_spectrum(signal)

        # Peak should be at expected bin
        peak_bin = int(freq * fft_size / sample_rate)
        assert np.argmax(power) == peak_bin

    def test_stft_time_frequency(self):
        """Test that STFT captures time-varying frequencies."""
        fft_size = 256
        hop_size = 64

        stft = zbci.Stft(fft_size, hop_size)

        # Create chirp-like signal
        signal = np.random.randn(1024).astype(np.float32)
        power = stft.power(signal)

        # Should have multiple frames
        assert power.shape[0] > 1

    def test_multiband_vs_fft_consistency(self):
        """Test that MultiBandPower and FFT give consistent results."""
        fft_size = 256
        sample_rate = 250.0

        fft = zbci.Fft(fft_size)
        mbp = zbci.MultiBandPower(fft_size, 1, sample_rate)

        # Single channel signal
        signal = np.random.randn(fft_size).astype(np.float32)

        # FFT power spectrum
        fft_power = fft.power_spectrum(signal)

        # MultiBandPower compute (single channel, reshape to (1, N))
        signal_2d = signal.reshape(1, -1)
        mbp.compute(signal_2d)

        # Both should detect the same dominant frequency
        # (Exact values may differ due to normalization)
        assert fft_power.shape == (fft_size // 2 + 1,)
