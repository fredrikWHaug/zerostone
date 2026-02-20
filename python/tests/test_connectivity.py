"""Tests for connectivity metrics (coherence, PLV)."""

import numpy as np
import pytest
import zpybci as zbci


class TestCoherence:
    """Tests for single-window coherence."""

    def test_coherence_identical_signals(self):
        """Identical signals should have coherence = 1.0 at signal frequency."""
        t = np.arange(256) / 256.0
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        freqs, coh = zbci.coherence(sig, sig, fft_size=256)

        assert freqs.shape == (129,)
        assert coh.shape == (129,)
        # Bin 10 = 10 Hz (since fft_size = sample_rate = 256)
        assert coh[10] > 0.99, f"Coherence at signal freq should be ~1.0, got {coh[10]}"

    def test_coherence_identical_broadband(self):
        """Identical broadband signal should have coherence = 1.0 everywhere."""
        np.random.seed(42)
        sig = np.random.randn(256).astype(np.float32)

        _, coh = zbci.coherence(sig, sig, fft_size=256)

        # All bins should be ~1.0
        assert np.all(coh[1:128] > 0.99), "All bins should have coherence ~1.0"

    def test_coherence_range(self):
        """Coherence values must be in [0, 1]."""
        t = np.arange(256) / 256.0
        sig_a = (np.sin(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 30 * t)).astype(np.float32)
        sig_b = (np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)

        _, coh = zbci.coherence(sig_a, sig_b, fft_size=256)

        assert np.all(coh >= 0.0), "Coherence must be >= 0"
        assert np.all(coh <= 1.0 + 1e-6), "Coherence must be <= 1"

    def test_coherence_window_types(self):
        """Test coherence with different window types."""
        t = np.arange(256) / 256.0
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        for window in ['rectangular', 'hann', 'hamming', 'blackman', 'blackman_harris']:
            _, coh = zbci.coherence(sig, sig, fft_size=256, window=window)
            assert coh[10] > 0.99, f"Window '{window}' failed: coherence = {coh[10]}"

    def test_coherence_fft_sizes(self):
        """Test coherence with different FFT sizes."""
        np.random.seed(42)
        sig = np.random.randn(4096).astype(np.float32)

        for fft_size in [256, 512, 1024, 2048, 4096]:
            _, coh = zbci.coherence(sig[:fft_size], sig[:fft_size], fft_size=fft_size)
            assert coh.shape == (fft_size // 2 + 1,)
            assert np.all(coh[1:-1] > 0.99)

    def test_coherence_invalid_fft_size(self):
        """Test that invalid FFT size raises error."""
        sig = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="fft_size"):
            zbci.coherence(sig, sig, fft_size=128)

    def test_coherence_signal_too_short(self):
        """Test that too-short signal raises error."""
        sig = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="length"):
            zbci.coherence(sig, sig, fft_size=256)


class TestSpectralCoherence:
    """Tests for Welch-style averaged coherence."""

    def test_spectral_coherence_identical_signals(self):
        """Identical signals should have spectral coherence = 1.0."""
        t = np.arange(1024) / 256.0
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        freqs, coh = zbci.spectral_coherence(sig, sig, fft_size=256, sample_rate=256.0)

        assert freqs.shape == (129,)
        assert coh.shape == (129,)
        # Bin 10 = 10 Hz
        assert coh[10] > 0.99, f"Spectral coherence at 10 Hz should be ~1.0, got {coh[10]}"

    def test_spectral_coherence_frequency_axis(self):
        """Test that frequency axis is correctly computed."""
        t = np.arange(1024) / 250.0
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        freqs, _ = zbci.spectral_coherence(sig, sig, fft_size=256, sample_rate=250.0)

        # Frequency resolution = 250/256 ≈ 0.977 Hz
        assert abs(freqs[0]) < 1e-6, "DC should be 0 Hz"
        assert abs(freqs[-1] - 125.0) < 0.5, f"Nyquist should be ~125 Hz, got {freqs[-1]}"

    def test_spectral_coherence_independent_noise(self):
        """Independent noise should have low spectral coherence."""
        np.random.seed(42)
        sig_a = np.random.randn(4096).astype(np.float32)
        np.random.seed(99)
        sig_b = np.random.randn(4096).astype(np.float32)

        _, coh = zbci.spectral_coherence(sig_a, sig_b, fft_size=256, sample_rate=256.0)

        mean_coh = np.mean(coh[1:128])
        assert mean_coh < 0.3, f"Mean coherence of independent noise should be low, got {mean_coh}"

    def test_spectral_coherence_shared_component(self):
        """Signals sharing a frequency component should have high coherence there."""
        np.random.seed(42)
        t = np.arange(2048) / 256.0
        shared = np.sin(2 * np.pi * 10 * t)
        sig_a = (shared + np.random.randn(2048) * 0.1).astype(np.float32)
        np.random.seed(99)
        sig_b = (shared + np.random.randn(2048) * 0.1).astype(np.float32)

        freqs, coh = zbci.spectral_coherence(sig_a, sig_b, fft_size=256, sample_rate=256.0)

        # High coherence at 10 Hz
        assert coh[10] > 0.8, f"Coherence at shared freq should be high, got {coh[10]}"

    def test_spectral_coherence_range(self):
        """Spectral coherence must be in [0, 1]."""
        np.random.seed(42)
        sig_a = np.random.randn(1024).astype(np.float32)
        sig_b = np.random.randn(1024).astype(np.float32)

        _, coh = zbci.spectral_coherence(sig_a, sig_b, fft_size=256, sample_rate=256.0)

        assert np.all(coh >= 0.0), "Coherence must be >= 0"
        assert np.all(coh <= 1.0 + 1e-6), "Coherence must be <= 1"

    def test_spectral_coherence_overlap(self):
        """Test different overlap values."""
        t = np.arange(1024) / 256.0
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        for overlap in [0.0, 0.25, 0.5, 0.75]:
            _, coh = zbci.spectral_coherence(
                sig, sig, fft_size=256, sample_rate=256.0, overlap=overlap
            )
            assert coh[10] > 0.99, f"Overlap {overlap}: coherence = {coh[10]}"

    def test_spectral_coherence_unequal_lengths(self):
        """Test that unequal signal lengths raise error."""
        sig_a = np.zeros(1024, dtype=np.float32)
        sig_b = np.zeros(512, dtype=np.float32)
        with pytest.raises(ValueError, match="equal length"):
            zbci.spectral_coherence(sig_a, sig_b, fft_size=256, sample_rate=256.0)

    def test_spectral_coherence_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        sig = np.zeros(1024, dtype=np.float32)
        with pytest.raises(ValueError, match="overlap"):
            zbci.spectral_coherence(sig, sig, fft_size=256, sample_rate=256.0, overlap=1.0)

    def test_spectral_coherence_invalid_sample_rate(self):
        """Test that invalid sample rate raises error."""
        sig = np.zeros(1024, dtype=np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            zbci.spectral_coherence(sig, sig, fft_size=256, sample_rate=0.0)


class TestPhaseLockingValue:
    """Tests for Phase Locking Value."""

    def test_plv_constant_phase_difference(self):
        """Constant phase difference should give PLV = 1.0."""
        phases_a = np.linspace(0, 10, 100).astype(np.float32)
        phases_b = phases_a + 0.5  # constant offset

        plv = zbci.phase_locking_value(phases_a, phases_b)

        assert abs(plv - 1.0) < 1e-5, f"PLV should be 1.0, got {plv}"

    def test_plv_identical_phases(self):
        """Identical phases should give PLV = 1.0."""
        phases = np.linspace(0, 20, 200).astype(np.float32)

        plv = zbci.phase_locking_value(phases, phases)

        assert abs(plv - 1.0) < 1e-5, f"PLV should be 1.0, got {plv}"

    def test_plv_random_phases(self):
        """Random phases should give PLV close to 0."""
        np.random.seed(42)
        phases_a = (np.random.rand(10000) * 2 * np.pi).astype(np.float32)
        phases_b = (np.random.rand(10000) * 2 * np.pi).astype(np.float32)

        plv = zbci.phase_locking_value(phases_a, phases_b)

        assert plv < 0.05, f"PLV of random phases should be ~0, got {plv}"

    def test_plv_anti_phase(self):
        """Constant pi phase difference should give PLV = 1.0."""
        phases_a = np.linspace(0, 10, 100).astype(np.float32)
        phases_b = phases_a + np.pi  # anti-phase

        plv = zbci.phase_locking_value(phases_a, phases_b.astype(np.float32))

        assert abs(plv - 1.0) < 1e-5, f"PLV should be 1.0 for anti-phase, got {plv}"

    def test_plv_range(self):
        """PLV must be in [0, 1]."""
        np.random.seed(42)
        phases_a = np.random.randn(50).astype(np.float32)
        phases_b = np.random.randn(50).astype(np.float32)

        plv = zbci.phase_locking_value(phases_a, phases_b)

        assert plv >= 0.0, f"PLV must be >= 0, got {plv}"
        assert plv <= 1.0 + 1e-6, f"PLV must be <= 1, got {plv}"

    def test_plv_unequal_lengths(self):
        """Test that unequal phase arrays raise error."""
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="equal length"):
            zbci.phase_locking_value(a, b)

    def test_plv_empty_arrays(self):
        """Test that empty arrays raise error."""
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            zbci.phase_locking_value(a, b)

    def test_plv_with_hilbert(self):
        """Test PLV computed from Hilbert transform phases."""
        # Create two phase-locked sinusoids
        t = np.arange(1024) / 256.0
        freq = 10.0
        sig_a = np.sin(2 * np.pi * freq * t).astype(np.float32)
        sig_b = np.sin(2 * np.pi * freq * t + 0.3).astype(np.float32)

        # Extract phases via Hilbert transform
        ht = zbci.HilbertTransform(size=1024)
        phases_a = ht.instantaneous_phase(sig_a)
        phases_b = ht.instantaneous_phase(sig_b)

        plv = zbci.phase_locking_value(phases_a, phases_b)

        assert plv > 0.95, f"Phase-locked sinusoids should have high PLV, got {plv}"


class TestConnectivityIntegration:
    """Integration tests combining connectivity with other primitives."""

    def test_coherence_after_filtering(self):
        """Test coherence of signals processed through the same filter."""
        np.random.seed(42)
        t = np.arange(1024) / 256.0

        # Two signals with shared 10 Hz component
        shared = np.sin(2 * np.pi * 10 * t)
        sig_a = (shared + np.random.randn(1024) * 0.2).astype(np.float32)
        sig_b = (shared + np.random.randn(1024) * 0.2).astype(np.float32)

        # Bandpass filter to isolate shared component
        bpf_a = zbci.IirFilter.butterworth_bandpass(256.0, 8.0, 12.0)
        bpf_b = zbci.IirFilter.butterworth_bandpass(256.0, 8.0, 12.0)
        filtered_a = bpf_a.process(sig_a)
        filtered_b = bpf_b.process(sig_b)

        _, coh = zbci.spectral_coherence(
            filtered_a, filtered_b, fft_size=256, sample_rate=256.0
        )

        # Coherence in the passband should be high
        assert coh[10] > 0.5, f"Coherence at 10 Hz after filtering should be high, got {coh[10]}"
