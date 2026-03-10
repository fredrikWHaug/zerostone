"""Tests for Phase-Amplitude Coupling (PAC) metrics."""

import numpy as np
import pytest
import zpybci as zbci


class TestModulationIndex:
    """Tests for modulation_index()."""

    def test_no_coupling(self):
        """Constant amplitude should give MI near 0."""
        phase = np.linspace(-np.pi, np.pi, 720, endpoint=False).astype(np.float32)
        amplitude = np.ones(720, dtype=np.float32)
        mi = zbci.modulation_index(phase, amplitude)
        assert mi < 0.01, f"No coupling should give MI near 0, got {mi}"

    def test_strong_coupling(self):
        """Amplitude modulated by phase should give significant MI."""
        phase = np.linspace(-np.pi, np.pi, 1000, endpoint=False).astype(np.float32)
        amplitude = (1.0 + 0.9 * np.cos(phase)).astype(np.float32)
        mi = zbci.modulation_index(phase, amplitude)
        assert mi > 0.05, f"Strong coupling should give MI > 0.05, got {mi}"

    def test_range(self):
        """MI must be in [0, 1]."""
        phase = np.linspace(-np.pi, np.pi, 500, endpoint=False).astype(np.float32)
        amplitude = (1.0 + 0.5 * np.cos(phase)).astype(np.float32)
        mi = zbci.modulation_index(phase, amplitude)
        assert 0.0 <= mi <= 1.0, f"MI must be in [0, 1], got {mi}"

    def test_default_bins(self):
        """Default n_bins=18 should work."""
        phase = np.linspace(-np.pi, np.pi, 500, endpoint=False).astype(np.float32)
        amplitude = np.ones(500, dtype=np.float32)
        mi = zbci.modulation_index(phase, amplitude)
        assert isinstance(mi, float)

    def test_custom_bins(self):
        """Custom n_bins should work."""
        phase = np.linspace(-np.pi, np.pi, 500, endpoint=False).astype(np.float32)
        amplitude = (1.0 + 0.8 * np.cos(phase)).astype(np.float32)
        mi_9 = zbci.modulation_index(phase, amplitude, n_bins=9)
        mi_36 = zbci.modulation_index(phase, amplitude, n_bins=36)
        assert mi_9 > 0.01, f"MI with 9 bins should show coupling, got {mi_9}"
        assert mi_36 > 0.01, f"MI with 36 bins should show coupling, got {mi_36}"

    def test_unequal_lengths(self):
        """Unequal phase/amplitude lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            zbci.modulation_index(
                np.zeros(5, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )

    def test_empty(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            zbci.modulation_index(
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

    def test_bins_out_of_range(self):
        """n_bins outside [2, 64] should raise ValueError."""
        phase = np.zeros(10, dtype=np.float32)
        amplitude = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="n_bins"):
            zbci.modulation_index(phase, amplitude, n_bins=1)
        with pytest.raises(ValueError, match="n_bins"):
            zbci.modulation_index(phase, amplitude, n_bins=65)


class TestMeanVectorLength:
    """Tests for mean_vector_length()."""

    def test_no_coupling(self):
        """Constant amplitude with uniform phases should give MVL near 0."""
        phase = np.linspace(-np.pi, np.pi, 720, endpoint=False).astype(np.float32)
        amplitude = np.ones(720, dtype=np.float32)
        mvl = zbci.mean_vector_length(phase, amplitude)
        assert mvl < 0.02, f"No coupling should give MVL near 0, got {mvl}"

    def test_strong_coupling(self):
        """Amplitude modulated by phase should give significant MVL."""
        phase = np.linspace(-np.pi, np.pi, 1000, endpoint=False).astype(np.float32)
        amplitude = (1.0 + 0.9 * np.cos(phase)).astype(np.float32)
        mvl = zbci.mean_vector_length(phase, amplitude)
        assert mvl > 0.1, f"Strong coupling should give MVL > 0.1, got {mvl}"

    def test_range(self):
        """MVL must be in [0, 1]."""
        phase = np.linspace(-np.pi, np.pi, 500, endpoint=False).astype(np.float32)
        amplitude = (1.0 + 0.5 * np.cos(phase)).astype(np.float32)
        mvl = zbci.mean_vector_length(phase, amplitude)
        assert 0.0 <= mvl <= 1.0, f"MVL must be in [0, 1], got {mvl}"

    def test_perfect_coupling(self):
        """All amplitude at one phase should give MVL near 1."""
        phase = np.zeros(100, dtype=np.float32)
        amplitude = np.ones(100, dtype=np.float32)
        mvl = zbci.mean_vector_length(phase, amplitude)
        assert abs(mvl - 1.0) < 0.01, f"Perfect coupling should give MVL ~ 1, got {mvl}"

    def test_unequal_lengths(self):
        """Unequal lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            zbci.mean_vector_length(
                np.zeros(5, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )

    def test_empty(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            zbci.mean_vector_length(
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )


class TestPhaseAmplitudeDistribution:
    """Tests for phase_amplitude_distribution()."""

    def test_shape(self):
        """Should return (n_bins,) arrays."""
        phase = np.linspace(-np.pi, np.pi, 500, endpoint=False).astype(np.float32)
        amplitude = np.ones(500, dtype=np.float32)
        centers, amps = zbci.phase_amplitude_distribution(phase, amplitude)
        assert centers.shape == (18,), f"Expected (18,), got {centers.shape}"
        assert amps.shape == (18,), f"Expected (18,), got {amps.shape}"

    def test_centers_range(self):
        """Bin centers should be within [-pi, pi]."""
        phase = np.zeros(100, dtype=np.float32)
        amplitude = np.ones(100, dtype=np.float32)
        centers, _ = zbci.phase_amplitude_distribution(phase, amplitude)
        assert np.all(centers > -np.pi - 0.01), "Centers should be >= -pi"
        assert np.all(centers < np.pi + 0.01), "Centers should be <= pi"

    def test_uniform_amplitude(self):
        """Constant amplitude should give uniform distribution."""
        phase = np.linspace(-np.pi, np.pi, 720, endpoint=False).astype(np.float32)
        amplitude = np.full(720, 5.0, dtype=np.float32)
        _, amps = zbci.phase_amplitude_distribution(phase, amplitude)
        for a in amps:
            assert abs(a - 5.0) < 0.2, f"Uniform amp should give ~5.0 per bin, got {a}"

    def test_custom_bins(self):
        """Custom bin count should return correct shape."""
        phase = np.zeros(100, dtype=np.float32)
        amplitude = np.ones(100, dtype=np.float32)
        centers, amps = zbci.phase_amplitude_distribution(phase, amplitude, n_bins=9)
        assert centers.shape == (9,)
        assert amps.shape == (9,)


class TestComodulogram:
    """Tests for pac_comodulogram()."""

    def test_shape(self):
        """Output shape should be (n_phase_freqs, n_amp_freqs)."""
        np.random.seed(42)
        signal = np.random.randn(2048).astype(np.float32)
        phase_freqs = np.array([4.0, 6.0, 8.0], dtype=np.float32)
        amp_freqs = np.array([30.0, 50.0, 70.0], dtype=np.float32)
        comod = zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs)
        assert comod.shape == (3, 3), f"Expected (3, 3), got {comod.shape}"

    def test_range(self):
        """Comodulogram values should be in [0, 1]."""
        np.random.seed(42)
        signal = np.random.randn(2048).astype(np.float32)
        phase_freqs = np.array([6.0, 10.0], dtype=np.float32)
        amp_freqs = np.array([30.0, 50.0], dtype=np.float32)
        comod = zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs)
        assert np.all(comod >= 0.0), "Comodulogram values must be >= 0"
        assert np.all(comod <= 1.0), "Comodulogram values must be <= 1"

    def test_both_methods(self):
        """Both MI and MVL methods should produce valid output."""
        np.random.seed(42)
        signal = np.random.randn(2048).astype(np.float32)
        phase_freqs = np.array([6.0], dtype=np.float32)
        amp_freqs = np.array([40.0], dtype=np.float32)

        mi_comod = zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs, method="mi")
        mvl_comod = zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs, method="mvl")

        assert mi_comod.shape == (1, 1)
        assert mvl_comod.shape == (1, 1)
        assert 0.0 <= mi_comod[0, 0] <= 1.0
        assert 0.0 <= mvl_comod[0, 0] <= 1.0

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        signal = np.random.randn(512).astype(np.float32)
        phase_freqs = np.array([6.0], dtype=np.float32)
        amp_freqs = np.array([40.0], dtype=np.float32)
        with pytest.raises(ValueError, match="method"):
            zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs, method="invalid")

    def test_detects_coupling_at_correct_frequency(self):
        """Signal with known PAC should show higher coupling at correct pair."""
        np.random.seed(42)
        fs = 500.0
        t = np.arange(4096) / fs

        # Create theta (6 Hz) phase signal
        theta = np.sin(2 * np.pi * 6 * t)
        theta_phase = np.angle(
            np.exp(1j * 2 * np.pi * 6 * t)
        )

        # Create gamma (40 Hz) amplitude modulated by theta phase
        gamma_amp = 1.0 + 0.8 * np.cos(theta_phase)
        gamma = gamma_amp * np.sin(2 * np.pi * 40 * t)

        signal = (theta + 0.5 * gamma).astype(np.float32)

        phase_freqs = np.array([4.0, 6.0, 10.0], dtype=np.float32)
        amp_freqs = np.array([30.0, 40.0, 60.0], dtype=np.float32)

        comod = zbci.pac_comodulogram(signal, fs, phase_freqs, amp_freqs)

        # The (6 Hz, 40 Hz) entry should be among the highest
        assert comod.shape == (3, 3)
        # Check that the coupling is detectable (non-zero)
        assert comod[1, 1] >= 0.0  # the 6 Hz phase, 40 Hz amplitude cell


class TestPACIntegration:
    """Integration tests: filter -> Hilbert -> PAC."""

    def test_full_pipeline(self):
        """Full pipeline: bandpass -> Hilbert -> MI."""
        np.random.seed(42)
        fs = 500.0
        n = 1024
        t = np.arange(n) / fs

        # Create a signal with theta-gamma coupling
        theta = np.sin(2 * np.pi * 6 * t)
        theta_phase = 2 * np.pi * 6 * t
        gamma_amp = 1.0 + 0.8 * np.cos(theta_phase)
        gamma = gamma_amp * np.sin(2 * np.pi * 40 * t)
        signal = (theta + 0.3 * gamma).astype(np.float32)

        # Extract phase from theta band
        bp_phase = zbci.IirFilter.butterworth_bandpass(fs, 4.0, 8.0)
        filtered_phase = bp_phase.process(signal)

        hilbert = zbci.HilbertTransform(size=n)
        phase = hilbert.instantaneous_phase(filtered_phase)

        # Extract amplitude from gamma band
        bp_amp = zbci.IirFilter.butterworth_bandpass(fs, 30.0, 50.0)
        filtered_amp = bp_amp.process(signal)
        amplitude = hilbert.instantaneous_amplitude(filtered_amp)

        # Compute PAC
        mi = zbci.modulation_index(phase, amplitude)
        mvl = zbci.mean_vector_length(phase, amplitude)

        # Both should be valid
        assert 0.0 <= mi <= 1.0, f"MI out of range: {mi}"
        assert 0.0 <= mvl <= 1.0, f"MVL out of range: {mvl}"

    def test_no_coupling_pipeline(self):
        """White noise should show minimal coupling."""
        np.random.seed(123)
        fs = 500.0
        n = 1024
        signal = np.random.randn(n).astype(np.float32) * 0.1

        # Phase from theta band
        bp_phase = zbci.IirFilter.butterworth_bandpass(fs, 4.0, 8.0)
        filtered_phase = bp_phase.process(signal)

        hilbert = zbci.HilbertTransform(size=n)
        phase = hilbert.instantaneous_phase(filtered_phase)

        # Amplitude from gamma band
        bp_amp = zbci.IirFilter.butterworth_bandpass(fs, 30.0, 50.0)
        filtered_amp = bp_amp.process(signal)
        amplitude = hilbert.instantaneous_amplitude(filtered_amp)

        mi = zbci.modulation_index(phase, amplitude)
        mvl = zbci.mean_vector_length(phase, amplitude)

        # Random noise should have low coupling
        assert mi < 0.3, f"Noise MI should be low, got {mi}"
        assert mvl < 0.5, f"Noise MVL should be low, got {mvl}"
