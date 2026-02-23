"""Tests for CCA and SSVEP detection Python bindings."""
import numpy as np
import pytest


class TestSsvepReferences:
    """Tests for ssvep_references function."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'ssvep_references')

    def test_basic_shape(self):
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 250, 10.0, n_harmonics=2)
        assert refs.shape == (250, 4)
        assert refs.dtype == np.float64

    def test_single_harmonic(self):
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 100, 10.0, n_harmonics=1)
        assert refs.shape == (100, 2)

    def test_three_harmonics(self):
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 100, 10.0, n_harmonics=3)
        assert refs.shape == (100, 6)

    def test_four_harmonics(self):
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 100, 10.0, n_harmonics=4)
        assert refs.shape == (100, 8)

    def test_initial_values(self):
        """At t=0, sin=0 and cos=1 for all harmonics."""
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 100, 10.0, n_harmonics=2)
        np.testing.assert_allclose(refs[0, 0], 0.0, atol=1e-10)  # sin(0) = 0
        np.testing.assert_allclose(refs[0, 1], 1.0, atol=1e-10)  # cos(0) = 1
        np.testing.assert_allclose(refs[0, 2], 0.0, atol=1e-10)  # sin(0) = 0
        np.testing.assert_allclose(refs[0, 3], 1.0, atol=1e-10)  # cos(0) = 1

    def test_frequency_correctness(self):
        """Verify the fundamental frequency is correct."""
        import zpybci as zbci
        sample_rate = 1000.0
        freq = 10.0
        refs = zbci.ssvep_references(sample_rate, 1000, freq, n_harmonics=1)
        # At t=25 samples (0.025 sec), sin(2*pi*10*0.025) = sin(pi/2) = 1
        t = 25
        expected = np.sin(2 * np.pi * freq * t / sample_rate)
        np.testing.assert_allclose(refs[t, 0], expected, atol=1e-10)

    def test_invalid_sample_rate(self):
        import zpybci as zbci
        with pytest.raises(ValueError):
            zbci.ssvep_references(0.0, 100, 10.0)

    def test_invalid_frequency(self):
        import zpybci as zbci
        with pytest.raises(ValueError):
            zbci.ssvep_references(250.0, 100, 0.0)

    def test_invalid_n_samples(self):
        import zpybci as zbci
        with pytest.raises(ValueError):
            zbci.ssvep_references(250.0, 0, 10.0)


class TestCca:
    """Tests for cca function."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'cca')

    def test_perfect_correlation(self):
        """When signals are the references, correlation should be ~1."""
        import zpybci as zbci
        refs = zbci.ssvep_references(250.0, 200, 12.0, n_harmonics=2)
        signals = refs[:, :2].copy()  # Use first 2 columns as signals
        corr = zbci.cca(signals, refs)
        assert corr.shape == (2,)  # min(2, 4)
        assert corr[0] > 0.99

    def test_uncorrelated(self):
        """Signals at different frequencies should have low correlation."""
        import zpybci as zbci
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t),
            np.cos(2 * np.pi * 10 * t),
        ])
        refs = zbci.ssvep_references(250.0, 500, 37.0, n_harmonics=2)
        corr = zbci.cca(signals, refs)
        assert corr[0] < 0.3

    def test_descending_order(self):
        """Canonical correlations should be in descending order."""
        import zpybci as zbci
        np.random.seed(42)
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(500),
            np.cos(2 * np.pi * 10 * t) + 0.5 * np.random.randn(500),
            np.sin(2 * np.pi * 10 * t + 1) + 0.5 * np.random.randn(500),
            np.cos(2 * np.pi * 10 * t + 2) + 0.5 * np.random.randn(500),
        ])
        refs = zbci.ssvep_references(250.0, 500, 10.0, n_harmonics=2)
        corr = zbci.cca(signals, refs)
        for i in range(len(corr) - 1):
            assert corr[i] >= corr[i + 1] - 1e-10

    def test_correlations_bounded(self):
        """All correlations should be in [0, 1]."""
        import zpybci as zbci
        np.random.seed(42)
        signals = np.random.randn(200, 4)
        refs = zbci.ssvep_references(250.0, 200, 10.0, n_harmonics=2)
        corr = zbci.cca(signals, refs)
        assert np.all(corr >= 0.0)
        assert np.all(corr <= 1.0)

    def test_8_channels(self):
        """Test with 8 channels."""
        import zpybci as zbci
        t = np.arange(300) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t + i * 0.3) + 0.3 * np.random.randn(300)
            for i in range(8)
        ])
        refs = zbci.ssvep_references(250.0, 300, 10.0, n_harmonics=2)
        corr = zbci.cca(signals, refs)
        assert corr.shape == (4,)  # min(8, 4)
        assert corr[0] > 0.3

    def test_16_channels(self):
        """Test with 16 channels."""
        import zpybci as zbci
        np.random.seed(42)
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 12 * t + i * 0.2) + 0.5 * np.random.randn(500)
            for i in range(16)
        ])
        refs = zbci.ssvep_references(250.0, 500, 12.0, n_harmonics=2)
        corr = zbci.cca(signals, refs)
        assert corr.shape == (4,)
        assert corr[0] > 0.3

    def test_dimension_mismatch(self):
        import zpybci as zbci
        signals = np.random.randn(100, 4)
        refs = np.random.randn(50, 4)  # Different n_samples
        with pytest.raises(ValueError):
            zbci.cca(signals, refs)

    def test_unsupported_channels(self):
        import zpybci as zbci
        signals = np.random.randn(100, 3)  # 3 channels not supported
        refs = np.random.randn(100, 4)
        with pytest.raises(ValueError):
            zbci.cca(signals, refs)


class TestSsvepDetect:
    """Tests for ssvep_detect function."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'ssvep_detect')

    def test_basic_detection(self):
        """Detect a clean 10 Hz signal."""
        import zpybci as zbci
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t + i * 0.3)
            for i in range(4)
        ])
        idx, corr = zbci.ssvep_detect(signals, 250.0, [8.0, 10.0, 12.0, 15.0])
        assert idx == 1  # 10 Hz is index 1
        assert corr > 0.9

    def test_detect_different_frequencies(self):
        """Test detection of each target frequency."""
        import zpybci as zbci
        target_freqs = [8.0, 10.0, 12.0, 15.0]
        correct = 0
        for true_idx, true_freq in enumerate(target_freqs):
            t = np.arange(500) / 250.0
            signals = np.column_stack([
                np.sin(2 * np.pi * true_freq * t + i * 0.3)
                for i in range(4)
            ])
            idx, _ = zbci.ssvep_detect(signals, 250.0, target_freqs)
            if idx == true_idx:
                correct += 1
        assert correct >= 3, f"Only {correct}/4 correct"

    def test_8_channel_detection(self):
        """Test with 8 channels."""
        import zpybci as zbci
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 12 * t + i * 0.3)
            for i in range(8)
        ])
        idx, corr = zbci.ssvep_detect(signals, 250.0, [8.0, 10.0, 12.0, 15.0])
        assert idx == 2  # 12 Hz

    def test_with_noise(self):
        """Test detection with noisy signals."""
        import zpybci as zbci
        np.random.seed(42)
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 15 * t + i * 0.3) + 0.5 * np.random.randn(500)
            for i in range(4)
        ])
        idx, _ = zbci.ssvep_detect(signals, 250.0, [8.0, 10.0, 12.0, 15.0])
        assert idx == 3  # 15 Hz

    def test_custom_harmonics(self):
        """Test with different n_harmonics."""
        import zpybci as zbci
        t = np.arange(500) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t + i * 0.3)
            for i in range(4)
        ])
        idx, corr = zbci.ssvep_detect(
            signals, 250.0, [8.0, 10.0, 12.0],
            n_harmonics=3,
        )
        assert idx == 1

    def test_empty_frequencies(self):
        import zpybci as zbci
        signals = np.random.randn(100, 4)
        with pytest.raises(ValueError):
            zbci.ssvep_detect(signals, 250.0, [])

    def test_insufficient_samples(self):
        import zpybci as zbci
        signals = np.random.randn(1, 4)
        with pytest.raises(ValueError):
            zbci.ssvep_detect(signals, 250.0, [10.0])

    def test_returns_tuple(self):
        """Result should be a (int, float) tuple."""
        import zpybci as zbci
        t = np.arange(200) / 250.0
        signals = np.column_stack([
            np.sin(2 * np.pi * 10 * t),
            np.cos(2 * np.pi * 10 * t),
        ])
        result = zbci.ssvep_detect(signals, 250.0, [10.0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
