"""Tests for window function Python bindings."""
import numpy as np
import pytest


class TestApplyWindow:
    """Tests for apply_window."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'apply_window')

    def test_hann_window(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        windowed = npy.apply_window(signal, "hann")
        assert windowed.shape == (256,)
        assert windowed.dtype == np.float32
        # Endpoints should be near zero for Hann
        assert windowed[0] < 0.01
        assert windowed[-1] < 0.01
        # Middle should be near 1
        assert windowed[128] > 0.9

    def test_hamming_window(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        windowed = npy.apply_window(signal, "hamming")
        assert windowed.shape == (256,)
        # Hamming doesn't go to zero at endpoints
        assert windowed[0] > 0.05

    def test_blackman_window(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        windowed = npy.apply_window(signal, "blackman")
        assert windowed.shape == (256,)

    def test_blackman_harris_window(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        windowed = npy.apply_window(signal, "blackman_harris")
        assert windowed.shape == (256,)

    def test_rectangular_window(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        windowed = npy.apply_window(signal, "rectangular")
        # Rectangular window should not modify signal
        np.testing.assert_allclose(windowed, signal)

    def test_does_not_modify_input(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        original = signal.copy()
        _ = npy.apply_window(signal, "hann")
        np.testing.assert_array_equal(signal, original)

    def test_invalid_window_type(self):
        import npyci as npy
        signal = np.ones(256, dtype=np.float32)
        with pytest.raises(ValueError):
            npy.apply_window(signal, "invalid")

    def test_window_aliases(self):
        import npyci as npy
        signal = np.ones(64, dtype=np.float32)
        # hanning is alias for hann
        w1 = npy.apply_window(signal, "hann")
        w2 = npy.apply_window(signal, "hanning")
        np.testing.assert_allclose(w1, w2)


class TestWindowCoefficient:
    """Tests for window_coefficient."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'window_coefficient')

    def test_hann_coefficient(self):
        import npyci as npy
        # Middle of Hann window should be 1.0
        length = 256
        coeff = npy.window_coefficient("hann", length // 2, length)
        assert isinstance(coeff, float)
        assert coeff > 0.9

    def test_rectangular_coefficient(self):
        import npyci as npy
        # All rectangular coefficients should be 1.0
        for i in range(10):
            coeff = npy.window_coefficient("rectangular", i, 10)
            assert np.isclose(coeff, 1.0)


class TestCoherentGain:
    """Tests for coherent_gain."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'coherent_gain')

    def test_rectangular_gain(self):
        import npyci as npy
        gain = npy.coherent_gain("rectangular", 256)
        # Rectangular window has coherent gain of 1.0
        assert np.isclose(gain, 1.0)

    def test_hann_gain(self):
        import npyci as npy
        gain = npy.coherent_gain("hann", 256)
        # Hann window coherent gain is ~0.5
        assert 0.4 < gain < 0.6


class TestEquivalentNoiseBandwidth:
    """Tests for equivalent_noise_bandwidth."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'equivalent_noise_bandwidth')

    def test_rectangular_enbw(self):
        import npyci as npy
        enbw = npy.equivalent_noise_bandwidth("rectangular", 256)
        # Rectangular has ENBW = 1.0
        assert np.isclose(enbw, 1.0)

    def test_hann_enbw(self):
        import npyci as npy
        enbw = npy.equivalent_noise_bandwidth("hann", 256)
        # Hann ENBW is ~1.5
        assert 1.4 < enbw < 1.6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
