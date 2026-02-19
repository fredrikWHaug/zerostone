"""Tests for cross-correlation and auto-correlation bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestXcorr:
    """Tests for xcorr function."""

    def test_xcorr_identical_signals(self):
        """Test xcorr of signal with itself."""
        signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        corr = zbci.xcorr(signal, signal)

        # Output length should be 2*N - 1
        assert corr.shape == (9,)

        # Peak should be at center (lag 0)
        center = 4
        assert corr[center] == np.max(corr)

    def test_xcorr_output_length(self):
        """Test xcorr output length."""
        x = np.random.randn(10).astype(np.float32)
        y = np.random.randn(7).astype(np.float32)
        corr = zbci.xcorr(x, y)

        # Output length = N + M - 1
        assert corr.shape == (16,)

    def test_xcorr_delayed_signal(self):
        """Test detecting time delay with xcorr."""
        x = np.array([0, 0, 1, 2, 1, 0, 0, 0], dtype=np.float32)
        y = np.array([0, 0, 0, 0, 1, 2, 1, 0], dtype=np.float32)  # x delayed by 2

        corr = zbci.xcorr(x, y)
        peak_idx, _ = zbci.find_peak(corr)
        lag = zbci.index_to_lag(peak_idx, len(y))

        assert lag == 2

    def test_xcorr_negative_delay(self):
        """Test detecting negative time delay."""
        x = np.array([0, 0, 0, 1, 2, 1, 0, 0], dtype=np.float32)
        y = np.array([0, 1, 2, 1, 0, 0, 0, 0], dtype=np.float32)  # y leads x

        corr = zbci.xcorr(x, y)
        peak_idx, _ = zbci.find_peak(corr)
        lag = zbci.index_to_lag(peak_idx, len(y))

        assert lag == -2

    def test_xcorr_normalization_none(self):
        """Test xcorr with no normalization."""
        x = np.ones(4, dtype=np.float32)
        y = np.ones(4, dtype=np.float32)
        corr = zbci.xcorr(x, y, normalization='none')

        # At lag 0, sum = 4
        center = 3
        assert abs(corr[center] - 4.0) < 0.001

    def test_xcorr_normalization_biased(self):
        """Test xcorr with biased normalization."""
        x = np.ones(4, dtype=np.float32)
        y = np.ones(4, dtype=np.float32)
        corr = zbci.xcorr(x, y, normalization='biased')

        # At lag 0, sum/N = 4/4 = 1.0
        center = 3
        assert abs(corr[center] - 1.0) < 0.001

    def test_xcorr_normalization_unbiased(self):
        """Test xcorr with unbiased normalization."""
        x = np.ones(4, dtype=np.float32)
        y = np.ones(4, dtype=np.float32)
        corr = zbci.xcorr(x, y, normalization='unbiased')

        # Unbiased: divide by overlap count at each lag
        # At all lags, result should be 1.0 for constant signal
        for val in corr:
            if val != 0:
                assert abs(val - 1.0) < 0.001

    def test_xcorr_normalization_coeff(self):
        """Test xcorr with coefficient normalization."""
        signal = np.random.randn(32).astype(np.float32)
        corr = zbci.xcorr(signal, signal, normalization='coeff')

        # Autocorr at lag 0 should be 1.0
        center = 31
        assert abs(corr[center] - 1.0) < 0.001

    def test_xcorr_invalid_normalization(self):
        """Test that invalid normalization raises error."""
        x = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown normalization"):
            zbci.xcorr(x, x, normalization='invalid')

    def test_xcorr_single_element(self):
        """Test xcorr with single-element signals."""
        x = np.array([5.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        corr = zbci.xcorr(x, y)

        assert corr.shape == (1,)
        assert abs(corr[0] - 15.0) < 0.001

    def test_xcorr_zero_signal(self):
        """Test xcorr with zero signal."""
        x = np.zeros(4, dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        corr = zbci.xcorr(x, y)

        assert np.all(corr == 0)


class TestAutocorr:
    """Tests for autocorr function."""

    def test_autocorr_output_length(self):
        """Test autocorr output length."""
        x = np.random.randn(10).astype(np.float32)
        acorr = zbci.autocorr(x)

        # Output length = 2*N - 1
        assert acorr.shape == (19,)

    def test_autocorr_symmetry(self):
        """Test that autocorr is symmetric."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        acorr = zbci.autocorr(x)

        # Autocorrelation must be symmetric
        n = len(acorr)
        for i in range(n // 2):
            assert abs(acorr[i] - acorr[n - 1 - i]) < 1e-6

    def test_autocorr_peak_at_center(self):
        """Test that autocorr peak is at center."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0], dtype=np.float32)
        acorr = zbci.autocorr(x)

        center = 4
        assert acorr[center] == np.max(acorr)

    def test_autocorr_normalization_coeff(self):
        """Test autocorr with coeff normalization."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        acorr = zbci.autocorr(x, normalization='coeff')

        # Center (lag 0) should be 1.0
        center = 4
        assert abs(acorr[center] - 1.0) < 1e-6

    def test_autocorr_normalization_biased(self):
        """Test autocorr with biased normalization."""
        x = np.ones(4, dtype=np.float32)
        acorr = zbci.autocorr(x, normalization='biased')

        # Lag 0: sum = 4, biased = 4/4 = 1.0
        center = 3
        assert abs(acorr[center] - 1.0) < 1e-6

        # Lag 1: sum = 3, biased = 3/4 = 0.75
        assert abs(acorr[center + 1] - 0.75) < 1e-6

    def test_autocorr_normalization_unbiased(self):
        """Test autocorr with unbiased normalization."""
        x = np.ones(4, dtype=np.float32)
        acorr = zbci.autocorr(x, normalization='unbiased')

        # All values should be 1.0 for constant signal
        for val in acorr:
            assert abs(val - 1.0) < 1e-6


class TestFindPeak:
    """Tests for find_peak function."""

    def test_find_peak_basic(self):
        """Test basic find_peak functionality."""
        corr = np.array([1.0, 3.0, 7.0, 5.0, 2.0], dtype=np.float32)
        idx, val = zbci.find_peak(corr)

        assert idx == 2
        assert val == 7.0

    def test_find_peak_first_element(self):
        """Test find_peak when peak is at start."""
        corr = np.array([10.0, 3.0, 2.0, 1.0], dtype=np.float32)
        idx, val = zbci.find_peak(corr)

        assert idx == 0
        assert val == 10.0

    def test_find_peak_last_element(self):
        """Test find_peak when peak is at end."""
        corr = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float32)
        idx, val = zbci.find_peak(corr)

        assert idx == 3
        assert val == 10.0

    def test_find_peak_empty(self):
        """Test find_peak with empty array."""
        corr = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            zbci.find_peak(corr)


class TestLagConversion:
    """Tests for lag conversion functions."""

    def test_index_to_lag(self):
        """Test index_to_lag conversion."""
        # For signals of length N=5 and M=3, output has 7 elements
        assert zbci.index_to_lag(0, 3) == -2
        assert zbci.index_to_lag(2, 3) == 0
        assert zbci.index_to_lag(6, 3) == 4

    def test_lag_to_index(self):
        """Test lag_to_index conversion."""
        assert zbci.lag_to_index(-2, 3) == 0
        assert zbci.lag_to_index(0, 3) == 2
        assert zbci.lag_to_index(4, 3) == 6

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion."""
        m = 5
        for idx in range(10):
            lag = zbci.index_to_lag(idx, m)
            idx_back = zbci.lag_to_index(lag, m)
            assert idx_back == idx


class TestXcorrApplications:
    """Application-oriented tests for cross-correlation."""

    def test_time_delay_estimation(self):
        """Test time delay estimation between signals."""
        # Create original signal
        t = np.arange(64) / 256.0
        original = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        # Create delayed version (delay by 5 samples)
        delay = 5
        delayed = np.zeros_like(original)
        delayed[delay:] = original[:-delay]

        corr = zbci.xcorr(original, delayed)
        peak_idx, _ = zbci.find_peak(corr)
        estimated_delay = zbci.index_to_lag(peak_idx, len(delayed))

        assert abs(estimated_delay - delay) <= 1

    def test_template_matching(self):
        """Test template matching with xcorr."""
        # Create signal with embedded template
        signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0], dtype=np.float32)
        template = np.array([1, 2, 3, 2, 1], dtype=np.float32)

        corr = zbci.xcorr(signal, template)

        # Verify xcorr output length
        assert len(corr) == len(signal) + len(template) - 1

        peak_idx, peak_val = zbci.find_peak(corr)

        # The peak value should be sum of template * template = 19
        expected_max = np.sum(template * template)
        assert abs(peak_val - expected_max) < 0.1, f"Expected peak ~{expected_max}, got {peak_val}"

        # Verify correlation is highest where template matches signal
        assert peak_val == np.max(corr)

    def test_periodicity_detection(self):
        """Test using autocorr for periodicity detection."""
        # Create periodic signal (period of 20 samples)
        period = 20
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / period).astype(np.float32)

        acorr = zbci.autocorr(signal, normalization='coeff')

        # Find peaks in autocorr (excluding center)
        center = len(signal) - 1
        # Look for next peak after center
        right_half = acorr[center + 5:]  # Skip center region
        local_peak_idx = np.argmax(right_half)

        detected_period = local_peak_idx + 5
        assert abs(detected_period - period) <= 2
