"""Tests for StreamingPercentile bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestStreamingPercentile:
    """Tests for StreamingPercentile."""

    def test_create_percentile(self):
        """Test creating a streaming percentile estimator."""
        est = zbci.StreamingPercentile(channels=4, percentile=0.5)
        assert est.channels == 4
        assert abs(est.target - 0.5) < 0.001
        assert not est.is_initialized
        assert est.count == 0

    def test_create_median(self):
        """Test creating median estimator."""
        est = zbci.StreamingPercentile.median(channels=8)
        assert est.channels == 8
        assert abs(est.target - 0.5) < 0.001

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        # Invalid channels
        with pytest.raises(ValueError):
            zbci.StreamingPercentile(channels=3, percentile=0.5)

        # Percentile too low
        with pytest.raises(ValueError):
            zbci.StreamingPercentile(channels=4, percentile=0.0)

        # Percentile too high
        with pytest.raises(ValueError):
            zbci.StreamingPercentile(channels=4, percentile=1.0)

    def test_supported_channel_counts(self):
        """Test all supported channel counts."""
        for channels in [1, 4, 8, 16, 32, 64]:
            est = zbci.StreamingPercentile(channels=channels, percentile=0.5)
            assert est.channels == channels

    def test_initialization_requires_5_samples(self):
        """Test that estimator needs 5 samples to initialize."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for i in range(4):
            est.update(np.array([float(i)], dtype=np.float64))
            assert not est.is_initialized
            assert est.percentile() is None

        est.update(np.array([4.0], dtype=np.float64))
        assert est.is_initialized
        assert est.percentile() is not None

    def test_median_simple_sequence(self):
        """Test median estimation of simple sequence."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        # Feed 1, 2, 3, 4, 5 - median should be 3
        for i in range(1, 6):
            est.update(np.array([float(i)], dtype=np.float64))

        median = est.percentile()[0]
        assert abs(median - 3.0) < 0.5

    def test_median_convergence(self):
        """Test median convergence with many samples."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        # Feed uniform sequence 0..999
        for i in range(1000):
            est.update(np.array([float(i)], dtype=np.float64))

        # True median is ~499.5
        median = est.percentile()[0]
        assert abs(median - 499.5) < 50.0

    def test_8th_percentile(self):
        """Test 8th percentile (common for baseline estimation)."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.08)

        # Feed uniform sequence 0..999
        for i in range(1000):
            est.update(np.array([float(i)], dtype=np.float64))

        # True 8th percentile is ~80
        p8 = est.percentile()[0]
        assert abs(p8 - 80.0) < 50.0

    def test_multi_channel_independence(self):
        """Test that channels are processed independently."""
        est = zbci.StreamingPercentile(channels=4, percentile=0.5)

        # Channel 0: 0-999, Channel 1: 1000-1999, Channel 2: 2000-2999, Channel 3: 3000-3999
        for i in range(1000):
            est.update(np.array([float(i), float(i + 1000), float(i + 2000), float(i + 3000)], dtype=np.float64))

        medians = est.percentile()

        # Channel 0 median should be ~500
        assert abs(medians[0] - 500.0) < 100.0

        # Channel 1 median should be ~1500
        assert abs(medians[1] - 1500.0) < 100.0

        # Channel 2 median should be ~2500
        assert abs(medians[2] - 2500.0) < 100.0

        # Channel 3 median should be ~3500
        assert abs(medians[3] - 3500.0) < 100.0

    def test_min_max(self):
        """Test min and max tracking."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for i in range(100):
            est.update(np.array([float(i)], dtype=np.float64))

        min_val = est.min()[0]
        max_val = est.max()[0]

        assert min_val == 0.0
        assert max_val == 99.0

    def test_reset(self):
        """Test resetting estimator."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for i in range(100):
            est.update(np.array([float(i)], dtype=np.float64))

        assert est.is_initialized
        assert est.count == 100

        est.reset()

        assert not est.is_initialized
        assert est.count == 0
        assert est.percentile() is None

        # Should work again after reset
        for i in range(5):
            est.update(np.array([float(i)], dtype=np.float64))
        assert est.is_initialized

    def test_channel_mismatch(self):
        """Test that channel mismatch raises error."""
        est = zbci.StreamingPercentile(channels=4, percentile=0.5)

        with pytest.raises(ValueError, match="expected 4"):
            est.update(np.array([1.0, 2.0], dtype=np.float64))

    def test_repr(self):
        """Test string representation."""
        est = zbci.StreamingPercentile(channels=8, percentile=0.08)
        r = repr(est)
        assert "8" in r
        assert "0.08" in r


class TestStreamingPercentileApplications:
    """Application-oriented tests for streaming percentile."""

    def test_baseline_estimation(self):
        """Test using 8th percentile for baseline estimation."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.08)

        np.random.seed(42)

        # Simulate signal with baseline + spikes
        baseline = 100.0
        for _ in range(1000):
            # Mostly baseline values, occasional spikes
            if np.random.random() < 0.9:
                value = baseline + np.random.randn() * 5
            else:
                value = baseline + 50 + np.random.randn() * 10  # Spike
            est.update(np.array([value], dtype=np.float64))

        # 8th percentile should be close to baseline
        estimated_baseline = est.percentile()[0]
        assert abs(estimated_baseline - baseline) < 20.0

    def test_constant_signal(self):
        """Test with constant signal."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for _ in range(100):
            est.update(np.array([42.0], dtype=np.float64))

        median = est.percentile()[0]
        assert median == 42.0

    def test_reverse_sorted_input(self):
        """Test with reverse-sorted input."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for i in range(999, -1, -1):
            est.update(np.array([float(i)], dtype=np.float64))

        median = est.percentile()[0]
        assert abs(median - 499.5) < 50.0

    def test_sorted_input(self):
        """Test with sorted input."""
        est = zbci.StreamingPercentile(channels=1, percentile=0.5)

        for i in range(1000):
            est.update(np.array([float(i)], dtype=np.float64))

        median = est.percentile()[0]
        assert abs(median - 499.5) < 50.0

    def test_extreme_percentiles(self):
        """Test extreme percentile values."""
        # 1st percentile
        est_low = zbci.StreamingPercentile(channels=1, percentile=0.01)
        for i in range(10000):
            est_low.update(np.array([float(i)], dtype=np.float64))
        p1 = est_low.percentile()[0]
        assert p1 < 500.0

        # 99th percentile
        est_high = zbci.StreamingPercentile(channels=1, percentile=0.99)
        for i in range(10000):
            est_high.update(np.array([float(i)], dtype=np.float64))
        p99 = est_high.percentile()[0]
        assert p99 > 9500.0
