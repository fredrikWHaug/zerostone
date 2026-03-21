"""Tests for drift estimation bindings."""
import numpy as np
import pytest
import zpybci as zbci


class TestDriftEstimator:
    def test_basic_drift(self):
        est = zbci.DriftEstimator(1000)
        for i in range(8):
            est.add_spike(i * 1000 + 500, 100.0 + i * 10.0)
        est.fit()
        assert est.is_fitted
        assert est.slope > 0.0
        assert abs(est.slope - 0.01) < 0.002

    def test_no_drift(self):
        est = zbci.DriftEstimator(1000)
        for i in range(8):
            est.add_spike(i * 1000 + 500, 100.0)
        est.fit()
        assert abs(est.slope) < 1e-10

    def test_correct_position(self):
        est = zbci.DriftEstimator(1000)
        for i in range(8):
            est.add_spike(i * 1000 + 500, 100.0 + i * 10.0)
        est.fit()
        corrected_0 = est.correct_position(500, 100.0)
        corrected_7 = est.correct_position(7500, 170.0)
        assert abs(corrected_0 - corrected_7) < 3.0

    def test_estimate_drift_before_fit(self):
        est = zbci.DriftEstimator(1000)
        assert est.estimate_drift(5000) == 0.0

    def test_reset(self):
        est = zbci.DriftEstimator(1000)
        est.add_spike(500, 100.0)
        est.add_spike(1500, 110.0)
        est.fit()
        assert est.is_fitted
        est.reset()
        assert not est.is_fitted
        assert est.n_bins_used == 0

    def test_repr(self):
        est = zbci.DriftEstimator(1000)
        r = repr(est)
        assert "DriftEstimator" in r

    def test_zero_bin_duration_raises(self):
        with pytest.raises((ValueError, Exception)):
            zbci.DriftEstimator(0)


class TestEstimateDriftFromPositions:
    def test_basic(self):
        indices = np.array([100, 1100, 2100, 3100], dtype=np.int64)
        positions = np.array([50.0, 60.0, 70.0, 80.0])
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 8)
        assert result is not None
        slope, intercept = result
        assert abs(slope - 0.01) < 0.002

    def test_single_bin_returns_none(self):
        indices = np.array([100, 200, 300], dtype=np.int64)
        positions = np.array([50.0, 55.0, 60.0])
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 8)
        assert result is None

    def test_empty_returns_none(self):
        indices = np.array([], dtype=np.int64)
        positions = np.array([], dtype=np.float64)
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 8)
        assert result is None


class TestDriftExpanded:
    def test_drift_empty_positions(self):
        indices = np.array([], dtype=np.int64)
        positions = np.array([], dtype=np.float64)
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 256)
        assert result is None

    def test_drift_single_position(self):
        indices = np.array([500], dtype=np.int64)
        positions = np.array([100.0])
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 256)
        # Single bin, need at least 2 for regression
        assert result is None

    def test_drift_constant_position(self):
        indices = np.array([500, 1500, 2500, 3500, 4500], dtype=np.int64)
        positions = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 256)
        assert result is not None
        slope, _ = result
        assert abs(slope) < 1e-10

    def test_drift_linear(self):
        # Linear drift: position increases by 10 per bin
        indices = np.array([500, 1500, 2500, 3500, 4500, 5500, 6500, 7500],
                           dtype=np.int64)
        positions = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0])
        result = zbci.estimate_drift_from_positions(indices, positions, 1000, 256)
        assert result is not None
        slope, _ = result
        # slope should be ~0.01 (10 um / 1000 samples)
        assert abs(slope - 0.01) < 0.002

    def test_drift_estimator_no_spikes(self):
        est = zbci.DriftEstimator(1000)
        est.fit()
        assert not est.is_fitted or abs(est.slope) < 1e-10
        assert est.n_bins_used == 0

    def test_drift_estimator_single_spike(self):
        est = zbci.DriftEstimator(1000)
        est.add_spike(500, 100.0)
        est.fit()
        # Only one bin, can't fit regression
        assert est.n_bins_used <= 1

    def test_drift_estimator_multiple_bins(self):
        est = zbci.DriftEstimator(500)
        for i in range(10):
            est.add_spike(i * 500 + 250, 50.0 + i * 5.0)
        est.fit()
        assert est.is_fitted
        assert est.n_bins_used >= 2
        assert est.slope > 0.0

    def test_drift_estimator_negative_positions(self):
        est = zbci.DriftEstimator(1000)
        for i in range(8):
            est.add_spike(i * 1000 + 500, -100.0 + i * 10.0)
        est.fit()
        assert est.is_fitted
        assert est.slope > 0.0
        # Verify correct_position works with negative values
        corrected = est.correct_position(500, -100.0)
        assert isinstance(corrected, float)
