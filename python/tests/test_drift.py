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
