"""Tests for ISI analysis functions."""
import numpy as np
import pytest
import zpybci as zbci


class TestIsiCv:
    def test_constant_intervals(self):
        """CV of constant ISIs should be 0."""
        spike_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        cv = zbci.isi_cv(spike_times)
        assert cv < 1e-10

    def test_variable_intervals(self):
        """CV of variable ISIs should be > 0."""
        spike_times = np.array([0.0, 0.01, 0.05, 0.06, 0.15, 0.16])
        cv = zbci.isi_cv(spike_times)
        assert cv > 0.1

    def test_too_few_spikes(self):
        """CV with < 2 spikes should be 0."""
        assert zbci.isi_cv(np.array([0.0])) == 0.0
        assert zbci.isi_cv(np.array([])) == 0.0

    def test_two_spikes(self):
        """CV with exactly 2 spikes (1 ISI) should be 0."""
        cv = zbci.isi_cv(np.array([0.0, 0.1]))
        assert cv < 1e-10


class TestLocalVariation:
    def test_regular_firing(self):
        """Regular firing should have Lv near 0."""
        spike_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        lv = zbci.local_variation(spike_times)
        assert lv < 1e-10

    def test_too_few_spikes(self):
        """Lv with < 3 spikes should be 0."""
        assert zbci.local_variation(np.array([0.0, 0.1])) == 0.0
        assert zbci.local_variation(np.array([0.0])) == 0.0
        assert zbci.local_variation(np.array([])) == 0.0

    def test_irregular_firing(self):
        """Irregular firing should have Lv > 0."""
        spike_times = np.array([0.0, 0.01, 0.10, 0.11, 0.20, 0.21])
        lv = zbci.local_variation(spike_times)
        assert lv > 0.1


class TestIsiHistogram:
    def test_basic(self):
        """Histogram bins should accumulate correctly."""
        # Use integer-scale times to avoid fp precision issues
        spike_times = np.array([0.0, 20.0, 40.0, 60.0, 80.0])  # ms units
        bins, overflow, mean, std, cv = zbci.isi_histogram(spike_times, 10.0)
        assert overflow == 0
        assert bins[2] == 4  # bin 2 covers [20, 30)
        assert abs(mean - 20.0) < 1e-10
        assert std < 1e-10
        assert cv < 1e-10

    def test_overflow(self):
        """Intervals beyond histogram range go to overflow."""
        spike_times = np.array([0.0, 1.0])  # 1s interval, default 100 bins * any width
        bins, overflow, mean, std, cv = zbci.isi_histogram(spike_times, 0.001, n_bins=10)
        assert overflow == 1
        assert bins.sum() == 0

    def test_custom_bins(self):
        """Custom bin count should work."""
        spike_times = np.array([0.0, 0.005, 0.010])
        bins, overflow, mean, std, cv = zbci.isi_histogram(spike_times, 0.001, n_bins=50)
        assert len(bins) == 50
        assert bins[5] == 2  # 5ms intervals

    def test_empty(self):
        """Empty spike train should return zeros."""
        spike_times = np.array([])
        bins, overflow, mean, std, cv = zbci.isi_histogram(spike_times, 0.001)
        assert overflow == 0
        assert mean == 0.0

    def test_invalid_bin_width(self):
        """Negative bin width should raise."""
        with pytest.raises(ValueError):
            zbci.isi_histogram(np.array([0.0, 0.1]), -0.001)


class TestBurstIndex:
    def test_no_bursts(self):
        """Long intervals should give burst index 0."""
        spike_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        bi = zbci.burst_index(spike_times, 0.010)
        assert bi < 1e-10

    def test_all_bursts(self):
        """Short intervals should give burst index 1."""
        spike_times = np.array([0.0, 0.003, 0.006, 0.009, 0.012])
        bi = zbci.burst_index(spike_times, 0.010)
        assert abs(bi - 1.0) < 1e-10

    def test_mixed(self):
        """Mixed intervals should give proportional burst index."""
        spike_times = np.array([0.0, 0.005, 0.010, 0.110, 0.210])
        bi = zbci.burst_index(spike_times, 0.010)
        assert abs(bi - 0.5) < 1e-10

    def test_empty(self):
        """Empty train should return 0."""
        assert zbci.burst_index(np.array([]), 0.01) == 0.0
        assert zbci.burst_index(np.array([0.0]), 0.01) == 0.0

    def test_range(self):
        """Burst index should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        spike_times = np.sort(rng.uniform(0, 10, size=200))
        bi = zbci.burst_index(spike_times, 0.01)
        assert 0.0 <= bi <= 1.0


class TestAutocorrelogram:
    def test_regular_spikes(self):
        """Regular spikes should show peaks at multiples of the interval."""
        spike_times = np.array([0.0, 0.010, 0.020, 0.030, 0.040])
        acg = zbci.autocorrelogram(spike_times, 0.005, 0.050)
        assert len(acg) == 10
        # Should have counts at lag = 10ms (bin 2), 20ms (bin 4), etc.
        assert acg[2] > 0
        assert acg[4] > 0

    def test_empty(self):
        """Empty spike train should produce zero autocorrelogram."""
        acg = zbci.autocorrelogram(np.array([]), 0.001, 0.050)
        assert all(v == 0 for v in acg)

    def test_single_spike(self):
        """Single spike should produce zero autocorrelogram."""
        acg = zbci.autocorrelogram(np.array([0.0]), 0.001, 0.050)
        assert all(v == 0 for v in acg)

    def test_invalid_params(self):
        """Invalid parameters should raise."""
        with pytest.raises(ValueError):
            zbci.autocorrelogram(np.array([0.0, 0.1]), -0.001, 0.050)
        with pytest.raises(ValueError):
            zbci.autocorrelogram(np.array([0.0, 0.1]), 0.001, -0.050)

    def test_refractory_gap(self):
        """Autocorrelogram should show refractory gap at very short lags."""
        # Spikes with 10ms refractory period
        spike_times = np.arange(0, 1.0, 0.010)  # 100 spikes, 10ms apart
        acg = zbci.autocorrelogram(spike_times, 0.001, 0.050)
        # Bin 0 ([0, 1ms)) should be empty (no spikes within 1ms)
        assert acg[0] == 0
