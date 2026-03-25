"""Tests for multi-channel sorting pipeline and online sorter."""

import numpy as np
import pytest

import zpybci as zbci


# ---- Helpers ----


def make_multichannel_data(n_samples, n_channels, rng, spike_positions=None):
    """Generate multi-channel data with optional injected spikes."""
    data = rng.standard_normal((n_samples, n_channels)) * 0.5
    if spike_positions:
        for sample, channel, amplitude in spike_positions:
            if 0 <= sample < n_samples and 0 <= channel < n_channels:
                data[sample, channel] += -amplitude
    return data


# ---- sort_multichannel tests ----


class TestSortMultichannel:
    def test_sort_basic_4ch_keys(self):
        np.random.seed(42)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        # Inject a few large spikes
        for t in [500, 1500, 2500, 3500]:
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        assert "n_spikes" in result
        assert "n_clusters" in result
        assert "labels" in result
        assert "clusters" in result

    def test_sort_basic_4ch(self):
        np.random.seed(43)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert result["n_spikes"] >= 0
        assert result["n_clusters"] >= 0

    def test_sort_returns_labels(self):
        np.random.seed(44)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        for t in [500, 1500, 2500, 3500]:
            data[t, 1] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        labels = result["labels"]
        assert isinstance(labels, np.ndarray)
        assert len(labels) == result["n_spikes"]

    def test_sort_returns_clusters(self):
        np.random.seed(45)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(8000, 4)
        for t in range(200, 7000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        clusters = result["clusters"]
        assert isinstance(clusters, list)
        assert len(clusters) == result["n_clusters"]
        for cl in clusters:
            assert "count" in cl
            assert "snr" in cl
            assert "isi_violation_rate" in cl

    def test_sort_zero_data(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.zeros((5000, 4))
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert result["n_spikes"] == 0

    def test_sort_high_threshold(self):
        np.random.seed(46)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4) * 0.1
        result = zbci.sort_multichannel(data, probe, threshold=100.0)
        assert result["n_spikes"] == 0

    def test_sort_custom_params(self):
        np.random.seed(47)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        for t in [500, 1500, 2500]:
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe,
            threshold=3.0,
            refractory=20,
            spatial_radius=100.0,
        )
        assert "n_spikes" in result

    def test_sort_requires_probe(self):
        data = np.random.randn(5000, 4)
        with pytest.raises(TypeError):
            zbci.sort_multichannel(data)

    def test_sort_invalid_channels(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        # 5 channels doesn't match the probe's 4 channels
        data = np.random.randn(5000, 5)
        with pytest.raises(ValueError):
            zbci.sort_multichannel(data, probe, threshold=5.0)

    def test_sort_short_data(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.zeros((10, 4))
        with pytest.raises(ValueError, match="Insufficient"):
            zbci.sort_multichannel(data, probe, threshold=5.0)

    def test_sort_noise_only(self):
        np.random.seed(48)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        # Should not crash, even if some threshold crossings happen
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert result["n_spikes"] >= 0

    def test_sort_deterministic(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(99)
        data = rng.standard_normal((5000, 4))
        for t in [500, 1500, 2500]:
            data[t, 0] += -15.0
        data2 = data.copy()
        r1 = zbci.sort_multichannel(data, probe, threshold=4.0)
        r2 = zbci.sort_multichannel(data2, probe, threshold=4.0)
        assert r1["n_spikes"] == r2["n_spikes"]
        assert r1["n_clusters"] == r2["n_clusters"]
        np.testing.assert_array_equal(r1["labels"], r2["labels"])

    def test_sort_8ch(self):
        np.random.seed(50)
        probe = zbci.ProbeLayout.linear(8, 25.0)
        data = np.random.randn(5000, 8)
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert result["n_spikes"] >= 0

    def test_sort_16ch(self):
        np.random.seed(51)
        probe = zbci.ProbeLayout.linear(16, 25.0)
        data = np.random.randn(5000, 16)
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert result["n_spikes"] >= 0

    def test_sort_returns_spike_times(self):
        np.random.seed(60)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(8000, 4)
        for t in range(200, 7000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        assert "spike_times" in result
        times = result["spike_times"]
        assert isinstance(times, np.ndarray)
        assert len(times) == result["n_spikes"]
        # Times should be non-negative
        if len(times) > 0:
            assert np.all(times >= 0)

    def test_sort_returns_spike_channels(self):
        np.random.seed(61)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(8000, 4)
        for t in range(200, 7000, 200):
            data[t, 1] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        assert "spike_channels" in result
        channels = result["spike_channels"]
        assert isinstance(channels, np.ndarray)
        assert len(channels) == result["n_spikes"]
        # Channels should be in valid range
        if len(channels) > 0:
            assert np.all(channels >= 0)
            assert np.all(channels < 4)

    def test_sort_spike_times_sorted(self):
        np.random.seed(62)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(8000, 4)
        for t in range(200, 7000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        times = result["spike_times"]
        if len(times) > 1:
            # After dedup + alignment, spike times should be non-decreasing
            assert np.all(np.diff(times) >= 0)

    def test_sort_zero_data_spike_times(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.zeros((5000, 4))
        result = zbci.sort_multichannel(data, probe, threshold=5.0)
        assert "spike_times" in result
        assert len(result["spike_times"]) == 0
        assert "spike_channels" in result
        assert len(result["spike_channels"]) == 0

    def test_sort_spike_times_consistent_with_labels(self):
        np.random.seed(63)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(8000, 4)
        for t in range(200, 7000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(data, probe, threshold=4.0)
        # spike_times, spike_channels, and labels should all have same length
        assert len(result["spike_times"]) == len(result["labels"])
        assert len(result["spike_channels"]) == len(result["labels"])


# ---- OnlineSorter tests (in test_sorter.py) ----


class TestOnlineSorterInSorter:
    def test_online_sorter_create(self):
        sorter = zbci.OnlineSorter()
        assert sorter.n_templates == 0
        assert sorter.n_classified == 0
        assert sorter.n_rejected == 0

    def test_online_sorter_add_template(self):
        sorter = zbci.OnlineSorter()
        idx = sorter.add_template([1.0, 0.0, 0.0])
        assert idx == 0

    def test_online_sorter_classify(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        label, dist = sorter.classify([1.0, 0.0, 0.0])
        assert isinstance(label, int)
        assert isinstance(dist, float)

    def test_online_sorter_closest_match(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        sorter.add_template([0.0, 1.0, 0.0])
        sorter.add_template([0.0, 0.0, 1.0])
        label, _ = sorter.classify([0.9, 0.1, 0.0])
        assert label == 0
        label, _ = sorter.classify([0.1, 0.9, 0.0])
        assert label == 1
        label, _ = sorter.classify([0.0, 0.1, 0.9])
        assert label == 2

    def test_online_sorter_reject(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.set_max_distance(0.5)
        result = sorter.classify_or_reject([100.0, 100.0, 100.0])
        assert result is None

    def test_online_sorter_counters(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.set_max_distance(1.0)
        sorter.classify([0.1, 0.1, 0.1])
        sorter.classify_or_reject([0.2, 0.2, 0.2])
        sorter.classify_or_reject([100.0, 100.0, 100.0])
        assert sorter.n_classified == 3
        assert sorter.n_rejected == 1

    def test_online_sorter_reset(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        sorter.add_template([0.0, 1.0, 0.0])
        assert sorter.n_templates == 2
        sorter.reset()
        assert sorter.n_templates == 0

    def test_online_sorter_from_centroids(self):
        centroids = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        sorter = zbci.OnlineSorter.from_centroids(centroids)
        assert sorter.n_templates == 3
        label, _ = sorter.classify([1.0, 0.0, 0.0])
        assert label == 0

    def test_online_sorter_multiple_templates(self):
        sorter = zbci.OnlineSorter()
        templates = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
        for t in templates:
            sorter.add_template(t)
        assert sorter.n_templates == 5
        # Each template should be closest to itself
        for i, t in enumerate(templates):
            label, dist = sorter.classify(t)
            assert label == i
            assert dist < 1e-10

    def test_online_sorter_repr(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        r = repr(sorter)
        assert "OnlineSorter" in r
        assert "1" in r  # n_templates=1

    def test_online_sorter_reset_counters(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.classify([0.1, 0.1, 0.1])
        sorter.classify([0.2, 0.2, 0.2])
        assert sorter.n_classified == 2
        sorter.reset_counters()
        assert sorter.n_classified == 0
        assert sorter.n_rejected == 0
        # Templates should still be there
        assert sorter.n_templates == 1

    def test_online_sorter_n_templates(self):
        sorter = zbci.OnlineSorter()
        assert sorter.n_templates == 0
        sorter.add_template([1.0, 0.0, 0.0])
        assert sorter.n_templates == 1
        sorter.add_template([0.0, 1.0, 0.0])
        assert sorter.n_templates == 2

    def test_online_sorter_wrong_dim(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="3"):
            sorter.classify([1.0, 2.0])


# ---- Template subtraction integration tests ----


class TestTemplateSubtraction:
    def test_template_subtract_flag_accepted(self):
        """Verify template_subtract parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(100)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=True
        )
        assert result["n_spikes"] >= 0
        assert result["n_clusters"] >= 0

    def test_template_subtract_disabled(self):
        """Sorting with template_subtract=False should also work."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(101)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=False
        )
        assert result["n_spikes"] >= 0

    def test_template_subtract_finds_more_spikes(self):
        """Template subtraction should find >= as many spikes as without."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(102)
        data = rng.standard_normal((10000, 4))
        # Inject overlapping spikes at nearby times
        for t in range(200, 9000, 100):
            data[t, 0] += -15.0
            if t + 20 < 10000:
                data[t + 20, 1] += -12.0

        data_copy = data.copy()
        r_no = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=False
        )
        r_yes = zbci.sort_multichannel(
            data_copy, probe, threshold=4.0, template_subtract=True
        )
        assert r_yes["n_spikes"] >= r_no["n_spikes"]

    def test_template_min_count_param(self):
        """Verify template_min_count parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(103)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0,
            template_subtract=True, template_min_count=5,
        )
        assert result["n_spikes"] >= 0

    def test_template_subtract_deterministic(self):
        """Template subtraction should be deterministic."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(104)
        data = rng.standard_normal((8000, 4))
        for t in range(200, 7000, 200):
            data[t, 0] += -15.0
        data2 = data.copy()

        r1 = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=True
        )
        r2 = zbci.sort_multichannel(
            data2, probe, threshold=4.0, template_subtract=True
        )
        assert r1["n_spikes"] == r2["n_spikes"]
        assert r1["n_clusters"] == r2["n_clusters"]
        np.testing.assert_array_equal(r1["labels"], r2["labels"])
        np.testing.assert_array_equal(r1["spike_times"], r2["spike_times"])

    def test_template_subtract_valid_labels(self):
        """All labels should be < n_clusters after template subtraction."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(105)
        data = rng.standard_normal((8000, 4))
        for t in range(200, 7000, 150):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=True
        )
        if result["n_spikes"] > 0:
            assert np.all(result["labels"] < result["n_clusters"])
            assert np.all(result["labels"] >= 0)

    def test_min_cluster_snr_param(self):
        """Verify min_cluster_snr parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(106)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, min_cluster_snr=3.0,
        )
        assert result["n_spikes"] >= 0

    def test_spatial_merge_dprime_param(self):
        """Verify spatial_merge_dprime parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(107)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, spatial_merge_dprime=2.0,
        )
        assert result["n_spikes"] >= 0

    def test_template_subtract_32ch(self):
        """Template subtraction should work on 32-channel data."""
        probe = zbci.ProbeLayout.linear(32, 25.0)
        rng = np.random.default_rng(108)
        data = rng.standard_normal((10000, 32))
        for t in range(200, 9000, 200):
            data[t, 5] += -15.0
            data[t, 6] += -10.0  # spatial spread
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, template_subtract=True
        )
        assert result["n_spikes"] >= 0
        if result["n_spikes"] > 0:
            assert np.all(result["labels"] < result["n_clusters"])


class TestDetectionModes:
    """Tests for NEO/SNEO detection modes."""

    def test_amplitude_mode_accepted(self):
        """Verify detection_mode='amplitude' is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(200)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, detection_mode="amplitude"
        )
        assert result["n_spikes"] >= 0

    def test_neo_mode_accepted(self):
        """Verify detection_mode='neo' is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(201)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, detection_mode="neo"
        )
        assert result["n_spikes"] >= 0

    def test_sneo_mode_accepted(self):
        """Verify detection_mode='sneo' is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(202)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, detection_mode="sneo", sneo_smooth_window=3
        )
        assert result["n_spikes"] >= 0

    def test_invalid_detection_mode_raises(self):
        """Verify invalid detection_mode raises ValueError."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        data = np.random.randn(5000, 4)
        try:
            zbci.sort_multichannel(data, probe, detection_mode="invalid")
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_sneo_smooth_window_param(self):
        """Verify sneo_smooth_window parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(203)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        for w in [1, 3, 5, 7]:
            result = zbci.sort_multichannel(
                data.copy(), probe, threshold=4.0, detection_mode="sneo",
                sneo_smooth_window=w
            )
            assert result["n_spikes"] >= 0

    def test_sneo_detects_spikes(self):
        """SNEO should detect clear spikes."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(204)
        data = rng.standard_normal((5000, 4)) * 0.3
        for t in range(500, 4500, 500):
            data[t, 0] += -10.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, detection_mode="sneo"
        )
        assert result["n_spikes"] >= 1


class TestCCGMerge:
    """Tests for CCG-based cluster merging."""

    def test_ccg_merge_flag_accepted(self):
        """Verify ccg_merge parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(300)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, ccg_merge=True
        )
        assert result["n_spikes"] >= 0

    def test_ccg_merge_disabled(self):
        """Verify ccg_merge=False works (default)."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(301)
        data = rng.standard_normal((5000, 4))
        data[1000, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, ccg_merge=False
        )
        assert result["n_spikes"] >= 0

    def test_ccg_template_corr_threshold(self):
        """Verify ccg_template_corr_threshold parameter is accepted."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(302)
        data = rng.standard_normal((5000, 4))
        for t in range(200, 4000, 200):
            data[t, 0] += -15.0
        result = zbci.sort_multichannel(
            data, probe, threshold=4.0, ccg_merge=True,
            ccg_template_corr_threshold=0.3
        )
        assert result["n_spikes"] >= 0

    def test_ccg_merge_reduces_clusters(self):
        """CCG merge with low threshold should merge at least as many as without."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(303)
        data = rng.standard_normal((8000, 4)) * 0.5
        for t in range(200, 7000, 200):
            data[t, 0] += -12.0
        result_no = zbci.sort_multichannel(
            data.copy(), probe, threshold=4.0, ccg_merge=False
        )
        result_yes = zbci.sort_multichannel(
            data.copy(), probe, threshold=4.0, ccg_merge=True,
            ccg_template_corr_threshold=0.3
        )
        assert result_yes["n_clusters"] <= result_no["n_clusters"]


class TestCrossCorrelogram:
    """Tests for cross-correlogram Python binding."""

    def test_basic(self):
        """Cross-correlogram of known trains."""
        train_a = np.array([0.0, 0.010, 0.020, 0.030])
        train_b = np.array([0.005, 0.015, 0.025, 0.035])
        ccg = zbci.cross_correlogram(train_a, train_b, 0.005, 0.050)
        assert ccg.shape[0] == 10
        assert ccg.sum() > 0

    def test_identical_trains(self):
        """Self-CCG should have counts."""
        train = np.array([0.0, 0.010, 0.020, 0.030, 0.040])
        ccg = zbci.cross_correlogram(train, train, 0.005, 0.050)
        assert ccg.sum() > 0

    def test_no_overlap(self):
        """Distant trains should produce zero CCG."""
        train_a = np.array([0.0, 0.010])
        train_b = np.array([10.0, 10.010])
        ccg = zbci.cross_correlogram(train_a, train_b, 0.005, 0.050)
        assert ccg.sum() == 0

    def test_empty_train(self):
        """Empty train should produce zero CCG."""
        train_a = np.array([], dtype=np.float64)
        train_b = np.array([0.0, 0.010])
        ccg = zbci.cross_correlogram(train_a, train_b, 0.005, 0.050)
        assert ccg.sum() == 0

    def test_invalid_bin_width(self):
        """Negative bin_width should raise."""
        try:
            zbci.cross_correlogram(
                np.array([0.0]), np.array([0.0]), -1.0, 0.050
            )
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_invalid_max_lag(self):
        """Negative max_lag should raise."""
        try:
            zbci.cross_correlogram(
                np.array([0.0]), np.array([0.0]), 0.005, -1.0
            )
            assert False, "Should have raised"
        except ValueError:
            pass


# ---- StreamingSorter tests ----


class TestStreamingSorter:
    def test_create_4ch(self):
        """Create a 4-channel streaming sorter."""
        sorter = zbci.StreamingSorter(4)
        assert sorter.n_templates == 0
        assert sorter.segment_count == 0

    def test_create_with_params(self):
        """Create with custom parameters."""
        sorter = zbci.StreamingSorter(
            8, decay=0.9, threshold=4.0, detection_mode="neo"
        )
        assert sorter.n_templates == 0

    def test_invalid_channels(self):
        """Invalid channel count should raise."""
        with pytest.raises(ValueError, match="must be 4, 8, 16, or 32"):
            zbci.StreamingSorter(3)

    def test_invalid_detection_mode(self):
        """Invalid detection mode should raise."""
        with pytest.raises(ValueError):
            zbci.StreamingSorter(4, detection_mode="invalid")

    def test_feed_basic(self):
        """Feed a segment and get result dict."""
        rng = np.random.default_rng(42)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        sorter = zbci.StreamingSorter(4)
        data = rng.standard_normal((5000, 4))
        result = sorter.feed(data, probe)
        assert "n_spikes" in result
        assert "n_clusters" in result
        assert "labels" in result
        assert "spike_times" in result
        assert "spike_channels" in result
        assert "clusters" in result
        assert sorter.segment_count == 1

    def test_feed_multiple_segments(self):
        """Feeding multiple segments increments segment_count."""
        rng = np.random.default_rng(123)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        sorter = zbci.StreamingSorter(4)
        for _ in range(3):
            data = rng.standard_normal((5000, 4))
            sorter.feed(data, probe)
        assert sorter.segment_count == 3

    def test_feed_wrong_channels(self):
        """Feeding data with wrong channel count should raise."""
        probe = zbci.ProbeLayout.linear(4, 25.0)
        sorter = zbci.StreamingSorter(4)
        data = np.random.randn(5000, 8)
        with pytest.raises(ValueError, match="expected 4 channels"):
            sorter.feed(data, probe)

    def test_feed_with_spikes(self):
        """Feed data with injected spikes and check detection."""
        rng = np.random.default_rng(99)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        data = rng.standard_normal((10000, 4)) * 0.3
        # Inject spikes
        for pos in range(1000, 9000, 200):
            data[pos, 0] -= 8.0
        result = sorter.feed(data, probe)
        assert result["n_spikes"] > 0

    def test_reset(self):
        """Reset clears templates and segment count."""
        rng = np.random.default_rng(55)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        sorter = zbci.StreamingSorter(4)
        data = rng.standard_normal((5000, 4))
        sorter.feed(data, probe)
        assert sorter.segment_count == 1
        sorter.reset()
        assert sorter.segment_count == 0
        assert sorter.n_templates == 0

    def test_8ch(self):
        """8-channel streaming sort."""
        rng = np.random.default_rng(77)
        probe = zbci.ProbeLayout.linear(8, 25.0)
        sorter = zbci.StreamingSorter(8)
        data = rng.standard_normal((5000, 8))
        result = sorter.feed(data, probe)
        assert "n_spikes" in result
        assert sorter.segment_count == 1

    def test_sneo_mode(self):
        """Streaming sorter with SNEO detection mode."""
        sorter = zbci.StreamingSorter(
            4, detection_mode="sneo", sneo_smooth_window=5
        )
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5000, 4))
        result = sorter.feed(data, probe)
        assert "n_spikes" in result

    def test_ccg_merge_flag(self):
        """Streaming sorter accepts ccg_merge flag."""
        sorter = zbci.StreamingSorter(4, ccg_merge=True)
        probe = zbci.ProbeLayout.linear(4, 25.0)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5000, 4))
        result = sorter.feed(data, probe)
        assert "n_spikes" in result
