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
