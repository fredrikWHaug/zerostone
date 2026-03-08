"""Tests for online k-means clustering."""

import numpy as np
import pytest

import zpybci as zbci


# ---- Helpers ----


def make_gaussian_clusters(n_per_cluster, centers, std, rng):
    """Generate Gaussian cluster data.

    Returns (points, true_labels) as 2D and 1D arrays.
    """
    n_clusters = len(centers)
    d = len(centers[0])
    points = np.zeros((n_per_cluster * n_clusters, d), dtype=np.float64)
    labels = np.zeros(n_per_cluster * n_clusters, dtype=np.int64)
    for i, center in enumerate(centers):
        start = i * n_per_cluster
        end = start + n_per_cluster
        points[start:end] = rng.standard_normal((n_per_cluster, d)) * std + center
        labels[start:end] = i
    return points, labels


# ---- Tests ----


class TestOnlineKMeans:
    def test_create_and_update(self):
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4)
        assert km.n_active == 0
        assert km.dimensions == 2
        assert km.max_clusters == 4

        result = km.update(np.array([1.0, 2.0]))
        assert "cluster" in result
        assert "distance" in result
        assert "created" in result
        assert result["created"] is True
        assert km.n_active == 1

    def test_auto_cluster_discovery(self):
        rng = np.random.default_rng(42)
        km = zbci.OnlineKMeans(
            dimensions=2, max_clusters=8, create_threshold=3.0
        )

        centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]
        for center in centers:
            for _ in range(30):
                point = np.array(center) + rng.standard_normal(2) * 0.5
                km.update(point)

        assert km.n_active == 3, f"Expected 3 clusters, got {km.n_active}"

    def test_batch_update(self):
        rng = np.random.default_rng(99)
        km = zbci.OnlineKMeans(
            dimensions=3, max_clusters=8, create_threshold=5.0
        )

        points = rng.standard_normal((50, 3)).astype(np.float64)
        labels, distances = km.update_batch(points)
        assert labels.shape == (50,)
        assert distances.shape == (50,)
        assert km.n_active >= 1

    def test_predict(self):
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4)
        km.seed_centroid(np.array([0.0, 0.0]))
        km.seed_centroid(np.array([10.0, 0.0]))

        points = np.array([[0.1, 0.1], [9.9, 0.1]], dtype=np.float64)
        labels, distances = km.predict(points)
        assert labels[0] == 0
        assert labels[1] == 1
        assert distances[0] < 1.0
        assert distances[1] < 1.0

    def test_merge_closest(self):
        km = zbci.OnlineKMeans(
            dimensions=2, max_clusters=8, merge_threshold=2.0
        )
        km.seed_centroid(np.array([0.0, 0.0]))
        km.seed_centroid(np.array([1.0, 0.0]))
        km.seed_centroid(np.array([10.0, 10.0]))
        assert km.n_active == 3

        result = km.merge_closest()
        assert result is not None
        assert result == (0, 1)
        assert km.n_active == 2

    def test_remove_cluster(self):
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4)
        km.seed_centroid(np.array([0.0, 0.0]))
        km.seed_centroid(np.array([5.0, 5.0]))
        km.seed_centroid(np.array([10.0, 10.0]))
        assert km.n_active == 3

        km.remove_cluster(1)
        assert km.n_active == 2

        # Centroid at index 1 should now be what was at index 2
        centroids = km.centroids
        assert centroids.shape == (2, 2)
        np.testing.assert_allclose(centroids[1], [10.0, 10.0], atol=0.01)

    def test_seed_centroids(self):
        km = zbci.OnlineKMeans(dimensions=3, max_clusters=4)
        idx0 = km.seed_centroid(np.array([1.0, 2.0, 3.0]))
        idx1 = km.seed_centroid(np.array([4.0, 5.0, 6.0]))
        assert idx0 == 0
        assert idx1 == 1
        assert km.n_active == 2

        centroids = km.centroids
        np.testing.assert_array_equal(centroids[0], [1.0, 2.0, 3.0])

    def test_count_capping(self):
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4, max_count=100)

        # Feed many points
        for i in range(500):
            km.update(np.array([float(i) * 0.001, 0.0]))

        counts = km.counts
        assert counts[0] == 100

    def test_reset(self):
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4)
        km.seed_centroid(np.array([1.0, 2.0]))
        km.update(np.array([3.0, 4.0]))
        assert km.n_active > 0

        km.reset()
        assert km.n_active == 0
        assert km.centroids.shape == (0, 2)

    def test_cluster_variance(self):
        rng = np.random.default_rng(77)
        km = zbci.OnlineKMeans(dimensions=2, max_clusters=4, max_count=100000)

        # Feed points from a known distribution
        for _ in range(5000):
            point = np.array([rng.standard_normal() * 2.0, rng.standard_normal() * 1.0])
            km.update(point)

        var = km.cluster_variance(0)
        assert var is not None
        assert len(var) == 2
        # Variance should be near 4.0 and 1.0
        assert abs(var[0] - 4.0) < 1.0, f"var[0]={var[0]}, expected ~4.0"
        assert abs(var[1] - 1.0) < 0.5, f"var[1]={var[1]}, expected ~1.0"

    def test_repr(self):
        km = zbci.OnlineKMeans(dimensions=3, max_clusters=8)
        r = repr(km)
        assert "OnlineKMeans" in r
        assert "3" in r
        assert "8" in r

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="Unsupported"):
            zbci.OnlineKMeans(dimensions=7, max_clusters=4)

        with pytest.raises(ValueError, match="Unsupported"):
            zbci.OnlineKMeans(dimensions=2, max_clusters=6)

    def test_spike_sort_integration(self):
        """End-to-end: detect -> extract -> PCA -> OnlineKMeans cluster."""
        rng = np.random.default_rng(42)
        sample_rate = 30000.0
        n_samples = 20000
        window = 64
        pre = window // 4

        # Generate synthetic recording with embedded spikes
        data = rng.standard_normal(n_samples) * 1.0

        # Two distinct neuron types
        t = np.arange(window)
        template1 = -8.0 * np.exp(-0.5 * ((t - pre) / 2.0) ** 2)
        template2 = -6.0 * np.exp(-0.5 * ((t - pre) / 4.0) ** 2)

        # Insert spikes at known positions
        positions1 = np.arange(500, 10000, 500)
        positions2 = np.arange(750, 10000, 500)
        for p in positions1:
            data[p - pre : p - pre + window] += template1
        for p in positions2:
            data[p - pre : p - pre + window] += template2

        # 1. Detect
        noise = zbci.estimate_noise_mad(data)
        spike_times = zbci.detect_spikes(data, threshold=4.0 * noise)
        assert len(spike_times) > 10

        # 2. Extract
        waveforms = zbci.extract_waveforms(data, spike_times, window=window)
        assert waveforms.shape[0] > 10

        # 3. PCA
        pca = zbci.WaveformPca(window=window, n_components=3)
        pca.fit(waveforms)
        features = pca.transform(waveforms)

        # 4. Cluster with OnlineKMeans
        km = zbci.OnlineKMeans(
            dimensions=3, max_clusters=8, create_threshold=5.0
        )
        labels, distances = km.update_batch(features)
        assert labels.shape == (features.shape[0],)
        assert km.n_active >= 2, f"Expected >=2 clusters, got {km.n_active}"
