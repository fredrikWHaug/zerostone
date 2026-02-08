"""Tests for OnlineStats Python bindings."""
import numpy as np
import pytest


class TestOnlineStats:
    """Tests for OnlineStats."""

    def test_import(self):
        """Test that OnlineStats can be imported."""
        import npyci as npy
        assert hasattr(npy, 'OnlineStats')

    def test_create(self):
        """Test creating OnlineStats."""
        import npyci as npy

        stats = npy.OnlineStats()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std == 0.0

    def test_update_single(self):
        """Test updating with single values."""
        import npyci as npy

        stats = npy.OnlineStats()
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)

        assert stats.count == 3
        assert np.isclose(stats.mean, 2.0)

    def test_update_batch(self):
        """Test updating with batch of values."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update_batch(values)

        assert stats.count == 5
        assert np.isclose(stats.mean, 3.0)

    def test_mean_correctness(self):
        """Test that mean is computed correctly."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        assert np.isclose(stats.mean, np.mean(values))

    def test_variance_correctness(self):
        """Test that variance is computed correctly (sample variance, n-1)."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for v in values:
            stats.update(v)

        # NumPy var with ddof=1 gives sample variance
        expected_var = np.var(values, ddof=1)
        assert np.isclose(stats.variance, expected_var)

    def test_std_correctness(self):
        """Test that std is computed correctly."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for v in values:
            stats.update(v)

        expected_std = np.std(values, ddof=1)
        assert np.isclose(stats.std, expected_std)

    def test_variance_needs_two_samples(self):
        """Test that variance is 0 with fewer than 2 samples."""
        import npyci as npy

        stats = npy.OnlineStats()
        assert stats.variance == 0.0

        stats.update(5.0)
        assert stats.variance == 0.0

        stats.update(10.0)
        assert stats.variance > 0.0

    def test_reset(self):
        """Test that reset clears state."""
        import npyci as npy

        stats = npy.OnlineStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.update(v)

        assert stats.count == 5

        stats.reset()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std == 0.0

    def test_large_dataset(self):
        """Test with larger dataset for numerical stability."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = np.random.randn(10000)
        stats.update_batch(values)

        assert stats.count == 10000
        assert np.isclose(stats.mean, np.mean(values), atol=1e-10)
        assert np.isclose(stats.variance, np.var(values, ddof=1), rtol=1e-10)

    def test_constant_values(self):
        """Test with constant values (zero variance)."""
        import npyci as npy

        stats = npy.OnlineStats()
        for _ in range(10):
            stats.update(5.0)

        assert np.isclose(stats.mean, 5.0)
        assert np.isclose(stats.variance, 0.0, atol=1e-10)
        assert np.isclose(stats.std, 0.0, atol=1e-10)

    def test_negative_values(self):
        """Test with negative values."""
        import npyci as npy

        stats = npy.OnlineStats()
        values = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        for v in values:
            stats.update(v)

        assert np.isclose(stats.mean, 0.0)
        assert np.isclose(stats.variance, np.var(values, ddof=1))

    def test_repr(self):
        """Test string representation."""
        import npyci as npy

        stats = npy.OnlineStats()
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)

        repr_str = repr(stats)
        assert 'OnlineStats' in repr_str
        assert 'count=3' in repr_str

    def test_incremental_updates(self):
        """Test that incremental and batch updates give same results."""
        import npyci as npy

        values = np.random.randn(100)

        # Incremental
        stats1 = npy.OnlineStats()
        for v in values:
            stats1.update(v)

        # Batch
        stats2 = npy.OnlineStats()
        stats2.update_batch(values)

        assert stats1.count == stats2.count
        assert np.isclose(stats1.mean, stats2.mean)
        assert np.isclose(stats1.variance, stats2.variance)
        assert np.isclose(stats1.std, stats2.std)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
