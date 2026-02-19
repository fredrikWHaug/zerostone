"""Tests for OnlineCov Python bindings."""
import numpy as np
import pytest


class TestOnlineCov:
    """Tests for OnlineCov."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'OnlineCov')

    def test_create(self):
        import zpybci as zbci
        for channels in [4, 8, 16, 32, 64]:
            cov = zbci.OnlineCov(channels=channels)
            assert cov.count == 0
            assert cov.channels == channels

    def test_invalid_channels(self):
        import zpybci as zbci
        with pytest.raises(ValueError):
            zbci.OnlineCov(channels=3)

    def test_update_and_count(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        for _ in range(10):
            cov.update(np.random.randn(4))
        assert cov.count == 10

    def test_wrong_sample_size(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        with pytest.raises(ValueError):
            cov.update(np.random.randn(8))

    def test_mean_shape(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        for _ in range(10):
            cov.update(np.random.randn(4))
        mean = cov.mean
        assert mean.shape == (4,)
        assert mean.dtype == np.float64

    def test_covariance_shape(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        for _ in range(10):
            cov.update(np.random.randn(4))
        c = cov.covariance
        assert c.shape == (4, 4)
        assert c.dtype == np.float64

    def test_correlation_shape(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        for _ in range(10):
            cov.update(np.random.randn(4))
        corr = cov.correlation
        assert corr.shape == (4, 4)

    def test_mean_correctness(self):
        import zpybci as zbci
        np.random.seed(42)
        cov = zbci.OnlineCov(channels=4)
        samples = np.random.randn(100, 4)
        for s in samples:
            cov.update(s)
        np.testing.assert_allclose(cov.mean, samples.mean(axis=0), atol=1e-10)

    def test_covariance_correctness(self):
        import zpybci as zbci
        np.random.seed(42)
        cov = zbci.OnlineCov(channels=4)
        samples = np.random.randn(200, 4)
        for s in samples:
            cov.update(s)
        expected = np.cov(samples, rowvar=False)
        np.testing.assert_allclose(cov.covariance, expected, atol=1e-10)

    def test_covariance_symmetric(self):
        import zpybci as zbci
        np.random.seed(42)
        cov = zbci.OnlineCov(channels=4)
        for _ in range(50):
            cov.update(np.random.randn(4))
        c = cov.covariance
        np.testing.assert_allclose(c, c.T, atol=1e-15)

    def test_get_element(self):
        import zpybci as zbci
        np.random.seed(42)
        cov = zbci.OnlineCov(channels=4)
        for _ in range(50):
            cov.update(np.random.randn(4))
        c = cov.covariance
        assert np.isclose(cov.get(0, 1), c[0, 1])
        assert np.isclose(cov.get(2, 3), c[2, 3])

    def test_reset(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        for _ in range(10):
            cov.update(np.random.randn(4))
        assert cov.count == 10
        cov.reset()
        assert cov.count == 0

    def test_repr(self):
        import zpybci as zbci
        cov = zbci.OnlineCov(channels=4)
        assert 'OnlineCov' in repr(cov)
        assert 'channels=4' in repr(cov)

    def test_8_channels(self):
        import zpybci as zbci
        np.random.seed(42)
        cov = zbci.OnlineCov(channels=8)
        samples = np.random.randn(100, 8)
        for s in samples:
            cov.update(s)
        assert cov.covariance.shape == (8, 8)
        np.testing.assert_allclose(cov.mean, samples.mean(axis=0), atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
