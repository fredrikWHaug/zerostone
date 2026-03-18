"""Tests for spatial whitening (ZCA/PCA) bindings."""

import numpy as np
import pytest

import zpybci as zbci


class TestWhiteningCreation:
    def test_zca_2ch(self):
        cov = np.array([[4.0, 2.0], [2.0, 3.0]])
        wm = zbci.WhiteningMatrix(cov, mode="zca")
        assert wm.n_channels == 2

    def test_pca_2ch(self):
        cov = np.array([[4.0, 2.0], [2.0, 3.0]])
        wm = zbci.WhiteningMatrix(cov, mode="pca")
        assert wm.n_channels == 2

    def test_4ch(self):
        cov = np.eye(4, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        assert wm.n_channels == 4

    def test_8ch(self):
        cov = np.eye(8, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        assert wm.n_channels == 8

    def test_16ch(self):
        cov = np.eye(16, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        assert wm.n_channels == 16

    def test_32ch(self):
        cov = np.eye(32, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        assert wm.n_channels == 32

    def test_64ch(self):
        cov = np.eye(64, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        assert wm.n_channels == 64

    def test_invalid_channel_count(self):
        cov = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="n_channels"):
            zbci.WhiteningMatrix(cov)

    def test_non_square_error(self):
        cov = np.ones((2, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="square"):
            zbci.WhiteningMatrix(cov)

    def test_invalid_mode(self):
        cov = np.eye(2, dtype=np.float64)
        with pytest.raises(ValueError, match="mode"):
            zbci.WhiteningMatrix(cov, mode="foo")

    def test_default_mode_is_zca(self):
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        wm = zbci.WhiteningMatrix(cov)
        out = wm.apply(np.array([2.0, 3.0]))
        # ZCA of diagonal [4,9]: W = diag(1/2, 1/3)
        np.testing.assert_allclose(out, [1.0, 1.0], atol=0.01)


class TestZcaWhitening:
    def test_identity_covariance_passthrough(self):
        """Identity covariance: whitening should not change the signal."""
        cov = np.eye(2, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov, mode="zca", epsilon=1e-10)
        sample = np.array([3.0, 7.0])
        out = wm.apply(sample)
        np.testing.assert_allclose(out, sample, atol=0.01)

    def test_known_diagonal(self):
        """Diagonal covariance [4,9]: W = diag(1/2, 1/3)."""
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        wm = zbci.WhiteningMatrix(cov, mode="zca", epsilon=1e-10)
        out = wm.apply(np.array([2.0, 3.0]))
        np.testing.assert_allclose(out, [1.0, 1.0], atol=0.01)

    def test_correlated_produces_identity_covariance(self):
        """After ZCA whitening, output covariance should be near identity."""
        rng = np.random.default_rng(42)
        n = 2000
        # Correlated 2-channel data
        z = rng.standard_normal((n, 2))
        L = np.array([[2.0, 0.0], [1.0, np.sqrt(2.0)]])
        data = z @ L.T

        emp_cov = np.cov(data.T)
        wm = zbci.WhiteningMatrix(emp_cov, mode="zca", epsilon=1e-10)

        whitened = np.array([wm.apply(data[i]) for i in range(n)])
        white_cov = np.cov(whitened.T)
        np.testing.assert_allclose(white_cov, np.eye(2), atol=0.1)

    def test_4ch_diagonal(self):
        cov = np.diag([1.0, 4.0, 9.0, 16.0])
        wm = zbci.WhiteningMatrix(cov, mode="zca", epsilon=1e-10)
        out = wm.apply(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0], atol=0.01)

    def test_3ch_identity(self):
        """3-channel identity covariance: passthrough."""
        # 3 channels are not supported -- should raise
        cov = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError):
            zbci.WhiteningMatrix(cov)


class TestPcaWhitening:
    def test_identity_covariance(self):
        """PCA whitening of identity: output magnitudes should match input."""
        cov = np.eye(2, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov, mode="pca", epsilon=1e-10)
        sample = np.array([3.0, 7.0])
        out = wm.apply(sample)
        # Output may be permuted/sign-flipped
        mags = sorted(np.abs(out))
        np.testing.assert_allclose(mags, [3.0, 7.0], atol=0.1)

    def test_decorrelates(self):
        """PCA whitening should produce uncorrelated unit-variance output."""
        rng = np.random.default_rng(77)
        n = 2000
        z = rng.standard_normal((n, 2))
        mix = np.array([[3.0, 1.0], [1.0, 2.0]])
        data = z @ mix.T

        emp_cov = np.cov(data.T)
        wm = zbci.WhiteningMatrix(emp_cov, mode="pca", epsilon=1e-10)

        whitened = np.array([wm.apply(data[i]) for i in range(n)])
        white_cov = np.cov(whitened.T)
        np.testing.assert_allclose(np.diag(white_cov), [1.0, 1.0], atol=0.1)
        np.testing.assert_allclose(white_cov[0, 1], 0.0, atol=0.1)


class TestRegularization:
    def test_singular_covariance(self):
        """Rank-1 covariance with regularization should not error."""
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        wm = zbci.WhiteningMatrix(cov, mode="zca", epsilon=1e-3)
        out = wm.apply(np.array([1.0, 1.0]))
        assert np.all(np.isfinite(out))

    def test_large_epsilon_shrinks_to_identity(self):
        """With very large epsilon, W ~ (1/sqrt(eps)) * I."""
        cov = np.array([[100.0, 50.0], [50.0, 100.0]])
        eps = 1e6
        wm = zbci.WhiteningMatrix(cov, mode="zca", epsilon=eps)
        out = wm.apply(np.array([1.0, 0.0]))
        expected_scale = 1.0 / np.sqrt(eps)
        assert abs(out[0] - expected_scale) / expected_scale < 0.1


class TestApplyValidation:
    def test_wrong_length(self):
        cov = np.eye(4, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        with pytest.raises(ValueError, match="expected 4"):
            wm.apply(np.array([1.0, 2.0]))

    def test_repr(self):
        cov = np.eye(8, dtype=np.float64)
        wm = zbci.WhiteningMatrix(cov)
        r = repr(wm)
        assert "WhiteningMatrix" in r
        assert "8" in r


class TestEstimateNoiseCovariance:
    def test_all_quiet(self):
        """When all data is below threshold, result equals full covariance."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 2)) * 0.1
        noise_std = np.ones(2, dtype=np.float64)
        cov = zbci.estimate_noise_covariance(
            data, noise_std, threshold_multiplier=3.0, min_quiet_samples=10
        )
        assert cov.shape == (2, 2)
        full_cov = np.cov(data.T, bias=False)
        np.testing.assert_allclose(cov, full_cov, atol=1e-10)

    def test_with_spikes(self):
        """Noise covariance should be smaller than full covariance when spikes present."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((500, 2))
        # Inject large spikes
        data[50] = [50.0, 50.0]
        data[100] = [-40.0, 60.0]
        data[200] = [45.0, -55.0]

        noise_std = np.ones(2, dtype=np.float64)
        noise_cov = zbci.estimate_noise_covariance(
            data, noise_std, threshold_multiplier=5.0, min_quiet_samples=10
        )
        full_cov = np.cov(data.T, bias=False)
        # Noise variance should be smaller than full variance
        assert noise_cov[0, 0] < full_cov[0, 0]
        assert noise_cov[1, 1] < full_cov[1, 1]

    def test_symmetry(self):
        """Covariance must be symmetric."""
        rng = np.random.default_rng(555)
        data = rng.standard_normal((300, 4))
        noise_std = np.ones(4, dtype=np.float64)
        cov = zbci.estimate_noise_covariance(data, noise_std)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_fallback_to_full(self):
        """When all samples exceed threshold, falls back to full covariance."""
        data = np.array([
            [10.0, 10.0],
            [11.0, 9.0],
            [-10.0, 10.0],
            [9.0, -11.0],
            [-11.0, -9.0],
        ], dtype=np.float64)
        noise_std = np.ones(2, dtype=np.float64)
        noise_cov = zbci.estimate_noise_covariance(
            data, noise_std, threshold_multiplier=3.0, min_quiet_samples=2
        )
        full_cov = np.cov(data.T, bias=False)
        np.testing.assert_allclose(noise_cov, full_cov, atol=1e-10)

    def test_wrong_noise_length(self):
        data = np.zeros((50, 4), dtype=np.float64)
        noise_std = np.ones(3, dtype=np.float64)
        with pytest.raises(ValueError, match="expected 4"):
            zbci.estimate_noise_covariance(data, noise_std)

    def test_invalid_channel_count(self):
        data = np.zeros((50, 3), dtype=np.float64)
        noise_std = np.ones(3, dtype=np.float64)
        with pytest.raises(ValueError, match="n_channels"):
            zbci.estimate_noise_covariance(data, noise_std)

    def test_default_params(self):
        """Should work with default threshold_multiplier and min_quiet_samples."""
        rng = np.random.default_rng(77)
        data = rng.standard_normal((200, 2))
        noise_std = np.ones(2, dtype=np.float64)
        cov = zbci.estimate_noise_covariance(data, noise_std)
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov))

    def test_4_channels(self):
        rng = np.random.default_rng(88)
        data = rng.standard_normal((300, 4))
        noise_std = np.ones(4, dtype=np.float64)
        cov = zbci.estimate_noise_covariance(data, noise_std, threshold_multiplier=3.0)
        assert cov.shape == (4, 4)
        # Diagonal should be positive
        for i in range(4):
            assert cov[i, i] > 0
