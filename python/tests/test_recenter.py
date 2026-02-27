"""Tests for recenter() Python binding."""
import numpy as np
import pytest
import zpybci as zbci


def _make_spd(c, rng):
    """Generate a random SPD matrix of size (c, c)."""
    A = rng.standard_normal((c, c))
    return A @ A.T + np.eye(c) * 0.5


class TestRecenter:
    def test_recenter_exists(self):
        assert hasattr(zbci, "recenter")

    def test_recenter_identity_mean(self):
        """Recentering with identity mean should be a no-op."""
        rng = np.random.default_rng(42)
        c = 4
        matrices = np.stack([_make_spd(c, rng) for _ in range(5)])
        mean = np.eye(c, dtype=np.float64)

        recentered = zbci.recenter(matrices, mean)

        assert recentered.shape == matrices.shape
        np.testing.assert_allclose(recentered, matrices, atol=1e-8)

    def test_recenter_frechet_mean_near_identity(self):
        """After recentering, the Frechet mean should be close to identity."""
        rng = np.random.default_rng(123)
        c = 4
        n = 8
        matrices = np.stack([_make_spd(c, rng) for _ in range(n)])

        # Compute Frechet mean of original matrices
        mean = zbci.frechet_mean(matrices)

        # Recenter
        recentered = zbci.recenter(matrices, mean)

        # Compute Frechet mean of recentered matrices
        new_mean = zbci.frechet_mean(recentered)

        np.testing.assert_allclose(new_mean, np.eye(c), atol=1e-3)

    def test_recenter_preserves_spd(self):
        """Recentered matrices should remain SPD."""
        rng = np.random.default_rng(99)
        c = 8
        matrices = np.stack([_make_spd(c, rng) for _ in range(4)])
        mean = zbci.frechet_mean(matrices)

        recentered = zbci.recenter(matrices, mean)

        for i in range(recentered.shape[0]):
            eigvals = np.linalg.eigvalsh(recentered[i])
            assert np.all(eigvals > 0), f"Matrix {i} has non-positive eigenvalue"

    def test_recenter_preserves_symmetry(self):
        """Recentered matrices should be symmetric."""
        rng = np.random.default_rng(7)
        c = 4
        matrices = np.stack([_make_spd(c, rng) for _ in range(3)])
        mean = zbci.frechet_mean(matrices)

        recentered = zbci.recenter(matrices, mean)

        for i in range(recentered.shape[0]):
            np.testing.assert_allclose(
                recentered[i], recentered[i].T, atol=1e-10
            )

    def test_recenter_invalid_channels(self):
        """Should reject unsupported channel counts."""
        matrices = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        mean = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError):
            zbci.recenter(matrices, mean)
