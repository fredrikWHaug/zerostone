"""Tests for TangentSpace Python bindings."""
import numpy as np
import pytest


class TestTangentSpace:
    """Tests for TangentSpace."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'TangentSpace')

    def test_create(self):
        import npyci as npy
        for ch in [4, 8, 16, 32]:
            ts = npy.TangentSpace(channels=ch)
            assert ts.channels == ch
            assert ts.vector_length == ch * (ch + 1) // 2

    def test_invalid_channels(self):
        import npyci as npy
        with pytest.raises(ValueError):
            npy.TangentSpace(channels=3)

    def test_vector_lengths(self):
        import npyci as npy
        assert npy.TangentSpace(channels=4).vector_length == 10
        assert npy.TangentSpace(channels=8).vector_length == 36
        assert npy.TangentSpace(channels=16).vector_length == 136
        assert npy.TangentSpace(channels=32).vector_length == 528

    def test_fit_identity(self):
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        ref = np.eye(4, dtype=np.float64)
        ts.fit(ref)  # Should not raise

    def test_fit_wrong_shape(self):
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        with pytest.raises(ValueError):
            ts.fit(np.eye(8, dtype=np.float64))

    def test_transform_identity(self):
        """Transform of identity at identity reference should be zero vector."""
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        ref = np.eye(4, dtype=np.float64)
        ts.fit(ref)
        vec = ts.transform(ref)
        assert vec.shape == (10,)
        assert vec.dtype == np.float64
        # log(I) = 0, so tangent vector should be all zeros
        np.testing.assert_allclose(vec, 0.0, atol=1e-10)

    def test_transform_spd_matrix(self):
        """Transform an SPD matrix."""
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        ref = np.eye(4, dtype=np.float64)
        ts.fit(ref)
        # Create an SPD matrix
        A = np.random.randn(4, 4)
        spd = A @ A.T + np.eye(4) * 2.0  # Ensure positive definite
        vec = ts.transform(spd)
        assert vec.shape == (10,)

    def test_inverse_transform(self):
        """Inverse transform should recover the original matrix."""
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        ref = np.eye(4, dtype=np.float64)
        ts.fit(ref)
        # Create an SPD matrix
        np.random.seed(42)
        A = np.random.randn(4, 4) * 0.1
        spd = A @ A.T + np.eye(4)
        vec = ts.transform(spd)
        recovered = ts.inverse_transform(vec)
        assert recovered.shape == (4, 4)
        np.testing.assert_allclose(recovered, spd, atol=1e-6)

    def test_inverse_transform_wrong_length(self):
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        ts.fit(np.eye(4, dtype=np.float64))
        with pytest.raises(ValueError):
            ts.inverse_transform(np.zeros(5))

    def test_transform_without_fit_raises(self):
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        with pytest.raises(ValueError):
            ts.transform(np.eye(4, dtype=np.float64))

    def test_roundtrip_8_channels(self):
        """Test roundtrip with 8 channels using a diagonal SPD matrix."""
        import npyci as npy
        ts = npy.TangentSpace(channels=8)
        ref = np.eye(8, dtype=np.float64)
        ts.fit(ref)
        # Diagonal SPD matrix (guaranteed convergence)
        spd = np.diag([1.1, 1.2, 0.9, 1.0, 1.05, 0.95, 1.15, 1.03])
        vec = ts.transform(spd)
        assert vec.shape == (36,)
        recovered = ts.inverse_transform(vec)
        np.testing.assert_allclose(recovered, spd, atol=1e-4)

    def test_repr(self):
        import npyci as npy
        ts = npy.TangentSpace(channels=4)
        r = repr(ts)
        assert 'TangentSpace' in r
        assert 'channels=4' in r
        assert 'vector_length=10' in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
