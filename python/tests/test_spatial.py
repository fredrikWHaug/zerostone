"""Tests for spatial filter Python bindings (CAR, SurfaceLaplacian)."""
import numpy as np
import pytest


class TestCAR:
    """Tests for Common Average Reference (CAR)."""

    def test_import(self):
        """Test that CAR can be imported."""
        import npyci as npy
        assert hasattr(npy, 'CAR')

    def test_create(self):
        """Test creating CAR with various channel counts."""
        import npyci as npy

        for channels in [4, 8, 16, 32, 64]:
            car = npy.CAR(channels=channels)
            assert car.channels == channels

    def test_create_dynamic(self):
        """Test creating CAR with non-optimized channel counts."""
        import npyci as npy

        for channels in [3, 5, 10, 128]:
            car = npy.CAR(channels=channels)
            assert car.channels == channels

    def test_create_invalid_channels(self):
        """Test that zero channels raises error."""
        import npyci as npy

        with pytest.raises(ValueError):
            npy.CAR(channels=0)

    def test_process_output_shape(self):
        """Test that process returns correct shape."""
        import npyci as npy

        car = npy.CAR(channels=8)
        data = np.random.randn(100, 8).astype(np.float32)
        filtered = car.process(data)

        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.float32
        assert filtered.shape == data.shape

    def test_zero_mean_property(self):
        """Test that CAR output has zero mean across channels."""
        import npyci as npy

        car = npy.CAR(channels=8)
        data = np.random.randn(100, 8).astype(np.float32)
        filtered = car.process(data)

        # Each row (sample) should have zero mean across channels
        row_means = filtered.mean(axis=1)
        assert np.allclose(row_means, 0, atol=1e-5)

    def test_known_values(self):
        """Test CAR with known input/output."""
        import npyci as npy

        car = npy.CAR(channels=4)
        # [1, 2, 3, 4] has mean 2.5
        data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        filtered = car.process(data)

        # Expected: [-1.5, -0.5, 0.5, 1.5]
        expected = np.array([[-1.5, -0.5, 0.5, 1.5]], dtype=np.float32)
        assert np.allclose(filtered, expected, atol=1e-5)

    def test_reference_independence(self):
        """Test that CAR removes constant offset."""
        import npyci as npy

        car = npy.CAR(channels=4)

        data1 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        data2 = np.array([[101.0, 102.0, 103.0, 104.0]], dtype=np.float32)  # Same + 100

        filtered1 = car.process(data1)
        filtered2 = car.process(data2)

        # Results should be identical
        assert np.allclose(filtered1, filtered2, atol=1e-5)

    def test_constant_input(self):
        """Test that constant input yields zeros."""
        import npyci as npy

        car = npy.CAR(channels=8)
        data = np.ones((10, 8), dtype=np.float32) * 5.0
        filtered = car.process(data)

        assert np.allclose(filtered, 0, atol=1e-5)

    def test_channel_mismatch_error(self):
        """Test that channel mismatch raises error."""
        import npyci as npy

        car = npy.CAR(channels=8)
        data = np.random.randn(100, 4).astype(np.float32)  # Wrong channel count

        with pytest.raises(ValueError, match="Channel count mismatch"):
            car.process(data)

    def test_reset(self):
        """Test that reset is a no-op (CAR is stateless)."""
        import npyci as npy

        car = npy.CAR(channels=8)
        data = np.random.randn(100, 8).astype(np.float32)

        # Process once
        out1 = car.process(data)

        # Reset and process again
        car.reset()
        out2 = car.process(data)

        # Results should be identical
        assert np.allclose(out1, out2)

    def test_repr(self):
        """Test string representation."""
        import npyci as npy

        car = npy.CAR(channels=8)
        assert 'CAR' in repr(car)
        assert '8' in repr(car)

    def test_dynamic_channels(self):
        """Test that dynamic implementation works correctly."""
        import npyci as npy

        # Use non-optimized channel count
        car = npy.CAR(channels=10)
        data = np.random.randn(100, 10).astype(np.float32)
        filtered = car.process(data)

        # Should still have zero mean
        row_means = filtered.mean(axis=1)
        assert np.allclose(row_means, 0, atol=1e-5)


class TestSurfaceLaplacian:
    """Tests for Surface Laplacian."""

    def test_import(self):
        """Test that SurfaceLaplacian can be imported."""
        import npyci as npy
        assert hasattr(npy, 'SurfaceLaplacian')

    def test_create_linear(self):
        """Test creating linear SurfaceLaplacian."""
        import npyci as npy

        for channels in [4, 8, 16, 32, 64]:
            lap = npy.SurfaceLaplacian.linear(channels=channels)
            assert lap.channels == channels

    def test_create_linear_dynamic(self):
        """Test creating linear SurfaceLaplacian with non-optimized channel counts."""
        import npyci as npy

        for channels in [3, 5, 10, 128]:
            lap = npy.SurfaceLaplacian.linear(channels=channels)
            assert lap.channels == channels

    def test_create_linear_invalid(self):
        """Test that linear with <2 channels raises error."""
        import npyci as npy

        with pytest.raises(ValueError):
            npy.SurfaceLaplacian.linear(channels=1)

    def test_create_custom(self):
        """Test creating SurfaceLaplacian with custom neighbors."""
        import npyci as npy

        neighbors = [[1], [0, 2], [1, 3], [2]]
        lap = npy.SurfaceLaplacian.custom(channels=4, neighbors=neighbors)
        assert lap.channels == 4

    def test_custom_invalid_neighbor(self):
        """Test that invalid neighbor index raises error."""
        import npyci as npy

        neighbors = [[1], [0, 5], [1, 3], [2]]  # 5 is out of bounds

        with pytest.raises(ValueError, match="Invalid neighbor index"):
            npy.SurfaceLaplacian.custom(channels=4, neighbors=neighbors)

    def test_custom_self_reference(self):
        """Test that self-reference raises error."""
        import npyci as npy

        neighbors = [[1], [1, 2], [1, 3], [2]]  # Channel 1 references itself

        with pytest.raises(ValueError, match="own neighbor"):
            npy.SurfaceLaplacian.custom(channels=4, neighbors=neighbors)

    def test_process_output_shape(self):
        """Test that process returns correct shape."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=8)
        data = np.random.randn(100, 8).astype(np.float32)
        filtered = lap.process(data)

        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.float32
        assert filtered.shape == data.shape

    def test_edge_detection(self):
        """Test that Laplacian detects peaks/edges."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=5)

        # Create signal with peak at center
        data = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float32)
        filtered = lap.process(data)

        # Center channel (index 2): 5.0 - (2.0 + 2.0)/2 = 3.0
        assert np.isclose(filtered[0, 2], 3.0, atol=1e-5)

        # Peak should have positive Laplacian (higher than neighbors)
        assert filtered[0, 2] > 0

    def test_linear_known_values(self):
        """Test linear Laplacian with known input/output."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=3)

        # [1, 2, 3] linear gradient
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        filtered = lap.process(data)

        # Channel 0 (edge): 1.0 - 2.0 = -1.0
        assert np.isclose(filtered[0, 0], -1.0, atol=1e-5)

        # Channel 1 (middle): 2.0 - (1.0 + 3.0)/2 = 0.0
        assert np.isclose(filtered[0, 1], 0.0, atol=1e-5)

        # Channel 2 (edge): 3.0 - 2.0 = 1.0
        assert np.isclose(filtered[0, 2], 1.0, atol=1e-5)

    def test_reference_independence(self):
        """Test that Laplacian is reference-independent (second derivative)."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=5)

        data1 = np.array([[1.0, 2.0, 5.0, 2.0, 1.0]], dtype=np.float32)
        data2 = np.array([[101.0, 102.0, 105.0, 102.0, 101.0]], dtype=np.float32)

        filtered1 = lap.process(data1)
        filtered2 = lap.process(data2)

        # Results should be identical
        assert np.allclose(filtered1, filtered2, atol=1e-5)

    def test_channel_mismatch_error(self):
        """Test that channel mismatch raises error."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=8)
        data = np.random.randn(100, 4).astype(np.float32)

        with pytest.raises(ValueError, match="Channel count mismatch"):
            lap.process(data)

    def test_reset(self):
        """Test that reset is a no-op (Laplacian is stateless)."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=8)
        data = np.random.randn(100, 8).astype(np.float32)

        out1 = lap.process(data)
        lap.reset()
        out2 = lap.process(data)

        assert np.allclose(out1, out2)

    def test_repr(self):
        """Test string representation."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=8)
        assert 'SurfaceLaplacian' in repr(lap)
        assert '8' in repr(lap)

    def test_constant_input(self):
        """Test that constant input yields zeros for middle channels."""
        import npyci as npy

        lap = npy.SurfaceLaplacian.linear(channels=5)
        data = np.ones((10, 5), dtype=np.float32) * 5.0
        filtered = lap.process(data)

        # All channels should be zero (constant - constant = 0)
        assert np.allclose(filtered, 0, atol=1e-5)

    def test_dynamic_channels(self):
        """Test that dynamic implementation works correctly."""
        import npyci as npy

        # Use non-optimized channel count
        lap = npy.SurfaceLaplacian.linear(channels=10)

        # Create signal with peak at center (channel 5)
        data = np.zeros((1, 10), dtype=np.float32)
        data[0, 5] = 10.0

        filtered = lap.process(data)

        # Peak channel should have positive Laplacian
        assert filtered[0, 5] > 0

    def test_custom_neighbors_list_mismatch(self):
        """Test error when neighbors list length doesn't match channels."""
        import npyci as npy

        neighbors = [[1], [0, 2], [1]]  # Only 3 entries for 4 channels

        with pytest.raises(ValueError, match="must match channels"):
            npy.SurfaceLaplacian.custom(channels=4, neighbors=neighbors)


class TestSpatialIntegration:
    """Integration tests for spatial filters."""

    def test_car_then_laplacian(self):
        """Test CAR followed by Laplacian."""
        import npyci as npy

        car = npy.CAR(channels=8)
        lap = npy.SurfaceLaplacian.linear(channels=8)

        data = np.random.randn(100, 8).astype(np.float32)
        car_filtered = car.process(data)
        final = lap.process(car_filtered)

        assert final.shape == data.shape

    def test_multiple_samples(self):
        """Test processing multiple samples at once."""
        import npyci as npy

        car = npy.CAR(channels=8)

        # Process 1000 samples
        data = np.random.randn(1000, 8).astype(np.float32)
        filtered = car.process(data)

        assert filtered.shape == (1000, 8)
        assert np.allclose(filtered.mean(axis=1), 0, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
