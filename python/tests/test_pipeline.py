"""Tests for Pipeline and ChannelRouter Python bindings."""
import numpy as np
import pytest


class TestChannelRouter:
    """Tests for ChannelRouter."""

    def test_import(self):
        """Test that ChannelRouter can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'ChannelRouter')

    def test_select(self):
        """Test channel selection."""
        import zpybci as zbci

        router = zbci.ChannelRouter.select(in_channels=8, indices=[0, 2, 4])
        assert router.in_channels == 8
        assert router.out_channels == 3

    def test_select_process(self):
        """Test that select routes channels correctly."""
        import zpybci as zbci

        router = zbci.ChannelRouter.select(in_channels=8, indices=[0, 2, 4])
        data = np.arange(16).reshape(2, 8).astype(np.float32)
        result = router.process(data)

        assert result.shape == (2, 3)
        assert np.allclose(result[0], [0, 2, 4])
        assert np.allclose(result[1], [8, 10, 12])

    def test_permute(self):
        """Test channel permutation."""
        import zpybci as zbci

        router = zbci.ChannelRouter.permute(channels=4, indices=[3, 2, 1, 0])
        assert router.in_channels == 4
        assert router.out_channels == 4

    def test_permute_reverse(self):
        """Test that permute reverses channels correctly."""
        import zpybci as zbci

        router = zbci.ChannelRouter.permute(channels=4, indices=[3, 2, 1, 0])
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        result = router.process(data)

        assert result.shape == (2, 4)
        assert np.allclose(result[0], [4, 3, 2, 1])
        assert np.allclose(result[1], [8, 7, 6, 5])

    def test_identity(self):
        """Test identity router."""
        import zpybci as zbci

        router = zbci.ChannelRouter.identity(channels=8)
        assert router.in_channels == 8
        assert router.out_channels == 8

    def test_identity_passthrough(self):
        """Test that identity passes data unchanged."""
        import zpybci as zbci

        router = zbci.ChannelRouter.identity(channels=4)
        data = np.random.randn(10, 4).astype(np.float32)
        result = router.process(data)

        assert result.shape == data.shape
        assert np.allclose(result, data)

    def test_select_invalid_index(self):
        """Test that invalid indices raise error."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="out of bounds"):
            zbci.ChannelRouter.select(in_channels=4, indices=[0, 5])

    def test_permute_length_mismatch(self):
        """Test that permute with wrong length raises error."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="must match"):
            zbci.ChannelRouter.permute(channels=4, indices=[0, 1, 2])

    def test_channel_mismatch(self):
        """Test that channel mismatch raises error."""
        import zpybci as zbci

        router = zbci.ChannelRouter.select(in_channels=8, indices=[0, 1])
        data = np.random.randn(10, 4).astype(np.float32)

        with pytest.raises(ValueError, match="Channel count mismatch"):
            router.process(data)

    def test_reset(self):
        """Test that reset is a no-op."""
        import zpybci as zbci

        router = zbci.ChannelRouter.identity(channels=4)
        router.reset()  # Should not raise

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        router = zbci.ChannelRouter.select(in_channels=8, indices=[0, 2, 4])
        assert 'ChannelRouter' in repr(router)
        assert '8' in repr(router)
        assert '3' in repr(router)

    def test_optimized_sizes(self):
        """Test optimized channel configurations."""
        import zpybci as zbci

        for channels in [4, 8, 16, 32, 64]:
            router = zbci.ChannelRouter.identity(channels=channels)
            data = np.random.randn(10, channels).astype(np.float32)
            result = router.process(data)
            assert np.allclose(result, data)

    def test_dynamic_size(self):
        """Test non-optimized channel configurations."""
        import zpybci as zbci

        router = zbci.ChannelRouter.identity(channels=10)
        data = np.random.randn(10, 10).astype(np.float32)
        result = router.process(data)
        assert np.allclose(result, data)


class TestPipeline:
    """Tests for Pipeline."""

    def test_import(self):
        """Test that Pipeline can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'Pipeline')

    def test_create_single_stage(self):
        """Test creating single-stage pipeline."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        pipeline = zbci.Pipeline([car])
        assert pipeline.n_stages == 1
        assert len(pipeline) == 1

    def test_create_multi_stage(self):
        """Test creating multi-stage pipeline."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([car, lap])
        assert pipeline.n_stages == 2

    def test_empty_pipeline_error(self):
        """Test that empty pipeline raises error."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="at least one"):
            zbci.Pipeline([])

    def test_process_single_stage(self):
        """Test processing through single-stage pipeline."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        pipeline = zbci.Pipeline([car])

        data = np.random.randn(100, 8).astype(np.float32)
        result = pipeline.process(data)

        # Compare with direct CAR processing
        car2 = zbci.CAR(channels=8)
        expected = car2.process(data)

        assert result.shape == expected.shape
        assert np.allclose(result, expected)

    def test_process_multi_stage(self):
        """Test processing through multi-stage pipeline."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([car, lap])

        data = np.random.randn(100, 8).astype(np.float32)
        result = pipeline.process(data)

        # Compare with sequential processing
        car2 = zbci.CAR(channels=8)
        lap2 = zbci.SurfaceLaplacian.linear(channels=8)
        expected = lap2.process(car2.process(data))

        assert result.shape == expected.shape
        assert np.allclose(result, expected)

    def test_pipeline_with_router(self):
        """Test pipeline with ChannelRouter."""
        import zpybci as zbci

        # 8 channels -> select 4 -> CAR on 4
        router = zbci.ChannelRouter.select(in_channels=8, indices=[0, 2, 4, 6])
        car = zbci.CAR(channels=4)
        pipeline = zbci.Pipeline([router, car])

        data = np.random.randn(100, 8).astype(np.float32)
        result = pipeline.process(data)

        assert result.shape == (100, 4)

        # Verify CAR property (zero mean)
        assert np.allclose(result.mean(axis=1), 0, atol=1e-5)

    def test_three_stage_pipeline(self):
        """Test three-stage pipeline."""
        import zpybci as zbci

        router = zbci.ChannelRouter.identity(channels=8)
        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([router, car, lap])

        data = np.random.randn(100, 8).astype(np.float32)
        result = pipeline.process(data)

        assert result.shape == (100, 8)
        assert pipeline.n_stages == 3

    def test_reset(self):
        """Test pipeline reset."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([car, lap])

        data = np.random.randn(100, 8).astype(np.float32)
        result1 = pipeline.process(data)

        pipeline.reset()
        result2 = pipeline.process(data)

        # Stateless processors should give same result
        assert np.allclose(result1, result2)

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([car, lap])

        assert 'Pipeline' in repr(pipeline)
        assert '2' in repr(pipeline)

    def test_channel_changing_pipeline(self):
        """Test pipeline that changes channel count."""
        import zpybci as zbci

        # 16 channels -> select 8 -> CAR on 8
        router = zbci.ChannelRouter.select(in_channels=16, indices=[0, 2, 4, 6, 8, 10, 12, 14])
        car = zbci.CAR(channels=8)
        pipeline = zbci.Pipeline([router, car])

        data = np.random.randn(50, 16).astype(np.float32)
        result = pipeline.process(data)

        assert result.shape == (50, 8)

    def test_pipeline_preserves_sample_count(self):
        """Test that pipeline preserves sample count."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        pipeline = zbci.Pipeline([car])

        for n_samples in [1, 10, 100, 1000]:
            data = np.random.randn(n_samples, 8).astype(np.float32)
            result = pipeline.process(data)
            assert result.shape[0] == n_samples


class TestPipelineIntegration:
    """Integration tests for Pipeline with various processors."""

    def test_car_then_laplacian(self):
        """Test CAR followed by Laplacian."""
        import zpybci as zbci

        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([car, lap])

        # Create signal with common noise
        n_samples = 100
        common_noise = np.random.randn(n_samples, 1).astype(np.float32)
        data = np.random.randn(n_samples, 8).astype(np.float32) + common_noise

        result = pipeline.process(data)

        assert result.shape == (n_samples, 8)

    def test_router_car_laplacian(self):
        """Test Router -> CAR -> Laplacian pipeline."""
        import zpybci as zbci

        router = zbci.ChannelRouter.permute(channels=8, indices=[7, 6, 5, 4, 3, 2, 1, 0])
        car = zbci.CAR(channels=8)
        lap = zbci.SurfaceLaplacian.linear(channels=8)
        pipeline = zbci.Pipeline([router, car, lap])

        data = np.random.randn(100, 8).astype(np.float32)
        result = pipeline.process(data)

        assert result.shape == (100, 8)
        assert pipeline.n_stages == 3

    def test_channel_selection_pipeline(self):
        """Test channel selection with processing."""
        import zpybci as zbci

        # Select subset of channels, then process
        select = zbci.ChannelRouter.select(in_channels=32, indices=[0, 8, 16, 24])
        car = zbci.CAR(channels=4)
        pipeline = zbci.Pipeline([select, car])

        data = np.random.randn(100, 32).astype(np.float32)
        result = pipeline.process(data)

        assert result.shape == (100, 4)
        # CAR output should have zero mean
        assert np.allclose(result.mean(axis=1), 0, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
