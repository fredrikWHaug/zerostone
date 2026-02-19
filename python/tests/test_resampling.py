"""Tests for resampling module bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestDecimator:
    """Tests for Decimator."""

    def test_create_decimator(self):
        """Test creating a decimator."""
        dec = zbci.Decimator(factor=4, channels=8)
        assert dec.factor == 4
        assert dec.channels == 8

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            zbci.Decimator(factor=0, channels=8)  # factor must be >= 1
        with pytest.raises(ValueError):
            zbci.Decimator(factor=4, channels=0)  # channels must be >= 1

    def test_decimate_4x(self):
        """Test 4x decimation."""
        dec = zbci.Decimator(factor=4, channels=8)

        # 1000 samples, 8 channels
        data = np.random.randn(1000, 8).astype(np.float32)
        decimated = dec.process(data)

        # Should have 250 samples (1000 / 4)
        assert decimated.shape == (250, 8)

    def test_decimate_preserves_samples(self):
        """Test that decimator keeps every Nth sample."""
        dec = zbci.Decimator(factor=4, channels=4)

        # Create known signal
        data = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
        decimated = dec.process(data)

        # First decimated sample should be first input sample
        np.testing.assert_array_almost_equal(decimated[0], data[0])

        # Check that we kept every 4th sample
        for i in range(decimated.shape[0]):
            np.testing.assert_array_almost_equal(decimated[i], data[i * 4])

    def test_channel_count_mismatch(self):
        """Test that channel count mismatch raises error."""
        dec = zbci.Decimator(factor=4, channels=8)
        data = np.random.randn(100, 4).astype(np.float32)  # Wrong channel count

        with pytest.raises(ValueError, match="Channel count mismatch"):
            dec.process(data)

    def test_reset(self):
        """Test resetting decimator state."""
        dec = zbci.Decimator(factor=4, channels=4)

        # Process partial data (not multiple of factor)
        data1 = np.random.randn(6, 4).astype(np.float32)
        out1 = dec.process(data1)

        # Reset
        dec.reset()

        # Process again - should behave like fresh decimator
        data2 = np.random.randn(8, 4).astype(np.float32)
        out2 = dec.process(data2)

        # First output should be first input
        np.testing.assert_array_almost_equal(out2[0], data2[0])

    def test_optimized_channel_counts(self):
        """Test that optimized channel counts work correctly."""
        for channels in [1, 4, 8, 16, 32, 64]:
            dec = zbci.Decimator(factor=4, channels=channels)
            data = np.random.randn(100, channels).astype(np.float32)
            decimated = dec.process(data)
            assert decimated.shape == (25, channels)

    def test_dynamic_channel_count(self):
        """Test non-standard channel count uses dynamic implementation."""
        dec = zbci.Decimator(factor=4, channels=7)  # Non-standard
        data = np.random.randn(100, 7).astype(np.float32)
        decimated = dec.process(data)
        assert decimated.shape == (25, 7)

    def test_factor_1(self):
        """Test factor=1 (no decimation)."""
        dec = zbci.Decimator(factor=1, channels=4)
        data = np.random.randn(100, 4).astype(np.float32)
        decimated = dec.process(data)

        # Should keep all samples
        assert decimated.shape == data.shape
        np.testing.assert_array_almost_equal(decimated, data)

    def test_repr(self):
        """Test string representation."""
        dec = zbci.Decimator(factor=4, channels=8)
        assert "factor=4" in repr(dec)
        assert "channels=8" in repr(dec)


class TestInterpolator:
    """Tests for Interpolator."""

    def test_create_interpolator(self):
        """Test creating an interpolator."""
        interp = zbci.Interpolator(factor=4, channels=8, method='linear')
        assert interp.factor == 4
        assert interp.channels == 8
        assert interp.method == 'linear'

    def test_default_method(self):
        """Test that default method is linear."""
        interp = zbci.Interpolator(factor=4, channels=8)
        assert interp.method == 'linear'

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            zbci.Interpolator(factor=0, channels=8)  # factor must be >= 1
        with pytest.raises(ValueError):
            zbci.Interpolator(factor=4, channels=0)  # channels must be >= 1
        with pytest.raises(ValueError):
            zbci.Interpolator(factor=4, channels=8, method='invalid')

    def test_interpolate_4x(self):
        """Test 4x interpolation."""
        interp = zbci.Interpolator(factor=4, channels=8, method='linear')

        # 100 samples, 8 channels
        data = np.random.randn(100, 8).astype(np.float32)
        upsampled = interp.process(data)

        # Should have 400 samples (100 * 4)
        assert upsampled.shape == (400, 8)

    def test_zero_order_hold(self):
        """Test zero-order hold interpolation."""
        interp = zbci.Interpolator(factor=4, channels=2, method='zero_order')

        # Simple step signal
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        upsampled = interp.process(data)

        assert upsampled.shape == (8, 2)

        # Zero-order hold repeats each sample
        # First 4 samples should all be [1.0, 2.0]
        for i in range(4):
            np.testing.assert_array_almost_equal(upsampled[i], [1.0, 2.0])

        # Next 4 samples should all be [3.0, 4.0]
        for i in range(4, 8):
            np.testing.assert_array_almost_equal(upsampled[i], [3.0, 4.0])

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        interp = zbci.Interpolator(factor=4, channels=1, method='linear')

        # Ramp signal
        data = np.array([[0.0], [4.0]], dtype=np.float32)
        upsampled = interp.process(data)

        assert upsampled.shape == (8, 1)

        # After warmup, should interpolate linearly
        # First 4 values are from first sample warmup
        # Next 4 values should interpolate from 0 to 4
        # The second batch interpolates between prev (0) and current (4)
        # At k=0: 0.0, k=1: 1.0, k=2: 2.0, k=3: 3.0

    def test_zero_insert(self):
        """Test zero-insert interpolation."""
        interp = zbci.Interpolator(factor=4, channels=2, method='zero_insert')

        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        upsampled = interp.process(data)

        assert upsampled.shape == (8, 2)

        # Zero-insert keeps original samples and inserts zeros
        np.testing.assert_array_almost_equal(upsampled[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(upsampled[1], [0.0, 0.0])
        np.testing.assert_array_almost_equal(upsampled[2], [0.0, 0.0])
        np.testing.assert_array_almost_equal(upsampled[3], [0.0, 0.0])
        np.testing.assert_array_almost_equal(upsampled[4], [3.0, 4.0])

    def test_channel_count_mismatch(self):
        """Test that channel count mismatch raises error."""
        interp = zbci.Interpolator(factor=4, channels=8)
        data = np.random.randn(100, 4).astype(np.float32)  # Wrong channel count

        with pytest.raises(ValueError, match="Channel count mismatch"):
            interp.process(data)

    def test_reset(self):
        """Test resetting interpolator state."""
        interp = zbci.Interpolator(factor=4, channels=4, method='linear')

        # Process some data
        data1 = np.random.randn(10, 4).astype(np.float32)
        interp.process(data1)

        # Reset
        interp.reset()

        # Process again - should behave like fresh interpolator
        data2 = np.random.randn(10, 4).astype(np.float32)
        out = interp.process(data2)

        assert out.shape == (40, 4)

    def test_optimized_channel_counts(self):
        """Test that optimized channel counts work correctly."""
        for channels in [1, 4, 8, 16, 32, 64]:
            interp = zbci.Interpolator(factor=4, channels=channels)
            data = np.random.randn(100, channels).astype(np.float32)
            upsampled = interp.process(data)
            assert upsampled.shape == (400, channels)

    def test_dynamic_channel_count(self):
        """Test non-standard channel count uses dynamic implementation."""
        interp = zbci.Interpolator(factor=4, channels=7)  # Non-standard
        data = np.random.randn(100, 7).astype(np.float32)
        upsampled = interp.process(data)
        assert upsampled.shape == (400, 7)

    def test_factor_1(self):
        """Test factor=1 (no interpolation)."""
        interp = zbci.Interpolator(factor=1, channels=4)
        data = np.random.randn(100, 4).astype(np.float32)
        upsampled = interp.process(data)

        # Should keep same number of samples
        assert upsampled.shape == data.shape

    def test_repr(self):
        """Test string representation."""
        interp = zbci.Interpolator(factor=4, channels=8, method='linear')
        assert "factor=4" in repr(interp)
        assert "channels=8" in repr(interp)
        assert "linear" in repr(interp)


class TestDecimatorInterpolatorRoundtrip:
    """Test decimator and interpolator together."""

    def test_decimate_then_interpolate(self):
        """Test that decimate then interpolate preserves length."""
        dec = zbci.Decimator(factor=4, channels=4)
        interp = zbci.Interpolator(factor=4, channels=4, method='zero_order')

        # Original signal
        data = np.random.randn(100, 4).astype(np.float32)

        # Decimate then interpolate
        decimated = dec.process(data)
        reconstructed = interp.process(decimated)

        # Length should be preserved
        assert reconstructed.shape == data.shape
