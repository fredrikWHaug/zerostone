"""Tests for CSP (Common Spatial Patterns) module bindings."""

import numpy as np
import pytest
import npyci as npy


class TestAdaptiveCsp:
    """Tests for AdaptiveCsp."""

    def test_create_csp(self):
        """Test creating a CSP filter."""
        csp = npy.AdaptiveCsp(channels=8, filters=4)
        assert csp.channels == 8
        assert csp.num_filters == 4
        assert not csp.is_ready

    def test_create_with_min_samples(self):
        """Test creating CSP with custom min_samples."""
        csp = npy.AdaptiveCsp(channels=8, filters=4, min_samples=50)
        assert csp.channels == 8

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            npy.AdaptiveCsp(channels=3, filters=2)  # Invalid channel count
        with pytest.raises(ValueError):
            npy.AdaptiveCsp(channels=8, filters=3)  # Invalid filter count

    def test_supported_configurations(self):
        """Test all supported channel/filter combinations."""
        configs = [
            (4, 2), (4, 4),
            (8, 2), (8, 4), (8, 6),
            (16, 2), (16, 4), (16, 6),
            (32, 2), (32, 4), (32, 6),
            (64, 2), (64, 4), (64, 6),
        ]
        for channels, filters in configs:
            csp = npy.AdaptiveCsp(channels=channels, filters=filters)
            assert csp.channels == channels
            assert csp.num_filters == filters

    def test_update_class1(self):
        """Test updating with class 1 data."""
        csp = npy.AdaptiveCsp(channels=8, filters=4, min_samples=10)

        # Trial shape: (samples, channels)
        trial = np.random.randn(100, 8).astype(np.float64)
        csp.update_class1(trial)

        assert csp.class1_count == 100
        assert csp.class2_count == 0

    def test_update_class2(self):
        """Test updating with class 2 data."""
        csp = npy.AdaptiveCsp(channels=8, filters=4, min_samples=10)

        trial = np.random.randn(100, 8).astype(np.float64)
        csp.update_class2(trial)

        assert csp.class1_count == 0
        assert csp.class2_count == 100

    def test_channel_mismatch(self):
        """Test that channel mismatch raises error."""
        csp = npy.AdaptiveCsp(channels=8, filters=4)
        trial = np.random.randn(100, 4).astype(np.float64)  # Wrong channels

        with pytest.raises(ValueError, match="channels"):
            csp.update_class1(trial)

    def test_recompute_filters_insufficient_data(self):
        """Test that recompute fails with insufficient data."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=100)

        # Only add 50 samples
        trial = np.random.randn(50, 4).astype(np.float64)
        csp.update_class1(trial)
        csp.update_class2(trial)

        with pytest.raises(ValueError, match="Failed"):
            csp.recompute_filters()

    def test_recompute_filters_success(self):
        """Test successful filter computation."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=50)

        # Add sufficient data with different patterns
        # Class 1: signal in first two channels
        trial1 = np.zeros((100, 4), dtype=np.float64)
        trial1[:, 0] = np.random.randn(100)
        trial1[:, 1] = np.random.randn(100)

        # Class 2: signal in last two channels
        trial2 = np.zeros((100, 4), dtype=np.float64)
        trial2[:, 2] = np.random.randn(100)
        trial2[:, 3] = np.random.randn(100)

        csp.update_class1(trial1)
        csp.update_class2(trial2)

        csp.recompute_filters()
        assert csp.is_ready

    def test_apply_before_ready(self):
        """Test that apply fails before filters are ready."""
        csp = npy.AdaptiveCsp(channels=4, filters=2)
        sample = np.random.randn(4).astype(np.float64)

        with pytest.raises(ValueError):
            csp.apply(sample)

    def test_apply_single_sample(self):
        """Test applying CSP to a single sample."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=50)

        # Train
        trial1 = np.random.randn(100, 4).astype(np.float64)
        trial2 = np.random.randn(100, 4).astype(np.float64)
        csp.update_class1(trial1)
        csp.update_class2(trial2)
        csp.recompute_filters()

        # Apply
        sample = np.random.randn(4).astype(np.float64)
        features = csp.apply(sample)

        assert features.shape == (2,)

    def test_apply_block(self):
        """Test applying CSP to a block of samples."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=50)

        # Train
        trial1 = np.random.randn(100, 4).astype(np.float64)
        trial2 = np.random.randn(100, 4).astype(np.float64)
        csp.update_class1(trial1)
        csp.update_class2(trial2)
        csp.recompute_filters()

        # Apply to block
        block = np.random.randn(50, 4).astype(np.float64)
        features = csp.apply_block(block)

        assert features.shape == (50, 2)

    def test_get_filters(self):
        """Test getting computed filters."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=50)

        # Before training, filters should be None
        assert csp.filters() is None

        # Train
        trial1 = np.random.randn(100, 4).astype(np.float64)
        trial2 = np.random.randn(100, 4).astype(np.float64)
        csp.update_class1(trial1)
        csp.update_class2(trial2)
        csp.recompute_filters()

        # Now filters should be available
        filters = csp.filters()
        assert filters is not None
        assert filters.shape == (2, 4)  # (filters, channels)

    def test_reset(self):
        """Test resetting CSP state."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=50)

        # Train and compute
        trial = np.random.randn(100, 4).astype(np.float64)
        csp.update_class1(trial)
        csp.update_class2(trial)
        csp.recompute_filters()

        assert csp.is_ready
        assert csp.class1_count > 0

        # Reset
        csp.reset()

        assert not csp.is_ready
        assert csp.class1_count == 0
        assert csp.class2_count == 0

    def test_repr(self):
        """Test string representation."""
        csp = npy.AdaptiveCsp(channels=8, filters=4)
        assert "8" in repr(csp)
        assert "4" in repr(csp)


class TestCspMotorImagery:
    """Tests simulating motor imagery scenarios."""

    def test_two_class_separation(self):
        """Test that CSP can separate two classes with different spatial patterns."""
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=100)

        np.random.seed(42)

        # Class 1: Strong activity in channels 0-1
        for _ in range(5):
            trial = np.zeros((50, 4), dtype=np.float64)
            trial[:, 0] = np.sin(np.linspace(0, 10*np.pi, 50)) + np.random.randn(50) * 0.1
            trial[:, 1] = np.sin(np.linspace(0, 10*np.pi, 50)) + np.random.randn(50) * 0.1
            trial[:, 2] = np.random.randn(50) * 0.1
            trial[:, 3] = np.random.randn(50) * 0.1
            csp.update_class1(trial)

        # Class 2: Strong activity in channels 2-3
        for _ in range(5):
            trial = np.zeros((50, 4), dtype=np.float64)
            trial[:, 0] = np.random.randn(50) * 0.1
            trial[:, 1] = np.random.randn(50) * 0.1
            trial[:, 2] = np.sin(np.linspace(0, 10*np.pi, 50)) + np.random.randn(50) * 0.1
            trial[:, 3] = np.sin(np.linspace(0, 10*np.pi, 50)) + np.random.randn(50) * 0.1
            csp.update_class2(trial)

        csp.recompute_filters()
        assert csp.is_ready

        # Test samples from each class
        sample1 = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)  # Class 1 pattern
        sample2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)  # Class 2 pattern

        features1 = csp.apply(sample1)
        features2 = csp.apply(sample2)

        # Features should be different for different classes
        assert not np.allclose(features1, features2, atol=0.1)

    def test_online_processing(self):
        """Test CSP in an online processing scenario."""
        # Use 4 channels which is simpler and more reliable
        csp = npy.AdaptiveCsp(channels=4, filters=2, min_samples=100, regularization=1e-3)

        np.random.seed(42)

        # Calibration phase - generate clearly separable classes
        for _ in range(30):
            # Class 1: high variance in channels 0-1, low in 2-3
            trial1 = np.zeros((20, 4), dtype=np.float64)
            trial1[:, 0] = np.random.randn(20) * 2.0 + np.sin(np.linspace(0, 4*np.pi, 20))
            trial1[:, 1] = np.random.randn(20) * 2.0 + np.cos(np.linspace(0, 4*np.pi, 20))
            trial1[:, 2] = np.random.randn(20) * 0.2
            trial1[:, 3] = np.random.randn(20) * 0.2

            # Class 2: low variance in channels 0-1, high in 2-3
            trial2 = np.zeros((20, 4), dtype=np.float64)
            trial2[:, 0] = np.random.randn(20) * 0.2
            trial2[:, 1] = np.random.randn(20) * 0.2
            trial2[:, 2] = np.random.randn(20) * 2.0 + np.sin(np.linspace(0, 4*np.pi, 20))
            trial2[:, 3] = np.random.randn(20) * 2.0 + np.cos(np.linspace(0, 4*np.pi, 20))

            csp.update_class1(trial1)
            csp.update_class2(trial2)

        csp.recompute_filters()

        # Online processing phase
        for _ in range(10):
            sample = np.random.randn(4).astype(np.float64)
            features = csp.apply(sample)
            assert features.shape == (2,)
            assert np.all(np.isfinite(features))
