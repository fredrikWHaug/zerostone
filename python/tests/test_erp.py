"""Tests for ERP and xDAWN Python bindings."""
import numpy as np
import pytest


class TestEpochAverage:
    """Tests for epoch_average function."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'epoch_average')

    def test_basic_average(self):
        """Test averaging of simple epochs."""
        import zpybci as zbci

        # Create 3 trials of 2 channels × 100 samples
        epochs = np.array([
            [[1.0, 2.0]] * 100,
            [[3.0, 4.0]] * 100,
            [[5.0, 6.0]] * 100,
        ], dtype=np.float64)

        avg = zbci.epoch_average(epochs)

        assert avg.shape == (100, 2)
        assert avg.dtype == np.float64
        np.testing.assert_allclose(avg[0], [3.0, 4.0], rtol=1e-10)  # (1+3+5)/3 = 3, (2+4+6)/3 = 4

    def test_single_epoch(self):
        """Test averaging of a single epoch."""
        import zpybci as zbci

        epochs = np.array([[[1.5, 2.5]] * 50], dtype=np.float64)
        avg = zbci.epoch_average(epochs)

        assert avg.shape == (50, 2)
        np.testing.assert_allclose(avg[0], [1.5, 2.5], rtol=1e-10)

    def test_different_channels(self):
        """Test with different channel counts."""
        import zpybci as zbci

        # 4 channels
        epochs = np.random.randn(5, 30, 4).astype(np.float64)
        avg = zbci.epoch_average(epochs)
        assert avg.shape == (30, 4)

        # 8 channels
        epochs = np.random.randn(10, 50, 8).astype(np.float64)
        avg = zbci.epoch_average(epochs)
        assert avg.shape == (50, 8)

        # 16 channels
        epochs = np.random.randn(3, 100, 16).astype(np.float64)
        avg = zbci.epoch_average(epochs)
        assert avg.shape == (100, 16)

    def test_noise_cancellation(self):
        """Averaging should reduce uncorrelated noise."""
        import zpybci as zbci

        np.random.seed(42)
        # True signal + noise
        true_signal = np.sin(np.linspace(0, 2*np.pi, 100))
        epochs = []
        for _ in range(100):
            noise = np.random.randn(100) * 0.5
            trial = np.column_stack([true_signal + noise, true_signal + noise])
            epochs.append(trial)

        epochs = np.array(epochs, dtype=np.float64)
        avg = zbci.epoch_average(epochs)

        # Averaged result should be closer to true signal than individual trials
        np.testing.assert_allclose(avg[:, 0], true_signal, atol=0.2)

    def test_empty_epochs(self):
        """Test with empty epochs array."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="Empty epochs"):
            zbci.epoch_average(np.empty((0, 10, 2), dtype=np.float64))

    def test_unsupported_channels(self):
        """Test with unsupported channel count."""
        import zpybci as zbci

        epochs = np.random.randn(5, 100, 7).astype(np.float64)
        with pytest.raises(ValueError, match="Unsupported number of channels"):
            zbci.epoch_average(epochs)


class TestXDawnTransformer:
    """Tests for XDawnTransformer class."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'XDawnTransformer')

    def test_initialization(self):
        """Test XDawnTransformer initialization."""
        import zpybci as zbci

        xdawn = zbci.XDawnTransformer(channels=8, filters=2)
        repr_str = repr(xdawn)
        assert "XDawnTransformer" in repr_str
        assert "channels=8" in repr_str
        assert "filters=2" in repr_str
        assert "fitted=False" in repr_str

    def test_invalid_channels(self):
        """Test initialization with invalid channel count."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="Unsupported channels"):
            zbci.XDawnTransformer(channels=7, filters=2)

    def test_invalid_filters(self):
        """Test initialization with invalid filter count."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="Unsupported filters"):
            zbci.XDawnTransformer(channels=8, filters=3)

    def test_filters_exceed_channels(self):
        """Test that filters cannot exceed channels."""
        import zpybci as zbci

        with pytest.raises(ValueError, match="cannot exceed channels"):
            zbci.XDawnTransformer(channels=4, filters=6)

    def test_fit_transform_synthetic_p300(self):
        """Test fit/transform on synthetic P300 data."""
        import zpybci as zbci

        np.random.seed(42)

        # Create synthetic data: 2 channels, 100 samples per trial
        n_trials = 40
        n_samples = 100
        n_channels = 2

        # Target trials (label=1): strong evoked response
        target_epochs = []
        for _ in range(n_trials // 2):
            t = np.linspace(0, 1, n_samples)
            signal = 2.0 * np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
            noise = np.random.randn(n_samples) * 0.5
            trial = np.column_stack([signal + noise, signal * 1.2 + noise])
            target_epochs.append(trial)

        # Nontarget trials (label=0): weak response
        nontarget_epochs = []
        for _ in range(n_trials // 2):
            noise1 = np.random.randn(n_samples) * 0.5
            noise2 = np.random.randn(n_samples) * 0.5
            trial = np.column_stack([noise1, noise2])
            nontarget_epochs.append(trial)

        # Combine
        X = np.array(target_epochs + nontarget_epochs, dtype=np.float64)
        y = np.array([1] * (n_trials // 2) + [0] * (n_trials // 2), dtype=np.int64)

        # Fit transformer
        xdawn = zbci.XDawnTransformer(channels=n_channels, filters=1)
        xdawn.fit(X, y)

        assert "fitted=True" in repr(xdawn)

        # Transform
        X_filtered = xdawn.transform(X)

        assert X_filtered.shape == (n_trials, n_samples, 1)
        assert X_filtered.dtype == np.float64

    def test_fit_without_targets(self):
        """Test fit with no target epochs."""
        import zpybci as zbci

        X = np.random.randn(10, 50, 4).astype(np.float64)
        y = np.zeros(10, dtype=np.int64)  # All nontargets

        xdawn = zbci.XDawnTransformer(channels=4, filters=2)

        with pytest.raises(ValueError, match="at least one target"):
            xdawn.fit(X, y)

    def test_fit_without_nontargets(self):
        """Test fit with no nontarget epochs."""
        import zpybci as zbci

        X = np.random.randn(10, 50, 4).astype(np.float64)
        y = np.ones(10, dtype=np.int64)  # All targets

        xdawn = zbci.XDawnTransformer(channels=4, filters=2)

        with pytest.raises(ValueError, match="at least one.*nontarget"):
            xdawn.fit(X, y)

    def test_transform_before_fit(self):
        """Test transform before calling fit."""
        import zpybci as zbci

        xdawn = zbci.XDawnTransformer(channels=4, filters=2)
        X = np.random.randn(10, 50, 4).astype(np.float64)

        with pytest.raises(ValueError, match="not fitted"):
            xdawn.transform(X)

    def test_dimension_mismatch(self):
        """Test fit/transform with dimension mismatch."""
        import zpybci as zbci

        X_train = np.random.randn(10, 50, 4).astype(np.float64)
        y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

        xdawn = zbci.XDawnTransformer(channels=4, filters=2)
        xdawn.fit(X_train, y_train)

        # Try transform with wrong channels
        X_test = np.random.randn(5, 50, 8).astype(np.float64)
        with pytest.raises(ValueError, match="expected 4"):
            xdawn.transform(X_test)

    def test_multiple_filter_counts(self):
        """Test with different filter counts."""
        import zpybci as zbci

        np.random.seed(42)
        X = np.random.randn(20, 30, 8).astype(np.float64)
        y = np.array([1, 0] * 10, dtype=np.int64)

        for n_filters in [1, 2, 4, 6]:
            xdawn = zbci.XDawnTransformer(channels=8, filters=n_filters)
            xdawn.fit(X, y)
            X_filtered = xdawn.transform(X)
            assert X_filtered.shape == (20, 30, n_filters)

    def test_sklearn_pipeline_compatibility(self):
        """Test that XDawnTransformer has sklearn-like interface."""
        import zpybci as zbci

        # Check if it has the right interface
        xdawn = zbci.XDawnTransformer(channels=4, filters=2)
        assert hasattr(xdawn, 'fit')
        assert hasattr(xdawn, 'transform')

        # Test fit/transform workflow
        np.random.seed(42)
        X = np.random.randn(20, 30, 4).astype(np.float64)
        y = np.array([1, 0] * 10, dtype=np.int64)

        xdawn.fit(X, y)
        X_transformed = xdawn.transform(X)
        assert X_transformed.shape == (20, 30, 2)

    def test_end_to_end_classification(self):
        """Test XDawnTransformer in a complete BCI pipeline."""
        import zpybci as zbci

        np.random.seed(42)

        # Generate synthetic P300 data with clear difference
        n_trials = 60
        n_samples = 100
        n_channels = 4

        X = []
        y = []

        # Target trials: strong signal at 300ms (sample 30 at 100 Hz)
        for _ in range(n_trials // 2):
            t = np.arange(n_samples) / 100.0
            signal = np.exp(-((t - 0.3) ** 2) / 0.01) * 5.0  # Gaussian at 300ms
            trial = np.column_stack([
                signal + np.random.randn(n_samples) * 0.5,
                signal * 1.2 + np.random.randn(n_samples) * 0.5,
                signal * 0.8 + np.random.randn(n_samples) * 0.5,
                signal * 1.1 + np.random.randn(n_samples) * 0.5,
            ])
            X.append(trial)
            y.append(1)

        # Nontarget trials: just noise
        for _ in range(n_trials // 2):
            trial = np.random.randn(n_samples, n_channels) * 0.5
            X.append(trial)
            y.append(0)

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        # Fit xDAWN
        xdawn = zbci.XDawnTransformer(channels=n_channels, filters=2)
        xdawn.fit(X, y)
        X_filtered = xdawn.transform(X)

        # The filtered data should enhance the difference between classes
        # Compute mean power for target vs nontarget
        target_power = np.mean(X_filtered[y == 1] ** 2)
        nontarget_power = np.mean(X_filtered[y == 0] ** 2)

        # Target should have higher power after xDAWN filtering
        assert target_power > nontarget_power

    def test_different_channel_configs(self):
        """Test with all supported channel configurations."""
        import zpybci as zbci

        np.random.seed(42)

        for channels in [2, 4, 8, 16, 32, 64]:
            for filters in [1, 2]:
                if filters > channels:
                    continue

                X = np.random.randn(10, 50, channels).astype(np.float64)
                y = np.array([1, 0] * 5, dtype=np.int64)

                xdawn = zbci.XDawnTransformer(channels=channels, filters=filters)
                xdawn.fit(X, y)
                X_filtered = xdawn.transform(X)

                assert X_filtered.shape == (10, 50, filters), \
                    f"Failed for channels={channels}, filters={filters}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
