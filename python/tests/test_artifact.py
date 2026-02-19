"""Tests for artifact detection module bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestArtifactDetector:
    """Tests for ArtifactDetector."""

    def test_create_detector(self):
        """Test creating an artifact detector."""
        det = zbci.ArtifactDetector(channels=8, amplitude_threshold=100.0, gradient_threshold=50.0)
        assert det.channels == 8
        assert det.amplitude_threshold == 100.0
        assert det.gradient_threshold == 50.0

    def test_invalid_channels(self):
        """Test that zero channels raises error."""
        with pytest.raises(ValueError):
            zbci.ArtifactDetector(channels=0, amplitude_threshold=100.0, gradient_threshold=50.0)

    def test_amplitude_only(self):
        """Test amplitude-only detection."""
        det = zbci.ArtifactDetector.amplitude_only(channels=4, threshold=100.0)

        # Below threshold
        sample = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        artifacts = det.detect(sample)
        assert not np.any(artifacts)

        # Above threshold
        sample = np.array([50.0, 150.0, 50.0, 200.0], dtype=np.float32)
        artifacts = det.detect(sample)
        assert not artifacts[0]
        assert artifacts[1]
        assert not artifacts[2]
        assert artifacts[3]

    def test_gradient_only(self):
        """Test gradient-only detection."""
        det = zbci.ArtifactDetector.gradient_only(channels=2, threshold=50.0)

        # First sample - no gradient check
        sample1 = np.array([100.0, 100.0], dtype=np.float32)
        artifacts = det.detect(sample1)
        assert not np.any(artifacts)

        # Large gradient on channel 0
        sample2 = np.array([200.0, 120.0], dtype=np.float32)
        artifacts = det.detect(sample2)
        assert artifacts[0]  # |200 - 100| = 100 > 50
        assert not artifacts[1]  # |120 - 100| = 20 < 50

    def test_combined_detection(self):
        """Test combined amplitude and gradient detection."""
        det = zbci.ArtifactDetector(channels=2, amplitude_threshold=100.0, gradient_threshold=50.0)

        # Initialize with low values
        det.detect(np.array([10.0, 10.0], dtype=np.float32))

        # Test amplitude artifact on channel 0, gradient artifact on channel 1
        sample = np.array([150.0, 80.0], dtype=np.float32)
        artifacts = det.detect(sample)
        assert artifacts[0]  # |150| > 100 (amplitude)
        assert artifacts[1]  # |80 - 10| = 70 > 50 (gradient)

    def test_detect_any(self):
        """Test detect_any method."""
        det = zbci.ArtifactDetector(channels=4, amplitude_threshold=100.0, gradient_threshold=50.0)

        # No artifacts
        sample = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        assert not det.detect_any(sample)

        # One artifact
        sample = np.array([50.0, 150.0, 50.0, 50.0], dtype=np.float32)
        assert det.detect_any(sample)

    def test_detect_count(self):
        """Test detect_count method."""
        det = zbci.ArtifactDetector(channels=4, amplitude_threshold=100.0, gradient_threshold=50.0)

        sample = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        assert det.detect_count(sample) == 0

        sample = np.array([150.0, 150.0, 50.0, 50.0], dtype=np.float32)
        assert det.detect_count(sample) == 2

    def test_reset(self):
        """Test resetting detector state."""
        det = zbci.ArtifactDetector.gradient_only(channels=2, threshold=50.0)

        # Initialize and create gradient state
        det.detect(np.array([50.0, 50.0], dtype=np.float32))

        # Reset
        det.reset()

        # After reset, no gradient check on first sample
        sample = np.array([120.0, 50.0], dtype=np.float32)
        artifacts = det.detect(sample)
        assert not np.any(artifacts)

    def test_optimized_channel_counts(self):
        """Test all optimized channel counts."""
        for channels in [1, 4, 8, 16, 32, 64]:
            det = zbci.ArtifactDetector(channels=channels, amplitude_threshold=100.0, gradient_threshold=50.0)
            sample = np.zeros(channels, dtype=np.float32)
            sample[0] = 150.0
            artifacts = det.detect(sample)
            assert artifacts[0]

    def test_dynamic_channel_count(self):
        """Test non-standard channel count uses dynamic implementation."""
        det = zbci.ArtifactDetector(channels=7, amplitude_threshold=100.0, gradient_threshold=50.0)
        sample = np.array([150.0] + [50.0] * 6, dtype=np.float32)
        artifacts = det.detect(sample)
        assert artifacts[0]
        assert not np.any(artifacts[1:])

    def test_repr(self):
        """Test string representation."""
        det = zbci.ArtifactDetector(channels=8, amplitude_threshold=100.0, gradient_threshold=50.0)
        assert "100" in repr(det)
        assert "50" in repr(det)


class TestZscoreArtifact:
    """Tests for ZscoreArtifact."""

    def test_create_detector(self):
        """Test creating a z-score artifact detector."""
        det = zbci.ZscoreArtifact(channels=8, threshold=3.0, min_samples=100)
        assert det.channels == 8
        assert det.threshold == 3.0
        assert det.is_calibrating

    def test_invalid_channels(self):
        """Test that unsupported channel count raises error."""
        with pytest.raises(ValueError):
            zbci.ZscoreArtifact(channels=3, threshold=3.0, min_samples=100)

    def test_calibration_period(self):
        """Test calibration period behavior."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=100)

        assert det.is_calibrating

        # Feed 99 samples - still calibrating
        for _ in range(99):
            sample = np.random.randn(4).astype(np.float32)
            det.update(sample)

        assert det.is_calibrating

        # 100th sample completes calibration
        det.update(np.random.randn(4).astype(np.float32))
        assert not det.is_calibrating

    def test_detection_during_calibration(self):
        """Test that detection returns all False during calibration."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=100)

        # Even with extreme values, should return False during calibration
        sample = np.array([1000.0] * 4, dtype=np.float32)
        artifacts = det.detect(sample)
        assert not np.any(artifacts)

    def test_detection_after_calibration(self):
        """Test artifact detection after calibration."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=50)

        # Calibrate with alternating values (mean ~0, std ~1)
        for i in range(50):
            val = 1.0 if i % 2 == 0 else -1.0
            det.update(np.array([val] * 4, dtype=np.float32))

        # Below threshold
        artifacts = det.detect(np.array([0.5] * 4, dtype=np.float32))
        assert not np.any(artifacts)

        # Above threshold (|z| > 3)
        artifacts = det.detect(np.array([5.0, 0.5, 0.5, 0.5], dtype=np.float32))
        assert artifacts[0]
        assert not np.any(artifacts[1:])

    def test_update_and_detect(self):
        """Test combined update and detect."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=50)

        # First 50 samples: calibrating
        for i in range(50):
            val = 1.0 if i % 2 == 0 else -1.0
            artifacts = det.update_and_detect(np.array([val] * 4, dtype=np.float32))
            assert not np.any(artifacts)

        # After calibration
        artifacts = det.update_and_detect(np.array([10.0, 0.5, 0.5, 0.5], dtype=np.float32))
        assert artifacts[0]

    def test_zscore_computation(self):
        """Test z-score computation."""
        det = zbci.ZscoreArtifact(channels=1, threshold=3.0, min_samples=50)

        # During calibration
        assert det.zscore(np.array([1.0], dtype=np.float32)) is None

        # Calibrate
        for i in range(50):
            val = 1.0 if i % 2 == 0 else -1.0
            det.update(np.array([val], dtype=np.float32))

        # Z-score should be close to sample value (mean ~0, std ~1)
        z = det.zscore(np.array([2.0], dtype=np.float32))
        assert z is not None
        assert abs(z[0] - 2.0) < 0.5

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing statistics."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=50)

        # Calibrate
        for i in range(50):
            val = 1.0 if i % 2 == 0 else -1.0
            det.update(np.array([val] * 4, dtype=np.float32))

        mean_before = det.mean().copy()

        # Freeze
        det.freeze()
        assert det.is_frozen

        # Update with different data
        for _ in range(50):
            det.update(np.array([100.0] * 4, dtype=np.float32))

        # Mean should not change
        np.testing.assert_array_almost_equal(det.mean(), mean_before)

        # Unfreeze
        det.unfreeze()
        assert not det.is_frozen

    def test_reset(self):
        """Test resetting detector state."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=50)

        # Calibrate and freeze
        for i in range(50):
            det.update(np.array([1.0] * 4, dtype=np.float32))
        det.freeze()

        assert not det.is_calibrating
        assert det.sample_count > 0

        # Reset
        det.reset()

        assert det.is_calibrating
        assert det.sample_count == 0
        assert not det.is_frozen

    def test_mean_and_std(self):
        """Test mean and std_dev methods."""
        det = zbci.ZscoreArtifact(channels=4, threshold=3.0, min_samples=50)

        # Feed constant values
        for _ in range(50):
            det.update(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        mean = det.mean()
        std = det.std_dev()

        # Mean should be close to [1, 2, 3, 4]
        np.testing.assert_array_almost_equal(mean, [1.0, 2.0, 3.0, 4.0], decimal=1)

        # Std should be close to 0 for constant signal
        assert np.all(std < 0.1)

    def test_supported_channel_counts(self):
        """Test all supported channel counts."""
        for channels in [1, 4, 8, 16, 32, 64]:
            det = zbci.ZscoreArtifact(channels=channels, threshold=3.0, min_samples=10)

            for _ in range(10):
                det.update(np.random.randn(channels).astype(np.float32))

            sample = np.zeros(channels, dtype=np.float32)
            sample[0] = 100.0
            artifacts = det.detect(sample)
            assert artifacts[0]

    def test_repr(self):
        """Test string representation."""
        det = zbci.ZscoreArtifact(channels=8, threshold=3.0, min_samples=100)
        assert "8" in repr(det)
        assert "3" in repr(det)  # May be "3" or "3.0" depending on format


class TestArtifactIntegration:
    """Integration tests for artifact detection."""

    def test_combined_detection_pipeline(self):
        """Test using both detectors in a pipeline."""
        amp_det = zbci.ArtifactDetector(channels=8, amplitude_threshold=100.0, gradient_threshold=50.0)
        zscore_det = zbci.ZscoreArtifact(channels=8, threshold=3.0, min_samples=50)

        # Calibration phase
        for i in range(50):
            sample = np.random.randn(8).astype(np.float32) * 10
            zscore_det.update(sample)

        # Detection phase
        for _ in range(10):
            sample = np.random.randn(8).astype(np.float32) * 10
            amp_artifacts = amp_det.detect(sample)
            zscore_artifacts = zscore_det.detect(sample)

            # Combine detections (OR logic)
            combined = np.logical_or(amp_artifacts, zscore_artifacts)
            assert combined.shape == (8,)
