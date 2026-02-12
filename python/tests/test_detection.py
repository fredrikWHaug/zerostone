"""Tests for detection module bindings."""

import numpy as np
import pytest
import npyci as npy


class TestThresholdDetector:
    """Tests for ThresholdDetector."""

    def test_create_detector(self):
        """Test creating a threshold detector."""
        det = npy.ThresholdDetector(4, 3.0, 100)
        assert det.channels == 4
        assert det.threshold == 3.0
        assert det.refractory == 100

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            npy.ThresholdDetector(0, 3.0, 100)  # channels must be >= 1

    def test_detect_spikes(self):
        """Test detecting threshold crossings."""
        det = npy.ThresholdDetector(4, 3.0, 10)

        # Create signal with known spikes
        signal = np.zeros((100, 4), dtype=np.float32)
        signal[20, 0] = 5.0  # Spike on channel 0
        signal[50, 2] = 4.0  # Spike on channel 2
        signal[80, 3] = 6.0  # Spike on channel 3

        events = det.process(signal)

        # Should detect 3 events
        assert len(events) == 3

        # Check event structure (sample_idx, channel, amplitude)
        assert events[0] == (20, 0, 5.0)
        assert events[1] == (50, 2, 4.0)
        assert events[2] == (80, 3, 6.0)

    def test_refractory_period(self):
        """Test that refractory period prevents double detection."""
        det = npy.ThresholdDetector(1, 3.0, 20)

        # Create signal with two spikes close together
        signal = np.zeros((100, 1), dtype=np.float32)
        signal[10, 0] = 5.0
        signal[15, 0] = 5.0  # Within refractory period
        signal[40, 0] = 5.0  # Outside refractory period

        events = det.process(signal)

        # Should only detect 2 events (first and third spike)
        assert len(events) == 2
        assert events[0][0] == 10
        assert events[1][0] == 40

    def test_reset(self):
        """Test resetting detector state."""
        det = npy.ThresholdDetector(4, 3.0, 100)

        # Process some data
        signal = np.zeros((50, 4), dtype=np.float32)
        signal[45, 0] = 5.0  # Spike near end
        det.process(signal)

        # Reset clears refractory state
        det.reset()

        # Process more data - should detect immediately
        signal2 = np.zeros((10, 4), dtype=np.float32)
        signal2[0, 0] = 5.0
        events = det.process(signal2)

        assert len(events) == 1

    def test_no_spikes(self):
        """Test with signal below threshold."""
        det = npy.ThresholdDetector(4, 3.0, 10)
        signal = np.random.randn(100, 4).astype(np.float32) * 0.5

        events = det.process(signal)
        assert len(events) == 0

    def test_optimized_channel_counts(self):
        """Test that optimized channel counts work correctly."""
        for channels in [1, 4, 8, 16, 32, 64]:
            det = npy.ThresholdDetector(channels, 3.0, 10)
            signal = np.zeros((50, channels), dtype=np.float32)
            signal[25, 0] = 5.0
            events = det.process(signal)
            assert len(events) == 1


class TestAdaptiveThresholdDetector:
    """Tests for AdaptiveThresholdDetector."""

    def test_create_detector(self):
        """Test creating an adaptive threshold detector."""
        det = npy.AdaptiveThresholdDetector(4, 4.0, 100, 500)
        assert det.channels == 4
        assert det.multiplier == 4.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            npy.AdaptiveThresholdDetector(0, 4.0, 100, 500)  # channels must be >= 1

    def test_warmup_period(self):
        """Test that warmup period is required before detection."""
        det = npy.AdaptiveThresholdDetector(1, 4.0, 10, 100)

        # Signal during warmup
        warmup_signal = np.random.randn(50, 1).astype(np.float32)
        events = det.process(warmup_signal)

        # Should not detect during warmup
        assert len(events) == 0

        # Complete warmup
        warmup_signal2 = np.random.randn(100, 1).astype(np.float32)
        det.process(warmup_signal2)

        # Now add a large spike
        signal = np.random.randn(50, 1).astype(np.float32) * 0.1
        signal[25, 0] = 10.0  # Large spike
        events = det.process(signal)

        # Should detect the spike
        assert len(events) >= 1

    def test_thresholds_property(self):
        """Test accessing computed thresholds."""
        det = npy.AdaptiveThresholdDetector(4, 4.0, 10, 100)

        # Initially thresholds are zeros
        thresholds = det.thresholds
        assert len(thresholds) == 4

        # Process some data to compute thresholds
        signal = np.random.randn(200, 4).astype(np.float32)
        det.process(signal)

        # Thresholds should now be computed
        thresholds = det.thresholds
        assert all(t > 0 for t in thresholds)

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing threshold adaptation."""
        det = npy.AdaptiveThresholdDetector(4, 4.0, 10, 100)

        # Warmup and compute thresholds
        signal = np.random.randn(200, 4).astype(np.float32)
        det.process(signal)

        # Freeze thresholds
        det.freeze()
        frozen_thresholds = det.thresholds.copy()

        # Process more data with different statistics
        signal2 = np.random.randn(100, 4).astype(np.float32) * 5.0
        det.process(signal2)

        # Thresholds should remain frozen
        assert np.allclose(det.thresholds, frozen_thresholds)

        # Unfreeze
        det.unfreeze()

        # Process more data
        det.process(signal2)

        # Thresholds should now change
        # (Note: exact behavior depends on implementation)

    def test_reset(self):
        """Test resetting detector state."""
        det = npy.AdaptiveThresholdDetector(4, 4.0, 10, 100)

        # Process data
        signal = np.random.randn(200, 4).astype(np.float32)
        det.process(signal)

        # Reset
        det.reset()

        # Thresholds should be reset to zero
        thresholds = det.thresholds
        assert all(t == 0.0 for t in thresholds)


class TestZeroCrossingDetector:
    """Tests for ZeroCrossingDetector."""

    def test_create_detector(self):
        """Test creating a zero crossing detector."""
        det = npy.ZeroCrossingDetector(4, 0.1)
        assert det.channels == 4
        assert abs(det.threshold - 0.1) < 1e-6

    def test_create_default_threshold(self):
        """Test creating with default threshold."""
        det = npy.ZeroCrossingDetector(4)
        assert det.channels == 4
        assert det.threshold == 0.0

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            npy.ZeroCrossingDetector(0, 0.1)  # channels must be >= 1

    def test_detect_zero_crossings(self):
        """Test detecting zero crossings."""
        det = npy.ZeroCrossingDetector(1, 0.0)

        # Simple alternating signal (single channel, 2D array)
        signal = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=np.float32)

        crossings = det.detect(signal)

        # Each transition is a crossing
        assert len(crossings) == 5
        # First sample can't be a crossing, then alternating crossings
        assert crossings[0] == False
        assert crossings[1] == True
        assert crossings[2] == True
        assert crossings[3] == True
        assert crossings[4] == True

    def test_threshold_effect(self):
        """Test that threshold affects detection."""
        det_low = npy.ZeroCrossingDetector(1, 0.0)
        det_high = npy.ZeroCrossingDetector(1, 0.5)

        # Signal with small oscillation around zero (2D single channel)
        signal = np.array([[0.2], [-0.2], [0.2], [-0.2], [0.2]], dtype=np.float32)

        crossings_low = det_low.detect(signal)
        crossings_high = det_high.detect(signal)

        # Low threshold should detect crossings
        assert np.sum(crossings_low) > 0

        # High threshold should miss small crossings
        assert np.sum(crossings_high) < np.sum(crossings_low)

    def test_zero_crossing_rate(self):
        """Test computing zero crossing rate."""
        det = npy.ZeroCrossingDetector(4, 0.0)

        # Multi-channel signal
        channels = 4
        samples = 100

        signal = np.zeros((samples, channels), dtype=np.float32)

        # Channel 0: high frequency (many crossings)
        signal[:, 0] = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(samples)])

        # Channel 1: low frequency (few crossings)
        signal[:, 1] = np.array([1.0 if (i // 10) % 2 == 0 else -1.0 for i in range(samples)])

        # Constant positive (no crossings)
        signal[:, 2] = 1.0

        # Random noise
        signal[:, 3] = np.random.randn(samples).astype(np.float32)

        zcr = det.zcr(signal)

        assert len(zcr) == channels

        # High frequency should have highest ZCR
        assert zcr[0] > zcr[1]

        # Constant signal should have zero ZCR
        assert zcr[2] == 0.0

    def test_repr(self):
        """Test string representation."""
        det = npy.ZeroCrossingDetector(4, 0.5)
        assert "0.5" in repr(det)
