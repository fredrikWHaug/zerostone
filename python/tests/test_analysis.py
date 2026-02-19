"""Tests for analysis module bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestEnvelopeFollower:
    """Tests for EnvelopeFollower."""

    def test_create_envelope_follower(self):
        """Test creating an envelope follower."""
        env = zbci.EnvelopeFollower(
            channels=8,
            sample_rate=250.0,
            attack_time=0.010,
            release_time=0.100,
            rectification='absolute'
        )
        assert env.channels == 8
        assert env.sample_rate == 250.0
        assert abs(env.attack_time - 0.010) < 1e-5
        assert abs(env.release_time - 0.100) < 1e-5
        assert env.rectification == 'absolute'

    def test_create_symmetric(self):
        """Test creating with symmetric smoothing."""
        env = zbci.EnvelopeFollower.symmetric(
            channels=4,
            sample_rate=250.0,
            smoothing_time=0.050,
            rectification='squared'
        )
        assert env.attack_time == env.release_time
        assert env.rectification == 'squared'

    def test_invalid_channels(self):
        """Test that unsupported channel count raises error."""
        with pytest.raises(ValueError):
            zbci.EnvelopeFollower(channels=3, sample_rate=250.0, attack_time=0.01, release_time=0.1)

    def test_invalid_rectification(self):
        """Test that invalid rectification raises error."""
        with pytest.raises(ValueError):
            zbci.EnvelopeFollower(channels=4, sample_rate=250.0, attack_time=0.01, release_time=0.1, rectification='invalid')

    def test_process_single_sample(self):
        """Test processing a single sample."""
        env = zbci.EnvelopeFollower(channels=4, sample_rate=250.0, attack_time=0.01, release_time=0.1)

        sample = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        envelope = env.process(sample)

        assert envelope.shape == (4,)
        assert np.all(envelope >= 0)  # Envelope should be non-negative

    def test_process_block(self):
        """Test processing a block of samples."""
        env = zbci.EnvelopeFollower(channels=4, sample_rate=250.0, attack_time=0.01, release_time=0.1)

        block = np.random.randn(100, 4).astype(np.float32)
        envelope = env.process_block(block)

        assert envelope.shape == (100, 4)
        assert np.all(envelope >= 0)

    def test_channel_mismatch(self):
        """Test that channel mismatch raises error."""
        env = zbci.EnvelopeFollower(channels=8, sample_rate=250.0, attack_time=0.01, release_time=0.1)

        sample = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # Wrong channels
        with pytest.raises(ValueError):
            env.process(sample)

    def test_rectification_absolute(self):
        """Test absolute value rectification."""
        env = zbci.EnvelopeFollower(channels=1, sample_rate=1000.0, attack_time=0.001, release_time=0.001, rectification='absolute')

        # Negative input should produce positive envelope
        envelope = env.process(np.array([-1.0], dtype=np.float32))
        assert envelope[0] > 0

    def test_rectification_squared(self):
        """Test squared rectification."""
        env = zbci.EnvelopeFollower(channels=1, sample_rate=1000.0, attack_time=0.001, release_time=0.001, rectification='squared')

        env.reset()
        # With very fast attack, envelope should be close to input²
        envelope = env.process(np.array([2.0], dtype=np.float32))
        assert envelope[0] > 3.0  # Should be close to 4.0

    def test_envelope_tracks_amplitude(self):
        """Test that envelope follows signal amplitude."""
        env = zbci.EnvelopeFollower.symmetric(channels=1, sample_rate=250.0, smoothing_time=0.020, rectification='absolute')

        # Feed increasing amplitude
        for i in range(1, 11):
            env.process(np.array([i * 0.1], dtype=np.float32))

        # Envelope should have increased
        current = env.current()
        assert current[0] > 0.5

    def test_attack_release_asymmetry(self):
        """Test asymmetric attack/release behavior."""
        env = zbci.EnvelopeFollower(channels=1, sample_rate=1000.0, attack_time=0.001, release_time=0.100, rectification='absolute')

        # Fast attack
        for _ in range(10):
            env.process(np.array([1.0], dtype=np.float32))
        peak = env.current()[0]

        # Slow release
        for _ in range(10):
            env.process(np.array([0.0], dtype=np.float32))
        after_release = env.current()[0]

        # Should retain most envelope due to slow release
        assert after_release > peak * 0.5

    def test_reset(self):
        """Test resetting envelope."""
        env = zbci.EnvelopeFollower(channels=4, sample_rate=250.0, attack_time=0.01, release_time=0.1)

        # Build up envelope
        for _ in range(100):
            env.process(np.array([1.0] * 4, dtype=np.float32))

        current = env.current()
        assert current[0] > 0.5

        # Reset
        env.reset()

        current = env.current()
        assert current[0] == 0.0

    def test_supported_channel_counts(self):
        """Test all supported channel counts."""
        for channels in [1, 4, 8, 16, 32, 64]:
            env = zbci.EnvelopeFollower(channels=channels, sample_rate=250.0, attack_time=0.01, release_time=0.1)
            sample = np.random.randn(channels).astype(np.float32)
            envelope = env.process(sample)
            assert envelope.shape == (channels,)

    def test_repr(self):
        """Test string representation."""
        env = zbci.EnvelopeFollower(channels=8, sample_rate=250.0, attack_time=0.01, release_time=0.1, rectification='absolute')
        assert "8" in repr(env)
        assert "250" in repr(env)
        assert "absolute" in repr(env)


class TestWindowedRms:
    """Tests for WindowedRms."""

    def test_create_windowed_rms(self):
        """Test creating a windowed RMS tracker."""
        rms = zbci.WindowedRms(channels=8, window_size=64)
        assert rms.channels == 8
        assert rms.window_size == 64
        assert not rms.is_ready

    def test_invalid_params(self):
        """Test that invalid parameters raise error."""
        with pytest.raises(ValueError):
            zbci.WindowedRms(channels=3, window_size=64)  # Invalid channels
        with pytest.raises(ValueError):
            zbci.WindowedRms(channels=8, window_size=50)  # Invalid window

    def test_supported_configurations(self):
        """Test all supported channel/window combinations."""
        for channels in [1, 4, 8, 16, 32, 64]:
            for window in [16, 32, 64, 128]:
                rms = zbci.WindowedRms(channels=channels, window_size=window)
                assert rms.channels == channels
                assert rms.window_size == window

    def test_process_sample(self):
        """Test processing single samples."""
        rms = zbci.WindowedRms(channels=4, window_size=16)

        for i in range(16):
            sample = np.array([1.0] * 4, dtype=np.float32)
            rms.process(sample)

        assert rms.is_ready
        assert rms.count == 16

    def test_rms_before_ready(self):
        """Test that RMS returns None before window is full."""
        rms = zbci.WindowedRms(channels=4, window_size=16)

        for i in range(15):
            rms.process(np.array([1.0] * 4, dtype=np.float32))
            assert rms.rms() is None

        rms.process(np.array([1.0] * 4, dtype=np.float32))
        assert rms.rms() is not None

    def test_rms_constant_signal(self):
        """Test RMS of constant signal."""
        rms = zbci.WindowedRms(channels=1, window_size=16)

        # Feed constant value
        for _ in range(16):
            rms.process(np.array([2.0], dtype=np.float32))

        # RMS should equal the constant value
        rms_vals = rms.rms()
        assert abs(rms_vals[0] - 2.0) < 1e-5

    def test_power_computation(self):
        """Test power computation."""
        rms = zbci.WindowedRms(channels=1, window_size=16)

        # Feed constant value
        for _ in range(16):
            rms.process(np.array([3.0], dtype=np.float32))

        rms_vals = rms.rms()
        power_vals = rms.power()

        # Power = RMS²
        assert abs(power_vals[0] - rms_vals[0] ** 2) < 1e-5

    def test_multi_channel(self):
        """Test multi-channel RMS."""
        rms = zbci.WindowedRms(channels=4, window_size=16)

        # Feed different values per channel
        for _ in range(16):
            rms.process(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        rms_vals = rms.rms()
        np.testing.assert_array_almost_equal(rms_vals, [1.0, 2.0, 3.0, 4.0], decimal=5)

    def test_sliding_window(self):
        """Test sliding window behavior."""
        rms = zbci.WindowedRms(channels=1, window_size=16)

        # Fill window with known values [1, 1, 1, ..., 1] (16 ones)
        for _ in range(16):
            rms.process(np.array([1.0], dtype=np.float32))

        # RMS of constant 1.0 should be 1.0
        rms_val = rms.rms()[0]
        assert abs(rms_val - 1.0) < 1e-5

        # Add 2.0, window slides - now [1.0]*15 + [2.0]
        rms.process(np.array([2.0], dtype=np.float32))

        # RMS = sqrt((15*1 + 4) / 16) = sqrt(19/16)
        rms_val = rms.rms()[0]
        expected = np.sqrt(19.0 / 16.0)
        assert abs(rms_val - expected) < 1e-5

    def test_reset(self):
        """Test resetting tracker."""
        rms = zbci.WindowedRms(channels=4, window_size=16)

        # Build up state
        for _ in range(16):
            rms.process(np.array([5.0] * 4, dtype=np.float32))

        assert rms.is_ready
        assert rms.rms() is not None

        # Reset
        rms.reset()

        assert not rms.is_ready
        assert rms.count == 0
        assert rms.rms() is None

    def test_channel_mismatch(self):
        """Test that channel mismatch raises error."""
        rms = zbci.WindowedRms(channels=8, window_size=16)

        sample = np.array([1.0] * 4, dtype=np.float32)  # Wrong channels
        with pytest.raises(ValueError):
            rms.process(sample)

    def test_repr(self):
        """Test string representation."""
        rms = zbci.WindowedRms(channels=8, window_size=64)
        assert "8" in repr(rms)
        assert "64" in repr(rms)


class TestAnalysisIntegration:
    """Integration tests for analysis components."""

    def test_envelope_to_rms_pipeline(self):
        """Test envelope followed by RMS computation."""
        env = zbci.EnvelopeFollower(channels=4, sample_rate=250.0, attack_time=0.01, release_time=0.1)
        rms = zbci.WindowedRms(channels=4, window_size=32)

        # Process signal through envelope then RMS
        for _ in range(100):
            sample = np.random.randn(4).astype(np.float32)
            envelope = env.process(sample)
            rms.process(envelope)

        assert rms.is_ready
        rms_vals = rms.rms()
        assert np.all(rms_vals >= 0)

    def test_power_tracking(self):
        """Test power tracking over time."""
        rms = zbci.WindowedRms(channels=8, window_size=64)

        # Simulate varying power signal
        powers = []
        for i in range(200):
            amplitude = 1.0 + np.sin(i / 20.0)  # Modulated amplitude
            sample = (np.random.randn(8) * amplitude).astype(np.float32)
            rms.process(sample)

            if rms.is_ready:
                powers.append(rms.power().copy())

        # Power should vary over time
        powers = np.array(powers)
        assert powers.std(axis=0).mean() > 0.01  # Should have variance
