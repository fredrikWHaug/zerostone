"""Tests for IirFilter Python bindings."""
import numpy as np
import pytest


def test_import():
    """Test that we can import npyci."""
    import npyci as npy
    assert hasattr(npy, '__version__')
    assert hasattr(npy, 'IirFilter')


def test_butterworth_lowpass_creation():
    """Test creating a Butterworth lowpass filter."""
    import npyci as npy

    lpf = npy.IirFilter.butterworth_lowpass(1000.0, 40.0)
    assert lpf.sample_rate == 1000.0


def test_butterworth_highpass_creation():
    """Test creating a Butterworth highpass filter."""
    import npyci as npy

    hpf = npy.IirFilter.butterworth_highpass(1000.0, 1.0)
    assert hpf.sample_rate == 1000.0


def test_butterworth_bandpass_creation():
    """Test creating a Butterworth bandpass filter."""
    import npyci as npy

    bpf = npy.IirFilter.butterworth_bandpass(1000.0, 8.0, 12.0)
    assert bpf.sample_rate == 1000.0


def test_invalid_cutoff():
    """Test that invalid cutoff frequencies raise errors."""
    import npyci as npy

    # Cutoff above Nyquist
    with pytest.raises(ValueError):
        npy.IirFilter.butterworth_lowpass(1000.0, 600.0)

    # Negative cutoff
    with pytest.raises(ValueError):
        npy.IirFilter.butterworth_lowpass(1000.0, -10.0)

    # Bandpass with invalid range
    with pytest.raises(ValueError):
        npy.IirFilter.butterworth_bandpass(1000.0, 50.0, 40.0)


def test_process_signal():
    """Test processing a signal through the filter."""
    import npyci as npy

    lpf = npy.IirFilter.butterworth_lowpass(1000.0, 40.0)

    # Create test signal
    signal = np.random.randn(1000).astype(np.float32)

    # Process
    filtered = lpf.process(signal)

    # Check output properties
    assert isinstance(filtered, np.ndarray)
    assert filtered.dtype == np.float32
    assert len(filtered) == len(signal)


def test_filter_reset():
    """Test that reset clears filter state."""
    import npyci as npy

    lpf = npy.IirFilter.butterworth_lowpass(1000.0, 40.0)

    # Process some data
    signal1 = np.ones(100, dtype=np.float32)
    out1 = lpf.process(signal1)

    # Reset
    lpf.reset()

    # Process same signal again
    out2 = lpf.process(signal1)

    # First few samples should match (same initial state after reset)
    assert np.allclose(out1[:10], out2[:10], rtol=1e-5)


def test_lowpass_attenuates_high_freq():
    """Test that lowpass filter actually attenuates high frequencies."""
    import npyci as npy

    sample_rate = 1000.0
    lpf = npy.IirFilter.butterworth_lowpass(sample_rate, 30.0)

    # Create signal with passband (10 Hz) and stopband (100 Hz) frequency components
    t = np.arange(0, 2, 1/sample_rate, dtype=np.float32)  # 2 seconds for settling
    passband = np.sin(2 * np.pi * 10 * t)
    stopband = np.sin(2 * np.pi * 100 * t)
    signal = passband + stopband

    # Filter
    filtered = lpf.process(signal)

    # Compare RMS of filtered signal (should be closer to passband amplitude)
    # Skip first second for filter settling
    signal_rms = np.sqrt(np.mean(signal[1000:] ** 2))
    filtered_rms = np.sqrt(np.mean(filtered[1000:] ** 2))
    passband_rms = np.sqrt(np.mean(passband[1000:] ** 2))

    # Filtered signal should have lower RMS than original (high freq removed)
    # and be closer to passband RMS
    assert filtered_rms < signal_rms * 0.9, "Filter should reduce overall RMS"
    assert abs(filtered_rms - passband_rms) < abs(signal_rms - passband_rms), \
        "Filtered signal should be closer to passband"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
