"""Tests for NotchFilter powerline noise removal."""

import numpy as np
import pytest

import npyci as npy


def test_import():
    assert hasattr(npy, "NotchFilter")


def test_powerline_60hz_creation():
    f = npy.NotchFilter.powerline_60hz(1000.0, channels=8)
    assert f.channels == 8
    assert f.sample_rate == 1000.0


def test_powerline_50hz_creation():
    f = npy.NotchFilter.powerline_50hz(1000.0, channels=8)
    assert f.channels == 8
    assert f.sample_rate == 1000.0


def test_custom_creation():
    f = npy.NotchFilter.custom(1000.0, 8, [60.0])
    assert f.channels == 8
    assert f.sample_rate == 1000.0


def test_invalid_channels():
    with pytest.raises((ValueError, Exception)):
        npy.NotchFilter.powerline_60hz(1000.0, channels=22)


def test_invalid_sample_rate():
    with pytest.raises((ValueError, Exception)):
        npy.NotchFilter.powerline_60hz(0.0, channels=8)


def test_process_shape():
    f = npy.NotchFilter.powerline_60hz(1000.0, channels=8)
    data = np.random.randn(500, 8).astype(np.float32)
    out = f.process(data)
    assert out.shape == (500, 8)
    assert out.dtype == np.float32


def test_60hz_attenuation():
    """60 Hz sine fed through 60 Hz notch should be attenuated >40 dB."""
    sample_rate = 1000.0
    f = npy.NotchFilter.powerline_60hz(sample_rate, channels=1)

    n_settle = 1000
    n_measure = 100
    t = np.arange(n_settle + n_measure, dtype=np.float32) / sample_rate
    sine = np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)

    # Process as (n_samples, 1)
    data = sine.reshape(-1, 1)
    out = f.process(data)

    # Measure peak amplitude after settling
    peak = float(np.abs(out[n_settle:]).max())
    assert peak < 0.01, f"Expected >40 dB attenuation at 60 Hz, got peak={peak:.6f}"


def test_passband_preserved():
    """10 Hz sine through 60 Hz notch should pass with <3 dB loss."""
    sample_rate = 1000.0
    f = npy.NotchFilter.powerline_60hz(sample_rate, channels=1)

    n_settle = 500
    n_measure = 100
    t = np.arange(n_settle + n_measure, dtype=np.float32) / sample_rate
    sine = np.sin(2.0 * np.pi * 10.0 * t).astype(np.float32)

    data = sine.reshape(-1, 1)
    out = f.process(data)

    peak = float(np.abs(out[n_settle:]).max())
    assert peak > 0.9, f"Expected >0.9 passband amplitude at 10 Hz, got peak={peak:.4f}"


def test_batch_processing():
    """2D array input should return float32 array of same shape."""
    f = npy.NotchFilter.powerline_60hz(1000.0, channels=8)
    data = np.random.randn(1000, 8).astype(np.float32)
    out = f.process(data)
    assert out.shape == data.shape
    assert out.dtype == np.float32


def test_reset():
    """After reset, re-processing same signal should produce identical outputs."""
    sample_rate = 1000.0
    f = npy.NotchFilter.powerline_60hz(sample_rate, channels=8)

    t = np.arange(200, dtype=np.float32) / sample_rate
    sine = np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)
    data = np.tile(sine[:, None], (1, 8))

    out1 = f.process(data)
    f.reset()
    out2 = f.process(data)

    np.testing.assert_array_equal(out1, out2)


def test_repr():
    f = npy.NotchFilter.powerline_60hz(250.0, channels=8)
    r = repr(f)
    assert "NotchFilter" in r
    assert "250" in r
    assert "8" in r
