"""Tests for zpybci.spikeinterface module."""

import numpy as np
import pytest

import zpybci as zbci


# ---------------------------------------------------------------------------
# Tests that work without SpikeInterface installed
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Test module-level import behavior."""

    def test_import_module(self):
        """Module imports without error even if spikeinterface is absent."""
        from zpybci import spikeinterface  # noqa: F401

    def test_has_spikeinterface_flag(self):
        """HAS_SPIKEINTERFACE flag is a bool."""
        from zpybci.spikeinterface import HAS_SPIKEINTERFACE

        assert isinstance(HAS_SPIKEINTERFACE, bool)

    def test_run_zerostone_importable(self):
        """run_zerostone function is always importable."""
        from zpybci.spikeinterface import run_zerostone

        assert callable(run_zerostone)

    def test_default_params(self):
        """Default parameters dict is available."""
        from zpybci.spikeinterface import _DEFAULT_PARAMS

        assert isinstance(_DEFAULT_PARAMS, dict)
        assert "threshold" in _DEFAULT_PARAMS
        assert "refractory" in _DEFAULT_PARAMS
        assert "probe_pitch" in _DEFAULT_PARAMS

    def test_supported_channels(self):
        """Supported channel counts are defined."""
        from zpybci.spikeinterface import _SUPPORTED_CHANNELS

        assert _SUPPORTED_CHANNELS == {4, 8, 16, 32, 64, 96, 128}


class TestParameterValidation:
    """Test parameter validation without SpikeInterface."""

    def test_valid_params(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        # Should not raise
        _validate_params(dict(_DEFAULT_PARAMS))

    def test_negative_threshold(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["threshold"] = -1.0
        with pytest.raises(ValueError, match="threshold must be positive"):
            _validate_params(params)

    def test_zero_threshold(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["threshold"] = 0.0
        with pytest.raises(ValueError, match="threshold must be positive"):
            _validate_params(params)

    def test_zero_refractory(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["refractory"] = 0
        with pytest.raises(ValueError, match="refractory must be >= 1"):
            _validate_params(params)

    def test_negative_spatial_radius(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["spatial_radius"] = -10.0
        with pytest.raises(ValueError, match="spatial_radius must be positive"):
            _validate_params(params)

    def test_negative_cluster_threshold(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["cluster_threshold"] = -1.0
        with pytest.raises(ValueError, match="cluster_threshold must be positive"):
            _validate_params(params)

    def test_negative_whitening_epsilon(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["whitening_epsilon"] = -0.001
        with pytest.raises(ValueError, match="whitening_epsilon must be positive"):
            _validate_params(params)

    def test_negative_probe_pitch(self):
        from zpybci.spikeinterface import _validate_params, _DEFAULT_PARAMS

        params = dict(_DEFAULT_PARAMS)
        params["probe_pitch"] = 0.0
        with pytest.raises(ValueError, match="probe_pitch must be positive"):
            _validate_params(params)


class TestNoiseEstimation:
    """Test the numpy-based MAD noise estimator."""

    def test_noise_shape(self):
        from zpybci.spikeinterface import _estimate_noise_multichannel

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10000, 4))
        noise = _estimate_noise_multichannel(data)
        assert noise.shape == (4,)

    def test_noise_positive(self):
        from zpybci.spikeinterface import _estimate_noise_multichannel

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10000, 4))
        noise = _estimate_noise_multichannel(data)
        assert np.all(noise > 0)

    def test_noise_close_to_std(self):
        """For Gaussian noise, MAD estimate should be close to true std."""
        from zpybci.spikeinterface import _estimate_noise_multichannel

        rng = np.random.default_rng(42)
        data = rng.standard_normal((100000, 2)) * 3.0
        noise = _estimate_noise_multichannel(data)
        np.testing.assert_allclose(noise, 3.0, rtol=0.05)

    def test_different_scales(self):
        from zpybci.spikeinterface import _estimate_noise_multichannel

        rng = np.random.default_rng(42)
        data = np.column_stack([
            rng.standard_normal(50000) * 1.0,
            rng.standard_normal(50000) * 5.0,
        ])
        noise = _estimate_noise_multichannel(data)
        # Channel 1 should have ~5x the noise of channel 0
        assert noise[1] / noise[0] > 3.0


# ---------------------------------------------------------------------------
# Tests that require SpikeInterface
# ---------------------------------------------------------------------------

try:
    import spikeinterface as si
    import spikeinterface.core as si_core

    HAS_SI = True
except ImportError:
    HAS_SI = False

needs_si = pytest.mark.skipif(not HAS_SI, reason="spikeinterface not installed")


if HAS_SI:

    def _make_recording(n_channels=4, duration=1.0, fs=30000.0, seed=42):
        """Create a simple synthetic SpikeInterface recording."""
        return si_core.generate_recording(
            num_channels=n_channels,
            sampling_frequency=fs,
            durations=[duration],
            seed=seed,
        )

    def _make_recording_with_spikes(n_channels=4, duration=2.0, fs=30000.0, seed=42):
        """Create a recording with embedded spikes for sorting."""
        rng = np.random.default_rng(seed)
        n_samples = int(duration * fs)
        data = rng.standard_normal((n_samples, n_channels)) * 0.5

        # Insert large negative spikes on channel 0
        spike_window = 32
        pre = spike_window // 4
        t = np.arange(spike_window, dtype=np.float64)
        template = -15.0 * np.exp(-0.5 * ((t - pre) / 2.0) ** 2)

        spike_times = np.arange(1000, n_samples - spike_window, int(fs * 0.01))
        for st in spike_times:
            data[st - pre : st - pre + spike_window, 0] += template

        traces = np.ascontiguousarray(data, dtype=np.float32)
        recording = si_core.NumpyRecording(
            traces_list=[traces], sampling_frequency=fs
        )
        return recording, spike_times


@needs_si
class TestSpikeInterfaceFlag:
    """Verify flag is True when spikeinterface is installed."""

    def test_has_spikeinterface_true(self):
        from zpybci.spikeinterface import HAS_SPIKEINTERFACE

        assert HAS_SPIKEINTERFACE is True


@needs_si
class TestZerostoneSorterClass:
    """Test the ZerostoneSorter class attributes and methods."""

    def test_class_exists(self):
        from zpybci.spikeinterface import ZerostoneSorter

        assert ZerostoneSorter.sorter_name == "zerostone"

    def test_is_installed(self):
        from zpybci.spikeinterface import ZerostoneSorter

        assert ZerostoneSorter.is_installed() is True

    def test_get_version(self):
        from zpybci.spikeinterface import ZerostoneSorter

        version = ZerostoneSorter.get_sorter_version()
        assert isinstance(version, str)

    def test_default_params_present(self):
        from zpybci.spikeinterface import ZerostoneSorter

        params = ZerostoneSorter._default_params
        assert "threshold" in params
        assert "refractory" in params

    def test_requires_locations_false(self):
        from zpybci.spikeinterface import ZerostoneSorter

        assert ZerostoneSorter.requires_locations is False

    def test_no_parallel(self):
        from zpybci.spikeinterface import ZerostoneSorter

        for key, val in ZerostoneSorter.compatible_with_parallel.items():
            assert val is False


@needs_si
class TestRunZerostone:
    """Test the convenience run_zerostone() function."""

    def test_unknown_param_raises(self):
        from zpybci.spikeinterface import run_zerostone

        recording = _make_recording(n_channels=4, duration=0.5)
        with pytest.raises(ValueError, match="Unknown parameter"):
            run_zerostone(recording, bogus_param=42)

    def test_unsupported_channels(self):
        from zpybci.spikeinterface import run_zerostone

        recording = _make_recording(n_channels=3, duration=0.5)
        with pytest.raises(ValueError, match="not supported"):
            run_zerostone(recording)

    def test_noise_only_recording(self):
        """Sorting pure noise should return a valid (possibly empty) NumpySorting."""
        from zpybci.spikeinterface import run_zerostone

        recording = _make_recording(n_channels=4, duration=0.5)
        sorting = run_zerostone(recording, threshold=8.0)
        assert hasattr(sorting, "get_unit_ids")

    def test_recording_with_spikes(self):
        """Sorting a recording with embedded spikes should find units."""
        from zpybci.spikeinterface import run_zerostone

        recording, spike_times = _make_recording_with_spikes(
            n_channels=4, duration=2.0
        )
        sorting = run_zerostone(recording, threshold=4.0)
        assert hasattr(sorting, "get_unit_ids")
        # Should detect at least some spikes
        total_spikes = sum(
            len(sorting.get_unit_spike_train(uid))
            for uid in sorting.get_unit_ids()
        )
        assert total_spikes > 0, "Expected at least some spikes to be detected"

    def test_custom_params(self):
        """Custom parameters are accepted and do not crash."""
        from zpybci.spikeinterface import run_zerostone

        recording = _make_recording(n_channels=4, duration=0.5)
        sorting = run_zerostone(
            recording,
            threshold=6.0,
            refractory=20,
            cluster_threshold=2.5,
        )
        assert hasattr(sorting, "get_unit_ids")

    def test_four_channel_recording(self):
        """Minimum supported channel count (4) works."""
        from zpybci.spikeinterface import run_zerostone

        recording = _make_recording(n_channels=4, duration=0.5)
        sorting = run_zerostone(recording, threshold=8.0)
        assert hasattr(sorting, "get_unit_ids")


@needs_si
class TestRecordingToNumpy:
    """Test the recording-to-numpy conversion helper."""

    def test_shape_and_dtype(self):
        from zpybci.spikeinterface import _recording_to_numpy

        recording = _make_recording(n_channels=4, duration=0.5, fs=30000.0)
        data, fs = _recording_to_numpy(recording)
        assert data.dtype == np.float64
        assert data.ndim == 2
        assert data.shape[1] == 4
        assert fs == 30000.0

    def test_contiguous(self):
        from zpybci.spikeinterface import _recording_to_numpy

        recording = _make_recording(n_channels=4, duration=0.5)
        data, _ = _recording_to_numpy(recording)
        assert data.flags["C_CONTIGUOUS"]


@needs_si
class TestSortRecordingDirect:
    """Test the internal _sort_recording function."""

    def test_returns_sorting_and_result(self):
        from zpybci.spikeinterface import _sort_recording, _DEFAULT_PARAMS

        recording, _ = _make_recording_with_spikes(n_channels=4, duration=1.0)
        params = dict(_DEFAULT_PARAMS)
        sorting, result = _sort_recording(recording, params)
        assert hasattr(sorting, "get_unit_ids")
        assert "n_spikes" in result
        assert "n_clusters" in result
        assert "labels" in result
        assert "clusters" in result

    def test_unsupported_channels_raises(self):
        from zpybci.spikeinterface import _sort_recording, _DEFAULT_PARAMS

        recording = _make_recording(n_channels=3, duration=0.5)
        params = dict(_DEFAULT_PARAMS)
        with pytest.raises(ValueError, match="not supported"):
            _sort_recording(recording, params)
