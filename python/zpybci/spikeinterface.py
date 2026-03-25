"""SpikeInterface integration for Zerostone spike sorter.

Provides a BaseSorter subclass that wraps zpybci.sort_multichannel(),
enabling Zerostone to participate in SpikeInterface comparison benchmarks
and standardized spike sorting workflows.

Usage::

    import spikeinterface.core as si
    from zpybci.spikeinterface import ZerostoneSorter, run_zerostone

    recording = si.load_extractor("path/to/recording")
    sorting = run_zerostone(recording, threshold=5.0)

    # Or via the sorter class directly:
    sorting = ZerostoneSorter.run(recording, output_folder="zs_output")

Requires ``spikeinterface >= 0.100`` as an optional dependency.
"""

import json
from pathlib import Path

import numpy as np

import zpybci as zbci

try:
    import spikeinterface
    from spikeinterface.core import NumpySorting
    from spikeinterface.sorters import BaseSorter

    HAS_SPIKEINTERFACE = True
except ImportError:
    HAS_SPIKEINTERFACE = False


def _check_spikeinterface():
    """Raise ImportError with install instructions if SpikeInterface is missing."""
    if not HAS_SPIKEINTERFACE:
        raise ImportError(
            "spikeinterface is required for zpybci.spikeinterface. "
            "Install it with: pip install spikeinterface"
        )


def _estimate_noise_multichannel(data):
    """Estimate per-channel noise using MAD (median absolute deviation).

    Parameters
    ----------
    data : np.ndarray
        2D float64 array of shape ``(n_samples, n_channels)``.

    Returns
    -------
    np.ndarray
        1D float64 array of per-channel noise estimates.
    """
    # MAD noise: sigma = median(|x|) / 0.6745
    return np.median(np.abs(data), axis=0) / 0.6745


def _recording_to_numpy(recording):
    """Extract contiguous float64 array from a SpikeInterface recording.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording extractor.

    Returns
    -------
    np.ndarray
        2D float64 array of shape ``(n_samples, n_channels)``.
    float
        Sampling frequency in Hz.
    """
    traces = recording.get_traces(return_scaled=True)
    return np.ascontiguousarray(traces, dtype=np.float64), recording.get_sampling_frequency()


_SUPPORTED_CHANNELS = {4, 8, 16, 32, 64, 128}

_DEFAULT_PARAMS = {
    "threshold": 5.0,
    "refractory": 15,
    "spatial_radius": 75.0,
    "temporal_radius": 5,
    "align_half_window": 5,
    "pre_samples": 16,
    "cluster_threshold": 5.0,
    "cluster_max_count": 1000,
    "whitening_epsilon": 1e-6,
    "probe_pitch": 25.0,
    "detection_mode": "amplitude",
    "sneo_smooth_window": 3,
    "ccg_merge": False,
}

_PARAM_DESCRIPTIONS = {
    "threshold": "Detection threshold in MAD units.",
    "refractory": "Minimum samples between detections per channel.",
    "spatial_radius": "Deduplication radius in micrometers.",
    "temporal_radius": "Deduplication radius in samples.",
    "align_half_window": "Half-window for fine peak alignment.",
    "pre_samples": "Samples before peak in extracted waveforms.",
    "cluster_threshold": "Distance threshold for creating new clusters.",
    "cluster_max_count": "Maximum observation count per cluster centroid.",
    "whitening_epsilon": "Regularization for whitening eigenvalues.",
    "probe_pitch": "Inter-electrode pitch in micrometers (for linear probe fallback).",
    "detection_mode": "Detection mode: 'amplitude', 'neo', or 'sneo'.",
    "sneo_smooth_window": "SNEO smoothing window (only for detection_mode='sneo').",
    "ccg_merge": "Enable CCG-based cluster merging to reduce over-splitting.",
}


def _validate_params(params):
    """Validate sorter parameters, raising ValueError for invalid values.

    Parameters
    ----------
    params : dict
        Sorter parameters.

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    if params["threshold"] <= 0:
        raise ValueError(f"threshold must be positive, got {params['threshold']}")
    if params["refractory"] < 1:
        raise ValueError(f"refractory must be >= 1, got {params['refractory']}")
    if params["spatial_radius"] <= 0:
        raise ValueError(
            f"spatial_radius must be positive, got {params['spatial_radius']}"
        )
    if params["cluster_threshold"] <= 0:
        raise ValueError(
            f"cluster_threshold must be positive, got {params['cluster_threshold']}"
        )
    if params["whitening_epsilon"] <= 0:
        raise ValueError(
            f"whitening_epsilon must be positive, got {params['whitening_epsilon']}"
        )
    if params["probe_pitch"] <= 0:
        raise ValueError(
            f"probe_pitch must be positive, got {params['probe_pitch']}"
        )


def _sort_recording(recording, params):
    """Run Zerostone sorting on a SpikeInterface recording.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording extractor.
    params : dict
        Sorter parameters (validated).

    Returns
    -------
    NumpySorting
        SpikeInterface sorting object with spike times and labels.
    dict
        Raw sort result from zpybci.sort_multichannel().
    """
    data, fs = _recording_to_numpy(recording)
    n_samples, n_channels = data.shape

    if n_channels not in _SUPPORTED_CHANNELS:
        raise ValueError(
            f"n_channels={n_channels} not supported. "
            f"Supported: {sorted(_SUPPORTED_CHANNELS)}"
        )

    # Build probe geometry -- use linear layout as fallback
    probe = zbci.ProbeLayout.linear(n_channels, params["probe_pitch"])

    # Step 1: Run the full sorting pipeline (detect + whiten + cluster)
    sort_result = zbci.sort_multichannel(
        data,
        probe,
        threshold=params["threshold"],
        refractory=params["refractory"],
        spatial_radius=params["spatial_radius"],
        temporal_radius=params["temporal_radius"],
        align_half_window=params["align_half_window"],
        pre_samples=params["pre_samples"],
        cluster_threshold=params["cluster_threshold"],
        cluster_max_count=params["cluster_max_count"],
        whitening_epsilon=params["whitening_epsilon"],
        detection_mode=params["detection_mode"],
        sneo_smooth_window=params["sneo_smooth_window"],
        ccg_merge=params["ccg_merge"],
    )

    n_spikes = sort_result["n_spikes"]
    labels = np.asarray(sort_result["labels"])

    if n_spikes == 0:
        sorting = NumpySorting.from_unit_dict({}, sampling_frequency=fs)
        return sorting, sort_result

    # Step 2: Extract spike times directly from sort result
    spike_times = np.asarray(sort_result["spike_times"], dtype=np.int64)

    # Step 3: Build NumpySorting from sample indices and unit labels
    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[spike_times],
        labels_list=[labels],
        sampling_frequency=fs,
    )

    return sorting, sort_result


# -- BaseSorter subclass (only defined when SpikeInterface is available) -----

if HAS_SPIKEINTERFACE:

    class ZerostoneSorter(BaseSorter):
        """SpikeInterface BaseSorter wrapper for Zerostone spike sorter.

        Zerostone is a real-time, deterministic spike sorter designed for
        embedded and closed-loop BCI applications. It supports 2-64 channels
        and runs entirely on CPU with no GPU dependency.

        The sorting pipeline:
        1. MAD noise estimation
        2. Spatial whitening (ZCA)
        3. Threshold detection with refractory period
        4. Cross-channel deduplication
        5. Peak alignment
        6. Single-channel waveform extraction
        7. PCA feature reduction (3 components)
        8. Online k-means clustering
        9. Quality metrics (SNR, ISI violations)
        """

        sorter_name = "zerostone"
        requires_locations = False
        compatible_with_parallel = {
            "loky": False,
            "multiprocessing": False,
            "threading": False,
        }
        requires_binary_data = False

        _default_params = dict(_DEFAULT_PARAMS)
        _params_description = dict(_PARAM_DESCRIPTIONS)

        sorter_description = (
            "Zerostone: real-time deterministic spike sorter for embedded BCI. "
            "CPU-only, 2-64 channels, sub-ms latency."
        )

        installation_mesg = (
            "pip install zpybci  # Zerostone Python bindings"
        )

        @classmethod
        def is_installed(cls):
            """Check if zpybci is installed."""
            try:
                import zpybci  # noqa: F401
                return True
            except ImportError:
                return False

        @classmethod
        def get_sorter_version(cls):
            """Return zpybci version string."""
            return getattr(zbci, "__version__", "unknown")

        @classmethod
        def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
            """Validate recording and save parameters to output folder.

            Called by BaseSorter before _run_from_folder. Saves the recording
            info and parameters as JSON for reproducibility.
            """
            sorter_output_folder = Path(sorter_output_folder)
            sorter_output_folder.mkdir(parents=True, exist_ok=True)

            n_channels = recording.get_num_channels()
            if n_channels not in _SUPPORTED_CHANNELS:
                raise ValueError(
                    f"n_channels={n_channels} not supported by Zerostone. "
                    f"Supported: {sorted(_SUPPORTED_CHANNELS)}"
                )

            _validate_params(params)

            # Save params for _run_from_folder
            info = {
                "params": params,
                "n_channels": n_channels,
                "sampling_frequency": recording.get_sampling_frequency(),
                "n_samples": recording.get_num_samples(),
            }
            with open(sorter_output_folder / "zerostone_params.json", "w") as f:
                json.dump(info, f, indent=2)

        @classmethod
        def _run_from_folder(cls, sorter_output_folder, params, verbose):
            """Run sorting. Called by BaseSorter after _setup_recording."""
            sorter_output_folder = Path(sorter_output_folder)

            # Load the recording that BaseSorter cached
            recording = cls.load_recording_from_folder(
                sorter_output_folder.parent, with_warnings=False
            )

            sorting, sort_result = _sort_recording(recording, params)

            # Save results
            sorting.save(folder=sorter_output_folder / "sorting")

            # Save raw metrics for inspection
            metrics = {
                "n_spikes": int(sort_result["n_spikes"]),
                "n_clusters": int(sort_result["n_clusters"]),
                "clusters": [
                    {
                        "count": int(c["count"]),
                        "snr": float(c["snr"]),
                        "isi_violation_rate": float(c["isi_violation_rate"]),
                    }
                    for c in sort_result["clusters"]
                ],
            }
            with open(sorter_output_folder / "zerostone_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        @classmethod
        def _get_result_from_folder(cls, sorter_output_folder):
            """Load sorting result from output folder."""
            sorter_output_folder = Path(sorter_output_folder)
            result_folder = sorter_output_folder / "sorting"
            if not result_folder.exists():
                raise FileNotFoundError(
                    f"No sorting results found in {sorter_output_folder}"
                )
            return NumpySorting.load(result_folder)


def run_zerostone(recording, **kwargs):
    """Convenience function to run Zerostone on a SpikeInterface recording.

    This is the simplest way to use Zerostone with SpikeInterface. It
    bypasses the folder-based BaseSorter workflow and returns results
    directly in memory.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording extractor.
    **kwargs
        Sorter parameters. See ``ZerostoneSorter._default_params`` for
        defaults and ``ZerostoneSorter._params_description`` for docs.

    Returns
    -------
    NumpySorting
        SpikeInterface sorting object with spike times and unit labels.

    Raises
    ------
    ImportError
        If spikeinterface is not installed.
    ValueError
        If recording has unsupported channel count or invalid parameters.

    Examples
    --------
    >>> import spikeinterface.core as si  # doctest: +SKIP
    >>> from zpybci.spikeinterface import run_zerostone  # doctest: +SKIP
    >>> rec = si.generate_recording(num_channels=4, sampling_frequency=30000.0)
    >>> sorting = run_zerostone(rec, threshold=5.0)  # doctest: +SKIP
    """
    _check_spikeinterface()

    params = dict(_DEFAULT_PARAMS)
    for key, val in kwargs.items():
        if key not in params:
            raise ValueError(
                f"Unknown parameter '{key}'. "
                f"Valid parameters: {sorted(params.keys())}"
            )
        params[key] = val

    _validate_params(params)

    sorting, _ = _sort_recording(recording, params)
    return sorting
