"""Synthetic multi-channel recording generation for spike sorting benchmarks.

Pure-numpy module that generates realistic multi-channel extracellular recordings
with known ground truth. Useful for benchmarking and regression-testing the
Zerostone spike sorter without external dependencies beyond numpy.

Waveform model: biphasic (negative-then-positive) extracellular action potentials
with per-unit random amplitude, width, and primary channel. Spatial falloff uses
exponential decay from the primary channel on a linear probe geometry.

Spike timing: Poisson process with refractory period enforcement.

Usage::

    from zpybci.synthetic import generate_recording

    rec = generate_recording(n_channels=32, duration_s=10.0, n_units=5, seed=0)
    data = rec["data"]            # (n_samples, n_channels) float64
    labels = rec["spike_labels"]  # ground-truth cluster labels
"""

import numpy as np


def generate_templates(n_units, n_channels, n_samples=61, seed=42):
    """Generate realistic biphasic spike templates.

    Each unit gets a primary channel with strongest amplitude,
    with exponential spatial falloff to neighboring channels on a
    linear probe geometry.

    Parameters
    ----------
    n_units : int
        Number of distinct neural units.
    n_channels : int
        Number of recording channels.
    n_samples : int
        Template length in samples (~2 ms at 30 kHz).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    templates : np.ndarray, shape (n_units, n_channels, n_samples)
        Spike waveform templates.
    primary_channels : np.ndarray, shape (n_units,)
        Primary (strongest) channel for each unit.
    """
    rng = np.random.default_rng(seed)
    templates = np.zeros((n_units, n_channels, n_samples), dtype=np.float64)

    # Time axis centered on trough
    t = np.arange(n_samples, dtype=np.float64)
    center = n_samples // 2

    # Assign each unit a primary channel (spread across probe)
    primary_channels = rng.integers(0, n_channels, size=n_units)

    for u in range(n_units):
        # Random waveform parameters
        amplitude = rng.uniform(3.0, 10.0)       # peak-to-peak in noise units
        trough_width = rng.uniform(0.2, 0.5)     # ms-scale width parameter
        peak_ratio = rng.uniform(0.3, 0.7)       # positive peak relative to trough

        # Biphasic waveform: negative Gaussian trough then positive peak
        sigma_trough = trough_width * n_samples / 4.0
        sigma_peak = sigma_trough * 1.3  # positive phase slightly wider

        trough = -np.exp(-0.5 * ((t - center) / sigma_trough) ** 2)
        peak = peak_ratio * np.exp(
            -0.5 * ((t - center - sigma_trough * 1.5) / sigma_peak) ** 2
        )
        waveform = amplitude * (trough + peak)

        # Spatial falloff: exponential decay from primary channel
        # decay_length controls how many channels away the signal extends
        decay_length = rng.uniform(1.5, 4.0)
        for ch in range(n_channels):
            dist = abs(ch - primary_channels[u])
            spatial_weight = np.exp(-dist / decay_length)
            templates[u, ch, :] = waveform * spatial_weight

    return templates, primary_channels


def generate_recording(
    n_channels=32,
    duration_s=60.0,
    sampling_rate=30000.0,
    n_units=5,
    firing_rate=5.0,
    noise_std=1.0,
    refractory_ms=1.0,
    seed=42,
):
    """Generate synthetic multi-channel recording with ground truth.

    Creates a realistic multi-channel extracellular recording by superimposing
    biphasic spike templates at Poisson-distributed times onto Gaussian noise.
    Output shape matches ``sort_multichannel()`` input format.

    Parameters
    ----------
    n_channels : int
        Number of recording channels.
    duration_s : float
        Recording duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    n_units : int
        Number of distinct neural units to simulate.
    firing_rate : float
        Mean firing rate per unit in Hz.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    refractory_ms : float
        Absolute refractory period in milliseconds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        data : np.ndarray, shape (n_samples, n_channels), float64
            Simulated recording.
        spike_times : list of np.ndarray
            Per-unit spike sample indices.
        spike_labels : np.ndarray
            All spike labels sorted by time.
        all_spike_times : np.ndarray
            All spike times sorted.
        templates : np.ndarray, shape (n_units, n_channels, n_samples)
            The waveform templates used.
        primary_channels : np.ndarray, shape (n_units,)
            Primary channel per unit.
        sampling_rate : float
        n_units : int
    """
    rng = np.random.default_rng(seed)
    n_total_samples = int(duration_s * sampling_rate)
    refractory_samples = int(refractory_ms * sampling_rate / 1000.0)

    # Template length: ~2ms at 30kHz = 61 samples (odd for symmetry)
    template_len = 61
    half_template = template_len // 2

    # Generate templates (use a sub-seed so templates are independent of spike times)
    templates, primary_channels = generate_templates(
        n_units, n_channels, n_samples=template_len, seed=seed + 1000
    )

    # Generate Poisson spike trains with refractory enforcement
    spike_times_per_unit = []
    for u in range(n_units):
        # Generate inter-spike intervals from exponential distribution
        mean_isi = sampling_rate / firing_rate
        # Over-generate to account for refractory rejection
        n_expected = int(duration_s * firing_rate * 1.5) + 100
        isis = rng.exponential(mean_isi, size=n_expected)

        # Enforce refractory period
        isis = np.maximum(isis, refractory_samples)

        # Cumulative sum to get spike times
        times = np.cumsum(isis).astype(np.int64)

        # Keep only spikes within valid range (template must fit)
        margin = half_template + 1
        valid = times[(times >= margin) & (times < n_total_samples - margin)]
        spike_times_per_unit.append(valid)

    # Build recording: start with noise
    data = rng.normal(0.0, noise_std, size=(n_total_samples, n_channels))

    # Superimpose spikes (scale by noise_std so template amplitudes are in
    # units of noise standard deviation, i.e., the amplitude parameter in
    # generate_templates truly represents SNR)
    for u in range(n_units):
        template = templates[u]  # (n_channels, template_len)
        for t_spike in spike_times_per_unit[u]:
            start = t_spike - half_template
            end = start + template_len
            # template is (n_channels, template_len), data is (n_samples, n_channels)
            data[start:end, :] += template.T * noise_std

    # Merge all spike times and labels, sorted by time
    all_times = []
    all_labels = []
    for u in range(n_units):
        all_times.append(spike_times_per_unit[u])
        all_labels.append(np.full(len(spike_times_per_unit[u]), u, dtype=np.int64))

    if len(all_times) > 0 and any(len(t) > 0 for t in all_times):
        all_times = np.concatenate(all_times)
        all_labels = np.concatenate(all_labels)
        sort_idx = np.argsort(all_times)
        all_times = all_times[sort_idx]
        all_labels = all_labels[sort_idx]
    else:
        all_times = np.array([], dtype=np.int64)
        all_labels = np.array([], dtype=np.int64)

    return {
        "data": data,
        "spike_times": spike_times_per_unit,
        "spike_labels": all_labels,
        "all_spike_times": all_times,
        "templates": templates,
        "primary_channels": primary_channels,
        "sampling_rate": sampling_rate,
        "n_units": n_units,
    }


def generate_drifting_recording(
    n_channels=32,
    duration_s=60.0,
    sampling_rate=30000.0,
    n_units=5,
    firing_rate=5.0,
    noise_std=1.0,
    refractory_ms=1.0,
    drift_um=50.0,
    pitch_um=25.0,
    seed=42,
):
    """Generate synthetic recording with linear probe drift.

    Simulates rigid vertical drift where the entire probe moves linearly
    over the recording duration. Each unit's primary channel shifts over
    time, modeling the effect of tissue drift in chronic recordings.

    The drift is implemented by linearly interpolating each unit's spatial
    footprint toward neighboring channels as the probe moves.

    Parameters
    ----------
    n_channels : int
        Number of recording channels.
    duration_s : float
        Recording duration in seconds.
    sampling_rate : float
        Sampling rate in Hz.
    n_units : int
        Number of distinct neural units to simulate.
    firing_rate : float
        Mean firing rate per unit in Hz.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    refractory_ms : float
        Absolute refractory period in milliseconds.
    drift_um : float
        Total probe drift in micrometers over the recording duration.
        Positive = probe moves down (neurons appear to move up).
    pitch_um : float
        Inter-electrode spacing in micrometers (for converting drift to channels).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        data : np.ndarray, shape (n_samples, n_channels), float64
            Simulated recording with drift.
        spike_times : list of np.ndarray
            Per-unit spike sample indices.
        spike_labels : np.ndarray
            All spike labels sorted by time.
        all_spike_times : np.ndarray
            All spike times sorted.
        templates : np.ndarray, shape (n_units, n_channels, n_samples)
            The initial waveform templates (before drift).
        primary_channels : np.ndarray, shape (n_units,)
            Initial primary channel per unit.
        drift_um : float
            Total drift applied.
        drift_trace : np.ndarray, shape (n_samples,)
            Instantaneous drift in micrometers at each sample.
        sampling_rate : float
        n_units : int
    """
    rng = np.random.default_rng(seed)
    n_total_samples = int(duration_s * sampling_rate)
    refractory_samples = int(refractory_ms * sampling_rate / 1000.0)

    template_len = 61
    half_template = template_len // 2

    # Generate templates at initial position
    templates, primary_channels = generate_templates(
        n_units, n_channels, n_samples=template_len, seed=seed + 1000
    )

    # Linear drift trace: 0 at start, drift_um at end
    drift_trace = np.linspace(0.0, drift_um, n_total_samples)

    # Generate Poisson spike trains
    spike_times_per_unit = []
    for u in range(n_units):
        mean_isi = sampling_rate / firing_rate
        n_expected = int(duration_s * firing_rate * 1.5) + 100
        isis = rng.exponential(mean_isi, size=n_expected)
        isis = np.maximum(isis, refractory_samples)
        times = np.cumsum(isis).astype(np.int64)
        margin = half_template + 1
        valid = times[(times >= margin) & (times < n_total_samples - margin)]
        spike_times_per_unit.append(valid)

    # Build recording with drift-shifted templates
    data = rng.normal(0.0, noise_std, size=(n_total_samples, n_channels))

    drift_channels = drift_um / pitch_um  # total drift in channel units

    for u in range(n_units):
        base_template = templates[u]  # (n_channels, template_len)
        for t_spike in spike_times_per_unit[u]:
            # Fractional channel shift at this time point
            frac = t_spike / max(n_total_samples - 1, 1)
            shift = frac * drift_channels  # channel shift at this time

            # Shift the template spatially by interpolating between channels
            shifted = np.zeros_like(base_template)
            for ch in range(n_channels):
                # Source channel in the original template
                src = ch + shift
                src_lo = int(np.floor(src))
                src_hi = src_lo + 1
                alpha = src - src_lo

                if 0 <= src_lo < n_channels:
                    shifted[ch] += (1.0 - alpha) * base_template[src_lo]
                if 0 <= src_hi < n_channels:
                    shifted[ch] += alpha * base_template[src_hi]

            start = t_spike - half_template
            end = start + template_len
            data[start:end, :] += shifted.T * noise_std

    # Merge all spike times and labels
    all_times = []
    all_labels = []
    for u in range(n_units):
        all_times.append(spike_times_per_unit[u])
        all_labels.append(np.full(len(spike_times_per_unit[u]), u, dtype=np.int64))

    if len(all_times) > 0 and any(len(t) > 0 for t in all_times):
        all_times = np.concatenate(all_times)
        all_labels = np.concatenate(all_labels)
        sort_idx = np.argsort(all_times)
        all_times = all_times[sort_idx]
        all_labels = all_labels[sort_idx]
    else:
        all_times = np.array([], dtype=np.int64)
        all_labels = np.array([], dtype=np.int64)

    return {
        "data": data,
        "spike_times": spike_times_per_unit,
        "spike_labels": all_labels,
        "all_spike_times": all_times,
        "templates": templates,
        "primary_channels": primary_channels,
        "drift_um": drift_um,
        "drift_trace": drift_trace,
        "sampling_rate": sampling_rate,
        "n_units": n_units,
    }
