//! Python bindings for Phase-Amplitude Coupling (PAC) metrics.
//!
//! Provides modulation index, mean vector length, phase-amplitude distribution,
//! and comodulogram computation.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::float::Float;
use zerostone::hilbert::HilbertTransform;
use zerostone::pac as zs_pac;
use zerostone::{BiquadCoeffs, IirFilter};

/// Compute the Modulation Index (Tort et al. 2010).
///
/// Measures phase-amplitude coupling by binning amplitude values by phase
/// and computing the KL divergence of the distribution from uniform.
///
/// Args:
///     phase (np.ndarray): Instantaneous phase values in radians (1D float32).
///     amplitude (np.ndarray): Instantaneous amplitude envelope (1D float32).
///     n_bins (int): Number of phase bins (2-64). Default: 18 (20-degree bins).
///
/// Returns:
///     float: MI in [0, 1] where 0 = no coupling, 1 = maximum coupling.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> phase = np.linspace(-np.pi, np.pi, 500).astype(np.float32)
///     >>> amplitude = (1 + 0.8 * np.cos(phase)).astype(np.float32)
///     >>> mi = zbci.modulation_index(phase, amplitude)
///     >>> assert mi > 0.01
#[pyfunction]
#[pyo3(signature = (phase, amplitude, n_bins=18))]
fn modulation_index(
    phase: PyReadonlyArray1<f32>,
    amplitude: PyReadonlyArray1<f32>,
    n_bins: usize,
) -> PyResult<Float> {
    let p_slice: Vec<Float> = phase.as_slice()?.iter().map(|&v| v as Float).collect();
    let a_slice: Vec<Float> = amplitude.as_slice()?.iter().map(|&v| v as Float).collect();

    if p_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Phase and amplitude arrays must have equal length: {} vs {}",
            p_slice.len(),
            a_slice.len()
        )));
    }
    if p_slice.is_empty() {
        return Err(PyValueError::new_err("Arrays must not be empty"));
    }
    if !(2..=64).contains(&n_bins) {
        return Err(PyValueError::new_err(format!(
            "n_bins must be in [2, 64], got {}",
            n_bins
        )));
    }

    Ok(zs_pac::modulation_index(&p_slice, &a_slice, n_bins))
}

/// Compute the Mean Vector Length (Canolty et al. 2006).
///
/// Measures phase-amplitude coupling as the normalized magnitude of the
/// mean amplitude-weighted phase vector.
///
/// Args:
///     phase (np.ndarray): Instantaneous phase values in radians (1D float32).
///     amplitude (np.ndarray): Instantaneous amplitude envelope (1D float32).
///
/// Returns:
///     float: MVL in [0, 1] where 0 = no coupling, 1 = perfect coupling.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> phase = np.linspace(-np.pi, np.pi, 500).astype(np.float32)
///     >>> amplitude = (1 + 0.8 * np.cos(phase)).astype(np.float32)
///     >>> mvl = zbci.mean_vector_length(phase, amplitude)
///     >>> assert mvl > 0.1
#[pyfunction]
fn mean_vector_length(
    phase: PyReadonlyArray1<f32>,
    amplitude: PyReadonlyArray1<f32>,
) -> PyResult<Float> {
    let p_slice: Vec<Float> = phase.as_slice()?.iter().map(|&v| v as Float).collect();
    let a_slice: Vec<Float> = amplitude.as_slice()?.iter().map(|&v| v as Float).collect();

    if p_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Phase and amplitude arrays must have equal length: {} vs {}",
            p_slice.len(),
            a_slice.len()
        )));
    }
    if p_slice.is_empty() {
        return Err(PyValueError::new_err("Arrays must not be empty"));
    }

    Ok(zs_pac::mean_vector_length(&p_slice, &a_slice))
}

/// Compute the phase-amplitude distribution for visualization.
///
/// Bins amplitude values by phase and returns bin centers and mean amplitudes.
///
/// Args:
///     phase (np.ndarray): Instantaneous phase values in radians (1D float32).
///     amplitude (np.ndarray): Instantaneous amplitude envelope (1D float32).
///     n_bins (int): Number of phase bins (2-64). Default: 18.
///
/// Returns:
///     tuple[np.ndarray, np.ndarray]: (bin_centers, mean_amplitudes) as 1D float32 arrays.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> phase = np.linspace(-np.pi, np.pi, 500).astype(np.float32)
///     >>> amplitude = np.ones(500, dtype=np.float32)
///     >>> centers, amps = zbci.phase_amplitude_distribution(phase, amplitude)
///     >>> assert len(centers) == 18
#[pyfunction]
#[pyo3(signature = (phase, amplitude, n_bins=18))]
#[allow(clippy::type_complexity)] // PyO3 return type with two numpy arrays
fn phase_amplitude_distribution<'py>(
    py: Python<'py>,
    phase: PyReadonlyArray1<f32>,
    amplitude: PyReadonlyArray1<f32>,
    n_bins: usize,
) -> PyResult<(
    pyo3::Bound<'py, PyArray1<f32>>,
    pyo3::Bound<'py, PyArray1<f32>>,
)> {
    let p_slice: Vec<Float> = phase.as_slice()?.iter().map(|&v| v as Float).collect();
    let a_slice: Vec<Float> = amplitude.as_slice()?.iter().map(|&v| v as Float).collect();

    if p_slice.len() != a_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Phase and amplitude arrays must have equal length: {} vs {}",
            p_slice.len(),
            a_slice.len()
        )));
    }
    if p_slice.is_empty() {
        return Err(PyValueError::new_err("Arrays must not be empty"));
    }
    if !(2..=64).contains(&n_bins) {
        return Err(PyValueError::new_err(format!(
            "n_bins must be in [2, 64], got {}",
            n_bins
        )));
    }

    let mut centers = vec![0.0; n_bins];
    let mut amps = vec![0.0; n_bins];
    zs_pac::phase_amplitude_distribution(&p_slice, &a_slice, n_bins, &mut centers, &mut amps);

    Ok((
        PyArray1::from_vec(py, centers.iter().map(|&v| v as f32).collect()),
        PyArray1::from_vec(py, amps.iter().map(|&v| v as f32).collect()),
    ))
}

/// Compute a PAC comodulogram over frequency pairs.
///
/// For each (phase_freq, amp_freq) pair, bandpass filters the signal,
/// extracts phase and amplitude via Hilbert transform, and computes
/// the coupling metric (MI or MVL).
///
/// Args:
///     signal (np.ndarray): Raw signal as 1D float32 array. Length must be
///         >= 256 samples.
///     sample_rate (float): Sampling frequency in Hz.
///     phase_freqs (np.ndarray): Center frequencies for phase extraction (1D float32).
///         Each frequency defines a bandpass band of [f-2, f+2] Hz.
///     amp_freqs (np.ndarray): Center frequencies for amplitude extraction (1D float32).
///         Each frequency defines a bandpass band of [f-f/4, f+f/4] Hz.
///     n_bins (int): Number of phase bins for MI method. Default: 18.
///     method (str): Coupling metric - 'mi' for Modulation Index or 'mvl'
///         for Mean Vector Length. Default: 'mi'.
///
/// Returns:
///     np.ndarray: 2D float32 array of shape (len(phase_freqs), len(amp_freqs))
///         with coupling values.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> t = np.arange(2048) / 500.0
///     >>> signal = np.sin(2 * np.pi * 6 * t).astype(np.float32)
///     >>> phase_freqs = np.array([4, 6, 8], dtype=np.float32)
///     >>> amp_freqs = np.array([30, 50, 70], dtype=np.float32)
///     >>> comod = zbci.pac_comodulogram(signal, 500.0, phase_freqs, amp_freqs)
///     >>> assert comod.shape == (3, 3)
#[pyfunction]
#[pyo3(signature = (signal, sample_rate, phase_freqs, amp_freqs, n_bins=18, method="mi"))]
fn pac_comodulogram<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f32>,
    sample_rate: Float,
    phase_freqs: PyReadonlyArray1<f32>,
    amp_freqs: PyReadonlyArray1<f32>,
    n_bins: usize,
    method: &str,
) -> PyResult<pyo3::Bound<'py, PyArray2<f32>>> {
    let sig: Vec<Float> = signal.as_slice()?.iter().map(|&v| v as Float).collect();
    let p_freqs: Vec<Float> = phase_freqs.as_slice()?.iter().map(|&v| v as Float).collect();
    let a_freqs: Vec<Float> = amp_freqs.as_slice()?.iter().map(|&v| v as Float).collect();

    if sig.is_empty() {
        return Err(PyValueError::new_err("Signal must not be empty"));
    }
    if sample_rate <= 0.0 {
        return Err(PyValueError::new_err("sample_rate must be positive"));
    }
    if p_freqs.is_empty() || a_freqs.is_empty() {
        return Err(PyValueError::new_err(
            "phase_freqs and amp_freqs must not be empty",
        ));
    }
    if !(2..=64).contains(&n_bins) {
        return Err(PyValueError::new_err(format!(
            "n_bins must be in [2, 64], got {}",
            n_bins
        )));
    }

    let use_mi = match method {
        "mi" => true,
        "mvl" => false,
        _ => return Err(PyValueError::new_err("method must be 'mi' or 'mvl'")),
    };

    let nyquist = sample_rate / 2.0;

    // Choose Hilbert size based on signal length
    let sig_len = sig.len();
    let hilbert_size = if sig_len >= 2048 {
        2048
    } else if sig_len >= 1024 {
        1024
    } else if sig_len >= 512 {
        512
    } else if sig_len >= 256 {
        256
    } else {
        return Err(PyValueError::new_err(format!(
            "Signal length {} must be >= 256 for Hilbert transform",
            sig_len
        )));
    };

    let n_phase = p_freqs.len();
    let n_amp = a_freqs.len();
    let mut result = vec![0.0; n_phase * n_amp];

    for (pi, &pf) in p_freqs.iter().enumerate() {
        // Phase band: [pf - 2, pf + 2] Hz (narrow band for phase)
        let p_low = pf - 2.0;
        let p_high = pf + 2.0;

        if p_low <= 0.0 || p_high >= nyquist {
            continue; // skip invalid frequencies
        }

        // Bandpass filter for phase (4th order = 2 sections)
        let p_sections =
            BiquadCoeffs::butterworth_bandpass_sections::<2>(sample_rate, p_low, p_high);
        let mut p_filter = IirFilter::new(p_sections);

        // Filter the signal for phase
        let mut p_filtered = vec![0.0; sig_len];
        for i in 0..sig_len {
            p_filtered[i] = p_filter.process_sample(sig[i]);
        }

        // Extract phase via Hilbert (process the last hilbert_size samples)
        let p_start = sig_len - hilbert_size;
        let p_chunk = &p_filtered[p_start..p_start + hilbert_size];
        let phase_vals = extract_phase(p_chunk, hilbert_size)?;

        for (ai, &af) in a_freqs.iter().enumerate() {
            // Amplitude band: [af - af/4, af + af/4] (proportional bandwidth)
            let a_low = af - af / 4.0;
            let a_high = af + af / 4.0;

            if a_low <= 0.0 || a_high >= nyquist {
                continue; // skip invalid frequencies
            }

            let a_sections =
                BiquadCoeffs::butterworth_bandpass_sections::<2>(sample_rate, a_low, a_high);
            let mut a_filter = IirFilter::new(a_sections);

            let mut a_filtered = vec![0.0; sig_len];
            for i in 0..sig_len {
                a_filtered[i] = a_filter.process_sample(sig[i]);
            }

            let a_chunk = &a_filtered[p_start..p_start + hilbert_size];
            let amp_vals = extract_amplitude(a_chunk, hilbert_size)?;

            let value = if use_mi {
                zs_pac::modulation_index(&phase_vals, &amp_vals, n_bins)
            } else {
                zs_pac::mean_vector_length(&phase_vals, &amp_vals)
            };

            result[pi * n_amp + ai] = value;
        }
    }

    // Build 2D array
    let rows: Vec<Vec<f32>> = result.chunks(n_amp).map(|c| c.iter().map(|&v| v as f32).collect()).collect();
    Ok(PyArray2::from_vec2(py, &rows)?)
}

/// Extract instantaneous phase from a signal chunk using HilbertTransform.
fn extract_phase(chunk: &[Float], size: usize) -> PyResult<Vec<Float>> {
    macro_rules! do_phase {
        ($n:expr) => {{
            let hilbert = HilbertTransform::<$n>::new();
            let input: [Float; $n] = chunk.try_into().map_err(|_| {
                PyValueError::new_err(format!("Chunk length {} != {}", chunk.len(), $n))
            })?;
            let mut output = [0.0; $n];
            hilbert.instantaneous_phase(&input, &mut output);
            Ok(output.to_vec())
        }};
    }

    match size {
        256 => do_phase!(256),
        512 => do_phase!(512),
        1024 => do_phase!(1024),
        2048 => do_phase!(2048),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported Hilbert size: {}",
            size
        ))),
    }
}

/// Extract instantaneous amplitude from a signal chunk using HilbertTransform.
fn extract_amplitude(chunk: &[Float], size: usize) -> PyResult<Vec<Float>> {
    macro_rules! do_amplitude {
        ($n:expr) => {{
            let hilbert = HilbertTransform::<$n>::new();
            let input: [Float; $n] = chunk.try_into().map_err(|_| {
                PyValueError::new_err(format!("Chunk length {} != {}", chunk.len(), $n))
            })?;
            let mut output = [0.0; $n];
            hilbert.instantaneous_amplitude(&input, &mut output);
            Ok(output.to_vec())
        }};
    }

    match size {
        256 => do_amplitude!(256),
        512 => do_amplitude!(512),
        1024 => do_amplitude!(1024),
        2048 => do_amplitude!(2048),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported Hilbert size: {}",
            size
        ))),
    }
}

/// Register PAC functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(modulation_index, m)?)?;
    m.add_function(wrap_pyfunction!(mean_vector_length, m)?)?;
    m.add_function(wrap_pyfunction!(phase_amplitude_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(pac_comodulogram, m)?)?;
    Ok(())
}
