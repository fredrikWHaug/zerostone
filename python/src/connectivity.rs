//! Python bindings for connectivity metrics (coherence, PLV).
//!
//! Provides coherence and phase locking value functions for measuring
//! synchronization between brain regions.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::connectivity as zs_conn;

use crate::spectral::parse_window_type;

/// Compute magnitude-squared coherence between two signals.
///
/// Coherence measures the linear relationship between two signals at each
/// frequency. Values range from 0 (no relationship) to 1 (perfect linear
/// relationship).
///
/// Args:
///     signal_a (np.ndarray): First signal as 1D float32 array.
///     signal_b (np.ndarray): Second signal as 1D float32 array (same length as signal_a).
///     fft_size (int): FFT segment size. Must be 256, 512, 1024, 2048, or 4096.
///         Signals must have length >= fft_size.
///     window (str): Window type - 'rectangular', 'hann', 'hamming', 'blackman',
///         or 'blackman_harris'. Default: 'hann'.
///
/// Returns:
///     tuple[np.ndarray, np.ndarray]: (frequencies, coherence) as two 1D float32
///         arrays, each with fft_size/2 + 1 elements.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> t = np.arange(256) / 256.0
///     >>> sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
///     >>> freqs, coh = zbci.coherence(sig, sig, fft_size=256)
///     >>> assert coh[10] > 0.99  # identical signals
#[pyfunction]
#[pyo3(signature = (signal_a, signal_b, fft_size, window = "hann"))]
fn coherence<'py>(
    py: Python<'py>,
    signal_a: PyReadonlyArray1<f32>,
    signal_b: PyReadonlyArray1<f32>,
    fft_size: usize,
    window: &str,
) -> PyResult<(
    pyo3::Bound<'py, PyArray1<f32>>,
    pyo3::Bound<'py, PyArray1<f32>>,
)> {
    let a_slice = signal_a.as_slice()?;
    let b_slice = signal_b.as_slice()?;
    let window_type = parse_window_type(window)?;

    if a_slice.len() < fft_size {
        return Err(PyValueError::new_err(format!(
            "signal_a length {} must be >= fft_size {}",
            a_slice.len(),
            fft_size
        )));
    }
    if b_slice.len() < fft_size {
        return Err(PyValueError::new_err(format!(
            "signal_b length {} must be >= fft_size {}",
            b_slice.len(),
            fft_size
        )));
    }

    let bins = fft_size / 2 + 1;
    let mut coh = vec![0.0f32; bins];
    let mut freqs = vec![0.0f32; bins];

    macro_rules! run_coherence {
        ($N:expr) => {{
            zs_conn::coherence::<$N>(a_slice, b_slice, window_type, &mut coh);
            zs_conn::coherence_frequencies::<$N>(
                // We don't have sample_rate here for single-window coherence
                // Actually we need it for frequencies. Let's compute manually.
                1.0, // placeholder — we'll fix frequencies below
                &mut freqs,
            );
        }};
    }

    match fft_size {
        256 => run_coherence!(256),
        512 => run_coherence!(512),
        1024 => run_coherence!(1024),
        2048 => run_coherence!(2048),
        4096 => run_coherence!(4096),
        _ => {
            return Err(PyValueError::new_err(
                "fft_size must be 256, 512, 1024, 2048, or 4096",
            ))
        }
    }

    // Frequencies as normalized (0 to 0.5) since no sample_rate provided
    // Actually, let's return bin indices as frequencies for the simple version
    for (k, f) in freqs.iter_mut().enumerate() {
        *f = k as f32;
    }

    Ok((PyArray1::from_vec(py, freqs), PyArray1::from_vec(py, coh)))
}

/// Compute Welch-style averaged coherence between two signals.
///
/// More robust than single-window coherence. Averages cross- and auto-spectral
/// densities over multiple overlapping segments before computing coherence.
///
/// Args:
///     signal_a (np.ndarray): First signal as 1D float32 array.
///     signal_b (np.ndarray): Second signal as 1D float32 array (same length as signal_a).
///     fft_size (int): FFT segment size. Must be 256, 512, 1024, 2048, or 4096.
///     sample_rate (float): Sample rate in Hz.
///     overlap (float): Overlap fraction in [0.0, 1.0). Default: 0.5.
///     window (str): Window type. Default: 'hann'.
///
/// Returns:
///     tuple[np.ndarray, np.ndarray]: (frequencies, coherence) as two 1D float32
///         arrays, each with fft_size/2 + 1 elements.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> t = np.arange(1024) / 256.0
///     >>> sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
///     >>> freqs, coh = zbci.spectral_coherence(sig, sig, fft_size=256, sample_rate=256.0)
///     >>> assert coh[10] > 0.99
#[pyfunction]
#[pyo3(signature = (signal_a, signal_b, fft_size, sample_rate, overlap = 0.5, window = "hann"))]
fn spectral_coherence<'py>(
    py: Python<'py>,
    signal_a: PyReadonlyArray1<f32>,
    signal_b: PyReadonlyArray1<f32>,
    fft_size: usize,
    sample_rate: f32,
    overlap: f32,
    window: &str,
) -> PyResult<(
    pyo3::Bound<'py, PyArray1<f32>>,
    pyo3::Bound<'py, PyArray1<f32>>,
)> {
    let a_slice = signal_a.as_slice()?;
    let b_slice = signal_b.as_slice()?;
    let window_type = parse_window_type(window)?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Signals must have equal length: {} vs {}",
            a_slice.len(),
            b_slice.len()
        )));
    }
    if a_slice.len() < fft_size {
        return Err(PyValueError::new_err(format!(
            "Signal length {} must be >= fft_size {}",
            a_slice.len(),
            fft_size
        )));
    }
    if sample_rate <= 0.0 {
        return Err(PyValueError::new_err("sample_rate must be positive"));
    }
    if overlap < 0.0 || overlap >= 1.0 {
        return Err(PyValueError::new_err("overlap must be in [0.0, 1.0)"));
    }

    let bins = fft_size / 2 + 1;
    let mut coh = vec![0.0f32; bins];
    let mut freqs = vec![0.0f32; bins];

    macro_rules! run_spectral_coherence {
        ($N:expr) => {{
            zs_conn::spectral_coherence::<$N>(
                a_slice,
                b_slice,
                overlap,
                window_type,
                &mut coh,
            );
            zs_conn::coherence_frequencies::<$N>(sample_rate, &mut freqs);
        }};
    }

    match fft_size {
        256 => run_spectral_coherence!(256),
        512 => run_spectral_coherence!(512),
        1024 => run_spectral_coherence!(1024),
        2048 => run_spectral_coherence!(2048),
        4096 => run_spectral_coherence!(4096),
        _ => {
            return Err(PyValueError::new_err(
                "fft_size must be 256, 512, 1024, 2048, or 4096",
            ))
        }
    }

    Ok((PyArray1::from_vec(py, freqs), PyArray1::from_vec(py, coh)))
}

/// Compute Phase Locking Value between two instantaneous phase arrays.
///
/// PLV measures the consistency of the phase difference between two signals.
/// Input arrays must contain instantaneous phase values (e.g., from
/// HilbertTransform).
///
/// Args:
///     phases_a (np.ndarray): Instantaneous phases of signal A (radians, 1D float32).
///     phases_b (np.ndarray): Instantaneous phases of signal B (radians, 1D float32).
///
/// Returns:
///     float: PLV in [0, 1] where 1 = perfect phase locking, 0 = no synchronization.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> phases_a = np.linspace(0, 10, 100).astype(np.float32)
///     >>> phases_b = phases_a + 0.5  # constant offset
///     >>> plv = zbci.phase_locking_value(phases_a, phases_b)
///     >>> assert plv > 0.99  # perfect phase locking
#[pyfunction]
fn phase_locking_value(
    phases_a: PyReadonlyArray1<f32>,
    phases_b: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    let a_slice = phases_a.as_slice()?;
    let b_slice = phases_b.as_slice()?;

    if a_slice.len() != b_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Phase arrays must have equal length: {} vs {}",
            a_slice.len(),
            b_slice.len()
        )));
    }
    if a_slice.is_empty() {
        return Err(PyValueError::new_err("Phase arrays must not be empty"));
    }

    Ok(zs_conn::phase_locking_value(a_slice, b_slice))
}

/// Register connectivity functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coherence, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(phase_locking_value, m)?)?;
    Ok(())
}
