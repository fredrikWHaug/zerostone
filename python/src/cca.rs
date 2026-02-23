//! Python bindings for Canonical Correlation Analysis (CCA) and SSVEP detection.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::cca as zs_cca;

/// Compute canonical correlations between two sets of multivariate signals.
///
/// Given signals X (T x C) and references Y (T x H), finds linear combinations
/// that maximize correlation. Returns canonical correlations in descending order.
///
/// Args:
///     signals (np.ndarray): Signal matrix of shape (n_samples, n_channels), float64.
///     references (np.ndarray): Reference matrix of shape (n_samples, n_components), float64.
///     regularization (float): Tikhonov regularization parameter. Default: 1e-6.
///
/// Returns:
///     np.ndarray: 1D array of canonical correlations in descending order,
///         length min(n_channels, n_components).
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> t = np.arange(200) / 250.0
///     >>> signals = np.column_stack([np.sin(2*np.pi*10*t), np.cos(2*np.pi*10*t)])
///     >>> refs = zbci.ssvep_references(250.0, 200, 10.0, n_harmonics=2)
///     >>> corr = zbci.cca(signals, refs)
///     >>> assert corr[0] > 0.9
#[pyfunction]
#[pyo3(signature = (signals, references, regularization = 1e-6))]
fn cca<'py>(
    py: Python<'py>,
    signals: PyReadonlyArray2<f64>,
    references: PyReadonlyArray2<f64>,
    regularization: f64,
) -> PyResult<pyo3::Bound<'py, PyArray1<f64>>> {
    let sig_shape = signals.shape();
    let ref_shape = references.shape();
    let n_samples = sig_shape[0];
    let n_channels = sig_shape[1];
    let n_components = ref_shape[1];

    if n_samples != ref_shape[0] {
        return Err(PyValueError::new_err(format!(
            "signals and references must have same number of samples: {} vs {}",
            n_samples, ref_shape[0]
        )));
    }
    if n_samples < 2 {
        return Err(PyValueError::new_err("Need at least 2 samples"));
    }

    let sig_slice = signals
        .as_slice()
        .map_err(|_| PyValueError::new_err("signals must be C-contiguous float64 array"))?;
    let ref_slice = references
        .as_slice()
        .map_err(|_| PyValueError::new_err("references must be C-contiguous float64 array"))?;

    let n_corr = n_channels.min(n_components);

    macro_rules! run_cca {
        ($C:expr, $H:expr) => {{
            let sig_arrays: Vec<[f64; $C]> = (0..n_samples)
                .map(|t| {
                    let mut arr = [0.0f64; $C];
                    for c in 0..$C {
                        arr[c] = sig_slice[t * n_channels + c];
                    }
                    arr
                })
                .collect();

            let ref_arrays: Vec<[f64; $H]> = (0..n_samples)
                .map(|t| {
                    let mut arr = [0.0f64; $H];
                    for h in 0..$H {
                        arr[h] = ref_slice[t * n_components + h];
                    }
                    arr
                })
                .collect();

            let mut correlations = vec![0.0f64; n_corr];
            zs_cca::cca::<$C, { $C * $C }, $H, { $H * $H }>(
                &sig_arrays,
                &ref_arrays,
                &mut correlations,
                regularization,
            )
            .map_err(|e| PyValueError::new_err(format!("CCA failed: {:?}", e)))?;

            Ok(PyArray1::from_vec(py, correlations))
        }};
    }

    match (n_channels, n_components) {
        (2, 2) => run_cca!(2, 2),
        (2, 4) => run_cca!(2, 4),
        (2, 6) => run_cca!(2, 6),
        (2, 8) => run_cca!(2, 8),
        (4, 2) => run_cca!(4, 2),
        (4, 4) => run_cca!(4, 4),
        (4, 6) => run_cca!(4, 6),
        (4, 8) => run_cca!(4, 8),
        (8, 2) => run_cca!(8, 2),
        (8, 4) => run_cca!(8, 4),
        (8, 6) => run_cca!(8, 6),
        (8, 8) => run_cca!(8, 8),
        (16, 2) => run_cca!(16, 2),
        (16, 4) => run_cca!(16, 4),
        (16, 6) => run_cca!(16, 6),
        (16, 8) => run_cca!(16, 8),
        (32, 4) => run_cca!(32, 4),
        (32, 6) => run_cca!(32, 6),
        (32, 8) => run_cca!(32, 8),
        (64, 4) => run_cca!(64, 4),
        (64, 6) => run_cca!(64, 6),
        (64, 8) => run_cca!(64, 8),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported channel/component combination: ({}, {}). \
             Channels must be 2, 4, 8, 16, 32, or 64. \
             Components must be 2, 4, 6, or 8.",
            n_channels, n_components
        ))),
    }
}

/// Generate SSVEP sinusoidal reference signals.
///
/// Creates sin/cos pairs at the fundamental frequency and its harmonics:
/// [sin(2*pi*f*t), cos(2*pi*f*t), sin(2*pi*2f*t), cos(2*pi*2f*t), ...]
///
/// Args:
///     sample_rate (float): Sampling frequency in Hz.
///     n_samples (int): Number of time samples to generate.
///     frequency (float): Fundamental SSVEP frequency in Hz.
///     n_harmonics (int): Number of harmonics to include. Default: 2.
///
/// Returns:
///     np.ndarray: 2D array of shape (n_samples, 2 * n_harmonics), float64.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> refs = zbci.ssvep_references(250.0, 250, 10.0, n_harmonics=2)
///     >>> assert refs.shape == (250, 4)
///     >>> assert abs(refs[0, 0]) < 1e-10  # sin(0) = 0
///     >>> assert abs(refs[0, 1] - 1.0) < 1e-10  # cos(0) = 1
#[pyfunction]
#[pyo3(signature = (sample_rate, n_samples, frequency, n_harmonics = 2))]
fn ssvep_references<'py>(
    py: Python<'py>,
    sample_rate: f64,
    n_samples: usize,
    frequency: f64,
    n_harmonics: usize,
) -> PyResult<pyo3::Bound<'py, PyArray2<f64>>> {
    if sample_rate <= 0.0 {
        return Err(PyValueError::new_err("sample_rate must be positive"));
    }
    if frequency <= 0.0 {
        return Err(PyValueError::new_err("frequency must be positive"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let n_components = 2 * n_harmonics;

    macro_rules! gen_refs {
        ($H:expr) => {{
            let mut output = vec![[0.0f64; $H]; n_samples];
            zs_cca::fill_ssvep_references::<$H>(sample_rate, frequency, &mut output);

            // Flatten to row-major for ndarray
            let flat: Vec<f64> = output.iter().flat_map(|row| row.iter().copied()).collect();
            let array = Array2::from_shape_vec((n_samples, $H), flat)
                .map_err(|e| PyValueError::new_err(format!("Array creation failed: {}", e)))?;
            Ok(PyArray2::from_owned_array(py, array))
        }};
    }

    match n_components {
        2 => gen_refs!(2),
        4 => gen_refs!(4),
        6 => gen_refs!(6),
        8 => gen_refs!(8),
        _ => Err(PyValueError::new_err(format!(
            "n_harmonics must be 1, 2, 3, or 4 (got {})",
            n_harmonics
        ))),
    }
}

/// Detect SSVEP frequency using Canonical Correlation Analysis.
///
/// Runs CCA between EEG signals and sinusoidal references at each target
/// frequency, returning the frequency with the highest canonical correlation.
///
/// Args:
///     signals (np.ndarray): EEG data of shape (n_samples, n_channels), float64.
///     sample_rate (float): Sampling frequency in Hz.
///     target_frequencies (list[float]): Target SSVEP frequencies to test.
///     n_harmonics (int): Number of harmonics for references. Default: 2.
///     regularization (float): Tikhonov regularization. Default: 1e-6.
///
/// Returns:
///     tuple[int, float]: (best_frequency_index, max_canonical_correlation).
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> t = np.arange(500) / 250.0
///     >>> sig = np.column_stack([np.sin(2*np.pi*10*t + i*0.3) for i in range(4)])
///     >>> idx, corr = zbci.ssvep_detect(sig, 250.0, [8.0, 10.0, 12.0, 15.0])
///     >>> assert idx == 1  # 10 Hz detected
#[pyfunction]
#[pyo3(signature = (signals, sample_rate, target_frequencies, n_harmonics = 2, regularization = 1e-6))]
fn ssvep_detect(
    signals: PyReadonlyArray2<f64>,
    sample_rate: f64,
    target_frequencies: Vec<f64>,
    n_harmonics: usize,
    regularization: f64,
) -> PyResult<(usize, f64)> {
    let shape = signals.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let n_components = 2 * n_harmonics;

    if n_samples < 2 {
        return Err(PyValueError::new_err("Need at least 2 samples"));
    }
    if target_frequencies.is_empty() {
        return Err(PyValueError::new_err(
            "target_frequencies must not be empty",
        ));
    }

    let sig_slice = signals
        .as_slice()
        .map_err(|_| PyValueError::new_err("signals must be C-contiguous float64 array"))?;

    macro_rules! run_detect {
        ($C:expr, $H:expr) => {{
            let sig_arrays: Vec<[f64; $C]> = (0..n_samples)
                .map(|t| {
                    let mut arr = [0.0f64; $C];
                    for c in 0..$C {
                        arr[c] = sig_slice[t * n_channels + c];
                    }
                    arr
                })
                .collect();

            let mut ref_buffer = vec![[0.0f64; $H]; n_samples];

            zs_cca::ssvep_detect::<$C, { $C * $C }, $H, { $H * $H }>(
                &sig_arrays,
                sample_rate,
                &target_frequencies,
                &mut ref_buffer,
                regularization,
            )
            .map_err(|e| PyValueError::new_err(format!("SSVEP detection failed: {:?}", e)))
        }};
    }

    match (n_channels, n_components) {
        (2, 2) => run_detect!(2, 2),
        (2, 4) => run_detect!(2, 4),
        (2, 6) => run_detect!(2, 6),
        (2, 8) => run_detect!(2, 8),
        (4, 2) => run_detect!(4, 2),
        (4, 4) => run_detect!(4, 4),
        (4, 6) => run_detect!(4, 6),
        (4, 8) => run_detect!(4, 8),
        (8, 2) => run_detect!(8, 2),
        (8, 4) => run_detect!(8, 4),
        (8, 6) => run_detect!(8, 6),
        (8, 8) => run_detect!(8, 8),
        (16, 2) => run_detect!(16, 2),
        (16, 4) => run_detect!(16, 4),
        (16, 6) => run_detect!(16, 6),
        (16, 8) => run_detect!(16, 8),
        (32, 4) => run_detect!(32, 4),
        (32, 6) => run_detect!(32, 6),
        (32, 8) => run_detect!(32, 8),
        (64, 4) => run_detect!(64, 4),
        (64, 6) => run_detect!(64, 6),
        (64, 8) => run_detect!(64, 8),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported channel/harmonics combination: ({}, {} harmonics). \
             Channels must be 2, 4, 8, 16, 32, or 64. \
             n_harmonics must be 1, 2, 3, or 4.",
            n_channels, n_harmonics
        ))),
    }
}

/// Register CCA functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cca, m)?)?;
    m.add_function(wrap_pyfunction!(ssvep_references, m)?)?;
    m.add_function(wrap_pyfunction!(ssvep_detect, m)?)?;
    Ok(())
}
