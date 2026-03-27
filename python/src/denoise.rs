use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;
use zerostone::denoise;

/// Denoise a signal using the Stationary Wavelet Transform with Haar wavelet.
///
/// Applies Donoho-Johnstone universal thresholding to suppress broadband noise
/// while preserving sharp transients like action potentials.
///
/// Args:
///     signal (np.ndarray): Input signal as 1D float64 numpy array
///     levels (int, optional): Number of SWT decomposition levels. Default: 3
///     mode (str, optional): Thresholding mode, "soft" or "hard". Default: "soft"
///
/// Returns:
///     np.ndarray: Denoised signal as 1D float64 numpy array
///
/// Example:
///     >>> import numpy as np
///     >>> signal = np.random.randn(256)
///     >>> denoised = denoise_haar(signal, levels=3, mode="soft")
#[pyfunction]
#[pyo3(signature = (signal, levels=3, mode="soft"))]
fn denoise_haar<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    levels: usize,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = signal.as_slice()?;
    let n = input.len();

    if n < 2 {
        return Err(PyValueError::new_err("Signal must have at least 2 samples"));
    }
    if levels == 0 {
        return Err(PyValueError::new_err("levels must be >= 1"));
    }

    let threshold_mode = match mode {
        "soft" => denoise::ThresholdMode::Soft,
        "hard" => denoise::ThresholdMode::Hard,
        _ => return Err(PyValueError::new_err("mode must be 'soft' or 'hard'")),
    };

    let mut buf = vec![0.0f64; n];
    buf.copy_from_slice(input);

    let scratch_len = n * (levels + 2);
    let mut scratch = vec![0.0f64; scratch_len];

    denoise::denoise_haar(&mut buf, &mut scratch, levels, threshold_mode);

    Ok(PyArray1::from_vec(py, buf))
}

/// Compute the universal threshold (Donoho-Johnstone 1994).
///
/// Args:
///     sigma (float): Noise standard deviation
///     n (int): Signal length
///
/// Returns:
///     float: Threshold value lambda = sigma * sqrt(2 * ln(n))
#[pyfunction]
fn universal_threshold_py(sigma: f64, n: usize) -> f64 {
    denoise::universal_threshold(sigma, n)
}

/// Apply soft thresholding: sign(x) * max(|x| - lambda, 0).
///
/// Args:
///     signal (np.ndarray): Input signal as 1D float64 numpy array
///     lambda_val (float): Threshold value
///
/// Returns:
///     np.ndarray: Thresholded signal
#[pyfunction]
fn soft_threshold<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    lambda_val: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = signal.as_slice()?;
    let output: Vec<f64> = input
        .iter()
        .map(|&x| denoise::soft_threshold(x, lambda_val))
        .collect();
    Ok(PyArray1::from_vec(py, output))
}

/// Apply hard thresholding: x if |x| >= lambda, else 0.
///
/// Args:
///     signal (np.ndarray): Input signal as 1D float64 numpy array
///     lambda_val (float): Threshold value
///
/// Returns:
///     np.ndarray: Thresholded signal
#[pyfunction]
fn hard_threshold<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    lambda_val: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = signal.as_slice()?;
    let output: Vec<f64> = input
        .iter()
        .map(|&x| denoise::hard_threshold(x, lambda_val))
        .collect();
    Ok(PyArray1::from_vec(py, output))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(denoise_haar, m)?)?;
    m.add_function(wrap_pyfunction!(universal_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(hard_threshold, m)?)?;
    Ok(())
}
