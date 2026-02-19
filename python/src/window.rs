//! Python bindings for window functions.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    coherent_gain as zs_coherent_gain, equivalent_noise_bandwidth as zs_enbw,
    window_coefficient as zs_window_coefficient, WindowType,
};

/// Parse a window type string to the Rust enum.
fn parse_window_type(name: &str) -> PyResult<WindowType> {
    match name.to_lowercase().as_str() {
        "rectangular" | "rect" => Ok(WindowType::Rectangular),
        "hann" | "hanning" => Ok(WindowType::Hann),
        "hamming" => Ok(WindowType::Hamming),
        "blackman" => Ok(WindowType::Blackman),
        "blackman_harris" | "blackmanharris" => Ok(WindowType::BlackmanHarris),
        _ => Err(PyValueError::new_err(format!(
            "Unknown window type '{}'. Use: 'rectangular', 'hann', 'hamming', 'blackman', or 'blackman_harris'",
            name
        ))),
    }
}

/// Apply a window function to a signal and return the windowed signal.
///
/// Args:
///     signal (np.ndarray): Input signal as 1D float32 array.
///     window_type (str): Window type name.
///
/// Returns:
///     np.ndarray: Windowed signal as 1D float32 array.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> signal = np.ones(256, dtype=np.float32)
///     >>> windowed = zbci.apply_window(signal, "hann")
#[pyfunction]
fn apply_window<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f32>,
    window_type: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let wt = parse_window_type(window_type)?;
    let mut output = signal.as_slice()?.to_vec();
    zerostone::apply_window(&mut output, wt);
    Ok(PyArray1::from_vec(py, output))
}

/// Get a single window coefficient.
///
/// Args:
///     window_type (str): Window type name.
///     index (int): Sample index within the window.
///     length (int): Total window length.
///
/// Returns:
///     float: The window coefficient at the given index.
#[pyfunction]
fn window_coefficient(window_type: &str, index: usize, length: usize) -> PyResult<f32> {
    let wt = parse_window_type(window_type)?;
    Ok(zs_window_coefficient(wt, index, length))
}

/// Compute the coherent gain of a window function.
///
/// Args:
///     window_type (str): Window type name.
///     length (int): Window length.
///
/// Returns:
///     float: The coherent gain (sum of coefficients / length).
#[pyfunction]
fn coherent_gain(window_type: &str, length: usize) -> PyResult<f64> {
    let wt = parse_window_type(window_type)?;
    Ok(zs_coherent_gain(wt, length))
}

/// Compute the equivalent noise bandwidth of a window function.
///
/// Args:
///     window_type (str): Window type name.
///     length (int): Window length.
///
/// Returns:
///     float: The equivalent noise bandwidth in bins.
#[pyfunction]
fn equivalent_noise_bandwidth(window_type: &str, length: usize) -> PyResult<f64> {
    let wt = parse_window_type(window_type)?;
    Ok(zs_enbw(wt, length))
}

/// Register window functions with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_window, m)?)?;
    m.add_function(wrap_pyfunction!(window_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(coherent_gain, m)?)?;
    m.add_function(wrap_pyfunction!(equivalent_noise_bandwidth, m)?)?;
    Ok(())
}
