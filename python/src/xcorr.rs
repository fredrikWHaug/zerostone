//! Python bindings for cross-correlation and auto-correlation.
//!
//! Provides signal correlation functions for similarity measurement,
//! time delay estimation, and feature extraction.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::xcorr::{find_peak as zs_find_peak, Normalization as ZsNormalization};

/// Convert Python normalization string to Rust enum.
fn parse_normalization(norm: &str) -> PyResult<ZsNormalization> {
    match norm.to_lowercase().as_str() {
        "none" => Ok(ZsNormalization::None),
        "biased" => Ok(ZsNormalization::Biased),
        "unbiased" => Ok(ZsNormalization::Unbiased),
        "coeff" | "coefficient" => Ok(ZsNormalization::Coeff),
        _ => Err(PyValueError::new_err(format!(
            "Unknown normalization '{}'. Use: 'none', 'biased', 'unbiased', or 'coeff'",
            norm
        ))),
    }
}

/// Compute cross-correlation of two signals.
///
/// Cross-correlation measures the similarity between signals as a function of
/// time lag. Useful for time delay estimation, template matching, and feature
/// extraction in BCI applications.
///
/// Args:
///     x (np.ndarray): First signal as 1D float32 array (length N).
///     y (np.ndarray): Second signal as 1D float32 array (length M).
///     normalization (str): Normalization method:
///         - 'none': Raw correlation values (default)
///         - 'biased': Divide by max(N, M)
///         - 'unbiased': Divide by overlap count at each lag
///         - 'coeff': Normalize so autocorrelation at lag 0 equals 1.0
///
/// Returns:
///     np.ndarray: Cross-correlation output (length N + M - 1).
///         The lag at index k is k - (M - 1), ranging from -(M-1) to N-1.
///
/// Example:
///     >>> import npyci as npy
///     >>> import numpy as np
///     >>>
///     >>> # Detect time delay between signals
///     >>> x = np.array([0, 0, 1, 2, 1, 0, 0, 0], dtype=np.float32)
///     >>> y = np.array([0, 0, 0, 0, 1, 2, 1, 0], dtype=np.float32)  # x delayed by 2
///     >>> corr = npy.xcorr(x, y)
///     >>> peak_idx, peak_val = npy.find_peak(corr)
///     >>> lag = peak_idx - (len(y) - 1)  # Should be 2
#[pyfunction]
#[pyo3(signature = (x, y, normalization="none"))]
fn xcorr<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
    y: PyReadonlyArray1<f32>,
    normalization: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let y_slice = y.as_slice()?;
    let norm = parse_normalization(normalization)?;

    let out_len = x_slice.len() + y_slice.len() - 1;
    let mut output = vec![0.0f32; out_len];

    // Implement dynamic cross-correlation (Rust API uses const generics)
    let n = x_slice.len();
    let m = y_slice.len();

    // Compute energy for coefficient normalization
    let (energy_x, energy_y) = if norm == ZsNormalization::Coeff {
        let ex: f32 = x_slice.iter().map(|&v| v * v).sum();
        let ey: f32 = y_slice.iter().map(|&v| v * v).sum();
        (ex, ey)
    } else {
        (0.0, 0.0)
    };

    let norm_factor = if norm == ZsNormalization::Coeff {
        let denom = (energy_x * energy_y).sqrt();
        if denom > 1e-10 {
            denom
        } else {
            1.0
        }
    } else {
        1.0
    };

    let max_len = n.max(m) as f32;

    for (k, out_val) in output.iter_mut().enumerate() {
        let lag = k as i32 - (m as i32 - 1);

        let mut sum = 0.0f32;
        let mut count = 0usize;

        let n_start = (-lag).max(0) as usize;
        let n_end = (n as i32).min(m as i32 - lag).max(0) as usize;

        if n_end > n_start {
            for i in n_start..n_end {
                let y_idx = (i as i32 + lag) as usize;
                if y_idx < m {
                    sum += x_slice[i] * y_slice[y_idx];
                    count += 1;
                }
            }
        }

        *out_val = match norm {
            ZsNormalization::None => sum,
            ZsNormalization::Biased => sum / max_len,
            ZsNormalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            ZsNormalization::Coeff => sum / norm_factor,
        };
    }

    Ok(PyArray1::from_vec(py, output))
}

/// Compute auto-correlation of a signal.
///
/// Auto-correlation is the cross-correlation of a signal with itself.
/// Useful for detecting periodicity, estimating pitch, and analyzing
/// signal structure.
///
/// Args:
///     x (np.ndarray): Input signal as 1D float32 array (length N).
///     normalization (str): Normalization method (same as xcorr).
///
/// Returns:
///     np.ndarray: Auto-correlation output (length 2*N - 1).
///         The lag at index k is k - (N - 1), ranging from -(N-1) to N-1.
///         Auto-correlation is symmetric: output[k] == output[2*N - 2 - k]
///
/// Example:
///     >>> import npyci as npy
///     >>> import numpy as np
///     >>>
///     >>> signal = np.sin(np.linspace(0, 4*np.pi, 64)).astype(np.float32)
///     >>> acorr = npy.autocorr(signal, normalization='coeff')
///     >>> # Peak at center (lag 0) should be 1.0
///     >>> assert abs(acorr[len(signal) - 1] - 1.0) < 1e-6
#[pyfunction]
#[pyo3(signature = (x, normalization="none"))]
fn autocorr<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
    normalization: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x_slice = x.as_slice()?;
    let norm = parse_normalization(normalization)?;

    let n = x_slice.len();
    let out_len = 2 * n - 1;
    let center = n - 1;

    // Compute energy at lag 0
    let energy: f32 = x_slice.iter().map(|&v| v * v).sum();

    let norm_factor = if norm == ZsNormalization::Coeff {
        if energy > 1e-10 {
            energy
        } else {
            1.0
        }
    } else {
        1.0
    };

    let mut output = vec![0.0f32; out_len];

    // Compute only non-negative lags and mirror (autocorr is symmetric)
    for lag in 0..n {
        let mut sum = 0.0f32;
        let count = n - lag;

        for i in 0..count {
            sum += x_slice[i] * x_slice[i + lag];
        }

        let normalized = match norm {
            ZsNormalization::None => sum,
            ZsNormalization::Biased => sum / n as f32,
            ZsNormalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            ZsNormalization::Coeff => sum / norm_factor,
        };

        // Place at center + lag (positive lag)
        if center + lag < out_len {
            output[center + lag] = normalized;
        }

        // Mirror to center - lag (negative lag), except for lag 0
        if lag > 0 && center >= lag {
            output[center - lag] = normalized;
        }
    }

    Ok(PyArray1::from_vec(py, output))
}

/// Find the peak (maximum) in a correlation array.
///
/// Utility function to find the lag at which correlation is maximum,
/// commonly used for time delay estimation.
///
/// Args:
///     correlation (np.ndarray): Correlation output from xcorr or autocorr.
///
/// Returns:
///     tuple: (peak_index, peak_value)
///
/// Example:
///     >>> corr = npy.xcorr(x, y)
///     >>> peak_idx, peak_val = npy.find_peak(corr)
///     >>> lag = peak_idx - (len(y) - 1)  # Convert to lag
#[pyfunction]
fn find_peak(correlation: PyReadonlyArray1<f32>) -> PyResult<(usize, f32)> {
    let corr_slice = correlation.as_slice()?;
    if corr_slice.is_empty() {
        return Err(PyValueError::new_err("Correlation array is empty"));
    }

    let (idx, val) = zs_find_peak(corr_slice);
    Ok((idx, val))
}

/// Convert correlation index to lag value.
///
/// For cross-correlation of signals with lengths N and M,
/// converts array index to the corresponding lag value.
///
/// Args:
///     index (int): Index into correlation output array.
///     m (int): Length of the second signal (y in xcorr(x, y)).
///
/// Returns:
///     int: Lag value at that index.
///
/// Example:
///     >>> lag = npy.index_to_lag(peak_idx, len(y))
#[pyfunction]
fn index_to_lag(index: usize, m: usize) -> i32 {
    index as i32 - (m as i32 - 1)
}

/// Convert lag value to correlation index.
///
/// Args:
///     lag (int): Lag value.
///     m (int): Length of the second signal.
///
/// Returns:
///     int: Index into correlation output array.
#[pyfunction]
fn lag_to_index(lag: i32, m: usize) -> usize {
    (lag + (m as i32 - 1)) as usize
}

/// Register xcorr functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xcorr, m)?)?;
    m.add_function(wrap_pyfunction!(autocorr, m)?)?;
    m.add_function(wrap_pyfunction!(find_peak, m)?)?;
    m.add_function(wrap_pyfunction!(index_to_lag, m)?)?;
    m.add_function(wrap_pyfunction!(lag_to_index, m)?)?;
    Ok(())
}
