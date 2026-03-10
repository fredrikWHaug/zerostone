//! Python bindings for entropy measures.
//!
//! Provides sample entropy, approximate entropy, spectral entropy,
//! and multiscale entropy functions.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::entropy as zs_entropy;

/// Compute Sample Entropy (SampEn).
///
/// Measures signal complexity by counting template matches at embedding
/// dimensions m and m+1. Lower values indicate more regularity.
///
/// Args:
///     data (np.ndarray): Input time series as 1D float64 array.
///     m (int): Embedding dimension. Default: 2.
///     r (float): Tolerance threshold. Default: 0.2.
///
/// Returns:
///     float: SampEn value. Returns inf if no matches at dimension m+1.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> constant = np.ones(100)
///     >>> se = zbci.sample_entropy(constant)
///     >>> assert se < 1e-10
#[pyfunction]
#[pyo3(signature = (data, m=2, r=0.2))]
fn sample_entropy(data: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    let slice = data.as_slice()?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    if m < 1 {
        return Err(PyValueError::new_err(format!("m must be >= 1, got {}", m)));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err(format!("r must be > 0, got {}", r)));
    }
    if slice.len() <= m + 1 {
        return Err(PyValueError::new_err(format!(
            "data length {} must be > m + 1 = {}",
            slice.len(),
            m + 1
        )));
    }

    Ok(zs_entropy::sample_entropy(slice, m, r))
}

/// Compute Approximate Entropy (ApEn).
///
/// Similar to sample entropy but includes self-matches, guaranteeing the
/// result is always finite.
///
/// Args:
///     data (np.ndarray): Input time series as 1D float64 array.
///     m (int): Embedding dimension. Default: 2.
///     r (float): Tolerance threshold. Default: 0.2.
///
/// Returns:
///     float: ApEn value (always finite and non-negative).
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> constant = np.ones(100)
///     >>> ae = zbci.approximate_entropy(constant)
///     >>> assert ae < 0.01
#[pyfunction]
#[pyo3(signature = (data, m=2, r=0.2))]
fn approximate_entropy(data: PyReadonlyArray1<f64>, m: usize, r: f64) -> PyResult<f64> {
    let slice = data.as_slice()?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    if m < 1 {
        return Err(PyValueError::new_err(format!("m must be >= 1, got {}", m)));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err(format!("r must be > 0, got {}", r)));
    }
    if slice.len() <= m {
        return Err(PyValueError::new_err(format!(
            "data length {} must be > m = {}",
            slice.len(),
            m
        )));
    }

    Ok(zs_entropy::approximate_entropy(slice, m, r))
}

/// Compute Spectral Entropy.
///
/// Shannon entropy of the normalized power spectral density.
///
/// Args:
///     psd (np.ndarray): Power spectral density values as 1D float64 array.
///     normalize (bool): If True, normalize by ln(N) for [0,1] range. Default: True.
///
/// Returns:
///     float: Spectral entropy. If normalized, 0 = single peak, 1 = flat spectrum.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> flat = np.ones(64)
///     >>> h = zbci.spectral_entropy(flat)
///     >>> assert abs(h - 1.0) < 1e-10
#[pyfunction]
#[pyo3(signature = (psd, normalize=true))]
fn spectral_entropy(psd: PyReadonlyArray1<f64>, normalize: bool) -> PyResult<f64> {
    let slice = psd.as_slice()?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("PSD must not be empty"));
    }

    Ok(zs_entropy::spectral_entropy(slice, normalize))
}

/// Compute Multiscale Entropy (MSE).
///
/// Coarse-grains the signal at the given scale factor and computes
/// sample entropy on the result.
///
/// Args:
///     data (np.ndarray): Input time series as 1D float64 array.
///     scale (int): Scale factor (>= 1). Default: 1.
///     m (int): Embedding dimension. Default: 2.
///     r (float): Tolerance threshold. Default: 0.2.
///
/// Returns:
///     float: Sample entropy at the given scale.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> data = np.tile([1.0, 2.0], 20)
///     >>> mse1 = zbci.multiscale_entropy(data, scale=1)
///     >>> se = zbci.sample_entropy(data)
///     >>> assert abs(mse1 - se) < 1e-10
#[pyfunction]
#[pyo3(signature = (data, scale=1, m=2, r=0.2))]
fn multiscale_entropy(
    data: PyReadonlyArray1<f64>,
    scale: usize,
    m: usize,
    r: f64,
) -> PyResult<f64> {
    let slice = data.as_slice()?;

    if slice.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    if scale < 1 {
        return Err(PyValueError::new_err(format!(
            "scale must be >= 1, got {}",
            scale
        )));
    }
    if m < 1 {
        return Err(PyValueError::new_err(format!("m must be >= 1, got {}", m)));
    }
    if r <= 0.0 {
        return Err(PyValueError::new_err(format!("r must be > 0, got {}", r)));
    }

    if scale == 1 {
        if slice.len() <= m + 1 {
            return Err(PyValueError::new_err(format!(
                "data length {} must be > m + 1 = {}",
                slice.len(),
                m + 1
            )));
        }
        return Ok(zs_entropy::sample_entropy(slice, m, r));
    }

    let coarse_len = slice.len() / scale;
    if coarse_len <= m + 1 {
        return Err(PyValueError::new_err(format!(
            "data too short after coarse-graining: coarse_len {} must be > m + 1 = {}",
            coarse_len,
            m + 1
        )));
    }

    let mut scratch = vec![0.0f64; coarse_len];
    Ok(zs_entropy::multiscale_entropy(
        slice,
        scale,
        m,
        r,
        &mut scratch,
    ))
}

/// Register entropy functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(multiscale_entropy, m)?)?;
    Ok(())
}
