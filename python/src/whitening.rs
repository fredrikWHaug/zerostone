//! Python bindings for spatial whitening (ZCA/PCA).

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::whitening::{
    estimate_noise_covariance as zs_estimate_noise_covariance,
    WhiteningMatrix as ZsWhiteningMatrix, WhiteningMode,
};

// --- Macro for const-generic dispatch ---

macro_rules! make_whitening_inner {
    ($name:ident, $c:expr, $m:expr) => {
        struct $name(ZsWhiteningMatrix<$c, $m>);

        impl $name {
            fn from_covariance(
                cov_flat: &[f64],
                mode: WhiteningMode,
                epsilon: f64,
            ) -> Result<Self, String> {
                let mut cov = [[0.0f64; $c]; $c];
                for i in 0..$c {
                    for j in 0..$c {
                        cov[i][j] = cov_flat[i * $c + j];
                    }
                }
                ZsWhiteningMatrix::<$c, $m>::from_covariance(&cov, mode, epsilon)
                    .map(Self)
                    .map_err(|e| format!("LinalgError: {:?}", e))
            }

            fn apply(&self, sample: &[f64]) -> Vec<f64> {
                let mut input = [0.0f64; $c];
                input.copy_from_slice(sample);
                self.0.apply(&input).to_vec()
            }

            fn n_channels(&self) -> usize {
                $c
            }
        }
    };
}

make_whitening_inner!(Whiten2, 2, 4);
make_whitening_inner!(Whiten4, 4, 16);
make_whitening_inner!(Whiten8, 8, 64);
make_whitening_inner!(Whiten16, 16, 256);
make_whitening_inner!(Whiten32, 32, 1024);
make_whitening_inner!(Whiten64, 64, 4096);

enum WhiteningInner {
    C2(Whiten2),
    C4(Whiten4),
    C8(Box<Whiten8>),
    C16(Box<Whiten16>),
    C32(Box<Whiten32>),
    C64(Box<Whiten64>),
}

/// Spatial whitening matrix for multi-channel neural recordings.
///
/// Computes a whitening transform (ZCA or PCA) from a covariance matrix
/// and applies it to multi-channel samples. Supports 2, 4, 8, 16, 32,
/// or 64 channels.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// cov = np.array([[4.0, 2.0], [2.0, 3.0]])
/// wm = zbci.WhiteningMatrix(cov, mode="zca")
/// whitened = wm.apply(np.array([1.0, 0.5]))
/// ```
#[pyclass]
pub struct WhiteningMatrix {
    inner: WhiteningInner,
}

fn parse_mode(mode: &str) -> PyResult<WhiteningMode> {
    match mode {
        "zca" => Ok(WhiteningMode::Zca),
        "pca" => Ok(WhiteningMode::Pca),
        _ => Err(PyValueError::new_err("mode must be 'zca' or 'pca'")),
    }
}

#[pymethods]
impl WhiteningMatrix {
    /// Compute a whitening matrix from a covariance matrix.
    ///
    /// Args:
    ///     covariance (np.ndarray): 2D float64 symmetric covariance matrix of
    ///         shape (C, C). C must be 2, 4, 8, 16, 32, or 64.
    ///     mode (str): Whitening mode, "zca" or "pca". Default: "zca".
    ///     epsilon (float): Regularization added to eigenvalues. Default: 1e-6.
    ///
    /// Returns:
    ///     WhiteningMatrix: A fitted whitening transform.
    #[new]
    #[pyo3(signature = (covariance, mode="zca", epsilon=1e-6))]
    fn new(covariance: PyReadonlyArray2<f64>, mode: &str, epsilon: f64) -> PyResult<Self> {
        let shape = covariance.shape();
        let c = shape[0];
        if shape[1] != c {
            return Err(PyValueError::new_err(format!(
                "Covariance must be square, got ({}, {})",
                shape[0], shape[1]
            )));
        }
        let wmode = parse_mode(mode)?;
        let flat = covariance.as_slice()?;

        let inner = match c {
            2 => WhiteningInner::C2(
                Whiten2::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            ),
            4 => WhiteningInner::C4(
                Whiten4::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            ),
            8 => WhiteningInner::C8(Box::new(
                Whiten8::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            )),
            16 => WhiteningInner::C16(Box::new(
                Whiten16::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            )),
            32 => WhiteningInner::C32(Box::new(
                Whiten32::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            )),
            64 => WhiteningInner::C64(Box::new(
                Whiten64::from_covariance(flat, wmode, epsilon).map_err(PyValueError::new_err)?,
            )),
            _ => {
                return Err(PyValueError::new_err(
                    "n_channels must be 2, 4, 8, 16, 32, or 64",
                ));
            }
        };
        Ok(Self { inner })
    }

    /// Apply the whitening transform to a single multi-channel sample.
    ///
    /// Args:
    ///     sample (np.ndarray): 1D float64 array of length C.
    ///
    /// Returns:
    ///     np.ndarray: 1D float64 whitened sample of length C.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = sample.as_slice()?;
        let n = self.n_channels();
        if s.len() != n {
            return Err(PyValueError::new_err(format!(
                "sample has {} elements, expected {}",
                s.len(),
                n
            )));
        }

        let result = match &self.inner {
            WhiteningInner::C2(w) => w.apply(s),
            WhiteningInner::C4(w) => w.apply(s),
            WhiteningInner::C8(w) => w.apply(s),
            WhiteningInner::C16(w) => w.apply(s),
            WhiteningInner::C32(w) => w.apply(s),
            WhiteningInner::C64(w) => w.apply(s),
        };

        Ok(PyArray1::from_owned_array(py, Array1::from_vec(result)))
    }

    /// Number of channels.
    #[getter]
    fn n_channels(&self) -> usize {
        match &self.inner {
            WhiteningInner::C2(w) => w.n_channels(),
            WhiteningInner::C4(w) => w.n_channels(),
            WhiteningInner::C8(w) => w.n_channels(),
            WhiteningInner::C16(w) => w.n_channels(),
            WhiteningInner::C32(w) => w.n_channels(),
            WhiteningInner::C64(w) => w.n_channels(),
        }
    }

    fn __repr__(&self) -> String {
        format!("WhiteningMatrix(n_channels={})", self.n_channels())
    }
}

/// Estimate noise covariance from spike-free segments of multi-channel data.
///
/// Identifies quiet periods (where all channels are below threshold) and
/// computes covariance only from those samples. Falls back to full-data
/// covariance if fewer than ``min_quiet_samples`` quiet samples are found.
///
/// Args:
///     data (np.ndarray): 2D float64 array of shape ``(n_samples, n_channels)``.
///         n_channels must be 2, 4, 8, 16, 32, or 64.
///     noise_std (np.ndarray): 1D float64 array of per-channel noise standard deviations.
///     threshold_multiplier (float): Amplitude threshold in noise units. Default: 3.0.
///     min_quiet_samples (int): Minimum quiet samples required before fallback. Default: 100.
///
/// Returns:
///     np.ndarray: 2D float64 covariance matrix of shape ``(n_channels, n_channels)``.
#[pyfunction]
#[pyo3(signature = (data, noise_std, threshold_multiplier=3.0, min_quiet_samples=100))]
fn estimate_noise_covariance<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    noise_std: PyReadonlyArray1<f64>,
    threshold_multiplier: f64,
    min_quiet_samples: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let data_slice = data.as_slice()?;
    let noise_slice = noise_std.as_slice()?;

    if noise_slice.len() != n_channels {
        return Err(PyValueError::new_err(format!(
            "noise_std has {} elements, expected {}",
            noise_slice.len(),
            n_channels
        )));
    }

    macro_rules! do_estimate {
        ($c:expr) => {{
            let typed_data: &[[f64; $c]] = {
                assert!(data_slice.len() == n_samples * $c);
                unsafe {
                    core::slice::from_raw_parts(data_slice.as_ptr() as *const [f64; $c], n_samples)
                }
            };
            let mut noise_arr = [0.0f64; $c];
            noise_arr.copy_from_slice(noise_slice);
            let cov = zs_estimate_noise_covariance::<$c>(
                typed_data,
                &noise_arr,
                threshold_multiplier,
                min_quiet_samples,
            );
            let mut flat = Vec::with_capacity($c * $c);
            for i in 0..$c {
                for j in 0..$c {
                    flat.push(cov[i][j]);
                }
            }
            let arr = Array2::from_shape_vec(($c, $c), flat)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }};
    }

    match n_channels {
        2 => do_estimate!(2),
        4 => do_estimate!(4),
        8 => do_estimate!(8),
        16 => do_estimate!(16),
        32 => do_estimate!(32),
        64 => do_estimate!(64),
        _ => Err(PyValueError::new_err(
            "n_channels must be 2, 4, 8, 16, 32, or 64",
        )),
    }
}

/// Register whitening classes and functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WhiteningMatrix>()?;
    m.add_function(wrap_pyfunction!(estimate_noise_covariance, m)?)?;
    Ok(())
}
