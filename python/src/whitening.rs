//! Python bindings for spatial whitening (ZCA/PCA).

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::whitening::{WhiteningMatrix as ZsWhiteningMatrix, WhiteningMode};

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

/// Register whitening classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WhiteningMatrix>()?;
    Ok(())
}
