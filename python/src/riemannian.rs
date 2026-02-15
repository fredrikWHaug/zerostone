//! Python bindings for Riemannian geometry (tangent space projection).

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::linalg::Matrix;
use zerostone::TangentSpace as ZsTangentSpace;

enum TangentSpaceInner {
    C4(ZsTangentSpace<4, 16, 10>),
    C8(ZsTangentSpace<8, 64, 36>),
    C16(ZsTangentSpace<16, 256, 136>),
    C32(ZsTangentSpace<32, 1024, 528>),
}

/// Tangent space projection for symmetric positive definite matrices.
///
/// Projects SPD matrices (e.g. covariance matrices) to the tangent space
/// at a reference point, enabling use of Euclidean classifiers on
/// Riemannian manifold data.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// ts = npy.TangentSpace(channels=4)
/// ref_matrix = np.eye(4, dtype=np.float64)
/// ts.fit(ref_matrix)
/// vec = ts.transform(ref_matrix)  # shape (10,) for 4 channels
/// ```
#[pyclass]
pub struct TangentSpace {
    inner: TangentSpaceInner,
    channels: usize,
}

#[pymethods]
impl TangentSpace {
    /// Create a new tangent space projector.
    ///
    /// Args:
    ///     channels (int): Matrix dimension. Must be 4, 8, 16, or 32.
    #[new]
    fn new(channels: usize) -> PyResult<Self> {
        let inner = match channels {
            4 => TangentSpaceInner::C4(ZsTangentSpace::new()),
            8 => TangentSpaceInner::C8(ZsTangentSpace::new()),
            16 => TangentSpaceInner::C16(ZsTangentSpace::new()),
            32 => TangentSpaceInner::C32(ZsTangentSpace::new()),
            _ => return Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        };
        Ok(Self { inner, channels })
    }

    /// Fit the reference point (SPD matrix).
    ///
    /// Args:
    ///     reference (np.ndarray): 2D float64 array of shape (C, C).
    fn fit(&mut self, reference: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = reference.shape();
        if shape[0] != self.channels || shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrix, got ({}, {})",
                self.channels, self.channels, shape[0], shape[1]
            )));
        }
        let data = reference.as_slice()?;

        macro_rules! do_fit {
            ($ts:expr, $m:expr) => {{
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(data);
                $ts.fit(&Matrix::new(arr))
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))
            }};
        }
        match &mut self.inner {
            TangentSpaceInner::C4(ts) => do_fit!(ts, 16),
            TangentSpaceInner::C8(ts) => do_fit!(ts, 64),
            TangentSpaceInner::C16(ts) => do_fit!(ts, 256),
            TangentSpaceInner::C32(ts) => do_fit!(ts, 1024),
        }
    }

    /// Project an SPD matrix to the tangent space.
    ///
    /// Args:
    ///     matrix (np.ndarray): 2D float64 array of shape (C, C).
    ///
    /// Returns:
    ///     np.ndarray: 1D float64 vector of length C*(C+1)/2.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.channels || shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrix, got ({}, {})",
                self.channels, self.channels, shape[0], shape[1]
            )));
        }
        let data = matrix.as_slice()?;

        macro_rules! do_transform {
            ($ts:expr, $m:expr) => {{
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(data);
                let vec = $ts
                    .transform(&Matrix::new(arr))
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                Ok(PyArray1::from_vec(py, vec.to_vec()))
            }};
        }
        match &self.inner {
            TangentSpaceInner::C4(ts) => do_transform!(ts, 16),
            TangentSpaceInner::C8(ts) => do_transform!(ts, 64),
            TangentSpaceInner::C16(ts) => do_transform!(ts, 256),
            TangentSpaceInner::C32(ts) => do_transform!(ts, 1024),
        }
    }

    /// Reconstruct an SPD matrix from a tangent space vector.
    ///
    /// Args:
    ///     vector (np.ndarray): 1D float64 array of length C*(C+1)/2.
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (C, C).
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        vector: numpy::PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = vector.as_slice()?;

        macro_rules! do_inverse {
            ($ts:expr, $c:expr, $m:expr, $v:expr) => {{
                if data.len() != $v {
                    return Err(PyValueError::new_err(format!(
                        "Expected vector of length {}, got {}",
                        $v,
                        data.len()
                    )));
                }
                let mut arr = [0.0f64; $v];
                arr.copy_from_slice(data);
                let mat = $ts
                    .inverse_transform(&arr)
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                let flat = mat.data().to_vec();
                let array = Array2::from_shape_vec(($c, $c), flat).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }
        match &self.inner {
            TangentSpaceInner::C4(ts) => do_inverse!(ts, 4, 16, 10),
            TangentSpaceInner::C8(ts) => do_inverse!(ts, 8, 64, 36),
            TangentSpaceInner::C16(ts) => do_inverse!(ts, 16, 256, 136),
            TangentSpaceInner::C32(ts) => do_inverse!(ts, 32, 1024, 528),
        }
    }

    /// Number of channels (matrix dimension).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Length of the tangent space vector: C*(C+1)/2.
    #[getter]
    fn vector_length(&self) -> usize {
        self.channels * (self.channels + 1) / 2
    }

    fn __repr__(&self) -> String {
        format!(
            "TangentSpace(channels={}, vector_length={})",
            self.channels,
            self.vector_length()
        )
    }
}
