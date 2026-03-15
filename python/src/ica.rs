//! Python bindings for Independent Component Analysis (FastICA).

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::ica::{ContrastFunction, Ica as ZsIca, IcaError};

#[allow(clippy::large_enum_variant)] // PyO3 dispatch enum, lives on heap via #[pyclass]
enum IcaInner {
    C4(ZsIca<4, 16>),
    C8(ZsIca<8, 64>),
    C16(ZsIca<16, 256>),
    C32(Box<ZsIca<32, 1024>>),
    C64(Box<ZsIca<64, 4096>>),
}

fn parse_contrast(s: &str) -> PyResult<ContrastFunction> {
    match s.to_lowercase().as_str() {
        "logcosh" | "log_cosh" => Ok(ContrastFunction::LogCosh),
        "exp" => Ok(ContrastFunction::Exp),
        "cube" => Ok(ContrastFunction::Cube),
        _ => Err(PyValueError::new_err(format!(
            "Unknown contrast function '{}'. Must be 'logcosh', 'exp', or 'cube'.",
            s
        ))),
    }
}

fn contrast_name(c: ContrastFunction) -> &'static str {
    match c {
        ContrastFunction::LogCosh => "logcosh",
        ContrastFunction::Exp => "exp",
        ContrastFunction::Cube => "cube",
    }
}

fn ica_error_to_py(e: IcaError) -> PyErr {
    let msg = match e {
        IcaError::InsufficientData => "Insufficient data: need at least 2 * channels samples",
        IcaError::NotFitted => "ICA model has not been fitted yet",
        IcaError::ConvergenceFailed => "FastICA did not converge within max_iter iterations",
        IcaError::EigenFailed => "Eigenvalue decomposition failed",
        IcaError::NumericalInstability => "Numerical instability detected (NaN or Inf)",
    };
    PyValueError::new_err(msg)
}

/// Independent Component Analysis (FastICA) for blind source separation.
///
/// Decomposes multi-channel signals into statistically independent sources.
/// The primary use case is artifact removal in EEG: decompose, identify
/// artifact components (eye blinks, muscle, heartbeat), then reconstruct
/// without them.
///
/// Uses the symmetric (parallel) FastICA algorithm which extracts all
/// components simultaneously, avoiding error accumulation.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// ica = zbci.Ica(channels=4, contrast="logcosh")
///
/// # data shape: (samples, channels)
/// data = np.random.randn(1000, 4)
/// ica.fit(data)
///
/// # Get independent components
/// sources = ica.transform(data)
///
/// # Remove artifact component 0
/// cleaned = ica.remove_components(data, [0])
/// ```
#[pyclass]
pub struct Ica {
    inner: IcaInner,
    channels: usize,
    contrast: ContrastFunction,
}

#[pymethods]
impl Ica {
    /// Create a new ICA decomposer.
    ///
    /// Args:
    ///     channels (int): Number of channels (4, 8, 16, 32, or 64).
    ///     contrast (str): Contrast function - 'logcosh' (default), 'exp', or 'cube'.
    #[new]
    #[pyo3(signature = (channels, contrast="logcosh"))]
    fn new(channels: usize, contrast: &str) -> PyResult<Self> {
        let cf = parse_contrast(contrast)?;
        let inner = match channels {
            4 => IcaInner::C4(ZsIca::new(cf)),
            8 => IcaInner::C8(ZsIca::new(cf)),
            16 => IcaInner::C16(ZsIca::new(cf)),
            32 => IcaInner::C32(Box::new(ZsIca::new(cf))),
            64 => IcaInner::C64(Box::new(ZsIca::new(cf))),
            _ => {
                return Err(PyValueError::new_err(
                    "channels must be 4, 8, 16, 32, or 64",
                ))
            }
        };
        Ok(Self {
            inner,
            channels,
            contrast: cf,
        })
    }

    /// Fit the ICA model to data.
    ///
    /// Args:
    ///     data (np.ndarray): 2D float64 array of shape (samples, channels).
    ///     max_iter (int): Maximum iterations. Default: 200.
    ///     tolerance (float): Convergence tolerance. Default: 1e-4.
    #[pyo3(signature = (data, max_iter=200, tolerance=1e-4))]
    fn fit(
        &mut self,
        data: PyReadonlyArray2<f64>,
        max_iter: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        let shape = data.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Data has {} channels, expected {}",
                shape[1], self.channels
            )));
        }
        let flat = data.as_slice()?;
        let n = shape[0];

        macro_rules! do_fit {
            ($ica:expr, $c:expr) => {{
                let samples: Vec<[f64; $c]> = (0..n)
                    .map(|i| {
                        let mut s = [0.0f64; $c];
                        s.copy_from_slice(&flat[i * $c..(i + 1) * $c]);
                        s
                    })
                    .collect();
                $ica.fit(&samples, max_iter, tolerance)
                    .map_err(ica_error_to_py)
            }};
        }

        match &mut self.inner {
            IcaInner::C4(ica) => do_fit!(ica, 4),
            IcaInner::C8(ica) => do_fit!(ica, 8),
            IcaInner::C16(ica) => do_fit!(ica, 16),
            IcaInner::C32(ica) => do_fit!(ica, 32),
            IcaInner::C64(ica) => do_fit!(ica, 64),
        }
    }

    /// Transform data to independent components.
    ///
    /// Args:
    ///     data (np.ndarray): 2D float64 array of shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (samples, channels) with independent components.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = data.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Data has {} channels, expected {}",
                shape[1], self.channels
            )));
        }
        let flat = data.as_slice()?;
        let n = shape[0];

        macro_rules! do_transform {
            ($ica:expr, $c:expr) => {{
                let samples: Vec<[f64; $c]> = (0..n)
                    .map(|i| {
                        let mut s = [0.0f64; $c];
                        s.copy_from_slice(&flat[i * $c..(i + 1) * $c]);
                        s
                    })
                    .collect();
                let mut output = vec![[0.0f64; $c]; n];
                $ica.transform(&samples, &mut output)
                    .map_err(ica_error_to_py)?;
                let result: Vec<f64> = output.iter().flat_map(|s| s.iter().copied()).collect();
                let array = Array2::from_shape_vec((n, $c), result).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }

        match &self.inner {
            IcaInner::C4(ica) => do_transform!(ica, 4),
            IcaInner::C8(ica) => do_transform!(ica, 8),
            IcaInner::C16(ica) => do_transform!(ica, 16),
            IcaInner::C32(ica) => do_transform!(ica, 32),
            IcaInner::C64(ica) => do_transform!(ica, 64),
        }
    }

    /// Reconstruct data from independent components.
    ///
    /// Args:
    ///     sources (np.ndarray): 2D float64 array of shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (samples, channels) with reconstructed data.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        sources: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = sources.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sources has {} channels, expected {}",
                shape[1], self.channels
            )));
        }
        let flat = sources.as_slice()?;
        let n = shape[0];

        macro_rules! do_inverse {
            ($ica:expr, $c:expr) => {{
                let samples: Vec<[f64; $c]> = (0..n)
                    .map(|i| {
                        let mut s = [0.0f64; $c];
                        s.copy_from_slice(&flat[i * $c..(i + 1) * $c]);
                        s
                    })
                    .collect();
                let mut output = vec![[0.0f64; $c]; n];
                $ica.inverse_transform(&samples, &mut output)
                    .map_err(ica_error_to_py)?;
                let result: Vec<f64> = output.iter().flat_map(|s| s.iter().copied()).collect();
                let array = Array2::from_shape_vec((n, $c), result).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }

        match &self.inner {
            IcaInner::C4(ica) => do_inverse!(ica, 4),
            IcaInner::C8(ica) => do_inverse!(ica, 8),
            IcaInner::C16(ica) => do_inverse!(ica, 16),
            IcaInner::C32(ica) => do_inverse!(ica, 32),
            IcaInner::C64(ica) => do_inverse!(ica, 64),
        }
    }

    /// Remove specified components and reconstruct.
    ///
    /// Transform to IC space, zero out excluded components, then inverse transform.
    ///
    /// Args:
    ///     data (np.ndarray): 2D float64 array of shape (samples, channels).
    ///     exclude (list[int]): Indices of components to remove.
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (samples, channels) with cleaned data.
    fn remove_components<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        exclude: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = data.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Data has {} channels, expected {}",
                shape[1], self.channels
            )));
        }
        let flat = data.as_slice()?;
        let n = shape[0];

        macro_rules! do_remove {
            ($ica:expr, $c:expr) => {{
                let samples: Vec<[f64; $c]> = (0..n)
                    .map(|i| {
                        let mut s = [0.0f64; $c];
                        s.copy_from_slice(&flat[i * $c..(i + 1) * $c]);
                        s
                    })
                    .collect();
                let mut output = vec![[0.0f64; $c]; n];
                $ica.remove_components(&samples, &exclude, &mut output)
                    .map_err(ica_error_to_py)?;
                let result: Vec<f64> = output.iter().flat_map(|s| s.iter().copied()).collect();
                let array = Array2::from_shape_vec((n, $c), result).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }

        match &self.inner {
            IcaInner::C4(ica) => do_remove!(ica, 4),
            IcaInner::C8(ica) => do_remove!(ica, 8),
            IcaInner::C16(ica) => do_remove!(ica, 16),
            IcaInner::C32(ica) => do_remove!(ica, 32),
            IcaInner::C64(ica) => do_remove!(ica, 64),
        }
    }

    /// Get the mixing matrix A (channels x channels).
    ///
    /// x = A * s + mean
    #[getter]
    fn mixing_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        macro_rules! get_mixing {
            ($ica:expr, $c:expr) => {{
                let mat = $ica.mixing_matrix();
                let flat = mat.data().to_vec();
                let array = Array2::from_shape_vec(($c, $c), flat).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }

        match &self.inner {
            IcaInner::C4(ica) => get_mixing!(ica, 4),
            IcaInner::C8(ica) => get_mixing!(ica, 8),
            IcaInner::C16(ica) => get_mixing!(ica, 16),
            IcaInner::C32(ica) => get_mixing!(ica, 32),
            IcaInner::C64(ica) => get_mixing!(ica, 64),
        }
    }

    /// Get the unmixing matrix W (channels x channels).
    ///
    /// s = W * (x - mean)
    #[getter]
    fn unmixing_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        macro_rules! get_unmixing {
            ($ica:expr, $c:expr) => {{
                let mat = $ica.unmixing_matrix();
                let flat = mat.data().to_vec();
                let array = Array2::from_shape_vec(($c, $c), flat).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }

        match &self.inner {
            IcaInner::C4(ica) => get_unmixing!(ica, 4),
            IcaInner::C8(ica) => get_unmixing!(ica, 8),
            IcaInner::C16(ica) => get_unmixing!(ica, 16),
            IcaInner::C32(ica) => get_unmixing!(ica, 32),
            IcaInner::C64(ica) => get_unmixing!(ica, 64),
        }
    }

    /// Number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Whether the model has been fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        match &self.inner {
            IcaInner::C4(ica) => ica.is_fitted(),
            IcaInner::C8(ica) => ica.is_fitted(),
            IcaInner::C16(ica) => ica.is_fitted(),
            IcaInner::C32(ica) => ica.is_fitted(),
            IcaInner::C64(ica) => ica.is_fitted(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Ica(channels={}, contrast={}, fitted={})",
            self.channels,
            contrast_name(self.contrast),
            self.is_fitted()
        )
    }
}
