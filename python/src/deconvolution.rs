//! Python bindings for OASIS calcium deconvolution.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::OasisDeconvolution as ZsOasis;

enum OasisInner {
    C1(ZsOasis<1, 256>),
    C4(ZsOasis<4, 256>),
    C8(ZsOasis<8, 256>),
    C16(ZsOasis<16, 256>),
    C32(ZsOasis<32, 256>),
    C64(ZsOasis<64, 256>),
}

/// Online OASIS deconvolution for calcium imaging.
///
/// Performs online deconvolution of fluorescence signals to extract
/// estimated calcium traces and spike events.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
/// fluor = np.array([1.0], dtype=np.float32)
/// baseline = np.array([0.0], dtype=np.float32)
/// calcium, spike = deconv.update(fluor, baseline)
/// ```
#[pyclass]
pub struct OasisDeconvolution {
    inner: OasisInner,
    channels: usize,
}

#[pymethods]
impl OasisDeconvolution {
    /// Create a new OASIS deconvolution instance.
    ///
    /// Args:
    ///     channels (int): Number of channels. Must be 1, 4, 8, 16, 32, or 64.
    ///     gamma (float): Decay rate parameter (0 < gamma < 1).
    ///     lambda_ (float): Sparsity penalty parameter (>= 0).
    #[new]
    #[pyo3(signature = (channels, gamma, lambda_))]
    fn new(channels: usize, gamma: f32, lambda_: f32) -> PyResult<Self> {
        let inner = match channels {
            1 => OasisInner::C1(ZsOasis::new(gamma, lambda_)),
            4 => OasisInner::C4(ZsOasis::new(gamma, lambda_)),
            8 => OasisInner::C8(ZsOasis::new(gamma, lambda_)),
            16 => OasisInner::C16(ZsOasis::new(gamma, lambda_)),
            32 => OasisInner::C32(ZsOasis::new(gamma, lambda_)),
            64 => OasisInner::C64(ZsOasis::new(gamma, lambda_)),
            _ => {
                return Err(PyValueError::new_err(
                    "channels must be 1, 4, 8, 16, 32, or 64",
                ))
            }
        };
        Ok(Self { inner, channels })
    }

    /// Create from time constant tau.
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///     sample_rate (float): Sample rate in Hz.
    ///     tau (float): Decay time constant in seconds.
    ///     lambda_ (float): Sparsity penalty parameter.
    #[staticmethod]
    #[pyo3(signature = (channels, sample_rate, tau, lambda_))]
    fn from_tau(channels: usize, sample_rate: f32, tau: f32, lambda_: f32) -> PyResult<Self> {
        let inner = match channels {
            1 => OasisInner::C1(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            4 => OasisInner::C4(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            8 => OasisInner::C8(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            16 => OasisInner::C16(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            32 => OasisInner::C32(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            64 => OasisInner::C64(ZsOasis::from_tau(sample_rate, tau, lambda_)),
            _ => {
                return Err(PyValueError::new_err(
                    "channels must be 1, 4, 8, 16, 32, or 64",
                ))
            }
        };
        Ok(Self { inner, channels })
    }

    /// Process one sample of fluorescence data.
    ///
    /// Args:
    ///     fluorescence (np.ndarray): 1D float32 array of fluorescence values.
    ///     baseline (np.ndarray): 1D float32 array of baseline values.
    ///
    /// Returns:
    ///     tuple: (calcium, spike) as 1D float32 arrays.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        fluorescence: PyReadonlyArray1<f32>,
        baseline: PyReadonlyArray1<f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let fluor_slice = fluorescence.as_slice()?;
        let base_slice = baseline.as_slice()?;

        if fluor_slice.len() != self.channels || base_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} channels, got fluorescence={}, baseline={}",
                self.channels,
                fluor_slice.len(),
                base_slice.len()
            )));
        }

        macro_rules! do_update {
            ($oasis:expr, $c:expr) => {{
                let mut fluor = [0.0f32; $c];
                let mut base = [0.0f32; $c];
                fluor.copy_from_slice(fluor_slice);
                base.copy_from_slice(base_slice);
                let result = $oasis.update(&fluor, &base);
                (
                    PyArray1::from_slice(py, &result.calcium),
                    PyArray1::from_slice(py, &result.spike),
                )
            }};
        }

        let (calcium, spike) = match &mut self.inner {
            OasisInner::C1(o) => do_update!(o, 1),
            OasisInner::C4(o) => do_update!(o, 4),
            OasisInner::C8(o) => do_update!(o, 8),
            OasisInner::C16(o) => do_update!(o, 16),
            OasisInner::C32(o) => do_update!(o, 32),
            OasisInner::C64(o) => do_update!(o, 64),
        };
        Ok((calcium, spike))
    }

    /// Reset the deconvolution state.
    fn reset(&mut self) {
        match &mut self.inner {
            OasisInner::C1(o) => o.reset(),
            OasisInner::C4(o) => o.reset(),
            OasisInner::C8(o) => o.reset(),
            OasisInner::C16(o) => o.reset(),
            OasisInner::C32(o) => o.reset(),
            OasisInner::C64(o) => o.reset(),
        }
    }

    /// Decay rate parameter.
    #[getter]
    fn gamma(&self) -> f32 {
        match &self.inner {
            OasisInner::C1(o) => o.gamma(),
            OasisInner::C4(o) => o.gamma(),
            OasisInner::C8(o) => o.gamma(),
            OasisInner::C16(o) => o.gamma(),
            OasisInner::C32(o) => o.gamma(),
            OasisInner::C64(o) => o.gamma(),
        }
    }

    /// Sparsity penalty parameter.
    #[getter]
    fn lambda_(&self) -> f32 {
        match &self.inner {
            OasisInner::C1(o) => o.lambda(),
            OasisInner::C4(o) => o.lambda(),
            OasisInner::C8(o) => o.lambda(),
            OasisInner::C16(o) => o.lambda(),
            OasisInner::C32(o) => o.lambda(),
            OasisInner::C64(o) => o.lambda(),
        }
    }

    /// Set the sparsity penalty parameter.
    fn set_lambda(&mut self, lambda_: f32) {
        match &mut self.inner {
            OasisInner::C1(o) => o.set_lambda(lambda_),
            OasisInner::C4(o) => o.set_lambda(lambda_),
            OasisInner::C8(o) => o.set_lambda(lambda_),
            OasisInner::C16(o) => o.set_lambda(lambda_),
            OasisInner::C32(o) => o.set_lambda(lambda_),
            OasisInner::C64(o) => o.set_lambda(lambda_),
        }
    }

    /// Number of samples processed.
    #[getter]
    fn sample_count(&self) -> u64 {
        match &self.inner {
            OasisInner::C1(o) => o.sample_count(),
            OasisInner::C4(o) => o.sample_count(),
            OasisInner::C8(o) => o.sample_count(),
            OasisInner::C16(o) => o.sample_count(),
            OasisInner::C32(o) => o.sample_count(),
            OasisInner::C64(o) => o.sample_count(),
        }
    }

    /// Number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __repr__(&self) -> String {
        format!(
            "OasisDeconvolution(channels={}, gamma={}, lambda={})",
            self.channels,
            self.gamma(),
            self.lambda_()
        )
    }
}
