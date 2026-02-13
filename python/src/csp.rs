//! Python bindings for Common Spatial Patterns (CSP).
//!
//! CSP is THE algorithm for motor imagery BCI - it finds spatial filters that
//! maximize variance between two classes (e.g., left vs right hand imagination).

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{AdaptiveCsp as ZsAdaptiveCsp, UpdateConfig as ZsUpdateConfig};

// ============================================================================
// AdaptiveCsp - Predefined configurations
// ============================================================================

/// Enum dispatch for different CSP configurations.
/// Format: Channels_Filters (C channels, K filters)
/// M = C*C, F = K*C
enum AdaptiveCspInner {
    // 4 channels (common for basic motor imagery)
    C4K2(ZsAdaptiveCsp<4, 16, 2, 8>),
    C4K4(ZsAdaptiveCsp<4, 16, 4, 16>),

    // 8 channels (standard montage)
    C8K2(ZsAdaptiveCsp<8, 64, 2, 16>),
    C8K4(ZsAdaptiveCsp<8, 64, 4, 32>),
    C8K6(ZsAdaptiveCsp<8, 64, 6, 48>),

    // 16 channels
    C16K2(ZsAdaptiveCsp<16, 256, 2, 32>),
    C16K4(ZsAdaptiveCsp<16, 256, 4, 64>),
    C16K6(ZsAdaptiveCsp<16, 256, 6, 96>),

    // 32 channels
    C32K2(ZsAdaptiveCsp<32, 1024, 2, 64>),
    C32K4(ZsAdaptiveCsp<32, 1024, 4, 128>),
    C32K6(ZsAdaptiveCsp<32, 1024, 6, 192>),

    // 64 channels (high-density EEG)
    C64K2(ZsAdaptiveCsp<64, 4096, 2, 128>),
    C64K4(ZsAdaptiveCsp<64, 4096, 4, 256>),
    C64K6(ZsAdaptiveCsp<64, 4096, 6, 384>),
}

/// Common Spatial Patterns (CSP) for two-class motor imagery BCI.
///
/// CSP finds spatial filters that maximize the variance ratio between two classes.
/// Essential for motor imagery tasks like left vs right hand imagination.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create CSP with 8 channels, extract 4 spatial filters
/// csp = npy.AdaptiveCsp(channels=8, filters=4)
///
/// # Training: feed trials from each class
/// for trial in left_hand_trials:  # shape: (samples, channels)
///     csp.update_class1(trial)
///
/// for trial in right_hand_trials:
///     csp.update_class2(trial)
///
/// # Compute the spatial filters
/// csp.recompute_filters()
///
/// # Apply to new data - returns K features per sample
/// sample = np.random.randn(8).astype(np.float64)
/// features = csp.apply(sample)
/// ```
#[pyclass]
pub struct AdaptiveCsp {
    inner: AdaptiveCspInner,
    channels: usize,
    num_filters: usize,
}

#[pymethods]
impl AdaptiveCsp {
    /// Create a new CSP filter.
    ///
    /// Args:
    ///     channels (int): Number of EEG channels (4, 8, 16, 32, or 64).
    ///     filters (int): Number of spatial filters to extract (2, 4, or 6).
    ///     min_samples (int): Minimum samples per class before computing filters. Default: 100.
    ///     regularization (float): Regularization parameter for covariance. Default: 1e-6.
    ///
    /// Returns:
    ///     AdaptiveCsp: A new CSP filter instance.
    ///
    /// Raises:
    ///     ValueError: If channels/filters combination is not supported.
    #[new]
    #[pyo3(signature = (channels, filters, min_samples=100, regularization=1e-6))]
    fn new(
        channels: usize,
        filters: usize,
        min_samples: u64,
        regularization: f64,
    ) -> PyResult<Self> {
        let config = ZsUpdateConfig {
            min_samples,
            update_interval: 0, // Manual updates only
            regularization,
            max_eigen_iters: 30,
            eigen_tol: 1e-10,
        };

        let inner = match (channels, filters) {
            // 4 channels
            (4, 2) => AdaptiveCspInner::C4K2(ZsAdaptiveCsp::new(config)),
            (4, 4) => AdaptiveCspInner::C4K4(ZsAdaptiveCsp::new(config)),

            // 8 channels
            (8, 2) => AdaptiveCspInner::C8K2(ZsAdaptiveCsp::new(config)),
            (8, 4) => AdaptiveCspInner::C8K4(ZsAdaptiveCsp::new(config)),
            (8, 6) => AdaptiveCspInner::C8K6(ZsAdaptiveCsp::new(config)),

            // 16 channels
            (16, 2) => AdaptiveCspInner::C16K2(ZsAdaptiveCsp::new(config)),
            (16, 4) => AdaptiveCspInner::C16K4(ZsAdaptiveCsp::new(config)),
            (16, 6) => AdaptiveCspInner::C16K6(ZsAdaptiveCsp::new(config)),

            // 32 channels
            (32, 2) => AdaptiveCspInner::C32K2(ZsAdaptiveCsp::new(config)),
            (32, 4) => AdaptiveCspInner::C32K4(ZsAdaptiveCsp::new(config)),
            (32, 6) => AdaptiveCspInner::C32K6(ZsAdaptiveCsp::new(config)),

            // 64 channels
            (64, 2) => AdaptiveCspInner::C64K2(ZsAdaptiveCsp::new(config)),
            (64, 4) => AdaptiveCspInner::C64K4(ZsAdaptiveCsp::new(config)),
            (64, 6) => AdaptiveCspInner::C64K6(ZsAdaptiveCsp::new(config)),

            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported channels/filters combination: ({}, {}). \
                     Supported channels: 4, 8, 16, 32, 64. \
                     Supported filters: 2, 4, 6 (must be <= channels).",
                    channels, filters
                )))
            }
        };

        Ok(Self {
            inner,
            channels,
            num_filters: filters,
        })
    }

    /// Update with a class 1 trial (e.g., left hand).
    ///
    /// Args:
    ///     trial (np.ndarray): Trial data as 2D float64 array with shape (samples, channels).
    ///
    /// Example:
    ///     >>> trial = np.random.randn(500, 8).astype(np.float64)
    ///     >>> csp.update_class1(trial)
    fn update_class1(&mut self, trial: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = trial.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Trial has {} channels, expected {}",
                shape[1], self.channels
            )));
        }

        let data = trial.as_slice()?;

        macro_rules! update_class1 {
            ($csp:expr, $c:expr) => {{
                let mut samples: Vec<[f64; $c]> = Vec::with_capacity(shape[0]);
                for row in 0..shape[0] {
                    let mut sample = [0.0f64; $c];
                    for col in 0..$c {
                        sample[col] = data[row * $c + col];
                    }
                    samples.push(sample);
                }
                $csp.update_class1(&samples);
            }};
        }

        match &mut self.inner {
            AdaptiveCspInner::C4K2(csp) => update_class1!(csp, 4),
            AdaptiveCspInner::C4K4(csp) => update_class1!(csp, 4),
            AdaptiveCspInner::C8K2(csp) => update_class1!(csp, 8),
            AdaptiveCspInner::C8K4(csp) => update_class1!(csp, 8),
            AdaptiveCspInner::C8K6(csp) => update_class1!(csp, 8),
            AdaptiveCspInner::C16K2(csp) => update_class1!(csp, 16),
            AdaptiveCspInner::C16K4(csp) => update_class1!(csp, 16),
            AdaptiveCspInner::C16K6(csp) => update_class1!(csp, 16),
            AdaptiveCspInner::C32K2(csp) => update_class1!(csp, 32),
            AdaptiveCspInner::C32K4(csp) => update_class1!(csp, 32),
            AdaptiveCspInner::C32K6(csp) => update_class1!(csp, 32),
            AdaptiveCspInner::C64K2(csp) => update_class1!(csp, 64),
            AdaptiveCspInner::C64K4(csp) => update_class1!(csp, 64),
            AdaptiveCspInner::C64K6(csp) => update_class1!(csp, 64),
        }

        Ok(())
    }

    /// Update with a class 2 trial (e.g., right hand).
    ///
    /// Args:
    ///     trial (np.ndarray): Trial data as 2D float64 array with shape (samples, channels).
    ///
    /// Example:
    ///     >>> trial = np.random.randn(500, 8).astype(np.float64)
    ///     >>> csp.update_class2(trial)
    fn update_class2(&mut self, trial: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = trial.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Trial has {} channels, expected {}",
                shape[1], self.channels
            )));
        }

        let data = trial.as_slice()?;

        macro_rules! update_class2 {
            ($csp:expr, $c:expr) => {{
                let mut samples: Vec<[f64; $c]> = Vec::with_capacity(shape[0]);
                for row in 0..shape[0] {
                    let mut sample = [0.0f64; $c];
                    for col in 0..$c {
                        sample[col] = data[row * $c + col];
                    }
                    samples.push(sample);
                }
                $csp.update_class2(&samples);
            }};
        }

        match &mut self.inner {
            AdaptiveCspInner::C4K2(csp) => update_class2!(csp, 4),
            AdaptiveCspInner::C4K4(csp) => update_class2!(csp, 4),
            AdaptiveCspInner::C8K2(csp) => update_class2!(csp, 8),
            AdaptiveCspInner::C8K4(csp) => update_class2!(csp, 8),
            AdaptiveCspInner::C8K6(csp) => update_class2!(csp, 8),
            AdaptiveCspInner::C16K2(csp) => update_class2!(csp, 16),
            AdaptiveCspInner::C16K4(csp) => update_class2!(csp, 16),
            AdaptiveCspInner::C16K6(csp) => update_class2!(csp, 16),
            AdaptiveCspInner::C32K2(csp) => update_class2!(csp, 32),
            AdaptiveCspInner::C32K4(csp) => update_class2!(csp, 32),
            AdaptiveCspInner::C32K6(csp) => update_class2!(csp, 32),
            AdaptiveCspInner::C64K2(csp) => update_class2!(csp, 64),
            AdaptiveCspInner::C64K4(csp) => update_class2!(csp, 64),
            AdaptiveCspInner::C64K6(csp) => update_class2!(csp, 64),
        }

        Ok(())
    }

    /// Recompute CSP filters from accumulated training data.
    ///
    /// Must be called after training with update_class1/update_class2 before
    /// applying filters to new data.
    ///
    /// Raises:
    ///     ValueError: If not enough samples in either class.
    fn recompute_filters(&mut self) -> PyResult<()> {
        macro_rules! recompute {
            ($csp:expr) => {
                $csp.recompute_filters().map_err(|e| {
                    PyValueError::new_err(format!("Failed to compute CSP filters: {:?}", e))
                })
            };
        }

        match &mut self.inner {
            AdaptiveCspInner::C4K2(csp) => recompute!(csp),
            AdaptiveCspInner::C4K4(csp) => recompute!(csp),
            AdaptiveCspInner::C8K2(csp) => recompute!(csp),
            AdaptiveCspInner::C8K4(csp) => recompute!(csp),
            AdaptiveCspInner::C8K6(csp) => recompute!(csp),
            AdaptiveCspInner::C16K2(csp) => recompute!(csp),
            AdaptiveCspInner::C16K4(csp) => recompute!(csp),
            AdaptiveCspInner::C16K6(csp) => recompute!(csp),
            AdaptiveCspInner::C32K2(csp) => recompute!(csp),
            AdaptiveCspInner::C32K4(csp) => recompute!(csp),
            AdaptiveCspInner::C32K6(csp) => recompute!(csp),
            AdaptiveCspInner::C64K2(csp) => recompute!(csp),
            AdaptiveCspInner::C64K4(csp) => recompute!(csp),
            AdaptiveCspInner::C64K6(csp) => recompute!(csp),
        }
    }

    /// Apply CSP spatial filters to a sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Single sample as 1D float64 array with shape (channels,).
    ///
    /// Returns:
    ///     np.ndarray: CSP features as 1D float64 array with shape (filters,).
    ///
    /// Raises:
    ///     ValueError: If filters not yet computed.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        macro_rules! apply_csp {
            ($csp:expr, $c:expr, $k:expr) => {{
                let mut input = [0.0f64; $c];
                for (i, &v) in sample_slice.iter().enumerate() {
                    input[i] = v;
                }
                let output: [f64; $k] = $csp.apply(&input).map_err(|e| {
                    PyValueError::new_err(format!("CSP apply failed: {:?}", e))
                })?;
                Ok(PyArray1::from_vec(py, output.to_vec()))
            }};
        }

        match &self.inner {
            AdaptiveCspInner::C4K2(csp) => apply_csp!(csp, 4, 2),
            AdaptiveCspInner::C4K4(csp) => apply_csp!(csp, 4, 4),
            AdaptiveCspInner::C8K2(csp) => apply_csp!(csp, 8, 2),
            AdaptiveCspInner::C8K4(csp) => apply_csp!(csp, 8, 4),
            AdaptiveCspInner::C8K6(csp) => apply_csp!(csp, 8, 6),
            AdaptiveCspInner::C16K2(csp) => apply_csp!(csp, 16, 2),
            AdaptiveCspInner::C16K4(csp) => apply_csp!(csp, 16, 4),
            AdaptiveCspInner::C16K6(csp) => apply_csp!(csp, 16, 6),
            AdaptiveCspInner::C32K2(csp) => apply_csp!(csp, 32, 2),
            AdaptiveCspInner::C32K4(csp) => apply_csp!(csp, 32, 4),
            AdaptiveCspInner::C32K6(csp) => apply_csp!(csp, 32, 6),
            AdaptiveCspInner::C64K2(csp) => apply_csp!(csp, 64, 2),
            AdaptiveCspInner::C64K4(csp) => apply_csp!(csp, 64, 4),
            AdaptiveCspInner::C64K6(csp) => apply_csp!(csp, 64, 6),
        }
    }

    /// Apply CSP to a block of samples.
    ///
    /// Args:
    ///     block (np.ndarray): Block of samples as 2D float64 array (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: CSP features as 2D float64 array (samples, filters).
    fn apply_block<'py>(
        &self,
        py: Python<'py>,
        block: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = block.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Block has {} channels, expected {}",
                shape[1], self.channels
            )));
        }

        let data = block.as_slice()?;
        let num_samples = shape[0];
        let mut output = vec![0.0f64; num_samples * self.num_filters];

        macro_rules! apply_block {
            ($csp:expr, $c:expr, $k:expr) => {{
                for row in 0..num_samples {
                    let mut input = [0.0f64; $c];
                    for col in 0..$c {
                        input[col] = data[row * $c + col];
                    }
                    let features: [f64; $k] = $csp.apply(&input).map_err(|e| {
                        PyValueError::new_err(format!("CSP apply failed: {:?}", e))
                    })?;
                    for (k, &f) in features.iter().enumerate() {
                        output[row * $k + k] = f;
                    }
                }
            }};
        }

        match &self.inner {
            AdaptiveCspInner::C4K2(csp) => apply_block!(csp, 4, 2),
            AdaptiveCspInner::C4K4(csp) => apply_block!(csp, 4, 4),
            AdaptiveCspInner::C8K2(csp) => apply_block!(csp, 8, 2),
            AdaptiveCspInner::C8K4(csp) => apply_block!(csp, 8, 4),
            AdaptiveCspInner::C8K6(csp) => apply_block!(csp, 8, 6),
            AdaptiveCspInner::C16K2(csp) => apply_block!(csp, 16, 2),
            AdaptiveCspInner::C16K4(csp) => apply_block!(csp, 16, 4),
            AdaptiveCspInner::C16K6(csp) => apply_block!(csp, 16, 6),
            AdaptiveCspInner::C32K2(csp) => apply_block!(csp, 32, 2),
            AdaptiveCspInner::C32K4(csp) => apply_block!(csp, 32, 4),
            AdaptiveCspInner::C32K6(csp) => apply_block!(csp, 32, 6),
            AdaptiveCspInner::C64K2(csp) => apply_block!(csp, 64, 2),
            AdaptiveCspInner::C64K4(csp) => apply_block!(csp, 64, 4),
            AdaptiveCspInner::C64K6(csp) => apply_block!(csp, 64, 6),
        }

        Ok(PyArray2::from_vec2(
            py,
            &output
                .chunks(self.num_filters)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )?)
    }

    /// Get the computed spatial filters.
    ///
    /// Returns:
    ///     np.ndarray: Filter coefficients as 2D float64 array (filters, channels).
    ///     Returns None if filters not yet computed.
    fn filters<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        macro_rules! get_filters {
            ($csp:expr, $c:expr, $k:expr) => {{
                if let Some(f) = $csp.filters() {
                    let filter_rows: Vec<Vec<f64>> = (0..$k)
                        .map(|k| (0..$c).map(|c| f[k * $c + c]).collect())
                        .collect();
                    Some(PyArray2::from_vec2(py, &filter_rows)?)
                } else {
                    None
                }
            }};
        }

        let result = match &self.inner {
            AdaptiveCspInner::C4K2(csp) => get_filters!(csp, 4, 2),
            AdaptiveCspInner::C4K4(csp) => get_filters!(csp, 4, 4),
            AdaptiveCspInner::C8K2(csp) => get_filters!(csp, 8, 2),
            AdaptiveCspInner::C8K4(csp) => get_filters!(csp, 8, 4),
            AdaptiveCspInner::C8K6(csp) => get_filters!(csp, 8, 6),
            AdaptiveCspInner::C16K2(csp) => get_filters!(csp, 16, 2),
            AdaptiveCspInner::C16K4(csp) => get_filters!(csp, 16, 4),
            AdaptiveCspInner::C16K6(csp) => get_filters!(csp, 16, 6),
            AdaptiveCspInner::C32K2(csp) => get_filters!(csp, 32, 2),
            AdaptiveCspInner::C32K4(csp) => get_filters!(csp, 32, 4),
            AdaptiveCspInner::C32K6(csp) => get_filters!(csp, 32, 6),
            AdaptiveCspInner::C64K2(csp) => get_filters!(csp, 64, 2),
            AdaptiveCspInner::C64K4(csp) => get_filters!(csp, 64, 4),
            AdaptiveCspInner::C64K6(csp) => get_filters!(csp, 64, 6),
        };

        Ok(result)
    }

    /// Check if filters are ready for use.
    #[getter]
    fn is_ready(&self) -> bool {
        macro_rules! check_ready {
            ($csp:expr) => {
                $csp.is_ready()
            };
        }

        match &self.inner {
            AdaptiveCspInner::C4K2(csp) => check_ready!(csp),
            AdaptiveCspInner::C4K4(csp) => check_ready!(csp),
            AdaptiveCspInner::C8K2(csp) => check_ready!(csp),
            AdaptiveCspInner::C8K4(csp) => check_ready!(csp),
            AdaptiveCspInner::C8K6(csp) => check_ready!(csp),
            AdaptiveCspInner::C16K2(csp) => check_ready!(csp),
            AdaptiveCspInner::C16K4(csp) => check_ready!(csp),
            AdaptiveCspInner::C16K6(csp) => check_ready!(csp),
            AdaptiveCspInner::C32K2(csp) => check_ready!(csp),
            AdaptiveCspInner::C32K4(csp) => check_ready!(csp),
            AdaptiveCspInner::C32K6(csp) => check_ready!(csp),
            AdaptiveCspInner::C64K2(csp) => check_ready!(csp),
            AdaptiveCspInner::C64K4(csp) => check_ready!(csp),
            AdaptiveCspInner::C64K6(csp) => check_ready!(csp),
        }
    }

    /// Get the number of samples accumulated for class 1.
    #[getter]
    fn class1_count(&self) -> u64 {
        macro_rules! get_count {
            ($csp:expr) => {
                $csp.class1_count()
            };
        }

        match &self.inner {
            AdaptiveCspInner::C4K2(csp) => get_count!(csp),
            AdaptiveCspInner::C4K4(csp) => get_count!(csp),
            AdaptiveCspInner::C8K2(csp) => get_count!(csp),
            AdaptiveCspInner::C8K4(csp) => get_count!(csp),
            AdaptiveCspInner::C8K6(csp) => get_count!(csp),
            AdaptiveCspInner::C16K2(csp) => get_count!(csp),
            AdaptiveCspInner::C16K4(csp) => get_count!(csp),
            AdaptiveCspInner::C16K6(csp) => get_count!(csp),
            AdaptiveCspInner::C32K2(csp) => get_count!(csp),
            AdaptiveCspInner::C32K4(csp) => get_count!(csp),
            AdaptiveCspInner::C32K6(csp) => get_count!(csp),
            AdaptiveCspInner::C64K2(csp) => get_count!(csp),
            AdaptiveCspInner::C64K4(csp) => get_count!(csp),
            AdaptiveCspInner::C64K6(csp) => get_count!(csp),
        }
    }

    /// Get the number of samples accumulated for class 2.
    #[getter]
    fn class2_count(&self) -> u64 {
        macro_rules! get_count {
            ($csp:expr) => {
                $csp.class2_count()
            };
        }

        match &self.inner {
            AdaptiveCspInner::C4K2(csp) => get_count!(csp),
            AdaptiveCspInner::C4K4(csp) => get_count!(csp),
            AdaptiveCspInner::C8K2(csp) => get_count!(csp),
            AdaptiveCspInner::C8K4(csp) => get_count!(csp),
            AdaptiveCspInner::C8K6(csp) => get_count!(csp),
            AdaptiveCspInner::C16K2(csp) => get_count!(csp),
            AdaptiveCspInner::C16K4(csp) => get_count!(csp),
            AdaptiveCspInner::C16K6(csp) => get_count!(csp),
            AdaptiveCspInner::C32K2(csp) => get_count!(csp),
            AdaptiveCspInner::C32K4(csp) => get_count!(csp),
            AdaptiveCspInner::C32K6(csp) => get_count!(csp),
            AdaptiveCspInner::C64K2(csp) => get_count!(csp),
            AdaptiveCspInner::C64K4(csp) => get_count!(csp),
            AdaptiveCspInner::C64K6(csp) => get_count!(csp),
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the number of filters.
    #[getter]
    fn num_filters(&self) -> usize {
        self.num_filters
    }

    /// Reset all state (training data and filters).
    fn reset(&mut self) {
        macro_rules! do_reset {
            ($csp:expr) => {
                $csp.reset()
            };
        }

        match &mut self.inner {
            AdaptiveCspInner::C4K2(csp) => do_reset!(csp),
            AdaptiveCspInner::C4K4(csp) => do_reset!(csp),
            AdaptiveCspInner::C8K2(csp) => do_reset!(csp),
            AdaptiveCspInner::C8K4(csp) => do_reset!(csp),
            AdaptiveCspInner::C8K6(csp) => do_reset!(csp),
            AdaptiveCspInner::C16K2(csp) => do_reset!(csp),
            AdaptiveCspInner::C16K4(csp) => do_reset!(csp),
            AdaptiveCspInner::C16K6(csp) => do_reset!(csp),
            AdaptiveCspInner::C32K2(csp) => do_reset!(csp),
            AdaptiveCspInner::C32K4(csp) => do_reset!(csp),
            AdaptiveCspInner::C32K6(csp) => do_reset!(csp),
            AdaptiveCspInner::C64K2(csp) => do_reset!(csp),
            AdaptiveCspInner::C64K4(csp) => do_reset!(csp),
            AdaptiveCspInner::C64K6(csp) => do_reset!(csp),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaptiveCsp(channels={}, filters={}, is_ready={})",
            self.channels,
            self.num_filters,
            self.is_ready()
        )
    }
}
