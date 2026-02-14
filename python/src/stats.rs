//! Python bindings for statistics primitives.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::OnlineCov as ZsOnlineCov;
use zerostone::OnlineStats as ZsOnlineStats;

/// Online statistics calculator using Welford's algorithm.
///
/// Computes running mean, variance, and standard deviation in a numerically
/// stable way without storing all samples. Memory usage is O(1).
///
/// # Example
/// ```python
/// import npyci as npy
///
/// stats = npy.OnlineStats()
///
/// # Update with individual values
/// for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
///     stats.update(x)
///
/// print(f"Mean: {stats.mean}")     # 3.0
/// print(f"Std: {stats.std}")       # ~1.58
/// print(f"Count: {stats.count}")   # 5
///
/// # Or update with a batch of values
/// import numpy as np
/// stats.reset()
/// stats.update_batch(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
/// ```
#[pyclass]
pub struct OnlineStats {
    inner: ZsOnlineStats<1>,
}

#[pymethods]
impl OnlineStats {
    /// Create a new online statistics calculator.
    ///
    /// Returns:
    ///     OnlineStats: A new statistics calculator with zero state.
    ///
    /// Example:
    ///     >>> stats = OnlineStats()
    #[new]
    fn new() -> Self {
        Self {
            inner: ZsOnlineStats::new(),
        }
    }

    /// Update statistics with a single value.
    ///
    /// Args:
    ///     value (float): The value to add to the statistics.
    ///
    /// Example:
    ///     >>> stats.update(42.0)
    fn update(&mut self, value: f64) {
        self.inner.update(&[value]);
    }

    /// Update statistics with a batch of values.
    ///
    /// More efficient than calling update() repeatedly for large arrays.
    ///
    /// Args:
    ///     values (np.ndarray): 1D array of float64 values.
    ///
    /// Example:
    ///     >>> stats.update_batch(np.array([1.0, 2.0, 3.0]))
    fn update_batch(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        for &value in slice {
            self.inner.update(&[value]);
        }
        Ok(())
    }

    /// Get the current mean.
    ///
    /// Returns 0.0 if no samples have been added.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()[0]
    }

    /// Get the sample variance (normalized by n-1).
    ///
    /// Returns 0.0 if fewer than 2 samples have been added.
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()[0]
    }

    /// Get the sample standard deviation.
    ///
    /// This is the square root of the sample variance.
    /// Returns 0.0 if fewer than 2 samples have been added.
    #[getter]
    fn std(&self) -> f64 {
        self.inner.std_dev()[0]
    }

    /// Get the number of samples processed.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Reset statistics to initial zero state.
    ///
    /// Example:
    ///     >>> stats.reset()
    ///     >>> stats.count
    ///     0
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineStats(count={}, mean={:.4}, std={:.4})",
            self.count(),
            self.mean(),
            self.std()
        )
    }
}

// --- OnlineCov enum dispatch ---

enum OnlineCovInner {
    C4(ZsOnlineCov<4, 16>),
    C8(ZsOnlineCov<8, 64>),
    C16(ZsOnlineCov<16, 256>),
    C32(ZsOnlineCov<32, 1024>),
    C64(ZsOnlineCov<64, 4096>),
}

/// Online covariance matrix estimator.
///
/// Computes running mean, covariance, and correlation matrices
/// using Welford's online algorithm. Memory usage is O(C^2).
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// cov = npy.OnlineCov(channels=4)
/// for _ in range(100):
///     sample = np.random.randn(4)
///     cov.update(sample)
/// print(cov.mean)        # shape (4,)
/// print(cov.covariance)  # shape (4, 4)
/// ```
#[pyclass]
pub struct OnlineCov {
    inner: OnlineCovInner,
    channels: usize,
}

#[pymethods]
impl OnlineCov {
    /// Create a new online covariance estimator.
    ///
    /// Args:
    ///     channels (int): Number of channels. Must be 4, 8, 16, 32, or 64.
    #[new]
    fn new(channels: usize) -> PyResult<Self> {
        let inner = match channels {
            4 => OnlineCovInner::C4(ZsOnlineCov::new()),
            8 => OnlineCovInner::C8(ZsOnlineCov::new()),
            16 => OnlineCovInner::C16(ZsOnlineCov::new()),
            32 => OnlineCovInner::C32(ZsOnlineCov::new()),
            64 => OnlineCovInner::C64(ZsOnlineCov::new()),
            _ => {
                return Err(PyValueError::new_err(
                    "channels must be 4, 8, 16, 32, or 64",
                ))
            }
        };
        Ok(Self { inner, channels })
    }

    /// Update with a new sample vector.
    ///
    /// Args:
    ///     sample (np.ndarray): 1D float64 array of length `channels`.
    fn update(&mut self, sample: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = sample.as_slice()?;
        if slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} channels, got {}",
                self.channels,
                slice.len()
            )));
        }

        macro_rules! update {
            ($cov:expr, $c:expr) => {{
                let mut arr = [0.0f64; $c];
                arr.copy_from_slice(slice);
                $cov.update(&arr);
            }};
        }
        match &mut self.inner {
            OnlineCovInner::C4(c) => update!(c, 4),
            OnlineCovInner::C8(c) => update!(c, 8),
            OnlineCovInner::C16(c) => update!(c, 16),
            OnlineCovInner::C32(c) => update!(c, 32),
            OnlineCovInner::C64(c) => update!(c, 64),
        }
        Ok(())
    }

    /// Get the current mean vector.
    ///
    /// Returns:
    ///     np.ndarray: 1D float64 array of shape (channels,).
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        match &self.inner {
            OnlineCovInner::C4(c) => PyArray1::from_slice(py, c.mean()),
            OnlineCovInner::C8(c) => PyArray1::from_slice(py, c.mean()),
            OnlineCovInner::C16(c) => PyArray1::from_slice(py, c.mean()),
            OnlineCovInner::C32(c) => PyArray1::from_slice(py, c.mean()),
            OnlineCovInner::C64(c) => PyArray1::from_slice(py, c.mean()),
        }
    }

    /// Get the covariance matrix.
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (channels, channels).
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        macro_rules! cov_matrix {
            ($cov:expr, $c:expr) => {{
                let flat = $cov.covariance();
                let arr = Array2::from_shape_vec(($c, $c), flat.to_vec()).unwrap();
                PyArray2::from_owned_array(py, arr)
            }};
        }
        match &self.inner {
            OnlineCovInner::C4(c) => cov_matrix!(c, 4),
            OnlineCovInner::C8(c) => cov_matrix!(c, 8),
            OnlineCovInner::C16(c) => cov_matrix!(c, 16),
            OnlineCovInner::C32(c) => cov_matrix!(c, 32),
            OnlineCovInner::C64(c) => cov_matrix!(c, 64),
        }
    }

    /// Get the correlation matrix.
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (channels, channels).
    #[getter]
    fn correlation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        macro_rules! corr_matrix {
            ($cov:expr, $c:expr) => {{
                let flat = $cov.correlation();
                let arr = Array2::from_shape_vec(($c, $c), flat.to_vec()).unwrap();
                PyArray2::from_owned_array(py, arr)
            }};
        }
        match &self.inner {
            OnlineCovInner::C4(c) => corr_matrix!(c, 4),
            OnlineCovInner::C8(c) => corr_matrix!(c, 8),
            OnlineCovInner::C16(c) => corr_matrix!(c, 16),
            OnlineCovInner::C32(c) => corr_matrix!(c, 32),
            OnlineCovInner::C64(c) => corr_matrix!(c, 64),
        }
    }

    /// Get a single covariance matrix element.
    ///
    /// Args:
    ///     i (int): Row index.
    ///     j (int): Column index.
    ///
    /// Returns:
    ///     float: Covariance between channels i and j.
    fn get(&self, i: usize, j: usize) -> f64 {
        match &self.inner {
            OnlineCovInner::C4(c) => c.get(i, j),
            OnlineCovInner::C8(c) => c.get(i, j),
            OnlineCovInner::C16(c) => c.get(i, j),
            OnlineCovInner::C32(c) => c.get(i, j),
            OnlineCovInner::C64(c) => c.get(i, j),
        }
    }

    /// Number of samples processed.
    #[getter]
    fn count(&self) -> u64 {
        match &self.inner {
            OnlineCovInner::C4(c) => c.count(),
            OnlineCovInner::C8(c) => c.count(),
            OnlineCovInner::C16(c) => c.count(),
            OnlineCovInner::C32(c) => c.count(),
            OnlineCovInner::C64(c) => c.count(),
        }
    }

    /// Number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Reset the estimator state.
    fn reset(&mut self) {
        match &mut self.inner {
            OnlineCovInner::C4(c) => c.reset(),
            OnlineCovInner::C8(c) => c.reset(),
            OnlineCovInner::C16(c) => c.reset(),
            OnlineCovInner::C32(c) => c.reset(),
            OnlineCovInner::C64(c) => c.reset(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineCov(channels={}, count={})",
            self.channels,
            self.count()
        )
    }
}
