//! Python bindings for statistics primitives.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
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
