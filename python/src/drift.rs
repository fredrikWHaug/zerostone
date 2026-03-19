//! Python bindings for drift estimation.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// DriftEstimator for tracking position drift in chronic recordings.
///
/// Bins spike positions over time and fits a linear regression to
/// estimate the drift rate. Use `correct_position()` to remove drift.
///
/// Args:
///     bin_duration_samples (int): Bin width in samples. Default: 1000.
///
/// Example:
///     >>> est = zbci.DriftEstimator(1000)
///     >>> for i in range(8):
///     ...     est.add_spike(i * 1000 + 500, 100.0 + i * 10.0)
///     >>> est.fit()
///     >>> est.slope > 0
///     True
#[pyclass]
pub struct DriftEstimator {
    inner: zerostone::drift::DriftEstimator<256>,
}

#[pymethods]
impl DriftEstimator {
    #[new]
    #[pyo3(signature = (bin_duration_samples=1000))]
    fn new(bin_duration_samples: usize) -> PyResult<Self> {
        if bin_duration_samples == 0 {
            return Err(PyValueError::new_err("bin_duration_samples must be > 0"));
        }
        Ok(Self {
            inner: zerostone::drift::DriftEstimator::<256>::new(bin_duration_samples),
        })
    }

    /// Add a spike's position and time.
    fn add_spike(&mut self, sample_index: usize, position_y: f64) {
        self.inner.add_spike(sample_index, position_y);
    }

    /// Fit linear regression on binned positions.
    fn fit(&mut self) {
        self.inner.fit();
    }

    /// Estimate drift at a given sample index.
    fn estimate_drift(&self, sample_index: usize) -> f64 {
        self.inner.estimate_drift(sample_index)
    }

    /// Correct a position by removing estimated drift.
    fn correct_position(&self, sample_index: usize, position_y: f64) -> f64 {
        self.inner.correct_position(sample_index, position_y)
    }

    #[getter]
    fn slope(&self) -> f64 {
        self.inner.slope()
    }

    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    #[getter]
    fn n_bins_used(&self) -> usize {
        self.inner.n_bins_used()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "DriftEstimator(fitted={}, slope={:.6}, n_bins={})",
            self.inner.is_fitted(),
            self.inner.slope(),
            self.inner.n_bins_used()
        )
    }
}

/// Estimate drift from arrays of spike times and positions (batch mode).
///
/// Args:
///     sample_indices (np.ndarray): 1D int64 array of spike sample indices.
///     positions_y (np.ndarray): 1D float64 array of spike y-positions.
///     bin_duration_samples (int): Bin width in samples. Default: 1000.
///     max_bins (int): Maximum number of bins. Default: 256.
///
/// Returns:
///     tuple or None: (slope, intercept) if at least 2 bins have data, else None.
///
/// Example:
///     >>> indices = np.array([100, 1100, 2100, 3100], dtype=np.int64)
///     >>> positions = np.array([50.0, 60.0, 70.0, 80.0])
///     >>> result = zbci.estimate_drift_from_positions(indices, positions)
#[pyfunction]
#[pyo3(signature = (sample_indices, positions_y, bin_duration_samples=1000, max_bins=256))]
fn estimate_drift_from_positions(
    sample_indices: PyReadonlyArray1<i64>,
    positions_y: PyReadonlyArray1<f64>,
    bin_duration_samples: usize,
    max_bins: usize,
) -> PyResult<Option<(f64, f64)>> {
    let indices_slice = sample_indices.as_slice()?;
    let positions_slice = positions_y.as_slice()?;

    let indices_usize: Vec<usize> = indices_slice.iter().map(|&x| x as usize).collect();

    Ok(zerostone::drift::estimate_drift_from_positions(
        &indices_usize,
        positions_slice,
        bin_duration_samples,
        max_bins,
    ))
}

/// Register drift estimation functions and classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DriftEstimator>()?;
    m.add_function(wrap_pyfunction!(estimate_drift_from_positions, m)?)?;
    Ok(())
}
