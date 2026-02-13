//! Python bindings for streaming percentile estimation.
//!
//! Provides the P² algorithm for computing percentiles in a streaming fashion.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::StreamingPercentile as ZsStreamingPercentile;

// ============================================================================
// StreamingPercentile
// ============================================================================

/// Internal enum for handling different channel counts.
enum StreamingPercentileInner {
    C1(ZsStreamingPercentile<1>),
    C4(ZsStreamingPercentile<4>),
    C8(ZsStreamingPercentile<8>),
    C16(ZsStreamingPercentile<16>),
    C32(ZsStreamingPercentile<32>),
    C64(ZsStreamingPercentile<64>),
}

/// Streaming percentile estimator using the P² algorithm.
///
/// Estimates percentiles over streaming multi-channel data without storing
/// observations. Uses O(1) memory per channel regardless of sample count.
/// Ideal for real-time baseline estimation in BCI applications.
///
/// # Algorithm
///
/// The P² algorithm maintains 5 markers to track the desired percentile.
/// After 5 samples, it provides estimates that improve with more data.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create 8th percentile estimator for baseline (common in calcium imaging)
/// est = npy.StreamingPercentile(channels=4, percentile=0.08)
///
/// # Feed streaming data
/// for _ in range(1000):
///     sample = np.random.randn(4).astype(np.float64)
///     est.update(sample)
///
/// # Get estimate (returns None if < 5 samples)
/// baseline = est.percentile()  # Array of 4 values
/// ```
#[pyclass]
pub struct StreamingPercentile {
    inner: StreamingPercentileInner,
    channels: usize,
    target_percentile: f64,
}

#[pymethods]
impl StreamingPercentile {
    /// Create a new streaming percentile estimator.
    ///
    /// Args:
    ///     channels (int): Number of channels (1, 4, 8, 16, 32, or 64).
    ///     percentile (float): Target percentile as fraction (0.0 to 1.0 exclusive).
    ///         Example: 0.5 for median, 0.08 for 8th percentile.
    ///
    /// Returns:
    ///     StreamingPercentile: A new estimator.
    #[new]
    fn new(channels: usize, percentile: f64) -> PyResult<Self> {
        if percentile <= 0.0 || percentile >= 1.0 {
            return Err(PyValueError::new_err(
                "percentile must be in (0.0, 1.0) exclusive",
            ));
        }

        let inner = match channels {
            1 => StreamingPercentileInner::C1(ZsStreamingPercentile::new(percentile)),
            4 => StreamingPercentileInner::C4(ZsStreamingPercentile::new(percentile)),
            8 => StreamingPercentileInner::C8(ZsStreamingPercentile::new(percentile)),
            16 => StreamingPercentileInner::C16(ZsStreamingPercentile::new(percentile)),
            32 => StreamingPercentileInner::C32(ZsStreamingPercentile::new(percentile)),
            64 => StreamingPercentileInner::C64(ZsStreamingPercentile::new(percentile)),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "channels must be 1, 4, 8, 16, 32, or 64, got {}",
                    channels
                )))
            }
        };

        Ok(Self {
            inner,
            channels,
            target_percentile: percentile,
        })
    }

    /// Create a median estimator (50th percentile).
    #[staticmethod]
    fn median(channels: usize) -> PyResult<Self> {
        Self::new(channels, 0.5)
    }

    /// Update the estimator with a new sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float64 array with shape (channels,).
    fn update(&mut self, sample: PyReadonlyArray1<f64>) -> PyResult<()> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        macro_rules! do_update {
            ($est:expr, $c:expr) => {{
                let input: [f64; $c] = sample_slice.try_into().unwrap();
                $est.update(&input);
            }};
        }

        match &mut self.inner {
            StreamingPercentileInner::C1(est) => do_update!(est, 1),
            StreamingPercentileInner::C4(est) => do_update!(est, 4),
            StreamingPercentileInner::C8(est) => do_update!(est, 8),
            StreamingPercentileInner::C16(est) => do_update!(est, 16),
            StreamingPercentileInner::C32(est) => do_update!(est, 32),
            StreamingPercentileInner::C64(est) => do_update!(est, 64),
        }

        Ok(())
    }

    /// Get the current percentile estimate.
    ///
    /// Returns:
    ///     np.ndarray: Percentile estimate for each channel, or None if < 5 samples.
    fn percentile<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        macro_rules! get_percentile {
            ($est:expr) => {
                $est.percentile().map(|v| v.to_vec())
            };
        }

        let vals = match &self.inner {
            StreamingPercentileInner::C1(est) => get_percentile!(est),
            StreamingPercentileInner::C4(est) => get_percentile!(est),
            StreamingPercentileInner::C8(est) => get_percentile!(est),
            StreamingPercentileInner::C16(est) => get_percentile!(est),
            StreamingPercentileInner::C32(est) => get_percentile!(est),
            StreamingPercentileInner::C64(est) => get_percentile!(est),
        };

        vals.map(|v| PyArray1::from_vec(py, v))
    }

    /// Get the current minimum value for each channel.
    ///
    /// Returns:
    ///     np.ndarray: Minimum values, or None if < 5 samples.
    fn min<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        macro_rules! get_min {
            ($est:expr) => {
                $est.min().map(|v| v.to_vec())
            };
        }

        let vals = match &self.inner {
            StreamingPercentileInner::C1(est) => get_min!(est),
            StreamingPercentileInner::C4(est) => get_min!(est),
            StreamingPercentileInner::C8(est) => get_min!(est),
            StreamingPercentileInner::C16(est) => get_min!(est),
            StreamingPercentileInner::C32(est) => get_min!(est),
            StreamingPercentileInner::C64(est) => get_min!(est),
        };

        vals.map(|v| PyArray1::from_vec(py, v))
    }

    /// Get the current maximum value for each channel.
    ///
    /// Returns:
    ///     np.ndarray: Maximum values, or None if < 5 samples.
    fn max<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        macro_rules! get_max {
            ($est:expr) => {
                $est.max().map(|v| v.to_vec())
            };
        }

        let vals = match &self.inner {
            StreamingPercentileInner::C1(est) => get_max!(est),
            StreamingPercentileInner::C4(est) => get_max!(est),
            StreamingPercentileInner::C8(est) => get_max!(est),
            StreamingPercentileInner::C16(est) => get_max!(est),
            StreamingPercentileInner::C32(est) => get_max!(est),
            StreamingPercentileInner::C64(est) => get_max!(est),
        };

        vals.map(|v| PyArray1::from_vec(py, v))
    }

    /// Reset the estimator to initial state.
    fn reset(&mut self) {
        match &mut self.inner {
            StreamingPercentileInner::C1(est) => est.reset(),
            StreamingPercentileInner::C4(est) => est.reset(),
            StreamingPercentileInner::C8(est) => est.reset(),
            StreamingPercentileInner::C16(est) => est.reset(),
            StreamingPercentileInner::C32(est) => est.reset(),
            StreamingPercentileInner::C64(est) => est.reset(),
        }
    }

    /// Check if estimator is initialized (has >= 5 samples).
    #[getter]
    fn is_initialized(&self) -> bool {
        match &self.inner {
            StreamingPercentileInner::C1(est) => est.is_initialized(),
            StreamingPercentileInner::C4(est) => est.is_initialized(),
            StreamingPercentileInner::C8(est) => est.is_initialized(),
            StreamingPercentileInner::C16(est) => est.is_initialized(),
            StreamingPercentileInner::C32(est) => est.is_initialized(),
            StreamingPercentileInner::C64(est) => est.is_initialized(),
        }
    }

    /// Get the number of samples processed.
    #[getter]
    fn count(&self) -> u64 {
        match &self.inner {
            StreamingPercentileInner::C1(est) => est.count(),
            StreamingPercentileInner::C4(est) => est.count(),
            StreamingPercentileInner::C8(est) => est.count(),
            StreamingPercentileInner::C16(est) => est.count(),
            StreamingPercentileInner::C32(est) => est.count(),
            StreamingPercentileInner::C64(est) => est.count(),
        }
    }

    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    fn target(&self) -> f64 {
        self.target_percentile
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingPercentile(channels={}, percentile={}, count={}, initialized={})",
            self.channels,
            self.target_percentile,
            self.count(),
            self.is_initialized()
        )
    }
}
