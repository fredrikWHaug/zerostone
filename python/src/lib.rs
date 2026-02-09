use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::Bound;
use ::zerostone::{BiquadCoeffs, IirFilter as ZsIirFilter};

mod filters;
mod spatial;
mod stats;
mod utils;

use filters::{AcCoupler, FirFilter, MedianFilter};
use spatial::{CAR, SurfaceLaplacian};
use stats::OnlineStats;

/// IIR (Infinite Impulse Response) filter with cascaded biquad sections.
///
/// This filter implements a 4th-order Butterworth filter (2 cascaded biquad sections)
/// for high-quality lowpass, highpass, and bandpass filtering.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create a lowpass filter at 30 Hz (1000 Hz sample rate)
/// lpf = npy.IirFilter.butterworth_lowpass(1000.0, 30.0)
///
/// # Process a signal
/// signal = np.random.randn(1000).astype(np.float32)
/// filtered = lpf.process(signal)
/// ```
#[pyclass]
struct IirFilter {
    // Using 2 sections for 4th-order filter (standard for Butterworth)
    filter: ZsIirFilter<2>,
    sample_rate: f32,
}

#[pymethods]
impl IirFilter {
    /// Create a Butterworth lowpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     cutoff (float): Cutoff frequency in Hz
    ///
    /// Returns:
    ///     IirFilter: A 4th-order Butterworth lowpass filter
    ///
    /// Example:
    ///     >>> lpf = IirFilter.butterworth_lowpass(1000.0, 40.0)
    #[staticmethod]
    fn butterworth_lowpass(sample_rate: f32, cutoff: f32) -> PyResult<Self> {
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequency must be between 0 and {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }

        let biquad = BiquadCoeffs::butterworth_lowpass(sample_rate, cutoff);
        let filter = ZsIirFilter::new([biquad, biquad]);

        Ok(Self {
            filter,
            sample_rate,
        })
    }

    /// Create a Butterworth highpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     cutoff (float): Cutoff frequency in Hz
    ///
    /// Returns:
    ///     IirFilter: A 4th-order Butterworth highpass filter
    ///
    /// Example:
    ///     >>> hpf = IirFilter.butterworth_highpass(1000.0, 1.0)
    #[staticmethod]
    fn butterworth_highpass(sample_rate: f32, cutoff: f32) -> PyResult<Self> {
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequency must be between 0 and {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }

        let biquad = BiquadCoeffs::butterworth_highpass(sample_rate, cutoff);
        let filter = ZsIirFilter::new([biquad, biquad]);

        Ok(Self {
            filter,
            sample_rate,
        })
    }

    /// Create a Butterworth bandpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     low_cutoff (float): Lower cutoff frequency in Hz
    ///     high_cutoff (float): Upper cutoff frequency in Hz
    ///
    /// Returns:
    ///     IirFilter: A 4th-order Butterworth bandpass filter
    ///
    /// Example:
    ///     >>> bpf = IirFilter.butterworth_bandpass(1000.0, 8.0, 12.0)
    #[staticmethod]
    fn butterworth_bandpass(sample_rate: f32, low_cutoff: f32, high_cutoff: f32) -> PyResult<Self> {
        if low_cutoff <= 0.0 || high_cutoff >= sample_rate / 2.0 || low_cutoff >= high_cutoff {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequencies must satisfy: 0 < low_cutoff < high_cutoff < {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }

        let biquad = BiquadCoeffs::butterworth_bandpass(
            sample_rate,
            low_cutoff,
            high_cutoff,
        );
        let filter = ZsIirFilter::new([biquad, biquad]);

        Ok(Self {
            filter,
            sample_rate,
        })
    }

    /// Process a signal through the filter.
    ///
    /// Args:
    ///     input (np.ndarray): Input signal as 1D float32 numpy array
    ///
    /// Returns:
    ///     np.ndarray: Filtered signal as 1D float32 numpy array
    ///
    /// Example:
    ///     >>> signal = np.random.randn(1000).astype(np.float32)
    ///     >>> filtered = lpf.process(signal)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice()?;
        let mut output = vec![0.0f32; input_slice.len()];

        // Process each sample through the filter
        for (i, &sample) in input_slice.iter().enumerate() {
            output[i] = self.filter.process_sample(sample);
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Reset the filter state (clear delay lines).
    ///
    /// This is useful when processing discontinuous segments of data.
    fn reset(&mut self) {
        self.filter.reset();
    }

    /// Get the sample rate of the filter.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn __repr__(&self) -> String {
        format!("IirFilter(sample_rate={} Hz)", self.sample_rate)
    }
}

/// npyci: High-performance signal processing for BCI research.
///
/// This module provides zero-allocation, real-time signal processing primitives
/// implemented in Rust for maximum performance.
#[pymodule]
fn npyci(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IirFilter>()?;
    m.add_class::<FirFilter>()?;
    m.add_class::<AcCoupler>()?;
    m.add_class::<MedianFilter>()?;
    m.add_class::<CAR>()?;
    m.add_class::<SurfaceLaplacian>()?;
    m.add_class::<OnlineStats>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
