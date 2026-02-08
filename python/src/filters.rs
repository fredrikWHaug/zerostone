//! Python bindings for filter primitives.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{AcCoupler as ZsAcCoupler, FirFilter as ZsFirFilter, MedianFilter as ZsMedianFilter};

// ============================================================================
// FirFilter
// ============================================================================

/// Internal enum for handling different FIR filter tap counts.
enum FirFilterInner {
    Taps8(ZsFirFilter<8>),
    Taps16(ZsFirFilter<16>),
    Taps32(ZsFirFilter<32>),
    Taps64(ZsFirFilter<64>),
    /// Dynamic implementation for non-standard tap counts
    Dynamic {
        coeffs: Vec<f32>,
        delay_line: Vec<f32>,
        index: usize,
    },
}

/// FIR (Finite Impulse Response) filter.
///
/// Implements direct-form FIR filtering for linear phase response and
/// guaranteed stability. Supports arbitrary filter coefficients.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create a 5-tap moving average filter
/// fir = npy.FirFilter.moving_average(5)
///
/// # Or create with custom coefficients
/// fir = npy.FirFilter(taps=[0.1, 0.2, 0.4, 0.2, 0.1])
///
/// # Process a signal
/// signal = np.random.randn(1000).astype(np.float32)
/// filtered = fir.process(signal)
/// ```
#[pyclass]
pub struct FirFilter {
    inner: FirFilterInner,
    num_taps: usize,
}

#[pymethods]
impl FirFilter {
    /// Create a new FIR filter with the given tap coefficients.
    ///
    /// Args:
    ///     taps (list[float]): Filter tap coefficients. Length must be >= 1.
    ///
    /// Returns:
    ///     FirFilter: A new FIR filter instance.
    ///
    /// Example:
    ///     >>> fir = FirFilter(taps=[0.2, 0.2, 0.2, 0.2, 0.2])
    #[new]
    #[pyo3(signature = (taps))]
    fn new(taps: Vec<f32>) -> PyResult<Self> {
        let num_taps = taps.len();
        if num_taps == 0 {
            return Err(PyValueError::new_err("taps must have at least 1 element"));
        }

        let inner = match num_taps {
            8 => {
                let arr: [f32; 8] = taps.try_into().unwrap();
                FirFilterInner::Taps8(ZsFirFilter::new(arr))
            }
            16 => {
                let arr: [f32; 16] = taps.try_into().unwrap();
                FirFilterInner::Taps16(ZsFirFilter::new(arr))
            }
            32 => {
                let arr: [f32; 32] = taps.try_into().unwrap();
                FirFilterInner::Taps32(ZsFirFilter::new(arr))
            }
            64 => {
                let arr: [f32; 64] = taps.try_into().unwrap();
                FirFilterInner::Taps64(ZsFirFilter::new(arr))
            }
            _ => FirFilterInner::Dynamic {
                coeffs: taps,
                delay_line: vec![0.0; num_taps],
                index: 0,
            },
        };

        Ok(Self { inner, num_taps })
    }

    /// Create a moving average filter with equal weights.
    ///
    /// Args:
    ///     size (int): Number of taps (window size). Must be 1-64.
    ///
    /// Returns:
    ///     FirFilter: A moving average filter.
    ///
    /// Example:
    ///     >>> fir = FirFilter.moving_average(5)
    #[staticmethod]
    fn moving_average(size: usize) -> PyResult<Self> {
        if size == 0 || size > 64 {
            return Err(PyValueError::new_err(
                "size must be between 1 and 64 for moving_average",
            ));
        }

        let weight = 1.0 / size as f32;
        let taps = vec![weight; size];
        Self::new(taps)
    }

    /// Process a signal through the filter.
    ///
    /// Args:
    ///     input (np.ndarray): Input signal as 1D float32 numpy array.
    ///
    /// Returns:
    ///     np.ndarray: Filtered signal as 1D float32 numpy array.
    ///
    /// Example:
    ///     >>> signal = np.random.randn(1000).astype(np.float32)
    ///     >>> filtered = fir.process(signal)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice()?;
        let mut output = vec![0.0f32; input_slice.len()];

        match &mut self.inner {
            FirFilterInner::Taps8(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    output[i] = filter.process_sample(sample);
                }
            }
            FirFilterInner::Taps16(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    output[i] = filter.process_sample(sample);
                }
            }
            FirFilterInner::Taps32(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    output[i] = filter.process_sample(sample);
                }
            }
            FirFilterInner::Taps64(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    output[i] = filter.process_sample(sample);
                }
            }
            FirFilterInner::Dynamic {
                coeffs,
                delay_line,
                index,
            } => {
                let num_taps = coeffs.len();
                for (i, &sample) in input_slice.iter().enumerate() {
                    delay_line[*index] = sample;

                    let mut out = 0.0;
                    let mut delay_idx = *index;
                    for tap in 0..num_taps {
                        out += coeffs[tap] * delay_line[delay_idx];
                        delay_idx = if delay_idx == 0 {
                            num_taps - 1
                        } else {
                            delay_idx - 1
                        };
                    }

                    *index = (*index + 1) % num_taps;
                    output[i] = out;
                }
            }
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Reset the filter state (clear delay line).
    ///
    /// This is useful when processing discontinuous segments of data.
    fn reset(&mut self) {
        match &mut self.inner {
            FirFilterInner::Taps8(filter) => filter.reset(),
            FirFilterInner::Taps16(filter) => filter.reset(),
            FirFilterInner::Taps32(filter) => filter.reset(),
            FirFilterInner::Taps64(filter) => filter.reset(),
            FirFilterInner::Dynamic {
                delay_line, index, ..
            } => {
                delay_line.fill(0.0);
                *index = 0;
            }
        }
    }

    /// Get the number of filter taps.
    #[getter]
    fn num_taps(&self) -> usize {
        self.num_taps
    }

    fn __repr__(&self) -> String {
        format!("FirFilter(num_taps={})", self.num_taps)
    }
}

// ============================================================================
// AcCoupler
// ============================================================================

/// AC coupling filter for removing DC offset from signals.
///
/// Uses a single-pole high-pass filter specifically designed for AC coupling
/// (DC removal). More efficient than a general IIR filter when you only need
/// to remove DC offset.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create AC coupler with 0.1 Hz cutoff at 1000 Hz sample rate
/// ac = npy.AcCoupler(1000.0, 0.1)
///
/// # Process a signal with DC offset
/// signal = np.ones(1000, dtype=np.float32) + np.random.randn(1000).astype(np.float32) * 0.1
/// clean = ac.process(signal)
/// # DC component will be removed, leaving only AC variations
/// ```
#[pyclass]
pub struct AcCoupler {
    inner: ZsAcCoupler<1>,
    sample_rate: f32,
    cutoff: f32,
}

#[pymethods]
impl AcCoupler {
    /// Create a new AC coupling filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz.
    ///     cutoff (float): Cutoff frequency in Hz (typically 0.1-1.0 Hz for BCI).
    ///
    /// Returns:
    ///     AcCoupler: A new AC coupling filter.
    ///
    /// Example:
    ///     >>> ac = AcCoupler(1000.0, 0.1)
    #[new]
    fn new(sample_rate: f32, cutoff: f32) -> PyResult<Self> {
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err(format!(
                "cutoff must be between 0 and {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }

        Ok(Self {
            inner: ZsAcCoupler::new(sample_rate, cutoff),
            sample_rate,
            cutoff,
        })
    }

    /// Process a signal through the AC coupler.
    ///
    /// Args:
    ///     input (np.ndarray): Input signal as 1D float32 numpy array.
    ///
    /// Returns:
    ///     np.ndarray: AC-coupled signal as 1D float32 numpy array.
    ///
    /// Example:
    ///     >>> signal = np.ones(1000, dtype=np.float32) * 2.0
    ///     >>> clean = ac.process(signal)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice()?;
        let mut output = vec![0.0f32; input_slice.len()];

        for (i, &sample) in input_slice.iter().enumerate() {
            let out = self.inner.process(&[sample]);
            output[i] = out[0];
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Reset the filter state.
    ///
    /// Call this when starting to process a new signal segment
    /// to avoid transients from previous state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the sample rate.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the cutoff frequency.
    #[getter]
    fn cutoff(&self) -> f32 {
        self.cutoff
    }

    fn __repr__(&self) -> String {
        format!(
            "AcCoupler(sample_rate={}, cutoff={})",
            self.sample_rate, self.cutoff
        )
    }
}

// ============================================================================
// MedianFilter
// ============================================================================

/// Internal enum for handling different median filter window sizes.
enum MedianFilterInner {
    Window3(ZsMedianFilter<1, 3>),
    Window5(ZsMedianFilter<1, 5>),
    Window7(ZsMedianFilter<1, 7>),
}

/// Non-linear median filter for impulsive noise rejection.
///
/// Replaces each sample with the median of a sliding window, effective for:
/// - Salt-and-pepper noise (electrode artifacts)
/// - Motion artifact spikes
/// - Outlier rejection
/// - Edge-preserving smoothing
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create median filter with window size 5
/// mf = npy.MedianFilter(5)
///
/// # Remove spike noise from signal
/// noisy = np.array([1, 1, 100, 1, 1, 1, 1, 1], dtype=np.float32)
/// clean = mf.process(noisy)
/// # The spike at index 2 will be rejected
/// ```
#[pyclass]
pub struct MedianFilter {
    inner: MedianFilterInner,
    window_size: usize,
}

#[pymethods]
impl MedianFilter {
    /// Create a new median filter.
    ///
    /// Args:
    ///     window_size (int): Size of the sliding window. Must be 3, 5, or 7.
    ///
    /// Returns:
    ///     MedianFilter: A new median filter.
    ///
    /// Example:
    ///     >>> mf = MedianFilter(5)
    #[new]
    fn new(window_size: usize) -> PyResult<Self> {
        let inner = match window_size {
            3 => MedianFilterInner::Window3(ZsMedianFilter::new()),
            5 => MedianFilterInner::Window5(ZsMedianFilter::new()),
            7 => MedianFilterInner::Window7(ZsMedianFilter::new()),
            _ => {
                return Err(PyValueError::new_err(
                    "window_size must be 3, 5, or 7 (optimized sorting network sizes)",
                ))
            }
        };

        Ok(Self { inner, window_size })
    }

    /// Process a signal through the median filter.
    ///
    /// Note: The first (window_size - 1) outputs use zero-padding.
    ///
    /// Args:
    ///     input (np.ndarray): Input signal as 1D float32 numpy array.
    ///
    /// Returns:
    ///     np.ndarray: Median-filtered signal as 1D float32 numpy array.
    ///
    /// Example:
    ///     >>> noisy = np.array([1, 1, 100, 1, 1], dtype=np.float32)
    ///     >>> clean = mf.process(noisy)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice()?;
        let mut output = vec![0.0f32; input_slice.len()];

        match &mut self.inner {
            MedianFilterInner::Window3(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    let out = filter.process(&[sample]);
                    output[i] = out[0];
                }
            }
            MedianFilterInner::Window5(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    let out = filter.process(&[sample]);
                    output[i] = out[0];
                }
            }
            MedianFilterInner::Window7(filter) => {
                for (i, &sample) in input_slice.iter().enumerate() {
                    let out = filter.process(&[sample]);
                    output[i] = out[0];
                }
            }
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Reset the filter state.
    ///
    /// Clears all circular buffers and restarts zero-padding phase.
    fn reset(&mut self) {
        match &mut self.inner {
            MedianFilterInner::Window3(filter) => filter.reset(),
            MedianFilterInner::Window5(filter) => filter.reset(),
            MedianFilterInner::Window7(filter) => filter.reset(),
        }
    }

    /// Get the window size.
    #[getter]
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn __repr__(&self) -> String {
        format!("MedianFilter(window_size={})", self.window_size)
    }
}
