//! Python bindings for the NotchFilter powerline noise removal bank.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::NotchFilter as ZsNotchFilter;

// ============================================================================
// Channel-count dispatch enum (SECTIONS hardcoded to 4)
// ============================================================================

enum NotchFilterInner {
    Ch1(ZsNotchFilter<1, 4>),
    Ch4(ZsNotchFilter<4, 4>),
    Ch8(ZsNotchFilter<8, 4>),
    Ch16(ZsNotchFilter<16, 4>),
    Ch32(ZsNotchFilter<32, 4>),
    Ch64(ZsNotchFilter<64, 4>),
}

impl NotchFilterInner {
    fn process_row(&mut self, row: &[f32], out: &mut [f32]) {
        match self {
            NotchFilterInner::Ch1(f) => {
                let inp = [row[0]];
                let y = f.process_sample(&inp);
                out[0] = y[0];
            }
            NotchFilterInner::Ch4(f) => {
                let inp = [row[0], row[1], row[2], row[3]];
                let y = f.process_sample(&inp);
                out.copy_from_slice(&y);
            }
            NotchFilterInner::Ch8(f) => {
                let mut inp = [0.0f32; 8];
                inp.copy_from_slice(row);
                let y = f.process_sample(&inp);
                out.copy_from_slice(&y);
            }
            NotchFilterInner::Ch16(f) => {
                let mut inp = [0.0f32; 16];
                inp.copy_from_slice(row);
                let y = f.process_sample(&inp);
                out.copy_from_slice(&y);
            }
            NotchFilterInner::Ch32(f) => {
                let mut inp = [0.0f32; 32];
                inp.copy_from_slice(row);
                let y = f.process_sample(&inp);
                out.copy_from_slice(&y);
            }
            NotchFilterInner::Ch64(f) => {
                let mut inp = [0.0f32; 64];
                inp.copy_from_slice(row);
                let y = f.process_sample(&inp);
                out.copy_from_slice(&y);
            }
        }
    }

    fn reset(&mut self) {
        match self {
            NotchFilterInner::Ch1(f) => f.reset(),
            NotchFilterInner::Ch4(f) => f.reset(),
            NotchFilterInner::Ch8(f) => f.reset(),
            NotchFilterInner::Ch16(f) => f.reset(),
            NotchFilterInner::Ch32(f) => f.reset(),
            NotchFilterInner::Ch64(f) => f.reset(),
        }
    }
}

// ============================================================================
// Public Python class
// ============================================================================

/// Multi-channel notch filter bank for powerline interference removal.
///
/// Applies cascaded biquad notch filters at powerline harmonics (up to 4
/// sections: fundamental + 3 harmonics). Harmonics above the Nyquist
/// frequency are automatically replaced with passthroughs.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # 8-channel filter for 60 Hz US powerline at 250 Hz sample rate
/// f = zbci.NotchFilter.powerline_60hz(250.0, channels=8)
///
/// data = np.random.randn(500, 8).astype(np.float32)
/// filtered = f.process(data)
/// ```
#[pyclass]
pub struct NotchFilter {
    inner: NotchFilterInner,
    channels: usize,
    sample_rate: f32,
}

fn make_inner_powerline(
    channels: usize,
    sample_rate: f32,
    fundamental: f32,
) -> PyResult<NotchFilterInner> {
    Ok(match channels {
        1 => NotchFilterInner::Ch1(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        4 => NotchFilterInner::Ch4(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        8 => NotchFilterInner::Ch8(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        16 => NotchFilterInner::Ch16(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        32 => NotchFilterInner::Ch32(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        64 => NotchFilterInner::Ch64(if fundamental == 60.0 {
            ZsNotchFilter::powerline_60hz(sample_rate)
        } else {
            ZsNotchFilter::powerline_50hz(sample_rate)
        }),
        _ => {
            return Err(PyValueError::new_err(format!(
                "channels must be one of {{1, 4, 8, 16, 32, 64}}, got {}",
                channels
            )))
        }
    })
}

fn make_inner_custom(
    channels: usize,
    sample_rate: f32,
    freqs: [f32; 4],
    q: f32,
) -> PyResult<NotchFilterInner> {
    Ok(match channels {
        1 => NotchFilterInner::Ch1(ZsNotchFilter::custom(sample_rate, freqs, q)),
        4 => NotchFilterInner::Ch4(ZsNotchFilter::custom(sample_rate, freqs, q)),
        8 => NotchFilterInner::Ch8(ZsNotchFilter::custom(sample_rate, freqs, q)),
        16 => NotchFilterInner::Ch16(ZsNotchFilter::custom(sample_rate, freqs, q)),
        32 => NotchFilterInner::Ch32(ZsNotchFilter::custom(sample_rate, freqs, q)),
        64 => NotchFilterInner::Ch64(ZsNotchFilter::custom(sample_rate, freqs, q)),
        _ => {
            return Err(PyValueError::new_err(format!(
                "channels must be one of {{1, 4, 8, 16, 32, 64}}, got {}",
                channels
            )))
        }
    })
}

#[pymethods]
impl NotchFilter {
    /// Create a notch filter bank targeting 60 Hz powerline noise (US/Japan).
    ///
    /// Places notches at 60, 120, 180, 240 Hz. Harmonics above Nyquist
    /// become passthroughs automatically.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz (must be > 0).
    ///     channels (int): Number of channels. Must be one of {1, 4, 8, 16, 32, 64}.
    ///
    /// Returns:
    ///     NotchFilter: Ready-to-use filter instance.
    ///
    /// Example:
    ///     >>> f = NotchFilter.powerline_60hz(250.0, channels=8)
    #[staticmethod]
    #[pyo3(signature = (sample_rate, channels))]
    fn powerline_60hz(sample_rate: f32, channels: usize) -> PyResult<Self> {
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }
        let inner = make_inner_powerline(channels, sample_rate, 60.0)?;
        Ok(Self {
            inner,
            channels,
            sample_rate,
        })
    }

    /// Create a notch filter bank targeting 50 Hz powerline noise (EU/Asia).
    ///
    /// Places notches at 50, 100, 150, 200 Hz. Harmonics above Nyquist
    /// become passthroughs automatically.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz (must be > 0).
    ///     channels (int): Number of channels. Must be one of {1, 4, 8, 16, 32, 64}.
    ///
    /// Returns:
    ///     NotchFilter: Ready-to-use filter instance.
    ///
    /// Example:
    ///     >>> f = NotchFilter.powerline_50hz(1000.0, channels=16)
    #[staticmethod]
    #[pyo3(signature = (sample_rate, channels))]
    fn powerline_50hz(sample_rate: f32, channels: usize) -> PyResult<Self> {
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }
        let inner = make_inner_powerline(channels, sample_rate, 50.0)?;
        Ok(Self {
            inner,
            channels,
            sample_rate,
        })
    }

    /// Create a notch filter bank with user-specified center frequencies.
    ///
    /// Provide 1–4 frequencies in Hz. Missing entries are treated as
    /// passthrough sections. Frequencies of 0.0 also become passthroughs.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz (must be > 0).
    ///     channels (int): Number of channels. Must be one of {1, 4, 8, 16, 32, 64}.
    ///     freqs (list[float]): Center frequencies in Hz (1–4 values).
    ///     q (float): Quality factor, default 30.0.
    ///
    /// Returns:
    ///     NotchFilter: Ready-to-use filter instance.
    ///
    /// Example:
    ///     >>> f = NotchFilter.custom(1000.0, 8, [60.0, 120.0])
    #[staticmethod]
    #[pyo3(signature = (sample_rate, channels, freqs, q = 30.0))]
    fn custom(sample_rate: f32, channels: usize, freqs: Vec<f32>, q: f32) -> PyResult<Self> {
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }
        if freqs.is_empty() || freqs.len() > 4 {
            return Err(PyValueError::new_err(
                "freqs must have between 1 and 4 elements",
            ));
        }

        // Pad to exactly 4 entries with 0.0 (passthrough)
        let mut freq_arr = [0.0f32; 4];
        for (i, &f) in freqs.iter().enumerate() {
            freq_arr[i] = f;
        }

        let inner = make_inner_custom(channels, sample_rate, freq_arr, q)?;
        Ok(Self {
            inner,
            channels,
            sample_rate,
        })
    }

    /// Process a block of multi-channel data through the notch filter.
    ///
    /// Args:
    ///     input (np.ndarray): Float32 array of shape (n_samples, n_channels).
    ///
    /// Returns:
    ///     np.ndarray: Filtered data with the same shape and dtype.
    ///
    /// Raises:
    ///     ValueError: If the array is not 2D or channel count doesn't match.
    ///
    /// Example:
    ///     >>> data = np.random.randn(500, 8).astype(np.float32)
    ///     >>> filtered = f.process(data)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: NotchFilter configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut output = vec![0.0f32; n_samples * n_channels];

        for (i, row) in input_array.rows().into_iter().enumerate() {
            let row_slice: Vec<f32> = row.iter().copied().collect();
            let out_slice = &mut output[i * n_channels..(i + 1) * n_channels];
            self.inner.process_row(&row_slice, out_slice);
        }

        let output_array = Array2::from_shape_vec((n_samples, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset all filter state to zero.
    ///
    /// Call this when processing a new, discontinuous data segment to
    /// avoid transients from previous state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the sampling rate this filter was configured for.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the number of channels this filter was configured for.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __repr__(&self) -> String {
        format!(
            "NotchFilter(sample_rate={} Hz, channels={})",
            self.sample_rate, self.channels
        )
    }
}
