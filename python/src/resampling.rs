//! Python bindings for resampling primitives.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    Decimator as ZsDecimator, InterpolationMethod as ZsInterpolationMethod,
    Interpolator as ZsInterpolator,
};

// ============================================================================
// Decimator
// ============================================================================

/// Internal enum for handling different channel counts.
enum DecimatorInner {
    Ch1(ZsDecimator<1>),
    Ch4(ZsDecimator<4>),
    Ch8(ZsDecimator<8>),
    Ch16(ZsDecimator<16>),
    Ch32(ZsDecimator<32>),
    Ch64(ZsDecimator<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic {
        factor: usize,
        counter: usize,
    },
}

/// Decimator for reducing sample rate.
///
/// Reduces sample rate by keeping every Nth sample. For proper decimation,
/// apply an anti-aliasing lowpass filter before the decimator.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create decimator: 4x downsampling, 8 channels
/// dec = zbci.Decimator(factor=4, channels=8)
///
/// # Process multi-channel data (samples x channels)
/// data = np.random.randn(1000, 8).astype(np.float32)
/// decimated = dec.process(data)
/// assert decimated.shape == (250, 8)
/// ```
#[pyclass]
pub struct Decimator {
    inner: DecimatorInner,
    factor: usize,
    channels: usize,
}

#[pymethods]
impl Decimator {
    /// Create a new decimator.
    ///
    /// Args:
    ///     factor (int): Decimation factor. Factor of 4 means output rate is 1/4 of input.
    ///     channels (int): Number of channels.
    ///
    /// Returns:
    ///     Decimator: A new decimator instance.
    ///
    /// Example:
    ///     >>> dec = Decimator(factor=4, channels=8)
    #[new]
    fn new(factor: usize, channels: usize) -> PyResult<Self> {
        if factor == 0 {
            return Err(PyValueError::new_err("factor must be at least 1"));
        }
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let inner = match channels {
            1 => DecimatorInner::Ch1(ZsDecimator::new(factor)),
            4 => DecimatorInner::Ch4(ZsDecimator::new(factor)),
            8 => DecimatorInner::Ch8(ZsDecimator::new(factor)),
            16 => DecimatorInner::Ch16(ZsDecimator::new(factor)),
            32 => DecimatorInner::Ch32(ZsDecimator::new(factor)),
            64 => DecimatorInner::Ch64(ZsDecimator::new(factor)),
            _ => DecimatorInner::Dynamic { factor, counter: 0 },
        };

        Ok(Self {
            inner,
            factor,
            channels,
        })
    }

    /// Process multi-channel data through the decimator.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Decimated data with shape (samples // factor, channels).
    ///
    /// Example:
    ///     >>> data = np.random.randn(1000, 8).astype(np.float32)
    ///     >>> decimated = dec.process(data)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: Decimator configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        // Calculate output size
        let out_samples = (n_samples + self.factor - 1) / self.factor;
        let input_array = input.as_array();
        let mut output = vec![0.0f32; out_samples * n_channels];
        let mut out_idx = 0;

        macro_rules! process_decimator {
            ($dec:expr, $C:expr) => {{
                for row in input_array.rows() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    if let Some(out) = $dec.process(&samples) {
                        for (j, &val) in out.iter().enumerate() {
                            output[out_idx * n_channels + j] = val;
                        }
                        out_idx += 1;
                    }
                }
            }};
        }

        match &mut self.inner {
            DecimatorInner::Ch1(dec) => process_decimator!(dec, 1),
            DecimatorInner::Ch4(dec) => process_decimator!(dec, 4),
            DecimatorInner::Ch8(dec) => process_decimator!(dec, 8),
            DecimatorInner::Ch16(dec) => process_decimator!(dec, 16),
            DecimatorInner::Ch32(dec) => process_decimator!(dec, 32),
            DecimatorInner::Ch64(dec) => process_decimator!(dec, 64),
            DecimatorInner::Dynamic { factor, counter } => {
                for row in input_array.rows() {
                    if *counter == 0 {
                        for (j, &val) in row.iter().enumerate() {
                            output[out_idx * n_channels + j] = val;
                        }
                        out_idx += 1;
                    }
                    *counter += 1;
                    if *counter >= *factor {
                        *counter = 0;
                    }
                }
            }
        }

        // Trim output to actual size
        output.truncate(out_idx * n_channels);
        let output_array = Array2::from_shape_vec((out_idx, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset the decimator state.
    fn reset(&mut self) {
        match &mut self.inner {
            DecimatorInner::Ch1(dec) => dec.reset(),
            DecimatorInner::Ch4(dec) => dec.reset(),
            DecimatorInner::Ch8(dec) => dec.reset(),
            DecimatorInner::Ch16(dec) => dec.reset(),
            DecimatorInner::Ch32(dec) => dec.reset(),
            DecimatorInner::Ch64(dec) => dec.reset(),
            DecimatorInner::Dynamic { counter, .. } => *counter = 0,
        }
    }

    /// Get the decimation factor.
    #[getter]
    fn factor(&self) -> usize {
        self.factor
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __repr__(&self) -> String {
        format!(
            "Decimator(factor={}, channels={})",
            self.factor, self.channels
        )
    }
}

// ============================================================================
// Interpolator
// ============================================================================

/// Internal enum for handling different channel counts.
enum InterpolatorInner {
    Ch1(ZsInterpolator<1>),
    Ch4(ZsInterpolator<4>),
    Ch8(ZsInterpolator<8>),
    Ch16(ZsInterpolator<16>),
    Ch32(ZsInterpolator<32>),
    Ch64(ZsInterpolator<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic {
        factor: usize,
        method: ZsInterpolationMethod,
        prev_sample: Vec<f32>,
        initialized: bool,
    },
}

/// Interpolator for increasing sample rate.
///
/// Increases sample rate by an integer factor using various interpolation methods.
/// After interpolation, apply a lowpass filter to remove imaging artifacts.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create interpolator: 4x upsampling, 8 channels, linear interpolation
/// interp = zbci.Interpolator(factor=4, channels=8, method='linear')
///
/// # Process multi-channel data (samples x channels)
/// data = np.random.randn(100, 8).astype(np.float32)
/// upsampled = interp.process(data)
/// assert upsampled.shape == (400, 8)
/// ```
#[pyclass]
pub struct Interpolator {
    inner: InterpolatorInner,
    factor: usize,
    channels: usize,
    method: String,
}

#[pymethods]
impl Interpolator {
    /// Create a new interpolator.
    ///
    /// Args:
    ///     factor (int): Interpolation factor. Factor of 4 means output rate is 4x input.
    ///     channels (int): Number of channels.
    ///     method (str): Interpolation method - 'zero_order', 'linear', or 'zero_insert'.
    ///
    /// Returns:
    ///     Interpolator: A new interpolator instance.
    ///
    /// Example:
    ///     >>> interp = Interpolator(factor=4, channels=8, method='linear')
    #[new]
    #[pyo3(signature = (factor, channels, method = "linear"))]
    fn new(factor: usize, channels: usize, method: &str) -> PyResult<Self> {
        if factor == 0 {
            return Err(PyValueError::new_err("factor must be at least 1"));
        }
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let interp_method = match method {
            "zero_order" => ZsInterpolationMethod::ZeroOrder,
            "linear" => ZsInterpolationMethod::Linear,
            "zero_insert" => ZsInterpolationMethod::ZeroInsert,
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'zero_order', 'linear', or 'zero_insert'",
                ))
            }
        };

        let inner = match channels {
            1 => InterpolatorInner::Ch1(ZsInterpolator::new(factor, interp_method)),
            4 => InterpolatorInner::Ch4(ZsInterpolator::new(factor, interp_method)),
            8 => InterpolatorInner::Ch8(ZsInterpolator::new(factor, interp_method)),
            16 => InterpolatorInner::Ch16(ZsInterpolator::new(factor, interp_method)),
            32 => InterpolatorInner::Ch32(ZsInterpolator::new(factor, interp_method)),
            64 => InterpolatorInner::Ch64(ZsInterpolator::new(factor, interp_method)),
            _ => InterpolatorInner::Dynamic {
                factor,
                method: interp_method,
                prev_sample: vec![0.0; channels],
                initialized: false,
            },
        };

        Ok(Self {
            inner,
            factor,
            channels,
            method: method.to_string(),
        })
    }

    /// Process multi-channel data through the interpolator.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Interpolated data with shape (samples * factor, channels).
    ///
    /// Example:
    ///     >>> data = np.random.randn(100, 8).astype(np.float32)
    ///     >>> upsampled = interp.process(data)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: Interpolator configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let out_samples = n_samples * self.factor;
        let input_array = input.as_array();
        let mut output = vec![0.0f32; out_samples * n_channels];
        let mut out_idx = 0;

        macro_rules! process_interpolator {
            ($interp:expr, $C:expr) => {{
                for row in input_array.rows() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let mut out_buf = [[0.0f32; $C]; 64]; // Max factor we support
                    let count = $interp.process(&samples, &mut out_buf[..self.factor]);
                    for k in 0..count {
                        for (j, &val) in out_buf[k].iter().enumerate() {
                            output[out_idx * n_channels + j] = val;
                        }
                        out_idx += 1;
                    }
                }
            }};
        }

        match &mut self.inner {
            InterpolatorInner::Ch1(interp) => process_interpolator!(interp, 1),
            InterpolatorInner::Ch4(interp) => process_interpolator!(interp, 4),
            InterpolatorInner::Ch8(interp) => process_interpolator!(interp, 8),
            InterpolatorInner::Ch16(interp) => process_interpolator!(interp, 16),
            InterpolatorInner::Ch32(interp) => process_interpolator!(interp, 32),
            InterpolatorInner::Ch64(interp) => process_interpolator!(interp, 64),
            InterpolatorInner::Dynamic {
                factor,
                method,
                prev_sample,
                initialized,
            } => {
                for row in input_array.rows() {
                    let current: Vec<f32> = row.iter().cloned().collect();

                    match method {
                        ZsInterpolationMethod::ZeroOrder => {
                            for _ in 0..*factor {
                                for (j, &val) in current.iter().enumerate() {
                                    output[out_idx * n_channels + j] = val;
                                }
                                out_idx += 1;
                            }
                        }
                        ZsInterpolationMethod::Linear => {
                            if !*initialized {
                                for _ in 0..*factor {
                                    for (j, &val) in current.iter().enumerate() {
                                        output[out_idx * n_channels + j] = val;
                                    }
                                    out_idx += 1;
                                }
                            } else {
                                for k in 0..*factor {
                                    let t = k as f32 / *factor as f32;
                                    for (j, &val) in current.iter().enumerate() {
                                        let interp_val =
                                            prev_sample[j] + t * (val - prev_sample[j]);
                                        output[out_idx * n_channels + j] = interp_val;
                                    }
                                    out_idx += 1;
                                }
                            }
                        }
                        ZsInterpolationMethod::ZeroInsert => {
                            for (j, &val) in current.iter().enumerate() {
                                output[out_idx * n_channels + j] = val;
                            }
                            out_idx += 1;
                            for _ in 1..*factor {
                                for j in 0..n_channels {
                                    output[out_idx * n_channels + j] = 0.0;
                                }
                                out_idx += 1;
                            }
                        }
                    }

                    *prev_sample = current;
                    *initialized = true;
                }
            }
        }

        let output_array = Array2::from_shape_vec((out_idx, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset the interpolator state.
    fn reset(&mut self) {
        match &mut self.inner {
            InterpolatorInner::Ch1(interp) => interp.reset(),
            InterpolatorInner::Ch4(interp) => interp.reset(),
            InterpolatorInner::Ch8(interp) => interp.reset(),
            InterpolatorInner::Ch16(interp) => interp.reset(),
            InterpolatorInner::Ch32(interp) => interp.reset(),
            InterpolatorInner::Ch64(interp) => interp.reset(),
            InterpolatorInner::Dynamic {
                prev_sample,
                initialized,
                ..
            } => {
                prev_sample.fill(0.0);
                *initialized = false;
            }
        }
    }

    /// Get the interpolation factor.
    #[getter]
    fn factor(&self) -> usize {
        self.factor
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the interpolation method.
    #[getter]
    fn method(&self) -> &str {
        &self.method
    }

    fn __repr__(&self) -> String {
        format!(
            "Interpolator(factor={}, channels={}, method='{}')",
            self.factor, self.channels, self.method
        )
    }
}
