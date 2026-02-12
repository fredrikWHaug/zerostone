//! Python bindings for detection primitives.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    AdaptiveThresholdDetector as ZsAdaptiveThresholdDetector,
    ThresholdDetector as ZsThresholdDetector, ZeroCrossingDetector as ZsZeroCrossingDetector,
};

// ============================================================================
// ThresholdDetector
// ============================================================================

/// Internal enum for handling different channel counts.
enum ThresholdDetectorInner {
    Ch1(ZsThresholdDetector<1>),
    Ch4(ZsThresholdDetector<4>),
    Ch8(ZsThresholdDetector<8>),
    Ch16(ZsThresholdDetector<16>),
    Ch32(ZsThresholdDetector<32>),
    Ch64(ZsThresholdDetector<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic {
        threshold: f32,
        refractory_samples: u32,
        refractory_counter: Vec<u32>,
    },
}

/// Multi-channel threshold detector with refractory period.
///
/// Detects when signal amplitude crosses a threshold and enforces a refractory
/// period where subsequent crossings are ignored. Essential for spike detection.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create detector for 8 channels with threshold 3.0 and 100-sample refractory
/// det = npy.ThresholdDetector(channels=8, threshold=3.0, refractory=100)
///
/// # Process multi-channel data
/// data = np.random.randn(1000, 8).astype(np.float32)
/// events = det.process(data)
/// # events is a list of (sample_idx, channel, amplitude) tuples
/// ```
#[pyclass]
pub struct ThresholdDetector {
    inner: ThresholdDetectorInner,
    channels: usize,
    threshold: f32,
    refractory: u32,
}

#[pymethods]
impl ThresholdDetector {
    /// Create a new threshold detector.
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///     threshold (float): Amplitude threshold for detection.
    ///     refractory (int): Refractory period in samples.
    ///
    /// Returns:
    ///     ThresholdDetector: A new detector instance.
    ///
    /// Example:
    ///     >>> det = ThresholdDetector(channels=8, threshold=3.0, refractory=100)
    #[new]
    fn new(channels: usize, threshold: f32, refractory: u32) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let inner = match channels {
            1 => ThresholdDetectorInner::Ch1(ZsThresholdDetector::new(threshold, refractory)),
            4 => ThresholdDetectorInner::Ch4(ZsThresholdDetector::new(threshold, refractory)),
            8 => ThresholdDetectorInner::Ch8(ZsThresholdDetector::new(threshold, refractory)),
            16 => ThresholdDetectorInner::Ch16(ZsThresholdDetector::new(threshold, refractory)),
            32 => ThresholdDetectorInner::Ch32(ZsThresholdDetector::new(threshold, refractory)),
            64 => ThresholdDetectorInner::Ch64(ZsThresholdDetector::new(threshold, refractory)),
            _ => ThresholdDetectorInner::Dynamic {
                threshold,
                refractory_samples: refractory,
                refractory_counter: vec![0; channels],
            },
        };

        Ok(Self {
            inner,
            channels,
            threshold,
            refractory,
        })
    }

    /// Process multi-channel data and detect threshold crossings.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     list[tuple[int, int, float]]: List of events as (sample_idx, channel, amplitude).
    ///
    /// Example:
    ///     >>> events = det.process(data)
    ///     >>> for sample_idx, channel, amplitude in events:
    ///     ...     print(f"Spike at sample {sample_idx}, channel {channel}: {amplitude}")
    fn process(&mut self, input: PyReadonlyArray2<f32>) -> PyResult<Vec<(usize, usize, f32)>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: detector configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut events = Vec::new();

        macro_rules! process_detector {
            ($det:expr, $C:expr) => {{
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let spike_events = $det.process_sample_all(&samples);
                    for event in spike_events.iter() {
                        events.push((i, event.channel, event.amplitude));
                    }
                }
            }};
        }

        match &mut self.inner {
            ThresholdDetectorInner::Ch1(det) => process_detector!(det, 1),
            ThresholdDetectorInner::Ch4(det) => process_detector!(det, 4),
            ThresholdDetectorInner::Ch8(det) => process_detector!(det, 8),
            ThresholdDetectorInner::Ch16(det) => process_detector!(det, 16),
            ThresholdDetectorInner::Ch32(det) => process_detector!(det, 32),
            ThresholdDetectorInner::Ch64(det) => process_detector!(det, 64),
            ThresholdDetectorInner::Dynamic {
                threshold,
                refractory_samples,
                refractory_counter,
            } => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    for (ch, &amplitude) in row.iter().enumerate() {
                        if refractory_counter[ch] > 0 {
                            refractory_counter[ch] -= 1;
                            continue;
                        }
                        if amplitude.abs() > *threshold {
                            refractory_counter[ch] = *refractory_samples;
                            events.push((i, ch, amplitude));
                        }
                    }
                }
            }
        }

        Ok(events)
    }

    /// Reset all channels to armed state.
    fn reset(&mut self) {
        match &mut self.inner {
            ThresholdDetectorInner::Ch1(det) => det.reset(),
            ThresholdDetectorInner::Ch4(det) => det.reset(),
            ThresholdDetectorInner::Ch8(det) => det.reset(),
            ThresholdDetectorInner::Ch16(det) => det.reset(),
            ThresholdDetectorInner::Ch32(det) => det.reset(),
            ThresholdDetectorInner::Ch64(det) => det.reset(),
            ThresholdDetectorInner::Dynamic {
                refractory_counter, ..
            } => {
                refractory_counter.fill(0);
            }
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the detection threshold.
    #[getter]
    fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the refractory period in samples.
    #[getter]
    fn refractory(&self) -> u32 {
        self.refractory
    }

    fn __repr__(&self) -> String {
        format!(
            "ThresholdDetector(channels={}, threshold={}, refractory={})",
            self.channels, self.threshold, self.refractory
        )
    }
}

// ============================================================================
// AdaptiveThresholdDetector
// ============================================================================

/// Internal enum for handling different channel counts.
enum AdaptiveThresholdDetectorInner {
    Ch1(ZsAdaptiveThresholdDetector<1>),
    Ch4(ZsAdaptiveThresholdDetector<4>),
    Ch8(ZsAdaptiveThresholdDetector<8>),
    Ch16(ZsAdaptiveThresholdDetector<16>),
    Ch32(ZsAdaptiveThresholdDetector<32>),
    Ch64(ZsAdaptiveThresholdDetector<64>),
}

/// Adaptive multi-channel threshold detector using N×σ detection.
///
/// Automatically computes detection thresholds based on signal statistics,
/// using a multiplier times the standard deviation (e.g., 4×σ).
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create detector: 4×σ threshold, 100-sample refractory, 500-sample warmup
/// det = npy.AdaptiveThresholdDetector(channels=8, multiplier=4.0, refractory=100, min_samples=500)
///
/// # Process data (first 500 samples are calibration)
/// data = np.random.randn(2000, 8).astype(np.float32)
/// events = det.process(data)
/// ```
#[pyclass]
pub struct AdaptiveThresholdDetector {
    inner: AdaptiveThresholdDetectorInner,
    channels: usize,
    multiplier: f32,
    refractory: u32,
    min_samples: u64,
}

#[pymethods]
impl AdaptiveThresholdDetector {
    /// Create a new adaptive threshold detector.
    ///
    /// Args:
    ///     channels (int): Number of channels. Must be 1, 4, 8, 16, 32, or 64.
    ///     multiplier (float): Threshold multiplier (e.g., 4.0 for 4×σ detection).
    ///     refractory (int): Refractory period in samples.
    ///     min_samples (int): Minimum samples before detection starts (warm-up).
    ///
    /// Returns:
    ///     AdaptiveThresholdDetector: A new detector instance.
    ///
    /// Example:
    ///     >>> det = AdaptiveThresholdDetector(channels=8, multiplier=4.0, refractory=100, min_samples=500)
    #[new]
    fn new(channels: usize, multiplier: f32, refractory: u32, min_samples: u64) -> PyResult<Self> {
        let inner = match channels {
            1 => AdaptiveThresholdDetectorInner::Ch1(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            4 => AdaptiveThresholdDetectorInner::Ch4(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            8 => AdaptiveThresholdDetectorInner::Ch8(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            16 => AdaptiveThresholdDetectorInner::Ch16(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            32 => AdaptiveThresholdDetectorInner::Ch32(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            64 => AdaptiveThresholdDetectorInner::Ch64(ZsAdaptiveThresholdDetector::new(
                multiplier,
                refractory,
                min_samples,
            )),
            _ => {
                return Err(PyValueError::new_err(
                    "channels must be 1, 4, 8, 16, 32, or 64 for AdaptiveThresholdDetector",
                ))
            }
        };

        Ok(Self {
            inner,
            channels,
            multiplier,
            refractory,
            min_samples,
        })
    }

    /// Process multi-channel data and detect threshold crossings.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     list[tuple[int, int, float]]: List of events as (sample_idx, channel, amplitude).
    fn process(&mut self, input: PyReadonlyArray2<f32>) -> PyResult<Vec<(usize, usize, f32)>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: detector configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut events = Vec::new();

        macro_rules! process_detector {
            ($det:expr, $C:expr) => {{
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let spike_events = $det.process_sample_all(&samples);
                    for event in spike_events.iter() {
                        events.push((i, event.channel, event.amplitude));
                    }
                }
            }};
        }

        match &mut self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => process_detector!(det, 1),
            AdaptiveThresholdDetectorInner::Ch4(det) => process_detector!(det, 4),
            AdaptiveThresholdDetectorInner::Ch8(det) => process_detector!(det, 8),
            AdaptiveThresholdDetectorInner::Ch16(det) => process_detector!(det, 16),
            AdaptiveThresholdDetectorInner::Ch32(det) => process_detector!(det, 32),
            AdaptiveThresholdDetectorInner::Ch64(det) => process_detector!(det, 64),
        }

        Ok(events)
    }

    /// Freeze the current thresholds (stop adapting).
    fn freeze(&mut self) {
        match &mut self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.freeze(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.freeze(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.freeze(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.freeze(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.freeze(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.freeze(),
        }
    }

    /// Unfreeze thresholds and resume adapting.
    fn unfreeze(&mut self) {
        match &mut self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.unfreeze(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.unfreeze(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.unfreeze(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.unfreeze(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.unfreeze(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.unfreeze(),
        }
    }

    /// Get current thresholds for all channels.
    #[getter]
    fn thresholds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let t: Vec<f32> = match &self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.thresholds().to_vec(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.thresholds().to_vec(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.thresholds().to_vec(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.thresholds().to_vec(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.thresholds().to_vec(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.thresholds().to_vec(),
        };
        PyArray1::from_vec(py, t)
    }

    /// Check if the detector is still in calibration period.
    #[getter]
    fn is_calibrating(&self) -> bool {
        match &self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.is_calibrating(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.is_calibrating(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.is_calibrating(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.is_calibrating(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.is_calibrating(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.is_calibrating(),
        }
    }

    /// Check if the detector is frozen.
    #[getter]
    fn is_frozen(&self) -> bool {
        match &self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.is_frozen(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.is_frozen(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.is_frozen(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.is_frozen(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.is_frozen(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.is_frozen(),
        }
    }

    /// Reset all state including statistics.
    fn reset(&mut self) {
        match &mut self.inner {
            AdaptiveThresholdDetectorInner::Ch1(det) => det.reset(),
            AdaptiveThresholdDetectorInner::Ch4(det) => det.reset(),
            AdaptiveThresholdDetectorInner::Ch8(det) => det.reset(),
            AdaptiveThresholdDetectorInner::Ch16(det) => det.reset(),
            AdaptiveThresholdDetectorInner::Ch32(det) => det.reset(),
            AdaptiveThresholdDetectorInner::Ch64(det) => det.reset(),
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the multiplier.
    #[getter]
    fn multiplier(&self) -> f32 {
        self.multiplier
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaptiveThresholdDetector(channels={}, multiplier={}, refractory={}, min_samples={})",
            self.channels, self.multiplier, self.refractory, self.min_samples
        )
    }
}

// ============================================================================
// ZeroCrossingDetector
// ============================================================================

/// Internal enum for handling different channel counts.
enum ZeroCrossingDetectorInner {
    Ch1(ZsZeroCrossingDetector<1>),
    Ch4(ZsZeroCrossingDetector<4>),
    Ch8(ZsZeroCrossingDetector<8>),
    Ch16(ZsZeroCrossingDetector<16>),
    Ch32(ZsZeroCrossingDetector<32>),
    Ch64(ZsZeroCrossingDetector<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic {
        threshold: f32,
        prev_sample: Vec<f32>,
        initialized: bool,
    },
}

/// Multi-channel zero-crossing detector for BCI signal analysis.
///
/// Detects when a signal changes sign (crosses zero). Essential for computing
/// zero-crossing rate (ZCR) features.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create detector for 8 channels with 0.1 noise threshold
/// det = npy.ZeroCrossingDetector(channels=8, threshold=0.1)
///
/// # Detect zero crossings
/// data = np.random.randn(1000, 8).astype(np.float32)
/// crossings = det.detect(data)  # Returns bool array
/// zcr = det.zcr(data)  # Returns ZCR per channel
/// ```
#[pyclass]
pub struct ZeroCrossingDetector {
    inner: ZeroCrossingDetectorInner,
    channels: usize,
    threshold: f32,
}

#[pymethods]
impl ZeroCrossingDetector {
    /// Create a new zero-crossing detector.
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///     threshold (float): Noise rejection threshold. Values within [-threshold, +threshold]
    ///         are treated as zero to prevent false crossings.
    ///
    /// Returns:
    ///     ZeroCrossingDetector: A new detector instance.
    ///
    /// Example:
    ///     >>> det = ZeroCrossingDetector(channels=8, threshold=0.1)
    #[new]
    #[pyo3(signature = (channels, threshold = 0.0))]
    fn new(channels: usize, threshold: f32) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let inner = match channels {
            1 => ZeroCrossingDetectorInner::Ch1(ZsZeroCrossingDetector::new(threshold)),
            4 => ZeroCrossingDetectorInner::Ch4(ZsZeroCrossingDetector::new(threshold)),
            8 => ZeroCrossingDetectorInner::Ch8(ZsZeroCrossingDetector::new(threshold)),
            16 => ZeroCrossingDetectorInner::Ch16(ZsZeroCrossingDetector::new(threshold)),
            32 => ZeroCrossingDetectorInner::Ch32(ZsZeroCrossingDetector::new(threshold)),
            64 => ZeroCrossingDetectorInner::Ch64(ZsZeroCrossingDetector::new(threshold)),
            _ => ZeroCrossingDetectorInner::Dynamic {
                threshold,
                prev_sample: vec![0.0; channels],
                initialized: false,
            },
        };

        Ok(Self {
            inner,
            channels,
            threshold,
        })
    }

    /// Detect zero crossings in multi-channel data.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Boolean array with shape (samples, channels), True where crossing detected.
    ///
    /// Example:
    ///     >>> crossings = det.detect(data)
    fn detect<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: detector configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut output = vec![false; n_samples * n_channels];

        macro_rules! process_detector {
            ($det:expr, $C:expr) => {{
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let crossings = $det.detect(&samples);
                    for (j, &crossed) in crossings.iter().enumerate() {
                        output[i * n_channels + j] = crossed;
                    }
                }
            }};
        }

        match &mut self.inner {
            ZeroCrossingDetectorInner::Ch1(det) => process_detector!(det, 1),
            ZeroCrossingDetectorInner::Ch4(det) => process_detector!(det, 4),
            ZeroCrossingDetectorInner::Ch8(det) => process_detector!(det, 8),
            ZeroCrossingDetectorInner::Ch16(det) => process_detector!(det, 16),
            ZeroCrossingDetectorInner::Ch32(det) => process_detector!(det, 32),
            ZeroCrossingDetectorInner::Ch64(det) => process_detector!(det, 64),
            ZeroCrossingDetectorInner::Dynamic {
                threshold,
                prev_sample,
                initialized,
            } => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    if !*initialized {
                        for (j, &val) in row.iter().enumerate() {
                            prev_sample[j] = val;
                        }
                        *initialized = true;
                        continue;
                    }

                    for (j, &val) in row.iter().enumerate() {
                        let prev_sign = get_sign(prev_sample[j], *threshold);
                        let curr_sign = get_sign(val, *threshold);
                        let crossed = prev_sign != 0 && curr_sign != 0 && prev_sign != curr_sign;
                        output[i * n_channels + j] = crossed;
                        prev_sample[j] = val;
                    }
                }
            }
        }

        let output_array = Array2::from_shape_vec((n_samples, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Compute zero-crossing rate for a block of samples.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: ZCR values per channel (0.0-1.0), shape (channels,).
    ///
    /// Example:
    ///     >>> zcr = det.zcr(data)
    fn zcr<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: detector configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut crossing_counts = vec![0usize; n_channels];

        macro_rules! process_detector {
            ($det:expr, $C:expr) => {{
                for row in input_array.rows() {
                    let mut samples = [0.0f32; $C];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let crossings = $det.detect(&samples);
                    for (j, &crossed) in crossings.iter().enumerate() {
                        if crossed {
                            crossing_counts[j] += 1;
                        }
                    }
                }
            }};
        }

        match &mut self.inner {
            ZeroCrossingDetectorInner::Ch1(det) => process_detector!(det, 1),
            ZeroCrossingDetectorInner::Ch4(det) => process_detector!(det, 4),
            ZeroCrossingDetectorInner::Ch8(det) => process_detector!(det, 8),
            ZeroCrossingDetectorInner::Ch16(det) => process_detector!(det, 16),
            ZeroCrossingDetectorInner::Ch32(det) => process_detector!(det, 32),
            ZeroCrossingDetectorInner::Ch64(det) => process_detector!(det, 64),
            ZeroCrossingDetectorInner::Dynamic {
                threshold,
                prev_sample,
                initialized,
            } => {
                for row in input_array.rows() {
                    if !*initialized {
                        for (j, &val) in row.iter().enumerate() {
                            prev_sample[j] = val;
                        }
                        *initialized = true;
                        continue;
                    }

                    for (j, &val) in row.iter().enumerate() {
                        let prev_sign = get_sign(prev_sample[j], *threshold);
                        let curr_sign = get_sign(val, *threshold);
                        if prev_sign != 0 && curr_sign != 0 && prev_sign != curr_sign {
                            crossing_counts[j] += 1;
                        }
                        prev_sample[j] = val;
                    }
                }
            }
        }

        let zcr: Vec<f32> = crossing_counts
            .iter()
            .map(|&c| c as f32 / n_samples as f32)
            .collect();
        Ok(PyArray1::from_vec(py, zcr))
    }

    /// Reset the detector state.
    fn reset(&mut self) {
        match &mut self.inner {
            ZeroCrossingDetectorInner::Ch1(det) => det.reset(),
            ZeroCrossingDetectorInner::Ch4(det) => det.reset(),
            ZeroCrossingDetectorInner::Ch8(det) => det.reset(),
            ZeroCrossingDetectorInner::Ch16(det) => det.reset(),
            ZeroCrossingDetectorInner::Ch32(det) => det.reset(),
            ZeroCrossingDetectorInner::Ch64(det) => det.reset(),
            ZeroCrossingDetectorInner::Dynamic {
                prev_sample,
                initialized,
                ..
            } => {
                prev_sample.fill(0.0);
                *initialized = false;
            }
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the noise rejection threshold.
    #[getter]
    fn threshold(&self) -> f32 {
        self.threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "ZeroCrossingDetector(channels={}, threshold={})",
            self.channels, self.threshold
        )
    }
}

/// Helper function to get sign with threshold.
fn get_sign(value: f32, threshold: f32) -> i8 {
    if value.abs() <= threshold {
        0
    } else if value > 0.0 {
        1
    } else {
        -1
    }
}
