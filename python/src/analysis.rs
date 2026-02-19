//! Python bindings for signal analysis primitives.
//!
//! Provides envelope detection, power tracking, and Hilbert transform for BCI signals.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    hilbert::HilbertTransform as ZsHilbertTransform, EnvelopeFollower as ZsEnvelopeFollower,
    Rectification as ZsRectification, WindowedRms as ZsWindowedRms,
};

// ============================================================================
// EnvelopeFollower
// ============================================================================

/// Internal enum for handling different channel counts.
enum EnvelopeFollowerInner {
    C1(ZsEnvelopeFollower<1>),
    C4(ZsEnvelopeFollower<4>),
    C8(ZsEnvelopeFollower<8>),
    C16(ZsEnvelopeFollower<16>),
    C32(ZsEnvelopeFollower<32>),
    C64(ZsEnvelopeFollower<64>),
}

/// Envelope follower that tracks signal amplitude.
///
/// Extracts the amplitude envelope by rectifying the signal and applying
/// exponential smoothing. Supports separate attack and release time constants.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create envelope follower: 10ms attack, 100ms release at 250 Hz
/// env = zbci.EnvelopeFollower(
///     channels=8,
///     sample_rate=250.0,
///     attack_time=0.010,
///     release_time=0.100,
///     rectification='absolute'
/// )
///
/// # Process samples
/// sample = np.array([0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.6, 0.3], dtype=np.float32)
/// envelope = env.process(sample)
/// ```
#[pyclass]
pub struct EnvelopeFollower {
    inner: EnvelopeFollowerInner,
    channels: usize,
    sample_rate: f32,
    attack_time: f32,
    release_time: f32,
    rectification: String,
}

#[pymethods]
impl EnvelopeFollower {
    /// Create a new envelope follower.
    ///
    /// Args:
    ///     channels (int): Number of channels (1, 4, 8, 16, 32, or 64).
    ///     sample_rate (float): Sample rate in Hz.
    ///     attack_time (float): Attack time in seconds (response to rising amplitude).
    ///     release_time (float): Release time in seconds (response to falling amplitude).
    ///     rectification (str): Rectification method - 'absolute' or 'squared'.
    ///
    /// Returns:
    ///     EnvelopeFollower: A new envelope follower.
    #[new]
    #[pyo3(signature = (channels, sample_rate, attack_time, release_time, rectification="absolute"))]
    fn new(
        channels: usize,
        sample_rate: f32,
        attack_time: f32,
        release_time: f32,
        rectification: &str,
    ) -> PyResult<Self> {
        let rect = match rectification {
            "absolute" => ZsRectification::Absolute,
            "squared" => ZsRectification::Squared,
            _ => {
                return Err(PyValueError::new_err(
                    "rectification must be 'absolute' or 'squared'",
                ))
            }
        };

        let inner = match channels {
            1 => EnvelopeFollowerInner::C1(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
            4 => EnvelopeFollowerInner::C4(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
            8 => EnvelopeFollowerInner::C8(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
            16 => EnvelopeFollowerInner::C16(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
            32 => EnvelopeFollowerInner::C32(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
            64 => EnvelopeFollowerInner::C64(ZsEnvelopeFollower::new(
                sample_rate,
                attack_time,
                release_time,
                rect,
            )),
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
            sample_rate,
            attack_time,
            release_time,
            rectification: rectification.to_string(),
        })
    }

    /// Create an envelope follower with symmetric attack/release.
    #[staticmethod]
    #[pyo3(signature = (channels, sample_rate, smoothing_time, rectification="absolute"))]
    fn symmetric(
        channels: usize,
        sample_rate: f32,
        smoothing_time: f32,
        rectification: &str,
    ) -> PyResult<Self> {
        Self::new(
            channels,
            sample_rate,
            smoothing_time,
            smoothing_time,
            rectification,
        )
    }

    /// Process a single multi-channel sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float32 array with shape (channels,).
    ///
    /// Returns:
    ///     np.ndarray: Current envelope values for each channel.
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &mut self.inner {
            EnvelopeFollowerInner::C1(env) => {
                let s: [f32; 1] = [sample_slice[0]];
                env.process(&s).to_vec()
            }
            EnvelopeFollowerInner::C4(env) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                env.process(&s).to_vec()
            }
            EnvelopeFollowerInner::C8(env) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                env.process(&s).to_vec()
            }
            EnvelopeFollowerInner::C16(env) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                env.process(&s).to_vec()
            }
            EnvelopeFollowerInner::C32(env) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                env.process(&s).to_vec()
            }
            EnvelopeFollowerInner::C64(env) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                env.process(&s).to_vec()
            }
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Process a block of samples.
    ///
    /// Args:
    ///     block (np.ndarray): Block as 2D float32 array (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Envelope values as 2D float32 array (samples, channels).
    fn process_block<'py>(
        &mut self,
        py: Python<'py>,
        block: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = block.shape();
        if shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Block has {} channels, expected {}",
                shape[1], self.channels
            )));
        }

        let data = block.as_slice()?;
        let num_samples = shape[0];
        let mut output = vec![0.0f32; num_samples * self.channels];

        macro_rules! process_block {
            ($env:expr, $c:expr) => {{
                for row in 0..num_samples {
                    let mut input = [0.0f32; $c];
                    for col in 0..$c {
                        input[col] = data[row * $c + col];
                    }
                    let envelope = $env.process(&input);
                    for (col, &e) in envelope.iter().enumerate() {
                        output[row * $c + col] = e;
                    }
                }
            }};
        }

        match &mut self.inner {
            EnvelopeFollowerInner::C1(env) => process_block!(env, 1),
            EnvelopeFollowerInner::C4(env) => process_block!(env, 4),
            EnvelopeFollowerInner::C8(env) => process_block!(env, 8),
            EnvelopeFollowerInner::C16(env) => process_block!(env, 16),
            EnvelopeFollowerInner::C32(env) => process_block!(env, 32),
            EnvelopeFollowerInner::C64(env) => process_block!(env, 64),
        }

        Ok(PyArray2::from_vec2(
            py,
            &output
                .chunks(self.channels)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )?)
    }

    /// Get the current envelope values.
    fn current<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let vals: Vec<f32> = match &self.inner {
            EnvelopeFollowerInner::C1(env) => env.current().to_vec(),
            EnvelopeFollowerInner::C4(env) => env.current().to_vec(),
            EnvelopeFollowerInner::C8(env) => env.current().to_vec(),
            EnvelopeFollowerInner::C16(env) => env.current().to_vec(),
            EnvelopeFollowerInner::C32(env) => env.current().to_vec(),
            EnvelopeFollowerInner::C64(env) => env.current().to_vec(),
        };
        PyArray1::from_vec(py, vals)
    }

    /// Reset the envelope to zero.
    fn reset(&mut self) {
        match &mut self.inner {
            EnvelopeFollowerInner::C1(env) => env.reset(),
            EnvelopeFollowerInner::C4(env) => env.reset(),
            EnvelopeFollowerInner::C8(env) => env.reset(),
            EnvelopeFollowerInner::C16(env) => env.reset(),
            EnvelopeFollowerInner::C32(env) => env.reset(),
            EnvelopeFollowerInner::C64(env) => env.reset(),
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the sample rate.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the attack time.
    #[getter]
    fn attack_time(&self) -> f32 {
        self.attack_time
    }

    /// Get the release time.
    #[getter]
    fn release_time(&self) -> f32 {
        self.release_time
    }

    /// Get the rectification method.
    #[getter]
    fn rectification(&self) -> &str {
        &self.rectification
    }

    fn __repr__(&self) -> String {
        format!(
            "EnvelopeFollower(channels={}, sample_rate={}, attack={}, release={}, rectification='{}')",
            self.channels, self.sample_rate, self.attack_time, self.release_time, self.rectification
        )
    }
}

// ============================================================================
// WindowedRms
// ============================================================================

/// Internal enum for handling different channel/window combinations.
enum WindowedRmsInner {
    // Common window sizes for 1, 4, 8 channels
    C1W16(ZsWindowedRms<1, 16>),
    C1W32(ZsWindowedRms<1, 32>),
    C1W64(ZsWindowedRms<1, 64>),
    C1W128(ZsWindowedRms<1, 128>),
    C4W16(ZsWindowedRms<4, 16>),
    C4W32(ZsWindowedRms<4, 32>),
    C4W64(ZsWindowedRms<4, 64>),
    C4W128(ZsWindowedRms<4, 128>),
    C8W16(ZsWindowedRms<8, 16>),
    C8W32(ZsWindowedRms<8, 32>),
    C8W64(ZsWindowedRms<8, 64>),
    C8W128(ZsWindowedRms<8, 128>),
    C16W16(ZsWindowedRms<16, 16>),
    C16W32(ZsWindowedRms<16, 32>),
    C16W64(ZsWindowedRms<16, 64>),
    C16W128(ZsWindowedRms<16, 128>),
    C32W16(ZsWindowedRms<32, 16>),
    C32W32(ZsWindowedRms<32, 32>),
    C32W64(ZsWindowedRms<32, 64>),
    C32W128(ZsWindowedRms<32, 128>),
    C64W16(ZsWindowedRms<64, 16>),
    C64W32(ZsWindowedRms<64, 32>),
    C64W64(ZsWindowedRms<64, 64>),
    C64W128(ZsWindowedRms<64, 128>),
}

/// Windowed RMS (root mean square) tracker.
///
/// Efficiently tracks RMS and power for multi-channel signals using a sliding
/// window. Essential for EEG power features and amplitude tracking.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create RMS tracker: 64-sample window for 8 channels
/// rms = zbci.WindowedRms(channels=8, window_size=64)
///
/// # Process samples
/// for _ in range(100):
///     sample = np.random.randn(8).astype(np.float32)
///     rms.process(sample)
///
/// # Get RMS values (available after window_size samples)
/// rms_vals = rms.rms()  # Returns array or None if not ready
/// ```
#[pyclass]
pub struct WindowedRms {
    inner: WindowedRmsInner,
    channels: usize,
    window_size: usize,
}

#[pymethods]
impl WindowedRms {
    /// Create a new windowed RMS tracker.
    ///
    /// Args:
    ///     channels (int): Number of channels (1, 4, 8, 16, 32, or 64).
    ///     window_size (int): Window size in samples (16, 32, 64, or 128).
    ///
    /// Returns:
    ///     WindowedRms: A new RMS tracker.
    #[new]
    fn new(channels: usize, window_size: usize) -> PyResult<Self> {
        let inner = match (channels, window_size) {
            (1, 16) => WindowedRmsInner::C1W16(ZsWindowedRms::new()),
            (1, 32) => WindowedRmsInner::C1W32(ZsWindowedRms::new()),
            (1, 64) => WindowedRmsInner::C1W64(ZsWindowedRms::new()),
            (1, 128) => WindowedRmsInner::C1W128(ZsWindowedRms::new()),
            (4, 16) => WindowedRmsInner::C4W16(ZsWindowedRms::new()),
            (4, 32) => WindowedRmsInner::C4W32(ZsWindowedRms::new()),
            (4, 64) => WindowedRmsInner::C4W64(ZsWindowedRms::new()),
            (4, 128) => WindowedRmsInner::C4W128(ZsWindowedRms::new()),
            (8, 16) => WindowedRmsInner::C8W16(ZsWindowedRms::new()),
            (8, 32) => WindowedRmsInner::C8W32(ZsWindowedRms::new()),
            (8, 64) => WindowedRmsInner::C8W64(ZsWindowedRms::new()),
            (8, 128) => WindowedRmsInner::C8W128(ZsWindowedRms::new()),
            (16, 16) => WindowedRmsInner::C16W16(ZsWindowedRms::new()),
            (16, 32) => WindowedRmsInner::C16W32(ZsWindowedRms::new()),
            (16, 64) => WindowedRmsInner::C16W64(ZsWindowedRms::new()),
            (16, 128) => WindowedRmsInner::C16W128(ZsWindowedRms::new()),
            (32, 16) => WindowedRmsInner::C32W16(ZsWindowedRms::new()),
            (32, 32) => WindowedRmsInner::C32W32(ZsWindowedRms::new()),
            (32, 64) => WindowedRmsInner::C32W64(ZsWindowedRms::new()),
            (32, 128) => WindowedRmsInner::C32W128(ZsWindowedRms::new()),
            (64, 16) => WindowedRmsInner::C64W16(ZsWindowedRms::new()),
            (64, 32) => WindowedRmsInner::C64W32(ZsWindowedRms::new()),
            (64, 64) => WindowedRmsInner::C64W64(ZsWindowedRms::new()),
            (64, 128) => WindowedRmsInner::C64W128(ZsWindowedRms::new()),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported channels/window_size: ({}, {}). \
                     Channels: 1, 4, 8, 16, 32, 64. Window sizes: 16, 32, 64, 128.",
                    channels, window_size
                )))
            }
        };

        Ok(Self {
            inner,
            channels,
            window_size,
        })
    }

    /// Process a single multi-channel sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float32 array.
    fn process(&mut self, sample: PyReadonlyArray1<f32>) -> PyResult<()> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        macro_rules! do_process {
            ($rms:expr, $c:expr) => {{
                let mut input = [0.0f32; $c];
                for (i, &v) in sample_slice.iter().enumerate() {
                    input[i] = v;
                }
                $rms.process(&input);
            }};
        }

        match &mut self.inner {
            WindowedRmsInner::C1W16(rms) => do_process!(rms, 1),
            WindowedRmsInner::C1W32(rms) => do_process!(rms, 1),
            WindowedRmsInner::C1W64(rms) => do_process!(rms, 1),
            WindowedRmsInner::C1W128(rms) => do_process!(rms, 1),
            WindowedRmsInner::C4W16(rms) => do_process!(rms, 4),
            WindowedRmsInner::C4W32(rms) => do_process!(rms, 4),
            WindowedRmsInner::C4W64(rms) => do_process!(rms, 4),
            WindowedRmsInner::C4W128(rms) => do_process!(rms, 4),
            WindowedRmsInner::C8W16(rms) => do_process!(rms, 8),
            WindowedRmsInner::C8W32(rms) => do_process!(rms, 8),
            WindowedRmsInner::C8W64(rms) => do_process!(rms, 8),
            WindowedRmsInner::C8W128(rms) => do_process!(rms, 8),
            WindowedRmsInner::C16W16(rms) => do_process!(rms, 16),
            WindowedRmsInner::C16W32(rms) => do_process!(rms, 16),
            WindowedRmsInner::C16W64(rms) => do_process!(rms, 16),
            WindowedRmsInner::C16W128(rms) => do_process!(rms, 16),
            WindowedRmsInner::C32W16(rms) => do_process!(rms, 32),
            WindowedRmsInner::C32W32(rms) => do_process!(rms, 32),
            WindowedRmsInner::C32W64(rms) => do_process!(rms, 32),
            WindowedRmsInner::C32W128(rms) => do_process!(rms, 32),
            WindowedRmsInner::C64W16(rms) => do_process!(rms, 64),
            WindowedRmsInner::C64W32(rms) => do_process!(rms, 64),
            WindowedRmsInner::C64W64(rms) => do_process!(rms, 64),
            WindowedRmsInner::C64W128(rms) => do_process!(rms, 64),
        }

        Ok(())
    }

    /// Get the current RMS values.
    ///
    /// Returns:
    ///     np.ndarray: RMS values for each channel, or None if window not full.
    fn rms<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        macro_rules! get_rms {
            ($rms:expr) => {
                $rms.rms().map(|v| v.to_vec())
            };
        }

        let vals = match &self.inner {
            WindowedRmsInner::C1W16(rms) => get_rms!(rms),
            WindowedRmsInner::C1W32(rms) => get_rms!(rms),
            WindowedRmsInner::C1W64(rms) => get_rms!(rms),
            WindowedRmsInner::C1W128(rms) => get_rms!(rms),
            WindowedRmsInner::C4W16(rms) => get_rms!(rms),
            WindowedRmsInner::C4W32(rms) => get_rms!(rms),
            WindowedRmsInner::C4W64(rms) => get_rms!(rms),
            WindowedRmsInner::C4W128(rms) => get_rms!(rms),
            WindowedRmsInner::C8W16(rms) => get_rms!(rms),
            WindowedRmsInner::C8W32(rms) => get_rms!(rms),
            WindowedRmsInner::C8W64(rms) => get_rms!(rms),
            WindowedRmsInner::C8W128(rms) => get_rms!(rms),
            WindowedRmsInner::C16W16(rms) => get_rms!(rms),
            WindowedRmsInner::C16W32(rms) => get_rms!(rms),
            WindowedRmsInner::C16W64(rms) => get_rms!(rms),
            WindowedRmsInner::C16W128(rms) => get_rms!(rms),
            WindowedRmsInner::C32W16(rms) => get_rms!(rms),
            WindowedRmsInner::C32W32(rms) => get_rms!(rms),
            WindowedRmsInner::C32W64(rms) => get_rms!(rms),
            WindowedRmsInner::C32W128(rms) => get_rms!(rms),
            WindowedRmsInner::C64W16(rms) => get_rms!(rms),
            WindowedRmsInner::C64W32(rms) => get_rms!(rms),
            WindowedRmsInner::C64W64(rms) => get_rms!(rms),
            WindowedRmsInner::C64W128(rms) => get_rms!(rms),
        };

        vals.map(|v| PyArray1::from_vec(py, v))
    }

    /// Get the current power values (more efficient than RMS).
    ///
    /// Returns:
    ///     np.ndarray: Power values for each channel, or None if window not full.
    fn power<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        macro_rules! get_power {
            ($rms:expr) => {
                $rms.power().map(|v| v.to_vec())
            };
        }

        let vals = match &self.inner {
            WindowedRmsInner::C1W16(rms) => get_power!(rms),
            WindowedRmsInner::C1W32(rms) => get_power!(rms),
            WindowedRmsInner::C1W64(rms) => get_power!(rms),
            WindowedRmsInner::C1W128(rms) => get_power!(rms),
            WindowedRmsInner::C4W16(rms) => get_power!(rms),
            WindowedRmsInner::C4W32(rms) => get_power!(rms),
            WindowedRmsInner::C4W64(rms) => get_power!(rms),
            WindowedRmsInner::C4W128(rms) => get_power!(rms),
            WindowedRmsInner::C8W16(rms) => get_power!(rms),
            WindowedRmsInner::C8W32(rms) => get_power!(rms),
            WindowedRmsInner::C8W64(rms) => get_power!(rms),
            WindowedRmsInner::C8W128(rms) => get_power!(rms),
            WindowedRmsInner::C16W16(rms) => get_power!(rms),
            WindowedRmsInner::C16W32(rms) => get_power!(rms),
            WindowedRmsInner::C16W64(rms) => get_power!(rms),
            WindowedRmsInner::C16W128(rms) => get_power!(rms),
            WindowedRmsInner::C32W16(rms) => get_power!(rms),
            WindowedRmsInner::C32W32(rms) => get_power!(rms),
            WindowedRmsInner::C32W64(rms) => get_power!(rms),
            WindowedRmsInner::C32W128(rms) => get_power!(rms),
            WindowedRmsInner::C64W16(rms) => get_power!(rms),
            WindowedRmsInner::C64W32(rms) => get_power!(rms),
            WindowedRmsInner::C64W64(rms) => get_power!(rms),
            WindowedRmsInner::C64W128(rms) => get_power!(rms),
        };

        vals.map(|v| PyArray1::from_vec(py, v))
    }

    /// Get the number of samples processed.
    #[getter]
    fn count(&self) -> usize {
        macro_rules! get_count {
            ($rms:expr) => {
                $rms.count()
            };
        }

        match &self.inner {
            WindowedRmsInner::C1W16(rms) => get_count!(rms),
            WindowedRmsInner::C1W32(rms) => get_count!(rms),
            WindowedRmsInner::C1W64(rms) => get_count!(rms),
            WindowedRmsInner::C1W128(rms) => get_count!(rms),
            WindowedRmsInner::C4W16(rms) => get_count!(rms),
            WindowedRmsInner::C4W32(rms) => get_count!(rms),
            WindowedRmsInner::C4W64(rms) => get_count!(rms),
            WindowedRmsInner::C4W128(rms) => get_count!(rms),
            WindowedRmsInner::C8W16(rms) => get_count!(rms),
            WindowedRmsInner::C8W32(rms) => get_count!(rms),
            WindowedRmsInner::C8W64(rms) => get_count!(rms),
            WindowedRmsInner::C8W128(rms) => get_count!(rms),
            WindowedRmsInner::C16W16(rms) => get_count!(rms),
            WindowedRmsInner::C16W32(rms) => get_count!(rms),
            WindowedRmsInner::C16W64(rms) => get_count!(rms),
            WindowedRmsInner::C16W128(rms) => get_count!(rms),
            WindowedRmsInner::C32W16(rms) => get_count!(rms),
            WindowedRmsInner::C32W32(rms) => get_count!(rms),
            WindowedRmsInner::C32W64(rms) => get_count!(rms),
            WindowedRmsInner::C32W128(rms) => get_count!(rms),
            WindowedRmsInner::C64W16(rms) => get_count!(rms),
            WindowedRmsInner::C64W32(rms) => get_count!(rms),
            WindowedRmsInner::C64W64(rms) => get_count!(rms),
            WindowedRmsInner::C64W128(rms) => get_count!(rms),
        }
    }

    /// Check if window is full (RMS values are ready).
    #[getter]
    fn is_ready(&self) -> bool {
        self.count() >= self.window_size
    }

    /// Reset the tracker to initial state.
    fn reset(&mut self) {
        macro_rules! do_reset {
            ($rms:expr) => {
                $rms.reset()
            };
        }

        match &mut self.inner {
            WindowedRmsInner::C1W16(rms) => do_reset!(rms),
            WindowedRmsInner::C1W32(rms) => do_reset!(rms),
            WindowedRmsInner::C1W64(rms) => do_reset!(rms),
            WindowedRmsInner::C1W128(rms) => do_reset!(rms),
            WindowedRmsInner::C4W16(rms) => do_reset!(rms),
            WindowedRmsInner::C4W32(rms) => do_reset!(rms),
            WindowedRmsInner::C4W64(rms) => do_reset!(rms),
            WindowedRmsInner::C4W128(rms) => do_reset!(rms),
            WindowedRmsInner::C8W16(rms) => do_reset!(rms),
            WindowedRmsInner::C8W32(rms) => do_reset!(rms),
            WindowedRmsInner::C8W64(rms) => do_reset!(rms),
            WindowedRmsInner::C8W128(rms) => do_reset!(rms),
            WindowedRmsInner::C16W16(rms) => do_reset!(rms),
            WindowedRmsInner::C16W32(rms) => do_reset!(rms),
            WindowedRmsInner::C16W64(rms) => do_reset!(rms),
            WindowedRmsInner::C16W128(rms) => do_reset!(rms),
            WindowedRmsInner::C32W16(rms) => do_reset!(rms),
            WindowedRmsInner::C32W32(rms) => do_reset!(rms),
            WindowedRmsInner::C32W64(rms) => do_reset!(rms),
            WindowedRmsInner::C32W128(rms) => do_reset!(rms),
            WindowedRmsInner::C64W16(rms) => do_reset!(rms),
            WindowedRmsInner::C64W32(rms) => do_reset!(rms),
            WindowedRmsInner::C64W64(rms) => do_reset!(rms),
            WindowedRmsInner::C64W128(rms) => do_reset!(rms),
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the window size.
    #[getter]
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn __repr__(&self) -> String {
        format!(
            "WindowedRms(channels={}, window_size={}, is_ready={})",
            self.channels,
            self.window_size,
            self.is_ready()
        )
    }
}

// ============================================================================
// HilbertTransform
// ============================================================================

/// Internal enum for handling different FFT sizes.
enum HilbertTransformInner {
    N64(ZsHilbertTransform<64>),
    N128(ZsHilbertTransform<128>),
    N256(ZsHilbertTransform<256>),
    N512(ZsHilbertTransform<512>),
    N1024(ZsHilbertTransform<1024>),
    N2048(ZsHilbertTransform<2048>),
}

/// Hilbert Transform for computing analytic signals.
///
/// Computes the analytic signal from a real signal using FFT, enabling
/// extraction of instantaneous amplitude, phase, and frequency. Essential
/// for phase-amplitude coupling analysis and real-time neurofeedback.
///
/// # Important
///
/// The Hilbert transform assumes **narrowband signals**. Bandpass filter your
/// signal before applying this transform to get meaningful results.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create Hilbert transform for 256-sample signals
/// hilbert = zbci.HilbertTransform(size=256)
///
/// # Generate a narrowband signal (should bandpass filter real data)
/// t = np.linspace(0, 1, 256)
/// signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
///
/// # Extract instantaneous amplitude (envelope)
/// amplitude = hilbert.instantaneous_amplitude(signal)
///
/// # Extract instantaneous phase
/// phase = hilbert.instantaneous_phase(signal)
/// ```
#[pyclass]
pub struct HilbertTransform {
    inner: HilbertTransformInner,
    size: usize,
}

#[pymethods]
impl HilbertTransform {
    /// Create a new Hilbert transform processor.
    ///
    /// Args:
    ///     size (int): Signal length (64, 128, 256, 512, 1024, or 2048).
    ///
    /// Returns:
    ///     HilbertTransform: A new Hilbert transform processor.
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let inner = match size {
            64 => HilbertTransformInner::N64(ZsHilbertTransform::new()),
            128 => HilbertTransformInner::N128(ZsHilbertTransform::new()),
            256 => HilbertTransformInner::N256(ZsHilbertTransform::new()),
            512 => HilbertTransformInner::N512(ZsHilbertTransform::new()),
            1024 => HilbertTransformInner::N1024(ZsHilbertTransform::new()),
            2048 => HilbertTransformInner::N2048(ZsHilbertTransform::new()),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "size must be 64, 128, 256, 512, 1024, or 2048, got {}",
                    size
                )))
            }
        };

        Ok(Self { inner, size })
    }

    /// Compute the Hilbert transform (imaginary part of analytic signal).
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Hilbert-transformed signal (90° phase shifted).
    fn transform<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_transform {
            ($hilbert:expr, $n:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = [0.0f32; $n];
                $hilbert.transform(&input, &mut output);
                output.to_vec()
            }};
        }

        let result = match &self.inner {
            HilbertTransformInner::N64(h) => do_transform!(h, 64),
            HilbertTransformInner::N128(h) => do_transform!(h, 128),
            HilbertTransformInner::N256(h) => do_transform!(h, 256),
            HilbertTransformInner::N512(h) => do_transform!(h, 512),
            HilbertTransformInner::N1024(h) => do_transform!(h, 1024),
            HilbertTransformInner::N2048(h) => do_transform!(h, 2048),
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Compute the instantaneous amplitude (envelope).
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Instantaneous amplitude envelope.
    fn instantaneous_amplitude<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_amplitude {
            ($hilbert:expr, $n:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = [0.0f32; $n];
                $hilbert.instantaneous_amplitude(&input, &mut output);
                output.to_vec()
            }};
        }

        let result = match &self.inner {
            HilbertTransformInner::N64(h) => do_amplitude!(h, 64),
            HilbertTransformInner::N128(h) => do_amplitude!(h, 128),
            HilbertTransformInner::N256(h) => do_amplitude!(h, 256),
            HilbertTransformInner::N512(h) => do_amplitude!(h, 512),
            HilbertTransformInner::N1024(h) => do_amplitude!(h, 1024),
            HilbertTransformInner::N2048(h) => do_amplitude!(h, 2048),
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Compute the instantaneous phase.
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Instantaneous phase in radians [-π, π].
    fn instantaneous_phase<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_phase {
            ($hilbert:expr, $n:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = [0.0f32; $n];
                $hilbert.instantaneous_phase(&input, &mut output);
                output.to_vec()
            }};
        }

        let result = match &self.inner {
            HilbertTransformInner::N64(h) => do_phase!(h, 64),
            HilbertTransformInner::N128(h) => do_phase!(h, 128),
            HilbertTransformInner::N256(h) => do_phase!(h, 256),
            HilbertTransformInner::N512(h) => do_phase!(h, 512),
            HilbertTransformInner::N1024(h) => do_phase!(h, 1024),
            HilbertTransformInner::N2048(h) => do_phase!(h, 2048),
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Compute the instantaneous frequency.
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///     sample_rate (float): Sample rate in Hz.
    ///
    /// Returns:
    ///     np.ndarray: Instantaneous frequency in Hz (length N-1).
    fn instantaneous_frequency<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
        sample_rate: f32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_frequency {
            ($hilbert:expr, $n:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = vec![0.0f32; $n - 1];
                $hilbert.instantaneous_frequency(&input, &mut output, sample_rate);
                output
            }};
        }

        let result = match &self.inner {
            HilbertTransformInner::N64(h) => do_frequency!(h, 64),
            HilbertTransformInner::N128(h) => do_frequency!(h, 128),
            HilbertTransformInner::N256(h) => do_frequency!(h, 256),
            HilbertTransformInner::N512(h) => do_frequency!(h, 512),
            HilbertTransformInner::N1024(h) => do_frequency!(h, 1024),
            HilbertTransformInner::N2048(h) => do_frequency!(h, 2048),
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Get the signal size.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    fn __repr__(&self) -> String {
        format!("HilbertTransform(size={})", self.size)
    }
}
