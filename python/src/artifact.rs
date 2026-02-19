//! Python bindings for artifact detection primitives.
//!
//! Provides building blocks for detecting and rejecting artifacts in neural signals.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{ArtifactDetector as ZsArtifactDetector, ZscoreArtifact as ZsZscoreArtifact};

// ============================================================================
// ArtifactDetector
// ============================================================================

/// Internal enum for handling different channel counts.
enum ArtifactDetectorInner {
    C1(ZsArtifactDetector<1>),
    C4(ZsArtifactDetector<4>),
    C8(ZsArtifactDetector<8>),
    C16(ZsArtifactDetector<16>),
    C32(ZsArtifactDetector<32>),
    C64(ZsArtifactDetector<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic {
        channels: usize,
        amplitude_threshold: f32,
        gradient_threshold: f32,
        prev_sample: Vec<f32>,
        initialized: bool,
    },
}

/// Multi-channel artifact detector using amplitude and gradient thresholds.
///
/// Detects artifacts sample-by-sample based on:
/// - **Amplitude**: Flags samples where |value| exceeds amplitude threshold
/// - **Gradient**: Flags samples where |value - previous| exceeds gradient threshold
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create detector for 8 channels with amplitude=100, gradient=50
/// detector = zbci.ArtifactDetector(channels=8, amplitude_threshold=100.0, gradient_threshold=50.0)
///
/// # Process samples
/// sample = np.array([10.0, 200.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32)
/// artifacts = detector.detect(sample)  # Returns boolean array
/// # artifacts[1] is True because |200| > 100
/// ```
#[pyclass]
pub struct ArtifactDetector {
    inner: ArtifactDetectorInner,
    channels: usize,
}

#[pymethods]
impl ArtifactDetector {
    /// Create a new artifact detector.
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///     amplitude_threshold (float): Maximum allowed absolute amplitude.
    ///     gradient_threshold (float): Maximum allowed sample-to-sample difference.
    ///
    /// Returns:
    ///     ArtifactDetector: A new artifact detector.
    #[new]
    fn new(channels: usize, amplitude_threshold: f32, gradient_threshold: f32) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be >= 1"));
        }

        let inner = match channels {
            1 => ArtifactDetectorInner::C1(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            4 => ArtifactDetectorInner::C4(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            8 => ArtifactDetectorInner::C8(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            16 => ArtifactDetectorInner::C16(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            32 => ArtifactDetectorInner::C32(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            64 => ArtifactDetectorInner::C64(ZsArtifactDetector::new(
                amplitude_threshold,
                gradient_threshold,
            )),
            _ => ArtifactDetectorInner::Dynamic {
                channels,
                amplitude_threshold,
                gradient_threshold,
                prev_sample: vec![0.0; channels],
                initialized: false,
            },
        };

        Ok(Self { inner, channels })
    }

    /// Create a detector with only amplitude threshold (no gradient check).
    #[staticmethod]
    fn amplitude_only(channels: usize, threshold: f32) -> PyResult<Self> {
        Self::new(channels, threshold, f32::INFINITY)
    }

    /// Create a detector with only gradient threshold (no amplitude check).
    #[staticmethod]
    fn gradient_only(channels: usize, threshold: f32) -> PyResult<Self> {
        Self::new(channels, f32::INFINITY, threshold)
    }

    /// Detect artifacts in a multi-channel sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float32 array with shape (channels,).
    ///
    /// Returns:
    ///     np.ndarray: Boolean array where True indicates artifact.
    fn detect<'py>(
        &mut self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &mut self.inner {
            ArtifactDetectorInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ArtifactDetectorInner::Dynamic {
                channels,
                amplitude_threshold,
                gradient_threshold,
                prev_sample,
                initialized,
            } => {
                let mut result = vec![false; *channels];
                for (i, &s) in sample_slice.iter().enumerate() {
                    let amplitude_bad = s.abs() > *amplitude_threshold;
                    let gradient_bad =
                        *initialized && (s - prev_sample[i]).abs() > *gradient_threshold;
                    result[i] = amplitude_bad || gradient_bad;
                }
                prev_sample.copy_from_slice(sample_slice);
                *initialized = true;
                result
            }
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Check if any channel has an artifact.
    fn detect_any(&mut self, sample: PyReadonlyArray1<f32>) -> PyResult<bool> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &mut self.inner {
            ArtifactDetectorInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.detect_any(&s)
            }
            ArtifactDetectorInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.detect_any(&s)
            }
            ArtifactDetectorInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.detect_any(&s)
            }
            ArtifactDetectorInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.detect_any(&s)
            }
            ArtifactDetectorInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.detect_any(&s)
            }
            ArtifactDetectorInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.detect_any(&s)
            }
            ArtifactDetectorInner::Dynamic {
                channels,
                amplitude_threshold,
                gradient_threshold,
                prev_sample,
                initialized,
            } => {
                let mut any_artifact = false;
                for (i, &s) in sample_slice.iter().enumerate() {
                    let amplitude_bad = s.abs() > *amplitude_threshold;
                    let gradient_bad =
                        *initialized && (s - prev_sample[i]).abs() > *gradient_threshold;
                    if amplitude_bad || gradient_bad {
                        any_artifact = true;
                    }
                }
                prev_sample.copy_from_slice(sample_slice);
                *initialized = true;
                any_artifact
            }
        };

        Ok(result)
    }

    /// Count the number of channels with artifacts.
    fn detect_count(&mut self, sample: PyReadonlyArray1<f32>) -> PyResult<usize> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &mut self.inner {
            ArtifactDetectorInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.detect_count(&s)
            }
            ArtifactDetectorInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.detect_count(&s)
            }
            ArtifactDetectorInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.detect_count(&s)
            }
            ArtifactDetectorInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.detect_count(&s)
            }
            ArtifactDetectorInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.detect_count(&s)
            }
            ArtifactDetectorInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.detect_count(&s)
            }
            ArtifactDetectorInner::Dynamic {
                channels,
                amplitude_threshold,
                gradient_threshold,
                prev_sample,
                initialized,
            } => {
                let mut count = 0;
                for (i, &s) in sample_slice.iter().enumerate() {
                    let amplitude_bad = s.abs() > *amplitude_threshold;
                    let gradient_bad =
                        *initialized && (s - prev_sample[i]).abs() > *gradient_threshold;
                    if amplitude_bad || gradient_bad {
                        count += 1;
                    }
                }
                prev_sample.copy_from_slice(sample_slice);
                *initialized = true;
                count
            }
        };

        Ok(result)
    }

    /// Reset the detector state.
    fn reset(&mut self) {
        match &mut self.inner {
            ArtifactDetectorInner::C1(det) => det.reset(),
            ArtifactDetectorInner::C4(det) => det.reset(),
            ArtifactDetectorInner::C8(det) => det.reset(),
            ArtifactDetectorInner::C16(det) => det.reset(),
            ArtifactDetectorInner::C32(det) => det.reset(),
            ArtifactDetectorInner::C64(det) => det.reset(),
            ArtifactDetectorInner::Dynamic {
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

    /// Get the amplitude threshold.
    #[getter]
    fn amplitude_threshold(&self) -> f32 {
        match &self.inner {
            ArtifactDetectorInner::C1(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::C4(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::C8(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::C16(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::C32(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::C64(det) => det.amplitude_threshold(),
            ArtifactDetectorInner::Dynamic {
                amplitude_threshold,
                ..
            } => *amplitude_threshold,
        }
    }

    /// Get the gradient threshold.
    #[getter]
    fn gradient_threshold(&self) -> f32 {
        match &self.inner {
            ArtifactDetectorInner::C1(det) => det.gradient_threshold(),
            ArtifactDetectorInner::C4(det) => det.gradient_threshold(),
            ArtifactDetectorInner::C8(det) => det.gradient_threshold(),
            ArtifactDetectorInner::C16(det) => det.gradient_threshold(),
            ArtifactDetectorInner::C32(det) => det.gradient_threshold(),
            ArtifactDetectorInner::C64(det) => det.gradient_threshold(),
            ArtifactDetectorInner::Dynamic {
                gradient_threshold, ..
            } => *gradient_threshold,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ArtifactDetector(channels={}, amplitude={}, gradient={})",
            self.channels,
            self.amplitude_threshold(),
            self.gradient_threshold()
        )
    }
}

// ============================================================================
// ZscoreArtifact
// ============================================================================

/// Internal enum for handling different channel counts.
enum ZscoreArtifactInner {
    C1(ZsZscoreArtifact<1>),
    C4(ZsZscoreArtifact<4>),
    C8(ZsZscoreArtifact<8>),
    C16(ZsZscoreArtifact<16>),
    C32(ZsZscoreArtifact<32>),
    C64(ZsZscoreArtifact<64>),
}

/// Adaptive artifact detection using z-score computation.
///
/// Detects artifacts by computing z-scores from streaming statistics.
/// Samples with |z| > threshold are flagged as artifacts.
///
/// # Operation Modes
/// - **Calibrating**: Collecting samples, building statistics (count < min_samples)
/// - **Adapting**: Continuously updating statistics and detecting
/// - **Frozen**: Using fixed statistics captured at freeze time
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create detector with 3-sigma threshold, 100 sample warmup
/// detector = zbci.ZscoreArtifact(channels=8, threshold=3.0, min_samples=100)
///
/// # Calibration phase
/// for _ in range(100):
///     sample = np.random.randn(8).astype(np.float32)
///     detector.update(sample)
///
/// # Detection phase
/// sample = np.array([0.5] * 7 + [10.0], dtype=np.float32)  # Channel 7 is outlier
/// artifacts = detector.detect(sample)
/// # artifacts[7] is True
/// ```
#[pyclass]
pub struct ZscoreArtifact {
    inner: ZscoreArtifactInner,
    channels: usize,
}

#[pymethods]
impl ZscoreArtifact {
    /// Create a new z-score artifact detector.
    ///
    /// Args:
    ///     channels (int): Number of channels (1, 4, 8, 16, 32, or 64).
    ///     threshold (float): Z-score threshold for artifact detection (e.g., 3.0 for 3σ).
    ///     min_samples (int): Minimum samples before detection starts (warm-up period).
    ///
    /// Returns:
    ///     ZscoreArtifact: A new z-score artifact detector.
    #[new]
    fn new(channels: usize, threshold: f32, min_samples: u64) -> PyResult<Self> {
        let inner = match channels {
            1 => ZscoreArtifactInner::C1(ZsZscoreArtifact::new(threshold, min_samples)),
            4 => ZscoreArtifactInner::C4(ZsZscoreArtifact::new(threshold, min_samples)),
            8 => ZscoreArtifactInner::C8(ZsZscoreArtifact::new(threshold, min_samples)),
            16 => ZscoreArtifactInner::C16(ZsZscoreArtifact::new(threshold, min_samples)),
            32 => ZscoreArtifactInner::C32(ZsZscoreArtifact::new(threshold, min_samples)),
            64 => ZscoreArtifactInner::C64(ZsZscoreArtifact::new(threshold, min_samples)),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "channels must be 1, 4, 8, 16, 32, or 64, got {}",
                    channels
                )))
            }
        };

        Ok(Self { inner, channels })
    }

    /// Update statistics with a new sample without performing detection.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float32 array.
    fn update(&mut self, sample: PyReadonlyArray1<f32>) -> PyResult<()> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        match &mut self.inner {
            ZscoreArtifactInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.update(&s);
            }
            ZscoreArtifactInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.update(&s);
            }
            ZscoreArtifactInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.update(&s);
            }
            ZscoreArtifactInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.update(&s);
            }
            ZscoreArtifactInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.update(&s);
            }
            ZscoreArtifactInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.update(&s);
            }
        }

        Ok(())
    }

    /// Detect artifacts in a sample.
    ///
    /// Args:
    ///     sample (np.ndarray): Sample as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Boolean array where True indicates artifact.
    ///
    /// Note: During calibration (count < min_samples), returns all False.
    fn detect<'py>(
        &self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &self.inner {
            ZscoreArtifactInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.detect(&s).to_vec()
            }
            ZscoreArtifactInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ZscoreArtifactInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ZscoreArtifactInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ZscoreArtifactInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
            ZscoreArtifactInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.detect(&s).to_vec()
            }
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Update statistics and detect artifacts in one call.
    fn update_and_detect<'py>(
        &mut self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &mut self.inner {
            ZscoreArtifactInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.update_and_detect(&s).to_vec()
            }
            ZscoreArtifactInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.update_and_detect(&s).to_vec()
            }
            ZscoreArtifactInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.update_and_detect(&s).to_vec()
            }
            ZscoreArtifactInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.update_and_detect(&s).to_vec()
            }
            ZscoreArtifactInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.update_and_detect(&s).to_vec()
            }
            ZscoreArtifactInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.update_and_detect(&s).to_vec()
            }
        };

        Ok(PyArray1::from_vec(py, result))
    }

    /// Compute z-scores for each channel.
    ///
    /// Returns:
    ///     np.ndarray: Z-scores for each channel, or None during calibration.
    fn zscore<'py>(
        &self,
        py: Python<'py>,
        sample: PyReadonlyArray1<f32>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let sample_slice = sample.as_slice()?;
        if sample_slice.len() != self.channels {
            return Err(PyValueError::new_err(format!(
                "Sample has {} elements, expected {}",
                sample_slice.len(),
                self.channels
            )));
        }

        let result = match &self.inner {
            ZscoreArtifactInner::C1(det) => {
                let s: [f32; 1] = [sample_slice[0]];
                det.zscore(&s).map(|z| z.to_vec())
            }
            ZscoreArtifactInner::C4(det) => {
                let s: [f32; 4] = sample_slice.try_into().unwrap();
                det.zscore(&s).map(|z| z.to_vec())
            }
            ZscoreArtifactInner::C8(det) => {
                let s: [f32; 8] = sample_slice.try_into().unwrap();
                det.zscore(&s).map(|z| z.to_vec())
            }
            ZscoreArtifactInner::C16(det) => {
                let s: [f32; 16] = sample_slice.try_into().unwrap();
                det.zscore(&s).map(|z| z.to_vec())
            }
            ZscoreArtifactInner::C32(det) => {
                let s: [f32; 32] = sample_slice.try_into().unwrap();
                det.zscore(&s).map(|z| z.to_vec())
            }
            ZscoreArtifactInner::C64(det) => {
                let s: [f32; 64] = sample_slice.try_into().unwrap();
                det.zscore(&s).map(|z| z.to_vec())
            }
        };

        Ok(result.map(|z| PyArray1::from_vec(py, z)))
    }

    /// Freeze the current statistics.
    fn freeze(&mut self) {
        match &mut self.inner {
            ZscoreArtifactInner::C1(det) => det.freeze(),
            ZscoreArtifactInner::C4(det) => det.freeze(),
            ZscoreArtifactInner::C8(det) => det.freeze(),
            ZscoreArtifactInner::C16(det) => det.freeze(),
            ZscoreArtifactInner::C32(det) => det.freeze(),
            ZscoreArtifactInner::C64(det) => det.freeze(),
        }
    }

    /// Unfreeze statistics, resuming adaptive behavior.
    fn unfreeze(&mut self) {
        match &mut self.inner {
            ZscoreArtifactInner::C1(det) => det.unfreeze(),
            ZscoreArtifactInner::C4(det) => det.unfreeze(),
            ZscoreArtifactInner::C8(det) => det.unfreeze(),
            ZscoreArtifactInner::C16(det) => det.unfreeze(),
            ZscoreArtifactInner::C32(det) => det.unfreeze(),
            ZscoreArtifactInner::C64(det) => det.unfreeze(),
        }
    }

    /// Reset all state including statistics.
    fn reset(&mut self) {
        match &mut self.inner {
            ZscoreArtifactInner::C1(det) => det.reset(),
            ZscoreArtifactInner::C4(det) => det.reset(),
            ZscoreArtifactInner::C8(det) => det.reset(),
            ZscoreArtifactInner::C16(det) => det.reset(),
            ZscoreArtifactInner::C32(det) => det.reset(),
            ZscoreArtifactInner::C64(det) => det.reset(),
        }
    }

    /// Check if detector is still in calibration period.
    #[getter]
    fn is_calibrating(&self) -> bool {
        match &self.inner {
            ZscoreArtifactInner::C1(det) => det.is_calibrating(),
            ZscoreArtifactInner::C4(det) => det.is_calibrating(),
            ZscoreArtifactInner::C8(det) => det.is_calibrating(),
            ZscoreArtifactInner::C16(det) => det.is_calibrating(),
            ZscoreArtifactInner::C32(det) => det.is_calibrating(),
            ZscoreArtifactInner::C64(det) => det.is_calibrating(),
        }
    }

    /// Check if statistics are frozen.
    #[getter]
    fn is_frozen(&self) -> bool {
        match &self.inner {
            ZscoreArtifactInner::C1(det) => det.is_frozen(),
            ZscoreArtifactInner::C4(det) => det.is_frozen(),
            ZscoreArtifactInner::C8(det) => det.is_frozen(),
            ZscoreArtifactInner::C16(det) => det.is_frozen(),
            ZscoreArtifactInner::C32(det) => det.is_frozen(),
            ZscoreArtifactInner::C64(det) => det.is_frozen(),
        }
    }

    /// Get the number of samples processed.
    #[getter]
    fn sample_count(&self) -> u64 {
        match &self.inner {
            ZscoreArtifactInner::C1(det) => det.sample_count(),
            ZscoreArtifactInner::C4(det) => det.sample_count(),
            ZscoreArtifactInner::C8(det) => det.sample_count(),
            ZscoreArtifactInner::C16(det) => det.sample_count(),
            ZscoreArtifactInner::C32(det) => det.sample_count(),
            ZscoreArtifactInner::C64(det) => det.sample_count(),
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the z-score threshold.
    #[getter]
    fn threshold(&self) -> f32 {
        match &self.inner {
            ZscoreArtifactInner::C1(det) => det.threshold(),
            ZscoreArtifactInner::C4(det) => det.threshold(),
            ZscoreArtifactInner::C8(det) => det.threshold(),
            ZscoreArtifactInner::C16(det) => det.threshold(),
            ZscoreArtifactInner::C32(det) => det.threshold(),
            ZscoreArtifactInner::C64(det) => det.threshold(),
        }
    }

    /// Get the current mean for each channel.
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let means: Vec<f64> = match &self.inner {
            ZscoreArtifactInner::C1(det) => det.mean().to_vec(),
            ZscoreArtifactInner::C4(det) => det.mean().to_vec(),
            ZscoreArtifactInner::C8(det) => det.mean().to_vec(),
            ZscoreArtifactInner::C16(det) => det.mean().to_vec(),
            ZscoreArtifactInner::C32(det) => det.mean().to_vec(),
            ZscoreArtifactInner::C64(det) => det.mean().to_vec(),
        };
        PyArray1::from_vec(py, means)
    }

    /// Get the current standard deviation for each channel.
    fn std_dev<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let stds: Vec<f64> = match &self.inner {
            ZscoreArtifactInner::C1(det) => det.std_dev().to_vec(),
            ZscoreArtifactInner::C4(det) => det.std_dev().to_vec(),
            ZscoreArtifactInner::C8(det) => det.std_dev().to_vec(),
            ZscoreArtifactInner::C16(det) => det.std_dev().to_vec(),
            ZscoreArtifactInner::C32(det) => det.std_dev().to_vec(),
            ZscoreArtifactInner::C64(det) => det.std_dev().to_vec(),
        };
        PyArray1::from_vec(py, stds)
    }

    fn __repr__(&self) -> String {
        format!(
            "ZscoreArtifact(channels={}, threshold={}, is_calibrating={})",
            self.channels,
            self.threshold(),
            self.is_calibrating()
        )
    }
}
