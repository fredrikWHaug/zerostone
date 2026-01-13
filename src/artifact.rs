//! Artifact detection primitives for BCI signal processing.
//!
//! Provides building blocks for detecting and rejecting artifacts in neural signals.
//! Common artifact sources include eye blinks, muscle activity, and electrode pops.
//!
//! # Primitives
//!
//! - [`ArtifactDetector`] - Threshold-based detection using amplitude and gradient criteria
//! - [`ZscoreArtifact`] - Adaptive detection using streaming z-score computation
//!
//! # Example
//!
//! ```
//! use zerostone::ArtifactDetector;
//!
//! // Create detector with amplitude threshold 100.0 and gradient threshold 50.0
//! let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);
//!
//! // Process samples and check for artifacts
//! let samples = [10.0, 20.0, 150.0, 15.0]; // Channel 2 exceeds amplitude threshold
//! let artifacts = detector.detect(&samples);
//!
//! assert!(!artifacts[0]); // Clean
//! assert!(!artifacts[1]); // Clean
//! assert!(artifacts[2]);  // Artifact (amplitude)
//! assert!(!artifacts[3]); // Clean
//! ```

/// Result of artifact detection for a single sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    /// No artifact detected
    Clean,
    /// Amplitude exceeded threshold
    Amplitude,
    /// Gradient (sample-to-sample difference) exceeded threshold
    Gradient,
    /// Both amplitude and gradient exceeded thresholds
    Both,
}

impl ArtifactType {
    /// Returns true if any artifact was detected.
    #[inline]
    pub fn is_artifact(&self) -> bool {
        !matches!(self, ArtifactType::Clean)
    }

    /// Returns true if no artifact was detected.
    #[inline]
    pub fn is_clean(&self) -> bool {
        matches!(self, ArtifactType::Clean)
    }
}

/// Multi-channel artifact detector using amplitude and gradient thresholds.
///
/// Detects artifacts sample-by-sample based on:
/// - **Amplitude**: Flags samples where |value| exceeds amplitude threshold
/// - **Gradient**: Flags samples where |value - previous| exceeds gradient threshold
///
/// Both thresholds can be disabled by setting to infinity (`f32::INFINITY`).
///
/// # Example
///
/// ```
/// use zerostone::ArtifactDetector;
///
/// // Amplitude threshold 100.0, gradient threshold 50.0
/// let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);
///
/// // First sample - only amplitude check (no previous sample for gradient)
/// let artifacts = detector.detect(&[10.0, 200.0]);
/// assert!(!artifacts[0]); // Clean (|10| < 100)
/// assert!(artifacts[1]);  // Artifact (|200| > 100)
///
/// // Second sample - both amplitude and gradient checks apply
/// let artifacts = detector.detect(&[80.0, 190.0]);
/// assert!(artifacts[0]);  // Artifact (|80 - 10| = 70 > 50 gradient)
/// assert!(artifacts[1]);  // Artifact (|190| > 100 amplitude)
/// ```
pub struct ArtifactDetector<const C: usize> {
    amplitude_threshold: f32,
    gradient_threshold: f32,
    prev_sample: [f32; C],
    initialized: bool,
}

impl<const C: usize> ArtifactDetector<C> {
    /// Creates a new artifact detector with specified thresholds.
    ///
    /// # Arguments
    /// * `amplitude_threshold` - Maximum allowed absolute amplitude
    /// * `gradient_threshold` - Maximum allowed sample-to-sample difference
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// // EEG typically: amplitude ~100-200 µV, gradient ~50-100 µV/sample
    /// let detector: ArtifactDetector<32> = ArtifactDetector::new(150.0, 75.0);
    /// ```
    pub fn new(amplitude_threshold: f32, gradient_threshold: f32) -> Self {
        Self {
            amplitude_threshold,
            gradient_threshold,
            prev_sample: [0.0; C],
            initialized: false,
        }
    }

    /// Creates a detector with only amplitude threshold (no gradient check).
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// let detector: ArtifactDetector<8> = ArtifactDetector::amplitude_only(100.0);
    /// ```
    pub fn amplitude_only(threshold: f32) -> Self {
        Self::new(threshold, f32::INFINITY)
    }

    /// Creates a detector with only gradient threshold (no amplitude check).
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// let detector: ArtifactDetector<8> = ArtifactDetector::gradient_only(50.0);
    /// ```
    pub fn gradient_only(threshold: f32) -> Self {
        Self::new(f32::INFINITY, threshold)
    }

    /// Detects artifacts in a multi-channel sample.
    ///
    /// Returns a boolean array where `true` indicates an artifact was detected
    /// on that channel.
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);
    /// let artifacts = detector.detect(&[10.0, 200.0, 30.0, 40.0]);
    /// assert!(artifacts[1]); // Channel 1 has artifact
    /// ```
    pub fn detect(&mut self, samples: &[f32; C]) -> [bool; C] {
        let mut result = [false; C];

        for (i, (&sample, result_out)) in samples.iter().zip(result.iter_mut()).enumerate() {
            let amplitude_bad = libm::fabsf(sample) > self.amplitude_threshold;
            let gradient_bad = self.initialized
                && libm::fabsf(sample - self.prev_sample[i]) > self.gradient_threshold;

            *result_out = amplitude_bad || gradient_bad;
        }

        // Update state
        self.prev_sample = *samples;
        self.initialized = true;

        result
    }

    /// Detects artifacts with detailed type information.
    ///
    /// Returns the specific type of artifact detected on each channel.
    ///
    /// # Example
    /// ```
    /// use zerostone::{ArtifactDetector, ArtifactType};
    ///
    /// let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);
    ///
    /// // First sample
    /// detector.detect(&[10.0, 10.0]);
    ///
    /// // Second sample with gradient artifact
    /// let types = detector.detect_detailed(&[80.0, 10.0]);
    /// assert_eq!(types[0], ArtifactType::Gradient);
    /// assert_eq!(types[1], ArtifactType::Clean);
    /// ```
    pub fn detect_detailed(&mut self, samples: &[f32; C]) -> [ArtifactType; C] {
        let mut result = [ArtifactType::Clean; C];

        for (i, (&sample, result_out)) in samples.iter().zip(result.iter_mut()).enumerate() {
            let amplitude_bad = libm::fabsf(sample) > self.amplitude_threshold;
            let gradient_bad = self.initialized
                && libm::fabsf(sample - self.prev_sample[i]) > self.gradient_threshold;

            *result_out = match (amplitude_bad, gradient_bad) {
                (false, false) => ArtifactType::Clean,
                (true, false) => ArtifactType::Amplitude,
                (false, true) => ArtifactType::Gradient,
                (true, true) => ArtifactType::Both,
            };
        }

        // Update state
        self.prev_sample = *samples;
        self.initialized = true;

        result
    }

    /// Checks if any channel has an artifact.
    ///
    /// Convenience method that returns true if any channel triggered.
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);
    /// let has_artifact = detector.detect_any(&[10.0, 200.0, 30.0, 40.0]);
    /// assert!(has_artifact); // Channel 1 exceeded amplitude threshold
    /// ```
    pub fn detect_any(&mut self, samples: &[f32; C]) -> bool {
        let artifacts = self.detect(samples);
        artifacts.iter().any(|&a| a)
    }

    /// Counts the number of channels with artifacts.
    ///
    /// # Example
    /// ```
    /// use zerostone::ArtifactDetector;
    ///
    /// let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);
    /// let count = detector.detect_count(&[200.0, 200.0, 30.0, 40.0]);
    /// assert_eq!(count, 2); // Channels 0 and 1 exceeded threshold
    /// ```
    pub fn detect_count(&mut self, samples: &[f32; C]) -> usize {
        let artifacts = self.detect(samples);
        artifacts.iter().filter(|&&a| a).count()
    }

    /// Resets the detector state.
    ///
    /// Clears the previous sample buffer, so the next call to `detect`
    /// will not perform gradient checking.
    pub fn reset(&mut self) {
        self.prev_sample = [0.0; C];
        self.initialized = false;
    }

    /// Returns the amplitude threshold.
    pub fn amplitude_threshold(&self) -> f32 {
        self.amplitude_threshold
    }

    /// Sets the amplitude threshold.
    pub fn set_amplitude_threshold(&mut self, threshold: f32) {
        self.amplitude_threshold = threshold;
    }

    /// Returns the gradient threshold.
    pub fn gradient_threshold(&self) -> f32 {
        self.gradient_threshold
    }

    /// Sets the gradient threshold.
    pub fn set_gradient_threshold(&mut self, threshold: f32) {
        self.gradient_threshold = threshold;
    }

    /// Returns the number of channels.
    pub const fn num_channels(&self) -> usize {
        C
    }
}

impl<const C: usize> Default for ArtifactDetector<C> {
    /// Creates a detector with conservative default thresholds.
    ///
    /// Defaults: amplitude = 100.0, gradient = 50.0
    fn default() -> Self {
        Self::new(100.0, 50.0)
    }
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
///
/// ```
/// use zerostone::ZscoreArtifact;
///
/// let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 100);
///
/// // Calibrate with normal data
/// for i in 0..100 {
///     let val = if i % 2 == 0 { 0.5 } else { -0.5 };
///     detector.update(&[val, val]);
/// }
///
/// // Now detecting: samples with |z| > 3.0 flagged as artifacts
/// let artifacts = detector.detect(&[10.0, 0.5]); // Channel 0 is outlier
/// assert!(artifacts[0]);
/// assert!(!artifacts[1]);
/// ```
pub struct ZscoreArtifact<const C: usize> {
    stats: crate::OnlineStats<C>,
    threshold: f32,
    min_samples: u64,
    frozen_mean: Option<[f64; C]>,
    frozen_std: Option<[f64; C]>,
}

impl<const C: usize> ZscoreArtifact<C> {
    /// Creates a new z-score artifact detector.
    ///
    /// # Arguments
    /// * `threshold` - Z-score threshold for artifact detection (e.g., 3.0 for 3σ)
    /// * `min_samples` - Minimum samples before detection starts (warm-up period)
    ///
    /// # Example
    /// ```
    /// use zerostone::ZscoreArtifact;
    ///
    /// // 3σ threshold with 500-sample calibration period
    /// let detector: ZscoreArtifact<32> = ZscoreArtifact::new(3.0, 500);
    /// ```
    pub fn new(threshold: f32, min_samples: u64) -> Self {
        Self {
            stats: crate::OnlineStats::new(),
            threshold,
            min_samples,
            frozen_mean: None,
            frozen_std: None,
        }
    }

    /// Updates statistics with a new sample without performing detection.
    ///
    /// Use this during calibration to build statistics without checking for artifacts.
    ///
    /// # Example
    /// ```
    /// use zerostone::ZscoreArtifact;
    ///
    /// let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 100);
    ///
    /// // Calibration phase
    /// for _ in 0..100 {
    ///     detector.update(&[0.5, -0.5]);
    /// }
    ///
    /// assert!(!detector.is_calibrating());
    /// ```
    pub fn update(&mut self, samples: &[f32; C]) {
        if self.frozen_mean.is_none() {
            let samples_f64 = samples.map(|x| x as f64);
            self.stats.update(&samples_f64);
        }
    }

    /// Detects artifacts in a sample.
    ///
    /// Returns a boolean array where `true` indicates an artifact (|z| > threshold).
    /// During calibration (count < min_samples), returns all false.
    ///
    /// Note: This method does NOT update statistics. Call [`update`](Self::update)
    /// first if you want to include the sample in statistics.
    ///
    /// # Example
    /// ```
    /// use zerostone::ZscoreArtifact;
    ///
    /// let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);
    ///
    /// // Calibrate
    /// for i in 0..50 {
    ///     let val = if i % 2 == 0 { 1.0 } else { -1.0 };
    ///     detector.update(&[val, val]);
    /// }
    ///
    /// // Detect outliers (std ≈ 1.0, so threshold ≈ 3.0)
    /// let artifacts = detector.detect(&[0.5, 10.0]);
    /// assert!(!artifacts[0]); // |0.5| / 1.0 < 3.0
    /// assert!(artifacts[1]);  // |10.0| / 1.0 > 3.0
    /// ```
    pub fn detect(&self, samples: &[f32; C]) -> [bool; C] {
        if self.stats.count() < self.min_samples {
            return [false; C];
        }

        let mean = self.frozen_mean.unwrap_or_else(|| *self.stats.mean());
        let std = self.frozen_std.unwrap_or_else(|| self.stats.std_dev());

        let mut result = [false; C];
        for (i, (&sample, result_out)) in samples.iter().zip(result.iter_mut()).enumerate() {
            if std[i] > 0.0 {
                let z = libm::fabs((sample as f64 - mean[i]) / std[i]);
                *result_out = z > self.threshold as f64;
            }
        }

        result
    }

    /// Computes z-scores for each channel.
    ///
    /// Returns None during calibration period.
    ///
    /// # Example
    /// ```
    /// use zerostone::ZscoreArtifact;
    ///
    /// let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);
    ///
    /// // During calibration
    /// assert!(detector.zscore(&[1.0, 1.0]).is_none());
    ///
    /// // After calibration
    /// for i in 0..50 {
    ///     let val = if i % 2 == 0 { 1.0 } else { -1.0 };
    ///     detector.update(&[val, val]);
    /// }
    ///
    /// let zscores = detector.zscore(&[0.0, 2.0]).unwrap();
    /// // Mean ≈ 0, std ≈ 1, so z ≈ |sample - mean| / std
    /// ```
    pub fn zscore(&self, samples: &[f32; C]) -> Option<[f64; C]> {
        if self.stats.count() < self.min_samples {
            return None;
        }

        let mean = self.frozen_mean.unwrap_or_else(|| *self.stats.mean());
        let std = self.frozen_std.unwrap_or_else(|| self.stats.std_dev());

        let mut result = [0.0f64; C];
        for (i, &sample) in samples.iter().enumerate() {
            if std[i] > 0.0 {
                result[i] = (sample as f64 - mean[i]) / std[i];
            }
        }

        Some(result)
    }

    /// Updates statistics and detects artifacts in one call.
    ///
    /// Convenience method that combines [`update`](Self::update) and
    /// [`detect`](Self::detect).
    ///
    /// # Example
    /// ```
    /// use zerostone::ZscoreArtifact;
    ///
    /// let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);
    ///
    /// // Process samples in a loop
    /// for i in 0..100 {
    ///     let val = if i % 2 == 0 { 0.5 } else { -0.5 };
    ///     let artifacts = detector.update_and_detect(&[val, val]);
    ///     // First 50 samples: all false (calibrating)
    ///     // After 50 samples: actual detection
    /// }
    /// ```
    pub fn update_and_detect(&mut self, samples: &[f32; C]) -> [bool; C] {
        self.update(samples);
        self.detect(samples)
    }

    /// Freezes the current statistics.
    ///
    /// After freezing, mean and standard deviation remain constant
    /// and statistics are no longer updated.
    pub fn freeze(&mut self) {
        self.frozen_mean = Some(*self.stats.mean());
        self.frozen_std = Some(self.stats.std_dev());
    }

    /// Unfreezes statistics, resuming adaptive behavior.
    pub fn unfreeze(&mut self) {
        self.frozen_mean = None;
        self.frozen_std = None;
    }

    /// Returns whether the detector is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen_mean.is_some()
    }

    /// Returns whether the detector is still in calibration period.
    pub fn is_calibrating(&self) -> bool {
        self.stats.count() < self.min_samples
    }

    /// Resets all state including statistics.
    pub fn reset(&mut self) {
        self.stats.reset();
        self.frozen_mean = None;
        self.frozen_std = None;
    }

    /// Returns the number of samples processed.
    pub fn sample_count(&self) -> u64 {
        self.stats.count()
    }

    /// Returns the z-score threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the z-score threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Returns the current mean for each channel.
    ///
    /// Returns the frozen mean if frozen, otherwise the current streaming mean.
    pub fn mean(&self) -> [f64; C] {
        self.frozen_mean.unwrap_or_else(|| *self.stats.mean())
    }

    /// Returns the current standard deviation for each channel.
    ///
    /// Returns the frozen std if frozen, otherwise the current streaming std.
    pub fn std_dev(&self) -> [f64; C] {
        self.frozen_std.unwrap_or_else(|| self.stats.std_dev())
    }

    /// Returns the number of channels.
    pub const fn num_channels(&self) -> usize {
        C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ArtifactDetector tests

    #[test]
    fn test_detector_amplitude_only() {
        let mut detector: ArtifactDetector<4> = ArtifactDetector::amplitude_only(100.0);

        // Below threshold - all clean
        let artifacts = detector.detect(&[50.0, 50.0, 50.0, 50.0]);
        assert!(artifacts.iter().all(|&a| !a));

        // Above threshold on some channels
        let artifacts = detector.detect(&[50.0, 150.0, 50.0, 200.0]);
        assert!(!artifacts[0]);
        assert!(artifacts[1]);
        assert!(!artifacts[2]);
        assert!(artifacts[3]);
    }

    #[test]
    fn test_detector_gradient_only() {
        let mut detector: ArtifactDetector<2> = ArtifactDetector::gradient_only(50.0);

        // First sample - no gradient check
        let artifacts = detector.detect(&[100.0, 100.0]);
        assert!(artifacts.iter().all(|&a| !a));

        // Large gradient on channel 0
        let artifacts = detector.detect(&[200.0, 120.0]);
        assert!(artifacts[0]); // |200 - 100| = 100 > 50
        assert!(!artifacts[1]); // |120 - 100| = 20 < 50
    }

    #[test]
    fn test_detector_combined_thresholds() {
        let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);

        // First sample
        detector.detect(&[50.0, 50.0]);

        // Second sample: amplitude ok, gradient bad on ch0
        let artifacts = detector.detect(&[120.0, 60.0]);
        assert!(artifacts[0]); // |120 - 50| = 70 > 50 (gradient)
        assert!(!artifacts[1]); // |60 - 50| = 10 < 50 (ok)
    }

    #[test]
    fn test_detector_detailed() {
        let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);

        // First sample to initialize with different values per channel
        detector.detect(&[50.0, 80.0, 50.0, 80.0]);

        // Test various artifact types
        let types = detector.detect_detailed(&[50.0, 110.0, 90.0, 200.0]);
        // ch0: |50| < 100 (ok), |50-50| = 0 < 50 (ok) -> Clean
        // ch1: |110| > 100 (bad), |110-80| = 30 < 50 (ok) -> Amplitude
        // ch2: |90| < 100 (ok), |90-50| = 40 < 50 (ok) -> Clean
        // ch3: |200| > 100 (bad), |200-80| = 120 > 50 (bad) -> Both

        assert_eq!(types[0], ArtifactType::Clean);
        assert_eq!(types[1], ArtifactType::Amplitude);
        assert_eq!(types[2], ArtifactType::Clean);
        assert_eq!(types[3], ArtifactType::Both);
    }

    #[test]
    fn test_detector_detailed_gradient_only_type() {
        let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);

        // Initialize
        detector.detect(&[50.0, 50.0]);

        // Test gradient-only artifact: amplitude ok but gradient bad
        let types = detector.detect_detailed(&[90.0, 50.0]);
        // ch0: |90| < 100 (ok), |90-50| = 40 < 50 (ok) -> Clean
        assert_eq!(types[0], ArtifactType::Clean);

        // Now create gradient artifact
        let types = detector.detect_detailed(&[50.0, 50.0]);
        // ch0: |50| < 100 (ok), |50-90| = 40 < 50 (ok) -> Clean
        assert_eq!(types[0], ArtifactType::Clean);

        // Jump to trigger gradient
        let _types = detector.detect_detailed(&[90.0, 50.0]);
        // ch0: |90| < 100 (ok), |90-50| = 40 < 50 (ok) -> still Clean

        // Need bigger jump for gradient artifact
        detector.detect(&[20.0, 20.0]);
        let types = detector.detect_detailed(&[80.0, 20.0]);
        // ch0: |80| < 100 (ok), |80-20| = 60 > 50 (bad) -> Gradient
        assert_eq!(types[0], ArtifactType::Gradient);
        assert_eq!(types[1], ArtifactType::Clean);
    }

    #[test]
    fn test_detector_negative_values() {
        let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);

        // Negative values should use absolute value
        let artifacts = detector.detect(&[-150.0, -50.0]);
        assert!(artifacts[0]); // |-150| = 150 > 100
        assert!(!artifacts[1]); // |-50| = 50 < 100
    }

    #[test]
    fn test_detector_any() {
        let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);

        // No artifacts
        assert!(!detector.detect_any(&[50.0, 50.0, 50.0, 50.0]));

        // One artifact
        assert!(detector.detect_any(&[50.0, 150.0, 50.0, 50.0]));
    }

    #[test]
    fn test_detector_count() {
        let mut detector: ArtifactDetector<4> = ArtifactDetector::new(100.0, 50.0);

        assert_eq!(detector.detect_count(&[50.0, 50.0, 50.0, 50.0]), 0);
        assert_eq!(detector.detect_count(&[150.0, 150.0, 50.0, 50.0]), 2);
        assert_eq!(detector.detect_count(&[150.0, 150.0, 150.0, 150.0]), 4);
    }

    #[test]
    fn test_detector_reset() {
        let mut detector: ArtifactDetector<2> = ArtifactDetector::new(100.0, 50.0);

        // Initialize with first sample
        detector.detect(&[50.0, 50.0]);

        // This would trigger gradient artifact
        let artifacts = detector.detect(&[120.0, 50.0]);
        assert!(artifacts[0]);

        // Reset
        detector.reset();

        // Now same sample should not trigger (no previous sample)
        let artifacts = detector.detect(&[120.0, 50.0]);
        assert!(artifacts[0]); // Still amplitude artifact

        // But this time we test gradient specifically
        let mut detector2: ArtifactDetector<2> = ArtifactDetector::gradient_only(50.0);
        detector2.detect(&[50.0, 50.0]);
        detector2.reset();
        // After reset, no gradient check on first sample
        let artifacts = detector2.detect(&[120.0, 50.0]);
        assert!(!artifacts[0]); // No previous sample, so no gradient
    }

    #[test]
    fn test_detector_setters() {
        let mut detector: ArtifactDetector<1> = ArtifactDetector::new(100.0, 50.0);

        assert_eq!(detector.amplitude_threshold(), 100.0);
        assert_eq!(detector.gradient_threshold(), 50.0);

        detector.set_amplitude_threshold(200.0);
        detector.set_gradient_threshold(75.0);

        assert_eq!(detector.amplitude_threshold(), 200.0);
        assert_eq!(detector.gradient_threshold(), 75.0);
    }

    #[test]
    fn test_detector_default() {
        let detector: ArtifactDetector<4> = ArtifactDetector::default();

        assert_eq!(detector.amplitude_threshold(), 100.0);
        assert_eq!(detector.gradient_threshold(), 50.0);
        assert_eq!(detector.num_channels(), 4);
    }

    #[test]
    fn test_artifact_type_methods() {
        assert!(!ArtifactType::Clean.is_artifact());
        assert!(ArtifactType::Clean.is_clean());

        assert!(ArtifactType::Amplitude.is_artifact());
        assert!(!ArtifactType::Amplitude.is_clean());

        assert!(ArtifactType::Gradient.is_artifact());
        assert!(!ArtifactType::Gradient.is_clean());

        assert!(ArtifactType::Both.is_artifact());
        assert!(!ArtifactType::Both.is_clean());
    }

    // ZscoreArtifact tests

    #[test]
    fn test_zscore_calibration_period() {
        let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 100);

        assert!(detector.is_calibrating());

        // Feed 99 samples - still calibrating
        for _ in 0..99 {
            detector.update(&[1.0, 1.0]);
        }
        assert!(detector.is_calibrating());

        // During calibration, detect returns all false
        let artifacts = detector.detect(&[1000.0, 1000.0]);
        assert!(artifacts.iter().all(|&a| !a));

        // 100th sample completes calibration
        detector.update(&[1.0, 1.0]);
        assert!(!detector.is_calibrating());
    }

    #[test]
    fn test_zscore_detection() {
        let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);

        // Calibrate with alternating values (mean ≈ 0, std ≈ 1)
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val, val]);
        }

        // Below threshold (|z| < 3)
        let artifacts = detector.detect(&[0.5, 0.5]);
        assert!(!artifacts[0]);
        assert!(!artifacts[1]);

        // Above threshold (|z| > 3, since std ≈ 1, need |x| > 3)
        let artifacts = detector.detect(&[5.0, 0.5]);
        assert!(artifacts[0]);
        assert!(!artifacts[1]);
    }

    #[test]
    fn test_zscore_computation() {
        let mut detector: ZscoreArtifact<1> = ZscoreArtifact::new(3.0, 50);

        // Calibrate with known values
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val]);
        }

        // z-score should be close to sample value (since mean ≈ 0, std ≈ 1)
        let z = detector.zscore(&[2.0]).unwrap();
        assert!((z[0] - 2.0).abs() < 0.1, "Expected z ≈ 2.0, got {}", z[0]);
    }

    #[test]
    fn test_zscore_update_and_detect() {
        let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);

        // First 50 calls: calibrating, all return false
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            let artifacts = detector.update_and_detect(&[val, val]);
            assert!(artifacts.iter().all(|&a| !a));
        }

        // After calibration, should detect outliers
        let artifacts = detector.update_and_detect(&[10.0, 0.5]);
        assert!(artifacts[0]);
        assert!(!artifacts[1]);
    }

    #[test]
    fn test_zscore_freeze_unfreeze() {
        let mut detector: ZscoreArtifact<1> = ZscoreArtifact::new(3.0, 50);

        // Calibrate
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val]);
        }

        let mean_before = detector.mean()[0];
        let std_before = detector.std_dev()[0];

        // Freeze
        detector.freeze();
        assert!(detector.is_frozen());

        // Feed different data
        for _ in 0..50 {
            detector.update(&[100.0]);
        }

        // Statistics should not change
        assert_eq!(detector.mean()[0], mean_before);
        assert_eq!(detector.std_dev()[0], std_before);

        // Unfreeze and feed data
        detector.unfreeze();
        assert!(!detector.is_frozen());

        for _ in 0..50 {
            detector.update(&[100.0]);
        }

        // Now statistics should have changed
        assert!(detector.mean()[0] > mean_before);
    }

    #[test]
    fn test_zscore_reset() {
        let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);

        // Calibrate and freeze
        for i in 0..50 {
            detector.update(&[if i % 2 == 0 { 1.0 } else { -1.0 }; 2]);
        }
        detector.freeze();

        assert!(!detector.is_calibrating());
        assert!(detector.is_frozen());
        assert!(detector.sample_count() > 0);

        // Reset
        detector.reset();

        assert!(detector.is_calibrating());
        assert!(!detector.is_frozen());
        assert_eq!(detector.sample_count(), 0);
    }

    #[test]
    fn test_zscore_per_channel() {
        let mut detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);

        // Feed different variances per channel
        for i in 0..50 {
            let ch0 = if i % 2 == 0 { 1.0 } else { -1.0 }; // std ≈ 1
            let ch1 = if i % 2 == 0 { 10.0 } else { -10.0 }; // std ≈ 10
            detector.update(&[ch0, ch1]);
        }

        // 5.0 is outlier for ch0 (|z| ≈ 5) but not for ch1 (|z| ≈ 0.5)
        let artifacts = detector.detect(&[5.0, 5.0]);
        assert!(artifacts[0]);
        assert!(!artifacts[1]);
    }

    #[test]
    fn test_zscore_setters() {
        let mut detector: ZscoreArtifact<1> = ZscoreArtifact::new(3.0, 50);

        assert_eq!(detector.threshold(), 3.0);

        detector.set_threshold(4.0);

        assert_eq!(detector.threshold(), 4.0);
    }

    #[test]
    fn test_zscore_zero_std_handling() {
        let mut detector: ZscoreArtifact<1> = ZscoreArtifact::new(3.0, 50);

        // All same values -> std = 0
        for _ in 0..50 {
            detector.update(&[5.0]);
        }

        // Should not flag as artifact when std = 0 (avoid division by zero)
        let artifacts = detector.detect(&[100.0]);
        assert!(!artifacts[0]);
    }

    #[test]
    fn test_zscore_none_during_calibration() {
        let detector: ZscoreArtifact<2> = ZscoreArtifact::new(3.0, 50);

        // zscore should return None during calibration
        assert!(detector.zscore(&[1.0, 1.0]).is_none());
    }
}
