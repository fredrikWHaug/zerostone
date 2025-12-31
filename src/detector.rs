/// Detector state for threshold crossing detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorState {
    /// Ready to detect threshold crossings
    Armed,
    /// Threshold was crossed on this sample
    Triggered,
    /// In refractory period, cannot trigger
    Refractory,
}

/// Event emitted when a threshold is crossed.
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Channel index where spike was detected
    pub channel: usize,
    /// Sample value that triggered the detection
    pub amplitude: f32,
}

/// Multi-channel threshold detector with refractory period.
///
/// Detects when signal amplitude crosses a threshold and enforces a refractory
/// period where subsequent crossings are ignored. Essential for spike sorting
/// and event detection in BCI systems.
///
/// # Memory Layout
/// - `threshold`: Detection threshold value
/// - `refractory_samples`: Number of samples to ignore after detection
/// - `refractory_counter`: Per-channel countdown timers
/// - `state`: Per-channel detector states
///
/// # Performance
/// Target: <10 μs for 1024 channels
///
/// # Example
/// ```
/// # use zerostone::ThresholdDetector;
/// let mut detector: ThresholdDetector<8> = ThresholdDetector::new(3.0, 100);
///
/// let samples = [1.0, 2.0, 4.0, 1.0, 5.0, 1.0, 1.0, 1.0];
/// if let Some(event) = detector.process_sample(&samples) {
///     println!("Spike on channel {} with amplitude {}", event.channel, event.amplitude);
/// }
/// ```
pub struct ThresholdDetector<const C: usize> {
    threshold: f32,
    refractory_samples: u32,
    refractory_counter: [u32; C],
}

impl<const C: usize> ThresholdDetector<C> {
    /// Creates a new threshold detector.
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold for detection
    /// * `refractory_samples` - Number of samples to wait after detection
    ///
    /// # Example
    /// ```
    /// # use zerostone::ThresholdDetector;
    /// // Detect spikes above 3.0 with 100-sample refractory period
    /// let detector: ThresholdDetector<32> = ThresholdDetector::new(3.0, 100);
    /// ```
    pub fn new(threshold: f32, refractory_samples: u32) -> Self {
        Self {
            threshold,
            refractory_samples,
            refractory_counter: [0; C],
        }
    }

    /// Processes a multi-channel sample.
    ///
    /// Returns the first detected event, or None if no channels crossed threshold.
    ///
    /// # Performance
    /// O(C) - checks all channels sequentially
    pub fn process_sample(&mut self, samples: &[f32; C]) -> Option<SpikeEvent> {
        for (ch, &amplitude) in samples.iter().enumerate() {
            // Update refractory counter
            if self.refractory_counter[ch] > 0 {
                self.refractory_counter[ch] -= 1;
                continue;
            }

            // Check threshold crossing
            if libm::fabsf(amplitude) > self.threshold {
                // Trigger detection
                self.refractory_counter[ch] = self.refractory_samples;
                return Some(SpikeEvent {
                    channel: ch,
                    amplitude,
                });
            }
        }

        None
    }

    /// Processes a single channel sample, returning whether threshold was crossed.
    ///
    /// # Arguments
    /// * `channel` - Channel index
    /// * `sample` - Sample value
    ///
    /// # Returns
    /// `Some(amplitude)` if threshold crossed, `None` otherwise
    pub fn process_channel(&mut self, channel: usize, sample: f32) -> Option<f32> {
        if channel >= C {
            return None;
        }

        // Update refractory counter
        if self.refractory_counter[channel] > 0 {
            self.refractory_counter[channel] -= 1;
            return None;
        }

        // Check threshold crossing
        if libm::fabsf(sample) > self.threshold {
            self.refractory_counter[channel] = self.refractory_samples;
            return Some(sample);
        }

        None
    }

    /// Returns the current detector state for a channel.
    pub fn state(&self, channel: usize) -> DetectorState {
        if channel >= C {
            return DetectorState::Armed;
        }

        if self.refractory_counter[channel] > 0 {
            DetectorState::Refractory
        } else {
            DetectorState::Armed
        }
    }

    /// Resets all channels to armed state.
    pub fn reset(&mut self) {
        self.refractory_counter = [0; C];
    }

    /// Updates the detection threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Returns the current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Updates the refractory period.
    pub fn set_refractory_period(&mut self, samples: u32) {
        self.refractory_samples = samples;
    }

    /// Returns the refractory period in samples.
    pub fn refractory_period(&self) -> u32 {
        self.refractory_samples
    }

    /// Returns number of channels.
    pub const fn num_channels(&self) -> usize {
        C
    }
}

/// Adaptive multi-channel threshold detector using N×σ detection.
///
/// Automatically computes detection thresholds based on signal statistics,
/// using a multiplier times the standard deviation (e.g., 4×σ). More robust
/// than fixed thresholds for varying signal conditions.
///
/// # Operation Modes
/// - **Calibrating**: Collecting samples, not yet detecting (count < min_samples)
/// - **Adapting**: Continuously updating threshold from signal statistics
/// - **Frozen**: Using fixed threshold captured at freeze time
///
/// # Example
/// ```
/// # use zerostone::AdaptiveThresholdDetector;
/// // 4×σ threshold, 100-sample refractory, 500-sample warm-up
/// let mut detector: AdaptiveThresholdDetector<8> = AdaptiveThresholdDetector::new(4.0, 100, 500);
///
/// // Feed calibration data
/// for _ in 0..500 {
///     let samples = [0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1];
///     detector.process_sample(&samples);
/// }
///
/// // Now detecting with adaptive threshold
/// let spike_samples = [0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1];
/// if let Some(event) = detector.process_sample(&spike_samples) {
///     println!("Spike on channel {}", event.channel);
/// }
/// ```
pub struct AdaptiveThresholdDetector<const C: usize> {
    stats: crate::OnlineStats<C>,
    multiplier: f32,
    min_samples: u64,
    refractory_samples: u32,
    refractory_counter: [u32; C],
    frozen_threshold: Option<[f32; C]>,
}

impl<const C: usize> AdaptiveThresholdDetector<C> {
    /// Creates a new adaptive threshold detector.
    ///
    /// # Arguments
    /// * `multiplier` - Threshold multiplier (e.g., 4.0 for 4×σ detection)
    /// * `refractory_samples` - Number of samples to wait after detection
    /// * `min_samples` - Minimum samples before detection starts (warm-up period)
    ///
    /// # Example
    /// ```
    /// # use zerostone::AdaptiveThresholdDetector;
    /// // Typical settings: 4×σ, 1ms refractory at 30kHz, 500-sample warm-up
    /// let detector: AdaptiveThresholdDetector<32> = AdaptiveThresholdDetector::new(4.0, 30, 500);
    /// ```
    pub fn new(multiplier: f32, refractory_samples: u32, min_samples: u64) -> Self {
        Self {
            stats: crate::OnlineStats::new(),
            multiplier,
            min_samples,
            refractory_samples,
            refractory_counter: [0; C],
            frozen_threshold: None,
        }
    }

    /// Processes a multi-channel sample.
    ///
    /// Updates statistics and detects threshold crossings. During warm-up
    /// (count < min_samples), only statistics are updated—no detection occurs.
    ///
    /// Returns the first detected event, or None if no detection.
    pub fn process_sample(&mut self, samples: &[f32; C]) -> Option<SpikeEvent> {
        // Convert to f64 for stats update
        let mut samples_f64 = [0.0f64; C];
        for (i, &s) in samples.iter().enumerate() {
            samples_f64[i] = s as f64;
        }

        // Always update statistics (unless frozen)
        if self.frozen_threshold.is_none() {
            self.stats.update(&samples_f64);
        }

        // Don't detect during warm-up period
        if self.stats.count() < self.min_samples {
            return None;
        }

        // Get current thresholds
        let thresholds = self.thresholds();

        // Check each channel
        for (ch, &amplitude) in samples.iter().enumerate() {
            // Update refractory counter
            if self.refractory_counter[ch] > 0 {
                self.refractory_counter[ch] -= 1;
                continue;
            }

            // Check threshold crossing
            if libm::fabsf(amplitude) > thresholds[ch] {
                self.refractory_counter[ch] = self.refractory_samples;
                return Some(SpikeEvent {
                    channel: ch,
                    amplitude,
                });
            }
        }

        None
    }

    /// Returns current threshold for each channel.
    ///
    /// If frozen, returns the frozen thresholds. Otherwise computes
    /// multiplier × std_dev for each channel.
    pub fn thresholds(&self) -> [f32; C] {
        if let Some(frozen) = self.frozen_threshold {
            return frozen;
        }

        let std_dev = self.stats.std_dev();
        let mut thresholds = [0.0f32; C];
        for (i, &s) in std_dev.iter().enumerate() {
            thresholds[i] = self.multiplier * (s as f32);
        }
        thresholds
    }

    /// Returns the threshold for a specific channel.
    pub fn threshold(&self, channel: usize) -> f32 {
        if channel >= C {
            return 0.0;
        }
        self.thresholds()[channel]
    }

    /// Freezes the current thresholds.
    ///
    /// After freezing, thresholds remain constant and statistics are no
    /// longer updated. Useful after a calibration period.
    pub fn freeze(&mut self) {
        self.frozen_threshold = Some(self.thresholds());
    }

    /// Unfreezes thresholds, resuming adaptive behavior.
    pub fn unfreeze(&mut self) {
        self.frozen_threshold = None;
    }

    /// Returns whether the detector is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen_threshold.is_some()
    }

    /// Returns whether the detector is still in warm-up period.
    pub fn is_calibrating(&self) -> bool {
        self.stats.count() < self.min_samples
    }

    /// Returns the current detector state for a channel.
    pub fn state(&self, channel: usize) -> DetectorState {
        if channel >= C {
            return DetectorState::Armed;
        }

        if self.refractory_counter[channel] > 0 {
            DetectorState::Refractory
        } else {
            DetectorState::Armed
        }
    }

    /// Resets all state including statistics.
    pub fn reset(&mut self) {
        self.stats.reset();
        self.refractory_counter = [0; C];
        self.frozen_threshold = None;
    }

    /// Returns the number of samples processed.
    pub fn sample_count(&self) -> u64 {
        self.stats.count()
    }

    /// Returns the multiplier.
    pub fn multiplier(&self) -> f32 {
        self.multiplier
    }

    /// Sets the multiplier.
    pub fn set_multiplier(&mut self, multiplier: f32) {
        self.multiplier = multiplier;
    }

    /// Returns the refractory period in samples.
    pub fn refractory_period(&self) -> u32 {
        self.refractory_samples
    }

    /// Sets the refractory period.
    pub fn set_refractory_period(&mut self, samples: u32) {
        self.refractory_samples = samples;
    }

    /// Returns number of channels.
    pub const fn num_channels(&self) -> usize {
        C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_basic_detection() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        // Below threshold - no detection
        let samples = [1.0, 1.0, 1.0, 1.0];
        assert!(detector.process_sample(&samples).is_none());

        // Above threshold - should detect on channel 2
        let samples = [1.0, 1.0, 3.0, 1.0];
        let event = detector.process_sample(&samples).unwrap();
        assert_eq!(event.channel, 2);
        assert_eq!(event.amplitude, 3.0);
    }

    #[test]
    fn test_threshold_refractory_period() {
        let mut detector: ThresholdDetector<1> = ThresholdDetector::new(2.0, 5);

        // First detection
        assert!(detector.process_sample(&[3.0]).is_some());

        // During refractory period - should not detect
        for _ in 0..5 {
            assert!(detector.process_sample(&[3.0]).is_none());
        }

        // After refractory period - should detect again
        assert!(detector.process_sample(&[3.0]).is_some());
    }

    #[test]
    fn test_threshold_negative_values() {
        let mut detector: ThresholdDetector<2> = ThresholdDetector::new(2.0, 10);

        // Negative value above threshold (absolute value)
        let samples = [1.0, -3.0];
        let event = detector.process_sample(&samples).unwrap();
        assert_eq!(event.channel, 1);
        assert_eq!(event.amplitude, -3.0);
    }

    #[test]
    fn test_threshold_process_channel() {
        let mut detector: ThresholdDetector<8> = ThresholdDetector::new(2.0, 10);

        // Test single channel processing
        assert!(detector.process_channel(0, 1.0).is_none());
        assert!(detector.process_channel(0, 3.0).is_some());

        // In refractory period
        assert!(detector.process_channel(0, 5.0).is_none());

        // Different channel not affected
        assert!(detector.process_channel(1, 3.0).is_some());
    }

    #[test]
    fn test_threshold_state() {
        let mut detector: ThresholdDetector<2> = ThresholdDetector::new(2.0, 5);

        // Initially armed
        assert_eq!(detector.state(0), DetectorState::Armed);

        // Trigger detection
        detector.process_channel(0, 3.0);

        // Now in refractory
        assert_eq!(detector.state(0), DetectorState::Refractory);

        // Other channel still armed
        assert_eq!(detector.state(1), DetectorState::Armed);
    }

    #[test]
    fn test_threshold_reset() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 100);

        // Trigger all channels
        for ch in 0..4 {
            detector.process_channel(ch, 3.0);
        }

        // All should be in refractory
        for ch in 0..4 {
            assert_eq!(detector.state(ch), DetectorState::Refractory);
        }

        // Reset
        detector.reset();

        // All should be armed again
        for ch in 0..4 {
            assert_eq!(detector.state(ch), DetectorState::Armed);
        }
    }

    #[test]
    fn test_threshold_setters() {
        let mut detector: ThresholdDetector<1> = ThresholdDetector::new(2.0, 10);

        assert_eq!(detector.threshold(), 2.0);
        assert_eq!(detector.refractory_period(), 10);

        detector.set_threshold(5.0);
        detector.set_refractory_period(20);

        assert_eq!(detector.threshold(), 5.0);
        assert_eq!(detector.refractory_period(), 20);

        // Should respect new threshold
        assert!(detector.process_sample(&[3.0]).is_none());
        assert!(detector.process_sample(&[6.0]).is_some());
    }

    #[test]
    fn test_threshold_multi_channel_priority() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        // Multiple channels above threshold - should return first one
        let samples = [1.0, 3.0, 4.0, 5.0];
        let event = detector.process_sample(&samples).unwrap();
        assert_eq!(event.channel, 1);
    }

    #[test]
    fn test_threshold_large_channel_count() {
        let mut detector: ThresholdDetector<1024> = ThresholdDetector::new(2.0, 10);

        assert_eq!(detector.num_channels(), 1024);

        // Test a specific channel works
        assert!(detector.process_channel(512, 3.0).is_some());
        assert_eq!(detector.state(512), DetectorState::Refractory);
    }

    // Adaptive threshold detector tests

    #[test]
    fn test_adaptive_calibration_period() {
        let mut detector: AdaptiveThresholdDetector<2> =
            AdaptiveThresholdDetector::new(4.0, 10, 100);

        assert!(detector.is_calibrating());

        // Feed 99 samples - still calibrating
        for _ in 0..99 {
            let samples = [0.5, -0.5];
            assert!(detector.process_sample(&samples).is_none());
        }
        assert!(detector.is_calibrating());

        // 100th sample completes calibration
        let samples = [0.5, -0.5];
        detector.process_sample(&samples);
        assert!(!detector.is_calibrating());
    }

    #[test]
    fn test_adaptive_threshold_computation() {
        let mut detector: AdaptiveThresholdDetector<1> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        // Feed samples with known statistics
        // Standard deviation of alternating 1.0 and -1.0 is 1.0
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.process_sample(&[val]);
        }

        // Threshold should be approximately 4.0 (4 × 1.0)
        let threshold = detector.threshold(0);
        assert!(
            (threshold - 4.0).abs() < 0.1,
            "Expected threshold ~4.0, got {}",
            threshold
        );
    }

    #[test]
    fn test_adaptive_detection() {
        let mut detector: AdaptiveThresholdDetector<2> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        // Calibrate with low-variance noise
        for i in 0..50 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val, val]);
        }

        // Threshold should be ~0.4 (4 × 0.1)
        // Signal within threshold - no detection
        assert!(detector.process_sample(&[0.3, 0.3]).is_none());

        // Signal above threshold - should detect
        let event = detector.process_sample(&[1.0, 0.0]).unwrap();
        assert_eq!(event.channel, 0);
    }

    #[test]
    fn test_adaptive_freeze_unfreeze() {
        let mut detector: AdaptiveThresholdDetector<1> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        // Calibrate
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.process_sample(&[val]);
        }

        let threshold_before = detector.threshold(0);
        assert!(!detector.is_frozen());

        // Freeze
        detector.freeze();
        assert!(detector.is_frozen());

        // Feed different data - threshold should NOT change
        for _ in 0..50 {
            detector.process_sample(&[10.0]); // Much larger values
        }

        let threshold_after = detector.threshold(0);
        assert_eq!(threshold_before, threshold_after);

        // Unfreeze and feed more data - threshold should change
        detector.unfreeze();
        assert!(!detector.is_frozen());

        for _ in 0..50 {
            detector.process_sample(&[10.0]);
        }

        let threshold_unfrozen = detector.threshold(0);
        assert!(threshold_unfrozen > threshold_before);
    }

    #[test]
    fn test_adaptive_refractory_period() {
        let mut detector: AdaptiveThresholdDetector<1> = AdaptiveThresholdDetector::new(4.0, 5, 50);

        // Calibrate
        for i in 0..50 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val]);
        }

        // Freeze to prevent threshold from changing during test
        detector.freeze();

        // First detection (threshold ~0.4, spike at 1.0)
        assert!(detector.process_sample(&[1.0]).is_some());
        assert_eq!(detector.state(0), DetectorState::Refractory);

        // Should not detect during refractory
        for _ in 0..5 {
            assert!(detector.process_sample(&[1.0]).is_none());
        }

        // Should detect after refractory
        assert!(detector.process_sample(&[1.0]).is_some());
    }

    #[test]
    fn test_adaptive_reset() {
        let mut detector: AdaptiveThresholdDetector<2> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        // Calibrate and trigger
        for i in 0..50 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.process_sample(&[val, val]);
        }
        detector.freeze();
        detector.process_sample(&[10.0, 10.0]); // Trigger both channels

        // Reset
        detector.reset();

        // Should be back to initial state
        assert!(detector.is_calibrating());
        assert!(!detector.is_frozen());
        assert_eq!(detector.sample_count(), 0);
        assert_eq!(detector.state(0), DetectorState::Armed);
        assert_eq!(detector.state(1), DetectorState::Armed);
    }

    #[test]
    fn test_adaptive_per_channel_thresholds() {
        let mut detector: AdaptiveThresholdDetector<2> =
            AdaptiveThresholdDetector::new(4.0, 10, 100);

        // Feed different variances to each channel
        for i in 0..100 {
            let ch0 = if i % 2 == 0 { 1.0 } else { -1.0 }; // std = 1.0
            let ch1 = if i % 2 == 0 { 2.0 } else { -2.0 }; // std = 2.0
            detector.process_sample(&[ch0, ch1]);
        }

        let thresholds = detector.thresholds();
        // Channel 1 should have roughly 2x the threshold of channel 0
        assert!(
            (thresholds[1] / thresholds[0] - 2.0).abs() < 0.1,
            "Expected ratio ~2.0, got {}",
            thresholds[1] / thresholds[0]
        );
    }

    #[test]
    fn test_adaptive_setters() {
        let mut detector: AdaptiveThresholdDetector<1> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        assert_eq!(detector.multiplier(), 4.0);
        assert_eq!(detector.refractory_period(), 10);

        detector.set_multiplier(5.0);
        detector.set_refractory_period(20);

        assert_eq!(detector.multiplier(), 5.0);
        assert_eq!(detector.refractory_period(), 20);
    }
}
