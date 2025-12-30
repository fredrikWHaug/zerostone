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
}
