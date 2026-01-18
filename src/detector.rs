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

/// Direction of zero-crossing to detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossingDirection {
    /// Detect both rising and falling crossings
    #[default]
    Any,
    /// Detect only rising crossings (negative to positive)
    Rising,
    /// Detect only falling crossings (positive to negative)
    Falling,
}

/// Event emitted when a threshold is crossed.
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Channel index where spike was detected
    pub channel: usize,
    /// Sample value that triggered the detection
    pub amplitude: f32,
}

/// Collection of spike events from a single multi-channel sample.
///
/// Holds up to C events (one per channel maximum). Use [`iter()`](Self::iter)
/// or [`get()`](Self::get) to access detected events.
///
/// # Example
/// ```
/// # use zerostone::{ThresholdDetector, SpikeEvents};
/// let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);
/// let samples = [1.0, 3.0, 4.0, 1.0]; // Channels 1 and 2 above threshold
///
/// let events: SpikeEvents<4> = detector.process_sample_all(&samples);
/// assert_eq!(events.len(), 2);
///
/// for event in events.iter() {
///     println!("Spike on channel {} with amplitude {}", event.channel, event.amplitude);
/// }
/// ```
pub struct SpikeEvents<const C: usize> {
    events: [SpikeEvent; C],
    len: usize,
}

impl<const C: usize> SpikeEvents<C> {
    /// Creates an empty SpikeEvents collection.
    fn new() -> Self {
        Self {
            events: [SpikeEvent {
                channel: 0,
                amplitude: 0.0,
            }; C],
            len: 0,
        }
    }

    /// Adds an event to the collection.
    #[inline]
    fn push(&mut self, event: SpikeEvent) {
        if self.len < C {
            self.events[self.len] = event;
            self.len += 1;
        }
    }

    /// Returns the number of detected events.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if no events were detected.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over the detected events.
    pub fn iter(&self) -> impl Iterator<Item = &SpikeEvent> {
        self.events[..self.len].iter()
    }

    /// Returns the event at the given index, if it exists.
    pub fn get(&self, index: usize) -> Option<&SpikeEvent> {
        if index < self.len {
            Some(&self.events[index])
        } else {
            None
        }
    }

    /// Returns the underlying slice of detected events.
    pub fn as_slice(&self) -> &[SpikeEvent] {
        &self.events[..self.len]
    }
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

    /// Processes a multi-channel sample, returning all detected events.
    ///
    /// Unlike [`process_sample`](Self::process_sample) which returns only the first
    /// detection, this method returns all channels that crossed the threshold.
    ///
    /// # Example
    /// ```
    /// # use zerostone::ThresholdDetector;
    /// let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);
    /// let samples = [1.0, 3.0, 4.0, 1.0]; // Channels 1 and 2 above threshold
    ///
    /// let events = detector.process_sample_all(&samples);
    /// assert_eq!(events.len(), 2);
    /// ```
    pub fn process_sample_all(&mut self, samples: &[f32; C]) -> SpikeEvents<C> {
        let mut events = SpikeEvents::new();

        for (ch, &amplitude) in samples.iter().enumerate() {
            // Update refractory counter
            if self.refractory_counter[ch] > 0 {
                self.refractory_counter[ch] -= 1;
                continue;
            }

            // Check threshold crossing
            if libm::fabsf(amplitude) > self.threshold {
                self.refractory_counter[ch] = self.refractory_samples;
                events.push(SpikeEvent {
                    channel: ch,
                    amplitude,
                });
            }
        }

        events
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

    /// Processes a multi-channel sample, returning all detected events.
    ///
    /// Unlike [`process_sample`](Self::process_sample) which returns only the first
    /// detection, this method returns all channels that crossed the threshold.
    ///
    /// # Example
    /// ```
    /// # use zerostone::AdaptiveThresholdDetector;
    /// let mut detector: AdaptiveThresholdDetector<4> = AdaptiveThresholdDetector::new(4.0, 10, 50);
    ///
    /// // Calibrate
    /// for i in 0..50 {
    ///     let val = if i % 2 == 0 { 0.1 } else { -0.1 };
    ///     detector.process_sample(&[val; 4]);
    /// }
    ///
    /// // Detect multiple spikes
    /// let samples = [0.0, 1.0, 1.0, 0.0]; // Channels 1 and 2 above threshold
    /// let events = detector.process_sample_all(&samples);
    /// ```
    pub fn process_sample_all(&mut self, samples: &[f32; C]) -> SpikeEvents<C> {
        let mut events = SpikeEvents::new();

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
            return events;
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
                events.push(SpikeEvent {
                    channel: ch,
                    amplitude,
                });
            }
        }

        events
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

/// Multi-channel zero-crossing detector for BCI signal analysis.
///
/// Detects when a signal changes sign (crosses zero). Essential for computing
/// zero-crossing rate (ZCR) features and detecting specific patterns in EEG,
/// such as epileptic spike-wave complexes.
///
/// # Algorithm
///
/// For each sample, computes the sign considering a noise threshold:
/// - Values in `[-threshold, +threshold]` are treated as zero (sign = 0)
/// - Values > threshold have sign = +1
/// - Values < -threshold have sign = -1
///
/// A crossing occurs when the sign changes from non-zero to a different non-zero value.
/// Zero samples are ignored to prevent false crossings near the threshold band.
///
/// # Memory Layout
/// - `threshold`: Noise rejection threshold
/// - `direction`: Which crossings to detect (Any, Rising, Falling)
/// - `prev_sample`: Previous sample value per channel
/// - `initialized`: Whether first sample has been received
///
/// # Performance
/// Target: <10 μs for 1024 channels
///
/// # BCI Applications
///
/// ## Zero-Crossing Rate (ZCR)
/// Compute the proportion of samples where sign changes occur. High ZCR indicates
/// complex, rapidly-varying signals (e.g., seizure activity), while low ZCR
/// indicates smooth signals (e.g., alpha waves during rest).
///
/// ## Epilepsy Detection
/// Spike-wave patterns in epileptic seizures exhibit characteristic high crossing rates.
/// Use `CrossingDirection::Rising` to detect sharp upward transitions.
///
/// ## Pitch Detection
/// In speech signals, voiced segments have low ZCR while unvoiced segments
/// (fricatives, stops) have high ZCR.
///
/// ## Event Detection
/// Use directional detection to identify specific transitions:
/// - `Rising`: Motor imagery onset, P300 rising edge
/// - `Falling`: Motor imagery offset, P300 falling edge
///
/// # Example
/// ```
/// # use zerostone::{ZeroCrossingDetector, CrossingDirection};
/// // Detect any zero-crossings with 0.1 threshold for noise rejection
/// let mut detector: ZeroCrossingDetector<8> = ZeroCrossingDetector::new(0.1);
///
/// let samples = [-0.5, 0.3, 0.2, -0.4, 0.6, -0.1, 0.0, 0.5];
/// let crossings = detector.detect(&samples);
///
/// // Count crossings
/// let count: usize = crossings.iter().filter(|&&x| x).count();
/// println!("Detected {} crossings", count);
/// ```
///
/// # Example: Computing Zero-Crossing Rate
/// ```
/// # use zerostone::ZeroCrossingDetector;
/// let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);
///
/// // Create a block of samples (e.g., 1-second window at 256 Hz)
/// let mut block = Vec::new();
/// for i in 0..256 {
///     let t = i as f32 * 0.01;
///     // Channel 0: 10 Hz sine wave
///     let ch0 = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * t);
///     // Channel 1: 50 Hz sine wave (higher frequency)
///     let ch1 = libm::sinf(2.0 * core::f32::consts::PI * 50.0 * t);
///     block.push([ch0, ch1]);
/// }
///
/// // Compute ZCR for the block
/// let zcr = detector.zcr(&block);
/// // zcr[0] ≈ 0.078 (10 Hz has ~20 crossings in 256 samples)
/// // zcr[1] ≈ 0.391 (50 Hz has ~100 crossings in 256 samples)
/// ```
///
/// # Example: Epilepsy Detection
/// ```
/// # use zerostone::{ZeroCrossingDetector, CrossingDirection};
/// // Detect rising edges for seizure onset
/// let mut detector: ZeroCrossingDetector<1> =
///     ZeroCrossingDetector::with_direction(0.05, CrossingDirection::Rising);
///
/// // Simulate spike-wave pattern (characteristic of absence seizures)
/// let spike_wave = [
///     -0.1, -0.2, 0.8,  // Sharp spike (rising)
///     0.5, 0.2, -0.3,   // Wave descent
///     -0.4, -0.2, 0.7,  // Another spike
///     0.4, 0.1, -0.2,   // Wave descent
/// ];
///
/// let mut rising_count = 0;
/// for &sample in &spike_wave {
///     if detector.detect(&[sample])[0] {
///         rising_count += 1;
///     }
/// }
///
/// // High rising crossing count indicates spike-wave pattern
/// if rising_count > 2 {
///     println!("Potential seizure activity detected");
/// }
/// ```
pub struct ZeroCrossingDetector<const C: usize> {
    threshold: f32,
    direction: CrossingDirection,
    prev_sample: [f32; C],
    initialized: bool,
}

impl<const C: usize> ZeroCrossingDetector<C> {
    /// Creates a new zero-crossing detector.
    ///
    /// # Arguments
    /// * `threshold` - Noise rejection threshold. Values within `[-threshold, +threshold]`
    ///   are treated as zero to prevent false crossings from noise.
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// // Detect crossings with 0.1 threshold for noise rejection
    /// let detector: ZeroCrossingDetector<32> = ZeroCrossingDetector::new(0.1);
    /// ```
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            direction: CrossingDirection::default(),
            prev_sample: [0.0; C],
            initialized: false,
        }
    }

    /// Creates a new zero-crossing detector with specific crossing direction.
    ///
    /// # Arguments
    /// * `threshold` - Noise rejection threshold
    /// * `direction` - Which crossings to detect (Any, Rising, or Falling)
    ///
    /// # Example
    /// ```
    /// # use zerostone::{ZeroCrossingDetector, CrossingDirection};
    /// // Detect only rising crossings (for event onset detection)
    /// let detector: ZeroCrossingDetector<8> =
    ///     ZeroCrossingDetector::with_direction(0.1, CrossingDirection::Rising);
    /// ```
    pub fn with_direction(threshold: f32, direction: CrossingDirection) -> Self {
        Self {
            threshold,
            direction,
            prev_sample: [0.0; C],
            initialized: false,
        }
    }

    /// Detects zero-crossings in a multi-channel sample.
    ///
    /// Returns a boolean array indicating which channels had a zero-crossing.
    /// The first sample after creation or reset always returns all `false`.
    ///
    /// # Arguments
    /// * `samples` - Multi-channel sample values
    ///
    /// # Returns
    /// Boolean array where `true` indicates a crossing was detected on that channel
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);
    ///
    /// // Initialize with negative values
    /// detector.detect(&[-1.0, -0.5, -0.3, -0.2]);
    ///
    /// // Detect crossings (channels 0, 1, 2 cross to positive)
    /// let crossings = detector.detect(&[1.0, 0.8, 0.5, -0.1]);
    /// assert!(crossings[0]); // -1.0 → 1.0 crossed
    /// assert!(crossings[1]); // -0.5 → 0.8 crossed
    /// assert!(crossings[2]); // -0.3 → 0.5 crossed
    /// assert!(!crossings[3]); // -0.2 → -0.1 no crossing
    /// ```
    pub fn detect(&mut self, samples: &[f32; C]) -> [bool; C] {
        let mut result = [false; C];

        // First sample has no previous - return all false
        if !self.initialized {
            self.prev_sample = *samples;
            self.initialized = true;
            return result;
        }

        for ch in 0..C {
            let prev_sign = self.get_sign(self.prev_sample[ch]);
            let curr_sign = self.get_sign(samples[ch]);

            // Check for crossing based on direction
            let crossed = match self.direction {
                CrossingDirection::Any => {
                    prev_sign != 0 && curr_sign != 0 && prev_sign != curr_sign
                }
                CrossingDirection::Rising => prev_sign == -1 && curr_sign == 1,
                CrossingDirection::Falling => prev_sign == 1 && curr_sign == -1,
            };

            result[ch] = crossed;
        }

        // Update previous samples
        self.prev_sample = *samples;

        result
    }

    /// Returns true if any channel had a zero-crossing.
    ///
    /// # Arguments
    /// * `samples` - Multi-channel sample values
    ///
    /// # Returns
    /// `true` if at least one channel crossed zero, `false` otherwise
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);
    ///
    /// detector.detect(&[-1.0, -1.0, -1.0, -1.0]);
    ///
    /// // One channel crosses
    /// assert!(detector.detect_any(&[1.0, -1.0, -1.0, -1.0]));
    ///
    /// // No channels cross
    /// assert!(!detector.detect_any(&[0.5, -0.5, -0.8, -0.3]));
    /// ```
    pub fn detect_any(&mut self, samples: &[f32; C]) -> bool {
        let crossings = self.detect(samples);
        crossings.iter().any(|&x| x)
    }

    /// Returns the number of channels that had a zero-crossing.
    ///
    /// # Arguments
    /// * `samples` - Multi-channel sample values
    ///
    /// # Returns
    /// Count of channels where a crossing was detected
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);
    ///
    /// detector.detect(&[-1.0, -1.0, 1.0, 1.0]);
    ///
    /// // All four channels cross (ch0,ch1: -1→1, ch2,ch3: 1→-1)
    /// let count = detector.detect_count(&[1.0, 1.0, -1.0, -1.0]);
    /// assert_eq!(count, 4);
    /// ```
    pub fn detect_count(&mut self, samples: &[f32; C]) -> usize {
        let crossings = self.detect(samples);
        crossings.iter().filter(|&&x| x).count()
    }

    /// Computes zero-crossing rate (ZCR) for a block of samples.
    ///
    /// Returns the proportion of samples where crossings occurred (0.0-1.0) for each channel.
    /// This is the primary method for computing ZCR features for BCI applications.
    ///
    /// # Arguments
    /// * `block` - Slice of multi-channel samples (e.g., a 1-second window)
    ///
    /// # Returns
    /// Array of ZCR values per channel (0.0 = no crossings, 1.0 = all samples crossed)
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);
    ///
    /// // Create test signals
    /// let mut block = Vec::new();
    /// for i in 0..100 {
    ///     let t = i as f32 * 0.1;
    ///     // Channel 0: slow oscillation (low ZCR)
    ///     let ch0 = if i < 50 { -1.0 } else { 1.0 };
    ///     // Channel 1: fast alternation (high ZCR)
    ///     let ch1 = if i % 2 == 0 { -1.0 } else { 1.0 };
    ///     block.push([ch0, ch1]);
    /// }
    ///
    /// let zcr = detector.zcr(&block);
    /// assert!(zcr[0] < 0.1);  // Low ZCR for slow signal
    /// assert!(zcr[1] > 0.4);  // High ZCR for fast signal
    /// ```
    pub fn zcr(&mut self, block: &[[f32; C]]) -> [f32; C] {
        if block.is_empty() {
            return [0.0; C];
        }

        let mut crossing_counts = [0usize; C];

        for sample in block {
            let crossings = self.detect(sample);
            for (ch, &crossed) in crossings.iter().enumerate() {
                if crossed {
                    crossing_counts[ch] += 1;
                }
            }
        }

        // Compute proportion
        let mut zcr = [0.0f32; C];
        let total_samples = block.len() as f32;
        for ch in 0..C {
            zcr[ch] = crossing_counts[ch] as f32 / total_samples;
        }

        zcr
    }

    /// Returns the current noise rejection threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the noise rejection threshold.
    ///
    /// # Arguments
    /// * `threshold` - New threshold value
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<1> = ZeroCrossingDetector::new(0.1);
    /// assert_eq!(detector.threshold(), 0.1);
    ///
    /// detector.set_threshold(0.2);
    /// assert_eq!(detector.threshold(), 0.2);
    /// ```
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Returns the current crossing direction mode.
    pub fn direction(&self) -> CrossingDirection {
        self.direction
    }

    /// Sets the crossing direction mode.
    ///
    /// # Arguments
    /// * `direction` - New crossing direction
    ///
    /// # Example
    /// ```
    /// # use zerostone::{ZeroCrossingDetector, CrossingDirection};
    /// let mut detector: ZeroCrossingDetector<1> = ZeroCrossingDetector::new(0.0);
    /// assert_eq!(detector.direction(), CrossingDirection::Any);
    ///
    /// detector.set_direction(CrossingDirection::Rising);
    /// assert_eq!(detector.direction(), CrossingDirection::Rising);
    /// ```
    pub fn set_direction(&mut self, direction: CrossingDirection) {
        self.direction = direction;
    }

    /// Resets the detector state.
    ///
    /// Clears previous sample history. The next call to `detect()` will return all `false`.
    ///
    /// # Example
    /// ```
    /// # use zerostone::ZeroCrossingDetector;
    /// let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);
    ///
    /// detector.detect(&[-1.0, -1.0]);
    /// detector.detect(&[1.0, 1.0]); // Would detect crossings
    ///
    /// detector.reset();
    ///
    /// // After reset, no crossing on first sample
    /// let crossings = detector.detect(&[1.0, 1.0]);
    /// assert!(!crossings[0]);
    /// assert!(!crossings[1]);
    /// ```
    pub fn reset(&mut self) {
        self.prev_sample = [0.0; C];
        self.initialized = false;
    }

    /// Returns the number of channels.
    pub const fn num_channels(&self) -> usize {
        C
    }

    /// Helper function to get the sign of a value with threshold.
    ///
    /// Returns:
    /// - 0 if |value| <= threshold (treated as zero)
    /// - 1 if value > threshold
    /// - -1 if value < -threshold
    #[inline]
    fn get_sign(&self, value: f32) -> i8 {
        if libm::fabsf(value) <= self.threshold {
            0
        } else if value > 0.0 {
            1
        } else {
            -1
        }
    }
}

impl<const C: usize> Default for ZeroCrossingDetector<C> {
    fn default() -> Self {
        Self::new(0.0)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

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

    // SpikeEvents and process_sample_all tests

    #[test]
    fn test_spike_events_empty() {
        let events: SpikeEvents<4> = SpikeEvents::new();
        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
        assert!(events.get(0).is_none());
        assert_eq!(events.as_slice().len(), 0);
    }

    #[test]
    fn test_process_sample_all_no_detections() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        // All samples below threshold
        let samples = [1.0, 1.0, 1.0, 1.0];
        let events = detector.process_sample_all(&samples);

        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_process_sample_all_single_detection() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        // One sample above threshold
        let samples = [1.0, 3.0, 1.0, 1.0];
        let events = detector.process_sample_all(&samples);

        assert_eq!(events.len(), 1);
        assert_eq!(events.get(0).unwrap().channel, 1);
        assert_eq!(events.get(0).unwrap().amplitude, 3.0);
    }

    #[test]
    fn test_process_sample_all_multiple_detections() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        // Multiple samples above threshold
        let samples = [1.0, 3.0, 4.0, 5.0];
        let events = detector.process_sample_all(&samples);

        assert_eq!(events.len(), 3);

        // Check all events
        assert_eq!(events.get(0).unwrap().channel, 1);
        assert_eq!(events.get(0).unwrap().amplitude, 3.0);
        assert_eq!(events.get(1).unwrap().channel, 2);
        assert_eq!(events.get(1).unwrap().amplitude, 4.0);
        assert_eq!(events.get(2).unwrap().channel, 3);
        assert_eq!(events.get(2).unwrap().amplitude, 5.0);
    }

    #[test]
    fn test_process_sample_all_respects_refractory() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 5);

        // First sample - multiple detections
        let samples = [3.0, 3.0, 3.0, 3.0];
        let events = detector.process_sample_all(&samples);
        assert_eq!(events.len(), 4);

        // Second sample - all in refractory
        let events = detector.process_sample_all(&samples);
        assert_eq!(events.len(), 0);

        // After refractory period
        for _ in 0..5 {
            detector.process_sample_all(&[1.0, 1.0, 1.0, 1.0]);
        }

        // Should detect again
        let events = detector.process_sample_all(&samples);
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn test_process_sample_all_vs_process_sample() {
        // Verify process_sample_all detects more than process_sample
        let mut detector1: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);
        let mut detector2: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        let samples = [3.0, 4.0, 5.0, 6.0]; // All above threshold

        // process_sample returns only first
        let single = detector1.process_sample(&samples);
        assert!(single.is_some());
        assert_eq!(single.unwrap().channel, 0);

        // process_sample_all returns all
        let all = detector2.process_sample_all(&samples);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_adaptive_process_sample_all() {
        let mut detector: AdaptiveThresholdDetector<4> =
            AdaptiveThresholdDetector::new(4.0, 10, 50);

        // Calibrate
        for i in 0..50 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val; 4]);
        }
        detector.freeze();

        // Detect multiple spikes (threshold ~0.4)
        let samples = [0.0, 1.0, 1.0, 0.0];
        let events = detector.process_sample_all(&samples);

        assert_eq!(events.len(), 2);
        assert_eq!(events.get(0).unwrap().channel, 1);
        assert_eq!(events.get(1).unwrap().channel, 2);
    }

    #[test]
    fn test_spike_events_iteration() {
        let mut detector: ThresholdDetector<4> = ThresholdDetector::new(2.0, 10);

        let samples = [3.0, 4.0, 1.0, 5.0];
        let events = detector.process_sample_all(&samples);

        // Test iteration
        let mut count = 0;
        for event in events.iter() {
            assert!(event.amplitude > 2.0);
            count += 1;
        }
        assert_eq!(count, 3);

        // Test as_slice
        let slice = events.as_slice();
        assert_eq!(slice.len(), 3);
    }

    // Zero-crossing detector tests

    #[test]
    fn test_zero_crossing_basic_detection() {
        let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);

        // Initialize with negative values
        let crossings = detector.detect(&[-1.0, -1.0, -1.0, -1.0]);
        // First sample returns all false
        assert!(!crossings[0]);
        assert!(!crossings[1]);
        assert!(!crossings[2]);
        assert!(!crossings[3]);

        // Detect crossings (channels 0, 1, 2 cross to positive)
        let crossings = detector.detect(&[1.0, 0.8, 0.5, -0.5]);
        assert!(crossings[0]); // -1.0 → 1.0 crossed
        assert!(crossings[1]); // -1.0 → 0.8 crossed
        assert!(crossings[2]); // -1.0 → 0.5 crossed
        assert!(!crossings[3]); // -1.0 → -0.5 no crossing
    }

    #[test]
    fn test_zero_crossing_threshold_noise_rejection() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.2);

        // Initialize
        detector.detect(&[-1.0, -1.0]);

        // Values within [-0.2, 0.2] are treated as zero
        let crossings = detector.detect(&[0.1, 0.15]); // Within threshold
        assert!(!crossings[0]); // -1.0 → 0.1 (0.1 treated as zero, no crossing)
        assert!(!crossings[1]); // -1.0 → 0.15 (0.15 treated as zero, no crossing)

        // Now go from zero to above threshold - this should cross
        let crossings = detector.detect(&[0.5, 0.3]);
        assert!(!crossings[0]); // 0 → 0.5 doesn't cross (zero to positive)
        assert!(!crossings[1]); // 0 → 0.3 doesn't cross (zero to positive)

        // Now from positive to negative should cross
        let crossings = detector.detect(&[-0.5, -0.3]);
        assert!(crossings[0]); // 0.5 → -0.5 crosses
        assert!(crossings[1]); // 0.3 → -0.3 crosses
    }

    #[test]
    fn test_zero_crossing_rising_only() {
        let mut detector: ZeroCrossingDetector<2> =
            ZeroCrossingDetector::with_direction(0.0, CrossingDirection::Rising);

        // Initialize
        detector.detect(&[-1.0, 1.0]);

        // Rising crossing should be detected
        let crossings = detector.detect(&[1.0, -1.0]);
        assert!(crossings[0]); // -1.0 → 1.0 rising
        assert!(!crossings[1]); // 1.0 → -1.0 falling (not detected)
    }

    #[test]
    fn test_zero_crossing_falling_only() {
        let mut detector: ZeroCrossingDetector<2> =
            ZeroCrossingDetector::with_direction(0.0, CrossingDirection::Falling);

        // Initialize
        detector.detect(&[1.0, -1.0]);

        // Falling crossing should be detected
        let crossings = detector.detect(&[-1.0, 1.0]);
        assert!(crossings[0]); // 1.0 → -1.0 falling
        assert!(!crossings[1]); // -1.0 → 1.0 rising (not detected)
    }

    #[test]
    fn test_zero_crossing_zcr_calculation() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);

        // Create block with different crossing rates
        let mut block = Vec::new();
        for i in 0..100 {
            // Channel 0: slow (one crossing at i=50)
            let ch0 = if i < 50 { -1.0 } else { 1.0 };
            // Channel 1: fast alternation (many crossings)
            let ch1 = if i % 2 == 0 { -1.0 } else { 1.0 };
            block.push([ch0, ch1]);
        }

        let zcr = detector.zcr(&block);
        assert!(zcr[0] < 0.05); // Low ZCR for slow signal (~1/100)
        assert!(zcr[1] > 0.4); // High ZCR for fast signal (~50/100)
    }

    #[test]
    fn test_zero_crossing_detect_any() {
        let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);

        detector.detect(&[-1.0, -1.0, -1.0, -1.0]);

        // One channel crosses
        assert!(detector.detect_any(&[1.0, -1.0, -1.0, -1.0]));

        // No channels cross
        assert!(!detector.detect_any(&[0.5, -0.5, -0.8, -0.3]));
    }

    #[test]
    fn test_zero_crossing_detect_count() {
        let mut detector: ZeroCrossingDetector<4> = ZeroCrossingDetector::new(0.0);

        detector.detect(&[-1.0, -1.0, 1.0, 1.0]);

        // Four channels cross (all change sign: ch0,ch1: -1→1, ch2,ch3: 1→-1)
        let count = detector.detect_count(&[1.0, 1.0, -1.0, -1.0]);
        assert_eq!(count, 4);

        // All channels cross back
        let count = detector.detect_count(&[-1.0, -1.0, 1.0, 1.0]);
        assert_eq!(count, 4);

        // No channels cross (same sign as previous)
        let count = detector.detect_count(&[-0.5, -0.5, 0.5, 0.5]);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_zero_crossing_reset() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);

        detector.detect(&[-1.0, -1.0]);
        let crossings = detector.detect(&[1.0, 1.0]);
        assert!(crossings[0]);
        assert!(crossings[1]);

        detector.reset();

        // After reset, first sample returns all false
        let crossings = detector.detect(&[1.0, 1.0]);
        assert!(!crossings[0]);
        assert!(!crossings[1]);
    }

    #[test]
    fn test_zero_crossing_setters() {
        let mut detector: ZeroCrossingDetector<1> = ZeroCrossingDetector::new(0.1);

        assert_eq!(detector.threshold(), 0.1);
        assert_eq!(detector.direction(), CrossingDirection::Any);

        detector.set_threshold(0.2);
        detector.set_direction(CrossingDirection::Rising);

        assert_eq!(detector.threshold(), 0.2);
        assert_eq!(detector.direction(), CrossingDirection::Rising);
    }

    #[test]
    fn test_zero_crossing_default() {
        let detector: ZeroCrossingDetector<8> = ZeroCrossingDetector::default();

        assert_eq!(detector.threshold(), 0.0);
        assert_eq!(detector.direction(), CrossingDirection::Any);
        assert_eq!(detector.num_channels(), 8);
    }

    #[test]
    fn test_zero_crossing_large_channel_count() {
        let mut detector: ZeroCrossingDetector<1024> = ZeroCrossingDetector::new(0.0);

        assert_eq!(detector.num_channels(), 1024);

        // Initialize with all negative
        let mut samples = [-1.0; 1024];
        detector.detect(&samples);

        // All channels cross to positive
        samples = [1.0; 1024];
        let crossings = detector.detect(&samples);

        // Verify all channels detected crossings
        let count: usize = crossings.iter().filter(|&&x| x).count();
        assert_eq!(count, 1024);
    }

    #[test]
    fn test_zero_crossing_zero_to_positive() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.1);

        // Start at zero (within threshold)
        detector.detect(&[0.05, 0.0]);

        // Moving from zero to positive doesn't count as crossing
        let crossings = detector.detect(&[0.5, 0.5]);
        assert!(!crossings[0]); // 0 → positive: no crossing
        assert!(!crossings[1]); // 0 → positive: no crossing

        // Now from positive to negative should cross
        let crossings = detector.detect(&[-0.5, -0.5]);
        assert!(crossings[0]); // positive → negative: crossing
        assert!(crossings[1]); // positive → negative: crossing
    }

    #[test]
    fn test_zero_crossing_staying_at_zero() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.1);

        // Both samples within threshold
        detector.detect(&[0.05, -0.05]);

        let crossings = detector.detect(&[0.08, -0.08]);
        assert!(!crossings[0]); // 0 → 0: no crossing
        assert!(!crossings[1]); // 0 → 0: no crossing
    }

    #[test]
    fn test_zero_crossing_epilepsy_pattern() {
        let mut detector: ZeroCrossingDetector<1> =
            ZeroCrossingDetector::with_direction(0.0, CrossingDirection::Rising);

        // Simulate spike-wave pattern (characteristic of epilepsy)
        let spike_wave = [
            -0.1, -0.2, 0.8, // Sharp spike (rising)
            0.5, 0.2, -0.3, // Wave descent
            -0.4, -0.2, 0.7, // Another spike
            0.4, 0.1, -0.2, // Wave descent
        ];

        let mut rising_count = 0;
        for &sample in &spike_wave {
            if detector.detect(&[sample])[0] {
                rising_count += 1;
            }
        }

        // Should detect 2 rising crossings (spike onsets)
        assert_eq!(rising_count, 2);
    }

    #[test]
    fn test_zero_crossing_block_processing() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);

        // Create signals with different frequencies
        let mut low_freq_block = Vec::new();
        let mut high_freq_block = Vec::new();

        for i in 0..200 {
            // Low frequency: 5 Hz (10 crossings in 200 samples @ arbitrary rate)
            let low_freq = if i < 100 { -1.0 } else { 1.0 };
            // High frequency: alternating (100 crossings in 200 samples)
            let high_freq = if i % 2 == 0 { -1.0 } else { 1.0 };

            low_freq_block.push([low_freq, low_freq]);
            high_freq_block.push([high_freq, high_freq]);
        }

        // Low frequency signal
        let zcr_low = detector.zcr(&low_freq_block);
        assert!(zcr_low[0] < 0.05); // ~1 crossing / 200 samples

        // Reset and test high frequency
        detector.reset();
        let zcr_high = detector.zcr(&high_freq_block);
        assert!(zcr_high[0] > 0.4); // ~100 crossings / 200 samples
    }

    #[test]
    fn test_zero_crossing_empty_block() {
        let mut detector: ZeroCrossingDetector<2> = ZeroCrossingDetector::new(0.0);

        let empty_block: Vec<[f32; 2]> = Vec::new();
        let zcr = detector.zcr(&empty_block);

        assert_eq!(zcr[0], 0.0);
        assert_eq!(zcr[1], 0.0);
    }

    #[test]
    fn test_zero_crossing_negative_to_negative() {
        let mut detector: ZeroCrossingDetector<1> = ZeroCrossingDetector::new(0.0);

        detector.detect(&[-1.0]);

        // Staying negative - no crossing
        let crossings = detector.detect(&[-0.5]);
        assert!(!crossings[0]);
    }

    #[test]
    fn test_zero_crossing_multi_channel_independence() {
        let mut detector: ZeroCrossingDetector<3> = ZeroCrossingDetector::new(0.0);

        // Initialize each channel differently
        detector.detect(&[-1.0, 0.5, -0.5]);

        // Each channel crosses independently
        let crossings = detector.detect(&[1.0, -0.5, 0.5]);
        assert!(crossings[0]); // -1.0 → 1.0 crosses
        assert!(crossings[1]); // 0.5 → -0.5 crosses
        assert!(crossings[2]); // -0.5 → 0.5 crosses
    }
}
