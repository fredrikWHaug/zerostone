//! Timestamp synchronization primitives for multi-stream BCI systems.
//!
//! Essential building blocks for aligning data from multiple sources (EEG + eye tracker,
//! stimulus markers, etc.) in real-time BCI applications.
//!
//! # Primitives
//!
//! - [`ClockOffset`] - Single timestamp offset measurement (NTP-style)
//! - [`SampleClock`] - Sample index ↔ timestamp conversion
//! - [`LinearDrift`] - Online linear regression for clock drift estimation
//! - [`OffsetBuffer`] - Filtered clock offset buffer with quality selection
//!
//! # Example: Clock Drift Estimation
//!
//! ```
//! use zerostone::{ClockOffset, LinearDrift};
//!
//! // Track drift between EEG system clock and stimulus computer
//! let mut drift = LinearDrift::new();
//!
//! // Collect clock offset measurements over time
//! for i in 0..100 {
//!     let local_time = i as f64 * 0.1;  // Local clock
//!     let offset = 0.5 + local_time * 0.001;  // Offset grows linearly (drift)
//!     let measurement = ClockOffset::new(offset, local_time, 0.01);
//!     drift.add_offset(&measurement);
//! }
//!
//! // Estimate drift rate and correct timestamps
//! let slope = drift.slope();  // ~0.001 seconds/second
//! let corrected_time = drift.correct(50.0);
//! ```
//!
//! # Example: Sample Timestamp Generation
//!
//! ```
//! use zerostone::SampleClock;
//!
//! // EEG recorded at 250 Hz, starting at t=100.5 seconds
//! let clock = SampleClock::new(100.5, 250.0);
//!
//! // Convert sample indices to timestamps
//! let t0 = clock.sample_to_time(0);      // 100.5
//! let t250 = clock.sample_to_time(250);  // 101.5 (1 second later)
//!
//! // Convert timestamps back to sample indices
//! let sample = clock.time_to_sample(101.5);  // 250
//! ```

/// Single timestamp offset measurement (NTP-style).
///
/// Represents a point-in-time measurement of the offset between two clocks.
/// Used as input to drift estimation algorithms and quality-based filtering.
///
/// # Algorithm
///
/// For NTP-style round-trip measurements:
/// ```text
/// offset = ((t2 - t1) + (t3 - t4)) / 2
/// rtt = (t4 - t1) - (t3 - t2)
/// ```
///
/// where:
/// - `t1` = local time when request sent
/// - `t2` = remote time when request received
/// - `t3` = remote time when response sent
/// - `t4` = local time when response received
///
/// # Example
///
/// ```
/// use zerostone::ClockOffset;
///
/// // Direct construction
/// let offset = ClockOffset::new(0.5, 100.0, 0.01);
/// assert_eq!(offset.offset(), 0.5);
/// assert_eq!(offset.rtt(), 0.01);
///
/// // NTP-style construction from round-trip timestamps
/// let offset = ClockOffset::from_ntp(1.0, 1.5, 1.51, 1.03);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ClockOffset {
    /// Clock offset in seconds
    offset: f64,
    /// Local time when measurement was taken (seconds)
    local_time: f64,
    /// Round-trip time in seconds (quality metric, lower is better)
    rtt: f64,
}

impl ClockOffset {
    /// Creates a new clock offset measurement.
    ///
    /// # Arguments
    ///
    /// * `offset` - Clock offset in seconds
    /// * `local_time` - Local time when measurement was taken (seconds)
    /// * `rtt` - Round-trip time in seconds (quality metric)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::ClockOffset;
    ///
    /// let offset = ClockOffset::new(0.5, 100.0, 0.01);
    /// assert_eq!(offset.offset(), 0.5);
    /// ```
    pub fn new(offset: f64, local_time: f64, rtt: f64) -> Self {
        Self {
            offset,
            local_time,
            rtt,
        }
    }

    /// Creates a clock offset from NTP-style round-trip timestamps.
    ///
    /// # Arguments
    ///
    /// * `t1` - Local time when request sent
    /// * `t2` - Remote time when request received
    /// * `t3` - Remote time when response sent
    /// * `t4` - Local time when response received
    ///
    /// # Formula
    ///
    /// ```text
    /// offset = ((t2 - t1) + (t3 - t4)) / 2
    /// rtt = (t4 - t1) - (t3 - t2)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::ClockOffset;
    ///
    /// let offset = ClockOffset::from_ntp(1.0, 1.5, 1.51, 1.03);
    /// assert!(offset.offset().abs() < 0.5);
    /// assert!(offset.rtt() > 0.0);
    /// ```
    pub fn from_ntp(t1: f64, t2: f64, t3: f64, t4: f64) -> Self {
        let offset = ((t2 - t1) + (t3 - t4)) / 2.0;
        let rtt = (t4 - t1) - (t3 - t2);
        let local_time = t4;
        Self {
            offset,
            local_time,
            rtt,
        }
    }

    /// Returns the clock offset in seconds.
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Returns the local time when measurement was taken (seconds).
    pub fn local_time(&self) -> f64 {
        self.local_time
    }

    /// Returns the round-trip time in seconds.
    ///
    /// Lower RTT indicates higher quality measurement (less network jitter).
    pub fn rtt(&self) -> f64 {
        self.rtt
    }

    /// Returns a quality score (inverse of RTT).
    ///
    /// Higher values indicate better quality. Useful for weighted averaging.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::ClockOffset;
    ///
    /// let good = ClockOffset::new(0.5, 100.0, 0.01);
    /// let bad = ClockOffset::new(0.5, 100.0, 0.1);
    /// assert!(good.quality() > bad.quality());
    /// ```
    pub fn quality(&self) -> f64 {
        if self.rtt > 0.0 {
            1.0 / self.rtt
        } else {
            f64::INFINITY
        }
    }
}

/// Sample index ↔ timestamp conversion.
///
/// Converts between discrete sample indices and continuous timestamps for
/// constant-rate data streams. Essential for synchronizing sample-based
/// data (EEG, eye tracking) with event timestamps.
///
/// # Formula
///
/// ```text
/// time = start_time + sample_index / sample_rate
/// sample_index = (time - start_time) * sample_rate
/// ```
///
/// # Example
///
/// ```
/// use zerostone::SampleClock;
///
/// // EEG at 250 Hz starting at t=100.0
/// let clock = SampleClock::new(100.0, 250.0);
///
/// // Sample 250 is at t=101.0 (1 second later)
/// assert!((clock.sample_to_time(250) - 101.0).abs() < 1e-10);
///
/// // t=101.0 corresponds to sample 250
/// assert_eq!(clock.time_to_sample(101.0), 250);
///
/// // Fractional conversion for interpolation
/// assert!((clock.time_to_sample_frac(101.002) - 250.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SampleClock {
    /// Start time in seconds (timestamp of sample 0)
    start_time: f64,
    /// Sample rate in Hz
    sample_rate: f64,
    /// Precomputed 1.0 / sample_rate for efficiency
    sample_period: f64,
}

impl SampleClock {
    /// Creates a new sample clock.
    ///
    /// # Arguments
    ///
    /// * `start_time` - Timestamp of sample 0 (seconds)
    /// * `sample_rate` - Sample rate in Hz (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate <= 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::SampleClock;
    ///
    /// let clock = SampleClock::new(100.0, 250.0);
    /// assert_eq!(clock.start_time(), 100.0);
    /// assert_eq!(clock.sample_rate(), 250.0);
    /// ```
    pub fn new(start_time: f64, sample_rate: f64) -> Self {
        assert!(sample_rate > 0.0, "sample_rate must be positive");
        Self {
            start_time,
            sample_rate,
            sample_period: 1.0 / sample_rate,
        }
    }

    /// Converts sample index to timestamp.
    ///
    /// # Arguments
    ///
    /// * `sample_index` - Sample index (0-based)
    ///
    /// # Returns
    ///
    /// Timestamp in seconds.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::SampleClock;
    ///
    /// let clock = SampleClock::new(100.0, 250.0);
    /// assert_eq!(clock.sample_to_time(0), 100.0);
    /// assert_eq!(clock.sample_to_time(250), 101.0);
    /// ```
    pub fn sample_to_time(&self, sample_index: u64) -> f64 {
        self.start_time + sample_index as f64 * self.sample_period
    }

    /// Converts timestamp to sample index (rounded down).
    ///
    /// # Arguments
    ///
    /// * `time` - Timestamp in seconds
    ///
    /// # Returns
    ///
    /// Sample index (rounded down to nearest integer).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::SampleClock;
    ///
    /// let clock = SampleClock::new(100.0, 250.0);
    /// assert_eq!(clock.time_to_sample(100.0), 0);
    /// assert_eq!(clock.time_to_sample(101.0), 250);
    /// assert_eq!(clock.time_to_sample(100.5), 125);
    /// ```
    pub fn time_to_sample(&self, time: f64) -> u64 {
        ((time - self.start_time) * self.sample_rate) as u64
    }

    /// Converts timestamp to fractional sample index.
    ///
    /// Useful for interpolation when timestamps fall between samples.
    ///
    /// # Arguments
    ///
    /// * `time` - Timestamp in seconds
    ///
    /// # Returns
    ///
    /// Fractional sample index.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::SampleClock;
    ///
    /// let clock = SampleClock::new(100.0, 250.0);
    /// assert_eq!(clock.time_to_sample_frac(100.0), 0.0);
    /// assert_eq!(clock.time_to_sample_frac(101.0), 250.0);
    /// assert!((clock.time_to_sample_frac(100.002) - 0.5).abs() < 1e-6);
    /// ```
    pub fn time_to_sample_frac(&self, time: f64) -> f64 {
        (time - self.start_time) * self.sample_rate
    }

    /// Returns the start time (timestamp of sample 0).
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// Returns the sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

/// Online linear regression for clock drift estimation.
///
/// Estimates linear clock drift using incremental least-squares regression.
/// Computes slope (drift rate) and intercept without storing full measurement history.
///
/// # Algorithm
///
/// Uses centered online formulas for numerically stable least-squares regression:
/// ```text
/// slope = Cov(time, offset) / Var(time)
/// intercept = mean(offset) - slope * mean(time)
/// ```
///
/// where covariance and variance are computed incrementally using deviations from
/// the first measurement to improve numerical stability with large timestamps.
///
/// # BCI Applications
///
/// - Synchronize EEG system clock with stimulus computer
/// - Track drift between eye tracker and neural recording
/// - Correct timestamps across distributed recording systems
///
/// # Example
///
/// ```
/// use zerostone::{ClockOffset, LinearDrift};
///
/// let mut drift = LinearDrift::new();
///
/// // Simulate clock with 1ms/s drift (slope=0.001)
/// for i in 0..100 {
///     let t = i as f64;
///     let offset = 0.5 + t * 0.001;  // Linear drift
///     let measurement = ClockOffset::new(offset, t, 0.01);
///     drift.add_offset(&measurement);
/// }
///
/// // Verify drift estimation
/// assert!((drift.slope() - 0.001).abs() < 1e-6);
/// assert!((drift.intercept() - 0.5).abs() < 1e-6);
///
/// // Correct a timestamp
/// let corrected = drift.correct(50.0);
/// assert!((corrected - 50.55).abs() < 1e-6);  // 50.0 + (0.5 + 50*0.001)
/// ```
#[derive(Debug, Clone)]
pub struct LinearDrift {
    count: u64,
    t0: f64,       // First time value (for centering)
    o0: f64,       // First offset value (for centering)
    sum_dt: f64,   // Sum of (time - t0)
    sum_do: f64,   // Sum of (offset - o0)
    sum_dt2: f64,  // Sum of (time - t0)²
    sum_dtdo: f64, // Sum of (time - t0) * (offset - o0)
}

impl Default for LinearDrift {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearDrift {
    /// Creates a new linear drift estimator.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let drift = LinearDrift::new();
    /// assert_eq!(drift.count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            count: 0,
            t0: 0.0,
            o0: 0.0,
            sum_dt: 0.0,
            sum_do: 0.0,
            sum_dt2: 0.0,
            sum_dtdo: 0.0,
        }
    }

    /// Adds a measurement to the drift estimator.
    ///
    /// # Arguments
    ///
    /// * `time` - Local time when measurement was taken (seconds)
    /// * `offset` - Clock offset at that time (seconds)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let mut drift = LinearDrift::new();
    /// drift.add_measurement(0.0, 0.5);
    /// drift.add_measurement(1.0, 0.501);
    /// assert_eq!(drift.count(), 2);
    /// ```
    pub fn add_measurement(&mut self, time: f64, offset: f64) {
        if self.count == 0 {
            // Store first measurement as reference point
            self.t0 = time;
            self.o0 = offset;
            self.count = 1;
        } else {
            // Compute deviations from first measurement
            let dt = time - self.t0;
            let do_val = offset - self.o0;

            self.sum_dt += dt;
            self.sum_do += do_val;
            self.sum_dt2 += dt * dt;
            self.sum_dtdo += dt * do_val;
            self.count += 1;
        }
    }

    /// Adds a clock offset measurement to the drift estimator.
    ///
    /// # Arguments
    ///
    /// * `offset` - Clock offset measurement
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, LinearDrift};
    ///
    /// let mut drift = LinearDrift::new();
    /// let measurement = ClockOffset::new(0.5, 100.0, 0.01);
    /// drift.add_offset(&measurement);
    /// assert_eq!(drift.count(), 1);
    /// ```
    pub fn add_offset(&mut self, offset: &ClockOffset) {
        self.add_measurement(offset.local_time, offset.offset);
    }

    /// Returns the estimated drift slope (seconds/second).
    ///
    /// Returns 0.0 if fewer than 2 measurements or if variance is too small.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let mut drift = LinearDrift::new();
    /// // Add measurements with linear drift: offset = 0.5 + 0.001 * time
    /// for i in 0..10 {
    ///     let t = (i * 10) as f64;
    ///     let offset = 0.5 + 0.001 * t;
    ///     drift.add_measurement(t, offset);
    /// }
    /// assert!((drift.slope() - 0.001).abs() < 1e-6);
    /// ```
    pub fn slope(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let n = (self.count - 1) as f64; // Exclude first measurement from sums

        // Variance of centered time
        let mean_dt = self.sum_dt / n;
        let var_t = self.sum_dt2 / n - mean_dt * mean_dt;

        // Avoid division by zero
        if var_t < 1e-12 {
            return 0.0;
        }

        // Covariance of centered time and offset
        let mean_do = self.sum_do / n;
        let cov = self.sum_dtdo / n - mean_dt * mean_do;

        cov / var_t
    }

    /// Returns the estimated intercept (seconds).
    ///
    /// Returns mean offset if fewer than 2 measurements.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let mut drift = LinearDrift::new();
    /// // Add measurements with linear drift: offset = 0.5 + 0.001 * time
    /// for i in 0..10 {
    ///     let t = (i * 10) as f64;
    ///     let offset = 0.5 + 0.001 * t;
    ///     drift.add_measurement(t, offset);
    /// }
    /// assert!((drift.intercept() - 0.5).abs() < 1e-6);
    /// ```
    pub fn intercept(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        if self.count == 1 {
            return self.o0;
        }

        let n = (self.count - 1) as f64;
        let mean_dt = self.sum_dt / n;
        let mean_do = self.sum_do / n;

        // Intercept at t0
        let intercept_at_t0 = self.o0 + mean_do - self.slope() * mean_dt;

        // Convert to intercept at t=0
        intercept_at_t0 - self.slope() * self.t0
    }

    /// Corrects a local timestamp using the estimated drift.
    ///
    /// # Arguments
    ///
    /// * `local_time` - Local timestamp to correct (seconds)
    ///
    /// # Returns
    ///
    /// Corrected timestamp accounting for drift.
    ///
    /// # Formula
    ///
    /// ```text
    /// corrected_time = local_time + intercept + slope * local_time
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let mut drift = LinearDrift::new();
    /// drift.add_measurement(0.0, 0.5);
    /// drift.add_measurement(100.0, 0.6);
    ///
    /// // Correct timestamp at t=50
    /// let corrected = drift.correct(50.0);
    /// ```
    pub fn correct(&self, local_time: f64) -> f64 {
        local_time + self.intercept() + self.slope() * local_time
    }

    /// Returns the number of measurements.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Resets the estimator to initial state.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::LinearDrift;
    ///
    /// let mut drift = LinearDrift::new();
    /// drift.add_measurement(0.0, 0.5);
    /// drift.reset();
    /// assert_eq!(drift.count(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.count = 0;
        self.t0 = 0.0;
        self.o0 = 0.0;
        self.sum_dt = 0.0;
        self.sum_do = 0.0;
        self.sum_dt2 = 0.0;
        self.sum_dtdo = 0.0;
    }
}

/// Circular buffer of clock offsets with quality filtering.
///
/// Stores recent clock offset measurements and selects the best based on
/// quality metrics (minimum RTT). Essential for robust synchronization in
/// the presence of network jitter.
///
/// # Type Parameters
///
/// * `N` - Buffer capacity (const generic)
///
/// # Selection Strategies
///
/// - `best_offset()` - Minimum RTT (highest quality)
/// - `latest_offset()` - Most recent measurement
/// - `median_offset()` - Median of buffer (robust to outliers)
///
/// # Example
///
/// ```
/// use zerostone::{ClockOffset, OffsetBuffer};
///
/// let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();
///
/// // Add measurements with varying quality
/// buffer.add(ClockOffset::new(0.5, 100.0, 0.10));  // Poor quality
/// buffer.add(ClockOffset::new(0.51, 100.1, 0.02)); // Good quality
/// buffer.add(ClockOffset::new(0.49, 100.2, 0.05)); // Medium quality
///
/// // Select best quality measurement (minimum RTT)
/// let best = buffer.best_offset().unwrap();
/// assert_eq!(best.rtt(), 0.02);
/// ```
#[derive(Debug, Clone)]
pub struct OffsetBuffer<const N: usize> {
    buffer: [ClockOffset; N],
    index: usize,
    count: usize,
}

impl<const N: usize> OffsetBuffer<N> {
    /// Creates a new empty offset buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::OffsetBuffer;
    ///
    /// let buffer: OffsetBuffer<16> = OffsetBuffer::new();
    /// assert_eq!(buffer.count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            buffer: [ClockOffset::new(0.0, 0.0, 0.0); N],
            index: 0,
            count: 0,
        }
    }

    /// Adds a clock offset measurement to the buffer.
    ///
    /// When buffer is full, overwrites oldest measurement (circular).
    ///
    /// # Arguments
    ///
    /// * `offset` - Clock offset measurement to add
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, OffsetBuffer};
    ///
    /// let mut buffer: OffsetBuffer<4> = OffsetBuffer::new();
    /// buffer.add(ClockOffset::new(0.5, 100.0, 0.01));
    /// assert_eq!(buffer.count(), 1);
    /// ```
    pub fn add(&mut self, offset: ClockOffset) {
        self.buffer[self.index] = offset;
        self.index = (self.index + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// Returns the best quality offset (minimum RTT).
    ///
    /// Returns `None` if buffer is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, OffsetBuffer};
    ///
    /// let mut buffer: OffsetBuffer<4> = OffsetBuffer::new();
    /// buffer.add(ClockOffset::new(0.5, 100.0, 0.10));
    /// buffer.add(ClockOffset::new(0.51, 100.1, 0.02));
    ///
    /// let best = buffer.best_offset().unwrap();
    /// assert_eq!(best.rtt(), 0.02);
    /// ```
    pub fn best_offset(&self) -> Option<ClockOffset> {
        if self.count == 0 {
            return None;
        }

        let mut best = self.buffer[0];
        for i in 1..self.count {
            if self.buffer[i].rtt < best.rtt {
                best = self.buffer[i];
            }
        }

        Some(best)
    }

    /// Returns the most recent offset.
    ///
    /// Returns `None` if buffer is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, OffsetBuffer};
    ///
    /// let mut buffer: OffsetBuffer<4> = OffsetBuffer::new();
    /// buffer.add(ClockOffset::new(0.5, 100.0, 0.01));
    /// buffer.add(ClockOffset::new(0.6, 101.0, 0.02));
    ///
    /// let latest = buffer.latest_offset().unwrap();
    /// assert_eq!(latest.offset(), 0.6);
    /// ```
    pub fn latest_offset(&self) -> Option<ClockOffset> {
        if self.count == 0 {
            return None;
        }

        let latest_idx = if self.index == 0 {
            self.count - 1
        } else {
            self.index - 1
        };

        Some(self.buffer[latest_idx])
    }

    /// Returns the median offset value.
    ///
    /// Returns `None` if buffer is empty. For even-sized buffers, returns
    /// the lower median.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, OffsetBuffer};
    ///
    /// let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();
    /// buffer.add(ClockOffset::new(0.1, 100.0, 0.01));
    /// buffer.add(ClockOffset::new(0.5, 100.1, 0.01));
    /// buffer.add(ClockOffset::new(0.9, 100.2, 0.01));
    ///
    /// let median = buffer.median_offset().unwrap();
    /// assert_eq!(median.offset(), 0.5);
    /// ```
    pub fn median_offset(&self) -> Option<ClockOffset> {
        if self.count == 0 {
            return None;
        }

        // Copy offsets to temporary array for sorting
        let mut offsets = [0.0; N];
        for (i, offset_val) in offsets.iter_mut().enumerate().take(self.count) {
            *offset_val = self.buffer[i].offset;
        }

        // Simple selection sort (fine for small N)
        for i in 0..self.count {
            for j in i + 1..self.count {
                if offsets[j] < offsets[i] {
                    offsets.swap(i, j);
                }
            }
        }

        let median_value = offsets[self.count / 2];

        // Find the measurement with this offset
        for i in 0..self.count {
            if self.buffer[i].offset == median_value {
                return Some(self.buffer[i]);
            }
        }

        None
    }

    /// Returns the number of measurements in the buffer.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Resets the buffer to empty state.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{ClockOffset, OffsetBuffer};
    ///
    /// let mut buffer: OffsetBuffer<4> = OffsetBuffer::new();
    /// buffer.add(ClockOffset::new(0.5, 100.0, 0.01));
    /// buffer.reset();
    /// assert_eq!(buffer.count(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.index = 0;
        self.count = 0;
    }
}

impl<const N: usize> Default for OffsetBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ClockOffset tests
    #[test]
    fn test_clock_offset_new() {
        let offset = ClockOffset::new(0.5, 100.0, 0.01);
        assert_eq!(offset.offset(), 0.5);
        assert_eq!(offset.local_time(), 100.0);
        assert_eq!(offset.rtt(), 0.01);
    }

    #[test]
    fn test_clock_offset_from_ntp() {
        // Simple NTP exchange with known result
        let t1 = 1.0;
        let t2 = 1.5;
        let t3 = 1.51;
        let t4 = 1.03;

        let offset = ClockOffset::from_ntp(t1, t2, t3, t4);

        // offset = ((t2-t1) + (t3-t4)) / 2 = ((1.5-1.0) + (1.51-1.03)) / 2 = (0.5 + 0.48) / 2 = 0.49
        assert!((offset.offset() - 0.49).abs() < 1e-10);

        // rtt = (t4-t1) - (t3-t2) = (1.03-1.0) - (1.51-1.5) = 0.03 - 0.01 = 0.02
        assert!((offset.rtt() - 0.02).abs() < 1e-10);

        assert_eq!(offset.local_time(), t4);
    }

    #[test]
    fn test_clock_offset_quality() {
        let good = ClockOffset::new(0.5, 100.0, 0.01);
        let bad = ClockOffset::new(0.5, 100.0, 0.1);
        assert!(good.quality() > bad.quality());
        assert!((good.quality() - 100.0).abs() < 1e-6);
        assert!((bad.quality() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_clock_offset_copy() {
        let offset1 = ClockOffset::new(0.5, 100.0, 0.01);
        let offset2 = offset1;
        assert_eq!(offset1.offset(), offset2.offset());
    }

    // SampleClock tests
    #[test]
    fn test_sample_clock_new() {
        let clock = SampleClock::new(100.0, 250.0);
        assert_eq!(clock.start_time(), 100.0);
        assert_eq!(clock.sample_rate(), 250.0);
    }

    #[test]
    fn test_sample_clock_sample_to_time() {
        let clock = SampleClock::new(100.0, 250.0);
        assert_eq!(clock.sample_to_time(0), 100.0);
        assert_eq!(clock.sample_to_time(250), 101.0);
        assert_eq!(clock.sample_to_time(500), 102.0);
    }

    #[test]
    fn test_sample_clock_time_to_sample() {
        let clock = SampleClock::new(100.0, 250.0);
        assert_eq!(clock.time_to_sample(100.0), 0);
        assert_eq!(clock.time_to_sample(101.0), 250);
        assert_eq!(clock.time_to_sample(102.0), 500);
        assert_eq!(clock.time_to_sample(100.5), 125);
    }

    #[test]
    fn test_sample_clock_time_to_sample_frac() {
        let clock = SampleClock::new(100.0, 250.0);
        assert_eq!(clock.time_to_sample_frac(100.0), 0.0);
        assert_eq!(clock.time_to_sample_frac(101.0), 250.0);
        assert!((clock.time_to_sample_frac(100.002) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sample_clock_roundtrip() {
        let clock = SampleClock::new(100.0, 250.0);
        for sample in [0, 1, 100, 1000, 10000] {
            let time = clock.sample_to_time(sample);
            let recovered = clock.time_to_sample(time);
            assert_eq!(recovered, sample);
        }
    }

    #[test]
    fn test_sample_clock_long_recording() {
        // Test with 10M samples to verify no accumulation error
        let clock = SampleClock::new(0.0, 1000.0);
        let sample = 10_000_000;
        let time = clock.sample_to_time(sample);
        let recovered = clock.time_to_sample(time);
        assert_eq!(recovered, sample);
    }

    #[test]
    fn test_sample_clock_fractional_start() {
        let clock = SampleClock::new(100.5, 250.0);
        assert_eq!(clock.sample_to_time(0), 100.5);
        assert_eq!(clock.sample_to_time(250), 101.5);
    }

    #[test]
    fn test_sample_clock_various_rates() {
        for rate in [1.0, 250.0, 1000.0, 44100.0] {
            let clock = SampleClock::new(0.0, rate);
            let time = clock.sample_to_time(1000);
            let recovered = clock.time_to_sample(time);
            assert_eq!(recovered, 1000);
        }
    }

    // LinearDrift tests
    #[test]
    fn test_linear_drift_new() {
        let drift = LinearDrift::new();
        assert_eq!(drift.count(), 0);
        assert_eq!(drift.slope(), 0.0);
        assert_eq!(drift.intercept(), 0.0);
    }

    #[test]
    fn test_linear_drift_zero_measurements() {
        let drift = LinearDrift::new();
        assert_eq!(drift.slope(), 0.0);
        assert_eq!(drift.intercept(), 0.0);
    }

    #[test]
    fn test_linear_drift_one_measurement() {
        let mut drift = LinearDrift::new();
        drift.add_measurement(0.0, 0.5);
        assert_eq!(drift.count(), 1);
        assert_eq!(drift.slope(), 0.0);
        assert_eq!(drift.intercept(), 0.5);
    }

    #[test]
    fn test_linear_drift_perfect_fit() {
        let mut drift = LinearDrift::new();

        // y = 0.5 + 2.0 * t
        for i in 0..10 {
            let t = i as f64;
            let offset = 0.5 + 2.0 * t;
            drift.add_measurement(t, offset);
        }

        assert!((drift.slope() - 2.0).abs() < 1e-10);
        assert!((drift.intercept() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_linear_drift_noisy() {
        let mut drift = LinearDrift::new();

        // y = 0.5 + 0.001 * t + noise
        for i in 0..100 {
            let t = i as f64;
            let noise = ((i % 7) as f64 - 3.0) * 0.0001;
            let offset = 0.5 + 0.001 * t + noise;
            drift.add_measurement(t, offset);
        }

        // Should converge close to true values
        assert!((drift.slope() - 0.001).abs() < 0.0001);
        assert!((drift.intercept() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_linear_drift_numerical_stability() {
        let mut drift = LinearDrift::new();

        // Large timestamps (~ 1 billion seconds, ~31 years)
        // Centered algorithm should handle this robustly
        let base_time = 1e9;
        for i in 0..100 {
            let t = base_time + i as f64;
            let offset = 0.5 + 0.001 * (i as f64);
            drift.add_measurement(t, offset);
        }

        // Should compute slope correctly even with large timestamps
        assert!((drift.slope() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_linear_drift_add_offset() {
        let mut drift = LinearDrift::new();

        let offset = ClockOffset::new(0.5, 100.0, 0.01);
        drift.add_offset(&offset);

        assert_eq!(drift.count(), 1);
    }

    #[test]
    fn test_linear_drift_correct() {
        let mut drift = LinearDrift::new();

        // y = 0.5 + 0.001 * t
        for i in 0..100 {
            let t = i as f64;
            let offset = 0.5 + 0.001 * t;
            drift.add_measurement(t, offset);
        }

        // Correct timestamp at t=50
        let corrected = drift.correct(50.0);
        // corrected = 50 + 0.5 + 0.001*50 = 50 + 0.5 + 0.05 = 50.55
        assert!((corrected - 50.55).abs() < 1e-6);
    }

    #[test]
    fn test_linear_drift_reset() {
        let mut drift = LinearDrift::new();
        drift.add_measurement(0.0, 0.5);
        drift.add_measurement(1.0, 0.6);
        assert_eq!(drift.count(), 2);

        drift.reset();
        assert_eq!(drift.count(), 0);
        assert_eq!(drift.slope(), 0.0);
        assert_eq!(drift.intercept(), 0.0);
    }

    #[test]
    fn test_linear_drift_zero_variance() {
        let mut drift = LinearDrift::new();

        // All measurements at same time (zero variance)
        for i in 0..10 {
            drift.add_measurement(100.0, i as f64);
        }

        // Should return slope=0 to avoid division by zero
        assert_eq!(drift.slope(), 0.0);
    }

    // OffsetBuffer tests
    #[test]
    fn test_offset_buffer_new() {
        let buffer: OffsetBuffer<8> = OffsetBuffer::new();
        assert_eq!(buffer.count(), 0);
        assert!(buffer.best_offset().is_none());
        assert!(buffer.latest_offset().is_none());
        assert!(buffer.median_offset().is_none());
    }

    #[test]
    fn test_offset_buffer_add_single() {
        let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();
        let offset = ClockOffset::new(0.5, 100.0, 0.01);
        buffer.add(offset);

        assert_eq!(buffer.count(), 1);
        assert_eq!(buffer.best_offset().unwrap().offset(), 0.5);
        assert_eq!(buffer.latest_offset().unwrap().offset(), 0.5);
    }

    #[test]
    fn test_offset_buffer_wraparound() {
        let mut buffer: OffsetBuffer<4> = OffsetBuffer::new();

        // Add 5 measurements (more than capacity)
        for i in 0..5 {
            buffer.add(ClockOffset::new(i as f64, 100.0 + i as f64, 0.01));
        }

        assert_eq!(buffer.count(), 4); // Should be capped at capacity
    }

    #[test]
    fn test_offset_buffer_best_offset() {
        let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();

        buffer.add(ClockOffset::new(0.5, 100.0, 0.10));
        buffer.add(ClockOffset::new(0.51, 100.1, 0.02)); // Best (min RTT)
        buffer.add(ClockOffset::new(0.49, 100.2, 0.05));

        let best = buffer.best_offset().unwrap();
        assert_eq!(best.rtt(), 0.02);
        assert_eq!(best.offset(), 0.51);
    }

    #[test]
    fn test_offset_buffer_latest_offset() {
        let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();

        buffer.add(ClockOffset::new(0.5, 100.0, 0.01));
        buffer.add(ClockOffset::new(0.6, 101.0, 0.02));
        buffer.add(ClockOffset::new(0.7, 102.0, 0.03)); // Latest

        let latest = buffer.latest_offset().unwrap();
        assert_eq!(latest.offset(), 0.7);
    }

    #[test]
    fn test_offset_buffer_median_offset() {
        let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();

        buffer.add(ClockOffset::new(0.1, 100.0, 0.01));
        buffer.add(ClockOffset::new(0.5, 100.1, 0.01)); // Median
        buffer.add(ClockOffset::new(0.9, 100.2, 0.01));

        let median = buffer.median_offset().unwrap();
        assert_eq!(median.offset(), 0.5);
    }

    #[test]
    fn test_offset_buffer_reset() {
        let mut buffer: OffsetBuffer<8> = OffsetBuffer::new();

        buffer.add(ClockOffset::new(0.5, 100.0, 0.01));
        buffer.add(ClockOffset::new(0.6, 101.0, 0.02));

        assert_eq!(buffer.count(), 2);

        buffer.reset();

        assert_eq!(buffer.count(), 0);
        assert!(buffer.best_offset().is_none());
    }

    #[test]
    fn test_offset_buffer_default() {
        let buffer: OffsetBuffer<16> = Default::default();
        assert_eq!(buffer.count(), 0);
    }
}
