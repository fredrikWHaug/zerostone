#![no_std]

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

mod stats;
pub use stats::OnlineStats;

/// Biquad (2nd-order IIR) filter coefficients.
///
/// Implements the difference equation:
/// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
///
/// Note: a0 is assumed to be 1.0 (normalized form)
#[derive(Clone, Copy, Debug)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl BiquadCoeffs {
    /// Creates coefficients for a simple passthrough filter (no filtering)
    pub const fn passthrough() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    /// Creates 2nd-order Butterworth lowpass filter coefficients.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `cutoff` - Cutoff frequency in Hz
    ///
    /// # Example
    /// ```
    /// # use zerostone::BiquadCoeffs;
    /// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 40.0);
    /// ```
    pub fn butterworth_lowpass(sample_rate: f32, cutoff: f32) -> Self {
        use core::f32::consts::PI;

        let omega = 2.0 * PI * cutoff / sample_rate;
        let cos_omega = libm::cosf(omega);
        let sin_omega = libm::sinf(omega);
        let alpha = sin_omega / (2.0 * core::f32::consts::SQRT_2);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = b0;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates 2nd-order Butterworth highpass filter coefficients.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `cutoff` - Cutoff frequency in Hz
    pub fn butterworth_highpass(sample_rate: f32, cutoff: f32) -> Self {
        use core::f32::consts::PI;

        let omega = 2.0 * PI * cutoff / sample_rate;
        let cos_omega = libm::cosf(omega);
        let sin_omega = libm::sinf(omega);
        let alpha = sin_omega / (2.0 * core::f32::consts::SQRT_2);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = b0;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates 2nd-order Butterworth bandpass filter coefficients.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `low_cutoff` - Lower cutoff frequency in Hz
    /// * `high_cutoff` - Upper cutoff frequency in Hz
    pub fn butterworth_bandpass(sample_rate: f32, low_cutoff: f32, high_cutoff: f32) -> Self {
        use core::f32::consts::PI;

        let center = libm::sqrtf(low_cutoff * high_cutoff);
        let bandwidth = high_cutoff - low_cutoff;

        let omega = 2.0 * PI * center / sample_rate;
        let cos_omega = libm::cosf(omega);
        let sin_omega = libm::sinf(omega);
        let bw = 2.0 * PI * bandwidth / sample_rate;
        let alpha = sin_omega * libm::sinhf(bw / 2.0);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Zero-allocation IIR filter using cascaded biquad sections.
///
/// Implements infinite impulse response filtering with compile-time known
/// number of sections. Each section is a 2nd-order (biquad) filter.
///
/// # Memory Layout
/// - `coeffs`: Array of biquad coefficient sets
/// - `state`: Delay line storing [x1, x2, y1, y2] for each section
///
/// # Performance
/// Target: <100 ns/sample for 32 channels @ 4th order (2 sections)
///
/// # Example
/// ```
/// # use zerostone::{IirFilter, BiquadCoeffs};
/// // 4th-order Butterworth lowpass at 40 Hz
/// let mut filter: IirFilter<2> = IirFilter::new([
///     BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
///     BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
/// ]);
///
/// let filtered = filter.process_sample(0.5);
/// ```
pub struct IirFilter<const SECTIONS: usize> {
    coeffs: [BiquadCoeffs; SECTIONS],
    // State: [x1, x2, y1, y2] for each section
    state: [[f32; 4]; SECTIONS],
}

impl<const SECTIONS: usize> IirFilter<SECTIONS> {
    /// Creates a new IIR filter from biquad coefficient array.
    pub fn new(coeffs: [BiquadCoeffs; SECTIONS]) -> Self {
        Self {
            coeffs,
            state: [[0.0; 4]; SECTIONS],
        }
    }

    /// Processes a single sample through all cascaded sections.
    ///
    /// # Performance
    /// Optimized for cache locality with sequential state access.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut x = input;

        for i in 0..SECTIONS {
            let c = &self.coeffs[i];
            let s = &mut self.state[i];

            // Direct Form I: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            let y = c.b0 * x + c.b1 * s[0] + c.b2 * s[1] - c.a1 * s[2] - c.a2 * s[3];

            // Update state: shift delay line
            s[1] = s[0]; // x[n-2] = x[n-1]
            s[0] = x;    // x[n-1] = x[n]
            s[3] = s[2]; // y[n-2] = y[n-1]
            s[2] = y;    // y[n-1] = y[n]

            x = y; // Output becomes input to next section
        }

        x
    }

    /// Processes multiple samples in place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Resets filter state to zero (clears delay lines).
    pub fn reset(&mut self) {
        self.state = [[0.0; 4]; SECTIONS];
    }

    /// Returns a reference to the filter coefficients.
    pub fn coefficients(&self) -> &[BiquadCoeffs; SECTIONS] {
        &self.coeffs
    }

    /// Updates filter coefficients (useful for adaptive filtering).
    pub fn set_coefficients(&mut self, coeffs: [BiquadCoeffs; SECTIONS]) {
        self.coeffs = coeffs;
    }
}

/// Zero-allocation FIR (Finite Impulse Response) filter.
///
/// Implements direct-form FIR filtering using a circular buffer for state storage.
/// FIR filters have linear phase response and guaranteed stability.
///
/// # Memory Layout
/// - `coeffs`: Filter tap weights [b0, b1, ..., b_{TAPS-1}]
/// - `delay_line`: Circular buffer storing past TAPS input samples
///
/// # Example
/// ```
/// # use zerostone::FirFilter;
/// // 5-tap moving average filter
/// let coeffs = [0.2, 0.2, 0.2, 0.2, 0.2];
/// let mut filter = FirFilter::new(coeffs);
///
/// let output = filter.process_sample(1.0);
/// ```
pub struct FirFilter<const TAPS: usize> {
    coeffs: [f32; TAPS],
    delay_line: [f32; TAPS],
    index: usize,
}

impl<const TAPS: usize> FirFilter<TAPS> {
    /// Creates a new FIR filter with given tap coefficients.
    ///
    /// # Example
    /// ```
    /// # use zerostone::FirFilter;
    /// // 3-tap moving average
    /// let filter = FirFilter::new([1.0/3.0, 1.0/3.0, 1.0/3.0]);
    /// ```
    pub fn new(coeffs: [f32; TAPS]) -> Self {
        Self {
            coeffs,
            delay_line: [0.0; TAPS],
            index: 0,
        }
    }

    /// Processes a single sample through the FIR filter.
    ///
    /// Implements: y[n] = sum(b[k] * x[n-k]) for k = 0 to TAPS-1
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Store new sample in delay line
        self.delay_line[self.index] = input;

        // Compute output as dot product of coefficients and delay line
        let mut output = 0.0;
        let mut delay_idx = self.index;

        for tap in 0..TAPS {
            output += self.coeffs[tap] * self.delay_line[delay_idx];

            // Move backward through delay line (with wrap-around)
            delay_idx = if delay_idx == 0 { TAPS - 1 } else { delay_idx - 1 };
        }

        // Update index for next sample
        self.index = (self.index + 1) % TAPS;

        output
    }

    /// Processes multiple samples in place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Resets filter state (clears delay line).
    pub fn reset(&mut self) {
        self.delay_line = [0.0; TAPS];
        self.index = 0;
    }

    /// Returns a reference to the filter coefficients.
    pub fn coefficients(&self) -> &[f32; TAPS] {
        &self.coeffs
    }

    /// Updates filter coefficients.
    pub fn set_coefficients(&mut self, coeffs: [f32; TAPS]) {
        self.coeffs = coeffs;
    }

    /// Creates a moving average filter with equal weights.
    ///
    /// # Example
    /// ```
    /// # use zerostone::FirFilter;
    /// let mut filter: FirFilter<5> = FirFilter::moving_average();
    /// ```
    pub fn moving_average() -> Self {
        let weight = 1.0 / TAPS as f32;
        Self::new([weight; TAPS])
    }
}

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
            if amplitude.abs() > self.threshold {
                // Trigger detection
                self.refractory_counter[ch] = self.refractory_samples;
                return Some(SpikeEvent { channel: ch, amplitude });
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
        if sample.abs() > self.threshold {
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

/// Lock-free circular buffer for single-producer/single-consumer scenarios.
/// Uses atomic operations for wait-free push/pop with deterministic latency.
///
/// # Memory Layout
/// - `buffer`: Static allocation of N elements using `MaybeUninit<T>`
/// - `head`: Atomic write position (producer increments)
/// - `tail`: Atomic read position (consumer increments)
///
/// # Thread Safety
/// Safe for concurrent access with one producer thread (push) and one consumer thread (pop).
pub struct CircularBuffer<T, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T, const N: usize> CircularBuffer<T, N> {
    /// Creates a new empty circular buffer.
    ///
    /// # Panics
    /// Panics if N is 0 or not a power of 2 (for efficient modulo via bit masking).
    pub const fn new() -> Self {
        assert!(N > 0 && N.is_power_of_two(), "N must be a power of 2");
        Self {
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Pushes a value into the buffer (producer operation).
    ///
    /// Returns `Err(value)` if the buffer is full.
    ///
    /// # Memory Ordering
    /// Uses `Relaxed` for tail read and `Release` for head write to ensure
    /// the written data is visible to the consumer thread.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);

        // Check if buffer is full
        let next_head = (head + 1) & (N - 1);
        if next_head == tail {
            return Err(value);
        }

        // Write the value
        self.buffer[head].write(value);

        // Publish the write with Release ordering
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Attempts to push without requiring mutable reference.
    /// Useful for scenarios where the buffer is behind shared reference.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let next_head = (head + 1) & (N - 1);
        if next_head == tail {
            return Err(value);
        }

        unsafe {
            let buffer_ptr = self.buffer.as_ptr() as *mut MaybeUninit<T>;
            (*buffer_ptr.add(head)).write(value);
        }

        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Pops a value from the buffer (consumer operation).
    ///
    /// Returns `None` if the buffer is empty.
    ///
    /// # Memory Ordering
    /// Uses `Acquire` for head read to ensure we see all writes from the producer,
    /// and `Release` for tail write to publish consumption.
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if buffer is empty
        if tail == head {
            return None;
        }

        // Read the value
        let value = unsafe { self.buffer[tail].assume_init_read() };

        // Publish the read with Release ordering
        let next_tail = (tail + 1) & (N - 1);
        self.tail.store(next_tail, Ordering::Release);

        Some(value)
    }

    /// Returns the number of elements currently in the buffer.
    ///
    /// Note: In concurrent scenarios, this value may be stale immediately after reading.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);

        if head >= tail {
            head - tail
        } else {
            N - tail + head
        }
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head == tail
    }

    /// Returns `true` if the buffer is full.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let next_head = (head + 1) & (N - 1);
        next_head == tail
    }

    /// Returns the capacity of the buffer.
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<T, const N: usize> Drop for CircularBuffer<T, N> {
    fn drop(&mut self) {
        // Drain all remaining elements to ensure proper cleanup
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer_new() {
        let buffer: CircularBuffer<i32, 8> = CircularBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.capacity(), 8);
    }

    #[test]
    fn test_push_pop_single() {
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        assert_eq!(buffer.push(42), Ok(()));
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());

        assert_eq!(buffer.pop(), Some(42));
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_push_pop_multiple() {
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        for i in 0..5 {
            assert_eq!(buffer.push(i), Ok(()));
        }
        assert_eq!(buffer.len(), 5);

        for i in 0..5 {
            assert_eq!(buffer.pop(), Some(i));
        }
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_full() {
        let mut buffer: CircularBuffer<i32, 4> = CircularBuffer::new();

        // Fill buffer (capacity - 1 = 3 elements for power-of-2 sized buffer)
        assert_eq!(buffer.push(1), Ok(()));
        assert_eq!(buffer.push(2), Ok(()));
        assert_eq!(buffer.push(3), Ok(()));

        // Buffer should now be full
        assert!(buffer.is_full());
        assert_eq!(buffer.push(4), Err(4));
    }

    #[test]
    fn test_wrap_around() {
        let mut buffer: CircularBuffer<i32, 4> = CircularBuffer::new();

        // Fill and drain multiple times to test wrap-around
        for cycle in 0..3 {
            for i in 0..3 {
                assert_eq!(buffer.push(cycle * 10 + i), Ok(()));
            }
            for i in 0..3 {
                assert_eq!(buffer.pop(), Some(cycle * 10 + i));
            }
        }
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_try_push() {
        let buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        assert_eq!(buffer.try_push(1), Ok(()));
        assert_eq!(buffer.try_push(2), Ok(()));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_drop_cleanup() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROP_COUNT.store(0, Ordering::Relaxed);
        {
            let mut buffer: CircularBuffer<DropCounter, 8> = CircularBuffer::new();
            buffer.push(DropCounter).unwrap();
            buffer.push(DropCounter).unwrap();
            buffer.push(DropCounter).unwrap();
            // Buffer goes out of scope here
        }

        // All 3 elements should have been dropped
        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_streaming_scenario() {
        let mut buffer: CircularBuffer<f32, 16> = CircularBuffer::new();

        // Simulate streaming: push some, pop some, repeat
        for chunk in 0..5 {
            // Push 10 samples
            for i in 0..10 {
                let sample = (chunk * 10 + i) as f32;
                buffer.push(sample).unwrap();
            }

            // Process 10 samples
            for i in 0..10 {
                let expected = (chunk * 10 + i) as f32;
                assert_eq!(buffer.pop(), Some(expected));
            }
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_capacity_minus_one_elements() {
        // For a power-of-2 buffer, we can store N-1 elements
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        // Fill with 7 elements (8-1)
        for i in 0..7 {
            assert_eq!(buffer.push(i), Ok(()));
        }

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 7);
    }

    // IirFilter tests
    #[test]
    fn test_iir_passthrough() {
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::passthrough()]);

        // Passthrough should not change the signal
        assert_eq!(filter.process_sample(1.0), 1.0);
        assert_eq!(filter.process_sample(2.5), 2.5);
        assert_eq!(filter.process_sample(-1.5), -1.5);
    }

    #[test]
    fn test_iir_lowpass_dc() {
        // DC (0 Hz) should pass through a lowpass filter
        let mut filter: IirFilter<2> = IirFilter::new([
            BiquadCoeffs::butterworth_lowpass(1000.0, 100.0),
            BiquadCoeffs::butterworth_lowpass(1000.0, 100.0),
        ]);

        // Feed DC signal and let it settle
        for _ in 0..100 {
            filter.process_sample(1.0);
        }

        // After settling, DC should pass through
        let output = filter.process_sample(1.0);
        assert!((output - 1.0).abs() < 0.01, "DC should pass through lowpass");
    }

    #[test]
    fn test_iir_highpass_dc_rejection() {
        // DC should be rejected by a highpass filter
        let mut filter: IirFilter<2> = IirFilter::new([
            BiquadCoeffs::butterworth_highpass(1000.0, 10.0),
            BiquadCoeffs::butterworth_highpass(1000.0, 10.0),
        ]);

        // Feed DC signal
        for _ in 0..100 {
            filter.process_sample(1.0);
        }

        // After settling, DC should be significantly attenuated
        let output = filter.process_sample(1.0);
        assert!(output.abs() < 0.2, "DC should be rejected by highpass, got {}", output);
    }

    #[test]
    fn test_iir_process_block() {
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 100.0,
        )]);

        let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        filter.process_block(&mut samples);

        // All samples should have been processed (modified)
        assert_ne!(samples, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_iir_reset() {
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 40.0,
        )]);

        // Process some samples to build up state
        for i in 0..10 {
            filter.process_sample(i as f32);
        }

        // Reset should clear state
        filter.reset();

        // After reset, same input should produce same output as fresh filter
        let mut filter2: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 40.0,
        )]);

        let out1 = filter.process_sample(5.0);
        let out2 = filter2.process_sample(5.0);

        assert_eq!(out1, out2, "Reset filter should match fresh filter");
    }

    #[test]
    fn test_iir_bandpass() {
        // Bandpass should pass frequencies in the passband
        let mut filter: IirFilter<2> = IirFilter::new([
            BiquadCoeffs::butterworth_bandpass(1000.0, 8.0, 12.0),
            BiquadCoeffs::butterworth_bandpass(1000.0, 8.0, 12.0),
        ]);

        // Just test that it runs without panicking
        for i in 0..100 {
            let sample = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * i as f32 / 1000.0);
            let _ = filter.process_sample(sample);
        }
    }

    #[test]
    fn test_iir_coefficients_access() {
        let coeffs = [
            BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
            BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
        ];
        let mut filter: IirFilter<2> = IirFilter::new(coeffs);

        // Test coefficient access
        let retrieved = filter.coefficients();
        assert_eq!(retrieved.len(), 2);

        // Test coefficient update
        let new_coeffs = [
            BiquadCoeffs::passthrough(),
            BiquadCoeffs::passthrough(),
        ];
        filter.set_coefficients(new_coeffs);

        // Should now behave as passthrough
        assert_eq!(filter.process_sample(3.14), 3.14);
    }

    // FirFilter tests
    #[test]
    fn test_fir_moving_average() {
        let mut filter: FirFilter<5> = FirFilter::moving_average();

        // Feed 5 ones
        for _ in 0..5 {
            filter.process_sample(1.0);
        }

        // After 5 samples of 1.0, moving average should output 1.0
        let output = filter.process_sample(1.0);
        assert!((output - 1.0).abs() < 0.001, "Moving average of ones should be 1.0");
    }

    #[test]
    fn test_fir_impulse_response() {
        let coeffs = [1.0, 2.0, 3.0, 2.0, 1.0];
        let mut filter = FirFilter::new(coeffs);

        // Feed an impulse (1.0 followed by zeros)
        let output1 = filter.process_sample(1.0);
        assert_eq!(output1, 1.0, "First output should be first coefficient");

        let output2 = filter.process_sample(0.0);
        assert_eq!(output2, 2.0, "Second output should be second coefficient");

        let output3 = filter.process_sample(0.0);
        assert_eq!(output3, 3.0, "Third output should be third coefficient");
    }

    #[test]
    fn test_fir_dc_gain() {
        // Coefficients that sum to 1.0
        let coeffs = [0.2, 0.2, 0.2, 0.2, 0.2];
        let mut filter = FirFilter::new(coeffs);

        // Feed DC signal
        for _ in 0..10 {
            filter.process_sample(1.0);
        }

        // DC gain should be sum of coefficients = 1.0
        let output = filter.process_sample(1.0);
        assert!((output - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fir_reset() {
        let mut filter: FirFilter<3> = FirFilter::moving_average();

        // Process some samples
        for i in 0..5 {
            filter.process_sample(i as f32);
        }

        // Reset
        filter.reset();

        // Should produce same output as fresh filter
        let mut fresh_filter: FirFilter<3> = FirFilter::moving_average();
        let out1 = filter.process_sample(10.0);
        let out2 = fresh_filter.process_sample(10.0);

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_fir_process_block() {
        let mut filter: FirFilter<3> = FirFilter::new([1.0, 0.0, 0.0]);

        let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        filter.process_block(&mut samples);

        // With coeffs [1,0,0], output = input (but delayed)
        // First sample: only sees first input
        assert_eq!(samples[0], 1.0);
    }

    #[test]
    fn test_fir_coefficients_access() {
        let coeffs = [0.5, 0.3, 0.2];
        let mut filter = FirFilter::new(coeffs);

        let retrieved = filter.coefficients();
        assert_eq!(retrieved, &[0.5, 0.3, 0.2]);

        // Update coefficients
        filter.set_coefficients([1.0, 0.0, 0.0]);
        assert_eq!(filter.coefficients(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fir_single_tap() {
        // Single tap is just a gain
        let mut filter = FirFilter::new([2.0]);

        assert_eq!(filter.process_sample(1.0), 2.0);
        assert_eq!(filter.process_sample(3.0), 6.0);
    }

    // ThresholdDetector tests
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
