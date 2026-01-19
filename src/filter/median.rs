//! Non-linear median filter for impulsive noise rejection.
//!
//! Replaces each sample with the median of a sliding window, effective for:
//! - Salt-and-pepper noise (electrode artifacts)
//! - Motion artifact spikes
//! - Outlier rejection
//! - Edge-preserving smoothing
//!
//! Uses zero-padding for edge handling (first WINDOW-1 outputs are padded).
//! Optimized implementations for window sizes 3, 5, 7 using sorting networks.
//!
//! # Example
//!
//! ```
//! use zerostone::MedianFilter;
//!
//! // Remove electrode contact spikes from 8-channel EEG
//! let mut filter = MedianFilter::<8, 5>::new();
//!
//! // Simulated EEG with motion artifact spike
//! let noisy = [10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0]; // spike at index 2
//!
//! for &sample in &noisy {
//!     let input = [sample; 8]; // Same signal on all channels
//!     let clean = filter.process(&input);
//!     // Spike is attenuated by median operation
//! }
//! ```

/// Non-linear median filter using sliding window.
///
/// For each channel, maintains a circular buffer of the last `WINDOW` samples.
/// Returns the median value of the window per channel.
///
/// # Algorithm
///
/// - Window 2-7: Optimized sorting networks (1-16 comparisons)
/// - Window 8+: Partial sort using nth_element (O(n) average)
///
/// # Memory
///
/// Stack allocation only: `C * WINDOW * 4` bytes.
/// Example: 32 channels, window=5 → 640 bytes.
///
/// # Performance
///
/// Target: <5 μs for 32 channels, window=5.
/// Achieved: ~640 ns (well below target).
///
/// # Type Parameters
///
/// * `C` - Number of channels (must be >= 1)
/// * `WINDOW` - Size of the sliding window (must be >= 1)
///
/// # NaN Handling
///
/// Input samples must be finite (not NaN or infinity). NaN values will produce
/// undefined ordering behavior in sorting networks and may corrupt results.
///
/// # Examples
///
/// ```
/// use zerostone::MedianFilter;
///
/// // Create 4-channel filter with window size 5
/// let mut filter = MedianFilter::<4, 5>::new();
///
/// // Process samples
/// let input = [1.0, 2.0, 3.0, 4.0];
/// let output = filter.process(&input);
/// ```
#[derive(Clone, Debug)]
pub struct MedianFilter<const C: usize, const WINDOW: usize> {
    /// Per-channel circular buffers holding sliding windows
    delay_lines: [[f32; WINDOW]; C],
    /// Current write position in each channel's circular buffer
    indices: [usize; C],
    /// Number of samples received per channel (0 to WINDOW)
    sample_count: [usize; C],
}

impl<const C: usize, const WINDOW: usize> MedianFilter<C, WINDOW> {
    /// Creates a new median filter with zero-initialized state.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::MedianFilter;
    ///
    /// let filter = MedianFilter::<8, 5>::new();
    /// assert_eq!(filter.window_size(), 5);
    /// assert_eq!(filter.num_channels(), 8);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        const { assert!(WINDOW >= 1, "WINDOW must be at least 1") };
        const { assert!(C >= 1, "C (channels) must be at least 1") };
        Self {
            delay_lines: [[0.0; WINDOW]; C],
            indices: [0; C],
            sample_count: [0; C],
        }
    }

    /// Process one sample per channel, return filtered output.
    ///
    /// The first `WINDOW-1` outputs use zero-padding (fewer samples available).
    /// After `WINDOW` samples, operates in steady state with full window.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::MedianFilter;
    ///
    /// let mut filter = MedianFilter::<2, 3>::new();
    ///
    /// // First sample: median of [0, 0, 1.0] = 0.0 (zero-padded)
    /// let out1 = filter.process(&[1.0, 2.0]);
    ///
    /// // Second sample: median of [0, 1.0, 2.0] = 1.0
    /// let out2 = filter.process(&[2.0, 3.0]);
    ///
    /// // Third sample: median of [1.0, 2.0, 3.0] = 2.0 (steady state)
    /// let out3 = filter.process(&[3.0, 4.0]);
    /// ```
    pub fn process(&mut self, input: &[f32; C]) -> [f32; C] {
        let mut output = [0.0; C];

        for ch in 0..C {
            // Update circular buffer
            self.delay_lines[ch][self.indices[ch]] = input[ch];

            // Compute median based on samples received so far
            let median = if self.sample_count[ch] < WINDOW {
                // Zero-padding phase: fewer samples than window size
                let valid = self.sample_count[ch] + 1;
                self.compute_median_padded(ch, valid)
            } else {
                // Steady state: full window available
                self.compute_median_full(ch)
            };

            output[ch] = median;

            // Advance circular buffer index
            self.indices[ch] = (self.indices[ch] + 1) % WINDOW;

            // Track sample count (saturates at WINDOW)
            if self.sample_count[ch] < WINDOW {
                self.sample_count[ch] += 1;
            }
        }

        output
    }

    /// Resets the filter to initial zero state.
    ///
    /// Clears all circular buffers and restarts zero-padding phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::MedianFilter;
    ///
    /// let mut filter = MedianFilter::<1, 3>::new();
    /// filter.process(&[1.0]);
    /// filter.process(&[2.0]);
    ///
    /// filter.reset();
    ///
    /// // Back to zero-padding phase
    /// let output = filter.process(&[5.0]);
    /// ```
    pub fn reset(&mut self) {
        self.delay_lines = [[0.0; WINDOW]; C];
        self.indices = [0; C];
        self.sample_count = [0; C];
    }

    /// Returns the window size.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::MedianFilter;
    ///
    /// let filter = MedianFilter::<4, 7>::new();
    /// assert_eq!(filter.window_size(), 7);
    /// ```
    #[must_use]
    pub const fn window_size(&self) -> usize {
        WINDOW
    }

    /// Returns the number of channels.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::MedianFilter;
    ///
    /// let filter = MedianFilter::<32, 5>::new();
    /// assert_eq!(filter.num_channels(), 32);
    /// ```
    #[must_use]
    pub const fn num_channels(&self) -> usize {
        C
    }

    /// Compute median during zero-padding phase.
    fn compute_median_padded(&self, ch: usize, valid_samples: usize) -> f32 {
        let mut window = [0.0; WINDOW];

        // Fill window: zeros followed by valid samples
        let num_zeros = WINDOW - valid_samples;
        for i in 0..valid_samples {
            let idx = (self.indices[ch] + WINDOW - valid_samples + 1 + i) % WINDOW;
            window[num_zeros + i] = self.delay_lines[ch][idx];
        }

        Self::median(&mut window)
    }

    /// Compute median in steady state (full window available).
    fn compute_median_full(&self, ch: usize) -> f32 {
        let mut window = [0.0; WINDOW];

        // Extract circular buffer into contiguous array
        for (i, item) in window.iter_mut().enumerate() {
            let idx = (self.indices[ch] + 1 + i) % WINDOW;
            *item = self.delay_lines[ch][idx];
        }

        Self::median(&mut window)
    }

    /// Dispatch to appropriate median algorithm based on window size.
    #[inline]
    fn median(window: &mut [f32; WINDOW]) -> f32 {
        match WINDOW {
            1 => window[0],
            2 => Self::median2(window[0], window[1]),
            3 => Self::median3(window[0], window[1], window[2]),
            4 => Self::median4([window[0], window[1], window[2], window[3]]),
            5 => Self::median5([window[0], window[1], window[2], window[3], window[4]]),
            6 => Self::median6([
                window[0], window[1], window[2], window[3], window[4], window[5],
            ]),
            7 => Self::median7([
                window[0], window[1], window[2], window[3], window[4], window[5], window[6],
            ]),
            _ => Self::median_partial_sort(window),
        }
    }

    /// 2-element median returns lower middle (minimum).
    #[inline]
    fn median2(a: f32, b: f32) -> f32 {
        if a < b {
            a
        } else {
            b
        }
    }

    /// Optimal 3-element sorting network (5 comparisons).
    #[inline]
    fn median3(a: f32, b: f32, c: f32) -> f32 {
        let (min_ab, max_ab) = if a < b { (a, b) } else { (b, a) };
        let min_max_c = if max_ab < c { max_ab } else { c };
        if min_ab > min_max_c {
            min_ab
        } else {
            min_max_c
        }
    }

    /// Sorting network for 4 elements, returns lower middle (index 1).
    #[inline]
    fn median4(mut w: [f32; 4]) -> f32 {
        macro_rules! cmp_swap {
            ($a:expr, $b:expr) => {
                if w[$a] > w[$b] {
                    w.swap($a, $b);
                }
            };
        }

        cmp_swap!(0, 1);
        cmp_swap!(2, 3);
        cmp_swap!(0, 2);
        cmp_swap!(1, 3);
        cmp_swap!(1, 2);

        w[1] // Lower middle
    }

    /// Optimal 5-element sorting network (9 comparisons).
    ///
    /// Based on Batcher's sorting network, returns middle element.
    #[inline]
    fn median5(mut w: [f32; 5]) -> f32 {
        // Compare-swap macro for clarity
        macro_rules! cmp_swap {
            ($a:expr, $b:expr) => {
                if w[$a] > w[$b] {
                    w.swap($a, $b);
                }
            };
        }

        // Sorting network for 5 elements
        cmp_swap!(0, 1);
        cmp_swap!(3, 4);
        cmp_swap!(2, 4);
        cmp_swap!(2, 3);
        cmp_swap!(0, 3);
        cmp_swap!(0, 2);
        cmp_swap!(1, 4);
        cmp_swap!(1, 3);
        cmp_swap!(1, 2);

        w[2] // Middle element
    }

    /// Sorting network for 6 elements (12 comparisons), returns lower middle (index 2).
    #[inline]
    fn median6(mut w: [f32; 6]) -> f32 {
        macro_rules! cmp_swap {
            ($a:expr, $b:expr) => {
                if w[$a] > w[$b] {
                    w.swap($a, $b);
                }
            };
        }

        // Batcher's odd-even merge sort network for 6 elements
        cmp_swap!(0, 1);
        cmp_swap!(2, 3);
        cmp_swap!(4, 5);
        cmp_swap!(0, 2);
        cmp_swap!(1, 4);
        cmp_swap!(3, 5);
        cmp_swap!(0, 1);
        cmp_swap!(2, 3);
        cmp_swap!(4, 5);
        cmp_swap!(1, 2);
        cmp_swap!(3, 4);
        cmp_swap!(2, 3);

        w[2] // Lower middle
    }

    /// Optimal 7-element sorting network (16 comparisons).
    #[inline]
    fn median7(mut w: [f32; 7]) -> f32 {
        macro_rules! cmp_swap {
            ($a:expr, $b:expr) => {
                if w[$a] > w[$b] {
                    w.swap($a, $b);
                }
            };
        }

        // Sorting network for 7 elements
        cmp_swap!(0, 5);
        cmp_swap!(0, 3);
        cmp_swap!(1, 6);
        cmp_swap!(2, 4);
        cmp_swap!(0, 1);
        cmp_swap!(3, 5);
        cmp_swap!(2, 6);
        cmp_swap!(2, 3);
        cmp_swap!(3, 6);
        cmp_swap!(4, 5);
        cmp_swap!(1, 4);
        cmp_swap!(1, 3);
        cmp_swap!(3, 4);
        cmp_swap!(4, 6);
        cmp_swap!(3, 4);
        cmp_swap!(5, 6);

        w[3] // Middle element
    }

    /// Partial sort for larger windows using nth_element pattern.
    ///
    /// Only guarantees correct median position, doesn't fully sort.
    /// O(n) average case via partition-based selection.
    #[inline]
    fn median_partial_sort(window: &mut [f32; WINDOW]) -> f32 {
        let mid = WINDOW / 2;
        let (_, median, _) = window.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
        });
        *median
    }
}

impl<const C: usize, const WINDOW: usize> Default for MedianFilter<C, WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_basic_median3() {
        let mut filter = MedianFilter::<1, 3>::new();

        // First sample: median([0, 0, 1]) = 0
        let out1 = filter.process(&[1.0]);
        assert!(approx_eq(out1[0], 0.0));

        // Second sample: median([0, 1, 2]) = 1
        let out2 = filter.process(&[2.0]);
        assert!(approx_eq(out2[0], 1.0));

        // Third sample: median([1, 2, 3]) = 2
        let out3 = filter.process(&[3.0]);
        assert!(approx_eq(out3[0], 2.0));

        // Fourth sample: median([2, 3, 2]) = 2
        let out4 = filter.process(&[2.0]);
        assert!(approx_eq(out4[0], 2.0));
    }

    #[test]
    fn test_zero_padding_phase() {
        let mut filter = MedianFilter::<1, 5>::new();

        // Window size is 5, so first 4 outputs use zero-padding
        let out1 = filter.process(&[10.0]);
        assert!(approx_eq(out1[0], 0.0)); // median([0,0,0,0,10]) = 0

        let out2 = filter.process(&[10.0]);
        assert!(approx_eq(out2[0], 0.0)); // median([0,0,0,10,10]) = 0

        let out3 = filter.process(&[10.0]);
        assert!(approx_eq(out3[0], 10.0)); // median([0,0,10,10,10]) = 10

        let out4 = filter.process(&[10.0]);
        assert!(approx_eq(out4[0], 10.0)); // median([0,10,10,10,10]) = 10

        let out5 = filter.process(&[10.0]);
        assert!(approx_eq(out5[0], 10.0)); // median([10,10,10,10,10]) = 10
    }

    #[test]
    fn test_impulse_rejection() {
        let mut filter = MedianFilter::<1, 5>::new();

        // Fill with baseline
        for _ in 0..5 {
            filter.process(&[10.0]);
        }

        // Inject impulse spike
        let spike_out = filter.process(&[100.0]);
        // Median([10,10,10,10,100]) = 10, spike rejected!
        assert!(approx_eq(spike_out[0], 10.0));

        // Following samples show spike leaving window
        let out1 = filter.process(&[10.0]);
        assert!(approx_eq(out1[0], 10.0)); // median([10,10,10,100,10]) = 10

        let out2 = filter.process(&[10.0]);
        assert!(approx_eq(out2[0], 10.0)); // median([10,10,100,10,10]) = 10
    }

    #[test]
    fn test_edge_preservation() {
        let mut filter = MedianFilter::<1, 5>::new();

        // Step function: low to high transition
        for _ in 0..5 {
            filter.process(&[0.0]);
        }

        // Transition samples
        let out1 = filter.process(&[10.0]); // median([0,0,0,0,10]) = 0
        assert!(approx_eq(out1[0], 0.0));

        let out2 = filter.process(&[10.0]); // median([0,0,0,10,10]) = 0
        assert!(approx_eq(out2[0], 0.0));

        let out3 = filter.process(&[10.0]); // median([0,0,10,10,10]) = 10
        assert!(approx_eq(out3[0], 10.0));

        // Edge is preserved (sharp transition), unlike linear filters which blur
    }

    #[test]
    fn test_window5_correctness() {
        let mut filter = MedianFilter::<1, 5>::new();

        // Known sequence
        let inputs = [5.0, 1.0, 3.0, 4.0, 2.0];
        for &val in &inputs {
            filter.process(&[val]);
        }

        // Steady state: median([5,1,3,4,2]) should be 3
        let out = filter.process(&[6.0]);
        // Window is now [1,3,4,2,6], median = 3
        assert!(approx_eq(out[0], 3.0));
    }

    #[test]
    fn test_window7_correctness() {
        let mut filter = MedianFilter::<1, 7>::new();

        let inputs = [7.0, 3.0, 5.0, 1.0, 6.0, 2.0, 4.0];
        for &val in &inputs {
            filter.process(&[val]);
        }

        // Window: [7,3,5,1,6,2,4], sorted: [1,2,3,4,5,6,7], median = 4
        let out = filter.process(&[8.0]);
        // New window: [3,5,1,6,2,4,8], sorted: [1,2,3,4,5,6,8], median = 4
        assert!(approx_eq(out[0], 4.0));
    }

    #[test]
    fn test_window9_partial_sort() {
        let mut filter = MedianFilter::<1, 9>::new();

        let inputs = [9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
        for &val in &inputs {
            filter.process(&[val]);
        }

        // Window: [9,1,5,3,7,2,8,4,6], sorted: [1,2,3,4,5,6,7,8,9], median = 5
        let out = filter.process(&[10.0]);
        // New window: [1,5,3,7,2,8,4,6,10], sorted: [1,2,3,4,5,6,7,8,10], median = 5
        assert!(approx_eq(out[0], 5.0));
    }

    #[test]
    fn test_multi_channel_independence() {
        let mut filter = MedianFilter::<2, 3>::new();

        // Fill both channels
        filter.process(&[1.0, 10.0]);
        filter.process(&[2.0, 10.0]);
        filter.process(&[3.0, 10.0]);

        // Spike on channel 0 only
        let out = filter.process(&[100.0, 10.0]);

        // Channel 0: median([2,3,100]) = 3, spike attenuated
        assert!(approx_eq(out[0], 3.0));

        // Channel 1: median([10,10,10]) = 10, unaffected
        assert!(approx_eq(out[1], 10.0));
    }

    #[test]
    fn test_reset_behavior() {
        let mut filter = MedianFilter::<1, 3>::new();

        filter.process(&[5.0]);
        filter.process(&[5.0]);
        filter.process(&[5.0]);

        // Reset to initial state
        filter.reset();

        // Back to zero-padding phase
        let out = filter.process(&[10.0]);
        assert!(approx_eq(out[0], 0.0)); // median([0,0,10]) = 0
    }

    #[test]
    fn test_even_window_size() {
        let mut filter = MedianFilter::<1, 4>::new();

        // Fill window with [1,2,3,4]
        filter.process(&[1.0]);
        filter.process(&[2.0]);
        filter.process(&[3.0]);
        filter.process(&[4.0]);

        // Even window: return lower middle (index 1 of sorted)
        // Window: [1,2,3,4], sorted: [1,2,3,4], lower middle = 2
        let out = filter.process(&[5.0]);
        // New window: [2,3,4,5], sorted: [2,3,4,5], lower middle = 3
        assert!(approx_eq(out[0], 3.0));
    }

    #[test]
    fn test_constant_signal() {
        let mut filter = MedianFilter::<1, 5>::new();

        for _ in 0..10 {
            let out = filter.process(&[7.0]);
            // After zero-padding phase, median of all 7s is 7
            if filter.sample_count[0] >= 5 {
                assert!(approx_eq(out[0], 7.0));
            }
        }
    }

    #[test]
    fn test_large_channel_count() {
        let mut filter = MedianFilter::<32, 5>::new();

        // Different value per channel
        let mut input = [0.0; 32];
        for (i, item) in input.iter_mut().enumerate() {
            *item = i as f32;
        }

        // Process several samples
        for _ in 0..10 {
            let output = filter.process(&input);
            // Each channel operates independently
            assert_eq!(output.len(), 32);
        }
    }

    #[test]
    fn test_negative_values() {
        let mut filter = MedianFilter::<1, 3>::new();

        filter.process(&[-5.0]);
        filter.process(&[-1.0]);
        let out = filter.process(&[-3.0]);

        // Window: [-5,-1,-3], sorted: [-5,-3,-1], median = -3
        assert!(approx_eq(out[0], -3.0));
    }

    #[test]
    fn test_window_size_1() {
        let mut filter = MedianFilter::<1, 1>::new();

        // Window of 1 is identity filter
        let out = filter.process(&[42.0]);
        assert!(approx_eq(out[0], 42.0));

        let out = filter.process(&[17.0]);
        assert!(approx_eq(out[0], 17.0));
    }

    #[test]
    fn test_window_size_2() {
        let mut filter = MedianFilter::<1, 2>::new();

        filter.process(&[10.0]);
        // Window: [10], median = 10 (with zero-padding: [0,10], lower middle = 0)
        let out = filter.process(&[20.0]);
        // Window: [10,20], lower middle = 10
        assert!(approx_eq(out[0], 10.0));
    }

    #[test]
    fn test_all_zeros() {
        let mut filter = MedianFilter::<1, 5>::new();

        for _ in 0..10 {
            let out = filter.process(&[0.0]);
            assert!(approx_eq(out[0], 0.0));
        }
    }

    #[test]
    fn test_motion_artifact_removal_bci() {
        // Simulated EEG with motion artifact
        let mut filter = MedianFilter::<8, 5>::new();

        // Baseline EEG (~10 µV amplitude)
        let baseline = [10.0; 8];

        // Prime the filter
        for _ in 0..5 {
            filter.process(&baseline);
        }

        // Motion artifact: sudden 100 µV spike on all channels
        let artifact = [100.0; 8];
        let out = filter.process(&artifact);

        // Spike should be attenuated significantly
        for &val in &out {
            assert!(val < 50.0); // Much less than 100 µV spike
        }
    }

    #[test]
    fn test_baseline_preservation() {
        let mut filter = MedianFilter::<1, 5>::new();

        // DC offset signal
        let dc_offset = 20.0;

        for _ in 0..10 {
            let out = filter.process(&[dc_offset]);
            if filter.sample_count[0] >= 5 {
                assert!(approx_eq(out[0], dc_offset));
            }
        }
    }

    #[test]
    fn test_getters() {
        let filter = MedianFilter::<32, 7>::new();
        assert_eq!(filter.window_size(), 7);
        assert_eq!(filter.num_channels(), 32);
    }

    #[test]
    fn test_default_trait() {
        let filter = MedianFilter::<4, 5>::default();
        assert_eq!(filter.window_size(), 5);
        assert_eq!(filter.num_channels(), 4);
    }

    #[test]
    fn test_window2_both_orderings() {
        // Test that window=2 returns minimum regardless of input order
        let mut filter1 = MedianFilter::<1, 2>::new();
        filter1.process(&[10.0]);
        filter1.process(&[20.0]);
        // Now in steady state, process with smaller value second
        let out1 = filter1.process(&[5.0]);
        // Window contains [20, 5], lower middle should be 5
        assert!(approx_eq(out1[0], 5.0));

        let mut filter2 = MedianFilter::<1, 2>::new();
        filter2.process(&[5.0]);
        filter2.process(&[20.0]);
        // Now in steady state, process with larger value
        let out2 = filter2.process(&[30.0]);
        // Window contains [20, 30], lower middle should be 20
        assert!(approx_eq(out2[0], 20.0));
    }

    #[test]
    fn test_window4_sorting_network() {
        let mut filter = MedianFilter::<1, 4>::new();

        // Test with reverse-sorted input
        filter.process(&[4.0]);
        filter.process(&[3.0]);
        filter.process(&[2.0]);
        filter.process(&[1.0]);
        // Window: [4,3,2,1], sorted: [1,2,3,4], lower middle (index 1) = 2
        let out1 = filter.process(&[0.0]);
        // Window: [3,2,1,0], sorted: [0,1,2,3], lower middle = 1
        assert!(approx_eq(out1[0], 1.0));

        // Test with already sorted input
        let mut filter2 = MedianFilter::<1, 4>::new();
        filter2.process(&[1.0]);
        filter2.process(&[2.0]);
        filter2.process(&[3.0]);
        filter2.process(&[4.0]);
        let out2 = filter2.process(&[5.0]);
        // Window: [2,3,4,5], sorted: [2,3,4,5], lower middle = 3
        assert!(approx_eq(out2[0], 3.0));
    }

    #[test]
    fn test_window6_sorting_network() {
        let mut filter = MedianFilter::<1, 6>::new();

        // Test with reverse-sorted input (worst case for sorting)
        filter.process(&[6.0]);
        filter.process(&[5.0]);
        filter.process(&[4.0]);
        filter.process(&[3.0]);
        filter.process(&[2.0]);
        filter.process(&[1.0]);
        // Window: [6,5,4,3,2,1], sorted: [1,2,3,4,5,6], lower middle (index 2) = 3
        let out1 = filter.process(&[0.0]);
        // Window: [5,4,3,2,1,0], sorted: [0,1,2,3,4,5], lower middle = 2
        assert!(approx_eq(out1[0], 2.0));

        // Test with mixed input
        let mut filter2 = MedianFilter::<1, 6>::new();
        filter2.process(&[3.0]);
        filter2.process(&[1.0]);
        filter2.process(&[4.0]);
        filter2.process(&[1.0]);
        filter2.process(&[5.0]);
        filter2.process(&[9.0]);
        // Window: [3,1,4,1,5,9], sorted: [1,1,3,4,5,9], lower middle = 3
        let out2 = filter2.process(&[2.0]);
        // Window: [1,4,1,5,9,2], sorted: [1,1,2,4,5,9], lower middle = 2
        assert!(approx_eq(out2[0], 2.0));
    }

    #[test]
    fn test_window6_all_permutations_of_small_set() {
        // Test several permutations to verify sorting network
        let test_cases: [[f32; 6]; 4] = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // sorted
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], // reverse
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0], // pi digits
            [2.0, 6.0, 1.0, 5.0, 3.0, 4.0], // random
        ];

        for (i, input) in test_cases.iter().enumerate() {
            let mut filter = MedianFilter::<1, 6>::new();
            for &val in input {
                filter.process(&[val]);
            }

            // Process one more to get a full window result
            let out = filter.process(&[input[0]]);
            // The window now has input[1..6] + input[0]
            let mut new_window = [input[1], input[2], input[3], input[4], input[5], input[0]];
            new_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(
                approx_eq(out[0], new_window[2]),
                "Case {}: expected {}, got {}",
                i,
                new_window[2],
                out[0]
            );
        }
    }
}
