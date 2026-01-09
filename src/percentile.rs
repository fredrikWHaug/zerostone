//! Streaming percentile estimation using the P² algorithm.
//!
//! This module provides [`StreamingPercentile`], an implementation of the P² algorithm
//! by Jain and Chlamtac (1985) for computing percentiles in a streaming fashion without
//! storing observations.
//!
//! # Algorithm
//!
//! The P² algorithm maintains 5 markers to estimate a single percentile:
//! - Marker 0: minimum value
//! - Marker 1: p/2 quantile estimate
//! - Marker 2: p quantile estimate (the target percentile)
//! - Marker 3: (1+p)/2 quantile estimate
//! - Marker 4: maximum value
//!
//! As observations arrive, marker heights are adjusted using piecewise-parabolic
//! interpolation (hence "P²") to track the desired percentile.
//!
//! # Features
//!
//! - O(1) memory per channel (5 markers regardless of sample count)
//! - O(1) update time per sample
//! - No heap allocation
//! - Multi-channel support via const generics
//!
//! # Example
//!
//! ```
//! use zerostone::StreamingPercentile;
//!
//! // Create an estimator for the 8th percentile (common for calcium imaging baseline)
//! let mut estimator: StreamingPercentile<4> = StreamingPercentile::new(0.08);
//!
//! // Feed samples (need at least 5 for valid estimate)
//! for i in 0..100 {
//!     let sample = [i as f64, (i * 2) as f64, (i * 3) as f64, (i * 4) as f64];
//!     estimator.update(&sample);
//! }
//!
//! // Get the percentile estimate
//! if let Some(p8) = estimator.percentile() {
//!     // p8 contains the 8th percentile estimate for each channel
//! }
//! ```
//!
//! # Performance
//!
//! - Update: < 100 ns per channel on modern CPUs
//! - Memory: ~120 bytes per channel (5 markers × 3 values × 8 bytes)
//!
//! # References
//!
//! Jain, R., & Chlamtac, I. (1985). The P² algorithm for dynamic calculation of
//! quantiles and histograms without storing observations. Communications of the ACM,
//! 28(10), 1076-1085.

/// Streaming percentile estimator using the P² algorithm.
///
/// Estimates a single percentile over streaming multi-channel data without storing
/// observations. Uses O(1) memory per channel regardless of sample count.
///
/// # Type Parameters
///
/// * `C` - Number of channels to process independently
///
/// # Example
///
/// ```
/// use zerostone::StreamingPercentile;
///
/// // Estimate the median (50th percentile) for 2 channels
/// let mut median: StreamingPercentile<2> = StreamingPercentile::new(0.5);
///
/// // Feed data
/// for i in 0..1000 {
///     median.update(&[i as f64, (1000 - i) as f64]);
/// }
///
/// // Get estimate (returns None if fewer than 5 samples)
/// let estimate = median.percentile().unwrap();
/// ```
pub struct StreamingPercentile<const C: usize> {
    /// Target percentile (0.0 to 1.0)
    p: f64,
    /// Marker heights (quantile estimates) - 5 markers per channel
    heights: [[f64; 5]; C],
    /// Actual marker positions (sample counts below each marker)
    positions: [[u64; 5]; C],
    /// Desired marker positions
    desired: [[f64; 5]; C],
    /// Position increments (same for all channels)
    increments: [f64; 5],
    /// Total samples processed
    count: u64,
    /// Buffer for first 5 samples during initialization
    init_buffer: [[f64; 5]; C],
    /// Number of samples in init_buffer (0-5)
    init_count: usize,
}

impl<const C: usize> Default for StreamingPercentile<C> {
    /// Creates a median (50th percentile) estimator.
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl<const C: usize> StreamingPercentile<C> {
    /// Creates a new streaming percentile estimator.
    ///
    /// # Arguments
    ///
    /// * `p` - Target percentile as a fraction (0.0 to 1.0). For example, 0.5 for median,
    ///   0.08 for 8th percentile (common baseline in calcium imaging).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the range (0.0, 1.0) exclusive.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::StreamingPercentile;
    ///
    /// // 8th percentile estimator for baseline estimation
    /// let baseline: StreamingPercentile<8> = StreamingPercentile::new(0.08);
    ///
    /// // Median estimator
    /// let median: StreamingPercentile<8> = StreamingPercentile::new(0.5);
    ///
    /// // 95th percentile for peak detection
    /// let peaks: StreamingPercentile<8> = StreamingPercentile::new(0.95);
    /// ```
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p < 1.0, "Percentile p must be in (0.0, 1.0)");

        Self {
            p,
            heights: [[0.0; 5]; C],
            positions: [[0; 5]; C],
            desired: [[0.0; 5]; C],
            increments: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            count: 0,
            init_buffer: [[0.0; 5]; C],
            init_count: 0,
        }
    }

    /// Returns the target percentile value.
    #[inline]
    pub fn target_percentile(&self) -> f64 {
        self.p
    }

    /// Returns the number of samples processed.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns true if enough samples have been processed for a valid estimate.
    ///
    /// The P² algorithm requires at least 5 samples to initialize markers.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.init_count >= 5
    }

    /// Updates the estimator with a new sample.
    ///
    /// Each channel is processed independently. The first 5 samples are buffered
    /// to initialize the markers; subsequent samples update markers using the
    /// P² algorithm.
    ///
    /// # Arguments
    ///
    /// * `sample` - Array of values, one per channel
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::StreamingPercentile;
    ///
    /// let mut est: StreamingPercentile<2> = StreamingPercentile::new(0.5);
    ///
    /// // Process streaming data
    /// est.update(&[1.0, 10.0]);
    /// est.update(&[2.0, 20.0]);
    /// est.update(&[3.0, 30.0]);
    /// est.update(&[4.0, 40.0]);
    /// est.update(&[5.0, 50.0]);  // Markers now initialized
    ///
    /// assert!(est.is_initialized());
    /// ```
    pub fn update(&mut self, sample: &[f64; C]) {
        self.count += 1;

        if self.init_count < 5 {
            // Buffering phase: store first 5 samples
            for (c, &s) in sample.iter().enumerate() {
                self.init_buffer[c][self.init_count] = s;
            }
            self.init_count += 1;

            if self.init_count == 5 {
                // Initialize markers from buffered samples
                self.initialize_markers();
            }
        } else {
            // Normal update using P² algorithm
            for (c, &s) in sample.iter().enumerate() {
                self.update_channel(c, s);
            }
        }
    }

    /// Returns the current percentile estimate for each channel.
    ///
    /// Returns `None` if fewer than 5 samples have been processed.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::StreamingPercentile;
    ///
    /// let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);
    ///
    /// // Not enough samples yet
    /// assert!(est.percentile().is_none());
    ///
    /// // Add 5 samples
    /// for i in 1..=5 {
    ///     est.update(&[i as f64]);
    /// }
    ///
    /// // Now we have an estimate
    /// let median = est.percentile().unwrap();
    /// ```
    pub fn percentile(&self) -> Option<[f64; C]> {
        if self.init_count < 5 {
            return None;
        }

        let mut result = [0.0; C];
        for (r, h) in result.iter_mut().zip(self.heights.iter()) {
            *r = h[2]; // Marker 2 is the p-quantile estimate
        }
        Some(result)
    }

    /// Returns the current minimum value for each channel.
    ///
    /// Returns `None` if fewer than 5 samples have been processed.
    pub fn min(&self) -> Option<[f64; C]> {
        if self.init_count < 5 {
            return None;
        }

        let mut result = [0.0; C];
        for (r, h) in result.iter_mut().zip(self.heights.iter()) {
            *r = h[0];
        }
        Some(result)
    }

    /// Returns the current maximum value for each channel.
    ///
    /// Returns `None` if fewer than 5 samples have been processed.
    pub fn max(&self) -> Option<[f64; C]> {
        if self.init_count < 5 {
            return None;
        }

        let mut result = [0.0; C];
        for (r, h) in result.iter_mut().zip(self.heights.iter()) {
            *r = h[4];
        }
        Some(result)
    }

    /// Resets the estimator to its initial state.
    ///
    /// Clears all markers and sample count. The target percentile is preserved.
    pub fn reset(&mut self) {
        self.heights = [[0.0; 5]; C];
        self.positions = [[0; 5]; C];
        self.desired = [[0.0; 5]; C];
        self.count = 0;
        self.init_buffer = [[0.0; 5]; C];
        self.init_count = 0;
    }

    /// Initializes markers from the first 5 samples.
    fn initialize_markers(&mut self) {
        for c in 0..C {
            // Sort the 5 samples for this channel
            let mut sorted = self.init_buffer[c];
            self.sort5(&mut sorted);

            // Set marker heights to sorted values
            self.heights[c] = sorted;

            // Set initial positions [0, 1, 2, 3, 4]
            self.positions[c] = [0, 1, 2, 3, 4];

            // Set desired positions based on percentile
            // n' = [0, 2p, 4p, 2+2p, 4] for the first 5 samples
            // But we express as fractional positions that grow with sample count
            self.desired[c] = [0.0, 2.0 * self.p, 4.0 * self.p, 2.0 + 2.0 * self.p, 4.0];
        }
    }

    /// Sorts exactly 5 elements in place using an optimal sorting network.
    #[inline]
    fn sort5(&self, arr: &mut [f64; 5]) {
        // Optimal sorting network for 5 elements (9 comparisons)
        Self::compare_swap(arr, 0, 1);
        Self::compare_swap(arr, 3, 4);
        Self::compare_swap(arr, 2, 4);
        Self::compare_swap(arr, 2, 3);
        Self::compare_swap(arr, 0, 3);
        Self::compare_swap(arr, 0, 2);
        Self::compare_swap(arr, 1, 4);
        Self::compare_swap(arr, 1, 3);
        Self::compare_swap(arr, 1, 2);
    }

    #[inline]
    fn compare_swap(arr: &mut [f64; 5], i: usize, j: usize) {
        if arr[i] > arr[j] {
            arr.swap(i, j);
        }
    }

    /// Updates a single channel with a new observation using the P² algorithm.
    #[inline]
    fn update_channel(&mut self, c: usize, x: f64) {
        let heights = &mut self.heights[c];
        let positions = &mut self.positions[c];
        let desired = &mut self.desired[c];
        let increments = &self.increments;

        // Step 1: Find the cell k where x falls
        let k = find_cell(heights, x);

        // Step 2: Handle extreme values
        if x < heights[0] {
            heights[0] = x;
        }
        if x > heights[4] {
            heights[4] = x;
        }

        // Step 3: Increment positions of markers to the right of k
        for pos in positions.iter_mut().skip(k + 1) {
            *pos += 1;
        }

        // Step 4: Update desired positions
        for (d, &inc) in desired.iter_mut().zip(increments.iter()) {
            *d += inc;
        }

        // Step 5: Adjust middle markers (1, 2, 3) if needed
        for i in 1..4 {
            let d = desired[i] - positions[i] as f64;

            if (d >= 1.0 && positions[i + 1] > positions[i] + 1)
                || (d <= -1.0 && positions[i - 1] + 1 < positions[i])
            {
                let d_sign = if d >= 0.0 { 1i64 } else { -1i64 };

                // Try parabolic interpolation
                let q_new = parabolic(heights, positions, i, d_sign);

                if heights[i - 1] < q_new && q_new < heights[i + 1] {
                    // Parabolic result is valid
                    heights[i] = q_new;
                } else {
                    // Fall back to linear interpolation
                    heights[i] = linear(heights, positions, i, d_sign);
                }

                // Update position
                positions[i] = (positions[i] as i64 + d_sign) as u64;
            }
        }
    }
}

/// Finds the cell index k where heights[k] <= x < heights[k+1].
/// Returns the rightmost valid k (0..4).
#[inline]
fn find_cell(heights: &[f64; 5], x: f64) -> usize {
    if x < heights[1] {
        0
    } else if x < heights[2] {
        1
    } else if x < heights[3] {
        2
    } else {
        3
    }
}

/// Computes the parabolic (P²) interpolation for marker adjustment.
#[inline]
fn parabolic(heights: &[f64; 5], positions: &[u64; 5], i: usize, d: i64) -> f64 {
    let d_f = d as f64;
    let n_i = positions[i] as f64;
    let n_im1 = positions[i - 1] as f64;
    let n_ip1 = positions[i + 1] as f64;

    let q_i = heights[i];
    let q_im1 = heights[i - 1];
    let q_ip1 = heights[i + 1];

    // P² formula
    q_i + d_f / (n_ip1 - n_im1)
        * ((n_i - n_im1 + d_f) * (q_ip1 - q_i) / (n_ip1 - n_i)
            + (n_ip1 - n_i - d_f) * (q_i - q_im1) / (n_i - n_im1))
}

/// Computes linear interpolation for marker adjustment (fallback).
#[inline]
fn linear(heights: &[f64; 5], positions: &[u64; 5], i: usize, d: i64) -> f64 {
    let idx = if d >= 0 { i + 1 } else { i - 1 };

    let n_i = positions[i] as f64;
    let n_other = positions[idx] as f64;
    let q_i = heights[i];
    let q_other = heights[idx];

    q_i + (d as f64) * (q_other - q_i) / (n_other - n_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let est: StreamingPercentile<4> = StreamingPercentile::new(0.5);
        assert_eq!(est.target_percentile(), 0.5);
        assert_eq!(est.count(), 0);
        assert!(!est.is_initialized());
    }

    #[test]
    fn test_default_is_median() {
        let est: StreamingPercentile<1> = StreamingPercentile::default();
        assert_eq!(est.target_percentile(), 0.5);
    }

    #[test]
    #[should_panic(expected = "Percentile p must be in (0.0, 1.0)")]
    fn test_invalid_percentile_zero() {
        let _: StreamingPercentile<1> = StreamingPercentile::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Percentile p must be in (0.0, 1.0)")]
    fn test_invalid_percentile_one() {
        let _: StreamingPercentile<1> = StreamingPercentile::new(1.0);
    }

    #[test]
    fn test_initialization_requires_5_samples() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        for i in 1..5 {
            est.update(&[i as f64]);
            assert!(!est.is_initialized());
            assert!(est.percentile().is_none());
        }

        est.update(&[5.0]);
        assert!(est.is_initialized());
        assert!(est.percentile().is_some());
    }

    #[test]
    fn test_median_simple_sequence() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // Feed 1, 2, 3, 4, 5 - median should be 3
        for i in 1..=5 {
            est.update(&[i as f64]);
        }

        let median = est.percentile().unwrap()[0];
        assert!(
            (median - 3.0).abs() < 0.1,
            "Expected median ~3.0, got {}",
            median
        );
    }

    #[test]
    fn test_median_convergence() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // Feed uniform sequence 0..1000
        for i in 0..1000 {
            est.update(&[i as f64]);
        }

        // True median is 499.5
        let median = est.percentile().unwrap()[0];
        let error = (median - 499.5).abs();

        assert!(
            error < 50.0,
            "Median error {} too large (expected ~499.5, got {})",
            error,
            median
        );
    }

    #[test]
    fn test_8th_percentile_convergence() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.08);

        // Feed uniform sequence 0..1000
        for i in 0..1000 {
            est.update(&[i as f64]);
        }

        // True 8th percentile is ~80
        let p8 = est.percentile().unwrap()[0];
        let error = (p8 - 80.0).abs();

        assert!(
            error < 50.0,
            "8th percentile error {} too large (expected ~80, got {})",
            error,
            p8
        );
    }

    #[test]
    fn test_multi_channel_independence() {
        let mut est: StreamingPercentile<2> = StreamingPercentile::new(0.5);

        // Channel 0: 1-1000, Channel 1: 1001-2000
        for i in 0..1000 {
            est.update(&[i as f64, (i + 1000) as f64]);
        }

        let medians = est.percentile().unwrap();

        // Channel 0 median should be ~500
        assert!(
            (medians[0] - 500.0).abs() < 100.0,
            "Channel 0 median {} not near 500",
            medians[0]
        );

        // Channel 1 median should be ~1500
        assert!(
            (medians[1] - 1500.0).abs() < 100.0,
            "Channel 1 median {} not near 1500",
            medians[1]
        );
    }

    #[test]
    fn test_min_max() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        for i in 0..100 {
            est.update(&[i as f64]);
        }

        let min = est.min().unwrap()[0];
        let max = est.max().unwrap()[0];

        assert_eq!(min, 0.0);
        assert_eq!(max, 99.0);
    }

    #[test]
    fn test_reset() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // Feed some data
        for i in 0..100 {
            est.update(&[i as f64]);
        }
        assert!(est.is_initialized());
        assert_eq!(est.count(), 100);

        // Reset
        est.reset();

        assert!(!est.is_initialized());
        assert_eq!(est.count(), 0);
        assert!(est.percentile().is_none());

        // Should work again after reset
        for i in 0..5 {
            est.update(&[i as f64]);
        }
        assert!(est.is_initialized());
    }

    #[test]
    fn test_constant_values() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // All same value
        for _ in 0..100 {
            est.update(&[42.0]);
        }

        let median = est.percentile().unwrap()[0];
        assert_eq!(median, 42.0);
    }

    #[test]
    fn test_reverse_sorted_input() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // Feed in reverse order
        for i in (0..1000).rev() {
            est.update(&[i as f64]);
        }

        let median = est.percentile().unwrap()[0];
        let error = (median - 499.5).abs();

        assert!(
            error < 50.0,
            "Median error {} too large for reverse input",
            error
        );
    }

    #[test]
    fn test_sorted_input() {
        let mut est: StreamingPercentile<1> = StreamingPercentile::new(0.5);

        // Already sorted
        for i in 0..1000 {
            est.update(&[i as f64]);
        }

        let median = est.percentile().unwrap()[0];
        let error = (median - 499.5).abs();

        assert!(
            error < 50.0,
            "Median error {} too large for sorted input",
            error
        );
    }

    #[test]
    fn test_extreme_percentiles() {
        // Test 1st percentile
        let mut est_low: StreamingPercentile<1> = StreamingPercentile::new(0.01);
        for i in 0..10000 {
            est_low.update(&[i as f64]);
        }
        let p1 = est_low.percentile().unwrap()[0];
        assert!(p1 < 500.0, "1st percentile {} should be low", p1);

        // Test 99th percentile
        let mut est_high: StreamingPercentile<1> = StreamingPercentile::new(0.99);
        for i in 0..10000 {
            est_high.update(&[i as f64]);
        }
        let p99 = est_high.percentile().unwrap()[0];
        assert!(p99 > 9500.0, "99th percentile {} should be high", p99);
    }
}
