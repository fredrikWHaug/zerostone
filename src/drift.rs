//! Drift estimation for spike sorting in chronic recordings.
//!
//! During long electrophysiology recordings, neural tissue can move relative
//! to the probe ("drift"), causing the same neuron's spikes to appear at
//! shifting positions over time. This module provides a simple linear drift
//! model that:
//!
//! 1. Bins detected spikes into time windows
//! 2. Computes the average position in each bin
//! 3. Fits a linear regression (position vs time) to estimate drift rate
//! 4. Corrects spike positions by subtracting the estimated drift
//!
//! For real-time 32-128 channel systems, a linear model is sufficient
//! for recordings up to a few hours. For longer sessions or large-scale
//! probes, piecewise-linear or registration-based methods (Kilosort4)
//! may be needed.
//!
//! # Example
//!
//! ```
//! use zerostone::drift::{DriftEstimator, estimate_drift_from_positions};
//!
//! let mut est = DriftEstimator::<16>::new(1000);
//! // Spikes drifting 10um per 1000 samples
//! for i in 0..8 {
//!     est.add_spike(i * 1000 + 500, 100.0 + i as f64 * 10.0);
//! }
//! est.fit();
//! assert!(est.slope() > 0.0); // positive drift
//! ```

/// Drift estimator using binned spike positions and linear regression.
///
/// Accumulates spike positions over time bins, then fits a linear model
/// to estimate drift rate (slope) and baseline position (intercept).
///
/// # Type Parameters
///
/// * `MAX_BINS` -- Maximum number of time bins to track
///
/// # Example
///
/// ```
/// use zerostone::drift::DriftEstimator;
///
/// let mut est = DriftEstimator::<8>::new(500);
/// est.add_spike(100, 50.0);
/// est.add_spike(600, 55.0);
/// est.add_spike(1100, 60.0);
/// est.fit();
/// assert!(est.slope() > 0.0);
/// let corrected = est.correct_position(1100, 60.0);
/// // Corrected position should be closer to the initial position
/// assert!((corrected - 50.0f64).abs() < (60.0f64 - 50.0).abs());
/// ```
pub struct DriftEstimator<const MAX_BINS: usize> {
    bin_sum_y: [f64; MAX_BINS],
    bin_counts: [usize; MAX_BINS],
    bin_duration: usize,
    n_bins: usize,
    slope: f64,
    intercept: f64,
    fitted: bool,
}

impl<const MAX_BINS: usize> DriftEstimator<MAX_BINS> {
    /// Create a new drift estimator with a given bin duration in samples.
    ///
    /// # Panics
    ///
    /// Panics if `bin_duration_samples` is zero.
    pub fn new(bin_duration_samples: usize) -> Self {
        assert!(bin_duration_samples > 0, "bin_duration_samples must be > 0");
        Self {
            bin_sum_y: [0.0; MAX_BINS],
            bin_counts: [0; MAX_BINS],
            bin_duration: bin_duration_samples,
            n_bins: 0,
            slope: 0.0,
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Add a spike's position and time to the estimator.
    ///
    /// The spike is assigned to a bin based on `sample_index / bin_duration`.
    /// If the bin index exceeds `MAX_BINS`, the spike is ignored.
    pub fn add_spike(&mut self, sample_index: usize, position_y: f64) {
        let bin = sample_index / self.bin_duration;
        if bin >= MAX_BINS {
            return;
        }
        self.bin_sum_y[bin] += position_y;
        self.bin_counts[bin] += 1;
        if bin >= self.n_bins {
            self.n_bins = bin + 1;
        }
    }

    /// Fit a linear regression on the bin-averaged positions.
    ///
    /// Uses ordinary least squares: position_y = slope * bin_time + intercept.
    /// Requires at least 2 bins with spikes. If fewer, slope and intercept
    /// remain at 0.0 and `fitted` stays false.
    pub fn fit(&mut self) {
        // Collect bins that have data
        let mut xs = [0.0f64; MAX_BINS];
        let mut ys = [0.0f64; MAX_BINS];
        let mut n = 0usize;

        let mut bin = 0;
        while bin < self.n_bins && bin < MAX_BINS {
            if self.bin_counts[bin] > 0 {
                // Use bin center time in sample units
                xs[n] = (bin as f64 + 0.5) * self.bin_duration as f64;
                ys[n] = self.bin_sum_y[bin] / self.bin_counts[bin] as f64;
                n += 1;
            }
            bin += 1;
        }

        if n < 2 {
            self.fitted = false;
            self.slope = 0.0;
            self.intercept = 0.0;
            return;
        }

        // OLS: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        let mut i = 0;
        while i < n {
            sum_x += xs[i];
            sum_y += ys[i];
            sum_xy += xs[i] * ys[i];
            sum_x2 += xs[i] * xs[i];
            i += 1;
        }

        let nf = n as f64;
        let denom = nf * sum_x2 - sum_x * sum_x;
        if libm::fabs(denom) < 1e-30 {
            self.slope = 0.0;
            self.intercept = sum_y / nf;
            self.fitted = true;
            return;
        }

        self.slope = (nf * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / nf;
        self.fitted = true;
    }

    /// Estimate the position drift at a given sample index.
    ///
    /// Returns the estimated position offset relative to time zero.
    /// If the model has not been fitted, returns 0.0.
    pub fn estimate_drift(&self, sample_index: usize) -> f64 {
        if !self.fitted {
            return 0.0;
        }
        let t = sample_index as f64;
        // Drift = (predicted position at t) - (predicted position at t=0)
        // predicted(t) = slope * t + intercept
        // predicted(0) = intercept
        // drift = slope * t
        self.slope * t
    }

    /// Correct a spike's position by removing estimated drift.
    ///
    /// Returns `position_y - estimate_drift(sample_index)`.
    pub fn correct_position(&self, sample_index: usize, position_y: f64) -> f64 {
        position_y - self.estimate_drift(sample_index)
    }

    /// The fitted drift rate (position units per sample).
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// The fitted intercept (position at time zero).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Number of bins that contain at least one spike.
    pub fn n_bins_used(&self) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i < self.n_bins && i < MAX_BINS {
            if self.bin_counts[i] > 0 {
                count += 1;
            }
            i += 1;
        }
        count
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        let mut i = 0;
        while i < MAX_BINS {
            self.bin_sum_y[i] = 0.0;
            self.bin_counts[i] = 0;
            i += 1;
        }
        self.n_bins = 0;
        self.slope = 0.0;
        self.intercept = 0.0;
        self.fitted = false;
    }
}

impl<const MAX_BINS: usize> Default for DriftEstimator<MAX_BINS> {
    fn default() -> Self {
        Self {
            bin_sum_y: [0.0; MAX_BINS],
            bin_counts: [0; MAX_BINS],
            bin_duration: 1000,
            n_bins: 0,
            slope: 0.0,
            intercept: 0.0,
            fitted: false,
        }
    }
}

/// Estimate drift from spike sample indices and y-positions (batch mode).
///
/// Bins the spikes, computes per-bin average position, and fits a linear
/// regression. Returns `Some((slope, intercept))` if at least 2 bins
/// have data, or `None` otherwise.
///
/// # Arguments
///
/// * `sample_indices` -- sample index of each spike
/// * `positions_y` -- y-position of each spike
/// * `bin_duration_samples` -- bin width in samples
/// * `max_bins` -- maximum number of bins (bins beyond this are ignored)
///
/// # Example
///
/// ```
/// use zerostone::drift::estimate_drift_from_positions;
///
/// let samples = [100, 1100, 2100, 3100];
/// let positions = [50.0, 60.0, 70.0, 80.0];
/// let result = estimate_drift_from_positions(&samples, &positions, 1000, 8);
/// assert!(result.is_some());
/// let (slope, _intercept) = result.unwrap();
/// assert!(slope > 0.0);
/// ```
pub fn estimate_drift_from_positions(
    sample_indices: &[usize],
    positions_y: &[f64],
    bin_duration_samples: usize,
    max_bins: usize,
) -> Option<(f64, f64)> {
    if sample_indices.is_empty() || positions_y.is_empty() || bin_duration_samples == 0 {
        return None;
    }
    let n = if sample_indices.len() < positions_y.len() {
        sample_indices.len()
    } else {
        positions_y.len()
    };

    // Use stack-allocated arrays for small bin counts, heap for larger
    // Since we're no_std, use a fixed upper bound
    const STACK_BINS: usize = 256;
    let effective_max = if max_bins < STACK_BINS {
        max_bins
    } else {
        STACK_BINS
    };

    let mut bin_sum = [0.0f64; STACK_BINS];
    let mut bin_count = [0usize; STACK_BINS];

    let mut i = 0;
    while i < n {
        let bin = sample_indices[i] / bin_duration_samples;
        if bin < effective_max {
            bin_sum[bin] += positions_y[i];
            bin_count[bin] += 1;
        }
        i += 1;
    }

    // Collect bins with data and fit OLS
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut count = 0usize;

    let mut bin = 0;
    while bin < effective_max {
        if bin_count[bin] > 0 {
            let x = (bin as f64 + 0.5) * bin_duration_samples as f64;
            let y = bin_sum[bin] / bin_count[bin] as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            count += 1;
        }
        bin += 1;
    }

    if count < 2 {
        return None;
    }

    let nf = count as f64;
    let denom = nf * sum_x2 - sum_x * sum_x;
    if libm::fabs(denom) < 1e-30 {
        return Some((0.0, sum_y / nf));
    }

    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / nf;
    Some((slope, intercept))
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that add_spike + fit + estimate_drift don't panic for bounded inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn drift_estimator_no_panic() {
        let mut est = DriftEstimator::<4>::new(100);
        let sample: usize = kani::any();
        let pos: f64 = kani::any();
        kani::assume(sample < 1000);
        kani::assume(pos.is_finite() && pos >= -1e4 && pos <= 1e4);
        est.add_spike(sample, pos);
        est.fit();
        let d = est.estimate_drift(sample);
        assert!(d.is_finite());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_drift_zero_slope() {
        let mut est = DriftEstimator::<16>::new(1000);
        // Constant position across bins
        let mut i = 0;
        while i < 8 {
            est.add_spike(i * 1000 + 500, 100.0);
            i += 1;
        }
        est.fit();
        assert!(est.is_fitted());
        assert!(
            libm::fabs(est.slope()) < 1e-10,
            "Slope should be ~0, got {}",
            est.slope()
        );
    }

    #[test]
    fn test_linear_drift_recovery() {
        let mut est = DriftEstimator::<16>::new(1000);
        // Position increases by 10um per 1000 samples
        let mut i = 0;
        while i < 8 {
            est.add_spike(i * 1000 + 500, 100.0 + i as f64 * 10.0);
            i += 1;
        }
        est.fit();
        assert!(est.is_fitted());
        // slope should be ~10/1000 = 0.01
        assert!(
            libm::fabs(est.slope() - 0.01) < 0.002,
            "Slope should be ~0.01, got {}",
            est.slope()
        );
    }

    #[test]
    fn test_correct_position_removes_drift() {
        let mut est = DriftEstimator::<16>::new(1000);
        let mut i = 0;
        while i < 8 {
            est.add_spike(i * 1000 + 500, 100.0 + i as f64 * 10.0);
            i += 1;
        }
        est.fit();

        // Corrected positions should be approximately constant
        let base = est.correct_position(500, 100.0);
        i = 1;
        while i < 8 {
            let corrected = est.correct_position(i * 1000 + 500, 100.0 + i as f64 * 10.0);
            assert!(
                libm::fabs(corrected - base) < 2.0,
                "Corrected position at bin {} = {}, expected ~{}",
                i,
                corrected,
                base
            );
            i += 1;
        }
    }

    #[test]
    fn test_single_bin_no_fit() {
        let mut est = DriftEstimator::<8>::new(1000);
        est.add_spike(500, 100.0);
        est.fit();
        assert!(!est.is_fitted());
        assert!((est.slope()).abs() < 1e-15);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut est = DriftEstimator::<8>::new(1000);
        est.add_spike(500, 100.0);
        est.add_spike(1500, 110.0);
        est.fit();
        assert!(est.is_fitted());

        est.reset();
        assert!(!est.is_fitted());
        assert!((est.slope()).abs() < 1e-15);
        assert!((est.intercept()).abs() < 1e-15);
        assert_eq!(est.n_bins_used(), 0);
    }

    #[test]
    fn test_estimate_drift_before_fit() {
        let est = DriftEstimator::<8>::new(1000);
        assert!((est.estimate_drift(5000)).abs() < 1e-15);
    }

    #[test]
    fn test_batch_function() {
        let samples = [100, 1100, 2100, 3100];
        let positions = [50.0, 60.0, 70.0, 80.0];
        let result = estimate_drift_from_positions(&samples, &positions, 1000, 8);
        assert!(result.is_some());
        let (slope, _intercept) = result.unwrap();
        // 10um per 1000 samples = 0.01
        assert!(
            libm::fabs(slope - 0.01) < 0.002,
            "Slope should be ~0.01, got {}",
            slope
        );
    }

    #[test]
    fn test_empty_input_returns_none() {
        let result = estimate_drift_from_positions(&[], &[], 1000, 8);
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_single_bin_returns_none() {
        let samples = [100, 200, 300];
        let positions = [50.0, 55.0, 60.0];
        // All in bin 0, so only 1 bin -> None
        let result = estimate_drift_from_positions(&samples, &positions, 1000, 8);
        assert!(result.is_none());
    }

    #[test]
    fn test_n_bins_used() {
        let mut est = DriftEstimator::<8>::new(1000);
        est.add_spike(500, 100.0);
        est.add_spike(2500, 110.0);
        // bins 0 and 2 have data, bin 1 is empty
        assert_eq!(est.n_bins_used(), 2);
    }

    #[test]
    fn test_overflow_bin_ignored() {
        let mut est = DriftEstimator::<4>::new(1000);
        // Bin 5 exceeds MAX_BINS=4, should be ignored
        est.add_spike(5500, 100.0);
        assert_eq!(est.n_bins_used(), 0);
    }
}
