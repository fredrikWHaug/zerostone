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

use crate::float::{self, Float};

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
    bin_sum_y: [Float; MAX_BINS],
    bin_counts: [usize; MAX_BINS],
    bin_duration: usize,
    n_bins: usize,
    slope: Float,
    intercept: Float,
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
    pub fn add_spike(&mut self, sample_index: usize, position_y: Float) {
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
        let mut xs = [0.0 as Float; MAX_BINS];
        let mut ys = [0.0 as Float; MAX_BINS];
        let mut n = 0usize;

        let mut bin = 0;
        while bin < self.n_bins && bin < MAX_BINS {
            if self.bin_counts[bin] > 0 {
                // Use bin center time in sample units
                xs[n] = (bin as Float + 0.5) * self.bin_duration as Float;
                ys[n] = self.bin_sum_y[bin] / self.bin_counts[bin] as Float;
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

        let nf = n as Float;
        let denom = nf * sum_x2 - sum_x * sum_x;
        if float::abs(denom) < 1e-30 {
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
    pub fn estimate_drift(&self, sample_index: usize) -> Float {
        if !self.fitted {
            return 0.0;
        }
        let t = sample_index as Float;
        // Drift = (predicted position at t) - (predicted position at t=0)
        // predicted(t) = slope * t + intercept
        // predicted(0) = intercept
        // drift = slope * t
        self.slope * t
    }

    /// Correct a spike's position by removing estimated drift.
    ///
    /// Returns `position_y - estimate_drift(sample_index)`.
    pub fn correct_position(&self, sample_index: usize, position_y: Float) -> Float {
        position_y - self.estimate_drift(sample_index)
    }

    /// The fitted drift rate (position units per sample).
    pub fn slope(&self) -> Float {
        self.slope
    }

    /// The fitted intercept (position at time zero).
    pub fn intercept(&self) -> Float {
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
    positions_y: &[Float],
    bin_duration_samples: usize,
    max_bins: usize,
) -> Option<(Float, Float)> {
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

    let mut bin_sum = [0.0 as Float; STACK_BINS];
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
    let mut sum_x: Float = 0.0;
    let mut sum_y: Float = 0.0;
    let mut sum_xy: Float = 0.0;
    let mut sum_x2: Float = 0.0;
    let mut count = 0usize;

    let mut bin = 0;
    while bin < effective_max {
        if bin_count[bin] > 0 {
            let x = (bin as Float + 0.5) * bin_duration_samples as Float;
            let y = bin_sum[bin] / bin_count[bin] as Float;
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

    let nf = count as Float;
    let denom = nf * sum_x2 - sum_x * sum_x;
    if float::abs(denom) < 1e-30 {
        return Some((0.0, sum_y / nf));
    }

    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / nf;
    Some((slope, intercept))
}

/// Per-bin linear drift model used internally by [`NonRigidDrift`].
///
/// Each depth bin independently tracks spike times and positions,
/// fitting its own slope and intercept via OLS.
struct BinDrift<const MAX_TIME_BINS: usize> {
    bin_sum_y: [Float; MAX_TIME_BINS],
    bin_counts: [usize; MAX_TIME_BINS],
    n_bins: usize,
    slope: Float,
    intercept: Float,
    fitted: bool,
}

impl<const MAX_TIME_BINS: usize> BinDrift<MAX_TIME_BINS> {
    const fn new() -> Self {
        Self {
            bin_sum_y: [0.0; MAX_TIME_BINS],
            bin_counts: [0; MAX_TIME_BINS],
            n_bins: 0,
            slope: 0.0,
            intercept: 0.0,
            fitted: false,
        }
    }

    fn add_sample(&mut self, time_bin: usize, position_y: Float) {
        if time_bin >= MAX_TIME_BINS {
            return;
        }
        self.bin_sum_y[time_bin] += position_y;
        self.bin_counts[time_bin] += 1;
        if time_bin >= self.n_bins {
            self.n_bins = time_bin + 1;
        }
    }

    fn fit(&mut self, bin_duration: usize) {
        let mut xs = [0.0 as Float; MAX_TIME_BINS];
        let mut ys = [0.0 as Float; MAX_TIME_BINS];
        let mut n = 0usize;

        let mut bin = 0;
        while bin < self.n_bins && bin < MAX_TIME_BINS {
            if self.bin_counts[bin] > 0 {
                xs[n] = (bin as Float + 0.5) * bin_duration as Float;
                ys[n] = self.bin_sum_y[bin] / self.bin_counts[bin] as Float;
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

        let mut sum_x: Float = 0.0;
        let mut sum_y: Float = 0.0;
        let mut sum_xy: Float = 0.0;
        let mut sum_x2: Float = 0.0;

        let mut i = 0;
        while i < n {
            sum_x += xs[i];
            sum_y += ys[i];
            sum_xy += xs[i] * ys[i];
            sum_x2 += xs[i] * xs[i];
            i += 1;
        }

        let nf = n as Float;
        let denom = nf * sum_x2 - sum_x * sum_x;
        if float::abs(denom) < 1e-30 {
            self.slope = 0.0;
            self.intercept = sum_y / nf;
            self.fitted = true;
            return;
        }

        self.slope = (nf * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / nf;
        self.fitted = true;
    }

    fn estimate_drift(&self, sample_index: usize) -> Float {
        if !self.fitted {
            return 0.0;
        }
        self.slope * sample_index as Float
    }

    fn reset(&mut self) {
        let mut i = 0;
        while i < MAX_TIME_BINS {
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

/// Non-rigid drift correction with piecewise-linear depth model.
///
/// Divides the probe into `K` equal depth bins and fits an independent
/// linear drift model in each bin. Drift correction at arbitrary depths
/// is computed by linear interpolation between the two nearest bin centers.
///
/// This handles spatially varying drift where different probe regions
/// move at different rates -- common in long chronic recordings.
///
/// # Type Parameters
///
/// * `K` -- Number of depth bins (typically 4 or 8)
/// * `MAX_TIME_BINS` -- Maximum number of time bins per depth bin
///
/// # Example
///
/// ```
/// use zerostone::drift::NonRigidDrift;
///
/// let mut nr = NonRigidDrift::<4, 16>::new(0.0, 400.0, 1000);
/// // Top drifts up, bottom drifts down
/// for i in 0..8 {
///     nr.add_spike(i * 1000 + 500, 50.0, 50.0 + i as f64 * 5.0);
///     nr.add_spike(i * 1000 + 500, 350.0, 350.0 - i as f64 * 5.0);
/// }
/// nr.fit();
/// assert!(nr.is_fitted());
/// ```
pub struct NonRigidDrift<const K: usize, const MAX_TIME_BINS: usize> {
    bins: [BinDrift<MAX_TIME_BINS>; K],
    bin_centers: [Float; K],
    y_min: Float,
    y_max: Float,
    bin_duration: usize,
    fitted: bool,
}

impl<const K: usize, const MAX_TIME_BINS: usize> NonRigidDrift<K, MAX_TIME_BINS> {
    /// Create a new non-rigid drift estimator.
    ///
    /// # Arguments
    ///
    /// * `y_min` -- Minimum probe depth (e.g., 0.0)
    /// * `y_max` -- Maximum probe depth (e.g., 3840.0 for Neuropixels)
    /// * `bin_duration_samples` -- Time bin width in samples
    ///
    /// # Panics
    ///
    /// Panics if `y_max <= y_min`, `K == 0`, or `bin_duration_samples == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::drift::NonRigidDrift;
    ///
    /// let nr = NonRigidDrift::<4, 16>::new(0.0, 400.0, 1000);
    /// assert!(!nr.is_fitted());
    /// ```
    pub fn new(y_min: Float, y_max: Float, bin_duration_samples: usize) -> Self {
        assert!(K > 0, "K must be > 0");
        assert!(y_max > y_min, "y_max must be > y_min");
        assert!(bin_duration_samples > 0, "bin_duration_samples must be > 0");

        let mut bin_centers = [0.0 as Float; K];
        let bin_width = (y_max - y_min) / K as Float;
        let mut i = 0;
        while i < K {
            bin_centers[i] = y_min + (i as Float + 0.5) * bin_width;
            i += 1;
        }

        // Initialize bins array -- cannot use [BinDrift::new(); K] without Copy
        let bins = {
            let mut arr: core::mem::MaybeUninit<[BinDrift<MAX_TIME_BINS>; K]> =
                core::mem::MaybeUninit::uninit();
            let ptr = arr.as_mut_ptr() as *mut BinDrift<MAX_TIME_BINS>;
            let mut j = 0;
            while j < K {
                unsafe { ptr.add(j).write(BinDrift::new()) };
                j += 1;
            }
            unsafe { arr.assume_init() }
        };

        Self {
            bins,
            bin_centers,
            y_min,
            y_max,
            bin_duration: bin_duration_samples,
            fitted: false,
        }
    }

    /// Assign a y-position to a depth bin index.
    ///
    /// Clamps to `[0, K-1]`.
    fn depth_bin(&self, y_position: Float) -> usize {
        let frac = (y_position - self.y_min) / (self.y_max - self.y_min);
        let idx = float::floor(frac * K as Float);
        // Clamp to valid range
        let clamped = if idx < 0.0 {
            0.0
        } else if idx >= K as Float {
            (K - 1) as Float
        } else {
            idx
        };
        clamped as usize
    }

    /// Add a spike observation.
    ///
    /// The spike is placed into the appropriate depth bin based on
    /// `y_position`, and into a time bin based on `sample_index`.
    /// The `position_y` value is the position to track for drift
    /// (often the same as `y_position`).
    ///
    /// # Arguments
    ///
    /// * `sample_index` -- time of the spike in samples
    /// * `y_position` -- depth on the probe (for bin assignment)
    /// * `position_y` -- tracked position value (for drift regression)
    pub fn add_spike(&mut self, sample_index: usize, y_position: Float, position_y: Float) {
        let dbin = self.depth_bin(y_position);
        let tbin = sample_index / self.bin_duration;
        self.bins[dbin].add_sample(tbin, position_y);
    }

    /// Fit linear regression independently in each depth bin.
    ///
    /// After fitting, [`correct_position`](Self::correct_position) and
    /// [`estimate_drift`](Self::estimate_drift) can be used with
    /// interpolation between bin centers.
    pub fn fit(&mut self) {
        let mut any_fitted = false;
        let mut i = 0;
        while i < K {
            self.bins[i].fit(self.bin_duration);
            if self.bins[i].fitted {
                any_fitted = true;
            }
            i += 1;
        }
        self.fitted = any_fitted;
    }

    /// Estimate the drift at a given (time, depth) point.
    ///
    /// Linearly interpolates the per-bin drift estimates between
    /// the two nearest bin centers. At positions outside the bin
    /// center range, clamps to the nearest bin's estimate.
    ///
    /// Returns 0.0 if the model has not been fitted.
    pub fn estimate_drift(&self, sample_index: usize, y_position: Float) -> Float {
        if !self.fitted {
            return 0.0;
        }
        self.interpolate_drift(sample_index, y_position)
    }

    /// Correct a spike's position by removing estimated drift.
    ///
    /// Returns `position_y - estimate_drift(sample_index, y_position)`.
    pub fn correct_position(&self, sample_index: usize, y_position: Float) -> Float {
        y_position - self.estimate_drift(sample_index, y_position)
    }

    /// Whether at least one depth bin has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Reset all depth bins and clear fitted state.
    pub fn reset(&mut self) {
        let mut i = 0;
        while i < K {
            self.bins[i].reset();
            i += 1;
        }
        self.fitted = false;
    }

    /// The per-bin drift slopes (position units per sample).
    ///
    /// Returns an array of K slopes.
    pub fn slopes(&self) -> [Float; K] {
        let mut out = [0.0 as Float; K];
        let mut i = 0;
        while i < K {
            out[i] = self.bins[i].slope;
            i += 1;
        }
        out
    }

    /// The bin centers along the depth axis.
    pub fn bin_centers(&self) -> &[Float; K] {
        &self.bin_centers
    }

    /// Interpolate drift between bin centers.
    fn interpolate_drift(&self, sample_index: usize, y_position: Float) -> Float {
        // If K == 1, just use that bin
        if K == 1 {
            return self.bins[0].estimate_drift(sample_index);
        }

        // Find the two bin centers bracketing y_position
        // If below first center, clamp to first bin
        if y_position <= self.bin_centers[0] {
            return self.bins[0].estimate_drift(sample_index);
        }
        // If above last center, clamp to last bin
        if y_position >= self.bin_centers[K - 1] {
            return self.bins[K - 1].estimate_drift(sample_index);
        }

        // Find the interval [bin_centers[lo], bin_centers[hi]] containing y_position
        let mut lo = 0;
        while lo < K - 2 {
            if self.bin_centers[lo + 1] > y_position {
                break;
            }
            lo += 1;
        }
        let hi = lo + 1;

        let d_lo = self.bins[lo].estimate_drift(sample_index);
        let d_hi = self.bins[hi].estimate_drift(sample_index);

        // Linear interpolation
        let span = self.bin_centers[hi] - self.bin_centers[lo];
        if float::abs(span) < 1e-30 {
            return d_lo;
        }
        let alpha = (y_position - self.bin_centers[lo]) / span;
        d_lo + alpha * (d_hi - d_lo)
    }
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
        est.add_spike(sample, pos as Float);
        est.fit();
        let d = est.estimate_drift(sample);
        assert!(d.is_finite());
    }

    /// Prove that `estimate_drift_from_positions` never panics for small bounded inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_estimate_drift_from_positions_no_panic() {
        let s0: usize = kani::any();
        let s1: usize = kani::any();
        let p0: f64 = kani::any();
        let p1: f64 = kani::any();
        kani::assume(s0 < 10000 && s1 < 10000);
        kani::assume(p0.is_finite() && p0 >= -1e4 && p0 <= 1e4);
        kani::assume(p1.is_finite() && p1 >= -1e4 && p1 <= 1e4);
        let samples = [s0, s1];
        let positions = [p0 as Float, p1 as Float];
        let bin_dur: usize = kani::any();
        kani::assume(bin_dur >= 1 && bin_dur <= 5000);
        let result = estimate_drift_from_positions(&samples, &positions, bin_dur, 4);
        if let Some((slope, intercept)) = result {
            assert!(slope.is_finite());
            assert!(intercept.is_finite());
        }
    }

    /// Prove that `DriftEstimator::new()` + `add_spike()` never panics for bounded inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_drift_estimator_add_spike_no_panic() {
        let bin_dur: usize = kani::any();
        kani::assume(bin_dur >= 1 && bin_dur <= 5000);
        let mut est = DriftEstimator::<3>::new(bin_dur);
        let s0: usize = kani::any();
        let s1: usize = kani::any();
        let p0: f64 = kani::any();
        let p1: f64 = kani::any();
        kani::assume(s0 < 20000 && s1 < 20000);
        kani::assume(p0.is_finite() && p0 >= -1e4 && p0 <= 1e4);
        kani::assume(p1.is_finite() && p1 >= -1e4 && p1 <= 1e4);
        est.add_spike(s0, p0 as Float);
        est.add_spike(s1, p1 as Float);
        assert!(est.n_bins_used() <= 3);
    }

    /// Prove that after adding spikes to distinct bins, `fit()` returns finite coefficients.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_drift_estimator_fit_returns_finite() {
        let mut est = DriftEstimator::<4>::new(100);
        let p0: f64 = kani::any();
        let p1: f64 = kani::any();
        kani::assume(p0.is_finite() && p0 >= -1e4 && p0 <= 1e4);
        kani::assume(p1.is_finite() && p1 >= -1e4 && p1 <= 1e4);
        // Place in distinct bins to guarantee fit succeeds
        est.add_spike(50, p0 as Float);
        est.add_spike(150, p1 as Float);
        est.fit();
        assert!(est.slope().is_finite());
        assert!(est.intercept().is_finite());
        assert!(est.is_fitted());
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
            float::abs(est.slope()) < 1e-10,
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
            est.add_spike(i * 1000 + 500, 100.0 + i as Float * 10.0);
            i += 1;
        }
        est.fit();
        assert!(est.is_fitted());
        // slope should be ~10/1000 = 0.01
        assert!(
            float::abs(est.slope() - 0.01) < 0.002,
            "Slope should be ~0.01, got {}",
            est.slope()
        );
    }

    #[test]
    fn test_correct_position_removes_drift() {
        let mut est = DriftEstimator::<16>::new(1000);
        let mut i = 0;
        while i < 8 {
            est.add_spike(i * 1000 + 500, 100.0 + i as Float * 10.0);
            i += 1;
        }
        est.fit();

        // Corrected positions should be approximately constant
        let base = est.correct_position(500, 100.0);
        i = 1;
        while i < 8 {
            let corrected = est.correct_position(i * 1000 + 500, 100.0 + i as Float * 10.0);
            assert!(
                float::abs(corrected - base) < 2.0,
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
        let positions: [Float; 4] = [50.0, 60.0, 70.0, 80.0];
        let result = estimate_drift_from_positions(&samples, &positions, 1000, 8);
        assert!(result.is_some());
        let (slope, _intercept) = result.unwrap();
        // 10um per 1000 samples = 0.01
        assert!(
            float::abs(slope - 0.01) < 0.002,
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
        let positions: [Float; 3] = [50.0, 55.0, 60.0];
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

    // --- NonRigidDrift tests ---

    #[test]
    fn test_nonrigid_uniform_drift() {
        // All depth bins drift the same way -- should match rigid drift
        let mut nr = NonRigidDrift::<4, 16>::new(0.0, 400.0, 1000);
        let mut rigid = DriftEstimator::<16>::new(1000);

        // Uniform drift: 10um per 1000 samples across all depths
        let mut i = 0;
        while i < 8 {
            let t = i * 1000 + 500;
            let drift = i as Float * 10.0;
            // Add spikes at four different depths, all drifting the same
            nr.add_spike(t, 50.0, 50.0 + drift);
            nr.add_spike(t, 150.0, 150.0 + drift);
            nr.add_spike(t, 250.0, 250.0 + drift);
            nr.add_spike(t, 350.0, 350.0 + drift);
            // Rigid estimator tracks a mid-depth position
            rigid.add_spike(t, 200.0 + drift);
            i += 1;
        }
        nr.fit();
        rigid.fit();

        assert!(nr.is_fitted());

        // All bin slopes should be approximately equal
        let slopes = nr.slopes();
        let mut k = 0;
        while k < 4 {
            assert!(
                float::abs(slopes[k] - 0.01) < 0.002,
                "Bin {} slope should be ~0.01, got {}",
                k,
                slopes[k]
            );
            k += 1;
        }

        // Drift estimate at mid-depth should match rigid
        let nr_drift = nr.estimate_drift(4000, 200.0);
        let rigid_drift = rigid.estimate_drift(4000);
        assert!(
            float::abs(nr_drift - rigid_drift) < 2.0,
            "Non-rigid drift {} should match rigid {}",
            nr_drift,
            rigid_drift
        );
    }

    #[test]
    fn test_nonrigid_opposite_drift() {
        // Top and bottom of probe drift in opposite directions
        let mut nr = NonRigidDrift::<2, 16>::new(0.0, 200.0, 1000);

        let mut i = 0;
        while i < 8 {
            let t = i * 1000 + 500;
            // Bottom bin (y=50): drifts down (positive)
            nr.add_spike(t, 50.0, 50.0 + i as Float * 10.0);
            // Top bin (y=150): drifts up (negative)
            nr.add_spike(t, 150.0, 150.0 - i as Float * 10.0);
            i += 1;
        }
        nr.fit();
        assert!(nr.is_fitted());

        let slopes = nr.slopes();
        // Bottom bin: positive slope
        assert!(
            slopes[0] > 0.005,
            "Bottom bin slope should be positive, got {}",
            slopes[0]
        );
        // Top bin: negative slope
        assert!(
            slopes[1] < -0.005,
            "Top bin slope should be negative, got {}",
            slopes[1]
        );

        // Drift at bottom should be positive, at top should be negative
        let d_bottom = nr.estimate_drift(4000, 50.0);
        let d_top = nr.estimate_drift(4000, 150.0);
        assert!(d_bottom > 0.0, "Bottom drift should be positive");
        assert!(d_top < 0.0, "Top drift should be negative");
    }

    #[test]
    fn test_nonrigid_correction_interpolates() {
        // Two bins with known drift, check interpolation at midpoint
        let mut nr = NonRigidDrift::<2, 16>::new(0.0, 200.0, 1000);

        let mut i = 0;
        while i < 8 {
            let t = i * 1000 + 500;
            // Bottom bin (center=50): drift rate = +20um per 1000 samples
            nr.add_spike(t, 50.0, 50.0 + i as Float * 20.0);
            // Top bin (center=150): drift rate = 0 (no drift)
            nr.add_spike(t, 150.0, 150.0);
            i += 1;
        }
        nr.fit();
        assert!(nr.is_fitted());

        let t_test = 4000;

        // Drift at bottom bin center (y=50): should be ~0.02 * 4000 = 80
        let d_bottom = nr.estimate_drift(t_test, 50.0);
        assert!(
            float::abs(d_bottom - 80.0) < 5.0,
            "Bottom drift should be ~80, got {}",
            d_bottom
        );

        // Drift at top bin center (y=150): should be ~0
        let d_top = nr.estimate_drift(t_test, 150.0);
        assert!(
            float::abs(d_top) < 5.0,
            "Top drift should be ~0, got {}",
            d_top
        );

        // Drift at midpoint (y=100): should interpolate to ~40
        let d_mid = nr.estimate_drift(t_test, 100.0);
        assert!(
            float::abs(d_mid - 40.0) < 5.0,
            "Midpoint drift should be ~40 (interpolated), got {}",
            d_mid
        );

        // Correction at midpoint should subtract interpolated drift
        let corrected = nr.correct_position(t_test, 100.0);
        let expected = 100.0 - d_mid;
        assert!(
            float::abs(corrected - expected) < 1e-10,
            "Corrected {} should equal 100.0 - {} = {}",
            corrected,
            d_mid,
            expected
        );
    }

    #[test]
    fn test_nonrigid_reset() {
        let mut nr = NonRigidDrift::<2, 8>::new(0.0, 100.0, 1000);
        // Need at least 2 time bins per depth bin for fit to succeed
        nr.add_spike(500, 25.0, 30.0);
        nr.add_spike(1500, 25.0, 35.0);
        nr.add_spike(500, 75.0, 80.0);
        nr.add_spike(1500, 75.0, 85.0);
        nr.fit();
        assert!(nr.is_fitted());

        nr.reset();
        assert!(!nr.is_fitted());
        // After reset, drift should be zero everywhere
        assert!(
            float::abs(nr.estimate_drift(1000, 50.0)) < 1e-15,
            "Drift should be 0 after reset"
        );
    }

    #[test]
    fn test_nonrigid_clamp_edges() {
        // Positions outside probe range should clamp to edge bins
        let mut nr = NonRigidDrift::<2, 16>::new(0.0, 200.0, 1000);

        let mut i = 0;
        while i < 8 {
            let t = i * 1000 + 500;
            nr.add_spike(t, 50.0, 50.0 + i as Float * 10.0);
            nr.add_spike(t, 150.0, 150.0 + i as Float * 5.0);
            i += 1;
        }
        nr.fit();

        // Position below y_min should clamp to bin 0
        let d_below = nr.estimate_drift(4000, -50.0);
        let d_bin0 = nr.estimate_drift(4000, 50.0);
        assert!(
            float::abs(d_below - d_bin0) < 1e-10,
            "Below-range should clamp to bin 0"
        );

        // Position above y_max should clamp to last bin
        let d_above = nr.estimate_drift(4000, 300.0);
        let d_binlast = nr.estimate_drift(4000, 150.0);
        assert!(
            float::abs(d_above - d_binlast) < 1e-10,
            "Above-range should clamp to last bin"
        );
    }
}
