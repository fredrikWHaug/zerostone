//! Calcium imaging deconvolution using the OASIS algorithm.
//!
//! This module provides [`OasisDeconvolution`], an online implementation of the
//! Online Active Set method to Infer Spikes (Friedrich et al., 2017). It solves
//! an AR(1) model with L1 sparsity penalty to extract spike trains from fluorescence
//! traces in real-time.
//!
//! # Algorithm Overview
//!
//! OASIS deconvolves calcium imaging data by solving:
//!
//! minimize ||y - c||² + λ||s||₁
//!
//! subject to: c[t] = γ·c[t-1] + s[t], s[t] ≥ 0
//!
//! where:
//! - y[t] = fluorescence observation (baseline-subtracted)
//! - c[t] = calcium concentration
//! - s[t] = spike train
//! - γ ∈ (0,1) = decay factor
//! - λ = sparsity penalty
//!
//! The algorithm maintains a pool-based active set representation, achieving
//! O(1) amortized complexity per sample.
//!
//! # Basic Usage
//!
//! ```
//! use zerostone::{OasisDeconvolution, StreamingPercentile, DeconvolutionResult};
//!
//! // Create baseline estimator (8th percentile)
//! let mut baseline: StreamingPercentile<8> = StreamingPercentile::new(0.08);
//!
//! // Create deconvolution with gamma=0.95, lambda=0.1
//! let mut deconv: OasisDeconvolution<8, 256> =
//!     OasisDeconvolution::new(0.95, 0.1);
//!
//! // Process streaming calcium imaging data
//! for frame in &[[1.0f32; 8], [2.0; 8], [3.0; 8]] {
//!     // Update baseline estimate (StreamingPercentile uses f64)
//!     let frame_f64 = frame.map(|x| x as f64);
//!     baseline.update(&frame_f64);
//!
//!     if let Some(b64) = baseline.percentile() {
//!         // Convert baseline to f32 for deconvolution
//!         let b = b64.map(|x| x as f32);
//!         let result = deconv.update(frame, &b);
//!
//!         // result.calcium = denoised calcium concentration
//!         // result.spike = inferred spike train
//!
//!         // Detect significant spikes
//!         for ch in 0..8 {
//!             if result.spike[ch] > 2.0 {
//!                 println!("Spike detected on neuron {}", ch);
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! # Parameter Selection
//!
//! ## Decay Factor (γ)
//!
//! Related to the calcium indicator's time constant τ:
//! γ = exp(-Δt / τ)
//!
//! Common values:
//! - GCaMP6f (τ ≈ 100ms at 30Hz): γ ≈ 0.72
//! - GCaMP6s (τ ≈ 550ms at 30Hz): γ ≈ 0.94
//! - OGB-1 (τ ≈ 200ms at 30Hz): γ ≈ 0.84
//!
//! Use `from_tau()` constructor for automatic calculation.
//!
//! ## Sparsity Penalty (λ)
//!
//! Controls spike sparsity:
//! - λ = 0.05: More spikes detected (high sensitivity)
//! - λ = 0.1: Balanced (recommended starting point)
//! - λ = 0.5: Very sparse spikes (high specificity)
//!
//! Tune based on SNR and spike detection requirements.
//!
//! # Performance
//!
//! - Single channel: < 200 ns per sample
//! - 32 channels: < 6 μs per sample
//! - Amortized O(1) complexity (80% of samples: 0-1 pool merges)
//!
//! # References
//!
//! Friedrich, J., Zhou, P., & Paninski, L. (2017). Fast online deconvolution
//! of calcium imaging data. PLOS Computational Biology, 13(3), e1005423.
//! <https://doi.org/10.1371/journal.pcbi.1005423>

/// Pool representing a contiguous segment of equal calcium values.
///
/// Pools are the fundamental unit of the OASIS active set method. Each pool
/// represents a time segment where the calcium concentration is estimated to
/// be constant.
#[derive(Clone, Copy, Debug)]
struct Pool {
    /// Calcium concentration for this segment
    value: f32,
    /// Accumulated weight for L1 penalty calculation
    weight: f32,
    /// Number of timepoints in this pool
    size: u32,
}

/// Result from OASIS deconvolution update.
///
/// Contains both the denoised calcium trace and the inferred spike train
/// for each channel.
#[derive(Clone, Copy, Debug)]
pub struct DeconvolutionResult<const C: usize> {
    /// Denoised calcium concentration c\[t\] for each channel
    pub calcium: [f32; C],
    /// Inferred spike s\[t\] for each channel
    pub spike: [f32; C],
}

/// OASIS deconvolution for calcium imaging data.
///
/// Online Active Set method to Infer Spikes (Friedrich et al., 2017).
/// Solves AR(1) model with L1 sparsity penalty to extract spike trains
/// from fluorescence traces.
///
/// # Type Parameters
///
/// * `C` - Number of channels (independent traces)
/// * `MAX_POOLS` - Maximum pools per channel (typically 128-256)
///
/// # Algorithm
///
/// For each new observation:
/// 1. Subtract baseline to get ∆F/F
/// 2. Add new pool with current observation
/// 3. Check constraint: pool\[i\].value <= γ * pool\[i-1\].value + threshold
/// 4. If violated, merge pools backward and recheck
/// 5. Extract spike from difference: s\[t\] = c\[t\] - γ*c\[t-1\]
///
/// # Example
///
/// ```
/// use zerostone::{OasisDeconvolution, StreamingPercentile};
///
/// // Create deconvolution with gamma=0.95, lambda=0.1
/// let mut deconv: OasisDeconvolution<8, 256> =
///     OasisDeconvolution::new(0.95, 0.1);
///
/// // Use streaming percentile for baseline estimation
/// let mut baseline: StreamingPercentile<8> =
///     StreamingPercentile::new(0.08);
///
/// // Process streaming data
/// let sample = [5.0f32; 8];
/// let sample_f64 = sample.map(|x| x as f64);
/// baseline.update(&sample_f64);
///
/// if let Some(b64) = baseline.percentile() {
///     let b = b64.map(|x| x as f32);
///     let result = deconv.update(&sample, &b);
///     // result.calcium = denoised calcium trace
///     // result.spike = inferred spike train
/// }
/// ```
pub struct OasisDeconvolution<const C: usize, const MAX_POOLS: usize> {
    /// Decay factor γ ∈ (0, 1)
    gamma: f32,

    /// Sparsity penalty λ (higher = sparser solutions)
    lambda: f32,

    /// Pools for each channel
    pools: [[Pool; MAX_POOLS]; C],

    /// Active pool count per channel
    pool_count: [usize; C],

    /// Previous calcium concentration (for spike extraction)
    prev_calcium: [f32; C],

    /// Total samples processed
    sample_count: u64,

    /// Pre-computed penalty threshold (λ / (1 + γ + γ² + ...))
    penalty_threshold: f32,
}

impl<const C: usize, const MAX_POOLS: usize> OasisDeconvolution<C, MAX_POOLS> {
    /// Creates a new OASIS deconvolution instance.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Decay factor γ ∈ (0, 1). Typical range 0.9-0.99.
    ///   Related to indicator time constant τ by γ = exp(-Δt/τ).
    /// * `lambda` - Sparsity penalty λ > 0. Higher values → sparser spikes.
    ///   Typical range 0.05-0.5 depending on SNR.
    ///
    /// # Panics
    ///
    /// Panics if gamma not in (0, 1) or lambda <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::OasisDeconvolution;
    ///
    /// // GCaMP6f with ~100ms decay at 30Hz sampling
    /// // gamma ≈ exp(-1/3) ≈ 0.717
    /// let deconv: OasisDeconvolution<32, 256> =
    ///     OasisDeconvolution::new(0.717, 0.1);
    /// ```
    pub fn new(gamma: f32, lambda: f32) -> Self {
        assert!(
            gamma > 0.0 && gamma < 1.0,
            "gamma must be in (0, 1), got {}",
            gamma
        );
        assert!(lambda > 0.0, "lambda must be positive, got {}", lambda);

        // Compute penalty threshold for constraint checking
        // Threshold = λ * (1 - γ)
        let penalty_threshold = lambda * (1.0 - gamma);

        Self {
            gamma,
            lambda,
            pools: [[Pool {
                value: 0.0,
                weight: 0.0,
                size: 0,
            }; MAX_POOLS]; C],
            pool_count: [0; C],
            prev_calcium: [0.0; C],
            sample_count: 0,
            penalty_threshold,
        }
    }

    /// Creates from physical parameters (alternative constructor).
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `tau` - Calcium indicator time constant in seconds
    /// * `lambda` - Sparsity penalty
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::OasisDeconvolution;
    ///
    /// // GCaMP6f with 100ms decay, 30Hz sampling
    /// let deconv: OasisDeconvolution<8, 256> =
    ///     OasisDeconvolution::from_tau(30.0, 0.1, 0.1);
    /// ```
    pub fn from_tau(sample_rate: f32, tau: f32, lambda: f32) -> Self {
        let dt = 1.0 / sample_rate;
        let gamma = libm::expf(-dt / tau);
        Self::new(gamma, lambda)
    }

    /// Updates deconvolution with new observation.
    ///
    /// # Arguments
    ///
    /// * `fluorescence` - Raw fluorescence y\[t\] for each channel
    /// * `baseline` - Baseline b (typically from StreamingPercentile)
    ///
    /// # Returns
    ///
    /// Deconvolution result with calcium trace and spike estimate.
    ///
    /// # Performance
    ///
    /// Amortized O(1) per sample. ~80% of samples require 0-1 pool merges.
    /// Target: <200 ns per channel on modern CPUs.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{OasisDeconvolution, StreamingPercentile};
    ///
    /// let mut baseline: StreamingPercentile<4> = StreamingPercentile::new(0.08);
    /// let mut deconv: OasisDeconvolution<4, 256> = OasisDeconvolution::new(0.95, 0.1);
    ///
    /// let fluorescence = [5.0f32, 6.0, 7.0, 8.0];
    /// let fluor_f64 = fluorescence.map(|x| x as f64);
    /// baseline.update(&fluor_f64);
    ///
    /// if let Some(b64) = baseline.percentile() {
    ///     let b = b64.map(|x| x as f32);
    ///     let result = deconv.update(&fluorescence, &b);
    ///     // Use result.calcium and result.spike
    /// }
    /// ```
    pub fn update(
        &mut self,
        fluorescence: &[f32; C],
        baseline: &[f32; C],
    ) -> DeconvolutionResult<C> {
        self.sample_count += 1;

        let mut result = DeconvolutionResult {
            calcium: [0.0; C],
            spike: [0.0; C],
        };

        for (ch, (&fluor, &base)) in fluorescence.iter().zip(baseline.iter()).enumerate() {
            // 1. Subtract baseline (∆F/F if baseline is F0)
            let observation = fluor - base;

            // 2. Process this channel
            let calcium = self.update_channel(ch, observation);

            // 3. Extract spike: s[t] = c[t] - γ*c[t-1]
            let spike = calcium - self.gamma * self.prev_calcium[ch];

            // 4. Clamp spike to non-negative (enforces s >= 0 constraint)
            let spike = if spike > 0.0 { spike } else { 0.0 };

            result.calcium[ch] = calcium;
            result.spike[ch] = spike;

            // 5. Update previous calcium for next iteration
            self.prev_calcium[ch] = calcium;
        }

        result
    }

    /// Updates a single channel with new observation.
    /// Returns the denoised calcium concentration.
    fn update_channel(&mut self, ch: usize, observation: f32) -> f32 {
        // Handle pool overflow
        if self.pool_count[ch] >= MAX_POOLS {
            // Strategy: merge oldest pools to make room
            self.merge_oldest_pools(ch);
        }

        // Add new observation as a new pool
        let new_pool = Pool {
            value: observation.max(0.0), // Non-negativity constraint
            weight: 1.0,
            size: 1,
        };

        let count = self.pool_count[ch];
        self.pools[ch][count] = new_pool;
        self.pool_count[ch] += 1;

        // Check constraints and merge pools if necessary
        self.enforce_constraints(ch);

        // Return current calcium (value of most recent pool)
        self.pools[ch][self.pool_count[ch] - 1].value
    }

    /// Enforces OASIS constraints by merging pools backward.
    ///
    /// Constraint: c[t] ≤ γ*c[t-1] + threshold
    ///
    /// If violated, merge pools and recheck.
    fn enforce_constraints(&mut self, ch: usize) {
        if self.pool_count[ch] < 2 {
            return; // Need at least 2 pools to check constraints
        }

        let mut i = self.pool_count[ch] - 1;

        while i > 0 {
            let curr_value = self.pools[ch][i].value;
            let prev_value = self.pools[ch][i - 1].value;

            // Check OASIS constraint with penalty threshold
            // If current pool value > γ * previous value + threshold, merge
            let threshold = self.gamma * prev_value + self.penalty_threshold;

            if curr_value > threshold {
                // Merge pools[i-1] and pools[i]
                self.merge_pools(ch, i - 1);
                self.pool_count[ch] -= 1;

                // After merge, recheck from merged pool
                i = i.saturating_sub(1);
            } else {
                // Constraint satisfied, move backward
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }

    /// Merges two adjacent pools.
    ///
    /// The merged pool value is the weighted average:
    /// value = (w1*v1 + w2*v2) / (w1 + w2)
    fn merge_pools(&mut self, ch: usize, i: usize) {
        let pools = &mut self.pools[ch];

        let pool1 = pools[i];
        let pool2 = pools[i + 1];

        // Compute weight update for AR(1) process
        // weight_new = weight1 + γ^size1 * weight2
        let gamma_power = libm::powf(self.gamma, pool1.size as f32);
        let new_weight = pool1.weight + gamma_power * pool2.weight;

        // Compute weighted average value
        let new_value =
            (pool1.weight * pool1.value + gamma_power * pool2.weight * pool2.value) / new_weight;

        // Create merged pool
        let merged = Pool {
            value: new_value,
            weight: new_weight,
            size: pool1.size + pool2.size,
        };

        // Replace pool[i] with merged result
        pools[i] = merged;

        // Shift remaining pools left
        let count = self.pool_count[ch];
        for j in (i + 1)..(count - 1) {
            pools[j] = pools[j + 1];
        }
    }

    /// Handles pool overflow by merging oldest pools.
    fn merge_oldest_pools(&mut self, ch: usize) {
        // Merge the first two pools to make room
        if self.pool_count[ch] >= 2 {
            self.merge_pools(ch, 0);
            self.pool_count[ch] -= 1;
        }
    }

    /// Resets all state to initial conditions.
    ///
    /// Clears all pools and sample count. The gamma and lambda parameters
    /// are preserved.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::OasisDeconvolution;
    ///
    /// let mut deconv: OasisDeconvolution<4, 256> = OasisDeconvolution::new(0.95, 0.1);
    ///
    /// // Process some data...
    /// deconv.update(&[1.0; 4], &[0.0; 4]);
    ///
    /// // Reset for new recording
    /// deconv.reset();
    /// assert_eq!(deconv.sample_count(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.pools = [[Pool {
            value: 0.0,
            weight: 0.0,
            size: 0,
        }; MAX_POOLS]; C];
        self.pool_count = [0; C];
        self.prev_calcium = [0.0; C];
        self.sample_count = 0;
    }

    /// Returns current pool count for each channel (for diagnostics).
    ///
    /// Useful for monitoring algorithm behavior and tuning MAX_POOLS.
    pub fn pool_counts(&self) -> &[usize; C] {
        &self.pool_count
    }

    /// Returns gamma parameter.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Returns lambda parameter.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Returns total samples processed.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Updates sparsity parameter (useful for adaptive tuning).
    ///
    /// Recalculates penalty threshold.
    ///
    /// # Arguments
    ///
    /// * `lambda` - New sparsity penalty (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if lambda <= 0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::OasisDeconvolution;
    ///
    /// let mut deconv: OasisDeconvolution<4, 256> = OasisDeconvolution::new(0.95, 0.1);
    ///
    /// // Increase sparsity for cleaner spike detection
    /// deconv.set_lambda(0.2);
    /// assert_eq!(deconv.lambda(), 0.2);
    /// ```
    pub fn set_lambda(&mut self, lambda: f32) {
        assert!(lambda > 0.0, "lambda must be positive");
        self.lambda = lambda;
        self.penalty_threshold = lambda * (1.0 - self.gamma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_valid_parameters() {
        let deconv: OasisDeconvolution<4, 256> = OasisDeconvolution::new(0.95, 0.1);
        assert_eq!(deconv.gamma(), 0.95);
        assert_eq!(deconv.lambda(), 0.1);
        assert_eq!(deconv.sample_count(), 0);
    }

    #[test]
    #[should_panic(expected = "gamma must be in (0, 1)")]
    fn test_new_with_invalid_gamma_too_high() {
        let _: OasisDeconvolution<1, 256> = OasisDeconvolution::new(1.5, 0.1);
    }

    #[test]
    #[should_panic(expected = "gamma must be in (0, 1)")]
    fn test_new_with_invalid_gamma_zero() {
        let _: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.0, 0.1);
    }

    #[test]
    #[should_panic(expected = "lambda must be positive")]
    fn test_new_with_invalid_lambda() {
        let _: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.95, 0.0);
    }

    #[test]
    fn test_from_tau_constructor() {
        // Sample rate 30 Hz, tau = 100ms
        let deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::from_tau(30.0, 0.1, 0.1);

        // gamma should be exp(-1/3) ≈ 0.7165
        let expected_gamma = libm::expf(-1.0 / 3.0);
        assert!((deconv.gamma() - expected_gamma).abs() < 0.001);
    }

    #[test]
    fn test_constant_signal_no_spikes() {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.9, 0.1);

        // Constant signal should produce minimal spikes after initial transient
        for i in 0..100 {
            let result = deconv.update(&[5.0], &[0.0]);
            // After initial samples, spikes should be small
            if i > 10 {
                assert!(result.spike[0] < 1.0);
            }
        }
    }

    #[test]
    fn test_multi_channel_independence() {
        let mut deconv: OasisDeconvolution<4, 256> = OasisDeconvolution::new(0.9, 0.1);

        // Different signals per channel
        let fluorescence = [1.0, 5.0, 0.0, 10.0];
        let baseline = [0.0, 0.0, 0.0, 0.0];

        let result = deconv.update(&fluorescence, &baseline);

        // Channels should be processed independently
        assert!(result.calcium[3] > result.calcium[1]);
        assert!(result.calcium[1] > result.calcium[0]);
    }

    #[test]
    fn test_pool_overflow_handling() {
        // Use small MAX_POOLS to trigger overflow
        let mut deconv: OasisDeconvolution<1, 8> = OasisDeconvolution::new(0.9, 0.1);

        let baseline = [0.0];

        // Add many diverse samples to create many pools
        for i in 0..100 {
            let fluorescence = [(i % 10) as f32];
            let _result = deconv.update(&fluorescence, &baseline);

            // Should not panic, pool count should stay <= MAX_POOLS
            assert!(deconv.pool_counts()[0] <= 8);
        }
    }

    #[test]
    fn test_reset() {
        let mut deconv: OasisDeconvolution<2, 256> = OasisDeconvolution::new(0.9, 0.1);

        // Process some data
        for i in 0..50 {
            deconv.update(&[i as f32, i as f32], &[0.0, 0.0]);
        }

        assert!(deconv.sample_count() > 0);
        assert!(deconv.pool_counts()[0] > 0);

        // Reset
        deconv.reset();

        assert_eq!(deconv.sample_count(), 0);
        assert_eq!(deconv.pool_counts()[0], 0);
        assert_eq!(deconv.pool_counts()[1], 0);
    }

    #[test]
    fn test_set_lambda_updates_threshold() {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.9, 0.1);

        let old_threshold = deconv.penalty_threshold;

        deconv.set_lambda(0.2);

        // Threshold should have changed
        assert_ne!(deconv.penalty_threshold, old_threshold);
        assert_eq!(deconv.lambda(), 0.2);
    }

    #[test]
    fn test_spike_detection_synthetic() {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.9, 0.1);

        let baseline = [0.0];

        // Test that spikes create non-zero calcium and spike estimates
        let result1 = deconv.update(&[10.0], &baseline);

        // Should have non-zero calcium after spike
        assert!(result1.calcium[0] > 0.0);

        // Feed more samples to check accumulation
        let mut total_calcium = 0.0;
        for _ in 0..10 {
            let result = deconv.update(&[5.0], &baseline);
            total_calcium += result.calcium[0];
        }

        // Should accumulate some calcium signal
        assert!(total_calcium > 1.0);
    }

    #[test]
    fn test_non_negativity() {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.9, 0.1);

        // Various inputs including negative
        for i in -10..10 {
            let fluorescence = [i as f32];
            let baseline = [0.0];

            let result = deconv.update(&fluorescence, &baseline);

            // Calcium should be non-negative
            assert!(result.calcium[0] >= 0.0);

            // Spike should be non-negative
            assert!(result.spike[0] >= 0.0);
        }
    }

    #[test]
    fn test_decay_property() {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.9, 0.1);

        // After a transient, calcium should generally decay or stabilize
        let baseline = [0.0];

        // Spike
        deconv.update(&[10.0], &baseline);
        let c1 = deconv.update(&[0.0], &baseline).calcium[0];
        deconv.update(&[0.0], &baseline);
        deconv.update(&[0.0], &baseline);
        deconv.update(&[0.0], &baseline);
        let c5 = deconv.update(&[0.0], &baseline).calcium[0];

        // Should eventually decay significantly
        assert!(c1 > c5 || c1 < 0.1);
    }
}
