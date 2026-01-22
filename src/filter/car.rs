//! Common Average Reference (CAR) spatial filter for multi-channel neural recordings.
//!
//! The Common Average Reference is a spatial filtering technique that removes common-mode
//! noise by subtracting the mean of all channels from each channel. This is particularly
//! useful for:
//!
//! - Removing reference electrode bias
//! - Reducing global noise sources (e.g., powerline interference, movement artifacts)
//! - Making recordings reference-independent
//!
//! CAR is widely used across electrophysiology modalities including microelectrode arrays
//! (Utah arrays, Neuropixels), electrocorticography (ECoG), stereo-EEG, and scalp EEG.
//!
//! # Mathematical Background
//!
//! ```text
//! V_car[i] = V[i] - (1/C) × Σ(V[j])
//! ```
//!
//! Where:
//! - `V[i]` = voltage at electrode i
//! - `C` = total number of channels
//! - `Σ(V[j])` = sum of all channel voltages
//!
//! # Properties
//!
//! - Output has zero mean across channels (sum of CAR-filtered channels = 0)
//! - Reference-independent (immune to reference electrode choice)
//! - Reduces global/common-mode artifacts
//! - Zero memory footprint (stateless computation)
//!
//! # Example
//!
//! ```
//! use zerostone::CommonAverageReference;
//!
//! let car: CommonAverageReference<4> = CommonAverageReference::new();
//!
//! let samples = [1.0, 2.0, 3.0, 4.0];  // mean = 2.5
//! let filtered = car.process(&samples);
//!
//! // Each channel minus mean: [-1.5, -0.5, 0.5, 1.5]
//! assert!((filtered[0] - (-1.5)).abs() < 1e-6);
//! assert!((filtered[2] - 0.5).abs() < 1e-6);
//!
//! // Output sums to zero
//! let sum: f32 = filtered.iter().sum();
//! assert!(sum.abs() < 1e-6);
//! ```

/// Common Average Reference spatial filter.
///
/// A zero-sized type that computes the common average reference by subtracting
/// the mean of all channels from each channel. Since CAR requires no configuration
/// or state, this struct has zero memory footprint.
///
/// # Type Parameters
///
/// - `C`: Number of channels (must be at least 1)
///
/// # Examples
///
/// ```
/// use zerostone::CommonAverageReference;
///
/// // Create a 4-channel CAR filter
/// let car: CommonAverageReference<4> = CommonAverageReference::new();
///
/// let samples = [1.0, 2.0, 3.0, 4.0];
/// let filtered = car.process(&samples);
///
/// // Mean was 2.5, so each value has 2.5 subtracted
/// assert!((filtered[0] - (-1.5)).abs() < 1e-6);
/// assert!((filtered[1] - (-0.5)).abs() < 1e-6);
/// assert!((filtered[2] - 0.5).abs() < 1e-6);
/// assert!((filtered[3] - 1.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommonAverageReference<const C: usize>;

impl<const C: usize> CommonAverageReference<C> {
    /// Compile-time assertion: C must be at least 1
    const _ASSERT_C: () = assert!(C >= 1, "C (channels) must be at least 1");

    /// Creates a new Common Average Reference filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::CommonAverageReference;
    ///
    /// let car: CommonAverageReference<8> = CommonAverageReference::new();
    /// ```
    pub fn new() -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_C;
        Self
    }

    /// Processes a single multi-channel sample through the CAR filter.
    ///
    /// Computes the mean of all channels and subtracts it from each channel.
    ///
    /// # Arguments
    ///
    /// - `samples`: Array of sample values for all channels
    ///
    /// # Returns
    ///
    /// Filtered sample array with zero mean across channels
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::CommonAverageReference;
    ///
    /// let car: CommonAverageReference<3> = CommonAverageReference::new();
    ///
    /// // Input: [1, 2, 3] with mean = 2
    /// let samples = [1.0, 2.0, 3.0];
    /// let filtered = car.process(&samples);
    ///
    /// // Output: [-1, 0, 1]
    /// assert!((filtered[0] - (-1.0)).abs() < 1e-6);
    /// assert!((filtered[1] - 0.0).abs() < 1e-6);
    /// assert!((filtered[2] - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn process(&self, samples: &[f32; C]) -> [f32; C] {
        // Compute mean
        let mut sum = 0.0f32;
        for &s in samples {
            sum += s;
        }
        let mean = sum / C as f32;

        // Subtract mean from each channel
        let mut output = [0.0; C];
        for (i, &s) in samples.iter().enumerate() {
            output[i] = s - mean;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_mean_property() {
        let car: CommonAverageReference<5> = CommonAverageReference::new();
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = car.process(&samples);

        let sum: f32 = filtered.iter().sum();
        assert!(sum.abs() < 1e-6, "Output should sum to zero, got {}", sum);
    }

    #[test]
    fn test_known_values() {
        let car: CommonAverageReference<3> = CommonAverageReference::new();
        let samples = [1.0, 2.0, 3.0]; // mean = 2
        let filtered = car.process(&samples);

        // Expected: [-1, 0, 1]
        assert!((filtered[0] - (-1.0)).abs() < 1e-6);
        assert!(filtered[1].abs() < 1e-6);
        assert!((filtered[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_constant_input() {
        let car: CommonAverageReference<4> = CommonAverageReference::new();
        let samples = [5.0, 5.0, 5.0, 5.0]; // All same value
        let filtered = car.process(&samples);

        // All outputs should be zero
        for val in &filtered {
            assert!(val.abs() < 1e-6, "Constant input should yield zeros");
        }
    }

    #[test]
    fn test_single_channel() {
        let car: CommonAverageReference<1> = CommonAverageReference::new();
        let samples = [5.0];
        let filtered = car.process(&samples);

        // Single channel: 5.0 - 5.0 = 0.0
        assert!(filtered[0].abs() < 1e-6);
    }

    #[test]
    fn test_negative_values() {
        let car: CommonAverageReference<3> = CommonAverageReference::new();
        let samples = [-3.0, -2.0, -1.0]; // mean = -2
        let filtered = car.process(&samples);

        // Expected: [-1, 0, 1]
        assert!((filtered[0] - (-1.0)).abs() < 1e-6);
        assert!(filtered[1].abs() < 1e-6);
        assert!((filtered[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reference_independence() {
        let car: CommonAverageReference<3> = CommonAverageReference::new();

        let samples1 = [1.0, 2.0, 3.0];
        let samples2 = [101.0, 102.0, 103.0]; // Same pattern, different offset

        let filtered1 = car.process(&samples1);
        let filtered2 = car.process(&samples2);

        // Results should be identical (CAR removes constant offset)
        for ch in 0..3 {
            assert!(
                (filtered1[ch] - filtered2[ch]).abs() < 1e-6,
                "CAR should produce same output regardless of reference"
            );
        }
    }

    #[test]
    fn test_large_channel_count() {
        let car: CommonAverageReference<128> = CommonAverageReference::new();
        let mut samples = [0.0f32; 128];
        samples[64] = 128.0; // Single peak

        let filtered = car.process(&samples);

        // Verify zero-sum property
        let sum: f32 = filtered.iter().sum();
        assert!(sum.abs() < 1e-4, "Sum should be zero, got {}", sum);

        // Peak channel should be 128 - 1 = 127 (mean is 1.0)
        assert!((filtered[64] - 127.0).abs() < 1e-4);

        // Other channels should be -1.0
        assert!((filtered[0] - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_zero_input() {
        let car: CommonAverageReference<4> = CommonAverageReference::new();
        let samples = [0.0, 0.0, 0.0, 0.0];
        let filtered = car.process(&samples);

        for val in &filtered {
            assert!(val.abs() < 1e-6, "Zero input should yield zeros");
        }
    }

    #[test]
    fn test_mixed_signs() {
        let car: CommonAverageReference<4> = CommonAverageReference::new();
        let samples = [-2.0, -1.0, 1.0, 2.0]; // mean = 0
        let filtered = car.process(&samples);

        // With mean 0, output should equal input
        for (i, &val) in filtered.iter().enumerate() {
            assert!(
                (val - samples[i]).abs() < 1e-6,
                "With zero mean, output should equal input"
            );
        }
    }

    #[test]
    fn test_amplitude_preservation() {
        let car: CommonAverageReference<3> = CommonAverageReference::new();
        let samples = [1.0, 3.0, 5.0];
        let filtered = car.process(&samples);

        // Differences between channels should be preserved
        // Original: 3-1=2, 5-3=2
        // After CAR: filtered[1]-filtered[0], filtered[2]-filtered[1]
        let diff1_before = samples[1] - samples[0];
        let diff2_before = samples[2] - samples[1];
        let diff1_after = filtered[1] - filtered[0];
        let diff2_after = filtered[2] - filtered[1];

        assert!(
            (diff1_before - diff1_after).abs() < 1e-6,
            "Inter-channel differences should be preserved"
        );
        assert!(
            (diff2_before - diff2_after).abs() < 1e-6,
            "Inter-channel differences should be preserved"
        );
    }

    #[test]
    fn test_default_trait() {
        let car: CommonAverageReference<4> = Default::default();
        let samples = [1.0, 2.0, 3.0, 4.0];
        let filtered = car.process(&samples);

        // Should work the same as new()
        let sum: f32 = filtered.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn test_copy_trait() {
        let car1: CommonAverageReference<4> = CommonAverageReference::new();
        let car2 = car1; // Copy, not move

        // Both should still work
        let samples = [1.0, 2.0, 3.0, 4.0];
        let _filtered1 = car1.process(&samples);
        let _filtered2 = car2.process(&samples);
    }

    #[test]
    fn test_zero_sized_type() {
        assert_eq!(
            core::mem::size_of::<CommonAverageReference<32>>(),
            0,
            "CommonAverageReference should be zero-sized"
        );
        assert_eq!(
            core::mem::size_of::<CommonAverageReference<128>>(),
            0,
            "CommonAverageReference should be zero-sized regardless of C"
        );
    }

    #[test]
    fn test_four_channel_example() {
        // Example from the plan
        let car: CommonAverageReference<4> = CommonAverageReference::new();
        let samples = [1.0, 2.0, 3.0, 4.0]; // mean = 2.5
        let filtered = car.process(&samples);

        // Expected: [-1.5, -0.5, 0.5, 1.5]
        assert!((filtered[0] - (-1.5)).abs() < 1e-6);
        assert!((filtered[1] - (-0.5)).abs() < 1e-6);
        assert!((filtered[2] - 0.5).abs() < 1e-6);
        assert!((filtered[3] - 1.5).abs() < 1e-6);
    }
}
