//! Interpolation (upsampling) for increasing sample rate.
//!
//! This module provides an interpolator that increases sample rate by an integer factor.
//! It complements the [`Decimator`](crate::Decimator) for sample rate conversion.
//!
//! # Anti-imaging
//!
//! After interpolation, you should apply a lowpass filter to remove imaging artifacts
//! (spectral copies at multiples of the original sample rate). The cutoff should be
//! at the original Nyquist frequency.
//!
//! # Example
//!
//! ```
//! use zerostone::{Interpolator, InterpolationMethod};
//!
//! // Upsample from 250 Hz to 1000 Hz (factor 4) with linear interpolation
//! let mut interp: Interpolator<2> = Interpolator::new(4, InterpolationMethod::Linear);
//!
//! // Process one input sample, get 4 output samples
//! let input = [1.0, 2.0];
//! let mut output = [[0.0f32; 2]; 4];
//! let count = interp.process(&input, &mut output);
//! assert_eq!(count, 4);
//! ```

/// Interpolation method for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Zero-order hold: repeat each sample `factor` times.
    /// Simple but introduces staircase artifacts.
    ZeroOrder,

    /// Linear interpolation between consecutive samples.
    /// Smoother output, slight delay (half sample at original rate).
    Linear,

    /// Insert zeros between samples.
    /// Requires post-filtering but preserves original samples exactly.
    ZeroInsert,
}

/// Multi-channel interpolator for upsampling by an integer factor.
///
/// Takes one input sample and produces `factor` output samples using the
/// specified interpolation method.
///
/// # Type Parameters
///
/// * `C` - Number of channels
///
/// # Example
///
/// ```
/// use zerostone::{Interpolator, InterpolationMethod};
///
/// // Create 4x interpolator with linear interpolation for 8 channels
/// let mut interp: Interpolator<8> = Interpolator::new(4, InterpolationMethod::Linear);
///
/// // First sample initializes the interpolator
/// let mut output = [[0.0f32; 8]; 4];
/// let count = interp.process(&[1.0; 8], &mut output);
/// // First call with Linear method outputs factor samples
/// assert_eq!(count, 4);
/// ```
pub struct Interpolator<const C: usize> {
    /// Interpolation factor (output rate / input rate)
    factor: usize,
    /// Interpolation method
    method: InterpolationMethod,
    /// Previous sample (for linear interpolation)
    prev_sample: [f32; C],
    /// Whether we have a previous sample
    initialized: bool,
}

impl<const C: usize> Interpolator<C> {
    /// Creates a new interpolator with the specified factor and method.
    ///
    /// # Arguments
    ///
    /// * `factor` - Interpolation factor (must be >= 1). A factor of 4 means
    ///   the output rate is 4x the input rate.
    /// * `method` - Interpolation method to use
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Interpolator, InterpolationMethod};
    ///
    /// // Upsample by 4x with zero-order hold
    /// let interp: Interpolator<4> = Interpolator::new(4, InterpolationMethod::ZeroOrder);
    /// ```
    pub fn new(factor: usize, method: InterpolationMethod) -> Self {
        assert!(factor >= 1, "Interpolation factor must be at least 1");
        Self {
            factor,
            method,
            prev_sample: [0.0; C],
            initialized: false,
        }
    }

    /// Creates a new interpolator with zero-order hold method.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Interpolator;
    ///
    /// let interp: Interpolator<2> = Interpolator::zero_order(4);
    /// ```
    pub fn zero_order(factor: usize) -> Self {
        Self::new(factor, InterpolationMethod::ZeroOrder)
    }

    /// Creates a new interpolator with linear interpolation.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Interpolator;
    ///
    /// let interp: Interpolator<2> = Interpolator::linear(4);
    /// ```
    pub fn linear(factor: usize) -> Self {
        Self::new(factor, InterpolationMethod::Linear)
    }

    /// Process a single input sample, producing `factor` output samples.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample array with one value per channel
    /// * `output` - Output buffer, must have at least `factor` elements
    ///
    /// # Returns
    ///
    /// Number of samples written to output (always `factor` after initialization)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Interpolator, InterpolationMethod};
    ///
    /// let mut interp: Interpolator<1> = Interpolator::new(4, InterpolationMethod::Linear);
    ///
    /// let mut output = [[0.0f32; 1]; 4];
    /// interp.process(&[0.0], &mut output); // Initialize
    /// interp.process(&[4.0], &mut output); // Interpolate 0.0 -> 4.0
    ///
    /// // With linear interpolation: [0.0, 1.0, 2.0, 3.0] (approaching 4.0)
    /// ```
    pub fn process(&mut self, input: &[f32; C], output: &mut [[f32; C]]) -> usize {
        let count = self.factor.min(output.len());

        match self.method {
            InterpolationMethod::ZeroOrder => {
                // Repeat the input sample
                for out in output.iter_mut().take(count) {
                    *out = *input;
                }
            }
            InterpolationMethod::Linear => {
                if !self.initialized {
                    // First sample: output all as the input value
                    for out in output.iter_mut().take(count) {
                        *out = *input;
                    }
                } else {
                    // Interpolate from prev_sample to input
                    for (i, out) in output.iter_mut().enumerate().take(count) {
                        let t = i as f32 / self.factor as f32;
                        for (c, out_c) in out.iter_mut().enumerate() {
                            *out_c = self.prev_sample[c] + t * (input[c] - self.prev_sample[c]);
                        }
                    }
                }
            }
            InterpolationMethod::ZeroInsert => {
                // First sample is input, rest are zeros
                if count > 0 {
                    output[0] = *input;
                }
                for out in output.iter_mut().take(count).skip(1) {
                    *out = [0.0; C];
                }
            }
        }

        self.prev_sample = *input;
        self.initialized = true;

        count
    }

    /// Process a block of input samples.
    ///
    /// # Arguments
    ///
    /// * `input` - Block of input samples
    /// * `output` - Output buffer, must have at least `input.len() * factor` elements
    ///
    /// # Returns
    ///
    /// Number of samples written to output
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Interpolator, InterpolationMethod};
    ///
    /// let mut interp: Interpolator<1> = Interpolator::new(2, InterpolationMethod::ZeroOrder);
    ///
    /// let input = [[1.0], [2.0], [3.0]];
    /// let mut output = [[0.0f32; 1]; 6];
    /// let count = interp.process_block(&input, &mut output);
    ///
    /// assert_eq!(count, 6);
    /// // Output: [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    /// ```
    pub fn process_block(&mut self, input: &[[f32; C]], output: &mut [[f32; C]]) -> usize {
        let mut write_idx = 0;

        for sample in input {
            let remaining = output.len().saturating_sub(write_idx);
            if remaining == 0 {
                break;
            }

            let out_slice = &mut output[write_idx..];
            let count = self.process(sample, out_slice);
            write_idx += count;
        }

        write_idx
    }

    /// Reset the interpolator state.
    ///
    /// Clears the previous sample, so the next input will be treated as the first.
    pub fn reset(&mut self) {
        self.prev_sample = [0.0; C];
        self.initialized = false;
    }

    /// Get the interpolation factor.
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// Set a new interpolation factor.
    ///
    /// Also resets the interpolator state.
    pub fn set_factor(&mut self, factor: usize) {
        assert!(factor >= 1, "Interpolation factor must be at least 1");
        self.factor = factor;
        self.reset();
    }

    /// Get the interpolation method.
    pub fn method(&self) -> InterpolationMethod {
        self.method
    }

    /// Set a new interpolation method.
    ///
    /// Also resets the interpolator state.
    pub fn set_method(&mut self, method: InterpolationMethod) {
        self.method = method;
        self.reset();
    }

    /// Returns whether the interpolator has been initialized with at least one sample.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the number of output samples per input sample.
    pub fn output_count(&self) -> usize {
        self.factor
    }
}

impl<const C: usize> Default for Interpolator<C> {
    /// Creates an interpolator with factor 1 and linear method (passthrough).
    fn default() -> Self {
        Self::new(1, InterpolationMethod::Linear)
    }
}

impl<const C: usize> crate::pipeline::BlockProcessor<C> for Interpolator<C> {
    type Sample = f32;

    fn process_block(&mut self, input: &[[f32; C]], output: &mut [[f32; C]]) -> usize {
        let mut total_written = 0;

        for sample in input {
            let remaining = output.len() - total_written;
            if remaining == 0 {
                break;
            }

            let written = self.process(sample, &mut output[total_written..]);
            total_written += written;
        }

        total_written
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        "Interpolator"
    }
}

impl<const C: usize> crate::pipeline::RateChangingProcessor<C> for Interpolator<C> {
    fn output_length(&self, input_length: usize) -> Option<usize> {
        Some(input_length * self.factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolator_factor_1() {
        let mut interp: Interpolator<2> = Interpolator::new(1, InterpolationMethod::Linear);

        let input = [1.0, 2.0];
        let mut output = [[0.0f32; 2]; 1];
        let count = interp.process(&input, &mut output);

        assert_eq!(count, 1);
        assert_eq!(output[0], input);
    }

    #[test]
    fn test_interpolator_zero_order() {
        let mut interp: Interpolator<1> = Interpolator::zero_order(4);

        let mut output = [[0.0f32; 1]; 4];
        let count = interp.process(&[5.0], &mut output);

        assert_eq!(count, 4);
        for out in &output {
            assert_eq!(out[0], 5.0);
        }
    }

    #[test]
    fn test_interpolator_zero_order_multi_channel() {
        let mut interp: Interpolator<3> = Interpolator::zero_order(2);

        let input = [1.0, 2.0, 3.0];
        let mut output = [[0.0f32; 3]; 2];
        let count = interp.process(&input, &mut output);

        assert_eq!(count, 2);
        assert_eq!(output[0], input);
        assert_eq!(output[1], input);
    }

    #[test]
    fn test_interpolator_linear_basic() {
        let mut interp: Interpolator<1> = Interpolator::linear(4);

        // First sample: initializes, outputs all same value
        let mut output = [[0.0f32; 1]; 4];
        interp.process(&[0.0], &mut output);

        // Second sample: interpolates from 0.0 to 4.0
        interp.process(&[4.0], &mut output);

        // Expected: [0.0, 1.0, 2.0, 3.0] (t = 0, 0.25, 0.5, 0.75)
        assert!((output[0][0] - 0.0).abs() < 1e-6);
        assert!((output[1][0] - 1.0).abs() < 1e-6);
        assert!((output[2][0] - 2.0).abs() < 1e-6);
        assert!((output[3][0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolator_linear_multi_channel() {
        let mut interp: Interpolator<2> = Interpolator::linear(2);

        let mut output = [[0.0f32; 2]; 2];

        // First sample
        interp.process(&[0.0, 10.0], &mut output);

        // Second sample: interpolate to [2.0, 20.0]
        interp.process(&[2.0, 20.0], &mut output);

        // t=0: [0.0, 10.0], t=0.5: [1.0, 15.0]
        assert!((output[0][0] - 0.0).abs() < 1e-6);
        assert!((output[0][1] - 10.0).abs() < 1e-6);
        assert!((output[1][0] - 1.0).abs() < 1e-6);
        assert!((output[1][1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolator_zero_insert() {
        let mut interp: Interpolator<1> = Interpolator::new(4, InterpolationMethod::ZeroInsert);

        let mut output = [[0.0f32; 1]; 4];
        interp.process(&[5.0], &mut output);

        assert_eq!(output[0][0], 5.0);
        assert_eq!(output[1][0], 0.0);
        assert_eq!(output[2][0], 0.0);
        assert_eq!(output[3][0], 0.0);
    }

    #[test]
    fn test_interpolator_process_block_zero_order() {
        let mut interp: Interpolator<1> = Interpolator::zero_order(2);

        let input = [[1.0], [2.0], [3.0]];
        let mut output = [[0.0f32; 1]; 6];
        let count = interp.process_block(&input, &mut output);

        assert_eq!(count, 6);
        assert_eq!(output[0][0], 1.0);
        assert_eq!(output[1][0], 1.0);
        assert_eq!(output[2][0], 2.0);
        assert_eq!(output[3][0], 2.0);
        assert_eq!(output[4][0], 3.0);
        assert_eq!(output[5][0], 3.0);
    }

    #[test]
    fn test_interpolator_process_block_linear() {
        let mut interp: Interpolator<1> = Interpolator::linear(2);

        let input = [[0.0], [2.0], [4.0]];
        let mut output = [[0.0f32; 1]; 6];
        let count = interp.process_block(&input, &mut output);

        assert_eq!(count, 6);
        // First input [0.0]: outputs [0.0, 0.0] (initialization)
        assert!((output[0][0] - 0.0).abs() < 1e-6);
        assert!((output[1][0] - 0.0).abs() < 1e-6);
        // Second input [2.0]: interpolates from 0.0 to 2.0 -> [0.0, 1.0]
        assert!((output[2][0] - 0.0).abs() < 1e-6);
        assert!((output[3][0] - 1.0).abs() < 1e-6);
        // Third input [4.0]: interpolates from 2.0 to 4.0 -> [2.0, 3.0]
        assert!((output[4][0] - 2.0).abs() < 1e-6);
        assert!((output[5][0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolator_reset() {
        let mut interp: Interpolator<1> = Interpolator::linear(2);

        // Initialize
        let mut output = [[0.0f32; 1]; 2];
        interp.process(&[10.0], &mut output);
        assert!(interp.is_initialized());

        // Reset
        interp.reset();
        assert!(!interp.is_initialized());

        // After reset, next sample treated as first
        interp.process(&[20.0], &mut output);
        // Should output [20.0, 20.0] not interpolated from 10.0
        assert_eq!(output[0][0], 20.0);
        assert_eq!(output[1][0], 20.0);
    }

    #[test]
    fn test_interpolator_setters() {
        let mut interp: Interpolator<1> = Interpolator::new(2, InterpolationMethod::Linear);

        assert_eq!(interp.factor(), 2);
        assert_eq!(interp.method(), InterpolationMethod::Linear);

        interp.set_factor(4);
        assert_eq!(interp.factor(), 4);

        interp.set_method(InterpolationMethod::ZeroOrder);
        assert_eq!(interp.method(), InterpolationMethod::ZeroOrder);
    }

    #[test]
    fn test_interpolator_default() {
        let interp: Interpolator<4> = Interpolator::default();
        assert_eq!(interp.factor(), 1);
        assert_eq!(interp.method(), InterpolationMethod::Linear);
    }

    #[test]
    #[should_panic(expected = "Interpolation factor must be at least 1")]
    fn test_interpolator_factor_zero_panics() {
        let _interp: Interpolator<1> = Interpolator::new(0, InterpolationMethod::Linear);
    }

    #[test]
    fn test_interpolator_output_count() {
        let interp: Interpolator<2> = Interpolator::new(8, InterpolationMethod::Linear);
        assert_eq!(interp.output_count(), 8);
    }

    #[test]
    fn test_interpolator_small_output_buffer() {
        let mut interp: Interpolator<1> = Interpolator::zero_order(4);

        // Output buffer smaller than factor
        let mut output = [[0.0f32; 1]; 2];
        let count = interp.process(&[5.0], &mut output);

        // Should only write what fits
        assert_eq!(count, 2);
        assert_eq!(output[0][0], 5.0);
        assert_eq!(output[1][0], 5.0);
    }

    #[test]
    fn test_interpolator_complements_decimator() {
        // Upsample then downsample should approximate identity
        let mut interp: Interpolator<1> = Interpolator::zero_order(4);

        let original = [1.0f32, 2.0, 3.0, 4.0, 5.0];

        // Upsample
        let mut upsampled = [[0.0f32; 1]; 20];
        let mut idx = 0;
        for &val in &original {
            let mut out = [[0.0f32; 1]; 4];
            interp.process(&[val], &mut out);
            for o in out {
                upsampled[idx] = o;
                idx += 1;
            }
        }

        // Check upsampled values
        assert_eq!(upsampled[0][0], 1.0);
        assert_eq!(upsampled[3][0], 1.0);
        assert_eq!(upsampled[4][0], 2.0);
        assert_eq!(upsampled[7][0], 2.0);
    }

    #[test]
    fn test_block_processor_trait() {
        use crate::pipeline::{BlockProcessor, RateChangingProcessor};

        let mut interp: Interpolator<2> = Interpolator::new(4, InterpolationMethod::Linear);

        // 2 input samples → 8 output samples (interpolation by 4)
        let input = [[1.0, 2.0], [5.0, 6.0]];
        let mut output = [[0.0; 2]; 8];

        let n_written = BlockProcessor::process_block(&mut interp, &input, &mut output);
        assert_eq!(n_written, 8);

        // Verify RateChangingProcessor marker trait
        fn assert_rate_changing<P: RateChangingProcessor<2>>() {}
        assert_rate_changing::<Interpolator<2>>();
    }

    #[test]
    fn test_rate_changing_output_length() {
        use crate::pipeline::RateChangingProcessor;

        let interp: Interpolator<1> = Interpolator::new(4, InterpolationMethod::Linear);
        assert_eq!(interp.output_length(10), Some(40));
        assert_eq!(interp.output_length(1), Some(4));
        assert_eq!(interp.output_length(0), Some(0));
    }
}
