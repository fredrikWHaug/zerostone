//! AC coupling filter for removing DC offset from signals.
//!
//! This module provides a simple, efficient single-pole high-pass filter
//! specifically designed for AC coupling (DC removal). It's more efficient
//! than a general IIR filter when you only need to remove DC offset.

use core::f32::consts::PI;

/// An AC coupling filter that removes the DC component from a signal.
///
/// Uses a single-pole high-pass filter with the transfer function:
/// `H(z) = (1 - z^-1) / (1 - α*z^-1)`
///
/// This is equivalent to: `y[n] = x[n] - x[n-1] + α * y[n-1]`
///
/// # Example
///
/// ```
/// use zerostone::AcCoupler;
///
/// // Create a AC coupler with 0.1 Hz cutoff at 250 Hz sample rate
/// let mut blocker: AcCoupler<4> = AcCoupler::new(250.0, 0.1);
///
/// // Process samples - DC offset will be removed
/// let input = [1.5, 1.6, 1.4, 1.5]; // Signal with ~1.5V DC offset
/// let output = blocker.process(&input);
/// // Output will trend toward zero mean
/// ```
pub struct AcCoupler<const C: usize> {
    /// Filter coefficient (pole location)
    alpha: f32,
    /// Previous input sample per channel
    x_prev: [f32; C],
    /// Previous output sample per channel
    y_prev: [f32; C],
}

impl<const C: usize> AcCoupler<C> {
    /// Creates a new AC coupling filter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - The sample rate in Hz
    /// * `cutoff_freq` - The -3dB cutoff frequency in Hz (typically 0.1-1.0 Hz for BCI)
    ///
    /// Lower cutoff frequencies result in slower DC tracking but less signal distortion.
    /// Higher cutoff frequencies track DC faster but may attenuate low-frequency content.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::AcCoupler;
    ///
    /// // 0.1 Hz cutoff - slow DC tracking, minimal signal distortion
    /// let slow: AcCoupler<4> = AcCoupler::new(250.0, 0.1);
    ///
    /// // 1.0 Hz cutoff - fast DC tracking, may affect sub-1Hz signals
    /// let fast: AcCoupler<4> = AcCoupler::new(250.0, 1.0);
    /// ```
    pub fn new(sample_rate: f32, cutoff_freq: f32) -> Self {
        // Compute alpha from cutoff frequency
        // α = exp(-2π * fc / fs)
        let omega = 2.0 * PI * cutoff_freq / sample_rate;
        let alpha = libm::expf(-omega);

        Self {
            alpha,
            x_prev: [0.0; C],
            y_prev: [0.0; C],
        }
    }

    /// Creates a AC coupler with a specified alpha coefficient directly.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The filter coefficient (0 < α < 1). Higher values = lower cutoff.
    ///
    /// Common values:
    /// - 0.995: ~0.2 Hz cutoff at 250 Hz sample rate
    /// - 0.999: ~0.04 Hz cutoff at 250 Hz sample rate
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::AcCoupler;
    ///
    /// let blocker: AcCoupler<4> = AcCoupler::with_alpha(0.995);
    /// ```
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            alpha,
            x_prev: [0.0; C],
            y_prev: [0.0; C],
        }
    }

    /// Process a single multi-channel sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample array with one value per channel
    ///
    /// # Returns
    ///
    /// Output sample array with DC removed from each channel
    pub fn process(&mut self, input: &[f32; C]) -> [f32; C] {
        let mut output = [0.0; C];

        for i in 0..C {
            // y[n] = x[n] - x[n-1] + α * y[n-1]
            output[i] = input[i] - self.x_prev[i] + self.alpha * self.y_prev[i];
            self.x_prev[i] = input[i];
            self.y_prev[i] = output[i];
        }

        output
    }

    /// Process a block of samples in place.
    ///
    /// # Arguments
    ///
    /// * `block` - Mutable slice of sample arrays to process
    pub fn process_block(&mut self, block: &mut [[f32; C]]) {
        for sample in block.iter_mut() {
            *sample = self.process(sample);
        }
    }

    /// Reset the filter state to zero.
    ///
    /// Call this when starting to process a new signal segment
    /// to avoid transients from previous state.
    pub fn reset(&mut self) {
        self.x_prev = [0.0; C];
        self.y_prev = [0.0; C];
    }

    /// Get the current alpha coefficient.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set a new alpha coefficient.
    ///
    /// # Arguments
    ///
    /// * `alpha` - New filter coefficient (0 < α < 1)
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }

    /// Set the cutoff frequency.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - The sample rate in Hz
    /// * `cutoff_freq` - The new cutoff frequency in Hz
    pub fn set_cutoff(&mut self, sample_rate: f32, cutoff_freq: f32) {
        let omega = 2.0 * PI * cutoff_freq / sample_rate;
        self.alpha = libm::expf(-omega);
    }
}

impl<const C: usize> Default for AcCoupler<C> {
    /// Creates a AC coupler with alpha = 0.995 (suitable for most applications).
    fn default() -> Self {
        Self::with_alpha(0.995)
    }
}

impl<const C: usize> crate::pipeline::BlockProcessor<C> for AcCoupler<C> {
    type Sample = f32;

    fn process_block_inplace(&mut self, block: &mut [[f32; C]]) {
        for sample in block.iter_mut() {
            *sample = self.process(sample);
        }
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        "AcCoupler"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_coupler_removes_dc() {
        let mut blocker: AcCoupler<1> = AcCoupler::new(250.0, 0.5);

        // Feed constant DC signal
        let mut output = 0.0;
        for _ in 0..1000 {
            let result = blocker.process(&[1.0]);
            output = result[0];
        }

        // Output should converge to near zero
        assert!(libm::fabsf(output) < 0.01);
    }

    #[test]
    fn test_ac_coupler_preserves_ac() {
        let mut blocker: AcCoupler<1> = AcCoupler::new(250.0, 0.1);

        // Generate 10 Hz sine wave (well above cutoff)
        let sample_rate = 250.0;
        let freq = 10.0;
        let mut max_output = 0.0f32;

        // Let filter settle
        for i in 0..500 {
            let t = i as f32 / sample_rate;
            let input = libm::sinf(2.0 * PI * freq * t);
            let output = blocker.process(&[input]);
            if i > 250 {
                // After settling
                max_output = if libm::fabsf(output[0]) > max_output {
                    libm::fabsf(output[0])
                } else {
                    max_output
                };
            }
        }

        // Should preserve most of the amplitude (>90%)
        assert!(max_output > 0.9);
    }

    #[test]
    fn test_ac_coupler_multi_channel() {
        let mut blocker: AcCoupler<4> = AcCoupler::new(250.0, 0.5);

        // Different DC offsets per channel
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];

        for _ in 0..1000 {
            output = blocker.process(&input);
        }

        // All channels should converge to near zero
        for val in &output {
            assert!(libm::fabsf(*val) < 0.01);
        }
    }

    #[test]
    fn test_ac_coupler_with_alpha() {
        let blocker: AcCoupler<1> = AcCoupler::with_alpha(0.999);
        assert!((blocker.alpha() - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_ac_coupler_reset() {
        let mut blocker: AcCoupler<2> = AcCoupler::new(250.0, 0.5);

        // Process some samples
        blocker.process(&[1.0, 2.0]);
        blocker.process(&[1.0, 2.0]);

        // Reset
        blocker.reset();

        // First sample after reset should behave like fresh start
        let output = blocker.process(&[1.0, 2.0]);
        // First output = x[0] - 0 + α*0 = x[0]
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_ac_coupler_process_block() {
        let mut blocker: AcCoupler<2> = AcCoupler::new(250.0, 0.5);

        let mut block = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];

        blocker.process_block(&mut block);

        // Verify block was modified
        // First sample: output = input (no previous)
        assert!((block[0][0] - 1.0).abs() < 1e-6);
        // Subsequent samples should start removing DC
        assert!(block[3][0] < 1.0);
    }

    #[test]
    fn test_ac_coupler_set_cutoff() {
        let mut blocker: AcCoupler<1> = AcCoupler::new(250.0, 0.1);
        let alpha1 = blocker.alpha();

        blocker.set_cutoff(250.0, 1.0);
        let alpha2 = blocker.alpha();

        // Higher cutoff = lower alpha
        assert!(alpha2 < alpha1);
    }

    #[test]
    fn test_ac_coupler_default() {
        let blocker: AcCoupler<4> = AcCoupler::default();
        assert!((blocker.alpha() - 0.995).abs() < 1e-6);
    }

    #[test]
    fn test_block_processor_inplace() {
        use crate::pipeline::BlockProcessor;

        let mut coupler: AcCoupler<2> = AcCoupler::new(250.0, 0.5);

        let mut block = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        BlockProcessor::process_block_inplace(&mut coupler, &mut block);

        // DC should start being removed
        assert!(block[3][0] < 1.0);
        assert!(block[3][1] < 2.0);
    }

    #[test]
    fn test_block_processor_out_of_place() {
        use crate::pipeline::BlockProcessor;

        let mut coupler: AcCoupler<3> = AcCoupler::new(250.0, 0.5);

        let input = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let mut output = [[0.0; 3]; 3];

        let n = BlockProcessor::process_block(&mut coupler, &input, &mut output);

        assert_eq!(n, 3);
        // First sample passes through
        assert!((output[0][0] - 1.0).abs() < 1e-5);
        // Subsequent samples have DC removal
        assert!(output[2][0] < 1.0);
    }

    #[test]
    fn test_block_processor_reset() {
        use crate::pipeline::BlockProcessor;

        let mut coupler: AcCoupler<2> = AcCoupler::new(250.0, 0.5);

        // Process some data
        let mut block = [[1.0, 2.0], [1.0, 2.0]];
        BlockProcessor::process_block_inplace(&mut coupler, &mut block);

        // Reset using trait method
        BlockProcessor::reset(&mut coupler);

        // Should match fresh coupler
        let mut fresh: AcCoupler<2> = AcCoupler::new(250.0, 0.5);

        let mut block1 = [[5.0, 10.0]];
        let mut block2 = [[5.0, 10.0]];

        BlockProcessor::process_block_inplace(&mut coupler, &mut block1);
        BlockProcessor::process_block_inplace(&mut fresh, &mut block2);

        assert!((block1[0][0] - block2[0][0]).abs() < 1e-5);
        assert!((block1[0][1] - block2[0][1]).abs() < 1e-5);
    }
}
