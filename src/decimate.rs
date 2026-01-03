//! Decimation (downsampling) for reducing sample rate.
//!
//! This module provides a simple decimator that keeps every Nth sample.
//! For proper decimation, apply an anti-aliasing lowpass filter before
//! the decimator to prevent frequency aliasing.

/// A decimator that reduces sample rate by keeping every Nth sample.
///
/// The decimator counts input samples and outputs one sample for every
/// `factor` inputs. This is useful for reducing data rate after filtering.
///
/// # Anti-aliasing
///
/// Before decimating by factor N, you should lowpass filter the signal
/// to remove frequencies above `fs_new / 2` (the new Nyquist frequency).
/// For example, decimating from 1000 Hz to 250 Hz (factor 4) requires
/// a lowpass filter with cutoff below 125 Hz.
///
/// # Example
///
/// ```
/// use zerostone::{BiquadCoeffs, IirFilter, Decimator};
///
/// // Decimate from 1000 Hz to 250 Hz (factor 4)
/// // First, create anti-alias filter (cutoff < 125 Hz)
/// let mut filter: IirFilter<2> = IirFilter::new([
///     BiquadCoeffs::butterworth_lowpass(1000.0, 100.0),
///     BiquadCoeffs::butterworth_lowpass(1000.0, 100.0),
/// ]);
///
/// // Then create decimator
/// let mut decimator: Decimator<4> = Decimator::new(4);
///
/// // Process samples
/// for i in 0..100 {
///     let sample = [i as f32; 4];
///     let filtered = filter.process_sample(sample[0]);
///     let filtered_sample = [filtered; 4];
///
///     if let Some(output) = decimator.process(&filtered_sample) {
///         // Output available every 4th sample
///     }
/// }
/// ```
pub struct Decimator<const C: usize> {
    /// Decimation factor (keep 1 out of every N samples)
    factor: usize,
    /// Current sample counter
    counter: usize,
}

impl<const C: usize> Decimator<C> {
    /// Creates a new decimator with the specified decimation factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Decimation factor (must be >= 1). A factor of 4 means
    ///   the output rate is 1/4 of the input rate.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Decimator;
    ///
    /// // Decimate by factor of 4 (1000 Hz -> 250 Hz)
    /// let decimator: Decimator<8> = Decimator::new(4);
    /// ```
    pub fn new(factor: usize) -> Self {
        assert!(factor >= 1, "Decimation factor must be at least 1");
        Self { factor, counter: 0 }
    }

    /// Process a single multi-channel sample.
    ///
    /// Returns `Some(sample)` every `factor` samples, `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample array with one value per channel
    ///
    /// # Returns
    ///
    /// `Some(input)` for every Nth sample, `None` for samples to be dropped
    pub fn process(&mut self, input: &[f32; C]) -> Option<[f32; C]> {
        let output = if self.counter == 0 {
            Some(*input)
        } else {
            None
        };

        self.counter += 1;
        if self.counter >= self.factor {
            self.counter = 0;
        }

        output
    }

    /// Process a block of samples, returning decimated output.
    ///
    /// The output slice must be large enough to hold `input.len() / factor`
    /// samples (rounded up). Returns the number of samples written.
    ///
    /// # Arguments
    ///
    /// * `input` - Input block of samples
    /// * `output` - Output buffer for decimated samples
    ///
    /// # Returns
    ///
    /// Number of samples written to output
    pub fn process_block(&mut self, input: &[[f32; C]], output: &mut [[f32; C]]) -> usize {
        let mut write_idx = 0;

        for sample in input {
            if let Some(out) = self.process(sample) {
                if write_idx < output.len() {
                    output[write_idx] = out;
                    write_idx += 1;
                }
            }
        }

        write_idx
    }

    /// Reset the decimator state.
    ///
    /// Resets the internal counter so the next sample will be output.
    pub fn reset(&mut self) {
        self.counter = 0;
    }

    /// Get the decimation factor.
    pub fn factor(&self) -> usize {
        self.factor
    }

    /// Set a new decimation factor.
    ///
    /// Also resets the counter.
    pub fn set_factor(&mut self, factor: usize) {
        assert!(factor >= 1, "Decimation factor must be at least 1");
        self.factor = factor;
        self.counter = 0;
    }

    /// Get the current counter value.
    ///
    /// Returns how many samples until the next output (0 means next sample outputs).
    pub fn samples_until_output(&self) -> usize {
        if self.counter == 0 {
            0
        } else {
            self.factor - self.counter
        }
    }
}

impl<const C: usize> Default for Decimator<C> {
    /// Creates a decimator with factor 1 (passthrough).
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimator_factor_1() {
        let mut dec: Decimator<2> = Decimator::new(1);

        // Factor 1 = passthrough, every sample should output
        for i in 0..10 {
            let input = [i as f32, (i * 2) as f32];
            let output = dec.process(&input);
            assert!(output.is_some());
            assert_eq!(output.unwrap(), input);
        }
    }

    #[test]
    fn test_decimator_factor_4() {
        let mut dec: Decimator<1> = Decimator::new(4);

        let mut outputs = 0;
        for i in 0..16 {
            let input = [i as f32];
            if let Some(out) = dec.process(&input) {
                // Should output samples 0, 4, 8, 12
                assert_eq!(out[0], (outputs * 4) as f32);
                outputs += 1;
            }
        }
        assert_eq!(outputs, 4);
    }

    #[test]
    fn test_decimator_multi_channel() {
        let mut dec: Decimator<4> = Decimator::new(2);

        // First sample - should output
        let input1 = [1.0, 2.0, 3.0, 4.0];
        assert!(dec.process(&input1).is_some());

        // Second sample - should drop
        let input2 = [5.0, 6.0, 7.0, 8.0];
        assert!(dec.process(&input2).is_none());

        // Third sample - should output
        let input3 = [9.0, 10.0, 11.0, 12.0];
        let out = dec.process(&input3);
        assert!(out.is_some());
        assert_eq!(out.unwrap(), input3);
    }

    #[test]
    fn test_decimator_reset() {
        let mut dec: Decimator<1> = Decimator::new(4);

        // Advance counter
        dec.process(&[1.0]); // outputs
        dec.process(&[2.0]); // drops
        dec.process(&[3.0]); // drops

        // Counter is now at 3, next would drop
        assert_eq!(dec.samples_until_output(), 1);

        // Reset
        dec.reset();

        // Now first sample should output
        assert!(dec.process(&[4.0]).is_some());
    }

    #[test]
    fn test_decimator_process_block() {
        let mut dec: Decimator<2> = Decimator::new(4);

        let input = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ];

        let mut output = [[0.0, 0.0]; 4];
        let count = dec.process_block(&input, &mut output);

        assert_eq!(count, 2);
        assert_eq!(output[0], [0.0, 0.0]);
        assert_eq!(output[1], [4.0, 4.0]);
    }

    #[test]
    fn test_decimator_set_factor() {
        let mut dec: Decimator<1> = Decimator::new(2);
        assert_eq!(dec.factor(), 2);

        dec.set_factor(8);
        assert_eq!(dec.factor(), 8);
        assert_eq!(dec.samples_until_output(), 0); // Reset on set
    }

    #[test]
    fn test_decimator_samples_until_output() {
        let mut dec: Decimator<1> = Decimator::new(4);

        assert_eq!(dec.samples_until_output(), 0); // Next outputs
        dec.process(&[1.0]);

        assert_eq!(dec.samples_until_output(), 3); // 3 more until output
        dec.process(&[2.0]);

        assert_eq!(dec.samples_until_output(), 2);
        dec.process(&[3.0]);

        assert_eq!(dec.samples_until_output(), 1);
        dec.process(&[4.0]);

        assert_eq!(dec.samples_until_output(), 0); // Back to 0
    }

    #[test]
    fn test_decimator_default() {
        let dec: Decimator<4> = Decimator::default();
        assert_eq!(dec.factor(), 1);
    }

    #[test]
    #[should_panic(expected = "Decimation factor must be at least 1")]
    fn test_decimator_factor_zero_panics() {
        let _dec: Decimator<1> = Decimator::new(0);
    }
}
