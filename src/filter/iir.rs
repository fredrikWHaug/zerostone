/// Biquad (2nd-order IIR) filter coefficients.
///
/// Implements the difference equation:
/// `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`
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

    /// Creates 2nd-order notch (band-reject) filter coefficients.
    ///
    /// Attenuates a narrow band around the center frequency while passing
    /// all other frequencies. Essential for removing powerline interference
    /// (50/60 Hz) in BCI applications.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `center_freq` - Center frequency to reject in Hz
    /// * `q` - Quality factor controlling notch width (higher Q = narrower notch)
    ///
    /// # Quality Factor Guidelines
    /// - Q = 30: Narrow notch, typical for powerline removal
    /// - Q = 10: Moderate notch width
    /// - Q = 1: Very wide notch (approaches highpass/lowpass)
    ///
    /// # Example
    /// ```
    /// # use zerostone::BiquadCoeffs;
    /// // Remove 60 Hz powerline noise (US) with narrow notch
    /// let coeffs = BiquadCoeffs::notch(1000.0, 60.0, 30.0);
    ///
    /// // Remove 50 Hz powerline noise (EU) with narrow notch
    /// let coeffs = BiquadCoeffs::notch(1000.0, 50.0, 30.0);
    /// ```
    pub fn notch(sample_rate: f32, center_freq: f32, q: f32) -> Self {
        use core::f32::consts::PI;

        let omega = 2.0 * PI * center_freq / sample_rate;
        let cos_omega = libm::cosf(omega);
        let sin_omega = libm::sinf(omega);
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;

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
            s[0] = x; // x[n-1] = x[n]
            s[3] = s[2]; // y[n-2] = y[n-1]
            s[2] = y; // y[n-1] = y[n]

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

impl<const SECTIONS: usize> crate::pipeline::BlockProcessor<1> for IirFilter<SECTIONS> {
    type Sample = f32;

    fn process_block_inplace(&mut self, block: &mut [[f32; 1]]) {
        for sample in block.iter_mut() {
            sample[0] = self.process_sample(sample[0]);
        }
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn name(&self) -> &str {
        "IirFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            (output - 1.0).abs() < 0.01,
            "DC should pass through lowpass"
        );
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
        assert!(
            output.abs() < 0.2,
            "DC should be rejected by highpass, got {}",
            output
        );
    }

    #[test]
    fn test_iir_process_block() {
        let mut filter: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 100.0)]);

        let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        filter.process_block(&mut samples);

        // All samples should have been processed (modified)
        assert_ne!(samples, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_iir_reset() {
        let mut filter: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 40.0)]);

        // Process some samples to build up state
        for i in 0..10 {
            filter.process_sample(i as f32);
        }

        // Reset should clear state
        filter.reset();

        // After reset, same input should produce same output as fresh filter
        let mut filter2: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 40.0)]);

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
        let new_coeffs = [BiquadCoeffs::passthrough(), BiquadCoeffs::passthrough()];
        filter.set_coefficients(new_coeffs);

        // Should now behave as passthrough
        assert_eq!(filter.process_sample(2.5), 2.5);
    }

    #[test]
    fn test_notch_dc_passthrough() {
        // DC (0 Hz) should pass through a notch filter
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::notch(1000.0, 60.0, 30.0)]);

        // Feed DC signal and let it settle
        for _ in 0..100 {
            filter.process_sample(1.0);
        }

        // After settling, DC should pass through
        let output = filter.process_sample(1.0);
        assert!(
            (output - 1.0).abs() < 0.01,
            "DC should pass through notch filter, got {}",
            output
        );
    }

    #[test]
    fn test_notch_rejects_center_frequency() {
        // A 60 Hz signal should be heavily attenuated by a 60 Hz notch filter
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::notch(1000.0, 60.0, 30.0)]);

        // Generate 60 Hz sine wave and measure output amplitude after settling
        let sample_rate = 1000.0;
        let freq = 60.0;

        // Let filter settle
        for i in 0..500 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            filter.process_sample(sample);
        }

        // Measure output amplitude over one cycle
        let mut max_output = 0.0_f32;
        for i in 500..600 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            let output = filter.process_sample(sample);
            max_output = max_output.max(output.abs());
        }

        // 60 Hz should be heavily attenuated (at least -20 dB, so < 0.1)
        assert!(
            max_output < 0.1,
            "60 Hz should be rejected by 60 Hz notch, got amplitude {}",
            max_output
        );
    }

    #[test]
    fn test_notch_passes_other_frequencies() {
        // A 30 Hz signal should pass through a 60 Hz notch filter with minimal attenuation
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::notch(1000.0, 60.0, 30.0)]);

        let sample_rate = 1000.0;
        let freq = 30.0; // Well away from 60 Hz notch

        // Let filter settle
        for i in 0..500 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            filter.process_sample(sample);
        }

        // Measure output amplitude over one cycle
        let mut max_output = 0.0_f32;
        for i in 500..600 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            let output = filter.process_sample(sample);
            max_output = max_output.max(output.abs());
        }

        // 30 Hz should pass with minimal attenuation (> 0.9)
        assert!(
            max_output > 0.9,
            "30 Hz should pass through 60 Hz notch, got amplitude {}",
            max_output
        );
    }

    #[test]
    fn test_notch_50hz_powerline() {
        // Test 50 Hz notch for EU powerline removal
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::notch(1000.0, 50.0, 30.0)]);

        let sample_rate = 1000.0;

        // 50 Hz should be rejected
        let mut max_50hz = 0.0_f32;
        for i in 0..1000 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * 50.0 * t);
            let output = filter.process_sample(sample);
            if i > 500 {
                max_50hz = max_50hz.max(output.abs());
            }
        }

        assert!(max_50hz < 0.1, "50 Hz should be rejected, got {}", max_50hz);
    }

    #[test]
    fn test_block_processor_inplace() {
        use crate::pipeline::BlockProcessor;

        let mut filter: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 100.0)]);

        // Create block of single-channel samples
        let mut block = [[1.0], [2.0], [3.0], [4.0]];
        BlockProcessor::process_block_inplace(&mut filter, &mut block);

        // All samples should be processed (modified)
        assert_ne!(block[0][0], 1.0);
        assert_ne!(block[1][0], 2.0);
    }

    #[test]
    fn test_block_processor_out_of_place() {
        use crate::pipeline::BlockProcessor;

        let mut filter: IirFilter<2> = IirFilter::new([
            BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
            BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
        ]);

        let input = [[1.0], [2.0], [3.0], [4.0]];
        let mut output = [[0.0]; 4];

        let n = BlockProcessor::process_block(&mut filter, &input, &mut output);

        assert_eq!(n, 4);
        // Output should be different from input (filtered)
        assert_ne!(output[0][0], 1.0);
    }

    #[test]
    fn test_block_processor_reset() {
        use crate::pipeline::BlockProcessor;

        let mut filter: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 40.0)]);

        // Process some data
        let mut block = [[1.0], [2.0], [3.0]];
        BlockProcessor::process_block_inplace(&mut filter, &mut block);

        // Reset using trait method
        BlockProcessor::reset(&mut filter);

        // After reset, should match fresh filter
        let mut fresh: IirFilter<1> =
            IirFilter::new([BiquadCoeffs::butterworth_lowpass(1000.0, 40.0)]);

        let mut block1 = [[5.0]];
        let mut block2 = [[5.0]];

        BlockProcessor::process_block_inplace(&mut filter, &mut block1);
        BlockProcessor::process_block_inplace(&mut fresh, &mut block2);

        assert_eq!(block1[0][0], block2[0][0]);
    }
}
