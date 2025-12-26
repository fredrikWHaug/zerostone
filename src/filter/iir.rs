/// Biquad (2nd-order IIR) filter coefficients.
///
/// Implements the difference equation:
/// y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
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
            s[0] = x;    // x[n-1] = x[n]
            s[3] = s[2]; // y[n-2] = y[n-1]
            s[2] = y;    // y[n-1] = y[n]

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
        assert!((output - 1.0).abs() < 0.01, "DC should pass through lowpass");
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
        assert!(output.abs() < 0.2, "DC should be rejected by highpass, got {}", output);
    }

    #[test]
    fn test_iir_process_block() {
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 100.0,
        )]);

        let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        filter.process_block(&mut samples);

        // All samples should have been processed (modified)
        assert_ne!(samples, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_iir_reset() {
        let mut filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 40.0,
        )]);

        // Process some samples to build up state
        for i in 0..10 {
            filter.process_sample(i as f32);
        }

        // Reset should clear state
        filter.reset();

        // After reset, same input should produce same output as fresh filter
        let mut filter2: IirFilter<1> = IirFilter::new([BiquadCoeffs::butterworth_lowpass(
            1000.0, 40.0,
        )]);

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
        let new_coeffs = [
            BiquadCoeffs::passthrough(),
            BiquadCoeffs::passthrough(),
        ];
        filter.set_coefficients(new_coeffs);

        // Should now behave as passthrough
        assert_eq!(filter.process_sample(3.14), 3.14);
    }
}
