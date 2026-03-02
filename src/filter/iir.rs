/// Minimal complex number for filter design (f64 precision, no_std compatible).
#[derive(Clone, Copy, Debug)]
struct Complex64 {
    re: f64,
    im: f64,
}

impl Complex64 {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn abs(self) -> f64 {
        libm::sqrt(self.abs2())
    }

    fn sqrt(self) -> Self {
        let r = self.abs();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let re = libm::sqrt((r + self.re) / 2.0);
        let im = if self.im >= 0.0 { 1.0 } else { -1.0 } * libm::sqrt((r - self.re) / 2.0);
        Self { re, im }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    fn div(self, other: Self) -> Self {
        let denom = other.abs2();
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
}

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

    /// Creates N sections for a 2*N-order Butterworth lowpass filter with proper pole placement.
    ///
    /// Each section has a distinct Q factor derived from the Butterworth pole positions,
    /// producing a true maximally-flat magnitude response matching scipy.signal.butter.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `cutoff` - Cutoff frequency in Hz
    ///
    /// # Example
    /// ```
    /// # use zerostone::BiquadCoeffs;
    /// // 4th-order Butterworth lowpass (2 sections)
    /// let sections = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 100.0);
    /// ```
    pub fn butterworth_lowpass_sections<const N: usize>(
        sample_rate: f32,
        cutoff: f32,
    ) -> [BiquadCoeffs; N] {
        use core::f64::consts::PI;

        let fs = sample_rate as f64;
        let fc = cutoff as f64;
        let omega = 2.0 * PI * fc / fs;
        let cos_omega = libm::cos(omega);
        let sin_omega = libm::sin(omega);

        let mut sections = [BiquadCoeffs::passthrough(); N];

        for (k, section) in sections.iter_mut().enumerate() {
            let q_k = 1.0 / (2.0 * libm::cos(PI * (2 * k + 1) as f64 / (4 * N) as f64));
            let alpha_k = sin_omega / (2.0 * q_k);

            let b0 = (1.0 - cos_omega) / 2.0;
            let b1 = 1.0 - cos_omega;
            let b2 = b0;
            let a0 = 1.0 + alpha_k;
            let a1 = -2.0 * cos_omega;
            let a2 = 1.0 - alpha_k;

            *section = BiquadCoeffs {
                b0: (b0 / a0) as f32,
                b1: (b1 / a0) as f32,
                b2: (b2 / a0) as f32,
                a1: (a1 / a0) as f32,
                a2: (a2 / a0) as f32,
            };
        }

        sections
    }

    /// Creates N sections for a 2*N-order Butterworth highpass filter with proper pole placement.
    ///
    /// Each section has a distinct Q factor derived from the Butterworth pole positions.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `cutoff` - Cutoff frequency in Hz
    pub fn butterworth_highpass_sections<const N: usize>(
        sample_rate: f32,
        cutoff: f32,
    ) -> [BiquadCoeffs; N] {
        use core::f64::consts::PI;

        let fs = sample_rate as f64;
        let fc = cutoff as f64;
        let omega = 2.0 * PI * fc / fs;
        let cos_omega = libm::cos(omega);
        let sin_omega = libm::sin(omega);

        let mut sections = [BiquadCoeffs::passthrough(); N];

        for (k, section) in sections.iter_mut().enumerate() {
            let q_k = 1.0 / (2.0 * libm::cos(PI * (2 * k + 1) as f64 / (4 * N) as f64));
            let alpha_k = sin_omega / (2.0 * q_k);

            let b0 = (1.0 + cos_omega) / 2.0;
            let b1 = -(1.0 + cos_omega);
            let b2 = b0;
            let a0 = 1.0 + alpha_k;
            let a1 = -2.0 * cos_omega;
            let a2 = 1.0 - alpha_k;

            *section = BiquadCoeffs {
                b0: (b0 / a0) as f32,
                b1: (b1 / a0) as f32,
                b2: (b2 / a0) as f32,
                a1: (a1 / a0) as f32,
                a2: (a2 / a0) as f32,
            };
        }

        sections
    }

    /// Creates N sections for a 2*N-order Butterworth bandpass filter with proper pole placement.
    ///
    /// Uses the full ZPK (zero-pole-gain) pipeline: analog prototype -> lp2bp transform
    /// -> bilinear transform -> biquad sections. Matches scipy.signal.butter output.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    /// * `low_cutoff` - Lower cutoff frequency in Hz
    /// * `high_cutoff` - Upper cutoff frequency in Hz
    pub fn butterworth_bandpass_sections<const N: usize>(
        sample_rate: f32,
        low_cutoff: f32,
        high_cutoff: f32,
    ) -> [BiquadCoeffs; N] {
        use core::f64::consts::PI;

        let fs = sample_rate as f64;
        let fl = low_cutoff as f64;
        let fh = high_cutoff as f64;

        // Pre-warp edge frequencies
        let wl = 2.0 * fs * libm::tan(PI * fl / fs);
        let wh = 2.0 * fs * libm::tan(PI * fh / fs);
        let w0 = libm::sqrt(wl * wh); // center frequency
        let bw = wh - wl; // bandwidth

        // Generate analog Butterworth prototype poles (order N for N sections)
        // Each prototype pole generates 2 bandpass poles -> 2nd order section
        let mut digital_poles = [Complex64::new(0.0, 0.0); 8]; // max 4 sections * 2 poles
        let mut n_poles = 0;

        for k in 0..N {
            // Analog prototype pole for Butterworth of order N
            let angle = PI * (2 * k + 1) as f64 / (2 * N) as f64 + PI / 2.0;
            let p_proto = Complex64::new(libm::cos(angle), libm::sin(angle));

            // lp2bp: scale by bandwidth, then find bandpass poles
            // p_lp = p_proto * bw/2
            let p_lp = p_proto.scale(bw / 2.0);

            // p_bp = p_lp +/- sqrt(p_lp^2 - w0^2)
            let p_lp_sq = p_lp.mul(p_lp);
            let discriminant = p_lp_sq.sub(Complex64::new(w0 * w0, 0.0));
            let sq = discriminant.sqrt();

            digital_poles[n_poles] = bilinear_transform(p_lp.add(sq), fs);
            digital_poles[n_poles + 1] = bilinear_transform(p_lp.sub(sq), fs);
            n_poles += 2;
        }

        // Group into conjugate pairs and form biquad sections
        let mut sections = [BiquadCoeffs::passthrough(); N];

        for k in 0..N {
            let p1 = digital_poles[2 * k];
            let p2 = digital_poles[2 * k + 1];

            // Denominator: (z - p1)(z - p2) = z^2 - (p1+p2)z + p1*p2
            let sum = p1.add(p2);
            let prod = p1.mul(p2);
            let a1 = -sum.re;
            let a2 = prod.re;

            // Bandpass numerator: zeros at z=1 and z=-1 -> (z-1)(z+1) = z^2 - 1
            // So b0 = 1, b1 = 0, b2 = -1 (before gain normalization)
            let mut b0 = 1.0;
            let b1 = 0.0;
            let mut b2 = -1.0;

            // Compute gain so that peak response = 1
            // Use the pre-warped center frequency mapped back through bilinear
            let omega_c_digital = 2.0 * libm::atan2(w0, 2.0 * fs);
            let z_c = Complex64::new(libm::cos(omega_c_digital), libm::sin(omega_c_digital));

            // Numerator at z_c: z_c^2 - 1
            let num = z_c.mul(z_c).sub(Complex64::new(1.0, 0.0));
            // Denominator at z_c: z_c^2 + a1*z_c + a2
            let den = z_c.mul(z_c).add(z_c.scale(a1)).add(Complex64::new(a2, 0.0));

            let gain = den.abs() / num.abs();
            b0 *= gain;
            b2 *= gain;

            sections[k] = BiquadCoeffs {
                b0: b0 as f32,
                b1: b1 as f32,
                b2: b2 as f32,
                a1: a1 as f32,
                a2: a2 as f32,
            };
        }

        sections
    }
}

/// Bilinear transform: maps analog pole to digital pole.
/// z = (2*fs + s) / (2*fs - s)
fn bilinear_transform(s: Complex64, fs: f64) -> Complex64 {
    let two_fs = Complex64::new(2.0 * fs, 0.0);
    two_fs.add(s).div(two_fs.sub(s))
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
/// // 4th-order Butterworth lowpass at 40 Hz (proper pole placement)
/// let sections = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 40.0);
/// let mut filter: IirFilter<2> = IirFilter::new(sections);
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

    #[test]
    fn test_butterworth_sections_distinct_coefficients() {
        let sections = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 100.0);
        // Two sections should have DIFFERENT coefficients (distinct Q factors)
        assert_ne!(sections[0].b0, sections[1].b0);
        assert_ne!(sections[0].a2, sections[1].a2);
    }

    #[test]
    fn test_butterworth_lowpass_sections_dc_passthrough() {
        // DC should pass through lowpass for all orders
        for n_sections in [1, 2, 3, 4] {
            let mut filter: IirFilter<4> = match n_sections {
                1 => {
                    let s = BiquadCoeffs::butterworth_lowpass_sections::<1>(1000.0, 100.0);
                    let mut arr = [BiquadCoeffs::passthrough(); 4];
                    arr[0] = s[0];
                    IirFilter::new(arr)
                }
                2 => {
                    let s = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 100.0);
                    let mut arr = [BiquadCoeffs::passthrough(); 4];
                    arr[0] = s[0];
                    arr[1] = s[1];
                    IirFilter::new(arr)
                }
                3 => {
                    let s = BiquadCoeffs::butterworth_lowpass_sections::<3>(1000.0, 100.0);
                    let mut arr = [BiquadCoeffs::passthrough(); 4];
                    arr[0] = s[0];
                    arr[1] = s[1];
                    arr[2] = s[2];
                    IirFilter::new(arr)
                }
                4 => {
                    let s = BiquadCoeffs::butterworth_lowpass_sections::<4>(1000.0, 100.0);
                    IirFilter::new(s)
                }
                _ => unreachable!(),
            };

            for _ in 0..200 {
                filter.process_sample(1.0);
            }
            let output = filter.process_sample(1.0);
            assert!(
                (output - 1.0).abs() < 0.01,
                "DC should pass through {}-section lowpass, got {}",
                n_sections,
                output
            );
        }
    }

    #[test]
    fn test_butterworth_highpass_sections_dc_rejection() {
        // DC should be rejected by highpass for all orders
        let sections = BiquadCoeffs::butterworth_highpass_sections::<2>(1000.0, 10.0);
        let mut filter: IirFilter<2> = IirFilter::new(sections);

        for _ in 0..200 {
            filter.process_sample(1.0);
        }
        let output = filter.process_sample(1.0);
        assert!(
            output.abs() < 0.01,
            "DC should be rejected by highpass, got {}",
            output
        );
    }

    #[test]
    fn test_butterworth_4th_order_cutoff_gain() {
        // For Butterworth of any order, gain at cutoff = 1/sqrt(2) = -3dB
        let sections = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 100.0);
        let mut filter: IirFilter<2> = IirFilter::new(sections);

        let sample_rate = 1000.0;
        let freq = 100.0;

        // Let filter settle
        for i in 0..2000 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            filter.process_sample(sample);
        }

        // Measure output amplitude
        let mut max_output = 0.0_f32;
        for i in 2000..2500 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            let output = filter.process_sample(sample);
            max_output = max_output.max(output.abs());
        }

        // Should be approximately 1/sqrt(2) = 0.707, NOT the old resonance of ~2.0
        let expected = 1.0 / libm::sqrtf(2.0);
        assert!(
            (max_output - expected).abs() < 0.05,
            "Gain at cutoff should be ~{:.3}, got {:.3}",
            expected,
            max_output
        );
    }

    #[test]
    fn test_butterworth_bandpass_sections_center_passthrough() {
        let sections = BiquadCoeffs::butterworth_bandpass_sections::<2>(1000.0, 8.0, 12.0);
        let mut filter: IirFilter<2> = IirFilter::new(sections);

        let sample_rate = 1000.0;
        let center = libm::sqrtf(8.0 * 12.0);

        // Let filter settle
        for i in 0..5000 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * center * t);
            filter.process_sample(sample);
        }

        // Measure output amplitude
        let mut max_output = 0.0_f32;
        for i in 5000..6000 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * center * t);
            let output = filter.process_sample(sample);
            max_output = max_output.max(output.abs());
        }

        // Center frequency should pass with reasonable amplitude (>0.5)
        assert!(
            max_output > 0.5,
            "Center frequency should pass through bandpass, got {}",
            max_output
        );
    }

    #[test]
    fn test_single_section_is_true_2nd_order_butterworth() {
        // butterworth_lowpass_sections::<1> should produce a true 2nd-order Butterworth
        // with Q = 1/sqrt(2), giving -3dB at cutoff
        let sections = BiquadCoeffs::butterworth_lowpass_sections::<1>(1000.0, 100.0);
        let mut filter: IirFilter<1> = IirFilter::new(sections);

        let sample_rate = 1000.0;
        let freq = 100.0;

        for i in 0..2000 {
            let t = i as f32 / sample_rate;
            filter.process_sample(libm::sinf(2.0 * core::f32::consts::PI * freq * t));
        }

        let mut max_output = 0.0_f32;
        for i in 2000..2500 {
            let t = i as f32 / sample_rate;
            let sample = libm::sinf(2.0 * core::f32::consts::PI * freq * t);
            let output = filter.process_sample(sample);
            max_output = max_output.max(output.abs());
        }

        let expected = 1.0 / libm::sqrtf(2.0);
        assert!(
            (max_output - expected).abs() < 0.05,
            "2nd-order Butterworth gain at cutoff should be ~{:.3}, got {:.3}",
            expected,
            max_output
        );
    }
}
