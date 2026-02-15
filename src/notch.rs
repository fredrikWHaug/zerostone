//! Multi-channel notch filter bank for powerline interference removal.
//!
//! [`NotchFilter`] cascades multiple biquad notch sections targeting powerline
//! harmonics (50/60 Hz and up). Coefficients are shared across channels;
//! state is per-channel for complete independence.

use crate::filter::BiquadCoeffs;

/// Multi-channel, multi-section notch filter bank for powerline noise removal.
///
/// Cascades `SECTIONS` biquad notch filters in series, one per harmonic.
/// Coefficients are shared across all `C` channels; filter state is
/// per-channel and per-section for complete channel independence.
///
/// # Type Parameters
/// * `C` - Number of channels (compile-time constant, >= 1)
/// * `SECTIONS` - Number of notch sections / harmonics (>= 1)
///
/// # Example
/// ```
/// use zerostone::NotchFilter;
///
/// // 8-channel, 4-section filter targeting 60/120/180/240 Hz at 1000 Hz fs
/// let mut filter = NotchFilter::<8, 4>::powerline_60hz(1000.0);
///
/// let input = [0.0f32; 8];
/// let output = filter.process_sample(&input);
/// assert_eq!(output.len(), 8);
/// ```
pub struct NotchFilter<const C: usize, const SECTIONS: usize> {
    /// Biquad coefficients shared across all channels
    coeffs: [BiquadCoeffs; SECTIONS],
    /// Per-channel, per-section state: [x1, x2, y1, y2]
    state: [[[f32; 4]; SECTIONS]; C],
}

impl<const C: usize, const SECTIONS: usize> NotchFilter<C, SECTIONS> {
    const _ASSERT_C: () = assert!(C >= 1, "C must be at least 1");
    const _ASSERT_S: () = assert!(SECTIONS >= 1, "SECTIONS must be at least 1");

    /// Internal constructor: fills sections with notch at each harmonic;
    /// passthrough for harmonics at or above Nyquist.
    fn from_powerline(sample_rate: f32, fundamental: f32, q: f32) -> Self {
        let () = Self::_ASSERT_C;
        let () = Self::_ASSERT_S;

        let nyquist = sample_rate / 2.0;
        let mut coeffs = [BiquadCoeffs::passthrough(); SECTIONS];
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            let freq = fundamental * (i + 1) as f32;
            if freq > 0.0 && freq < nyquist {
                *coeff = BiquadCoeffs::notch(sample_rate, freq, q);
            }
            // freq >= nyquist → passthrough (already initialized)
        }

        Self {
            coeffs,
            state: [[[0.0; 4]; SECTIONS]; C],
        }
    }

    /// Creates a notch filter bank targeting 60 Hz powerline noise (US/Japan).
    ///
    /// Notches at 60, 120, 180, 240 Hz (up to Nyquist). Harmonics above
    /// Nyquist automatically become passthroughs — no panic, no cost.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    ///
    /// # Example
    /// ```
    /// use zerostone::NotchFilter;
    /// let mut f = NotchFilter::<8, 4>::powerline_60hz(1000.0);
    /// let out = f.process_sample(&[0.0; 8]);
    /// assert_eq!(out, [0.0; 8]);
    /// ```
    pub fn powerline_60hz(sample_rate: f32) -> Self {
        Self::from_powerline(sample_rate, 60.0, 30.0)
    }

    /// Creates a notch filter bank targeting 50 Hz powerline noise (EU/Asia).
    ///
    /// Notches at 50, 100, 150, 200 Hz (up to Nyquist). Harmonics above
    /// Nyquist automatically become passthroughs.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz
    ///
    /// # Example
    /// ```
    /// use zerostone::NotchFilter;
    /// let mut f = NotchFilter::<4, 4>::powerline_50hz(1000.0);
    /// let out = f.process_sample(&[0.0; 4]);
    /// assert_eq!(out, [0.0; 4]);
    /// ```
    pub fn powerline_50hz(sample_rate: f32) -> Self {
        Self::from_powerline(sample_rate, 50.0, 30.0)
    }

    /// Creates a notch filter bank with user-specified center frequencies.
    ///
    /// A frequency of `0.0` creates a passthrough for that section.
    /// Frequencies at or above Nyquist also become passthroughs automatically.
    ///
    /// # Arguments
    /// * `sample_rate` - Sampling frequency in Hz (must be > 0)
    /// * `freqs` - Center frequency for each section in Hz; `0.0` = passthrough
    /// * `q` - Quality factor (higher = narrower notch; 30 is standard)
    ///
    /// # Panics
    /// Panics if `sample_rate <= 0.0`.
    pub fn custom(sample_rate: f32, freqs: [f32; SECTIONS], q: f32) -> Self {
        let () = Self::_ASSERT_C;
        let () = Self::_ASSERT_S;

        assert!(sample_rate > 0.0, "sample_rate must be positive");

        let nyquist = sample_rate / 2.0;
        let mut coeffs = [BiquadCoeffs::passthrough(); SECTIONS];
        for (coeff, &freq) in coeffs.iter_mut().zip(freqs.iter()) {
            if freq > 0.0 && freq < nyquist {
                *coeff = BiquadCoeffs::notch(sample_rate, freq, q);
            }
        }

        Self {
            coeffs,
            state: [[[0.0; 4]; SECTIONS]; C],
        }
    }

    /// Processes a single multi-channel sample through all notch sections.
    ///
    /// Each channel is filtered independently through all `SECTIONS` biquad
    /// stages in series using Direct Form I.
    #[inline]
    pub fn process_sample(&mut self, input: &[f32; C]) -> [f32; C] {
        let mut output = [0.0f32; C];

        for ch in 0..C {
            let mut x = input[ch];

            for sec in 0..SECTIONS {
                let c = &self.coeffs[sec];
                let s = &mut self.state[ch][sec];

                // Direct Form I:
                // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                let y = c.b0 * x + c.b1 * s[0] + c.b2 * s[1] - c.a1 * s[2] - c.a2 * s[3];

                s[1] = s[0]; // x[n-2] = x[n-1]
                s[0] = x; // x[n-1] = x[n]
                s[3] = s[2]; // y[n-2] = y[n-1]
                s[2] = y; // y[n-1] = y[n]

                x = y;
            }

            output[ch] = x;
        }

        output
    }

    /// Processes a block of multi-channel samples in place.
    pub fn process_block(&mut self, block: &mut [[f32; C]]) {
        for sample in block.iter_mut() {
            *sample = self.process_sample(sample);
        }
    }

    /// Resets all filter state to zero (clears all delay lines).
    pub fn reset(&mut self) {
        self.state = [[[0.0; 4]; SECTIONS]; C];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::IirFilter;

    /// Helper: measure peak amplitude of a sine after settling.
    fn measure_amplitude(
        freq: f32,
        sample_rate: f32,
        settle: usize,
        measure: usize,
        filter_fn: &mut impl FnMut(f32) -> f32,
    ) -> f32 {
        use core::f32::consts::PI;

        for i in 0..settle {
            let t = i as f32 / sample_rate;
            filter_fn(libm::sinf(2.0 * PI * freq * t));
        }

        let mut max = 0.0f32;
        for i in settle..(settle + measure) {
            let t = i as f32 / sample_rate;
            let y = filter_fn(libm::sinf(2.0 * PI * freq * t));
            max = max.max(y.abs());
        }
        max
    }

    #[test]
    fn test_passthrough_on_coeff() {
        // BiquadCoeffs::passthrough() should pass 1.0 → 1.0 exactly
        let mut f: IirFilter<1> = IirFilter::new([BiquadCoeffs::passthrough()]);
        assert_eq!(f.process_sample(1.0), 1.0);
        assert_eq!(f.process_sample(2.5), 2.5);
        assert_eq!(f.process_sample(-0.5), -0.5);
    }

    #[test]
    fn test_60hz_attenuated_40db() {
        let mut f = NotchFilter::<1, 4>::powerline_60hz(1000.0);
        let amp = measure_amplitude(60.0, 1000.0, 1000, 100, &mut |s| f.process_sample(&[s])[0]);
        assert!(
            amp < 0.01,
            "60 Hz should be attenuated >40 dB, got amplitude {}",
            amp
        );
    }

    #[test]
    fn test_10hz_passes_through() {
        let mut f = NotchFilter::<1, 4>::powerline_60hz(1000.0);
        let amp = measure_amplitude(10.0, 1000.0, 500, 100, &mut |s| f.process_sample(&[s])[0]);
        assert!(
            amp > 0.9,
            "10 Hz should pass through 60 Hz notch, got amplitude {}",
            amp
        );
    }

    #[test]
    fn test_120hz_attenuated() {
        let mut f = NotchFilter::<1, 4>::powerline_60hz(1000.0);
        let amp = measure_amplitude(120.0, 1000.0, 1000, 100, &mut |s| f.process_sample(&[s])[0]);
        assert!(
            amp < 0.01,
            "120 Hz harmonic should be attenuated >40 dB, got amplitude {}",
            amp
        );
    }

    #[test]
    fn test_50hz_attenuated() {
        let mut f = NotchFilter::<1, 4>::powerline_50hz(1000.0);
        let amp = measure_amplitude(50.0, 1000.0, 1000, 100, &mut |s| f.process_sample(&[s])[0]);
        assert!(
            amp < 0.01,
            "50 Hz should be attenuated >40 dB by 50 Hz notch, got amplitude {}",
            amp
        );
    }

    #[test]
    fn test_multichannel_independence() {
        use core::f32::consts::PI;

        let mut f = NotchFilter::<2, 4>::powerline_60hz(1000.0);
        let sample_rate = 1000.0_f32;

        // Channel 0: 60 Hz (should be attenuated)
        // Channel 1: 10 Hz (should pass through)
        // Settle
        for i in 0..1000usize {
            let t = i as f32 / sample_rate;
            let ch0 = libm::sinf(2.0 * PI * 60.0 * t);
            let ch1 = libm::sinf(2.0 * PI * 10.0 * t);
            f.process_sample(&[ch0, ch1]);
        }

        let mut max_ch0 = 0.0f32;
        let mut max_ch1 = 0.0f32;
        for i in 1000..1100usize {
            let t = i as f32 / sample_rate;
            let ch0 = libm::sinf(2.0 * PI * 60.0 * t);
            let ch1 = libm::sinf(2.0 * PI * 10.0 * t);
            let out = f.process_sample(&[ch0, ch1]);
            max_ch0 = max_ch0.max(out[0].abs());
            max_ch1 = max_ch1.max(out[1].abs());
        }

        assert!(
            max_ch0 < 0.01,
            "ch0 (60 Hz) should be attenuated, got {}",
            max_ch0
        );
        assert!(
            max_ch1 > 0.9,
            "ch1 (10 Hz) should pass through, got {}",
            max_ch1
        );
    }

    #[test]
    fn test_reset_clears_state() {
        use core::f32::consts::PI;

        let sample_rate = 1000.0_f32;
        let mut f = NotchFilter::<1, 4>::powerline_60hz(sample_rate);

        // Process some samples to build up state
        for i in 0..200usize {
            let t = i as f32 / sample_rate;
            f.process_sample(&[libm::sinf(2.0 * PI * 60.0 * t)]);
        }

        // Reset and re-process same signal from start
        f.reset();

        // Fresh filter for comparison
        let mut f2 = NotchFilter::<1, 4>::powerline_60hz(sample_rate);

        for i in 0..10usize {
            let t = i as f32 / sample_rate;
            let x = libm::sinf(2.0 * PI * 60.0 * t);
            let y1 = f.process_sample(&[x])[0];
            let y2 = f2.process_sample(&[x])[0];
            assert_eq!(
                y1, y2,
                "reset output must match fresh filter at sample {}",
                i
            );
        }
    }

    #[test]
    fn test_custom_single_notch() {
        // Only section 0 is a notch (60 Hz); sections 1-3 are passthrough (freq=0.0)
        let mut f = NotchFilter::<1, 4>::custom(1000.0, [60.0, 0.0, 0.0, 0.0], 30.0);
        let amp = measure_amplitude(60.0, 1000.0, 1000, 100, &mut |s| f.process_sample(&[s])[0]);
        assert!(
            amp < 0.01,
            "custom notch: 60 Hz should be attenuated, got {}",
            amp
        );

        // 120 Hz should pass (no notch at 120 Hz in custom)
        let mut f2 = NotchFilter::<1, 4>::custom(1000.0, [60.0, 0.0, 0.0, 0.0], 30.0);
        let amp2 = measure_amplitude(120.0, 1000.0, 500, 100, &mut |s| f2.process_sample(&[s])[0]);
        assert!(
            amp2 > 0.9,
            "custom notch: 120 Hz should pass (no notch there), got {}",
            amp2
        );
    }

    #[test]
    fn test_low_sample_rate_passthrough() {
        // At 200 Hz fs, Nyquist = 100 Hz. Fundamental 60 Hz:
        //   section 0 = 60 Hz notch (< 100 Hz, valid)
        //   sections 1-3 = passthrough (120, 180, 240 Hz all >= 100 Hz)
        // Should construct without panic.
        let mut f = NotchFilter::<1, 4>::powerline_60hz(200.0);

        // 60 Hz is above Nyquist at 200 Hz — section 0 should also be passthrough
        // because 60 < 100 is valid... wait, 200/2 = 100, so 60 < 100: valid notch.
        // But 60 Hz at 200 Hz sample rate has omega = 2π*60/200 = 1.885 rad (close to π)
        // The filter should still work; just verify no panic and output is finite.
        let out = f.process_sample(&[0.5]);
        assert!(out[0].is_finite(), "output should be finite at low fs");
    }

    #[test]
    fn test_doc_example() {
        // Verify the doc example compiles and runs
        let mut filter = NotchFilter::<8, 4>::powerline_60hz(1000.0);
        let input = [0.0f32; 8];
        let output = filter.process_sample(&input);
        assert_eq!(output, [0.0; 8]);
    }
}
