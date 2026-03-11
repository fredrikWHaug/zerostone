use proptest::prelude::*;
use zerostone::{BiquadCoeffs, Complex, Fft, IirFilter};

proptest! {
    /// Biquad filter stability: arbitrary valid (sr, cutoff) and input signal
    /// must never produce NaN or infinity.
    #[test]
    fn biquad_output_is_finite(
        sr in 100.0f32..48000.0,
        cutoff_ratio in 0.01f32..0.49,
        samples in prop::collection::vec(-1e6f32..1e6, 1000..=1000),
    ) {
        let cutoff = sr * cutoff_ratio;
        let coeffs = BiquadCoeffs::butterworth_lowpass(sr, cutoff);
        let mut filter = IirFilter::<1>::new([coeffs]);
        for &x in &samples {
            let y = filter.process_sample(x);
            prop_assert!(y.is_finite(), "output {y} is not finite for sr={sr}, cutoff={cutoff}, input={x}");
        }
    }

    /// FFT roundtrip: forward then inverse recovers original signal within epsilon.
    #[test]
    fn fft_roundtrip(
        signal in prop::collection::vec(-1e3f32..1e3, 64..=64),
    ) {
        let fft = Fft::<64>::new();
        let mut data: [Complex; 64] = core::array::from_fn(|i| Complex::from_real(signal[i]));
        fft.forward(&mut data);
        fft.inverse(&mut data);
        for (i, &original) in signal.iter().enumerate() {
            let recovered = data[i].re;
            let diff = (recovered - original).abs();
            prop_assert!(
                diff < 1e-2,
                "roundtrip mismatch at index {i}: original={original}, recovered={recovered}, diff={diff}"
            );
        }
    }

    /// Parseval's theorem: time-domain energy equals frequency-domain energy / N.
    #[test]
    fn parsevals_theorem(
        signal in prop::collection::vec(-1e3f32..1e3, 64..=64),
    ) {
        let fft = Fft::<64>::new();
        let time_energy: f32 = signal.iter().map(|x| x * x).sum();

        let mut data: [Complex; 64] = core::array::from_fn(|i| Complex::from_real(signal[i]));
        fft.forward(&mut data);
        let freq_energy: f32 = data.iter().map(|c| c.magnitude_squared()).sum::<f32>() / 64.0;

        let relative_err = if time_energy > 1e-10 {
            (time_energy - freq_energy).abs() / time_energy
        } else {
            (time_energy - freq_energy).abs()
        };

        prop_assert!(
            relative_err < 1e-4,
            "Parseval violated: time_energy={time_energy}, freq_energy={freq_energy}, relative_err={relative_err}"
        );
    }
}
