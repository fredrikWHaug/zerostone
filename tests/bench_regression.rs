//! Lightweight benchmark regression tests.
//!
//! Times critical signal processing pipelines and asserts they complete
//! within generous ceilings (5-10x typical baseline). NOT a substitute
//! for criterion benchmarks -- catches catastrophic O(n^2) regressions only.

use std::time::Instant;
use zerostone::{BiquadCoeffs, Complex, Fft, IirFilter, ThresholdDetector, WelchPsd, WindowType};

/// Test signal: 1024 samples at 250 Hz.
fn test_signal() -> [f32; 1024] {
    core::array::from_fn(|i| {
        let t = i as f32 / 250.0;
        libm::sinf(2.0 * std::f32::consts::PI * 10.0 * t)
            + 0.5 * libm::sinf(2.0 * std::f32::consts::PI * 25.0 * t)
    })
}

/// Bandpass filter: 1024 samples through a 2-stage IIR.
/// Baseline (release): <1 ms. Ceiling: 500 ms (debug mode safe).
#[test]
fn bench_regression_bandpass() {
    let signal = test_signal();
    let coeffs = BiquadCoeffs::butterworth_bandpass(250.0, 8.0, 12.0);

    let start = Instant::now();
    for _ in 0..100 {
        let mut filter = IirFilter::<1>::new([coeffs]);
        for &s in &signal {
            std::hint::black_box(filter.process_sample(s));
        }
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "bandpass 100x1024 took {}ms (ceiling 500ms)",
        elapsed.as_millis()
    );
}

/// FFT forward: 1024-point complex FFT.
/// Baseline (release): <1 ms. Ceiling: 500 ms (debug mode safe).
#[test]
fn bench_regression_fft() {
    let signal = test_signal();
    let fft = Fft::<1024>::new();

    let start = Instant::now();
    for _ in 0..100 {
        let mut data: [Complex; 1024] = core::array::from_fn(|i| Complex::from_real(signal[i]));
        fft.forward(&mut data);
        std::hint::black_box(&data);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "FFT 100x1024 took {}ms (ceiling 500ms)",
        elapsed.as_millis()
    );
}

/// Spike detection: 1024 samples through 32-channel detector.
/// Baseline (release): <1 ms. Ceiling: 500 ms (debug mode safe).
#[test]
fn bench_regression_spike_detection() {
    let mut detector: ThresholdDetector<32> = ThresholdDetector::new(3.0, 30);

    let start = Instant::now();
    for _ in 0..100 {
        for i in 0..1024 {
            let mut samples = [0.0f32; 32];
            samples[0] = if i % 100 == 0 { 5.0 } else { 0.1 };
            std::hint::black_box(detector.process_sample(&samples));
        }
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "spike detection 100x1024x32ch took {}ms (ceiling 500ms)",
        elapsed.as_millis()
    );
}

/// Welch PSD: 1024-point with 256-sample segments.
/// Baseline (release): <5 ms. Ceiling: 2000 ms (debug mode safe).
#[test]
fn bench_regression_welch() {
    let signal = test_signal();
    let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);

    let start = Instant::now();
    for _ in 0..100 {
        let mut psd = [0.0f32; 129];
        welch.estimate(&signal, 250.0, &mut psd);
        std::hint::black_box(&psd);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 2000,
        "Welch PSD 100x took {}ms (ceiling 2000ms)",
        elapsed.as_millis()
    );
}
