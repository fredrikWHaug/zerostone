//! Lightweight benchmark regression tests.
//!
//! Times critical signal processing pipelines and asserts they complete
//! within generous ceilings (5-10x typical baseline). NOT a substitute
//! for criterion benchmarks -- catches catastrophic O(n^2) regressions only.

use std::time::Instant;
use zerostone::float::{self, Float};
use zerostone::probe::ProbeLayout;
use zerostone::sorter::{sort_multichannel, DetectionMode, SortConfig};
use zerostone::spike_sort::MultiChannelEvent;
use zerostone::{BiquadCoeffs, Complex, Fft, IirFilter, ThresholdDetector, WelchPsd, WindowType};

/// Test signal: 1024 samples at 250 Hz.
fn test_signal() -> [Float; 1024] {
    core::array::from_fn(|i| {
        let t = i as Float / 250.0;
        float::sin(2.0 * float::PI * 10.0 * t) + 0.5 * float::sin(2.0 * float::PI * 25.0 * t)
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
    let mut detector: ThresholdDetector<32> = ThresholdDetector::new(3.0, 50);

    let start = Instant::now();
    for _ in 0..100 {
        for i in 0..1024 {
            let mut samples = [0.0 as Float; 32];
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
        let mut psd = [0.0 as Float; 129];
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

/// Multi-channel spike sorting: 2 channels, 5000 samples, full pipeline.
/// Baseline (release): <50 ms. Ceiling: 5000 ms (debug mode safe).
#[test]
fn bench_regression_sort_multichannel() {
    // Simple xorshift RNG for deterministic test data.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn gaussian(&mut self) -> Float {
            let u1 = (self.next_u64() % 1_000_000 + 1) as f64 / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as f64 / 1_000_000.0;
            (libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * std::f64::consts::PI * u2)) as Float
        }
    }

    let n = 5000;
    let config = SortConfig {
        threshold_multiplier: 4.0,
        pre_samples: 4,
        detection_mode: DetectionMode::Amplitude,
        ccg_merge: false,
        ccg_template_corr_threshold: 0.5,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);

    // Generate noisy 2-channel data with injected spikes.
    let mut rng = Rng::new(42);
    let mut base_data = vec![[0.0 as Float; 2]; n];
    for s in base_data.iter_mut() {
        s[0] = rng.gaussian();
        s[1] = rng.gaussian();
    }

    // Inject spikes on ch0 every 150 samples.
    let mut pos = 200;
    while pos + 6 < n {
        for dt in 0..6 {
            let t = (dt as f64 - 2.0) / 1.0;
            base_data[pos + dt][0] += (-10.0 * libm::exp(-0.5 * t * t)) as Float;
        }
        pos += 150;
    }

    // Inject spikes on ch1 every 200 samples.
    pos = 300;
    while pos + 6 < n {
        for dt in 0..6 {
            let t = (dt as f64 - 2.0) / 1.0;
            base_data[pos + dt][1] += (-8.0 * libm::exp(-0.5 * t * t)) as Float;
        }
        pos += 200;
    }

    let start = Instant::now();
    for _ in 0..10 {
        let mut data = base_data.clone();
        let mut scratch = vec![0.0 as Float; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0 as Float; 32]; 200];
        let mut features = vec![[0.0 as Float; 3]; 200];
        let mut labels = vec![0usize; 200];

        let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 5000,
        "sort_multichannel 10x5000x2ch took {}ms (ceiling 5000ms)",
        elapsed.as_millis()
    );
}
