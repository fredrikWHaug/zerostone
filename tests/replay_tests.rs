//! Process replay tests: deterministic pipeline outputs frozen as canonical references.
//!
//! These tests ensure that critical signal processing pipelines produce bit-identical
//! output across code changes. Any change to these outputs requires explicit acknowledgment.
//!
//! Convention: commits tagged [refactor] must NOT change any hash in this file.
//!
//! All trig uses `libm` (pure Rust) instead of platform `sin()`/`cos()` to guarantee
//! identical results on macOS ARM64, Linux x86_64, etc.

use zerostone::{BiquadCoeffs, Complex, Fft, IirFilter, ThresholdDetector, WelchPsd, WindowType};

/// FNV-1a hash of an f32 slice -- detects any bit-level change in output.
fn hash_f32(data: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in data {
        for &b in &v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Deterministic test signal: 10 Hz + 25 Hz sinusoids at 250 Hz sample rate.
/// Uses libm::sinf for cross-platform bit-exact reproducibility.
fn test_signal_1024() -> [f32; 1024] {
    core::array::from_fn(|i| {
        let t = i as f32 / 250.0;
        libm::sinf(2.0 * std::f32::consts::PI * 10.0 * t)
            + 0.5 * libm::sinf(2.0 * std::f32::consts::PI * 25.0 * t)
    })
}

/// Replay 1: Butterworth bandpass 8-12 Hz on the test signal.
#[test]
fn replay_bandpass_filter() {
    let signal = test_signal_1024();
    let coeffs = BiquadCoeffs::butterworth_bandpass(250.0, 8.0, 12.0);
    let mut filter = IirFilter::<1>::new([coeffs]);

    let mut output = [0.0f32; 1024];
    for (i, &s) in signal.iter().enumerate() {
        output[i] = filter.process_sample(s);
    }

    let h = hash_f32(&output);
    assert_eq!(
        h, HASH_BANDPASS,
        "bandpass replay changed: got {h:#018x}, expected {HASH_BANDPASS:#018x}"
    );
}

/// Replay 2: FFT magnitude spectrum of the test signal.
#[test]
fn replay_fft_magnitude() {
    let signal = test_signal_1024();
    let fft = Fft::<1024>::new();
    let mut data: [Complex; 1024] = core::array::from_fn(|i| Complex::from_real(signal[i]));
    fft.forward(&mut data);

    let mut magnitudes = [0.0f32; 513]; // N/2 + 1
    for (i, mag) in magnitudes.iter_mut().enumerate() {
        *mag = data[i].magnitude();
    }

    let h = hash_f32(&magnitudes);
    assert_eq!(
        h, HASH_FFT,
        "FFT replay changed: got {h:#018x}, expected {HASH_FFT:#018x}"
    );
}

/// Replay 3: Spike detection on a signal with known threshold crossings.
#[test]
fn replay_spike_detection() {
    // Signal with deliberate spikes at known positions
    let mut signal = [0.0f32; 1024];
    for (i, s) in signal.iter_mut().enumerate() {
        let t = i as f32 / 250.0;
        *s = libm::sinf(2.0 * std::f32::consts::PI * 5.0 * t);
    }
    // Inject spikes
    signal[100] = 5.0;
    signal[300] = -5.0;
    signal[500] = 5.0;
    signal[700] = -5.0;
    signal[900] = 5.0;

    let mut detector: ThresholdDetector<1> = ThresholdDetector::new(3.0, 50);
    let mut events = Vec::new();
    for (i, &s) in signal.iter().enumerate() {
        if let Some(evt) = detector.process_sample(&[s]) {
            events.push((i as f32, evt.amplitude));
        }
    }

    // Hash the event positions and amplitudes
    let flat: Vec<f32> = events.iter().flat_map(|(pos, amp)| [*pos, *amp]).collect();
    let h = hash_f32(&flat);
    assert_eq!(
        h, HASH_SPIKE,
        "spike detection replay changed: got {h:#018x}, expected {HASH_SPIKE:#018x}"
    );
}

/// Replay 4: Welch PSD of the test signal.
#[test]
fn replay_welch_psd() {
    let signal = test_signal_1024();
    let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
    let mut psd = [0.0f32; 129]; // 256/2 + 1
    welch.estimate(&signal, 250.0, &mut psd);

    let h = hash_f32(&psd);
    assert_eq!(
        h, HASH_WELCH,
        "Welch PSD replay changed: got {h:#018x}, expected {HASH_WELCH:#018x}"
    );
}

/// Replay 5: CSP spatial filter extraction from known covariance structure.
#[test]
fn replay_csp_filters() {
    use zerostone::{AdaptiveCsp, UpdateConfig};

    let config = UpdateConfig {
        min_samples: 10,
        update_interval: 0,
        regularization: 1e-6,
        max_eigen_iters: 100,
        eigen_tol: 1e-12,
    };

    // 4 channels, 4x4=16, extract 2 filters, 2x4=8
    let mut csp: AdaptiveCsp<4, 16, 2, 8> = AdaptiveCsp::new(config);

    // Deterministic training data: class 1 has power in channels 0-1, class 2 in channels 2-3
    let mut class1_trial = [[0.0f64; 4]; 100];
    let mut class2_trial = [[0.0f64; 4]; 100];
    for i in 0..100 {
        let t = i as f64 / 250.0;
        let s1 = libm::sin(2.0 * std::f64::consts::PI * 10.0 * t);
        let s2 = libm::sin(2.0 * std::f64::consts::PI * 12.0 * t);
        class1_trial[i] = [s1, s1 * 0.8, 0.1 * s2, 0.05 * s2];
        class2_trial[i] = [0.1 * s1, 0.05 * s1, s2, s2 * 0.8];
    }

    csp.update_class1(&class1_trial);
    csp.update_class2(&class2_trial);
    csp.recompute_filters()
        .expect("CSP filter computation failed");

    let filters = csp.filters().expect("CSP filters not ready");
    // Hash the filter coefficients (f64 -> pairs of f32 for hashing)
    let as_f32: Vec<f32> = filters.iter().map(|&v| v as f32).collect();
    let h = hash_f32(&as_f32);
    assert_eq!(
        h, HASH_CSP,
        "CSP replay changed: got {h:#018x}, expected {HASH_CSP:#018x}"
    );
}

// Canonical hashes -- regenerate by running tests and updating from failure messages.
// Any change here means pipeline output changed. Commits tagged [refactor] must not change these.
const HASH_BANDPASS: u64 = 0x2cfb75bb77f0f1ec;
const HASH_FFT: u64 = 0x536017a5e587159d;
const HASH_WELCH: u64 = 0x22c723d86dad4098;
const HASH_SPIKE: u64 = 0x271153b6466ce591;
const HASH_CSP: u64 = 0x6beb8f7318b7cb14;
