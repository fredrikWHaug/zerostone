//! Process replay tests: deterministic pipeline outputs frozen as canonical references.
//!
//! These tests ensure that critical signal processing pipelines produce bit-identical
//! output across code changes. Any change to these outputs requires explicit acknowledgment.
//!
//! Convention: commits tagged [refactor] must NOT change any hash in this file.
//!
//! All trig uses `libm` (pure Rust) instead of platform `sin()`/`cos()` to guarantee
//! identical results on macOS ARM64, Linux x86_64, etc.

use zerostone::localize::{center_of_mass, center_of_mass_threshold};
use zerostone::probe::ProbeLayout;
use zerostone::sorter::{sort_multichannel, SortConfig};
use zerostone::spike_sort::{deduplicate_events, detect_spikes_multichannel, MultiChannelEvent};
use zerostone::whitening::{apply_whitening, WhiteningMatrix, WhiteningMode};
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

/// FNV-1a hash of an f64 slice.
fn hash_f64(data: &[f64]) -> u64 {
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

/// Replay 6: Multi-channel detection + deduplication pipeline.
#[test]
fn replay_multichannel_detect_dedup() {
    // Deterministic 2-channel signal: 1000 samples at 250 Hz
    let n = 1000;
    let fs = 250.0f64;
    let mut data = vec![[0.0f64; 2]; n];
    for (i, sample) in data.iter_mut().enumerate().take(n) {
        let t = i as f64 / fs;
        sample[0] = libm::sin(2.0 * std::f64::consts::PI * 8.0 * t);
        sample[1] = libm::sin(2.0 * std::f64::consts::PI * 12.0 * t);
    }

    // Inject deterministic spikes on channel 0
    for &pos in &[100usize, 350, 600, 850] {
        data[pos][0] = -12.0;
        if pos + 1 < n {
            data[pos + 1][0] = -8.0;
        }
        if pos + 2 < n {
            data[pos + 2][0] = 4.0;
        }
    }
    // Inject deterministic spikes on channel 1
    for &pos in &[200usize, 450, 700] {
        data[pos][1] = -15.0;
        if pos + 1 < n {
            data[pos + 1][1] = -10.0;
        }
        if pos + 2 < n {
            data[pos + 2][1] = 5.0;
        }
    }

    // Detect
    let noise = [1.0f64; 2];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        50
    ];
    let n_detected = detect_spikes_multichannel::<2>(&data, 5.0, &noise, 10, &mut events);

    // Dedup
    let probe = ProbeLayout::<2>::linear(25.0);
    let n_dedup = deduplicate_events::<2>(&mut events, n_detected, &probe, 30.0, 5);

    // Hash event data: flatten (sample, channel, amplitude) for each surviving event
    let mut flat = Vec::new();
    for event in events.iter().take(n_dedup) {
        flat.push(event.sample as f64);
        flat.push(event.channel as f64);
        flat.push(event.amplitude);
    }
    let h = hash_f64(&flat);
    assert_eq!(
        h, HASH_MULTICHANNEL_DETECT,
        "multichannel detect+dedup replay changed: got {h:#018x}, expected {HASH_MULTICHANNEL_DETECT:#018x}"
    );
}

/// Replay 7: Whitening matrix computation + application.
#[test]
fn replay_whitening_pipeline() {
    // Generate deterministic 2-channel signal: 100 samples at 250 Hz
    let n = 100;
    let fs = 250.0f64;
    let mut data = [[0.0f64; 2]; 100];
    for (i, sample) in data.iter_mut().enumerate().take(n) {
        let t = i as f64 / fs;
        sample[0] = libm::sin(2.0 * std::f64::consts::PI * 10.0 * t);
        sample[1] = libm::sin(2.0 * std::f64::consts::PI * 15.0 * t);
    }

    // Compute sample covariance
    let mut mean = [0.0f64; 2];
    for sample in data.iter() {
        mean[0] += sample[0];
        mean[1] += sample[1];
    }
    mean[0] /= n as f64;
    mean[1] /= n as f64;

    let mut cov = [[0.0f64; 2]; 2];
    for sample in data.iter() {
        let d0 = sample[0] - mean[0];
        let d1 = sample[1] - mean[1];
        cov[0][0] += d0 * d0;
        cov[0][1] += d0 * d1;
        cov[1][0] += d1 * d0;
        cov[1][1] += d1 * d1;
    }
    let inv = 1.0 / (n - 1) as f64;
    cov[0][0] *= inv;
    cov[0][1] *= inv;
    cov[1][0] *= inv;
    cov[1][1] *= inv;

    // Compute whitening matrix (ZCA, epsilon=1e-6)
    let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6).unwrap();

    // Apply whitening to all samples
    let mut whitened = Vec::with_capacity(n * 2);
    for sample in data.iter() {
        let out = apply_whitening(&wm, sample);
        whitened.push(out[0]);
        whitened.push(out[1]);
    }

    let h = hash_f64(&whitened);
    assert_eq!(
        h, HASH_WHITENING,
        "whitening pipeline replay changed: got {h:#018x}, expected {HASH_WHITENING:#018x}"
    );
}

/// Replay 8: Spike localization from amplitude distributions.
#[test]
fn replay_localization() {
    // Create a deterministic 4-channel linear probe with 25um pitch
    let probe = ProbeLayout::<4>::linear(25.0);
    let positions = *probe.positions();

    // Deterministic amplitude array (simulates a spike strongest on channel 0)
    let amplitudes: [f64; 4] = [10.0, 5.0, 2.0, 1.0];

    // Center-of-mass localization
    let com = center_of_mass(&amplitudes, &positions);

    // Thresholded center-of-mass (threshold=0.3 -- all channels pass)
    let com_thresh = center_of_mass_threshold(&amplitudes, &positions, 0.3).unwrap();

    // Hash both results: com_x, com_y, thresh_x, thresh_y
    let flat = [com[0], com[1], com_thresh[0], com_thresh[1]];
    let h = hash_f64(&flat);
    assert_eq!(
        h, HASH_LOCALIZATION,
        "localization replay changed: got {h:#018x}, expected {HASH_LOCALIZATION:#018x}"
    );
}

/// Replay 9: Full multi-channel sorting pipeline.
#[test]
fn replay_sort_multichannel() {
    // Deterministic 2-channel signal: 1000 samples at 250 Hz (same pattern as detect+dedup)
    let n = 1000;
    let fs = 250.0f64;
    let mut data = vec![[0.0f64; 2]; n];
    for (i, sample) in data.iter_mut().enumerate().take(n) {
        let t = i as f64 / fs;
        sample[0] = libm::sin(2.0 * std::f64::consts::PI * 8.0 * t);
        sample[1] = libm::sin(2.0 * std::f64::consts::PI * 12.0 * t);
    }

    // Inject deterministic spikes on channel 0
    for &pos in &[100usize, 350, 600, 850] {
        data[pos][0] = -12.0;
        if pos + 1 < n {
            data[pos + 1][0] = -8.0;
        }
        if pos + 2 < n {
            data[pos + 2][0] = 4.0;
        }
    }
    // Inject deterministic spikes on channel 1
    for &pos in &[200usize, 450, 700] {
        data[pos][1] = -15.0;
        if pos + 1 < n {
            data[pos + 1][1] = -10.0;
        }
        if pos + 2 < n {
            data[pos + 2][1] = 5.0;
        }
    }

    let config = SortConfig {
        pre_samples: 4,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);

    let mut scratch = vec![0.0f64; n];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        200
    ];
    let mut waveforms = vec![[0.0f64; 32]; 200];
    let mut features = vec![[0.0f64; 3]; 200];
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
    )
    .expect("sort_multichannel failed");

    // Hash: labels (as f64) + n_spikes + n_clusters
    let mut flat: Vec<f64> = labels
        .iter()
        .take(result.n_spikes)
        .map(|&l| l as f64)
        .collect();
    flat.push(result.n_spikes as f64);
    flat.push(result.n_clusters as f64);

    let h = hash_f64(&flat);
    assert_eq!(
        h, HASH_SORT_MULTICHANNEL,
        "sort_multichannel replay changed: got {h:#018x}, expected {HASH_SORT_MULTICHANNEL:#018x}"
    );
}

// Canonical hashes -- regenerate by running tests and updating from failure messages.
// Any change here means pipeline output changed. Commits tagged [refactor] must not change these.
const HASH_BANDPASS: u64 = 0x2cfb75bb77f0f1ec;
const HASH_FFT: u64 = 0x536017a5e587159d;
const HASH_WELCH: u64 = 0x22c723d86dad4098;
const HASH_SPIKE: u64 = 0x271153b6466ce591;
const HASH_CSP: u64 = 0x6beb8f7318b7cb14;
const HASH_MULTICHANNEL_DETECT: u64 = 0x0b237a058d038e32;
const HASH_WHITENING: u64 = 0xf05e0280890d2b3d;
const HASH_LOCALIZATION: u64 = 0xcef05e22f2464b21;
const HASH_SORT_MULTICHANNEL: u64 = 0x640c847077948964;
