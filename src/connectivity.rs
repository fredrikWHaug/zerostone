//! Connectivity metrics for brain signal analysis.
//!
//! This module provides coherence and phase locking value (PLV) for measuring
//! synchronization between brain regions. These are fundamental metrics in
//! BCI research for connectivity analysis — determining how brain areas
//! communicate and synchronize.
//!
//! # Overview
//!
//! - [`coherence`] — Single-window magnitude-squared coherence via FFT
//! - [`spectral_coherence`] — Welch-style averaged coherence (more robust)
//! - [`phase_locking_value`] — Phase synchronization metric from instantaneous phases
//! - [`coherence_frequencies`] — Frequency bin centers for coherence output
//!
//! # Coherence
//!
//! Magnitude-squared coherence measures the linear relationship between two
//! signals at each frequency:
//!
//! ```text
//! Cxy(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))
//! ```
//!
//! where Sxy is the cross-spectral density and Sxx, Syy are the auto-spectral
//! densities. Values range from 0 (no linear relationship) to 1 (perfect
//! linear relationship).
//!
//! # Phase Locking Value
//!
//! PLV measures phase synchronization independent of amplitude:
//!
//! ```text
//! PLV = |mean(exp(j · (φ_a - φ_b)))|
//! ```
//!
//! Values range from 0 (random phase relationship) to 1 (constant phase
//! difference). Requires narrowband-filtered signals and Hilbert transform
//! for instantaneous phase extraction.
//!
//! # Example
//!
//! ```
//! use zerostone::connectivity::{coherence, phase_locking_value};
//! use zerostone::WindowType;
//!
//! // Two identical signals have coherence = 1.0
//! let signal = [0.0f32; 256];
//! let mut coh = [0.0f32; 129]; // N/2 + 1
//! coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);
//! // coh[k] ≈ 1.0 for all bins with signal energy
//!
//! // Phase-locked signals have PLV ≈ 1.0
//! let phases_a = [0.0f32; 100];
//! let phases_b = [0.5f32; 100]; // constant offset
//! let plv = phase_locking_value(&phases_a, &phases_b);
//! assert!((plv - 1.0).abs() < 1e-6); // perfect phase locking
//! ```

use crate::fft::{Complex, Fft};
use crate::window::{window_coefficient, WindowType};

/// Compute magnitude-squared coherence between two signals.
///
/// Uses a single FFT window. For more robust estimates with lower variance,
/// use [`spectral_coherence`] which averages over multiple overlapping segments.
///
/// # Arguments
///
/// * `signal_a` - First signal (length >= N)
/// * `signal_b` - Second signal (length >= N)
/// * `window` - Window function to apply before FFT
/// * `output` - Output buffer for coherence values (length >= N/2 + 1)
///
/// # Output
///
/// Writes N/2 + 1 coherence values in \[0, 1\] to `output`.
///
/// # Panics
///
/// Panics if signals are shorter than N or output is too small.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::coherence;
/// use zerostone::WindowType;
///
/// let mut signal = [0.0f32; 256];
/// for (i, s) in signal.iter_mut().enumerate() {
///     let t = i as f32 / 256.0;
///     *s = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * t);
/// }
///
/// let mut coh = [0.0f32; 129];
/// coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);
///
/// // Identical signals: coherence = 1.0 at the signal frequency
/// assert!(coh[10] > 0.99);
/// ```
pub fn coherence<const N: usize>(
    signal_a: &[f32],
    signal_b: &[f32],
    window: WindowType,
    output: &mut [f32],
) {
    assert!(
        signal_a.len() >= N,
        "signal_a length {} must be >= {}",
        signal_a.len(),
        N
    );
    assert!(
        signal_b.len() >= N,
        "signal_b length {} must be >= {}",
        signal_b.len(),
        N
    );
    let bins = N / 2 + 1;
    assert!(
        output.len() >= bins,
        "output length {} must be >= {}",
        output.len(),
        bins
    );

    let fft = Fft::<N>::new();

    // Window and FFT signal A
    let mut data_a: [Complex; N] = core::array::from_fn(|i| {
        let w = window_coefficient(window, i, N);
        Complex::from_real(signal_a[i] * w)
    });
    fft.forward(&mut data_a);

    // Window and FFT signal B
    let mut data_b: [Complex; N] = core::array::from_fn(|i| {
        let w = window_coefficient(window, i, N);
        Complex::from_real(signal_b[i] * w)
    });
    fft.forward(&mut data_b);

    // Compute coherence: |Sxy|² / (Sxx · Syy)
    for k in 0..bins {
        let sxy = data_a[k].cmul(data_b[k].conj());
        let sxx = data_a[k].magnitude_squared();
        let syy = data_b[k].magnitude_squared();

        let denom = sxx * syy;
        output[k] = if denom > 1e-20 {
            sxy.magnitude_squared() / denom
        } else {
            0.0
        };
    }
}

/// Compute Welch-style averaged coherence between two signals.
///
/// More robust than single-window [`coherence`]. Divides signals into
/// overlapping segments, computes cross- and auto-spectral densities for
/// each segment, averages them, then computes coherence from the averages.
///
/// # Arguments
///
/// * `signal_a` - First signal (length >= N, must equal signal_b length)
/// * `signal_b` - Second signal (length >= N, must equal signal_a length)
/// * `overlap_frac` - Overlap fraction in \[0.0, 1.0). 0.5 = 50% overlap (recommended).
/// * `window` - Window function to apply to each segment
/// * `output_coh` - Output buffer for coherence values (length >= N/2 + 1)
///
/// # Returns
///
/// Number of segments averaged.
///
/// # Panics
///
/// Panics if signals have different lengths, are shorter than N, or output is too small.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::spectral_coherence;
/// use zerostone::WindowType;
///
/// let mut sig_a = [0.0f32; 1024];
/// let mut sig_b = [0.0f32; 1024];
/// for (i, (a, b)) in sig_a.iter_mut().zip(sig_b.iter_mut()).enumerate() {
///     let t = i as f32 / 256.0;
///     *a = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * t);
///     *b = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * t);
/// }
///
/// let mut coh = [0.0f32; 129];
/// let segments = spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);
/// assert!(segments > 1);
/// assert!(coh[10] > 0.99); // 10 Hz bin
/// ```
pub fn spectral_coherence<const N: usize>(
    signal_a: &[f32],
    signal_b: &[f32],
    overlap_frac: f32,
    window: WindowType,
    output_coh: &mut [f32],
) -> usize {
    assert!(
        signal_a.len() >= N,
        "signal_a length {} must be >= {}",
        signal_a.len(),
        N
    );
    assert!(
        signal_b.len() >= N,
        "signal_b length {} must be >= {}",
        signal_b.len(),
        N
    );
    assert_eq!(
        signal_a.len(),
        signal_b.len(),
        "Signals must have equal length"
    );
    let bins = N / 2 + 1;
    assert!(
        output_coh.len() >= bins,
        "output length {} must be >= {}",
        output_coh.len(),
        bins
    );
    assert!(
        (0.0..1.0).contains(&overlap_frac),
        "overlap_frac must be in [0.0, 1.0)"
    );

    let fft = Fft::<N>::new();
    let overlap = (N as f32 * overlap_frac) as usize;
    let hop = N - overlap;
    let signal_len = signal_a.len();
    let num_segments = (signal_len - N) / hop + 1;

    // Accumulators for averaged spectra (use full N arrays, only first `bins` used)
    let mut sxx_acc = [0.0f32; N];
    let mut syy_acc = [0.0f32; N];
    let mut sxy_re_acc = [0.0f32; N];
    let mut sxy_im_acc = [0.0f32; N];

    for seg in 0..num_segments {
        let start = seg * hop;

        let mut data_a: [Complex; N] = core::array::from_fn(|i| {
            let w = window_coefficient(window, i, N);
            Complex::from_real(signal_a[start + i] * w)
        });
        fft.forward(&mut data_a);

        let mut data_b: [Complex; N] = core::array::from_fn(|i| {
            let w = window_coefficient(window, i, N);
            Complex::from_real(signal_b[start + i] * w)
        });
        fft.forward(&mut data_b);

        for k in 0..bins {
            sxx_acc[k] += data_a[k].magnitude_squared();
            syy_acc[k] += data_b[k].magnitude_squared();
            let cross = data_a[k].cmul(data_b[k].conj());
            sxy_re_acc[k] += cross.re;
            sxy_im_acc[k] += cross.im;
        }
    }

    // Compute coherence from averaged spectra
    for k in 0..bins {
        let sxy_mag_sq = sxy_re_acc[k] * sxy_re_acc[k] + sxy_im_acc[k] * sxy_im_acc[k];
        let denom = sxx_acc[k] * syy_acc[k];
        output_coh[k] = if denom > 1e-20 {
            sxy_mag_sq / denom
        } else {
            0.0
        };
    }

    num_segments
}

/// Compute Phase Locking Value between two instantaneous phase arrays.
///
/// PLV measures the consistency of the phase difference between two signals:
///
/// ```text
/// PLV = |mean(exp(j · (φ_a - φ_b)))|
/// ```
///
/// # Arguments
///
/// * `phases_a` - Instantaneous phases of signal A (radians, from Hilbert transform)
/// * `phases_b` - Instantaneous phases of signal B (radians, from Hilbert transform)
///
/// # Returns
///
/// PLV in \[0, 1\] where:
/// - 1.0 = constant phase difference (perfect synchronization)
/// - 0.0 = random phase relationship (no synchronization)
///
/// # Panics
///
/// Panics if phase arrays have different lengths or are empty.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::phase_locking_value;
///
/// // Constant phase difference → PLV = 1.0
/// let phases_a = [0.0f32, 0.5, 1.0, 1.5, 2.0];
/// let phases_b = [0.3f32, 0.8, 1.3, 1.8, 2.3]; // constant 0.3 rad offset
/// let plv = phase_locking_value(&phases_a, &phases_b);
/// assert!((plv - 1.0).abs() < 1e-6);
/// ```
pub fn phase_locking_value(phases_a: &[f32], phases_b: &[f32]) -> f32 {
    assert_eq!(
        phases_a.len(),
        phases_b.len(),
        "Phase arrays must have equal length"
    );
    assert!(!phases_a.is_empty(), "Phase arrays must not be empty");

    let n = phases_a.len();
    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;

    for i in 0..n {
        let diff = (phases_a[i] - phases_b[i]) as f64;
        sum_re += libm::cos(diff);
        sum_im += libm::sin(diff);
    }

    sum_re /= n as f64;
    sum_im /= n as f64;

    libm::sqrt(sum_re * sum_re + sum_im * sum_im) as f32
}

/// Compute frequency bin centers for coherence output.
///
/// # Arguments
///
/// * `sample_rate` - Sample rate in Hz
/// * `output` - Output buffer (length >= N/2 + 1)
///
/// # Example
///
/// ```
/// use zerostone::connectivity::coherence_frequencies;
///
/// let mut freqs = [0.0f32; 129];
/// coherence_frequencies::<256>(256.0, &mut freqs);
/// assert!((freqs[0] - 0.0).abs() < 1e-6);
/// assert!((freqs[1] - 1.0).abs() < 1e-6);
/// assert!((freqs[128] - 128.0).abs() < 1e-6);
/// ```
pub fn coherence_frequencies<const N: usize>(sample_rate: f32, output: &mut [f32]) {
    let bins = N / 2 + 1;
    assert!(
        output.len() >= bins,
        "output length {} must be >= {}",
        output.len(),
        bins
    );
    let freq_res = sample_rate / N as f32;
    for (k, val) in output[..bins].iter_mut().enumerate() {
        *val = k as f32 * freq_res;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_coherence_identical_signals() {
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / 256.0;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut coh = [0.0f32; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);

        // At the signal frequency (bin 10), coherence should be 1.0
        assert!(
            coh[10] > 0.99,
            "Coherence at signal frequency should be ~1.0, got {}",
            coh[10]
        );
    }

    #[test]
    fn test_coherence_identical_all_bins() {
        // For identical signals, all bins with energy should have coherence 1.0
        let mut signal = [0.0f32; 256];
        let mut state: u32 = 42;
        for s in signal.iter_mut() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *s = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut coh = [0.0f32; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);

        // All bins should be ~1.0 (identical signals)
        for (k, &c) in coh[1..128].iter().enumerate() {
            assert!(
                c > 0.99,
                "Coherence at bin {} should be ~1.0, got {}",
                k + 1,
                c
            );
        }
    }

    #[test]
    fn test_coherence_single_window_bias() {
        // Single-window magnitude-squared coherence is always 1.0 for any two
        // non-zero signals (known property). This is why spectral_coherence
        // with Welch averaging is needed for meaningful estimates.
        let mut sig_a = [0.0f32; 256];
        let mut sig_b = [0.0f32; 256];
        let mut state_a: u32 = 42;
        let mut state_b: u32 = 99999;
        for i in 0..256 {
            state_a = state_a.wrapping_mul(1103515245).wrapping_add(12345);
            state_b = state_b.wrapping_mul(1103515245).wrapping_add(12345);
            sig_a[i] = (state_a as f32 / u32::MAX as f32) * 2.0 - 1.0;
            sig_b[i] = (state_b as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut coh = [0.0f32; 129];
        coherence::<256>(&sig_a, &sig_b, WindowType::Hann, &mut coh);

        // Single-window coherence is biased to 1.0 — this is expected behavior
        let mean_coh: f32 = coh[1..128].iter().sum::<f32>() / 127.0;
        assert!(
            mean_coh > 0.99,
            "Single-window coherence should be ~1.0 (known bias), got {}",
            mean_coh
        );
    }

    #[test]
    fn test_coherence_range() {
        // Coherence values must be in [0, 1]
        let mut sig_a = [0.0f32; 256];
        let mut sig_b = [0.0f32; 256];
        for (i, (a, b)) in sig_a.iter_mut().zip(sig_b.iter_mut()).enumerate() {
            let t = i as f32 / 256.0;
            *a = libm::sinf(2.0 * PI * 10.0 * t) + libm::sinf(2.0 * PI * 30.0 * t);
            *b = libm::sinf(2.0 * PI * 10.0 * t) + libm::cosf(2.0 * PI * 50.0 * t);
        }

        let mut coh = [0.0f32; 129];
        coherence::<256>(&sig_a, &sig_b, WindowType::Hann, &mut coh);

        for (k, &c) in coh.iter().enumerate() {
            assert!(c >= 0.0, "Coherence at bin {} is negative: {}", k, c);
            assert!(c <= 1.0 + 1e-6, "Coherence at bin {} exceeds 1.0: {}", k, c);
        }
    }

    #[test]
    fn test_spectral_coherence_identical_signals() {
        let mut signal = [0.0f32; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / 256.0;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut coh = [0.0f32; 129];
        let segments = spectral_coherence::<256>(&signal, &signal, 0.5, WindowType::Hann, &mut coh);

        assert!(segments > 1, "Should have multiple segments");
        assert!(
            coh[10] > 0.99,
            "Spectral coherence at 10 Hz should be ~1.0, got {}",
            coh[10]
        );
    }

    #[test]
    fn test_spectral_coherence_independent_noise() {
        let mut sig_a = [0.0f32; 2048];
        let mut sig_b = [0.0f32; 2048];
        let mut state_a: u32 = 42;
        let mut state_b: u32 = 99999;
        for i in 0..2048 {
            state_a = state_a.wrapping_mul(1103515245).wrapping_add(12345);
            state_b = state_b.wrapping_mul(1103515245).wrapping_add(12345);
            sig_a[i] = (state_a as f32 / u32::MAX as f32) * 2.0 - 1.0;
            sig_b[i] = (state_b as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut coh = [0.0f32; 129];
        let segments = spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        assert!(segments > 3);

        // With multiple segments, averaged coherence of independent noise should be low
        let mean_coh: f32 = coh[1..128].iter().sum::<f32>() / 127.0;
        assert!(
            mean_coh < 0.3,
            "Mean spectral coherence of independent noise should be low, got {}",
            mean_coh
        );
    }

    #[test]
    fn test_spectral_coherence_range() {
        let mut sig_a = [0.0f32; 1024];
        let mut sig_b = [0.0f32; 1024];
        let mut state: u32 = 42;
        for i in 0..1024 {
            let t = i as f32 / 256.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            sig_a[i] = libm::sinf(2.0 * PI * 10.0 * t) + noise * 0.3;
            sig_b[i] = libm::sinf(2.0 * PI * 10.0 * t) + noise * 0.5;
        }

        let mut coh = [0.0f32; 129];
        spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        for (k, &c) in coh.iter().enumerate() {
            assert!(c >= 0.0, "Coherence at bin {} is negative: {}", k, c);
            assert!(c <= 1.0 + 1e-6, "Coherence at bin {} exceeds 1.0: {}", k, c);
        }
    }

    #[test]
    fn test_plv_constant_phase_difference() {
        // Constant phase difference → PLV = 1.0
        let phases_a: [f32; 100] = core::array::from_fn(|i| i as f32 * 0.1);
        let phases_b: [f32; 100] = core::array::from_fn(|i| i as f32 * 0.1 + 0.5);

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(
            (plv - 1.0).abs() < 1e-6,
            "PLV with constant phase difference should be 1.0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_zero_phase_difference() {
        let phases: [f32; 100] = core::array::from_fn(|i| i as f32 * 0.1);
        let plv = phase_locking_value(&phases, &phases);
        assert!(
            (plv - 1.0).abs() < 1e-6,
            "PLV with zero phase difference should be 1.0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_random_phases() {
        // Uniformly distributed phase differences → PLV ≈ 0
        let mut phases_a = [0.0f32; 1000];
        let mut phases_b = [0.0f32; 1000];
        let mut state: u32 = 42;
        for i in 0..1000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            phases_a[i] = (state as f32 / u32::MAX as f32) * 2.0 * PI;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            phases_b[i] = (state as f32 / u32::MAX as f32) * 2.0 * PI;
        }

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(
            plv < 0.15,
            "PLV with random phases should be ~0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_range() {
        // PLV should always be in [0, 1]
        let phases_a = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let phases_b = [0.5f32, 1.5, 0.0, 3.0, 5.0];

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(plv >= 0.0, "PLV must be >= 0, got {}", plv);
        assert!(plv <= 1.0 + 1e-6, "PLV must be <= 1, got {}", plv);
    }

    #[test]
    #[should_panic(expected = "Phase arrays must have equal length")]
    fn test_plv_unequal_lengths() {
        let a = [0.0f32; 5];
        let b = [0.0f32; 3];
        phase_locking_value(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Phase arrays must not be empty")]
    fn test_plv_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        phase_locking_value(&a, &b);
    }

    #[test]
    fn test_coherence_frequencies() {
        let mut freqs = [0.0f32; 129];
        coherence_frequencies::<256>(256.0, &mut freqs);

        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - 1.0).abs() < 1e-6);
        assert!((freqs[128] - 128.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_frequencies_250hz() {
        let mut freqs = [0.0f32; 129];
        coherence_frequencies::<256>(250.0, &mut freqs);

        let freq_res = 250.0 / 256.0;
        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - freq_res).abs() < 1e-4);
        assert!((freqs[128] - 125.0).abs() < 1e-3);
    }

    #[test]
    fn test_spectral_coherence_single_segment() {
        // With exactly N samples, spectral coherence should equal single-window coherence
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / 256.0;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut coh_single = [0.0f32; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh_single);

        let mut coh_welch = [0.0f32; 129];
        let segments =
            spectral_coherence::<256>(&signal, &signal, 0.5, WindowType::Hann, &mut coh_welch);

        assert_eq!(segments, 1);

        for k in 0..129 {
            assert!(
                (coh_single[k] - coh_welch[k]).abs() < 1e-5,
                "Bin {}: single={} vs welch={}",
                k,
                coh_single[k],
                coh_welch[k]
            );
        }
    }

    #[test]
    fn test_coherence_shared_component() {
        // Two signals sharing a common component should have high coherence at that frequency
        let mut sig_a = [0.0f32; 1024];
        let mut sig_b = [0.0f32; 1024];
        let mut state: u32 = 42;
        for i in 0..1024 {
            let t = i as f32 / 256.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise_a = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise_b = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;

            let shared = libm::sinf(2.0 * PI * 10.0 * t);
            sig_a[i] = shared + noise_a * 0.1;
            sig_b[i] = shared + noise_b * 0.1;
        }

        let mut coh = [0.0f32; 129];
        spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        // High coherence at 10 Hz (bin 10)
        assert!(
            coh[10] > 0.8,
            "Coherence at shared frequency should be high, got {}",
            coh[10]
        );
    }
}
