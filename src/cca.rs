//! Canonical Correlation Analysis (CCA) for SSVEP-based BCIs.
//!
//! CCA finds linear combinations of two sets of multivariate signals that are
//! maximally correlated. For SSVEP detection, it correlates multi-channel EEG
//! with sinusoidal reference templates at target frequencies.
//!
//! # Algorithm
//!
//! Given signals X (T x C) and references Y (T x H):
//! 1. Compute covariance matrices Cxx, Cyy, and cross-covariance Cxy
//! 2. Whiten: K = Lx^{-1} * Cxy * Ly^{-T} where Lx = chol(Cxx), Ly = chol(Cyy)
//! 3. Eigendecompose K^T K (H x H symmetric matrix)
//! 4. Canonical correlations = sqrt(eigenvalues)
//!
//! # SSVEP Detection
//!
//! SSVEP (Steady-State Visual Evoked Potential) BCIs detect which visual stimulus
//! frequency the user attends to. Reference signals are sinusoids at target
//! frequencies with harmonics: sin(2*pi*f*t), cos(2*pi*f*t), sin(2*pi*2f*t), ...
//!
//! The frequency with the highest canonical correlation is the detected target.
//!
//! # References
//!
//! - Lin et al. (2007): "Frequency recognition based on CCA for SSVEP-based BCIs"
//! - Chen et al. (2015): "Filter bank CCA for SSVEP frequency recognition"

use crate::linalg::{LinalgError, Matrix};
use core::f64::consts::PI;

/// Errors that can occur during CCA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcaError {
    /// Not enough samples for CCA
    InsufficientSamples,

    /// Signal and reference length mismatch
    LengthMismatch,

    /// Linear algebra operation failed
    LinalgFailed,

    /// No target frequencies provided
    NoTargetFrequencies,
}

/// Compute canonical correlations between two sets of multivariate signals.
///
/// Given signals X (T x C) and references Y (T x H), computes canonical
/// correlations using the whitened eigenvalue approach. Returns the number
/// of canonical correlations computed (min(C, H)).
///
/// # Type Parameters
///
/// * `C` - Number of signal channels (EEG channels)
/// * `MC` - Signal covariance matrix size (must equal C * C)
/// * `H` - Number of reference components (2 * n_harmonics)
/// * `MH` - Reference covariance matrix size (must equal H * H)
///
/// # Arguments
///
/// * `signals` - Signal data, T samples of C channels each
/// * `references` - Reference data, T samples of H components each
/// * `correlations` - Output buffer for canonical correlations (at least min(C,H) elements)
/// * `regularization` - Tikhonov regularization parameter (e.g., 1e-6)
///
/// # Returns
///
/// Number of canonical correlations written to output buffer.
///
/// # Example
///
/// ```
/// use zerostone::cca::cca;
///
/// // 2 channels, 4 reference components (2 harmonics)
/// let signals: Vec<[f64; 2]> = (0..100).map(|t| {
///     let time = t as f64 / 250.0;
///     [libm::sin(2.0 * core::f64::consts::PI * 10.0 * time),
///      libm::cos(2.0 * core::f64::consts::PI * 10.0 * time)]
/// }).collect();
/// let references: Vec<[f64; 4]> = (0..100).map(|t| {
///     let time = t as f64 / 250.0;
///     [libm::sin(2.0 * core::f64::consts::PI * 10.0 * time),
///      libm::cos(2.0 * core::f64::consts::PI * 10.0 * time),
///      libm::sin(2.0 * core::f64::consts::PI * 20.0 * time),
///      libm::cos(2.0 * core::f64::consts::PI * 20.0 * time)]
/// }).collect();
/// let mut correlations = [0.0; 2];
/// let n = cca::<2, 4, 4, 16>(&signals, &references, &mut correlations, 1e-6).unwrap();
/// assert!(correlations[0] > 0.9); // High correlation expected
/// ```
pub fn cca<const C: usize, const MC: usize, const H: usize, const MH: usize>(
    signals: &[[f64; C]],
    references: &[[f64; H]],
    correlations: &mut [f64],
    regularization: f64,
) -> Result<usize, LinalgError> {
    assert!(MC == C * C, "MC must equal C * C");
    assert!(MH == H * H, "MH must equal H * H");

    let t = signals.len();
    if t != references.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    if t < 2 {
        return Err(LinalgError::NumericalInstability);
    }

    let n_corr = if C < H { C } else { H };

    // 1. Compute means
    let mut mean_x = [0.0; C];
    let mut mean_y = [0.0; H];
    for i in 0..t {
        for c in 0..C {
            mean_x[c] += signals[i][c];
        }
        for h in 0..H {
            mean_y[h] += references[i][h];
        }
    }
    let t_f64 = t as f64;
    for val in mean_x.iter_mut() {
        *val /= t_f64;
    }
    for val in mean_y.iter_mut() {
        *val /= t_f64;
    }

    // 2. Compute covariance matrices and cross-covariance
    let mut cxx = Matrix::<C, MC>::zeros();
    let mut cyy = Matrix::<H, MH>::zeros();
    // Cross-covariance stored as H columns of C-vectors: cxy_cols[h][c] = Cxy[c][h]
    let mut cxy_cols = [[0.0; C]; H];

    for i in 0..t {
        // Centered samples
        let mut dx = [0.0; C];
        let mut dy = [0.0; H];
        for c in 0..C {
            dx[c] = signals[i][c] - mean_x[c];
        }
        for h in 0..H {
            dy[h] = references[i][h] - mean_y[h];
        }

        // Accumulate Cxx = sum(dx * dx^T)
        for r in 0..C {
            for c in 0..C {
                let cur = cxx.get(r, c);
                cxx.set(r, c, cur + dx[r] * dx[c]);
            }
        }

        // Accumulate Cyy = sum(dy * dy^T)
        for r in 0..H {
            for c in 0..H {
                let cur = cyy.get(r, c);
                cyy.set(r, c, cur + dy[r] * dy[c]);
            }
        }

        // Accumulate Cxy: cxy_cols[h][c] += dx[c] * dy[h]
        for h in 0..H {
            for c in 0..C {
                cxy_cols[h][c] += dx[c] * dy[h];
            }
        }
    }

    // Normalize by T-1
    let scale = 1.0 / (t_f64 - 1.0);
    for r in 0..C {
        for c in 0..C {
            let cur = cxx.get(r, c);
            cxx.set(r, c, cur * scale);
        }
    }
    for r in 0..H {
        for c in 0..H {
            let cur = cyy.get(r, c);
            cyy.set(r, c, cur * scale);
        }
    }
    for col in cxy_cols.iter_mut() {
        for val in col.iter_mut() {
            *val *= scale;
        }
    }

    // 3. Regularize
    cxx.add_diagonal(regularization);
    cyy.add_diagonal(regularization);

    // 4. Cholesky decomposition
    let lx = cxx.cholesky()?;
    let ly = cyy.cholesky()?;

    // 5. Compute W = Lx^{-1} * Cxy (H columns of C-vectors)
    // w_cols[h] = Lx.forward_substitute(cxy_cols[h])
    let mut w_cols = [[0.0; C]; H];
    for (w_col, cxy_col) in w_cols.iter_mut().zip(cxy_cols.iter()) {
        *w_col = lx.forward_substitute(cxy_col);
    }

    // 6. Compute M = K^T K where K = W * Ly^{-T}
    // Process one row of K at a time to avoid storing the full C x H matrix.
    // For each row c of W:
    //   Extract w_row[h] = W[c][h] = w_cols[h][c]
    //   k_row = Ly.forward_substitute(w_row)  (gives Ly^{-1} * w_row)
    //   M += k_row * k_row^T (outer product)
    let mut m = Matrix::<H, MH>::zeros();
    #[allow(clippy::needless_range_loop)]
    for c in 0..C {
        let mut w_row = [0.0; H];
        for h in 0..H {
            w_row[h] = w_cols[h][c];
        }

        let k_row = ly.forward_substitute(&w_row);

        // Accumulate outer product
        for i in 0..H {
            for j in 0..H {
                let cur = m.get(i, j);
                m.set(i, j, cur + k_row[i] * k_row[j]);
            }
        }
    }

    // 7. Eigendecompose M (H x H symmetric PSD matrix)
    let eigen = m.eigen_symmetric(30, 1e-10)?;

    // 8. Canonical correlations = sqrt(eigenvalues), clamped to [0, 1]
    for (corr, &ev) in correlations
        .iter_mut()
        .zip(eigen.eigenvalues.iter())
        .take(n_corr)
    {
        if ev > 0.0 {
            let rho = libm::sqrt(ev);
            *corr = if rho > 1.0 { 1.0 } else { rho };
        } else {
            *corr = 0.0;
        }
    }

    Ok(n_corr)
}

/// Fill a buffer with SSVEP sinusoidal reference signals.
///
/// Generates sin/cos pairs at the fundamental frequency and its harmonics:
/// `[sin(2*pi*f*t), cos(2*pi*f*t), sin(2*pi*2f*t), cos(2*pi*2f*t), ...]`
///
/// # Type Parameters
///
/// * `H` - Number of reference components (must equal 2 * n_harmonics)
///
/// # Arguments
///
/// * `sample_rate` - Sampling frequency in Hz
/// * `frequency` - Fundamental SSVEP frequency in Hz
/// * `output` - Output buffer, each element is one time sample with H components
///
/// # Example
///
/// ```
/// use zerostone::cca::fill_ssvep_references;
///
/// // Generate references for 10 Hz with 2 harmonics (H=4) over 100 samples
/// let mut refs = [[0.0f64; 4]; 100];
/// fill_ssvep_references::<4>(250.0, 10.0, &mut refs);
/// // refs[t] = [sin(2*pi*10*t/250), cos(2*pi*10*t/250),
/// //            sin(2*pi*20*t/250), cos(2*pi*20*t/250)]
/// ```
pub fn fill_ssvep_references<const H: usize>(
    sample_rate: f64,
    frequency: f64,
    output: &mut [[f64; H]],
) {
    assert!(H >= 2, "Need at least 2 reference components (1 harmonic)");
    assert!(H.is_multiple_of(2), "H must be even (sin/cos pairs)");

    let n_harmonics = H / 2;

    for (t, sample) in output.iter_mut().enumerate() {
        let time = t as f64 / sample_rate;
        for k in 0..n_harmonics {
            let freq = frequency * (k + 1) as f64;
            let angle = 2.0 * PI * freq * time;
            sample[2 * k] = libm::sin(angle);
            sample[2 * k + 1] = libm::cos(angle);
        }
    }
}

/// Detect SSVEP frequency using CCA.
///
/// Runs CCA between signals and sinusoidal references at each target frequency,
/// returning the frequency index with the highest canonical correlation.
///
/// This is the core SSVEP detection function. The caller must provide a buffer
/// for reference signals (stack-allocated, size T x H).
///
/// # Type Parameters
///
/// * `C` - Number of EEG channels
/// * `MC` - C * C
/// * `H` - Number of reference components (2 * n_harmonics)
/// * `MH` - H * H
///
/// # Arguments
///
/// * `signals` - EEG data, T samples of C channels
/// * `sample_rate` - Sampling frequency in Hz
/// * `target_frequencies` - Target SSVEP frequencies to test
/// * `ref_buffer` - Pre-allocated buffer for reference signals (same length as signals)
/// * `regularization` - Tikhonov regularization parameter
///
/// # Returns
///
/// `(best_frequency_index, max_canonical_correlation)`
pub fn ssvep_detect<const C: usize, const MC: usize, const H: usize, const MH: usize>(
    signals: &[[f64; C]],
    sample_rate: f64,
    target_frequencies: &[f64],
    ref_buffer: &mut [[f64; H]],
    regularization: f64,
) -> Result<(usize, f64), CcaError> {
    if target_frequencies.is_empty() {
        return Err(CcaError::NoTargetFrequencies);
    }
    if signals.len() < 2 {
        return Err(CcaError::InsufficientSamples);
    }
    if signals.len() != ref_buffer.len() {
        return Err(CcaError::LengthMismatch);
    }

    let mut best_idx = 0;
    let mut best_corr = -1.0;
    let mut correlations = [0.0; H]; // max H correlations per frequency

    for (freq_idx, &freq) in target_frequencies.iter().enumerate() {
        // Generate reference signals for this frequency
        fill_ssvep_references::<H>(sample_rate, freq, ref_buffer);

        // Run CCA
        match cca::<C, MC, H, MH>(signals, ref_buffer, &mut correlations, regularization) {
            Ok(n) => {
                if n > 0 && correlations[0] > best_corr {
                    best_corr = correlations[0];
                    best_idx = freq_idx;
                }
            }
            Err(_) => {
                // Skip frequencies that fail (e.g., singular covariance)
                continue;
            }
        }
    }

    if best_corr < 0.0 {
        return Err(CcaError::LinalgFailed);
    }

    Ok((best_idx, best_corr))
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    // Helper: generate synthetic SSVEP signal
    fn generate_ssvep_signal<const C: usize>(
        sample_rate: f64,
        n_samples: usize,
        frequency: f64,
        snr_linear: f64,
    ) -> Vec<[f64; C]> {
        let mut signals = vec![[0.0; C]; n_samples];
        // Simple LCG for deterministic pseudo-random noise
        let mut rng_state: u64 = 12345;

        for (t, sample) in signals.iter_mut().enumerate() {
            let time = t as f64 / sample_rate;
            for (c, val) in sample.iter_mut().enumerate() {
                // LCG pseudo-random
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                // Scale noise and different phase per channel
                let phase = c as f64 * 0.3;
                let signal = libm::sin(2.0 * PI * frequency * time + phase);
                *val = signal * snr_linear + noise;
            }
        }
        signals
    }

    #[test]
    fn test_fill_ssvep_references_basic() {
        let mut refs = [[0.0f64; 4]; 250];
        fill_ssvep_references::<4>(250.0, 10.0, &mut refs);

        // At t=0, all sin should be 0, all cos should be 1
        assert!(refs[0][0].abs() < 1e-10); // sin(0) = 0
        assert!((refs[0][1] - 1.0).abs() < 1e-10); // cos(0) = 1
        assert!(refs[0][2].abs() < 1e-10); // sin(0) = 0
        assert!((refs[0][3] - 1.0).abs() < 1e-10); // cos(0) = 1

        // Check frequency: at t=25 samples (0.1 sec), sin(2*pi*10*0.1) = sin(2*pi) = 0
        assert!(refs[25][0].abs() < 1e-10);
    }

    #[test]
    fn test_fill_ssvep_references_correct_harmonics() {
        let mut refs = [[0.0f64; 6]; 1000];
        fill_ssvep_references::<6>(1000.0, 10.0, &mut refs);

        // 3 harmonics: 10 Hz, 20 Hz, 30 Hz
        // At t=250 samples (0.25 sec):
        // sin(2*pi*10*0.25) = sin(5*pi/2) = 1 (approx, actually sin(pi/2) for the period)
        let t = 250;
        let time = t as f64 / 1000.0; // 0.25 sec
        let expected_sin_10 = libm::sin(2.0 * PI * 10.0 * time);
        let expected_cos_10 = libm::cos(2.0 * PI * 10.0 * time);
        let expected_sin_20 = libm::sin(2.0 * PI * 20.0 * time);

        assert!((refs[t][0] - expected_sin_10).abs() < 1e-10);
        assert!((refs[t][1] - expected_cos_10).abs() < 1e-10);
        assert!((refs[t][2] - expected_sin_20).abs() < 1e-10);
    }

    #[test]
    fn test_cca_perfect_correlation() {
        // When signals ARE the references, correlation should be ~1.0
        let sample_rate = 250.0;
        let freq = 12.0;

        let mut refs = [[0.0f64; 4]; 200];
        fill_ssvep_references::<4>(sample_rate, freq, &mut refs);

        // Use first 2 reference columns as "signals"
        let signals: Vec<[f64; 2]> = refs.iter().map(|r| [r[0], r[1]]).collect();

        let mut correlations = [0.0; 2];
        let n_corr = cca::<2, 4, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        assert_eq!(n_corr, 2);
        assert!(
            correlations[0] > 0.99,
            "Expected near-perfect correlation, got {}",
            correlations[0]
        );
    }

    #[test]
    fn test_cca_uncorrelated_signals() {
        // When signals are at a different frequency than references
        let n = 500;
        let sample_rate = 250.0;

        // Signals at 10 Hz
        let signals: Vec<[f64; 2]> = (0..n)
            .map(|t| {
                let time = t as f64 / sample_rate;
                [
                    libm::sin(2.0 * PI * 10.0 * time),
                    libm::cos(2.0 * PI * 10.0 * time),
                ]
            })
            .collect();

        // References at 37 Hz (incommensurate with 10 Hz)
        let mut refs = [[0.0f64; 4]; 500];
        fill_ssvep_references::<4>(sample_rate, 37.0, &mut refs);

        let mut correlations = [0.0; 2];
        let _ = cca::<2, 4, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        // Correlation should be low (not exactly 0 due to finite samples)
        assert!(
            correlations[0] < 0.3,
            "Expected low correlation, got {}",
            correlations[0]
        );
    }

    #[test]
    fn test_cca_frequency_discrimination() {
        // Generate SSVEP at 12 Hz, test CCA against 8, 10, 12, 15 Hz
        // 12 Hz should have the highest correlation
        let n = 500;
        let sample_rate = 250.0;
        let target_freq = 12.0;

        let signals: Vec<[f64; 4]> = (0..n)
            .map(|t| {
                let time = t as f64 / sample_rate;
                [
                    libm::sin(2.0 * PI * target_freq * time) + 0.1,
                    libm::cos(2.0 * PI * target_freq * time) + 0.2,
                    libm::sin(2.0 * PI * target_freq * time + 0.5),
                    libm::cos(2.0 * PI * target_freq * time + 0.8),
                ]
            })
            .collect();

        let test_freqs = [8.0, 10.0, 12.0, 15.0];
        let mut best_idx = 0;
        let mut best_corr = 0.0;

        for (idx, &freq) in test_freqs.iter().enumerate() {
            let mut refs = [[0.0f64; 4]; 500];
            fill_ssvep_references::<4>(sample_rate, freq, &mut refs);

            let mut correlations = [0.0; 4];
            let _ = cca::<4, 16, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

            if correlations[0] > best_corr {
                best_corr = correlations[0];
                best_idx = idx;
            }
        }

        assert_eq!(
            best_idx, 2,
            "Should detect 12 Hz (index 2), got {}",
            best_idx
        );
        assert!(
            best_corr > 0.9,
            "Correlation should be high, got {}",
            best_corr
        );
    }

    #[test]
    fn test_ssvep_detect_basic() {
        let n = 500;
        let sample_rate = 250.0;
        let target_freq = 10.0;

        let signals: Vec<[f64; 4]> = (0..n)
            .map(|t| {
                let time = t as f64 / sample_rate;
                [
                    libm::sin(2.0 * PI * target_freq * time),
                    libm::cos(2.0 * PI * target_freq * time),
                    libm::sin(2.0 * PI * target_freq * time + 1.0),
                    libm::cos(2.0 * PI * target_freq * time + 2.0),
                ]
            })
            .collect();

        let target_freqs = [8.0, 10.0, 12.0, 15.0];
        let mut ref_buffer = [[0.0f64; 4]; 500];

        let (best_idx, best_corr) = ssvep_detect::<4, 16, 4, 16>(
            &signals,
            sample_rate,
            &target_freqs,
            &mut ref_buffer,
            1e-6,
        )
        .unwrap();

        assert_eq!(best_idx, 1, "Should detect 10 Hz (index 1)");
        assert!(best_corr > 0.9);
    }

    #[test]
    fn test_ssvep_detect_with_noise() {
        // Test detection with noisy signals
        let n = 500;
        let sample_rate = 250.0;
        let target_freq = 15.0;

        let signals = generate_ssvep_signal::<8>(sample_rate, n, target_freq, 2.0);

        let target_freqs = [8.0, 10.0, 12.0, 15.0, 20.0];
        let mut ref_buffer = [[0.0f64; 4]; 500];

        let (best_idx, _) = ssvep_detect::<8, 64, 4, 16>(
            &signals,
            sample_rate,
            &target_freqs,
            &mut ref_buffer,
            1e-6,
        )
        .unwrap();

        assert_eq!(best_idx, 3, "Should detect 15 Hz (index 3)");
    }

    #[test]
    fn test_cca_more_channels_than_references() {
        // Typical SSVEP: 8 channels, 4 reference components
        let n = 300;
        let sample_rate = 250.0;
        let freq = 10.0;

        let signals = generate_ssvep_signal::<8>(sample_rate, n, freq, 3.0);

        let mut refs = [[0.0f64; 4]; 300];
        fill_ssvep_references::<4>(sample_rate, freq, &mut refs);

        let mut correlations = [0.0; 4];
        let n_corr = cca::<8, 64, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        assert_eq!(n_corr, 4); // min(8, 4) = 4
        assert!(
            correlations[0] > 0.5,
            "Expected decent correlation with noisy signal"
        );
    }

    #[test]
    fn test_cca_returns_descending_correlations() {
        let n = 500;
        let sample_rate = 250.0;
        let freq = 10.0;

        let signals = generate_ssvep_signal::<4>(sample_rate, n, freq, 3.0);

        let mut refs = [[0.0f64; 4]; 500];
        fill_ssvep_references::<4>(sample_rate, freq, &mut refs);

        let mut correlations = [0.0; 4];
        let n_corr = cca::<4, 16, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        // Canonical correlations should be in descending order
        for i in 1..n_corr {
            assert!(
                correlations[i - 1] >= correlations[i] - 1e-10,
                "Correlations not descending: {} < {}",
                correlations[i - 1],
                correlations[i]
            );
        }
    }

    #[test]
    fn test_cca_correlations_bounded() {
        // All canonical correlations should be in [0, 1]
        let n = 200;
        let sample_rate = 250.0;

        let signals = generate_ssvep_signal::<4>(sample_rate, n, 10.0, 1.0);

        let mut refs = [[0.0f64; 4]; 200];
        fill_ssvep_references::<4>(sample_rate, 10.0, &mut refs);

        let mut correlations = [0.0; 4];
        let n_corr = cca::<4, 16, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        for &corr in correlations.iter().take(n_corr) {
            assert!(
                (0.0..=1.0).contains(&corr),
                "Correlation out of bounds: {}",
                corr
            );
        }
    }

    #[test]
    fn test_ssvep_detect_empty_frequencies() {
        let signals = [[0.0f64; 4]; 100];
        let mut ref_buffer = [[0.0f64; 4]; 100];

        let result = ssvep_detect::<4, 16, 4, 16>(&signals, 250.0, &[], &mut ref_buffer, 1e-6);
        assert_eq!(result, Err(CcaError::NoTargetFrequencies));
    }

    #[test]
    fn test_ssvep_detect_insufficient_samples() {
        let signals = [[0.0f64; 4]; 1]; // Only 1 sample
        let mut ref_buffer = [[0.0f64; 4]; 1];

        let result = ssvep_detect::<4, 16, 4, 16>(&signals, 250.0, &[10.0], &mut ref_buffer, 1e-6);
        assert_eq!(result, Err(CcaError::InsufficientSamples));
    }

    #[test]
    fn test_cca_dimension_mismatch() {
        let signals = [[0.0f64; 2]; 100];
        let refs = [[0.0f64; 4]; 50]; // Different length
        let mut correlations = [0.0; 2];

        let result = cca::<2, 4, 4, 16>(&signals, &refs, &mut correlations, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_cca_16_channels() {
        // Test with 16 channels to verify larger matrix handling
        let n = 500;
        let sample_rate = 250.0;
        let freq = 12.0;

        let signals = generate_ssvep_signal::<16>(sample_rate, n, freq, 3.0);

        let mut refs = [[0.0f64; 4]; 500];
        fill_ssvep_references::<4>(sample_rate, freq, &mut refs);

        let mut correlations = [0.0; 4];
        let n_corr = cca::<16, 256, 4, 16>(&signals, &refs, &mut correlations, 1e-6).unwrap();

        assert_eq!(n_corr, 4);
        assert!(
            correlations[0] > 0.3,
            "Expected reasonable correlation with 16 channels"
        );
    }

    #[test]
    fn test_multi_frequency_detection_accuracy() {
        // Test detection across multiple frequencies - each should be correctly identified
        let sample_rate = 250.0;
        let n = 500;
        let target_freqs = [8.0, 10.0, 12.0, 15.0];

        let mut correct = 0;
        for (true_idx, &true_freq) in target_freqs.iter().enumerate() {
            let signals = generate_ssvep_signal::<4>(sample_rate, n, true_freq, 3.0);
            let mut ref_buffer = [[0.0f64; 4]; 500];

            let (detected_idx, _) = ssvep_detect::<4, 16, 4, 16>(
                &signals,
                sample_rate,
                &target_freqs,
                &mut ref_buffer,
                1e-6,
            )
            .unwrap();

            if detected_idx == true_idx {
                correct += 1;
            }
        }

        // With SNR=3, at least 3 out of 4 should be correct
        assert!(
            correct >= 3,
            "Detection accuracy too low: {}/4 correct",
            correct
        );
    }
}
