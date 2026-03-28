//! Phase-Amplitude Coupling (PAC) metrics for neural oscillation analysis.
//!
//! This module provides functions for measuring how the phase of a slow oscillation
//! (e.g., theta 4-8 Hz) modulates the amplitude envelope of a fast oscillation
//! (e.g., gamma 30-100 Hz). PAC is widely used in epilepsy research (HFO detection),
//! memory consolidation (theta-gamma coupling), and sleep studies.
//!
//! # Overview
//!
//! - [`modulation_index`] -- Tort et al. (2010) KL-divergence-based MI in \[0, 1\]
//! - [`mean_vector_length`] -- Canolty et al. (2006) circular mean of amplitude-weighted phase
//! - [`phase_amplitude_distribution`] -- Binned phase-amplitude histogram for visualization
//!
//! # Usage
//!
//! All functions take pre-computed phase and amplitude arrays. The caller is responsible
//! for bandpass filtering and Hilbert transform extraction before calling these functions.
//! This matches the [`crate::connectivity::phase_locking_value`] pattern.
//!
//! # Example
//!
//! ```
//! use zerostone::pac::{modulation_index, mean_vector_length};
//! use zerostone::float::{self, Float, PI};
//!
//! // Synthetic coupled signal: amplitude modulated by phase
//! let n = 500;
//! let mut phase = [0.0 as Float; 500];
//! let mut amplitude = [0.0 as Float; 500];
//! for i in 0..n {
//!     let p = (i as Float / n as Float) * 2.0 * PI - PI;
//!     phase[i] = p;
//!     // Amplitude peaks at phase = 0 (strong coupling)
//!     amplitude[i] = 1.0 + 0.8 * float::cos(p);
//! }
//!
//! let mi = modulation_index(&phase, &amplitude, 18);
//! assert!(mi > 0.01); // significant coupling
//!
//! let mvl = mean_vector_length(&phase, &amplitude);
//! assert!(mvl > 0.1); // significant coupling
//! ```

use crate::float::Float;

/// Compute the Modulation Index (Tort et al. 2010).
///
/// Measures phase-amplitude coupling by binning amplitude values by phase,
/// computing the normalized amplitude distribution, and measuring its
/// divergence from a uniform distribution using KL divergence.
///
/// # Arguments
///
/// * `phase` - Instantaneous phase values in radians (from Hilbert transform)
/// * `amplitude` - Instantaneous amplitude envelope (from Hilbert transform)
/// * `n_bins` - Number of phase bins (must be 2..=64, typically 18 for 20-degree bins)
///
/// # Returns
///
/// MI in \[0, 1\] where 0 = no coupling (uniform distribution) and
/// 1 = maximum coupling (all amplitude in one bin).
///
/// # Panics
///
/// Panics if:
/// - `phase` and `amplitude` have different lengths
/// - Either array is empty
/// - `n_bins` < 2 or `n_bins` > 64
///
/// # Example
///
/// ```
/// use zerostone::pac::modulation_index;
/// use zerostone::float::{Float, PI};
///
/// // No coupling: constant amplitude regardless of phase
/// let mut phase = [0.0 as Float; 360];
/// for i in 0..360 {
///     phase[i] = (i as Float) * (PI / 180.0) - PI;
/// }
/// let amplitude = [1.0 as Float; 360];
/// let mi = modulation_index(&phase, &amplitude, 18);
/// assert!(mi < 0.01, "No coupling should give MI near 0, got {}", mi);
/// ```
#[allow(clippy::unnecessary_cast)]
pub fn modulation_index(phase: &[Float], amplitude: &[Float], n_bins: usize) -> Float {
    assert_eq!(
        phase.len(),
        amplitude.len(),
        "Phase and amplitude arrays must have equal length"
    );
    assert!(!phase.is_empty(), "Arrays must not be empty");
    assert!(
        (2..=64).contains(&n_bins),
        "n_bins must be in [2, 64], got {}",
        n_bins
    );

    let n = phase.len();

    // Bin width: 2*pi / n_bins
    let bin_width = 2.0 * core::f64::consts::PI / n_bins as f64;

    // Accumulate amplitude into phase bins (f64 for precision)
    let mut bin_sum = [0.0f64; 64];
    let mut bin_count = [0u32; 64];

    for i in 0..n {
        // Wrap phase to [-pi, pi] then map to bin index
        let p = phase[i] as f64;
        // Shift to [0, 2*pi]
        let p_shifted = p + core::f64::consts::PI;
        let mut bin = (p_shifted / bin_width) as usize;
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        bin_sum[bin] += amplitude[i] as f64;
        bin_count[bin] += 1;
    }

    // Compute mean amplitude per bin
    let mut mean_amp = [0.0f64; 64];
    for b in 0..n_bins {
        if bin_count[b] > 0 {
            mean_amp[b] = bin_sum[b] / bin_count[b] as f64;
        }
    }

    // Normalize to probability distribution
    let total: f64 = mean_amp[..n_bins].iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut p_dist = [0.0f64; 64];
    for b in 0..n_bins {
        p_dist[b] = mean_amp[b] / total;
    }

    // KL divergence from uniform distribution
    // D_KL(P || U) = sum(P(i) * log(P(i) * N))
    let log_n = libm::log(n_bins as f64);
    let epsilon = 1e-12;
    let mut d_kl = 0.0f64;
    for p_b in p_dist.iter().take(n_bins) {
        let p_b = if *p_b < epsilon { epsilon } else { *p_b };
        d_kl += p_b * libm::log(p_b * n_bins as f64);
    }

    // Normalize: MI = D_KL / log(N), clamp to [0, 1]
    let mi = d_kl / log_n;
    mi.clamp(0.0, 1.0) as Float
}

/// Compute the Mean Vector Length (Canolty et al. 2006).
///
/// Measures phase-amplitude coupling as the magnitude of the mean
/// amplitude-weighted phase vector, normalized by mean amplitude.
///
/// # Arguments
///
/// * `phase` - Instantaneous phase values in radians (from Hilbert transform)
/// * `amplitude` - Instantaneous amplitude envelope (from Hilbert transform)
///
/// # Returns
///
/// MVL in \[0, 1\] where 0 = no coupling and 1 = perfect coupling
/// (all amplitude concentrated at a single phase).
///
/// # Panics
///
/// Panics if:
/// - `phase` and `amplitude` have different lengths
/// - Either array is empty
///
/// # Example
///
/// ```
/// use zerostone::pac::mean_vector_length;
/// use zerostone::float::{Float, PI};
///
/// // No coupling: constant amplitude
/// let mut phase = [0.0 as Float; 360];
/// for i in 0..360 {
///     phase[i] = (i as Float) * (PI / 180.0) - PI;
/// }
/// let amplitude = [1.0 as Float; 360];
/// let mvl = mean_vector_length(&phase, &amplitude);
/// assert!(mvl < 0.05, "No coupling should give MVL near 0, got {}", mvl);
/// ```
#[allow(clippy::unnecessary_cast)]
pub fn mean_vector_length(phase: &[Float], amplitude: &[Float]) -> Float {
    assert_eq!(
        phase.len(),
        amplitude.len(),
        "Phase and amplitude arrays must have equal length"
    );
    assert!(!phase.is_empty(), "Arrays must not be empty");

    let n = phase.len();
    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let mut sum_amp = 0.0f64;

    for i in 0..n {
        let a = amplitude[i] as f64;
        let p = phase[i] as f64;
        sum_re += a * libm::cos(p);
        sum_im += a * libm::sin(p);
        sum_amp += a;
    }

    if sum_amp <= 0.0 {
        return 0.0;
    }

    // Mean vector length normalized by mean amplitude
    let mean_re = sum_re / n as f64;
    let mean_im = sum_im / n as f64;
    let mean_amp = sum_amp / n as f64;

    let mvl = libm::sqrt(mean_re * mean_re + mean_im * mean_im) / mean_amp;
    let mvl_clamped = if mvl > 1.0 { 1.0 } else { mvl };
    mvl_clamped as Float
}

/// Compute the phase-amplitude distribution for visualization.
///
/// Bins amplitude values by phase and returns bin centers and mean amplitudes.
/// This is the intermediate computation used by [`modulation_index`], exposed
/// for plotting phase-amplitude histograms.
///
/// # Arguments
///
/// * `phase` - Instantaneous phase values in radians
/// * `amplitude` - Instantaneous amplitude envelope
/// * `n_bins` - Number of phase bins (must be 2..=64)
/// * `output_centers` - Output buffer for bin center angles (length >= n_bins)
/// * `output_amplitudes` - Output buffer for mean amplitude per bin (length >= n_bins)
///
/// # Returns
///
/// Number of bins written (equal to `n_bins`).
///
/// # Panics
///
/// Panics if:
/// - `phase` and `amplitude` have different lengths
/// - Either array is empty
/// - `n_bins` < 2 or `n_bins` > 64
/// - Output buffers are too small
///
/// # Example
///
/// ```
/// use zerostone::pac::phase_amplitude_distribution;
/// use zerostone::Float;
///
/// let phase = [0.0 as Float; 100];
/// let amplitude = [1.0 as Float; 100];
/// let mut centers = [0.0 as Float; 18];
/// let mut amps = [0.0 as Float; 18];
/// let n = phase_amplitude_distribution(&phase, &amplitude, 18, &mut centers, &mut amps);
/// assert_eq!(n, 18);
/// ```
#[allow(clippy::unnecessary_cast)]
pub fn phase_amplitude_distribution(
    phase: &[Float],
    amplitude: &[Float],
    n_bins: usize,
    output_centers: &mut [Float],
    output_amplitudes: &mut [Float],
) -> usize {
    assert_eq!(
        phase.len(),
        amplitude.len(),
        "Phase and amplitude arrays must have equal length"
    );
    assert!(!phase.is_empty(), "Arrays must not be empty");
    assert!(
        (2..=64).contains(&n_bins),
        "n_bins must be in [2, 64], got {}",
        n_bins
    );
    assert!(
        output_centers.len() >= n_bins,
        "output_centers length {} must be >= n_bins {}",
        output_centers.len(),
        n_bins
    );
    assert!(
        output_amplitudes.len() >= n_bins,
        "output_amplitudes length {} must be >= n_bins {}",
        output_amplitudes.len(),
        n_bins
    );

    let n = phase.len();
    let bin_width = 2.0 * core::f64::consts::PI / n_bins as f64;

    // Write bin centers
    for (b, center_out) in output_centers.iter_mut().enumerate().take(n_bins) {
        let center = -core::f64::consts::PI + (b as f64 + 0.5) * bin_width;
        *center_out = center as Float;
    }

    // Accumulate
    let mut bin_sum = [0.0f64; 64];
    let mut bin_count = [0u32; 64];

    for i in 0..n {
        let p = phase[i] as f64;
        let p_shifted = p + core::f64::consts::PI;
        let mut bin = (p_shifted / bin_width) as usize;
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        bin_sum[bin] += amplitude[i] as f64;
        bin_count[bin] += 1;
    }

    // Write mean amplitudes
    for b in 0..n_bins {
        output_amplitudes[b] = if bin_count[b] > 0 {
            (bin_sum[b] / bin_count[b] as f64) as Float
        } else {
            0.0
        };
    }

    n_bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float;

    // Helper: fill phase array with linear ramp from -PI to PI
    fn fill_phase_ramp(buf: &mut [Float]) {
        let n = buf.len();
        for (i, val) in buf.iter_mut().enumerate() {
            *val = (i as Float / n as Float) * 2.0 * float::PI - float::PI;
        }
    }

    // Helper: fill coupled amplitude (peaks at phase=0)
    fn fill_coupled_amplitude(phase: &[Float], amplitude: &mut [Float], strength: Float) {
        for i in 0..phase.len() {
            amplitude[i] = 1.0 + strength * float::cos(phase[i]);
        }
    }

    // =========================================================================
    // Modulation Index tests
    // =========================================================================

    #[test]
    fn test_mi_no_coupling() {
        // Constant amplitude = uniform distribution = MI ~ 0
        let mut phase = [0.0 as Float; 720];
        fill_phase_ramp(&mut phase);
        let amplitude = [1.0 as Float; 720];
        let mi = modulation_index(&phase, &amplitude, 18);
        assert!(mi < 0.01, "No coupling should give MI near 0, got {}", mi);
    }

    #[test]
    fn test_mi_strong_coupling() {
        let mut phase = [0.0 as Float; 1000];
        let mut amplitude = [0.0 as Float; 1000];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amplitude, 0.9);
        let mi = modulation_index(&phase, &amplitude, 18);
        assert!(
            mi > 0.05,
            "Strong coupling should give significant MI, got {}",
            mi
        );
    }

    #[test]
    fn test_mi_range() {
        let mut phase = [0.0 as Float; 500];
        let mut amplitude = [0.0 as Float; 500];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amplitude, 0.5);
        let mi = modulation_index(&phase, &amplitude, 18);
        assert!(mi >= 0.0, "MI must be >= 0, got {}", mi);
        assert!(mi <= 1.0, "MI must be <= 1, got {}", mi);
    }

    #[test]
    fn test_mi_increases_with_coupling() {
        let mut phase = [0.0 as Float; 1000];
        let mut amp_w = [0.0 as Float; 1000];
        let mut amp_s = [0.0 as Float; 1000];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amp_w, 0.3);
        fill_coupled_amplitude(&phase, &mut amp_s, 0.9);
        let mi_weak = modulation_index(&phase, &amp_w, 18);
        let mi_strong = modulation_index(&phase, &amp_s, 18);
        assert!(
            mi_strong > mi_weak,
            "Stronger coupling should give higher MI: {} vs {}",
            mi_strong,
            mi_weak
        );
    }

    #[test]
    fn test_mi_custom_bins() {
        let mut phase = [0.0 as Float; 500];
        let mut amplitude = [0.0 as Float; 500];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amplitude, 0.8);
        let mi_9 = modulation_index(&phase, &amplitude, 9);
        let mi_36 = modulation_index(&phase, &amplitude, 36);
        assert!(
            mi_9 > 0.01,
            "MI with 9 bins should show coupling, got {}",
            mi_9
        );
        assert!(
            mi_36 > 0.01,
            "MI with 36 bins should show coupling, got {}",
            mi_36
        );
    }

    #[test]
    #[should_panic(expected = "Phase and amplitude arrays must have equal length")]
    fn test_mi_unequal_lengths() {
        modulation_index(&[0.0; 5], &[1.0; 3], 18);
    }

    #[test]
    #[should_panic(expected = "Arrays must not be empty")]
    fn test_mi_empty() {
        modulation_index(&[], &[], 18);
    }

    #[test]
    #[should_panic(expected = "n_bins must be in [2, 64]")]
    fn test_mi_bins_too_small() {
        modulation_index(&[0.0; 10], &[1.0; 10], 1);
    }

    #[test]
    #[should_panic(expected = "n_bins must be in [2, 64]")]
    fn test_mi_bins_too_large() {
        modulation_index(&[0.0; 10], &[1.0; 10], 65);
    }

    // =========================================================================
    // Mean Vector Length tests
    // =========================================================================

    #[test]
    fn test_mvl_no_coupling() {
        let mut phase = [0.0 as Float; 720];
        fill_phase_ramp(&mut phase);
        let amplitude = [1.0 as Float; 720];
        let mvl = mean_vector_length(&phase, &amplitude);
        assert!(
            mvl < 0.02,
            "No coupling should give MVL near 0, got {}",
            mvl
        );
    }

    #[test]
    fn test_mvl_strong_coupling() {
        let mut phase = [0.0 as Float; 1000];
        let mut amplitude = [0.0 as Float; 1000];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amplitude, 0.9);
        let mvl = mean_vector_length(&phase, &amplitude);
        assert!(
            mvl > 0.1,
            "Strong coupling should give significant MVL, got {}",
            mvl
        );
    }

    #[test]
    fn test_mvl_range() {
        let mut phase = [0.0 as Float; 500];
        let mut amplitude = [0.0 as Float; 500];
        fill_phase_ramp(&mut phase);
        fill_coupled_amplitude(&phase, &mut amplitude, 0.5);
        let mvl = mean_vector_length(&phase, &amplitude);
        assert!(mvl >= 0.0, "MVL must be >= 0, got {}", mvl);
        assert!(mvl <= 1.0, "MVL must be <= 1, got {}", mvl);
    }

    #[test]
    fn test_mvl_perfect_coupling() {
        // All amplitude at a single phase -> MVL = 1
        let phase = [0.0 as Float; 100];
        let amplitude = [1.0 as Float; 100];
        let mvl = mean_vector_length(&phase, &amplitude);
        assert!(
            (mvl - 1.0).abs() < 0.01,
            "All amplitude at one phase should give MVL ~ 1, got {}",
            mvl
        );
    }

    #[test]
    #[should_panic(expected = "Phase and amplitude arrays must have equal length")]
    fn test_mvl_unequal_lengths() {
        mean_vector_length(&[0.0; 5], &[1.0; 3]);
    }

    #[test]
    #[should_panic(expected = "Arrays must not be empty")]
    fn test_mvl_empty() {
        mean_vector_length(&[], &[]);
    }

    // =========================================================================
    // Phase-Amplitude Distribution tests
    // =========================================================================

    #[test]
    fn test_distribution_shape() {
        let mut phase = [0.0 as Float; 360];
        fill_phase_ramp(&mut phase);
        let amplitude = [1.0 as Float; 360];
        let mut centers = [0.0 as Float; 18];
        let mut amps = [0.0 as Float; 18];
        let n = phase_amplitude_distribution(&phase, &amplitude, 18, &mut centers, &mut amps);
        assert_eq!(n, 18);
    }

    #[test]
    fn test_distribution_centers_range() {
        let phase = [0.0 as Float; 100];
        let amplitude = [1.0 as Float; 100];
        let mut centers = [0.0 as Float; 18];
        let mut amps = [0.0 as Float; 18];
        phase_amplitude_distribution(&phase, &amplitude, 18, &mut centers, &mut amps);

        for &c in &centers {
            assert!(
                c > -float::PI - 0.01 && c < float::PI + 0.01,
                "Bin center {} outside [-pi, pi]",
                c
            );
        }
    }

    #[test]
    fn test_distribution_uniform_amplitude() {
        let mut phase = [0.0 as Float; 720];
        fill_phase_ramp(&mut phase);
        let amplitude = [5.0 as Float; 720];
        let mut centers = [0.0 as Float; 18];
        let mut amps = [0.0 as Float; 18];
        phase_amplitude_distribution(&phase, &amplitude, 18, &mut centers, &mut amps);

        for &a in &amps[..18] {
            assert!(
                (a - 5.0).abs() < 0.1,
                "Uniform amplitude should give ~5.0 per bin, got {}",
                a
            );
        }
    }

    #[test]
    fn test_mi_zero_amplitude() {
        let mut phase = [0.0 as Float; 100];
        fill_phase_ramp(&mut phase);
        let amplitude = [0.0 as Float; 100];
        let mi = modulation_index(&phase, &amplitude, 18);
        assert!(mi < 0.001, "Zero amplitude should give MI ~ 0, got {}", mi);
    }

    #[test]
    fn test_mvl_zero_amplitude() {
        let mut phase = [0.0 as Float; 100];
        fill_phase_ramp(&mut phase);
        let amplitude = [0.0 as Float; 100];
        let mvl = mean_vector_length(&phase, &amplitude);
        assert!(
            mvl < 0.001,
            "Zero amplitude should give MVL ~ 0, got {}",
            mvl
        );
    }
}
