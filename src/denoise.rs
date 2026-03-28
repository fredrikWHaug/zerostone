//! Wavelet denoising for spike sorting preprocessing.
//!
//! Implements the Stationary Wavelet Transform (SWT) with the Haar wavelet for
//! signal denoising, following the Donoho-Johnstone (1994) universal threshold
//! framework. This is a standard preprocessing step in extracellular spike sorting
//! pipelines: broadband thermal noise is suppressed while preserving the sharp,
//! transient waveform morphology of action potentials.
//!
//! # Algorithm
//!
//! The SWT (a trous algorithm) decomposes a signal into approximation and detail
//! coefficients at each level without downsampling, preserving the original signal
//! length. For the Haar wavelet:
//!
//! - Lowpass filter: `h = [1/sqrt(2), 1/sqrt(2)]`
//! - Highpass filter: `g = [1/sqrt(2), -1/sqrt(2)]`
//!
//! At each decomposition level `j`, the filters are upsampled by inserting `2^(j-1) - 1`
//! zeros between taps (equivalent to spacing the taps `2^(j-1)` apart in the input).
//!
//! Denoising proceeds in three steps:
//! 1. **Decompose**: apply the SWT for `L` levels, producing detail coefficients `d_1, ..., d_L`
//! 2. **Threshold**: apply soft or hard thresholding to each level's detail coefficients
//!    using the universal threshold `lambda = sigma * sqrt(2 * ln(N))`, where sigma is
//!    estimated from the finest-level details via MAD / 0.6745
//! 3. **Reconstruct**: invert the SWT using the thresholded coefficients
//!
//! # Why Haar for spikes
//!
//! The Haar wavelet is piecewise-constant with compact support (2 taps), making it
//! well-suited for detecting sharp transients like extracellular spikes. Its minimal
//! filter length also means minimal boundary effects and low computational cost --
//! important for real-time and embedded contexts.
//!
//! # Example
//!
//! ```
//! use zerostone::denoise::{denoise_haar, ThresholdMode};
//! use zerostone::Float;
//!
//! let mut signal = [0.0 as Float; 64];
//! // Add a "spike" at sample 32
//! signal[32] = -5.0;
//! signal[33] = 3.0;
//!
//! let mut scratch = [0.0 as Float; 64 * 5]; // need signal.len() * (levels + 2)
//! denoise_haar(&mut signal, &mut scratch, 3, ThresholdMode::Soft);
//!
//! // The spike shape is largely preserved (large amplitude survives thresholding)
//! assert!(signal[32] < -1.0, "Spike peak should be preserved");
//! ```

#![allow(clippy::needless_range_loop)]

use crate::float::{self, Float};

/// Thresholding mode for wavelet denoising.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMode {
    /// Soft thresholding: `sign(x) * max(|x| - lambda, 0)`.
    /// Produces less bias at the threshold boundary; preferred for most applications.
    Soft,
    /// Hard thresholding: `x` if `|x| >= lambda`, else `0`.
    /// Preserves amplitude of large coefficients exactly.
    Hard,
}

/// Soft threshold function.
///
/// Returns `sign(x) * max(|x| - lambda, 0)`. For `lambda <= 0`, returns `x` unchanged.
///
/// # Example
///
/// ```
/// use zerostone::denoise::soft_threshold;
///
/// assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-12);
/// assert!((soft_threshold(-3.0, 1.0) - (-2.0)).abs() < 1e-12);
/// assert!((soft_threshold(0.5, 1.0) - 0.0).abs() < 1e-12);
/// ```
#[inline]
pub fn soft_threshold(x: Float, lambda: Float) -> Float {
    if lambda <= 0.0 {
        return x;
    }
    let ax = float::abs(x);
    if ax <= lambda {
        0.0
    } else if x > 0.0 {
        ax - lambda
    } else {
        -(ax - lambda)
    }
}

/// Hard threshold function.
///
/// Returns `x` if `|x| >= lambda`, else `0`. For `lambda <= 0`, returns `x` unchanged.
///
/// # Example
///
/// ```
/// use zerostone::denoise::hard_threshold;
///
/// assert!((hard_threshold(3.0, 1.0) - 3.0).abs() < 1e-12);
/// assert!((hard_threshold(0.5, 1.0) - 0.0).abs() < 1e-12);
/// assert!((hard_threshold(-2.0, 1.0) - (-2.0)).abs() < 1e-12);
/// ```
#[inline]
pub fn hard_threshold(x: Float, lambda: Float) -> Float {
    if lambda <= 0.0 {
        return x;
    }
    if float::abs(x) >= lambda {
        x
    } else {
        0.0
    }
}

/// Compute the universal threshold (Donoho-Johnstone 1994).
///
/// `lambda = sigma * sqrt(2 * ln(N))`
///
/// where `sigma` is the noise standard deviation and `N` is the signal length.
/// Returns 0.0 if `n <= 1`.
///
/// # Example
///
/// ```
/// use zerostone::denoise::universal_threshold;
///
/// let lambda = universal_threshold(1.0, 100);
/// // sqrt(2 * ln(100)) ~ 3.034
/// assert!(lambda > 3.0 && lambda < 3.1, "lambda = {}", lambda);
/// ```
#[inline]
pub fn universal_threshold(sigma: Float, n: usize) -> Float {
    if n <= 1 {
        return 0.0;
    }
    sigma * float::sqrt(2.0 * float::log(n as Float))
}

/// Estimate noise standard deviation from detail coefficients using MAD / 0.6745.
///
/// The Median Absolute Deviation is a robust estimator of scale. For Gaussian noise,
/// MAD / 0.6745 is a consistent estimator of sigma.
///
/// `scratch` is used for sorting and must be at least as long as `details`.
#[cfg(test)]
fn estimate_sigma_mad(details: &[Float], scratch: &mut [Float]) -> Float {
    let n = details.len();
    if n == 0 {
        return 0.0;
    }
    for i in 0..n {
        scratch[i] = float::abs(details[i]);
    }
    let s = &mut scratch[..n];
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) * 0.5
    };
    median / 0.6745
}

/// Denoise a signal in-place using the Stationary Wavelet Transform with the Haar wavelet.
///
/// Applies `levels` decomposition levels, estimates noise from the finest-level detail
/// coefficients, computes the universal threshold, and applies soft or hard thresholding
/// to all detail coefficients before reconstructing.
///
/// # Arguments
///
/// * `signal` - Input/output signal (modified in-place)
/// * `scratch` - Working buffer, must be at least `signal.len() * (levels + 2)` long.
///   A generous allocation is `signal.len() * (levels + 3)`.
/// * `levels` - Number of decomposition levels (must be >= 1)
/// * `mode` - Soft or hard thresholding
///
/// # Panics
///
/// Panics if `levels == 0`, `signal.len() < 2`, or `scratch` is too small.
pub fn denoise_haar(
    signal: &mut [Float],
    scratch: &mut [Float],
    levels: usize,
    mode: ThresholdMode,
) {
    let n = signal.len();
    assert!(levels >= 1, "levels must be >= 1");
    assert!(n >= 2, "signal must have at least 2 samples");
    let required = n * (levels + 2);
    assert!(
        scratch.len() >= required,
        "scratch must be at least signal.len() * (levels + 2) = {}",
        required
    );

    // Layout within scratch:
    // [0..n)                     : current approximation
    // [n..n*(levels+1))          : detail coefficients for each level (levels * n)
    // [n*(levels+1)..n*(levels+2)) : temp buffer for MAD / inverse reconstruction

    // Copy signal into approximation buffer.
    let (approx_buf, rest) = scratch.split_at_mut(n);
    approx_buf.copy_from_slice(signal);

    // Forward SWT decomposition.
    // We store details in rest[level*n .. (level+1)*n].
    // After each level, the new approximation overwrites signal (as temp storage).
    for level in 0..levels {
        let step = 1 << level; // 2^level
        let detail_start = level * n;

        // Read from approx_buf, write new approx to signal (temp), detail to rest.
        let inv_sqrt2 = 1.0 / float::sqrt(2.0);
        for i in 0..n {
            let j = if i >= step { i - step } else { n + i - step };
            let a = approx_buf[j];
            let b = approx_buf[i];
            signal[i] = (a + b) * inv_sqrt2;
            rest[detail_start + i] = (a - b) * inv_sqrt2;
        }

        // Copy new approximation back to approx_buf.
        approx_buf.copy_from_slice(signal);
    }

    // Estimate noise from finest-level (level 0) detail coefficients.
    // Copy detail[0] into the temp region for MAD sorting (preserves original details).
    let temp_start = levels * n;
    for i in 0..n {
        rest[temp_start + i] = float::abs(rest[i]);
    }
    // Save the coarsest approximation into signal temporarily (we need approx_buf later).
    signal.copy_from_slice(approx_buf);
    let sigma = {
        let s = &mut rest[temp_start..temp_start + n];
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) * 0.5
        };
        median / 0.6745
    };
    // Restore the coarsest approximation.
    approx_buf.copy_from_slice(signal);

    let lambda = universal_threshold(sigma, n);

    // Threshold detail coefficients at all levels.
    let threshold_fn = match mode {
        ThresholdMode::Soft => soft_threshold,
        ThresholdMode::Hard => hard_threshold,
    };
    for level in 0..levels {
        let start = level * n;
        for i in 0..n {
            rest[start + i] = threshold_fn(rest[start + i], lambda);
        }
    }

    // Inverse SWT reconstruction (from coarsest to finest).
    // approx_buf still holds the coarsest approximation.
    for level in (0..levels).rev() {
        let step = 1 << level;
        let detail_start = level * n;

        // Inverse Haar: reconstruct into signal (temp).
        let inv_sqrt2 = 1.0 / float::sqrt(2.0);
        for i in 0..n {
            let k = (i + step) % n;
            let a_i = approx_buf[i];
            let d_i = rest[detail_start + i];
            let a_k = approx_buf[k];
            let d_k = rest[detail_start + k];
            let r1 = (a_i - d_i) * inv_sqrt2;
            let r2 = (a_k + d_k) * inv_sqrt2;
            signal[i] = (r1 + r2) * 0.5;
        }

        // The reconstructed signal becomes the approximation for the next (finer) level.
        approx_buf.copy_from_slice(signal);
    }

    // signal now holds the denoised result (already in place from last inverse step).
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float::Float;

    #[test]
    fn test_soft_threshold_values() {
        assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((soft_threshold(-3.0, 1.0) - (-2.0)).abs() < 1e-12);
        assert!((soft_threshold(0.5, 1.0) - 0.0).abs() < 1e-12);
        assert!((soft_threshold(-0.5, 1.0) - 0.0).abs() < 1e-12);
        assert!((soft_threshold(1.0, 1.0) - 0.0).abs() < 1e-12);
        assert!((soft_threshold(0.0, 1.0) - 0.0).abs() < 1e-12);
        // Negative lambda returns x unchanged
        assert!((soft_threshold(2.5, -1.0) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_hard_threshold_values() {
        assert!((hard_threshold(3.0, 1.0) - 3.0).abs() < 1e-12);
        assert!((hard_threshold(-3.0, 1.0) - (-3.0)).abs() < 1e-12);
        assert!((hard_threshold(0.5, 1.0) - 0.0).abs() < 1e-12);
        assert!((hard_threshold(-0.5, 1.0) - 0.0).abs() < 1e-12);
        assert!((hard_threshold(1.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((hard_threshold(0.0, 1.0) - 0.0).abs() < 1e-12);
        // Negative lambda returns x unchanged
        assert!((hard_threshold(0.3, -1.0) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_universal_threshold_positive() {
        let lambda = universal_threshold(1.0, 100);
        assert!(lambda > 0.0, "Threshold must be positive");
        // sqrt(2 * ln(100)) ~ 3.034
        assert!(
            (lambda - 3.034).abs() < 0.1,
            "Expected ~3.034, got {}",
            lambda
        );

        let lambda2 = universal_threshold(2.0, 100);
        assert!(
            (lambda2 - 2.0 * lambda).abs() < 1e-10,
            "Should scale linearly with sigma"
        );

        assert!(universal_threshold(1.0, 1) == 0.0, "n=1 should return 0");
    }

    #[test]
    fn test_denoise_zero_input() {
        let mut signal = [0.0 as Float; 32];
        let mut scratch = [0.0 as Float; 32 * 5];
        denoise_haar(&mut signal, &mut scratch, 3, ThresholdMode::Soft);
        for &s in signal.iter() {
            assert!(s.abs() < 1e-15, "Zero input should stay zero, got {}", s);
        }
    }

    #[test]
    fn test_denoise_pure_signal_unchanged() {
        // A large-amplitude sine wave should be mostly preserved (well above noise threshold).
        let n = 128;
        let mut signal = [0.0 as Float; 128];
        let mut original = [0.0 as Float; 128];
        for i in 0..n {
            let val = 100.0 * float::sin(2.0 * float::PI * 5.0 * i as Float / n as Float);
            signal[i] = val;
            original[i] = val;
        }

        let mut scratch = [0.0 as Float; 128 * 6];
        denoise_haar(&mut signal, &mut scratch, 3, ThresholdMode::Soft);

        // Compute RMS error relative to original
        let mut err_sq = 0.0;
        let mut sig_sq = 0.0;
        for i in 0..n {
            let e = signal[i] - original[i];
            err_sq += e * e;
            sig_sq += original[i] * original[i];
        }
        let snr = 10.0 * float::log10(sig_sq / (err_sq + 1e-30));
        assert!(
            snr > 10.0,
            "Clean large-amplitude signal should be well-preserved, SNR = {:.1} dB",
            snr
        );
    }

    #[test]
    fn test_denoise_reduces_noise() {
        // Signal with added noise: denoising should reduce variance of the noise portion.
        let n = 256;
        let mut signal = [0.0 as Float; 256];
        let mut scratch = [0.0 as Float; 256 * 6];

        // Simple LCG pseudo-random noise
        let mut rng_state: u64 = 42;
        for i in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((rng_state >> 33) as Float / (1u64 << 31) as Float) - 1.0;
            signal[i] = noise * 0.5; // noise with std ~ 0.29
        }

        // Compute variance before
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &s in signal.iter() {
            sum += s;
            sum_sq += s * s;
        }
        let var_before = sum_sq / n as Float - (sum / n as Float) * (sum / n as Float);

        denoise_haar(&mut signal, &mut scratch, 3, ThresholdMode::Soft);

        // Compute variance after
        let mut sum2 = 0.0;
        let mut sum_sq2 = 0.0;
        for &s in signal.iter() {
            sum2 += s;
            sum_sq2 += s * s;
        }
        let var_after = sum_sq2 / n as Float - (sum2 / n as Float) * (sum2 / n as Float);

        assert!(
            var_after < var_before,
            "Denoising should reduce variance: before={:.4}, after={:.4}",
            var_before,
            var_after
        );
    }

    #[test]
    fn test_denoise_single_level() {
        let mut signal = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let mut scratch = [0.0 as Float; 8 * 4];
        denoise_haar(&mut signal, &mut scratch, 1, ThresholdMode::Hard);

        // With only 1 level and hard thresholding, large coefficients should survive.
        // Just verify it doesn't panic and output is finite.
        for &s in signal.iter() {
            assert!(s.is_finite(), "Output must be finite");
        }
        // The alternating pattern has strong detail coefficients; should be partially preserved.
        let mut energy = 0.0;
        for &s in signal.iter() {
            energy += s * s;
        }
        assert!(
            energy > 0.0,
            "Non-zero input should produce non-zero output"
        );
    }

    #[test]
    fn test_denoise_multi_level() {
        let _n = 64;
        let mut signal = [0.0 as Float; 64];
        // Put a spike-like feature
        signal[30] = -4.0;
        signal[31] = -8.0;
        signal[32] = 3.0;
        signal[33] = 1.0;

        let mut scratch = [0.0 as Float; 64 * 7];
        denoise_haar(&mut signal, &mut scratch, 4, ThresholdMode::Soft);

        // The spike peak should be partially preserved (large amplitude)
        assert!(
            signal[31] < -1.0,
            "Spike peak should survive denoising, got {}",
            signal[31]
        );

        // Regions far from the spike should be near zero
        let mut far_energy = 0.0;
        for i in 0..20 {
            far_energy += signal[i] * signal[i];
        }
        assert!(
            far_energy < 1.0,
            "Quiet region should stay quiet, energy = {}",
            far_energy
        );
    }

    #[test]
    fn test_denoise_hard_vs_soft() {
        let _n = 32;
        let mut sig_soft = [0.0 as Float; 32];
        let mut sig_hard = [0.0 as Float; 32];
        sig_soft[16] = -10.0;
        sig_hard[16] = -10.0;

        let mut scratch_s = [0.0 as Float; 32 * 5];
        let mut scratch_h = [0.0 as Float; 32 * 5];

        denoise_haar(&mut sig_soft, &mut scratch_s, 2, ThresholdMode::Soft);
        denoise_haar(&mut sig_hard, &mut scratch_h, 2, ThresholdMode::Hard);

        // Hard thresholding preserves amplitudes better than soft
        assert!(
            float::abs(sig_hard[16]) >= float::abs(sig_soft[16]),
            "Hard threshold should preserve amplitude >= soft: hard={}, soft={}",
            sig_hard[16],
            sig_soft[16]
        );
    }

    #[test]
    fn test_estimate_sigma_mad() {
        // Known Gaussian: MAD/0.6745 should approximate sigma
        let details = [
            -0.67, 0.32, -1.21, 0.78, 0.11, -0.45, 1.03, -0.89, 0.54, -0.15, 0.66, -1.10, 0.27,
            -0.58, 0.91, -0.33,
        ];
        let mut scratch = [0.0 as Float; 16];
        let sigma = estimate_sigma_mad(&details, &mut scratch);
        assert!(sigma > 0.0, "Sigma must be positive");
        assert!(sigma < 3.0, "Sigma should be reasonable, got {}", sigma);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::float::Float;

    #[kani::proof]
    fn soft_threshold_panic_free() {
        let x: Float = kani::any();
        let lambda: Float = kani::any();
        kani::assume(x.is_finite());
        kani::assume(lambda.is_finite());
        let result = soft_threshold(x, lambda);
        // Should not panic; result should be finite for finite inputs.
        assert!(result.is_finite());
    }

    #[kani::proof]
    fn hard_threshold_panic_free() {
        let x: Float = kani::any();
        let lambda: Float = kani::any();
        kani::assume(x.is_finite());
        kani::assume(lambda.is_finite());
        let result = hard_threshold(x, lambda);
        assert!(result.is_finite());
    }
}
