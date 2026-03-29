//! Entropy measures for neural signal complexity analysis.
//!
//! This module provides entropy-based biomarkers used in seizure detection
//! (SampEn drops before onset), anesthesia depth monitoring (BIS index uses
//! spectral entropy), and sleep staging.
//!
//! # Overview
//!
//! - [`sample_entropy`] -- SampEn(m, r, N): template-matching complexity measure
//! - [`approximate_entropy`] -- ApEn(m, r, N): self-matching variant (always finite)
//! - [`spectral_entropy`] -- Shannon entropy of normalized power spectral density
//! - [`multiscale_entropy`] -- MSE: sample entropy at coarse-grained time scales
//!
//! # Example
//!
//! ```
//! use zerostone::entropy::{sample_entropy, spectral_entropy};
//!
//! // Constant signal has zero complexity
//! let constant = [5.0; 100];
//! let se = sample_entropy(&constant, 2, 0.2);
//! assert!(se < 1e-10, "Constant signal should have SampEn ~ 0, got {}", se);
//!
//! // Flat PSD (white noise) has maximum spectral entropy
//! let flat_psd = [1.0; 64];
//! let h = spectral_entropy(&flat_psd, true);
//! assert!((h - 1.0).abs() < 1e-10, "Flat PSD normalized should be 1.0, got {}", h);
//! ```

use crate::float::{self, Float};

/// Compute Sample Entropy (SampEn).
///
/// Measures the complexity / irregularity of a time series by counting
/// template matches at embedding dimensions `m` and `m+1`, using Chebyshev
/// distance with threshold `r`. SampEn = -ln(A/B) where B = matches at dim m,
/// A = matches at dim m+1. Self-matches are excluded.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `m` - Embedding dimension (template length), must be >= 1
/// * `r` - Tolerance threshold, must be > 0
///
/// # Returns
///
/// SampEn value. Returns `Float::INFINITY` when no matches at dimension m+1 (A == 0).
/// Returns 0.0 for constant signals (all templates match at both dimensions).
///
/// # Panics
///
/// Panics if `data` is empty, `m < 1`, `r <= 0`, or `data.len() <= m + 1`.
///
/// # Example
///
/// ```
/// use zerostone::entropy::sample_entropy;
///
/// // Constant signal: perfectly regular -> SampEn = 0
/// let data = [3.0; 50];
/// let se = sample_entropy(&data, 2, 0.2);
/// assert!(se.abs() < 1e-10);
///
/// // Periodic signal: moderate complexity
/// let mut periodic = [0.0; 100];
/// for i in 0..100 {
///     periodic[i] = zerostone::float::sin(i as zerostone::float::Float * 0.5);
/// }
/// let se = sample_entropy(&periodic, 2, 0.2);
/// assert!(se > 0.0);
/// ```
pub fn sample_entropy(data: &[Float], m: usize, r: Float) -> Float {
    assert!(!data.is_empty(), "data must not be empty");
    assert!(m >= 1, "m must be >= 1, got {}", m);
    assert!(r > 0.0, "r must be > 0, got {}", r);
    assert!(
        data.len() > m + 1,
        "data.len() {} must be > m + 1 = {}",
        data.len(),
        m + 1
    );

    let n = data.len();
    let n_templates_m = n - m;

    let mut count_b = 0u64; // matches at dim m
    let mut count_a = 0u64; // matches at dim m+1

    // Only iterate j > i (symmetric pairs, exclude self-matches)
    for i in 0..n_templates_m {
        for j in (i + 1)..n_templates_m {
            // Check dim m: Chebyshev distance with early termination
            let mut match_m = true;
            for k in 0..m {
                if float::abs(data[i + k] - data[j + k]) >= r {
                    match_m = false;
                    break;
                }
            }

            if match_m {
                count_b += 1;

                // Check dim m+1: only need to check the extra element
                if i + m < n && j + m < n && float::abs(data[i + m] - data[j + m]) < r {
                    count_a += 1;
                }
            }
        }
    }

    if count_b == 0 {
        return float::INFINITY;
    }
    if count_a == 0 {
        return float::INFINITY;
    }

    -float::log(count_a as Float / count_b as Float)
}

/// Compute Approximate Entropy (ApEn).
///
/// Similar to sample entropy but includes self-matches, guaranteeing the result
/// is always finite. ApEn = phi(m) - phi(m+1) where phi(dim) is the mean of
/// ln(C_i) over all templates, and C_i counts matches including self.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `m` - Embedding dimension, must be >= 1
/// * `r` - Tolerance threshold, must be > 0
///
/// # Returns
///
/// ApEn value, always finite and non-negative.
///
/// # Panics
///
/// Panics if `data` is empty, `m < 1`, `r <= 0`, or `data.len() <= m`.
///
/// # Example
///
/// ```
/// use zerostone::entropy::approximate_entropy;
///
/// // Constant signal: perfectly regular -> ApEn near 0
/// let data = [3.0; 50];
/// let ae = approximate_entropy(&data, 2, 0.2);
/// assert!(ae < 0.01, "Constant signal ApEn should be near 0, got {}", ae);
/// ```
pub fn approximate_entropy(data: &[Float], m: usize, r: Float) -> Float {
    assert!(!data.is_empty(), "data must not be empty");
    assert!(m >= 1, "m must be >= 1, got {}", m);
    assert!(r > 0.0, "r must be > 0, got {}", r);
    assert!(
        data.len() > m,
        "data.len() {} must be > m = {}",
        data.len(),
        m
    );

    let phi_m = phi(data, m, r);
    let phi_m1 = phi(data, m + 1, r);
    let result = phi_m - phi_m1;
    // Clamp to non-negative (numerical noise can make it slightly negative)
    if result < 0.0 {
        0.0
    } else {
        result
    }
}

/// Compute phi(dim) for approximate entropy.
fn phi(data: &[Float], dim: usize, r: Float) -> Float {
    let n = data.len();
    if n < dim {
        return 0.0;
    }
    let n_templates = n - dim + 1;
    let mut sum_log: Float = 0.0;

    for i in 0..n_templates {
        let mut count = 0u64;
        for j in 0..n_templates {
            // Check if template j matches template i at dimension dim
            let mut matches = true;
            for k in 0..dim {
                if float::abs(data[i + k] - data[j + k]) >= r {
                    matches = false;
                    break;
                }
            }
            if matches {
                count += 1;
            }
        }
        // C_i = count / n_templates (includes self-match, so count >= 1)
        sum_log += float::log(count as Float / n_templates as Float);
    }

    sum_log / n_templates as Float
}

/// Compute Spectral Entropy.
///
/// Shannon entropy of the normalized power spectral density.
/// H = -sum(p_i * ln(p_i)) where p_i = psd\[i\] / sum(psd).
///
/// # Arguments
///
/// * `psd` - Power spectral density values (must be non-negative)
/// * `normalize` - If true, normalize by ln(N) to get result in \[0, 1\]
///
/// # Returns
///
/// Spectral entropy. If `normalize` is true, result is in \[0, 1\] where
/// 0 = single peak (pure tone) and 1 = flat spectrum (white noise).
/// Returns 0.0 for all-zero PSD.
///
/// # Panics
///
/// Panics if `psd` is empty.
///
/// # Example
///
/// ```
/// use zerostone::entropy::spectral_entropy;
///
/// // Flat spectrum: maximum entropy
/// let psd = [1.0; 32];
/// let h = spectral_entropy(&psd, true);
/// assert!((h - 1.0).abs() < 1e-10, "Flat PSD should give 1.0, got {}", h);
///
/// // Single peak: minimum entropy
/// let mut peak_psd = [0.0; 32];
/// peak_psd[5] = 10.0;
/// let h = spectral_entropy(&peak_psd, true);
/// assert!(h < 0.01, "Single peak should give near 0, got {}", h);
/// ```
pub fn spectral_entropy(psd: &[Float], normalize: bool) -> Float {
    assert!(!psd.is_empty(), "PSD must not be empty");

    let n = psd.len();

    // Sum for normalization
    let total: Float = psd.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    // Shannon entropy
    let mut h: Float = 0.0;
    for &val in psd {
        let p = val / total;
        if p > 0.0 {
            h -= p * float::log(p);
        }
    }

    if normalize {
        let log_n = float::log(n as Float);
        if log_n > 0.0 {
            h / log_n
        } else {
            0.0
        }
    } else {
        h
    }
}

/// Compute Multiscale Entropy (MSE).
///
/// Coarse-grains the signal at the given scale factor and then computes
/// sample entropy on the resulting signal. At scale 1, this is equivalent
/// to calling [`sample_entropy`] directly.
///
/// # Arguments
///
/// * `data` - Input time series
/// * `scale` - Scale factor (must be >= 1)
/// * `m` - Embedding dimension for sample entropy
/// * `r` - Tolerance threshold for sample entropy
/// * `scratch` - Scratch buffer for coarse-grained signal (length >= data.len() / scale)
///
/// # Returns
///
/// Sample entropy at the given scale.
///
/// # Panics
///
/// Panics if `data` is empty, `scale < 1`, `m < 1`, `r <= 0`,
/// `scratch` is too small, or coarse-grained length <= m + 1.
///
/// # Example
///
/// ```
/// use zerostone::entropy::multiscale_entropy;
///
/// // Scale 1 should equal sample_entropy
/// let data = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
///             1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
/// let mut scratch = [0.0; 20];
/// let mse1 = multiscale_entropy(&data, 1, 2, 0.5, &mut scratch);
/// let se = zerostone::entropy::sample_entropy(&data, 2, 0.5);
/// assert!((mse1 - se).abs() < 1e-10);
/// ```
pub fn multiscale_entropy(
    data: &[Float],
    scale: usize,
    m: usize,
    r: Float,
    scratch: &mut [Float],
) -> Float {
    assert!(!data.is_empty(), "data must not be empty");
    assert!(scale >= 1, "scale must be >= 1, got {}", scale);
    assert!(m >= 1, "m must be >= 1, got {}", m);
    assert!(r > 0.0, "r must be > 0, got {}", r);

    if scale == 1 {
        return sample_entropy(data, m, r);
    }

    let coarse_len = data.len() / scale;
    assert!(
        scratch.len() >= coarse_len,
        "scratch length {} must be >= coarse_len {}",
        scratch.len(),
        coarse_len
    );
    assert!(
        coarse_len > m + 1,
        "coarse_len {} must be > m + 1 = {} (data too short for scale {})",
        coarse_len,
        m + 1,
        scale
    );

    // Coarse-grain: average non-overlapping windows
    for (j, out) in scratch.iter_mut().enumerate().take(coarse_len) {
        let start = j * scale;
        let mut sum: Float = 0.0;
        for &val in &data[start..start + scale] {
            sum += val;
        }
        *out = sum / scale as Float;
    }

    sample_entropy(&scratch[..coarse_len], m, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Sample Entropy tests
    // =========================================================================

    #[test]
    fn test_sampen_constant_signal() {
        // Constant signal: all templates match at all dimensions -> SampEn = 0
        let data = [5.0 as Float; 50];
        let se = sample_entropy(&data, 2, 0.2);
        assert!(
            se.abs() < 1e-10,
            "Constant signal SampEn should be 0, got {}",
            se
        );
    }

    #[test]
    fn test_sampen_periodic_signal() {
        // Periodic signal: moderate complexity
        let mut data = [0.0 as Float; 200];
        for (i, val) in data.iter_mut().enumerate() {
            *val = float::sin(i as Float * 0.3);
        }
        let se = sample_entropy(&data, 2, 0.2);
        assert!(
            se > 0.0,
            "Periodic signal should have SampEn > 0, got {}",
            se
        );
    }

    #[test]
    fn test_sampen_regularity_ordering() {
        // Constant < periodic complexity
        let constant = [3.0 as Float; 100];
        let mut periodic = [0.0 as Float; 100];
        for (i, val) in periodic.iter_mut().enumerate() {
            *val = float::sin(i as Float * 0.5);
        }
        let se_const = sample_entropy(&constant, 2, 0.2);
        let se_periodic = sample_entropy(&periodic, 2, 0.2);
        assert!(
            se_const < se_periodic,
            "Constant ({}) should have lower SampEn than periodic ({})",
            se_const,
            se_periodic
        );
    }

    #[test]
    fn test_sampen_returns_inf() {
        // Monotonic data: templates of length m=2 may match but m+1 won't
        // with tight enough tolerance, or no matches at m at all -> inf
        let mut data = [0.0 as Float; 50];
        for (i, val) in data.iter_mut().enumerate() {
            *val = i as Float;
        }
        let se = sample_entropy(&data, 2, 0.001);
        assert!(se.is_infinite(), "Should return inf, got {}", se);
    }

    #[test]
    fn test_sampen_non_negative() {
        let data = [
            1.0,
            2.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            2.0 as Float,
        ];
        let se = sample_entropy(&data, 2, 0.5);
        assert!(se >= 0.0, "SampEn should be non-negative, got {}", se);
    }

    #[test]
    #[should_panic(expected = "data must not be empty")]
    fn test_sampen_empty() {
        sample_entropy(&[], 2, 0.2);
    }

    #[test]
    #[should_panic(expected = "m must be >= 1")]
    fn test_sampen_m_zero() {
        sample_entropy(&[1.0, 2.0, 3.0, 4.0], 0, 0.2);
    }

    #[test]
    #[should_panic(expected = "r must be > 0")]
    fn test_sampen_r_zero() {
        sample_entropy(&[1.0, 2.0, 3.0, 4.0], 2, 0.0);
    }

    #[test]
    #[should_panic(expected = "data.len()")]
    fn test_sampen_too_short() {
        sample_entropy(&[1.0, 2.0, 3.0], 2, 0.2);
    }

    // =========================================================================
    // Approximate Entropy tests
    // =========================================================================

    #[test]
    fn test_apen_constant_signal() {
        let data = [3.0 as Float; 50];
        let ae = approximate_entropy(&data, 2, 0.2);
        assert!(
            ae < 0.01,
            "Constant signal ApEn should be near 0, got {}",
            ae
        );
    }

    #[test]
    fn test_apen_always_finite() {
        // Even with tight tolerance, ApEn is finite (self-matches guarantee count >= 1)
        let mut data = [0.0 as Float; 50];
        for (i, val) in data.iter_mut().enumerate() {
            *val = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let ae = approximate_entropy(&data, 2, 0.001);
        assert!(ae.is_finite(), "ApEn should always be finite, got {}", ae);
    }

    #[test]
    fn test_apen_non_negative() {
        let data = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0 as Float];
        let ae = approximate_entropy(&data, 2, 0.5);
        assert!(ae >= 0.0, "ApEn should be non-negative, got {}", ae);
    }

    #[test]
    #[should_panic(expected = "data must not be empty")]
    fn test_apen_empty() {
        approximate_entropy(&[], 2, 0.2);
    }

    #[test]
    #[should_panic(expected = "m must be >= 1")]
    fn test_apen_m_zero() {
        approximate_entropy(&[1.0, 2.0, 3.0, 4.0], 0, 0.2);
    }

    // =========================================================================
    // Spectral Entropy tests
    // =========================================================================

    #[test]
    fn test_spectral_entropy_flat_normalized() {
        let psd = [1.0 as Float; 64];
        let h = spectral_entropy(&psd, true);
        let tol = if cfg!(feature = "f32") { 1e-4 } else { 1e-10 };
        assert!(
            (h - 1.0).abs() < tol,
            "Flat PSD normalized should be 1.0, got {}",
            h
        );
    }

    #[test]
    fn test_spectral_entropy_flat_unnormalized() {
        let n = 64;
        let psd = [1.0 as Float; 64];
        let h = spectral_entropy(&psd, false);
        let expected = float::log(n as Float);
        let tol = if cfg!(feature = "f32") { 1e-4 } else { 1e-10 };
        assert!(
            (h - expected).abs() < tol,
            "Flat PSD unnormalized should be ln({}), got {}",
            n,
            h
        );
    }

    #[test]
    fn test_spectral_entropy_single_peak() {
        let mut psd = [0.0 as Float; 64];
        psd[10] = 100.0;
        let h = spectral_entropy(&psd, true);
        assert!(h < 0.01, "Single peak should give near 0, got {}", h);
    }

    #[test]
    fn test_spectral_entropy_two_peaks() {
        let mut psd = [0.0 as Float; 64];
        psd[10] = 1.0;
        psd[30] = 1.0;
        let h = spectral_entropy(&psd, true);
        let expected = float::log(2.0) / float::log(64.0);
        assert!(
            (h - expected).abs() < 1e-10,
            "Two equal peaks should give ln(2)/ln(N), got {} expected {}",
            h,
            expected
        );
    }

    #[test]
    fn test_spectral_entropy_normalized_range() {
        // Random-ish PSD values
        let psd = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 as Float];
        let h = spectral_entropy(&psd, true);
        assert!(
            h >= 0.0,
            "Normalized spectral entropy should be >= 0, got {}",
            h
        );
        assert!(
            h <= 1.0,
            "Normalized spectral entropy should be <= 1, got {}",
            h
        );
    }

    #[test]
    fn test_spectral_entropy_all_zero() {
        let psd = [0.0 as Float; 32];
        let h = spectral_entropy(&psd, true);
        assert!(h.abs() < 1e-10, "All-zero PSD should give 0, got {}", h);
    }

    #[test]
    #[should_panic(expected = "PSD must not be empty")]
    fn test_spectral_entropy_empty() {
        spectral_entropy(&[], true);
    }

    // =========================================================================
    // Multiscale Entropy tests
    // =========================================================================

    #[test]
    fn test_mse_scale1_equals_sampen() {
        let data = [
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0,
            1.0,
            2.0 as Float,
        ];
        let mut scratch = [0.0 as Float; 20];
        let mse1 = multiscale_entropy(&data, 1, 2, 0.5, &mut scratch);
        let se = sample_entropy(&data, 2, 0.5);
        assert!(
            (mse1 - se).abs() < 1e-10,
            "MSE scale=1 ({}) should equal SampEn ({})",
            mse1,
            se
        );
    }

    #[test]
    fn test_mse_constant_zero() {
        let data = [5.0 as Float; 100];
        let mut scratch = [0.0 as Float; 50];
        // Scale 2: coarse_len = 50, still constant
        let mse = multiscale_entropy(&data, 2, 2, 0.2, &mut scratch);
        assert!(
            mse.abs() < 1e-10,
            "Constant signal MSE should be 0, got {}",
            mse
        );
    }

    #[test]
    fn test_mse_non_negative() {
        let mut data = [0.0 as Float; 100];
        for (i, val) in data.iter_mut().enumerate() {
            *val = float::sin(i as Float * 0.3);
        }
        let mut scratch = [0.0 as Float; 50];
        let mse = multiscale_entropy(&data, 2, 2, 0.2, &mut scratch);
        assert!(mse >= 0.0, "MSE should be non-negative, got {}", mse);
    }

    #[test]
    #[should_panic(expected = "scale must be >= 1")]
    fn test_mse_scale_zero() {
        let data = [1.0 as Float; 100];
        let mut scratch = [0.0 as Float; 100];
        multiscale_entropy(&data, 0, 2, 0.2, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "coarse_len")]
    fn test_mse_too_short_after_coarsening() {
        // 10 samples, scale=5 -> coarse_len=2, which is <= m+1=3
        let data = [1.0 as Float; 10];
        let mut scratch = [0.0 as Float; 10];
        multiscale_entropy(&data, 5, 2, 0.2, &mut scratch);
    }
}
