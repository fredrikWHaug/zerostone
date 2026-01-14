//! Cross-correlation and auto-correlation primitives.
//!
//! This module provides signal correlation functions for similarity measurement,
//! time delay estimation, and feature extraction in BCI applications.
//!
//! # Overview
//!
//! Cross-correlation measures the similarity between two signals as a function of
//! time lag. It is widely used in BCI for:
//! - Inter-channel correlation analysis
//! - Template matching for evoked potentials
//! - Time delay estimation between electrode pairs
//! - Feature extraction for motor imagery classification
//!
//! # Functions
//!
//! - [`xcorr`] - Cross-correlation of two signals (full output)
//! - [`xcorr_into`] - Cross-correlation with explicit output buffer
//! - [`autocorr`] - Auto-correlation (optimized for symmetry)
//! - [`autocorr_into`] - Auto-correlation with explicit output buffer
//! - [`correlation_lags`] - Compute lag indices for correlation output
//!
//! # Normalization Options
//!
//! The [`Normalization`] enum provides four options matching MATLAB's `xcorr`:
//! - `None` - Raw correlation values
//! - `Biased` - Divided by N (length of longer signal)
//! - `Unbiased` - Divided by (N - |lag|), removes bias from edge effects
//! - `Coeff` - Normalized so autocorrelation at lag 0 equals 1.0
//!
//! # Example
//!
//! ```
//! use zerostone::xcorr::{xcorr, autocorr, Normalization};
//!
//! // Detect time delay between two signals
//! let x = [0.0f32, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0];
//! let y = [0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0];  // x delayed by 2
//!
//! let mut output = [0.0f32; 15];  // N + M - 1 = 8 + 8 - 1
//! xcorr(&x, &y, &mut output, Normalization::None);
//!
//! // Find peak lag
//! let (max_idx, _) = output.iter().enumerate()
//!     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//!     .unwrap();
//! let peak_lag = max_idx as i32 - 7;  // lag = index - (M - 1)
//! assert_eq!(peak_lag, 2);  // Detected 2-sample delay
//! ```
//!
//! # Auto-correlation Example
//!
//! ```
//! use zerostone::xcorr::{autocorr, Normalization};
//!
//! let signal = [1.0f32, 2.0, 3.0, 2.0, 1.0];
//! let mut output = [0.0f32; 9];  // 2*N - 1
//!
//! autocorr(&signal, &mut output, Normalization::Coeff);
//!
//! // With Coeff normalization, center (lag 0) should be 1.0
//! let center = output.len() / 2;
//! assert!((output[center] - 1.0).abs() < 1e-6);
//!
//! // Auto-correlation is symmetric
//! for i in 0..center {
//!     assert!((output[i] - output[output.len() - 1 - i]).abs() < 1e-6);
//! }
//! ```

/// Normalization method for correlation output.
///
/// Different normalization options are useful for different applications:
/// - `None`: Raw correlation, preserves absolute magnitude information
/// - `Biased`: Simple scaling by signal length, introduces edge bias
/// - `Unbiased`: Corrects for reduced overlap at large lags
/// - `Coeff`: Produces correlation coefficients in [-1, 1] range
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// Raw correlation values (no normalization).
    #[default]
    None,
    /// Biased estimate: divide by N (length of longer signal).
    Biased,
    /// Unbiased estimate: divide by (N - |lag|).
    Unbiased,
    /// Coefficient normalization: autocorrelation at lag 0 equals 1.0.
    Coeff,
}

/// Compute the cross-correlation of two signals.
///
/// Computes the discrete cross-correlation:
/// ```text
/// (x ⋆ y)[k] = Σ_n x[n] · y[n + k]
/// ```
///
/// The output has length `N + M - 1` where `N` and `M` are the input lengths.
/// The lag at output index `k` is `k - (M - 1)`, ranging from `-(M-1)` to `N-1`.
///
/// # Arguments
///
/// * `x` - First input signal of length N
/// * `y` - Second input signal of length M
/// * `output` - Output buffer of length N + M - 1
/// * `norm` - Normalization method
///
/// # Performance
///
/// Time complexity: O(N × M). For typical BCI segment sizes (256-1024 samples),
/// direct computation is efficient. For very large signals, consider FFT-based
/// correlation.
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{xcorr, Normalization};
///
/// let x = [1.0f32, 2.0, 3.0];
/// let y = [1.0f32, 0.5];
/// let mut output = [0.0f32; 4];  // 3 + 2 - 1 = 4
///
/// xcorr(&x, &y, &mut output, Normalization::None);
///
/// // output[0] is lag -(M-1) = -1: x[0]*0 + x[1]*y[0] doesn't exist, only partial overlap
/// // output[1] is lag 0: x[0]*y[0] + x[1]*y[1] = 1*1 + 2*0.5 = 2.0
/// assert!((output[1] - 2.0).abs() < 1e-6);
/// ```
pub fn xcorr<const N: usize, const M: usize, const OUT: usize>(
    x: &[f32; N],
    y: &[f32; M],
    output: &mut [f32; OUT],
    norm: Normalization,
) {
    // Compute energy for coefficient normalization
    let (energy_x, energy_y) = if norm == Normalization::Coeff {
        let ex: f32 = x.iter().map(|&v| v * v).sum();
        let ey: f32 = y.iter().map(|&v| v * v).sum();
        (ex, ey)
    } else {
        (0.0, 0.0)
    };

    let norm_factor = if norm == Normalization::Coeff {
        let denom = libm::sqrtf(energy_x * energy_y);
        if denom > 1e-10 { denom } else { 1.0 }
    } else {
        1.0
    };

    let n = N as i32;
    let m = M as i32;
    let max_len = N.max(M) as f32;

    for (k, out_val) in output.iter_mut().enumerate() {
        let lag = k as i32 - (m - 1);

        let mut sum = 0.0f32;
        let mut count = 0usize;

        // Standard cross-correlation: (x ⋆ y)[lag] = Σ_n x[n] * y[n + lag]
        // n must satisfy: 0 <= n < N and 0 <= n + lag < M
        // So: n >= max(0, -lag) and n < min(N, M - lag)
        let n_start = (-lag).max(0) as usize;
        let n_end_signed = (n.min(m - lag)).max(0);
        let n_end = n_end_signed as usize;

        if n_end > n_start {
            for (i, &x_val) in x.iter().enumerate().take(n_end).skip(n_start) {
                let y_idx = (i as i32 + lag) as usize;
                sum += x_val * y[y_idx];
                count += 1;
            }
        }

        *out_val = match norm {
            Normalization::None => sum,
            Normalization::Biased => sum / max_len,
            Normalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            Normalization::Coeff => sum / norm_factor,
        };
    }
}

/// Compute cross-correlation into a dynamically-sized slice.
///
/// This variant accepts a slice instead of a fixed-size array, returning
/// the number of output samples written.
///
/// # Arguments
///
/// * `x` - First input signal
/// * `y` - Second input signal
/// * `output` - Output slice (must have length >= N + M - 1)
/// * `norm` - Normalization method
///
/// # Returns
///
/// Number of output samples written (N + M - 1).
///
/// # Panics
///
/// Panics if `output.len() < N + M - 1`.
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{xcorr_into, Normalization};
///
/// let x = [1.0f32, 2.0, 3.0, 4.0];
/// let y = [1.0f32, 1.0];
/// let mut output = vec![0.0f32; 10];  // Oversized buffer
///
/// let written = xcorr_into(&x, &y, &mut output, Normalization::None);
/// assert_eq!(written, 5);  // 4 + 2 - 1 = 5
/// ```
pub fn xcorr_into<const N: usize, const M: usize>(
    x: &[f32; N],
    y: &[f32; M],
    output: &mut [f32],
    norm: Normalization,
) -> usize {
    let out_len = N + M - 1;
    assert!(
        output.len() >= out_len,
        "Output buffer too small: need {}, got {}",
        out_len,
        output.len()
    );

    // Compute energy for coefficient normalization
    let (energy_x, energy_y) = if norm == Normalization::Coeff {
        let ex: f32 = x.iter().map(|&v| v * v).sum();
        let ey: f32 = y.iter().map(|&v| v * v).sum();
        (ex, ey)
    } else {
        (0.0, 0.0)
    };

    let norm_factor = if norm == Normalization::Coeff {
        let denom = libm::sqrtf(energy_x * energy_y);
        if denom > 1e-10 { denom } else { 1.0 }
    } else {
        1.0
    };

    let n = N as i32;
    let m = M as i32;
    let max_len = N.max(M) as f32;

    for (k, out_val) in output.iter_mut().enumerate().take(out_len) {
        let lag = k as i32 - (m - 1);

        let mut sum = 0.0f32;
        let mut count = 0usize;

        // Standard cross-correlation: (x ⋆ y)[lag] = Σ_n x[n] * y[n + lag]
        let n_start = (-lag).max(0) as usize;
        let n_end_signed = (n.min(m - lag)).max(0);
        let n_end = n_end_signed as usize;

        if n_end > n_start {
            for (i, &x_val) in x.iter().enumerate().take(n_end).skip(n_start) {
                let y_idx = (i as i32 + lag) as usize;
                sum += x_val * y[y_idx];
                count += 1;
            }
        }

        *out_val = match norm {
            Normalization::None => sum,
            Normalization::Biased => sum / max_len,
            Normalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            Normalization::Coeff => sum / norm_factor,
        };
    }

    out_len
}

/// Compute the auto-correlation of a signal.
///
/// Auto-correlation is the cross-correlation of a signal with itself.
/// This function is optimized to exploit the symmetry property:
/// `autocorr[k] == autocorr[-k]`.
///
/// The output has length `2*N - 1`. The lag at output index `k` is
/// `k - (N - 1)`, ranging from `-(N-1)` to `N-1`.
///
/// # Arguments
///
/// * `x` - Input signal of length N
/// * `output` - Output buffer of length 2*N - 1
/// * `norm` - Normalization method
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{autocorr, Normalization};
///
/// let signal = [1.0f32, 2.0, 3.0, 2.0, 1.0];
/// let mut output = [0.0f32; 9];  // 2*5 - 1 = 9
///
/// autocorr(&signal, &mut output, Normalization::None);
///
/// // Peak at center (lag 0)
/// let center = 4;
/// let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
/// assert_eq!(output[center], max_val);
///
/// // Symmetric
/// assert!((output[0] - output[8]).abs() < 1e-6);
/// assert!((output[1] - output[7]).abs() < 1e-6);
/// ```
pub fn autocorr<const N: usize, const OUT: usize>(
    x: &[f32; N],
    output: &mut [f32; OUT],
    norm: Normalization,
) {
    let out_len = 2 * N - 1;
    let center = N - 1;

    // Compute energy at lag 0 (needed for all normalizations, and for symmetry optimization)
    let energy: f32 = x.iter().map(|&v| v * v).sum();

    // For coefficient normalization
    let norm_factor = if norm == Normalization::Coeff {
        if energy > 1e-10 {
            energy
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Compute only non-negative lags and mirror
    // Lag 0 is at index center
    for lag in 0..N {
        let mut sum = 0.0f32;
        let count = N - lag;

        for i in 0..count {
            sum += x[i] * x[i + lag];
        }

        let normalized = match norm {
            Normalization::None => sum,
            Normalization::Biased => sum / N as f32,
            Normalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            Normalization::Coeff => sum / norm_factor,
        };

        // Place at center + lag (positive lag)
        if center + lag < out_len && center + lag < OUT {
            output[center + lag] = normalized;
        }

        // Mirror to center - lag (negative lag), except for lag 0
        if lag > 0 && center >= lag && center - lag < OUT {
            output[center - lag] = normalized;
        }
    }
}

/// Compute auto-correlation into a dynamically-sized slice.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `output` - Output slice (must have length >= 2*N - 1)
/// * `norm` - Normalization method
///
/// # Returns
///
/// Number of output samples written (2*N - 1).
///
/// # Panics
///
/// Panics if `output.len() < 2*N - 1`.
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{autocorr_into, Normalization};
///
/// let signal = [1.0f32, 2.0, 3.0];
/// let mut output = vec![0.0f32; 10];
///
/// let written = autocorr_into(&signal, &mut output, Normalization::Coeff);
/// assert_eq!(written, 5);  // 2*3 - 1 = 5
///
/// // Lag 0 should be 1.0 with Coeff normalization
/// assert!((output[2] - 1.0).abs() < 1e-6);
/// ```
pub fn autocorr_into<const N: usize>(
    x: &[f32; N],
    output: &mut [f32],
    norm: Normalization,
) -> usize {
    let out_len = 2 * N - 1;
    assert!(
        output.len() >= out_len,
        "Output buffer too small: need {}, got {}",
        out_len,
        output.len()
    );

    let center = N - 1;
    let energy: f32 = x.iter().map(|&v| v * v).sum();

    let norm_factor = if norm == Normalization::Coeff {
        if energy > 1e-10 {
            energy
        } else {
            1.0
        }
    } else {
        1.0
    };

    for lag in 0..N {
        let mut sum = 0.0f32;
        let count = N - lag;

        for i in 0..count {
            sum += x[i] * x[i + lag];
        }

        let normalized = match norm {
            Normalization::None => sum,
            Normalization::Biased => sum / N as f32,
            Normalization::Unbiased => {
                if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                }
            }
            Normalization::Coeff => sum / norm_factor,
        };

        output[center + lag] = normalized;
        if lag > 0 {
            output[center - lag] = normalized;
        }
    }

    out_len
}

/// Compute lag indices for correlation output.
///
/// Returns the minimum and maximum lag values for a cross-correlation
/// of signals with lengths N and M.
///
/// # Returns
///
/// Tuple of (min_lag, max_lag) where:
/// - min_lag = -(M - 1)
/// - max_lag = N - 1
///
/// # Example
///
/// ```
/// use zerostone::xcorr::correlation_lags;
///
/// // For signals of length 5 and 3
/// let (min_lag, max_lag) = correlation_lags::<5, 3>();
/// assert_eq!(min_lag, -2);  // -(3-1)
/// assert_eq!(max_lag, 4);   // 5-1
///
/// // Output length = max_lag - min_lag + 1 = 4 - (-2) + 1 = 7 = N + M - 1
/// ```
pub const fn correlation_lags<const N: usize, const M: usize>() -> (i32, i32) {
    let min_lag = -((M as i32) - 1);
    let max_lag = (N as i32) - 1;
    (min_lag, max_lag)
}

/// Convert output index to lag value.
///
/// For cross-correlation output of signals with length N and M,
/// converts array index to the corresponding lag value.
///
/// # Arguments
///
/// * `index` - Index into correlation output array
///
/// # Returns
///
/// Lag value at that index.
///
/// # Example
///
/// ```
/// use zerostone::xcorr::index_to_lag;
///
/// // For correlation of signals length 5 and 3, output has 7 elements
/// // Index 0 corresponds to lag -(M-1) = -2
/// assert_eq!(index_to_lag::<5, 3>(0), -2);
/// assert_eq!(index_to_lag::<5, 3>(2), 0);   // Center
/// assert_eq!(index_to_lag::<5, 3>(6), 4);
/// ```
pub const fn index_to_lag<const N: usize, const M: usize>(index: usize) -> i32 {
    index as i32 - (M as i32 - 1)
}

/// Convert lag value to output index.
///
/// For cross-correlation output of signals with length N and M,
/// converts lag value to the corresponding array index.
///
/// # Arguments
///
/// * `lag` - Lag value
///
/// # Returns
///
/// Index into correlation output array.
///
/// # Example
///
/// ```
/// use zerostone::xcorr::lag_to_index;
///
/// // For correlation of signals length 5 and 3
/// assert_eq!(lag_to_index::<5, 3>(-2), 0);
/// assert_eq!(lag_to_index::<5, 3>(0), 2);
/// assert_eq!(lag_to_index::<5, 3>(4), 6);
/// ```
pub const fn lag_to_index<const N: usize, const M: usize>(lag: i32) -> usize {
    (lag + (M as i32 - 1)) as usize
}

/// Batch cross-correlation for multiple channel pairs.
///
/// Computes cross-correlation for each corresponding pair of channels.
///
/// # Arguments
///
/// * `x` - First set of signals (C channels, each of length N)
/// * `y` - Second set of signals (C channels, each of length M)
/// * `output` - Output buffers (C channels, each of length N + M - 1)
/// * `norm` - Normalization method
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{xcorr_batch, Normalization};
///
/// let x: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]];
/// let y: [[f32; 3]; 2] = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]];
/// let mut output = [[0.0f32; 6]; 2];  // 4 + 3 - 1 = 6
///
/// xcorr_batch(&x, &y, &mut output, Normalization::None);
/// // Each channel processed independently
/// ```
pub fn xcorr_batch<const N: usize, const M: usize, const OUT: usize, const C: usize>(
    x: &[[f32; N]; C],
    y: &[[f32; M]; C],
    output: &mut [[f32; OUT]; C],
    norm: Normalization,
) {
    for ch in 0..C {
        xcorr(&x[ch], &y[ch], &mut output[ch], norm);
    }
}

/// Batch auto-correlation for multiple channels.
///
/// Computes auto-correlation for each channel independently.
///
/// # Arguments
///
/// * `x` - Input signals (C channels, each of length N)
/// * `output` - Output buffers (C channels, each of length 2*N - 1)
/// * `norm` - Normalization method
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{autocorr_batch, Normalization};
///
/// let signals: [[f32; 4]; 3] = [
///     [1.0, 2.0, 2.0, 1.0],
///     [0.0, 1.0, 1.0, 0.0],
///     [1.0, 0.0, 0.0, 1.0],
/// ];
/// let mut output = [[0.0f32; 7]; 3];  // 2*4 - 1 = 7
///
/// autocorr_batch(&signals, &mut output, Normalization::Coeff);
///
/// // Each channel's autocorr at lag 0 should be 1.0
/// for ch in 0..3 {
///     assert!((output[ch][3] - 1.0).abs() < 1e-6);
/// }
/// ```
pub fn autocorr_batch<const N: usize, const OUT: usize, const C: usize>(
    x: &[[f32; N]; C],
    output: &mut [[f32; OUT]; C],
    norm: Normalization,
) {
    for ch in 0..C {
        autocorr(&x[ch], &mut output[ch], norm);
    }
}

/// Find the peak (maximum) correlation and its lag.
///
/// Utility function to find the lag at which correlation is maximum,
/// commonly used for time delay estimation.
///
/// # Arguments
///
/// * `correlation` - Correlation output array
///
/// # Returns
///
/// Tuple of (peak_index, peak_value).
///
/// # Example
///
/// ```
/// use zerostone::xcorr::{xcorr, find_peak, index_to_lag, Normalization};
///
/// // Signal delayed by 2 samples
/// let x = [0.0f32, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0];
/// let y = [0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0];
///
/// let mut corr = [0.0f32; 15];
/// xcorr(&x, &y, &mut corr, Normalization::None);
///
/// let (peak_idx, _peak_val) = find_peak(&corr);
/// let lag = index_to_lag::<8, 8>(peak_idx);
/// assert_eq!(lag, 2);  // y is x delayed by 2
/// ```
pub fn find_peak(correlation: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (i, &val) in correlation.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;

    #[test]
    fn test_xcorr_identical_signals() {
        let signal = [1.0f32, 2.0, 3.0, 2.0, 1.0];
        let mut output = [0.0f32; 9];
        xcorr(&signal, &signal, &mut output, Normalization::None);

        // Peak should be at center (lag 0)
        let center = 4;
        let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(output[center], max_val);
    }

    #[test]
    fn test_xcorr_delayed_signal() {
        // x has a peak at index 2
        let x = [0.0f32, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        // y is x shifted right by 2 (peak at index 4)
        let y = [0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0];

        let mut output = [0.0f32; 15];
        xcorr(&x, &y, &mut output, Normalization::None);

        let (peak_idx, _) = find_peak(&output);
        let lag = index_to_lag::<8, 8>(peak_idx);

        // y is x delayed by 2, so lag should be +2
        assert_eq!(lag, 2);
    }

    #[test]
    fn test_xcorr_negative_delay() {
        // y leads x (negative lag should be peak)
        let x = [0.0f32, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0];
        let y = [0.0f32, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let mut output = [0.0f32; 15];
        xcorr(&x, &y, &mut output, Normalization::None);

        let (peak_idx, _) = find_peak(&output);
        let lag = index_to_lag::<8, 8>(peak_idx);

        assert_eq!(lag, -2);
    }

    #[test]
    fn test_autocorr_symmetry() {
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 9];
        autocorr(&signal, &mut output, Normalization::None);

        // Auto-correlation must be symmetric
        for i in 0..4 {
            assert!(
                (output[i] - output[8 - i]).abs() < 1e-6,
                "Asymmetry at index {}: {} vs {}",
                i,
                output[i],
                output[8 - i]
            );
        }
    }

    #[test]
    fn test_autocorr_peak_at_center() {
        let signal = [1.0f32, 3.0, 2.0, 4.0, 1.0];
        let mut output = [0.0f32; 9];
        autocorr(&signal, &mut output, Normalization::None);

        // Peak must be at center (lag 0)
        let center = 4;
        let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(output[center], max_val);
    }

    #[test]
    fn test_normalization_coeff() {
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 9];
        autocorr(&signal, &mut output, Normalization::Coeff);

        // With Coeff normalization, autocorr at lag 0 should be 1.0
        let center = 4;
        assert!(
            (output[center] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {}",
            output[center]
        );
    }

    #[test]
    fn test_normalization_biased() {
        let signal = [1.0f32, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 7];
        autocorr(&signal, &mut output, Normalization::Biased);

        // Lag 0: sum = 4, biased = 4/4 = 1.0
        let center = 3;
        assert!((output[center] - 1.0).abs() < 1e-6);

        // Lag 1: sum = 3, biased = 3/4 = 0.75
        assert!((output[center + 1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_unbiased() {
        let signal = [1.0f32, 1.0, 1.0, 1.0];
        let mut output = [0.0f32; 7];
        autocorr(&signal, &mut output, Normalization::Unbiased);

        // Lag 0: sum = 4, count = 4, unbiased = 4/4 = 1.0
        let center = 3;
        assert!((output[center] - 1.0).abs() < 1e-6);

        // Lag 1: sum = 3, count = 3, unbiased = 3/3 = 1.0
        assert!((output[center + 1] - 1.0).abs() < 1e-6);

        // Lag 2: sum = 2, count = 2, unbiased = 2/2 = 1.0
        assert!((output[center + 2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_lags() {
        let (min_lag, max_lag) = correlation_lags::<5, 3>();
        assert_eq!(min_lag, -2);
        assert_eq!(max_lag, 4);

        let (min_lag, max_lag) = correlation_lags::<8, 8>();
        assert_eq!(min_lag, -7);
        assert_eq!(max_lag, 7);
    }

    #[test]
    fn test_index_to_lag() {
        // For signals of length 5 and 3
        assert_eq!(index_to_lag::<5, 3>(0), -2);
        assert_eq!(index_to_lag::<5, 3>(2), 0);
        assert_eq!(index_to_lag::<5, 3>(6), 4);
    }

    #[test]
    fn test_lag_to_index() {
        assert_eq!(lag_to_index::<5, 3>(-2), 0);
        assert_eq!(lag_to_index::<5, 3>(0), 2);
        assert_eq!(lag_to_index::<5, 3>(4), 6);
    }

    #[test]
    fn test_xcorr_into() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let y = [1.0f32, 1.0];
        let mut output = vec![0.0f32; 10];

        let written = xcorr_into(&x, &y, &mut output, Normalization::None);
        assert_eq!(written, 5);
    }

    #[test]
    fn test_autocorr_into() {
        let signal = [1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 10];

        let written = autocorr_into(&signal, &mut output, Normalization::Coeff);
        assert_eq!(written, 5);
        assert!((output[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_xcorr_batch() {
        let x: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]];
        let y: [[f32; 3]; 2] = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let mut output = [[0.0f32; 6]; 2];

        xcorr_batch(&x, &y, &mut output, Normalization::None);

        // Verify both channels computed
        assert!(output[0].iter().any(|&v| v != 0.0));
        assert!(output[1].iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_autocorr_batch() {
        let signals: [[f32; 4]; 3] = [
            [1.0, 2.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let mut output = [[0.0f32; 7]; 3];

        autocorr_batch(&signals, &mut output, Normalization::Coeff);

        // Each channel's autocorr at lag 0 should be 1.0
        for (ch, ch_output) in output.iter().enumerate() {
            assert!(
                (ch_output[3] - 1.0).abs() < 1e-6,
                "Channel {} has autocorr[0] = {}",
                ch,
                ch_output[3]
            );
        }
    }

    #[test]
    fn test_find_peak() {
        let corr = [1.0f32, 3.0, 7.0, 5.0, 2.0];
        let (idx, val) = find_peak(&corr);
        assert_eq!(idx, 2);
        assert_eq!(val, 7.0);
    }

    #[test]
    fn test_different_length_signals() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0f32, 1.0];
        let mut output = [0.0f32; 6]; // 5 + 2 - 1 = 6

        xcorr(&x, &y, &mut output, Normalization::None);

        // Verify output has correct length worth of data
        // At lag 0: x[0]*y[0] + x[1]*y[1] = 1 + 2 = 3
        let lag0_idx = lag_to_index::<5, 2>(0);
        assert!((output[lag0_idx] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_element_signals() {
        let x = [5.0f32];
        let y = [3.0f32];
        let mut output = [0.0f32; 1];

        xcorr(&x, &y, &mut output, Normalization::None);
        assert!((output[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_signal() {
        let x = [0.0f32; 4];
        let y = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 7];

        xcorr(&x, &y, &mut output, Normalization::None);

        // All zeros correlated with anything should give zeros
        for val in output {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_impulse_response() {
        // Correlation with impulse gives shifted copy of signal
        let x = [1.0f32, 2.0, 3.0, 2.0, 1.0];
        let impulse = [0.0f32, 0.0, 1.0, 0.0, 0.0];
        let mut output = [0.0f32; 9];

        xcorr(&x, &impulse, &mut output, Normalization::None);

        // Peak should be at lag 0
        let (peak_idx, _) = find_peak(&output);
        assert_eq!(index_to_lag::<5, 5>(peak_idx), 0);
    }
}
