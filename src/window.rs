//! Window functions for spectral analysis.
//!
//! Window functions reduce spectral leakage when computing the Fourier transform
//! of finite-length signals. This module provides common window types used in
//! BCI signal processing, computed on-the-fly with zero allocation.
//!
//! # Window Types
//!
//! | Window | Main Lobe Width | Sidelobe Level | Use Case |
//! |--------|-----------------|----------------|----------|
//! | Rectangular | Narrowest | -13 dB | Maximum frequency resolution |
//! | Hann | Moderate | -31 dB | General purpose |
//! | Hamming | Moderate | -42 dB | Speech/audio analysis |
//! | Blackman | Wide | -58 dB | Low sidelobe requirements |
//! | BlackmanHarris | Widest | -92 dB | Minimum spectral leakage |
//!
//! # Example
//!
//! ```
//! use zerostone::{WindowType, apply_window, Fft, Complex};
//!
//! // Prepare signal for spectral analysis
//! let mut signal = [1.0f32; 256];
//!
//! // Apply Hann window to reduce spectral leakage
//! apply_window(&mut signal, WindowType::Hann);
//!
//! // Now compute FFT
//! let fft = Fft::<256>::new();
//! let mut spectrum: [Complex; 256] = core::array::from_fn(|i| Complex::from_real(signal[i]));
//! fft.forward(&mut spectrum);
//! ```
//!
//! # On-the-fly Computation
//!
//! For streaming applications where you need individual coefficients:
//!
//! ```
//! use zerostone::{WindowType, window_coefficient};
//!
//! // Compute window coefficient for sample 10 of a 256-point window
//! let coeff = window_coefficient(WindowType::Hamming, 10, 256);
//! ```

use core::f64::consts::PI;

/// Window function types for spectral analysis.
///
/// Each window type provides a different trade-off between main lobe width
/// (frequency resolution) and sidelobe level (spectral leakage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular window (no windowing).
    ///
    /// Provides maximum frequency resolution but worst spectral leakage (-13 dB).
    /// Use only when the signal is already periodic within the frame.
    Rectangular,

    /// Hann window (raised cosine).
    ///
    /// Good general-purpose window with -31 dB sidelobe level.
    /// Also known as "Hanning" window. Commonly used in EEG spectral analysis.
    Hann,

    /// Hamming window.
    ///
    /// Similar to Hann but with better sidelobe suppression (-42 dB).
    /// Does not taper to zero at endpoints, which can be advantageous
    /// for some applications.
    Hamming,

    /// Blackman window.
    ///
    /// Three-term window with excellent sidelobe suppression (-58 dB).
    /// Wider main lobe than Hann/Hamming but much lower leakage.
    Blackman,

    /// Blackman-Harris window (4-term).
    ///
    /// Four-term window with exceptional sidelobe suppression (-92 dB).
    /// Best choice when spectral leakage must be minimized.
    /// Widest main lobe of the provided windows.
    BlackmanHarris,
}

/// Computes a single window coefficient.
///
/// This function computes the window value at a specific index, enabling
/// on-the-fly computation without storing the entire window in memory.
///
/// # Arguments
///
/// * `window` - The window type to compute
/// * `index` - Sample index (0 to length-1)
/// * `length` - Total window length
///
/// # Returns
///
/// Window coefficient in range \[0.0, 1.0\] for most windows.
///
/// # Panics
///
/// Panics if `index >= length` or `length == 0`.
///
/// # Example
///
/// ```
/// use zerostone::{WindowType, window_coefficient};
///
/// // Hann window is symmetric and zero at endpoints
/// let n = 64;
/// let first = window_coefficient(WindowType::Hann, 0, n);
/// let last = window_coefficient(WindowType::Hann, n - 1, n);
/// let middle = window_coefficient(WindowType::Hann, n / 2, n);
///
/// assert!(first < 0.01);  // Near zero at start
/// assert!(last < 0.01);   // Near zero at end
/// assert!(middle > 0.99); // Maximum at center
/// ```
#[inline]
pub fn window_coefficient(window: WindowType, index: usize, length: usize) -> f32 {
    assert!(length > 0, "Window length must be positive");
    assert!(index < length, "Index must be less than length");

    if length == 1 {
        return 1.0;
    }

    let n = index as f64;
    let len = (length - 1) as f64;
    let ratio = n / len;

    let value = match window {
        WindowType::Rectangular => 1.0,

        // Hann: 0.5 * (1 - cos(2*pi*n/(N-1)))
        WindowType::Hann => 0.5 * (1.0 - libm::cos(2.0 * PI * ratio)),

        // Hamming: 0.54 - 0.46 * cos(2*pi*n/(N-1))
        WindowType::Hamming => 0.54 - 0.46 * libm::cos(2.0 * PI * ratio),

        // Blackman: 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
        WindowType::Blackman => {
            0.42 - 0.5 * libm::cos(2.0 * PI * ratio) + 0.08 * libm::cos(4.0 * PI * ratio)
        }

        // Blackman-Harris (4-term):
        // 0.35875 - 0.48829*cos(2*pi*n/(N-1)) + 0.14128*cos(4*pi*n/(N-1)) - 0.01168*cos(6*pi*n/(N-1))
        WindowType::BlackmanHarris => {
            0.35875 - 0.48829 * libm::cos(2.0 * PI * ratio) + 0.14128 * libm::cos(4.0 * PI * ratio)
                - 0.01168 * libm::cos(6.0 * PI * ratio)
        }
    };

    // Clamp to non-negative (some windows can have tiny negative values at endpoints
    // due to floating point precision)
    if value < 0.0 {
        0.0
    } else {
        value as f32
    }
}

/// Applies a window function to a signal buffer in-place.
///
/// This is the primary function for preparing signals for spectral analysis.
/// Each sample is multiplied by the corresponding window coefficient.
///
/// # Arguments
///
/// * `signal` - Signal buffer to window (modified in-place)
/// * `window` - Window type to apply
///
/// # Performance
///
/// O(N) time complexity where N is the signal length.
/// Zero allocation - window coefficients are computed on-the-fly.
///
/// # Example
///
/// ```
/// use zerostone::{WindowType, apply_window};
///
/// let mut signal = [1.0f32; 128];
/// apply_window(&mut signal, WindowType::Hann);
///
/// // Endpoints are now near zero
/// assert!(signal[0] < 0.01);
/// assert!(signal[127] < 0.01);
/// ```
#[inline]
pub fn apply_window(signal: &mut [f32], window: WindowType) {
    let len = signal.len();
    if len == 0 {
        return;
    }

    for (i, sample) in signal.iter_mut().enumerate() {
        *sample *= window_coefficient(window, i, len);
    }
}

/// Applies a window function to a signal buffer in-place (f64 precision).
///
/// Same as [`apply_window`] but for double-precision signals.
/// Use this when working with high dynamic range signals or when
/// maximum numerical precision is required.
///
/// # Arguments
///
/// * `signal` - Signal buffer to window (modified in-place)
/// * `window` - Window type to apply
///
/// # Example
///
/// ```
/// use zerostone::{WindowType, apply_window_f64};
///
/// let mut signal = [1.0f64; 128];
/// apply_window_f64(&mut signal, WindowType::Blackman);
/// ```
#[inline]
pub fn apply_window_f64(signal: &mut [f64], window: WindowType) {
    let len = signal.len();
    if len == 0 {
        return;
    }

    for (i, sample) in signal.iter_mut().enumerate() {
        *sample *= window_coefficient(window, i, len) as f64;
    }
}

/// Computes the coherent gain of a window.
///
/// The coherent gain is the sum of all window coefficients divided by N.
/// It represents the DC gain of the window and is used to normalize
/// amplitude measurements after windowing.
///
/// # Arguments
///
/// * `window` - Window type
/// * `length` - Window length
///
/// # Returns
///
/// Coherent gain (typically between 0.3 and 1.0 depending on window type).
///
/// # Example
///
/// ```
/// use zerostone::{WindowType, coherent_gain};
///
/// // Rectangular window has coherent gain of 1.0
/// let rect_gain = coherent_gain(WindowType::Rectangular, 256);
/// assert!((rect_gain - 1.0).abs() < 1e-6);
///
/// // Hann window has coherent gain of ~0.5
/// let hann_gain = coherent_gain(WindowType::Hann, 256);
/// assert!((hann_gain - 0.5).abs() < 0.01);
/// ```
pub fn coherent_gain(window: WindowType, length: usize) -> f64 {
    if length == 0 {
        return 0.0;
    }

    let sum: f64 = (0..length)
        .map(|i| window_coefficient(window, i, length) as f64)
        .sum();

    sum / length as f64
}

/// Computes the noise equivalent bandwidth (ENBW) of a window.
///
/// ENBW is the bandwidth of an ideal rectangular filter that would
/// pass the same amount of white noise power as the window.
/// Expressed as a ratio to the frequency bin width.
///
/// # Arguments
///
/// * `window` - Window type
/// * `length` - Window length
///
/// # Returns
///
/// ENBW in bins (typically between 1.0 and 2.0).
///
/// # Example
///
/// ```
/// use zerostone::{WindowType, equivalent_noise_bandwidth};
///
/// // Rectangular window has ENBW of 1.0 bin
/// let rect_enbw = equivalent_noise_bandwidth(WindowType::Rectangular, 256);
/// assert!((rect_enbw - 1.0).abs() < 1e-6);
///
/// // Hann window has ENBW of ~1.5 bins
/// let hann_enbw = equivalent_noise_bandwidth(WindowType::Hann, 256);
/// assert!((hann_enbw - 1.5).abs() < 0.05);
/// ```
pub fn equivalent_noise_bandwidth(window: WindowType, length: usize) -> f64 {
    if length == 0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..length {
        let w = window_coefficient(window, i, length) as f64;
        sum += w;
        sum_sq += w * w;
    }

    if sum == 0.0 {
        return 0.0;
    }

    // ENBW = N * sum(w^2) / sum(w)^2
    length as f64 * sum_sq / (sum * sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_window() {
        // Rectangular window should be all ones
        for i in 0..64 {
            let coeff = window_coefficient(WindowType::Rectangular, i, 64);
            assert!((coeff - 1.0).abs() < 1e-6, "Rectangular should be 1.0");
        }
    }

    #[test]
    fn test_hann_window_symmetry() {
        let n = 64;
        for i in 0..n / 2 {
            let left = window_coefficient(WindowType::Hann, i, n);
            let right = window_coefficient(WindowType::Hann, n - 1 - i, n);
            assert!(
                (left - right).abs() < 1e-6,
                "Hann window should be symmetric"
            );
        }
    }

    #[test]
    fn test_hann_window_endpoints() {
        let n = 128;
        let first = window_coefficient(WindowType::Hann, 0, n);
        let last = window_coefficient(WindowType::Hann, n - 1, n);

        // Hann window tapers to zero at endpoints
        assert!(first < 1e-6, "Hann should be zero at start");
        assert!(last < 1e-6, "Hann should be zero at end");
    }

    #[test]
    fn test_hann_window_peak() {
        let n = 128;
        let middle = window_coefficient(WindowType::Hann, n / 2, n);

        // Hann window peaks at center (value = 1.0)
        assert!((middle - 1.0).abs() < 0.01, "Hann should peak at center");
    }

    #[test]
    fn test_hamming_window_endpoints() {
        let n = 128;
        let first = window_coefficient(WindowType::Hamming, 0, n);
        let last = window_coefficient(WindowType::Hamming, n - 1, n);

        // Hamming window does NOT taper to zero (minimum is 0.08)
        assert!(
            (first - 0.08).abs() < 0.01,
            "Hamming should be ~0.08 at endpoints"
        );
        assert!(
            (last - 0.08).abs() < 0.01,
            "Hamming should be ~0.08 at endpoints"
        );
    }

    #[test]
    fn test_blackman_window_symmetry() {
        let n = 64;
        for i in 0..n / 2 {
            let left = window_coefficient(WindowType::Blackman, i, n);
            let right = window_coefficient(WindowType::Blackman, n - 1 - i, n);
            assert!(
                (left - right).abs() < 1e-6,
                "Blackman window should be symmetric"
            );
        }
    }

    #[test]
    fn test_blackman_harris_symmetry() {
        let n = 64;
        for i in 0..n / 2 {
            let left = window_coefficient(WindowType::BlackmanHarris, i, n);
            let right = window_coefficient(WindowType::BlackmanHarris, n - 1 - i, n);
            assert!(
                (left - right).abs() < 1e-6,
                "Blackman-Harris window should be symmetric"
            );
        }
    }

    #[test]
    fn test_apply_window_in_place() {
        let mut signal = [1.0f32; 64];
        apply_window(&mut signal, WindowType::Hann);

        // Check endpoints are near zero
        assert!(signal[0] < 1e-5);
        assert!(signal[63] < 1e-5);

        // Check middle is near 1.0
        assert!((signal[32] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_window_f64() {
        let mut signal = [1.0f64; 64];
        apply_window_f64(&mut signal, WindowType::Hamming);

        // Check Hamming endpoints (~0.08)
        assert!((signal[0] - 0.08).abs() < 0.01);
        assert!((signal[63] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_apply_window_empty() {
        let mut signal: [f32; 0] = [];
        apply_window(&mut signal, WindowType::Hann); // Should not panic
    }

    #[test]
    fn test_coherent_gain_rectangular() {
        let gain = coherent_gain(WindowType::Rectangular, 256);
        assert!(
            (gain - 1.0).abs() < 1e-6,
            "Rectangular coherent gain should be 1.0"
        );
    }

    #[test]
    fn test_coherent_gain_hann() {
        let gain = coherent_gain(WindowType::Hann, 256);
        // Hann window coherent gain is exactly 0.5
        assert!(
            (gain - 0.5).abs() < 0.01,
            "Hann coherent gain should be ~0.5"
        );
    }

    #[test]
    fn test_coherent_gain_hamming() {
        let gain = coherent_gain(WindowType::Hamming, 256);
        // Hamming window coherent gain is 0.54
        assert!(
            (gain - 0.54).abs() < 0.01,
            "Hamming coherent gain should be ~0.54"
        );
    }

    #[test]
    fn test_enbw_rectangular() {
        let enbw = equivalent_noise_bandwidth(WindowType::Rectangular, 256);
        assert!((enbw - 1.0).abs() < 1e-6, "Rectangular ENBW should be 1.0");
    }

    #[test]
    fn test_enbw_hann() {
        let enbw = equivalent_noise_bandwidth(WindowType::Hann, 256);
        // Hann window ENBW is 1.5 bins
        assert!((enbw - 1.5).abs() < 0.05, "Hann ENBW should be ~1.5");
    }

    #[test]
    fn test_enbw_blackman() {
        let enbw = equivalent_noise_bandwidth(WindowType::Blackman, 256);
        // Blackman window ENBW is ~1.73 bins
        assert!((enbw - 1.73).abs() < 0.05, "Blackman ENBW should be ~1.73");
    }

    #[test]
    fn test_window_ordering_by_sidelobe() {
        // Windows ordered by increasing sidelobe suppression have increasing ENBW
        let rect = equivalent_noise_bandwidth(WindowType::Rectangular, 256);
        let hann = equivalent_noise_bandwidth(WindowType::Hann, 256);
        let hamming = equivalent_noise_bandwidth(WindowType::Hamming, 256);
        let blackman = equivalent_noise_bandwidth(WindowType::Blackman, 256);
        let bh = equivalent_noise_bandwidth(WindowType::BlackmanHarris, 256);

        assert!(rect < hann, "Rectangular should have lowest ENBW");
        assert!(hann < blackman, "Hann should have lower ENBW than Blackman");
        assert!(
            blackman < bh,
            "Blackman should have lower ENBW than Blackman-Harris"
        );
        // Hamming is similar to Hann
        assert!((hamming - hann).abs() < 0.2);
    }

    #[test]
    fn test_single_sample_window() {
        // Edge case: single sample window should return 1.0
        let coeff = window_coefficient(WindowType::Hann, 0, 1);
        assert!((coeff - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_two_sample_window() {
        // Edge case: two sample window
        let first = window_coefficient(WindowType::Hann, 0, 2);
        let second = window_coefficient(WindowType::Hann, 1, 2);

        // For N=2, Hann gives [0, 0] since both endpoints are at cos extremes
        assert!(first < 1e-6);
        assert!(second < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Index must be less than length")]
    fn test_coefficient_out_of_bounds() {
        window_coefficient(WindowType::Hann, 64, 64);
    }

    #[test]
    #[should_panic(expected = "Window length must be positive")]
    fn test_coefficient_zero_length() {
        window_coefficient(WindowType::Hann, 0, 0);
    }

    #[test]
    fn test_all_window_types_produce_valid_output() {
        let windows = [
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::BlackmanHarris,
        ];

        for window in windows {
            for i in 0..128 {
                let coeff = window_coefficient(window, i, 128);
                assert!(coeff >= 0.0, "Window coefficient should be non-negative");
                assert!(coeff <= 1.01, "Window coefficient should be <= 1.0");
            }
        }
    }
}
