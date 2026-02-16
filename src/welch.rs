//! Welch's method for power spectral density estimation.
//!
//! Welch's method reduces variance in PSD estimates by averaging multiple
//! overlapping windowed periodograms. This is the standard technique for PSD
//! estimation in EEG/BCI research (equivalent to `scipy.signal.welch`,
//! MATLAB's `pwelch`, MNE-Python's `compute_psd`).
//!
//! # Algorithm
//!
//! 1. Divide the signal into overlapping segments of length N
//! 2. Apply a window function to each segment
//! 3. Compute the FFT and |X[k]|² for each segment
//! 4. Average the periodograms across segments
//! 5. Normalize to V²/Hz
//!
//! # Example
//!
//! ```
//! use zerostone::{WelchPsd, WindowType};
//!
//! // 256-point segments, Hann window, 50% overlap
//! let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
//!
//! // Estimate PSD from a long signal
//! let signal = [0.0f32; 1024];
//! let mut psd = [0.0f32; 129]; // N/2 + 1 bins
//! let num_segments = welch.estimate(&signal, 250.0, &mut psd);
//!
//! assert_eq!(num_segments, 7); // (1024 - 256) / 128 + 1
//! ```

use crate::fft::{Complex, Fft};
use crate::window::{window_coefficient, WindowType};

/// Welch's method PSD estimator.
///
/// Estimates the power spectral density by averaging overlapping windowed
/// periodograms. The segment size `N` must be a power of 2.
///
/// # Type Parameters
///
/// * `N` - Segment/FFT size (must be a power of 2)
///
/// # Example
///
/// ```
/// use zerostone::{WelchPsd, WindowType};
///
/// let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
///
/// // Generate a 10 Hz sine wave at 250 Hz sample rate
/// let mut signal = [0.0f32; 1024];
/// for (i, s) in signal.iter_mut().enumerate() {
///     let t = i as f32 / 250.0;
///     *s = libm::sinf(2.0 * core::f32::consts::PI * 10.0 * t);
/// }
///
/// let mut psd = [0.0f32; 129];
/// let segments = welch.estimate(&signal, 250.0, &mut psd);
/// assert!(segments > 1);
/// ```
pub struct WelchPsd<const N: usize> {
    fft: Fft<N>,
    window: WindowType,
    overlap: usize,
    window_s2: f32,
}

impl<const N: usize> WelchPsd<N> {
    /// Creates a new Welch PSD estimator.
    ///
    /// # Arguments
    ///
    /// * `window` - Window function to apply to each segment
    /// * `overlap_frac` - Overlap fraction in \[0.0, 1.0). 0.5 = 50% overlap (recommended).
    ///
    /// # Panics
    ///
    /// Panics if `N` is not a power of 2, or if `overlap_frac` is outside \[0.0, 1.0).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{WelchPsd, WindowType};
    ///
    /// let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
    /// ```
    pub fn new(window: WindowType, overlap_frac: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&overlap_frac),
            "overlap_frac must be in [0.0, 1.0)"
        );

        let overlap = (N as f32 * overlap_frac) as usize;

        let window_s2: f32 = (0..N)
            .map(|i| {
                let w = window_coefficient(window, i, N);
                w * w
            })
            .sum();

        Self {
            fft: Fft::new(),
            window,
            overlap,
            window_s2,
        }
    }

    /// Estimates the PSD of a signal using Welch's method.
    ///
    /// Writes the one-sided PSD (N/2 + 1 bins) into `output`, normalized
    /// to V²/Hz.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal slice (must have length >= N)
    /// * `sample_rate` - Sample rate in Hz
    /// * `output` - Output buffer for PSD (must have length >= N/2 + 1)
    ///
    /// # Returns
    ///
    /// Number of segments averaged.
    ///
    /// # Panics
    ///
    /// Panics if `signal.len() < N` or `output.len() < N/2 + 1`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{WelchPsd, WindowType};
    ///
    /// let welch = WelchPsd::<64>::new(WindowType::Hann, 0.5);
    /// let signal = [1.0f32; 256];
    /// let mut psd = [0.0f32; 33];
    /// let segments = welch.estimate(&signal, 100.0, &mut psd);
    /// assert_eq!(segments, 7); // (256 - 64) / 32 + 1
    /// ```
    pub fn estimate(&self, signal: &[f32], sample_rate: f32, output: &mut [f32]) -> usize {
        assert!(
            signal.len() >= N,
            "Signal length {} must be >= segment size {}",
            signal.len(),
            N
        );
        assert!(
            output.len() > N / 2,
            "Output buffer length {} must be >= {}",
            output.len(),
            N / 2 + 1
        );

        let num_segments = self.num_segments(signal.len());
        let hop = N - self.overlap;

        // Zero output
        for val in output[..N / 2 + 1].iter_mut() {
            *val = 0.0;
        }

        // Accumulate periodograms
        for seg in 0..num_segments {
            let start = seg * hop;
            let segment = &signal[start..start + N];

            // Apply window and convert to complex
            let mut data: [Complex; N] = core::array::from_fn(|i| {
                let w = window_coefficient(self.window, i, N);
                Complex::from_real(segment[i] * w)
            });

            // FFT
            self.fft.forward(&mut data);

            // Accumulate |X[k]|²
            output[0] += data[0].magnitude_squared();
            for k in 1..N / 2 {
                output[k] += data[k].magnitude_squared();
            }
            output[N / 2] += data[N / 2].magnitude_squared();
        }

        // Average and normalize to V²/Hz
        let k = num_segments as f32;
        let norm = 2.0 / (sample_rate * self.window_s2);
        let dc_nyquist_norm = 1.0 / (sample_rate * self.window_s2);

        output[0] = output[0] / k * dc_nyquist_norm;
        for val in output[1..N / 2].iter_mut() {
            *val = *val / k * norm;
        }
        output[N / 2] = output[N / 2] / k * dc_nyquist_norm;

        num_segments
    }

    /// Computes the number of segments that fit in a signal of the given length.
    ///
    /// # Arguments
    ///
    /// * `signal_len` - Length of the input signal
    ///
    /// # Returns
    ///
    /// Number of complete segments. Returns 0 if `signal_len < N`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{WelchPsd, WindowType};
    ///
    /// let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
    /// assert_eq!(welch.num_segments(1024), 7);
    /// assert_eq!(welch.num_segments(256), 1);
    /// assert_eq!(welch.num_segments(100), 0);
    /// ```
    pub fn num_segments(&self, signal_len: usize) -> usize {
        if signal_len < N {
            return 0;
        }
        let hop = N - self.overlap;
        (signal_len - N) / hop + 1
    }

    /// Fills `output` with frequency bin centers in Hz.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `output` - Output buffer (must have length >= N/2 + 1)
    ///
    /// # Panics
    ///
    /// Panics if `output.len() < N/2 + 1`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{WelchPsd, WindowType};
    ///
    /// let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
    /// let mut freqs = [0.0f32; 129];
    /// welch.frequencies(250.0, &mut freqs);
    /// assert!((freqs[0] - 0.0).abs() < 1e-6);
    /// assert!((freqs[128] - 125.0).abs() < 1e-6);
    /// ```
    pub fn frequencies(&self, sample_rate: f32, output: &mut [f32]) {
        assert!(
            output.len() > N / 2,
            "Output buffer length {} must be >= {}",
            output.len(),
            N / 2 + 1
        );

        let freq_resolution = sample_rate / N as f32;
        for (k, val) in output[..N / 2 + 1].iter_mut().enumerate() {
            *val = k as f32 * freq_resolution;
        }
    }

    /// Returns the window type.
    pub fn window(&self) -> WindowType {
        self.window
    }

    /// Returns the overlap in samples.
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Returns the segment size (N).
    pub fn segment_size(&self) -> usize {
        N
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_sinusoid_peak_at_correct_frequency() {
        // 10 Hz sine at 256 Hz sample rate, 256-point segments
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut psd = [0.0f32; 129];
        let segments = welch.estimate(&signal, sample_rate, &mut psd);
        assert!(segments > 1);

        // Find peak bin (skip DC)
        let peak_bin = psd[1..129]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap();

        // Frequency resolution = 256/256 = 1 Hz per bin, so bin 10 = 10 Hz
        assert_eq!(peak_bin, 10, "Peak should be at bin 10 (10 Hz)");
    }

    #[test]
    fn test_sinusoid_power_level() {
        // A pure sinusoid with amplitude A has power A²/2
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let sample_rate = 256.0;
        let amplitude = 2.0;

        let mut signal = [0.0f32; 2048];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = amplitude * libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut psd = [0.0f32; 129];
        welch.estimate(&signal, sample_rate, &mut psd);

        // Integrate PSD to get total power
        let freq_res = sample_rate / 256.0;
        let total_power: f32 = psd.iter().map(|&v| v * freq_res).sum();

        // Expected: A²/2 = 4/2 = 2.0
        let expected = amplitude * amplitude / 2.0;
        let ratio = total_power / expected;
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "Total power ratio: {} (expected ~1.0)",
            ratio
        );
    }

    #[test]
    fn test_white_noise_approximately_flat() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let sample_rate = 256.0;

        // Deterministic pseudo-white noise
        let mut signal = [0.0f32; 4096];
        let mut state: u32 = 12345;
        for s in signal.iter_mut() {
            // Simple LCG for deterministic "noise"
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *s = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut psd = [0.0f32; 129];
        welch.estimate(&signal, sample_rate, &mut psd);

        // Check flatness: ratio of max to min in bins 5..120 (avoid DC/edge)
        let mid_psd = &psd[5..120];
        let mean: f32 = mid_psd.iter().sum::<f32>() / mid_psd.len() as f32;
        let max = mid_psd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = mid_psd.iter().cloned().fold(f32::INFINITY, f32::min);

        // For averaged white noise PSD, max/min should be reasonable
        let flatness = max / min;
        assert!(
            flatness < 5.0,
            "White noise PSD should be roughly flat (max/min = {})",
            flatness
        );
        assert!(mean > 0.0, "Mean PSD should be positive");
    }

    #[test]
    fn test_parseval_theorem() {
        // Total PSD power should match time-domain variance (rectangular window)
        let welch = WelchPsd::<256>::new(WindowType::Rectangular, 0.5);
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = libm::sinf(2.0 * PI * 20.0 * t) + 0.5 * libm::cosf(2.0 * PI * 50.0 * t);
        }

        let time_power: f32 = signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32;

        let mut psd = [0.0f32; 129];
        welch.estimate(&signal, sample_rate, &mut psd);

        let freq_res = sample_rate / 256.0;
        let freq_power: f32 = psd.iter().map(|&v| v * freq_res).sum();

        let ratio = freq_power / time_power;
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "Parseval's theorem: ratio = {} (expected ~1.0)",
            ratio
        );
    }

    #[test]
    fn test_single_segment_equivalent_to_windowed_periodogram() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let sample_rate = 256.0;

        // Signal exactly N samples -> 1 segment
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = libm::sinf(2.0 * PI * 15.0 * t);
        }

        let mut psd = [0.0f32; 129];
        let segments = welch.estimate(&signal, sample_rate, &mut psd);
        assert_eq!(segments, 1);

        // Compute the same thing manually
        let window_s2: f32 = (0..256)
            .map(|i| {
                let w = window_coefficient(WindowType::Hann, i, 256);
                w * w
            })
            .sum();

        let mut data: [Complex; 256] = core::array::from_fn(|i| {
            let w = window_coefficient(WindowType::Hann, i, 256);
            Complex::from_real(signal[i] * w)
        });

        let fft = Fft::<256>::new();
        fft.forward(&mut data);

        let dc_norm = 1.0 / (sample_rate * window_s2);
        let norm = 2.0 / (sample_rate * window_s2);

        let expected_dc = data[0].magnitude_squared() * dc_norm;
        let expected_10 = data[10].magnitude_squared() * norm;
        let expected_nyquist = data[128].magnitude_squared() * dc_norm;

        assert!(
            (psd[0] - expected_dc).abs() < 1e-6,
            "DC mismatch: {} vs {}",
            psd[0],
            expected_dc
        );
        assert!(
            (psd[10] - expected_10).abs() / (expected_10 + 1e-10) < 1e-4,
            "Bin 10 mismatch: {} vs {}",
            psd[10],
            expected_10
        );
        assert!(
            (psd[128] - expected_nyquist).abs() < 1e-6,
            "Nyquist mismatch: {} vs {}",
            psd[128],
            expected_nyquist
        );
    }

    #[test]
    fn test_more_segments_lower_variance() {
        let sample_rate = 256.0;

        // Generate noisy sinusoid
        let mut signal_long = [0.0f32; 4096];
        let mut state: u32 = 42;
        for (i, s) in signal_long.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *s = libm::sinf(2.0 * PI * 10.0 * t) + noise * 0.5;
        }

        // Short signal -> few segments
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let mut psd_short = [0.0f32; 129];
        let seg_short = welch.estimate(&signal_long[..512], sample_rate, &mut psd_short);

        // Long signal -> many segments
        let mut psd_long = [0.0f32; 129];
        let seg_long = welch.estimate(&signal_long, sample_rate, &mut psd_long);

        assert!(seg_long > seg_short);

        // Both should peak at bin 10 (10 Hz)
        let peak_short = psd_short[1..129]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap();
        let peak_long = psd_long[1..129]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap();

        assert_eq!(peak_short, 10);
        assert_eq!(peak_long, 10);
    }

    #[test]
    fn test_different_window_types() {
        let sample_rate = 256.0;
        let mut signal = [0.0f32; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        for window in [
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::BlackmanHarris,
        ] {
            let welch = WelchPsd::<256>::new(window, 0.5);
            let mut psd = [0.0f32; 129];
            let segments = welch.estimate(&signal, sample_rate, &mut psd);
            assert!(segments > 0, "Window {:?} should produce segments", window);

            // All should find peak at 10 Hz
            let peak = psd[1..129]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i + 1)
                .unwrap();
            assert_eq!(peak, 10, "Window {:?} should find 10 Hz peak", window);
        }
    }

    #[test]
    #[should_panic(expected = "overlap_frac must be in [0.0, 1.0)")]
    fn test_overlap_validation_high() {
        let _ = WelchPsd::<256>::new(WindowType::Hann, 1.0);
    }

    #[test]
    #[should_panic(expected = "overlap_frac must be in [0.0, 1.0)")]
    fn test_overlap_validation_negative() {
        let _ = WelchPsd::<256>::new(WindowType::Hann, -0.1);
    }

    #[test]
    #[should_panic(expected = "Signal length")]
    fn test_signal_too_short() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let signal = [0.0f32; 100];
        let mut psd = [0.0f32; 129];
        welch.estimate(&signal, 256.0, &mut psd);
    }

    #[test]
    fn test_frequency_bins() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        let mut freqs = [0.0f32; 129];
        welch.frequencies(256.0, &mut freqs);

        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - 1.0).abs() < 1e-6);
        assert!((freqs[128] - 128.0).abs() < 1e-6);
    }

    #[test]
    fn test_num_segments() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.5);
        // hop = 256 - 128 = 128
        assert_eq!(welch.num_segments(1024), 7); // (1024-256)/128 + 1
        assert_eq!(welch.num_segments(256), 1);
        assert_eq!(welch.num_segments(100), 0);
        assert_eq!(welch.num_segments(384), 2); // (384-256)/128 + 1

        // Zero overlap
        let welch0 = WelchPsd::<256>::new(WindowType::Hann, 0.0);
        assert_eq!(welch0.num_segments(1024), 4); // (1024-256)/256 + 1
    }

    #[test]
    fn test_zero_overlap() {
        let welch = WelchPsd::<256>::new(WindowType::Hann, 0.0);
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *s = libm::sinf(2.0 * PI * 10.0 * t);
        }

        let mut psd = [0.0f32; 129];
        let segments = welch.estimate(&signal, sample_rate, &mut psd);
        assert_eq!(segments, 4);

        let peak = psd[1..129]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)
            .unwrap();
        assert_eq!(peak, 10);
    }

    #[test]
    fn test_accessors() {
        let welch = WelchPsd::<256>::new(WindowType::Hamming, 0.5);
        assert_eq!(welch.window(), WindowType::Hamming);
        assert_eq!(welch.overlap(), 128);
        assert_eq!(welch.segment_size(), 256);
    }
}
