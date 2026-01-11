//! Wavelet transforms for time-frequency analysis.
//!
//! Provides Continuous Wavelet Transform (CWT) using Morlet wavelets,
//! enabling multi-resolution analysis of neural signals. Essential for
//! event-related synchronization/desynchronization detection in BCIs.
//!
//! # Features
//!
//! - Morlet wavelet with configurable central frequency (omega0)
//! - Logarithmically-spaced scales for uniform frequency resolution
//! - Zero-allocation computation with on-the-fly wavelet coefficients
//! - Multi-channel batch processing
//!
//! # Example
//!
//! ```
//! use zerostone::{Cwt, Complex};
//!
//! // Create CWT for motor imagery analysis (8-30 Hz)
//! let cwt = Cwt::<256, 8>::new(250.0, 8.0, 30.0);
//!
//! // Process EEG signal
//! let signal = [0.0f32; 256];
//! let mut power = [[0.0f32; 256]; 8];
//! cwt.power(&signal, &mut power);
//!
//! // Access power at each time-frequency point
//! let alpha_power_at_center = power[4][128];
//! ```

use core::f32::consts::PI;

use crate::Complex;

/// Threshold for wavelet truncation (exp(-8) ≈ 0.00034).
///
/// Wavelet coefficients below this amplitude are considered negligible.
const WAVELET_THRESHOLD: f32 = 0.000335;

/// Default Morlet wavelet central frequency.
///
/// omega0 = 6.0 provides a good balance between time and frequency resolution
/// for neural signal analysis.
const DEFAULT_OMEGA0: f32 = 6.0;

/// Wavelet type for CWT analysis.
///
/// Currently supports Morlet wavelet, with potential for future expansion
/// to other wavelet types (Mexican hat, Paul, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Morlet wavelet: complex sinusoid modulated by Gaussian.
    ///
    /// The wavelet is defined as:
    /// `w(t) = exp(i * omega0 * t) * exp(-t² / 2)`
    ///
    /// The `omega0` parameter controls the trade-off between time and frequency
    /// resolution:
    /// - omega0 = 5.0: Better time resolution
    /// - omega0 = 6.0 (default): Good balance for neural signals
    /// - omega0 = 8.0: Better frequency resolution
    Morlet {
        /// Central frequency parameter
        omega0: f32,
    },
}

impl Default for WaveletType {
    fn default() -> Self {
        WaveletType::Morlet {
            omega0: DEFAULT_OMEGA0,
        }
    }
}

/// Computes a single Morlet wavelet coefficient.
///
/// The Morlet wavelet at time offset `t` and scale `s` is:
/// `w(t, s) = (1/√s) * exp(i * omega0 * t/s) * exp(-t² / (2s²))`
///
/// # Arguments
///
/// * `t` - Time offset from center in samples
/// * `scale` - Wavelet scale (controls frequency)
/// * `omega0` - Central frequency parameter
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Complex wavelet coefficient at the given time and scale.
///
/// # Example
///
/// ```
/// use zerostone::wavelet::morlet_coefficient;
///
/// // Compute wavelet coefficient at center (t=0)
/// let w = morlet_coefficient(0.0, 1.0, 6.0, 250.0);
/// assert!(w.re > 0.0); // Peak at center
/// ```
#[inline]
pub fn morlet_coefficient(t: f32, scale: f32, omega0: f32, sample_rate: f32) -> Complex {
    // Convert time offset to scaled units
    let dt = 1.0 / sample_rate;
    let t_scaled = t * dt / scale;

    // Gaussian envelope: exp(-t²/2)
    let gaussian = libm::expf(-0.5 * t_scaled * t_scaled);

    // Complex exponential: exp(i * omega0 * t)
    let phase = omega0 * t_scaled;
    let cos_phase = libm::cosf(phase);
    let sin_phase = libm::sinf(phase);

    // Normalization: 1/√scale for energy preservation
    let norm = 1.0 / libm::sqrtf(scale);

    Complex::new(norm * gaussian * cos_phase, norm * gaussian * sin_phase)
}

/// Computes the effective half-width of the wavelet at a given scale.
///
/// The wavelet is truncated where its amplitude falls below a threshold,
/// as coefficients beyond this point contribute negligibly to the transform.
///
/// # Arguments
///
/// * `scale` - Wavelet scale
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Number of samples on each side of the wavelet center.
///
/// # Example
///
/// ```
/// use zerostone::wavelet::wavelet_half_width;
///
/// // Larger scales have wider effective wavelets
/// let w1 = wavelet_half_width(1.0, 250.0);
/// let w2 = wavelet_half_width(2.0, 250.0);
/// assert!(w2 > w1);
/// ```
#[inline]
pub fn wavelet_half_width(scale: f32, sample_rate: f32) -> usize {
    // Solve: exp(-0.5 * t²) = threshold
    // t = sqrt(-2 * ln(threshold))
    let t_limit = libm::sqrtf(-2.0 * libm::logf(WAVELET_THRESHOLD));

    // Convert to samples: t_limit is in scaled units, convert back
    let samples = (t_limit * scale * sample_rate) as usize;
    samples.max(1)
}

/// Generates logarithmically-spaced scales covering a frequency range.
///
/// For the Morlet wavelet, the relationship between scale and frequency is:
/// `frequency = omega0 / (2π * scale)`
///
/// Log-spacing provides uniform resolution on a logarithmic frequency scale,
/// which matches human perception and is appropriate for neural oscillations.
fn generate_scales<const S: usize>(min_freq: f32, max_freq: f32, omega0: f32) -> [f32; S] {
    // Convert frequencies to scales (inverse relationship)
    // Higher frequency → smaller scale
    let max_scale = omega0 / (2.0 * PI * min_freq);
    let min_scale = omega0 / (2.0 * PI * max_freq);

    let log_min = libm::logf(min_scale);
    let log_max = libm::logf(max_scale);

    let mut scales = [0.0f32; S];

    if S == 1 {
        scales[0] = libm::expf((log_min + log_max) / 2.0);
    } else {
        for (i, scale) in scales.iter_mut().enumerate() {
            let t = i as f32 / (S - 1) as f32;
            let log_scale = log_min + t * (log_max - log_min);
            *scale = libm::expf(log_scale);
        }
    }

    scales
}

/// Continuous Wavelet Transform using Morlet wavelet.
///
/// Provides time-frequency analysis of neural signals with multi-resolution
/// capability: better time resolution at high frequencies, better frequency
/// resolution at low frequencies.
///
/// # Type Parameters
///
/// * `N` - Signal length (number of time samples)
/// * `S` - Number of scales (frequency resolutions)
///
/// # Example
///
/// ```
/// use zerostone::{Cwt, Complex};
///
/// // Create CWT processor for 256-sample signals with 8 scales
/// let cwt = Cwt::<256, 8>::new(250.0, 1.0, 30.0);
///
/// // Process a signal
/// let signal = [0.0f32; 256];
/// let mut power = [[0.0f32; 256]; 8];
/// cwt.power(&signal, &mut power);
/// ```
#[derive(Debug, Clone)]
pub struct Cwt<const N: usize, const S: usize> {
    /// Sample rate in Hz
    sample_rate: f32,
    /// Scales array (computed from frequency range)
    scales: [f32; S],
    /// Central frequency of Morlet wavelet
    omega0: f32,
}

impl<const N: usize, const S: usize> Cwt<N, S> {
    /// Creates a new CWT processor with logarithmically-spaced scales.
    ///
    /// Uses the default Morlet wavelet (omega0 = 6.0).
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `min_freq` - Minimum frequency of interest in Hz
    /// * `max_freq` - Maximum frequency of interest in Hz
    ///
    /// # Panics
    ///
    /// Panics if `min_freq >= max_freq` or if frequencies are non-positive.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Cwt;
    ///
    /// // Create CWT for 1-50 Hz range at 250 Hz sample rate
    /// let cwt = Cwt::<256, 16>::new(250.0, 1.0, 50.0);
    /// ```
    pub fn new(sample_rate: f32, min_freq: f32, max_freq: f32) -> Self {
        assert!(min_freq > 0.0, "min_freq must be positive");
        assert!(
            max_freq > min_freq,
            "max_freq must be greater than min_freq"
        );
        assert!(
            max_freq <= sample_rate / 2.0,
            "max_freq must not exceed Nyquist frequency"
        );

        let scales = generate_scales::<S>(min_freq, max_freq, DEFAULT_OMEGA0);

        Self {
            sample_rate,
            scales,
            omega0: DEFAULT_OMEGA0,
        }
    }

    /// Creates CWT with custom wavelet parameters.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `min_freq` - Minimum frequency in Hz
    /// * `max_freq` - Maximum frequency in Hz
    /// * `wavelet` - Wavelet type and parameters
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Cwt, WaveletType};
    ///
    /// // Use Morlet with higher omega0 for better frequency resolution
    /// let cwt = Cwt::<256, 8>::with_wavelet(
    ///     250.0, 8.0, 30.0,
    ///     WaveletType::Morlet { omega0: 8.0 }
    /// );
    /// ```
    pub fn with_wavelet(
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
        wavelet: WaveletType,
    ) -> Self {
        assert!(min_freq > 0.0, "min_freq must be positive");
        assert!(
            max_freq > min_freq,
            "max_freq must be greater than min_freq"
        );
        assert!(
            max_freq <= sample_rate / 2.0,
            "max_freq must not exceed Nyquist frequency"
        );

        let omega0 = match wavelet {
            WaveletType::Morlet { omega0 } => omega0,
        };

        let scales = generate_scales::<S>(min_freq, max_freq, omega0);

        Self {
            sample_rate,
            scales,
            omega0,
        }
    }

    /// Creates CWT with explicit scale values.
    ///
    /// Use this when you need precise control over the scales used.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `scales` - Pre-computed scale values
    /// * `omega0` - Morlet wavelet central frequency
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Cwt;
    ///
    /// // Custom scales for specific frequencies
    /// let scales = [0.5, 1.0, 2.0, 4.0];
    /// let cwt = Cwt::<256, 4>::from_scales(250.0, scales, 6.0);
    /// ```
    pub fn from_scales(sample_rate: f32, scales: [f32; S], omega0: f32) -> Self {
        Self {
            sample_rate,
            scales,
            omega0,
        }
    }

    /// Computes the CWT, returning complex coefficients.
    ///
    /// Output layout: `output[scale_idx][time_idx]`
    ///
    /// The complex coefficients contain both amplitude and phase information,
    /// useful for phase-based analysis like event-related synchronization.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal (real-valued)
    /// * `output` - Output buffer for complex coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Cwt, Complex};
    ///
    /// let cwt = Cwt::<128, 4>::new(250.0, 5.0, 30.0);
    /// let signal = [0.0f32; 128];
    /// let mut coeffs = [[Complex::new(0.0, 0.0); 128]; 4];
    /// cwt.transform(&signal, &mut coeffs);
    /// ```
    pub fn transform(&self, signal: &[f32; N], output: &mut [[Complex; N]; S]) {
        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            self.convolve_scale(signal, scale, &mut output[scale_idx]);
        }
    }

    /// Computes the CWT power (magnitude squared).
    ///
    /// More efficient than computing full complex coefficients when
    /// only power is needed (common for band power analysis).
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `output` - Output buffer for power values
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Cwt;
    ///
    /// let cwt = Cwt::<256, 8>::new(250.0, 1.0, 50.0);
    /// let signal = [0.5f32; 256];
    /// let mut power = [[0.0f32; 256]; 8];
    /// cwt.power(&signal, &mut power);
    /// ```
    pub fn power(&self, signal: &[f32; N], output: &mut [[f32; N]; S]) {
        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            self.convolve_scale_power(signal, scale, &mut output[scale_idx]);
        }
    }

    /// Computes CWT magnitude (absolute value of coefficients).
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `output` - Output buffer for magnitude values
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Cwt;
    ///
    /// let cwt = Cwt::<256, 8>::new(250.0, 1.0, 50.0);
    /// let signal = [0.5f32; 256];
    /// let mut magnitude = [[0.0f32; 256]; 8];
    /// cwt.magnitude(&signal, &mut magnitude);
    /// ```
    pub fn magnitude(&self, signal: &[f32; N], output: &mut [[f32; N]; S]) {
        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            self.convolve_scale_magnitude(signal, scale, &mut output[scale_idx]);
        }
    }

    /// Computes CWT for a single scale.
    ///
    /// Useful for streaming applications where only specific frequencies
    /// are of interest.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `scale_idx` - Index of the scale to compute (0 to S-1)
    /// * `output` - Output buffer for complex coefficients at this scale
    ///
    /// # Panics
    ///
    /// Panics if `scale_idx >= S`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Cwt, Complex};
    ///
    /// let cwt = Cwt::<128, 8>::new(250.0, 5.0, 30.0);
    /// let signal = [0.0f32; 128];
    /// let mut coeffs = [Complex::new(0.0, 0.0); 128];
    ///
    /// // Compute only the 4th scale
    /// cwt.transform_scale(&signal, 3, &mut coeffs);
    /// ```
    pub fn transform_scale(&self, signal: &[f32; N], scale_idx: usize, output: &mut [Complex; N]) {
        assert!(scale_idx < S, "scale_idx out of bounds");
        let scale = self.scales[scale_idx];
        self.convolve_scale(signal, scale, output);
    }

    /// Converts scale to pseudo-frequency in Hz.
    ///
    /// For the Morlet wavelet: `f = omega0 / (2π * scale)`
    ///
    /// # Arguments
    ///
    /// * `scale` - Wavelet scale
    ///
    /// # Returns
    ///
    /// Corresponding pseudo-frequency in Hz.
    #[inline]
    pub fn scale_to_frequency(&self, scale: f32) -> f32 {
        self.omega0 / (2.0 * PI * scale)
    }

    /// Converts frequency to scale.
    ///
    /// For the Morlet wavelet: `scale = omega0 / (2π * f)`
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    ///
    /// # Returns
    ///
    /// Corresponding wavelet scale.
    #[inline]
    pub fn frequency_to_scale(&self, frequency: f32) -> f32 {
        self.omega0 / (2.0 * PI * frequency)
    }

    /// Returns the scales array.
    #[inline]
    pub fn scales(&self) -> &[f32; S] {
        &self.scales
    }

    /// Returns pseudo-frequencies for all scales.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Cwt;
    ///
    /// let cwt = Cwt::<256, 4>::new(250.0, 5.0, 20.0);
    /// let freqs = cwt.frequencies();
    /// // freqs[0] corresponds to highest frequency (smallest scale)
    /// // freqs[S-1] corresponds to lowest frequency (largest scale)
    /// ```
    pub fn frequencies(&self) -> [f32; S] {
        let mut freqs = [0.0f32; S];
        for (i, &scale) in self.scales.iter().enumerate() {
            freqs[i] = self.scale_to_frequency(scale);
        }
        freqs
    }

    /// Returns the sample rate.
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the omega0 parameter.
    #[inline]
    pub fn omega0(&self) -> f32 {
        self.omega0
    }

    /// Performs convolution of signal with wavelet at a single scale.
    ///
    /// Uses direct time-domain convolution with zero-padding boundary handling.
    fn convolve_scale(&self, signal: &[f32; N], scale: f32, output: &mut [Complex; N]) {
        let half_width = wavelet_half_width(scale, self.sample_rate);

        for (t, out) in output.iter_mut().enumerate() {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;

            // Compute convolution bounds with zero-padding
            let start = t.saturating_sub(half_width);
            let end = (t + half_width + 1).min(N);

            for (tau, &sample) in signal.iter().enumerate().take(end).skip(start) {
                let dt = tau as f32 - t as f32;
                let wavelet = morlet_coefficient(dt, scale, self.omega0, self.sample_rate);

                // Use conjugate for convolution (not correlation)
                sum_re += sample * wavelet.re;
                sum_im += sample * (-wavelet.im);
            }

            *out = Complex::new(sum_re, sum_im);
        }
    }

    /// Computes power directly without storing complex coefficients.
    ///
    /// Optimization for when only power spectrum is needed.
    fn convolve_scale_power(&self, signal: &[f32; N], scale: f32, output: &mut [f32; N]) {
        let half_width = wavelet_half_width(scale, self.sample_rate);

        for (t, out) in output.iter_mut().enumerate() {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;

            let start = t.saturating_sub(half_width);
            let end = (t + half_width + 1).min(N);

            for (tau, &sample) in signal.iter().enumerate().take(end).skip(start) {
                let dt = tau as f32 - t as f32;
                let wavelet = morlet_coefficient(dt, scale, self.omega0, self.sample_rate);

                sum_re += sample * wavelet.re;
                sum_im += sample * (-wavelet.im);
            }

            *out = sum_re * sum_re + sum_im * sum_im;
        }
    }

    /// Computes magnitude directly.
    fn convolve_scale_magnitude(&self, signal: &[f32; N], scale: f32, output: &mut [f32; N]) {
        let half_width = wavelet_half_width(scale, self.sample_rate);

        for (t, out) in output.iter_mut().enumerate() {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;

            let start = t.saturating_sub(half_width);
            let end = (t + half_width + 1).min(N);

            for (tau, &sample) in signal.iter().enumerate().take(end).skip(start) {
                let dt = tau as f32 - t as f32;
                let wavelet = morlet_coefficient(dt, scale, self.omega0, self.sample_rate);

                sum_re += sample * wavelet.re;
                sum_im += sample * (-wavelet.im);
            }

            *out = libm::sqrtf(sum_re * sum_re + sum_im * sum_im);
        }
    }
}

/// Multi-channel CWT processor for batch processing.
///
/// Processes multiple channels independently, useful for multi-electrode
/// EEG or other multi-channel neural recordings.
///
/// # Type Parameters
///
/// * `N` - Signal length
/// * `S` - Number of scales
/// * `C` - Number of channels
///
/// # Example
///
/// ```
/// use zerostone::MultiChannelCwt;
///
/// // 8-channel CWT processor
/// let cwt = MultiChannelCwt::<256, 8, 8>::new(250.0, 1.0, 50.0);
///
/// let signals = [[0.0f32; 256]; 8];
/// let mut power = [[[0.0f32; 256]; 8]; 8];
/// cwt.power(&signals, &mut power);
/// ```
#[derive(Debug, Clone)]
pub struct MultiChannelCwt<const N: usize, const S: usize, const C: usize> {
    cwt: Cwt<N, S>,
}

impl<const N: usize, const S: usize, const C: usize> MultiChannelCwt<N, S, C> {
    /// Creates a multi-channel CWT processor.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `min_freq` - Minimum frequency in Hz
    /// * `max_freq` - Maximum frequency in Hz
    pub fn new(sample_rate: f32, min_freq: f32, max_freq: f32) -> Self {
        Self {
            cwt: Cwt::new(sample_rate, min_freq, max_freq),
        }
    }

    /// Creates multi-channel CWT with custom wavelet parameters.
    pub fn with_wavelet(
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
        wavelet: WaveletType,
    ) -> Self {
        Self {
            cwt: Cwt::with_wavelet(sample_rate, min_freq, max_freq, wavelet),
        }
    }

    /// Computes CWT for all channels.
    ///
    /// # Arguments
    ///
    /// * `signals` - Input signals for all channels `[channel][time]`
    /// * `output` - Output buffer `[channel][scale][time]`
    pub fn transform(&self, signals: &[[f32; N]; C], output: &mut [[[Complex; N]; S]; C]) {
        for ch in 0..C {
            self.cwt.transform(&signals[ch], &mut output[ch]);
        }
    }

    /// Computes power for all channels.
    ///
    /// # Arguments
    ///
    /// * `signals` - Input signals `[channel][time]`
    /// * `output` - Output buffer `[channel][scale][time]`
    pub fn power(&self, signals: &[[f32; N]; C], output: &mut [[[f32; N]; S]; C]) {
        for ch in 0..C {
            self.cwt.power(&signals[ch], &mut output[ch]);
        }
    }

    /// Computes magnitude for all channels.
    pub fn magnitude(&self, signals: &[[f32; N]; C], output: &mut [[[f32; N]; S]; C]) {
        for ch in 0..C {
            self.cwt.magnitude(&signals[ch], &mut output[ch]);
        }
    }

    /// Returns the underlying CWT processor.
    pub fn cwt(&self) -> &Cwt<N, S> {
        &self.cwt
    }

    /// Returns the scales array.
    pub fn scales(&self) -> &[f32; S] {
        self.cwt.scales()
    }

    /// Returns pseudo-frequencies for all scales.
    pub fn frequencies(&self) -> [f32; S] {
        self.cwt.frequencies()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morlet_coefficient_at_center() {
        // At t=0, the wavelet should have maximum magnitude
        let w_center = morlet_coefficient(0.0, 1.0, 6.0, 250.0);
        let w_offset = morlet_coefficient(10.0, 1.0, 6.0, 250.0);

        assert!(w_center.magnitude() > w_offset.magnitude());
    }

    #[test]
    fn test_morlet_coefficient_symmetry() {
        // Magnitude should be symmetric around t=0
        let w_pos = morlet_coefficient(5.0, 1.0, 6.0, 250.0);
        let w_neg = morlet_coefficient(-5.0, 1.0, 6.0, 250.0);

        assert!((w_pos.magnitude() - w_neg.magnitude()).abs() < 1e-6);
    }

    #[test]
    fn test_morlet_coefficient_normalization() {
        // Normalization factor is 1/sqrt(scale)
        let w_s1 = morlet_coefficient(0.0, 1.0, 6.0, 250.0);
        let w_s4 = morlet_coefficient(0.0, 4.0, 6.0, 250.0);

        // w_s1 should be 2x the magnitude of w_s4 (sqrt(4) = 2)
        let ratio = w_s1.magnitude() / w_s4.magnitude();
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_wavelet_half_width_scaling() {
        // Larger scales should have larger effective widths
        let w1 = wavelet_half_width(1.0, 250.0);
        let w2 = wavelet_half_width(2.0, 250.0);
        let w4 = wavelet_half_width(4.0, 250.0);

        assert!(w2 > w1);
        assert!(w4 > w2);
    }

    #[test]
    fn test_scale_frequency_roundtrip() {
        let cwt = Cwt::<256, 8>::new(250.0, 1.0, 50.0);

        for &scale in cwt.scales().iter() {
            let freq = cwt.scale_to_frequency(scale);
            let scale_back = cwt.frequency_to_scale(freq);
            assert!((scale - scale_back).abs() < 1e-5);
        }
    }

    #[test]
    fn test_frequency_range() {
        let cwt = Cwt::<256, 8>::new(250.0, 5.0, 40.0);
        let freqs = cwt.frequencies();

        // First scale (smallest) should give highest frequency
        // Last scale (largest) should give lowest frequency
        assert!(freqs[0] > freqs[S - 1]);

        // All frequencies should be in the specified range (approximately)
        for &f in freqs.iter() {
            assert!((4.0..=45.0).contains(&f)); // Allow some tolerance
        }
    }

    #[test]
    fn test_cwt_pure_sine() {
        let cwt = Cwt::<256, 16>::new(250.0, 5.0, 50.0);

        // Generate 20 Hz sine wave
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / 250.0;
            *s = libm::sinf(2.0 * PI * 20.0 * t);
        }

        let mut power = [[0.0f32; 256]; 16];
        cwt.power(&signal, &mut power);

        // Find scale with maximum average power
        let frequencies = cwt.frequencies();
        let mut max_power = 0.0f32;
        let mut max_freq = 0.0f32;

        for (s, &freq) in frequencies.iter().enumerate() {
            let avg_power: f32 = power[s].iter().sum::<f32>() / 256.0;
            if avg_power > max_power {
                max_power = avg_power;
                max_freq = freq;
            }
        }

        // Maximum power should be near 20 Hz (allow ±5 Hz tolerance)
        assert!(
            (max_freq - 20.0).abs() < 5.0,
            "Expected peak near 20 Hz, got {} Hz",
            max_freq
        );
    }

    #[test]
    fn test_cwt_impulse_response() {
        let cwt = Cwt::<64, 4>::new(250.0, 5.0, 30.0);

        // Impulse signal
        let mut signal = [0.0f32; 64];
        signal[32] = 1.0;

        let mut power = [[0.0f32; 64]; 4];
        cwt.power(&signal, &mut power);

        // All scales should have some response to impulse
        for scale_power in &power {
            let total: f32 = scale_power.iter().sum();
            assert!(total > 0.0);
        }

        // Power should be centered around the impulse
        for scale_power in &power {
            let center_power = scale_power[32];
            let edge_power = scale_power[0];
            assert!(center_power > edge_power);
        }
    }

    #[test]
    fn test_transform_vs_power() {
        let cwt = Cwt::<64, 4>::new(250.0, 5.0, 30.0);

        let mut signal = [0.0f32; 64];
        for (i, s) in signal.iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI * 10.0 * i as f32 / 250.0);
        }

        let mut coeffs = [[Complex::new(0.0, 0.0); 64]; 4];
        let mut power = [[0.0f32; 64]; 4];

        cwt.transform(&signal, &mut coeffs);
        cwt.power(&signal, &mut power);

        // power[s][t] should equal coeffs[s][t].magnitude_squared()
        for s in 0..4 {
            for t in 0..64 {
                let computed_power = coeffs[s][t].magnitude_squared();
                assert!(
                    (power[s][t] - computed_power).abs() < 1e-6,
                    "Power mismatch at scale {}, time {}",
                    s,
                    t
                );
            }
        }
    }

    #[test]
    fn test_multi_channel_independence() {
        let cwt = MultiChannelCwt::<64, 4, 2>::new(250.0, 5.0, 30.0);

        // Different signals per channel
        let mut signals = [[0.0f32; 64]; 2];
        for (i, s) in signals[0].iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI * 10.0 * i as f32 / 250.0);
        }
        for (i, s) in signals[1].iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI * 25.0 * i as f32 / 250.0);
        }

        let mut output = [[[0.0f32; 64]; 4]; 2];
        cwt.power(&signals, &mut output);

        // Both channels should have non-zero power
        let ch0_power: f32 = output[0].iter().flat_map(|r| r.iter()).sum();
        let ch1_power: f32 = output[1].iter().flat_map(|r| r.iter()).sum();

        assert!(ch0_power > 0.0);
        assert!(ch1_power > 0.0);
    }

    #[test]
    fn test_transform_scale_matches_full() {
        let cwt = Cwt::<64, 4>::new(250.0, 5.0, 30.0);

        let signal = [0.5f32; 64];
        let mut full_output = [[Complex::new(0.0, 0.0); 64]; 4];
        let mut single_output = [Complex::new(0.0, 0.0); 64];

        cwt.transform(&signal, &mut full_output);
        cwt.transform_scale(&signal, 2, &mut single_output);

        // Single scale should match corresponding row of full transform
        for t in 0..64 {
            assert!((full_output[2][t].re - single_output[t].re).abs() < 1e-6);
            assert!((full_output[2][t].im - single_output[t].im).abs() < 1e-6);
        }
    }

    #[test]
    fn test_wavelet_type_default() {
        let wavelet = WaveletType::default();
        match wavelet {
            WaveletType::Morlet { omega0 } => {
                assert!((omega0 - 6.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_cwt_with_custom_wavelet() {
        let cwt =
            Cwt::<128, 4>::with_wavelet(250.0, 5.0, 30.0, WaveletType::Morlet { omega0: 8.0 });

        assert!((cwt.omega0() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_cwt_from_scales() {
        let scales = [0.5, 1.0, 2.0, 4.0];
        let cwt = Cwt::<128, 4>::from_scales(250.0, scales, 6.0);

        assert_eq!(cwt.scales(), &scales);
    }

    const S: usize = 8;
}
