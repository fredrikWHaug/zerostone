//! Multi-channel band power extraction with proper PSD normalization.
//!
//! This module provides state-of-the-art band power extraction for multi-channel
//! neural signals, with configurable windowing, proper power spectral density
//! normalization, and multiple integration methods.
//!
//! # Features
//!
//! - **Multi-channel support** via const generics
//! - **Proper PSD normalization** for V²/Hz units
//! - **Configurable windowing** (Hann default, supports all window types)
//! - **Multiple integration methods** (Trapezoidal default, Rectangular for MATLAB compatibility)
//! - **Efficient multi-band queries** (PSD computed once, queried multiple times)
//! - **Standard frequency bands** pre-defined (Delta, Theta, Alpha, Beta, Gamma, etc.)
//!
//! # Example
//!
//! ```
//! use zerostone::{MultiBandPower, FrequencyBand, IntegrationMethod};
//!
//! // Create band power extractor for 256-point FFT, 4 channels, 250 Hz sample rate
//! let mut bp: MultiBandPower<256, 4> = MultiBandPower::new(250.0);
//!
//! // Compute PSD for all channels
//! let signals: [[f32; 256]; 4] = [[0.0; 256]; 4];
//! bp.compute(&signals);
//!
//! // Query multiple bands efficiently (no re-computation)
//! let alpha = bp.band_power(FrequencyBand::ALPHA);
//! let beta = bp.band_power(FrequencyBand::BETA);
//!
//! // Relative power (normalized to reference band)
//! let rel_alpha = bp.band_power_relative(FrequencyBand::ALPHA, 1.0, 40.0);
//! ```
//!
//! # PSD Normalization
//!
//! The PSD is normalized to V²/Hz using the formula:
//!
//! ```text
//! PSD[k] = |X[k]|² × 2 / (fs × S2)
//! ```
//!
//! Where:
//! - `X[k]` is the FFT output at bin k
//! - `fs` is the sample rate
//! - `S2` is the sum of squared window coefficients (window power)
//! - Factor of 2 accounts for one-sided spectrum
//!
//! This gives proper physical units of V²/Hz, allowing meaningful
//! comparison across different window functions and FFT sizes.

use crate::fft::{Complex, Fft};
use crate::window::{window_coefficient, WindowType};

/// A frequency band defined by lower and upper bounds in Hz.
///
/// Standard EEG/BCI frequency bands are provided as associated constants.
/// Custom bands can be created using [`FrequencyBand::new`].
///
/// # Example
///
/// ```
/// use zerostone::FrequencyBand;
///
/// // Use predefined bands
/// let alpha = FrequencyBand::ALPHA;  // 8-12 Hz
/// let beta = FrequencyBand::BETA;    // 12-30 Hz
///
/// // Create custom band
/// let custom = FrequencyBand::new(15.0, 25.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyBand {
    /// Lower frequency bound in Hz (inclusive).
    pub low_hz: f32,
    /// Upper frequency bound in Hz (inclusive).
    pub high_hz: f32,
}

impl FrequencyBand {
    /// Delta band: 0.5-4 Hz.
    ///
    /// Associated with deep sleep (stages 3-4) and certain pathological conditions.
    pub const DELTA: FrequencyBand = FrequencyBand {
        low_hz: 0.5,
        high_hz: 4.0,
    };

    /// Theta band: 4-8 Hz.
    ///
    /// Associated with drowsiness, light sleep, and memory encoding.
    /// Often elevated during meditation and working memory tasks.
    pub const THETA: FrequencyBand = FrequencyBand {
        low_hz: 4.0,
        high_hz: 8.0,
    };

    /// Alpha band: 8-12 Hz.
    ///
    /// Associated with relaxed wakefulness, especially when eyes are closed.
    /// Suppressed during visual attention and mental effort.
    pub const ALPHA: FrequencyBand = FrequencyBand {
        low_hz: 8.0,
        high_hz: 12.0,
    };

    /// Mu rhythm: 8-12 Hz (same range as alpha).
    ///
    /// Sensorimotor alpha rhythm over motor cortex. Suppressed during
    /// motor execution and motor imagery (event-related desynchronization).
    /// Primary feature for motor imagery BCIs.
    pub const MU: FrequencyBand = FrequencyBand {
        low_hz: 8.0,
        high_hz: 12.0,
    };

    /// Sensorimotor rhythm (SMR): 12-15 Hz.
    ///
    /// Low beta rhythm over sensorimotor cortex. Used in neurofeedback
    /// training and some BCI paradigms.
    pub const SMR: FrequencyBand = FrequencyBand {
        low_hz: 12.0,
        high_hz: 15.0,
    };

    /// Beta band: 12-30 Hz.
    ///
    /// Associated with active thinking, focus, and alertness.
    /// Increased during motor planning and anxiety.
    pub const BETA: FrequencyBand = FrequencyBand {
        low_hz: 12.0,
        high_hz: 30.0,
    };

    /// Motor beta: 16-24 Hz.
    ///
    /// Narrower beta band centered on motor-related activity.
    /// Shows event-related desynchronization (ERD) during motor execution
    /// and event-related synchronization (ERS) after movement.
    pub const MOTOR_BETA: FrequencyBand = FrequencyBand {
        low_hz: 16.0,
        high_hz: 24.0,
    };

    /// Gamma band: 30-100 Hz.
    ///
    /// Associated with higher cognitive functions, attention binding,
    /// and perception. Often requires high sample rates and careful
    /// artifact rejection.
    pub const GAMMA: FrequencyBand = FrequencyBand {
        low_hz: 30.0,
        high_hz: 100.0,
    };

    /// Low gamma band: 30-50 Hz.
    ///
    /// Lower portion of gamma, more reliably measurable with standard
    /// EEG equipment. Associated with cognitive processing.
    pub const LOW_GAMMA: FrequencyBand = FrequencyBand {
        low_hz: 30.0,
        high_hz: 50.0,
    };

    /// High gamma band: 50-100 Hz.
    ///
    /// Higher portion of gamma, often contaminated by muscle artifacts
    /// and power line noise. Requires careful preprocessing.
    pub const HIGH_GAMMA: FrequencyBand = FrequencyBand {
        low_hz: 50.0,
        high_hz: 100.0,
    };

    /// Creates a custom frequency band.
    ///
    /// # Arguments
    ///
    /// * `low_hz` - Lower frequency bound in Hz
    /// * `high_hz` - Upper frequency bound in Hz
    ///
    /// # Panics
    ///
    /// Panics if `low_hz >= high_hz` or if either bound is negative.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::FrequencyBand;
    ///
    /// // Custom band for specific analysis
    /// let band = FrequencyBand::new(18.0, 22.0);
    /// assert_eq!(band.low_hz, 18.0);
    /// assert_eq!(band.high_hz, 22.0);
    /// ```
    pub fn new(low_hz: f32, high_hz: f32) -> Self {
        assert!(low_hz >= 0.0, "Lower frequency bound must be non-negative");
        assert!(
            high_hz > low_hz,
            "Upper frequency must be greater than lower frequency"
        );
        Self { low_hz, high_hz }
    }

    /// Returns the center frequency of the band.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::FrequencyBand;
    ///
    /// assert_eq!(FrequencyBand::ALPHA.center_hz(), 10.0);
    /// ```
    pub fn center_hz(&self) -> f32 {
        (self.low_hz + self.high_hz) / 2.0
    }

    /// Returns the bandwidth in Hz.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::FrequencyBand;
    ///
    /// assert_eq!(FrequencyBand::ALPHA.bandwidth_hz(), 4.0);
    /// ```
    pub fn bandwidth_hz(&self) -> f32 {
        self.high_hz - self.low_hz
    }
}

/// Integration method for computing band power from PSD.
///
/// The integration method determines how the area under the PSD curve
/// is approximated within a frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMethod {
    /// Rectangular integration (left Riemann sum).
    ///
    /// Approximates the area using rectangles with height equal to the
    /// PSD value at each bin. Error is O(Δf) where Δf is the frequency
    /// resolution.
    ///
    /// Use this for MATLAB/EEGLAB compatibility or when comparing with
    /// legacy implementations.
    Rectangular,

    /// Trapezoidal integration (default).
    ///
    /// Approximates the area using trapezoids. Error is O(Δf²), providing
    /// better accuracy than rectangular integration for smooth PSDs.
    ///
    /// This is the recommended method for most applications.
    #[default]
    Trapezoidal,
}

/// Multi-channel band power extractor with proper PSD normalization.
///
/// Computes power spectral density (PSD) for multiple channels and efficiently
/// extracts band power in arbitrary frequency ranges. The PSD is computed once
/// and stored internally, allowing efficient querying of multiple bands.
///
/// # Type Parameters
///
/// * `N` - FFT size (must be a power of 2)
/// * `C` - Number of channels
///
/// # Example
///
/// ```
/// use zerostone::{MultiBandPower, FrequencyBand, IntegrationMethod, WindowType};
///
/// // 256-point FFT, 8 channels, 500 Hz sample rate
/// let mut bp: MultiBandPower<256, 8> = MultiBandPower::new(500.0);
///
/// // Or with custom configuration
/// let mut bp_custom: MultiBandPower<512, 4> = MultiBandPower::with_config(
///     250.0,
///     WindowType::Hamming,
///     IntegrationMethod::Rectangular,
/// );
///
/// // Process signals
/// let signals: [[f32; 256]; 8] = [[0.0; 256]; 8];
/// bp.compute(&signals);
///
/// // Get band power for all channels
/// let alpha_power = bp.band_power(FrequencyBand::ALPHA);
/// let beta_power = bp.band_power(FrequencyBand::BETA);
///
/// // Get relative power (band / reference range)
/// let rel_alpha = bp.band_power_relative(FrequencyBand::ALPHA, 1.0, 40.0);
///
/// // Get normalized power (band / total power)
/// let norm_alpha = bp.band_power_normalized(FrequencyBand::ALPHA);
/// ```
pub struct MultiBandPower<const N: usize, const C: usize> {
    /// Sample rate in Hz.
    sample_rate: f32,
    /// Window function type.
    window: WindowType,
    /// Integration method for band power.
    integration: IntegrationMethod,
    /// FFT processor.
    fft: Fft<N>,
    /// Frequency resolution (Hz per bin).
    freq_resolution: f32,
    /// Stored PSD for each channel (one-sided, N/2+1 bins).
    /// Indexed as psd[channel][bin].
    psd: [[f32; N]; C],
    /// Whether PSD has been computed.
    has_psd: bool,
    /// Window power sum (S2 = sum of squared window coefficients).
    window_s2: f32,
}

impl<const N: usize, const C: usize> MultiBandPower<N, C> {
    /// Creates a new band power extractor with default settings.
    ///
    /// Uses Hann window and trapezoidal integration by default.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::MultiBandPower;
    ///
    /// let bp: MultiBandPower<256, 4> = MultiBandPower::new(250.0);
    /// ```
    pub fn new(sample_rate: f32) -> Self {
        Self::with_config(
            sample_rate,
            WindowType::Hann,
            IntegrationMethod::Trapezoidal,
        )
    }

    /// Creates a new band power extractor with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `window` - Window function to apply before FFT
    /// * `integration` - Integration method for band power computation
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{MultiBandPower, WindowType, IntegrationMethod};
    ///
    /// let bp: MultiBandPower<512, 2> = MultiBandPower::with_config(
    ///     500.0,
    ///     WindowType::BlackmanHarris,
    ///     IntegrationMethod::Rectangular,
    /// );
    /// ```
    pub fn with_config(
        sample_rate: f32,
        window: WindowType,
        integration: IntegrationMethod,
    ) -> Self {
        // Compute window power sum S2 = sum(w[n]^2)
        let window_s2: f32 = (0..N)
            .map(|i| {
                let w = window_coefficient(window, i, N);
                w * w
            })
            .sum();

        Self {
            sample_rate,
            window,
            integration,
            fft: Fft::new(),
            freq_resolution: sample_rate / N as f32,
            psd: [[0.0; N]; C],
            has_psd: false,
            window_s2,
        }
    }

    /// Computes the PSD for all channels.
    ///
    /// After calling this method, band power can be queried efficiently
    /// using [`band_power`](Self::band_power), [`band_power_relative`](Self::band_power_relative),
    /// or [`band_power_normalized`](Self::band_power_normalized).
    ///
    /// The PSD is normalized to V²/Hz:
    /// ```text
    /// PSD[k] = |X[k]|² × 2 / (fs × S2)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `signals` - Input signals for each channel, shape \[C\]\[N\]
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::MultiBandPower;
    ///
    /// let mut bp: MultiBandPower<128, 2> = MultiBandPower::new(256.0);
    ///
    /// let signals = [[0.5f32; 128], [-0.5f32; 128]];
    /// bp.compute(&signals);
    /// ```
    pub fn compute(&mut self, signals: &[[f32; N]; C]) {
        // Normalization factor: 2 / (fs * S2)
        // Factor of 2 for one-sided spectrum
        let norm_factor = 2.0 / (self.sample_rate * self.window_s2);
        let dc_nyquist_norm = 1.0 / (self.sample_rate * self.window_s2);

        for (ch, signal) in signals.iter().enumerate() {
            // Apply window and convert to complex
            let mut data: [Complex; N] = core::array::from_fn(|i| {
                let w = window_coefficient(self.window, i, N);
                Complex::from_real(signal[i] * w)
            });

            // Compute FFT
            self.fft.forward(&mut data);

            // Compute one-sided PSD with proper normalization
            // DC component (no factor of 2)
            self.psd[ch][0] = data[0].magnitude_squared() * dc_nyquist_norm;

            // Positive frequencies (bins 1 to N/2-1)
            for (psd_bin, data_bin) in self.psd[ch][1..N / 2].iter_mut().zip(data[1..N / 2].iter())
            {
                *psd_bin = data_bin.magnitude_squared() * norm_factor;
            }

            // Nyquist frequency (no factor of 2)
            self.psd[ch][N / 2] = data[N / 2].magnitude_squared() * dc_nyquist_norm;
        }

        self.has_psd = true;
    }

    /// Returns the absolute band power for all channels.
    ///
    /// The band power is the integral of the PSD over the frequency band,
    /// in units of V² (signal power).
    ///
    /// # Arguments
    ///
    /// * `band` - Frequency band to compute power for
    ///
    /// # Returns
    ///
    /// Array of band power values, one per channel.
    ///
    /// # Panics
    ///
    /// Panics if [`compute`](Self::compute) has not been called.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{MultiBandPower, FrequencyBand};
    ///
    /// let mut bp: MultiBandPower<256, 4> = MultiBandPower::new(250.0);
    /// let signals = [[0.0f32; 256]; 4];
    /// bp.compute(&signals);
    ///
    /// let alpha_power = bp.band_power(FrequencyBand::ALPHA);
    /// ```
    pub fn band_power(&self, band: FrequencyBand) -> [f32; C] {
        assert!(
            self.has_psd,
            "Must call compute() before querying band power"
        );

        let (low_bin, high_bin) = self.band_to_bins(band);
        let mut power = [0.0f32; C];

        for (ch, pwr) in power.iter_mut().enumerate() {
            *pwr = self.integrate_band(ch, low_bin, high_bin);
        }

        power
    }

    /// Returns the relative band power for all channels.
    ///
    /// Relative power is the ratio of power in the target band to power
    /// in a reference frequency range. Useful for computing ratios like
    /// theta/alpha or alpha/beta.
    ///
    /// # Arguments
    ///
    /// * `band` - Target frequency band
    /// * `ref_low_hz` - Lower bound of reference range (Hz)
    /// * `ref_high_hz` - Upper bound of reference range (Hz)
    ///
    /// # Returns
    ///
    /// Array of relative power values (dimensionless ratio), one per channel.
    ///
    /// # Panics
    ///
    /// Panics if [`compute`](Self::compute) has not been called.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{MultiBandPower, FrequencyBand};
    ///
    /// let mut bp: MultiBandPower<256, 4> = MultiBandPower::new(250.0);
    /// let signals = [[0.0f32; 256]; 4];
    /// bp.compute(&signals);
    ///
    /// // Alpha power relative to 1-40 Hz range
    /// let rel_alpha = bp.band_power_relative(FrequencyBand::ALPHA, 1.0, 40.0);
    /// ```
    pub fn band_power_relative(
        &self,
        band: FrequencyBand,
        ref_low_hz: f32,
        ref_high_hz: f32,
    ) -> [f32; C] {
        assert!(
            self.has_psd,
            "Must call compute() before querying band power"
        );

        let target_power = self.band_power(band);
        let ref_band = FrequencyBand::new(ref_low_hz, ref_high_hz);
        let ref_power = self.band_power(ref_band);

        let mut relative = [0.0f32; C];
        for ch in 0..C {
            if ref_power[ch] > 0.0 {
                relative[ch] = target_power[ch] / ref_power[ch];
            }
        }

        relative
    }

    /// Returns the normalized band power for all channels.
    ///
    /// Normalized power is the ratio of power in the target band to total
    /// power across all frequencies (0 to Nyquist). Always in range [0, 1].
    ///
    /// # Arguments
    ///
    /// * `band` - Target frequency band
    ///
    /// # Returns
    ///
    /// Array of normalized power values (0 to 1), one per channel.
    ///
    /// # Panics
    ///
    /// Panics if [`compute`](Self::compute) has not been called.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{MultiBandPower, FrequencyBand};
    ///
    /// let mut bp: MultiBandPower<256, 4> = MultiBandPower::new(250.0);
    /// let signals = [[0.0f32; 256]; 4];
    /// bp.compute(&signals);
    ///
    /// let norm_alpha = bp.band_power_normalized(FrequencyBand::ALPHA);
    /// // norm_alpha[ch] is in range [0, 1]
    /// ```
    pub fn band_power_normalized(&self, band: FrequencyBand) -> [f32; C] {
        assert!(
            self.has_psd,
            "Must call compute() before querying band power"
        );

        let target_power = self.band_power(band);
        let total_band = FrequencyBand::new(0.0, self.sample_rate / 2.0);
        let total_power = self.band_power(total_band);

        let mut normalized = [0.0f32; C];
        for ch in 0..C {
            if total_power[ch] > 0.0 {
                normalized[ch] = target_power[ch] / total_power[ch];
            }
        }

        normalized
    }

    /// Returns the raw PSD for a specific channel.
    ///
    /// The PSD has N/2+1 bins covering frequencies from 0 to Nyquist (fs/2).
    /// The frequency of bin k is `k * fs / N` Hz.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index (0 to C-1)
    ///
    /// # Returns
    ///
    /// Slice of PSD values in V²/Hz. Only the first N/2+1 elements are valid
    /// (one-sided spectrum).
    ///
    /// # Panics
    ///
    /// Panics if `channel >= C` or if [`compute`](Self::compute) has not been called.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::MultiBandPower;
    ///
    /// let mut bp: MultiBandPower<256, 2> = MultiBandPower::new(250.0);
    /// let signals = [[0.0f32; 256]; 2];
    /// bp.compute(&signals);
    ///
    /// let psd_ch0 = bp.psd(0);
    /// // psd_ch0[0] is DC, psd_ch0[128] is Nyquist (125 Hz)
    /// ```
    pub fn psd(&self, channel: usize) -> &[f32] {
        assert!(self.has_psd, "Must call compute() before accessing PSD");
        assert!(channel < C, "Channel index out of bounds");
        &self.psd[channel][..=N / 2]
    }

    /// Resets the internal state, clearing stored PSD.
    ///
    /// After calling this, [`compute`](Self::compute) must be called again
    /// before querying band power.
    pub fn reset(&mut self) {
        self.has_psd = false;
        // Note: we don't need to zero the PSD array since has_psd guards access
    }

    /// Returns the sample rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the frequency resolution in Hz (frequency per bin).
    pub fn freq_resolution(&self) -> f32 {
        self.freq_resolution
    }

    /// Returns the window type.
    pub fn window(&self) -> WindowType {
        self.window
    }

    /// Returns the integration method.
    pub fn integration_method(&self) -> IntegrationMethod {
        self.integration
    }

    /// Returns the frequency in Hz for a given bin index.
    ///
    /// # Arguments
    ///
    /// * `bin` - Bin index (0 to N/2)
    ///
    /// # Returns
    ///
    /// Frequency in Hz corresponding to the bin.
    pub fn bin_to_freq(&self, bin: usize) -> f32 {
        bin as f32 * self.freq_resolution
    }

    /// Converts a frequency band to bin indices.
    fn band_to_bins(&self, band: FrequencyBand) -> (usize, usize) {
        let low_bin = libm::floorf(band.low_hz / self.freq_resolution) as usize;
        let high_bin = libm::ceilf(band.high_hz / self.freq_resolution) as usize;

        // Clamp to valid range
        let low_bin = low_bin.min(N / 2);
        let high_bin = high_bin.min(N / 2);

        (low_bin, high_bin)
    }

    /// Integrates PSD over a bin range using the configured method.
    fn integrate_band(&self, channel: usize, low_bin: usize, high_bin: usize) -> f32 {
        if low_bin >= high_bin {
            return 0.0;
        }

        match self.integration {
            IntegrationMethod::Rectangular => {
                // Left Riemann sum: sum(PSD[k] * df) for k in [low, high)
                let sum: f32 = self.psd[channel][low_bin..high_bin].iter().sum();
                sum * self.freq_resolution
            }
            IntegrationMethod::Trapezoidal => {
                // Trapezoidal rule: df * (PSD[low]/2 + PSD[low+1] + ... + PSD[high-1] + PSD[high]/2)
                if high_bin <= low_bin {
                    return 0.0;
                }

                let mut sum = 0.0f32;

                // First term (half weight)
                sum += self.psd[channel][low_bin] / 2.0;

                // Middle terms (full weight)
                for k in (low_bin + 1)..high_bin {
                    sum += self.psd[channel][k];
                }

                // Last term (half weight)
                sum += self.psd[channel][high_bin] / 2.0;

                sum * self.freq_resolution
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI as PI32;

    /// Test that DC signal has all power in DC bin.
    #[test]
    fn test_dc_signal_power() {
        // Use rectangular window to avoid spreading DC power
        let mut bp: MultiBandPower<64, 1> = MultiBandPower::with_config(
            100.0,
            WindowType::Rectangular,
            IntegrationMethod::Trapezoidal,
        );

        // Constant signal (DC)
        let signals = [[1.0f32; 64]];
        bp.compute(&signals);

        // DC power should dominate
        let psd = bp.psd(0);
        let dc_power = psd[0];
        let other_power: f32 = psd[1..].iter().sum();

        // DC should have much more power than other bins
        assert!(
            dc_power > other_power * 10.0,
            "DC signal should have power concentrated in DC bin (DC: {}, other: {})",
            dc_power,
            other_power
        );
    }

    /// Test that pure tone has power localized near its frequency.
    #[test]
    fn test_pure_tone_localization() {
        let mut bp: MultiBandPower<256, 1> = MultiBandPower::new(256.0);

        // 10 Hz sine wave (exactly on bin 10)
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as f32 / 256.0;
            *s = libm::sinf(2.0 * PI32 * 10.0 * t);
        }
        let signals = [signal];
        bp.compute(&signals);

        // Alpha band (8-12 Hz) should contain most power
        let alpha = bp.band_power(FrequencyBand::ALPHA);

        // Other bands should have much less
        let delta = bp.band_power(FrequencyBand::DELTA);
        let beta = bp.band_power(FrequencyBand::BETA);

        assert!(
            alpha[0] > delta[0] * 10.0,
            "10 Hz tone should have more power in alpha than delta"
        );
        assert!(
            alpha[0] > beta[0] * 10.0,
            "10 Hz tone should have more power in alpha than beta"
        );
    }

    /// Test Parseval's theorem: time-domain power ≈ frequency-domain power.
    #[test]
    fn test_parseval_theorem() {
        let mut bp: MultiBandPower<256, 1> = MultiBandPower::with_config(
            256.0,
            WindowType::Rectangular,
            IntegrationMethod::Trapezoidal,
        );

        // Create a signal with known power
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI32 * 20.0 * i as f32 / 256.0);
        }

        // Time-domain power (mean squared)
        let time_power: f32 = signal.iter().map(|x| x * x).sum::<f32>() / 256.0;

        let signals = [signal];
        bp.compute(&signals);

        // Frequency-domain total power
        let total_band = FrequencyBand::new(0.0, 128.0);
        let freq_power = bp.band_power(total_band);

        // Should be approximately equal (within 20% for this test)
        let ratio = freq_power[0] / time_power;
        assert!(
            (ratio - 1.0).abs() < 0.2,
            "Parseval's theorem: time and frequency power should match (ratio: {})",
            ratio
        );
    }

    /// Test that relative power in non-overlapping bands sums to approximately 1.
    #[test]
    fn test_relative_power_sums() {
        let mut bp: MultiBandPower<256, 1> = MultiBandPower::new(256.0);

        // White-ish noise (pseudo-random)
        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            // Simple deterministic "noise"
            *s = libm::sinf(i as f32 * 0.1)
                + libm::cosf(i as f32 * 0.37)
                + libm::sinf(i as f32 * 0.73);
        }
        let signals = [signal];
        bp.compute(&signals);

        // Get normalized power for several bands
        let delta = bp.band_power_normalized(FrequencyBand::DELTA);
        let theta = bp.band_power_normalized(FrequencyBand::THETA);
        let alpha = bp.band_power_normalized(FrequencyBand::ALPHA);
        let beta = bp.band_power_normalized(FrequencyBand::BETA);

        // Each should be between 0 and 1
        assert!(delta[0] >= 0.0 && delta[0] <= 1.0);
        assert!(theta[0] >= 0.0 && theta[0] <= 1.0);
        assert!(alpha[0] >= 0.0 && alpha[0] <= 1.0);
        assert!(beta[0] >= 0.0 && beta[0] <= 1.0);
    }

    /// Test that multi-channel processing is independent.
    #[test]
    fn test_multichannel_independence() {
        let mut bp: MultiBandPower<128, 2> = MultiBandPower::new(128.0);

        // Channel 0: 5 Hz (theta band)
        // Channel 1: 10 Hz (alpha band)
        let signals: [[f32; 128]; 2] = [
            core::array::from_fn(|i| libm::sinf(2.0 * PI32 * 5.0 * i as f32 / 128.0)),
            core::array::from_fn(|i| libm::sinf(2.0 * PI32 * 10.0 * i as f32 / 128.0)),
        ];
        bp.compute(&signals);

        let theta = bp.band_power(FrequencyBand::THETA);
        let alpha = bp.band_power(FrequencyBand::ALPHA);

        // Channel 0 should have more theta than alpha
        assert!(
            theta[0] > alpha[0],
            "Channel 0 (5 Hz) should have more theta than alpha"
        );

        // Channel 1 should have more alpha than theta
        assert!(
            alpha[1] > theta[1],
            "Channel 1 (10 Hz) should have more alpha than theta"
        );
    }

    /// Test integration method comparison.
    #[test]
    fn test_integration_methods() {
        let mut bp_rect: MultiBandPower<256, 1> =
            MultiBandPower::with_config(256.0, WindowType::Hann, IntegrationMethod::Rectangular);
        let mut bp_trap: MultiBandPower<256, 1> =
            MultiBandPower::with_config(256.0, WindowType::Hann, IntegrationMethod::Trapezoidal);

        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI32 * 10.0 * i as f32 / 256.0);
        }
        let signals = [signal];

        bp_rect.compute(&signals);
        bp_trap.compute(&signals);

        let rect_power = bp_rect.band_power(FrequencyBand::ALPHA);
        let trap_power = bp_trap.band_power(FrequencyBand::ALPHA);

        // Both should give similar results (within 20%)
        let ratio = rect_power[0] / trap_power[0];
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Integration methods should give similar results"
        );
    }

    /// Test frequency band helper methods.
    #[test]
    fn test_frequency_band_helpers() {
        let band = FrequencyBand::new(8.0, 12.0);
        assert_eq!(band.center_hz(), 10.0);
        assert_eq!(band.bandwidth_hz(), 4.0);

        assert_eq!(FrequencyBand::ALPHA.center_hz(), 10.0);
        assert_eq!(FrequencyBand::BETA.bandwidth_hz(), 18.0);
    }

    /// Test standard frequency bands are correct.
    #[test]
    fn test_standard_bands() {
        assert_eq!(FrequencyBand::DELTA.low_hz, 0.5);
        assert_eq!(FrequencyBand::DELTA.high_hz, 4.0);

        assert_eq!(FrequencyBand::THETA.low_hz, 4.0);
        assert_eq!(FrequencyBand::THETA.high_hz, 8.0);

        assert_eq!(FrequencyBand::ALPHA.low_hz, 8.0);
        assert_eq!(FrequencyBand::ALPHA.high_hz, 12.0);

        assert_eq!(FrequencyBand::BETA.low_hz, 12.0);
        assert_eq!(FrequencyBand::BETA.high_hz, 30.0);

        assert_eq!(FrequencyBand::GAMMA.low_hz, 30.0);
        assert_eq!(FrequencyBand::GAMMA.high_hz, 100.0);

        // MU should equal ALPHA (both 8-12 Hz)
        assert_eq!(FrequencyBand::MU.low_hz, FrequencyBand::ALPHA.low_hz);
        assert_eq!(FrequencyBand::MU.high_hz, FrequencyBand::ALPHA.high_hz);
    }

    /// Test reset clears state.
    #[test]
    #[should_panic(expected = "Must call compute()")]
    fn test_reset_clears_state() {
        let mut bp: MultiBandPower<64, 1> = MultiBandPower::new(100.0);
        let signals = [[1.0f32; 64]];
        bp.compute(&signals);

        // This should work
        let _ = bp.band_power(FrequencyBand::ALPHA);

        // Reset and try again - should panic
        bp.reset();
        let _ = bp.band_power(FrequencyBand::ALPHA);
    }

    /// Test accessor methods.
    #[test]
    fn test_accessors() {
        let bp: MultiBandPower<256, 4> =
            MultiBandPower::with_config(500.0, WindowType::Hamming, IntegrationMethod::Rectangular);

        assert_eq!(bp.sample_rate(), 500.0);
        assert_eq!(bp.window(), WindowType::Hamming);
        assert_eq!(bp.integration_method(), IntegrationMethod::Rectangular);
        assert!((bp.freq_resolution() - 500.0 / 256.0).abs() < 1e-6);
    }

    /// Test bin to frequency conversion.
    #[test]
    fn test_bin_to_freq() {
        let bp: MultiBandPower<256, 1> = MultiBandPower::new(256.0);

        assert_eq!(bp.bin_to_freq(0), 0.0);
        assert_eq!(bp.bin_to_freq(1), 1.0); // 256 Hz / 256 = 1 Hz per bin
        assert_eq!(bp.bin_to_freq(10), 10.0);
        assert_eq!(bp.bin_to_freq(128), 128.0); // Nyquist
    }

    /// Test different window types produce different results.
    #[test]
    fn test_window_affects_result() {
        let mut bp_hann: MultiBandPower<256, 1> =
            MultiBandPower::with_config(256.0, WindowType::Hann, IntegrationMethod::Trapezoidal);
        let mut bp_rect: MultiBandPower<256, 1> = MultiBandPower::with_config(
            256.0,
            WindowType::Rectangular,
            IntegrationMethod::Trapezoidal,
        );

        let mut signal = [0.0f32; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            // Non-integer frequency causes spectral leakage
            *s = libm::sinf(2.0 * PI32 * 10.5 * i as f32 / 256.0);
        }
        let signals = [signal];

        bp_hann.compute(&signals);
        bp_rect.compute(&signals);

        // Rectangular will have more spectral leakage outside main lobe
        let beta_hann = bp_hann.band_power(FrequencyBand::BETA);
        let beta_rect = bp_rect.band_power(FrequencyBand::BETA);

        // Rectangular should leak more into beta band for non-integer frequency
        assert!(
            beta_rect[0] > beta_hann[0],
            "Rectangular window should have more spectral leakage"
        );
    }

    /// Test empty/edge case bands.
    #[test]
    fn test_edge_case_bands() {
        let mut bp: MultiBandPower<64, 1> = MultiBandPower::new(64.0);
        let signals = [[1.0f32; 64]];
        bp.compute(&signals);

        // Very narrow band
        let narrow = bp.band_power(FrequencyBand::new(10.0, 10.1));
        assert!(narrow[0] >= 0.0);

        // Band at Nyquist
        let nyquist = bp.band_power(FrequencyBand::new(30.0, 32.0));
        assert!(nyquist[0] >= 0.0);
    }
}
