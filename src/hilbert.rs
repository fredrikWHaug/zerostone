//! Hilbert Transform for analytic signal computation.
//!
//! This module provides Hilbert transform primitives for extracting instantaneous
//! phase, amplitude, and frequency from narrowband signals. These are fundamental
//! operations in BCI for phase-amplitude coupling analysis, motor imagery classification,
//! and real-time neurofeedback.
//!
//! # Overview
//!
//! The Hilbert transform computes the **analytic signal** via FFT:
//! ```text
//! a(t) = x(t) + i·H[x(t)]
//! ```
//! where `H[x(t)]` is the Hilbert transform (90° phase shift).
//!
//! From the analytic signal, we can extract:
//! - **Instantaneous amplitude**: `|a(t)|`
//! - **Instantaneous phase**: `arg(a(t))`
//! - **Instantaneous frequency**: `d(phase)/dt`
//!
//! # Important Limitations
//!
//! **Narrowband assumption**: The Hilbert transform only provides meaningful
//! instantaneous parameters for narrowband signals (nearly sinusoidal). You must
//! bandpass filter your signal before applying the Hilbert transform.
//!
//! **Non-causal**: The FFT-based approach requires the entire signal, making it
//! unsuitable for true real-time processing without latency.
//!
//! # BCI Applications
//!
//! - **Phase-Amplitude Coupling (PAC)**: Measure coupling between low-frequency
//!   phase (e.g., theta) and high-frequency amplitude (e.g., gamma)
//! - **Motor Imagery**: Extract event-related patterns for BCI classification
//! - **Neurofeedback**: Real-time phase/amplitude tracking for closed-loop stimulation
//!
//! # Example
//!
//! ```
//! use zerostone::{BiquadCoeffs, IirFilter, hilbert::HilbertTransform};
//!
//! // 1. Bandpass filter signal to alpha band (8-13 Hz)
//! let coeffs = BiquadCoeffs::butterworth_bandpass(250.0, 8.0, 13.0);
//! let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
//!
//! // Simulate some EEG data (in practice, this comes from acquisition hardware)
//! let raw_eeg = [0.0f32; 256];
//!
//! let mut signal = [0.0f32; 256];
//! for i in 0..256 {
//!     signal[i] = filter.process_sample(raw_eeg[i]);
//! }
//!
//! // 2. Extract instantaneous amplitude (alpha power envelope)
//! let hilbert = HilbertTransform::<256>::new();
//! let mut amplitude = [0.0f32; 256];
//! hilbert.instantaneous_amplitude(&signal, &mut amplitude);
//!
//! // 3. Extract phase for PAC analysis
//! let mut phase = [0.0f32; 256];
//! hilbert.instantaneous_phase(&signal, &mut phase);
//! ```

use crate::fft::{Complex, Fft};

/// Hilbert Transform processor for analytic signal computation.
///
/// Computes the analytic signal `a(t) = x(t) + i·H[x(t)]` using the FFT method.
/// The imaginary part `H[x(t)]` is the Hilbert transform, which represents a 90°
/// phase shift of the input signal's frequency components.
///
/// # Type Parameters
///
/// * `N` - Signal length (must be a power of 2 for FFT)
///
/// # Algorithm
///
/// 1. Convert real signal to complex via FFT
/// 2. Zero out negative frequency components
/// 3. Double positive frequency components
/// 4. Inverse FFT to obtain analytic signal
///
/// # Important
///
/// The Hilbert transform assumes **narrowband signals**. Bandpass filter your
/// signal before applying this transform to get meaningful instantaneous parameters.
///
/// # Example
///
/// ```
/// use zerostone::hilbert::HilbertTransform;
///
/// let signal = [1.0f32, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, 0.0];
/// let mut amplitude = [0.0f32; 8];
///
/// let hilbert = HilbertTransform::<8>::new();
/// hilbert.instantaneous_amplitude(&signal, &mut amplitude);
/// ```
pub struct HilbertTransform<const N: usize> {
    fft: Fft<N>,
}

impl<const N: usize> HilbertTransform<N> {
    /// Creates a new Hilbert transform processor.
    ///
    /// # Panics
    ///
    /// Panics if `N` is not a power of 2 (required for FFT).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::hilbert::HilbertTransform;
    ///
    /// let hilbert = HilbertTransform::<64>::new();
    /// ```
    pub const fn new() -> Self {
        Self { fft: Fft::new() }
    }

    /// Computes the analytic signal.
    ///
    /// The analytic signal is a complex-valued representation where:
    /// - Real part = original signal `x(t)`
    /// - Imaginary part = Hilbert transform `H[x(t)]`
    ///
    /// From the analytic signal, you can extract:
    /// - Amplitude envelope: `magnitude()`
    /// - Instantaneous phase: `atan2(im, re)`
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal (length N)
    /// * `output` - Complex analytic signal output (length N)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Complex, hilbert::HilbertTransform};
    ///
    /// let signal = [1.0f32, 0.0, -1.0, 0.0];
    /// let mut analytic = [Complex::new(0.0, 0.0); 4];
    ///
    /// let hilbert = HilbertTransform::<4>::new();
    /// hilbert.analytic_signal(&signal, &mut analytic);
    ///
    /// // Real part matches original signal
    /// assert!((analytic[0].re - 1.0).abs() < 1e-6);
    /// ```
    #[allow(clippy::needless_range_loop)] // Need to modify both re and im fields
    pub fn analytic_signal(&self, signal: &[f32; N], output: &mut [Complex; N]) {
        // 1. Convert real signal to complex
        for (i, &sample) in signal.iter().enumerate() {
            output[i] = Complex::from_real(sample);
        }

        // 2. Forward FFT (time → frequency domain)
        self.fft.forward(output);

        // 3. Create analytic signal in frequency domain
        // - Keep DC component (index 0)
        // - Double positive frequencies (indices 1 to N/2-1)
        // - Keep Nyquist component (index N/2)
        // - Zero negative frequencies (indices N/2+1 to N-1)

        // DC: keep real, zero imaginary
        output[0].im = 0.0;

        // Double positive frequencies
        for i in 1..N / 2 {
            output[i].re *= 2.0;
            output[i].im *= 2.0;
        }

        // Nyquist: keep real, zero imaginary
        output[N / 2].im = 0.0;

        // Zero negative frequencies
        for i in (N / 2 + 1)..N {
            output[i].re = 0.0;
            output[i].im = 0.0;
        }

        // 4. Inverse FFT (frequency → time domain)
        self.fft.inverse(output);

        // Result: output contains analytic signal
        // real(output) = original signal
        // imag(output) = Hilbert transform
    }

    /// Computes the Hilbert transform only (imaginary part of analytic signal).
    ///
    /// This is equivalent to applying a 90° phase shift to all frequency components.
    /// For a pure cosine wave, the Hilbert transform produces a sine wave.
    ///
    /// # Mathematical Properties
    ///
    /// - `H[constant] = 0` (DC has no Hilbert transform)
    /// - `H[cos(ωt)] = sin(ωt)`
    /// - `H[sin(ωt)] = -cos(ωt)`
    /// - `H[H[x]] = -x` (inverse property)
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal
    /// * `output` - Hilbert-transformed output (real-valued)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::hilbert::HilbertTransform;
    ///
    /// // Hilbert transform of constant is zero
    /// let signal = [5.0f32; 64];
    /// let mut hilbert_out = [0.0f32; 64];
    ///
    /// let hilbert = HilbertTransform::<64>::new();
    /// hilbert.transform(&signal, &mut hilbert_out);
    ///
    /// for &val in &hilbert_out {
    ///     assert!(val.abs() < 1e-3);
    /// }
    /// ```
    pub fn transform(&self, signal: &[f32; N], output: &mut [f32; N]) {
        let mut analytic = [Complex::new(0.0, 0.0); N];
        self.analytic_signal(signal, &mut analytic);

        // Extract imaginary part (Hilbert transform)
        for (i, &z) in analytic.iter().enumerate() {
            output[i] = z.im;
        }
    }

    /// Computes the instantaneous amplitude (envelope) of the signal.
    ///
    /// The instantaneous amplitude is the magnitude of the analytic signal:
    /// ```text
    /// A(t) = |a(t)| = sqrt(x²(t) + H[x(t)]²)
    /// ```
    ///
    /// This represents the time-varying amplitude envelope of the oscillation.
    /// For BCI, this is used to track power fluctuations in specific frequency bands.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal (should be bandpass filtered)
    /// * `output` - Instantaneous amplitude output
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::hilbert::HilbertTransform;
    /// use core::f32::consts::PI;
    ///
    /// // Amplitude-modulated signal: A(t) = (1 + 0.5·cos(ωₘt))·cos(ωₓt)
    /// let mut signal = [0.0f32; 128];
    /// for (i, sample) in signal.iter_mut().enumerate() {
    ///     let t = i as f32 / 128.0;
    ///     let modulation = 1.0 + 0.5 * (2.0 * PI * 2.0 * t).cos();
    ///     *sample = modulation * (2.0 * PI * 10.0 * t).cos();
    /// }
    ///
    /// let mut amplitude = [0.0f32; 128];
    /// let hilbert = HilbertTransform::<128>::new();
    /// hilbert.instantaneous_amplitude(&signal, &mut amplitude);
    ///
    /// // Amplitude should follow modulation envelope
    /// assert!(amplitude[0] > 0.5);  // Rough check
    /// ```
    pub fn instantaneous_amplitude(&self, signal: &[f32; N], output: &mut [f32; N]) {
        let mut analytic = [Complex::new(0.0, 0.0); N];
        self.analytic_signal(signal, &mut analytic);

        for (i, &z) in analytic.iter().enumerate() {
            output[i] = z.magnitude(); // sqrt(re² + im²)
        }
    }

    /// Computes the instantaneous phase of the signal.
    ///
    /// The instantaneous phase is the angle of the analytic signal:
    /// ```text
    /// φ(t) = arg(a(t)) = atan2(H[x(t)], x(t))
    /// ```
    ///
    /// Phase is returned in radians in the range [-π, π]. For BCI applications,
    /// this is used in phase-amplitude coupling and phase-locked stimulation.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal (should be bandpass filtered)
    /// * `output` - Instantaneous phase in radians [-π, π]
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::hilbert::HilbertTransform;
    /// use core::f32::consts::PI;
    ///
    /// // Pure sine wave at 10 Hz
    /// let mut signal = [0.0f32; 64];
    /// for (i, sample) in signal.iter_mut().enumerate() {
    ///     let t = i as f32 / 64.0;
    ///     *sample = (2.0 * PI * 10.0 * t).sin();
    /// }
    ///
    /// let mut phase = [0.0f32; 64];
    /// let hilbert = HilbertTransform::<64>::new();
    /// hilbert.instantaneous_phase(&signal, &mut phase);
    ///
    /// // Phase should be approximately -π/2 at t=0 for sine
    /// assert!((phase[0] + PI / 2.0).abs() < 0.2);
    /// ```
    pub fn instantaneous_phase(&self, signal: &[f32; N], output: &mut [f32; N]) {
        let mut analytic = [Complex::new(0.0, 0.0); N];
        self.analytic_signal(signal, &mut analytic);

        for (i, &z) in analytic.iter().enumerate() {
            output[i] = libm::atan2f(z.im, z.re); // phase in radians [-π, π]
        }
    }

    /// Computes the instantaneous frequency of the signal.
    ///
    /// The instantaneous frequency is the time derivative of the unwrapped phase:
    /// ```text
    /// f(t) = (1/2π) · d(φ(t))/dt
    /// ```
    ///
    /// Since this is a discrete derivative, the output has length `N-1`.
    /// Phase unwrapping is performed to handle 2π discontinuities.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal (should be bandpass filtered)
    /// * `output` - Output buffer for instantaneous frequency (Hz)
    /// * `sample_rate` - Sampling rate in Hz
    ///
    /// # Returns
    ///
    /// Number of frequency values computed (N-1).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::hilbert::HilbertTransform;
    /// use core::f32::consts::PI;
    ///
    /// // Pure sine at 15 Hz
    /// let sample_rate = 256.0;
    /// let mut signal = [0.0f32; 128];
    /// for (i, sample) in signal.iter_mut().enumerate() {
    ///     let t = i as f32 / sample_rate;
    ///     *sample = (2.0 * PI * 15.0 * t).sin();
    /// }
    ///
    /// let mut freq = [0.0f32; 127];
    /// let hilbert = HilbertTransform::<128>::new();
    /// let n = hilbert.instantaneous_frequency(&signal, &mut freq, sample_rate);
    ///
    /// assert_eq!(n, 127);
    /// // Frequency should be close to 15 Hz (with edge effects)
    /// let mean_freq: f32 = freq[10..117].iter().sum::<f32>() / 107.0;
    /// assert!((mean_freq - 15.0).abs() < 1.0);
    /// ```
    pub fn instantaneous_frequency(
        &self,
        signal: &[f32; N],
        output: &mut [f32],
        sample_rate: f32,
    ) -> usize {
        let mut phase = [0.0f32; N];
        self.instantaneous_phase(signal, &mut phase);

        // Unwrap phase (handle 2π discontinuities)
        let mut unwrapped = [0.0f32; N];
        unwrapped[0] = phase[0];
        let mut cumulative = 0.0f32;

        for i in 1..N {
            let diff = phase[i] - phase[i - 1];
            // Detect and correct for phase wraps
            let corrected = if diff > core::f32::consts::PI {
                diff - 2.0 * core::f32::consts::PI
            } else if diff < -core::f32::consts::PI {
                diff + 2.0 * core::f32::consts::PI
            } else {
                diff
            };
            cumulative += corrected;
            unwrapped[i] = phase[0] + cumulative;
        }

        // Compute derivative: f(t) = (1/2π) · dφ/dt
        let n_out = (N - 1).min(output.len());
        for i in 0..n_out {
            let dphase_dt = (unwrapped[i + 1] - unwrapped[i]) * sample_rate;
            output[i] = dphase_dt / (2.0 * core::f32::consts::PI);
        }

        n_out
    }
}

impl<const N: usize> Default for HilbertTransform<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch Hilbert transform for multi-channel signals.
///
/// Processes multiple channels independently, computing the analytic signal
/// for each channel. This is useful for EEG/BCI applications where you need
/// to extract phase or amplitude from multiple electrodes simultaneously.
///
/// # Type Parameters
///
/// * `N` - Signal length per channel (must be power of 2)
/// * `C` - Number of channels
///
/// # Arguments
///
/// * `signals` - Input signals (C channels, each of length N)
/// * `output` - Analytic signal output (C channels, each of length N)
///
/// # Example
///
/// ```
/// use zerostone::{Complex, hilbert::hilbert_batch};
///
/// // 3 channels, 64 samples each
/// let signals: [[f32; 64]; 3] = [
///     [1.0; 64],
///     [2.0; 64],
///     [3.0; 64],
/// ];
/// let mut output = [[Complex::new(0.0, 0.0); 64]; 3];
///
/// hilbert_batch(&signals, &mut output);
///
/// // Each channel processed independently
/// for ch in 0..3 {
///     assert!((output[ch][0].re - signals[ch][0]).abs() < 1e-5);
/// }
/// ```
pub fn hilbert_batch<const N: usize, const C: usize>(
    signals: &[[f32; N]; C],
    output: &mut [[Complex; N]; C],
) {
    let hilbert = HilbertTransform::<N>::new();
    for ch in 0..C {
        hilbert.analytic_signal(&signals[ch], &mut output[ch]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_hilbert_dc_is_zero() {
        // Hilbert transform of constant signal should be zero
        let signal = [5.0f32; 64];
        let mut output = [0.0f32; 64];

        let hilbert = HilbertTransform::<64>::new();
        hilbert.transform(&signal, &mut output);

        for &val in &output {
            assert!(val.abs() < 1e-3, "Hilbert of DC should be ~0, got {}", val);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // Need index for time computation
    fn test_hilbert_cosine_gives_sine() {
        // H[cos(ωt)] = sin(ωt)
        let n = 128;
        let freq = 4.0; // 4 Hz - exactly periodic in window (bin 2)
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 128];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = libm::cosf(2.0 * PI * freq * t);
        }

        let mut hilbert_out = [0.0f32; 128];
        let hilbert = HilbertTransform::<128>::new();
        hilbert.transform(&signal, &mut hilbert_out);

        // Check against expected sine (skip edges due to FFT artifacts)
        for i in 10..n - 10 {
            let t = i as f32 / sample_rate;
            let expected = libm::sinf(2.0 * PI * freq * t);
            assert!(
                (hilbert_out[i] - expected).abs() < 0.15,
                "At i={}: expected {}, got {}",
                i,
                expected,
                hilbert_out[i]
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // Need index for time computation
    fn test_hilbert_sine_gives_negative_cosine() {
        // H[sin(ωt)] = -cos(ωt)
        let n = 128;
        let freq = 4.0; // 4 Hz - exactly periodic in window (bin 2)
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 128];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = libm::sinf(2.0 * PI * freq * t);
        }

        let mut hilbert_out = [0.0f32; 128];
        let hilbert = HilbertTransform::<128>::new();
        hilbert.transform(&signal, &mut hilbert_out);

        // Check against expected -cos (skip edges)
        for i in 10..n - 10 {
            let t = i as f32 / sample_rate;
            let expected = -libm::cosf(2.0 * PI * freq * t);
            assert!(
                (hilbert_out[i] - expected).abs() < 0.15,
                "At i={}: expected {}, got {}",
                i,
                expected,
                hilbert_out[i]
            );
        }
    }

    #[test]
    fn test_hilbert_inverse_property() {
        // H[H[x]] = -x (applying Hilbert transform twice negates the signal)
        // Note: This property only holds for zero-mean signals (DC is zeroed by Hilbert)
        let signal = [1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]; // Zero-mean
        let mut h1 = [0.0f32; 8];
        let mut h2 = [0.0f32; 8];

        let hilbert = HilbertTransform::<8>::new();
        hilbert.transform(&signal, &mut h1);
        hilbert.transform(&h1, &mut h2);

        for (i, (&result, &original)) in h2.iter().zip(signal.iter()).enumerate() {
            assert!(
                (result + original).abs() < 0.15,
                "H[H[x]] should equal -x at i={}: got {}, expected {}",
                i,
                result,
                -original
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // Need index for assertion message
    fn test_instantaneous_amplitude_constant() {
        // For pure tone, amplitude should be constant
        let freq = 10.0;
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 128];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = 2.0 * libm::sinf(2.0 * PI * freq * t);
        }

        let mut amplitude = [0.0f32; 128];
        let hilbert = HilbertTransform::<128>::new();
        hilbert.instantaneous_amplitude(&signal, &mut amplitude);

        // Amplitude should be close to 2.0 (skip edges)
        for i in 10..118 {
            assert!(
                (amplitude[i] - 2.0).abs() < 0.2,
                "At i={}: amplitude {}, expected ~2.0",
                i,
                amplitude[i]
            );
        }
    }

    #[test]
    fn test_instantaneous_phase_monotonic() {
        // For pure tone, phase should increase monotonically (when unwrapped)
        let freq = 8.0;
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 256];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = libm::sinf(2.0 * PI * freq * t);
        }

        let mut phase = [0.0f32; 256];
        let hilbert = HilbertTransform::<256>::new();
        hilbert.instantaneous_phase(&signal, &mut phase);

        // Phase differences should be roughly constant (within wrapping)
        let mut consistent_count = 0;
        for i in 1..255 {
            let diff = phase[i] - phase[i - 1];
            // Should be small positive value or large negative (wrapping)
            if diff.abs() < 0.5 || diff < -6.0 {
                consistent_count += 1;
            }
        }

        assert!(
            consistent_count > 200,
            "Phase should be monotonic, got {} consistent differences out of 254",
            consistent_count
        );
    }

    #[test]
    fn test_analytic_signal_real_part_matches_input() {
        // Real part of analytic signal should match original signal
        let signal = [1.0f32, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, 0.0];
        let mut analytic = [Complex::new(0.0, 0.0); 8];

        let hilbert = HilbertTransform::<8>::new();
        hilbert.analytic_signal(&signal, &mut analytic);

        for (i, &expected) in signal.iter().enumerate() {
            assert!(
                (analytic[i].re - expected).abs() < 1e-5,
                "Real part at i={} should be {}, got {}",
                i,
                expected,
                analytic[i].re
            );
        }
    }

    #[test]
    fn test_batch_processing() {
        let signals: [[f32; 64]; 3] = [[1.0; 64], [2.0; 64], [3.0; 64]];
        let mut output = [[Complex::new(0.0, 0.0); 64]; 3];

        hilbert_batch(&signals, &mut output);

        // Verify each channel processed independently
        for ch in 0..3 {
            for i in 0..64 {
                assert!(
                    (output[ch][i].re - signals[ch][i]).abs() < 1e-5,
                    "Channel {} index {}: real part mismatch",
                    ch,
                    i
                );
            }
        }
    }

    #[test]
    fn test_instantaneous_frequency_pure_tone() {
        // Pure sine at 15 Hz should have constant instantaneous frequency ~15 Hz
        let freq = 15.0;
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 256];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = libm::sinf(2.0 * PI * freq * t);
        }

        let mut inst_freq = [0.0f32; 255];
        let hilbert = HilbertTransform::<256>::new();
        let n = hilbert.instantaneous_frequency(&signal, &mut inst_freq, sample_rate);

        assert_eq!(n, 255);

        // Check middle section (skip edges)
        let mean_freq: f32 = inst_freq[30..225].iter().sum::<f32>() / 195.0;
        assert!(
            (mean_freq - freq).abs() < 1.5,
            "Mean frequency should be ~{} Hz, got {} Hz",
            freq,
            mean_freq
        );
    }

    #[test]
    fn test_hilbert_default() {
        let _hilbert: HilbertTransform<64> = Default::default();
        // Just verify it compiles and constructs
    }

    #[test]
    fn test_analytic_signal_power_preservation() {
        // Energy should be roughly preserved (doubled due to one-sided spectrum)
        let mut signal = [0.0f32; 64];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / 64.0;
            *sample = (2.0 * PI * 5.0 * t).sin();
        }

        let signal_energy: f32 = signal.iter().map(|&x| x * x).sum();

        let mut analytic = [Complex::new(0.0, 0.0); 64];
        let hilbert = HilbertTransform::<64>::new();
        hilbert.analytic_signal(&signal, &mut analytic);

        let analytic_energy: f32 = analytic.iter().map(|z| z.magnitude_squared()).sum();

        // Analytic signal energy should be roughly 2x original (one-sided spectrum)
        let ratio = analytic_energy / signal_energy;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "Energy ratio should be ~2.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_instantaneous_amplitude_extractable() {
        // Test amplitude modulation extraction
        let carrier_freq = 20.0;
        let mod_freq = 2.0;
        let sample_rate = 256.0;

        let mut signal = [0.0f32; 256];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            let modulation = 1.0 + 0.5 * libm::cosf(2.0 * PI * mod_freq * t);
            *sample = modulation * libm::sinf(2.0 * PI * carrier_freq * t);
        }

        let mut amplitude = [0.0f32; 256];
        let hilbert = HilbertTransform::<256>::new();
        hilbert.instantaneous_amplitude(&signal, &mut amplitude);

        // Amplitude should vary between 0.5 and 1.5 (modulation envelope)
        let max_amp = amplitude.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_amp = amplitude.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(
            max_amp > 1.2,
            "Max amplitude should be >1.2, got {}",
            max_amp
        );
        assert!(
            min_amp < 0.8,
            "Min amplitude should be <0.8, got {}",
            min_amp
        );
    }
}
