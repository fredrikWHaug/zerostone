//! Synthetic signal generators for testing and demonstration.
//!
//! This module provides functions to generate common test signals:
//! - Sine waves (single frequency)
//! - White noise
//! - Chirps (frequency sweeps)
//! - Impulse trains
//! - Composite signals (multiple components)

#![allow(dead_code)] // Functions used across multiple examples

use std::f32::consts::PI;
use zerostone::{BiquadCoeffs, Complex, Fft};

/// Generates a sine wave.
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `sample_rate` - Sample rate in Hz
/// * `frequency` - Sine frequency in Hz
/// * `amplitude` - Peak amplitude
/// * `phase` - Initial phase in radians
pub fn sine_wave(
    samples: usize,
    sample_rate: f32,
    frequency: f32,
    amplitude: f32,
    phase: f32,
) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            amplitude * (2.0 * PI * frequency * t + phase).sin()
        })
        .collect()
}

/// Generates white noise using a simple LCG random number generator.
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `amplitude` - Peak amplitude (noise will be in range [-amplitude, amplitude])
/// * `seed` - Random seed for reproducibility
pub fn white_noise(samples: usize, amplitude: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..samples)
        .map(|_| {
            // Simple LCG: x_{n+1} = (a * x_n + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [-1, 1] range
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            amplitude * normalized
        })
        .collect()
}

/// Generates a linear chirp (frequency sweep).
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `sample_rate` - Sample rate in Hz
/// * `start_freq` - Starting frequency in Hz
/// * `end_freq` - Ending frequency in Hz
/// * `amplitude` - Peak amplitude
pub fn chirp(
    samples: usize,
    sample_rate: f32,
    start_freq: f32,
    end_freq: f32,
    amplitude: f32,
) -> Vec<f32> {
    let duration = samples as f32 / sample_rate;
    let freq_rate = (end_freq - start_freq) / duration;

    (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let instantaneous_freq = start_freq + freq_rate * t / 2.0;
            amplitude * (2.0 * PI * instantaneous_freq * t).sin()
        })
        .collect()
}

/// Generates an impulse train (periodic impulses).
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `sample_rate` - Sample rate in Hz
/// * `impulse_rate` - Number of impulses per second
/// * `amplitude` - Impulse amplitude
pub fn impulse_train(
    samples: usize,
    sample_rate: f32,
    impulse_rate: f32,
    amplitude: f32,
) -> Vec<f32> {
    let period_samples = (sample_rate / impulse_rate) as usize;
    (0..samples)
        .map(|i| {
            if period_samples > 0 && i % period_samples == 0 {
                amplitude
            } else {
                0.0
            }
        })
        .collect()
}

/// Generates a composite signal with multiple sine components plus noise.
///
/// Useful for creating "EEG-like" signals with multiple frequency bands.
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `sample_rate` - Sample rate in Hz
/// * `components` - Vec of (frequency, amplitude) tuples
/// * `noise_amplitude` - Amplitude of additive white noise
/// * `seed` - Random seed for noise
pub fn composite_signal(
    samples: usize,
    sample_rate: f32,
    components: &[(f32, f32)],
    noise_amplitude: f32,
    seed: u64,
) -> Vec<f32> {
    let mut signal = vec![0.0; samples];

    // Add sine components
    for &(freq, amp) in components {
        let sine = sine_wave(samples, sample_rate, freq, amp, 0.0);
        for (i, &s) in sine.iter().enumerate() {
            signal[i] += s;
        }
    }

    // Add noise
    if noise_amplitude > 0.0 {
        let noise = white_noise(samples, noise_amplitude, seed);
        for (i, &n) in noise.iter().enumerate() {
            signal[i] += n;
        }
    }

    signal
}

/// Generates a step function (square wave with single transition).
///
/// # Arguments
/// * `samples` - Number of samples to generate
/// * `step_sample` - Sample index where step occurs
/// * `low_value` - Value before step
/// * `high_value` - Value after step
pub fn step_function(
    samples: usize,
    step_sample: usize,
    low_value: f32,
    high_value: f32,
) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            if i < step_sample {
                low_value
            } else {
                high_value
            }
        })
        .collect()
}

/// Computes power spectrum in dB from a signal.
///
/// # Arguments
/// * `signal` - Input signal (will be zero-padded or truncated to FFT size)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Tuple of (frequency_vec, power_dB_vec) where frequencies go from 0 to Nyquist
pub fn compute_power_spectrum(signal: &[f32], sample_rate: f32) -> (Vec<f32>, Vec<f32>) {
    const FFT_SIZE: usize = 1024;

    // Prepare FFT input (zero-pad or truncate)
    let mut fft_input = [0.0f32; FFT_SIZE];
    let copy_len = signal.len().min(FFT_SIZE);
    fft_input[..copy_len].copy_from_slice(&signal[..copy_len]);

    // Compute power spectrum
    let fft = Fft::<FFT_SIZE>::new();
    let mut power_spec = vec![0.0f32; FFT_SIZE / 2 + 1];
    fft.power_spectrum(&fft_input, &mut power_spec);

    // Convert to dB (10 * log10(power))
    // Use floor of -100 dB for numerical stability
    let power_db: Vec<f32> = power_spec
        .iter()
        .map(|&p| {
            if p > 0.0 {
                10.0 * p.log10()
            } else {
                -100.0 // Floor value
            }
        })
        .collect();

    // Generate frequency vector (0 to Nyquist)
    let freq_resolution = sample_rate / FFT_SIZE as f32;
    let frequencies: Vec<f32> = (0..=FFT_SIZE / 2)
        .map(|i| i as f32 * freq_resolution)
        .collect();

    (frequencies, power_db)
}

/// Computes filter frequency response (magnitude and phase).
///
/// # Arguments
/// * `coeffs` - Array of biquad coefficient sections
/// * `sample_rate` - Sample rate in Hz
/// * `num_points` - Number of frequency points to evaluate (default 512)
///
/// # Returns
/// Tuple of (frequency_vec, magnitude_dB_vec, phase_deg_vec)
pub fn compute_filter_response(
    coeffs: &[BiquadCoeffs],
    sample_rate: f32,
    num_points: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nyquist = sample_rate / 2.0;
    let mut frequencies = Vec::with_capacity(num_points);
    let mut magnitude_db = Vec::with_capacity(num_points);
    let mut phase_deg = Vec::with_capacity(num_points);

    for i in 0..num_points {
        // Frequency from 0 to Nyquist
        let freq = (i as f32 / (num_points - 1) as f32) * nyquist;
        frequencies.push(freq);

        // Angular frequency
        let omega = 2.0 * PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();

        // Start with unity response
        let mut h = Complex::new(1.0, 0.0);

        // Multiply response of each biquad section
        for coeff in coeffs {
            // Numerator: b0 + b1*z^-1 + b2*z^-2
            // z^-1 = e^(-jω) = cos(-ω) + j*sin(-ω)
            let z_inv = Complex::new(cos_omega, -sin_omega);
            let z_inv_2 = z_inv.cmul(z_inv);

            let numerator = Complex::new(coeff.b0, 0.0)
                .cadd(Complex::new(coeff.b1, 0.0).cmul(z_inv))
                .cadd(Complex::new(coeff.b2, 0.0).cmul(z_inv_2));

            // Denominator: 1 + a1*z^-1 + a2*z^-2
            let denominator = Complex::new(1.0, 0.0)
                .cadd(Complex::new(coeff.a1, 0.0).cmul(z_inv))
                .cadd(Complex::new(coeff.a2, 0.0).cmul(z_inv_2));

            // H(z) = numerator / denominator
            // Division: a/b = a * conj(b) / |b|^2
            let denom_mag_sq = denominator.magnitude_squared();
            let section_response = numerator.cmul(denominator.conj());
            let section_response = Complex::new(
                section_response.re / denom_mag_sq,
                section_response.im / denom_mag_sq,
            );

            // Cascade: multiply with accumulated response
            h = h.cmul(section_response);
        }

        // Convert to magnitude (dB) and phase (degrees)
        let mag = h.magnitude();
        let mag_db = if mag > 0.0 {
            20.0 * mag.log10()
        } else {
            -120.0
        };
        magnitude_db.push(mag_db);

        let phase = h.im.atan2(h.re) * 180.0 / PI;
        phase_deg.push(phase);
    }

    (frequencies, magnitude_db, phase_deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave() {
        let signal = sine_wave(100, 100.0, 10.0, 1.0, 0.0);
        assert_eq!(signal.len(), 100);
        // At t=0, sin(0) = 0
        assert!(signal[0].abs() < 0.01);
        // Peak should be around amplitude
        let max = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_white_noise() {
        let noise = white_noise(1000, 1.0, 12345);
        assert_eq!(noise.len(), 1000);
        // Check range
        for &n in &noise {
            assert!(n >= -1.0 && n <= 1.0);
        }
        // Check it's not all zeros
        let sum: f32 = noise.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_chirp() {
        let signal = chirp(1000, 1000.0, 10.0, 100.0, 1.0);
        assert_eq!(signal.len(), 1000);
    }

    #[test]
    fn test_composite() {
        let signal = composite_signal(1000, 1000.0, &[(10.0, 1.0), (50.0, 0.5)], 0.1, 42);
        assert_eq!(signal.len(), 1000);
    }
}
