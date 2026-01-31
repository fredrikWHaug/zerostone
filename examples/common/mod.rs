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
