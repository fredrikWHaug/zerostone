//! Envelope detection for tracking signal amplitude.
//!
//! This module provides an envelope follower that extracts the amplitude
//! envelope from a signal using rectification and smoothing. Useful for
//! motor imagery BCI, EMG processing, and amplitude-based event detection.

use core::f32::consts::PI;

/// Rectification method for envelope detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rectification {
    /// Absolute value: `|x|`
    Absolute,
    /// Squared value: `x²` (proportional to power)
    Squared,
}

/// An envelope follower that tracks signal amplitude.
///
/// Extracts the amplitude envelope by rectifying the signal and applying
/// exponential smoothing (single-pole lowpass filter). Supports separate
/// attack and release time constants for asymmetric response.
///
/// # Example
///
/// ```
/// use zerostone::{EnvelopeFollower, Rectification};
///
/// // Create envelope follower with 10ms attack, 100ms release at 250 Hz
/// let mut env: EnvelopeFollower<4> = EnvelopeFollower::new(
///     250.0,           // sample rate
///     0.010,           // attack time (seconds)
///     0.100,           // release time (seconds)
///     Rectification::Absolute,
/// );
///
/// // Process samples
/// let input = [0.5, -0.3, 0.8, -0.2];
/// let envelope = env.process(&input);
/// // envelope contains smoothed amplitude for each channel
/// ```
pub struct EnvelopeFollower<const C: usize> {
    /// Attack coefficient (for rising envelope)
    attack_coeff: f32,
    /// Release coefficient (for falling envelope)
    release_coeff: f32,
    /// Rectification method
    rectification: Rectification,
    /// Current envelope value per channel
    envelope: [f32; C],
}

impl<const C: usize> EnvelopeFollower<C> {
    /// Creates a new envelope follower with specified time constants.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `attack_time` - Attack time in seconds (response to increasing amplitude)
    /// * `release_time` - Release time in seconds (response to decreasing amplitude)
    /// * `rectification` - Rectification method (Absolute or Squared)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{EnvelopeFollower, Rectification};
    ///
    /// // Fast attack (10ms), slow release (100ms)
    /// let env: EnvelopeFollower<8> = EnvelopeFollower::new(
    ///     250.0,
    ///     0.010,
    ///     0.100,
    ///     Rectification::Absolute,
    /// );
    /// ```
    pub fn new(
        sample_rate: f32,
        attack_time: f32,
        release_time: f32,
        rectification: Rectification,
    ) -> Self {
        Self {
            attack_coeff: Self::time_to_coeff(sample_rate, attack_time),
            release_coeff: Self::time_to_coeff(sample_rate, release_time),
            rectification,
            envelope: [0.0; C],
        }
    }

    /// Creates an envelope follower with symmetric attack/release.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `smoothing_time` - Time constant in seconds for both attack and release
    /// * `rectification` - Rectification method
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{EnvelopeFollower, Rectification};
    ///
    /// // Symmetric 50ms smoothing
    /// let env: EnvelopeFollower<4> = EnvelopeFollower::symmetric(
    ///     250.0,
    ///     0.050,
    ///     Rectification::Absolute,
    /// );
    /// ```
    pub fn symmetric(sample_rate: f32, smoothing_time: f32, rectification: Rectification) -> Self {
        Self::new(sample_rate, smoothing_time, smoothing_time, rectification)
    }

    /// Convert time constant to filter coefficient.
    ///
    /// Uses the formula: coeff = 1 - exp(-2π / (sample_rate * time))
    fn time_to_coeff(sample_rate: f32, time: f32) -> f32 {
        if time <= 0.0 {
            1.0 // Instant response
        } else {
            1.0 - libm::expf(-2.0 * PI / (sample_rate * time))
        }
    }

    /// Process a single multi-channel sample.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample array with one value per channel
    ///
    /// # Returns
    ///
    /// Current envelope value for each channel
    pub fn process(&mut self, input: &[f32; C]) -> [f32; C] {
        for (i, &sample) in input.iter().enumerate() {
            // Rectify
            let rectified = match self.rectification {
                Rectification::Absolute => libm::fabsf(sample),
                Rectification::Squared => sample * sample,
            };

            // Apply attack or release coefficient based on direction
            let coeff = if rectified > self.envelope[i] {
                self.attack_coeff
            } else {
                self.release_coeff
            };

            // Exponential smoothing: env = env + coeff * (rectified - env)
            self.envelope[i] += coeff * (rectified - self.envelope[i]);
        }

        self.envelope
    }

    /// Process a block of samples in place.
    ///
    /// Each sample is replaced with its envelope value.
    pub fn process_block(&mut self, block: &mut [[f32; C]]) {
        for sample in block.iter_mut() {
            *sample = self.process(sample);
        }
    }

    /// Get the current envelope values.
    pub fn current(&self) -> &[f32; C] {
        &self.envelope
    }

    /// Reset the envelope to zero.
    pub fn reset(&mut self) {
        self.envelope = [0.0; C];
    }

    /// Set new attack time.
    pub fn set_attack_time(&mut self, sample_rate: f32, attack_time: f32) {
        self.attack_coeff = Self::time_to_coeff(sample_rate, attack_time);
    }

    /// Set new release time.
    pub fn set_release_time(&mut self, sample_rate: f32, release_time: f32) {
        self.release_coeff = Self::time_to_coeff(sample_rate, release_time);
    }

    /// Set rectification method.
    pub fn set_rectification(&mut self, rectification: Rectification) {
        self.rectification = rectification;
    }

    /// Get current rectification method.
    pub fn rectification(&self) -> Rectification {
        self.rectification
    }
}

impl<const C: usize> Default for EnvelopeFollower<C> {
    /// Creates an envelope follower with 10ms attack, 100ms release at 250 Hz.
    fn default() -> Self {
        Self::new(250.0, 0.010, 0.100, Rectification::Absolute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_follows_amplitude() {
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::symmetric(250.0, 0.020, Rectification::Absolute);

        // Feed increasing amplitude
        for i in 1..=10 {
            let input = [i as f32 * 0.1];
            env.process(&input);
        }

        // Envelope should have increased
        assert!(env.current()[0] > 0.5);
    }

    #[test]
    fn test_envelope_rectification_absolute() {
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::new(250.0, 0.001, 0.001, Rectification::Absolute);

        // Negative input should produce positive envelope
        let output = env.process(&[-1.0]);
        assert!(output[0] > 0.0);
    }

    #[test]
    fn test_envelope_rectification_squared() {
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::new(250.0, 0.001, 0.001, Rectification::Squared);

        // Process a known value
        env.reset();
        let output = env.process(&[2.0]);
        // With very fast attack, should be close to 4.0 (2²)
        assert!(output[0] > 3.0);
    }

    #[test]
    fn test_envelope_attack_release_asymmetry() {
        // Fast attack, slow release
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::new(1000.0, 0.001, 0.100, Rectification::Absolute);

        // Spike up
        for _ in 0..10 {
            env.process(&[1.0]);
        }
        let peak = env.current()[0];

        // Release - should decay slowly
        for _ in 0..10 {
            env.process(&[0.0]);
        }
        let after_release = env.current()[0];

        // Should still retain most of the envelope due to slow release
        assert!(after_release > peak * 0.5);
    }

    #[test]
    fn test_envelope_multi_channel() {
        let mut env: EnvelopeFollower<4> =
            EnvelopeFollower::symmetric(250.0, 0.010, Rectification::Absolute);

        // Different amplitudes per channel
        let input = [0.1, 0.5, 1.0, 0.0];

        for _ in 0..100 {
            env.process(&input);
        }

        let current = env.current();
        // Envelope should reflect relative amplitudes
        assert!(current[2] > current[1]);
        assert!(current[1] > current[0]);
        assert!(current[3] < current[0]);
    }

    #[test]
    fn test_envelope_reset() {
        let mut env: EnvelopeFollower<2> =
            EnvelopeFollower::symmetric(250.0, 0.010, Rectification::Absolute);

        // Build up envelope
        for _ in 0..100 {
            env.process(&[1.0, 1.0]);
        }
        assert!(env.current()[0] > 0.5);

        // Reset
        env.reset();
        assert_eq!(env.current()[0], 0.0);
        assert_eq!(env.current()[1], 0.0);
    }

    #[test]
    fn test_envelope_process_block() {
        let mut env: EnvelopeFollower<2> =
            EnvelopeFollower::symmetric(250.0, 0.010, Rectification::Absolute);

        let mut block = [[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]];
        env.process_block(&mut block);

        // Block should be modified to contain envelope values
        // Each successive value should be larger as envelope builds
        assert!(block[3][0] > block[0][0]);
    }

    #[test]
    fn test_envelope_setters() {
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::new(250.0, 0.010, 0.100, Rectification::Absolute);

        env.set_attack_time(250.0, 0.005);
        env.set_release_time(250.0, 0.050);
        env.set_rectification(Rectification::Squared);

        assert_eq!(env.rectification(), Rectification::Squared);
    }

    #[test]
    fn test_envelope_default() {
        let env: EnvelopeFollower<4> = EnvelopeFollower::default();
        assert_eq!(env.rectification(), Rectification::Absolute);
    }

    #[test]
    fn test_envelope_instant_attack() {
        // Zero attack time = instant response
        let mut env: EnvelopeFollower<1> =
            EnvelopeFollower::new(250.0, 0.0, 0.100, Rectification::Absolute);

        let output = env.process(&[1.0]);
        // Should immediately reach input value
        assert!((output[0] - 1.0).abs() < 0.01);
    }
}
