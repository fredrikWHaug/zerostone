//! NLMS (Normalized Least Mean Squares) adaptive filter.
//!
//! NLMS improves upon LMS by normalizing the step size by the input signal power,
//! providing better convergence and stability across varying signal amplitudes.
//!
//! # Algorithm
//!
//! The NLMS filter implements:
//! ```text
//! y(n) = w^T(n-1) * u(n)                                    // Filter output
//! e(n) = d(n) - y(n)                                         // Error signal
//! w(n) = w(n-1) + [μ/(ε + ||u(n)||²)] * e(n) * u(n)        // Normalized weight update
//! ```
//!
//! where:
//! - `||u(n)||²` = input power (sum of squared samples in delay line)
//! - `ε` = small regularization constant (prevents division by zero)
//! - `μ` = normalized step size (typically 0.1 - 1.0, larger than LMS)
//!
//! # Advantages over LMS
//!
//! - **Input power independence**: Step size auto-adjusts to signal level
//! - **Faster convergence**: Especially with varying amplitude signals
//! - **Better stability**: Less sensitive to input signal characteristics
//! - **Wider step size range**: Can use μ ∈ (0, 2) vs LMS requiring μ < 2/λ_max
//!
//! # BCI Applications
//!
//! Same as LMS but recommended when:
//! - Input signal amplitude varies over time
//! - Faster convergence is needed
//! - Signal power is uncertain or changing
//!
//! # Example
//!
//! ```
//! use zerostone::NlmsFilter;
//!
//! // NLMS can use larger step size (0.5) than LMS
//! let mut nlms = NlmsFilter::<64>::new(0.5, 0.01);
//!
//! # let eog_data = [0.0f32; 100];
//! # let eeg_data = [0.0f32; 100];
//! for i in 0..eeg_data.len() {
//!     let result = nlms.process_sample(eog_data[i], eeg_data[i]);
//!     let clean_eeg = result.output;
//! }
//! ```

use crate::filter::lms::AdaptiveOutput;

/// NLMS (Normalized Least Mean Squares) adaptive filter.
///
/// Provides improved convergence over LMS by normalizing the adaptation step
/// by the input signal power. Particularly effective when input amplitude varies.
///
/// # Type Parameters
///
/// * `N` - Number of filter taps (adaptive coefficients)
///
/// # Step Size Guidelines
///
/// - **Range**: 0 < μ < 2 (wider than LMS)
/// - **Practical**: 0.1 - 1.0 for BCI applications
/// - **Common**: μ = 0.5 or μ = 1.0 work well in most cases
///
/// # Example
///
/// ```
/// use zerostone::NlmsFilter;
///
/// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
/// let result = nlms.process_sample(0.5, 1.0);
/// ```
pub struct NlmsFilter<const N: usize> {
    /// Adaptive filter coefficients (weights)
    weights: [f32; N],
    /// Circular buffer storing past N input samples
    delay_line: [f32; N],
    /// Current write position in circular buffer
    index: usize,
    /// Normalized step size (typically 0.1 - 1.0)
    mu: f32,
    /// Regularization constant to prevent division by zero
    epsilon: f32,
}

impl<const N: usize> NlmsFilter<N> {
    /// Creates a new NLMS filter with zero-initialized weights.
    ///
    /// # Arguments
    ///
    /// * `mu` - Normalized step size, typically 0.1 - 1.0
    /// * `epsilon` - Regularization constant, typically 0.01
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let nlms = NlmsFilter::<64>::new(0.5, 0.01);
    /// ```
    pub fn new(mu: f32, epsilon: f32) -> Self {
        Self {
            weights: [0.0; N],
            delay_line: [0.0; N],
            index: 0,
            mu,
            epsilon,
        }
    }

    /// Creates an NLMS filter with specified initial weights.
    ///
    /// # Arguments
    ///
    /// * `mu` - Normalized step size
    /// * `epsilon` - Regularization constant
    /// * `weights` - Initial filter coefficients
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let weights = [0.5, 1.0, 0.5];
    /// let nlms = NlmsFilter::<3>::with_weights(0.5, 0.01, weights);
    /// ```
    pub fn with_weights(mu: f32, epsilon: f32, weights: [f32; N]) -> Self {
        Self {
            weights,
            delay_line: [0.0; N],
            index: 0,
            mu,
            epsilon,
        }
    }

    /// Processes a single sample through the NLMS adaptive filter.
    ///
    /// Performs the NLMS algorithm with power normalization for improved stability.
    ///
    /// # Arguments
    ///
    /// * `input` - Reference/input signal
    /// * `desired` - Desired signal
    ///
    /// # Returns
    ///
    /// [`AdaptiveOutput`] containing filtered output and error signal.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
    /// let result = nlms.process_sample(0.5, 1.0);
    /// ```
    pub fn process_sample(&mut self, input: f32, desired: f32) -> AdaptiveOutput {
        // 1. Store input in circular buffer
        self.delay_line[self.index] = input;

        // 2. Compute output: y(n) = w^T * u(n)
        let mut output = 0.0;
        let mut delay_idx = self.index;

        for tap in 0..N {
            output += self.weights[tap] * self.delay_line[delay_idx];
            delay_idx = if delay_idx == 0 { N - 1 } else { delay_idx - 1 };
        }

        // 3. Compute error: e(n) = d(n) - y(n)
        let error = desired - output;

        // 4. NLMS: Compute input power ||u(n)||²
        let mut input_power = 0.0;
        delay_idx = self.index;

        for _ in 0..N {
            let val = self.delay_line[delay_idx];
            input_power += val * val;
            delay_idx = if delay_idx == 0 { N - 1 } else { delay_idx - 1 };
        }

        // Normalized step size: μ/(ε + ||u||²)
        let norm_mu = self.mu / (self.epsilon + input_power);

        // 5. Update weights with normalized step
        delay_idx = self.index;
        for tap in 0..N {
            self.weights[tap] += norm_mu * error * self.delay_line[delay_idx];
            delay_idx = if delay_idx == 0 { N - 1 } else { delay_idx - 1 };
        }

        // 6. Advance circular buffer index
        self.index = (self.index + 1) % N;

        AdaptiveOutput { output, error }
    }

    /// Processes multiple samples in place.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Reference/input signal samples
    /// * `desired` - Desired signal samples
    /// * `outputs` - Output buffer for results
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{NlmsFilter, AdaptiveOutput};
    ///
    /// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
    ///
    /// let inputs = [0.1, 0.2, 0.3, 0.4];
    /// let desired = [1.0, 1.1, 0.9, 1.0];
    /// let mut outputs = [AdaptiveOutput { output: 0.0, error: 0.0 }; 4];
    ///
    /// nlms.process_block(&inputs, &desired, &mut outputs);
    /// ```
    pub fn process_block(
        &mut self,
        inputs: &[f32],
        desired: &[f32],
        outputs: &mut [AdaptiveOutput],
    ) {
        let len = inputs.len().min(desired.len()).min(outputs.len());
        for i in 0..len {
            outputs[i] = self.process_sample(inputs[i], desired[i]);
        }
    }

    /// Predicts the filter output without adapting weights.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal value
    ///
    /// # Returns
    ///
    /// Predicted output based on current weights.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let mut nlms = NlmsFilter::<16>::new(0.5, 0.01);
    ///
    /// // Train
    /// for _ in 0..100 {
    ///     nlms.process_sample(0.5, 1.0);
    /// }
    ///
    /// // Predict only
    /// let prediction = nlms.predict(0.5);
    /// ```
    pub fn predict(&mut self, input: f32) -> f32 {
        self.delay_line[self.index] = input;

        let mut output = 0.0;
        let mut delay_idx = self.index;

        for tap in 0..N {
            output += self.weights[tap] * self.delay_line[delay_idx];
            delay_idx = if delay_idx == 0 { N - 1 } else { delay_idx - 1 };
        }

        self.index = (self.index + 1) % N;

        output
    }

    /// Resets the filter state (delay line and index).
    ///
    /// Preserves learned weights.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
    ///
    /// for _ in 0..100 {
    ///     nlms.process_sample(0.5, 1.0);
    /// }
    ///
    /// nlms.reset(); // Clear state, keep weights
    /// ```
    pub fn reset(&mut self) {
        self.delay_line = [0.0; N];
        self.index = 0;
    }

    /// Resets the adaptive weights to zero.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
    ///
    /// for _ in 0..100 {
    ///     nlms.process_sample(0.5, 1.0);
    /// }
    ///
    /// nlms.reset_weights(); // Start fresh
    /// ```
    pub fn reset_weights(&mut self) {
        self.weights = [0.0; N];
    }

    /// Returns a reference to the current filter weights.
    pub fn weights(&self) -> &[f32; N] {
        &self.weights
    }

    /// Sets the filter weights.
    pub fn set_weights(&mut self, weights: [f32; N]) {
        self.weights = weights;
    }

    /// Returns the current step size.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Sets the step size.
    pub fn set_mu(&mut self, mu: f32) {
        self.mu = mu;
    }

    /// Returns the regularization constant epsilon.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Sets the regularization constant epsilon.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - New epsilon value (typically 0.001 - 0.1)
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::NlmsFilter;
    ///
    /// let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
    /// nlms.set_epsilon(0.001); // Lower for better precision
    /// ```
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }
}

impl<const N: usize> Default for NlmsFilter<N> {
    fn default() -> Self {
        Self::new(0.5, 0.01) // Default for BCI applications
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FirFilter;
    use core::f32::consts::PI;

    #[test]
    fn test_nlms_new() {
        let nlms = NlmsFilter::<32>::new(0.5, 0.01);
        assert_eq!(nlms.mu(), 0.5);
        assert_eq!(nlms.epsilon(), 0.01);
        for &w in nlms.weights() {
            assert_eq!(w, 0.0);
        }
    }

    #[test]
    fn test_nlms_with_weights() {
        let weights = [0.5, 1.0, 0.5];
        let nlms = NlmsFilter::with_weights(0.8, 0.01, weights);
        assert_eq!(nlms.mu(), 0.8);
        assert_eq!(nlms.weights()[1], 1.0);
    }

    #[test]
    fn test_nlms_system_identification() {
        let mut system = FirFilter::new([0.5, 1.0, 0.5]);
        let mut nlms = NlmsFilter::<3>::new(0.8, 0.01);

        for i in 0..1000 {
            let input = (i % 17) as f32 / 17.0 - 0.5;
            let desired = system.process_sample(input);
            nlms.process_sample(input, desired);
        }

        let w = nlms.weights();
        assert!(
            (w[0] - 0.5).abs() < 0.2,
            "Expected w[0] ≈ 0.5, got {}",
            w[0]
        );
        assert!(
            (w[1] - 1.0).abs() < 0.2,
            "Expected w[1] ≈ 1.0, got {}",
            w[1]
        );
        assert!(
            (w[2] - 0.5).abs() < 0.2,
            "Expected w[2] ≈ 0.5, got {}",
            w[2]
        );
    }

    #[test]
    fn test_nlms_converges_faster_than_lms() {
        // NLMS should converge faster with varying amplitude signals
        use crate::filter::LmsFilter;

        let mut lms = LmsFilter::<2>::new(0.01);
        let mut nlms = NlmsFilter::<2>::new(0.5, 0.01);

        let mut lms_error_sum = 0.0;
        let mut nlms_error_sum = 0.0;

        // Use varying amplitude input (more challenging for LMS)
        for i in 0..300 {
            let amplitude = 1.0 + 0.5 * libm::sinf(i as f32 * 0.05);
            let input = amplitude * ((i % 11) as f32 / 11.0 - 0.5);

            let mut sys_clone = FirFilter::new([0.8, 0.5]);
            let desired = sys_clone.process_sample(input);

            let lms_result = lms.process_sample(input, desired);
            let nlms_result = nlms.process_sample(input, desired);

            // Collect errors after initial convergence
            if i >= 200 {
                lms_error_sum += lms_result.error.abs();
                nlms_error_sum += nlms_result.error.abs();
            }
        }

        // NLMS should have lower average error
        assert!(
            nlms_error_sum < lms_error_sum,
            "NLMS error {} should be < LMS error {}",
            nlms_error_sum,
            lms_error_sum
        );
    }

    #[test]
    fn test_nlms_stability_with_varying_power() {
        let mut nlms = NlmsFilter::<16>::new(1.0, 0.01);

        // Low power input should not cause instability
        for _ in 0..100 {
            let result = nlms.process_sample(0.001, 0.0);
            assert!(result.output.is_finite());
        }

        // High power input should adapt correctly
        for _ in 0..100 {
            let result = nlms.process_sample(10.0, 5.0);
            assert!(result.output.is_finite());
        }

        // Verify weights are reasonable
        for &w in nlms.weights() {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_nlms_zero_input() {
        let mut nlms = NlmsFilter::<16>::new(0.5, 0.01);

        for _ in 0..100 {
            let result = nlms.process_sample(0.0, 0.0);
            assert_eq!(result.output, 0.0);
            assert_eq!(result.error, 0.0);
        }
    }

    #[test]
    fn test_nlms_reset() {
        let mut nlms = NlmsFilter::<16>::new(0.5, 0.01);

        for i in 0..10 {
            let input = i as f32 * 0.1;
            nlms.process_sample(input, input * 2.0);
        }

        nlms.reset();

        let result = nlms.process_sample(0.0, 0.0);
        assert_eq!(result.output, 0.0);
    }

    #[test]
    fn test_nlms_reset_weights() {
        let mut nlms = NlmsFilter::<16>::new(0.5, 0.01);

        for i in 0..100 {
            let input = (i % 7) as f32 / 7.0;
            nlms.process_sample(input, input * 2.0);
        }

        let has_nonzero = nlms.weights().iter().any(|&w| w.abs() > 0.01);
        assert!(has_nonzero);

        nlms.reset_weights();

        for &w in nlms.weights() {
            assert_eq!(w, 0.0);
        }
    }

    #[test]
    fn test_nlms_process_block() {
        let mut nlms = NlmsFilter::<8>::new(0.5, 0.01);

        let inputs = [0.1, 0.2, 0.3, 0.4];
        let desired = [1.0, 1.1, 0.9, 1.0];
        let mut outputs = [AdaptiveOutput {
            output: 0.0,
            error: 0.0,
        }; 4];

        nlms.process_block(&inputs, &desired, &mut outputs);

        for (i, out) in outputs.iter().enumerate() {
            assert_eq!(out.error, desired[i] - out.output);
        }
    }

    #[test]
    fn test_nlms_predict() {
        let mut nlms = NlmsFilter::<8>::with_weights(0.5, 0.01, [1.0; 8]);

        let weights_before = *nlms.weights();
        let output = nlms.predict(0.5);
        let weights_after = *nlms.weights();

        assert!(output.is_finite());
        assert_eq!(weights_before, weights_after);
    }

    #[test]
    fn test_nlms_setters() {
        let mut nlms = NlmsFilter::<8>::new(0.5, 0.01);

        nlms.set_mu(0.8);
        assert_eq!(nlms.mu(), 0.8);

        nlms.set_epsilon(0.001);
        assert_eq!(nlms.epsilon(), 0.001);

        let new_weights = [0.1; 8];
        nlms.set_weights(new_weights);
        assert_eq!(*nlms.weights(), new_weights);
    }

    #[test]
    fn test_nlms_default() {
        let nlms: NlmsFilter<32> = Default::default();
        assert_eq!(nlms.mu(), 0.5);
        assert_eq!(nlms.epsilon(), 0.01);
    }

    #[test]
    fn test_nlms_noise_reduction() {
        let mut nlms = NlmsFilter::<16>::new(1.0, 0.01);

        let mut initial_error_sum = 0.0;
        let mut final_error_sum = 0.0;

        for i in 0..500 {
            let input = libm::sinf(2.0 * PI * 0.1 * i as f32);
            let desired = 2.0 * input;

            let result = nlms.process_sample(input, desired);

            if i < 50 {
                initial_error_sum += result.error.abs();
            }
            if i >= 450 {
                final_error_sum += result.error.abs();
            }
        }

        let initial_avg_error = initial_error_sum / 50.0;
        let final_avg_error = final_error_sum / 50.0;

        assert!(
            final_avg_error < initial_avg_error * 0.3,
            // NLMS should converge faster than LMS
            "Error should decrease: initial {}, final {}",
            initial_avg_error,
            final_avg_error
        );
    }
}
