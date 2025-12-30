/// Zero-allocation FIR (Finite Impulse Response) filter.
///
/// Implements direct-form FIR filtering using a circular buffer for state storage.
/// FIR filters have linear phase response and guaranteed stability.
///
/// # Memory Layout
/// - `coeffs`: Filter tap weights [b0, b1, ..., b_{TAPS-1}]
/// - `delay_line`: Circular buffer storing past TAPS input samples
///
/// # Example
/// ```
/// # use zerostone::FirFilter;
/// // 5-tap moving average filter
/// let coeffs = [0.2, 0.2, 0.2, 0.2, 0.2];
/// let mut filter = FirFilter::new(coeffs);
///
/// let output = filter.process_sample(1.0);
/// ```
pub struct FirFilter<const TAPS: usize> {
    coeffs: [f32; TAPS],
    delay_line: [f32; TAPS],
    index: usize,
}

impl<const TAPS: usize> FirFilter<TAPS> {
    /// Creates a new FIR filter with given tap coefficients.
    ///
    /// # Example
    /// ```
    /// # use zerostone::FirFilter;
    /// // 3-tap moving average
    /// let filter = FirFilter::new([1.0/3.0, 1.0/3.0, 1.0/3.0]);
    /// ```
    pub fn new(coeffs: [f32; TAPS]) -> Self {
        Self {
            coeffs,
            delay_line: [0.0; TAPS],
            index: 0,
        }
    }

    /// Processes a single sample through the FIR filter.
    ///
    /// Implements: `y[n] = sum(b[k] * x[n-k])` for k = 0 to TAPS-1
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Store new sample in delay line
        self.delay_line[self.index] = input;

        // Compute output as dot product of coefficients and delay line
        let mut output = 0.0;
        let mut delay_idx = self.index;

        for tap in 0..TAPS {
            output += self.coeffs[tap] * self.delay_line[delay_idx];

            // Move backward through delay line (with wrap-around)
            delay_idx = if delay_idx == 0 {
                TAPS - 1
            } else {
                delay_idx - 1
            };
        }

        // Update index for next sample
        self.index = (self.index + 1) % TAPS;

        output
    }

    /// Processes multiple samples in place.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Resets filter state (clears delay line).
    pub fn reset(&mut self) {
        self.delay_line = [0.0; TAPS];
        self.index = 0;
    }

    /// Returns a reference to the filter coefficients.
    pub fn coefficients(&self) -> &[f32; TAPS] {
        &self.coeffs
    }

    /// Updates filter coefficients.
    pub fn set_coefficients(&mut self, coeffs: [f32; TAPS]) {
        self.coeffs = coeffs;
    }

    /// Creates a moving average filter with equal weights.
    ///
    /// # Example
    /// ```
    /// # use zerostone::FirFilter;
    /// let mut filter: FirFilter<5> = FirFilter::moving_average();
    /// ```
    pub fn moving_average() -> Self {
        let weight = 1.0 / TAPS as f32;
        Self::new([weight; TAPS])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fir_moving_average() {
        let mut filter: FirFilter<5> = FirFilter::moving_average();

        // Feed 5 ones
        for _ in 0..5 {
            filter.process_sample(1.0);
        }

        // After 5 samples of 1.0, moving average should output 1.0
        let output = filter.process_sample(1.0);
        assert!(
            (output - 1.0).abs() < 0.001,
            "Moving average of ones should be 1.0"
        );
    }

    #[test]
    fn test_fir_impulse_response() {
        let coeffs = [1.0, 2.0, 3.0, 2.0, 1.0];
        let mut filter = FirFilter::new(coeffs);

        // Feed an impulse (1.0 followed by zeros)
        let output1 = filter.process_sample(1.0);
        assert_eq!(output1, 1.0, "First output should be first coefficient");

        let output2 = filter.process_sample(0.0);
        assert_eq!(output2, 2.0, "Second output should be second coefficient");

        let output3 = filter.process_sample(0.0);
        assert_eq!(output3, 3.0, "Third output should be third coefficient");
    }

    #[test]
    fn test_fir_dc_gain() {
        // Coefficients that sum to 1.0
        let coeffs = [0.2, 0.2, 0.2, 0.2, 0.2];
        let mut filter = FirFilter::new(coeffs);

        // Feed DC signal
        for _ in 0..10 {
            filter.process_sample(1.0);
        }

        // DC gain should be sum of coefficients = 1.0
        let output = filter.process_sample(1.0);
        assert!((output - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fir_reset() {
        let mut filter: FirFilter<3> = FirFilter::moving_average();

        // Process some samples
        for i in 0..5 {
            filter.process_sample(i as f32);
        }

        // Reset
        filter.reset();

        // Should produce same output as fresh filter
        let mut fresh_filter: FirFilter<3> = FirFilter::moving_average();
        let out1 = filter.process_sample(10.0);
        let out2 = fresh_filter.process_sample(10.0);

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_fir_process_block() {
        let mut filter: FirFilter<3> = FirFilter::new([1.0, 0.0, 0.0]);

        let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        filter.process_block(&mut samples);

        // With coeffs [1,0,0], output = input (but delayed)
        // First sample: only sees first input
        assert_eq!(samples[0], 1.0);
    }

    #[test]
    fn test_fir_coefficients_access() {
        let coeffs = [0.5, 0.3, 0.2];
        let mut filter = FirFilter::new(coeffs);

        let retrieved = filter.coefficients();
        assert_eq!(retrieved, &[0.5, 0.3, 0.2]);

        // Update coefficients
        filter.set_coefficients([1.0, 0.0, 0.0]);
        assert_eq!(filter.coefficients(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fir_single_tap() {
        // Single tap is just a gain
        let mut filter = FirFilter::new([2.0]);

        assert_eq!(filter.process_sample(1.0), 2.0);
        assert_eq!(filter.process_sample(3.0), 6.0);
    }
}
