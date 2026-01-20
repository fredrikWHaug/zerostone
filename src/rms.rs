/// Computes windowed RMS (root mean square) and power over a sliding rectangular window.
///
/// Efficiently tracks RMS and power for multi-channel signals using a circular buffer
/// and running sum-of-squares. Essential for EEG power features, amplitude tracking,
/// and signal quality monitoring in BCI applications.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `WINDOW_SIZE` - Window size in samples (must be at least 1)
///
/// # Algorithm
///
/// Uses efficient incremental update:
/// - Maintains circular buffer of last WINDOW_SIZE samples
/// - Running sum: `sum² = sum² + new² - old²`
/// - RMS: `sqrt(sum² / WINDOW_SIZE)`
/// - Power: `sum² / WINDOW_SIZE` (more efficient, no sqrt)
///
/// # Example
///
/// ```
/// use zerostone::WindowedRms;
///
/// // 0.25s window at 256 Hz for 4-channel EEG
/// let mut rms: WindowedRms<4, 64> = WindowedRms::new();
///
/// // Process samples
/// for _ in 0..100 {
///     let sample = [1.0, 0.5, 0.8, 0.3];
///     rms.process(&sample);
/// }
///
/// // Get RMS values (available after 64 samples)
/// if let Some(rms_vals) = rms.rms() {
///     println!("Channel 0 RMS: {}", rms_vals[0]);
/// }
///
/// // Get power (more efficient, no sqrt)
/// if let Some(power_vals) = rms.power() {
///     println!("Channel 0 power: {}", power_vals[0]);
/// }
/// ```
pub struct WindowedRms<const C: usize, const WINDOW_SIZE: usize> {
    /// Circular buffer of multi-channel samples
    buffer: [[f32; C]; WINDOW_SIZE],
    /// Write position in circular buffer
    index: usize,
    /// Per-channel running sum of squared values
    sum_squared: [f32; C],
    /// Number of valid samples in buffer (0 to WINDOW_SIZE)
    count: usize,
}

impl<const C: usize, const WINDOW_SIZE: usize> WindowedRms<C, WINDOW_SIZE> {
    /// Creates a new windowed RMS tracker.
    ///
    /// # Panics
    ///
    /// Panics at compile time if `WINDOW_SIZE < 1`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// // 64-sample window for 8 channels
    /// let rms: WindowedRms<8, 64> = WindowedRms::new();
    /// ```
    pub fn new() -> Self {
        const { assert!(WINDOW_SIZE >= 1, "Window size must be at least 1") };
        Self {
            buffer: [[0.0; C]; WINDOW_SIZE],
            index: 0,
            sum_squared: [0.0; C],
            count: 0,
        }
    }

    /// Process a single multi-channel sample, updating the internal state.
    ///
    /// This method updates the circular buffer and running sum-of-squares.
    /// Call `rms()` or `power()` to retrieve the current values after processing.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<2, 4> = WindowedRms::new();
    ///
    /// // Process several samples
    /// rms.process(&[1.0, 2.0]);
    /// rms.process(&[1.0, 2.0]);
    /// rms.process(&[1.0, 2.0]);
    /// rms.process(&[1.0, 2.0]);
    ///
    /// // Now window is full, can retrieve RMS
    /// assert!(rms.rms().is_some());
    /// ```
    #[inline]
    pub fn process(&mut self, input: &[f32; C]) {
        // For each channel
        for (ch, &sample) in input.iter().enumerate() {
            let old_sample = self.buffer[self.index][ch];

            // Update running sum: remove old, add new
            if self.count >= WINDOW_SIZE {
                self.sum_squared[ch] -= old_sample * old_sample;
            }
            self.sum_squared[ch] += sample * sample;

            // Store new sample
            self.buffer[self.index][ch] = sample;
        }

        // Update circular buffer index
        self.index = (self.index + 1) % WINDOW_SIZE;

        // Track initialization
        if self.count < WINDOW_SIZE {
            self.count += 1;
        }
    }

    /// Returns the current RMS (root mean square) values for each channel.
    ///
    /// Returns `None` until at least `WINDOW_SIZE` samples have been processed.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<1, 4> = WindowedRms::new();
    ///
    /// // Not enough samples yet
    /// assert_eq!(rms.rms(), None);
    ///
    /// // Feed constant value of 2.0
    /// for _ in 0..4 {
    ///     rms.process(&[2.0]);
    /// }
    ///
    /// // RMS of constant signal equals the value
    /// let rms_vals = rms.rms().unwrap();
    /// assert!((rms_vals[0] - 2.0).abs() < 1e-6);
    /// ```
    pub fn rms(&self) -> Option<[f32; C]> {
        if self.count < WINDOW_SIZE {
            return None;
        }

        let mut rms_vals = [0.0; C];
        let n = WINDOW_SIZE as f32;
        for (i, &sum_sq) in self.sum_squared.iter().enumerate() {
            rms_vals[i] = libm::sqrtf(sum_sq / n);
        }
        Some(rms_vals)
    }

    /// Returns the current power (mean of squares) values for each channel.
    ///
    /// This is more efficient than `rms()` as it avoids the square root computation.
    /// Returns `None` until at least `WINDOW_SIZE` samples have been processed.
    ///
    /// Note: `power = rms²`
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<1, 4> = WindowedRms::new();
    ///
    /// // Feed constant value of 3.0
    /// for _ in 0..4 {
    ///     rms.process(&[3.0]);
    /// }
    ///
    /// // Power = value²
    /// let power_vals = rms.power().unwrap();
    /// assert!((power_vals[0] - 9.0).abs() < 1e-6);
    ///
    /// // Verify power = rms²
    /// let rms_vals = rms.rms().unwrap();
    /// assert!((power_vals[0] - rms_vals[0] * rms_vals[0]).abs() < 1e-6);
    /// ```
    pub fn power(&self) -> Option<[f32; C]> {
        if self.count < WINDOW_SIZE {
            return None;
        }

        let mut power_vals = [0.0; C];
        let n = WINDOW_SIZE as f32;
        for (i, &sum_sq) in self.sum_squared.iter().enumerate() {
            power_vals[i] = sum_sq / n;
        }
        Some(power_vals)
    }

    /// Process a block of samples in place, replacing each sample with the RMS values.
    ///
    /// Each sample in the block is first used to update the internal state, then
    /// replaced with the current RMS values. Samples processed before the window
    /// is full will be replaced with zeros.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<2, 4> = WindowedRms::new();
    ///
    /// let mut block = [
    ///     [1.0, 2.0],
    ///     [1.0, 2.0],
    ///     [1.0, 2.0],
    ///     [1.0, 2.0],
    /// ];
    ///
    /// rms.process_block(&mut block);
    ///
    /// // Last sample has RMS values (window is now full)
    /// assert!((block[3][0] - 1.0).abs() < 1e-6);
    /// assert!((block[3][1] - 2.0).abs() < 1e-6);
    /// ```
    pub fn process_block(&mut self, block: &mut [[f32; C]]) {
        for sample in block.iter_mut() {
            self.process(sample);
            if let Some(rms_vals) = self.rms() {
                *sample = rms_vals;
            } else {
                *sample = [0.0; C];
            }
        }
    }

    /// Returns the number of samples processed so far.
    ///
    /// This value increases from 0 up to `WINDOW_SIZE`, then remains constant.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<1, 10> = WindowedRms::new();
    ///
    /// assert_eq!(rms.count(), 0);
    ///
    /// rms.process(&[1.0]);
    /// assert_eq!(rms.count(), 1);
    ///
    /// for _ in 0..20 {
    ///     rms.process(&[1.0]);
    /// }
    /// assert_eq!(rms.count(), 10); // Caps at WINDOW_SIZE
    /// ```
    pub fn count(&self) -> usize {
        self.count
    }

    /// Resets the tracker to its initial state.
    ///
    /// Clears the buffer, running sums, and sample count.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::WindowedRms;
    ///
    /// let mut rms: WindowedRms<1, 4> = WindowedRms::new();
    ///
    /// // Process some samples
    /// for _ in 0..4 {
    ///     rms.process(&[5.0]);
    /// }
    /// assert!(rms.rms().is_some());
    ///
    /// // Reset to initial state
    /// rms.reset();
    /// assert_eq!(rms.count(), 0);
    /// assert_eq!(rms.rms(), None);
    /// ```
    pub fn reset(&mut self) {
        self.buffer = [[0.0; C]; WINDOW_SIZE];
        self.index = 0;
        self.sum_squared = [0.0; C];
        self.count = 0;
    }
}

impl<const C: usize, const WINDOW_SIZE: usize> Default for WindowedRms<C, WINDOW_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_constant_signal() {
        let mut rms: WindowedRms<1, 4> = WindowedRms::new();

        // Feed constant value of 2.0
        for _ in 0..4 {
            rms.process(&[2.0]);
        }

        // RMS of constant signal equals the value
        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_known_values() {
        let mut rms: WindowedRms<1, 4> = WindowedRms::new();

        // Feed alternating [1, -1, 1, -1]
        rms.process(&[1.0]);
        rms.process(&[-1.0]);
        rms.process(&[1.0]);
        rms.process(&[-1.0]);

        // RMS = sqrt((1² + 1² + 1² + 1²) / 4) = sqrt(4/4) = 1.0
        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_power_vs_rms() {
        let mut rms: WindowedRms<2, 8> = WindowedRms::new();

        // Feed some samples
        for _ in 0..8 {
            rms.process(&[3.0, 4.0]);
        }

        let rms_vals = rms.rms().unwrap();
        let power_vals = rms.power().unwrap();

        // Verify power = rms²
        for ch in 0..2 {
            let expected_power = rms_vals[ch] * rms_vals[ch];
            assert!((power_vals[ch] - expected_power).abs() < 1e-6);
        }
    }

    #[test]
    fn test_initialization() {
        let mut rms: WindowedRms<1, 5> = WindowedRms::new();

        // Should return None until window is full
        assert_eq!(rms.rms(), None);

        for i in 0..4 {
            rms.process(&[1.0]);
            assert_eq!(rms.rms(), None, "Should be None at sample {}", i + 1);
        }

        // After 5th sample, should return Some
        rms.process(&[1.0]);
        assert!(rms.rms().is_some());
    }

    #[test]
    fn test_multi_channel_independence() {
        let mut rms: WindowedRms<3, 4> = WindowedRms::new();

        // Different constant values per channel
        for _ in 0..4 {
            rms.process(&[1.0, 2.0, 3.0]);
        }

        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 1.0).abs() < 1e-6);
        assert!((rms_vals[1] - 2.0).abs() < 1e-6);
        assert!((rms_vals[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_channel_different_amplitudes() {
        let mut rms: WindowedRms<2, 4> = WindowedRms::new();

        // Channel 0: alternating ±2
        // Channel 1: alternating ±1
        rms.process(&[2.0, 1.0]);
        rms.process(&[-2.0, -1.0]);
        rms.process(&[2.0, 1.0]);
        rms.process(&[-2.0, -1.0]);

        let rms_vals = rms.rms().unwrap();
        // Both should have RMS equal to their amplitude
        assert!((rms_vals[0] - 2.0).abs() < 1e-6);
        assert!((rms_vals[1] - 1.0).abs() < 1e-6);

        // Verify ratio
        assert!((rms_vals[0] / rms_vals[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut rms: WindowedRms<1, 4> = WindowedRms::new();

        // Build up state
        for _ in 0..4 {
            rms.process(&[5.0]);
        }
        assert!(rms.rms().is_some());
        assert_eq!(rms.count(), 4);

        // Reset
        rms.reset();
        assert_eq!(rms.count(), 0);
        assert_eq!(rms.rms(), None);

        // Should work again after reset
        for _ in 0..4 {
            rms.process(&[2.0]);
        }
        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_count() {
        let mut rms: WindowedRms<1, 10> = WindowedRms::new();

        assert_eq!(rms.count(), 0);

        for i in 1..=10 {
            rms.process(&[1.0]);
            assert_eq!(rms.count(), i);
        }

        // Should cap at WINDOW_SIZE
        for _ in 0..20 {
            rms.process(&[1.0]);
            assert_eq!(rms.count(), 10);
        }
    }

    #[test]
    fn test_process_block() {
        let mut rms: WindowedRms<2, 4> = WindowedRms::new();

        let mut block = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];

        rms.process_block(&mut block);

        // First 3 samples should be zeros (window not full yet)
        for sample in block.iter().take(3) {
            assert_eq!(*sample, [0.0, 0.0]);
        }

        // Last sample should have RMS values (window is now full)
        assert!((block[3][0] - 1.0).abs() < 1e-6);
        assert!((block[3][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_input() {
        let mut rms: WindowedRms<2, 4> = WindowedRms::new();

        for _ in 0..4 {
            rms.process(&[0.0, 0.0]);
        }

        let rms_vals = rms.rms().unwrap();
        assert_eq!(rms_vals[0], 0.0);
        assert_eq!(rms_vals[1], 0.0);

        let power_vals = rms.power().unwrap();
        assert_eq!(power_vals[0], 0.0);
        assert_eq!(power_vals[1], 0.0);
    }

    #[test]
    fn test_single_sample_window() {
        let mut rms: WindowedRms<1, 1> = WindowedRms::new();

        rms.process(&[3.0]);

        // RMS of single sample = |sample|
        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 3.0).abs() < 1e-6);

        rms.process(&[-4.0]);
        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_values() {
        let mut rms: WindowedRms<1, 4> = WindowedRms::new();

        // Use large but not overflow-inducing values
        let large_val = 1e6;
        for _ in 0..4 {
            rms.process(&[large_val]);
        }

        let rms_vals = rms.rms().unwrap();
        assert!((rms_vals[0] - large_val).abs() < 1.0); // Relaxed tolerance for large values
    }

    #[test]
    fn test_sliding_window() {
        let mut rms: WindowedRms<1, 3> = WindowedRms::new();

        // Feed [1, 2, 3]
        rms.process(&[1.0]);
        rms.process(&[2.0]);
        rms.process(&[3.0]);

        // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.160
        let rms1 = rms.rms().unwrap()[0];
        let expected1 = libm::sqrtf(14.0 / 3.0);
        assert!((rms1 - expected1).abs() < 1e-6);

        // Add 4, window becomes [2, 3, 4]
        rms.process(&[4.0]);

        // RMS = sqrt((4 + 9 + 16) / 3) = sqrt(29/3) ≈ 3.109
        let rms2 = rms.rms().unwrap()[0];
        let expected2 = libm::sqrtf(29.0 / 3.0);
        assert!((rms2 - expected2).abs() < 1e-6);
    }

    #[test]
    fn test_default() {
        let rms1: WindowedRms<4, 16> = WindowedRms::default();
        let rms2: WindowedRms<4, 16> = WindowedRms::new();

        assert_eq!(rms1.count(), rms2.count());
    }
}
