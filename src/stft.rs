//! Short-Time Fourier Transform for time-frequency analysis.
//!
//! Provides STFT for analyzing how the frequency content of a signal changes
//! over time. Essential for non-stationary signal analysis in BCI applications
//! where neural oscillations vary temporally.
//!
//! # Overview
//!
//! The STFT divides a signal into overlapping segments, applies a window
//! function to each segment, and computes the FFT. This produces a
//! time-frequency representation (spectrogram).
//!
//! # Example
//!
//! ```
//! use zerostone::{Stft, Complex, WindowType};
//!
//! // Create STFT with 256-point window, 64-sample hop, Hann window
//! let stft = Stft::<256>::new(64, WindowType::Hann);
//!
//! // Process a 1024-sample signal (produces 13 frames)
//! let signal = [0.5f32; 1024];
//! let num_frames = stft.num_frames(signal.len());
//!
//! // Compute power spectrum
//! let mut power = [[0.0f32; 256]; 13];
//! stft.power(&signal, &mut power);
//! ```
//!
//! # Frame Calculation
//!
//! For a signal of length `L` with window size `N` and hop size `H`:
//! - Number of frames = `(L - N) / H + 1`
//! - Each frame covers samples `[i*H, i*H + N)` for frame index `i`

use crate::{window_coefficient, Complex, Fft, WindowType};

/// Short-Time Fourier Transform processor.
///
/// Computes the STFT of a signal using a sliding window approach.
/// The window size `N` must be a power of 2 for the FFT.
///
/// # Type Parameters
///
/// * `N` - Window size (must be power of 2)
///
/// # Example
///
/// ```
/// use zerostone::{Stft, WindowType};
///
/// // 256-point window, 50% overlap (hop = 128)
/// let stft = Stft::<256>::new(128, WindowType::Hann);
///
/// let signal = [0.0f32; 512];
/// assert_eq!(stft.num_frames(512), 3); // frames at 0, 128, 256
/// ```
pub struct Stft<const N: usize> {
    /// Hop size (samples between consecutive frames)
    hop_size: usize,
    /// Window type for spectral analysis
    window_type: WindowType,
    /// FFT processor
    fft: Fft<N>,
}

impl<const N: usize> Stft<N> {
    /// Creates a new STFT processor.
    ///
    /// # Arguments
    ///
    /// * `hop_size` - Number of samples to advance between frames.
    ///   Common choices: `N/2` (50% overlap), `N/4` (75% overlap).
    /// * `window_type` - Window function to apply to each frame.
    ///
    /// # Panics
    ///
    /// Panics if `N` is not a power of 2 or if `hop_size` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// // 50% overlap with Hann window
    /// let stft = Stft::<256>::new(128, WindowType::Hann);
    /// ```
    pub fn new(hop_size: usize, window_type: WindowType) -> Self {
        assert!(hop_size > 0, "hop_size must be positive");

        Self {
            hop_size,
            window_type,
            fft: Fft::new(),
        }
    }

    /// Creates STFT with default 50% overlap and Hann window.
    ///
    /// This is a common configuration for spectral analysis.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::Stft;
    ///
    /// let stft = Stft::<256>::default_overlap();
    /// assert_eq!(stft.hop_size(), 128);
    /// ```
    pub fn default_overlap() -> Self {
        Self::new(N / 2, WindowType::Hann)
    }

    /// Computes the number of frames for a given signal length.
    ///
    /// # Arguments
    ///
    /// * `signal_len` - Length of the input signal
    ///
    /// # Returns
    ///
    /// Number of complete frames that fit in the signal.
    /// Returns 0 if signal is shorter than window size.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// let stft = Stft::<256>::new(64, WindowType::Hann);
    ///
    /// // 1024 samples with 256 window and 64 hop = 13 frames
    /// assert_eq!(stft.num_frames(1024), 13);
    ///
    /// // Signal shorter than window = 0 frames
    /// assert_eq!(stft.num_frames(100), 0);
    /// ```
    #[inline]
    pub fn num_frames(&self, signal_len: usize) -> usize {
        if signal_len < N {
            0
        } else {
            (signal_len - N) / self.hop_size + 1
        }
    }

    /// Computes the STFT, returning complex coefficients.
    ///
    /// Each row of the output contains the FFT of one windowed frame.
    /// Output layout: `output[frame_idx][freq_bin]`
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal (real-valued)
    /// * `output` - Output buffer for complex coefficients.
    ///   Must have at least `num_frames(signal.len())` rows.
    ///
    /// # Returns
    ///
    /// Number of frames computed.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, Complex, WindowType};
    ///
    /// let stft = Stft::<64>::new(32, WindowType::Hann);
    /// let signal = [1.0f32; 128];
    ///
    /// let mut output = [[Complex::new(0.0, 0.0); 64]; 3];
    /// let frames = stft.transform(&signal, &mut output);
    /// assert_eq!(frames, 3);
    /// ```
    pub fn transform<const F: usize>(
        &self,
        signal: &[f32],
        output: &mut [[Complex; N]; F],
    ) -> usize {
        let num_frames = self.num_frames(signal.len()).min(F);

        for (frame_idx, frame_output) in output.iter_mut().enumerate().take(num_frames) {
            let start = frame_idx * self.hop_size;
            self.compute_frame(&signal[start..start + N], frame_output);
        }

        num_frames
    }

    /// Computes the STFT power spectrum (magnitude squared).
    ///
    /// More efficient when only power is needed, as it avoids storing
    /// intermediate complex values.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `output` - Output buffer for power values.
    ///   Must have at least `num_frames(signal.len())` rows.
    ///
    /// # Returns
    ///
    /// Number of frames computed.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// let stft = Stft::<64>::new(32, WindowType::Hann);
    /// let signal = [1.0f32; 128];
    ///
    /// let mut power = [[0.0f32; 64]; 3];
    /// let frames = stft.power(&signal, &mut power);
    /// assert_eq!(frames, 3);
    /// ```
    pub fn power<const F: usize>(&self, signal: &[f32], output: &mut [[f32; N]; F]) -> usize {
        let num_frames = self.num_frames(signal.len()).min(F);

        for (frame_idx, frame_output) in output.iter_mut().enumerate().take(num_frames) {
            let start = frame_idx * self.hop_size;
            self.compute_frame_power(&signal[start..start + N], frame_output);
        }

        num_frames
    }

    /// Computes the STFT magnitude (absolute value of coefficients).
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `output` - Output buffer for magnitude values.
    ///
    /// # Returns
    ///
    /// Number of frames computed.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// let stft = Stft::<64>::new(32, WindowType::Hann);
    /// let signal = [1.0f32; 128];
    ///
    /// let mut magnitude = [[0.0f32; 64]; 3];
    /// let frames = stft.magnitude(&signal, &mut magnitude);
    /// assert_eq!(frames, 3);
    /// ```
    pub fn magnitude<const F: usize>(&self, signal: &[f32], output: &mut [[f32; N]; F]) -> usize {
        let num_frames = self.num_frames(signal.len()).min(F);

        for (frame_idx, frame_output) in output.iter_mut().enumerate().take(num_frames) {
            let start = frame_idx * self.hop_size;
            self.compute_frame_magnitude(&signal[start..start + N], frame_output);
        }

        num_frames
    }

    /// Computes STFT for a single frame starting at a given offset.
    ///
    /// Useful for streaming applications where frames are processed one at a time.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal (must have at least `offset + N` samples)
    /// * `offset` - Starting sample index for this frame
    /// * `output` - Output buffer for complex coefficients
    ///
    /// # Panics
    ///
    /// Panics if `offset + N > signal.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, Complex, WindowType};
    ///
    /// let stft = Stft::<64>::new(32, WindowType::Hann);
    /// let signal = [1.0f32; 128];
    ///
    /// let mut frame = [Complex::new(0.0, 0.0); 64];
    /// stft.transform_frame(&signal, 32, &mut frame);
    /// ```
    pub fn transform_frame(&self, signal: &[f32], offset: usize, output: &mut [Complex; N]) {
        assert!(
            offset + N <= signal.len(),
            "frame extends beyond signal length"
        );
        self.compute_frame(&signal[offset..offset + N], output);
    }

    /// Returns the hop size.
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Returns the window type.
    #[inline]
    pub fn window_type(&self) -> WindowType {
        self.window_type
    }

    /// Returns the window size (same as `N`).
    #[inline]
    pub fn window_size(&self) -> usize {
        N
    }

    /// Sets the hop size.
    ///
    /// # Panics
    ///
    /// Panics if `hop_size` is 0.
    pub fn set_hop_size(&mut self, hop_size: usize) {
        assert!(hop_size > 0, "hop_size must be positive");
        self.hop_size = hop_size;
    }

    /// Sets the window type.
    pub fn set_window_type(&mut self, window_type: WindowType) {
        self.window_type = window_type;
    }

    /// Converts a frequency bin index to frequency in Hz.
    ///
    /// # Arguments
    ///
    /// * `bin` - Frequency bin index (0 to N-1)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Frequency in Hz corresponding to the bin.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// let stft = Stft::<256>::new(128, WindowType::Hann);
    ///
    /// // At 250 Hz sample rate, bin 1 = 250/256 ≈ 0.98 Hz
    /// let freq = stft.bin_to_frequency(1, 250.0);
    /// assert!((freq - 0.9765625).abs() < 0.001);
    /// ```
    #[inline]
    pub fn bin_to_frequency(&self, bin: usize, sample_rate: f32) -> f32 {
        bin as f32 * sample_rate / N as f32
    }

    /// Converts a frequency in Hz to the nearest bin index.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    ///
    /// Nearest frequency bin index.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::{Stft, WindowType};
    ///
    /// let stft = Stft::<256>::new(128, WindowType::Hann);
    ///
    /// // At 250 Hz sample rate, 10 Hz ≈ bin 10
    /// let bin = stft.frequency_to_bin(10.0, 250.0);
    /// assert_eq!(bin, 10);
    /// ```
    #[inline]
    pub fn frequency_to_bin(&self, frequency: f32, sample_rate: f32) -> usize {
        let bin = libm::roundf(frequency * N as f32 / sample_rate) as usize;
        bin.min(N - 1)
    }

    /// Computes a single frame with complex output.
    fn compute_frame(&self, segment: &[f32], output: &mut [Complex; N]) {
        // Apply window and convert to complex
        for (i, out) in output.iter_mut().enumerate() {
            let window_coeff = window_coefficient(self.window_type, i, N);
            *out = Complex::from_real(segment[i] * window_coeff);
        }

        // Compute FFT in-place
        self.fft.forward(output);
    }

    /// Computes a single frame with power output.
    fn compute_frame_power(&self, segment: &[f32], output: &mut [f32; N]) {
        // Temporary buffer for FFT
        let mut buffer = [Complex::new(0.0, 0.0); N];

        // Apply window and convert to complex
        for (i, buf) in buffer.iter_mut().enumerate() {
            let window_coeff = window_coefficient(self.window_type, i, N);
            *buf = Complex::from_real(segment[i] * window_coeff);
        }

        // Compute FFT
        self.fft.forward(&mut buffer);

        // Extract power
        for (i, out) in output.iter_mut().enumerate() {
            *out = buffer[i].magnitude_squared();
        }
    }

    /// Computes a single frame with magnitude output.
    fn compute_frame_magnitude(&self, segment: &[f32], output: &mut [f32; N]) {
        // Temporary buffer for FFT
        let mut buffer = [Complex::new(0.0, 0.0); N];

        // Apply window and convert to complex
        for (i, buf) in buffer.iter_mut().enumerate() {
            let window_coeff = window_coefficient(self.window_type, i, N);
            *buf = Complex::from_real(segment[i] * window_coeff);
        }

        // Compute FFT
        self.fft.forward(&mut buffer);

        // Extract magnitude
        for (i, out) in output.iter_mut().enumerate() {
            *out = buffer[i].magnitude();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn test_stft_new() {
        let stft = Stft::<256>::new(128, WindowType::Hann);
        assert_eq!(stft.hop_size(), 128);
        assert_eq!(stft.window_type(), WindowType::Hann);
        assert_eq!(stft.window_size(), 256);
    }

    #[test]
    fn test_stft_default_overlap() {
        let stft = Stft::<256>::default_overlap();
        assert_eq!(stft.hop_size(), 128);
        assert_eq!(stft.window_type(), WindowType::Hann);
    }

    #[test]
    #[should_panic(expected = "hop_size must be positive")]
    fn test_stft_zero_hop_panics() {
        let _ = Stft::<256>::new(0, WindowType::Hann);
    }

    #[test]
    fn test_num_frames() {
        let stft = Stft::<256>::new(64, WindowType::Hann);

        // (1024 - 256) / 64 + 1 = 13
        assert_eq!(stft.num_frames(1024), 13);

        // (512 - 256) / 64 + 1 = 5
        assert_eq!(stft.num_frames(512), 5);

        // Signal shorter than window
        assert_eq!(stft.num_frames(100), 0);

        // Exact window size = 1 frame
        assert_eq!(stft.num_frames(256), 1);
    }

    #[test]
    fn test_num_frames_50_percent_overlap() {
        let stft = Stft::<256>::new(128, WindowType::Hann);

        // (512 - 256) / 128 + 1 = 3
        assert_eq!(stft.num_frames(512), 3);
    }

    #[test]
    fn test_transform_dc_signal() {
        let stft = Stft::<64>::new(32, WindowType::Rectangular);
        let signal = [1.0f32; 128];

        let mut output = [[Complex::new(0.0, 0.0); 64]; 3];
        let frames = stft.transform(&signal, &mut output);

        assert_eq!(frames, 3);

        // DC component should be dominant for constant signal
        for frame in &output[..frames] {
            // DC bin (bin 0) should have significant energy
            assert!(frame[0].magnitude() > 0.1);
            // Higher frequency bins should be near zero
            assert!(frame[32].magnitude() < 0.01);
        }
    }

    #[test]
    fn test_transform_sine_wave() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let sample_rate = 256.0;
        let freq = 32.0; // Should appear at bin 8

        // Generate sine wave
        let mut signal = [0.0f32; 128];
        for (i, s) in signal.iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI * freq * i as f32 / sample_rate);
        }

        let mut power = [[0.0f32; 64]; 3];
        let frames = stft.power(&signal, &mut power);

        assert_eq!(frames, 3);

        // Find bin with maximum power in each frame (only positive frequencies 0..N/2)
        for frame_power in &power[..frames] {
            let max_bin = frame_power[..32] // Only search positive frequencies
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            // Should be near bin 8 (32 Hz at 256 Hz sample rate with 64-point FFT)
            assert!(
                (max_bin as i32 - 8).abs() <= 1,
                "Expected peak near bin 8, got bin {}",
                max_bin
            );
        }
    }

    #[test]
    fn test_power_vs_transform() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let signal = [0.5f32; 128];

        let mut complex_output = [[Complex::new(0.0, 0.0); 64]; 3];
        let mut power_output = [[0.0f32; 64]; 3];

        stft.transform(&signal, &mut complex_output);
        stft.power(&signal, &mut power_output);

        // Power should equal magnitude squared
        for frame in 0..3 {
            for bin in 0..64 {
                let expected = complex_output[frame][bin].magnitude_squared();
                assert!(
                    (power_output[frame][bin] - expected).abs() < 1e-6,
                    "Mismatch at frame {}, bin {}",
                    frame,
                    bin
                );
            }
        }
    }

    #[test]
    fn test_magnitude_vs_transform() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let signal = [0.5f32; 128];

        let mut complex_output = [[Complex::new(0.0, 0.0); 64]; 3];
        let mut magnitude_output = [[0.0f32; 64]; 3];

        stft.transform(&signal, &mut complex_output);
        stft.magnitude(&signal, &mut magnitude_output);

        // Magnitude should equal absolute value
        for frame in 0..3 {
            for bin in 0..64 {
                let expected = complex_output[frame][bin].magnitude();
                assert!(
                    (magnitude_output[frame][bin] - expected).abs() < 1e-6,
                    "Mismatch at frame {}, bin {}",
                    frame,
                    bin
                );
            }
        }
    }

    #[test]
    fn test_transform_frame() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let signal = [0.5f32; 128];

        // Compute using transform
        let mut full_output = [[Complex::new(0.0, 0.0); 64]; 3];
        stft.transform(&signal, &mut full_output);

        // Compute frame 1 directly
        let mut frame_output = [Complex::new(0.0, 0.0); 64];
        stft.transform_frame(&signal, 32, &mut frame_output);

        // Should match
        for bin in 0..64 {
            assert!(
                (full_output[1][bin].re - frame_output[bin].re).abs() < 1e-6,
                "Real mismatch at bin {}",
                bin
            );
            assert!(
                (full_output[1][bin].im - frame_output[bin].im).abs() < 1e-6,
                "Imag mismatch at bin {}",
                bin
            );
        }
    }

    #[test]
    fn test_bin_frequency_conversion() {
        let stft = Stft::<256>::new(128, WindowType::Hann);
        let sample_rate = 250.0;

        // Bin 0 = 0 Hz (DC)
        assert!((stft.bin_to_frequency(0, sample_rate) - 0.0).abs() < 0.001);

        // Bin N/2 = Nyquist
        let nyquist = stft.bin_to_frequency(128, sample_rate);
        assert!((nyquist - 125.0).abs() < 0.001);

        // Round-trip
        let freq = 10.0;
        let bin = stft.frequency_to_bin(freq, sample_rate);
        let freq_back = stft.bin_to_frequency(bin, sample_rate);
        assert!((freq - freq_back).abs() < 1.0);
    }

    #[test]
    fn test_setters() {
        let mut stft = Stft::<256>::new(128, WindowType::Hann);

        stft.set_hop_size(64);
        assert_eq!(stft.hop_size(), 64);

        stft.set_window_type(WindowType::Hamming);
        assert_eq!(stft.window_type(), WindowType::Hamming);
    }

    #[test]
    fn test_output_buffer_larger_than_needed() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let signal = [0.5f32; 128]; // 3 frames

        // Buffer has room for 10 frames
        let mut output = [[0.0f32; 64]; 10];
        let frames = stft.power(&signal, &mut output);

        // Should only compute 3 frames
        assert_eq!(frames, 3);
    }

    #[test]
    fn test_output_buffer_smaller_than_needed() {
        let stft = Stft::<64>::new(32, WindowType::Hann);
        let signal = [0.5f32; 256]; // Would produce 7 frames

        // Buffer only has room for 3 frames
        let mut output = [[0.0f32; 64]; 3];
        let frames = stft.power(&signal, &mut output);

        // Should only compute 3 frames (limited by buffer)
        assert_eq!(frames, 3);
    }

    #[test]
    fn test_different_window_types() {
        let signal = [1.0f32; 128];

        for window_type in [
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::BlackmanHarris,
        ] {
            let stft = Stft::<64>::new(32, window_type);
            let mut output = [[0.0f32; 64]; 3];
            let frames = stft.power(&signal, &mut output);
            assert_eq!(frames, 3);

            // All frames should have non-zero energy
            for frame_power in &output[..frames] {
                let total: f32 = frame_power.iter().sum();
                assert!(total > 0.0, "Window {:?} produced zero energy", window_type);
            }
        }
    }
}
