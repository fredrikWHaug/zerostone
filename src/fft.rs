//! Fast Fourier Transform for spectral analysis.
//!
//! Provides in-place radix-2 FFT for power-of-two sizes, enabling
//! frequency-domain analysis of neural signals. Essential for band power
//! computation in motor imagery BCIs and spectral feature extraction.

use core::f32::consts::PI;

/// A complex number with 32-bit floating point components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real component
    pub re: f32,
    /// Imaginary component
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Create a complex number from a real value (imaginary part = 0).
    pub const fn from_real(re: f32) -> Self {
        Self { re, im: 0.0 }
    }

    /// Compute the magnitude (absolute value).
    pub fn magnitude(self) -> f32 {
        libm::sqrtf(self.re * self.re + self.im * self.im)
    }

    /// Compute the squared magnitude (power).
    pub fn magnitude_squared(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Complex multiplication.
    pub fn cmul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex addition.
    pub fn cadd(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction.
    pub fn csub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// Fast Fourier Transform processor.
///
/// Implements in-place radix-2 Cooley-Tukey FFT for power-of-two sizes.
/// Suitable for spectral analysis of neural signals in BCI applications.
///
/// # Constraints
///
/// - N must be a power of 2
/// - Operates in-place with O(N log N) complexity
/// - No heap allocation
///
/// # Example
///
/// ```
/// use zerostone::{Fft, Complex};
///
/// // Create FFT processor for 256-point transforms
/// let fft = Fft::<256>::new();
///
/// // Prepare input (e.g., 256 samples of neural signal)
/// let mut signal: [Complex; 256] = [Complex::new(0.0, 0.0); 256];
/// for i in 0..256 {
///     signal[i] = Complex::from_real((i as f32 * 0.1).sin());
/// }
///
/// // Compute FFT in-place
/// fft.forward(&mut signal);
///
/// // Access frequency bins (0 to N-1)
/// let dc_component = signal[0];
/// let first_bin = signal[1];
/// ```
pub struct Fft<const N: usize> {
    // Stateless - all computation is in-place
}

impl<const N: usize> Fft<N> {
    /// Create a new FFT processor.
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2.
    pub const fn new() -> Self {
        assert!(N > 0 && (N & (N - 1)) == 0, "FFT size must be power of 2");
        Self {}
    }

    /// Compute forward FFT in-place.
    ///
    /// Transforms time-domain signal to frequency-domain.
    /// After transformation, `data[k]` contains the complex amplitude
    /// at frequency `k * sample_rate / N`.
    ///
    /// # Arguments
    ///
    /// * `data` - Complex array to transform in-place
    pub fn forward(&self, data: &mut [Complex; N]) {
        // Bit-reversal permutation
        self.bit_reverse(data);

        // Cooley-Tukey decimation-in-time
        let mut size = 2;
        while size <= N {
            let half_size = size / 2;
            let angle = -2.0 * PI / size as f32;

            for start in (0..N).step_by(size) {
                for k in 0..half_size {
                    let omega =
                        Complex::new(libm::cosf(angle * k as f32), libm::sinf(angle * k as f32));

                    let even_idx = start + k;
                    let odd_idx = start + k + half_size;

                    let even = data[even_idx];
                    let odd = data[odd_idx].cmul(omega);

                    data[even_idx] = even.cadd(odd);
                    data[odd_idx] = even.csub(odd);
                }
            }

            size *= 2;
        }
    }

    /// Compute inverse FFT in-place.
    ///
    /// Transforms frequency-domain back to time-domain.
    /// Result is scaled by 1/N.
    pub fn inverse(&self, data: &mut [Complex; N]) {
        // Conjugate input
        for x in data.iter_mut() {
            *x = x.conj();
        }

        // Forward FFT
        self.forward(data);

        // Conjugate output and scale
        let scale = 1.0 / N as f32;
        for x in data.iter_mut() {
            *x = Complex::new(x.re * scale, -x.im * scale);
        }
    }

    /// Compute power spectrum from real-valued signal.
    ///
    /// Returns one-sided power spectrum with N/2 + 1 bins.
    /// Useful for spectral analysis where input is real (e.g., EEG/LFP).
    ///
    /// # Arguments
    ///
    /// * `real_signal` - Real-valued input signal
    /// * `output` - Output power spectrum (length N/2 + 1)
    pub fn power_spectrum(&self, real_signal: &[f32; N], output: &mut [f32]) {
        assert!(output.len() > N / 2, "Output buffer too small");

        // Convert to complex
        let mut data = [Complex::new(0.0, 0.0); N];
        for (i, &x) in real_signal.iter().enumerate() {
            data[i] = Complex::from_real(x);
        }

        // Compute FFT
        self.forward(&mut data);

        // Compute one-sided power spectrum
        // DC component
        output[0] = data[0].magnitude_squared();

        // Positive frequencies (bins 1 to N/2-1)
        for k in 1..N / 2 {
            // Multiply by 2 since we're combining positive and negative frequencies
            output[k] = 2.0 * data[k].magnitude_squared();
        }

        // Nyquist frequency (if N is even)
        output[N / 2] = data[N / 2].magnitude_squared();
    }

    /// Perform bit-reversal permutation.
    fn bit_reverse(&self, data: &mut [Complex; N]) {
        let bits = N.trailing_zeros() as usize;

        for i in 0..N {
            let j = Self::reverse_bits(i, bits);
            if j > i {
                data.swap(i, j);
            }
        }
    }

    /// Reverse the bits of a number.
    fn reverse_bits(mut x: usize, bits: usize) -> usize {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }
}

impl<const N: usize> Default for Fft<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for computing band power in specific frequency ranges.
///
/// Integrates power spectrum over a frequency band, useful for
/// extracting alpha (8-12 Hz), beta (13-30 Hz), gamma (30-100 Hz) features.
///
/// # Example
///
/// ```
/// use zerostone::{Fft, BandPower};
///
/// let fft = Fft::<256>::new();
/// let mut band_power = BandPower::new(250.0); // 250 Hz sample rate
///
/// // Get alpha band power (8-12 Hz)
/// let signal = [0.0f32; 256]; // Your EEG data here
/// let alpha_power = band_power.compute::<256>(&fft, &signal, 8.0, 12.0);
/// ```
pub struct BandPower {
    sample_rate: f32,
}

impl BandPower {
    /// Create a new band power calculator.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Compute power in a frequency band.
    ///
    /// # Arguments
    ///
    /// * `fft` - FFT processor
    /// * `signal` - Real-valued input signal
    /// * `low_freq` - Lower frequency bound in Hz
    /// * `high_freq` - Upper frequency bound in Hz
    ///
    /// # Returns
    ///
    /// Integrated power in the specified band
    pub fn compute<const N: usize>(
        &mut self,
        fft: &Fft<N>,
        signal: &[f32; N],
        low_freq: f32,
        high_freq: f32,
    ) -> f32 {
        // Compute power spectrum (allocate N elements, only N/2+1 will be used)
        let mut power_spec = [0.0f32; N];
        fft.power_spectrum(signal, &mut power_spec[..]);

        // Convert frequencies to bin indices
        let freq_resolution = self.sample_rate / N as f32;
        let low_bin = (low_freq / freq_resolution) as usize;
        let high_bin = ((high_freq / freq_resolution) as usize).min(N / 2);

        // Integrate power over band
        power_spec[low_bin..=high_bin].iter().sum()
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, 2.0);

        let sum = a.cadd(b);
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);

        let diff = a.csub(b);
        assert_eq!(diff.re, 2.0);
        assert_eq!(diff.im, 2.0);

        let prod = a.cmul(b);
        assert_eq!(prod.re, -5.0); // 3*1 - 4*2
        assert_eq!(prod.im, 10.0); // 3*2 + 4*1
    }

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < 1e-6);
        assert!((c.magnitude_squared() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_fft_dc_signal() {
        let fft = Fft::<8>::new();
        let mut data = [Complex::from_real(1.0); 8];

        fft.forward(&mut data);

        // DC component should be N
        assert!((data[0].re - 8.0).abs() < 1e-5);
        assert!(data[0].im.abs() < 1e-5);

        // All other bins should be ~0
        for val in &data[1..8] {
            assert!(val.magnitude() < 1e-5);
        }
    }

    #[test]
    fn test_fft_sine_wave() {
        let fft = Fft::<16>::new();
        let mut data = [Complex::new(0.0, 0.0); 16];

        // Create a sine wave at bin 2 (2 cycles over 16 samples)
        for (i, val) in data.iter_mut().enumerate() {
            let angle = 2.0 * PI * 2.0 * i as f32 / 16.0;
            *val = Complex::from_real(libm::sinf(angle));
        }

        fft.forward(&mut data);

        // Energy should be concentrated in bin 2 and bin 14 (N-2, negative frequency)
        assert!(data[2].magnitude() > 5.0);
        assert!(data[14].magnitude() > 5.0);

        // Other bins should have minimal energy
        for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15] {
            assert!(data[i].magnitude() < 0.1);
        }
    }

    #[test]
    fn test_fft_inverse() {
        let fft = Fft::<8>::new();
        let original = [
            Complex::from_real(1.0),
            Complex::from_real(2.0),
            Complex::from_real(3.0),
            Complex::from_real(4.0),
            Complex::from_real(5.0),
            Complex::from_real(6.0),
            Complex::from_real(7.0),
            Complex::from_real(8.0),
        ];

        let mut data = original;
        fft.forward(&mut data);
        fft.inverse(&mut data);

        // Should recover original signal
        for i in 0..8 {
            assert!((data[i].re - original[i].re).abs() < 1e-5);
            assert!(data[i].im.abs() < 1e-5);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let fft = Fft::<16>::new();

        // DC signal
        let signal = [1.0f32; 16];
        let mut power = [0.0f32; 9]; // N/2 + 1

        fft.power_spectrum(&signal, &mut power);

        // All power in DC bin
        assert!(power[0] > 200.0); // 16^2
        for val in &power[1..9] {
            assert!(*val < 1e-5);
        }
    }

    #[test]
    fn test_band_power_alpha() {
        let fft = Fft::<256>::new();
        let mut bp = BandPower::new(250.0);

        // Create a 10 Hz sine wave (in alpha band)
        let mut signal = [0.0f32; 256];
        for (i, val) in signal.iter_mut().enumerate() {
            let t = i as f32 / 250.0;
            *val = libm::sinf(2.0 * PI * 10.0 * t);
        }

        // Alpha band (8-12 Hz) should have significant power
        let alpha_power = bp.compute(&fft, &signal, 8.0, 12.0);
        assert!(alpha_power > 1000.0);

        // Beta band (13-30 Hz) should have much less power (spectral leakage present)
        let beta_power = bp.compute(&fft, &signal, 13.0, 30.0);
        assert!(alpha_power > beta_power * 5.0); // At least 5x more in alpha band
    }

    #[test]
    fn test_bit_reversal() {
        let fft = Fft::<8>::new();
        let mut data = [
            Complex::from_real(0.0),
            Complex::from_real(1.0),
            Complex::from_real(2.0),
            Complex::from_real(3.0),
            Complex::from_real(4.0),
            Complex::from_real(5.0),
            Complex::from_real(6.0),
            Complex::from_real(7.0),
        ];

        fft.bit_reverse(&mut data);

        // Expected order after bit-reversal for N=8
        // Original: 0,1,2,3,4,5,6,7
        // Reversed: 0,4,2,6,1,5,3,7
        assert_eq!(data[0].re, 0.0);
        assert_eq!(data[1].re, 4.0);
        assert_eq!(data[2].re, 2.0);
        assert_eq!(data[3].re, 6.0);
        assert_eq!(data[4].re, 1.0);
        assert_eq!(data[5].re, 5.0);
        assert_eq!(data[6].re, 3.0);
        assert_eq!(data[7].re, 7.0);
    }

    #[test]
    #[should_panic(expected = "FFT size must be power of 2")]
    fn test_non_power_of_two_panics() {
        let _ = Fft::<100>::new();
    }
}
