//! Python bindings for spectral analysis primitives.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    Complex, Fft as ZsFft, FrequencyBand as ZsFrequencyBand, MultiBandPower as ZsMultiBandPower,
    Stft as ZsStft, WindowType as ZsWindowType,
};

// ============================================================================
// Fft
// ============================================================================

/// Internal enum for handling different FFT sizes.
enum FftInner {
    Size64(ZsFft<64>),
    Size128(ZsFft<128>),
    Size256(ZsFft<256>),
    Size512(ZsFft<512>),
    Size1024(ZsFft<1024>),
    Size2048(ZsFft<2048>),
}

/// Fast Fourier Transform for spectral analysis.
///
/// Implements radix-2 FFT for power-of-two sizes. Essential for frequency-domain
/// analysis of neural signals in BCI applications.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create FFT processor for 256-point transforms
/// fft = npy.Fft(size=256)
///
/// # Compute power spectrum
/// signal = np.random.randn(256).astype(np.float32)
/// power = fft.power_spectrum(signal)  # Shape: (129,) for one-sided spectrum
/// ```
#[pyclass]
pub struct Fft {
    inner: FftInner,
    size: usize,
}

#[pymethods]
impl Fft {
    /// Create a new FFT processor.
    ///
    /// Args:
    ///     size (int): FFT size. Must be 64, 128, 256, 512, 1024, or 2048.
    ///
    /// Returns:
    ///     Fft: A new FFT processor instance.
    ///
    /// Example:
    ///     >>> fft = Fft(size=256)
    #[new]
    fn new(size: usize) -> PyResult<Self> {
        let inner = match size {
            64 => FftInner::Size64(ZsFft::new()),
            128 => FftInner::Size128(ZsFft::new()),
            256 => FftInner::Size256(ZsFft::new()),
            512 => FftInner::Size512(ZsFft::new()),
            1024 => FftInner::Size1024(ZsFft::new()),
            2048 => FftInner::Size2048(ZsFft::new()),
            _ => {
                return Err(PyValueError::new_err(
                    "size must be 64, 128, 256, 512, 1024, or 2048",
                ))
            }
        };

        Ok(Self { inner, size })
    }

    /// Compute forward FFT and return complex result.
    ///
    /// Args:
    ///     signal (np.ndarray): Real-valued input signal as 1D float32 array.
    ///
    /// Returns:
    ///     tuple[np.ndarray, np.ndarray]: (real, imaginary) components as 1D float32 arrays.
    ///
    /// Example:
    ///     >>> real, imag = fft.forward(signal)
    fn forward<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let input_slice = signal.as_slice()?;
        if input_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal length {} must match FFT size {}",
                input_slice.len(),
                self.size
            )));
        }

        let mut real = vec![0.0f32; self.size];
        let mut imag = vec![0.0f32; self.size];

        macro_rules! compute_fft {
            ($fft:expr, $N:expr) => {{
                let mut data: [Complex; $N] =
                    core::array::from_fn(|i| Complex::from_real(input_slice[i]));
                $fft.forward(&mut data);
                for (i, c) in data.iter().enumerate() {
                    real[i] = c.re;
                    imag[i] = c.im;
                }
            }};
        }

        match &self.inner {
            FftInner::Size64(fft) => compute_fft!(fft, 64),
            FftInner::Size128(fft) => compute_fft!(fft, 128),
            FftInner::Size256(fft) => compute_fft!(fft, 256),
            FftInner::Size512(fft) => compute_fft!(fft, 512),
            FftInner::Size1024(fft) => compute_fft!(fft, 1024),
            FftInner::Size2048(fft) => compute_fft!(fft, 2048),
        }

        Ok((PyArray1::from_vec(py, real), PyArray1::from_vec(py, imag)))
    }

    /// Compute one-sided power spectrum from real-valued signal.
    ///
    /// Args:
    ///     signal (np.ndarray): Real-valued input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Power spectrum with size/2 + 1 bins.
    ///
    /// Example:
    ///     >>> power = fft.power_spectrum(signal)
    ///     >>> print(f"DC: {power[0]}, Nyquist: {power[-1]}")
    fn power_spectrum<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = signal.as_slice()?;
        if input_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal length {} must match FFT size {}",
                input_slice.len(),
                self.size
            )));
        }

        let out_size = self.size / 2 + 1;
        let mut output = vec![0.0f32; out_size];

        macro_rules! compute_power {
            ($fft:expr, $N:expr) => {{
                let mut signal_arr = [0.0f32; $N];
                for (i, &v) in input_slice.iter().enumerate() {
                    signal_arr[i] = v;
                }
                let mut power_arr = [0.0f32; $N];
                $fft.power_spectrum(&signal_arr, &mut power_arr);
                for (i, &v) in power_arr[..out_size].iter().enumerate() {
                    output[i] = v;
                }
            }};
        }

        match &self.inner {
            FftInner::Size64(fft) => compute_power!(fft, 64),
            FftInner::Size128(fft) => compute_power!(fft, 128),
            FftInner::Size256(fft) => compute_power!(fft, 256),
            FftInner::Size512(fft) => compute_power!(fft, 512),
            FftInner::Size1024(fft) => compute_power!(fft, 1024),
            FftInner::Size2048(fft) => compute_power!(fft, 2048),
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Get the FFT size.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    fn __repr__(&self) -> String {
        format!("Fft(size={})", self.size)
    }
}

// ============================================================================
// Stft
// ============================================================================

/// Internal enum for handling different STFT sizes.
enum StftInner {
    Size64(ZsStft<64>),
    Size128(ZsStft<128>),
    Size256(ZsStft<256>),
    Size512(ZsStft<512>),
    Size1024(ZsStft<1024>),
    Size2048(ZsStft<2048>),
}

/// Short-Time Fourier Transform for time-frequency analysis.
///
/// Computes the STFT of a signal using a sliding window approach,
/// producing a time-frequency representation (spectrogram).
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create STFT with 256-point window, 64-sample hop
/// stft = npy.Stft(size=256, hop_size=64)
///
/// # Compute power spectrogram
/// signal = np.random.randn(1024).astype(np.float32)
/// power = stft.power(signal)  # Shape: (num_frames, 256)
/// ```
#[pyclass]
pub struct Stft {
    inner: StftInner,
    size: usize,
    hop_size: usize,
}

fn parse_window_type(window: &str) -> PyResult<ZsWindowType> {
    match window.to_lowercase().as_str() {
        "rectangular" | "rect" => Ok(ZsWindowType::Rectangular),
        "hann" | "hanning" => Ok(ZsWindowType::Hann),
        "hamming" => Ok(ZsWindowType::Hamming),
        "blackman" => Ok(ZsWindowType::Blackman),
        "blackman_harris" | "blackmanharris" => Ok(ZsWindowType::BlackmanHarris),
        _ => Err(PyValueError::new_err(
            "window must be 'rectangular', 'hann', 'hamming', 'blackman', or 'blackman_harris'",
        )),
    }
}

#[pymethods]
impl Stft {
    /// Create a new STFT processor.
    ///
    /// Args:
    ///     size (int): Window size. Must be 64, 128, 256, 512, 1024, or 2048.
    ///     hop_size (int): Samples between consecutive frames.
    ///     window (str): Window type - 'rectangular', 'hann', 'hamming', 'blackman', 'blackman_harris'.
    ///
    /// Returns:
    ///     Stft: A new STFT processor instance.
    ///
    /// Example:
    ///     >>> stft = Stft(size=256, hop_size=64, window='hann')
    #[new]
    #[pyo3(signature = (size, hop_size, window = "hann"))]
    fn new(size: usize, hop_size: usize, window: &str) -> PyResult<Self> {
        if hop_size == 0 {
            return Err(PyValueError::new_err("hop_size must be positive"));
        }

        let window_type = parse_window_type(window)?;

        let inner = match size {
            64 => StftInner::Size64(ZsStft::new(hop_size, window_type)),
            128 => StftInner::Size128(ZsStft::new(hop_size, window_type)),
            256 => StftInner::Size256(ZsStft::new(hop_size, window_type)),
            512 => StftInner::Size512(ZsStft::new(hop_size, window_type)),
            1024 => StftInner::Size1024(ZsStft::new(hop_size, window_type)),
            2048 => StftInner::Size2048(ZsStft::new(hop_size, window_type)),
            _ => {
                return Err(PyValueError::new_err(
                    "size must be 64, 128, 256, 512, 1024, or 2048",
                ))
            }
        };

        Ok(Self {
            inner,
            size,
            hop_size,
        })
    }

    /// Compute STFT power (magnitude squared) spectrogram.
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Power spectrogram with shape (num_frames, size).
    ///
    /// Example:
    ///     >>> power = stft.power(signal)
    fn power<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let input_slice = signal.as_slice()?;
        let signal_len = input_slice.len();

        // Calculate number of frames
        let num_frames = if signal_len < self.size {
            0
        } else {
            (signal_len - self.size) / self.hop_size + 1
        };

        if num_frames == 0 {
            return Err(PyValueError::new_err(format!(
                "Signal length {} is shorter than window size {}",
                signal_len, self.size
            )));
        }

        let mut output = vec![0.0f32; num_frames * self.size];

        macro_rules! compute_power {
            ($stft:expr, $N:expr) => {{
                // We need to process frame by frame since the Rust API expects fixed-size output
                for frame_idx in 0..num_frames {
                    let start = frame_idx * self.hop_size;
                    let mut frame_output = [0.0f32; $N];
                    let mut temp_buffer = [Complex::new(0.0, 0.0); $N];

                    // Apply window and convert to complex
                    for (i, &sample) in input_slice[start..start + $N].iter().enumerate() {
                        let window_coeff =
                            zerostone::window_coefficient($stft.window_type(), i, $N);
                        temp_buffer[i] = Complex::from_real(sample * window_coeff);
                    }

                    // Compute FFT
                    let fft = ZsFft::<$N>::new();
                    fft.forward(&mut temp_buffer);

                    // Extract power
                    for (i, c) in temp_buffer.iter().enumerate() {
                        frame_output[i] = c.magnitude_squared();
                    }

                    // Copy to output
                    for (i, &v) in frame_output.iter().enumerate() {
                        output[frame_idx * self.size + i] = v;
                    }
                }
            }};
        }

        match &self.inner {
            StftInner::Size64(stft) => compute_power!(stft, 64),
            StftInner::Size128(stft) => compute_power!(stft, 128),
            StftInner::Size256(stft) => compute_power!(stft, 256),
            StftInner::Size512(stft) => compute_power!(stft, 512),
            StftInner::Size1024(stft) => compute_power!(stft, 1024),
            StftInner::Size2048(stft) => compute_power!(stft, 2048),
        }

        let output_array = Array2::from_shape_vec((num_frames, self.size), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Compute STFT and return complex result.
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     tuple[np.ndarray, np.ndarray]: (real, imag) with shape (num_frames, size) each.
    ///
    /// Example:
    ///     >>> real, imag = stft.transform(signal)
    fn transform<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let input_slice = signal.as_slice()?;
        let signal_len = input_slice.len();

        let num_frames = if signal_len < self.size {
            0
        } else {
            (signal_len - self.size) / self.hop_size + 1
        };

        if num_frames == 0 {
            return Err(PyValueError::new_err(format!(
                "Signal length {} is shorter than window size {}",
                signal_len, self.size
            )));
        }

        let mut real_out = vec![0.0f32; num_frames * self.size];
        let mut imag_out = vec![0.0f32; num_frames * self.size];

        macro_rules! compute_transform {
            ($stft:expr, $N:expr) => {{
                for frame_idx in 0..num_frames {
                    let start = frame_idx * self.hop_size;
                    let mut temp_buffer = [Complex::new(0.0, 0.0); $N];

                    for (i, &sample) in input_slice[start..start + $N].iter().enumerate() {
                        let window_coeff =
                            zerostone::window_coefficient($stft.window_type(), i, $N);
                        temp_buffer[i] = Complex::from_real(sample * window_coeff);
                    }

                    let fft = ZsFft::<$N>::new();
                    fft.forward(&mut temp_buffer);

                    for (i, c) in temp_buffer.iter().enumerate() {
                        real_out[frame_idx * self.size + i] = c.re;
                        imag_out[frame_idx * self.size + i] = c.im;
                    }
                }
            }};
        }

        match &self.inner {
            StftInner::Size64(stft) => compute_transform!(stft, 64),
            StftInner::Size128(stft) => compute_transform!(stft, 128),
            StftInner::Size256(stft) => compute_transform!(stft, 256),
            StftInner::Size512(stft) => compute_transform!(stft, 512),
            StftInner::Size1024(stft) => compute_transform!(stft, 1024),
            StftInner::Size2048(stft) => compute_transform!(stft, 2048),
        }

        let real_array = Array2::from_shape_vec((num_frames, self.size), real_out)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        let imag_array = Array2::from_shape_vec((num_frames, self.size), imag_out)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;

        Ok((
            PyArray2::from_owned_array(py, real_array),
            PyArray2::from_owned_array(py, imag_array),
        ))
    }

    /// Calculate number of frames for a given signal length.
    ///
    /// Args:
    ///     signal_len (int): Length of the input signal.
    ///
    /// Returns:
    ///     int: Number of STFT frames.
    fn num_frames(&self, signal_len: usize) -> usize {
        if signal_len < self.size {
            0
        } else {
            (signal_len - self.size) / self.hop_size + 1
        }
    }

    /// Get the window size.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    /// Get the hop size.
    #[getter]
    fn hop_size(&self) -> usize {
        self.hop_size
    }

    fn __repr__(&self) -> String {
        format!("Stft(size={}, hop_size={})", self.size, self.hop_size)
    }
}

// ============================================================================
// MultiBandPower
// ============================================================================

/// Internal enum for handling different FFT size / channel combinations.
/// We use common combinations for BCI applications.
enum MultiBandPowerInner {
    // 256-point FFT (common for BCI)
    N256C1(ZsMultiBandPower<256, 1>),
    N256C4(ZsMultiBandPower<256, 4>),
    N256C8(ZsMultiBandPower<256, 8>),
    N256C16(ZsMultiBandPower<256, 16>),
    N256C32(ZsMultiBandPower<256, 32>),
    N256C64(ZsMultiBandPower<256, 64>),
    // 512-point FFT
    N512C1(ZsMultiBandPower<512, 1>),
    N512C4(ZsMultiBandPower<512, 4>),
    N512C8(ZsMultiBandPower<512, 8>),
    N512C16(ZsMultiBandPower<512, 16>),
    N512C32(ZsMultiBandPower<512, 32>),
    N512C64(ZsMultiBandPower<512, 64>),
    // 1024-point FFT
    N1024C1(ZsMultiBandPower<1024, 1>),
    N1024C4(ZsMultiBandPower<1024, 4>),
    N1024C8(ZsMultiBandPower<1024, 8>),
    N1024C16(ZsMultiBandPower<1024, 16>),
    N1024C32(ZsMultiBandPower<1024, 32>),
    N1024C64(ZsMultiBandPower<1024, 64>),
}

/// Multi-channel band power extraction with proper PSD normalization.
///
/// Computes power spectral density for multiple channels and efficiently
/// extracts band power in arbitrary frequency ranges. Provides standard
/// EEG frequency bands (delta, theta, alpha, beta, gamma).
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create band power extractor: 256-point FFT, 8 channels, 250 Hz sample rate
/// bp = npy.MultiBandPower(fft_size=256, channels=8, sample_rate=250.0)
///
/// # Compute PSD for all channels (shape: channels x fft_size)
/// signals = np.random.randn(8, 256).astype(np.float32)
/// bp.compute(signals)
///
/// # Query band power (returns array of length channels)
/// alpha = bp.band_power(8.0, 12.0)  # Alpha band: 8-12 Hz
/// beta = bp.band_power(12.0, 30.0)  # Beta band: 12-30 Hz
/// ```
#[pyclass]
pub struct MultiBandPower {
    inner: MultiBandPowerInner,
    fft_size: usize,
    channels: usize,
    sample_rate: f32,
}

#[pymethods]
impl MultiBandPower {
    /// Create a new multi-channel band power extractor.
    ///
    /// Args:
    ///     fft_size (int): FFT size. Must be 256, 512, or 1024.
    ///     channels (int): Number of channels. Must be 1, 4, 8, 16, 32, or 64.
    ///     sample_rate (float): Sample rate in Hz.
    ///
    /// Returns:
    ///     MultiBandPower: A new band power extractor.
    ///
    /// Example:
    ///     >>> bp = MultiBandPower(fft_size=256, channels=8, sample_rate=250.0)
    #[new]
    fn new(fft_size: usize, channels: usize, sample_rate: f32) -> PyResult<Self> {
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }

        let inner = match (fft_size, channels) {
            (256, 1) => MultiBandPowerInner::N256C1(ZsMultiBandPower::new(sample_rate)),
            (256, 4) => MultiBandPowerInner::N256C4(ZsMultiBandPower::new(sample_rate)),
            (256, 8) => MultiBandPowerInner::N256C8(ZsMultiBandPower::new(sample_rate)),
            (256, 16) => MultiBandPowerInner::N256C16(ZsMultiBandPower::new(sample_rate)),
            (256, 32) => MultiBandPowerInner::N256C32(ZsMultiBandPower::new(sample_rate)),
            (256, 64) => MultiBandPowerInner::N256C64(ZsMultiBandPower::new(sample_rate)),
            (512, 1) => MultiBandPowerInner::N512C1(ZsMultiBandPower::new(sample_rate)),
            (512, 4) => MultiBandPowerInner::N512C4(ZsMultiBandPower::new(sample_rate)),
            (512, 8) => MultiBandPowerInner::N512C8(ZsMultiBandPower::new(sample_rate)),
            (512, 16) => MultiBandPowerInner::N512C16(ZsMultiBandPower::new(sample_rate)),
            (512, 32) => MultiBandPowerInner::N512C32(ZsMultiBandPower::new(sample_rate)),
            (512, 64) => MultiBandPowerInner::N512C64(ZsMultiBandPower::new(sample_rate)),
            (1024, 1) => MultiBandPowerInner::N1024C1(ZsMultiBandPower::new(sample_rate)),
            (1024, 4) => MultiBandPowerInner::N1024C4(ZsMultiBandPower::new(sample_rate)),
            (1024, 8) => MultiBandPowerInner::N1024C8(ZsMultiBandPower::new(sample_rate)),
            (1024, 16) => MultiBandPowerInner::N1024C16(ZsMultiBandPower::new(sample_rate)),
            (1024, 32) => MultiBandPowerInner::N1024C32(ZsMultiBandPower::new(sample_rate)),
            (1024, 64) => MultiBandPowerInner::N1024C64(ZsMultiBandPower::new(sample_rate)),
            _ => return Err(PyValueError::new_err(
                "fft_size must be 256, 512, or 1024 and channels must be 1, 4, 8, 16, 32, or 64",
            )),
        };

        Ok(Self {
            inner,
            fft_size,
            channels,
            sample_rate,
        })
    }

    /// Compute PSD for all channels.
    ///
    /// Args:
    ///     signals (np.ndarray): Input signals with shape (channels, fft_size).
    ///
    /// Example:
    ///     >>> signals = np.random.randn(8, 256).astype(np.float32)
    ///     >>> bp.compute(signals)
    fn compute(&mut self, signals: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = signals.shape();
        let (n_channels, n_samples) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: expected {}, got {}",
                self.channels, n_channels
            )));
        }
        if n_samples != self.fft_size {
            return Err(PyValueError::new_err(format!(
                "Sample count mismatch: expected {}, got {}",
                self.fft_size, n_samples
            )));
        }

        let input_array = signals.as_array();

        macro_rules! compute_psd {
            ($bp:expr, $N:expr, $C:expr) => {{
                let mut signal_arr: [[f32; $N]; $C] = [[0.0f32; $N]; $C];
                for (ch, row) in input_array.rows().into_iter().enumerate() {
                    for (i, &val) in row.iter().enumerate() {
                        signal_arr[ch][i] = val;
                    }
                }
                $bp.compute(&signal_arr);
            }};
        }

        match &mut self.inner {
            MultiBandPowerInner::N256C1(bp) => compute_psd!(bp, 256, 1),
            MultiBandPowerInner::N256C4(bp) => compute_psd!(bp, 256, 4),
            MultiBandPowerInner::N256C8(bp) => compute_psd!(bp, 256, 8),
            MultiBandPowerInner::N256C16(bp) => compute_psd!(bp, 256, 16),
            MultiBandPowerInner::N256C32(bp) => compute_psd!(bp, 256, 32),
            MultiBandPowerInner::N256C64(bp) => compute_psd!(bp, 256, 64),
            MultiBandPowerInner::N512C1(bp) => compute_psd!(bp, 512, 1),
            MultiBandPowerInner::N512C4(bp) => compute_psd!(bp, 512, 4),
            MultiBandPowerInner::N512C8(bp) => compute_psd!(bp, 512, 8),
            MultiBandPowerInner::N512C16(bp) => compute_psd!(bp, 512, 16),
            MultiBandPowerInner::N512C32(bp) => compute_psd!(bp, 512, 32),
            MultiBandPowerInner::N512C64(bp) => compute_psd!(bp, 512, 64),
            MultiBandPowerInner::N1024C1(bp) => compute_psd!(bp, 1024, 1),
            MultiBandPowerInner::N1024C4(bp) => compute_psd!(bp, 1024, 4),
            MultiBandPowerInner::N1024C8(bp) => compute_psd!(bp, 1024, 8),
            MultiBandPowerInner::N1024C16(bp) => compute_psd!(bp, 1024, 16),
            MultiBandPowerInner::N1024C32(bp) => compute_psd!(bp, 1024, 32),
            MultiBandPowerInner::N1024C64(bp) => compute_psd!(bp, 1024, 64),
        }

        Ok(())
    }

    /// Get band power for a frequency range.
    ///
    /// Args:
    ///     low_hz (float): Lower frequency bound in Hz.
    ///     high_hz (float): Upper frequency bound in Hz.
    ///
    /// Returns:
    ///     np.ndarray: Band power per channel.
    ///
    /// Example:
    ///     >>> alpha = bp.band_power(8.0, 12.0)  # Alpha band
    fn band_power<'py>(
        &self,
        py: Python<'py>,
        low_hz: f32,
        high_hz: f32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if low_hz >= high_hz {
            return Err(PyValueError::new_err("low_hz must be less than high_hz"));
        }

        let band = ZsFrequencyBand::new(low_hz, high_hz);

        macro_rules! get_band_power {
            ($bp:expr) => {{
                $bp.band_power(band).to_vec()
            }};
        }

        let power: Vec<f32> = match &self.inner {
            MultiBandPowerInner::N256C1(bp) => get_band_power!(bp),
            MultiBandPowerInner::N256C4(bp) => get_band_power!(bp),
            MultiBandPowerInner::N256C8(bp) => get_band_power!(bp),
            MultiBandPowerInner::N256C16(bp) => get_band_power!(bp),
            MultiBandPowerInner::N256C32(bp) => get_band_power!(bp),
            MultiBandPowerInner::N256C64(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C1(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C4(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C8(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C16(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C32(bp) => get_band_power!(bp),
            MultiBandPowerInner::N512C64(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C1(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C4(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C8(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C16(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C32(bp) => get_band_power!(bp),
            MultiBandPowerInner::N1024C64(bp) => get_band_power!(bp),
        };

        Ok(PyArray1::from_vec(py, power))
    }

    /// Reset internal state.
    fn reset(&mut self) {
        macro_rules! reset_bp {
            ($bp:expr) => {{
                $bp.reset();
            }};
        }

        match &mut self.inner {
            MultiBandPowerInner::N256C1(bp) => reset_bp!(bp),
            MultiBandPowerInner::N256C4(bp) => reset_bp!(bp),
            MultiBandPowerInner::N256C8(bp) => reset_bp!(bp),
            MultiBandPowerInner::N256C16(bp) => reset_bp!(bp),
            MultiBandPowerInner::N256C32(bp) => reset_bp!(bp),
            MultiBandPowerInner::N256C64(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C1(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C4(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C8(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C16(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C32(bp) => reset_bp!(bp),
            MultiBandPowerInner::N512C64(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C1(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C4(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C8(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C16(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C32(bp) => reset_bp!(bp),
            MultiBandPowerInner::N1024C64(bp) => reset_bp!(bp),
        }
    }

    /// Get the FFT size.
    #[getter]
    fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Get the sample rate.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiBandPower(fft_size={}, channels={}, sample_rate={})",
            self.fft_size, self.channels, self.sample_rate
        )
    }
}
