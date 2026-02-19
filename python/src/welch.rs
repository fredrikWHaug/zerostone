//! Python bindings for Welch's PSD estimation.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::WelchPsd as ZsWelchPsd;

use crate::spectral::parse_window_type;

/// Internal enum for handling different FFT sizes.
enum WelchInner {
    Size256(ZsWelchPsd<256>),
    Size512(ZsWelchPsd<512>),
    Size1024(ZsWelchPsd<1024>),
    Size2048(ZsWelchPsd<2048>),
    Size4096(ZsWelchPsd<4096>),
}

/// Welch's method PSD estimator.
///
/// Estimates the power spectral density by averaging overlapping windowed
/// periodograms. This reduces variance compared to a single-window PSD estimate.
///
/// Equivalent to ``scipy.signal.welch``, MATLAB's ``pwelch``, or MNE-Python's
/// ``compute_psd``.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// welch = zbci.WelchPsd(fft_size=256, window='hann', overlap=0.5)
///
/// # 10 Hz sine wave at 250 Hz sample rate
/// t = np.arange(2048) / 250.0
/// signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
///
/// freqs, psd = welch.estimate(signal, sample_rate=250.0)
/// ```
#[pyclass]
pub struct WelchPsd {
    inner: WelchInner,
    fft_size: usize,
    window_name: String,
    overlap: f32,
}

#[pymethods]
impl WelchPsd {
    /// Create a new Welch PSD estimator.
    ///
    /// Args:
    ///     fft_size (int): Segment/FFT size. Must be 256, 512, 1024, 2048, or 4096.
    ///     window (str): Window type - 'rectangular', 'hann', 'hamming', 'blackman',
    ///         or 'blackman_harris'. Default: 'hann'.
    ///     overlap (float): Overlap fraction in [0.0, 1.0). Default: 0.5.
    ///
    /// Returns:
    ///     WelchPsd: A new Welch PSD estimator.
    ///
    /// Example:
    ///     >>> welch = WelchPsd(fft_size=1024, window='hann', overlap=0.5)
    #[new]
    #[pyo3(signature = (fft_size, window = "hann", overlap = 0.5))]
    fn new(fft_size: usize, window: &str, overlap: f32) -> PyResult<Self> {
        if overlap < 0.0 || overlap >= 1.0 {
            return Err(PyValueError::new_err(
                "overlap must be in [0.0, 1.0)",
            ));
        }

        let window_type = parse_window_type(window)?;

        let inner = match fft_size {
            256 => WelchInner::Size256(ZsWelchPsd::new(window_type, overlap)),
            512 => WelchInner::Size512(ZsWelchPsd::new(window_type, overlap)),
            1024 => WelchInner::Size1024(ZsWelchPsd::new(window_type, overlap)),
            2048 => WelchInner::Size2048(ZsWelchPsd::new(window_type, overlap)),
            4096 => WelchInner::Size4096(ZsWelchPsd::new(window_type, overlap)),
            _ => {
                return Err(PyValueError::new_err(
                    "fft_size must be 256, 512, 1024, 2048, or 4096",
                ))
            }
        };

        Ok(Self {
            inner,
            fft_size,
            window_name: window.to_lowercase(),
            overlap,
        })
    }

    /// Estimate PSD of a 1D signal.
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array. Must have
    ///         length >= fft_size.
    ///     sample_rate (float): Sample rate in Hz.
    ///
    /// Returns:
    ///     tuple[np.ndarray, np.ndarray]: (frequencies, psd) as two 1D float32 arrays,
    ///         each with fft_size/2 + 1 elements.
    ///
    /// Example:
    ///     >>> freqs, psd = welch.estimate(signal, sample_rate=250.0)
    ///     >>> print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")
    fn estimate<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
        sample_rate: f32,
    ) -> PyResult<(
        pyo3::Bound<'py, PyArray1<f32>>,
        pyo3::Bound<'py, PyArray1<f32>>,
    )> {
        let input_slice = signal.as_slice()?;
        if input_slice.len() < self.fft_size {
            return Err(PyValueError::new_err(format!(
                "Signal length {} must be >= fft_size {}",
                input_slice.len(),
                self.fft_size
            )));
        }
        if sample_rate <= 0.0 {
            return Err(PyValueError::new_err("sample_rate must be positive"));
        }

        let out_len = self.fft_size / 2 + 1;
        let mut psd = vec![0.0f32; out_len];
        let mut freqs = vec![0.0f32; out_len];

        macro_rules! run_welch {
            ($welch:expr, $N:expr) => {{
                $welch.estimate(input_slice, sample_rate, &mut psd);
                $welch.frequencies(sample_rate, &mut freqs);
            }};
        }

        match &self.inner {
            WelchInner::Size256(w) => run_welch!(w, 256),
            WelchInner::Size512(w) => run_welch!(w, 512),
            WelchInner::Size1024(w) => run_welch!(w, 1024),
            WelchInner::Size2048(w) => run_welch!(w, 2048),
            WelchInner::Size4096(w) => run_welch!(w, 4096),
        }

        Ok((
            PyArray1::from_vec(py, freqs),
            PyArray1::from_vec(py, psd),
        ))
    }

    /// Get the FFT/segment size.
    #[getter]
    fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the window type name.
    #[getter]
    fn window(&self) -> &str {
        &self.window_name
    }

    /// Get the overlap fraction.
    #[getter]
    fn overlap(&self) -> f32 {
        self.overlap
    }

    fn __repr__(&self) -> String {
        format!(
            "WelchPsd(fft_size={}, window={}, overlap={})",
            self.fft_size, self.window_name, self.overlap
        )
    }
}
