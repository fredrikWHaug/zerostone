//! Python bindings for wavelet transform primitives.
//!
//! Provides Continuous Wavelet Transform (CWT) for time-frequency analysis.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{Cwt as ZsCwt, WaveletType as ZsWaveletType};

// ============================================================================
// Cwt - Continuous Wavelet Transform
// ============================================================================

/// Internal enum for handling different size/scale combinations.
/// We support common combinations: N in {64, 128, 256, 512, 1024}, S in {4, 8, 16, 32}
enum CwtInner {
    // N=64
    N64S4(ZsCwt<64, 4>),
    N64S8(ZsCwt<64, 8>),
    N64S16(ZsCwt<64, 16>),
    N64S32(ZsCwt<64, 32>),
    // N=128
    N128S4(ZsCwt<128, 4>),
    N128S8(ZsCwt<128, 8>),
    N128S16(ZsCwt<128, 16>),
    N128S32(ZsCwt<128, 32>),
    // N=256
    N256S4(ZsCwt<256, 4>),
    N256S8(ZsCwt<256, 8>),
    N256S16(ZsCwt<256, 16>),
    N256S32(ZsCwt<256, 32>),
    // N=512
    N512S4(ZsCwt<512, 4>),
    N512S8(ZsCwt<512, 8>),
    N512S16(ZsCwt<512, 16>),
    N512S32(ZsCwt<512, 32>),
    // N=1024
    N1024S4(ZsCwt<1024, 4>),
    N1024S8(ZsCwt<1024, 8>),
    N1024S16(ZsCwt<1024, 16>),
    N1024S32(ZsCwt<1024, 32>),
}

/// Continuous Wavelet Transform using Morlet wavelet.
///
/// Provides time-frequency analysis with multi-resolution capability:
/// better time resolution at high frequencies, better frequency resolution
/// at low frequencies. Essential for event-related synchronization/desynchronization
/// detection in BCIs.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create CWT for motor imagery analysis (8-30 Hz)
/// cwt = zbci.Cwt(
///     size=256,
///     num_scales=8,
///     sample_rate=250.0,
///     min_freq=8.0,
///     max_freq=30.0
/// )
///
/// # Process EEG signal
/// signal = np.random.randn(256).astype(np.float32)
/// power = cwt.power(signal)  # Shape: (8, 256)
///
/// # Get corresponding frequencies
/// freqs = cwt.frequencies()  # Array of 8 frequencies
/// ```
#[pyclass]
pub struct Cwt {
    inner: CwtInner,
    size: usize,
    num_scales: usize,
    sample_rate: f32,
    min_freq: f32,
    max_freq: f32,
    omega0: f32,
}

#[pymethods]
impl Cwt {
    /// Create a new CWT processor.
    ///
    /// Args:
    ///     size (int): Signal length (64, 128, 256, 512, or 1024).
    ///     num_scales (int): Number of frequency scales (4, 8, 16, or 32).
    ///     sample_rate (float): Sample rate in Hz.
    ///     min_freq (float): Minimum frequency in Hz.
    ///     max_freq (float): Maximum frequency in Hz.
    ///     omega0 (float): Morlet wavelet central frequency (default 6.0).
    ///
    /// Returns:
    ///     Cwt: A new CWT processor.
    #[new]
    #[pyo3(signature = (size, num_scales, sample_rate, min_freq, max_freq, omega0=6.0))]
    fn new(
        size: usize,
        num_scales: usize,
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
        omega0: f32,
    ) -> PyResult<Self> {
        if min_freq <= 0.0 {
            return Err(PyValueError::new_err("min_freq must be positive"));
        }
        if max_freq <= min_freq {
            return Err(PyValueError::new_err(
                "max_freq must be greater than min_freq",
            ));
        }
        if max_freq > sample_rate / 2.0 {
            return Err(PyValueError::new_err(
                "max_freq must not exceed Nyquist frequency",
            ));
        }

        let wavelet = ZsWaveletType::Morlet { omega0 };

        macro_rules! create_cwt {
            ($n:expr, $s:expr) => {
                ZsCwt::<$n, $s>::with_wavelet(sample_rate, min_freq, max_freq, wavelet)
            };
        }

        let inner = match (size, num_scales) {
            // N=64
            (64, 4) => CwtInner::N64S4(create_cwt!(64, 4)),
            (64, 8) => CwtInner::N64S8(create_cwt!(64, 8)),
            (64, 16) => CwtInner::N64S16(create_cwt!(64, 16)),
            (64, 32) => CwtInner::N64S32(create_cwt!(64, 32)),
            // N=128
            (128, 4) => CwtInner::N128S4(create_cwt!(128, 4)),
            (128, 8) => CwtInner::N128S8(create_cwt!(128, 8)),
            (128, 16) => CwtInner::N128S16(create_cwt!(128, 16)),
            (128, 32) => CwtInner::N128S32(create_cwt!(128, 32)),
            // N=256
            (256, 4) => CwtInner::N256S4(create_cwt!(256, 4)),
            (256, 8) => CwtInner::N256S8(create_cwt!(256, 8)),
            (256, 16) => CwtInner::N256S16(create_cwt!(256, 16)),
            (256, 32) => CwtInner::N256S32(create_cwt!(256, 32)),
            // N=512
            (512, 4) => CwtInner::N512S4(create_cwt!(512, 4)),
            (512, 8) => CwtInner::N512S8(create_cwt!(512, 8)),
            (512, 16) => CwtInner::N512S16(create_cwt!(512, 16)),
            (512, 32) => CwtInner::N512S32(create_cwt!(512, 32)),
            // N=1024
            (1024, 4) => CwtInner::N1024S4(create_cwt!(1024, 4)),
            (1024, 8) => CwtInner::N1024S8(create_cwt!(1024, 8)),
            (1024, 16) => CwtInner::N1024S16(create_cwt!(1024, 16)),
            (1024, 32) => CwtInner::N1024S32(create_cwt!(1024, 32)),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported size/num_scales: ({}, {}). \
                     Sizes: 64, 128, 256, 512, 1024. Scales: 4, 8, 16, 32.",
                    size, num_scales
                )))
            }
        };

        Ok(Self {
            inner,
            size,
            num_scales,
            sample_rate,
            min_freq,
            max_freq,
            omega0,
        })
    }

    /// Compute CWT power (magnitude squared).
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Power as 2D array (num_scales, size).
    fn power<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_power {
            ($cwt:expr, $n:expr, $s:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = [[0.0f32; $n]; $s];
                $cwt.power(&input, &mut output);
                // Convert to Vec<Vec<f32>> for PyArray2
                output.iter().map(|row| row.to_vec()).collect::<Vec<_>>()
            }};
        }

        let result = match &self.inner {
            CwtInner::N64S4(cwt) => do_power!(cwt, 64, 4),
            CwtInner::N64S8(cwt) => do_power!(cwt, 64, 8),
            CwtInner::N64S16(cwt) => do_power!(cwt, 64, 16),
            CwtInner::N64S32(cwt) => do_power!(cwt, 64, 32),
            CwtInner::N128S4(cwt) => do_power!(cwt, 128, 4),
            CwtInner::N128S8(cwt) => do_power!(cwt, 128, 8),
            CwtInner::N128S16(cwt) => do_power!(cwt, 128, 16),
            CwtInner::N128S32(cwt) => do_power!(cwt, 128, 32),
            CwtInner::N256S4(cwt) => do_power!(cwt, 256, 4),
            CwtInner::N256S8(cwt) => do_power!(cwt, 256, 8),
            CwtInner::N256S16(cwt) => do_power!(cwt, 256, 16),
            CwtInner::N256S32(cwt) => do_power!(cwt, 256, 32),
            CwtInner::N512S4(cwt) => do_power!(cwt, 512, 4),
            CwtInner::N512S8(cwt) => do_power!(cwt, 512, 8),
            CwtInner::N512S16(cwt) => do_power!(cwt, 512, 16),
            CwtInner::N512S32(cwt) => do_power!(cwt, 512, 32),
            CwtInner::N1024S4(cwt) => do_power!(cwt, 1024, 4),
            CwtInner::N1024S8(cwt) => do_power!(cwt, 1024, 8),
            CwtInner::N1024S16(cwt) => do_power!(cwt, 1024, 16),
            CwtInner::N1024S32(cwt) => do_power!(cwt, 1024, 32),
        };

        Ok(PyArray2::from_vec2(py, &result)?)
    }

    /// Compute CWT magnitude (absolute value).
    ///
    /// Args:
    ///     signal (np.ndarray): Input signal as 1D float32 array.
    ///
    /// Returns:
    ///     np.ndarray: Magnitude as 2D array (num_scales, size).
    fn magnitude<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let signal_slice = signal.as_slice()?;
        if signal_slice.len() != self.size {
            return Err(PyValueError::new_err(format!(
                "Signal has {} elements, expected {}",
                signal_slice.len(),
                self.size
            )));
        }

        macro_rules! do_magnitude {
            ($cwt:expr, $n:expr, $s:expr) => {{
                let input: [f32; $n] = signal_slice.try_into().unwrap();
                let mut output = [[0.0f32; $n]; $s];
                $cwt.magnitude(&input, &mut output);
                output.iter().map(|row| row.to_vec()).collect::<Vec<_>>()
            }};
        }

        let result = match &self.inner {
            CwtInner::N64S4(cwt) => do_magnitude!(cwt, 64, 4),
            CwtInner::N64S8(cwt) => do_magnitude!(cwt, 64, 8),
            CwtInner::N64S16(cwt) => do_magnitude!(cwt, 64, 16),
            CwtInner::N64S32(cwt) => do_magnitude!(cwt, 64, 32),
            CwtInner::N128S4(cwt) => do_magnitude!(cwt, 128, 4),
            CwtInner::N128S8(cwt) => do_magnitude!(cwt, 128, 8),
            CwtInner::N128S16(cwt) => do_magnitude!(cwt, 128, 16),
            CwtInner::N128S32(cwt) => do_magnitude!(cwt, 128, 32),
            CwtInner::N256S4(cwt) => do_magnitude!(cwt, 256, 4),
            CwtInner::N256S8(cwt) => do_magnitude!(cwt, 256, 8),
            CwtInner::N256S16(cwt) => do_magnitude!(cwt, 256, 16),
            CwtInner::N256S32(cwt) => do_magnitude!(cwt, 256, 32),
            CwtInner::N512S4(cwt) => do_magnitude!(cwt, 512, 4),
            CwtInner::N512S8(cwt) => do_magnitude!(cwt, 512, 8),
            CwtInner::N512S16(cwt) => do_magnitude!(cwt, 512, 16),
            CwtInner::N512S32(cwt) => do_magnitude!(cwt, 512, 32),
            CwtInner::N1024S4(cwt) => do_magnitude!(cwt, 1024, 4),
            CwtInner::N1024S8(cwt) => do_magnitude!(cwt, 1024, 8),
            CwtInner::N1024S16(cwt) => do_magnitude!(cwt, 1024, 16),
            CwtInner::N1024S32(cwt) => do_magnitude!(cwt, 1024, 32),
        };

        Ok(PyArray2::from_vec2(py, &result)?)
    }

    /// Get pseudo-frequencies for all scales.
    ///
    /// Returns:
    ///     np.ndarray: Frequency in Hz for each scale.
    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        macro_rules! get_freqs {
            ($cwt:expr) => {
                $cwt.frequencies().to_vec()
            };
        }

        let freqs = match &self.inner {
            CwtInner::N64S4(cwt) => get_freqs!(cwt),
            CwtInner::N64S8(cwt) => get_freqs!(cwt),
            CwtInner::N64S16(cwt) => get_freqs!(cwt),
            CwtInner::N64S32(cwt) => get_freqs!(cwt),
            CwtInner::N128S4(cwt) => get_freqs!(cwt),
            CwtInner::N128S8(cwt) => get_freqs!(cwt),
            CwtInner::N128S16(cwt) => get_freqs!(cwt),
            CwtInner::N128S32(cwt) => get_freqs!(cwt),
            CwtInner::N256S4(cwt) => get_freqs!(cwt),
            CwtInner::N256S8(cwt) => get_freqs!(cwt),
            CwtInner::N256S16(cwt) => get_freqs!(cwt),
            CwtInner::N256S32(cwt) => get_freqs!(cwt),
            CwtInner::N512S4(cwt) => get_freqs!(cwt),
            CwtInner::N512S8(cwt) => get_freqs!(cwt),
            CwtInner::N512S16(cwt) => get_freqs!(cwt),
            CwtInner::N512S32(cwt) => get_freqs!(cwt),
            CwtInner::N1024S4(cwt) => get_freqs!(cwt),
            CwtInner::N1024S8(cwt) => get_freqs!(cwt),
            CwtInner::N1024S16(cwt) => get_freqs!(cwt),
            CwtInner::N1024S32(cwt) => get_freqs!(cwt),
        };

        PyArray1::from_vec(py, freqs)
    }

    /// Get the scales array.
    ///
    /// Returns:
    ///     np.ndarray: Scale values for each frequency band.
    fn scales<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        macro_rules! get_scales {
            ($cwt:expr) => {
                $cwt.scales().to_vec()
            };
        }

        let scales = match &self.inner {
            CwtInner::N64S4(cwt) => get_scales!(cwt),
            CwtInner::N64S8(cwt) => get_scales!(cwt),
            CwtInner::N64S16(cwt) => get_scales!(cwt),
            CwtInner::N64S32(cwt) => get_scales!(cwt),
            CwtInner::N128S4(cwt) => get_scales!(cwt),
            CwtInner::N128S8(cwt) => get_scales!(cwt),
            CwtInner::N128S16(cwt) => get_scales!(cwt),
            CwtInner::N128S32(cwt) => get_scales!(cwt),
            CwtInner::N256S4(cwt) => get_scales!(cwt),
            CwtInner::N256S8(cwt) => get_scales!(cwt),
            CwtInner::N256S16(cwt) => get_scales!(cwt),
            CwtInner::N256S32(cwt) => get_scales!(cwt),
            CwtInner::N512S4(cwt) => get_scales!(cwt),
            CwtInner::N512S8(cwt) => get_scales!(cwt),
            CwtInner::N512S16(cwt) => get_scales!(cwt),
            CwtInner::N512S32(cwt) => get_scales!(cwt),
            CwtInner::N1024S4(cwt) => get_scales!(cwt),
            CwtInner::N1024S8(cwt) => get_scales!(cwt),
            CwtInner::N1024S16(cwt) => get_scales!(cwt),
            CwtInner::N1024S32(cwt) => get_scales!(cwt),
        };

        PyArray1::from_vec(py, scales)
    }

    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    #[getter]
    fn num_scales(&self) -> usize {
        self.num_scales
    }

    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    #[getter]
    fn min_freq(&self) -> f32 {
        self.min_freq
    }

    #[getter]
    fn max_freq(&self) -> f32 {
        self.max_freq
    }

    #[getter]
    fn omega0(&self) -> f32 {
        self.omega0
    }

    fn __repr__(&self) -> String {
        format!(
            "Cwt(size={}, num_scales={}, sample_rate={}, freq_range=[{}, {}] Hz)",
            self.size, self.num_scales, self.sample_rate, self.min_freq, self.max_freq
        )
    }
}
