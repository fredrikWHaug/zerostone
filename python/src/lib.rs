use ::zerostone::{BiquadCoeffs, IirFilter as ZsIirFilter};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Bound;

mod analysis;
mod artifact;
mod cca;
mod connectivity;
mod csp;
mod deconvolution;
mod detection;
mod erp;
mod filters;
mod notch;
mod percentile;
mod pipeline;
mod resampling;
mod riemannian;
mod spatial;
mod spectral;
mod stats;
mod sync;
mod utils;
mod wavelet;
mod welch;
mod window;
mod xcorr;

use analysis::{EnvelopeFollower, HilbertTransform, WindowedRms};
use artifact::{ArtifactDetector, ZscoreArtifact};
use csp::AdaptiveCsp;
use deconvolution::OasisDeconvolution;
use detection::{AdaptiveThresholdDetector, ThresholdDetector, ZeroCrossingDetector};
use filters::{AcCoupler, FirFilter, LmsFilter, MedianFilter, NlmsFilter};
use notch::NotchFilter as PyNotchFilter;
use percentile::StreamingPercentile;
use pipeline::Pipeline;
use resampling::{Decimator, Interpolator};
use riemannian::{MdmClassifier, TangentSpace};
use spatial::{ChannelRouter, SurfaceLaplacian, CAR};
use spectral::{Fft, MultiBandPower, Stft};
use stats::{OnlineCov, OnlineStats};
use sync::{ClockOffset, LinearDrift, OffsetBuffer, SampleClock};
use wavelet::Cwt;
use welch::WelchPsd as PyWelchPsd;

/// IIR (Infinite Impulse Response) filter with cascaded biquad sections.
///
/// Supports Butterworth filters of order 2, 4, 6, or 8 with proper pole placement.
/// Each section uses a distinct Q factor, matching scipy.signal.butter output.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// # Create a 4th-order lowpass filter at 30 Hz (default order=4)
/// lpf = zbci.IirFilter.butterworth_lowpass(1000.0, 30.0)
///
/// # Create a 2nd-order lowpass filter
/// lpf2 = zbci.IirFilter.butterworth_lowpass(1000.0, 30.0, order=2)
///
/// # Process a signal
/// signal = np.random.randn(1000).astype(np.float32)
/// filtered = lpf.process(signal)
/// ```
#[pyclass]
struct IirFilter {
    inner: IirFilterInner,
    sample_rate: f32,
    order: u32,
}

enum IirFilterInner {
    Sections1(ZsIirFilter<1>),
    Sections2(ZsIirFilter<2>),
    Sections3(ZsIirFilter<3>),
    Sections4(ZsIirFilter<4>),
}

impl IirFilterInner {
    fn process_sample(&mut self, sample: f32) -> f32 {
        match self {
            IirFilterInner::Sections1(f) => f.process_sample(sample),
            IirFilterInner::Sections2(f) => f.process_sample(sample),
            IirFilterInner::Sections3(f) => f.process_sample(sample),
            IirFilterInner::Sections4(f) => f.process_sample(sample),
        }
    }

    fn reset(&mut self) {
        match self {
            IirFilterInner::Sections1(f) => f.reset(),
            IirFilterInner::Sections2(f) => f.reset(),
            IirFilterInner::Sections3(f) => f.reset(),
            IirFilterInner::Sections4(f) => f.reset(),
        }
    }
}

fn validate_order(order: u32) -> PyResult<()> {
    match order {
        2 | 4 | 6 | 8 => Ok(()),
        _ => Err(PyValueError::new_err(
            "Order must be 2, 4, 6, or 8 (number of sections = order / 2)",
        )),
    }
}

#[pymethods]
impl IirFilter {
    /// Create a Butterworth lowpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     cutoff (float): Cutoff frequency in Hz
    ///     order (int, optional): Filter order (2, 4, 6, or 8). Default: 4
    ///
    /// Returns:
    ///     IirFilter: A Butterworth lowpass filter of the specified order
    ///
    /// Example:
    ///     >>> lpf = IirFilter.butterworth_lowpass(1000.0, 40.0)
    ///     >>> lpf2 = IirFilter.butterworth_lowpass(1000.0, 40.0, order=2)
    #[staticmethod]
    #[pyo3(signature = (sample_rate, cutoff, order=4))]
    fn butterworth_lowpass(sample_rate: f32, cutoff: f32, order: u32) -> PyResult<Self> {
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequency must be between 0 and {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }
        validate_order(order)?;

        let inner = match order {
            2 => IirFilterInner::Sections1(ZsIirFilter::new(
                BiquadCoeffs::butterworth_lowpass_sections::<1>(sample_rate, cutoff),
            )),
            4 => IirFilterInner::Sections2(ZsIirFilter::new(
                BiquadCoeffs::butterworth_lowpass_sections::<2>(sample_rate, cutoff),
            )),
            6 => IirFilterInner::Sections3(ZsIirFilter::new(
                BiquadCoeffs::butterworth_lowpass_sections::<3>(sample_rate, cutoff),
            )),
            8 => IirFilterInner::Sections4(ZsIirFilter::new(
                BiquadCoeffs::butterworth_lowpass_sections::<4>(sample_rate, cutoff),
            )),
            _ => unreachable!(),
        };

        Ok(Self {
            inner,
            sample_rate,
            order,
        })
    }

    /// Create a Butterworth highpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     cutoff (float): Cutoff frequency in Hz
    ///     order (int, optional): Filter order (2, 4, 6, or 8). Default: 4
    ///
    /// Returns:
    ///     IirFilter: A Butterworth highpass filter of the specified order
    ///
    /// Example:
    ///     >>> hpf = IirFilter.butterworth_highpass(1000.0, 1.0)
    #[staticmethod]
    #[pyo3(signature = (sample_rate, cutoff, order=4))]
    fn butterworth_highpass(sample_rate: f32, cutoff: f32, order: u32) -> PyResult<Self> {
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequency must be between 0 and {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }
        validate_order(order)?;

        let inner = match order {
            2 => IirFilterInner::Sections1(ZsIirFilter::new(
                BiquadCoeffs::butterworth_highpass_sections::<1>(sample_rate, cutoff),
            )),
            4 => IirFilterInner::Sections2(ZsIirFilter::new(
                BiquadCoeffs::butterworth_highpass_sections::<2>(sample_rate, cutoff),
            )),
            6 => IirFilterInner::Sections3(ZsIirFilter::new(
                BiquadCoeffs::butterworth_highpass_sections::<3>(sample_rate, cutoff),
            )),
            8 => IirFilterInner::Sections4(ZsIirFilter::new(
                BiquadCoeffs::butterworth_highpass_sections::<4>(sample_rate, cutoff),
            )),
            _ => unreachable!(),
        };

        Ok(Self {
            inner,
            sample_rate,
            order,
        })
    }

    /// Create a Butterworth bandpass filter.
    ///
    /// Args:
    ///     sample_rate (float): Sampling frequency in Hz
    ///     low_cutoff (float): Lower cutoff frequency in Hz
    ///     high_cutoff (float): Upper cutoff frequency in Hz
    ///     order (int, optional): Filter order (2, 4, 6, or 8). Default: 4
    ///
    /// Returns:
    ///     IirFilter: A Butterworth bandpass filter of the specified order
    ///
    /// Example:
    ///     >>> bpf = IirFilter.butterworth_bandpass(1000.0, 8.0, 12.0)
    #[staticmethod]
    #[pyo3(signature = (sample_rate, low_cutoff, high_cutoff, order=4))]
    fn butterworth_bandpass(
        sample_rate: f32,
        low_cutoff: f32,
        high_cutoff: f32,
        order: u32,
    ) -> PyResult<Self> {
        if low_cutoff <= 0.0 || high_cutoff >= sample_rate / 2.0 || low_cutoff >= high_cutoff {
            return Err(PyValueError::new_err(format!(
                "Cutoff frequencies must satisfy: 0 < low_cutoff < high_cutoff < {} Hz (Nyquist)",
                sample_rate / 2.0
            )));
        }
        validate_order(order)?;

        let inner = match order {
            2 => IirFilterInner::Sections1(ZsIirFilter::new(
                BiquadCoeffs::butterworth_bandpass_sections::<1>(sample_rate, low_cutoff, high_cutoff),
            )),
            4 => IirFilterInner::Sections2(ZsIirFilter::new(
                BiquadCoeffs::butterworth_bandpass_sections::<2>(sample_rate, low_cutoff, high_cutoff),
            )),
            6 => IirFilterInner::Sections3(ZsIirFilter::new(
                BiquadCoeffs::butterworth_bandpass_sections::<3>(sample_rate, low_cutoff, high_cutoff),
            )),
            8 => IirFilterInner::Sections4(ZsIirFilter::new(
                BiquadCoeffs::butterworth_bandpass_sections::<4>(sample_rate, low_cutoff, high_cutoff),
            )),
            _ => unreachable!(),
        };

        Ok(Self {
            inner,
            sample_rate,
            order,
        })
    }

    /// Process a signal through the filter.
    ///
    /// Args:
    ///     input (np.ndarray): Input signal as 1D float32 numpy array
    ///
    /// Returns:
    ///     np.ndarray: Filtered signal as 1D float32 numpy array
    ///
    /// Example:
    ///     >>> signal = np.random.randn(1000).astype(np.float32)
    ///     >>> filtered = lpf.process(signal)
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_slice = input.as_slice()?;
        let mut output = vec![0.0f32; input_slice.len()];

        for (i, &sample) in input_slice.iter().enumerate() {
            output[i] = self.inner.process_sample(sample);
        }

        Ok(PyArray1::from_vec(py, output))
    }

    /// Reset the filter state (clear delay lines).
    ///
    /// This is useful when processing discontinuous segments of data.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the sample rate of the filter.
    #[getter]
    fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the filter order.
    #[getter]
    fn order(&self) -> u32 {
        self.order
    }

    fn __repr__(&self) -> String {
        format!(
            "IirFilter(sample_rate={} Hz, order={})",
            self.sample_rate, self.order
        )
    }
}

/// zpybci: High-performance signal processing for BCI research.
///
/// This module provides zero-allocation, real-time signal processing primitives
/// implemented in Rust for maximum performance.
#[pymodule]
fn zpybci(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Filters
    m.add_class::<IirFilter>()?;
    m.add_class::<FirFilter>()?;
    m.add_class::<AcCoupler>()?;
    m.add_class::<MedianFilter>()?;
    m.add_class::<LmsFilter>()?;
    m.add_class::<NlmsFilter>()?;
    m.add_class::<PyNotchFilter>()?;

    // Spatial filters
    m.add_class::<CAR>()?;
    m.add_class::<SurfaceLaplacian>()?;
    m.add_class::<ChannelRouter>()?;

    // Pipeline
    m.add_class::<Pipeline>()?;

    // Statistics
    m.add_class::<OnlineStats>()?;

    // Detection
    m.add_class::<ThresholdDetector>()?;
    m.add_class::<AdaptiveThresholdDetector>()?;
    m.add_class::<ZeroCrossingDetector>()?;

    // Resampling
    m.add_class::<Decimator>()?;
    m.add_class::<Interpolator>()?;

    // Spectral
    m.add_class::<Fft>()?;
    m.add_class::<Stft>()?;
    m.add_class::<MultiBandPower>()?;
    m.add_class::<PyWelchPsd>()?;

    // CSP (Common Spatial Patterns)
    m.add_class::<AdaptiveCsp>()?;

    // Artifact Detection
    m.add_class::<ArtifactDetector>()?;
    m.add_class::<ZscoreArtifact>()?;

    // Analysis
    m.add_class::<EnvelopeFollower>()?;
    m.add_class::<WindowedRms>()?;
    m.add_class::<HilbertTransform>()?;

    // Wavelet
    m.add_class::<Cwt>()?;

    // Percentile
    m.add_class::<StreamingPercentile>()?;

    // Deconvolution
    m.add_class::<OasisDeconvolution>()?;

    // Riemannian geometry
    m.add_class::<TangentSpace>()?;
    m.add_class::<MdmClassifier>()?;
    riemannian::register(m)?;

    // Online covariance
    m.add_class::<OnlineCov>()?;

    // Clock synchronization
    m.add_class::<ClockOffset>()?;
    m.add_class::<SampleClock>()?;
    m.add_class::<LinearDrift>()?;
    m.add_class::<OffsetBuffer>()?;

    // Cross-correlation functions
    xcorr::register(m)?;

    // Connectivity metrics
    connectivity::register(m)?;

    // CCA / SSVEP detection
    cca::register(m)?;

    // ERP / xDAWN
    erp::register(m)?;

    // Window functions
    window::register(m)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
