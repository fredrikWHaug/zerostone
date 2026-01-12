//! High-performance, zero-allocation signal processing primitives for BCI and real-time systems.
//!
//! Zerostone provides lock-free, deterministic, and ultra-low-latency components for
//! processing multi-channel neural data streams. All data structures use fixed-size
//! arrays with no heap allocation, making them suitable for embedded and real-time
//! environments.
//!
//! # Features
//!
//! - **[`CircularBuffer`]** - Lock-free SPSC ring buffer for streaming data
//! - **[`IirFilter`]** - Cascaded biquad IIR filters (Butterworth lowpass/highpass/bandpass)
//! - **[`FirFilter`]** - Direct-form FIR filter with linear phase
//! - **[`AcCoupler`]** - AC coupling filter for DC offset removal
//! - **[`Decimator`]** - Sample rate reduction (downsampling)
//! - **[`EnvelopeFollower`]** - Amplitude envelope extraction
//! - **[`Fft`]** - Fast Fourier Transform for spectral analysis
//! - **[`ThresholdDetector`]** - Multi-channel spike detection with refractory period
//! - **[`OnlineStats`]** - Welford's algorithm for streaming mean/variance
//! - **[`OnlineCov`]** - Streaming covariance matrix estimation for CSP
//! - **[`StreamingPercentile`]** - P² algorithm for streaming percentile estimation
//! - **[`OasisDeconvolution`]** - OASIS calcium imaging deconvolution
//! - **[`WindowType`]** - Window functions for spectral analysis (Hann, Hamming, Blackman, etc.)
//! - **[`Cwt`]** - Continuous Wavelet Transform for time-frequency analysis
//! - **[`Stft`]** - Short-Time Fourier Transform for spectrograms
//!
//! # Example
//!
//! ```
//! use zerostone::{BiquadCoeffs, IirFilter, ThresholdDetector};
//!
//! // Create a 4th-order Butterworth lowpass filter at 100 Hz (1000 Hz sample rate)
//! // 4th order = 2 cascaded biquad sections
//! let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
//! let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
//!
//! // Create a spike detector for 8 channels with threshold 3.0 and 100-sample refractory
//! let mut detector: ThresholdDetector<8> = ThresholdDetector::new(3.0, 100);
//!
//! // Process a sample through the filter
//! let filtered = filter.process_sample(1.5);
//! ```
//!
//! # Performance Targets
//!
//! - Circular buffer: 30M+ samples/sec
//! - IIR filtering: <100 ns/sample for 32 channels at 4th order
//! - Threshold detection: <10 μs for 1024 channels

#![no_std]

mod buffer;
mod csp;
mod decimate;
mod deconvolution;
mod detector;
mod envelope;
mod fft;
mod filter;
pub mod linalg;
mod percentile;
mod stats;
mod stft;
pub mod wavelet;
mod window;

// Re-export at crate root for convenience
pub use buffer::CircularBuffer;
pub use csp::{AdaptiveCsp, CspError, UpdateConfig};
pub use decimate::Decimator;
pub use deconvolution::{DeconvolutionResult, OasisDeconvolution};
pub use detector::{
    AdaptiveThresholdDetector, DetectorState, SpikeEvent, SpikeEvents, ThresholdDetector,
};
pub use envelope::{EnvelopeFollower, Rectification};
pub use fft::{BandPower, Complex, Fft};
pub use filter::{AcCoupler, BiquadCoeffs, FirFilter, IirFilter};
pub use percentile::StreamingPercentile;
pub use stats::{OnlineCov, OnlineStats};
pub use stft::Stft;
pub use wavelet::{morlet_coefficient, wavelet_half_width, Cwt, MultiChannelCwt, WaveletType};
pub use window::{
    apply_window, apply_window_f64, coherent_gain, equivalent_noise_bandwidth, window_coefficient,
    WindowType,
};
