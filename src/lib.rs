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
//! - **[`MedianFilter`]** - Non-linear median filter for impulsive noise rejection
//! - **[`SurfaceLaplacian`]** - Surface Laplacian spatial filter for volume conduction reduction
//! - **[`CommonAverageReference`]** - Common Average Reference spatial filter for multi-channel recordings
//! - **[`LmsFilter`], [`NlmsFilter`]** - Adaptive filters for real-time noise cancellation
//! - **[`Decimator`]** - Sample rate reduction (downsampling)
//! - **[`Interpolator`]** - Sample rate increase (upsampling)
//! - **[`EnvelopeFollower`]** - Amplitude envelope extraction
//! - **[`Fft`]** - Fast Fourier Transform for spectral analysis
//! - **[`ThresholdDetector`]** - Multi-channel spike detection with refractory period
//! - **[`ZeroCrossingDetector`]** - Zero-crossing detection for ZCR features and epilepsy patterns
//! - **[`OnlineStats`]** - Welford's algorithm for streaming mean/variance
//! - **[`OnlineCov`]** - Streaming covariance matrix estimation for CSP
//! - **[`StreamingPercentile`]** - P² algorithm for streaming percentile estimation
//! - **[`WindowedRms`]** - Windowed RMS and power computation for amplitude tracking
//! - **[`OasisDeconvolution`]** - OASIS calcium imaging deconvolution
//! - **[`WindowType`]** - Window functions for spectral analysis (Hann, Hamming, Blackman, etc.)
//! - **[`Cwt`]** - Continuous Wavelet Transform for time-frequency analysis
//! - **[`Stft`]** - Short-Time Fourier Transform for spectrograms
//! - **[`ArtifactDetector`]** - Threshold-based artifact detection (amplitude/gradient)
//! - **[`ZscoreArtifact`]** - Adaptive z-score based artifact detection
//! - **[`xcorr`]** - Cross-correlation and auto-correlation functions
//! - **[`hilbert`]** - Hilbert transform for analytic signal and instantaneous parameters
//! - **[`ClockOffset`]** - Single timestamp offset measurement (NTP-style)
//! - **[`SampleClock`]** - Sample index ↔ timestamp conversion
//! - **[`LinearDrift`]** - Online linear regression for clock drift estimation
//! - **[`OffsetBuffer`]** - Filtered clock offset buffer with quality selection
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

mod artifact;
mod buffer;
mod csp;
mod decimate;
mod deconvolution;
mod detector;
mod envelope;
mod fft;
mod filter;
pub mod hilbert;
mod interpolate;
pub mod linalg;
mod percentile;
mod rms;
mod stats;
mod stft;
mod sync;
pub mod wavelet;
mod window;
pub mod xcorr;

// Re-export at crate root for convenience
pub use artifact::{ArtifactDetector, ArtifactType, ZscoreArtifact};
pub use buffer::CircularBuffer;
pub use csp::{AdaptiveCsp, CspError, UpdateConfig};
pub use decimate::Decimator;
pub use deconvolution::{DeconvolutionResult, OasisDeconvolution};
pub use detector::{
    AdaptiveThresholdDetector, CrossingDirection, DetectorState, SpikeEvent, SpikeEvents,
    ThresholdDetector, ZeroCrossingDetector,
};
pub use envelope::{EnvelopeFollower, Rectification};
pub use fft::{BandPower, Complex, Fft};
pub use filter::{
    AcCoupler, AdaptiveOutput, BiquadCoeffs, CommonAverageReference, FirFilter, IirFilter,
    LmsFilter, MedianFilter, NlmsFilter, SurfaceLaplacian,
};
pub use hilbert::HilbertTransform;
pub use interpolate::{InterpolationMethod, Interpolator};
pub use percentile::StreamingPercentile;
pub use rms::WindowedRms;
pub use stats::{OnlineCov, OnlineStats};
pub use stft::Stft;
pub use sync::{ClockOffset, LinearDrift, OffsetBuffer, SampleClock};
pub use wavelet::{morlet_coefficient, wavelet_half_width, Cwt, MultiChannelCwt, WaveletType};
pub use window::{
    apply_window, apply_window_f64, coherent_gain, equivalent_noise_bandwidth, window_coefficient,
    WindowType,
};
