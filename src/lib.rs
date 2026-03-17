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
//! - **[`NotchFilter`]** - Multi-channel powerline notch filter (50/60 Hz + harmonics)
//! - **[`IirFilter`]** - Cascaded biquad IIR filters (Butterworth lowpass/highpass/bandpass)
//! - **[`FirFilter`]** - Direct-form FIR filter with linear phase
//! - **[`AcCoupler`]** - AC coupling filter for DC offset removal
//! - **[`MedianFilter`]** - Non-linear median filter for impulsive noise rejection
//! - **[`SurfaceLaplacian`]** - Surface Laplacian spatial filter for volume conduction reduction
//! - **[`CommonAverageReference`]** - Common Average Reference spatial filter for multi-channel recordings
//! - **[`LmsFilter`], [`NlmsFilter`]** - Adaptive filters for real-time noise cancellation
//! - **[`Decimator`]** - Sample rate reduction (downsampling)
//! - **[`Interpolator`]** - Sample rate increase (upsampling)
//! - **[`ChannelRouter`]** - Channel selection, reordering, and duplication for pipeline composition
//! - **[`EnvelopeFollower`]** - Amplitude envelope extraction
//! - **[`Fft`]** - Fast Fourier Transform for spectral analysis
//! - **[`MultiBandPower`]** - Multi-channel band power extraction with proper PSD normalization
//! - **[`ThresholdDetector`]** - Multi-channel spike detection with refractory period
//! - **[`ZeroCrossingDetector`]** - Zero-crossing detection for ZCR features and epilepsy patterns
//! - **[`OnlineStats`]** - Welford's algorithm for streaming mean/variance
//! - **[`OnlineCov`]** - Streaming covariance matrix estimation for CSP
//! - **[`StreamingPercentile`]** - P² algorithm for streaming percentile estimation
//! - **[`WindowedRms`]** - Windowed RMS and power computation for amplitude tracking
//! - **[`OasisDeconvolution`]** - OASIS calcium imaging deconvolution
//! - **[`WelchPsd`]** - Welch's method PSD estimation with segment averaging
//! - **[`WindowType`]** - Window functions for spectral analysis (Hann, Hamming, Blackman, etc.)
//! - **[`Cwt`]** - Continuous Wavelet Transform for time-frequency analysis
//! - **[`Stft`]** - Short-Time Fourier Transform for spectrograms
//! - **[`ArtifactDetector`]** - Threshold-based artifact detection (amplitude/gradient)
//! - **[`ZscoreArtifact`]** - Adaptive z-score based artifact detection
//! - **[`xcorr`]** - Cross-correlation and auto-correlation functions
//! - **[`hilbert`]** - Hilbert transform for analytic signal and instantaneous parameters
//! - **[`connectivity`]** - Coherence and phase locking value for brain connectivity
//! - **[`ClockOffset`]** - Single timestamp offset measurement (NTP-style)
//! - **[`SampleClock`]** - Sample index ↔ timestamp conversion
//! - **[`LinearDrift`]** - Online linear regression for clock drift estimation
//! - **[`OffsetBuffer`]** - Filtered clock offset buffer with quality selection
//! - **[`TangentSpace`]** - Riemannian tangent space projection for SPD matrices
//! - **[`matrix_log`], [`matrix_exp`]** - Matrix logarithm/exponential for SPD matrices
//! - **[`OnlineKMeans`]** - Online k-means clustering for spike sorting
//! - **[`modulation_index`], [`mean_vector_length`]** - Phase-amplitude coupling metrics
//! - **[`entropy`]** - Sample, approximate, spectral, and multiscale entropy measures
//!
//! # Example
//!
//! ```
//! use zerostone::{BiquadCoeffs, IirFilter, ThresholdDetector};
//!
//! // Create a 4th-order Butterworth lowpass filter at 100 Hz (1000 Hz sample rate)
//! // 4th order = 2 cascaded biquad sections with proper pole placement
//! let sections = BiquadCoeffs::butterworth_lowpass_sections::<2>(1000.0, 100.0);
//! let mut filter: IirFilter<2> = IirFilter::new(sections);
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
mod bandpower;
mod buffer;
pub mod cca;
pub mod connectivity;
mod csp;
mod decimate;
mod deconvolution;
mod detector;
pub mod edf;
pub mod entropy;
mod envelope;
pub mod erp;
pub mod ersp;
mod fft;
mod filter;
pub mod hilbert;
pub mod ica;
mod interpolate;
pub mod isi;
pub mod kalman;
pub mod lda;
pub mod linalg;
pub mod localize;
pub mod mda;
pub mod metrics;
mod notch;
pub mod online_kmeans;
pub mod pac;
mod percentile;
mod pipeline;
pub mod probe;
pub mod quality;
pub mod riemannian;
mod rms;
mod router;
pub mod sorter;
pub mod spike_sort;
mod stats;
mod stft;
mod sync;
pub mod template_subtract;
pub mod wavelet;
mod welch;
pub mod whitening;
mod window;
pub mod xcorr;
pub mod xdf;

// Re-export at crate root for convenience
pub use artifact::{ArtifactDetector, ArtifactType, ZscoreArtifact};
pub use bandpower::{FrequencyBand, IntegrationMethod, MultiBandPower};
pub use buffer::CircularBuffer;
pub use csp::{AdaptiveCsp, CspError, MulticlassCsp, UpdateConfig};
pub use decimate::Decimator;
pub use deconvolution::{DeconvolutionResult, OasisDeconvolution};
pub use detector::{
    AdaptiveThresholdDetector, CrossingDirection, DetectorState, SpikeEvent, SpikeEvents,
    ThresholdDetector, ZeroCrossingDetector,
};
pub use edf::{
    parse_header, parse_signal_header, read_channel, read_record, EdfError, EdfHeader,
    EdfSignalHeader,
};
pub use entropy::{approximate_entropy, multiscale_entropy, sample_entropy, spectral_entropy};
pub use envelope::{EnvelopeFollower, Rectification};
pub use erp::{apply_spatial_filter, epoch_average, xdawn_filters, ErpError};
pub use ersp::{BaselineMode, ErspError};
pub use fft::{BandPower, Complex, Fft};
pub use filter::{
    AcCoupler, AdaptiveOutput, BiquadCoeffs, CommonAverageReference, FirFilter, IirFilter,
    LmsFilter, MedianFilter, NlmsFilter, SurfaceLaplacian,
};
pub use hilbert::HilbertTransform;
pub use ica::{ContrastFunction, Ica, IcaError};
pub use interpolate::{InterpolationMethod, Interpolator};
pub use isi::{autocorrelogram, isi_cv, local_variation, IsiHistogram};
pub use kalman::{KalmanError, KalmanFilter};
pub use lda::{Lda, LdaError};
pub use localize::{center_of_mass, center_of_mass_threshold, localize_spike, monopole_localize};
pub use mda::{
    mda_element_size, mda_num_elements, parse_mda_header, read_mda_f32, read_mda_f64, MdaDataType,
    MdaHeader, MAX_DIMS,
};
pub use metrics::{compare_sorting, compare_spike_trains, UnitMatch};
pub use notch::NotchFilter;
pub use online_kmeans::{KMeansError, KMeansResult, OnlineKMeans};
pub use pac::{mean_vector_length, modulation_index, phase_amplitude_distribution};
pub use percentile::StreamingPercentile;
pub use pipeline::{BlockProcessor, CloneableProcessor, Pipeline, RateChangingProcessor, Terminal};
pub use quality::{
    contamination_rate, d_prime, euclidean_distance, isi_violation_rate, isolation_distance,
    mean_silhouette, silhouette_score, waveform_snr,
};
pub use riemannian::{
    frechet_mean, matrix_exp, matrix_inv_sqrt, matrix_log, matrix_sqrt, mdm_classify, recenter,
    reconstruct_from_eigen, riemannian_distance, TangentSpace,
};
pub use rms::WindowedRms;
pub use router::ChannelRouter;
pub use sorter::{
    estimate_noise_multichannel, sort_multichannel, ClusterInfo, SortConfig, SortResult,
};
pub use spike_sort::{
    align_to_peak, deduplicate_events, detect_spikes, detect_spikes_multichannel,
    estimate_noise_mad, extract_multichannel, extract_peak_channel, MultiChannelEvent, SortError,
    SpikeCluster, TemplateMatch, WaveformExtractor, WaveformPca,
};
pub use stats::{OnlineCov, OnlineStats};
pub use stft::Stft;
pub use sync::{ClockOffset, LinearDrift, OffsetBuffer, SampleClock};
pub use template_subtract::{PeelResult, TemplateSubtractor};
pub use wavelet::{morlet_coefficient, wavelet_half_width, Cwt, MultiChannelCwt, WaveletType};
pub use welch::WelchPsd;
pub use whitening::{apply_whitening, WhiteningMatrix, WhiteningMode};
pub use window::{
    apply_window, apply_window_f64, coherent_gain, equivalent_noise_bandwidth, window_coefficient,
    WindowType,
};
pub use xdf::{
    count_chunk_samples, decode_samples_f64, find_tag_value, next_chunk, parse_clock_offset,
    parse_stream_info, read_varlen, validate_magic, ClockOffsetPair, XdfChannelFormat, XdfChunk,
    XdfError, XdfStreamInfo, TAG_BOUNDARY, TAG_CLOCK_OFFSET, TAG_FILE_HEADER, TAG_SAMPLES,
    TAG_STREAM_FOOTER, TAG_STREAM_HEADER,
};
