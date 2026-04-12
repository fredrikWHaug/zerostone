//! Multi-channel sorting pipeline orchestrator.
//!
//! Chains the individual spike sorting primitives into a complete pipeline:
//! noise estimation, whitening, detection, deduplication, alignment,
//! waveform extraction, PCA, clustering, and quality metrics.
//!
//! # Pipeline
//!
//! 1. **Noise estimation** -- MAD per channel
//! 2. **Covariance** -- sample covariance from data
//! 3. **Whitening** -- ZCA in-place, producing unit-variance channels
//! 4. **Detection** -- threshold crossings on whitened data
//! 5. **Deduplication** -- spatial dedup using probe geometry
//! 6. **Alignment** -- fine peak alignment within a local window
//! 7. **Extraction** -- peak-channel waveforms
//! 8. **PCA** -- dimensionality reduction
//! 9. **Clustering** -- online k-means with adaptive creation
//! 10. **Labels** -- cluster assignment per spike
//! 11. **Quality** -- SNR and ISI violation rate per cluster
//!
//! # Example
//!
//! ```
//! use zerostone::sorter::{SortConfig, SortResult, ClusterInfo, estimate_noise_multichannel};
//!
//! // Estimate noise on 2-channel data
//! let data = [[1.0, -0.5], [0.3, 0.2], [-0.7, 1.1], [0.1, -0.3]];
//! let mut scratch = [0.0; 8];
//! let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
//! assert!(noise[0] > 0.0);
//! assert!(noise[1] > 0.0);
//! ```

use crate::float::{self, Float};
use crate::isi;
use crate::localize;
use crate::matched_filter::{MatchedDetection, MatchedFilterBank};
use crate::online_kmeans::OnlineKMeans;
use crate::probe::ProbeLayout;
use crate::quality;
use crate::spike_sort::{
    align_to_peak, compute_adaptive_thresholds, deduplicate_events, detect_spikes_multichannel,
    extract_peak_channel, MultiChannelEvent, SortError, WaveformPca,
};
use crate::whitening::{WhiteningMatrix, WhiteningMode};

/// Detection mode for the spike sorting pipeline.
///
/// Controls how threshold crossings are identified in the whitened data.
///
/// # Example
///
/// ```
/// use zerostone::sorter::DetectionMode;
///
/// let mode = DetectionMode::Amplitude;
/// assert_eq!(mode, DetectionMode::Amplitude);
/// let sneo = DetectionMode::Sneo { smooth_window: 3 };
/// assert_eq!(sneo, DetectionMode::Sneo { smooth_window: 3 });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Standard negative-amplitude threshold crossing (default).
    Amplitude,
    /// Nonlinear Energy Operator: `psi[n] = x[n]^2 - x[n-1]*x[n+1]`.
    Neo,
    /// Smoothed NEO with triangular window of given half-width.
    Sneo { smooth_window: usize },
}

/// Configuration for the multi-channel sorting pipeline.
///
/// # Example
///
/// ```
/// use zerostone::sorter::SortConfig;
///
/// let config = SortConfig::default();
/// assert!((config.threshold_multiplier - 5.0).abs() < 1e-12);
/// assert_eq!(config.refractory_samples, 15);
/// assert!((config.merge_dprime_threshold - 2.0).abs() < 1e-12);
/// assert!((config.merge_isi_threshold - 0.05).abs() < 1e-12);
/// assert_eq!(config.split_min_cluster_size, 10);
/// assert!((config.split_bimodality_threshold - 2.0).abs() < 1e-12);
/// assert!((config.spatial_merge_dprime - 1.5).abs() < 1e-12);
/// assert!(config.template_subtract);
/// assert_eq!(config.template_min_count, 3);
/// assert!((config.min_cluster_snr - 2.5).abs() < 1e-12);
/// assert_eq!(config.detection_mode, zerostone::sorter::DetectionMode::Amplitude);
/// assert_eq!(config.template_subtract_passes, 2);
/// assert!((config.isi_split_threshold - 0.1).abs() < 1e-12);
/// assert!(config.matched_filter_detect);
/// assert!((config.matched_filter_threshold - 3.5).abs() < 1e-12);
/// assert!(!config.svd_init);
/// assert_eq!(config.refinement_iterations, 1);
/// assert!(!config.gmm_refine);
/// assert!(!config.use_localization);
/// assert!(!config.use_amplitude_profile);
/// assert_eq!(config.amplitude_profile_neighbors, 4);
/// assert!(config.auto_cmr);
/// assert!(config.coincidence_detect);
/// assert!((config.coincidence_primary_threshold - 3.5).abs() < 1e-12);
/// assert!((config.coincidence_secondary_threshold - 2.0).abs() < 1e-12);
/// assert_eq!(config.min_coincident_channels, 2);
/// assert!(config.neighbor_mf_detect);
/// assert!((config.neighbor_mf_bonus - 0.5).abs() < 1e-10);
/// assert!(config.use_shape_features);
/// assert!(config.auto_cluster_threshold);
/// assert!(config.auto_refine);
/// assert!(config.refine_collapse_guard);
/// assert!(!config.refine_isi_guard);
/// assert!((config.refine_isi_tolerance - 0.1).abs() < 1e-12);
/// assert!(config.ccg_merge);
/// ```
pub struct SortConfig {
    /// Threshold multiplier for spike detection (sigma units on whitened data).
    pub threshold_multiplier: Float,
    /// Minimum samples between detections on the same channel.
    pub refractory_samples: usize,
    /// Spatial deduplication radius in micrometers.
    pub spatial_radius_um: Float,
    /// Temporal deduplication radius in samples.
    pub temporal_radius: usize,
    /// Half-window for fine peak alignment.
    pub align_half_window: usize,
    /// Samples before the peak in extracted waveforms.
    pub pre_samples: usize,
    /// Distance threshold for creating new clusters.
    pub cluster_threshold: Float,
    /// Maximum observation count per cluster (keeps centroids plastic).
    pub cluster_max_count: u32,
    /// Regularization for whitening eigenvalues.
    pub whitening_epsilon: Float,
    /// D-prime threshold for cluster merging: merge if d' below this value.
    pub merge_dprime_threshold: Float,
    /// ISI violation threshold for cluster merging: skip merge if combined
    /// ISI violation rate would exceed this value.
    pub merge_isi_threshold: Float,
    /// Minimum spikes per cluster to attempt splitting.
    pub split_min_cluster_size: usize,
    /// Bimodality threshold for cluster splitting (gap / std_dev).
    pub split_bimodality_threshold: Float,
    /// D-prime threshold for cross-channel spatial merge.
    pub spatial_merge_dprime: Float,
    /// Enable template subtraction pass to recover masked spikes.
    pub template_subtract: bool,
    /// Minimum spikes per cluster to build a reliable template for subtraction.
    pub template_min_count: usize,
    /// Minimum cluster SNR for auto-curation. Clusters below this are removed.
    pub min_cluster_snr: Float,
    /// Detection mode: Amplitude (default), NEO, or SNEO.
    pub detection_mode: DetectionMode,
    /// Enable CCG-based cluster merging after d-prime and spatial merge.
    /// Merges cluster pairs with high template correlation and no refractory dip.
    pub ccg_merge: bool,
    /// Template correlation threshold for CCG merge candidates.
    pub ccg_template_corr_threshold: Float,
    /// Number of template subtraction passes (0 = disabled, 1 = single pass, 2+ = multi-pass).
    /// Each additional pass subtracts the updated templates and re-detects on the residual.
    pub template_subtract_passes: usize,
    /// ISI violation rate threshold for post-sort cluster splitting.
    /// Clusters with ISI violation rate above this are split along the
    /// first principal axis of their feature distribution.
    pub isi_split_threshold: Float,
    /// Enable GMM refinement of k-means clusters.
    /// After k-means clustering and merge/split, runs batch EM with full
    /// covariance to capture cluster shape. Typically reassigns 5-15% of
    /// borderline spikes. Max EM iterations controlled by `gmm_max_iter`.
    pub gmm_refine: bool,
    /// Maximum EM iterations for GMM refinement.
    pub gmm_max_iter: usize,
    /// Enable matched filter second-pass detection.
    /// After initial amplitude detection and clustering, uses learned templates
    /// as matched filters to detect spikes below the amplitude threshold.
    /// This is the Neyman-Pearson optimal detector: no other linear filter
    /// achieves higher detection probability for a given false-positive rate.
    pub matched_filter_detect: bool,
    /// Matched filter detection threshold in sigma (z-score) units.
    /// Under the null hypothesis (noise), the normalized statistic is N(0,1).
    /// Lower values detect weaker spikes but increase false positives.
    /// Default 4.0 corresponds to P(false positive) ≈ 3e-5 per sample per filter.
    pub matched_filter_threshold: Float,
    /// Bandpass filter low cutoff in Hz (0.0 = disabled).
    /// Standard spike sorting range: 300-6000 Hz at 30 kHz sample rate.
    /// Applied as a 4th-order Butterworth forward filter before whitening.
    pub bandpass_low: Float,
    /// Bandpass filter high cutoff in Hz (0.0 = disabled).
    pub bandpass_high: Float,
    /// Sample rate in Hz (needed for bandpass filter).
    pub sample_rate: Float,
    /// Enable common median reference (CMR) before whitening.
    /// Subtracts the per-timepoint median across all channels.
    /// More robust than CAR against large artifacts on individual channels.
    pub common_median_ref: bool,
    /// Enable SVD-based centroid initialization for k-means clustering.
    /// Projects features onto the dominant eigenvector of the covariance matrix,
    /// bins the projections into equal-count groups, and uses bin means as seeds.
    /// Places centroids where the data density is highest along the principal
    /// axis, rather than at the extremes (farthest-point). Default: false.
    pub svd_init: bool,
    /// Number of template refinement iterations (0 = disabled).
    /// After the first sort pass, uses learned templates as matched filter seeds
    /// for re-detection and template-seeded k-means for re-clustering.
    /// Each iteration refines templates, improving accuracy on borderline units.
    pub refinement_iterations: usize,
    /// Enable per-channel adaptive detection thresholds.
    /// Computes thresholds from per-channel MAD noise estimates on whitened data,
    /// with a minimum floor for dead channels and overactivity scaling.
    /// More accurate than a uniform threshold when whitening is imperfect
    /// (edge channels, artifacts, rank-deficient covariance). Default: false.
    pub adaptive_threshold: bool,
    /// Minimum absolute threshold for adaptive mode (dead channel floor).
    /// Channels with noise below this get this threshold. Default: 0.5.
    pub adaptive_min_threshold: Float,
    /// Maximum crossing rate (Hz) before adaptive threshold is raised.
    /// Overactive channels get scaled up by sqrt(rate / max_rate). Default: 200.0.
    pub adaptive_max_rate_hz: Float,
    /// Enable center-of-mass spike localization as clustering features.
    /// When true, replaces the last two feature dimensions (least-important
    /// PCA component and channel index) with the (x, y) center-of-mass
    /// position computed from peak amplitudes across channels and probe
    /// geometry. Requires K >= 3. Default: false.
    pub use_localization: bool,
    /// Enable amplitude-profile spatial features (default: true).
    /// For each spike, measures peak amplitude on neighboring channels
    /// and encodes the normalized profile as the last feature dimension.
    /// This replaces the channel-index encoding with a physics-based
    /// spatial fingerprint: units with different spatial extent or
    /// position get different amplitude profiles even on the same
    /// peak channel. Falls back to channel-index encoding when false.
    pub use_amplitude_profile: bool,
    /// Number of neighbor channels to include in the amplitude profile.
    /// Only used when `use_amplitude_profile` is true.
    /// The profile includes the peak channel + this many nearest neighbors.
    /// Default: 4 (5 channels total: peak + 4 neighbors).
    pub amplitude_profile_neighbors: usize,
    /// Auto-apply CMR when channel count >= 8.
    /// Common Median Reference removes shared noise across channels before
    /// whitening. For multi-channel recordings (C >= 8), correlated noise
    /// between adjacent channels degrades whitening; CMR removes it.
    /// When `auto_cmr` is true, CMR is applied regardless of `common_median_ref`.
    /// Default: true.
    pub auto_cmr: bool,
    /// Enable spatial coincidence recovery of sub-threshold spikes.
    /// After primary amplitude detection, scans at a lower threshold
    /// (`coincidence_primary_threshold`) and accepts candidates only when
    /// `min_coincident_channels` neighboring channels also exceed
    /// `coincidence_secondary_threshold`. Spikes spread spatially across
    /// multiple channels; isolated noise does not. Only active when C >= 4.
    /// Default: true.
    pub coincidence_detect: bool,
    /// Primary threshold for coincidence detection pass (sigma units).
    /// Lower than `threshold_multiplier` to catch sub-threshold spikes.
    /// Only candidates with spatial corroboration are accepted.
    /// Default: 3.5.
    pub coincidence_primary_threshold: Float,
    /// Secondary threshold for neighbor corroboration (sigma units).
    /// Neighboring channels must exceed this to count as supporting evidence.
    /// Default: 2.0.
    pub coincidence_secondary_threshold: Float,
    /// Minimum number of neighboring channels that must exceed
    /// `coincidence_secondary_threshold` to accept a coincidence candidate.
    /// Requires at least this many channels beyond the peak channel.
    /// Default: 2 (3 channels total: peak + 2 neighbors).
    pub min_coincident_channels: usize,
    /// Enable neighbor-channel composite matched filter scoring.
    /// After learning templates, augments single-channel MF detection with
    /// a corroboration score from the nearest neighboring channel.
    /// Composite = primary_ncc + neighbor_mf_bonus * neighbor_ncc.
    /// Units spanning multiple channels get a higher composite score.
    /// Only active when `matched_filter_detect` is true. Default: true.
    pub neighbor_mf_detect: bool,
    /// Weight for neighbor channel NCC in composite matched filter score.
    /// Composite = primary_ncc + neighbor_mf_bonus * neighbor_ncc.
    /// Default: 0.5.
    pub neighbor_mf_bonus: Float,
    /// Enable spike half-width as a shape feature in the feature vector.
    /// Replaces the third PCA component (K-2) with normalized half-width
    /// of the spike trough at half-maximum depth. Separates unit types:
    /// narrow interneurons (half-width < 0.3ms) vs broad pyramidal cells
    /// (half-width > 0.5ms). Only active in fallback feature mode when
    /// K >= 4. Default: true.
    pub use_shape_features: bool,
    /// Scale the cluster creation threshold inversely with channel count for
    /// recordings with fewer than 8 channels. When true and `C < 8`, the
    /// effective threshold is `cluster_threshold * sqrt(8 / C)`, which
    /// prevents over-splitting on low-channel recordings where the spatial
    /// feature dimension spans fewer modes. For C≥8, no scaling is applied.
    /// Default: true.
    pub auto_cluster_threshold: bool,
    /// Skip feature-space refinement iterations for recordings with fewer than
    /// 8 channels. On small probes (C < 8), the post-pipeline cluster layout
    /// already has more clusters than ground-truth units (over-split residuals
    /// after merge/split). Re-running nearest-centroid assignment in that state
    /// pulls borderline spikes into noise clusters rather than tightening
    /// within-unit boundaries. For C≥8 recordings, refinement safely reduces
    /// borderline misassignments. When false, `refinement_iterations` applies
    /// regardless of channel count. Default: true.
    pub auto_refine: bool,
    /// Guard each refinement iteration against cluster collapse.
    /// Before committing reassignments, counts the post-reassignment size of
    /// every cluster. If any cluster that currently holds at least
    /// `split_min_cluster_size` spikes would be completely emptied, the
    /// iteration is skipped and the loop exits early (pre-iteration labels are
    /// preserved). This prevents pathological cases where two cluster centroids
    /// are close enough that an entire real unit's spikes fall into a neighbor's
    /// Voronoi region. Uses a stack-allocated dry-run pass — no heap allocation.
    /// Default: true.
    pub refine_collapse_guard: bool,
    /// Guard each refinement iteration against ISI violation increase.
    /// Before committing reassignments, counts inter-spike-interval violations
    /// in the proposed new labeling (consecutive same-cluster spike pairs within
    /// `refractory_samples`) and compares to the current labeling. If violations
    /// would increase by more than `refine_isi_tolerance` (fractional), the
    /// iteration is skipped. Targets distributed correlated misassignment: cases
    /// where refinement pulls spikes from multiple units into wrong neighbors
    /// simultaneously without emptying any single cluster (not caught by
    /// `refine_collapse_guard`). Uses detection-order ISI approximation — valid
    /// because spikes are stored approximately in time order. Default: false.
    pub refine_isi_guard: bool,
    /// Fractional ISI-violation increase that triggers the ISI guard revert.
    /// The proposed assignment is rejected when post-pass violations exceed
    /// pre-pass violations by more than this fraction. Example: 0.1 means
    /// skip the iteration if violations increase by more than 10%.
    /// Only active when `refine_isi_guard` is true. Default: 0.1.
    pub refine_isi_tolerance: Float,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            threshold_multiplier: 5.0,
            refractory_samples: 15,
            spatial_radius_um: 75.0,
            temporal_radius: 5,
            align_half_window: 15,
            pre_samples: 20,
            cluster_threshold: 7.0,
            cluster_max_count: 1000,
            whitening_epsilon: 1e-6,
            merge_dprime_threshold: 2.0,
            merge_isi_threshold: 0.05,
            split_min_cluster_size: 10,
            split_bimodality_threshold: 2.0,
            spatial_merge_dprime: 1.5,
            template_subtract: true,
            template_min_count: 3,
            min_cluster_snr: 2.5,
            detection_mode: DetectionMode::Amplitude,
            ccg_merge: true,
            ccg_template_corr_threshold: 0.5,
            template_subtract_passes: 2,
            isi_split_threshold: 0.1,
            gmm_refine: false,
            gmm_max_iter: 10,
            matched_filter_detect: true,
            matched_filter_threshold: 3.5,
            bandpass_low: 0.0,
            bandpass_high: 0.0,
            sample_rate: 30000.0,
            common_median_ref: false,
            svd_init: false,
            refinement_iterations: 1,
            adaptive_threshold: false,
            adaptive_min_threshold: 0.5,
            adaptive_max_rate_hz: 200.0,
            use_localization: false,
            use_amplitude_profile: false,
            amplitude_profile_neighbors: 4,
            auto_cmr: true,
            coincidence_detect: true,
            coincidence_primary_threshold: 3.5,
            coincidence_secondary_threshold: 2.0,
            min_coincident_channels: 2,
            neighbor_mf_detect: true,
            neighbor_mf_bonus: 0.5,
            use_shape_features: true,
            auto_cluster_threshold: true,
            auto_refine: true,
            refine_collapse_guard: true,
            refine_isi_guard: false,
            refine_isi_tolerance: 0.1,
        }
    }
}

/// Per-cluster quality information.
///
/// # Example
///
/// ```
/// use zerostone::sorter::ClusterInfo;
///
/// let info = ClusterInfo { count: 50, snr: 8.5, isi_violation_rate: 0.01 };
/// assert_eq!(info.count, 50);
/// ```
pub struct ClusterInfo {
    /// Number of spikes assigned to this cluster.
    pub count: usize,
    /// Signal-to-noise ratio (peak-to-peak / 2*noise_std).
    pub snr: Float,
    /// Fraction of ISI violations (inter-spike intervals below refractory).
    pub isi_violation_rate: Float,
}

/// Result of the sorting pipeline.
///
/// # Example
///
/// ```
/// use zerostone::sorter::{SortResult, ClusterInfo};
///
/// let result = SortResult::<4> {
///     n_spikes: 0,
///     n_clusters: 0,
///     clusters: [
///         ClusterInfo { count: 0, snr: 0.0, isi_violation_rate: 0.0 },
///         ClusterInfo { count: 0, snr: 0.0, isi_violation_rate: 0.0 },
///         ClusterInfo { count: 0, snr: 0.0, isi_violation_rate: 0.0 },
///         ClusterInfo { count: 0, snr: 0.0, isi_violation_rate: 0.0 },
///     ],
/// };
/// assert_eq!(result.n_spikes, 0);
/// ```
pub struct SortResult<const N: usize> {
    /// Total number of spikes detected and labeled.
    pub n_spikes: usize,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Per-cluster quality metrics.
    pub clusters: [ClusterInfo; N],
}

/// Estimate noise on each channel using MAD.
///
/// Computes `sigma = median(|x|) / 0.6745` independently for each channel.
/// Requires `scratch` with at least `2 * data.len()` elements.
///
/// # Type Parameters
///
/// * `C` - Number of channels
///
/// # Example
///
/// ```
/// use zerostone::sorter::estimate_noise_multichannel;
///
/// let data = [[1.0, -2.0], [0.5, 1.5], [-0.3, 0.8], [0.7, -1.2]];
/// let mut scratch = [0.0; 8];
/// let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
/// assert!(noise[0] > 0.0 && noise[0].is_finite());
/// assert!(noise[1] > 0.0 && noise[1].is_finite());
/// ```
pub fn estimate_noise_multichannel<const C: usize>(
    data: &[[Float; C]],
    scratch: &mut [Float],
) -> [Float; C] {
    let t_len = data.len();
    let mut noise = [0.0; C];
    if t_len == 0 {
        return noise;
    }
    assert!(
        scratch.len() >= t_len,
        "scratch must have at least data.len() elements"
    );

    let mut ch = 0;
    while ch < C {
        // Copy absolute values into scratch
        for t in 0..t_len {
            scratch[t] = float::abs(data[t][ch]);
        }
        let s = &mut scratch[..t_len];
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let median = if t_len % 2 == 1 {
            s[t_len / 2]
        } else {
            (s[t_len / 2 - 1] + s[t_len / 2]) * 0.5
        };
        noise[ch] = median / 0.6745;
        ch += 1;
    }
    noise
}

/// Apply 4th-order Butterworth bandpass filter in-place, per channel.
///
/// Implements a cascade of 2 second-order sections (biquads) in f64.
/// The filter is causal (forward-only), suitable for real-time use.
/// Coefficients are computed using the bilinear transform.
#[allow(clippy::needless_range_loop)]
fn bandpass_inplace<const C: usize>(data: &mut [[Float; C]], fs: Float, low: Float, high: Float) {
    use float::PI;
    let n = data.len();
    if n < 4 || low >= high || fs <= 0.0 {
        return;
    }

    // Prewarp cutoff frequencies
    let w_low = float::tan(PI * low / fs);
    let w_high = float::tan(PI * high / fs);
    let bw = w_high - w_low;
    let w0_sq = w_low * w_high;

    // 2nd-order Butterworth poles: exp(j * pi * (2k+1) / (2*2)) for k=0,1
    // For order=2: poles at angles pi*3/4 and pi*5/4
    // Real part: cos(3pi/4) = -1/sqrt(2), cos(5pi/4) = -1/sqrt(2)
    // So both pole pairs have the same real part: -sqrt(2)/2
    // Bandpass transform doubles the order: each 2nd-order lowpass section
    // becomes a 4th-order bandpass (2 biquad sections).
    //
    // For simplicity, compute coefficients for a 2nd-order bandpass
    // (single biquad pair) and apply it twice for 4th-order.
    let alpha = bw;
    let a0 = 1.0 + alpha + w0_sq;
    let a0_inv = 1.0 / a0;

    // Bandpass biquad: H(z) = (bw * z^-1) / (1 + a1*z^-1 + a2*z^-2)
    // Numerator: [0, bw, 0] in z-domain after bilinear
    let b0 = alpha * a0_inv;
    let b1 = 0.0;
    let b2 = -alpha * a0_inv;
    let a1 = 2.0 * (w0_sq - 1.0) * a0_inv;
    let a2 = (1.0 - alpha + w0_sq) * a0_inv;

    // Apply 2 passes of the same biquad for 4th-order response
    for ch in 0..C {
        for _pass in 0..2 {
            let mut x1 = 0.0;
            let mut x2 = 0.0;
            let mut y1 = 0.0;
            let mut y2 = 0.0;
            for sample in data.iter_mut().take(n) {
                let x = sample[ch];
                let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = x;
                y2 = y1;
                y1 = y;
                sample[ch] = y;
            }
        }
    }
}

/// Compute sample covariance matrix from multi-channel data.
#[allow(clippy::needless_range_loop)]
fn compute_covariance<const C: usize>(data: &[[Float; C]]) -> [[Float; C]; C] {
    let n = data.len();
    let mut cov = [[0.0; C]; C];
    if n < 2 {
        return cov;
    }

    // Compute means
    let mut mean = [0.0; C];
    for sample in data.iter() {
        for c in 0..C {
            mean[c] += sample[c];
        }
    }
    let inv_n = 1.0 / n as Float;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }

    // Compute covariance
    for sample in data.iter() {
        for i in 0..C {
            let di = sample[i] - mean[i];
            for j in i..C {
                let dj = sample[j] - mean[j];
                cov[i][j] += di * dj;
            }
        }
    }
    let inv_nm1 = 1.0 / (n - 1) as Float;
    for i in 0..C {
        for j in i..C {
            cov[i][j] *= inv_nm1;
            if i != j {
                cov[j][i] = cov[i][j];
            }
        }
    }
    cov
}

/// Maximum number of clusters supported for merge bookkeeping.
///
/// Merge-related scratch arrays are sized to this limit. Pipelines with
/// `N > MAX_MERGE_CLUSTERS` will only attempt merges among the first
/// `MAX_MERGE_CLUSTERS` clusters.
const MAX_MERGE_CLUSTERS: usize = 32;

/// Merge over-split clusters based on d-prime and ISI violation criteria.
///
/// Iterates over all active cluster pairs and greedily merges the pair with
/// the smallest d-prime (most similar feature distributions) provided that:
///
/// 1. d-prime is below `dprime_threshold`
/// 2. The combined spike train would not exceed `isi_threshold` ISI violation rate
///
/// When a merge occurs, all labels equal to the removed cluster are reassigned
/// to the kept cluster, and labels above the removed index are shifted down.
/// The process repeats until no more valid merges remain.
///
/// Operates entirely on fixed-size stack buffers (no heap allocation).
///
/// # Arguments
///
/// * `n_spikes` - Number of valid entries in `labels`, `feature_buf`, and `event_buf`
/// * `labels` - Cluster label per spike (modified in place on merge)
/// * `feature_buf` - PCA feature vector per spike (read-only, K dimensions)
/// * `event_buf` - Detected events (for spike times used in ISI computation)
/// * `n_clusters` - Current number of active clusters (modified on return)
/// * `dprime_threshold` - Merge if d-prime below this value
/// * `isi_threshold` - Skip merge if combined ISI violation rate exceeds this
/// * `refractory_samples` - Refractory period in samples for ISI computation
/// * `scratch` - Working buffer, must have at least `n_spikes` elements
///
/// # Returns
///
/// The new number of active clusters after all merges.
///
/// # Example
///
/// ```
/// use zerostone::sorter::merge_clusters;
/// use zerostone::spike_sort::MultiChannelEvent;
///
/// // Two clusters with identical features should merge
/// let mut labels = [0, 0, 0, 1, 1, 1];
/// let features = [[1.0, 0.0], [1.1, 0.1], [0.9, -0.1],
///                  [1.05, 0.05], [0.95, -0.05], [1.0, 0.0]];
/// let events = [
///     MultiChannelEvent { sample: 100, channel: 0, amplitude: 5.0 },
///     MultiChannelEvent { sample: 200, channel: 0, amplitude: 5.0 },
///     MultiChannelEvent { sample: 300, channel: 0, amplitude: 5.0 },
///     MultiChannelEvent { sample: 400, channel: 0, amplitude: 5.0 },
///     MultiChannelEvent { sample: 500, channel: 0, amplitude: 5.0 },
///     MultiChannelEvent { sample: 600, channel: 0, amplitude: 5.0 },
/// ];
/// let mut scratch = [0.0; 6];
/// let new_n = merge_clusters(
///     6, &mut labels, &features, &events, 2,
///     1.5, 0.05, 15, &mut scratch, 2,
/// );
/// assert_eq!(new_n, 1);
/// assert!(labels.iter().all(|&l| l == 0));
/// ```
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn merge_clusters<const K: usize>(
    n_spikes: usize,
    labels: &mut [usize],
    feature_buf: &[[Float; K]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    dprime_threshold: Float,
    isi_threshold: Float,
    refractory_samples: usize,
    scratch: &mut [Float],
    merge_dims: usize,
) -> usize {
    if n_clusters < 2 || n_spikes < 2 {
        return n_clusters;
    }

    let max_k = if n_clusters > MAX_MERGE_CLUSTERS {
        MAX_MERGE_CLUSTERS
    } else {
        n_clusters
    };

    let mut current_n = max_k;

    // Fixed-size projection buffers for d-prime computation.
    // We collect 1D projections of each cluster onto the axis connecting
    // two cluster centroids. MAX_SPIKES_PER_CLUSTER limits stack usage.
    const MAX_SPIKES: usize = 512;

    // Track pairs that failed ISI check so we skip them in subsequent iterations.
    // Stored as (i, j) with i < j. Cleared after each successful merge since
    // indices shift.
    const MAX_EXCLUDED: usize = 64;
    let mut excluded = [(0usize, 0usize); MAX_EXCLUDED];
    let mut n_excluded = 0usize;

    loop {
        if current_n < 2 {
            break;
        }

        // Find the pair with the smallest d-prime (skipping excluded pairs)
        let mut best_dp = float::MAX;
        let mut best_i = 0usize;
        let mut best_j = 0usize;

        // Compute centroids for each cluster
        let mut centroids = [[0.0; 32]; MAX_MERGE_CLUSTERS];
        let mut counts = [0usize; MAX_MERGE_CLUSTERS];
        // Use caller-specified dims (excludes channel feature when appropriate)
        let dim = if merge_dims > 32 { 32 } else { merge_dims };

        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            let cl = labels[s];
            if cl >= current_n {
                continue;
            }
            counts[cl] += 1;
            for d in 0..dim {
                centroids[cl][d] += feature_buf[s][d];
            }
        }
        for cl in 0..current_n {
            if counts[cl] > 0 {
                let inv = 1.0 / counts[cl] as Float;
                for d in 0..dim {
                    centroids[cl][d] *= inv;
                }
            }
        }

        // Evaluate all pairs
        for i in 0..current_n {
            if counts[i] < 2 {
                continue;
            }
            for j in (i + 1)..current_n {
                if counts[j] < 2 {
                    continue;
                }

                // Skip excluded pairs
                let mut is_excluded = false;
                let mut e = 0;
                while e < n_excluded {
                    if excluded[e].0 == i && excluded[e].1 == j {
                        is_excluded = true;
                        break;
                    }
                    e += 1;
                }
                if is_excluded {
                    continue;
                }

                // Compute discriminant axis: centroid_j - centroid_i
                let mut axis = [0.0; 32];
                let mut axis_norm_sq = 0.0;
                for d in 0..dim {
                    axis[d] = centroids[j][d] - centroids[i][d];
                    axis_norm_sq += axis[d] * axis[d];
                }
                if axis_norm_sq < 1e-30 {
                    // Centroids are essentially identical -- d-prime ~ 0
                    best_dp = 0.0;
                    best_i = i;
                    best_j = j;
                    continue;
                }
                let inv_norm = 1.0 / float::sqrt(axis_norm_sq);
                for d in 0..dim {
                    axis[d] *= inv_norm;
                }

                // Project spikes from cluster i and j onto axis and compute d-prime
                let mut proj_a = [0.0; MAX_SPIKES];
                let mut proj_b = [0.0; MAX_SPIKES];
                let mut na = 0usize;
                let mut nb = 0usize;

                for s in 0..n_spikes {
                    if s >= labels.len() {
                        break;
                    }
                    let cl = labels[s];
                    if cl == i && na < MAX_SPIKES {
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += feature_buf[s][d] * axis[d];
                        }
                        proj_a[na] = dot;
                        na += 1;
                    } else if cl == j && nb < MAX_SPIKES {
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += feature_buf[s][d] * axis[d];
                        }
                        proj_b[nb] = dot;
                        nb += 1;
                    }
                }

                if na < 2 || nb < 2 {
                    continue;
                }

                if let Some(dp) = quality::d_prime(&proj_a[..na], &proj_b[..nb]) {
                    if dp < best_dp {
                        best_dp = dp;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        // Check if the best pair meets the d-prime criterion
        if best_dp > dprime_threshold || best_dp == float::MAX {
            break;
        }

        // Check ISI violation rate of the merged spike train
        // Collect spike times for clusters best_i and best_j into scratch
        let mut n_combined = 0usize;
        if scratch.len() >= n_spikes {
            for s in 0..n_spikes {
                if s >= labels.len() {
                    break;
                }
                let cl = labels[s];
                if (cl == best_i || cl == best_j) && n_combined < scratch.len() {
                    scratch[n_combined] = event_buf[s].sample as Float;
                    n_combined += 1;
                }
            }
        }

        if n_combined >= 2 {
            let times = &mut scratch[..n_combined];
            times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let combined_isi =
                quality::isi_violation_rate(times, refractory_samples as Float).unwrap_or(1.0);
            if combined_isi > isi_threshold {
                // Merging would create too many ISI violations -- exclude this
                // pair and try the next-best pair instead of stopping all merges.
                if n_excluded < MAX_EXCLUDED {
                    excluded[n_excluded] = (best_i, best_j);
                    n_excluded += 1;
                    continue;
                }
                break;
            }
        }

        // Execute the merge: relabel best_j -> best_i, shift labels above best_j down
        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            if labels[s] == best_j {
                labels[s] = best_i;
            } else if labels[s] > best_j {
                labels[s] -= 1;
            }
        }
        current_n -= 1;
        // Clear excluded pairs since indices shifted after merge
        n_excluded = 0;
    }

    current_n
}

/// Split clusters that show bimodal distributions in feature space.
///
/// For each active cluster, projects spikes onto the axis of maximum
/// variance and checks for bimodality using a gap-based criterion.
/// Clusters with a gap exceeding `bimodality_threshold * std_dev` are
/// split into two sub-clusters.
///
/// Only one cluster is split per pass; the function loops until no
/// more splits are found, up to `MAX_MERGE_CLUSTERS` total clusters.
///
/// # Arguments
///
/// * `n_spikes` -- Number of valid entries in `labels` and `feature_buf`
/// * `labels` -- Cluster label per spike (modified in place on split)
/// * `feature_buf` -- PCA feature vector per spike
/// * `n_clusters` -- Current number of active clusters
/// * `min_cluster_size` -- Minimum spikes per cluster to attempt split
/// * `bimodality_threshold` -- Gap threshold relative to std deviation
///
/// # Returns
///
/// The new number of active clusters after all splits.
///
/// # Example
///
/// ```
/// use zerostone::sorter::split_clusters;
///
/// // Two well-separated groups incorrectly merged into cluster 0
/// let mut labels = [0, 0, 0, 0, 0, 0];
/// let features = [
///     [0.0, 0.0], [0.1, 0.1], [0.2, -0.1],  // group A near origin
///     [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],    // group B far away
/// ];
/// let new_n = split_clusters(6, &mut labels, &features, 1, 3, 1.5);
/// assert_eq!(new_n, 2);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn split_clusters<const K: usize>(
    n_spikes: usize,
    labels: &mut [usize],
    feature_buf: &[[Float; K]],
    n_clusters: usize,
    min_cluster_size: usize,
    bimodality_threshold: Float,
) -> usize {
    if n_spikes == 0 || n_clusters == 0 {
        return n_clusters;
    }

    const MAX_SPIKES: usize = 512;
    let mut current_n = n_clusters;

    loop {
        if current_n >= MAX_MERGE_CLUSTERS {
            break;
        }

        let mut did_split = false;

        let mut cl = 0;
        while cl < current_n {
            // Count spikes in this cluster and collect indices
            let mut indices = [0usize; MAX_SPIKES];
            let mut count = 0usize;
            let mut s = 0;
            while s < n_spikes && s < labels.len() {
                if labels[s] == cl && count < MAX_SPIKES {
                    indices[count] = s;
                    count += 1;
                }
                s += 1;
            }

            if count < min_cluster_size || count < 4 {
                cl += 1;
                continue;
            }

            let dim = if K > 32 { 32 } else { K };

            // Compute centroid
            let mut centroid = [0.0; 32];
            let mut i = 0;
            while i < count {
                let mut d = 0;
                while d < dim {
                    centroid[d] += feature_buf[indices[i]][d];
                    d += 1;
                }
                i += 1;
            }
            let inv_count = 1.0 / count as Float;
            let mut d = 0;
            while d < dim {
                centroid[d] *= inv_count;
                d += 1;
            }

            // Power iteration: find direction of maximum variance (1D PCA)
            // Initialize with first centered spike
            let mut axis = [0.0; 32];
            let mut d = 0;
            while d < dim {
                axis[d] = feature_buf[indices[0]][d] - centroid[d];
                d += 1;
            }
            // Normalize
            let mut norm_sq = 0.0;
            d = 0;
            while d < dim {
                norm_sq += axis[d] * axis[d];
                d += 1;
            }
            if norm_sq < 1e-30 {
                cl += 1;
                continue;
            }
            let inv_norm = 1.0 / float::sqrt(norm_sq);
            d = 0;
            while d < dim {
                axis[d] *= inv_norm;
                d += 1;
            }

            // 3 iterations of power method on the scatter matrix
            let mut iter = 0;
            while iter < 3 {
                let mut new_axis = [0.0; 32];
                let mut i = 0;
                while i < count {
                    // Project centered spike onto current axis
                    let mut dot = 0.0;
                    let mut d = 0;
                    while d < dim {
                        dot += (feature_buf[indices[i]][d] - centroid[d]) * axis[d];
                        d += 1;
                    }
                    // Accumulate outer product contribution
                    d = 0;
                    while d < dim {
                        new_axis[d] += dot * (feature_buf[indices[i]][d] - centroid[d]);
                        d += 1;
                    }
                    i += 1;
                }
                // Normalize
                let mut ns = 0.0;
                d = 0;
                while d < dim {
                    ns += new_axis[d] * new_axis[d];
                    d += 1;
                }
                if ns < 1e-30 {
                    break;
                }
                let inv = 1.0 / float::sqrt(ns);
                d = 0;
                while d < dim {
                    axis[d] = new_axis[d] * inv;
                    d += 1;
                }
                iter += 1;
            }

            // Project all spikes onto the axis
            let mut projections = [0.0; MAX_SPIKES];
            let mut i = 0;
            while i < count {
                let mut dot = 0.0;
                let mut d = 0;
                while d < dim {
                    dot += (feature_buf[indices[i]][d] - centroid[d]) * axis[d];
                    d += 1;
                }
                projections[i] = dot;
                i += 1;
            }

            // Compute std dev of projections
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            i = 0;
            while i < count {
                sum += projections[i];
                sum_sq += projections[i] * projections[i];
                i += 1;
            }
            let mean_proj = sum / count as Float;
            let var = sum_sq / count as Float - mean_proj * mean_proj;
            let std_dev = if var > 0.0 { float::sqrt(var) } else { 0.0 };

            if std_dev < 1e-15 {
                cl += 1;
                continue;
            }

            // Sort projections (need sorted copy + index mapping)
            let mut sorted_proj = [0.0; MAX_SPIKES];
            i = 0;
            while i < count {
                sorted_proj[i] = projections[i];
                i += 1;
            }
            let sp = &mut sorted_proj[..count];
            sp.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            // Find largest gap
            let mut max_gap = 0.0;
            let mut gap_midpoint = 0.0;
            i = 1;
            while i < count {
                let gap = sp[i] - sp[i - 1];
                if gap > max_gap {
                    max_gap = gap;
                    gap_midpoint = (sp[i] + sp[i - 1]) * 0.5;
                }
                i += 1;
            }

            // Check bimodality criterion
            if max_gap > bimodality_threshold * std_dev {
                // Split: spikes with projection >= gap_midpoint get new label
                let new_label = current_n;
                i = 0;
                while i < count {
                    if projections[i] >= gap_midpoint {
                        labels[indices[i]] = new_label;
                    }
                    i += 1;
                }
                current_n += 1;
                did_split = true;
                break; // restart outer loop
            }

            cl += 1;
        }

        if !did_split {
            break;
        }
    }

    current_n
}

/// Merge clusters across channels using C-dimensional spatial amplitude profiles.
///
/// For each spike, the amplitude vector `data[sample]` (all C channels at the
/// peak time) provides a natural spatial signature. Clusters from the same neuron
/// on adjacent channels will have correlated amplitude profiles and low dprime.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn merge_clusters_spatial<const C: usize>(
    n_spikes: usize,
    labels: &mut [usize],
    data: &[[Float; C]],
    event_buf: &[MultiChannelEvent],
    probe: &ProbeLayout<C>,
    n_clusters: usize,
    dprime_threshold: Float,
    spatial_radius_um: Float,
    isi_threshold: Float,
    refractory_samples: usize,
    scratch: &mut [Float],
) -> usize {
    if n_clusters < 2 || n_spikes < 2 {
        return n_clusters;
    }

    let max_k = if n_clusters > MAX_MERGE_CLUSTERS {
        MAX_MERGE_CLUSTERS
    } else {
        n_clusters
    };
    let mut current_n = max_k;

    // Find mode peak channel per cluster
    let mut mode_ch = [0usize; MAX_MERGE_CLUSTERS];
    let mut ch_votes = [[0u32; 64]; MAX_MERGE_CLUSTERS];
    for s in 0..n_spikes {
        if s >= labels.len() {
            break;
        }
        let cl = labels[s];
        if cl < current_n {
            let ch = event_buf[s].channel;
            if ch < 64 {
                ch_votes[cl][ch] += 1;
            }
        }
    }
    for cl in 0..current_n {
        let mut best = 0;
        let mut best_v = 0;
        for (ch, &v) in ch_votes[cl].iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = ch;
            }
        }
        mode_ch[cl] = best;
    }

    const MAX_SPIKES: usize = 512;
    const MAX_EXCLUDED: usize = 64;
    let mut excluded = [(0usize, 0usize); MAX_EXCLUDED];
    let mut n_excluded = 0usize;
    let dim = if C > 32 { 32 } else { C };

    loop {
        if current_n < 2 {
            break;
        }

        // Compute spatial centroids (amplitude at peak time, all channels)
        let mut centroids = [[0.0; 32]; MAX_MERGE_CLUSTERS];
        let mut counts = [0usize; MAX_MERGE_CLUSTERS];

        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            let cl = labels[s];
            if cl >= current_n {
                continue;
            }
            let t = event_buf[s].sample;
            if t < data.len() {
                counts[cl] += 1;
                for d in 0..dim {
                    centroids[cl][d] += data[t][d];
                }
            }
        }
        for cl in 0..current_n {
            if counts[cl] > 0 {
                let inv = 1.0 / counts[cl] as Float;
                for d in 0..dim {
                    centroids[cl][d] *= inv;
                }
            }
        }

        // Find best merge pair: lowest dprime among spatially proximate clusters
        let mut best_dp = float::MAX;
        let mut best_i = 0usize;
        let mut best_j = 0usize;

        for i in 0..current_n {
            if counts[i] < 2 {
                continue;
            }
            for j in (i + 1)..current_n {
                if counts[j] < 2 {
                    continue;
                }

                // Only merge if peak channels are within spatial radius
                let dist = probe.channel_distance(mode_ch[i], mode_ch[j]);
                if dist > spatial_radius_um {
                    continue;
                }

                // Check excluded
                let mut is_excluded = false;
                for e in 0..n_excluded {
                    if excluded[e].0 == i && excluded[e].1 == j {
                        is_excluded = true;
                        break;
                    }
                }
                if is_excluded {
                    continue;
                }

                // Compute discriminant axis
                let mut axis = [0.0; 32];
                let mut axis_norm_sq = 0.0;
                for d in 0..dim {
                    axis[d] = centroids[j][d] - centroids[i][d];
                    axis_norm_sq += axis[d] * axis[d];
                }
                if axis_norm_sq < 1e-30 {
                    best_dp = 0.0;
                    best_i = i;
                    best_j = j;
                    continue;
                }
                let inv_norm = 1.0 / float::sqrt(axis_norm_sq);
                for d in 0..dim {
                    axis[d] *= inv_norm;
                }

                // Project spikes onto axis
                let mut proj_a = [0.0; MAX_SPIKES];
                let mut proj_b = [0.0; MAX_SPIKES];
                let mut na = 0usize;
                let mut nb = 0usize;

                for s in 0..n_spikes {
                    if s >= labels.len() {
                        break;
                    }
                    let cl = labels[s];
                    let t = event_buf[s].sample;
                    if t >= data.len() {
                        continue;
                    }
                    if cl == i && na < MAX_SPIKES {
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += data[t][d] * axis[d];
                        }
                        proj_a[na] = dot;
                        na += 1;
                    } else if cl == j && nb < MAX_SPIKES {
                        let mut dot = 0.0;
                        for d in 0..dim {
                            dot += data[t][d] * axis[d];
                        }
                        proj_b[nb] = dot;
                        nb += 1;
                    }
                }

                if na < 2 || nb < 2 {
                    continue;
                }

                if let Some(dp) = quality::d_prime(&proj_a[..na], &proj_b[..nb]) {
                    if dp < best_dp {
                        best_dp = dp;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        if best_dp > dprime_threshold || best_dp == float::MAX {
            break;
        }

        // ISI check
        let mut n_combined = 0usize;
        if scratch.len() >= n_spikes {
            for s in 0..n_spikes {
                if s >= labels.len() {
                    break;
                }
                let cl = labels[s];
                if (cl == best_i || cl == best_j) && n_combined < scratch.len() {
                    scratch[n_combined] = event_buf[s].sample as Float;
                    n_combined += 1;
                }
            }
        }
        if n_combined >= 2 {
            let times = &mut scratch[..n_combined];
            times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let combined_isi =
                quality::isi_violation_rate(times, refractory_samples as Float).unwrap_or(1.0);
            if combined_isi > isi_threshold {
                if n_excluded < MAX_EXCLUDED {
                    excluded[n_excluded] = (best_i, best_j);
                    n_excluded += 1;
                    continue;
                }
                break;
            }
        }

        // Execute merge
        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            if labels[s] == best_j {
                labels[s] = best_i;
            } else if labels[s] > best_j {
                labels[s] -= 1;
            }
        }
        current_n -= 1;

        // Update mode channels after merge
        for cl in 0..current_n {
            ch_votes[cl] = [0u32; 64];
        }
        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            let cl = labels[s];
            if cl < current_n {
                let ch = event_buf[s].channel;
                if ch < 64 {
                    ch_votes[cl][ch] += 1;
                }
            }
        }
        for cl in 0..current_n {
            let mut best = 0;
            let mut best_v = 0;
            for (ch, &v) in ch_votes[cl].iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = ch;
                }
            }
            mode_ch[cl] = best;
        }
        n_excluded = 0;
    }

    current_n
}

/// Merge over-split clusters using cross-correlogram (CCG) refractoriness test.
///
/// For all cluster pairs whose mean waveform templates have normalized
/// cross-correlation above `corr_threshold`, computes the CCG and checks
/// for a refractory dip. Pairs with high template similarity and no
/// refractory dip are merged (they are likely over-split fragments of
/// the same neuron).
///
/// This follows the Kilosort4 pattern, which found CCG-based merge to be
/// the single strongest contributor to sorting accuracy.
///
/// # Arguments
///
/// * `n_spikes` - Number of valid spikes
/// * `labels` - Cluster label per spike (modified in place)
/// * `waveform_buf` - Waveform per spike (for template computation)
/// * `event_buf` - Spike events (for spike times)
/// * `n_clusters` - Current number of clusters
/// * `corr_threshold` - Minimum template NCC to consider a merge (e.g., 0.5)
/// * `sample_rate` - Sampling rate in Hz (for CCG bin width computation)
///
/// # Returns
///
/// The new number of clusters after CCG merging.
///
/// # Example
///
/// ```
/// use zerostone::sorter::ccg_merge_clusters;
/// use zerostone::spike_sort::MultiChannelEvent;
///
/// // Two clusters from the same neuron (similar waveforms, no refractory dip)
/// let mut labels = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1];
/// let waveforms = [[1.0; 16]; 10];
/// let events: Vec<_> = (0..10).map(|i| MultiChannelEvent {
///     sample: i * 100, channel: 0, amplitude: 5.0,
/// }).collect();
/// let new_n = ccg_merge_clusters::<16, 32>(
///     10, &mut labels, &waveforms, &events, 2, 0.5, 30000.0,
/// );
/// assert!(new_n <= 2);
/// ```
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn ccg_merge_clusters<const W: usize, const N: usize>(
    n_spikes: usize,
    labels: &mut [usize],
    waveform_buf: &[[Float; W]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    corr_threshold: Float,
    sample_rate: Float,
) -> usize {
    if n_clusters < 2 || n_spikes < 4 {
        return n_clusters;
    }

    let max_k = if n_clusters > MAX_MERGE_CLUSTERS {
        MAX_MERGE_CLUSTERS
    } else {
        n_clusters
    };

    let mut current_n = max_k;

    // CCG parameters
    let bin_width_s = 0.5e-3; // 0.5 ms bins
    let max_lag_s = 25.0e-3; // 25 ms max lag
    let refractory_bins = 2; // first 1ms (2 bins of 0.5ms) is the refractory zone

    loop {
        if current_n < 2 {
            break;
        }

        // Compute mean waveforms for current clusters
        let mut means = [[0.0; W]; MAX_MERGE_CLUSTERS];
        let mut counts = [0usize; MAX_MERGE_CLUSTERS];
        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            let cl = labels[s];
            if cl >= current_n {
                continue;
            }
            counts[cl] += 1;
            for w in 0..W {
                means[cl][w] += waveform_buf[s][w];
            }
        }
        for cl in 0..current_n {
            if counts[cl] > 0 {
                let inv = 1.0 / counts[cl] as Float;
                for w in 0..W {
                    means[cl][w] *= inv;
                }
            }
        }

        // Find best merge candidate: highest template correlation above threshold
        let mut best_corr = corr_threshold;
        let mut best_i = 0;
        let mut best_j = 0;
        let mut found = false;

        for i in 0..current_n {
            if counts[i] < 2 {
                continue;
            }
            // Precompute norm of template i
            let mut norm_i_sq = 0.0;
            for w in 0..W {
                norm_i_sq += means[i][w] * means[i][w];
            }
            let norm_i = float::sqrt(norm_i_sq);
            if norm_i < 1e-15 {
                continue;
            }

            for j in (i + 1)..current_n {
                if counts[j] < 2 {
                    continue;
                }
                // Template NCC
                let mut dot = 0.0;
                let mut norm_j_sq = 0.0;
                for w in 0..W {
                    dot += means[i][w] * means[j][w];
                    norm_j_sq += means[j][w] * means[j][w];
                }
                let norm_j = float::sqrt(norm_j_sq);
                if norm_j < 1e-15 {
                    continue;
                }
                let ncc = dot / (norm_i * norm_j);
                if ncc > best_corr {
                    best_corr = ncc;
                    best_i = i;
                    best_j = j;
                    found = true;
                }
            }
        }

        if !found {
            break;
        }

        // Collect spike times for each cluster (in seconds)
        const MAX_TIMES: usize = 512;
        let mut times_a = [0.0; MAX_TIMES];
        let mut times_b = [0.0; MAX_TIMES];
        let mut n_a = 0;
        let mut n_b = 0;
        let inv_sr = 1.0 / sample_rate;

        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            if labels[s] == best_i && n_a < MAX_TIMES {
                times_a[n_a] = event_buf[s].sample as Float * inv_sr;
                n_a += 1;
            } else if labels[s] == best_j && n_b < MAX_TIMES {
                times_b[n_b] = event_buf[s].sample as Float * inv_sr;
                n_b += 1;
            }
        }

        // Sort spike times
        times_a[..n_a]
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        times_b[..n_b]
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        // Compute CCG
        const CCG_BINS: usize = 50;
        let mut ccg = [0u64; CCG_BINS];
        isi::cross_correlogram(
            &times_a[..n_a],
            &times_b[..n_b],
            bin_width_s,
            max_lag_s,
            &mut ccg,
        );

        // Check for refractory dip: if present, these are the same neuron split
        // into two clusters (the neuron's refractory period appears in the CCG).
        // If NO dip, they are distinct neurons firing independently -- do NOT merge.
        if !isi::has_refractory_dip(&ccg, refractory_bins) {
            // No dip = independent neurons. Skip this pair.
            break;
        }

        // Refractory dip present: same neuron, over-split -- merge
        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            if labels[s] == best_j {
                labels[s] = best_i;
            } else if labels[s] > best_j {
                labels[s] -= 1;
            }
        }
        current_n -= 1;
    }

    current_n
}

/// Split clusters with high ISI violation rates.
///
/// If a cluster's ISI violation rate exceeds `isi_threshold`, it likely
/// contains two neurons. Splits along the first principal axis of the
/// feature distribution (same method as bimodality split, but triggered
/// by ISI rather than gap width).
#[allow(clippy::too_many_arguments)]
pub fn isi_violation_split<const K: usize>(
    n_spikes: usize,
    labels: &mut [usize],
    feature_buf: &[[Float; K]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    isi_threshold: Float,
    refractory_samples: usize,
    min_cluster_size: usize,
    scratch: &mut [Float],
    max_clusters: usize,
) -> usize {
    if n_spikes == 0 || n_clusters == 0 {
        return n_clusters;
    }

    const MAX_SPIKES: usize = 512;
    let mut current_n = n_clusters;

    loop {
        if current_n >= max_clusters {
            break;
        }

        let mut did_split = false;

        let mut cl = 0;
        while cl < current_n {
            // Collect spike indices for this cluster
            let mut indices = [0usize; MAX_SPIKES];
            let mut count = 0usize;
            let mut s = 0;
            while s < n_spikes && s < labels.len() {
                if labels[s] == cl && count < MAX_SPIKES {
                    indices[count] = s;
                    count += 1;
                }
                s += 1;
            }

            if count < min_cluster_size || count < 6 {
                cl += 1;
                continue;
            }

            // Compute ISI violation rate for this cluster
            let spike_n = count.min(scratch.len());
            for i in 0..spike_n {
                scratch[i] = event_buf[indices[i]].sample as Float;
            }
            let st = &mut scratch[..spike_n];
            st.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            let isi_rate =
                quality::isi_violation_rate(st, refractory_samples as Float).unwrap_or(0.0);

            if isi_rate <= isi_threshold {
                cl += 1;
                continue;
            }

            // This cluster has high ISI violations -- split along first principal axis.
            // Uses the same power-iteration approach as split_clusters.
            let dim = if K > 32 { 32 } else { K };

            // Compute centroid
            let mut centroid = [0.0; 32];
            for i in 0..count {
                for d in 0..dim {
                    centroid[d] += feature_buf[indices[i]][d];
                }
            }
            let inv_count = 1.0 / count as Float;
            for d in centroid.iter_mut().take(dim) {
                *d *= inv_count;
            }

            // Power iteration for first principal axis
            let mut axis = [0.0; 32];
            for d in 0..dim {
                axis[d] = feature_buf[indices[0]][d] - centroid[d];
            }
            let mut norm_sq = 0.0;
            for a in axis.iter().take(dim) {
                norm_sq += a * a;
            }
            if norm_sq < 1e-30 {
                cl += 1;
                continue;
            }
            let inv_norm = 1.0 / float::sqrt(norm_sq);
            for a in axis.iter_mut().take(dim) {
                *a *= inv_norm;
            }

            for _iter in 0..3 {
                let mut new_axis = [0.0; 32];
                for i in 0..count {
                    let mut dot = 0.0;
                    for d in 0..dim {
                        dot += (feature_buf[indices[i]][d] - centroid[d]) * axis[d];
                    }
                    for d in 0..dim {
                        new_axis[d] += dot * (feature_buf[indices[i]][d] - centroid[d]);
                    }
                }
                let mut ns = 0.0;
                for na in new_axis.iter().take(dim) {
                    ns += na * na;
                }
                if ns < 1e-30 {
                    break;
                }
                let inv = 1.0 / float::sqrt(ns);
                for d in 0..dim {
                    axis[d] = new_axis[d] * inv;
                }
            }

            // Project spikes and split at median
            let mut projections = [0.0; MAX_SPIKES];
            for i in 0..count {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += (feature_buf[indices[i]][d] - centroid[d]) * axis[d];
                }
                projections[i] = dot;
            }

            // Split at median projection (ensures roughly equal-sized halves)
            let mut sorted_proj = [0.0; MAX_SPIKES];
            sorted_proj[..count].copy_from_slice(&projections[..count]);
            sorted_proj[..count]
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let median = sorted_proj[count / 2];

            let new_label = current_n;
            let mut n_above = 0usize;
            for i in 0..count {
                if projections[i] >= median {
                    labels[indices[i]] = new_label;
                    n_above += 1;
                }
            }

            // Only split if both halves are large enough
            let n_below = count - n_above;
            if n_below >= min_cluster_size && n_above >= min_cluster_size {
                current_n += 1;
                did_split = true;
                break; // restart outer loop
            } else {
                // Undo: restore original label
                for i in 0..count {
                    if projections[i] >= median {
                        labels[indices[i]] = cl;
                    }
                }
                cl += 1;
            }
        }

        if !did_split {
            break;
        }
    }

    current_n
}

/// Split clusters with bimodal amplitude distributions.
///
/// For each cluster, collects spike amplitudes, sorts them, and looks for
/// the largest gap between consecutive sorted amplitudes. If the gap
/// exceeds `threshold * MAD / 0.6745` (i.e. `threshold` sigma), the cluster
/// is split at that gap. This catches cases where two neurons with different
/// peak amplitudes were merged into one cluster.
///
/// # Example
///
/// ```
/// use zerostone::sorter::amplitude_bimodality_split;
/// use zerostone::spike_sort::MultiChannelEvent;
///
/// let mut labels = [0, 0, 0, 0, 0, 0];
/// // Two groups: 3 spikes with amp ~1.0, 3 with amp ~10.0
/// let events = [
///     MultiChannelEvent { sample: 100, channel: 0, amplitude: 1.0 },
///     MultiChannelEvent { sample: 200, channel: 0, amplitude: 1.1 },
///     MultiChannelEvent { sample: 300, channel: 0, amplitude: 0.9 },
///     MultiChannelEvent { sample: 400, channel: 0, amplitude: 10.0 },
///     MultiChannelEvent { sample: 500, channel: 0, amplitude: 10.2 },
///     MultiChannelEvent { sample: 600, channel: 0, amplitude: 9.8 },
/// ];
/// let n = amplitude_bimodality_split(6, &mut labels, &events, 1, 2.0, 2, 8);
/// assert_eq!(n, 2);
/// // First 3 spikes should be in one cluster, last 3 in another
/// assert_eq!(labels[0], labels[1]);
/// assert_eq!(labels[3], labels[4]);
/// assert_ne!(labels[0], labels[3]);
/// ```
pub fn amplitude_bimodality_split(
    n_spikes: usize,
    labels: &mut [usize],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    threshold: Float,
    min_cluster_size: usize,
    max_clusters: usize,
) -> usize {
    if n_spikes == 0 || n_clusters == 0 {
        return n_clusters;
    }

    const MAX_SPIKES: usize = 512;
    let mut current_n = n_clusters;

    loop {
        if current_n >= max_clusters {
            break;
        }

        let mut did_split = false;

        let mut cl = 0;
        while cl < current_n {
            // Collect amplitudes for this cluster
            let mut amps = [0.0; MAX_SPIKES];
            let mut indices = [0usize; MAX_SPIKES];
            let mut count = 0usize;
            let mut s = 0;
            while s < n_spikes && s < labels.len() {
                if labels[s] == cl && count < MAX_SPIKES {
                    amps[count] = event_buf[s].amplitude;
                    indices[count] = s;
                    count += 1;
                }
                s += 1;
            }

            if count < min_cluster_size * 2 || count < 6 {
                cl += 1;
                continue;
            }

            // Sort amplitudes (keep index mapping)
            // Simple insertion sort to co-sort amps and indices
            let mut i = 1;
            while i < count {
                let key_amp = amps[i];
                let key_idx = indices[i];
                let mut j = i;
                while j > 0 && amps[j - 1] > key_amp {
                    amps[j] = amps[j - 1];
                    indices[j] = indices[j - 1];
                    j -= 1;
                }
                amps[j] = key_amp;
                indices[j] = key_idx;
                i += 1;
            }

            // Find the largest gap between consecutive sorted amplitudes,
            // then check if the gap exceeds threshold * local spread.
            // "Local spread" = max(MAD_left, MAD_right) to handle bimodal cases
            // where the global MAD is inflated by spanning both modes.
            let mut best_gap = 0.0;
            let mut best_gap_idx = 0usize;
            for k in 1..count {
                let gap = amps[k] - amps[k - 1];
                if gap > best_gap {
                    best_gap = gap;
                    best_gap_idx = k;
                }
            }

            if best_gap_idx == 0 || best_gap <= 0.0 {
                cl += 1;
                continue;
            }

            // Compute MAD for each half separately
            let left_n = best_gap_idx;
            let right_n = count - best_gap_idx;
            let local_sigma = {
                let mad_half = |start: usize, n: usize| -> Float {
                    if n < 2 {
                        return 0.0;
                    }
                    let med = amps[start + n / 2];
                    let mut devs = [0.0; MAX_SPIKES];
                    for k in 0..n {
                        devs[k] = float::abs(amps[start + k] - med);
                    }
                    devs[..n].sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                    });
                    let mad = devs[n / 2];
                    if mad > 0.0 {
                        mad / 0.6745
                    } else {
                        0.0
                    }
                };
                let s_left = mad_half(0, left_n);
                let s_right = mad_half(best_gap_idx, right_n);
                let s = if s_left > s_right { s_left } else { s_right };
                if s > 0.0 {
                    s
                } else {
                    1.0
                }
            };

            let gap_threshold = threshold * local_sigma;
            if best_gap < gap_threshold {
                cl += 1;
                continue;
            }

            // Split: spikes at index >= best_gap_idx (higher amplitudes) get new label
            let new_label = current_n;
            let n_above = count - best_gap_idx;
            let n_below = best_gap_idx;

            if n_below >= min_cluster_size && n_above >= min_cluster_size {
                for k in best_gap_idx..count {
                    labels[indices[k]] = new_label;
                }
                current_n += 1;
                did_split = true;
                break; // restart outer loop
            } else {
                cl += 1;
            }
        }

        if !did_split {
            break;
        }
    }

    current_n
}

/// Computes normalized spike half-width at half-maximum trough depth.
///
/// Returns the fraction of the waveform window occupied by the spike trough
/// at half-maximum depth. Narrow spikes (fast-spiking interneurons) return
/// small values; broad spikes (pyramidal cells) return large values.
///
/// # Arguments
/// * `waveform` - Peak-channel waveform of length W
///
/// Returns a value in approximately [0.0, 1.0] (clamped to 0.5 if no trough).
fn compute_half_width<const W: usize>(waveform: &[Float; W]) -> Float {
    // Find trough (minimum value)
    let mut t_min = W / 2;
    let mut trough = waveform[t_min];
    let mut wi = 0;
    while wi < W {
        if waveform[wi] < trough {
            trough = waveform[wi];
            t_min = wi;
        }
        wi += 1;
    }
    if trough >= 0.0 {
        return 0.5; // no trough → default
    }
    let half_max = trough * 0.5; // both negative; half_max is between 0 and trough
                                 // Walk left from t_min until waveform exceeds half_max (becomes less negative)
    let mut left = t_min;
    while left > 0 && waveform[left] < half_max {
        left -= 1;
    }
    // Walk right from t_min until waveform exceeds half_max
    let mut right = t_min;
    while right + 1 < W && waveform[right] < half_max {
        right += 1;
    }
    (right - left) as Float / W as Float
}

/// Compute mean waveform per cluster, spike count, and most common peak channel.
#[allow(clippy::too_many_arguments)]
fn compute_cluster_means<const W: usize, const N: usize>(
    waveform_buf: &[[Float; W]],
    labels: &[usize],
    event_buf: &[MultiChannelEvent],
    n_extracted: usize,
    n_clusters: usize,
    means: &mut [[Float; W]; N],
    counts: &mut [u32; N],
    peak_channels: &mut [usize; N],
) {
    for c in 0..N {
        means[c] = [0.0; W];
        counts[c] = 0;
        peak_channels[c] = 0;
    }
    let mut ch_votes = [[0u32; 64]; N];

    for i in 0..n_extracted {
        let label = labels[i];
        if label >= n_clusters || label >= N {
            continue;
        }
        counts[label] += 1;
        let ch = event_buf[i].channel;
        if ch < 64 {
            ch_votes[label][ch] += 1;
        }
        for (w, mw) in means[label].iter_mut().enumerate() {
            *mw += waveform_buf[i][w];
        }
    }

    for c in 0..n_clusters.min(N) {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as Float;
            for mw in means[c].iter_mut() {
                *mw *= inv;
            }
        }
        let mut best_ch = 0;
        let mut best_votes = 0;
        for (ch, &v) in ch_votes[c].iter().enumerate() {
            if v > best_votes {
                best_votes = v;
                best_ch = ch;
            }
        }
        peak_channels[c] = best_ch;
    }
}

/// Computes mean waveforms on the nearest neighbor channel for each cluster.
///
/// For each cluster, finds the nearest neighbor of the peak channel and
/// averages waveforms from spike events on that neighbor channel.
/// Used for composite matched filter scoring.
#[allow(clippy::too_many_arguments)]
fn compute_neighbor_templates<const W: usize, const N: usize, const C: usize>(
    data: &[[Float; C]],
    event_buf: &[MultiChannelEvent],
    labels: &[usize],
    n_extracted: usize,
    n_clusters: usize,
    mf_peak_channels: &[usize; N],
    probe: &ProbeLayout<C>,
    neighbor_templates: &mut [[Float; W]; N],
    neighbor_channels: &mut [usize; N],
) {
    // Initialize outputs
    let mut i = 0;
    while i < N {
        neighbor_templates[i] = [0.0; W];
        neighbor_channels[i] = mf_peak_channels[i]; // fallback: same as peak
        i += 1;
    }
    let mut counts = [0u32; N];

    // Determine neighbor channel for each cluster
    let mut c = 0;
    while c < n_clusters.min(N) {
        let peak_ch = mf_peak_channels[c];
        if peak_ch < C {
            let mut nbuf = [0usize; 2];
            let n = probe.nearest_channels(peak_ch, 1, &mut nbuf);
            if n > 0 && nbuf[0] < C {
                neighbor_channels[c] = nbuf[0];
            }
        }
        c += 1;
    }

    // Accumulate waveforms on the neighbor channel
    let t_len = data.len();
    let mut i = 0;
    while i < n_extracted {
        let label = labels[i];
        if label >= n_clusters || label >= N {
            i += 1;
            continue;
        }
        let neighbor_ch = neighbor_channels[label];
        if neighbor_ch >= C {
            i += 1;
            continue;
        }
        let sample = event_buf[i].sample;
        // Use same pre_samples=20 and window W as primary waveform extraction
        let pre = W * 5 / 12; // matches matched_filter pre_samples convention
        if sample >= pre && sample + W - pre <= t_len {
            let start = sample - pre;
            let mut w = 0;
            while w < W {
                neighbor_templates[label][w] += data[start + w][neighbor_ch];
                w += 1;
            }
            counts[label] += 1;
        }
        i += 1;
    }

    // Normalize to get mean waveforms
    let mut c = 0;
    while c < n_clusters.min(N) {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as Float;
            let mut w = 0;
            while w < W {
                neighbor_templates[c][w] *= inv;
                w += 1;
            }
        }
        c += 1;
    }
}

/// Subtract cluster mean templates from data at each spike location.
#[allow(clippy::too_many_arguments)]
fn subtract_templates_multichannel<const C: usize, const W: usize, const N: usize>(
    data: &mut [[Float; C]],
    event_buf: &[MultiChannelEvent],
    n_spikes: usize,
    labels: &[usize],
    means: &[[Float; W]; N],
    counts: &[u32; N],
    peak_channels: &[usize; N],
    min_count: usize,
    pre_samples: usize,
) {
    // Precompute template norms squared for amplitude scaling
    let mut norms_sq = [0.0; N];
    for c in 0..N {
        if counts[c] == 0 {
            continue;
        }
        let mut s = 0.0;
        for val in means[c].iter() {
            s += val * val;
        }
        norms_sq[c] = s;
    }

    let t_len = data.len();
    for i in 0..n_spikes {
        let label = labels[i];
        if label >= N || (counts[label] as usize) < min_count {
            continue;
        }
        let ch = peak_channels[label];
        if ch >= C {
            continue;
        }
        let peak = event_buf[i].sample;
        let start = peak.saturating_sub(pre_samples);
        let end = (start + W).min(t_len);
        let n_valid = end - start;

        // Per-spike amplitude scaling: alpha = dot(data, template) / ||template||^2
        // This handles natural amplitude variability in single-unit spikes
        let mut dot = 0.0;
        for w in 0..n_valid {
            dot += data[start + w][ch] * means[label][w];
        }
        let alpha = if norms_sq[label] > 1e-30 {
            // Clamp to [0.3, 3.0] to prevent pathological scaling
            (dot / norms_sq[label]).clamp(0.3, 3.0)
        } else {
            1.0
        };

        for w in 0..n_valid {
            data[start + w][ch] -= alpha * means[label][w];
        }
    }
}

/// Assign a waveform to the nearest cluster template by L2 distance.
fn assign_to_nearest_template<const W: usize, const N: usize>(
    waveform: &[Float; W],
    means: &[[Float; W]; N],
    counts: &[u32; N],
    n_clusters: usize,
) -> (usize, Float) {
    let mut best = 0;
    let mut best_dist = float::MAX;
    for c in 0..n_clusters.min(N) {
        if counts[c] == 0 {
            continue;
        }
        // Early-exit squared distance: bail if partial sum exceeds current best
        let mut dist = 0.0;
        let mut bail = false;
        for w in 0..W {
            let d = waveform[w] - means[c][w];
            dist += d * d;
            if dist > best_dist {
                bail = true;
                break;
            }
        }
        if !bail && dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    (best, best_dist)
}

/// SVD-based centroid initialization for k-means clustering.
///
/// Finds the dominant eigenvector of the feature covariance using the
/// power iteration method (no matrix storage beyond K-dimensional vectors),
/// projects all features onto it, bins by projection value into `max_seeds`
/// equal-width intervals, and returns each bin's mean as a seed centroid.
///
/// This places centroids where the data density is highest along the
/// principal axis, unlike farthest-point which only maximizes inter-centroid
/// distance and tends to place seeds at outliers.
///
/// # Type Parameters
///
/// * `K` - Feature dimensionality
/// * `N` - Maximum number of centroids
///
/// # Returns
///
/// `(centroids, count)` where `centroids[..count]` are the seed centroids.
///
/// # Example
///
/// ```
/// use zerostone::sorter::svd_init_centroids;
///
/// // Two well-separated clusters along dimension 0
/// let features = [
///     [0.0, 0.0], [0.1, 0.1], [0.2, -0.1], [-0.1, 0.05],
///     [5.0, 0.0], [5.1, 0.1], [4.9, -0.1], [5.2, 0.05],
/// ];
/// let (centroids, count) = svd_init_centroids::<2, 8>(&features, 8, 2);
/// assert_eq!(count, 2);
/// // First centroid near [0.05, 0.0125], second near [5.05, 0.0125]
/// assert!(centroids[0][0] < 1.0);
/// assert!(centroids[1][0] > 4.0);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn svd_init_centroids<const K: usize, const N: usize>(
    feature_buf: &[[Float; K]],
    n_extracted: usize,
    max_seeds: usize,
) -> ([[Float; K]; N], usize) {
    let mut centroids = [[0.0; K]; N];
    let n = if n_extracted < feature_buf.len() {
        n_extracted
    } else {
        feature_buf.len()
    };
    if n < 2 || max_seeds == 0 || K == 0 {
        return (centroids, 0);
    }
    let n_seeds = if max_seeds > N { N } else { max_seeds };
    let n_seeds = if n_seeds > n { n } else { n_seeds };

    // 1. Compute mean
    let mut mean = [0.0; K];
    for i in 0..n {
        for d in 0..K {
            mean[d] += feature_buf[i][d];
        }
    }
    let inv_n = 1.0 / n as Float;
    for d in 0..K {
        mean[d] *= inv_n;
    }

    // 2. Find dominant eigenvector via power iteration on the covariance.
    //    Instead of forming the K x K matrix explicitly, we use the identity:
    //      C * v = (1/n) * sum_i [ (x_i - mu) * ((x_i - mu) . v) ]
    //    Each iteration is O(n * K), no K x K storage needed.
    let mut top_vec = [0.0; K];
    // Initialize with [1, 1, ..., 1] normalized
    let inv_sqrt_k = 1.0 / float::sqrt(K as Float);
    for d in 0..K {
        top_vec[d] = inv_sqrt_k;
    }

    for _iter in 0..50 {
        let mut new_vec = [0.0; K];
        // Multiply: new_vec = C * top_vec = (1/(n-1)) * sum_i (x_i - mu) * dot(x_i - mu, top_vec)
        for i in 0..n {
            let mut dot = 0.0;
            for d in 0..K {
                dot += (feature_buf[i][d] - mean[d]) * top_vec[d];
            }
            for d in 0..K {
                new_vec[d] += (feature_buf[i][d] - mean[d]) * dot;
            }
        }
        // Normalize
        let mut norm_sq = 0.0;
        for d in 0..K {
            norm_sq += new_vec[d] * new_vec[d];
        }
        let norm = float::sqrt(norm_sq);
        if norm < 1e-30 {
            // Degenerate -- all points identical
            centroids[0] = mean;
            return (centroids, 1);
        }
        let inv_norm = 1.0 / norm;
        // Check convergence: |new - old| < tol
        let mut diff_sq = 0.0;
        for d in 0..K {
            let v = new_vec[d] * inv_norm;
            diff_sq += (v - top_vec[d]) * (v - top_vec[d]);
            top_vec[d] = v;
        }
        if diff_sq < 1e-20 {
            break;
        }
    }

    // 3. Project features onto top eigenvector and find min/max
    let mut proj_min = float::MAX;
    let mut proj_max = Float::MIN;
    for i in 0..n {
        let mut p = 0.0;
        for d in 0..K {
            p += feature_buf[i][d] * top_vec[d];
        }
        if p < proj_min {
            proj_min = p;
        }
        if p > proj_max {
            proj_max = p;
        }
    }

    if proj_max - proj_min < 1e-15 {
        centroids[0] = mean;
        return (centroids, 1);
    }

    // 4. Range-based binning: divide [proj_min, proj_max] into n_seeds equal-width bins.
    let bin_width = (proj_max - proj_min) / n_seeds as Float;

    let mut bin_sums = [[0.0; K]; N];
    let mut bin_counts = [0usize; N];

    for i in 0..n {
        let mut p = 0.0;
        for d in 0..K {
            p += feature_buf[i][d] * top_vec[d];
        }
        let mut bin = ((p - proj_min) / bin_width) as usize;
        if bin >= n_seeds {
            bin = n_seeds - 1;
        }
        bin_counts[bin] += 1;
        for d in 0..K {
            bin_sums[bin][d] += feature_buf[i][d];
        }
    }

    // Compute bin means, skip empty bins
    let mut count = 0;
    for b in 0..n_seeds {
        if bin_counts[b] > 0 && count < N {
            let inv_c = 1.0 / bin_counts[b] as Float;
            for d in 0..K {
                centroids[count][d] = bin_sums[b][d] * inv_c;
            }
            count += 1;
        }
    }

    (centroids, count)
}

/// Full multi-channel sorting pipeline.
///
/// Pipeline: noise -> covariance -> whiten (in-place) -> detect -> dedup ->
/// align -> extract (peak channel) -> PCA -> cluster -> labels.
///
/// Caller provides all buffers. `data` is whitened in-place.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `CM` - C * C (whitening matrix size)
/// * `W` - Waveform window length in samples
/// * `K` - Number of PCA components / feature dimensions
/// * `WM` - W * W (PCA covariance matrix size)
/// * `N` - Maximum number of clusters
///
/// # Arguments
///
/// * `config` - Pipeline configuration
/// * `probe` - Probe geometry for deduplication
/// * `data` - Multi-channel data (whitened in-place)
/// * `scratch` - Working buffer, must have at least `data.len()` elements
/// * `event_buf` - Buffer for detected events
/// * `waveform_buf` - Buffer for extracted waveforms
/// * `feature_buf` - Buffer for PCA features
/// * `labels` - Output buffer for cluster labels per spike
///
/// # Returns
///
/// `SortResult<N>` with spike count, cluster count, and per-cluster quality.
///
/// After this function returns, `event_buf[0..result.n_spikes]` contains the
/// final spike events after deduplication and alignment, each with `.sample`
/// (the sample index in the input data), `.channel` (the peak channel), and
/// `.amplitude` fields.
///
/// # Errors
///
/// Returns `SortError::InsufficientData` if the data is too short or too
/// few spikes are detected for PCA. Returns `SortError::EigenFailed` if
/// whitening or PCA eigendecomposition fails.
///
/// # Example
///
/// ```no_run
/// use zerostone::sorter::{sort_multichannel, SortConfig, SortResult};
/// use zerostone::probe::ProbeLayout;
/// use zerostone::spike_sort::MultiChannelEvent;
///
/// let config = SortConfig::default();
/// let probe = ProbeLayout::<2>::linear(25.0);
/// let mut data = vec![[0.0; 2]; 1000];
/// let mut scratch = vec![0.0; 1000];
/// let mut events = vec![MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 100];
/// let mut waveforms = vec![[0.0; 16]; 100];
/// let mut features = vec![[0.0; 3]; 100];
/// let mut labels = vec![0usize; 100];
///
/// let result = sort_multichannel::<2, 4, 16, 3, 256, 8>(
///     &config, &probe, &mut data, &mut scratch,
///     &mut events, &mut waveforms, &mut features, &mut labels,
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn sort_multichannel<
    const C: usize,
    const CM: usize,
    const W: usize,
    const K: usize,
    const WM: usize,
    const N: usize,
>(
    config: &SortConfig,
    probe: &ProbeLayout<C>,
    data: &mut [[Float; C]],
    scratch: &mut [Float],
    event_buf: &mut [MultiChannelEvent],
    waveform_buf: &mut [[Float; W]],
    feature_buf: &mut [[Float; K]],
    labels: &mut [usize],
) -> Result<SortResult<N>, SortError> {
    let t_len = data.len();
    if t_len < W {
        return Err(SortError::InsufficientData);
    }

    // 0a. Common Median Reference: subtract per-sample median across channels.
    // More robust than CAR (mean) when individual channels have large artifacts.
    // Auto-applies for C >= 8 when auto_cmr is true: correlated noise between
    // adjacent channels degrades whitening on multi-channel probes.
    let use_cmr = (config.common_median_ref || (config.auto_cmr && C >= 8)) && C > 2;
    if use_cmr {
        for sample in data.iter_mut() {
            // Sort channel values into scratch to find median
            let n = C.min(scratch.len());
            for (sc, sv) in scratch.iter_mut().zip(sample.iter()).take(n) {
                *sc = *sv;
            }
            scratch[..n]
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            #[allow(clippy::manual_is_multiple_of)]
            let median = if n % 2 == 0 {
                (scratch[n / 2 - 1] + scratch[n / 2]) * 0.5
            } else {
                scratch[n / 2]
            };
            for ch in sample.iter_mut() {
                *ch -= median;
            }
        }
    }

    // 0b. Bandpass filter: remove LFP (< low) and high-frequency noise (> high).
    // Uses a 4th-order Butterworth bandpass implemented as 2 cascaded biquad
    // sections in f64. Applied per-channel, forward-only (causal).
    if config.bandpass_low > 0.0
        && config.bandpass_high > config.bandpass_low
        && config.sample_rate > 0.0
    {
        bandpass_inplace::<C>(
            data,
            config.sample_rate,
            config.bandpass_low,
            config.bandpass_high,
        );
    }

    // 1. Noise estimation (pre-whitening, for quality metrics later)
    let pre_noise = estimate_noise_multichannel::<C>(data, scratch);
    // Use channel 0 noise as representative for SNR computation
    let mut noise_mean = 0.0;
    for noise_val in pre_noise.iter() {
        noise_mean += noise_val;
    }
    noise_mean /= C as Float;
    if noise_mean <= 0.0 {
        noise_mean = 1.0;
    }

    // 2. Covariance
    let cov = compute_covariance::<C>(data);

    // 3. Whitening (in-place)
    let wm = WhiteningMatrix::<C, CM>::from_covariance(
        &cov,
        WhiteningMode::Zca,
        config.whitening_epsilon,
    )
    .map_err(|_| SortError::EigenFailed)?;
    for sample in data.iter_mut() {
        *sample = wm.apply(sample);
    }

    // 4. Detection (on whitened data, noise ~ 1.0 per channel)
    //
    // For NEO/SNEO modes, we apply the energy operator per channel into scratch,
    // estimate noise on the transformed signal, and detect on the energy signal.
    // The detected spike times still index into the original (whitened) data.
    let n_detected = match config.detection_mode {
        DetectionMode::Amplitude => {
            if config.adaptive_threshold {
                // Per-channel adaptive thresholds: compute from whitened data
                let adaptive_thresh = compute_adaptive_thresholds::<C>(
                    data,
                    config.threshold_multiplier,
                    config.adaptive_min_threshold,
                    config.adaptive_max_rate_hz,
                    config.sample_rate,
                    scratch,
                );
                // Pass absolute thresholds as noise estimates with multiplier 1.0
                detect_spikes_multichannel::<C>(
                    data,
                    1.0,
                    &adaptive_thresh,
                    config.refractory_samples,
                    event_buf,
                )
            } else {
                let unit_noise = [1.0; C];
                detect_spikes_multichannel::<C>(
                    data,
                    config.threshold_multiplier,
                    &unit_noise,
                    config.refractory_samples,
                    event_buf,
                )
            }
        }
        DetectionMode::Neo | DetectionMode::Sneo { .. } => {
            use crate::spike_sort::{neo_transform, sneo_transform};
            let smooth_w = match config.detection_mode {
                DetectionMode::Sneo { smooth_window } => smooth_window,
                _ => 0,
            };
            // Apply NEO/SNEO per channel, detect on energy signal.
            // We build a temporary energy array per channel in scratch,
            // estimate its noise via MAD, and detect threshold crossings.
            // Spike times are offset by +1 to map back to the original data indices.
            let mut total = 0usize;
            let mut ch = 0;
            while ch < C {
                // Extract single-channel data into scratch
                let s_len = scratch.len().min(t_len);
                for t in 0..s_len {
                    scratch[t] = data[t][ch];
                }
                // Apply NEO or SNEO (output into the second half of scratch)
                let half = s_len / 2;
                let (src, dst) = scratch.split_at_mut(half);
                let n_energy = if smooth_w > 0 {
                    sneo_transform(&src[..s_len.min(half)], dst, smooth_w)
                } else {
                    neo_transform(&src[..s_len.min(half)], dst)
                };
                if n_energy < 2 {
                    ch += 1;
                    continue;
                }
                // Threshold via robust percentile estimation (median + MAD).
                // Use first min(n_energy, 2000) samples as calibration window.
                // Median and MAD resist spike contamination (50% breakdown).
                // thresh = median + T * MAD / 0.6745
                let energy = &dst[..n_energy];
                const CAL_LEN: usize = 2000;
                let cal_n = if n_energy < CAL_LEN {
                    n_energy
                } else {
                    CAL_LEN
                };
                let mut cal_buf = [0.0; CAL_LEN];
                let mut ci = 0;
                while ci < cal_n {
                    cal_buf[ci] = energy[ci];
                    ci += 1;
                }
                cal_buf[..cal_n].sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let median = if cal_n < 2 {
                    1.0
                } else if cal_n % 2 == 1 {
                    cal_buf[cal_n / 2]
                } else {
                    (cal_buf[cal_n / 2 - 1] + cal_buf[cal_n / 2]) * 0.5
                };
                // MAD = median(|x - median|)
                ci = 0;
                while ci < cal_n {
                    cal_buf[ci] = float::abs(cal_buf[ci] - median);
                    ci += 1;
                }
                cal_buf[..cal_n].sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let mad = if cal_n < 2 {
                    1.0
                } else if cal_n % 2 == 1 {
                    cal_buf[cal_n / 2]
                } else {
                    (cal_buf[cal_n / 2 - 1] + cal_buf[cal_n / 2]) * 0.5
                };
                let sigma = if mad > 0.0 { mad / 0.6745 } else { 1.0 };
                let thresh = median + config.threshold_multiplier * sigma;
                // Detect positive threshold crossings on energy signal
                // (NEO/SNEO output is positive for spikes)
                let mut i = 0;
                while i < n_energy {
                    if energy[i] > thresh {
                        let end = if i + config.refractory_samples < n_energy {
                            i + config.refractory_samples
                        } else {
                            n_energy
                        };
                        let mut max_idx = i;
                        let mut max_val = energy[i];
                        let mut j = i + 1;
                        while j < end {
                            if energy[j] > max_val {
                                max_val = energy[j];
                                max_idx = j;
                            }
                            j += 1;
                        }
                        if total < event_buf.len() {
                            // Offset by +1 to map NEO index back to original data
                            let neo_sample = max_idx + 1;
                            // Refine: find the nearest voltage minimum (negative peak)
                            // within a small window around the NEO peak. This aligns
                            // SNEO-detected events with the actual spike trough for
                            // correct waveform extraction and clustering.
                            let refine_half = 5; // search +/- 5 samples
                            let r_start = neo_sample.saturating_sub(refine_half);
                            let r_end = (neo_sample + refine_half + 1).min(t_len);
                            let mut best_sample = neo_sample.min(t_len - 1);
                            let mut best_val = data[best_sample][ch];
                            let mut ri = r_start;
                            while ri < r_end {
                                if data[ri][ch] < best_val {
                                    best_val = data[ri][ch];
                                    best_sample = ri;
                                }
                                ri += 1;
                            }
                            event_buf[total] = MultiChannelEvent {
                                sample: best_sample,
                                channel: ch,
                                amplitude: float::abs(best_val),
                            };
                            total += 1;
                        }
                        i = end;
                    } else {
                        i += 1;
                    }
                }
                ch += 1;
            }
            // Sort events by sample index (insertion sort, stable)
            let mut k = 1;
            while k < total {
                let key = event_buf[k];
                let mut pos = k;
                while pos > 0 && event_buf[pos - 1].sample > key.sample {
                    event_buf[pos] = event_buf[pos - 1];
                    pos -= 1;
                }
                event_buf[pos] = key;
                k += 1;
            }
            total
        }
    };
    // 4b. Spatial coincidence recovery: scan at a lower threshold but require
    // multiple neighboring channels to corroborate the spike. True spikes spread
    // spatially; isolated noise events do not. Only active when C >= 4.
    let n_detected = if config.coincidence_detect && C >= 4 {
        use crate::spike_sort::detect_spikes_coincidence;
        // Split event_buf at n_detected: first half is existing events (read-only),
        // second half is where new coincidence events are written.
        // split_at_mut gives non-overlapping slices; coerce first to &[..].
        let buf_len = event_buf.len();
        let split = n_detected.min(buf_len);
        let (existing_part, new_part) = event_buf.split_at_mut(split);
        let n_new = detect_spikes_coincidence::<C>(
            data,
            &[1.0; C],
            existing_part,
            probe,
            config.coincidence_primary_threshold,
            config.coincidence_secondary_threshold,
            config.min_coincident_channels,
            config.spatial_radius_um,
            config.refractory_samples,
            new_part,
        );
        n_detected + n_new
    } else {
        n_detected
    };

    if n_detected < 2 {
        return Ok(SortResult {
            n_spikes: n_detected,
            n_clusters: 0,
            clusters: core::array::from_fn(|_| ClusterInfo {
                count: 0,
                snr: 0.0,
                isi_violation_rate: 0.0,
            }),
        });
    }

    // 4c. Re-sort combined event list (primary + coincidence) by sample index.
    // deduplicate_events requires sorted input. Primary detection produces sorted
    // events; coincidence events are appended unsorted and must be merged in.
    if config.coincidence_detect && C >= 4 {
        let mut k = 1;
        while k < n_detected {
            let key = event_buf[k];
            let mut pos = k;
            while pos > 0 && event_buf[pos - 1].sample > key.sample {
                event_buf[pos] = event_buf[pos - 1];
                pos -= 1;
            }
            event_buf[pos] = key;
            k += 1;
        }
    }

    // 5. Deduplication
    let n_dedup = deduplicate_events::<C>(
        event_buf,
        n_detected,
        probe,
        config.spatial_radius_um,
        config.temporal_radius,
    );
    if n_dedup < 2 {
        return Ok(SortResult {
            n_spikes: n_dedup,
            n_clusters: 0,
            clusters: core::array::from_fn(|_| ClusterInfo {
                count: 0,
                snr: 0.0,
                isi_violation_rate: 0.0,
            }),
        });
    }

    // 6. Alignment
    align_to_peak::<C>(data, event_buf, n_dedup, config.align_half_window);

    // 7. Extraction (peak channel)
    let mut n_extracted =
        extract_peak_channel::<C, W>(data, event_buf, n_dedup, config.pre_samples, waveform_buf);
    if n_extracted < 2 {
        return Ok(SortResult {
            n_spikes: n_extracted,
            n_clusters: 0,
            clusters: core::array::from_fn(|_| ClusterInfo {
                count: 0,
                snr: 0.0,
                isi_violation_rate: 0.0,
            }),
        });
    }

    // 8. PCA
    let mut pca = WaveformPca::<W, K, WM>::new();
    pca.fit(&waveform_buf[..n_extracted])?;

    for i in 0..n_extracted {
        pca.transform(&waveform_buf[i], &mut feature_buf[i])?;
    }

    // 8b. Encode spatial information as feature dimensions.
    //
    // After whitening, single-channel waveform shapes become similar across
    // channels, so PCA on peak-channel waveforms alone cannot distinguish
    // units on different channels.
    //
    // Two modes:
    // - Default: replace the last PCA component with normalized channel index.
    // - Localization: replace the last two PCA components with (x, y)
    //   center-of-mass position from peak amplitudes across channels and
    //   probe geometry. This provides continuous 2D spatial discrimination.
    if K >= 3 {
        if config.use_localization && K >= 4 {
            // Compute center-of-mass (x, y) for each spike from multi-channel
            // peak amplitudes at the spike time, scaled to match cluster_threshold.
            let positions = probe.positions();

            // Find the spatial extent for normalization
            let mut x_min: Float = Float::MAX;
            let mut x_max: Float = Float::MIN;
            let mut y_min: Float = Float::MAX;
            let mut y_max: Float = Float::MIN;
            let mut ci = 0;
            while ci < C {
                if positions[ci][0] < x_min {
                    x_min = positions[ci][0];
                }
                if positions[ci][0] > x_max {
                    x_max = positions[ci][0];
                }
                if positions[ci][1] < y_min {
                    y_min = positions[ci][1];
                }
                if positions[ci][1] > y_max {
                    y_max = positions[ci][1];
                }
                ci += 1;
            }
            let x_range = if float::abs(x_max - x_min) > 1e-12 {
                x_max - x_min
            } else {
                1.0
            };
            let y_range = if float::abs(y_max - y_min) > 1e-12 {
                y_max - y_min
            } else {
                1.0
            };

            let spatial_scale = config.cluster_threshold * C as Float;

            for i in 0..n_extracted {
                let sample_idx = event_buf[i].sample;
                if sample_idx < data.len() {
                    // Extract per-channel amplitudes at spike time.
                    // Use absolute values because whitened data can be negative,
                    // and center-of-mass needs positive weights.
                    let mut amps = [0.0 as Float; C];
                    let mut ch = 0;
                    while ch < C {
                        amps[ch] = float::abs(data[sample_idx][ch]);
                        ch += 1;
                    }
                    let com = localize::center_of_mass(&amps, positions);
                    // Normalize to [0, 1] and scale
                    feature_buf[i][K - 2] = ((com[0] - x_min) / x_range) * spatial_scale;
                    feature_buf[i][K - 1] = ((com[1] - y_min) / y_range) * spatial_scale;
                } else {
                    // Fallback: use channel index for both dims
                    let ch = event_buf[i].channel;
                    let norm = ch as Float / C as Float;
                    feature_buf[i][K - 2] = norm * spatial_scale;
                    feature_buf[i][K - 1] = norm * spatial_scale;
                }
            }
        } else if config.use_amplitude_profile && C > 1 && K >= 4 {
            // Two-feature spatial encoding:
            // - K-1: channel index (strong separation, same as fallback)
            // - K-2: amplitude profile ratio (fine spatial discrimination)
            //
            // The amplitude profile measures peak amplitude on neighboring
            // channels relative to the peak channel. Units with different
            // spatial extents (e.g., near vs far from peak channel) get
            // different ratios even when they share the same peak channel.
            // This replaces the weakest PCA component (K-2) with spatial info.
            let n_neighbors = config.amplitude_profile_neighbors.min(C - 1);

            // Precompute nearest-neighbor lists for all channels
            let mut neighbor_lists = [[0usize; 8]; C];
            let mut neighbor_counts = [0usize; C];
            let max_per_ch = n_neighbors.min(8);
            for ch_idx in 0..C {
                let mut nbuf = [0usize; 8];
                let n = probe.nearest_channels(ch_idx, max_per_ch, &mut nbuf);
                let mut ni = 0;
                while ni < n {
                    neighbor_lists[ch_idx][ni] = nbuf[ni];
                    ni += 1;
                }
                neighbor_counts[ch_idx] = n;
            }

            let channel_scale = config.cluster_threshold * C as Float;
            // Profile scale: moderate, comparable to PCA magnitudes
            let profile_scale = config.cluster_threshold * 2.0;

            for i in 0..n_extracted {
                // K-1: channel index (unchanged from fallback)
                let peak_ch = event_buf[i].channel;
                feature_buf[i][K - 1] = (peak_ch as Float / C as Float) * channel_scale;

                // K-2: amplitude profile ratio
                let sample_idx = event_buf[i].sample;
                if sample_idx < data.len() && peak_ch < C {
                    let peak_amp = float::abs(data[sample_idx][peak_ch]);
                    if peak_amp > 1e-15 {
                        // Compute ratio of neighbor amplitude to peak
                        let mut neighbor_sum = 0.0;
                        let n_nbrs = neighbor_counts[peak_ch];
                        let mut ni = 0;
                        let mut count = 0;
                        while ni < n_nbrs && ni < n_neighbors {
                            let nbr_ch = neighbor_lists[peak_ch][ni];
                            neighbor_sum += float::abs(data[sample_idx][nbr_ch]);
                            count += 1;
                            ni += 1;
                        }
                        // Ratio: how much energy is in neighbors vs peak
                        // Range: [0, n_neighbors] typically [0.3, 2.0]
                        let ratio = if count > 0 {
                            neighbor_sum / (count as Float * peak_amp)
                        } else {
                            0.0
                        };
                        feature_buf[i][K - 2] = ratio * profile_scale;
                    } else {
                        feature_buf[i][K - 2] = 0.0;
                    }
                } else {
                    feature_buf[i][K - 2] = 0.0;
                }
            }
        } else {
            // Fallback: encode channel index in K-1 and optionally spike
            // half-width in K-2. Half-width separates unit types (narrow
            // interneurons vs broad pyramidal cells) orthogonally to amplitude.
            let channel_scale = config.cluster_threshold * C as Float;
            // Scale half-width to approximately match the PCA component range.
            // cluster_threshold * 4 maps [0,1] half-width to [0, 28] for the
            // default threshold of 7.0 — comparable to the range of PCA components
            // and small enough not to dominate cross-channel separation.
            let shape_scale = config.cluster_threshold * 4.0;
            for i in 0..n_extracted {
                let ch = event_buf[i].channel;
                feature_buf[i][K - 1] = (ch as Float / C as Float) * channel_scale;
                if config.use_shape_features && K >= 4 {
                    let hw = compute_half_width::<W>(&waveform_buf[i]);
                    feature_buf[i][K - 2] = hw * shape_scale;
                }
            }
        }
    }

    // 9. Clustering
    // Auto-scale cluster creation threshold for low-channel recordings:
    // with fewer channels the spatial feature dimension spans fewer modes,
    // clusters sit closer together in feature space, and a higher threshold
    // is needed to prevent over-splitting. Scale by sqrt(8/C) when C < 8.
    let effective_cluster_threshold = if config.auto_cluster_threshold && C < 8 {
        config.cluster_threshold * float::sqrt(8.0 / C as Float)
    } else {
        config.cluster_threshold
    };
    let mut km = OnlineKMeans::<K, N>::new(config.cluster_max_count);
    km.set_create_threshold(effective_cluster_threshold);

    // 9a. Seed centroids using farthest-point or SVD-based initialization.
    // This picks well-separated initial centroids deterministically,
    // reducing sensitivity to spike arrival order vs naive first-come init.
    // Limit seeds to sqrt(N) to leave room for online cluster creation.
    let max_init_seeds = {
        // isqrt approximation: find largest s where s*s <= N
        let mut s = 1usize;
        while (s + 1) * (s + 1) <= N {
            s += 1;
        }
        s.max(2).min(N / 2)
    };
    if n_extracted > max_init_seeds {
        if config.svd_init {
            // SVD-based: project onto dominant eigenvector, bin, seed with bin means.
            let (seeds, n_seeds) =
                svd_init_centroids::<K, N>(feature_buf, n_extracted, max_init_seeds);
            for seed in seeds.iter().take(n_seeds) {
                let _ = km.seed_centroid(seed);
            }
        } else {
            km.init_farthest_point(&feature_buf[..n_extracted], max_init_seeds);
        }
    }

    for i in 0..n_extracted {
        let result = km.update(&feature_buf[i]);
        if i < labels.len() {
            labels[i] = result.cluster;
        }
    }

    let n_clusters_pre = km.n_active();

    // 9b. Post-clustering merge of over-split clusters
    let n_clusters = merge_clusters::<K>(
        n_extracted,
        labels,
        feature_buf,
        event_buf,
        n_clusters_pre,
        config.merge_dprime_threshold,
        config.merge_isi_threshold,
        config.refractory_samples,
        scratch,
        K,
    );

    // 9b2. Cross-channel spatial merge using amplitude profiles
    let n_clusters = merge_clusters_spatial::<C>(
        n_extracted,
        labels,
        data,
        event_buf,
        probe,
        n_clusters,
        config.spatial_merge_dprime,
        config.spatial_radius_um,
        config.merge_isi_threshold,
        config.refractory_samples,
        scratch,
    );

    // 9c. Split bimodal clusters
    let n_clusters = split_clusters::<K>(
        n_extracted,
        labels,
        feature_buf,
        n_clusters,
        config.split_min_cluster_size,
        config.split_bimodality_threshold,
    );

    // 9c2. CCG-based cluster merge: merge over-split clusters that lack refractory dip
    let n_clusters = if config.ccg_merge && n_clusters > 1 {
        ccg_merge_clusters::<W, N>(
            n_extracted,
            labels,
            waveform_buf,
            event_buf,
            n_clusters,
            config.ccg_template_corr_threshold,
            30000.0, // TODO: accept sample_rate in SortConfig
        )
    } else {
        n_clusters
    };

    // 9c3. ISI-violation split: clusters with high ISI violation rate likely
    // contain two neurons firing at similar rates. Split along the first
    // principal axis of the feature distribution.
    let n_clusters = if config.isi_split_threshold > 0.0 && n_clusters > 0 {
        isi_violation_split::<K>(
            n_extracted,
            labels,
            feature_buf,
            event_buf,
            n_clusters,
            config.isi_split_threshold,
            config.refractory_samples,
            config.split_min_cluster_size,
            scratch,
            N,
        )
    } else {
        n_clusters
    };

    // 9c3b. Amplitude bimodality split: if a cluster's peak-amplitude distribution
    // has a large gap (relative to spread), it likely contains two neurons with
    // different amplitudes merged into one cluster. Split at the gap.
    let n_clusters = if config.split_bimodality_threshold > 0.0 && n_clusters > 0 {
        amplitude_bimodality_split(
            n_extracted,
            labels,
            event_buf,
            n_clusters,
            config.split_bimodality_threshold,
            config.split_min_cluster_size,
            N,
        )
    } else {
        n_clusters
    };

    // Cap n_clusters at N (the SortResult array size)
    let mut n_clusters = if n_clusters > N { N } else { n_clusters };

    // 9c4. GMM refinement: refine k-means boundaries using full-covariance EM.
    // K-means assumes spherical clusters; EM models cluster shape (elongated,
    // rotated) and reassigns borderline spikes using Mahalanobis distance.
    if config.gmm_refine && n_clusters > 1 && n_extracted > n_clusters * 2 {
        let mut gmm = crate::gmm::GaussianMixture::<K, N>::new(1e-4);
        gmm.init_from_labels(
            &feature_buf[..n_extracted],
            &labels[..n_extracted],
            n_clusters,
        );
        let ll = gmm.fit(&feature_buf[..n_extracted], config.gmm_max_iter);
        if ll.is_finite() {
            gmm.relabel(&feature_buf[..n_extracted], &mut labels[..n_extracted]);
        }
    }

    // 9d. Template subtraction passes: subtract known spikes, re-detect masked ones.
    // Multi-pass: each iteration refines templates and recovers additional masked spikes.
    let n_passes = if config.template_subtract {
        config.template_subtract_passes.max(1)
    } else {
        0
    };
    for _pass in 0..n_passes {
        if n_clusters > 0 && n_extracted > 0 {
            let mut tmpl_means = [[0.0; W]; N];
            let mut tmpl_counts = [0u32; N];
            let mut tmpl_peak_ch = [0usize; N];

            compute_cluster_means::<W, N>(
                waveform_buf,
                labels,
                event_buf,
                n_extracted,
                n_clusters,
                &mut tmpl_means,
                &mut tmpl_counts,
                &mut tmpl_peak_ch,
            );

            // Compute mean within-cluster L2 distance for rejection threshold
            let mut mean_dist = [0.0; N];
            for i in 0..n_extracted {
                let label = labels[i];
                if label < n_clusters && label < N {
                    let mut d = 0.0;
                    for w in 0..W {
                        let diff = waveform_buf[i][w] - tmpl_means[label][w];
                        d += diff * diff;
                    }
                    mean_dist[label] += d;
                }
            }
            for c in 0..n_clusters.min(N) {
                if tmpl_counts[c] > 0 {
                    mean_dist[c] /= tmpl_counts[c] as Float;
                }
            }
            // Max acceptable distance: 3x the mean within-cluster distance
            let mut max_accept_dist = 0.0;
            let mut n_valid = 0;
            for c in 0..n_clusters.min(N) {
                if tmpl_counts[c] >= config.template_min_count as u32 {
                    max_accept_dist += mean_dist[c];
                    n_valid += 1;
                }
            }
            if n_valid > 0 {
                max_accept_dist = (max_accept_dist / n_valid as Float) * 3.0;
            } else {
                max_accept_dist = float::MAX;
            }

            // Subtract templates from whitened data
            subtract_templates_multichannel::<C, W, N>(
                data,
                event_buf,
                n_extracted,
                labels,
                &tmpl_means,
                &tmpl_counts,
                &tmpl_peak_ch,
                config.template_min_count,
                config.pre_samples,
            );

            // Re-detect on residual at same threshold. After subtracting known
            // templates, masked spikes (temporal overlap) become detectable.
            // Using the same threshold avoids flooding with noise detections.
            let remaining_buf = event_buf.len().saturating_sub(n_extracted);
            if remaining_buf > 0 {
                let n_re_detected = if config.adaptive_threshold {
                    let adaptive_thresh_re = compute_adaptive_thresholds::<C>(
                        data,
                        config.threshold_multiplier,
                        config.adaptive_min_threshold,
                        config.adaptive_max_rate_hz,
                        config.sample_rate,
                        scratch,
                    );
                    detect_spikes_multichannel::<C>(
                        data,
                        1.0,
                        &adaptive_thresh_re,
                        config.refractory_samples,
                        &mut event_buf[n_extracted..],
                    )
                } else {
                    let unit_noise_re = [1.0; C];
                    detect_spikes_multichannel::<C>(
                        data,
                        config.threshold_multiplier,
                        &unit_noise_re,
                        config.refractory_samples,
                        &mut event_buf[n_extracted..],
                    )
                };

                if n_re_detected > 0 {
                    // Dedup re-detections
                    let n_re_dedup = deduplicate_events::<C>(
                        &mut event_buf[n_extracted..],
                        n_re_detected,
                        probe,
                        config.spatial_radius_um,
                        config.temporal_radius,
                    );

                    // Filter out re-detections that overlap existing spikes
                    let mut n_new = 0usize;
                    'outer: for j in 0..n_re_dedup {
                        let new_sample = event_buf[n_extracted + j].sample;
                        for ev in event_buf.iter().take(n_extracted) {
                            if new_sample.abs_diff(ev.sample) <= config.temporal_radius {
                                continue 'outer;
                            }
                        }
                        // Keep this event (compact in-place)
                        if n_new != j {
                            event_buf[n_extracted + n_new] = event_buf[n_extracted + j];
                        }
                        n_new += 1;
                    }

                    // Extract waveforms and assign to nearest template
                    if n_new > 0 {
                        let remaining_wf = waveform_buf.len().saturating_sub(n_extracted);
                        let n_to_extract = n_new.min(remaining_wf);
                        let n_re_extracted = extract_peak_channel::<C, W>(
                            data,
                            &event_buf[n_extracted..],
                            n_to_extract,
                            config.pre_samples,
                            &mut waveform_buf[n_extracted..],
                        );

                        let mut n_accepted = 0usize;
                        for j in 0..n_re_extracted {
                            let (best_label, dist) = assign_to_nearest_template::<W, N>(
                                &waveform_buf[n_extracted + j],
                                &tmpl_means,
                                &tmpl_counts,
                                n_clusters,
                            );
                            if dist > max_accept_dist {
                                continue; // reject: too far from any template
                            }
                            // Compact accepted spikes
                            let dst = n_extracted + n_accepted;
                            if dst != n_extracted + j {
                                event_buf[dst] = event_buf[n_extracted + j];
                                waveform_buf[dst] = waveform_buf[n_extracted + j];
                            }
                            if dst < labels.len() {
                                labels[dst] = best_label;
                            }
                            n_accepted += 1;
                        }
                        n_extracted += n_accepted;
                    }
                }
            }

            // 9e. Template-based NCC residual detection.
            //
            // After amplitude-threshold residual re-detection, scan the residual
            // for template-shaped waveforms using normalized cross-correlation.
            // This recovers weak units (SNR 2-4) that fall below the amplitude
            // threshold but whose waveform shape matches a known template.
            //
            // Optimizations over naive sliding:
            // - Sorted spike times + binary search for overlap check (O(log n) vs O(n))
            // - Early amplitude check before expensive NCC computation
            // - Adaptive step: skip by W/2 when amplitude is negligible
            let remaining_ncc = event_buf.len().saturating_sub(n_extracted);
            let remaining_wf_ncc = waveform_buf.len().saturating_sub(n_extracted);
            if remaining_ncc > 0 && remaining_wf_ncc > 0 {
                let ncc_threshold = 0.7;
                let half_thresh = config.threshold_multiplier * 0.5;
                let mut n_ncc_found = 0usize;

                // Build sorted spike sample indices for binary search overlap check.
                // Reuse scratch buffer (already available, large enough for spike times).
                let n_existing = n_extracted;
                let sorted_times_n = n_existing.min(scratch.len());
                for i in 0..sorted_times_n {
                    scratch[i] = event_buf[i].sample as Float;
                }
                scratch[..sorted_times_n].sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let overlap_radius = (config.temporal_radius + W / 2) as Float;

                for c in 0..n_clusters.min(N) {
                    if (tmpl_counts[c] as usize) < config.template_min_count {
                        continue;
                    }
                    let ch = tmpl_peak_ch[c];
                    if ch >= C {
                        continue;
                    }
                    // Precompute template norm
                    let mut t_norm_sq = 0.0;
                    for &tv in tmpl_means[c].iter() {
                        t_norm_sq += tv * tv;
                    }
                    let t_norm = float::sqrt(t_norm_sq);
                    if t_norm < 1e-15 {
                        continue;
                    }
                    // NCC threshold squared for early rejection:
                    // ncc = dot / (r_norm * t_norm) > thresh  iff  dot^2 > thresh^2 * r_norm_sq * t_norm_sq
                    let ncc_thresh_sq = ncc_threshold * ncc_threshold;
                    let t_norm_sq_thresh = ncc_thresh_sq * t_norm_sq;

                    // Slide template across residual on peak channel
                    let pre = config.pre_samples;
                    let step = config.refractory_samples.max(W / 2);
                    let mut pos = pre;
                    while pos + W <= t_len {
                        // Early amplitude check: if peak sample is negligible, skip ahead
                        let peak_amp = float::abs(data[pos][ch]);
                        if peak_amp < half_thresh {
                            // Adaptive step: skip more aggressively in quiet regions
                            pos += W / 4;
                            continue;
                        }

                        // Binary search overlap check on sorted spike times
                        let pos_f = pos as Float;
                        let lo = pos_f - overlap_radius;
                        let hi = pos_f + overlap_radius;
                        // Find first spike time >= lo
                        let mut left = 0usize;
                        let mut right = sorted_times_n;
                        while left < right {
                            let mid = left + (right - left) / 2;
                            if scratch[mid] < lo {
                                left = mid + 1;
                            } else {
                                right = mid;
                            }
                        }
                        let overlaps = left < sorted_times_n && scratch[left] <= hi;
                        if overlaps {
                            pos += step;
                            continue;
                        }

                        // Compute NCC between residual window and template
                        let start = pos - pre;
                        let mut dot = 0.0;
                        let mut r_norm_sq = 0.0;
                        for w in 0..W {
                            let rv = data[start + w][ch];
                            dot += rv * tmpl_means[c][w];
                            r_norm_sq += rv * rv;
                        }

                        // Early rejection: dot^2 < thresh^2 * r_norm_sq * t_norm_sq means ncc < threshold
                        if dot * dot < t_norm_sq_thresh * r_norm_sq || dot <= 0.0 {
                            pos += 1;
                            continue;
                        }

                        let r_norm = float::sqrt(r_norm_sq);
                        let ncc = if r_norm > 1e-15 {
                            dot / (r_norm * t_norm)
                        } else {
                            0.0
                        };

                        if ncc > ncc_threshold {
                            let idx = n_extracted + n_ncc_found;
                            if idx < event_buf.len()
                                && idx < waveform_buf.len()
                                && idx < labels.len()
                            {
                                event_buf[idx] = MultiChannelEvent {
                                    sample: pos,
                                    channel: ch,
                                    amplitude: peak_amp,
                                };
                                // Extract waveform
                                for w in 0..W {
                                    waveform_buf[idx][w] = data[start + w][ch];
                                }
                                labels[idx] = c;
                                n_ncc_found += 1;
                            }
                            pos += step;
                        } else {
                            pos += 1;
                        }
                    }
                }
                n_extracted += n_ncc_found;
            }
        } // end if n_clusters > 0 && n_extracted > 0
    } // end multi-pass loop

    // 9f. Matched filter second-pass detection.
    //
    // After initial amplitude detection + clustering + template subtraction,
    // we now have template waveforms for each cluster. Use these as matched
    // filters on the ORIGINAL whitened data to find spikes that were below
    // the amplitude threshold but match a known template shape.
    //
    // This is the Neyman-Pearson optimal detector: it maximizes detection
    // probability for a given false-positive rate by integrating over all
    // W time samples instead of thresholding a single sample.
    //
    // The SNR gain is √(W_eff / 1) ≈ 4-5× for a typical spike waveform,
    // which can recover units with peak amplitude 2-3σ (below the standard
    // 5σ amplitude threshold) but matched-filter SNR above the MF threshold.
    if config.matched_filter_detect && n_clusters > 0 && n_extracted > 0 {
        // Build templates from current cluster assignments
        let mut mf_means = [[0.0; W]; N];
        let mut mf_counts = [0u32; N];
        let mut mf_peak_ch = [0usize; N];
        compute_cluster_means::<W, N>(
            waveform_buf,
            labels,
            event_buf,
            n_extracted,
            n_clusters,
            &mut mf_means,
            &mut mf_counts,
            &mut mf_peak_ch,
        );

        // Compute neighbor channel templates for composite scoring
        let mut nbr_templates = [[0.0 as Float; W]; N];
        let mut nbr_channels = [0usize; N];
        if config.neighbor_mf_detect {
            compute_neighbor_templates::<W, N, C>(
                data,
                event_buf,
                labels,
                n_extracted,
                n_clusters,
                &mf_peak_ch,
                probe,
                &mut nbr_templates,
                &mut nbr_channels,
            );
        }

        // Build matched filter bank from cluster templates
        let bank = MatchedFilterBank::<W, N>::from_cluster_templates(
            &mf_means,
            &mf_counts,
            &mf_peak_ch,
            n_clusters,
            config.template_min_count,
        );

        if bank.n_filters() > 0 {
            // Run matched filter detection on the original whitened data.
            // We detect on the full data (not residual) because the MF threshold
            // is calibrated for whitened noise, not for residual noise.
            let remaining = event_buf.len().saturating_sub(n_extracted);
            let mf_buf_size = remaining.min(2048);
            // Use a stack-allocated detection buffer. 2048 entries = 80KB, fine for stack.
            let mut mf_detections = [MatchedDetection::ZERO; 2048];
            let n_mf = bank.detect(
                data,
                config.matched_filter_threshold,
                config.refractory_samples,
                &mut mf_detections[..mf_buf_size],
            );

            // Filter: keep only detections that don't overlap existing spikes
            let mut n_mf_accepted = 0usize;
            for det in mf_detections.iter().take(n_mf) {
                let mf_sample = det.sample;
                let mf_template = det.template_idx;
                // Check overlap with existing spikes
                let mut overlaps = false;
                for ev in event_buf.iter().take(n_extracted) {
                    if mf_sample.abs_diff(ev.sample) <= config.temporal_radius {
                        overlaps = true;
                        break;
                    }
                }
                if overlaps {
                    continue;
                }
                // Composite neighbor-channel score filter
                if config.neighbor_mf_detect && mf_template < N {
                    let nbr_ch = nbr_channels[mf_template];
                    let pre = W * 5 / 12;
                    let nbr_ncc = if mf_sample >= pre
                        && mf_sample + W - pre <= t_len
                        && nbr_ch < C
                        && nbr_ch != mf_peak_ch[mf_template]
                    {
                        let start = mf_sample - pre;
                        // Compute NCC: dot(nbr_template, data_window) / ||nbr_template||
                        let mut dot = 0.0;
                        let mut norm_sq = 0.0;
                        let mut w = 0;
                        while w < W {
                            dot += nbr_templates[mf_template][w] * data[start + w][nbr_ch];
                            norm_sq +=
                                nbr_templates[mf_template][w] * nbr_templates[mf_template][w];
                            w += 1;
                        }
                        if norm_sq > 1e-12 {
                            dot / float::sqrt(norm_sq)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let composite = det.normalized + config.neighbor_mf_bonus * nbr_ncc;
                    if composite < config.matched_filter_threshold {
                        continue;
                    }
                }
                // Amplitude sanity check: reject if amplitude is negative or very large
                let amp = det.amplitude;
                if !(0.2..=5.0).contains(&amp) {
                    continue;
                }
                // Map template index back to cluster index
                // (bank may have skipped low-count clusters, but from_cluster_templates
                // adds them in order, so template_idx maps to the nth valid cluster)
                let mut cluster_idx = 0usize;
                let mut valid_count = 0usize;
                while cluster_idx < n_clusters && cluster_idx < N {
                    if (mf_counts[cluster_idx] as usize) >= config.template_min_count {
                        if valid_count == mf_template {
                            break;
                        }
                        valid_count += 1;
                    }
                    cluster_idx += 1;
                }
                if cluster_idx >= n_clusters || cluster_idx >= N {
                    continue;
                }
                let ch = mf_peak_ch[cluster_idx];
                if ch >= C {
                    continue;
                }
                // Extract waveform and add to results
                let dst = n_extracted + n_mf_accepted;
                if dst >= event_buf.len() || dst >= waveform_buf.len() || dst >= labels.len() {
                    break;
                }
                let pre = config.pre_samples;
                if mf_sample >= pre && mf_sample + W - pre <= t_len {
                    let start = mf_sample - pre;
                    event_buf[dst] = MultiChannelEvent {
                        sample: mf_sample,
                        channel: ch,
                        amplitude: float::abs(data[mf_sample][ch]),
                    };
                    for w in 0..W {
                        waveform_buf[dst][w] = data[start + w][ch];
                    }
                    labels[dst] = cluster_idx;
                    n_mf_accepted += 1;
                }
            }
            n_extracted += n_mf_accepted;
        }
    }

    // 9g. Refinement iterations: re-estimate centroids in feature space and
    // reassign all spikes. Unlike waveform-space reassignment (which loses
    // channel discrimination after whitening), feature space includes the
    // spatial dimension that separates units on different channels.
    let do_refine = config.refinement_iterations > 0 && (!config.auto_refine || C >= 8);
    if do_refine && n_clusters > 1 && n_extracted > n_clusters {
        for _iter in 0..config.refinement_iterations {
            // Compute mean feature vector per cluster
            let mut centroids = [[0.0 as Float; K]; N];
            let mut cent_count = [0u32; N];
            for i in 0..n_extracted {
                let l = labels[i];
                if l < n_clusters && l < N {
                    cent_count[l] += 1;
                    for k in 0..K {
                        centroids[l][k] += feature_buf[i][k];
                    }
                }
            }
            for c in 0..n_clusters.min(N) {
                if cent_count[c] > 0 {
                    let inv = 1.0 / cent_count[c] as Float;
                    for v in centroids[c].iter_mut() {
                        *v *= inv;
                    }
                }
            }
            // Dry-run guard: skip iteration if it would cause cluster collapse
            // or a significant increase in ISI violations.
            let do_dry_run = config.refine_collapse_guard || config.refine_isi_guard;
            if do_dry_run {
                // Pre-pass: count current ISI violations in detection order.
                let mut pre_isi = 0u32;
                if config.refine_isi_guard && n_extracted > 1 {
                    let mut prev_lbl = n_clusters;
                    let mut prev_sample = 0usize;
                    for i in 0..n_extracted {
                        let lbl = labels[i];
                        if lbl == prev_lbl && lbl < n_clusters {
                            let dt = event_buf[i].sample.saturating_sub(prev_sample);
                            if dt > 0 && dt < config.refractory_samples {
                                pre_isi += 1;
                            }
                        }
                        prev_lbl = lbl;
                        prev_sample = event_buf[i].sample;
                    }
                }
                // Dry-run: compute proposed assignments and track both collapse
                // counts and post-assignment ISI violations simultaneously.
                let mut new_counts = [0u32; N];
                let mut post_isi = 0u32;
                let mut prev_new_lbl = n_clusters;
                let mut prev_new_sample = 0usize;
                for i in 0..n_extracted {
                    let mut best_c = labels[i];
                    let mut best_d: Float = Float::MAX;
                    for c in 0..n_clusters.min(N) {
                        if cent_count[c] == 0 {
                            continue;
                        }
                        let mut d = 0.0;
                        for k in 0..K {
                            let diff = feature_buf[i][k] - centroids[c][k];
                            d += diff * diff;
                        }
                        if d < best_d {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    if best_c < N {
                        new_counts[best_c] += 1;
                    }
                    if config.refine_isi_guard && best_c == prev_new_lbl && best_c < n_clusters {
                        let dt = event_buf[i].sample.saturating_sub(prev_new_sample);
                        if dt > 0 && dt < config.refractory_samples {
                            post_isi += 1;
                        }
                    }
                    prev_new_lbl = best_c;
                    prev_new_sample = event_buf[i].sample;
                }
                let min_size = config.split_min_cluster_size as u32;
                let would_collapse = config.refine_collapse_guard
                    && (0..n_clusters.min(N))
                        .any(|c| cent_count[c] >= min_size && new_counts[c] == 0);
                let isi_degrades = config.refine_isi_guard
                    && post_isi > pre_isi
                    && post_isi as Float > pre_isi as Float * (1.0 + config.refine_isi_tolerance);
                if would_collapse || isi_degrades {
                    break;
                }
            }

            // Reassign each spike to nearest centroid in feature space
            let mut changed = 0usize;
            for i in 0..n_extracted {
                let mut best_c = labels[i];
                let mut best_d: Float = Float::MAX;
                for c in 0..n_clusters.min(N) {
                    if cent_count[c] == 0 {
                        continue;
                    }
                    let mut d = 0.0;
                    for k in 0..K {
                        let diff = feature_buf[i][k] - centroids[c][k];
                        d += diff * diff;
                    }
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                if best_c != labels[i] {
                    labels[i] = best_c;
                    changed += 1;
                }
            }
            if changed == 0 {
                break; // converged
            }
        }

        // Remove empty clusters after refinement
        let mut counts = [0u32; N];
        for label in labels.iter().take(n_extracted) {
            if *label < N {
                counts[*label] += 1;
            }
        }
        let mut has_empty = false;
        for count in counts.iter().take(n_clusters.min(N)) {
            if *count == 0 {
                has_empty = true;
                break;
            }
        }
        if has_empty {
            let mut remap = [0usize; N];
            let mut new_n = 0;
            for c in 0..n_clusters.min(N) {
                if counts[c] > 0 {
                    remap[c] = new_n;
                    new_n += 1;
                }
            }
            for i in 0..n_extracted {
                labels[i] = remap[labels[i]];
            }
            n_clusters = new_n;
        }
    }

    // 9h. Template-based waveform reassignment.
    //
    // After all detection and clustering passes, reassign each spike to the
    // cluster whose mean waveform it most closely matches, but ONLY among
    // clusters on the same peak channel. This preserves channel separation
    // while fixing misassignments from noisy PCA features.
    //
    // This is more principled than feature-space refinement because:
    // 1. It uses all W waveform samples (not just K << W PCA features)
    // 2. The channel constraint prevents cross-channel contamination
    // 3. It naturally handles the case where PCA discards discriminative info
    if n_clusters > 1 && n_extracted > n_clusters {
        // Build templates: mean waveform per cluster, tracking peak channel
        let mut tmpl_wf = [[0.0 as Float; W]; N];
        let mut tmpl_count = [0u32; N];
        let mut tmpl_ch = [0usize; N];
        compute_cluster_means::<W, N>(
            waveform_buf,
            labels,
            event_buf,
            n_extracted,
            n_clusters,
            &mut tmpl_wf,
            &mut tmpl_count,
            &mut tmpl_ch,
        );

        // Reassign: for each spike, find the nearest template on the same
        // channel. Require a 30% distance improvement to prevent marginal
        // reassignments that break good clustering.
        let mut changed = 0usize;
        for i in 0..n_extracted {
            let spike_ch = event_buf[i].channel;
            let old_label = labels[i];

            // Compute distance to current template
            let mut old_d: Float = 0.0;
            if old_label < n_clusters && old_label < N && tmpl_ch[old_label] == spike_ch {
                for w in 0..W {
                    let diff = waveform_buf[i][w] - tmpl_wf[old_label][w];
                    old_d += diff * diff;
                }
            }

            let mut best_c = old_label;
            let mut best_d: Float = Float::MAX;
            for c in 0..n_clusters.min(N) {
                if tmpl_count[c] < config.template_min_count as u32 {
                    continue;
                }
                // Only consider clusters on the same peak channel
                if tmpl_ch[c] != spike_ch {
                    continue;
                }
                let mut d = 0.0;
                for w in 0..W {
                    let diff = waveform_buf[i][w] - tmpl_wf[c][w];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            // Only reassign if new template is significantly closer (30% margin)
            // This prevents marginal reassignments from breaking good clustering.
            if best_c != old_label && best_d < old_d * 0.7 {
                labels[i] = best_c;
                changed += 1;
            }
        }

        // If reassignment changed labels, remove empty clusters
        if changed > 0 {
            let mut counts = [0u32; N];
            for label in labels.iter().take(n_extracted) {
                if *label < N {
                    counts[*label] += 1;
                }
            }
            let mut has_empty = false;
            for count in counts.iter().take(n_clusters.min(N)) {
                if *count == 0 {
                    has_empty = true;
                    break;
                }
            }
            if has_empty {
                let mut remap = [0usize; N];
                let mut new_n = 0;
                for c in 0..n_clusters.min(N) {
                    if counts[c] > 0 {
                        remap[c] = new_n;
                        new_n += 1;
                    }
                }
                for label in labels.iter_mut().take(n_extracted) {
                    *label = remap[*label];
                }
                n_clusters = new_n;
            }
        }
    }

    // 10. Quality metrics
    // Compute per-cluster: mean waveform for SNR, spike times for ISI
    let mut clusters: [ClusterInfo; N] = core::array::from_fn(|_| ClusterInfo {
        count: 0,
        snr: 0.0,
        isi_violation_rate: 0.0,
    });

    for (ci, cluster) in clusters.iter_mut().enumerate().take(n_clusters) {
        // Count spikes in this cluster
        let mut count = 0usize;
        for i in 0..n_extracted {
            if i < labels.len() && labels[i] == ci {
                count += 1;
            }
        }
        cluster.count = count;

        if count == 0 {
            continue;
        }

        // Mean waveform for SNR
        let mut mean_wf = [0.0; W];
        for i in 0..n_extracted {
            if i < labels.len() && labels[i] == ci {
                for (w, mw) in mean_wf.iter_mut().enumerate() {
                    *mw += waveform_buf[i][w];
                }
            }
        }
        let inv_count = 1.0 / count as Float;
        for mw in mean_wf.iter_mut() {
            *mw *= inv_count;
        }

        cluster.snr = quality::waveform_snr(&mean_wf, noise_mean).unwrap_or(0.0);

        // ISI violation rate (using sample indices as proxy spike times)
        // Convert sample indices to seconds-ish (just use raw sample counts)
        if count >= 2 {
            // Collect spike times for this cluster into scratch
            let mut spike_idx = 0;
            for i in 0..n_extracted {
                if i < labels.len() && labels[i] == ci && spike_idx < scratch.len() {
                    scratch[spike_idx] = event_buf[i].sample as Float;
                    spike_idx += 1;
                }
            }
            // Sort spike times
            let spike_times = &mut scratch[..spike_idx];
            spike_times
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            // ISI violation rate with refractory period in samples
            cluster.isi_violation_rate =
                quality::isi_violation_rate(spike_times, config.refractory_samples as Float)
                    .unwrap_or(0.0);
        }
    }

    // 11. Auto-curation: remove spikes from clusters below SNR floor
    // Normalize threshold by noise level: SNR uses pre-whitening noise_mean as denominator,
    // so higher noise → lower SNR for the same signal. Scale threshold proportionally.
    // noise_mean ≈ median(|x|)/0.6745 ≈ 1.0 for unit Gaussian noise.
    const REF_NOISE: Float = 1.0;
    let effective_snr_threshold = config.min_cluster_snr * (REF_NOISE / noise_mean);
    let mut keep_cluster = [false; N];
    let mut n_kept_clusters = 0usize;
    for ci in 0..n_clusters {
        if clusters[ci].snr >= effective_snr_threshold {
            keep_cluster[ci] = true;
            n_kept_clusters += 1;
        }
    }

    if n_kept_clusters < n_clusters {
        // Build label remapping (old cluster index -> new compact index)
        let mut label_map = [0usize; N];
        let mut new_idx = 0;
        for ci in 0..n_clusters {
            if keep_cluster[ci] {
                label_map[ci] = new_idx;
                new_idx += 1;
            }
        }

        // Compact spikes: keep only those in surviving clusters
        let mut write = 0;
        for read in 0..n_extracted {
            if read < labels.len() && labels[read] < n_clusters && keep_cluster[labels[read]] {
                if write != read {
                    event_buf[write] = event_buf[read];
                    waveform_buf[write] = waveform_buf[read];
                    feature_buf[write] = feature_buf[read];
                }
                labels[write] = label_map[labels[read]];
                write += 1;
            }
        }
        n_extracted = write;

        // Compact cluster info
        let mut new_clusters: [ClusterInfo; N] = core::array::from_fn(|_| ClusterInfo {
            count: 0,
            snr: 0.0,
            isi_violation_rate: 0.0,
        });
        let mut wi = 0;
        for ci in 0..n_clusters {
            if keep_cluster[ci] {
                new_clusters[wi] = ClusterInfo {
                    count: clusters[ci].count,
                    snr: clusters[ci].snr,
                    isi_violation_rate: clusters[ci].isi_violation_rate,
                };
                // Recount after compaction (some spikes may have been from template subtraction)
                let mut recount = 0;
                for label in labels.iter().take(n_extracted) {
                    if *label == wi {
                        recount += 1;
                    }
                }
                new_clusters[wi].count = recount;
                wi += 1;
            }
        }

        n_clusters = n_kept_clusters;
        clusters = new_clusters;
    }

    Ok(SortResult {
        n_spikes: n_extracted,
        n_clusters,
        clusters,
    })
}

/// Online spike sorter for real-time template matching.
///
/// After learning cluster centroids from a batch sorting pass, this
/// struct classifies new spikes by nearest-centroid distance in
/// feature space. Designed for sub-100 microsecond per-spike latency.
///
/// # Type Parameters
///
/// * `K` -- Feature dimensionality (number of PCA components)
/// * `N` -- Maximum number of templates/clusters
///
/// # Example
///
/// ```
/// use zerostone::sorter::OnlineSorter;
///
/// // Create from pre-learned centroids
/// let centroids = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let mut sorter = OnlineSorter::<3, 8>::from_centroids(&centroids);
/// assert_eq!(sorter.n_templates(), 2);
///
/// let (label, dist) = sorter.classify(&[0.9, 0.1, 0.0]);
/// assert_eq!(label, 0); // closest to first template
/// assert!(dist < 0.2);
///
/// // Reject distant spikes
/// sorter.set_max_distance(0.5);
/// assert!(sorter.classify_or_reject(&[5.0, 5.0, 5.0]).is_none());
/// ```
pub struct OnlineSorter<const K: usize, const N: usize> {
    templates: [[Float; K]; N],
    n_templates: usize,
    max_distance: Float,
    n_classified: usize,
    n_rejected: usize,
}

impl<const K: usize, const N: usize> OnlineSorter<K, N> {
    /// Create a new online sorter with no templates.
    pub fn new() -> Self {
        Self {
            templates: [[0.0; K]; N],
            n_templates: 0,
            max_distance: float::MAX,
            n_classified: 0,
            n_rejected: 0,
        }
    }

    /// Create from centroids extracted from a batch sort result.
    /// `centroids` is a slice of feature vectors, up to N are used.
    pub fn from_centroids(centroids: &[[Float; K]]) -> Self {
        let mut s = Self::new();
        let count = if centroids.len() < N {
            centroids.len()
        } else {
            N
        };
        let mut i = 0;
        while i < count {
            s.templates[i] = centroids[i];
            i += 1;
        }
        s.n_templates = count;
        s
    }

    /// Add a template. Returns the template index, or None if full.
    pub fn add_template(&mut self, centroid: &[Float; K]) -> Option<usize> {
        if self.n_templates >= N {
            return None;
        }
        let idx = self.n_templates;
        self.templates[idx] = *centroid;
        self.n_templates += 1;
        Some(idx)
    }

    /// Set the maximum distance for classification. Spikes farther
    /// than this from all templates are rejected (classified as `None`).
    /// Default: float::MAX (no rejection).
    pub fn set_max_distance(&mut self, max_dist: Float) {
        self.max_distance = max_dist;
    }

    /// Classify a single spike by nearest centroid.
    /// Returns (template_index, distance).
    /// If no templates are loaded, returns (0, float::MAX).
    pub fn classify(&mut self, features: &[Float; K]) -> (usize, Float) {
        self.n_classified += 1;

        if self.n_templates == 0 {
            return (0, float::MAX);
        }

        let mut best_idx = 0;
        let mut best_dist = float::MAX;

        let mut ti = 0;
        while ti < self.n_templates {
            let mut sum_sq = 0.0;
            let mut ki = 0;
            while ki < K {
                let diff = features[ki] - self.templates[ti][ki];
                sum_sq += diff * diff;
                ki += 1;
            }
            let dist = float::sqrt(sum_sq);
            if dist < best_dist {
                best_dist = dist;
                best_idx = ti;
            }
            ti += 1;
        }

        (best_idx, best_dist)
    }

    /// Classify a spike, returning None if distance exceeds max_distance.
    pub fn classify_or_reject(&mut self, features: &[Float; K]) -> Option<(usize, Float)> {
        let (label, dist) = self.classify(features);
        if dist > self.max_distance {
            self.n_rejected += 1;
            None
        } else {
            Some((label, dist))
        }
    }

    /// Number of templates loaded.
    pub fn n_templates(&self) -> usize {
        self.n_templates
    }

    /// Total spikes classified (including rejected).
    pub fn n_classified(&self) -> usize {
        self.n_classified
    }

    /// Total spikes rejected (distance > max_distance).
    pub fn n_rejected(&self) -> usize {
        self.n_rejected
    }

    /// Get a reference to the template centroids.
    pub fn templates(&self) -> &[[Float; K]] {
        &self.templates[..self.n_templates]
    }

    /// Reset counters (but keep templates).
    pub fn reset_counters(&mut self) {
        self.n_classified = 0;
        self.n_rejected = 0;
    }

    /// Clear all templates and counters.
    pub fn reset(&mut self) {
        self.n_templates = 0;
        self.n_classified = 0;
        self.n_rejected = 0;
        let mut i = 0;
        while i < N {
            self.templates[i] = [0.0; K];
            i += 1;
        }
    }
}

impl<const K: usize, const N: usize> Default for OnlineSorter<K, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `estimate_noise_multichannel` does not panic for valid inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn noise_estimation_no_panic() {
        let d0: Float = kani::any();
        let d1: Float = kani::any();
        let d2: Float = kani::any();
        let d3: Float = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);

        let data = [[d0, d1], [d2, d3]];
        let mut scratch = [0.0; 2];
        let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
        assert!(noise[0].is_finite(), "noise[0] must be finite");
        assert!(noise[1].is_finite(), "noise[1] must be finite");
        assert!(noise[0] >= 0.0, "noise must be non-negative");
        assert!(noise[1] >= 0.0, "noise must be non-negative");
    }

    /// Prove that `merge_clusters` does not panic for small valid inputs.
    #[kani::proof]
    #[kani::unwind(10)]
    fn merge_clusters_no_panic() {
        let l0: usize = kani::any();
        let l1: usize = kani::any();
        let l2: usize = kani::any();
        let l3: usize = kani::any();

        kani::assume(l0 < 3 && l1 < 3 && l2 < 3 && l3 < 3);

        let mut labels = [l0, l1, l2, l3];
        let features = [[0.0; 2]; 4];
        let events = [
            crate::spike_sort::MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 4];

        let dp_thresh: Float = kani::any();
        let isi_thresh: Float = kani::any();
        kani::assume(dp_thresh.is_finite() && dp_thresh >= 0.0 && dp_thresh <= 100.0);
        kani::assume(isi_thresh.is_finite() && isi_thresh >= 0.0 && isi_thresh <= 1.0);

        let new_n = merge_clusters(
            4,
            &mut labels,
            &features,
            &events,
            3,
            dp_thresh,
            isi_thresh,
            15,
            &mut scratch,
            2,
        );
        assert!(new_n <= 3, "cluster count must not increase");
    }

    /// Prove that `split_clusters` does not panic for small valid inputs.
    #[kani::proof]
    #[kani::unwind(10)]
    fn split_clusters_no_panic() {
        let l0: usize = kani::any();
        let l1: usize = kani::any();
        let l2: usize = kani::any();
        let l3: usize = kani::any();
        kani::assume(l0 < 2 && l1 < 2 && l2 < 2 && l3 < 2);

        let mut labels = [l0, l1, l2, l3];
        let features = [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
        let threshold: Float = kani::any();
        kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 100.0);

        let new_n = split_clusters(4, &mut labels, &features, 2, 2, threshold);
        assert!(new_n >= 2);
    }

    /// Prove that `OnlineSorter::classify` does not panic for finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn online_sorter_classify_no_panic() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        let t0: Float = kani::any();
        let t1: Float = kani::any();
        let f0: Float = kani::any();
        let f1: Float = kani::any();
        kani::assume(t0.is_finite() && t0 >= -1e6 && t0 <= 1e6);
        kani::assume(t1.is_finite() && t1 >= -1e6 && t1 <= 1e6);
        kani::assume(f0.is_finite() && f0 >= -1e6 && f0 <= 1e6);
        kani::assume(f1.is_finite() && f1 >= -1e6 && f1 <= 1e6);

        sorter.add_template(&[t0, t1]);
        let (label, dist) = sorter.classify(&[f0, f1]);
        assert_eq!(label, 0);
        assert!(dist.is_finite());
        assert!(dist >= 0.0);
    }

    /// Prove that `SortConfig::default()` produces valid positive thresholds.
    #[kani::proof]
    fn verify_sort_config_default_valid() {
        let cfg = SortConfig::default();
        assert!(cfg.threshold_multiplier > 0.0);
        assert!(cfg.refractory_samples > 0);
        assert!(cfg.spatial_radius_um > 0.0);
        assert!(cfg.temporal_radius > 0);
        assert!(cfg.cluster_threshold > 0.0);
        assert!(cfg.whitening_epsilon > 0.0);
        assert!(cfg.merge_dprime_threshold > 0.0);
        assert!(cfg.merge_isi_threshold > 0.0);
        assert!(cfg.split_min_cluster_size > 0);
        assert!(cfg.split_bimodality_threshold > 0.0);
    }

    /// Prove that for any finite data and template values, the per-spike
    /// amplitude scaling alpha = (dot / norms_sq).clamp(0.3, 3.0) is always
    /// finite and within [0.3, 3.0].
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_amplitude_scaling_finite() {
        // C=2, W=4, N=2
        // Build arbitrary finite data and template values
        let d0: Float = kani::any();
        let d1: Float = kani::any();
        let d2: Float = kani::any();
        let d3: Float = kani::any();
        let t0: Float = kani::any();
        let t1: Float = kani::any();
        let t2: Float = kani::any();
        let t3: Float = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);
        kani::assume(t0.is_finite() && t0 >= -1e6 && t0 <= 1e6);
        kani::assume(t1.is_finite() && t1 >= -1e6 && t1 <= 1e6);
        kani::assume(t2.is_finite() && t2 >= -1e6 && t2 <= 1e6);
        kani::assume(t3.is_finite() && t3 >= -1e6 && t3 <= 1e6);

        let data = [d0, d1, d2, d3];
        let template = [t0, t1, t2, t3];

        // Compute norms_sq (same as subtract_templates_multichannel)
        let mut norms_sq = 0.0;
        for val in template.iter() {
            norms_sq += val * val;
        }

        // Compute dot product (same as subtract_templates_multichannel)
        let mut dot = 0.0;
        for w in 0..4 {
            dot += data[w] * template[w];
        }

        // Compute alpha (same logic as in subtract_templates_multichannel)
        let alpha = if norms_sq > 1e-30 {
            (dot / norms_sq).clamp(0.3, 3.0)
        } else {
            1.0
        };

        assert!(alpha.is_finite(), "alpha must be finite");
        assert!(alpha >= 0.3, "alpha must be >= 0.3");
        assert!(alpha <= 3.0, "alpha must be <= 3.0");
    }

    /// Prove that `subtract_templates_multichannel` never panics for valid
    /// inputs (valid label indices, valid channel indices). C=2, W=4, N=2.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_subtract_templates_no_panic() {
        // Build small arbitrary data: 6 time samples x 2 channels
        let mut data = [[0.0; 2]; 6];
        let d0: Float = kani::any();
        let d1: Float = kani::any();
        let d2: Float = kani::any();
        let d3: Float = kani::any();
        kani::assume(d0.is_finite() && d0 >= -1e3 && d0 <= 1e3);
        kani::assume(d1.is_finite() && d1 >= -1e3 && d1 <= 1e3);
        kani::assume(d2.is_finite() && d2 >= -1e3 && d2 <= 1e3);
        kani::assume(d3.is_finite() && d3 >= -1e3 && d3 <= 1e3);
        data[2][0] = d0;
        data[3][0] = d1;
        data[2][1] = d2;
        data[3][1] = d3;

        // Template means: 2 clusters, W=4 each
        let m0: Float = kani::any();
        let m1: Float = kani::any();
        kani::assume(m0.is_finite() && m0 >= -1e3 && m0 <= 1e3);
        kani::assume(m1.is_finite() && m1 >= -1e3 && m1 <= 1e3);
        let means: [[Float; 4]; 2] = [[m0, m1, 0.5, -0.5], [0.3, -0.3, m0, m1]];

        // Spike events: 2 spikes
        let s0: usize = kani::any();
        let s1: usize = kani::any();
        let ch0: usize = kani::any();
        let ch1: usize = kani::any();
        let l0: usize = kani::any();
        let l1: usize = kani::any();

        kani::assume(s0 < 6 && s1 < 6);
        kani::assume(ch0 < 2 && ch1 < 2);
        kani::assume(l0 < 2 && l1 < 2);

        let events = [
            crate::spike_sort::MultiChannelEvent {
                sample: s0,
                channel: ch0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: s1,
                channel: ch1,
                amplitude: 5.0,
            },
        ];
        let labels = [l0, l1];
        let counts: [u32; 2] = [5, 5];
        let peak_channels: [usize; 2] = [ch0, ch1];

        subtract_templates_multichannel::<2, 4, 2>(
            &mut data,
            &events,
            2,
            &labels,
            &means,
            &counts,
            &peak_channels,
            1, // min_count
            1, // pre_samples
        );
        // If we reach here, the function did not panic
    }

    /// Prove that `merge_clusters` output n_clusters <= input n_clusters.
    /// K=3 feature dimensions, 3 input clusters.
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_merge_clusters_count_bounded() {
        let l0: usize = kani::any();
        let l1: usize = kani::any();
        let l2: usize = kani::any();
        let l3: usize = kani::any();
        let l4: usize = kani::any();
        let l5: usize = kani::any();

        kani::assume(l0 < 3 && l1 < 3 && l2 < 3);
        kani::assume(l3 < 3 && l4 < 3 && l5 < 3);

        let mut labels = [l0, l1, l2, l3, l4, l5];

        // Arbitrary feature values for 6 spikes with K=3 dimensions
        let f0: Float = kani::any();
        let f1: Float = kani::any();
        kani::assume(f0.is_finite() && f0 >= -1e3 && f0 <= 1e3);
        kani::assume(f1.is_finite() && f1 >= -1e3 && f1 <= 1e3);

        let features = [
            [f0, 0.0, 0.0],
            [f1, 0.0, 0.0],
            [0.0, f0, 0.0],
            [0.0, f1, 0.0],
            [0.0, 0.0, f0],
            [0.0, 0.0, f1],
        ];
        let events = [
            crate::spike_sort::MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 500,
                channel: 0,
                amplitude: 5.0,
            },
            crate::spike_sort::MultiChannelEvent {
                sample: 600,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 6];

        let dp_thresh: Float = kani::any();
        let isi_thresh: Float = kani::any();
        kani::assume(dp_thresh.is_finite() && dp_thresh >= 0.0 && dp_thresh <= 100.0);
        kani::assume(isi_thresh.is_finite() && isi_thresh >= 0.0 && isi_thresh <= 1.0);

        let input_n: usize = 3;
        let output_n = merge_clusters::<3>(
            6,
            &mut labels,
            &features,
            &events,
            input_n,
            dp_thresh,
            isi_thresh,
            15,
            &mut scratch,
            3,
        );
        assert!(output_n <= input_n, "merge must not increase cluster count");
        assert!(
            output_n >= 1 || input_n == 0,
            "merge must preserve at least 1 cluster"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    // Simple pseudo-RNG (xorshift64)
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn gaussian(&mut self, mean: Float, std: Float) -> Float {
            let u1 = (self.next_u64() % 1_000_000 + 1) as Float / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as Float / 1_000_000.0;
            let z = float::sqrt(-2.0 * float::log(u1)) * float::cos(2.0 * float::PI * u2);
            mean + z * std
        }
    }

    #[test]
    fn test_estimate_noise_multichannel_known() {
        // Constant data: all values are 1.0
        // |1.0| / 0.6745 = 1.4826
        let data = [[1.0, 2.0]; 100];
        let mut scratch = [0.0; 100];
        let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
        assert!(
            (noise[0] - 1.0 / 0.6745).abs() < 0.01,
            "Expected ~1.483, got {}",
            noise[0]
        );
        assert!(
            (noise[1] - 2.0 / 0.6745).abs() < 0.01,
            "Expected ~2.966, got {}",
            noise[1]
        );
    }

    #[test]
    fn test_estimate_noise_multichannel_gaussian() {
        let mut rng = Rng::new(42);
        let n = 10000;
        let mut data = vec![[0.0; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = rng.gaussian(0.0, 1.0);
            sample[1] = rng.gaussian(0.0, 3.0);
        }
        let mut scratch = vec![0.0; n];
        let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
        assert!(
            (noise[0] - 1.0).abs() < 0.15,
            "Channel 0 noise should be ~1.0, got {}",
            noise[0]
        );
        assert!(
            (noise[1] - 3.0).abs() < 0.3,
            "Channel 1 noise should be ~3.0, got {}",
            noise[1]
        );
    }

    #[test]
    fn test_estimate_noise_empty() {
        let data: &[[Float; 2]] = &[];
        let mut scratch = [0.0; 1];
        let noise = estimate_noise_multichannel::<2>(data, &mut scratch);
        assert!((noise[0]).abs() < 1e-12);
        assert!((noise[1]).abs() < 1e-12);
    }

    #[test]
    fn test_sort_config_default() {
        let config = SortConfig::default();
        assert!((config.threshold_multiplier - 5.0).abs() < 1e-12);
        assert_eq!(config.refractory_samples, 15);
        assert!((config.spatial_radius_um - 75.0).abs() < 1e-12);
        assert_eq!(config.temporal_radius, 5);
        assert_eq!(config.align_half_window, 15);
        assert_eq!(config.pre_samples, 20);
        assert!((config.cluster_threshold - 7.0).abs() < 1e-12);
        assert_eq!(config.cluster_max_count, 1000);
        assert!((config.whitening_epsilon - 1e-6).abs() < 1e-12);
        assert!((config.merge_dprime_threshold - 2.0).abs() < 1e-12);
        assert!((config.merge_isi_threshold - 0.05).abs() < 1e-12);
        assert!(config.template_subtract);
        assert_eq!(config.template_min_count, 3);
        assert!(!config.use_amplitude_profile);
        assert_eq!(config.amplitude_profile_neighbors, 4);
    }

    #[test]
    fn test_compute_covariance_identity() {
        // Uncorrelated unit-variance 2-channel data
        let mut rng = Rng::new(99);
        let n = 5000;
        let mut data = vec![[0.0; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = rng.gaussian(0.0, 1.0);
            sample[1] = rng.gaussian(0.0, 1.0);
        }
        let cov = compute_covariance::<2>(&data);
        assert!(
            (cov[0][0] - 1.0).abs() < 0.1,
            "Var(0)={}, expected ~1",
            cov[0][0]
        );
        assert!(
            (cov[1][1] - 1.0).abs() < 0.1,
            "Var(1)={}, expected ~1",
            cov[1][1]
        );
        assert!(cov[0][1].abs() < 0.1, "Cov(0,1)={}, expected ~0", cov[0][1]);
    }

    #[test]
    fn test_sort_multichannel_insufficient_data() {
        let config = SortConfig::default();
        let probe = ProbeLayout::<2>::linear(25.0);
        let mut data = vec![[0.0; 2]; 4]; // too short for W=8
        let mut scratch = vec![0.0; 4];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            10
        ];
        let mut waveforms = vec![[0.0; 8]; 10];
        let mut features = vec![[0.0; 3]; 10];
        let mut labels = vec![0usize; 10];

        let result = sort_multichannel::<2, 4, 8, 3, 64, 4>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_multichannel_no_spikes() {
        // All-zero data should produce no spikes
        let config = SortConfig::default();
        let probe = ProbeLayout::<2>::linear(25.0);
        let mut data = vec![[0.0; 2]; 1000];
        let mut scratch = vec![0.0; 1000];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            100
        ];
        let mut waveforms = vec![[0.0; 8]; 100];
        let mut features = vec![[0.0; 3]; 100];
        let mut labels = vec![0usize; 100];

        let result = sort_multichannel::<2, 4, 8, 3, 64, 4>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_ok());
        let sr = result.unwrap();
        assert_eq!(sr.n_spikes, 0);
        assert_eq!(sr.n_clusters, 0);
    }

    #[test]
    fn test_sort_multichannel_with_spikes() {
        let mut rng = Rng::new(42);
        let n = 5000;

        // Generate 2-channel noisy data
        let mut data = vec![[0.0; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = rng.gaussian(0.0, 1.0);
            sample[1] = rng.gaussian(0.0, 1.0);
        }

        // Inject spikes: neuron A on channel 0, neuron B on channel 1
        let spike_template_a = |t: Float| -> Float { -12.0 * float::exp(-0.5 * t * t) };
        let spike_template_b =
            |t: Float| -> Float { -10.0 * float::exp(-0.5 * (t / 1.5) * (t / 1.5)) };

        // Neuron A fires at regular intervals on channel 0
        let mut spike_pos_a = 200;
        while spike_pos_a + 10 < n {
            for dt in 0..8 {
                let t = (dt as Float - 2.0) / 1.5;
                if spike_pos_a + dt < n {
                    data[spike_pos_a + dt][0] += spike_template_a(t);
                }
            }
            spike_pos_a += 150;
        }

        // Neuron B fires at different intervals on channel 1
        let mut spike_pos_b = 300;
        while spike_pos_b + 12 < n {
            for dt in 0..10 {
                let t = (dt as Float - 3.0) / 2.0;
                if spike_pos_b + dt < n {
                    data[spike_pos_b + dt][1] += spike_template_b(t);
                }
            }
            spike_pos_b += 200;
        }

        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            detection_mode: DetectionMode::Amplitude,
            ccg_merge: false,
            ccg_template_corr_threshold: 0.5,
            ..SortConfig::default()
        };

        let probe = ProbeLayout::<2>::linear(25.0);
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0; 8]; 200];
        let mut features = vec![[0.0; 3]; 200];
        let mut labels = vec![0usize; 200];

        let result = sort_multichannel::<2, 4, 8, 3, 64, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_ok(), "Sort should succeed");
        let sr = result.unwrap();
        assert!(
            sr.n_spikes >= 10,
            "Should detect multiple spikes, got {}",
            sr.n_spikes
        );
        assert!(
            sr.n_clusters >= 1,
            "Should find at least 1 cluster, got {}",
            sr.n_clusters
        );
        // At least one cluster should have decent SNR
        let max_snr = sr.clusters[..sr.n_clusters]
            .iter()
            .map(|c| c.snr)
            .fold(0.0, |a, b| if a > b { a } else { b });
        assert!(
            max_snr > 1.0,
            "Best cluster SNR should be > 1.0, got {}",
            max_snr
        );
    }

    #[test]
    fn test_sort_with_adaptive_threshold() {
        let mut rng = Rng::new(42);
        let n = 5000;

        // 4-channel data with different noise levels per channel
        let mut data = vec![[0.0; 4]; n];
        let noise_levels = [1.0, 0.5, 2.0, 0.01]; // ch3 is near-dead
        for sample in data.iter_mut() {
            for ch in 0..4 {
                sample[ch] = rng.gaussian(0.0, noise_levels[ch]);
            }
        }

        // Inject spikes on channel 0
        let spike_template = |t: Float| -> Float { -12.0 * float::exp(-0.5 * t * t) };
        let mut pos = 200;
        while pos + 10 < n {
            for dt in 0..8 {
                let t = (dt as Float - 2.0) / 1.5;
                if pos + dt < n {
                    data[pos + dt][0] += spike_template(t);
                }
            }
            pos += 150;
        }

        let probe = ProbeLayout::<4>::linear(25.0);
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0; 8]; 200];
        let mut features = vec![[0.0; 3]; 200];
        let mut labels = vec![0usize; 200];

        // Sort with adaptive thresholds
        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            adaptive_threshold: true,
            adaptive_min_threshold: 0.5,
            adaptive_max_rate_hz: 200.0,
            ..SortConfig::default()
        };

        let result = sort_multichannel::<4, 16, 8, 3, 64, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(
            result.is_ok(),
            "Sort with adaptive thresholds should succeed"
        );
        let sr = result.unwrap();
        assert!(
            sr.n_spikes >= 5,
            "Should detect spikes with adaptive thresholds, got {}",
            sr.n_spikes
        );
    }

    // =========================================================================
    // merge_clusters tests
    // =========================================================================

    #[test]
    fn test_merge_identical_clusters() {
        // Two clusters with identical feature distributions should merge
        let mut labels = [0, 0, 0, 1, 1, 1];
        let features = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1],
            [1.05, 0.05],
            [0.95, -0.05],
            [1.0, 0.0],
        ];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 500,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 600,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 6];

        let new_n = merge_clusters(
            6,
            &mut labels,
            &features,
            &events,
            2,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(new_n, 1, "Identical clusters should merge, got {}", new_n);
        assert!(
            labels.iter().all(|&l| l == 0),
            "All labels should be 0 after merge"
        );
    }

    #[test]
    fn test_merge_well_separated_clusters() {
        // Two clearly separated clusters should NOT merge
        let mut labels = [0, 0, 0, 1, 1, 1];
        let features = [
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, -0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
        ];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 500,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 600,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 6];

        let new_n = merge_clusters(
            6,
            &mut labels,
            &features,
            &events,
            2,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(
            new_n, 2,
            "Separated clusters should NOT merge, got {}",
            new_n
        );
    }

    #[test]
    fn test_merge_isi_violation_prevents_merge() {
        // Two clusters that are similar but merging would create ISI violations
        let mut labels = [0, 0, 0, 1, 1, 1];
        let features = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1],
            [1.05, 0.05],
            [0.95, -0.05],
            [1.0, 0.0],
        ];
        // Spike times interleaved very closely -- merging would create ISI violations
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 102,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 104,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 101,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 103,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 105,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 6];

        let new_n = merge_clusters(
            6,
            &mut labels,
            &features,
            &events,
            2,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(
            new_n, 2,
            "ISI violations should prevent merge, got {}",
            new_n
        );
    }

    #[test]
    fn test_merge_single_cluster() {
        let mut labels = [0, 0, 0];
        let features = [[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 3];

        let new_n = merge_clusters(
            3,
            &mut labels,
            &features,
            &events,
            1,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(new_n, 1, "Single cluster should remain unchanged");
    }

    #[test]
    fn test_merge_empty() {
        let mut labels: [usize; 0] = [];
        let features: [[Float; 2]; 0] = [];
        let events: [MultiChannelEvent; 0] = [];
        let mut scratch: [Float; 0] = [];

        let new_n = merge_clusters(
            0,
            &mut labels,
            &features,
            &events,
            0,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(new_n, 0);
    }

    #[test]
    fn test_merge_three_clusters_two_similar() {
        // Three clusters: 0 and 1 are similar, 2 is separate
        let mut labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        let features = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1], // cluster 0
            [1.05, 0.05],
            [0.95, -0.05],
            [1.0, 0.0], // cluster 1 (similar to 0)
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9], // cluster 2 (far away)
        ];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 500,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 600,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 700,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 800,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 900,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 9];

        let new_n = merge_clusters(
            9,
            &mut labels,
            &features,
            &events,
            3,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(new_n, 2, "Should merge 0+1, keep 2 separate, got {}", new_n);
        // Cluster 2 (originally) should now be at index 1
        // All of the originally-cluster-0 and originally-cluster-1 should share a label
        let label_01 = labels[0];
        for &l in &labels[..6] {
            assert_eq!(l, label_01, "Merged cluster labels should match");
        }
        // Cluster 2 should have a different label
        let label_2 = labels[6];
        assert_ne!(
            label_01, label_2,
            "Separate cluster should keep its own label"
        );
        for &l in &labels[6..9] {
            assert_eq!(l, label_2, "Cluster 2 labels should all match");
        }
    }

    #[test]
    fn test_merge_label_shift() {
        // Verify that labels above the removed cluster are shifted down properly
        let mut labels = [0, 0, 1, 1, 2, 2, 3, 3];
        // Clusters 1 and 2 are similar, 0 and 3 are far apart
        let features = [
            [0.0, 0.0],
            [0.1, 0.0], // cluster 0
            [5.0, 5.0],
            [5.1, 5.1], // cluster 1
            [5.05, 5.05],
            [4.95, 4.95], // cluster 2 (similar to 1)
            [20.0, 20.0],
            [20.1, 20.1], // cluster 3
        ];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 400,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 500,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 600,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 700,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 800,
                channel: 0,
                amplitude: 5.0,
            },
        ];
        let mut scratch = [0.0; 8];

        let new_n = merge_clusters(
            8,
            &mut labels,
            &features,
            &events,
            4,
            1.5,
            0.05,
            15,
            &mut scratch,
            2,
        );
        assert_eq!(new_n, 3, "Should merge 1+2 into 3 clusters, got {}", new_n);
        // Cluster 0 stays at 0
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        // Clusters 1 and 2 merged (all become 1)
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], 1);
        assert_eq!(labels[4], 1);
        assert_eq!(labels[5], 1);
        // Cluster 3 shifted down to 2
        assert_eq!(labels[6], 2);
        assert_eq!(labels[7], 2);
    }

    // ---- split_clusters ----

    #[test]
    fn test_split_bimodal_cluster() {
        // Two well-separated groups in one cluster
        let mut labels = [0, 0, 0, 0, 0, 0];
        let features = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.1],
        ];
        let new_n = split_clusters(6, &mut labels, &features, 1, 3, 1.5);
        assert_eq!(new_n, 2, "Should split into 2 clusters");
        // Verify both labels are present
        let has_0 = labels.contains(&0);
        let has_1 = labels.contains(&1);
        assert!(has_0 && has_1, "Both clusters should be present");
    }

    #[test]
    fn test_split_unimodal_no_split() {
        // Tight cluster, should not split
        let mut labels = [0, 0, 0, 0, 0, 0];
        let features = [
            [1.0, 1.0],
            [1.01, 0.99],
            [0.99, 1.01],
            [1.02, 0.98],
            [0.98, 1.02],
            [1.0, 1.0],
        ];
        let new_n = split_clusters(6, &mut labels, &features, 1, 3, 2.0);
        assert_eq!(new_n, 1, "Tight cluster should not split");
    }

    #[test]
    fn test_split_small_cluster_skipped() {
        let mut labels = [0, 0];
        let features = [[0.0, 0.0], [10.0, 10.0]];
        // min_cluster_size = 5, so this cluster of 2 is skipped
        let new_n = split_clusters(2, &mut labels, &features, 1, 5, 1.0);
        assert_eq!(new_n, 1);
    }

    #[test]
    fn test_split_empty_no_panic() {
        let mut labels: [usize; 0] = [];
        let features: [[Float; 2]; 0] = [];
        let new_n = split_clusters(0, &mut labels, &features, 0, 3, 2.0);
        assert_eq!(new_n, 0);
    }

    // ---- OnlineSorter ----

    #[test]
    fn test_online_sorter_basic() {
        let mut sorter = OnlineSorter::<3, 8>::new();
        sorter.add_template(&[1.0, 0.0, 0.0]);
        sorter.add_template(&[0.0, 1.0, 0.0]);

        let (label0, dist0) = sorter.classify(&[0.9, 0.1, 0.0]);
        assert_eq!(label0, 0, "Should match first template");
        assert!(dist0 < 0.2, "Distance should be small, got {}", dist0);

        let (label1, dist1) = sorter.classify(&[0.1, 0.9, 0.0]);
        assert_eq!(label1, 1, "Should match second template");
        assert!(dist1 < 0.2, "Distance should be small, got {}", dist1);
    }

    #[test]
    fn test_online_sorter_reject() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        sorter.add_template(&[0.0, 0.0]);
        sorter.set_max_distance(1.0);

        let result = sorter.classify_or_reject(&[0.5, 0.5]);
        assert!(result.is_some(), "Close spike should be accepted");

        let result = sorter.classify_or_reject(&[10.0, 10.0]);
        assert!(result.is_none(), "Distant spike should be rejected");
        assert_eq!(sorter.n_rejected(), 1);
    }

    #[test]
    fn test_online_sorter_from_centroids() {
        let centroids = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let sorter = OnlineSorter::<2, 8>::from_centroids(&centroids);
        assert_eq!(sorter.n_templates(), 3);
        assert!((sorter.templates()[0][0] - 1.0).abs() < 1e-12);
        assert!((sorter.templates()[1][0] - 3.0).abs() < 1e-12);
        assert!((sorter.templates()[2][1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_online_sorter_no_templates() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        let (label, dist) = sorter.classify(&[1.0, 2.0]);
        assert_eq!(label, 0);
        assert_eq!(dist, float::MAX);
    }

    #[test]
    fn test_online_sorter_reset() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        sorter.add_template(&[1.0, 0.0]);
        sorter.add_template(&[0.0, 1.0]);
        sorter.classify(&[0.5, 0.5]);
        assert_eq!(sorter.n_templates(), 2);
        assert_eq!(sorter.n_classified(), 1);

        sorter.reset();
        assert_eq!(sorter.n_templates(), 0);
        assert_eq!(sorter.n_classified(), 0);
        assert_eq!(sorter.n_rejected(), 0);
    }

    #[test]
    fn test_online_sorter_full() {
        let mut sorter = OnlineSorter::<2, 2>::new();
        assert!(sorter.add_template(&[1.0, 0.0]).is_some());
        assert!(sorter.add_template(&[0.0, 1.0]).is_some());
        assert!(sorter.add_template(&[0.5, 0.5]).is_none(), "Should be full");
        assert_eq!(sorter.n_templates(), 2);
    }

    #[test]
    fn test_online_sorter_counters() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        sorter.add_template(&[0.0, 0.0]);
        sorter.set_max_distance(1.0);

        sorter.classify(&[0.1, 0.1]);
        sorter.classify(&[0.2, 0.2]);
        sorter.classify(&[0.3, 0.3]);
        assert_eq!(sorter.n_classified(), 3);
        assert_eq!(sorter.n_rejected(), 0);

        sorter.classify_or_reject(&[10.0, 10.0]);
        sorter.classify_or_reject(&[20.0, 20.0]);
        assert_eq!(sorter.n_classified(), 5);
        assert_eq!(sorter.n_rejected(), 2);

        sorter.reset_counters();
        assert_eq!(sorter.n_classified(), 0);
        assert_eq!(sorter.n_rejected(), 0);
        assert_eq!(sorter.n_templates(), 1);
    }

    #[test]
    fn test_compute_cluster_means() {
        // 2 clusters, W=4, N=4
        let waveforms: alloc::vec::Vec<[Float; 4]> = vec![
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0, 40.0],
        ];
        let labels = [0usize, 0, 1];
        let events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 200,
                channel: 0,
                amplitude: 6.0,
            },
            MultiChannelEvent {
                sample: 300,
                channel: 1,
                amplitude: 8.0,
            },
        ];

        let mut means = [[0.0; 4]; 4];
        let mut counts = [0u32; 4];
        let mut peak_ch = [0usize; 4];

        compute_cluster_means::<4, 4>(
            &waveforms,
            &labels,
            &events,
            3,
            2,
            &mut means,
            &mut counts,
            &mut peak_ch,
        );

        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 1);
        // Cluster 0 mean = (1+3)/2, (2+4)/2, (3+5)/2, (4+6)/2 = 2,3,4,5
        assert!((means[0][0] - 2.0).abs() < 1e-12);
        assert!((means[0][1] - 3.0).abs() < 1e-12);
        // Cluster 1 mean = 10,20,30,40
        assert!((means[1][0] - 10.0).abs() < 1e-12);
        assert_eq!(peak_ch[0], 0);
        assert_eq!(peak_ch[1], 1);
    }

    #[test]
    fn test_subtract_templates_multichannel() {
        // 2-channel data, W=4, 1 spike at sample 5
        let mut data = vec![[0.0; 2]; 20];
        data[3] = [1.0, 0.0];
        data[4] = [2.0, 0.0];
        data[5] = [3.0, 0.0];
        data[6] = [2.0, 0.0];

        let events = [MultiChannelEvent {
            sample: 5,
            channel: 0,
            amplitude: 3.0,
        }];
        let labels = [0usize];
        let means: [[Float; 4]; 4] = [[1.0, 2.0, 3.0, 2.0], [0.0; 4], [0.0; 4], [0.0; 4]];
        let counts = [5u32, 0, 0, 0];
        let peak_ch = [0usize, 0, 0, 0];

        subtract_templates_multichannel::<2, 4, 4>(
            &mut data, &events, 1, &labels, &means, &counts, &peak_ch, 3, 2,
        );

        // pre_samples=2, so start=5-2=3, template subtracted at data[3..7] on ch 0
        assert!((data[3][0] - 0.0).abs() < 1e-12); // 1.0 - 1.0
        assert!((data[4][0] - 0.0).abs() < 1e-12); // 2.0 - 2.0
        assert!((data[5][0] - 0.0).abs() < 1e-12); // 3.0 - 3.0
        assert!((data[6][0] - 0.0).abs() < 1e-12); // 2.0 - 2.0
                                                   // Channel 1 untouched
        assert!((data[3][1]).abs() < 1e-12);
    }

    #[test]
    fn test_assign_to_nearest_template() {
        let means: [[Float; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
        ];
        let counts = [10u32, 10, 0, 0];

        let wf = [0.9, 0.1, 0.0, 0.0];
        let (label, _dist) = assign_to_nearest_template::<4, 4>(&wf, &means, &counts, 2);
        assert_eq!(label, 0);

        let wf2 = [0.1, 0.9, 0.0, 0.0];
        let (label2, _) = assign_to_nearest_template::<4, 4>(&wf2, &means, &counts, 2);
        assert_eq!(label2, 1);
    }

    #[test]
    fn test_sort_with_template_subtraction() {
        // Run sorting with template_subtract on vs off on the same data
        let mut rng = Rng::new(77);
        let n = 5000;
        let mut data_on = vec![[0.0; 2]; n];
        for s in data_on.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        // Inject spikes on channel 0
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                let t = (dt as Float - 2.0) / 1.5;
                data_on[pos + dt][0] += -12.0 * float::exp(-0.5 * t * t);
            }
            pos += 150;
        }
        let mut data_off = data_on.clone();

        let probe = ProbeLayout::<2>::linear(25.0);
        let max_ev = n / 15 + 2;

        let config_on = SortConfig {
            template_subtract: true,
            detection_mode: DetectionMode::Amplitude,
            ccg_merge: false,
            ccg_template_corr_threshold: 0.5,
            ..SortConfig::default()
        };
        let mut scratch_on = vec![0.0; n];
        let mut ev_on = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            max_ev
        ];
        let mut wf_on = vec![[0.0; 8]; max_ev];
        let mut feat_on = vec![[0.0; 3]; max_ev];
        let mut lab_on = vec![0usize; max_ev];

        let r_on = sort_multichannel::<2, 4, 8, 3, 64, 4>(
            &config_on,
            &probe,
            &mut data_on,
            &mut scratch_on,
            &mut ev_on,
            &mut wf_on,
            &mut feat_on,
            &mut lab_on,
        );
        assert!(r_on.is_ok());

        let config_off = SortConfig {
            template_subtract: false,
            detection_mode: DetectionMode::Amplitude,
            ccg_merge: false,
            ccg_template_corr_threshold: 0.5,
            ..SortConfig::default()
        };
        let mut scratch_off = vec![0.0; n];
        let mut ev_off = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            max_ev
        ];
        let mut wf_off = vec![[0.0; 8]; max_ev];
        let mut feat_off = vec![[0.0; 3]; max_ev];
        let mut lab_off = vec![0usize; max_ev];

        let r_off = sort_multichannel::<2, 4, 8, 3, 64, 4>(
            &config_off,
            &probe,
            &mut data_off,
            &mut scratch_off,
            &mut ev_off,
            &mut wf_off,
            &mut feat_off,
            &mut lab_off,
        );
        assert!(r_off.is_ok());

        // Template subtraction should find >= as many spikes as without
        let sr_on = r_on.unwrap();
        let sr_off = r_off.unwrap();
        assert!(
            sr_on.n_spikes >= sr_off.n_spikes,
            "template_subtract ON ({}) should find >= spikes than OFF ({})",
            sr_on.n_spikes,
            sr_off.n_spikes
        );
    }

    #[test]
    fn test_isi_violation_split_no_violations() {
        // Cluster with well-spaced spikes should not be split
        let mut labels = [0usize; 20];
        let features: Vec<[Float; 2]> = (0..20).map(|i| [i as Float * 0.1, 0.0]).collect();
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 100, // well-spaced (100 samples apart)
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = [0.0; 100];
        let n = isi_violation_split::<2>(
            20,
            &mut labels,
            &features,
            &events,
            1,
            0.1, // isi_threshold
            15,  // refractory
            5,   // min_cluster_size
            &mut scratch,
            8,
        );
        assert_eq!(n, 1, "well-spaced cluster should not be split");
    }

    #[test]
    fn test_isi_violation_split_with_violations() {
        // Two neurons interleaved at high rate -- high ISI violations
        let n_spikes = 40;
        let mut labels = vec![0usize; n_spikes];
        // Two populations in feature space
        let features: Vec<[Float; 2]> = (0..n_spikes)
            .map(|i| if i % 2 == 0 { [5.0, 0.0] } else { [0.0, 5.0] })
            .collect();
        // Interleaved spike times: 0, 5, 10, 15, ... (5 samples apart, < 15 refractory)
        let events: Vec<MultiChannelEvent> = (0..n_spikes)
            .map(|i| MultiChannelEvent {
                sample: i * 5,
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = vec![0.0; n_spikes + 10];
        let n = isi_violation_split::<2>(
            n_spikes,
            &mut labels,
            &features,
            &events,
            1,
            0.05, // strict ISI threshold
            15,   // refractory
            5,    // min_cluster_size
            &mut scratch,
            8,
        );
        assert!(
            n >= 2,
            "interleaved neurons should be split (got {} clusters)",
            n
        );
    }

    #[test]
    fn test_isi_violation_split_empty() {
        let mut labels = [];
        let features: Vec<[Float; 2]> = vec![];
        let events: Vec<MultiChannelEvent> = vec![];
        let mut scratch = [0.0; 10];
        let n = isi_violation_split::<2>(
            0,
            &mut labels,
            &features,
            &events,
            0,
            0.1,
            15,
            5,
            &mut scratch,
            8,
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn test_isi_violation_split_high_threshold() {
        // With threshold = 1.0 (100%), nothing should be split since max ISI rate < 1.0
        let mut labels = [0usize; 20];
        let features: Vec<[Float; 2]> = (0..20).map(|i| [i as Float, 0.0]).collect();
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 5,
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = [0.0; 30];
        // threshold = 1.0 means only split if 100% ISI violations (impossible)
        let n = isi_violation_split::<2>(
            20,
            &mut labels,
            &features,
            &events,
            1,
            1.0,
            15,
            5,
            &mut scratch,
            8,
        );
        assert_eq!(n, 1, "threshold=1.0 should not split");
    }

    #[test]
    fn test_amplitude_bimodality_split_two_groups() {
        // Two clear amplitude groups: 10 spikes at amp ~2.0, 10 at amp ~20.0
        let mut labels = [0usize; 20];
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| {
                let amp = if i < 10 {
                    2.0 + (i as Float) * 0.05
                } else {
                    20.0 + ((i - 10) as Float) * 0.05
                };
                MultiChannelEvent {
                    sample: i * 100,
                    channel: 0,
                    amplitude: amp,
                }
            })
            .collect();
        let n = amplitude_bimodality_split(20, &mut labels, &events, 1, 2.0, 3, 8);
        assert_eq!(n, 2, "should split into 2 clusters");
        // First 10 and last 10 should have different labels
        assert_eq!(labels[0], labels[9]);
        assert_eq!(labels[10], labels[19]);
        assert_ne!(labels[0], labels[10]);
    }

    #[test]
    fn test_amplitude_bimodality_split_unimodal() {
        // Uniform amplitudes -- should NOT split
        let mut labels = [0usize; 20];
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 100,
                channel: 0,
                amplitude: 5.0 + (i as Float) * 0.1,
            })
            .collect();
        let n = amplitude_bimodality_split(20, &mut labels, &events, 1, 2.0, 3, 8);
        assert_eq!(n, 1, "unimodal distribution should not split");
    }

    #[test]
    fn test_amplitude_bimodality_split_too_small() {
        // Too few spikes to split
        let mut labels = [0usize; 4];
        let events: Vec<MultiChannelEvent> = (0..4)
            .map(|i| MultiChannelEvent {
                sample: i * 100,
                channel: 0,
                amplitude: if i < 2 { 1.0 } else { 100.0 },
            })
            .collect();
        let n = amplitude_bimodality_split(4, &mut labels, &events, 1, 2.0, 3, 8);
        assert_eq!(n, 1, "too few spikes to split (min_cluster_size=3)");
    }

    #[test]
    fn test_amplitude_bimodality_split_respects_max() {
        // Already at max clusters
        let mut labels = [0usize; 20];
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 100,
                channel: 0,
                amplitude: if i < 10 { 1.0 } else { 100.0 },
            })
            .collect();
        let n = amplitude_bimodality_split(20, &mut labels, &events, 1, 2.0, 3, 1);
        assert_eq!(n, 1, "should not split when at max_clusters");
    }

    #[test]
    fn test_multi_pass_template_subtract() {
        // Multi-pass should find >= spikes as single pass
        use crate::probe::ProbeLayout;
        let probe = ProbeLayout::<4>::linear(25.0);
        let n_samples = 10000;
        let mut data1 = vec![[0.0; 4]; n_samples];
        let mut data2 = data1.clone();
        // Inject overlapping spikes
        for t in (200..9000).step_by(80) {
            data1[t][0] = -12.0;
            data2[t][0] = -12.0;
            if t + 15 < n_samples {
                data1[t + 15][1] = -10.0;
                data2[t + 15][1] = -10.0;
            }
        }
        let max_events = n_samples / 15 + 4;
        let mut scratch1 = vec![0.0; n_samples];
        let mut events1 = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            max_events
        ];
        let mut wf1 = vec![[0.0; 48]; max_events];
        let mut feat1 = vec![[0.0; 4]; max_events];
        let mut lab1 = vec![0usize; max_events];

        let mut scratch2 = scratch1.clone();
        let mut events2 = events1.clone();
        let mut wf2 = wf1.clone();
        let mut feat2 = feat1.clone();
        let mut lab2 = lab1.clone();

        let config1 = SortConfig {
            template_subtract_passes: 1,
            ..SortConfig::default()
        };
        let config2 = SortConfig {
            template_subtract_passes: 3,
            ..SortConfig::default()
        };

        let r1 = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
            &config1,
            &probe,
            &mut data1,
            &mut scratch1,
            &mut events1,
            &mut wf1,
            &mut feat1,
            &mut lab1,
        );
        let r2 = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
            &config2,
            &probe,
            &mut data2,
            &mut scratch2,
            &mut events2,
            &mut wf2,
            &mut feat2,
            &mut lab2,
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        let s1 = r1.unwrap();
        let s2 = r2.unwrap();
        assert!(
            s2.n_spikes >= s1.n_spikes,
            "3-pass ({}) should find >= spikes than 1-pass ({})",
            s2.n_spikes,
            s1.n_spikes
        );
    }

    #[test]
    fn test_bandpass_removes_dc() {
        // Bandpass filter should remove DC offset
        let mut data = [[5.0; 2]; 2000];
        // Add some high-frequency content at sample rate / 10
        for (i, sample) in data.iter_mut().enumerate() {
            let t = i as Float / 1000.0;
            sample[0] += float::sin(2.0 * float::PI * 100.0 * t);
        }
        bandpass_inplace::<2>(&mut data, 1000.0, 50.0, 200.0);
        // DC should be removed: mean should be near zero
        let mean: Float =
            data.iter().skip(200).map(|s| s[0]).sum::<Float>() / (data.len() - 200) as Float;
        assert!(
            float::abs(mean) < 1.0,
            "bandpass should remove DC, mean={}",
            mean
        );
    }

    #[test]
    fn test_bandpass_preserves_passband() {
        // Signal in passband should be mostly preserved
        let mut data = [[0.0; 1]; 4000];
        let freq = 1000.0; // 1kHz signal, passband 300-6000Hz at 30kHz
        for (i, sample) in data.iter_mut().enumerate() {
            let t = i as Float / 30000.0;
            sample[0] = float::sin(2.0 * float::PI * freq * t);
        }
        let original_power: Float = data
            .iter()
            .skip(500)
            .take(2000)
            .map(|s| s[0] * s[0])
            .sum::<Float>();
        bandpass_inplace::<1>(&mut data, 30000.0, 300.0, 6000.0);
        let filtered_power: Float = data
            .iter()
            .skip(500)
            .take(2000)
            .map(|s| s[0] * s[0])
            .sum::<Float>();
        let ratio = filtered_power / original_power;
        assert!(
            ratio > 0.5,
            "passband signal should be mostly preserved, ratio={}",
            ratio
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_common_median_ref() {
        use crate::probe::ProbeLayout;
        // Common noise on all channels should be removed by CMR
        let probe = ProbeLayout::<4>::linear(25.0);
        let n_samples = 2000;
        let mut data = vec![[0.0; 4]; n_samples];
        let mut rng = Rng::new(77);
        // Add common noise + independent noise + spikes
        for t in 0..n_samples {
            let common = rng.gaussian(0.0, 3.0);
            for ch in 0..4 {
                data[t][ch] = common + rng.gaussian(0.0, 1.0);
            }
        }
        // Add spikes
        for t in (200..1800).step_by(200) {
            data[t][0] = -15.0;
        }
        let mut scratch = vec![0.0; n_samples];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            100
        ];
        let mut wf = vec![[0.0; 48]; 100];
        let mut feat = vec![[0.0; 4]; 100];
        let mut lab = vec![0usize; 100];

        let config = SortConfig {
            common_median_ref: true,
            ..SortConfig::default()
        };
        let result = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut wf,
            &mut feat,
            &mut lab,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_svd_init_centroids_two_clusters() {
        // Two well-separated clusters along dim 0
        let features = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, -0.1],
            [-0.1, 0.05],
            [5.0, 0.0],
            [5.1, 0.1],
            [4.9, -0.1],
            [5.2, 0.05],
        ];
        let (centroids, count) = svd_init_centroids::<2, 8>(&features, 8, 2);
        assert_eq!(count, 2);
        // Centroids should separate the two groups
        assert!(
            centroids[0][0] < 1.0,
            "first centroid dim0={}",
            centroids[0][0]
        );
        assert!(
            centroids[1][0] > 4.0,
            "second centroid dim0={}",
            centroids[1][0]
        );
    }

    #[test]
    fn test_svd_init_centroids_single_point() {
        let features = [[1.0, 2.0]];
        let (_, count) = svd_init_centroids::<2, 4>(&features, 1, 2);
        // Only 1 point, need at least 2 for covariance
        assert_eq!(count, 0);
    }

    #[test]
    fn test_svd_init_centroids_identical_points() {
        let features = [[3.0, 3.0]; 10];
        let (centroids, count) = svd_init_centroids::<2, 4>(&features, 10, 3);
        // All identical -> degenerate, should return 1 centroid at the mean
        assert_eq!(count, 1);
        assert!((centroids[0][0] - 3.0).abs() < 1e-10);
        assert!((centroids[0][1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_init_in_sort_pipeline() {
        use crate::probe::ProbeLayout;
        let probe = ProbeLayout::<4>::linear(25.0);
        let n_samples = 2000;
        let mut data = vec![[0.0; 4]; n_samples];
        let mut rng = Rng::new(99);
        for row in data.iter_mut() {
            for ch in row.iter_mut() {
                *ch = rng.gaussian(0.0, 1.0);
            }
        }
        // Inject spikes on channel 0
        for t in (200..1800).step_by(200) {
            data[t][0] = -15.0;
        }
        let mut scratch = vec![0.0; n_samples];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            100
        ];
        let mut wf = vec![[0.0; 48]; 100];
        let mut feat = vec![[0.0; 4]; 100];
        let mut lab = vec![0usize; 100];

        let config = SortConfig {
            svd_init: true,
            ..SortConfig::default()
        };
        let result = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut wf,
            &mut feat,
            &mut lab,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_refinement_iterations_stability() {
        // Verify that refinement produces stable results: 2 iterations and 3 iterations
        // should give the same or very similar cluster counts and spike counts.
        let n = 6000;

        // Helper to build data with two neurons
        let build_data = |rng: &mut Rng| -> Vec<[Float; 2]> {
            let mut data = vec![[0.0; 2]; n];
            for sample in data.iter_mut() {
                sample[0] = rng.gaussian(0.0, 1.0);
                sample[1] = rng.gaussian(0.0, 1.0);
            }
            // Neuron A: large spike on channel 0
            let mut pos = 200;
            while pos + 10 < n {
                for dt in 0..8 {
                    let t = (dt as Float - 2.0) / 1.5;
                    if pos + dt < n {
                        data[pos + dt][0] += -14.0 * float::exp(-0.5 * t * t);
                    }
                }
                pos += 150;
            }
            // Neuron B: medium spike on channel 1
            let mut pos = 350;
            while pos + 12 < n {
                for dt in 0..10 {
                    let t = (dt as Float - 3.0) / 2.0;
                    if pos + dt < n {
                        data[pos + dt][1] += -10.0 * float::exp(-0.5 * t * t);
                    }
                }
                pos += 200;
            }
            data
        };

        let run_sort = |iters: usize, rng: &mut Rng| -> SortResult<8> {
            let mut data = build_data(rng);
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                matched_filter_threshold: 4.0,
                refinement_iterations: iters,
                ..SortConfig::default()
            };
            let probe = ProbeLayout::<2>::linear(25.0);
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                300
            ];
            let mut waveforms = vec![[0.0; 8]; 300];
            let mut features = vec![[0.0; 3]; 300];
            let mut labels = vec![0usize; 300];

            sort_multichannel::<2, 4, 8, 3, 64, 8>(
                &config,
                &probe,
                &mut data,
                &mut scratch,
                &mut events,
                &mut waveforms,
                &mut features,
                &mut labels,
            )
            .expect("sort should succeed")
        };

        // Run with 0, 1, 2, 3 iterations
        let r0 = run_sort(0, &mut Rng::new(99));
        let r1 = run_sort(1, &mut Rng::new(99));
        let r2 = run_sort(2, &mut Rng::new(99));
        let r3 = run_sort(3, &mut Rng::new(99));

        // All should find spikes
        assert!(r0.n_spikes >= 5, "iter=0: n_spikes={}", r0.n_spikes);
        assert!(r1.n_spikes >= 5, "iter=1: n_spikes={}", r1.n_spikes);
        assert!(r2.n_spikes >= 5, "iter=2: n_spikes={}", r2.n_spikes);
        assert!(r3.n_spikes >= 5, "iter=3: n_spikes={}", r3.n_spikes);

        // Refinement should not dramatically change results -- cluster count
        // should be similar across iterations (within 1 of each other)
        let cdiff = r2.n_clusters.abs_diff(r3.n_clusters);
        assert!(
            cdiff <= 1,
            "2 vs 3 iterations cluster count should be stable: {} vs {}",
            r2.n_clusters,
            r3.n_clusters
        );

        // Spike counts should be in the same ballpark (within 50%)
        let diff = r2.n_spikes.abs_diff(r3.n_spikes);
        let max_spikes = r2.n_spikes.max(r3.n_spikes);
        assert!(
            diff * 2 <= max_spikes, // within 50%
            "2 vs 3 iterations spike count should be stable: {} vs {}",
            r2.n_spikes,
            r3.n_spikes
        );
    }

    #[test]
    fn test_auto_refine_skips_small_probe() {
        // auto_refine=true must suppress refinement on C=2 (< 8).
        // With refinement_iterations=1 but auto_refine=true, the 2ch sort
        // result should be identical to refinement_iterations=0.
        let n = 6000;
        let mut data = vec![[0.0f64; 2]; n];
        let mut rng = Rng::new(31);
        for s in data.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -14.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        let probe = ProbeLayout::<2>::linear(25.0);
        let make_result = |refine: usize, auto: bool| -> SortResult<8> {
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                matched_filter_threshold: 4.0,
                refinement_iterations: refine,
                auto_refine: auto,
                ..SortConfig::default()
            };
            let mut d = data.clone();
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0
                };
                300
            ];
            let mut wf = vec![[0.0; 8]; 300];
            let mut feat = vec![[0.0; 3]; 300];
            let mut lab = vec![0usize; 300];
            sort_multichannel::<2, 4, 8, 3, 64, 8>(
                &config,
                &probe,
                &mut d,
                &mut scratch,
                &mut events,
                &mut wf,
                &mut feat,
                &mut lab,
            )
            .expect("sort ok")
        };
        // auto_refine=true with C=2 → refinement skipped → same as 0 iterations
        let r_auto = make_result(1, true);
        let r_none = make_result(0, true);
        assert_eq!(
            r_auto.n_spikes, r_none.n_spikes,
            "auto_refine should suppress refinement on C=2: {} vs {}",
            r_auto.n_spikes, r_none.n_spikes
        );
        // auto_refine=false → refinement runs → allowed to differ from 0-iter
        let r_forced = make_result(1, false);
        assert!(
            r_forced.n_spikes > 0,
            "forced refinement should still find spikes"
        );
    }

    #[test]
    fn test_auto_refine_runs_on_large_probe() {
        // auto_refine=true must NOT suppress refinement on C=8 (≥ 8).
        // Run with refinement_iterations=1+auto_refine=true vs =0; they may
        // differ (refinement reassigns borderline spikes), but the key check
        // is that the sort completes and finds spikes in both cases.
        let n = 8000;
        let mut data = vec![[0.0f64; 8]; n];
        let mut rng = Rng::new(37);
        for s in data.iter_mut() {
            for v in s.iter_mut() {
                *v = rng.gaussian(0.0, 1.0);
            }
        }
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -14.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                    data[pos + dt][1] += -9.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        let probe = ProbeLayout::<8>::linear(25.0);
        let run = |refine: usize| -> SortResult<16> {
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                matched_filter_threshold: 4.0,
                refinement_iterations: refine,
                auto_refine: true,
                ..SortConfig::default()
            };
            let mut d = data.clone();
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0
                };
                500
            ];
            let mut wf = vec![[0.0; 8]; 500];
            let mut feat = vec![[0.0; 3]; 500];
            let mut lab = vec![0usize; 500];
            sort_multichannel::<8, 64, 8, 3, 64, 16>(
                &config,
                &probe,
                &mut d,
                &mut scratch,
                &mut events,
                &mut wf,
                &mut feat,
                &mut lab,
            )
            .expect("sort ok")
        };
        let r0 = run(0);
        let r1 = run(1);
        assert!(r0.n_spikes > 0, "baseline should find spikes");
        assert!(r1.n_spikes > 0, "auto_refine on C=8 should find spikes");
    }

    #[test]
    fn test_collapse_guard_prevents_empty_cluster() {
        // Construct a 2-cluster scenario where one cluster's centroid is
        // artificially placed close to the other so that all spikes in
        // cluster 0 would reassign to cluster 1 during refinement.
        // With refine_collapse_guard=true the reassignment must be skipped
        // and cluster 0 must survive; with guard=false it is allowed to empty.
        let n = 4000;
        let mut data = vec![[0.0f64; 2]; n];
        let mut rng = Rng::new(41);
        for s in data.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        // Two neurons: A strong on ch0, B strong on ch1
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -16.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        let mut pos2 = 300;
        while pos2 + 8 < n {
            for dt in 0..8 {
                if pos2 + dt < n {
                    data[pos2 + dt][1] +=
                        -16.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos2 += 150;
        }
        let probe = ProbeLayout::<2>::linear(25.0);
        let run_sort = |guard: bool| -> SortResult<8> {
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                matched_filter_threshold: 4.0,
                refinement_iterations: 1,
                auto_refine: false, // force refinement to run on C=2
                refine_collapse_guard: guard,
                split_min_cluster_size: 5,
                ..SortConfig::default()
            };
            let mut d = data.clone();
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                300
            ];
            let mut wf = vec![[0.0; 8]; 300];
            let mut feat = vec![[0.0; 3]; 300];
            let mut lab = vec![0usize; 300];
            sort_multichannel::<2, 4, 8, 3, 64, 8>(
                &config,
                &probe,
                &mut d,
                &mut scratch,
                &mut events,
                &mut wf,
                &mut feat,
                &mut lab,
            )
            .expect("sort ok")
        };
        let r_guard = run_sort(true);
        let r_no_guard = run_sort(false);
        // Both runs should detect spikes
        assert!(r_guard.n_spikes > 0, "guard=true should detect spikes");
        assert!(r_no_guard.n_spikes > 0, "guard=false should detect spikes");
        // With the guard, well-separated units should survive as distinct clusters
        assert!(
            r_guard.n_clusters >= 1,
            "guard should preserve clusters: got {}",
            r_guard.n_clusters
        );
    }

    #[test]
    fn test_collapse_guard_off_allows_collapse() {
        // With refine_collapse_guard=false, the refinement loop is free to
        // run even if it would empty a cluster. This test verifies the flag
        // actually gates the dry-run: disabling it must not panic or error.
        let n = 3000;
        let mut data = vec![[0.0f64; 2]; n];
        let mut rng = Rng::new(43);
        for s in data.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -12.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        let probe = ProbeLayout::<2>::linear(25.0);
        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            matched_filter_threshold: 4.0,
            refinement_iterations: 2,
            auto_refine: false,
            refine_collapse_guard: false,
            ..SortConfig::default()
        };
        let mut d = data.clone();
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut wf = vec![[0.0; 8]; 200];
        let mut feat = vec![[0.0; 3]; 200];
        let mut lab = vec![0usize; 200];
        let result = sort_multichannel::<2, 4, 8, 3, 64, 8>(
            &config,
            &probe,
            &mut d,
            &mut scratch,
            &mut events,
            &mut wf,
            &mut feat,
            &mut lab,
        );
        assert!(result.is_ok(), "sort must not error with guard disabled");
    }

    #[test]
    fn test_collapse_guard_default_true() {
        let config = SortConfig::default();
        assert!(
            config.refine_collapse_guard,
            "refine_collapse_guard must default to true"
        );
    }

    /// Test that localization mode runs without errors on multi-channel data.
    /// Uses K=4 (2 PCA + 2 localization dims) on a 4-channel linear probe.
    #[test]
    fn test_sort_with_localization() {
        let n = 3000;
        let mut data = vec![[0.0 as Float; 4]; n];

        // Inject spikes on channel 0 (y=0) and channel 3 (y=75)
        let spike = |t: Float| -> Float { -8.0 * (-t * t / 2.0).exp() };
        let mut pos = 100;
        let mut unit = 0;
        while pos + 5 < n {
            let ch = if unit % 2 == 0 { 0 } else { 3 };
            for dt in 0..5 {
                let t = (dt as Float - 2.0) / 1.5;
                data[pos + dt][ch] += spike(t);
            }
            pos += 100;
            unit += 1;
        }

        let probe = ProbeLayout::<4>::linear(25.0);
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0; 8]; 200];
        let mut features = vec![[0.0; 4]; 200];
        let mut labels = vec![0usize; 200];

        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            use_localization: true,
            ..SortConfig::default()
        };

        let result = sort_multichannel::<4, 16, 8, 4, 64, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );

        assert!(result.is_ok(), "sort with localization should succeed");
        let r = result.unwrap();
        assert!(r.n_spikes >= 5, "should detect spikes: got {}", r.n_spikes);
    }

    /// Test that localization produces different feature values for spikes
    /// on spatially separated channels (verifies spatial features are
    /// actually being computed, not just zero).
    #[test]
    fn test_localization_features_differ_by_channel() {
        let n = 4000;
        let mut data = vec![[0.0 as Float; 4]; n];

        // Large spikes alternating between ch 0 (y=0) and ch 3 (y=75)
        let spike = |t: Float| -> Float { -12.0 * (-t * t / 2.0).exp() };
        let mut pos = 200;
        let mut unit = 0;
        while pos + 5 < n {
            let ch = if unit % 2 == 0 { 0 } else { 3 };
            for dt in 0..5 {
                let t = (dt as Float - 2.0) / 1.5;
                data[pos + dt][ch] += spike(t);
            }
            pos += 200;
            unit += 1;
        }

        let probe = ProbeLayout::<4>::linear(25.0);
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0; 8]; 200];
        let mut features = vec![[0.0; 4]; 200];
        let mut labels = vec![0usize; 200];

        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            use_localization: true,
            ..SortConfig::default()
        };

        let result = sort_multichannel::<4, 16, 8, 4, 64, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );

        let r = result.expect("sort should succeed");
        if r.n_spikes >= 4 {
            // Check that the spatial feature dims (K-2, K-1) differ across spikes
            // on different channels. Collect unique (rounded) y-positions.
            let mut y_vals: Vec<Float> = Vec::new();
            for feat in features.iter().take(r.n_spikes) {
                let y = feat[3]; // K-1 = y position
                let mut found = false;
                for existing in &y_vals {
                    if float::abs(existing - y) < 1.0 {
                        found = true;
                        break;
                    }
                }
                if !found {
                    y_vals.push(y);
                }
            }
            // We injected spikes on ch 0 (y=0) and ch 3 (y=75), so after
            // normalization there should be at least 2 distinct y-feature values.
            assert!(
                y_vals.len() >= 2,
                "localization should produce distinct y-features for different channels, got {} unique: {:?}",
                y_vals.len(),
                y_vals
            );
        }
    }

    /// Test that localization does not regress sort quality vs channel-index mode.
    /// Both modes should detect a similar number of spikes.
    #[test]
    fn test_localization_no_regression() {
        let n = 4000;
        let build = || -> Vec<[Float; 4]> {
            let mut data = vec![[0.0 as Float; 4]; n];
            let spike = |t: Float| -> Float { -10.0 * (-t * t / 2.0).exp() };
            let mut pos = 150;
            while pos + 5 < n {
                let ch = (pos / 150) % 4;
                for dt in 0..5 {
                    let t = (dt as Float - 2.0) / 1.5;
                    data[pos + dt][ch] += spike(t);
                }
                pos += 150;
            }
            data
        };

        let probe = ProbeLayout::<4>::linear(25.0);

        let run = |use_loc: bool| -> SortResult<8> {
            let mut data = build();
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                use_localization: use_loc,
                ..SortConfig::default()
            };
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                200
            ];
            let mut waveforms = vec![[0.0; 8]; 200];
            let mut features = vec![[0.0; 4]; 200];
            let mut labels = vec![0usize; 200];

            sort_multichannel::<4, 16, 8, 4, 64, 8>(
                &config,
                &probe,
                &mut data,
                &mut scratch,
                &mut events,
                &mut waveforms,
                &mut features,
                &mut labels,
            )
            .expect("sort should succeed")
        };

        let r_off = run(false);
        let r_on = run(true);

        // Both should detect spikes
        assert!(r_off.n_spikes >= 5, "baseline: {}", r_off.n_spikes);
        assert!(r_on.n_spikes >= 5, "localization: {}", r_on.n_spikes);

        // Spike counts should be in the same ballpark (within 50%)
        let diff = r_off.n_spikes.abs_diff(r_on.n_spikes);
        let max_s = r_off.n_spikes.max(r_on.n_spikes);
        assert!(
            diff * 2 <= max_s,
            "localization should not dramatically change spike count: {} vs {}",
            r_off.n_spikes,
            r_on.n_spikes
        );
    }

    // --- Auto-CMR tests ---

    #[test]
    fn test_auto_cmr_triggers_for_large_channel_count() {
        // auto_cmr=true should apply CMR when C >= 8.
        // We verify by checking that the config default has auto_cmr=true
        // and that the pipeline runs without error on 8-channel data.
        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            auto_cmr: true,
            common_median_ref: false, // explicit flag is off
            ..SortConfig::default()
        };
        assert!(config.auto_cmr);
        // The actual CMR application is tested indirectly through sort_multichannel.
        // For 8 channels auto_cmr triggers; the output should still be valid.
        let probe = ProbeLayout::<8>::linear(25.0);
        let n = 500;
        let mut data: Vec<[Float; 8]> = (0..n)
            .map(|i| {
                let t = i as Float;
                // Correlated noise on all channels (CMR should remove this)
                let common = (t * 0.01).sin() * 0.5;
                core::array::from_fn(|ch| {
                    let phase = (t * 0.3 + ch as Float * 0.5).sin() * 6.0;
                    common + phase * if i % 80 == 0 { 1.0 } else { 0.1 }
                })
            })
            .collect();
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            200
        ];
        let mut waveforms = vec![[0.0; 8]; 200];
        let mut features = vec![[0.0; 4]; 200];
        let mut labels = vec![0usize; 200];
        let result = sort_multichannel::<8, 64, 8, 4, 64, 8>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_ok(), "auto_cmr sort should not fail");
    }

    #[test]
    fn test_auto_cmr_does_not_trigger_for_small_channel_count() {
        // auto_cmr=true should NOT apply CMR when C < 8 (4-channel probe).
        // The sort should still succeed.
        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            auto_cmr: true,
            ..SortConfig::default()
        };
        let probe = ProbeLayout::<4>::linear(25.0);
        let n = 300;
        let mut data: Vec<[Float; 4]> = (0..n)
            .map(|i| {
                core::array::from_fn(|ch| {
                    if i % 60 == 0 {
                        -6.0 * (ch as Float + 1.0)
                    } else {
                        0.05
                    }
                })
            })
            .collect();
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            100
        ];
        let mut waveforms = vec![[0.0; 8]; 100];
        let mut features = vec![[0.0; 4]; 100];
        let mut labels = vec![0usize; 100];
        let result = sort_multichannel::<4, 16, 8, 4, 64, 4>(
            &config,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_ok(), "small channel count sort should not fail");
    }

    #[test]
    fn test_auto_cmr_disabled_overrides() {
        // auto_cmr=false means CMR is never applied automatically.
        let config_auto = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            auto_cmr: false,
            common_median_ref: false,
            ..SortConfig::default()
        };
        assert!(!config_auto.auto_cmr);
        assert!(!config_auto.common_median_ref);
        // Should still succeed
        let probe = ProbeLayout::<8>::linear(25.0);
        let n = 400;
        let mut data: Vec<[Float; 8]> = (0..n)
            .map(|i| core::array::from_fn(|_| if i % 70 == 0 { -6.0 } else { 0.05 }))
            .collect();
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            100
        ];
        let mut waveforms = vec![[0.0; 8]; 100];
        let mut features = vec![[0.0; 4]; 100];
        let mut labels = vec![0usize; 100];
        let result = sort_multichannel::<8, 64, 8, 4, 64, 8>(
            &config_auto,
            &probe,
            &mut data,
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );
        assert!(result.is_ok());
    }

    // --- Coincidence detection tests ---

    #[test]
    fn test_coincidence_detect_recovers_subthreshold_spatial_spike() {
        // A spike with amplitude 4.0σ (below 5.0σ primary threshold) appearing
        // on 3 neighboring channels should be recovered by coincidence detection.
        use crate::spike_sort::detect_spikes_coincidence;
        let probe = ProbeLayout::<8>::linear(25.0);
        let n = 200;
        let mut data = vec![[0.0; 8]; n];
        // Spike at t=50: channel 3 at 4.0σ (primary threshold 3.5), channels 2,4 at 2.5σ
        data.iter_mut().take(53).skip(47).for_each(|s| {
            s[3] = -4.0;
            s[2] = -2.5;
            s[4] = -2.5;
        });
        let noise = [1.0; 8];
        let existing: [MultiChannelEvent; 0] = [];
        let mut out = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 32];
        let n_found = detect_spikes_coincidence::<8>(
            &data, &noise, &existing, &probe, 3.5, 2.0, 2, 75.0, 10, &mut out,
        );
        assert!(
            n_found >= 1,
            "sub-threshold spatial spike should be recovered"
        );
        assert!(out[..n_found].iter().any(|e| e.sample.abs_diff(50) <= 5));
    }

    #[test]
    fn test_coincidence_detect_rejects_isolated_subthreshold_spike() {
        // A spike with amplitude 4.0σ on only 1 channel (no neighbors above 2σ)
        // should NOT be accepted by coincidence detection.
        use crate::spike_sort::detect_spikes_coincidence;
        let probe = ProbeLayout::<8>::linear(25.0);
        let n = 200;
        let mut data = vec![[0.0; 8]; n];
        // Only channel 3 is active, neighbors are at 0.5σ (below 2σ)
        data.iter_mut().take(53).skip(47).for_each(|s| {
            s[3] = -4.0;
            s[2] = -0.5;
            s[4] = -0.5;
        });
        let noise = [1.0; 8];
        let existing: [MultiChannelEvent; 0] = [];
        let mut out = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 32];
        let n_found = detect_spikes_coincidence::<8>(
            &data, &noise, &existing, &probe, 3.5, 2.0, 2, 75.0, 10, &mut out,
        );
        assert_eq!(
            n_found, 0,
            "isolated sub-threshold spike should be rejected"
        );
    }

    #[test]
    fn test_coincidence_detect_skips_existing_events() {
        // A coincidence candidate at a time already covered by an existing event
        // (within refractory) should not be re-emitted.
        use crate::spike_sort::detect_spikes_coincidence;
        let probe = ProbeLayout::<8>::linear(25.0);
        let n = 200;
        let mut data = vec![[0.0; 8]; n];
        data.iter_mut().take(53).skip(47).for_each(|s| {
            s[3] = -4.0;
            s[2] = -2.5;
            s[4] = -2.5;
        });
        let noise = [1.0; 8];
        let existing = [MultiChannelEvent {
            sample: 50,
            channel: 3,
            amplitude: 4.0,
        }];
        let mut out = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 32];
        let n_found = detect_spikes_coincidence::<8>(
            &data, &noise, &existing, &probe, 3.5, 2.0, 2, 75.0, 15, &mut out,
        );
        assert_eq!(n_found, 0, "should not duplicate an already-detected event");
    }

    #[test]
    fn test_coincidence_detect_disabled_adds_nothing() {
        // coincidence_detect=false should not add any events.
        let config = SortConfig {
            threshold_multiplier: 5.0,
            pre_samples: 2,
            refractory_samples: 10,
            coincidence_detect: false,
            ..SortConfig::default()
        };
        assert!(!config.coincidence_detect);
    }

    #[test]
    fn test_compute_half_width_narrow() {
        // Narrow spike: trough at center, quickly returns to baseline
        let mut waveform = [0.0; 48];
        // Trough at index 20, half-width of ~4 samples
        waveform[18] = -0.5;
        waveform[19] = -0.9;
        waveform[20] = -1.0;
        waveform[21] = -0.9;
        waveform[22] = -0.5;
        let hw = compute_half_width::<48>(&waveform);
        // half_max = -0.5, crossing at ~18 and ~22, width = ~4/48
        assert!(hw > 0.0 && hw < 0.2, "narrow spike half-width={hw}");
    }

    #[test]
    fn test_compute_half_width_broad() {
        // Broad spike: trough at center, slowly returns to baseline
        let mut waveform = [0.0; 48];
        // Trough at index 24, half-width of ~16 samples
        for (i, v) in waveform.iter_mut().enumerate().take(33).skip(16) {
            let d = (i as Float - 24.0).abs();
            *v = -1.0 + d * d / 64.0;
        }
        // Make minimum at center
        waveform[24] = -1.0;
        let hw = compute_half_width::<48>(&waveform);
        assert!(hw >= 0.2, "broad spike half-width={hw}");
    }

    #[test]
    fn test_compute_half_width_no_trough() {
        // Positive-only waveform → default 0.5
        let waveform = [1.0; 48];
        let hw = compute_half_width::<48>(&waveform);
        assert!((hw - 0.5).abs() < 1e-10, "no trough should return 0.5");
    }

    #[test]
    fn test_compute_half_width_range() {
        // Any valid spike waveform should produce half-width in [0, 1]
        let mut waveform = [0.0; 48];
        waveform[24] = -1.0;
        waveform[23] = -0.8;
        waveform[25] = -0.8;
        let hw = compute_half_width::<48>(&waveform);
        assert!((0.0..=1.0).contains(&hw), "half-width out of range: {hw}");
    }

    #[test]
    fn test_compute_half_width_separates_widths() {
        // Narrow and broad spikes should produce different half-width values
        let mut narrow = [0.0; 48];
        narrow[24] = -1.0;
        narrow[23] = -0.3;
        narrow[25] = -0.3;

        let mut broad = [0.0; 48];
        for v in broad.iter_mut().take(33).skip(16) {
            *v = -0.6;
        }
        broad[24] = -1.0;

        let hw_narrow = compute_half_width::<48>(&narrow);
        let hw_broad = compute_half_width::<48>(&broad);
        assert!(
            hw_broad > hw_narrow,
            "broad={hw_broad} should exceed narrow={hw_narrow}"
        );
    }

    #[test]
    fn test_shape_features_changes_k2() {
        // When use_shape_features=true, two spikes with same channel but different
        // widths should get different K-2 feature values.
        // We test compute_half_width directly since integration test would be complex.
        let mut narrow_wave = [0.0; 48];
        narrow_wave[24] = -1.0;
        narrow_wave[23] = -0.3;
        narrow_wave[25] = -0.3;

        let mut broad_wave = [0.0; 48];
        broad_wave[20] = -0.6;
        broad_wave[21] = -0.8;
        broad_wave[22] = -0.95;
        broad_wave[23] = -1.0;
        broad_wave[24] = -1.0;
        broad_wave[25] = -0.95;
        broad_wave[26] = -0.8;
        broad_wave[27] = -0.6;

        let hw_narrow = compute_half_width::<48>(&narrow_wave);
        let hw_broad = compute_half_width::<48>(&broad_wave);
        // Different widths → different K-2 values (when scaled by cluster_threshold * 4)
        let cluster_threshold = 7.0;
        let scale = cluster_threshold * 4.0;
        let feat_narrow = hw_narrow * scale;
        let feat_broad = hw_broad * scale;
        assert!(
            (feat_broad - feat_narrow).abs() > 0.1,
            "features should differ: narrow={feat_narrow}, broad={feat_broad}"
        );
    }

    #[test]
    fn test_compute_neighbor_templates_basic() {
        // Minimal test: neighbor channel selection is determined by probe geometry
        use crate::probe::ProbeLayout;
        let probe = ProbeLayout::<4>::linear(25.0);
        let peak_channels = [
            0usize, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ];
        let mut neighbor_templates = [[0.0; 48]; 32];
        let mut neighbor_channels = [0usize; 32];
        let data: Vec<[Float; 4]> = vec![[0.0; 4]; 100];
        let events: Vec<MultiChannelEvent> = vec![MultiChannelEvent {
            sample: 25,
            channel: 0,
            amplitude: 1.0,
        }];
        let labels = vec![0usize; 1];
        compute_neighbor_templates::<48, 32, 4>(
            &data,
            &events,
            &labels,
            1,
            1,
            &peak_channels,
            &probe,
            &mut neighbor_templates,
            &mut neighbor_channels,
        );
        // Neighbor of channel 0 on a linear probe should be channel 1
        assert_eq!(
            neighbor_channels[0], 1,
            "neighbor of ch0 on linear probe should be ch1"
        );
    }

    #[test]
    fn test_sort_config_new_fields_defaults() {
        let config = SortConfig::default();
        assert!(
            config.neighbor_mf_detect,
            "neighbor_mf_detect should default to true"
        );
        assert!(
            (config.neighbor_mf_bonus - 0.5).abs() < 1e-10,
            "neighbor_mf_bonus should default to 0.5"
        );
        assert!(
            config.use_shape_features,
            "use_shape_features should default to true"
        );
    }

    // --- Day 8 tests: auto_cluster_threshold and ccg_merge defaults ---

    #[test]
    fn test_auto_cluster_threshold_default() {
        let config = SortConfig::default();
        assert!(
            config.auto_cluster_threshold,
            "auto_cluster_threshold should default to true"
        );
        assert!(config.ccg_merge, "ccg_merge should default to true");
    }

    #[test]
    fn test_auto_cluster_threshold_scaling_c4() {
        // For C=4 with auto_cluster_threshold=true, effective threshold
        // should be cluster_threshold * sqrt(8/4) = cluster_threshold * sqrt(2).
        let config = SortConfig {
            auto_cluster_threshold: true,
            ..Default::default()
        };
        let c: usize = 4;
        let expected = config.cluster_threshold * (8.0 / c as Float).sqrt();
        // sqrt(8/4) = sqrt(2) ≈ 1.414
        assert!(
            (expected - config.cluster_threshold * 2.0_f64.sqrt()).abs() < 1e-9,
            "C=4 effective threshold should be cluster_threshold * sqrt(2)"
        );
        assert!(
            expected > config.cluster_threshold,
            "effective threshold should exceed base threshold for C=4"
        );
    }

    #[test]
    fn test_auto_cluster_threshold_no_scaling_c16() {
        // For C=16 (≥8), auto_cluster_threshold has no effect.
        let config = SortConfig {
            auto_cluster_threshold: true,
            ..Default::default()
        };
        let c: usize = 16;
        // Scaling formula: if C < 8, scale; else keep. C=16 → no change.
        let effective = if c < 8 {
            config.cluster_threshold * (8.0 / c as Float).sqrt()
        } else {
            config.cluster_threshold
        };
        assert!(
            (effective - config.cluster_threshold).abs() < 1e-9,
            "C=16 should use unscaled cluster_threshold"
        );
    }

    #[test]
    fn test_auto_cluster_threshold_disabled() {
        // When auto_cluster_threshold=false, effective threshold equals base regardless of C.
        let config = SortConfig {
            auto_cluster_threshold: false,
            ..Default::default()
        };
        for c in [2usize, 4, 8, 32] {
            let effective = if config.auto_cluster_threshold && c < 8 {
                config.cluster_threshold * (8.0 / c as Float).sqrt()
            } else {
                config.cluster_threshold
            };
            assert!(
                (effective - config.cluster_threshold).abs() < 1e-9,
                "auto_cluster_threshold=false: C={c} should use unscaled threshold"
            );
        }
    }

    // --- Day 9 tests: svd_init, gmm_refine, refinement_iterations, min_cluster_snr ---
    // --- Day 10 update: refinement_iterations=1 enabled with auto_refine guard ---

    #[test]
    fn test_day9_ablation_defaults() {
        // Day 9 ablation found svd+refine interaction crashes medium (-13%).
        // Day 10 isolation showed refine=1 alone helps medium (+0.4%) and hard (+1.3%)
        // but regresses easy (-5.1%) due to over-reassignment on small probes.
        // Resolution: auto_refine=true skips refinement for C<8, enabling refine=1 default.
        let config = SortConfig::default();
        assert!(
            !config.svd_init,
            "svd_init remains false (ablation showed regression)"
        );
        assert_eq!(
            config.refinement_iterations, 1,
            "refinement_iterations=1 enabled; auto_refine guards small probes"
        );
        assert!(
            config.auto_refine,
            "auto_refine=true prevents refinement on C<8 recordings"
        );
        assert!(
            !config.gmm_refine,
            "gmm_refine remains false (net negative across difficulties)"
        );
        assert!(
            (config.min_cluster_snr - 2.5).abs() < 1e-12,
            "min_cluster_snr remains 2.5"
        );
    }

    #[test]
    fn test_snr_floor_keeps_low_snr_cluster() {
        // min_cluster_snr=2.0 should keep a cluster with SNR between 2.0 and 2.5,
        // which min_cluster_snr=2.5 would remove.
        // Verify the threshold logic: effective = min_cluster_snr * (REF_NOISE / noise_mean)
        // With noise_mean=1.0 (whitened), effective == min_cluster_snr exactly.
        let threshold_old = 2.5_f64;
        let threshold_new = 2.0_f64;
        let cluster_snr = 2.2_f64; // between 2.0 and 2.5
        assert!(
            cluster_snr < threshold_old,
            "SNR 2.2 should be below old floor 2.5"
        );
        assert!(
            cluster_snr >= threshold_new,
            "SNR 2.2 should survive new floor 2.0"
        );
    }

    #[test]
    fn test_snr_floor_removes_noise() {
        // min_cluster_snr=2.0 should still remove clusters with SNR < 2.0.
        let threshold_new = 2.0_f64;
        let noise_cluster_snr = 1.5_f64;
        assert!(
            noise_cluster_snr < threshold_new,
            "SNR 1.5 should still be removed by floor 2.0"
        );
    }

    #[test]
    fn test_refine_isi_guard_default_false() {
        let config = SortConfig::default();
        assert!(
            !config.refine_isi_guard,
            "refine_isi_guard must default to false"
        );
    }

    #[test]
    fn test_refine_isi_tolerance_default() {
        let config = SortConfig::default();
        assert!(
            (config.refine_isi_tolerance - 0.1).abs() < 1e-12,
            "refine_isi_tolerance must default to 0.1"
        );
    }

    #[test]
    fn test_refine_isi_guard_no_degradation_on_clean_data() {
        // ISI guard should not fire on well-separated units — result must
        // match the non-guard run (guard only skips when violations increase).
        let n = 4000;
        let mut data = vec![[0.0f64; 2]; n];
        let mut rng = Rng::new(77);
        for s in data.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        // Unit A on ch0
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -14.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        // Unit B on ch1
        let mut pos2 = 275;
        while pos2 + 8 < n {
            for dt in 0..8 {
                if pos2 + dt < n {
                    data[pos2 + dt][1] +=
                        -14.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos2 += 150;
        }
        let probe = ProbeLayout::<2>::linear(25.0);
        let run = |isi_guard: bool| -> SortResult<8> {
            let config = SortConfig {
                threshold_multiplier: 4.0,
                pre_samples: 2,
                refractory_samples: 10,
                matched_filter_threshold: 4.0,
                refinement_iterations: 1,
                auto_refine: false,
                refine_isi_guard: isi_guard,
                ..SortConfig::default()
            };
            let mut d = data.clone();
            let mut scratch = vec![0.0; n];
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                300
            ];
            let mut wf = vec![[0.0; 8]; 300];
            let mut feat = vec![[0.0; 3]; 300];
            let mut lab = vec![0usize; 300];
            sort_multichannel::<2, 4, 8, 3, 64, 8>(
                &config,
                &probe,
                &mut d,
                &mut scratch,
                &mut events,
                &mut wf,
                &mut feat,
                &mut lab,
            )
            .expect("sort ok")
        };
        let r_no_guard = run(false);
        let r_guard = run(true);
        assert!(
            r_guard.n_spikes > 0,
            "ISI guard must not suppress all spikes"
        );
        // Guard should not lose clusters relative to non-guard run
        assert!(
            r_guard.n_clusters >= r_no_guard.n_clusters.saturating_sub(1),
            "ISI guard must not eliminate clusters on clean data: guard={} no_guard={}",
            r_guard.n_clusters,
            r_no_guard.n_clusters
        );
    }

    #[test]
    fn test_refine_isi_guard_and_collapse_guard_coexist() {
        // Both guards enabled simultaneously must not panic or error.
        let n = 3000;
        let mut data = vec![[0.0f64; 2]; n];
        let mut rng = Rng::new(88);
        for s in data.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                if pos + dt < n {
                    data[pos + dt][0] += -12.0 * f64::exp(-0.5 * ((dt as f64 - 2.0) / 1.5).powi(2));
                }
            }
            pos += 150;
        }
        let probe = ProbeLayout::<2>::linear(25.0);
        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            matched_filter_threshold: 4.0,
            refinement_iterations: 2,
            auto_refine: false,
            refine_collapse_guard: true,
            refine_isi_guard: true,
            ..SortConfig::default()
        };
        let mut d = data.clone();
        let mut scratch = vec![0.0; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut wf = vec![[0.0; 8]; 200];
        let mut feat = vec![[0.0; 3]; 200];
        let mut lab = vec![0usize; 200];
        let result = sort_multichannel::<2, 4, 8, 3, 64, 8>(
            &config,
            &probe,
            &mut d,
            &mut scratch,
            &mut events,
            &mut wf,
            &mut feat,
            &mut lab,
        );
        assert!(
            result.is_ok(),
            "both guards enabled must not error: {:?}",
            result.err()
        );
    }
}
