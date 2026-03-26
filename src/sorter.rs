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
//! let mut scratch = [0.0f64; 8];
//! let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
//! assert!(noise[0] > 0.0);
//! assert!(noise[1] > 0.0);
//! ```

use crate::isi;
use crate::online_kmeans::OnlineKMeans;
use crate::probe::ProbeLayout;
use crate::quality;
use crate::spike_sort::{
    align_to_peak, deduplicate_events, detect_spikes_multichannel, extract_peak_channel,
    MultiChannelEvent, SortError, WaveformPca,
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
/// assert!((config.merge_dprime_threshold - 3.1).abs() < 1e-12);
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
/// ```
pub struct SortConfig {
    /// Threshold multiplier for spike detection (sigma units on whitened data).
    pub threshold_multiplier: f64,
    /// Minimum samples between detections on the same channel.
    pub refractory_samples: usize,
    /// Spatial deduplication radius in micrometers.
    pub spatial_radius_um: f64,
    /// Temporal deduplication radius in samples.
    pub temporal_radius: usize,
    /// Half-window for fine peak alignment.
    pub align_half_window: usize,
    /// Samples before the peak in extracted waveforms.
    pub pre_samples: usize,
    /// Distance threshold for creating new clusters.
    pub cluster_threshold: f64,
    /// Maximum observation count per cluster (keeps centroids plastic).
    pub cluster_max_count: u32,
    /// Regularization for whitening eigenvalues.
    pub whitening_epsilon: f64,
    /// D-prime threshold for cluster merging: merge if d' below this value.
    pub merge_dprime_threshold: f64,
    /// ISI violation threshold for cluster merging: skip merge if combined
    /// ISI violation rate would exceed this value.
    pub merge_isi_threshold: f64,
    /// Minimum spikes per cluster to attempt splitting.
    pub split_min_cluster_size: usize,
    /// Bimodality threshold for cluster splitting (gap / std_dev).
    pub split_bimodality_threshold: f64,
    /// D-prime threshold for cross-channel spatial merge.
    pub spatial_merge_dprime: f64,
    /// Enable template subtraction pass to recover masked spikes.
    pub template_subtract: bool,
    /// Minimum spikes per cluster to build a reliable template for subtraction.
    pub template_min_count: usize,
    /// Minimum cluster SNR for auto-curation. Clusters below this are removed.
    pub min_cluster_snr: f64,
    /// Detection mode: Amplitude (default), NEO, or SNEO.
    pub detection_mode: DetectionMode,
    /// Enable CCG-based cluster merging after d-prime and spatial merge.
    /// Merges cluster pairs with high template correlation and no refractory dip.
    pub ccg_merge: bool,
    /// Template correlation threshold for CCG merge candidates.
    pub ccg_template_corr_threshold: f64,
    /// Number of template subtraction passes (0 = disabled, 1 = single pass, 2+ = multi-pass).
    /// Each additional pass subtracts the updated templates and re-detects on the residual.
    pub template_subtract_passes: usize,
    /// ISI violation rate threshold for post-sort cluster splitting.
    /// Clusters with ISI violation rate above this are split along the
    /// first principal axis of their feature distribution.
    pub isi_split_threshold: f64,
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
            cluster_threshold: 5.0,
            cluster_max_count: 1000,
            whitening_epsilon: 1e-6,
            merge_dprime_threshold: 3.1,
            merge_isi_threshold: 0.05,
            split_min_cluster_size: 10,
            split_bimodality_threshold: 2.0,
            spatial_merge_dprime: 1.5,
            template_subtract: true,
            template_min_count: 3,
            min_cluster_snr: 2.5,
            detection_mode: DetectionMode::Amplitude,
            ccg_merge: false,
            ccg_template_corr_threshold: 0.5,
            template_subtract_passes: 2,
            isi_split_threshold: 0.1,
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
    pub snr: f64,
    /// Fraction of ISI violations (inter-spike intervals below refractory).
    pub isi_violation_rate: f64,
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
/// let mut scratch = [0.0f64; 8];
/// let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
/// assert!(noise[0] > 0.0 && noise[0].is_finite());
/// assert!(noise[1] > 0.0 && noise[1].is_finite());
/// ```
pub fn estimate_noise_multichannel<const C: usize>(
    data: &[[f64; C]],
    scratch: &mut [f64],
) -> [f64; C] {
    let t_len = data.len();
    let mut noise = [0.0f64; C];
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
            scratch[t] = libm::fabs(data[t][ch]);
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

/// Compute sample covariance matrix from multi-channel data.
#[allow(clippy::needless_range_loop)]
fn compute_covariance<const C: usize>(data: &[[f64; C]]) -> [[f64; C]; C] {
    let n = data.len();
    let mut cov = [[0.0f64; C]; C];
    if n < 2 {
        return cov;
    }

    // Compute means
    let mut mean = [0.0f64; C];
    for sample in data.iter() {
        for c in 0..C {
            mean[c] += sample[c];
        }
    }
    let inv_n = 1.0 / n as f64;
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
    let inv_nm1 = 1.0 / (n - 1) as f64;
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
/// let mut scratch = [0.0f64; 6];
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
    feature_buf: &[[f64; K]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    dprime_threshold: f64,
    isi_threshold: f64,
    refractory_samples: usize,
    scratch: &mut [f64],
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
        let mut best_dp = f64::MAX;
        let mut best_i = 0usize;
        let mut best_j = 0usize;

        // Compute centroids for each cluster
        let mut centroids = [[0.0f64; 32]; MAX_MERGE_CLUSTERS];
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
                let inv = 1.0 / counts[cl] as f64;
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
                let mut axis = [0.0f64; 32];
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
                let inv_norm = 1.0 / libm::sqrt(axis_norm_sq);
                for d in 0..dim {
                    axis[d] *= inv_norm;
                }

                // Project spikes from cluster i and j onto axis and compute d-prime
                let mut proj_a = [0.0f64; MAX_SPIKES];
                let mut proj_b = [0.0f64; MAX_SPIKES];
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
        if best_dp > dprime_threshold || best_dp == f64::MAX {
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
                    scratch[n_combined] = event_buf[s].sample as f64;
                    n_combined += 1;
                }
            }
        }

        if n_combined >= 2 {
            let times = &mut scratch[..n_combined];
            times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let combined_isi =
                quality::isi_violation_rate(times, refractory_samples as f64).unwrap_or(1.0);
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
    feature_buf: &[[f64; K]],
    n_clusters: usize,
    min_cluster_size: usize,
    bimodality_threshold: f64,
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
            let mut centroid = [0.0f64; 32];
            let mut i = 0;
            while i < count {
                let mut d = 0;
                while d < dim {
                    centroid[d] += feature_buf[indices[i]][d];
                    d += 1;
                }
                i += 1;
            }
            let inv_count = 1.0 / count as f64;
            let mut d = 0;
            while d < dim {
                centroid[d] *= inv_count;
                d += 1;
            }

            // Power iteration: find direction of maximum variance (1D PCA)
            // Initialize with first centered spike
            let mut axis = [0.0f64; 32];
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
            let inv_norm = 1.0 / libm::sqrt(norm_sq);
            d = 0;
            while d < dim {
                axis[d] *= inv_norm;
                d += 1;
            }

            // 3 iterations of power method on the scatter matrix
            let mut iter = 0;
            while iter < 3 {
                let mut new_axis = [0.0f64; 32];
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
                let inv = 1.0 / libm::sqrt(ns);
                d = 0;
                while d < dim {
                    axis[d] = new_axis[d] * inv;
                    d += 1;
                }
                iter += 1;
            }

            // Project all spikes onto the axis
            let mut projections = [0.0f64; MAX_SPIKES];
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
            let mean_proj = sum / count as f64;
            let var = sum_sq / count as f64 - mean_proj * mean_proj;
            let std_dev = if var > 0.0 { libm::sqrt(var) } else { 0.0 };

            if std_dev < 1e-15 {
                cl += 1;
                continue;
            }

            // Sort projections (need sorted copy + index mapping)
            let mut sorted_proj = [0.0f64; MAX_SPIKES];
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
    data: &[[f64; C]],
    event_buf: &[MultiChannelEvent],
    probe: &ProbeLayout<C>,
    n_clusters: usize,
    dprime_threshold: f64,
    spatial_radius_um: f64,
    isi_threshold: f64,
    refractory_samples: usize,
    scratch: &mut [f64],
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
        let mut centroids = [[0.0f64; 32]; MAX_MERGE_CLUSTERS];
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
                let inv = 1.0 / counts[cl] as f64;
                for d in 0..dim {
                    centroids[cl][d] *= inv;
                }
            }
        }

        // Find best merge pair: lowest dprime among spatially proximate clusters
        let mut best_dp = f64::MAX;
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
                let mut axis = [0.0f64; 32];
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
                let inv_norm = 1.0 / libm::sqrt(axis_norm_sq);
                for d in 0..dim {
                    axis[d] *= inv_norm;
                }

                // Project spikes onto axis
                let mut proj_a = [0.0f64; MAX_SPIKES];
                let mut proj_b = [0.0f64; MAX_SPIKES];
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

        if best_dp > dprime_threshold || best_dp == f64::MAX {
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
                    scratch[n_combined] = event_buf[s].sample as f64;
                    n_combined += 1;
                }
            }
        }
        if n_combined >= 2 {
            let times = &mut scratch[..n_combined];
            times.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let combined_isi =
                quality::isi_violation_rate(times, refractory_samples as f64).unwrap_or(1.0);
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
    waveform_buf: &[[f64; W]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    corr_threshold: f64,
    sample_rate: f64,
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
        let mut means = [[0.0f64; W]; MAX_MERGE_CLUSTERS];
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
                let inv = 1.0 / counts[cl] as f64;
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
            let norm_i = libm::sqrt(norm_i_sq);
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
                let norm_j = libm::sqrt(norm_j_sq);
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
        let mut times_a = [0.0f64; MAX_TIMES];
        let mut times_b = [0.0f64; MAX_TIMES];
        let mut n_a = 0;
        let mut n_b = 0;
        let inv_sr = 1.0 / sample_rate;

        for s in 0..n_spikes {
            if s >= labels.len() {
                break;
            }
            if labels[s] == best_i && n_a < MAX_TIMES {
                times_a[n_a] = event_buf[s].sample as f64 * inv_sr;
                n_a += 1;
            } else if labels[s] == best_j && n_b < MAX_TIMES {
                times_b[n_b] = event_buf[s].sample as f64 * inv_sr;
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
    feature_buf: &[[f64; K]],
    event_buf: &[MultiChannelEvent],
    n_clusters: usize,
    isi_threshold: f64,
    refractory_samples: usize,
    min_cluster_size: usize,
    scratch: &mut [f64],
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
                scratch[i] = event_buf[indices[i]].sample as f64;
            }
            let st = &mut scratch[..spike_n];
            st.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

            let isi_rate =
                quality::isi_violation_rate(st, refractory_samples as f64).unwrap_or(0.0);

            if isi_rate <= isi_threshold {
                cl += 1;
                continue;
            }

            // This cluster has high ISI violations -- split along first principal axis.
            // Uses the same power-iteration approach as split_clusters.
            let dim = if K > 32 { 32 } else { K };

            // Compute centroid
            let mut centroid = [0.0f64; 32];
            for i in 0..count {
                for d in 0..dim {
                    centroid[d] += feature_buf[indices[i]][d];
                }
            }
            let inv_count = 1.0 / count as f64;
            for d in centroid.iter_mut().take(dim) {
                *d *= inv_count;
            }

            // Power iteration for first principal axis
            let mut axis = [0.0f64; 32];
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
            let inv_norm = 1.0 / libm::sqrt(norm_sq);
            for a in axis.iter_mut().take(dim) {
                *a *= inv_norm;
            }

            for _iter in 0..3 {
                let mut new_axis = [0.0f64; 32];
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
                let inv = 1.0 / libm::sqrt(ns);
                for d in 0..dim {
                    axis[d] = new_axis[d] * inv;
                }
            }

            // Project spikes and split at median
            let mut projections = [0.0f64; MAX_SPIKES];
            for i in 0..count {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += (feature_buf[indices[i]][d] - centroid[d]) * axis[d];
                }
                projections[i] = dot;
            }

            // Split at median projection (ensures roughly equal-sized halves)
            let mut sorted_proj = [0.0f64; MAX_SPIKES];
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

/// Compute mean waveform per cluster, spike count, and most common peak channel.
#[allow(clippy::too_many_arguments)]
fn compute_cluster_means<const W: usize, const N: usize>(
    waveform_buf: &[[f64; W]],
    labels: &[usize],
    event_buf: &[MultiChannelEvent],
    n_extracted: usize,
    n_clusters: usize,
    means: &mut [[f64; W]; N],
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
            let inv = 1.0 / counts[c] as f64;
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

/// Subtract cluster mean templates from data at each spike location.
#[allow(clippy::too_many_arguments)]
fn subtract_templates_multichannel<const C: usize, const W: usize, const N: usize>(
    data: &mut [[f64; C]],
    event_buf: &[MultiChannelEvent],
    n_spikes: usize,
    labels: &[usize],
    means: &[[f64; W]; N],
    counts: &[u32; N],
    peak_channels: &[usize; N],
    min_count: usize,
    pre_samples: usize,
) {
    // Precompute template norms squared for amplitude scaling
    let mut norms_sq = [0.0f64; N];
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
    waveform: &[f64; W],
    means: &[[f64; W]; N],
    counts: &[u32; N],
    n_clusters: usize,
) -> (usize, f64) {
    let mut best = 0;
    let mut best_dist = f64::MAX;
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
/// let mut data = vec![[0.0f64; 2]; 1000];
/// let mut scratch = vec![0.0f64; 1000];
/// let mut events = vec![MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 100];
/// let mut waveforms = vec![[0.0f64; 16]; 100];
/// let mut features = vec![[0.0f64; 3]; 100];
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
    data: &mut [[f64; C]],
    scratch: &mut [f64],
    event_buf: &mut [MultiChannelEvent],
    waveform_buf: &mut [[f64; W]],
    feature_buf: &mut [[f64; K]],
    labels: &mut [usize],
) -> Result<SortResult<N>, SortError> {
    let t_len = data.len();
    if t_len < W {
        return Err(SortError::InsufficientData);
    }

    // 1. Noise estimation (pre-whitening, for quality metrics later)
    let pre_noise = estimate_noise_multichannel::<C>(data, scratch);
    // Use channel 0 noise as representative for SNR computation
    let mut noise_mean = 0.0;
    for noise_val in pre_noise.iter() {
        noise_mean += noise_val;
    }
    noise_mean /= C as f64;
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
            let unit_noise = [1.0f64; C];
            detect_spikes_multichannel::<C>(
                data,
                config.threshold_multiplier,
                &unit_noise,
                config.refractory_samples,
                event_buf,
            )
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
                let mut cal_buf = [0.0f64; CAL_LEN];
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
                    cal_buf[ci] = libm::fabs(cal_buf[ci] - median);
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
                            let sample = max_idx + 1;
                            event_buf[total] = MultiChannelEvent {
                                sample,
                                channel: ch,
                                amplitude: libm::fabs(data[sample.min(t_len - 1)][ch]),
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

    // 8b. Encode detection channel as a feature dimension.
    //
    // After whitening, single-channel waveform shapes become similar across
    // channels, so PCA on peak-channel waveforms alone cannot distinguish
    // units on different channels. Replacing the least-important PCA
    // component (last dimension) with the normalized channel index provides
    // spatial discrimination. The scale factor controls how strongly channel
    // identity influences clustering relative to waveform shape.
    if K >= 3 {
        let channel_scale = config.cluster_threshold * C as f64;
        for i in 0..n_extracted {
            let ch = event_buf[i].channel;
            feature_buf[i][K - 1] = (ch as f64 / C as f64) * channel_scale;
        }
    }

    // 9. Clustering
    let mut km = OnlineKMeans::<K, N>::new(config.cluster_max_count);
    km.set_create_threshold(config.cluster_threshold);

    // 9a. Seed centroids using farthest-point initialization.
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
        km.init_farthest_point(&feature_buf[..n_extracted], max_init_seeds);
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

    // Cap n_clusters at N (the SortResult array size)
    let mut n_clusters = if n_clusters > N { N } else { n_clusters };

    // 9d. Template subtraction passes: subtract known spikes, re-detect masked ones.
    // Multi-pass: each iteration refines templates and recovers additional masked spikes.
    let n_passes = if config.template_subtract {
        config.template_subtract_passes.max(1)
    } else {
        0
    };
    for _pass in 0..n_passes {
    if n_clusters > 0 && n_extracted > 0 {
        let mut tmpl_means = [[0.0f64; W]; N];
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
        let mut mean_dist = [0.0f64; N];
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
                mean_dist[c] /= tmpl_counts[c] as f64;
            }
        }
        // Max acceptable distance: 3x the mean within-cluster distance
        let mut max_accept_dist = 0.0f64;
        let mut n_valid = 0;
        for c in 0..n_clusters.min(N) {
            if tmpl_counts[c] >= config.template_min_count as u32 {
                max_accept_dist += mean_dist[c];
                n_valid += 1;
            }
        }
        if n_valid > 0 {
            max_accept_dist = (max_accept_dist / n_valid as f64) * 3.0;
        } else {
            max_accept_dist = f64::MAX;
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
            let unit_noise_re = [1.0f64; C];
            let n_re_detected = detect_spikes_multichannel::<C>(
                data,
                config.threshold_multiplier,
                &unit_noise_re,
                config.refractory_samples,
                &mut event_buf[n_extracted..],
            );

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
                scratch[i] = event_buf[i].sample as f64;
            }
            scratch[..sorted_times_n]
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let overlap_radius = (config.temporal_radius + W / 2) as f64;

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
                let t_norm = libm::sqrt(t_norm_sq);
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
                    let peak_amp = libm::fabs(data[pos][ch]);
                    if peak_amp < half_thresh {
                        // Adaptive step: skip more aggressively in quiet regions
                        pos += W / 4;
                        continue;
                    }

                    // Binary search overlap check on sorted spike times
                    let pos_f = pos as f64;
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

                    let r_norm = libm::sqrt(r_norm_sq);
                    let ncc = if r_norm > 1e-15 {
                        dot / (r_norm * t_norm)
                    } else {
                        0.0
                    };

                    if ncc > ncc_threshold {
                        let idx = n_extracted + n_ncc_found;
                        if idx < event_buf.len() && idx < waveform_buf.len() && idx < labels.len() {
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
        let mut mean_wf = [0.0f64; W];
        for i in 0..n_extracted {
            if i < labels.len() && labels[i] == ci {
                for (w, mw) in mean_wf.iter_mut().enumerate() {
                    *mw += waveform_buf[i][w];
                }
            }
        }
        let inv_count = 1.0 / count as f64;
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
                    scratch[spike_idx] = event_buf[i].sample as f64;
                    spike_idx += 1;
                }
            }
            // Sort spike times
            let spike_times = &mut scratch[..spike_idx];
            spike_times
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            // ISI violation rate with refractory period in samples
            cluster.isi_violation_rate =
                quality::isi_violation_rate(spike_times, config.refractory_samples as f64)
                    .unwrap_or(0.0);
        }
    }

    // 11. Auto-curation: remove spikes from clusters below SNR floor
    // Normalize threshold by noise level: SNR uses pre-whitening noise_mean as denominator,
    // so higher noise → lower SNR for the same signal. Scale threshold proportionally.
    // noise_mean ≈ median(|x|)/0.6745 ≈ 1.0 for unit Gaussian noise.
    const REF_NOISE: f64 = 1.0;
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
    templates: [[f64; K]; N],
    n_templates: usize,
    max_distance: f64,
    n_classified: usize,
    n_rejected: usize,
}

impl<const K: usize, const N: usize> OnlineSorter<K, N> {
    /// Create a new online sorter with no templates.
    pub fn new() -> Self {
        Self {
            templates: [[0.0; K]; N],
            n_templates: 0,
            max_distance: f64::MAX,
            n_classified: 0,
            n_rejected: 0,
        }
    }

    /// Create from centroids extracted from a batch sort result.
    /// `centroids` is a slice of feature vectors, up to N are used.
    pub fn from_centroids(centroids: &[[f64; K]]) -> Self {
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
    pub fn add_template(&mut self, centroid: &[f64; K]) -> Option<usize> {
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
    /// Default: f64::MAX (no rejection).
    pub fn set_max_distance(&mut self, max_dist: f64) {
        self.max_distance = max_dist;
    }

    /// Classify a single spike by nearest centroid.
    /// Returns (template_index, distance).
    /// If no templates are loaded, returns (0, f64::MAX).
    pub fn classify(&mut self, features: &[f64; K]) -> (usize, f64) {
        self.n_classified += 1;

        if self.n_templates == 0 {
            return (0, f64::MAX);
        }

        let mut best_idx = 0;
        let mut best_dist = f64::MAX;

        let mut ti = 0;
        while ti < self.n_templates {
            let mut sum_sq = 0.0f64;
            let mut ki = 0;
            while ki < K {
                let diff = features[ki] - self.templates[ti][ki];
                sum_sq += diff * diff;
                ki += 1;
            }
            let dist = libm::sqrt(sum_sq);
            if dist < best_dist {
                best_dist = dist;
                best_idx = ti;
            }
            ti += 1;
        }

        (best_idx, best_dist)
    }

    /// Classify a spike, returning None if distance exceeds max_distance.
    pub fn classify_or_reject(&mut self, features: &[f64; K]) -> Option<(usize, f64)> {
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
    pub fn templates(&self) -> &[[f64; K]] {
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
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);

        let data = [[d0, d1], [d2, d3]];
        let mut scratch = [0.0f64; 2];
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
        let features = [[0.0f64; 2]; 4];
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
        let mut scratch = [0.0f64; 4];

        let dp_thresh: f64 = kani::any();
        let isi_thresh: f64 = kani::any();
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
        let features = [[1.0f64, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
        let threshold: f64 = kani::any();
        kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 100.0);

        let new_n = split_clusters(4, &mut labels, &features, 2, 2, threshold);
        assert!(new_n >= 2);
    }

    /// Prove that `OnlineSorter::classify` does not panic for finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn online_sorter_classify_no_panic() {
        let mut sorter = OnlineSorter::<2, 4>::new();
        let t0: f64 = kani::any();
        let t1: f64 = kani::any();
        let f0: f64 = kani::any();
        let f1: f64 = kani::any();
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
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();
        let t0: f64 = kani::any();
        let t1: f64 = kani::any();
        let t2: f64 = kani::any();
        let t3: f64 = kani::any();

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
        let mut norms_sq = 0.0f64;
        for val in template.iter() {
            norms_sq += val * val;
        }

        // Compute dot product (same as subtract_templates_multichannel)
        let mut dot = 0.0f64;
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
        let mut data = [[0.0f64; 2]; 6];
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();
        kani::assume(d0.is_finite() && d0 >= -1e3 && d0 <= 1e3);
        kani::assume(d1.is_finite() && d1 >= -1e3 && d1 <= 1e3);
        kani::assume(d2.is_finite() && d2 >= -1e3 && d2 <= 1e3);
        kani::assume(d3.is_finite() && d3 >= -1e3 && d3 <= 1e3);
        data[2][0] = d0;
        data[3][0] = d1;
        data[2][1] = d2;
        data[3][1] = d3;

        // Template means: 2 clusters, W=4 each
        let m0: f64 = kani::any();
        let m1: f64 = kani::any();
        kani::assume(m0.is_finite() && m0 >= -1e3 && m0 <= 1e3);
        kani::assume(m1.is_finite() && m1 >= -1e3 && m1 <= 1e3);
        let means: [[f64; 4]; 2] = [[m0, m1, 0.5, -0.5], [0.3, -0.3, m0, m1]];

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
        let f0: f64 = kani::any();
        let f1: f64 = kani::any();
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
        let mut scratch = [0.0f64; 6];

        let dp_thresh: f64 = kani::any();
        let isi_thresh: f64 = kani::any();
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
        fn gaussian(&mut self, mean: f64, std: f64) -> f64 {
            let u1 = (self.next_u64() % 1_000_000 + 1) as f64 / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as f64 / 1_000_000.0;
            let z = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2);
            mean + z * std
        }
    }

    #[test]
    fn test_estimate_noise_multichannel_known() {
        // Constant data: all values are 1.0
        // |1.0| / 0.6745 = 1.4826
        let data = [[1.0, 2.0]; 100];
        let mut scratch = [0.0f64; 100];
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
        let mut data = vec![[0.0f64; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = rng.gaussian(0.0, 1.0);
            sample[1] = rng.gaussian(0.0, 3.0);
        }
        let mut scratch = vec![0.0f64; n];
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
        let data: &[[f64; 2]] = &[];
        let mut scratch = [0.0f64; 1];
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
        assert!((config.cluster_threshold - 5.0).abs() < 1e-12);
        assert_eq!(config.cluster_max_count, 1000);
        assert!((config.whitening_epsilon - 1e-6).abs() < 1e-12);
        assert!((config.merge_dprime_threshold - 3.1).abs() < 1e-12);
        assert!((config.merge_isi_threshold - 0.05).abs() < 1e-12);
        assert!(config.template_subtract);
        assert_eq!(config.template_min_count, 3);
    }

    #[test]
    fn test_compute_covariance_identity() {
        // Uncorrelated unit-variance 2-channel data
        let mut rng = Rng::new(99);
        let n = 5000;
        let mut data = vec![[0.0f64; 2]; n];
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
        let mut data = vec![[0.0f64; 2]; 4]; // too short for W=8
        let mut scratch = vec![0.0f64; 4];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            10
        ];
        let mut waveforms = vec![[0.0f64; 8]; 10];
        let mut features = vec![[0.0f64; 3]; 10];
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
        let mut data = vec![[0.0f64; 2]; 1000];
        let mut scratch = vec![0.0f64; 1000];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            100
        ];
        let mut waveforms = vec![[0.0f64; 8]; 100];
        let mut features = vec![[0.0f64; 3]; 100];
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
        let mut data = vec![[0.0f64; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = rng.gaussian(0.0, 1.0);
            sample[1] = rng.gaussian(0.0, 1.0);
        }

        // Inject spikes: neuron A on channel 0, neuron B on channel 1
        let spike_template_a = |t: f64| -> f64 { -12.0 * libm::exp(-0.5 * t * t) };
        let spike_template_b = |t: f64| -> f64 { -10.0 * libm::exp(-0.5 * (t / 1.5) * (t / 1.5)) };

        // Neuron A fires at regular intervals on channel 0
        let mut spike_pos_a = 200;
        while spike_pos_a + 10 < n {
            for dt in 0..8 {
                let t = (dt as f64 - 2.0) / 1.5;
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
                let t = (dt as f64 - 3.0) / 2.0;
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
        let mut scratch = vec![0.0f64; n];
        let mut events = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            };
            200
        ];
        let mut waveforms = vec![[0.0f64; 8]; 200];
        let mut features = vec![[0.0f64; 3]; 200];
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
            .fold(0.0f64, |a, b| if a > b { a } else { b });
        assert!(
            max_snr > 1.0,
            "Best cluster SNR should be > 1.0, got {}",
            max_snr
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
        let mut scratch = [0.0f64; 6];

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
        let mut scratch = [0.0f64; 6];

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
        let mut scratch = [0.0f64; 6];

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
        let mut scratch = [0.0f64; 3];

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
        let features: [[f64; 2]; 0] = [];
        let events: [MultiChannelEvent; 0] = [];
        let mut scratch: [f64; 0] = [];

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
        let mut scratch = [0.0f64; 9];

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
        let mut scratch = [0.0f64; 8];

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
        let features: [[f64; 2]; 0] = [];
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
        assert_eq!(dist, f64::MAX);
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
        let waveforms: alloc::vec::Vec<[f64; 4]> = vec![
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

        let mut means = [[0.0f64; 4]; 4];
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
        let mut data = vec![[0.0f64; 2]; 20];
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
        let means: [[f64; 4]; 4] = [[1.0, 2.0, 3.0, 2.0], [0.0; 4], [0.0; 4], [0.0; 4]];
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
        let means: [[f64; 4]; 4] = [
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
        let mut data_on = vec![[0.0f64; 2]; n];
        for s in data_on.iter_mut() {
            s[0] = rng.gaussian(0.0, 1.0);
            s[1] = rng.gaussian(0.0, 1.0);
        }
        // Inject spikes on channel 0
        let mut pos = 200;
        while pos + 8 < n {
            for dt in 0..8 {
                let t = (dt as f64 - 2.0) / 1.5;
                data_on[pos + dt][0] += -12.0 * libm::exp(-0.5 * t * t);
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
        let mut scratch_on = vec![0.0f64; n];
        let mut ev_on = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            max_ev
        ];
        let mut wf_on = vec![[0.0f64; 8]; max_ev];
        let mut feat_on = vec![[0.0f64; 3]; max_ev];
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
        let mut scratch_off = vec![0.0f64; n];
        let mut ev_off = vec![
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0
            };
            max_ev
        ];
        let mut wf_off = vec![[0.0f64; 8]; max_ev];
        let mut feat_off = vec![[0.0f64; 3]; max_ev];
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
        let features: Vec<[f64; 2]> = (0..20).map(|i| [i as f64 * 0.1, 0.0]).collect();
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 100, // well-spaced (100 samples apart)
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = [0.0f64; 100];
        let n = isi_violation_split::<2>(
            20,
            &mut labels,
            &features,
            &events,
            1,
            0.1,   // isi_threshold
            15,    // refractory
            5,     // min_cluster_size
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
        let features: Vec<[f64; 2]> = (0..n_spikes)
            .map(|i| {
                if i % 2 == 0 {
                    [5.0, 0.0]
                } else {
                    [0.0, 5.0]
                }
            })
            .collect();
        // Interleaved spike times: 0, 5, 10, 15, ... (5 samples apart, < 15 refractory)
        let events: Vec<MultiChannelEvent> = (0..n_spikes)
            .map(|i| MultiChannelEvent {
                sample: i * 5,
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = vec![0.0f64; n_spikes + 10];
        let n = isi_violation_split::<2>(
            n_spikes,
            &mut labels,
            &features,
            &events,
            1,
            0.05,  // strict ISI threshold
            15,    // refractory
            5,     // min_cluster_size
            &mut scratch,
            8,
        );
        assert!(n >= 2, "interleaved neurons should be split (got {} clusters)", n);
    }

    #[test]
    fn test_isi_violation_split_empty() {
        let mut labels = [];
        let features: Vec<[f64; 2]> = vec![];
        let events: Vec<MultiChannelEvent> = vec![];
        let mut scratch = [0.0f64; 10];
        let n = isi_violation_split::<2>(
            0, &mut labels, &features, &events, 0, 0.1, 15, 5, &mut scratch, 8,
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn test_isi_violation_split_high_threshold() {
        // With threshold = 1.0 (100%), nothing should be split since max ISI rate < 1.0
        let mut labels = [0usize; 20];
        let features: Vec<[f64; 2]> = (0..20).map(|i| [i as f64, 0.0]).collect();
        let events: Vec<MultiChannelEvent> = (0..20)
            .map(|i| MultiChannelEvent {
                sample: i * 5,
                channel: 0,
                amplitude: 5.0,
            })
            .collect();
        let mut scratch = [0.0f64; 30];
        // threshold = 1.0 means only split if 100% ISI violations (impossible)
        let n = isi_violation_split::<2>(
            20, &mut labels, &features, &events, 1, 1.0, 15, 5, &mut scratch, 8,
        );
        assert_eq!(n, 1, "threshold=1.0 should not split");
    }

    #[test]
    fn test_multi_pass_template_subtract() {
        // Multi-pass should find >= spikes as single pass
        use crate::probe::ProbeLayout;
        let probe = ProbeLayout::<4>::linear(25.0);
        let n_samples = 10000;
        let mut data1 = vec![[0.0f64; 4]; n_samples];
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
        let mut events1 = vec![MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; max_events];
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
            &config1, &probe, &mut data1, &mut scratch1,
            &mut events1, &mut wf1, &mut feat1, &mut lab1,
        );
        let r2 = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
            &config2, &probe, &mut data2, &mut scratch2,
            &mut events2, &mut wf2, &mut feat2, &mut lab2,
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        let s1 = r1.unwrap();
        let s2 = r2.unwrap();
        assert!(
            s2.n_spikes >= s1.n_spikes,
            "3-pass ({}) should find >= spikes than 1-pass ({})",
            s2.n_spikes, s1.n_spikes
        );
    }
}
