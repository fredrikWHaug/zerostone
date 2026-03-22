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

use crate::online_kmeans::OnlineKMeans;
use crate::probe::ProbeLayout;
use crate::quality;
use crate::spike_sort::{
    align_to_peak, deduplicate_events, detect_spikes_multichannel, extract_peak_channel,
    MultiChannelEvent, SortError, WaveformPca,
};
use crate::whitening::{WhiteningMatrix, WhiteningMode};

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
/// assert!((config.merge_dprime_threshold - 1.5).abs() < 1e-12);
/// assert!((config.merge_isi_threshold - 0.05).abs() < 1e-12);
/// assert_eq!(config.split_min_cluster_size, 10);
/// assert!((config.split_bimodality_threshold - 2.0).abs() < 1e-12);
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
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            threshold_multiplier: 5.0,
            refractory_samples: 15,
            spatial_radius_um: 75.0,
            temporal_radius: 5,
            align_half_window: 5,
            pre_samples: 16,
            cluster_threshold: 3.0,
            cluster_max_count: 1000,
            whitening_epsilon: 1e-6,
            merge_dprime_threshold: 1.5,
            merge_isi_threshold: 0.05,
            split_min_cluster_size: 10,
            split_bimodality_threshold: 2.0,
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
///     1.5, 0.05, 15, &mut scratch,
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

    loop {
        if current_n < 2 {
            break;
        }

        // Find the pair with the smallest d-prime
        let mut best_dp = f64::MAX;
        let mut best_i = 0usize;
        let mut best_j = 0usize;

        // Compute centroids for each cluster
        let mut centroids = [[0.0f64; 32]; MAX_MERGE_CLUSTERS];
        let mut counts = [0usize; MAX_MERGE_CLUSTERS];
        let dim = if K > 32 { 32 } else { K };

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
                // Merging would create too many ISI violations.
                // Mark this pair as ineligible by breaking (greedy -- we tried
                // the best pair and it failed, so stop).
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
    let unit_noise = [1.0f64; C];
    let n_detected = detect_spikes_multichannel::<C>(
        data,
        config.threshold_multiplier,
        &unit_noise,
        config.refractory_samples,
        event_buf,
    );
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
    let n_extracted =
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

    // 9. Clustering
    let mut km = OnlineKMeans::<K, N>::new(config.cluster_max_count);
    km.set_create_threshold(config.cluster_threshold);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

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
        assert_eq!(config.align_half_window, 5);
        assert_eq!(config.pre_samples, 16);
        assert!((config.cluster_threshold - 3.0).abs() < 1e-12);
        assert_eq!(config.cluster_max_count, 1000);
        assert!((config.whitening_epsilon - 1e-6).abs() < 1e-12);
        assert!((config.merge_dprime_threshold - 1.5).abs() < 1e-12);
        assert!((config.merge_isi_threshold - 0.05).abs() < 1e-12);
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
}
