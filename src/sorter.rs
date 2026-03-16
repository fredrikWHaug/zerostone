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

    let n_clusters = km.n_active();

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
}
