//! Dynamic-dispatch sorting for runtime channel counts.
//!
//! Wraps the const-generic `sort_multichannel` with runtime dispatch for
//! channel counts from 2 to 128. This module requires the `alloc` feature
//! since it uses `Vec` for dynamically-sized buffers.
//!
//! For channel counts <= 128, it dispatches to the zero-allocation const-generic
//! implementation. Future versions will add a heap-allocated path for >128
//! channels (Neuropixels-scale).
//!
//! # Example
//!
//! ```
//! use zerostone::sorter::SortConfig;
//! use zerostone::sorter_dyn::{DynSortResult, sort_dyn};
//! use zerostone::probe::ProbeLayout;
//!
//! let n_ch = 4;
//! let n_samples = 5000;
//! let mut data = vec![0.0f64; n_samples * n_ch];
//! let probe = ProbeLayout::<4>::linear(25.0);
//! let positions: Vec<[f64; 2]> = probe.positions().to_vec();
//! let config = SortConfig::default();
//!
//! let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch);
//! assert!(result.is_ok());
//! let r = result.unwrap();
//! assert_eq!(r.n_spikes, 0); // noise-only data
//! ```

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::probe::ProbeLayout;
use crate::sorter::{sort_multichannel, ClusterInfo, SortConfig};
use crate::spike_sort::{MultiChannelEvent, SortError};

/// Dynamic sort result with heap-allocated label and spike arrays.
#[derive(Debug)]
pub struct DynSortResult {
    /// Total number of spikes detected and sorted.
    pub n_spikes: usize,
    /// Number of clusters found.
    pub n_clusters: usize,
    /// Cluster label for each spike.
    pub labels: Vec<usize>,
    /// Sample index for each spike.
    pub spike_times: Vec<usize>,
    /// Peak channel for each spike.
    pub spike_channels: Vec<usize>,
    /// Per-cluster quality metrics.
    pub clusters: Vec<DynClusterInfo>,
}

/// Per-cluster quality information (heap-allocated version).
#[derive(Debug, Clone)]
pub struct DynClusterInfo {
    /// Number of spikes assigned to this cluster.
    pub count: usize,
    /// Signal-to-noise ratio.
    pub snr: f64,
    /// ISI violation rate.
    pub isi_violation_rate: f64,
}

impl From<&ClusterInfo> for DynClusterInfo {
    fn from(ci: &ClusterInfo) -> Self {
        Self {
            count: ci.count,
            snr: ci.snr,
            isi_violation_rate: ci.isi_violation_rate,
        }
    }
}

/// Sort multi-channel data with runtime channel count.
///
/// Dispatches to the const-generic implementation for channel counts
/// 2, 4, 8, 16, 32, 64, 128. Returns `SortError::InvalidInput` for
/// unsupported channel counts.
///
/// # Parameters
///
/// * `config` - Sorting configuration.
/// * `positions` - 2D electrode positions, one `[x, y]` per channel.
/// * `data` - Row-major flat array of shape `(n_samples, n_channels)`.
/// * `n_samples` - Number of time samples.
/// * `n_channels` - Number of channels (must be 2, 4, 8, 16, 32, 64, or 128).
///
/// # Returns
///
/// `DynSortResult` with labels, spike times, and cluster info.
pub fn sort_dyn(
    config: &SortConfig,
    positions: &[[f64; 2]],
    data: &mut [f64],
    n_samples: usize,
    n_channels: usize,
) -> Result<DynSortResult, SortError> {
    if data.len() != n_samples * n_channels {
        return Err(SortError::InvalidInput);
    }
    if positions.len() != n_channels {
        return Err(SortError::InvalidInput);
    }

    // W=48, K=4, N=32 (same as Python bindings)
    const W: usize = 48;
    const K: usize = 4;
    const N: usize = 32;

    let max_events = n_samples / config.refractory_samples.max(1) + n_channels;

    macro_rules! dispatch {
        ($c:expr, $cm:expr) => {{
            // Build const-generic probe from positions
            let mut pos_arr = [[0.0f64; 2]; $c];
            for i in 0..$c {
                if i < positions.len() {
                    pos_arr[i] = positions[i];
                }
            }
            let probe = ProbeLayout::<$c>::new(pos_arr);

            // Copy flat data into const-generic array format
            let mut data_arr: Vec<[f64; $c]> = Vec::with_capacity(n_samples);
            for t in 0..n_samples {
                let mut row = [0.0f64; $c];
                row.copy_from_slice(&data[t * $c..(t + 1) * $c]);
                data_arr.push(row);
            }

            let mut scratch = vec![0.0f64; n_samples];
            let mut event_buf = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                max_events
            ];
            let mut waveform_buf = vec![[0.0f64; W]; max_events];
            let mut feature_buf = vec![[0.0f64; K]; max_events];
            let mut labels = vec![0usize; max_events];

            let sr = sort_multichannel::<$c, $cm, W, K, 2304, N>(
                config,
                &probe,
                &mut data_arr,
                &mut scratch,
                &mut event_buf,
                &mut waveform_buf,
                &mut feature_buf,
                &mut labels,
            )?;

            let n = sr.n_spikes;
            Ok(DynSortResult {
                n_spikes: n,
                n_clusters: sr.n_clusters,
                labels: labels[..n].to_vec(),
                spike_times: event_buf[..n].iter().map(|e| e.sample).collect(),
                spike_channels: event_buf[..n].iter().map(|e| e.channel).collect(),
                clusters: sr.clusters[..sr.n_clusters]
                    .iter()
                    .map(DynClusterInfo::from)
                    .collect(),
            })
        }};
    }

    match n_channels {
        2 => dispatch!(2, 4),
        4 => dispatch!(4, 16),
        8 => dispatch!(8, 64),
        16 => dispatch!(16, 256),
        32 => dispatch!(32, 1024),
        64 => dispatch!(64, 4096),
        96 => dispatch!(96, 9216),
        128 => dispatch!(128, 16384),
        n if n > 128 => {
            crate::sorter_heap::sort_heap(config, positions, data, n_samples, n_channels)
        }
        _ => Err(SortError::InvalidInput),
    }
}

/// Sort a long recording in parallel by splitting into segments.
///
/// Divides the recording into `segment_samples`-long chunks and sorts
/// each independently using rayon's thread pool. Spike times in the
/// returned results are offset to reflect their position in the full
/// recording. Each segment's result is returned as a separate
/// `DynSortResult` in the output vector.
///
/// # Parameters
///
/// * `config` - Sorting configuration.
/// * `positions` - 2D electrode positions, one `[x, y]` per channel.
/// * `data` - Row-major flat array of shape `(n_samples, n_channels)`.
/// * `n_samples` - Total number of time samples.
/// * `n_channels` - Number of channels.
/// * `segment_samples` - Samples per segment (e.g. 30000 for 1s at 30kHz).
///
/// # Returns
///
/// A vector of `DynSortResult`, one per segment, with spike times
/// adjusted to global sample indices.
#[cfg(feature = "parallel")]
pub fn sort_batch_parallel(
    config: &SortConfig,
    positions: &[[f64; 2]],
    data: &mut [f64],
    n_samples: usize,
    n_channels: usize,
    segment_samples: usize,
) -> Result<Vec<DynSortResult>, SortError> {
    use rayon::prelude::*;

    if data.len() != n_samples * n_channels {
        return Err(SortError::InvalidInput);
    }
    if positions.len() != n_channels {
        return Err(SortError::InvalidInput);
    }
    if segment_samples < 48 {
        return Err(SortError::InsufficientData);
    }

    // Split into segments
    let n_segments = n_samples.div_ceil(segment_samples);
    let mut segments: Vec<(usize, Vec<f64>)> = Vec::with_capacity(n_segments);
    for seg in 0..n_segments {
        let start = seg * segment_samples;
        let end = (start + segment_samples).min(n_samples);
        let seg_data = data[start * n_channels..end * n_channels].to_vec();
        segments.push((start, seg_data));
    }

    // Sort each segment in parallel
    let results: Vec<Result<DynSortResult, SortError>> = segments
        .par_iter_mut()
        .map(|(offset, seg_data)| {
            let seg_samples = seg_data.len() / n_channels;
            let mut result = sort_dyn(config, positions, seg_data, seg_samples, n_channels)?;
            // Offset spike times to global indices
            for t in result.spike_times.iter_mut() {
                *t += *offset;
            }
            Ok(result)
        })
        .collect();

    // Collect results, propagating first error
    results.into_iter().collect()
}

/// Merge multiple segment results into a single `DynSortResult`.
///
/// Concatenates spikes from all segments. Labels are kept per-segment
/// (not unified across segments). For cross-segment label unification,
/// use `StreamingSorter` instead.
pub fn merge_batch_results(segments: &[DynSortResult]) -> DynSortResult {
    let total_spikes: usize = segments.iter().map(|s| s.n_spikes).sum();
    let mut labels = Vec::with_capacity(total_spikes);
    let mut spike_times = Vec::with_capacity(total_spikes);
    let mut spike_channels = Vec::with_capacity(total_spikes);
    let mut max_cluster_id = 0usize;
    let mut all_clusters = Vec::new();

    for seg in segments {
        // Offset labels by the running cluster count so they don't collide
        for &l in &seg.labels {
            labels.push(l + max_cluster_id);
        }
        spike_times.extend_from_slice(&seg.spike_times);
        spike_channels.extend_from_slice(&seg.spike_channels);
        for c in &seg.clusters {
            all_clusters.push(c.clone());
        }
        max_cluster_id += seg.n_clusters;
    }

    DynSortResult {
        n_spikes: total_spikes,
        n_clusters: all_clusters.len(),
        labels,
        spike_times,
        spike_channels,
        clusters: all_clusters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::ProbeLayout;

    #[test]
    fn test_sort_dyn_4ch_noise() {
        let n_ch = 4;
        let n_samples = 5000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        // Fill with small noise
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 100) as f64 * 0.001 - 0.05;
        }
        let probe = ProbeLayout::<4>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert_eq!(result.n_spikes, 0);
    }

    #[test]
    fn test_sort_dyn_32ch() {
        let n_ch = 32;
        let n_samples = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 13 + 7) % 100) as f64 * 0.001 - 0.05;
        }
        let probe = ProbeLayout::<32>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert!(result.labels.len() == result.n_spikes);
        assert!(result.spike_times.len() == result.n_spikes);
    }

    #[test]
    #[ignore] // Requires release mode (128ch const-generic puts ~130KB on stack)
    fn test_sort_dyn_128ch() {
        let n_ch = 128;
        let n_samples = 2000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 11 + 5) % 100) as f64 * 0.001 - 0.05;
        }
        let probe = ProbeLayout::<128>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert!(result.labels.len() == result.n_spikes);
    }

    #[test]
    fn test_sort_dyn_invalid_channels() {
        let n_ch = 3; // unsupported
        let n_samples = 1000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        let positions: Vec<[f64; 2]> = (0..n_ch).map(|i| [0.0, i as f64 * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_dyn_data_length_mismatch() {
        let n_ch = 4;
        let n_samples = 1000;
        let mut data = vec![0.0f64; 100]; // too short
        let positions: Vec<[f64; 2]> = (0..n_ch).map(|i| [0.0, i as f64 * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch);
        assert!(result.is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sort_batch_parallel_noise() {
        let n_ch = 4;
        let n_samples = 10000;
        let segment = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 100) as f64 * 0.001 - 0.05;
        }
        let probe = ProbeLayout::<4>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let results =
            super::sort_batch_parallel(&config, &positions, &mut data, n_samples, n_ch, segment)
                .unwrap();
        // 10000 / 3000 = 4 segments (ceil)
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.n_spikes, 0);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sort_batch_parallel_merge() {
        let n_ch = 4;
        let n_samples = 6000;
        let segment = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 100) as f64 * 0.001 - 0.05;
        }
        let probe = ProbeLayout::<4>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let results =
            super::sort_batch_parallel(&config, &positions, &mut data, n_samples, n_ch, segment)
                .unwrap();
        assert_eq!(results.len(), 2);
        let merged = super::merge_batch_results(&results);
        assert_eq!(merged.n_spikes, 0);
        assert_eq!(merged.labels.len(), 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sort_batch_parallel_spike_time_offsets() {
        // Inject a clear spike in the second segment to verify offset
        let n_ch = 4;
        let n_samples = 6000;
        let segment = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        // Small noise everywhere
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 7 + 11) % 50) as f64 * 0.001 - 0.025;
        }
        // Inject spike at sample 4000 (second segment, local sample 1000), channel 0
        for offset in 0..20 {
            let t = 4000 + offset;
            if t < n_samples {
                // Triangle shape with peak at center
                let amp = if offset < 10 {
                    -(offset as f64) * 3.0
                } else {
                    -((20 - offset) as f64) * 3.0
                };
                data[t * n_ch] = amp;
            }
        }
        let probe = ProbeLayout::<4>::linear(25.0);
        let positions: Vec<[f64; 2]> = probe.positions().to_vec();
        let config = SortConfig::default();
        let results =
            super::sort_batch_parallel(&config, &positions, &mut data, n_samples, n_ch, segment)
                .unwrap();
        assert_eq!(results.len(), 2);
        // If spikes detected in second segment, their times should be >= 3000
        for t in &results[1].spike_times {
            assert!(
                *t >= 3000,
                "spike time {} should be >= segment offset 3000",
                t
            );
        }
    }

    #[test]
    fn test_sort_dyn_256ch_heap_path() {
        let n_ch = 256;
        let n_samples = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 100) as f64 * 0.001 - 0.05;
        }
        let positions: Vec<[f64; 2]> = (0..n_ch).map(|i| [0.0, i as f64 * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert_eq!(result.n_spikes, 0);
    }

    #[test]
    fn test_sort_dyn_384ch_heap_path() {
        let n_ch = 384;
        let n_samples = 3000;
        let mut data = vec![0.0f64; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 13 + 7) % 100) as f64 * 0.001 - 0.05;
        }
        let positions: Vec<[f64; 2]> = (0..n_ch).map(|i| [0.0, i as f64 * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_dyn(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert_eq!(result.n_spikes, 0);
    }

    #[test]
    fn test_merge_batch_results_empty() {
        let merged = super::merge_batch_results(&[]);
        assert_eq!(merged.n_spikes, 0);
        assert_eq!(merged.n_clusters, 0);
    }
}
