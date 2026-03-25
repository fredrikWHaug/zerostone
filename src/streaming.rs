//! Streaming segment-based spike sorter.
//!
//! Processes continuous recordings in fixed-length segments,
//! maintaining a template library across segments for consistent
//! label assignment. This is the architectural foundation for
//! real-time spike sorting.
//!
//! # Design
//!
//! Each call to [`StreamingSorter::process_segment`] runs the full
//! [`sort_multichannel`] pipeline on one segment, then reconciles
//! the resulting cluster labels against a persistent template library
//! using normalized cross-correlation (NCC). Templates are updated
//! with an exponential moving average so they track slow waveform
//! drift without losing identity.
//!
//! # Example
//!
//! ```
//! use zerostone::streaming::StreamingSorter;
//! use zerostone::sorter::SortConfig;
//!
//! let config = SortConfig::default();
//! let sorter: StreamingSorter<4, 16, 48, 4, 2304, 32> =
//!     StreamingSorter::new(config, 0.95);
//! assert_eq!(sorter.n_templates(), 0);
//! assert_eq!(sorter.segment_count(), 0);
//! ```

use crate::probe::ProbeLayout;
use crate::sorter::{sort_multichannel, SortConfig, SortResult};
use crate::spike_sort::{MultiChannelEvent, SortError};

/// Streaming segment-based spike sorter.
///
/// Maintains a template library across segments for consistent label
/// assignment. Generic parameters match those of [`sort_multichannel`]:
///
/// - `C` -- number of channels
/// - `CM` -- C*C (for whitening matrix)
/// - `W` -- waveform window length in samples
/// - `K` -- PCA feature dimensions
/// - `WM` -- W*W (for PCA covariance)
/// - `N` -- maximum number of clusters/templates
pub struct StreamingSorter<
    const C: usize,
    const CM: usize,
    const W: usize,
    const K: usize,
    const WM: usize,
    const N: usize,
> {
    /// Mean waveforms from previous segments.
    template_library: [[f64; W]; N],
    /// How many spikes have contributed to each template.
    template_counts: [usize; N],
    /// Number of active templates.
    n_templates: usize,
    /// How many segments have been processed.
    segment_count: usize,
    /// Sorting configuration.
    config: SortConfig,
    /// Exponential moving average decay for template updates.
    decay: f64,
}

impl<
        const C: usize,
        const CM: usize,
        const W: usize,
        const K: usize,
        const WM: usize,
        const N: usize,
    > StreamingSorter<C, CM, W, K, WM, N>
{
    /// Create a new streaming sorter with an empty template library.
    ///
    /// `decay` controls the exponential moving average rate for template
    /// updates: `template = decay * old + (1 - decay) * new`. A value
    /// of 0.95 means the template adapts slowly, preserving identity
    /// across segments.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::streaming::StreamingSorter;
    /// use zerostone::sorter::SortConfig;
    ///
    /// let sorter: StreamingSorter<4, 16, 48, 4, 2304, 32> =
    ///     StreamingSorter::new(SortConfig::default(), 0.95);
    /// assert_eq!(sorter.segment_count(), 0);
    /// ```
    pub fn new(config: SortConfig, decay: f64) -> Self {
        Self {
            template_library: [[0.0; W]; N],
            template_counts: [0; N],
            n_templates: 0,
            segment_count: 0,
            config,
            decay,
        }
    }

    /// Process one segment of multi-channel data.
    ///
    /// Runs the full sorting pipeline on the segment, then reconciles
    /// cluster labels against the persistent template library. On the
    /// first segment, templates are initialized directly from the
    /// cluster means. On subsequent segments, new clusters are matched
    /// to existing templates via NCC, and templates are updated with
    /// an exponential moving average.
    ///
    /// All mutable buffers are caller-provided (no heap allocation).
    ///
    /// Returns the [`SortResult`] with labels remapped to the global
    /// template indices.
    #[allow(clippy::too_many_arguments)]
    pub fn process_segment(
        &mut self,
        probe: &ProbeLayout<C>,
        data: &mut [[f64; C]],
        scratch: &mut [f64],
        event_buf: &mut [MultiChannelEvent],
        waveform_buf: &mut [[f64; W]],
        feature_buf: &mut [[f64; K]],
        labels: &mut [usize],
    ) -> Result<SortResult<N>, SortError> {
        // Run the batch pipeline on this segment.
        let result = sort_multichannel::<C, CM, W, K, WM, N>(
            &self.config,
            probe,
            data,
            scratch,
            event_buf,
            waveform_buf,
            feature_buf,
            labels,
        )?;

        let n_spikes = result.n_spikes;
        let n_clusters = result.n_clusters;

        if n_clusters == 0 || n_spikes == 0 {
            self.segment_count += 1;
            return Ok(result);
        }

        // Compute mean waveforms for each cluster in this segment.
        let mut seg_means = [[0.0f64; W]; N];
        let mut seg_counts = [0usize; N];

        for i in 0..n_spikes {
            let label = if i < labels.len() {
                labels[i]
            } else {
                continue;
            };
            if label >= n_clusters || label >= N {
                continue;
            }
            seg_counts[label] += 1;
            for w in 0..W {
                seg_means[label][w] += waveform_buf[i][w];
            }
        }
        for c in 0..n_clusters.min(N) {
            if seg_counts[c] > 0 {
                let inv = 1.0 / seg_counts[c] as f64;
                for mw in seg_means[c].iter_mut() {
                    *mw *= inv;
                }
            }
        }

        if self.segment_count == 0 {
            // First segment: adopt cluster means directly as the template library.
            let nc = n_clusters.min(N);
            self.template_library[..nc].copy_from_slice(&seg_means[..nc]);
            self.template_counts[..nc].copy_from_slice(&seg_counts[..nc]);
            self.n_templates = nc;
        } else {
            // Subsequent segments: match new clusters to existing templates.
            // Build a remapping table: seg_label -> global template index.
            let mut label_remap = [0usize; N];
            let mut used_template = [false; N];

            for c in 0..n_clusters.min(N) {
                if seg_counts[c] == 0 {
                    label_remap[c] = c;
                    continue;
                }

                // Find best matching existing template by NCC.
                let mut best_ncc = -2.0f64;
                let mut best_idx = N; // sentinel: no match

                for (t, &used) in used_template.iter().enumerate().take(self.n_templates) {
                    if used {
                        continue;
                    }
                    let ncc =
                        normalized_cross_correlation::<W>(&seg_means[c], &self.template_library[t]);
                    if ncc > best_ncc {
                        best_ncc = ncc;
                        best_idx = t;
                    }
                }

                if best_ncc > 0.7 && best_idx < N {
                    // Match: remap to existing template, update with EMA.
                    label_remap[c] = best_idx;
                    used_template[best_idx] = true;

                    let alpha = self.decay;
                    let beta = 1.0 - alpha;
                    for (tw, sw) in self.template_library[best_idx]
                        .iter_mut()
                        .zip(seg_means[c].iter())
                    {
                        *tw = alpha * *tw + beta * *sw;
                    }
                    self.template_counts[best_idx] += seg_counts[c];
                } else if self.n_templates < N {
                    // New unit: add to template library.
                    let new_idx = self.n_templates;
                    self.template_library[new_idx] = seg_means[c];
                    self.template_counts[new_idx] = seg_counts[c];
                    label_remap[c] = new_idx;
                    self.n_templates += 1;
                } else {
                    // Library full, keep original label (best effort).
                    label_remap[c] = c;
                }
            }

            // Remap all labels in the output buffer.
            for i in 0..n_spikes.min(labels.len()) {
                let old = labels[i];
                if old < n_clusters && old < N {
                    labels[i] = label_remap[old];
                }
            }
        }

        self.segment_count += 1;

        // Return the result. n_clusters reflects the segment's clustering;
        // the caller can use n_templates() to get the global count.
        Ok(result)
    }

    /// Number of active templates in the library.
    pub fn n_templates(&self) -> usize {
        self.n_templates
    }

    /// Number of segments processed so far.
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// Get a template waveform by index, or `None` if out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::streaming::StreamingSorter;
    /// use zerostone::sorter::SortConfig;
    ///
    /// let sorter: StreamingSorter<4, 16, 48, 4, 2304, 32> =
    ///     StreamingSorter::new(SortConfig::default(), 0.95);
    /// assert!(sorter.template(0).is_none());
    /// ```
    pub fn template(&self, idx: usize) -> Option<&[f64; W]> {
        if idx < self.n_templates {
            Some(&self.template_library[idx])
        } else {
            None
        }
    }

    /// Reset the sorter, clearing all templates and the segment counter.
    pub fn reset(&mut self) {
        self.template_library = [[0.0; W]; N];
        self.template_counts = [0; N];
        self.n_templates = 0;
        self.segment_count = 0;
    }
}

/// Normalized cross-correlation between two waveforms.
///
/// Returns a value in [-1, 1]. Returns 0.0 if either waveform has
/// zero energy (all zeros).
fn normalized_cross_correlation<const W: usize>(a: &[f64; W], b: &[f64; W]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for w in 0..W {
        dot += a[w] * b[w];
        norm_a += a[w] * a[w];
        norm_b += b[w] * b[w];
    }

    let denom = libm::sqrt(norm_a * norm_b);
    if denom < 1e-15 {
        return 0.0;
    }
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::ProbeLayout;

    // Helper: create a simple linear probe with C channels.
    fn make_probe<const C: usize>() -> ProbeLayout<C> {
        let mut positions = [[0.0f64; 2]; C];
        for (i, pos) in positions.iter_mut().enumerate() {
            pos[0] = 0.0;
            pos[1] = i as f64 * 25.0; // 25 um spacing
        }
        ProbeLayout::new(positions)
    }

    // Helper: generate test data with clear spikes on a given channel.
    // Places negative spikes of given amplitude at the given sample positions.
    fn generate_test_data<const C: usize>(
        n_samples: usize,
        spike_channel: usize,
        spike_positions: &[usize],
        amplitude: f64,
    ) -> ([[f64; C]; 5000], usize) {
        assert!(n_samples <= 5000);
        let mut data = [[0.0f64; C]; 5000];

        // Add small background noise (deterministic PRNG).
        let mut rng_state: u64 = 12345;
        for row in data.iter_mut().take(n_samples) {
            for ch in row.iter_mut() {
                // Simple xorshift64 for deterministic noise.
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let noise = ((rng_state as f64) / (u64::MAX as f64) - 0.5) * 0.5;
                *ch = noise;
            }
        }

        // Insert spikes: negative peak with a simple triangular shape.
        for &pos in spike_positions {
            if pos + 20 < n_samples {
                // Triangular spike: ramp down then up, width ~20 samples.
                for off in 0..20 {
                    let shape = if off < 10 {
                        amplitude * (off as f64 / 10.0)
                    } else {
                        amplitude * ((20 - off) as f64 / 10.0)
                    };
                    data[pos + off][spike_channel] += shape; // negative amplitude
                }
            }
        }

        (data, n_samples)
    }

    #[test]
    fn test_streaming_single_segment_matches_batch() {
        const C: usize = 4;
        const CM: usize = 16;
        const W: usize = 48;
        const K: usize = 4;
        const WM: usize = 2304;
        const N: usize = 32;

        let probe = make_probe::<C>();
        let config = SortConfig::default();
        let positions: [usize; 10] = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900];
        let (data_orig, n_samples) = generate_test_data::<C>(5000, 0, &positions, -15.0);

        // Batch sort.
        let mut data_batch = data_orig;
        let mut scratch = [0.0f64; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0f64; W]; 512];
        let mut features = [[0.0f64; K]; 512];
        let mut labels_batch = [0usize; 512];

        let batch_result = sort_multichannel::<C, CM, W, K, WM, N>(
            &config,
            &probe,
            &mut data_batch[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels_batch,
        );

        // Streaming sort (single segment).
        let mut data_stream = data_orig;
        let mut scratch2 = [0.0f64; 20000];
        let mut events2 = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms2 = [[0.0f64; W]; 512];
        let mut features2 = [[0.0f64; K]; 512];
        let mut labels_stream = [0usize; 512];

        let mut sorter: StreamingSorter<C, CM, W, K, WM, N> =
            StreamingSorter::new(SortConfig::default(), 0.95);

        let stream_result = sorter.process_segment(
            &probe,
            &mut data_stream[..n_samples],
            &mut scratch2,
            &mut events2,
            &mut waveforms2,
            &mut features2,
            &mut labels_stream,
        );

        // Both should succeed or fail the same way.
        match (batch_result, stream_result) {
            (Ok(br), Ok(sr)) => {
                assert_eq!(br.n_spikes, sr.n_spikes, "n_spikes mismatch");
                assert_eq!(br.n_clusters, sr.n_clusters, "n_clusters mismatch");
            }
            (Err(_), Err(_)) => {} // Both errored, fine.
            _ => panic!("batch and streaming results diverged"),
        }

        assert_eq!(sorter.segment_count(), 1);
    }

    #[test]
    fn test_streaming_consistent_labels() {
        const C: usize = 4;
        const CM: usize = 16;
        const W: usize = 48;
        const K: usize = 4;
        const WM: usize = 2304;
        const N: usize = 32;

        let probe = make_probe::<C>();
        let positions: [usize; 10] = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900];
        let (data_orig, n_samples) = generate_test_data::<C>(5000, 0, &positions, -15.0);

        let mut sorter: StreamingSorter<C, CM, W, K, WM, N> =
            StreamingSorter::new(SortConfig::default(), 0.95);

        // Process segment 1.
        let mut data1 = data_orig;
        let mut scratch = [0.0f64; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0f64; W]; 512];
        let mut features = [[0.0f64; K]; 512];
        let mut labels1 = [0usize; 512];

        let r1 = sorter.process_segment(
            &probe,
            &mut data1[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels1,
        );

        let n_templates_after_first = sorter.n_templates();

        // Process segment 2 (same data pattern).
        let mut data2 = data_orig;
        let mut labels2 = [0usize; 512];

        let r2 = sorter.process_segment(
            &probe,
            &mut data2[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels2,
        );

        assert_eq!(sorter.segment_count(), 2);

        // The template count should be stable (same data, same clusters).
        if let (Ok(r1_val), Ok(r2_val)) = (r1, r2) {
            if r1_val.n_clusters > 0 && r2_val.n_clusters > 0 {
                // Template count should not have grown (same unit).
                assert_eq!(
                    sorter.n_templates(),
                    n_templates_after_first,
                    "template count should be stable for identical data"
                );
            }
        }
    }

    #[test]
    fn test_streaming_reset() {
        const C: usize = 4;
        const CM: usize = 16;
        const W: usize = 48;
        const K: usize = 4;
        const WM: usize = 2304;
        const N: usize = 32;

        let probe = make_probe::<C>();
        let positions: [usize; 10] = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900];
        let (data_orig, n_samples) = generate_test_data::<C>(5000, 0, &positions, -15.0);

        let mut sorter: StreamingSorter<C, CM, W, K, WM, N> =
            StreamingSorter::new(SortConfig::default(), 0.95);

        let mut data = data_orig;
        let mut scratch = [0.0f64; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0f64; W]; 512];
        let mut features = [[0.0f64; K]; 512];
        let mut labels = [0usize; 512];

        let _ = sorter.process_segment(
            &probe,
            &mut data[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );

        assert!(sorter.segment_count() > 0);

        sorter.reset();

        assert_eq!(sorter.n_templates(), 0);
        assert_eq!(sorter.segment_count(), 0);
        assert!(sorter.template(0).is_none());
    }

    #[test]
    fn test_streaming_new_unit_detection() {
        const C: usize = 4;
        const CM: usize = 16;
        const W: usize = 48;
        const K: usize = 4;
        const WM: usize = 2304;
        const N: usize = 32;

        let probe = make_probe::<C>();

        let mut sorter: StreamingSorter<C, CM, W, K, WM, N> =
            StreamingSorter::new(SortConfig::default(), 0.95);

        // Segment 1: spikes only on channel 0.
        let positions_ch0: [usize; 10] = [200, 500, 800, 1100, 1400, 1700, 2000, 2300, 2600, 2900];
        let (data1_orig, n_samples) = generate_test_data::<C>(5000, 0, &positions_ch0, -15.0);

        let mut data1 = data1_orig;
        let mut scratch = [0.0f64; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0f64; W]; 512];
        let mut features = [[0.0f64; K]; 512];
        let mut labels = [0usize; 512];

        let r1 = sorter.process_segment(
            &probe,
            &mut data1[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );

        let n_templates_seg1 = sorter.n_templates();

        // Segment 2: spikes on channel 0 AND channel 2 (a new unit).
        let positions_ch2: [usize; 8] = [300, 600, 900, 1200, 1500, 1800, 2100, 2400];
        let (mut data2, _) = generate_test_data::<C>(5000, 0, &positions_ch0, -15.0);
        // Add spikes on channel 2 with a different amplitude.
        for &pos in positions_ch2.iter() {
            if pos + 20 < n_samples {
                for off in 0..20 {
                    let shape = if off < 10 {
                        -20.0 * (off as f64 / 10.0)
                    } else {
                        -20.0 * ((20 - off) as f64 / 10.0)
                    };
                    data2[pos + off][2] += shape;
                }
            }
        }

        let r2 = sorter.process_segment(
            &probe,
            &mut data2[..n_samples],
            &mut scratch,
            &mut events,
            &mut waveforms,
            &mut features,
            &mut labels,
        );

        assert_eq!(sorter.segment_count(), 2);

        // If both segments produced clusters, the second segment should
        // have detected the new unit on channel 2.
        if let (Ok(r1_val), Ok(r2_val)) = (r1, r2) {
            if r1_val.n_clusters > 0 && r2_val.n_clusters > 1 {
                assert!(
                    sorter.n_templates() > n_templates_seg1,
                    "expected new template for unit on channel 2: had {}, now {}",
                    n_templates_seg1,
                    sorter.n_templates()
                );
            }
        }
    }

    #[test]
    fn test_ncc_identical() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let ncc = normalized_cross_correlation::<4>(&a, &b);
        assert!(
            (ncc - 1.0).abs() < 1e-12,
            "NCC of identical waveforms should be 1.0"
        );
    }

    #[test]
    fn test_ncc_opposite() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [-1.0, -2.0, -3.0, -4.0];
        let ncc = normalized_cross_correlation::<4>(&a, &b);
        assert!(
            (ncc + 1.0).abs() < 1e-12,
            "NCC of opposite waveforms should be -1.0"
        );
    }

    #[test]
    fn test_ncc_zero_energy() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let ncc = normalized_cross_correlation::<4>(&a, &b);
        assert!(
            (ncc).abs() < 1e-12,
            "NCC with zero-energy waveform should be 0.0"
        );
    }
}
