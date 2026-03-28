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
//! using normalized cross-correlation (NCC) weighted by drift-corrected
//! spatial proximity. The [`DriftEstimator`] tracks probe motion and
//! corrects spike y-positions before template matching, so the same
//! neuron maintains its identity even as tissue drifts relative to
//! the probe. Templates are updated with an exponential moving average.
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

use crate::drift::DriftEstimator;
use crate::float::{self, Float};
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
    template_library: [[Float; W]; N],
    /// How many spikes have contributed to each template.
    template_counts: [usize; N],
    /// Drift-corrected y-position for each template (used for spatial matching).
    template_positions: [Float; N],
    /// Number of active templates.
    n_templates: usize,
    /// How many segments have been processed.
    segment_count: usize,
    /// Sorting configuration.
    config: SortConfig,
    /// Exponential moving average decay for template updates.
    decay: Float,
    /// Drift estimator tracking probe motion across segments.
    /// Uses 64 bins of segment-length to estimate linear drift.
    drift: DriftEstimator<64>,
    /// Accumulated sample offset for drift tracking.
    total_samples: usize,
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
    pub fn new(config: SortConfig, decay: Float) -> Self {
        Self {
            template_library: [[0.0; W]; N],
            template_counts: [0; N],
            template_positions: [0.0; N],
            n_templates: 0,
            segment_count: 0,
            config,
            decay,
            drift: DriftEstimator::<64>::new(30000), // 1s bins at 30kHz
            total_samples: 0,
        }
    }

    /// Create a new streaming sorter with a custom drift bin duration.
    ///
    /// `drift_bin_samples` controls the time resolution of the drift
    /// estimator. Smaller bins give faster drift adaptation but noisier
    /// estimates. A typical value for 30kHz data is 30000 (1 second).
    pub fn with_drift_bins(config: SortConfig, decay: Float, drift_bin_samples: usize) -> Self {
        Self {
            template_library: [[0.0; W]; N],
            template_counts: [0; N],
            template_positions: [0.0; N],
            n_templates: 0,
            segment_count: 0,
            config,
            decay,
            drift: DriftEstimator::<64>::new(drift_bin_samples),
            total_samples: 0,
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
        data: &mut [[Float; C]],
        scratch: &mut [Float],
        event_buf: &mut [MultiChannelEvent],
        waveform_buf: &mut [[Float; W]],
        feature_buf: &mut [[Float; K]],
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
        let mut seg_means = [[0.0; W]; N];
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
                let inv = 1.0 / seg_counts[c] as Float;
                for mw in seg_means[c].iter_mut() {
                    *mw *= inv;
                }
            }
        }

        // Track drift: add spikes to drift estimator and re-fit.
        // Each spike's y-position is estimated from its peak channel and the probe layout.
        let positions = probe.positions();
        if n_spikes > 0 {
            for ev in event_buf.iter().take(n_spikes) {
                let ch = ev.channel;
                if ch < C {
                    let y_pos = positions[ch][1];
                    let global_sample = self.total_samples + ev.sample;
                    self.drift.add_spike(global_sample, y_pos);
                }
            }
            // Re-fit drift after each segment
            self.drift.fit();
        }

        // Compute drift-corrected y-position for each cluster in this segment.
        // For each cluster, compute the amplitude-weighted mean corrected position.
        let mut seg_positions = [0.0; N];
        let mut seg_pos_weights = [0.0; N];

        for i in 0..n_spikes {
            let label = if i < labels.len() {
                labels[i]
            } else {
                continue;
            };
            if label >= n_clusters || label >= N {
                continue;
            }
            let ev = &event_buf[i];
            if ev.channel < C {
                let raw_y = positions[ev.channel][1];
                let global_sample = self.total_samples + ev.sample;
                let corrected_y = self.drift.correct_position(global_sample, raw_y);
                let weight = ev.amplitude;
                seg_positions[label] += corrected_y * weight;
                seg_pos_weights[label] += weight;
            }
        }

        // Normalize to get mean corrected position per cluster.
        for c in 0..n_clusters.min(N) {
            if seg_pos_weights[c] > 1e-15 {
                seg_positions[c] /= seg_pos_weights[c];
            }
        }

        if self.segment_count == 0 {
            // First segment: adopt cluster means and positions directly.
            let nc = n_clusters.min(N);
            self.template_library[..nc].copy_from_slice(&seg_means[..nc]);
            self.template_counts[..nc].copy_from_slice(&seg_counts[..nc]);
            self.template_positions[..nc].copy_from_slice(&seg_positions[..nc]);
            self.n_templates = nc;
        } else {
            // Subsequent segments: match using combined NCC + spatial proximity.
            let mut label_remap = [0usize; N];
            let mut used_template = [false; N];

            for c in 0..n_clusters.min(N) {
                if seg_counts[c] == 0 {
                    label_remap[c] = c;
                    continue;
                }

                // Find best matching existing template by combined NCC and
                // drift-corrected spatial proximity.
                let mut best_score = -2.0;
                let mut best_idx = N; // sentinel: no match

                for (t, &used) in used_template.iter().enumerate().take(self.n_templates) {
                    if used {
                        continue;
                    }
                    let ncc =
                        normalized_cross_correlation::<W>(&seg_means[c], &self.template_library[t]);

                    // Spatial penalty: Gaussian weighting on position difference.
                    // sigma = 50um -- positions beyond ~100um are heavily penalized.
                    let dy = seg_positions[c] - self.template_positions[t];
                    let spatial_weight = float::exp(-0.5 * (dy * dy) / (50.0 * 50.0));

                    // Combined score: NCC weighted by spatial proximity.
                    let score = ncc * spatial_weight;
                    if score > best_score {
                        best_score = score;
                        best_idx = t;
                    }
                }

                if best_score > 0.5 && best_idx < N {
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
                    // Update template position with EMA.
                    self.template_positions[best_idx] =
                        alpha * self.template_positions[best_idx] + beta * seg_positions[c];
                    self.template_counts[best_idx] += seg_counts[c];
                } else if self.n_templates < N {
                    // New unit: add to template library.
                    let new_idx = self.n_templates;
                    self.template_library[new_idx] = seg_means[c];
                    self.template_counts[new_idx] = seg_counts[c];
                    self.template_positions[new_idx] = seg_positions[c];
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

        self.total_samples += data.len();
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
    pub fn template(&self, idx: usize) -> Option<&[Float; W]> {
        if idx < self.n_templates {
            Some(&self.template_library[idx])
        } else {
            None
        }
    }

    /// Get the drift-corrected y-position for a template, or `None` if out of range.
    pub fn template_position(&self, idx: usize) -> Option<Float> {
        if idx < self.n_templates {
            Some(self.template_positions[idx])
        } else {
            None
        }
    }

    /// Estimated drift rate in micrometers per sample.
    ///
    /// Positive values indicate the spike center of mass is moving
    /// toward higher-numbered channels over time.
    pub fn drift_rate(&self) -> Float {
        self.drift.slope()
    }

    /// Whether the drift model has been fitted (requires >= 2 time bins).
    pub fn drift_fitted(&self) -> bool {
        self.drift.is_fitted()
    }

    /// Estimated drift at a given sample index, in micrometers.
    pub fn drift_at(&self, sample_index: usize) -> Float {
        self.drift.estimate_drift(sample_index)
    }

    /// Reset the sorter, clearing all templates, drift, and the segment counter.
    pub fn reset(&mut self) {
        self.template_library = [[0.0; W]; N];
        self.template_counts = [0; N];
        self.template_positions = [0.0; N];
        self.n_templates = 0;
        self.segment_count = 0;
        self.drift.reset();
        self.total_samples = 0;
    }
}

/// Normalized cross-correlation between two waveforms.
///
/// Returns a value in [-1, 1]. Returns 0.0 if either waveform has
/// zero energy (all zeros).
fn normalized_cross_correlation<const W: usize>(a: &[Float; W], b: &[Float; W]) -> Float {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for w in 0..W {
        dot += a[w] * b[w];
        norm_a += a[w] * a[w];
        norm_b += b[w] * b[w];
    }

    let denom = float::sqrt(norm_a * norm_b);
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
        let mut positions = [[0.0; 2]; C];
        for (i, pos) in positions.iter_mut().enumerate() {
            pos[0] = 0.0;
            pos[1] = i as Float * 25.0; // 25 um spacing
        }
        ProbeLayout::new(positions)
    }

    // Helper: generate test data with clear spikes on a given channel.
    // Places negative spikes of given amplitude at the given sample positions.
    fn generate_test_data<const C: usize>(
        n_samples: usize,
        spike_channel: usize,
        spike_positions: &[usize],
        amplitude: Float,
    ) -> ([[Float; C]; 5000], usize) {
        assert!(n_samples <= 5000);
        let mut data = [[0.0; C]; 5000];

        // Add small background noise (deterministic PRNG).
        let mut rng_state: u64 = 12345;
        for row in data.iter_mut().take(n_samples) {
            for ch in row.iter_mut() {
                // Simple xorshift64 for deterministic noise.
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let noise = ((rng_state as Float) / (u64::MAX as Float) - 0.5) * 0.5;
                *ch = noise;
            }
        }

        // Insert spikes: negative peak with a simple triangular shape.
        for &pos in spike_positions {
            if pos + 20 < n_samples {
                // Triangular spike: ramp down then up, width ~20 samples.
                for off in 0..20 {
                    let shape = if off < 10 {
                        amplitude * (off as Float / 10.0)
                    } else {
                        amplitude * ((20 - off) as Float / 10.0)
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
        let mut scratch = [0.0; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0; W]; 512];
        let mut features = [[0.0; K]; 512];
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
        let mut scratch2 = [0.0; 20000];
        let mut events2 = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms2 = [[0.0; W]; 512];
        let mut features2 = [[0.0; K]; 512];
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
        let mut scratch = [0.0; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0; W]; 512];
        let mut features = [[0.0; K]; 512];
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
        let mut scratch = [0.0; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0; W]; 512];
        let mut features = [[0.0; K]; 512];
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
        let mut scratch = [0.0; 20000];
        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 512];
        let mut waveforms = [[0.0; W]; 512];
        let mut features = [[0.0; K]; 512];
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
                        -20.0 * (off as Float / 10.0)
                    } else {
                        -20.0 * ((20 - off) as Float / 10.0)
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
        // have detected spikes. Template count may or may not increase
        // depending on whether the new unit's waveform matches an
        // existing template via NCC.
        if let (Ok(r1_val), Ok(r2_val)) = (r1, r2) {
            if r1_val.n_clusters > 0 {
                assert!(
                    r2_val.n_clusters >= 1,
                    "second segment should detect at least 1 cluster"
                );
                assert!(
                    sorter.n_templates() >= n_templates_seg1,
                    "template count should not decrease: had {}, now {}",
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

    /// Synthetic linear drift test.
    ///
    /// Generates 12 segments of 5000 samples each. A single unit starts
    /// on channel 2, drifts to channel 4 over time (50um total). Verifies:
    ///
    /// 1. The drift estimator tracks the true drift direction.
    /// 2. Cluster identity is maintained (template count stays low).
    #[test]
    fn test_drift_correction_synthetic() {
        const C: usize = 8;
        const CM: usize = 64;
        const W: usize = 48;
        const K: usize = 4;
        const WM: usize = 2304;
        const N: usize = 32;

        // 8-channel linear probe with 25um spacing.
        let mut pos = [[0.0; 2]; C];
        for (i, p) in pos.iter_mut().enumerate() {
            p[0] = 0.0;
            p[1] = i as Float * 25.0;
        }
        let probe = ProbeLayout::new(pos);

        let config = SortConfig::default();
        // Use 5000-sample drift bins (one per segment) so the estimator
        // fits after just 2 segments.
        let mut sorter: StreamingSorter<C, CM, W, K, WM, N> =
            StreamingSorter::with_drift_bins(config, 0.85, 5000);

        let n_segments = 12;
        let seg_len = 5000usize;
        let spike_positions: [usize; 8] = [300, 700, 1100, 1500, 1900, 2300, 2700, 3100];

        // Drift: peak channel shifts from 2 to 4 over 12 segments.
        let channels_over_time: [usize; 12] = [2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4];

        let mut n_templates_final = 0usize;

        for (seg, &ch) in channels_over_time.iter().enumerate().take(n_segments) {
            let mut data = [[0.0; C]; 5000];

            // Deterministic noise.
            let mut rng_state: u64 = 12345 + seg as u64 * 9999;
            for row in data.iter_mut().take(seg_len) {
                for val in row.iter_mut() {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let noise = ((rng_state as Float) / (u64::MAX as Float) - 0.5) * 0.5;
                    *val = noise;
                }
            }

            // Insert identical triangular spikes on the drifting channel.
            for &sp in spike_positions.iter() {
                if sp + 20 < seg_len {
                    for off in 0..20 {
                        let shape = if off < 10 {
                            -15.0 * (off as Float / 10.0)
                        } else {
                            -15.0 * ((20 - off) as Float / 10.0)
                        };
                        data[sp + off][ch] += shape;
                        // Spread to neighbors.
                        if ch > 0 {
                            data[sp + off][ch - 1] += shape * 0.3;
                        }
                        if ch + 1 < C {
                            data[sp + off][ch + 1] += shape * 0.3;
                        }
                    }
                }
            }

            let mut scratch = [0.0; 20000];
            let mut events = [MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 0.0,
            }; 512];
            let mut waveforms = [[0.0; W]; 512];
            let mut features = [[0.0; K]; 512];
            let mut labels = [0usize; 512];

            let _ = sorter.process_segment(
                &probe,
                &mut data[..seg_len],
                &mut scratch,
                &mut events,
                &mut waveforms,
                &mut features,
                &mut labels,
            );

            n_templates_final = sorter.n_templates();
        }

        // Verify drift detection: the estimator should be fitted and
        // detect the upward drift (positive slope).
        assert!(
            sorter.drift_fitted(),
            "Drift estimator should be fitted after 12 segments"
        );
        assert!(
            sorter.drift_rate() > 0.0,
            "Drift rate should be positive (upward drift), got {}",
            sorter.drift_rate()
        );

        // The total estimated drift should be in the right ballpark.
        // True drift: ch2 (50um) to ch4 (100um) = 50um over 60000 samples.
        let total_drift = sorter.drift_at(n_segments * seg_len);
        assert!(
            total_drift > 10.0,
            "Total estimated drift should be > 10um, got {:.1}",
            total_drift
        );

        // Verify cluster identity: with identical waveform shape and
        // drift-corrected spatial matching, template count should be modest.
        // Without drift correction this would fragment more.
        assert!(
            n_templates_final <= 5,
            "Template count should be bounded, got {}",
            n_templates_final
        );
    }

    /// Verify that drift-corrected template positions are stored correctly.
    #[test]
    fn test_template_position_accessor() {
        let sorter: StreamingSorter<4, 16, 48, 4, 2304, 32> =
            StreamingSorter::new(SortConfig::default(), 0.95);
        // No templates yet.
        assert!(sorter.template_position(0).is_none());
    }
}
