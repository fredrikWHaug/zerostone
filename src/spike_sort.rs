//! Spike sorting foundations for extracellular electrophysiology.
//!
//! Provides the core building blocks for spike detection and classification:
//! - MAD-based noise estimation and threshold detection
//! - Waveform extraction with configurable pre/post ratio
//! - Peak alignment for consistent waveform representation
//! - PCA for dimensionality reduction of spike waveforms
//! - Template matching (Euclidean distance and NCC) for spike classification
//! - Multi-channel spike detection with per-channel thresholds
//! - Spatial deduplication of detected events using probe geometry
//! - Fine peak alignment within a local window

use crate::linalg::Matrix;
use crate::probe::ProbeLayout;

/// Errors that can occur during spike sorting operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortError {
    /// Not enough data to perform the operation
    InsufficientData,
    /// Model has not been fitted yet
    NotFitted,
    /// Eigendecomposition failed during PCA
    EigenFailed,
    /// Template storage is full
    TemplateFull,
    /// Invalid input parameters
    InvalidInput,
}

/// Extracts fixed-length waveform windows around detected spike times.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `W` - Window length in samples
pub struct WaveformExtractor<const C: usize, const W: usize> {
    pre_samples: usize,
}

impl<const C: usize, const W: usize> Default for WaveformExtractor<C, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const W: usize> WaveformExtractor<C, W> {
    /// Create a new extractor with default pre_samples = W/4.
    pub fn new() -> Self {
        Self { pre_samples: W / 4 }
    }

    /// Create a new extractor with custom pre_samples.
    pub fn with_pre_samples(pre: usize) -> Self {
        assert!(pre < W, "pre_samples must be less than window length");
        Self { pre_samples: pre }
    }

    /// Extract waveform windows around spike times from single-channel data.
    ///
    /// Spikes too close to the start or end of the data are skipped.
    /// Returns the number of waveforms successfully extracted.
    pub fn extract(&self, data: &[f64], spike_times: &[usize], output: &mut [[f64; W]]) -> usize {
        let n = data.len();
        let mut count = 0;
        for &t in spike_times {
            if count >= output.len() {
                break;
            }
            if t < self.pre_samples {
                continue;
            }
            let start = t - self.pre_samples;
            let end = start + W;
            if end > n {
                continue;
            }
            output[count][..W].copy_from_slice(&data[start..end]);
            count += 1;
        }
        count
    }

    /// Align waveforms so the minimum (negative spike peak) is at the pre_samples index.
    ///
    /// Operates on `n` waveforms in-place. For multi-channel data, `channel` selects
    /// which channel to use for peak finding (ignored for single-channel).
    pub fn align_to_peak(&self, waveforms: &mut [[f64; W]], n: usize, _channel: usize) {
        for wf in waveforms.iter_mut().take(n) {
            // Find index of minimum value (negative peak)
            let mut min_idx = 0;
            let mut min_val = wf[0];
            for (j, &val) in wf.iter().enumerate().skip(1) {
                if val < min_val {
                    min_val = val;
                    min_idx = j;
                }
            }

            let target = self.pre_samples;
            if min_idx == target {
                continue;
            }

            let shift = min_idx as isize - target as isize;
            if shift > 0 {
                // Peak is to the right of target, shift waveform left
                let shift = shift as usize;
                if shift >= W {
                    continue;
                }
                let mut temp = [0.0f64; W];
                let copyable = W - shift;
                temp[..copyable].copy_from_slice(&wf[shift..]);
                *wf = temp;
            } else {
                // Peak is to the left of target, shift waveform right
                let shift = (-shift) as usize;
                if shift >= W {
                    continue;
                }
                let mut temp = [0.0f64; W];
                let copyable = W - shift;
                temp[shift..shift + copyable].copy_from_slice(&wf[..copyable]);
                *wf = temp;
            }
        }
    }

    /// Number of samples before the spike time.
    pub fn pre_samples(&self) -> usize {
        self.pre_samples
    }

    /// Number of samples after the spike time.
    pub fn post_samples(&self) -> usize {
        W - self.pre_samples
    }
}

/// PCA for single-channel spike waveforms.
///
/// # Type Parameters
///
/// * `W` - Waveform length (dimensionality of input)
/// * `K` - Number of principal components to retain
/// * `WM` - Must equal W*W (for the covariance matrix)
pub struct WaveformPca<const W: usize, const K: usize, const WM: usize> {
    mean: [f64; W],
    components: [[f64; W]; K],
    explained_variance: [f64; K],
    total_variance: f64,
    fitted: bool,
}

impl<const W: usize, const K: usize, const WM: usize> Default for WaveformPca<W, K, WM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const W: usize, const K: usize, const WM: usize> WaveformPca<W, K, WM> {
    /// Create a new PCA instance.
    pub fn new() -> Self {
        assert!(WM == W * W, "WM must equal W * W");
        assert!(K <= W, "K must be <= W");
        Self {
            mean: [0.0; W],
            components: [[0.0; W]; K],
            explained_variance: [0.0; K],
            total_variance: 0.0,
            fitted: false,
        }
    }

    /// Fit PCA on a set of waveforms.
    ///
    /// Computes the mean, covariance matrix, and top K eigenvectors.
    pub fn fit(&mut self, waveforms: &[[f64; W]]) -> Result<(), SortError> {
        let n = waveforms.len();
        if n < 2 {
            return Err(SortError::InsufficientData);
        }

        // Compute mean
        let mut mean = [0.0f64; W];
        for wf in waveforms.iter() {
            for (j, &val) in wf.iter().enumerate() {
                mean[j] += val;
            }
        }
        let inv_n = 1.0 / n as f64;
        for m in mean.iter_mut() {
            *m *= inv_n;
        }

        // Build covariance matrix
        let mut cov = Matrix::<W, WM>::zeros();
        for wf in waveforms.iter() {
            let mut centered = [0.0f64; W];
            for (j, (&val, &m)) in wf.iter().zip(mean.iter()).enumerate() {
                centered[j] = val - m;
            }
            for r in 0..W {
                for c in r..W {
                    let val = cov.get(r, c) + centered[r] * centered[c];
                    cov.set(r, c, val);
                    if r != c {
                        cov.set(c, r, val);
                    }
                }
            }
        }
        // Normalize by n-1
        let inv_nm1 = 1.0 / (n - 1) as f64;
        for val in cov.data_mut().iter_mut() {
            *val *= inv_nm1;
        }

        // Eigendecomposition (eigenvalues returned in descending order)
        let eigen = cov
            .eigen_symmetric(200, 1e-12)
            .map_err(|_| SortError::EigenFailed)?;

        // Total variance = sum of all eigenvalues
        let total: f64 = eigen.eigenvalues.iter().sum();
        self.total_variance = total;

        // Extract top K components
        for k in 0..K {
            self.explained_variance[k] = eigen.eigenvalues[k];
            for j in 0..W {
                self.components[k][j] = eigen.eigenvectors.get(j, k);
            }
        }

        self.mean = mean;
        self.fitted = true;
        Ok(())
    }

    /// Project a waveform onto the K principal components.
    pub fn transform(&self, waveform: &[f64; W], output: &mut [f64; K]) -> Result<(), SortError> {
        if !self.fitted {
            return Err(SortError::NotFitted);
        }
        for (k, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for ((&wv, &mv), &cv) in waveform
                .iter()
                .zip(self.mean.iter())
                .zip(self.components[k].iter())
            {
                sum += (wv - mv) * cv;
            }
            *out = sum;
        }
        Ok(())
    }

    /// Fraction of variance explained by each component.
    pub fn explained_variance_ratio(&self) -> [f64; K] {
        let mut ratios = [0.0; K];
        if self.total_variance > 0.0 {
            for (r, &ev) in ratios.iter_mut().zip(self.explained_variance.iter()) {
                *r = ev / self.total_variance;
            }
        }
        ratios
    }

    /// Whether the PCA has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// The principal components (K vectors of length W).
    pub fn components(&self) -> &[[f64; W]; K] {
        &self.components
    }

    /// The mean waveform.
    pub fn mean(&self) -> &[f64; W] {
        &self.mean
    }
}

/// Template-based spike classification.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `W` - Window length in samples
/// * `N` - Maximum number of templates
pub struct TemplateMatch<const C: usize, const W: usize, const N: usize> {
    templates: [[f64; W]; N],
    counts: [usize; N],
    n_templates: usize,
}

impl<const C: usize, const W: usize, const N: usize> Default for TemplateMatch<C, W, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const W: usize, const N: usize> TemplateMatch<C, W, N> {
    /// Create a new empty template matcher.
    pub fn new() -> Self {
        Self {
            templates: [[0.0; W]; N],
            counts: [0; N],
            n_templates: 0,
        }
    }

    /// Add a template waveform. Returns the template index.
    pub fn add_template(&mut self, template: &[f64; W]) -> Result<usize, SortError> {
        if self.n_templates >= N {
            return Err(SortError::TemplateFull);
        }
        let idx = self.n_templates;
        self.templates[idx] = *template;
        self.counts[idx] = 1;
        self.n_templates += 1;
        Ok(idx)
    }

    /// Match a waveform to the best template using Euclidean distance.
    ///
    /// Returns `(best_template_index, distance)` or `None` if no templates exist.
    pub fn match_waveform(&self, waveform: &[f64; W]) -> Option<(usize, f64)> {
        if self.n_templates == 0 {
            return None;
        }
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for i in 0..self.n_templates {
            let mut dist = 0.0;
            for (&wv, &tv) in waveform.iter().zip(self.templates[i].iter()) {
                let d = wv - tv;
                dist += d * d;
            }
            dist = libm::sqrt(dist);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        Some((best_idx, best_dist))
    }

    /// Match a waveform using normalized cross-correlation.
    ///
    /// Returns `(best_template_index, ncc)` or `None` if no templates exist.
    /// NCC ranges from -1 to 1, where 1 is a perfect match.
    pub fn match_waveform_ncc(&self, waveform: &[f64; W]) -> Option<(usize, f64)> {
        if self.n_templates == 0 {
            return None;
        }

        // Compute waveform norm
        let mut wf_norm_sq = 0.0;
        for &val in waveform.iter() {
            wf_norm_sq += val * val;
        }
        let wf_norm = libm::sqrt(wf_norm_sq);
        if wf_norm < 1e-15 {
            return Some((0, 0.0));
        }

        let mut best_idx = 0;
        let mut best_ncc = f64::MIN;
        for i in 0..self.n_templates {
            let mut dot = 0.0;
            let mut t_norm_sq = 0.0;
            for (&wv, &tv) in waveform.iter().zip(self.templates[i].iter()) {
                dot += wv * tv;
                t_norm_sq += tv * tv;
            }
            let t_norm = libm::sqrt(t_norm_sq);
            let ncc = if t_norm < 1e-15 {
                0.0
            } else {
                dot / (wf_norm * t_norm)
            };
            if ncc > best_ncc {
                best_ncc = ncc;
                best_idx = i;
            }
        }
        Some((best_idx, best_ncc))
    }

    /// Update a template with a running average of a new waveform.
    pub fn update_template(&mut self, idx: usize, waveform: &[f64; W]) {
        if idx >= self.n_templates {
            return;
        }
        self.counts[idx] += 1;
        let n = self.counts[idx] as f64;
        let alpha = 1.0 / n;
        for (tv, &wv) in self.templates[idx].iter_mut().zip(waveform.iter()) {
            *tv = *tv * (1.0 - alpha) + wv * alpha;
        }
    }

    /// Number of templates currently stored.
    pub fn n_templates(&self) -> usize {
        self.n_templates
    }

    /// Get a reference to a template by index.
    pub fn template(&self, idx: usize) -> Option<&[f64; W]> {
        if idx < self.n_templates {
            Some(&self.templates[idx])
        } else {
            None
        }
    }

    /// Get the observation count for a template.
    pub fn count(&self, idx: usize) -> usize {
        if idx < self.n_templates {
            self.counts[idx]
        } else {
            0
        }
    }
}

/// Summary of a spike cluster.
pub struct SpikeCluster<const C: usize, const W: usize> {
    /// Mean waveform for this cluster
    pub mean_waveform: [f64; W],
    /// Number of spikes assigned to this cluster
    pub count: usize,
}

/// Estimate noise standard deviation using the Median Absolute Deviation.
///
/// sigma = median(|x|) / 0.6745
///
/// This is robust to spike contamination since spikes are rare events.
/// `scratch` must be at least as large as `data`.
pub fn estimate_noise_mad(data: &[f64], scratch: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    assert!(scratch.len() >= n);
    for (s, &d) in scratch.iter_mut().zip(data.iter()) {
        *s = libm::fabs(d);
    }
    let s = &mut scratch[..n];
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) * 0.5
    };
    median / 0.6745
}

/// Detect spikes via negative-threshold crossing with peak finding.
///
/// Finds samples where `data[i] < -threshold`, then within each refractory window
/// selects the sample with the most negative value as the spike time.
///
/// Returns the number of spikes detected. Spike times are written to `spike_times`.
pub fn detect_spikes(
    data: &[f64],
    threshold: f64,
    refractory: usize,
    spike_times: &mut [usize],
) -> usize {
    let n = data.len();
    let mut count = 0;
    let mut i = 0;
    while i < n {
        if data[i] < -threshold {
            // Found a crossing; search the refractory window for the peak
            let end = if i + refractory < n {
                i + refractory
            } else {
                n
            };
            let mut min_idx = i;
            let mut min_val = data[i];
            let mut j = i + 1;
            while j < end {
                if data[j] < min_val {
                    min_val = data[j];
                    min_idx = j;
                }
                j += 1;
            }
            if count < spike_times.len() {
                spike_times[count] = min_idx;
                count += 1;
            }
            i = end;
        } else {
            i += 1;
        }
    }
    count
}

/// A spike event detected on a multi-channel recording.
///
/// Stores the time index, channel, and absolute amplitude of a detected
/// threshold crossing. Used as the output element for [`detect_spikes_multichannel`]
/// and input to [`deduplicate_events`].
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::MultiChannelEvent;
///
/// let event = MultiChannelEvent { sample: 1000, channel: 3, amplitude: 5.2 };
/// assert_eq!(event.sample, 1000);
/// assert_eq!(event.channel, 3);
/// assert!((event.amplitude - 5.2).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiChannelEvent {
    /// Time index (sample offset) of the detected spike.
    pub sample: usize,
    /// Channel on which the spike was detected.
    pub channel: usize,
    /// Absolute amplitude at the peak (always positive).
    pub amplitude: f64,
}

/// Detect spikes across multiple channels with per-channel noise thresholds.
///
/// Scans each channel of a `T`-sample, `C`-channel recording for negative
/// threshold crossings, using the same refractory-window peak-finding logic
/// as [`detect_spikes`]. The threshold for channel `ch` is
/// `threshold_multiplier * noise_estimates[ch]`.
///
/// Detected events are written into `output` sorted by sample index (within
/// each channel, events are naturally time-ordered; the final output
/// interleaves channels by time). Returns the number of events written.
///
/// # Type Parameters
///
/// * `C` - Number of channels
///
/// # Arguments
///
/// * `data` - Multi-channel data as a slice of `[f64; C]` (one array per time step)
/// * `threshold_multiplier` - Scalar multiplied by per-channel noise to get thresholds
/// * `noise_estimates` - Per-channel noise standard deviation (e.g., from MAD)
/// * `refractory` - Minimum samples between detections on the same channel
/// * `output` - Caller-provided buffer for detected events
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{detect_spikes_multichannel, MultiChannelEvent};
///
/// // 2 channels, 20 time steps
/// let mut data = [[0.0f64; 2]; 20];
/// data[5][0] = -6.0; // spike on channel 0
/// data[15][1] = -8.0; // spike on channel 1
///
/// let noise = [1.0, 1.0];
/// let mut events = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 16];
/// let n = detect_spikes_multichannel::<2>(&data, 4.0, &noise, 5, &mut events);
/// assert_eq!(n, 2);
/// assert_eq!(events[0].channel, 0);
/// assert_eq!(events[0].sample, 5);
/// assert_eq!(events[1].channel, 1);
/// assert_eq!(events[1].sample, 15);
/// ```
pub fn detect_spikes_multichannel<const C: usize>(
    data: &[[f64; C]],
    threshold_multiplier: f64,
    noise_estimates: &[f64; C],
    refractory: usize,
    output: &mut [MultiChannelEvent],
) -> usize {
    let t_len = data.len();
    if t_len == 0 || refractory == 0 {
        return 0;
    }

    // Phase 1: detect per-channel, collecting into output buffer.
    // We process channels in order, appending events, then merge-sort by time.
    // To avoid allocation we use a two-pass approach: first count per channel
    // to partition the output buffer, but that requires scratch. Instead, we
    // do a simple approach: detect all channels sequentially, then insertion-sort
    // the result by sample index (stable sort preserves channel order for ties).

    let mut total = 0usize;

    let mut ch = 0;
    while ch < C {
        let thresh = threshold_multiplier * noise_estimates[ch];
        let mut i = 0;
        while i < t_len {
            if data[i][ch] < -thresh {
                // Found crossing: search refractory window for peak
                let end = if i + refractory < t_len {
                    i + refractory
                } else {
                    t_len
                };
                let mut min_idx = i;
                let mut min_val = data[i][ch];
                let mut j = i + 1;
                while j < end {
                    if data[j][ch] < min_val {
                        min_val = data[j][ch];
                        min_idx = j;
                    }
                    j += 1;
                }
                if total < output.len() {
                    output[total] = MultiChannelEvent {
                        sample: min_idx,
                        channel: ch,
                        amplitude: libm::fabs(min_val),
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

    // Insertion sort by sample index (stable: preserves channel order for ties)
    let mut k = 1;
    while k < total {
        let key = output[k];
        let mut pos = k;
        while pos > 0 && output[pos - 1].sample > key.sample {
            output[pos] = output[pos - 1];
            pos -= 1;
        }
        output[pos] = key;
        k += 1;
    }

    total
}

/// Compute per-channel adaptive detection thresholds from noise statistics.
///
/// For each channel:
/// 1. Estimates noise via MAD (robust to spike contamination)
/// 2. Base threshold = `base_multiplier * noise[ch]`
/// 3. Applies a minimum floor: `max(base_threshold, min_threshold)`
/// 4. Activity check: counts negative crossings at the base threshold.
///    If the crossing rate exceeds `max_rate_hz`, scales the threshold
///    up by `sqrt(rate / max_rate_hz)` to suppress over-detection.
///
/// Returns per-channel absolute thresholds suitable for comparison
/// against `-data[t][ch]` (i.e., threshold values are positive).
///
/// # Arguments
///
/// * `data` - Multi-channel recording, `data[t][ch]`
/// * `base_multiplier` - Multiplier for MAD noise estimate (typically 4.0-5.0)
/// * `min_threshold` - Minimum threshold floor for dead channel protection
/// * `max_rate_hz` - Maximum allowed crossing rate before threshold is raised
/// * `sample_rate` - Sampling rate in Hz
/// * `scratch` - Scratch buffer, must have at least `data.len()` elements
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::compute_adaptive_thresholds;
///
/// // 2-channel data: channel 0 has noise, channel 1 is dead
/// let data = [
///     [0.3, 0.0], [-0.5, 0.0], [0.1, 0.0], [0.4, 0.0],
///     [-0.2, 0.0], [0.6, 0.0], [-0.3, 0.0], [0.2, 0.0],
/// ];
/// let mut scratch = [0.0f64; 8];
/// let thresholds = compute_adaptive_thresholds::<2>(
///     &data, 4.0, 0.5, 100.0, 30000.0, &mut scratch,
/// );
/// // Channel 0: threshold based on noise level
/// assert!(thresholds[0] > 0.5);
/// // Channel 1: dead channel gets min_threshold
/// assert!((thresholds[1] - 0.5).abs() < 1e-12);
/// ```
pub fn compute_adaptive_thresholds<const C: usize>(
    data: &[[f64; C]],
    base_multiplier: f64,
    min_threshold: f64,
    max_rate_hz: f64,
    sample_rate: f64,
    scratch: &mut [f64],
) -> [f64; C] {
    let t_len = data.len();
    let mut thresholds = [0.0f64; C];

    // Handle empty data: return min_threshold for all channels.
    if t_len == 0 {
        let mut ch = 0;
        while ch < C {
            thresholds[ch] = min_threshold;
            ch += 1;
        }
        return thresholds;
    }

    assert!(
        scratch.len() >= t_len,
        "scratch buffer must be at least data.len() elements"
    );

    // Step 1 & 2: Estimate noise per channel via MAD and compute base thresholds.
    // Inline the MAD computation (same as estimate_noise_mad) to avoid
    // aliasing scratch as both data and scratch arguments.
    let mut ch = 0;
    while ch < C {
        // Copy absolute values of channel data into scratch.
        let mut t = 0;
        while t < t_len {
            scratch[t] = libm::fabs(data[t][ch]);
            t += 1;
        }
        let s = &mut scratch[..t_len];
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let median = if t_len % 2 == 1 {
            s[t_len / 2]
        } else {
            (s[t_len / 2 - 1] + s[t_len / 2]) * 0.5
        };
        let noise = median / 0.6745;
        let base = base_multiplier * noise;

        // Step 3: Apply minimum threshold floor (dead channel protection).
        thresholds[ch] = if base > min_threshold {
            base
        } else {
            min_threshold
        };

        ch += 1;
    }

    // Step 4: Activity check -- count crossings and raise threshold on
    // overactive channels.
    let duration_s = t_len as f64 / sample_rate;
    if duration_s > 0.0 && max_rate_hz > 0.0 {
        let mut ch = 0;
        while ch < C {
            let thresh = thresholds[ch];
            let mut crossings = 0usize;
            let mut t = 1;
            while t < t_len {
                // Count negative crossings: sample goes below -thresh
                // while the previous sample was above -thresh.
                if data[t][ch] < -thresh && data[t - 1][ch] >= -thresh {
                    crossings += 1;
                }
                t += 1;
            }

            let rate = crossings as f64 / duration_s;
            if rate > max_rate_hz {
                let scale = libm::sqrt(rate / max_rate_hz);
                thresholds[ch] *= scale;
                // Re-apply floor after scaling (should always be >=, but be safe).
                if thresholds[ch] < min_threshold {
                    thresholds[ch] = min_threshold;
                }
            }

            ch += 1;
        }
    }

    thresholds
}

/// Remove duplicate detections of the same spike across neighboring channels.
///
/// When a neuron fires, its extracellular waveform appears on multiple nearby
/// channels. This function keeps only the event with the largest amplitude
/// within each spatiotemporal neighborhood, following the "locally exclusive"
/// strategy used by SpikeInterface and MountainSort5.
///
/// Two events are considered duplicates if:
/// 1. Their sample indices differ by at most `temporal_radius` samples, AND
/// 2. Their channels are neighbors on the probe (within `spatial_radius_um`).
///
/// The deduplication is performed in-place. Retained events are packed to the
/// front of the slice; the function returns the number of retained events.
///
/// # Type Parameters
///
/// * `C` - Number of channels on the probe
///
/// # Arguments
///
/// * `events` - Mutable slice of detected events (modified in place)
/// * `n_events` - Number of valid events in the slice
/// * `probe` - Probe geometry for spatial neighbor queries
/// * `spatial_radius_um` - Maximum distance (micrometers) for two channels to be neighbors
/// * `temporal_radius` - Maximum sample offset for two events to be considered the same spike
///
/// # Returns
///
/// Number of retained (deduplicated) events. Always `<= n_events`.
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{deduplicate_events, MultiChannelEvent};
/// use zerostone::probe::ProbeLayout;
///
/// let probe = ProbeLayout::<4>::linear(25.0);
/// let mut events = [
///     MultiChannelEvent { sample: 100, channel: 1, amplitude: 5.0 },
///     MultiChannelEvent { sample: 101, channel: 2, amplitude: 7.0 }, // larger, wins
///     MultiChannelEvent { sample: 500, channel: 0, amplitude: 3.0 }, // far away, kept
///     MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 },
/// ];
/// let n = deduplicate_events::<4>(&mut events, 3, &probe, 30.0, 5);
/// assert_eq!(n, 2);
/// assert_eq!(events[0].sample, 101); // winner from first group
/// assert_eq!(events[1].sample, 500); // isolated event
/// ```
pub fn deduplicate_events<const C: usize>(
    events: &mut [MultiChannelEvent],
    n_events: usize,
    probe: &ProbeLayout<C>,
    spatial_radius_um: f64,
    temporal_radius: usize,
) -> usize {
    let n = if n_events < events.len() {
        n_events
    } else {
        events.len()
    };
    if n == 0 {
        return 0;
    }

    // Mark events for removal: use amplitude = -1.0 as a tombstone.
    // For each event, check if a later event in the temporal window on a
    // neighboring channel has larger amplitude. If so, mark the smaller one.
    // This is O(n * n * C) worst case but n is typically small per buffer.

    // We need neighbor lookup. Use the probe API with a stack buffer.
    let mut i = 0;
    while i < n {
        if events[i].amplitude < 0.0 {
            // Already removed
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < n {
            // Since events are sorted by sample, once we exceed temporal_radius
            // we can break early.
            if events[j].sample > events[i].sample + temporal_radius {
                break;
            }
            if events[j].amplitude < 0.0 {
                j += 1;
                continue;
            }

            // Check temporal proximity (bidirectional, but since sorted,
            // events[j].sample >= events[i].sample)
            let dt = events[j].sample - events[i].sample;
            if dt <= temporal_radius {
                // Check spatial proximity
                let dist = probe.channel_distance(events[i].channel, events[j].channel);
                if dist <= spatial_radius_um {
                    // Same spike: remove the one with smaller amplitude
                    if events[i].amplitude >= events[j].amplitude {
                        events[j].amplitude = -1.0; // tombstone
                    } else {
                        events[i].amplitude = -1.0; // tombstone
                        break; // event i is dead, no need to compare further
                    }
                }
            }
            j += 1;
        }
        i += 1;
    }

    // Compact: move surviving events to the front
    let mut write = 0;
    let mut read = 0;
    while read < n {
        if events[read].amplitude >= 0.0 {
            if write != read {
                events[write] = events[read];
            }
            write += 1;
        }
        read += 1;
    }
    write
}

/// Fine-align spike event times to the nearest negative peak within a window.
///
/// For each event in `events[..n_events]`, searches the data on the event's
/// channel within `+/- half_window` samples of the current `event.sample` for
/// the most negative value, and updates `event.sample` to that index. Also
/// updates `event.amplitude` to reflect the new peak value.
///
/// This improves waveform extraction quality by centering the extraction
/// window on the true peak rather than the threshold crossing point.
///
/// # Type Parameters
///
/// * `C` - Number of channels
///
/// # Arguments
///
/// * `data` - Multi-channel data `&[[f64; C]]` (time x channels)
/// * `events` - Mutable slice of detected events
/// * `n_events` - Number of valid events in `events`
/// * `half_window` - Search radius in samples (e.g., 5 for +/- 5 samples)
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{align_to_peak, MultiChannelEvent};
///
/// let mut data = [[0.0f64; 2]; 20];
/// data[9][0] = -3.0;
/// data[10][0] = -8.0; // true peak, 1 sample after detection
/// data[11][0] = -2.0;
///
/// let mut events = [MultiChannelEvent { sample: 9, channel: 0, amplitude: 3.0 }];
/// align_to_peak::<2>(&data, &mut events, 1, 5);
/// assert_eq!(events[0].sample, 10);
/// assert!((events[0].amplitude - 8.0).abs() < 1e-12);
/// ```
pub fn align_to_peak<const C: usize>(
    data: &[[f64; C]],
    events: &mut [MultiChannelEvent],
    n_events: usize,
    half_window: usize,
) {
    let t_len = data.len();
    let n = if n_events < events.len() {
        n_events
    } else {
        events.len()
    };

    let mut i = 0;
    while i < n {
        let ch = events[i].channel;
        if ch >= C {
            i += 1;
            continue;
        }

        let center = events[i].sample;
        let start = center.saturating_sub(half_window);
        let end = if center + half_window + 1 < t_len {
            center + half_window + 1
        } else {
            t_len
        };

        let mut best_idx = center;
        let mut best_val = if center < t_len {
            data[center][ch]
        } else {
            0.0
        };

        let mut j = start;
        while j < end {
            if data[j][ch] < best_val {
                best_val = data[j][ch];
                best_idx = j;
            }
            j += 1;
        }

        events[i].sample = best_idx;
        events[i].amplitude = libm::fabs(best_val);
        i += 1;
    }
}

/// Extract multi-channel waveform snippets around detected events.
///
/// For each event, extracts a W-sample window from ALL C channels centered
/// on the event's sample time. Events too close to data edges are skipped.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `W` - Window length in samples
///
/// # Arguments
///
/// * `data` - Multi-channel data as `&[[f64; C]]` (time x channels)
/// * `events` - Detected multi-channel events
/// * `n_events` - Number of valid events in `events`
/// * `pre_samples` - Samples before the event's sample index
/// * `output` - Buffer for extracted waveforms, `[[[f64; W]; C]]`
///
/// # Returns
///
/// Number of waveforms successfully extracted.
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{extract_multichannel, MultiChannelEvent};
///
/// let mut data = [[0.0f64; 2]; 20];
/// data[10][0] = -5.0;
/// data[10][1] = -3.0;
///
/// let events = [MultiChannelEvent { sample: 10, channel: 0, amplitude: 5.0 }];
/// let mut output = [[[0.0f64; 4]; 2]; 4];
/// let n = extract_multichannel::<2, 4>(&data, &events, 1, 1, &mut output);
/// assert_eq!(n, 1);
/// assert!((output[0][0][1] - (-5.0)).abs() < 1e-12); // channel 0, sample at peak
/// assert!((output[0][1][1] - (-3.0)).abs() < 1e-12); // channel 1, same time
/// ```
pub fn extract_multichannel<const C: usize, const W: usize>(
    data: &[[f64; C]],
    events: &[MultiChannelEvent],
    n_events: usize,
    pre_samples: usize,
    output: &mut [[[f64; W]; C]],
) -> usize {
    let t_len = data.len();
    let n = if n_events < events.len() {
        n_events
    } else {
        events.len()
    };
    let mut count = 0;

    let mut i = 0;
    while i < n {
        if count >= output.len() {
            break;
        }
        let t = events[i].sample;
        if t < pre_samples {
            i += 1;
            continue;
        }
        let start = t - pre_samples;
        let end = start + W;
        if end > t_len {
            i += 1;
            continue;
        }

        // Copy all channels for this window
        let mut ch = 0;
        while ch < C {
            let mut w = 0;
            while w < W {
                output[count][ch][w] = data[start + w][ch];
                w += 1;
            }
            ch += 1;
        }
        count += 1;
        i += 1;
    }
    count
}

/// Extract waveforms from each event's peak channel only.
///
/// Simpler than full multi-channel extraction. Returns single-channel
/// waveforms suitable for PCA with the existing [`WaveformPca`].
///
/// # Type Parameters
///
/// * `C` - Number of channels in the data
/// * `W` - Window length in samples
///
/// # Arguments
///
/// * `data` - Multi-channel data as `&[[f64; C]]` (time x channels)
/// * `events` - Detected multi-channel events (each has a `.channel` field)
/// * `n_events` - Number of valid events in `events`
/// * `pre_samples` - Samples before the event's sample index
/// * `output` - Buffer for extracted single-channel waveforms
///
/// # Returns
///
/// Number of waveforms successfully extracted.
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{extract_peak_channel, MultiChannelEvent};
///
/// let mut data = [[0.0f64; 3]; 20];
/// data[8][1] = -7.0;
/// data[9][1] = -10.0;
/// data[10][1] = -6.0;
///
/// let events = [MultiChannelEvent { sample: 9, channel: 1, amplitude: 10.0 }];
/// let mut output = [[0.0f64; 4]; 4];
/// let n = extract_peak_channel::<3, 4>(&data, &events, 1, 1, &mut output);
/// assert_eq!(n, 1);
/// assert!((output[0][1] - (-10.0)).abs() < 1e-12); // peak at pre_samples offset
/// ```
pub fn extract_peak_channel<const C: usize, const W: usize>(
    data: &[[f64; C]],
    events: &[MultiChannelEvent],
    n_events: usize,
    pre_samples: usize,
    output: &mut [[f64; W]],
) -> usize {
    let t_len = data.len();
    let n = if n_events < events.len() {
        n_events
    } else {
        events.len()
    };
    let mut count = 0;

    let mut i = 0;
    while i < n {
        if count >= output.len() {
            break;
        }
        let t = events[i].sample;
        let ch = events[i].channel;
        if t < pre_samples || ch >= C {
            i += 1;
            continue;
        }
        let start = t - pre_samples;
        let end = start + W;
        if end > t_len {
            i += 1;
            continue;
        }

        let mut w = 0;
        while w < W {
            output[count][w] = data[start + w][ch];
            w += 1;
        }
        count += 1;
        i += 1;
    }
    count
}

/// Combine PCA features with spatial position features.
///
/// Appends scaled spatial coordinates to PCA feature vectors.
/// The spatial features are scaled by `spatial_weight` to balance
/// their contribution relative to PCA components.
///
/// # Type Parameters
/// * `K` - Number of PCA components
/// * `S` - Number of spatial features (typically 2 for x,y)
/// * `T` - Total features (must equal K + S)
///
/// # Panics
///
/// Panics at compile time if `T != K + S`.
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::combine_features;
///
/// let pca = [[1.0, 2.0], [-0.5, 0.3]];
/// let spatial = [[10.0, 50.0], [20.0, 60.0]];
/// let mut output = [[0.0f64; 4]; 2];
/// combine_features::<2, 2, 4>(&pca, &spatial, 0.5, &mut output, 2);
/// assert!((output[0][0] - 1.0).abs() < 1e-12);
/// assert!((output[0][1] - 2.0).abs() < 1e-12);
/// assert!((output[0][2] - 5.0).abs() < 1e-12);  // 10.0 * 0.5
/// assert!((output[0][3] - 25.0).abs() < 1e-12); // 50.0 * 0.5
/// ```
pub fn combine_features<const K: usize, const S: usize, const T: usize>(
    pca_features: &[[f64; K]],
    spatial_features: &[[f64; S]],
    spatial_weight: f64,
    output: &mut [[f64; T]],
    n: usize,
) {
    // Runtime assertion (const generics checked at monomorphization)
    assert!(T == K + S, "T must equal K + S");

    let mut i = 0;
    while i < n {
        // Copy PCA features
        let mut k = 0;
        while k < K {
            output[i][k] = pca_features[i][k];
            k += 1;
        }
        // Append scaled spatial features
        let mut s = 0;
        while s < S {
            output[i][K + s] = spatial_features[i][s] * spatial_weight;
            s += 1;
        }
        i += 1;
    }
}

/// Extract spatial position features for each spike from multi-channel waveforms.
///
/// For each detected event, extracts the peak amplitude on each channel
/// and computes center-of-mass position using the probe geometry.
///
/// Returns the number of events processed (min of `n_events` and `output.len()`).
///
/// # Type Parameters
/// * `C` - Number of channels
///
/// # Example
///
/// ```
/// use zerostone::spike_sort::{extract_spatial_features, MultiChannelEvent};
///
/// // 4-channel data, 20 time steps
/// let mut data = [[0.0f64; 4]; 20];
/// // Spike on channel 2 at sample 10
/// data[10][2] = -10.0;
/// data[10][1] = -3.0;
///
/// let positions = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
/// let events = [MultiChannelEvent { sample: 10, channel: 2, amplitude: 10.0 }];
/// let mut output = [[0.0f64; 2]; 4];
/// let n = extract_spatial_features::<4>(&data, &events, 1, &positions, &mut output);
/// assert_eq!(n, 1);
/// // Dominated by channel 2 (amplitude 10 at y=50) with some pull from ch1 (amplitude 3 at y=25)
/// assert!(output[0][1] > 40.0);
/// ```
pub fn extract_spatial_features<const C: usize>(
    data: &[[f64; C]],
    events: &[MultiChannelEvent],
    n_events: usize,
    positions: &[[f64; 2]; C],
    output: &mut [[f64; 2]],
) -> usize {
    let t_len = data.len();
    let max_out = if n_events < output.len() {
        n_events
    } else {
        output.len()
    };

    let mut count = 0;
    let mut i = 0;
    while i < max_out {
        let sample = events[i].sample;
        if sample >= t_len {
            i += 1;
            continue;
        }

        // Extract peak absolute amplitude on each channel in a small window
        // around the event sample (use a +-2 sample window for robustness)
        let mut amps = [0.0f64; C];
        let mut ch = 0;
        while ch < C {
            let start = sample.saturating_sub(2);
            let end = if sample + 3 <= t_len {
                sample + 3
            } else {
                t_len
            };
            let mut max_abs = 0.0;
            let mut t = start;
            while t < end {
                let a = libm::fabs(data[t][ch]);
                if a > max_abs {
                    max_abs = a;
                }
                t += 1;
            }
            amps[ch] = max_abs;
            ch += 1;
        }

        output[count] = crate::localize::center_of_mass(&amps, positions);
        count += 1;
        i += 1;
    }
    count
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(8)]
    fn detect_spikes_no_panic() {
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();
        let threshold: f64 = kani::any();
        let refractory: usize = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);
        kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 1e6);
        kani::assume(refractory >= 1 && refractory <= 4);

        let data = [d0, d1, d2, d3];
        let mut spike_times = [0usize; 4];
        let count = detect_spikes(&data, threshold, refractory, &mut spike_times);
        assert!(count <= 4, "count must not exceed buffer length");
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn mad_noise_finite() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let c: f64 = kani::any();

        kani::assume(a.is_finite() && a >= -1e6 && a <= 1e6);
        kani::assume(b.is_finite() && b >= -1e6 && b <= 1e6);
        kani::assume(c.is_finite() && c >= -1e6 && c <= 1e6);

        let data = [a, b, c];
        let mut scratch = [0.0f64; 3];
        let result = estimate_noise_mad(&data, &mut scratch);
        assert!(result.is_finite(), "MAD result must be finite");
    }

    /// Prove that `deduplicate_events` never panics and output <= input.
    #[kani::proof]
    #[kani::unwind(6)]
    fn dedup_no_panic_and_count_invariant() {
        let probe = ProbeLayout::<2>::linear(25.0);

        let s0: usize = kani::any();
        let s1: usize = kani::any();
        let c0: usize = kani::any();
        let c1: usize = kani::any();
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        let tr: usize = kani::any();

        kani::assume(s0 <= 100 && s1 <= 100);
        kani::assume(c0 < 2 && c1 < 2);
        kani::assume(a0.is_finite() && a0 >= 0.0 && a0 <= 1e6);
        kani::assume(a1.is_finite() && a1 >= 0.0 && a1 <= 1e6);
        kani::assume(tr >= 1 && tr <= 10);

        let mut events = [
            MultiChannelEvent {
                sample: s0,
                channel: c0,
                amplitude: a0,
            },
            MultiChannelEvent {
                sample: s1,
                channel: c1,
                amplitude: a1,
            },
        ];
        let result = deduplicate_events::<2>(&mut events, 2, &probe, 30.0, tr);
        assert!(result <= 2, "dedup output must be <= input count");
    }

    /// Prove that `detect_spikes_multichannel` returns valid channel indices.
    #[kani::proof]
    #[kani::unwind(8)]
    fn multichannel_detect_valid_channels() {
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);

        let data = [[d0, d1], [d2, d3]];
        let noise = [1.0, 1.0];
        let mut output = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }; 4];

        let n = detect_spikes_multichannel::<2>(&data, 3.0, &noise, 1, &mut output);
        assert!(n <= 4, "count must not exceed buffer length");
        let mut i = 0;
        while i < n {
            assert!(output[i].channel < 2, "channel index must be < C");
            assert!(output[i].sample < 2, "sample index must be < data length");
            i += 1;
        }
    }

    /// Prove that `align_to_peak` never panics for valid channel indices.
    #[kani::proof]
    #[kani::unwind(8)]
    fn align_to_peak_no_panic() {
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);

        let data = [[d0, d1], [d2, d3]];
        let hw: usize = kani::any();
        kani::assume(hw <= 1);

        let mut events = [MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 5.0,
        }];

        align_to_peak::<2>(&data, &mut events, 1, hw);
        assert!(events[0].sample < 2, "aligned sample must be in range");
    }

    /// Prove that `extract_peak_channel` output count never exceeds buffer.
    #[kani::proof]
    #[kani::unwind(8)]
    fn extract_peak_channel_bounded() {
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();
        let d4: f64 = kani::any();
        let d5: f64 = kani::any();
        let d6: f64 = kani::any();
        let d7: f64 = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);
        kani::assume(d4.is_finite() && d4 >= -1e6 && d4 <= 1e6);
        kani::assume(d5.is_finite() && d5 >= -1e6 && d5 <= 1e6);
        kani::assume(d6.is_finite() && d6 >= -1e6 && d6 <= 1e6);
        kani::assume(d7.is_finite() && d7 >= -1e6 && d7 <= 1e6);

        let data = [[d0, d1], [d2, d3], [d4, d5], [d6, d7]];
        let pre: usize = kani::any();
        kani::assume(pre <= 2);

        let events = [MultiChannelEvent {
            sample: 1,
            channel: 0,
            amplitude: 5.0,
        }];
        let mut output = [[0.0f64; 4]; 2];
        let n = extract_peak_channel::<2, 4>(&data, &events, 1, pre, &mut output);
        assert!(n <= 2, "extracted count must not exceed output buffer");
    }

    /// Prove that `compute_adaptive_thresholds` never panics and
    /// returns finite thresholds >= min_threshold.
    #[kani::proof]
    #[kani::unwind(10)]
    fn adaptive_threshold_no_panic() {
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();
        let d2: f64 = kani::any();
        let d3: f64 = kani::any();
        let d4: f64 = kani::any();
        let d5: f64 = kani::any();
        let d6: f64 = kani::any();
        let d7: f64 = kani::any();

        kani::assume(d0.is_finite() && d0 >= -1e6 && d0 <= 1e6);
        kani::assume(d1.is_finite() && d1 >= -1e6 && d1 <= 1e6);
        kani::assume(d2.is_finite() && d2 >= -1e6 && d2 <= 1e6);
        kani::assume(d3.is_finite() && d3 >= -1e6 && d3 <= 1e6);
        kani::assume(d4.is_finite() && d4 >= -1e6 && d4 <= 1e6);
        kani::assume(d5.is_finite() && d5 >= -1e6 && d5 <= 1e6);
        kani::assume(d6.is_finite() && d6 >= -1e6 && d6 <= 1e6);
        kani::assume(d7.is_finite() && d7 >= -1e6 && d7 <= 1e6);

        let data = [[d0, d1], [d2, d3], [d4, d5], [d6, d7]];
        let mut scratch = [0.0f64; 4];
        let result =
            compute_adaptive_thresholds::<2>(&data, 4.0, 0.5, 100.0, 30000.0, &mut scratch);
        let mut ch = 0;
        while ch < 2 {
            kani::assert(result[ch].is_finite(), "threshold must be finite");
            kani::assert(result[ch] >= 0.5, "threshold must be >= min_threshold");
            ch += 1;
        }
    }

    /// Prove that `combine_features` never panics for bounded finite inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn combine_features_no_panic() {
        let p0: f64 = kani::any();
        let p1: f64 = kani::any();
        let s0: f64 = kani::any();
        let s1: f64 = kani::any();
        let w: f64 = kani::any();

        kani::assume(p0.is_finite() && p0 >= -1e6 && p0 <= 1e6);
        kani::assume(p1.is_finite() && p1 >= -1e6 && p1 <= 1e6);
        kani::assume(s0.is_finite() && s0 >= -1e6 && s0 <= 1e6);
        kani::assume(s1.is_finite() && s1 >= -1e6 && s1 <= 1e6);
        kani::assume(w.is_finite() && w >= 0.0 && w <= 10.0);

        let pca = [[p0, p1]];
        let spatial = [[s0, s1]];
        let mut output = [[0.0f64; 4]; 1];
        combine_features::<2, 2, 4>(&pca, &spatial, w, &mut output, 1);

        // All outputs must be finite
        let mut i = 0;
        while i < 4 {
            assert!(output[0][i].is_finite(), "output must be finite");
            i += 1;
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    // Simple pseudo-RNG for tests (xorshift64)
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

    /// Generate a synthetic spike waveform: a negative gaussian peak
    fn make_spike(amplitude: f64, peak_sample: usize, width: f64, window: usize) -> Vec<f64> {
        (0..window)
            .map(|i| {
                let x = (i as f64 - peak_sample as f64) / width;
                -amplitude * libm::exp(-0.5 * x * x)
            })
            .collect()
    }

    #[test]
    fn test_extract_known_positions() {
        let ext = WaveformExtractor::<1, 8>::new(); // pre_samples = 2
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let spike_times = [10, 50, 90];
        let mut output = [[0.0f64; 8]; 3];
        let count = ext.extract(&data, &spike_times, &mut output);
        assert_eq!(count, 3);
        assert_eq!(output[0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        assert_eq!(output[1], [48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]);
    }

    #[test]
    fn test_extract_edge_cases() {
        let ext = WaveformExtractor::<1, 8>::new();
        let data = vec![0.0; 20];
        let spike_times = [0, 1, 15, 19];
        let mut output = [[0.0f64; 8]; 4];
        let count = ext.extract(&data, &spike_times, &mut output);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_peak_alignment() {
        let ext = WaveformExtractor::<1, 16>::with_pre_samples(4);
        let mut waveforms = [[1.0f64; 16]; 1];
        waveforms[0][6] = -5.0; // minimum at index 6

        ext.align_to_peak(&mut waveforms, 1, 0);

        // After alignment, minimum should be at index 4
        let min_idx = waveforms[0]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(min_idx, 4, "Peak should be aligned to pre_samples index");
    }

    #[test]
    fn test_pca_fit_transform_roundtrip() {
        let mut rng = Rng::new(42);
        let mut pca = WaveformPca::<8, 3, 64>::new();

        let mut waveforms = Vec::new();
        for _ in 0..50 {
            let mut wf = [0.0f64; 8];
            let a = rng.gaussian(0.0, 5.0);
            let b = rng.gaussian(0.0, 3.0);
            let c = rng.gaussian(0.0, 1.0);
            wf[0] = a;
            wf[1] = a + rng.gaussian(0.0, 0.1);
            wf[2] = b;
            wf[3] = b + rng.gaussian(0.0, 0.1);
            wf[4] = c;
            wf[5] = c + rng.gaussian(0.0, 0.1);
            wf[6] = rng.gaussian(0.0, 0.01);
            wf[7] = rng.gaussian(0.0, 0.01);
            waveforms.push(wf);
        }

        pca.fit(&waveforms).unwrap();

        let mut output = [0.0f64; 3];
        pca.transform(&waveforms[0], &mut output).unwrap();
        let energy: f64 = output.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "PCA projection should be non-zero");
    }

    #[test]
    fn test_pca_explained_variance() {
        let mut rng = Rng::new(123);
        let mut pca = WaveformPca::<8, 3, 64>::new();

        let mut waveforms = Vec::new();
        for _ in 0..100 {
            let a = rng.gaussian(0.0, 10.0);
            let b = rng.gaussian(0.0, 5.0);
            let c = rng.gaussian(0.0, 2.0);
            let wf = [
                a,
                b,
                c,
                rng.gaussian(0.0, 0.01),
                rng.gaussian(0.0, 0.01),
                rng.gaussian(0.0, 0.01),
                rng.gaussian(0.0, 0.01),
                rng.gaussian(0.0, 0.01),
            ];
            waveforms.push(wf);
        }

        pca.fit(&waveforms).unwrap();
        let ratios = pca.explained_variance_ratio();
        let total: f64 = ratios.iter().sum();
        assert!(
            total > 0.90,
            "3 components should explain >90% variance, got {}",
            total
        );
    }

    #[test]
    fn test_pca_not_fitted() {
        let pca = WaveformPca::<8, 3, 64>::new();
        let wf = [0.0f64; 8];
        let mut out = [0.0f64; 3];
        assert_eq!(pca.transform(&wf, &mut out), Err(SortError::NotFitted));
        assert!(!pca.is_fitted());
    }

    #[test]
    fn test_pca_insufficient_data() {
        let mut pca = WaveformPca::<8, 3, 64>::new();
        let waveforms = [[0.0f64; 8]; 1];
        assert_eq!(pca.fit(&waveforms), Err(SortError::InsufficientData));
    }

    #[test]
    fn test_template_euclidean_match() {
        let mut tm = TemplateMatch::<1, 8, 4>::new();

        let t1 = [1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 0.5];
        let t2 = [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0];
        tm.add_template(&t1).unwrap();
        tm.add_template(&t2).unwrap();

        let wf = [1.1, 0.1, -0.9, -1.9, -0.9, 0.1, 1.1, 0.6];
        let (idx, dist) = tm.match_waveform(&wf).unwrap();
        assert_eq!(idx, 0, "Should match template 0");
        assert!(dist < 1.0, "Distance should be small");
    }

    #[test]
    fn test_template_ncc_match() {
        let mut tm = TemplateMatch::<1, 8, 4>::new();

        let t1 = [1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 0.5];
        let t2 = [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0];
        tm.add_template(&t1).unwrap();
        tm.add_template(&t2).unwrap();

        let wf: [f64; 8] = [2.0, 0.0, -2.0, -4.0, -2.0, 0.0, 2.0, 1.0];
        let (idx, ncc) = tm.match_waveform_ncc(&wf).unwrap();
        assert_eq!(idx, 0, "Scaled t1 should match template 0 via NCC");
        assert!(ncc > 0.99, "NCC should be ~1.0, got {}", ncc);
    }

    #[test]
    fn test_template_running_average() {
        let mut tm = TemplateMatch::<1, 4, 4>::new();
        let t = [1.0, 2.0, 3.0, 4.0];
        tm.add_template(&t).unwrap();

        // Update with same values shouldn't change template
        tm.update_template(0, &t);
        let tmpl = tm.template(0).unwrap();
        for (tv, &expected) in tmpl.iter().zip(t.iter()) {
            assert!(
                libm::fabs(*tv - expected) < 1e-10,
                "Template should be unchanged after update with same values"
            );
        }

        // Update with different values should shift template
        let new_wf = [5.0, 6.0, 7.0, 8.0];
        tm.update_template(0, &new_wf);
        let tmpl = tm.template(0).unwrap();
        for (tv, &orig) in tmpl.iter().zip(t.iter()) {
            assert!(*tv > orig, "Template should shift toward new values");
        }
    }

    #[test]
    fn test_template_full() {
        let mut tm = TemplateMatch::<1, 4, 2>::new();
        tm.add_template(&[1.0; 4]).unwrap();
        tm.add_template(&[2.0; 4]).unwrap();
        assert_eq!(tm.add_template(&[3.0; 4]), Err(SortError::TemplateFull));
    }

    #[test]
    fn test_mad_estimation() {
        let mut rng = Rng::new(42);
        let n = 10000;
        let true_std = 3.0;
        let mut data: Vec<f64> = (0..n).map(|_| rng.gaussian(0.0, true_std)).collect();
        let mut scratch = vec![0.0f64; n];
        let estimated = estimate_noise_mad(&data, &mut scratch);
        let ratio = estimated / true_std;
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "MAD estimate {:.3} should be within 10% of true std {:.3}, ratio={:.3}",
            estimated,
            true_std,
            ratio
        );
        // Silence unused mut warning -- data is built via collect
        data.push(0.0);
    }

    #[test]
    fn test_detect_spikes_refractory() {
        let mut data = vec![0.0f64; 100];
        data[20] = -5.0;
        data[21] = -3.0;
        data[50] = -4.0;
        data[55] = -6.0;

        let mut spike_times = [0usize; 10];
        let count = detect_spikes(&data, 2.0, 10, &mut spike_times);

        assert_eq!(count, 2, "Should detect 2 spikes with refractory period");
        assert_eq!(spike_times[0], 20);
        assert_eq!(spike_times[1], 55);
    }

    #[test]
    fn test_end_to_end_synthetic() {
        let mut rng = Rng::new(99);
        let n_samples = 10000;
        let noise_std = 1.0;

        let mut data: Vec<f64> = (0..n_samples)
            .map(|_| rng.gaussian(0.0, noise_std))
            .collect();

        let spike_wf1 = make_spike(8.0, 4, 2.0, 8);
        let spike_wf2 = make_spike(6.0, 4, 3.0, 8);
        let spike_wf3 = make_spike(10.0, 4, 1.5, 8);

        let positions1 = [200, 700, 1200, 1700, 2200];
        let positions2 = [400, 900, 1400, 1900, 2400];
        let positions3 = [600, 1100, 1600, 2100, 2600];

        for &p in &positions1 {
            for (j, &v) in spike_wf1.iter().enumerate() {
                data[p - 4 + j] += v;
            }
        }
        for &p in &positions2 {
            for (j, &v) in spike_wf2.iter().enumerate() {
                data[p - 4 + j] += v;
            }
        }
        for &p in &positions3 {
            for (j, &v) in spike_wf3.iter().enumerate() {
                data[p - 4 + j] += v;
            }
        }

        // Step 1: Estimate noise
        let mut scratch = vec![0.0; n_samples];
        let noise_est = estimate_noise_mad(&data, &mut scratch);
        assert!(
            noise_est > 0.5 && noise_est < 2.0,
            "Noise estimate should be ~1.0, got {}",
            noise_est
        );

        // Step 2: Detect spikes
        let threshold = 4.0 * noise_est;
        let mut spike_times = [0usize; 100];
        let n_spikes = detect_spikes(&data, threshold, 20, &mut spike_times);
        assert!(
            n_spikes >= 10,
            "Should detect at least 10 spikes, got {}",
            n_spikes
        );

        // Step 3: Extract waveforms
        let ext = WaveformExtractor::<1, 8>::new();
        let mut waveforms = vec![[0.0f64; 8]; n_spikes];
        let n_extracted = ext.extract(&data, &spike_times[..n_spikes], &mut waveforms);
        assert!(n_extracted > 0, "Should extract some waveforms");

        // Step 4: PCA
        let mut pca = WaveformPca::<8, 3, 64>::new();
        pca.fit(&waveforms[..n_extracted]).unwrap();

        for wf in waveforms.iter().take(n_extracted) {
            let mut f = [0.0f64; 3];
            pca.transform(wf, &mut f).unwrap();
        }

        // Step 5: Template matching
        let mut tm = TemplateMatch::<1, 8, 4>::new();
        tm.add_template(&waveforms[0]).unwrap();

        let (idx, _dist) = tm.match_waveform(&waveforms[0]).unwrap();
        assert_eq!(idx, 0, "Waveform should match its own template");
    }

    // ---- Multi-channel detection tests ----

    fn make_empty_event() -> MultiChannelEvent {
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        }
    }

    #[test]
    fn test_multichannel_detect_known_locations() {
        // 4 channels, 100 time steps
        let mut data = [[0.0f64; 4]; 100];
        // Place spikes at known locations on different channels
        data[20][0] = -6.0;
        data[50][1] = -8.0;
        data[80][2] = -5.0;

        let noise = [1.0, 1.0, 1.0, 1.0];
        let mut events = [make_empty_event(); 16];
        let n = detect_spikes_multichannel::<4>(&data, 4.0, &noise, 10, &mut events);

        assert_eq!(n, 3, "Should detect 3 spikes, got {}", n);
        // Events should be sorted by sample time
        assert_eq!(events[0].sample, 20);
        assert_eq!(events[0].channel, 0);
        assert_eq!(events[1].sample, 50);
        assert_eq!(events[1].channel, 1);
        assert_eq!(events[2].sample, 80);
        assert_eq!(events[2].channel, 2);
    }

    #[test]
    fn test_multichannel_detect_no_spikes() {
        // All noise below threshold
        let data = [[0.5f64; 2]; 50];
        let noise = [1.0, 1.0];
        let mut events = [make_empty_event(); 8];
        let n = detect_spikes_multichannel::<2>(&data, 4.0, &noise, 5, &mut events);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_multichannel_detect_single_channel() {
        let mut data = [[0.0f64; 1]; 50];
        data[10][0] = -7.0;
        data[30][0] = -5.0;

        let noise = [1.0];
        let mut events = [make_empty_event(); 8];
        let n = detect_spikes_multichannel::<1>(&data, 4.0, &noise, 5, &mut events);

        assert_eq!(n, 2);
        assert_eq!(events[0].sample, 10);
        assert_eq!(events[1].sample, 30);
    }

    #[test]
    fn test_multichannel_detect_simultaneous_all_channels() {
        // Same spike appears on all channels at the same time
        let mut data = [[0.0f64; 4]; 50];
        for ch in 0..4 {
            data[25][ch] = -10.0;
        }

        let noise = [1.0; 4];
        let mut events = [make_empty_event(); 16];
        let n = detect_spikes_multichannel::<4>(&data, 4.0, &noise, 5, &mut events);

        assert_eq!(n, 4, "Should detect on all 4 channels");
        // All at sample 25
        for e in events.iter().take(n) {
            assert_eq!(e.sample, 25);
        }
    }

    #[test]
    fn test_multichannel_detect_amplitude_correct() {
        let mut data = [[0.0f64; 2]; 20];
        data[10][0] = -7.5;

        let noise = [1.0, 1.0];
        let mut events = [make_empty_event(); 4];
        let n = detect_spikes_multichannel::<2>(&data, 4.0, &noise, 5, &mut events);

        assert_eq!(n, 1);
        assert!((events[0].amplitude - 7.5).abs() < 1e-12);
    }

    #[test]
    fn test_multichannel_detect_empty_data() {
        let data: &[[f64; 2]] = &[];
        let noise = [1.0, 1.0];
        let mut events = [make_empty_event(); 4];
        let n = detect_spikes_multichannel::<2>(data, 4.0, &noise, 5, &mut events);
        assert_eq!(n, 0);
    }

    // ---- Deduplication tests ----

    #[test]
    fn test_dedup_removes_neighbor_duplicate() {
        let probe = ProbeLayout::<4>::linear(25.0);

        // Two events at similar times on adjacent channels
        let mut events = [
            MultiChannelEvent {
                sample: 100,
                channel: 1,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 102,
                channel: 2,
                amplitude: 7.0,
            }, // larger, should win
        ];

        let n = deduplicate_events::<4>(&mut events, 2, &probe, 30.0, 5);
        assert_eq!(n, 1, "Should keep only the larger event");
        assert_eq!(events[0].channel, 2);
        assert!((events[0].amplitude - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_dedup_keeps_distant_events() {
        let probe = ProbeLayout::<4>::linear(25.0);

        // Two events far apart in time
        let mut events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 500,
                channel: 1,
                amplitude: 6.0,
            },
        ];

        let n = deduplicate_events::<4>(&mut events, 2, &probe, 30.0, 5);
        assert_eq!(n, 2, "Temporally distant events should both be kept");
    }

    #[test]
    fn test_dedup_keeps_spatially_distant() {
        let probe = ProbeLayout::<8>::linear(50.0);

        // Two events at similar times but on channels far apart
        let mut events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 101,
                channel: 7,
                amplitude: 6.0,
            }, // 350 um away
        ];

        let n = deduplicate_events::<8>(&mut events, 2, &probe, 60.0, 5);
        assert_eq!(
            n, 2,
            "Spatially distant events should both be kept (dist = 350 um)"
        );
    }

    #[test]
    fn test_dedup_no_events() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events = [make_empty_event(); 4];
        let n = deduplicate_events::<4>(&mut events, 0, &probe, 30.0, 5);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_dedup_single_event() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events = [MultiChannelEvent {
            sample: 50,
            channel: 0,
            amplitude: 4.0,
        }];
        let n = deduplicate_events::<4>(&mut events, 1, &probe, 30.0, 5);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_dedup_three_events_cluster() {
        // Three channels pick up the same spike
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events = [
            MultiChannelEvent {
                sample: 100,
                channel: 0,
                amplitude: 3.0,
            },
            MultiChannelEvent {
                sample: 101,
                channel: 1,
                amplitude: 8.0,
            }, // winner
            MultiChannelEvent {
                sample: 102,
                channel: 2,
                amplitude: 5.0,
            },
            make_empty_event(),
        ];

        let n = deduplicate_events::<4>(&mut events, 3, &probe, 30.0, 5);
        assert_eq!(n, 1, "Only the largest amplitude event should survive");
        assert_eq!(events[0].channel, 1);
        assert!((events[0].amplitude - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_dedup_preserves_order() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events = [
            MultiChannelEvent {
                sample: 50,
                channel: 0,
                amplitude: 5.0,
            },
            MultiChannelEvent {
                sample: 51,
                channel: 1,
                amplitude: 3.0,
            }, // dup of first
            MultiChannelEvent {
                sample: 200,
                channel: 2,
                amplitude: 7.0,
            },
            MultiChannelEvent {
                sample: 201,
                channel: 3,
                amplitude: 4.0,
            }, // dup of third
        ];

        let n = deduplicate_events::<4>(&mut events, 4, &probe, 30.0, 5);
        assert_eq!(n, 2);
        assert_eq!(events[0].sample, 50);
        assert_eq!(events[1].sample, 200);
    }

    // ---- Peak alignment tests ----

    #[test]
    fn test_align_to_peak_improves_timing() {
        // The spike's true peak is at sample 12 but was detected at 10
        let mut data = [[0.0f64; 2]; 30];
        data[10][0] = -5.0;
        data[11][0] = -7.0;
        data[12][0] = -9.0; // true peak
        data[13][0] = -6.0;

        let mut events = [MultiChannelEvent {
            sample: 10,
            channel: 0,
            amplitude: 5.0,
        }];

        align_to_peak::<2>(&data, &mut events, 1, 5);
        assert_eq!(events[0].sample, 12, "Should align to true peak at 12");
        assert!((events[0].amplitude - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_align_to_peak_already_aligned() {
        let mut data = [[0.0f64; 1]; 20];
        data[10][0] = -8.0; // already the peak

        let mut events = [MultiChannelEvent {
            sample: 10,
            channel: 0,
            amplitude: 8.0,
        }];

        align_to_peak::<1>(&data, &mut events, 1, 5);
        assert_eq!(events[0].sample, 10, "Already aligned, should not move");
    }

    #[test]
    fn test_align_to_peak_boundary_start() {
        // Event near the start of data
        let mut data = [[0.0f64; 1]; 20];
        data[0][0] = -10.0;
        data[2][0] = -3.0;

        let mut events = [MultiChannelEvent {
            sample: 2,
            channel: 0,
            amplitude: 3.0,
        }];

        align_to_peak::<1>(&data, &mut events, 1, 5);
        assert_eq!(events[0].sample, 0, "Should find peak at boundary");
        assert!((events[0].amplitude - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_align_to_peak_boundary_end() {
        // Event near the end of data
        let mut data = [[0.0f64; 1]; 20];
        data[18][0] = -4.0;
        data[19][0] = -9.0; // peak at last sample

        let mut events = [MultiChannelEvent {
            sample: 18,
            channel: 0,
            amplitude: 4.0,
        }];

        align_to_peak::<1>(&data, &mut events, 1, 5);
        assert_eq!(events[0].sample, 19);
        assert!((events[0].amplitude - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_align_no_events() {
        let data = [[0.0f64; 2]; 20];
        let mut events = [make_empty_event(); 4];
        align_to_peak::<2>(&data, &mut events, 0, 5);
        // Should be a no-op; no panic
    }

    #[test]
    fn test_align_multichannel_correct_channel() {
        // Spikes on different channels should align to peaks on their own channel
        let mut data = [[0.0f64; 2]; 30];
        data[10][0] = -3.0;
        data[12][0] = -8.0; // true peak ch0
        data[10][1] = -2.0;
        data[11][1] = -6.0; // true peak ch1

        let mut events = [
            MultiChannelEvent {
                sample: 10,
                channel: 0,
                amplitude: 3.0,
            },
            MultiChannelEvent {
                sample: 10,
                channel: 1,
                amplitude: 2.0,
            },
        ];

        align_to_peak::<2>(&data, &mut events, 2, 5);
        assert_eq!(events[0].sample, 12, "Ch0 peak at 12");
        assert_eq!(events[1].sample, 11, "Ch1 peak at 11");
    }

    // ---- End-to-end multi-channel test ----

    #[test]
    fn test_multichannel_end_to_end() {
        use crate::probe::ProbeLayout;

        // 4-channel linear probe, 25 um pitch
        let probe = ProbeLayout::<4>::linear(25.0);

        // Generate 1000-sample recording with noise and 2 distinct spikes
        let mut rng = Rng::new(77);
        let n_samples = 1000;
        let mut data = vec![[0.0f64; 4]; n_samples];

        // Add noise
        for t in 0..n_samples {
            for ch in 0..4 {
                data[t][ch] = rng.gaussian(0.0, 1.0);
            }
        }

        // Spike 1 at t=200, largest on channel 1, visible on channels 0,2
        data[200][0] += -6.0;
        data[200][1] += -10.0; // peak channel
        data[200][2] += -5.0;

        // Spike 2 at t=600, largest on channel 3, visible on channel 2
        data[600][2] += -4.5;
        data[600][3] += -9.0; // peak channel

        // Step 1: estimate per-channel noise
        let noise = [1.0f64; 4]; // We know it's unit gaussian

        // Step 2: detect multi-channel
        let mut events = [make_empty_event(); 64];
        let n_det = detect_spikes_multichannel::<4>(&data, 4.0, &noise, 10, &mut events);
        assert!(
            n_det >= 4,
            "Should detect spike on at least 4 channel-instances, got {}",
            n_det
        );

        // Step 3: align to peak
        align_to_peak::<4>(&data, &mut events, n_det, 3);

        // Step 4: deduplicate
        let n_dedup = deduplicate_events::<4>(&mut events, n_det, &probe, 30.0, 5);

        // After dedup, we should have roughly 2 events (one per true spike)
        // plus possibly some noise-triggered events
        assert!(
            n_dedup >= 2,
            "Should retain at least 2 events after dedup, got {}",
            n_dedup
        );
        assert!(
            n_dedup < n_det,
            "Dedup should remove some events: {} -> {}",
            n_det,
            n_dedup
        );

        // Verify the two main spikes are present (within alignment window)
        let spike1_found = events[..n_dedup]
            .iter()
            .any(|e| e.sample >= 197 && e.sample <= 203);
        let spike2_found = events[..n_dedup]
            .iter()
            .any(|e| e.sample >= 597 && e.sample <= 603);
        assert!(spike1_found, "Spike 1 near t=200 should be detected");
        assert!(spike2_found, "Spike 2 near t=600 should be detected");
    }

    // =========================================================================
    // Multi-channel waveform extraction tests
    // =========================================================================

    #[test]
    fn test_extract_multichannel_basic() {
        // 2-channel data, known values
        let mut data = [[0.0f64; 2]; 20];
        for t in 0..20 {
            data[t][0] = t as f64;
            data[t][1] = -(t as f64);
        }

        let events = [MultiChannelEvent {
            sample: 10,
            channel: 0,
            amplitude: 10.0,
        }];
        let mut output = [[[0.0f64; 4]; 2]; 4];
        let n = extract_multichannel::<2, 4>(&data, &events, 1, 1, &mut output);
        assert_eq!(n, 1);
        // Window starts at 10-1=9, ends at 9+4=13
        assert!((output[0][0][0] - 9.0).abs() < 1e-12);
        assert!((output[0][0][1] - 10.0).abs() < 1e-12);
        assert!((output[0][0][2] - 11.0).abs() < 1e-12);
        assert!((output[0][0][3] - 12.0).abs() < 1e-12);
        // Channel 1 should be negative
        assert!((output[0][1][0] - (-9.0)).abs() < 1e-12);
        assert!((output[0][1][1] - (-10.0)).abs() < 1e-12);
    }

    #[test]
    fn test_extract_multichannel_boundary_skip() {
        let data = [[1.0f64; 2]; 20];
        let events = [
            MultiChannelEvent {
                sample: 0,
                channel: 0,
                amplitude: 1.0,
            }, // too close to start (pre=2, 0 < 2)
            MultiChannelEvent {
                sample: 18,
                channel: 0,
                amplitude: 1.0,
            }, // too close to end (18-2+8=24 > 20)
            MultiChannelEvent {
                sample: 10,
                channel: 0,
                amplitude: 1.0,
            }, // valid (10-2=8, 8+8=16 <= 20)
        ];
        let mut output = [[[0.0f64; 8]; 2]; 4];
        let n = extract_multichannel::<2, 8>(&data, &events, 3, 2, &mut output);
        assert_eq!(n, 1, "Only middle event should be extractable");
    }

    #[test]
    fn test_extract_peak_channel_basic() {
        let mut data = [[0.0f64; 3]; 20];
        // Put a spike on channel 1
        data[8][1] = -7.0;
        data[9][1] = -10.0;
        data[10][1] = -6.0;
        // Channel 0 and 2 have different values
        data[9][0] = 1.0;
        data[9][2] = 2.0;

        let events = [MultiChannelEvent {
            sample: 9,
            channel: 1,
            amplitude: 10.0,
        }];
        let mut output = [[0.0f64; 4]; 4];
        let n = extract_peak_channel::<3, 4>(&data, &events, 1, 1, &mut output);
        assert_eq!(n, 1);
        // Window: [8..12] on channel 1
        assert!((output[0][0] - (-7.0)).abs() < 1e-12);
        assert!((output[0][1] - (-10.0)).abs() < 1e-12);
        assert!((output[0][2] - (-6.0)).abs() < 1e-12);
        assert!((output[0][3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_extract_peak_channel_boundary_skip() {
        let data = [[0.0f64; 2]; 10];
        let events = [
            MultiChannelEvent {
                sample: 1,
                channel: 0,
                amplitude: 1.0,
            }, // pre=2, 1 < 2, skip
            MultiChannelEvent {
                sample: 5,
                channel: 0,
                amplitude: 1.0,
            }, // valid: 5-2=3, 3+8=11 > 10, skip for W=8
        ];
        let mut output = [[0.0f64; 8]; 4];
        let n = extract_peak_channel::<2, 8>(&data, &events, 2, 2, &mut output);
        assert_eq!(n, 0, "Both events should be out of bounds");
    }

    #[test]
    fn test_extract_multichannel_vs_peak_consistency() {
        // Extract both ways and verify peak channel extraction matches
        let mut data = [[0.0f64; 2]; 20];
        for t in 0..20 {
            data[t][0] = t as f64;
            data[t][1] = 100.0 + t as f64;
        }

        let events = [MultiChannelEvent {
            sample: 10,
            channel: 1,
            amplitude: 110.0,
        }];
        let mut mc_output = [[[0.0f64; 4]; 2]; 2];
        let mut pc_output = [[0.0f64; 4]; 2];
        let n_mc = extract_multichannel::<2, 4>(&data, &events, 1, 1, &mut mc_output);
        let n_pc = extract_peak_channel::<2, 4>(&data, &events, 1, 1, &mut pc_output);
        assert_eq!(n_mc, 1);
        assert_eq!(n_pc, 1);

        // Peak channel (1) from multichannel should match peak_channel output
        for w in 0..4 {
            assert!(
                (mc_output[0][1][w] - pc_output[0][w]).abs() < 1e-12,
                "Mismatch at sample {}: mc={}, pc={}",
                w,
                mc_output[0][1][w],
                pc_output[0][w]
            );
        }
    }

    #[test]
    fn test_extract_empty_events() {
        let data = [[0.0f64; 2]; 20];
        let mut mc_output = [[[0.0f64; 4]; 2]; 4];
        let mut pc_output = [[0.0f64; 4]; 4];
        let n_mc = extract_multichannel::<2, 4>(&data, &[], 0, 1, &mut mc_output);
        let n_pc = extract_peak_channel::<2, 4>(&data, &[], 0, 1, &mut pc_output);
        assert_eq!(n_mc, 0);
        assert_eq!(n_pc, 0);
    }

    // =========================================================================
    // Feature combination tests
    // =========================================================================

    #[test]
    fn test_combine_features_basic() {
        let pca = [[1.0, 2.0], [-0.5, 0.3]];
        let spatial = [[10.0, 50.0], [20.0, 60.0]];
        let mut output = [[0.0f64; 4]; 2];
        combine_features::<2, 2, 4>(&pca, &spatial, 0.5, &mut output, 2);

        // First spike
        assert!((output[0][0] - 1.0).abs() < 1e-12);
        assert!((output[0][1] - 2.0).abs() < 1e-12);
        assert!((output[0][2] - 5.0).abs() < 1e-12); // 10.0 * 0.5
        assert!((output[0][3] - 25.0).abs() < 1e-12); // 50.0 * 0.5

        // Second spike
        assert!((output[1][0] - (-0.5)).abs() < 1e-12);
        assert!((output[1][1] - 0.3).abs() < 1e-12);
        assert!((output[1][2] - 10.0).abs() < 1e-12); // 20.0 * 0.5
        assert!((output[1][3] - 30.0).abs() < 1e-12); // 60.0 * 0.5
    }

    #[test]
    fn test_combine_features_zero_weight() {
        let pca = [[3.0, 4.0]];
        let spatial = [[100.0, 200.0]];
        let mut output = [[0.0f64; 4]; 1];
        combine_features::<2, 2, 4>(&pca, &spatial, 0.0, &mut output, 1);

        assert!((output[0][0] - 3.0).abs() < 1e-12);
        assert!((output[0][1] - 4.0).abs() < 1e-12);
        assert!((output[0][2]).abs() < 1e-12); // 100.0 * 0.0
        assert!((output[0][3]).abs() < 1e-12); // 200.0 * 0.0
    }

    #[test]
    fn test_combine_features_unit_weight() {
        let pca = [[1.0]];
        let spatial = [[5.0, 10.0, 15.0]];
        let mut output = [[0.0f64; 4]; 1];
        combine_features::<1, 3, 4>(&pca, &spatial, 1.0, &mut output, 1);

        assert!((output[0][0] - 1.0).abs() < 1e-12);
        assert!((output[0][1] - 5.0).abs() < 1e-12);
        assert!((output[0][2] - 10.0).abs() < 1e-12);
        assert!((output[0][3] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_combine_features_partial_n() {
        let pca = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let spatial = [[10.0], [20.0], [30.0]];
        let mut output = [[0.0f64; 3]; 3];
        // Only process first 2 of 3
        combine_features::<2, 1, 3>(&pca, &spatial, 0.1, &mut output, 2);

        assert!((output[0][2] - 1.0).abs() < 1e-12); // 10.0 * 0.1
        assert!((output[1][2] - 2.0).abs() < 1e-12); // 20.0 * 0.1
                                                     // Third entry untouched
        assert!((output[2][0]).abs() < 1e-12);
    }

    #[test]
    fn test_extract_spatial_features_basic() {
        let mut data = [[0.0f64; 4]; 20];
        // Spike at sample 10: dominant on channel 2, some on channel 1
        data[10][2] = -10.0;
        data[10][1] = -3.0;

        let positions = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let events = [MultiChannelEvent {
            sample: 10,
            channel: 2,
            amplitude: 10.0,
        }];
        let mut output = [[0.0f64; 2]; 4];
        let n = extract_spatial_features::<4>(&data, &events, 1, &positions, &mut output);
        assert_eq!(n, 1);
        // Weighted: (3*25 + 10*50)/(3+10) = 575/13 ~ 44.23
        let expected_y = (3.0 * 25.0 + 10.0 * 50.0) / (3.0 + 10.0);
        assert!(
            (output[0][1] - expected_y).abs() < 1e-9,
            "y={}, expected={}",
            output[0][1],
            expected_y
        );
    }

    #[test]
    fn test_extract_spatial_features_empty() {
        let data = [[0.0f64; 2]; 20];
        let positions = [[0.0, 0.0], [0.0, 25.0]];
        let mut output = [[0.0f64; 2]; 4];
        let n = extract_spatial_features::<2>(&data, &[], 0, &positions, &mut output);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_extract_spatial_features_out_of_bounds_sample() {
        let data = [[0.0f64; 2]; 10];
        let positions = [[0.0, 0.0], [0.0, 25.0]];
        let events = [MultiChannelEvent {
            sample: 100, // out of bounds
            channel: 0,
            amplitude: 5.0,
        }];
        let mut output = [[0.0f64; 2]; 4];
        let n = extract_spatial_features::<2>(&data, &events, 1, &positions, &mut output);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_extract_spatial_features_multiple_events() {
        let mut data = [[0.0f64; 2]; 30];
        // Event 1 at sample 5: dominant on channel 0
        data[5][0] = -8.0;
        // Event 2 at sample 20: dominant on channel 1
        data[20][1] = -6.0;

        let positions = [[0.0, 0.0], [0.0, 50.0]];
        let events = [
            MultiChannelEvent {
                sample: 5,
                channel: 0,
                amplitude: 8.0,
            },
            MultiChannelEvent {
                sample: 20,
                channel: 1,
                amplitude: 6.0,
            },
        ];
        let mut output = [[0.0f64; 2]; 4];
        let n = extract_spatial_features::<2>(&data, &events, 2, &positions, &mut output);
        assert_eq!(n, 2);
        // Event 1: dominated by ch0 at y=0
        assert!(output[0][1] < 5.0);
        // Event 2: dominated by ch1 at y=50
        assert!(output[1][1] > 45.0);
    }

    #[test]
    fn test_combine_and_extract_end_to_end() {
        // Simulate a simple pipeline: detect -> extract spatial -> combine with PCA
        let mut data = [[0.0f64; 4]; 30];
        data[15][1] = -10.0;
        data[15][2] = -4.0;

        let positions = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let events = [MultiChannelEvent {
            sample: 15,
            channel: 1,
            amplitude: 10.0,
        }];

        // Extract spatial features
        let mut spatial = [[0.0f64; 2]; 1];
        let n = extract_spatial_features::<4>(&data, &events, 1, &positions, &mut spatial);
        assert_eq!(n, 1);

        // Fake PCA features
        let pca = [[0.5, -0.3, 1.2]];

        // Combine
        let mut combined = [[0.0f64; 5]; 1];
        combine_features::<3, 2, 5>(&pca, &spatial, 0.2, &mut combined, 1);

        // PCA part unchanged
        assert!((combined[0][0] - 0.5).abs() < 1e-12);
        assert!((combined[0][1] - (-0.3)).abs() < 1e-12);
        assert!((combined[0][2] - 1.2).abs() < 1e-12);
        // Spatial part is scaled
        assert!((combined[0][3] - spatial[0][0] * 0.2).abs() < 1e-12);
        assert!((combined[0][4] - spatial[0][1] * 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_threshold_basic() {
        // 4 channels with different noise levels.
        let mut rng = Rng::new(99);
        let mut data = [[0.0f64; 4]; 500];
        // Channel 0: noise std=1.0, Channel 1: std=3.0,
        // Channel 2: std=0.5, Channel 3: std=2.0
        let stds = [1.0, 3.0, 0.5, 2.0];
        for t in 0..500 {
            for ch in 0..4 {
                data[t][ch] = rng.gaussian(0.0, stds[ch]);
            }
        }
        let mut scratch = [0.0f64; 500];
        let thresholds =
            compute_adaptive_thresholds::<4>(&data, 4.0, 0.1, 200.0, 30000.0, &mut scratch);

        // Thresholds should be roughly proportional to noise levels.
        // Channel 1 (std=3) should have higher threshold than channel 2 (std=0.5).
        assert!(
            thresholds[1] > thresholds[2],
            "noisier channel should have higher threshold: ch1={} ch2={}",
            thresholds[1],
            thresholds[2]
        );
        // All thresholds should be positive and >= min_threshold.
        for ch in 0..4 {
            assert!(thresholds[ch] >= 0.1, "threshold must be >= min_threshold");
            assert!(thresholds[ch].is_finite(), "threshold must be finite");
        }
    }

    #[test]
    fn test_adaptive_threshold_dead_channel() {
        // Channel 0 has noise, channel 1 is dead (all zeros).
        let mut data = [[0.0f64; 2]; 200];
        let mut rng = Rng::new(42);
        for t in 0..200 {
            data[t][0] = rng.gaussian(0.0, 2.0);
            // Channel 1 stays zero.
        }
        let mut scratch = [0.0f64; 200];
        let thresholds =
            compute_adaptive_thresholds::<2>(&data, 4.0, 0.5, 100.0, 30000.0, &mut scratch);

        // Dead channel should get exactly min_threshold.
        assert!(
            (thresholds[1] - 0.5).abs() < 1e-12,
            "dead channel should get min_threshold, got {}",
            thresholds[1]
        );
        // Active channel should have threshold > min_threshold.
        assert!(
            thresholds[0] > 0.5,
            "active channel threshold should exceed min_threshold, got {}",
            thresholds[0]
        );
    }

    #[test]
    fn test_adaptive_threshold_overactive() {
        // Create a channel that crosses threshold very frequently.
        // Use a fast oscillation that will produce many crossings.
        let mut data = [[0.0f64; 2]; 1000];
        let mut rng = Rng::new(77);
        for t in 0..1000 {
            // Channel 0: moderate noise
            data[t][0] = rng.gaussian(0.0, 1.0);
            // Channel 1: same noise plus a fast oscillation that creates many crossings
            data[t][1] = rng.gaussian(0.0, 1.0)
                + 5.0 * libm::sin(2.0 * core::f64::consts::PI * 500.0 * t as f64 / 30000.0);
        }

        let mut scratch = [0.0f64; 1000];
        // Use a very low max_rate_hz so channel 1's activity triggers scaling.
        let thresholds =
            compute_adaptive_thresholds::<2>(&data, 4.0, 0.1, 5.0, 30000.0, &mut scratch);

        // Both thresholds should be valid.
        assert!(thresholds[0] >= 0.1);
        assert!(thresholds[1] >= 0.1);
        assert!(thresholds[0].is_finite());
        assert!(thresholds[1].is_finite());
    }

    #[test]
    fn test_adaptive_threshold_empty_data() {
        let data: &[[f64; 3]] = &[];
        let mut scratch = [0.0f64; 0];
        let thresholds =
            compute_adaptive_thresholds::<3>(data, 4.0, 0.75, 100.0, 30000.0, &mut scratch);

        // All channels should get min_threshold.
        for ch in 0..3 {
            assert!(
                (thresholds[ch] - 0.75).abs() < 1e-12,
                "empty data: channel {} should get min_threshold, got {}",
                ch,
                thresholds[ch]
            );
        }
    }
}
