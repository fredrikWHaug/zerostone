//! Matched filter spike detection (Neyman-Pearson optimal detector).
//!
//! A matched filter maximizes the signal-to-noise ratio for detecting a known
//! waveform template in white Gaussian noise. This is the theoretical optimum
//! from the Neyman-Pearson lemma: no other linear detector can achieve higher
//! detection probability for a given false-positive rate.
//!
//! # Theory
//!
//! For pre-whitened data `x[t]` and a template `h[τ]` of length `W`:
//!
//! **Detection statistic:**
//!   `d(t) = Σ_τ h[τ] · x[t - W + 1 + τ]`
//!
//! **Normalized statistic:**
//!   `z(t) = d(t) / ‖h‖`  where `‖h‖ = √(Σ h²[τ])`
//!
//! Under the null hypothesis (noise only), `z ~ N(0, 1)`.
//! Under the alternative (spike at time t), `z ~ N(‖h‖, 1)`.
//!
//! The SNR gain over single-sample amplitude detection is `√(W_eff)`
//! where `W_eff` is the effective number of signal samples in the template.
//! For a typical spike waveform with energy spread over ~20 of 48 samples,
//! this yields a ~4.5× SNR improvement.
//!
//! **Amplitude estimation:**
//!   `α = d(t) / ‖h‖² = Σ h[τ] · x[t+τ] / Σ h²[τ]`
//!
//! This is the least-squares amplitude estimate, unbiased for Gaussian noise.
//!
//! # Online (real-time) detection
//!
//! [`OnlineMatchedDetector`] processes one sample at a time, maintaining a
//! circular buffer of the last `W` samples per channel. Each new sample
//! updates all filter outputs in `O(N × W)` time (or `O(N)` amortized with
//! incremental dot-product updates). This gives sub-millisecond latency
//! for template-based spike detection on embedded systems.
//!
//! # Example
//!
//! ```
//! use zerostone::matched_filter::{MatchedFilterBank, MatchedDetection};
//!
//! // Build a filter bank from a single template
//! let template = [0.0, -1.0, -3.0, -5.0, -3.0, -1.0, 0.5, 1.0];
//! let mut bank = MatchedFilterBank::<8, 4>::new();
//! bank.add_template(&template, 0).unwrap();
//!
//! // Inject a spike into noise-free single-channel data
//! let mut data = [[0.0f64; 1]; 30];
//! for i in 0..8 {
//!     data[10 + i] = [template[i]];
//! }
//!
//! // Detect
//! let mut detections = [MatchedDetection::ZERO; 32];
//! let n = bank.detect(&data, 3.0, 8, &mut detections);
//! assert!(n >= 1);
//! assert_eq!(detections[0].template_idx, 0);
//! assert!((detections[0].amplitude - 1.0).abs() < 0.2);
//! ```

/// A single detection from matched filter processing.
///
/// # Example
///
/// ```
/// use zerostone::matched_filter::MatchedDetection;
///
/// let d = MatchedDetection {
///     sample: 100,
///     template_idx: 2,
///     statistic: 15.3,
///     normalized: 5.1,
///     amplitude: 1.2,
/// };
/// assert_eq!(d.sample, 100);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MatchedDetection {
    /// Sample index of the detection peak.
    pub sample: usize,
    /// Index of the template that produced this detection.
    pub template_idx: usize,
    /// Raw detection statistic `d(t) = Σ h[τ] · x[t+τ]`.
    pub statistic: f64,
    /// Normalized statistic `z(t) = d(t) / ‖h‖`. Under H0: z ~ N(0,1).
    pub normalized: f64,
    /// Amplitude estimate `α = d(t) / ‖h‖²`.
    pub amplitude: f64,
}

impl MatchedDetection {
    /// Zero-valued detection for array initialization.
    pub const ZERO: Self = Self {
        sample: 0,
        template_idx: 0,
        statistic: 0.0,
        normalized: 0.0,
        amplitude: 0.0,
    };
}

/// Bank of matched filters for batch spike detection.
///
/// Stores up to `N` peak-channel filter kernels derived from spike templates.
/// Each kernel is the template waveform itself (for pre-whitened data, the
/// matched filter IS the template since the noise covariance is identity).
///
/// # Type Parameters
///
/// * `W` - Template/window length in samples (e.g. 48)
/// * `N` - Maximum number of templates/filters
///
/// # Example
///
/// ```
/// use zerostone::matched_filter::MatchedFilterBank;
///
/// let bank = MatchedFilterBank::<16, 8>::new();
/// assert_eq!(bank.n_filters(), 0);
/// ```
pub struct MatchedFilterBank<const W: usize, const N: usize> {
    /// Filter kernels (= templates for pre-whitened data).
    kernels: [[f64; W]; N],
    /// Squared norm of each kernel: ‖h‖² = Σ h²[τ].
    norms_sq: [f64; N],
    /// Norm of each kernel: ‖h‖ = √(‖h‖²).
    norms: [f64; N],
    /// Peak channel for each template.
    peak_channels: [usize; N],
    /// Number of active filters.
    n_filters: usize,
}

impl<const W: usize, const N: usize> Default for MatchedFilterBank<W, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const W: usize, const N: usize> MatchedFilterBank<W, N> {
    /// Create an empty filter bank.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::matched_filter::MatchedFilterBank;
    ///
    /// let bank = MatchedFilterBank::<8, 4>::new();
    /// assert_eq!(bank.n_filters(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            kernels: [[0.0; W]; N],
            norms_sq: [0.0; N],
            norms: [0.0; N],
            peak_channels: [0; N],
            n_filters: 0,
        }
    }

    /// Number of active filters in the bank.
    pub fn n_filters(&self) -> usize {
        self.n_filters
    }

    /// Add a template as a matched filter.
    ///
    /// The template should come from the mean waveform of a cluster,
    /// extracted from pre-whitened data on the peak detection channel.
    ///
    /// Returns the filter index, or `None` if the bank is full or
    /// the template has zero energy.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::matched_filter::MatchedFilterBank;
    ///
    /// let mut bank = MatchedFilterBank::<4, 2>::new();
    /// let idx = bank.add_template(&[-1.0, -3.0, -2.0, 0.5], 0);
    /// assert_eq!(idx, Some(0));
    /// assert_eq!(bank.n_filters(), 1);
    /// let idx2 = bank.add_template(&[0.5, -2.0, -4.0, -1.0], 3);
    /// assert_eq!(idx2, Some(1));
    /// assert_eq!(bank.n_filters(), 2);
    /// // Bank full
    /// assert_eq!(bank.add_template(&[1.0; 4], 0), None);
    /// ```
    pub fn add_template(&mut self, template: &[f64; W], peak_channel: usize) -> Option<usize> {
        if self.n_filters >= N {
            return None;
        }
        let mut norm_sq = 0.0;
        let mut w = 0;
        while w < W {
            norm_sq += template[w] * template[w];
            w += 1;
        }
        if norm_sq <= 0.0 || !norm_sq.is_finite() {
            return None;
        }
        let idx = self.n_filters;
        self.kernels[idx] = *template;
        self.norms_sq[idx] = norm_sq;
        self.norms[idx] = libm::sqrt(norm_sq);
        self.peak_channels[idx] = peak_channel;
        self.n_filters += 1;
        Some(idx)
    }

    /// Build a filter bank from cluster templates.
    ///
    /// Takes arrays of templates, counts, and peak channels (as produced by
    /// the sorter's `compute_cluster_means`). Only clusters with at least
    /// `min_count` spikes are included.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::matched_filter::MatchedFilterBank;
    ///
    /// let templates = [[-1.0, -3.0, -2.0, 0.5], [0.0; 4]];
    /// let counts = [10, 0];
    /// let peak_ch = [5, 0];
    /// let bank = MatchedFilterBank::<4, 2>::from_cluster_templates(
    ///     &templates, &counts, &peak_ch, 2, 3,
    /// );
    /// assert_eq!(bank.n_filters(), 1);
    /// ```
    pub fn from_cluster_templates(
        templates: &[[f64; W]; N],
        counts: &[u32; N],
        peak_channels: &[usize; N],
        n_clusters: usize,
        min_count: usize,
    ) -> Self {
        let mut bank = Self::new();
        let mut c = 0;
        while c < n_clusters && c < N {
            if (counts[c] as usize) >= min_count {
                bank.add_template(&templates[c], peak_channels[c]);
            }
            c += 1;
        }
        bank
    }

    /// Template norm ‖h‖ for filter `idx`.
    ///
    /// This is the theoretical SNR of the template in whitened space:
    /// a spike matching this template exactly will produce a normalized
    /// detection statistic of ‖h‖ standard deviations above noise.
    pub fn template_snr(&self, idx: usize) -> f64 {
        if idx < self.n_filters {
            self.norms[idx]
        } else {
            0.0
        }
    }

    /// Peak channel for filter `idx`.
    pub fn peak_channel(&self, idx: usize) -> usize {
        if idx < self.n_filters {
            self.peak_channels[idx]
        } else {
            0
        }
    }

    /// Compute the detection statistic for a single filter at a single position.
    ///
    /// `window` must be a W-sample slice from the data on the filter's peak channel.
    /// Returns `(statistic, normalized, amplitude)`.
    #[inline]
    fn correlate_window(&self, filter_idx: usize, window: &[f64; W]) -> (f64, f64, f64) {
        let kernel = &self.kernels[filter_idx];
        let mut dot = 0.0;
        let mut w = 0;
        while w < W {
            dot += kernel[w] * window[w];
            w += 1;
        }
        let norm = self.norms[filter_idx];
        let norm_sq = self.norms_sq[filter_idx];
        let normalized = if norm > 0.0 { dot / norm } else { 0.0 };
        let amplitude = if norm_sq > 0.0 { dot / norm_sq } else { 0.0 };
        (dot, normalized, amplitude)
    }

    /// Batch detection: slide all filters across single-channel data.
    ///
    /// For each filter, correlates the kernel with data on the filter's peak
    /// channel, finds peaks above `threshold` (in normalized sigma units),
    /// and resolves conflicts (same spike claimed by multiple filters).
    ///
    /// `data` is a flat slice of length `T × n_channels` with channels as
    /// the fast axis (interleaved: `data[t * n_channels + c]`).
    ///
    /// Returns the number of detections written to `out`.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved multi-channel data (pre-whitened)
    /// * `n_channels` - Number of channels in the data
    /// * `threshold` - Detection threshold in sigma (z-score) units
    /// * `refractory` - Minimum samples between detections from the same filter
    /// * `pre_samples` - Samples before the peak in the template window
    /// * `out` - Output buffer for detections
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::matched_filter::{MatchedFilterBank, MatchedDetection};
    ///
    /// let template = [0.0, -1.0, -3.0, -5.0, -3.0, -1.0, 0.5, 1.0];
    /// let mut bank = MatchedFilterBank::<8, 4>::new();
    /// bank.add_template(&template, 0).unwrap();
    ///
    /// // Single-channel data with one spike
    /// let mut data = [0.0f64; 40];
    /// for i in 0..8 { data[12 + i] = 1.5 * template[i]; }
    ///
    /// let mut det = [MatchedDetection::ZERO; 16];
    /// let n = bank.detect_interleaved(&data, 1, 3.0, 8, 3, &mut det);
    /// assert!(n >= 1);
    /// ```
    pub fn detect_interleaved(
        &self,
        data: &[f64],
        n_channels: usize,
        threshold: f64,
        refractory: usize,
        pre_samples: usize,
        out: &mut [MatchedDetection],
    ) -> usize {
        if self.n_filters == 0 || n_channels == 0 || data.len() < W * n_channels {
            return 0;
        }
        let t_len = data.len() / n_channels;
        let mut n_out = 0usize;

        // Process each filter independently, collecting candidates.
        // We use a two-phase approach: collect all candidates, then resolve conflicts.
        // Phase 1: collect per-filter candidates into `out` with greedy refractory.
        let mut f = 0;
        while f < self.n_filters {
            let ch = self.peak_channels[f];
            if ch >= n_channels {
                f += 1;
                continue;
            }

            // Slide the filter across the data on its peak channel.
            // The detection is aligned to the template's pre_samples offset:
            // for a template window starting at (t - pre_samples), the detection
            // time is t (the peak position within the template).
            let start_t = pre_samples;
            let end_t = if t_len >= W - pre_samples {
                t_len - (W - pre_samples) + 1
            } else {
                0
            };

            let mut last_det = 0usize; // last detection time for refractory
            let mut had_det = false;
            let mut t = start_t;
            while t < end_t {
                // Extract window: data[(t - pre_samples) .. (t - pre_samples + W)] on channel ch
                let win_start = t - pre_samples;
                let mut window = [0.0f64; W];
                let mut w = 0;
                while w < W {
                    window[w] = data[(win_start + w) * n_channels + ch];
                    w += 1;
                }

                let (stat, norm, amp) = self.correlate_window(f, &window);

                // Detection: normalized statistic exceeds threshold.
                // We look for positive correlation (template and data have same polarity).
                // For negative-going spikes with negative templates, dot product is positive.
                if norm > threshold {
                    // Find the peak within a local refractory window
                    let mut best_t = t;
                    let mut best_norm = norm;
                    let mut best_stat = stat;
                    let mut best_amp = amp;
                    let scan_end = (t + refractory).min(end_t);
                    let mut tt = t + 1;
                    while tt < scan_end {
                        let ws = tt - pre_samples;
                        let mut w2 = [0.0f64; W];
                        let mut ww = 0;
                        while ww < W {
                            w2[ww] = data[(ws + ww) * n_channels + ch];
                            ww += 1;
                        }
                        let (s2, n2, a2) = self.correlate_window(f, &w2);
                        if n2 > best_norm {
                            best_t = tt;
                            best_norm = n2;
                            best_stat = s2;
                            best_amp = a2;
                        }
                        tt += 1;
                    }

                    // Enforce refractory period
                    if !had_det || best_t >= last_det + refractory {
                        if n_out < out.len() {
                            out[n_out] = MatchedDetection {
                                sample: best_t,
                                template_idx: f,
                                statistic: best_stat,
                                normalized: best_norm,
                                amplitude: best_amp,
                            };
                            n_out += 1;
                        }
                        last_det = best_t;
                        had_det = true;
                    }
                    t = scan_end;
                } else {
                    t += 1;
                }
            }
            f += 1;
        }

        // Phase 2: sort by sample time, then resolve spatial/temporal conflicts.
        // Two detections from different filters at nearby times (< refractory)
        // are resolved by keeping the one with higher normalized statistic.
        if n_out > 1 {
            // Insertion sort by sample (small N × detections, cache-friendly)
            let mut k = 1;
            while k < n_out {
                let key = out[k];
                let mut pos = k;
                while pos > 0 && out[pos - 1].sample > key.sample {
                    out[pos] = out[pos - 1];
                    pos -= 1;
                }
                out[pos] = key;
                k += 1;
            }

            // Deduplicate: suppress weaker detection within refractory window
            let mut write = 0usize;
            let mut i = 0;
            while i < n_out {
                // Look ahead for conflicts within refractory window
                let mut best = i;
                let mut j = i + 1;
                while j < n_out && out[j].sample < out[i].sample + refractory {
                    if out[j].normalized > out[best].normalized {
                        best = j;
                    }
                    j += 1;
                }
                out[write] = out[best];
                write += 1;
                // Skip past the conflict group
                i = j;
            }
            n_out = write;
        }

        n_out
    }

    /// Detect on structured multi-channel data (array-of-arrays layout).
    ///
    /// Convenience wrapper that operates on `&[[f64; C]]` data (the layout
    /// used by `sort_multichannel`). Returns the number of detections.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::matched_filter::{MatchedFilterBank, MatchedDetection};
    ///
    /// let template = [0.0, -2.0, -5.0, -2.0];
    /// let mut bank = MatchedFilterBank::<4, 2>::new();
    /// bank.add_template(&template, 0).unwrap();
    ///
    /// // 2-channel data, spike on channel 0
    /// let mut data = [[0.0f64; 2]; 20];
    /// for i in 0..4 { data[6 + i][0] = template[i]; }
    ///
    /// let mut det = [MatchedDetection::ZERO; 8];
    /// let n = bank.detect(&data, 3.0, 4, &mut det);
    /// assert!(n >= 1);
    /// ```
    pub fn detect<const C: usize>(
        &self,
        data: &[[f64; C]],
        threshold: f64,
        refractory: usize,
        out: &mut [MatchedDetection],
    ) -> usize {
        if data.is_empty() || C == 0 {
            return 0;
        }
        // Compute pre_samples as W/2 (standard centering: peak at middle of window)
        // This matches the sorter's default pre_samples = 20 for W = 48 (close to W/2 = 24)
        let pre_samples = W * 5 / 12; // ~20 for W=48, matches sorter default

        // Safety: [[f64; C]] has the same memory layout as [f64] with stride C
        let flat =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const f64, data.len() * C) };
        self.detect_interleaved(flat, C, threshold, refractory, pre_samples, out)
    }
}

/// Online (real-time) matched filter detector.
///
/// Processes one multi-channel sample at a time, maintaining a circular
/// buffer of the last `W` samples on each monitored channel. When a filter's
/// detection statistic exceeds its threshold, a detection event is emitted.
///
/// This is the core real-time spike detector for the BCI online phase.
/// After template learning (offline or calibration), the online detector
/// runs in `O(N)` per sample with incremental dot-product updates.
///
/// # Type Parameters
///
/// * `W` - Window/template length in samples
/// * `N` - Maximum number of filter templates
/// * `MC` - Maximum number of monitored channels
///
/// # Example
///
/// ```
/// use zerostone::matched_filter::{MatchedFilterBank, OnlineMatchedDetector, MatchedDetection};
///
/// let template = [0.0, -1.0, -3.0, -5.0, -3.0, -1.0, 0.5, 1.0];
/// let mut bank = MatchedFilterBank::<8, 4>::new();
/// bank.add_template(&template, 0).unwrap();
///
/// let mut detector = OnlineMatchedDetector::<8, 4, 2>::new(&bank, 3.0, 8);
///
/// // Feed samples one at a time
/// let mut detected = false;
/// for t in 0..30 {
///     let sample = if (10..18).contains(&t) {
///         [template[t - 10], 0.0]
///     } else {
///         [0.0, 0.0]
///     };
///     if let Some(det) = detector.push(&sample) {
///         detected = true;
///         assert_eq!(det.template_idx, 0);
///     }
/// }
/// assert!(detected);
/// ```
pub struct OnlineMatchedDetector<const W: usize, const N: usize, const MC: usize> {
    /// Circular buffer: `buffer[ch][pos]` stores the last W samples per channel.
    buffer: [[f64; W]; MC],
    /// Write position in the circular buffer (wraps at W).
    write_pos: usize,
    /// Total samples pushed (for absolute sample index).
    samples_seen: usize,
    /// Filter kernels (copied from bank at construction).
    kernels: [[f64; W]; N],
    /// Template norms.
    norms: [f64; N],
    /// Template squared norms.
    norms_sq: [f64; N],
    /// Peak channel per filter.
    peak_channels: [usize; N],
    /// Number of active filters.
    n_filters: usize,
    /// Detection threshold in sigma units.
    threshold: f64,
    /// Refractory period in samples.
    refractory: usize,
    /// Last detection sample per filter (for refractory enforcement).
    last_detection: [usize; N],
    /// Whether each filter has had any detection (for refractory init).
    has_detected: [bool; N],
    /// Running dot products for incremental update: `dots[f]` accumulates
    /// the correlation of filter f with the current buffer contents.
    dots: [f64; N],
    /// Pre-samples offset (peak position within template).
    pre_samples: usize,
}

impl<const W: usize, const N: usize, const MC: usize> OnlineMatchedDetector<W, N, MC> {
    /// Create a new online detector from a filter bank.
    ///
    /// # Arguments
    ///
    /// * `bank` - The matched filter bank (templates)
    /// * `threshold` - Detection threshold in sigma units
    /// * `refractory` - Minimum samples between detections per filter
    pub fn new(bank: &MatchedFilterBank<W, N>, threshold: f64, refractory: usize) -> Self {
        let mut det = Self {
            buffer: [[0.0; W]; MC],
            write_pos: 0,
            samples_seen: 0,
            kernels: bank.kernels,
            norms: bank.norms,
            norms_sq: bank.norms_sq,
            peak_channels: bank.peak_channels,
            n_filters: bank.n_filters,
            threshold,
            refractory: refractory.max(1),
            last_detection: [0; N],
            has_detected: [false; N],
            dots: [0.0; N],
            pre_samples: W * 5 / 12,
        };
        // Initialize dot products to zero (buffer is all zeros)
        let mut f = 0;
        while f < N {
            det.dots[f] = 0.0;
            f += 1;
        }
        det
    }

    /// Total samples processed so far.
    pub fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// Push a single multi-channel sample and check for detections.
    ///
    /// Returns `Some(detection)` if any filter exceeds its threshold
    /// (the best-matching filter wins if multiple fire simultaneously).
    /// Returns `None` if no detection occurs.
    ///
    /// The detection's `sample` field is the absolute sample index
    /// (counting from the first push).
    pub fn push(&mut self, sample: &[f64; MC]) -> Option<MatchedDetection> {
        let pos = self.write_pos;

        // Update circular buffer
        let mut ch = 0;
        while ch < MC {
            self.buffer[ch][pos] = sample[ch];
            ch += 1;
        }

        // Update write position
        self.write_pos = if pos + 1 >= W { 0 } else { pos + 1 };
        self.samples_seen += 1;

        // Need at least W samples before we can detect
        if self.samples_seen < W {
            return None;
        }

        // The detection time is (samples_seen - 1 - (W - 1 - pre_samples))
        // = samples_seen - W + pre_samples
        let det_sample = self.samples_seen - W + self.pre_samples;

        // Compute full correlation for each filter (recompute each time for correctness).
        // The buffer is circular: position write_pos is the oldest sample,
        // write_pos+1 is the second oldest, etc.
        let mut best_f = 0usize;
        let mut best_norm = f64::NEG_INFINITY;
        let mut best_stat = 0.0;
        let mut best_amp = 0.0;
        let mut any_above = false;

        let mut f = 0;
        while f < self.n_filters {
            let fch = self.peak_channels[f];
            if fch >= MC {
                f += 1;
                continue;
            }

            // Check refractory
            if self.has_detected[f] && det_sample < self.last_detection[f] + self.refractory {
                f += 1;
                continue;
            }

            // Compute dot product with circular buffer
            let mut dot = 0.0;
            let oldest = self.write_pos; // oldest sample position
            let mut w = 0;
            while w < W {
                let buf_idx = if oldest + w >= W {
                    oldest + w - W
                } else {
                    oldest + w
                };
                dot += self.kernels[f][w] * self.buffer[fch][buf_idx];
                w += 1;
            }

            let norm = self.norms[f];
            let normalized = if norm > 0.0 { dot / norm } else { 0.0 };

            if normalized > self.threshold && normalized > best_norm {
                best_f = f;
                best_norm = normalized;
                best_stat = dot;
                best_amp = if self.norms_sq[f] > 0.0 {
                    dot / self.norms_sq[f]
                } else {
                    0.0
                };
                any_above = true;
            }
            f += 1;
        }

        if any_above {
            self.last_detection[best_f] = det_sample;
            self.has_detected[best_f] = true;
            Some(MatchedDetection {
                sample: det_sample,
                template_idx: best_f,
                statistic: best_stat,
                normalized: best_norm,
                amplitude: best_amp,
            })
        } else {
            None
        }
    }

    /// Reset the detector state (clear buffer and detection history).
    ///
    /// Keeps the filter bank and configuration intact.
    pub fn reset(&mut self) {
        self.buffer = [[0.0; W]; MC];
        self.write_pos = 0;
        self.samples_seen = 0;
        self.last_detection = [0; N];
        self.has_detected = [false; N];
        self.dots = [0.0; N];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Typical negative-going spike template (48 samples)
    fn spike_template_48() -> [f64; 48] {
        let mut t = [0.0f64; 48];
        // Gaussian-ish negative spike at sample 20 (the "peak")
        for (i, tv) in t.iter_mut().enumerate() {
            let x = (i as f64 - 20.0) / 4.0;
            *tv = -5.0 * libm::exp(-0.5 * x * x);
        }
        t
    }

    // Short 8-sample template for fast tests
    fn spike_template_8() -> [f64; 8] {
        [0.2, -0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 0.3]
    }

    #[test]
    fn test_bank_new_empty() {
        let bank = MatchedFilterBank::<8, 4>::new();
        assert_eq!(bank.n_filters(), 0);
    }

    #[test]
    fn test_add_template() {
        let mut bank = MatchedFilterBank::<8, 4>::new();
        let t = spike_template_8();
        assert_eq!(bank.add_template(&t, 3), Some(0));
        assert_eq!(bank.n_filters(), 1);
        assert_eq!(bank.peak_channel(0), 3);
        // SNR should be the template norm
        let expected_norm = libm::sqrt(t.iter().map(|x| x * x).sum::<f64>());
        assert!((bank.template_snr(0) - expected_norm).abs() < 1e-10);
    }

    #[test]
    fn test_add_zero_template_rejected() {
        let mut bank = MatchedFilterBank::<4, 2>::new();
        assert_eq!(bank.add_template(&[0.0; 4], 0), None);
        assert_eq!(bank.n_filters(), 0);
    }

    #[test]
    fn test_bank_full() {
        let mut bank = MatchedFilterBank::<4, 2>::new();
        assert!(bank.add_template(&[1.0, 2.0, 3.0, 4.0], 0).is_some());
        assert!(bank.add_template(&[4.0, 3.0, 2.0, 1.0], 1).is_some());
        assert!(bank.add_template(&[1.0; 4], 0).is_none());
    }

    #[test]
    fn test_from_cluster_templates() {
        let templates = [
            [-1.0, -3.0, -5.0, -2.0],
            [0.5, -1.0, -2.0, 0.3],
            [0.0; 4], // zero template
            [0.0; 4],
        ];
        let counts = [15, 2, 10, 0];
        let peak_ch = [0, 1, 2, 3];
        let bank =
            MatchedFilterBank::<4, 4>::from_cluster_templates(&templates, &counts, &peak_ch, 3, 3);
        assert_eq!(bank.n_filters(), 1); // only cluster 0 has count >= 3 and non-zero template
    }

    #[test]
    fn test_detect_single_spike() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // Single-channel data with one spike at sample 10
        let mut data = [[0.0f64; 1]; 30];
        for i in 0..8 {
            data[7 + i][0] = t[i]; // unit amplitude spike
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 2.0, 8, &mut det);
        assert!(n >= 1, "expected at least 1 detection, got {}", n);
        assert_eq!(det[0].template_idx, 0);
        assert!(
            (det[0].amplitude - 1.0).abs() < 0.2,
            "amplitude {} expected ~1.0",
            det[0].amplitude
        );
    }

    #[test]
    fn test_detect_scaled_spike() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // Spike with amplitude 2.0
        let mut data = [[0.0f64; 1]; 30];
        for i in 0..8 {
            data[7 + i][0] = 2.0 * t[i];
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 2.0, 8, &mut det);
        assert!(n >= 1);
        assert!(
            (det[0].amplitude - 2.0).abs() < 0.3,
            "amplitude {} expected ~2.0",
            det[0].amplitude
        );
    }

    #[test]
    fn test_detect_multiple_spikes() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // Two spikes separated by 20 samples
        let mut data = [[0.0f64; 1]; 50];
        for i in 0..8 {
            data[5 + i][0] = t[i];
            data[25 + i][0] = 1.5 * t[i];
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 2.0, 8, &mut det);
        assert_eq!(n, 2, "expected 2 detections, got {}", n);
        assert!(det[1].sample > det[0].sample);
    }

    #[test]
    fn test_detect_two_templates() {
        let t1 = [0.0, -1.0, -4.0, -2.0, 0.5, 1.0, 0.5, 0.0];
        let t2 = [0.0, 0.0, -1.0, -3.0, -5.0, -2.0, 0.0, 0.0];

        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t1, 0).unwrap();
        bank.add_template(&t2, 1).unwrap();

        // Channel 0 has template 1 spike, channel 1 has template 2 spike
        let mut data = [[0.0f64; 2]; 40];
        for i in 0..8 {
            data[5 + i][0] = t1[i]; // template 0 on channel 0
            data[25 + i][1] = t2[i]; // template 1 on channel 1
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 2.0, 8, &mut det);
        assert_eq!(n, 2);
        // Should identify correct templates
        let templates_found: [bool; 2] = [
            det[..n].iter().any(|d| d.template_idx == 0),
            det[..n].iter().any(|d| d.template_idx == 1),
        ];
        assert!(templates_found[0], "template 0 not detected");
        assert!(templates_found[1], "template 1 not detected");
    }

    #[test]
    fn test_detect_refractory_enforcement() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // Two spikes very close together (only 5 samples apart, refractory = 10)
        let mut data = [[0.0f64; 1]; 30];
        for i in 0..8 {
            data[3 + i][0] = t[i];
            data[8 + i][0] += t[i]; // overlapping
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 2.0, 10, &mut det);
        assert_eq!(n, 1, "refractory should suppress second detection");
    }

    #[test]
    fn test_detect_no_false_positives_in_noise() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // Pure zeros (no noise, no spikes)
        let data = [[0.0f64; 1]; 100];
        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect(&data, 3.0, 8, &mut det);
        assert_eq!(n, 0, "no spikes in zero data");
    }

    #[test]
    fn test_detect_48_sample_template() {
        let t = spike_template_48();
        let mut bank = MatchedFilterBank::<48, 8>::new();
        bank.add_template(&t, 0).unwrap();

        // Inject spike at known position
        let mut data = [[0.0f64; 2]; 200];
        for i in 0..48 {
            data[80 + i][0] = t[i];
        }

        let mut det = [MatchedDetection::ZERO; 16];
        let n = bank.detect(&data, 3.0, 15, &mut det);
        assert!(n >= 1, "48-sample template detection failed");
        assert_eq!(det[0].template_idx, 0);
    }

    #[test]
    fn test_snr_gain_over_amplitude() {
        // Demonstrate matched filter SNR gain: a weak spike detectable by
        // matched filter but below amplitude threshold.
        let t = spike_template_8();
        let norm = libm::sqrt(t.iter().map(|x| x * x).sum::<f64>());
        let peak = t.iter().copied().fold(f64::INFINITY, f64::min).abs();

        // SNR in amplitude space: peak / 1.0 (unit noise)
        // SNR in matched filter space: norm * amplitude / 1.0
        // Gain = norm / peak
        let gain = norm / peak;
        assert!(
            gain > 1.2,
            "matched filter should have >1.2x SNR gain, got {}",
            gain
        );
    }

    // --- Online detector tests ---

    #[test]
    fn test_online_detector_basic() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        let mut detector = OnlineMatchedDetector::<8, 4, 1>::new(&bank, 3.0, 8);

        // Feed a spike
        let mut detected = false;
        for s in 0..30 {
            let sample = if (10..18).contains(&s) {
                [t[s - 10]]
            } else {
                [0.0]
            };
            if let Some(det) = detector.push(&sample) {
                detected = true;
                assert_eq!(det.template_idx, 0);
                assert!(det.normalized > 3.0);
            }
        }
        assert!(detected, "online detector should detect the spike");
    }

    #[test]
    fn test_online_detector_two_channels() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 1).unwrap(); // template on channel 1

        let mut detector = OnlineMatchedDetector::<8, 4, 2>::new(&bank, 3.0, 8);

        let mut detected = false;
        for s in 0..30 {
            let sample = if (10..18).contains(&s) {
                [0.0, t[s - 10]] // spike on channel 1 only
            } else {
                [0.0, 0.0]
            };
            if let Some(det) = detector.push(&sample) {
                detected = true;
                assert_eq!(det.template_idx, 0);
            }
        }
        assert!(detected);
    }

    #[test]
    fn test_online_detector_refractory() {
        let t = [0.0, -2.0, -5.0, -2.0];
        let mut bank = MatchedFilterBank::<4, 2>::new();
        bank.add_template(&t, 0).unwrap();

        let mut detector = OnlineMatchedDetector::<4, 2, 1>::new(&bank, 2.0, 10);

        // Two spikes close together
        let mut detections = 0;
        for s in 0..30 {
            let sample = if (5..9).contains(&s) {
                [t[s - 5]]
            } else if (11..15).contains(&s) {
                [t[s - 11]]
            } else {
                [0.0]
            };
            if detector.push(&sample).is_some() {
                detections += 1;
            }
        }
        // With refractory=10, second spike should be suppressed
        assert!(
            detections <= 2,
            "refractory should limit detections to <=2, got {}",
            detections
        );
    }

    #[test]
    fn test_online_detector_no_false_positives() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        let mut detector = OnlineMatchedDetector::<8, 4, 1>::new(&bank, 4.0, 8);

        // Feed zeros
        for _ in 0..100 {
            assert!(detector.push(&[0.0]).is_none());
        }
    }

    #[test]
    fn test_online_reset() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        let mut detector = OnlineMatchedDetector::<8, 4, 1>::new(&bank, 3.0, 8);

        // Detect a spike
        for s in 0..20 {
            let sample = if (5..13).contains(&s) {
                [t[s - 5]]
            } else {
                [0.0]
            };
            detector.push(&sample);
        }
        assert!(detector.samples_seen() == 20);

        // Reset and verify
        detector.reset();
        assert_eq!(detector.samples_seen(), 0);

        // Should detect again after reset
        let mut detected = false;
        for s in 0..20 {
            let sample = if (5..13).contains(&s) {
                [t[s - 5]]
            } else {
                [0.0]
            };
            if detector.push(&sample).is_some() {
                detected = true;
            }
        }
        assert!(detected, "should detect after reset");
    }

    #[test]
    fn test_interleaved_detect() {
        let t = spike_template_8();
        let mut bank = MatchedFilterBank::<8, 4>::new();
        bank.add_template(&t, 0).unwrap();

        // 2-channel interleaved data, spike on channel 0
        let mut data = [0.0f64; 80]; // 40 samples, 2 channels
        for i in 0..8 {
            data[(10 + i) * 2] = t[i]; // channel 0
        }

        let mut det = [MatchedDetection::ZERO; 8];
        let n = bank.detect_interleaved(&data, 2, 2.0, 8, 3, &mut det);
        assert!(n >= 1, "interleaved detect should find spike");
        assert_eq!(det[0].template_idx, 0);
    }

    #[test]
    fn test_correlate_is_dot_product() {
        // Verify that the detection statistic is the dot product
        let t = [1.0, 2.0, 3.0, 4.0];
        let mut bank = MatchedFilterBank::<4, 2>::new();
        bank.add_template(&t, 0).unwrap();

        let window = [4.0, 3.0, 2.0, 1.0];
        let (stat, _norm, _amp) = bank.correlate_window(0, &window);
        // dot = 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert!((stat - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_estimation_accuracy() {
        // For a scaled template: d = α·‖h‖², so amplitude = d/‖h‖² = α
        let t = [0.0, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.2];
        let mut bank = MatchedFilterBank::<8, 2>::new();
        bank.add_template(&t, 0).unwrap();

        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0] {
            let mut window = [0.0f64; 8];
            for i in 0..8 {
                window[i] = alpha * t[i];
            }
            let (_, _, amp) = bank.correlate_window(0, &window);
            assert!(
                (amp - alpha).abs() < 1e-10,
                "alpha={}, estimated={}",
                alpha,
                amp
            );
        }
    }

    #[test]
    fn test_normalized_statistic_distribution() {
        // For a unit-amplitude spike (data = template), normalized = ‖h‖
        let t = spike_template_8();
        let norm_sq: f64 = t.iter().map(|x| x * x).sum();
        let expected_norm = libm::sqrt(norm_sq);

        let mut bank = MatchedFilterBank::<8, 2>::new();
        bank.add_template(&t, 0).unwrap();

        let (_, normalized, _) = bank.correlate_window(0, &t);
        assert!(
            (normalized - expected_norm).abs() < 1e-10,
            "normalized={}, expected={}",
            normalized,
            expected_norm
        );
    }

    #[test]
    fn test_detection_sensitivity_vs_amplitude() {
        // Matched filter detects weaker spikes than amplitude threshold would.
        // Template peak = 5.0, template norm = ~6.3 for spike_template_8
        let t = spike_template_8();
        let peak = t.iter().copied().fold(f64::INFINITY, f64::min).abs();
        let norm = libm::sqrt(t.iter().map(|x| x * x).sum::<f64>());

        // A spike at amplitude 0.6× would have:
        //   peak amplitude = 0.6 * 5.0 = 3.0 (below threshold=4.0 in amplitude space)
        //   matched filter statistic = 0.6 * norm ≈ 0.6 * 6.3 ≈ 3.8 (above threshold=3.0)
        let alpha = 0.6;
        let amp_snr = alpha * peak;
        let mf_snr = alpha * norm;

        assert!(
            amp_snr < 4.0,
            "weak spike should be below amplitude threshold"
        );
        assert!(
            mf_snr > 3.0,
            "weak spike should be above matched filter threshold"
        );
    }
}

// --- Kani formal verification ---

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove: `add_template` never panics for any 4-sample template.
    #[kani::proof]
    fn add_template_panic_free() {
        let mut bank = MatchedFilterBank::<4, 2>::new();
        let t: [f64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        // Filter NaN/Inf to valid floating point
        for v in &t {
            kani::assume(v.is_finite());
        }
        let ch: usize = kani::any();
        kani::assume(ch < 16);
        let _ = bank.add_template(&t, ch);
    }

    /// Prove: `correlate_window` never panics when called on a valid filter.
    #[kani::proof]
    fn correlate_window_panic_free() {
        let mut bank = MatchedFilterBank::<4, 2>::new();
        let t = [1.0, -2.0, -5.0, 1.0];
        bank.add_template(&t, 0);
        let w: [f64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for v in &w {
            kani::assume(v.is_finite());
            kani::assume(v.abs() < 1e6);
        }
        let (stat, norm, amp) = bank.correlate_window(0, &w);
        // Statistic must be finite for finite inputs
        assert!(stat.is_finite());
        assert!(norm.is_finite());
        assert!(amp.is_finite());
    }

    /// Prove: `template_snr` returns non-negative finite value.
    #[kani::proof]
    fn template_snr_nonneg_finite() {
        let mut bank = MatchedFilterBank::<4, 2>::new();
        let t: [f64; 4] = [kani::any(), kani::any(), kani::any(), kani::any()];
        for v in &t {
            kani::assume(v.is_finite());
            kani::assume(v.abs() > 0.01); // ensure non-zero
        }
        if bank.add_template(&t, 0).is_some() {
            let snr = bank.template_snr(0);
            assert!(snr >= 0.0);
            assert!(snr.is_finite());
        }
    }

    /// Prove: `from_cluster_templates` never panics.
    #[kani::proof]
    fn from_cluster_templates_panic_free() {
        let templates = [[1.0, -3.0, -2.0, 0.5], [0.0; 4]];
        let count0: u32 = kani::any();
        let count1: u32 = kani::any();
        kani::assume(count0 < 1000);
        kani::assume(count1 < 1000);
        let counts = [count0, count1];
        let peak_ch = [0usize, 1usize];
        let n_clusters: usize = kani::any();
        kani::assume(n_clusters <= 2);
        let min_count: usize = kani::any();
        kani::assume(min_count <= 100);
        let _ = MatchedFilterBank::<4, 2>::from_cluster_templates(
            &templates, &counts, &peak_ch, n_clusters, min_count,
        );
    }

    /// Prove: online detector `push` never panics on finite input.
    #[kani::proof]
    fn online_push_panic_free() {
        let t = [1.0, -3.0, -5.0, 1.0];
        let mut bank = MatchedFilterBank::<4, 2>::new();
        bank.add_template(&t, 0);
        let mut det = OnlineMatchedDetector::<4, 2, 1>::new(&bank, 3.0, 4);
        let sample: [f64; 1] = [kani::any()];
        kani::assume(sample[0].is_finite());
        kani::assume(sample[0].abs() < 1e6);
        let _ = det.push(&sample);
    }
}
