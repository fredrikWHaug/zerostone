//! Spike sorting foundations for extracellular electrophysiology.
//!
//! Provides the core building blocks for spike detection and classification:
//! - MAD-based noise estimation and threshold detection
//! - Waveform extraction with configurable pre/post ratio
//! - Peak alignment for consistent waveform representation
//! - PCA for dimensionality reduction of spike waveforms
//! - Template matching (Euclidean distance and NCC) for spike classification

use crate::linalg::Matrix;

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
}

#[cfg(test)]
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
}
