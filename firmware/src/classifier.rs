//! Template-based spike classification via normalized cross-correlation.
//!
//! After the first-stage amplitude detector in [`crate::pipeline`] emits
//! [`SpikeEvent`](crate::pipeline::SpikeEvent)s, this module extracts the
//! waveform around each spike and classifies it against a library of
//! pre-loaded templates using normalized cross-correlation (NCC).
//!
//! All structures are `no_std`, zero-heap, and use fixed-size arrays.

use zerostone::float::Float;

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// A pre-loaded spike template for one channel's waveform.
///
/// `W` is the waveform length in samples (e.g., 48 at 30 kHz covers ~1.6 ms).
#[derive(Clone, Copy)]
pub struct Template<const W: usize> {
    /// The template waveform samples.
    pub waveform: [Float; W],
    /// Precomputed `||waveform||^2` (sum of squares) for fast NCC.
    pub norm_sq: Float,
    /// Cluster ID assigned to spikes matching this template.
    pub cluster_id: u8,
}

impl<const W: usize> Template<W> {
    /// An empty (inactive) template used to fill unused library slots.
    const fn empty() -> Self {
        Self {
            waveform: [0.0; W],
            norm_sq: 0.0,
            cluster_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TemplateLibrary
// ---------------------------------------------------------------------------

/// Fixed-capacity library of spike templates.
///
/// - `W`: waveform length in samples.
/// - `N`: maximum number of templates the library can hold.
pub struct TemplateLibrary<const W: usize, const N: usize> {
    templates: [Template<W>; N],
    count: usize,
}

impl<const W: usize, const N: usize> TemplateLibrary<W, N> {
    /// Creates a new empty template library.
    pub fn new() -> Self {
        Self {
            templates: [Template::empty(); N],
            count: 0,
        }
    }

    /// Adds a template to the library.
    ///
    /// Precomputes `norm_sq = sum(waveform[i]^2)` for NCC.
    /// Returns `false` if the library is full.
    pub fn add_template(&mut self, waveform: &[Float; W], cluster_id: u8) -> bool {
        if self.count >= N {
            return false;
        }
        let mut norm_sq: Float = 0.0;
        let mut i = 0;
        while i < W {
            norm_sq += waveform[i] * waveform[i];
            i += 1;
        }
        self.templates[self.count] = Template {
            waveform: *waveform,
            norm_sq,
            cluster_id,
        };
        self.count += 1;
        true
    }

    /// Returns the number of templates in the library.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns a reference to the template at `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&Template<W>> {
        if index < self.count {
            Some(&self.templates[index])
        } else {
            None
        }
    }

    /// Removes all templates from the library.
    pub fn clear(&mut self) {
        let mut i = 0;
        while i < self.count {
            self.templates[i] = Template::empty();
            i += 1;
        }
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// WaveformExtractor
// ---------------------------------------------------------------------------

/// Circular buffer that stores recent per-channel samples for waveform
/// extraction around detected spikes.
///
/// - `C`: number of channels.
/// - `W`: waveform length (number of samples to buffer per channel).
///
/// When a spike is detected, call [`extract`](Self::extract) to retrieve
/// the last `W` samples for the relevant channel in chronological order.
pub struct WaveformExtractor<const C: usize, const W: usize> {
    /// Per-channel circular buffer of Float samples.
    /// Layout: `buf[channel][sample]`.
    buf: [[Float; W]; C],
    /// Write index (shared across all channels since frames arrive together).
    head: usize,
    /// Number of samples pushed so far (saturates at W).
    filled: usize,
}

impl<const C: usize, const W: usize> WaveformExtractor<C, W> {
    /// Creates a new extractor with all buffers zeroed.
    pub fn new() -> Self {
        Self {
            buf: [[0.0; W]; C],
            head: 0,
            filled: 0,
        }
    }

    /// Pushes one multi-channel ADC frame into the circular buffer.
    ///
    /// Each i16 sample is converted to Float by dividing by 32768.0
    /// (matching the normalization in [`crate::pipeline`]).
    pub fn push_frame(&mut self, frame: &[i16; C]) {
        let mut ch = 0;
        while ch < C {
            self.buf[ch][self.head] = frame[ch] as Float / 32768.0;
            ch += 1;
        }
        self.head = (self.head + 1) % W;
        if self.filled < W {
            self.filled += 1;
        }
    }

    /// Extracts the last `W` samples for `channel` in chronological order.
    ///
    /// If fewer than `W` samples have been pushed, earlier slots are zero.
    pub fn extract(&self, channel: usize) -> [Float; W] {
        let mut out = [0.0; W];
        // The oldest sample is at index `head` (it was overwritten least
        // recently), and the newest is at `head - 1` (mod W).
        let mut i = 0;
        while i < W {
            let idx = (self.head + i) % W;
            out[i] = self.buf[channel][idx];
            i += 1;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/// Spike classifier using normalized cross-correlation (NCC) against a
/// template library.
///
/// - `W`: waveform length in samples.
/// - `N`: maximum number of templates.
pub struct Classifier<const W: usize, const N: usize> {
    library: TemplateLibrary<W, N>,
    min_correlation: Float,
}

impl<const W: usize, const N: usize> Classifier<W, N> {
    /// Creates a new classifier with the given minimum NCC threshold.
    ///
    /// Waveforms that do not exceed `min_correlation` against any template
    /// are assigned cluster ID 0 (unclassified).
    pub fn new(min_correlation: Float) -> Self {
        Self {
            library: TemplateLibrary::new(),
            min_correlation,
        }
    }

    /// Adds a template to the classifier's library.
    ///
    /// Returns `false` if the library is full.
    pub fn add_template(&mut self, waveform: &[Float; W], cluster_id: u8) -> bool {
        self.library.add_template(waveform, cluster_id)
    }

    /// Classifies a waveform against the template library.
    ///
    /// Computes the normalized cross-correlation (NCC) against each template:
    ///
    /// ```text
    /// NCC = dot(waveform, template) / (||waveform|| * ||template||)
    /// ```
    ///
    /// Returns the `cluster_id` of the best-matching template if its NCC
    /// exceeds `min_correlation`, or `0` (unclassified) otherwise.
    ///
    /// Uses optimized DSP primitives from [`crate::dsp`] for the hot path.
    pub fn classify(&self, waveform: &[Float; W]) -> u8 {
        if self.library.count() == 0 {
            return 0;
        }

        let mut best_ncc: Float = -2.0; // NCC range is [-1, 1]
        let mut best_id: u8 = 0;

        let mut t = 0;
        while t < self.library.count() {
            let tmpl = &self.library.templates[t];

            let corr = crate::dsp::ncc(waveform, &tmpl.waveform, tmpl.norm_sq);

            // ncc() returns 0.0 for zero-energy inputs, so no special guard needed.
            if corr > best_ncc {
                best_ncc = corr;
                best_id = tmpl.cluster_id;
            }

            t += 1;
        }

        if best_ncc >= self.min_correlation {
            best_id
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TemplateLibrary tests -----------------------------------------------

    #[test]
    fn library_new_is_empty() {
        let lib = TemplateLibrary::<8, 4>::new();
        assert_eq!(lib.count(), 0);
        assert!(lib.get(0).is_none());
    }

    #[test]
    fn library_add_and_get() {
        let mut lib = TemplateLibrary::<4, 4>::new();
        let wf = [1.0, 0.0, -1.0, 0.0];
        assert!(lib.add_template(&wf, 1));
        assert_eq!(lib.count(), 1);

        let t = lib.get(0).unwrap();
        assert_eq!(t.cluster_id, 1);
        assert_eq!(t.waveform, wf);
        // norm_sq = 1^2 + 0^2 + (-1)^2 + 0^2 = 2.0
        assert!((t.norm_sq - 2.0).abs() < 1e-6);
    }

    #[test]
    fn library_full_rejects() {
        let mut lib = TemplateLibrary::<2, 2>::new();
        assert!(lib.add_template(&[1.0, 0.0], 1));
        assert!(lib.add_template(&[0.0, 1.0], 2));
        assert!(!lib.add_template(&[1.0, 1.0], 3));
        assert_eq!(lib.count(), 2);
    }

    #[test]
    fn library_get_out_of_range() {
        let lib = TemplateLibrary::<4, 4>::new();
        assert!(lib.get(0).is_none());
        assert!(lib.get(100).is_none());
    }

    #[test]
    fn library_clear() {
        let mut lib = TemplateLibrary::<2, 4>::new();
        assert!(lib.add_template(&[1.0, 2.0], 1));
        assert!(lib.add_template(&[3.0, 4.0], 2));
        assert_eq!(lib.count(), 2);

        lib.clear();
        assert_eq!(lib.count(), 0);
        assert!(lib.get(0).is_none());
    }

    // -- WaveformExtractor tests ---------------------------------------------

    #[test]
    fn extractor_new_returns_zeros() {
        let ext = WaveformExtractor::<2, 4>::new();
        let wf = ext.extract(0);
        assert_eq!(wf, [0.0; 4]);
    }

    #[test]
    fn extractor_push_and_extract_single_channel() {
        let mut ext = WaveformExtractor::<2, 4>::new();

        // Push 4 frames. Channel 0 gets values 1000, 2000, 3000, 4000.
        ext.push_frame(&[1000, 0]);
        ext.push_frame(&[2000, 0]);
        ext.push_frame(&[3000, 0]);
        ext.push_frame(&[4000, 0]);

        let wf = ext.extract(0);
        let expected: [Float; 4] = [
            1000.0 / 32768.0,
            2000.0 / 32768.0,
            3000.0 / 32768.0,
            4000.0 / 32768.0,
        ];
        for i in 0..4 {
            assert!(
                (wf[i] - expected[i]).abs() < 1e-6,
                "sample {i}: got {}, expected {}",
                wf[i],
                expected[i]
            );
        }
    }

    #[test]
    fn extractor_wrap_around() {
        let mut ext = WaveformExtractor::<1, 3>::new();

        // Push 5 frames into a buffer of size 3. Should keep the last 3.
        ext.push_frame(&[100]);
        ext.push_frame(&[200]);
        ext.push_frame(&[300]);
        ext.push_frame(&[400]);
        ext.push_frame(&[500]);

        let wf = ext.extract(0);
        let expected: [Float; 3] = [
            300.0 / 32768.0,
            400.0 / 32768.0,
            500.0 / 32768.0,
        ];
        for i in 0..3 {
            assert!(
                (wf[i] - expected[i]).abs() < 1e-6,
                "sample {i}: got {}, expected {}",
                wf[i],
                expected[i]
            );
        }
    }

    #[test]
    fn extractor_multichannel() {
        let mut ext = WaveformExtractor::<3, 2>::new();

        ext.push_frame(&[100, 200, 300]);
        ext.push_frame(&[400, 500, 600]);

        // Check each channel independently.
        let wf0 = ext.extract(0);
        assert!((wf0[0] - 100.0 / 32768.0).abs() < 1e-6);
        assert!((wf0[1] - 400.0 / 32768.0).abs() < 1e-6);

        let wf1 = ext.extract(1);
        assert!((wf1[0] - 200.0 / 32768.0).abs() < 1e-6);
        assert!((wf1[1] - 500.0 / 32768.0).abs() < 1e-6);

        let wf2 = ext.extract(2);
        assert!((wf2[0] - 300.0 / 32768.0).abs() < 1e-6);
        assert!((wf2[1] - 600.0 / 32768.0).abs() < 1e-6);
    }

    // -- Classifier tests ----------------------------------------------------

    #[test]
    fn classify_empty_library_returns_zero() {
        let clf = Classifier::<4, 4>::new(0.7);
        let wf = [1.0, 0.0, -1.0, 0.0];
        assert_eq!(clf.classify(&wf), 0);
    }

    #[test]
    fn classify_perfect_match() {
        let mut clf = Classifier::<4, 4>::new(0.7);
        let template = [1.0, -0.5, 0.3, -0.8];
        clf.add_template(&template, 5);

        // Identical waveform should yield NCC = 1.0 and return cluster 5.
        assert_eq!(clf.classify(&template), 5);
    }

    #[test]
    fn classify_scaled_match() {
        let mut clf = Classifier::<4, 4>::new(0.7);
        let template = [1.0, -0.5, 0.3, -0.8];
        clf.add_template(&template, 3);

        // Scaled version (NCC is scale-invariant for positive scaling).
        let scaled = [2.0, -1.0, 0.6, -1.6];
        assert_eq!(clf.classify(&scaled), 3);
    }

    #[test]
    fn classify_orthogonal_returns_zero() {
        let mut clf = Classifier::<4, 8>::new(0.7);
        // Template along one axis.
        let template = [1.0, 0.0, 0.0, 0.0];
        clf.add_template(&template, 1);

        // Waveform orthogonal to template: NCC = 0.
        let orthogonal = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(clf.classify(&orthogonal), 0);
    }

    #[test]
    fn classify_best_of_multiple_templates() {
        let mut clf = Classifier::<4, 4>::new(0.5);

        let t1 = [1.0, 0.0, 0.0, 0.0];
        let t2 = [0.0, 1.0, 0.0, 0.0];
        let t3 = [1.0, 1.0, 0.0, 0.0];
        clf.add_template(&t1, 1);
        clf.add_template(&t2, 2);
        clf.add_template(&t3, 3);

        // Waveform most similar to t3 (NCC with t3 > NCC with t1 or t2).
        let wf = [0.9, 1.1, 0.0, 0.0];
        assert_eq!(clf.classify(&wf), 3);
    }

    #[test]
    fn classify_below_threshold_returns_zero() {
        let mut clf = Classifier::<4, 4>::new(0.99);
        let template = [1.0, 0.0, 0.0, 0.0];
        clf.add_template(&template, 1);

        // Waveform partially aligned but NCC < 0.99.
        let wf = [1.0, 0.5, 0.0, 0.0];
        // NCC = 1.0 / (sqrt(1.25) * 1.0) ~ 0.894
        assert_eq!(clf.classify(&wf), 0);
    }

    #[test]
    fn classify_zero_waveform_returns_zero() {
        let mut clf = Classifier::<4, 4>::new(0.7);
        let template = [1.0, -0.5, 0.3, -0.8];
        clf.add_template(&template, 1);

        let zero = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(clf.classify(&zero), 0);
    }

    #[test]
    fn classify_inverted_waveform() {
        let mut clf = Classifier::<4, 4>::new(0.7);
        let template = [1.0, -0.5, 0.3, -0.8];
        clf.add_template(&template, 2);

        // Inverted waveform: NCC = -1.0 (below any positive threshold).
        let inverted = [-1.0, 0.5, -0.3, 0.8];
        assert_eq!(clf.classify(&inverted), 0);
    }
}
