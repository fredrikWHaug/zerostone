//! Online template learning for adaptive spike classification.
//!
//! The [`OnlineLearner`] accumulates incoming spike waveforms and builds
//! templates on-the-fly, enabling the classifier to adapt to new neural
//! units without requiring pre-loaded templates.
//!
//! All structures are `no_std`, zero-heap, and use fixed-size arrays.

use zerostone::float::{self, Float};

// ---------------------------------------------------------------------------
// NCC helper
// ---------------------------------------------------------------------------

/// Computes the normalized cross-correlation between two waveforms.
///
/// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector has
/// zero energy.
pub fn ncc_f32<const W: usize>(a: &[Float; W], b: &[Float; W]) -> Float {
    let mut dot: Float = 0.0;
    let mut norm_a: Float = 0.0;
    let mut norm_b: Float = 0.0;
    let mut i = 0;
    while i < W {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    dot / (float::sqrt(norm_a) * float::sqrt(norm_b))
}

// ---------------------------------------------------------------------------
// TemplateAccumulator
// ---------------------------------------------------------------------------

/// Accumulates aligned spike waveforms to compute a mean template.
pub struct TemplateAccumulator<const W: usize> {
    /// Running sum of all accumulated waveforms.
    pub sum: [Float; W],
    /// Number of waveforms accumulated.
    pub count: u32,
    /// Cluster ID for this accumulator.
    pub cluster_id: u8,
}

impl<const W: usize> TemplateAccumulator<W> {
    /// Creates a new empty accumulator for the given cluster ID.
    pub fn new(cluster_id: u8) -> Self {
        Self {
            sum: [0.0; W],
            count: 0,
            cluster_id,
        }
    }

    /// Adds a waveform to the running sum.
    pub fn accumulate(&mut self, waveform: &[Float; W]) {
        let mut i = 0;
        while i < W {
            self.sum[i] += waveform[i];
            i += 1;
        }
        self.count += 1;
    }

    /// Returns the mean waveform (sum / count).
    ///
    /// If count is 0, returns all zeros.
    pub fn template(&self) -> [Float; W] {
        if self.count == 0 {
            return [0.0; W];
        }
        let mut out = [0.0; W];
        let inv = 1.0 / self.count as Float;
        let mut i = 0;
        while i < W {
            out[i] = self.sum[i] * inv;
            i += 1;
        }
        out
    }

    /// Returns the number of accumulated waveforms.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Resets the accumulator to zero.
    pub fn reset(&mut self) {
        let mut i = 0;
        while i < W {
            self.sum[i] = 0.0;
            i += 1;
        }
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// OnlineLearner
// ---------------------------------------------------------------------------

/// Online template learner that adapts templates from incoming spike data.
///
/// - `W`: waveform length in samples.
/// - `N`: maximum number of template slots.
///
/// Waveforms are accumulated per cluster. Once enough spikes are collected
/// (>= `min_spikes_init`), the mean template becomes available for use
/// by the classifier. Similar templates can be merged via [`try_merge`](Self::try_merge).
pub struct OnlineLearner<const W: usize, const N: usize> {
    /// One accumulator per template slot.
    accumulators: [TemplateAccumulator<W>; N],
    /// Number of active (in-use) template slots.
    active: usize,
    /// Minimum spike count before a template is considered initialized.
    min_spikes_init: u32,
    /// NCC threshold above which two templates are merged.
    merge_threshold: Float,
    /// Next cluster ID to assign when creating a new template.
    next_cluster_id: u8,
}

impl<const W: usize, const N: usize> OnlineLearner<W, N> {
    /// Creates a new online learner.
    pub fn new(min_spikes_init: u32, merge_threshold: Float) -> Self {
        // Initialize all accumulators with cluster_id 0 (unused).
        // We use a const-generic compatible initialization.
        let accumulators = {
            let mut arr: [TemplateAccumulator<W>; N] = unsafe {
                // Safety: TemplateAccumulator is plain data (floats, u32, u8).
                // We immediately initialize every element below.
                core::mem::MaybeUninit::uninit().assume_init()
            };
            let mut i = 0;
            while i < N {
                arr[i] = TemplateAccumulator::new(0);
                i += 1;
            }
            arr
        };
        Self {
            accumulators,
            active: 0,
            min_spikes_init,
            merge_threshold,
            next_cluster_id: 1,
        }
    }

    /// Processes an incoming waveform.
    ///
    /// - If `cluster_id` matches an existing accumulator, the waveform is
    ///   added to that accumulator.
    /// - If `cluster_id` is 0 (unclassified), a new template slot is
    ///   started if capacity allows.
    pub fn learn(&mut self, waveform: &[Float; W], cluster_id: u8) {
        if cluster_id != 0 {
            // Find matching accumulator.
            let mut i = 0;
            while i < self.active {
                if self.accumulators[i].cluster_id == cluster_id {
                    self.accumulators[i].accumulate(waveform);
                    return;
                }
                i += 1;
            }
        }

        // Unclassified or no matching accumulator: start a new template.
        if self.active < N {
            let id = self.next_cluster_id;
            self.next_cluster_id = self.next_cluster_id.wrapping_add(1);
            if self.next_cluster_id == 0 {
                self.next_cluster_id = 1; // skip 0
            }
            self.accumulators[self.active] = TemplateAccumulator::new(id);
            self.accumulators[self.active].accumulate(waveform);
            self.active += 1;
        }
    }

    /// Returns initialized templates (those with count >= min_spikes_init).
    ///
    /// Returns a fixed-size array of `(template, cluster_id)` pairs and the
    /// number of valid entries.
    pub fn get_templates(&self) -> ([([Float; W], u8); N], usize) {
        let mut result: [([Float; W], u8); N] = [([0.0; W], 0); N];
        let mut count = 0;
        let mut i = 0;
        while i < self.active {
            if self.accumulators[i].count >= self.min_spikes_init {
                result[count] = (
                    self.accumulators[i].template(),
                    self.accumulators[i].cluster_id,
                );
                count += 1;
            }
            i += 1;
        }
        (result, count)
    }

    /// Checks all pairs of active templates; if their NCC exceeds
    /// `merge_threshold`, merges the smaller into the larger.
    pub fn try_merge(&mut self) {
        let mut i = 0;
        while i < self.active {
            let mut j = i + 1;
            while j < self.active {
                let tmpl_i = self.accumulators[i].template();
                let tmpl_j = self.accumulators[j].template();
                let ncc = ncc_f32(&tmpl_i, &tmpl_j);

                if ncc > self.merge_threshold {
                    // Merge the one with fewer counts into the other.
                    let (keep, remove) = if self.accumulators[i].count
                        >= self.accumulators[j].count
                    {
                        (i, j)
                    } else {
                        (j, i)
                    };

                    // Add the removed accumulator's sum into the kept one.
                    let mut k = 0;
                    while k < W {
                        self.accumulators[keep].sum[k] +=
                            self.accumulators[remove].sum[k];
                        k += 1;
                    }
                    self.accumulators[keep].count +=
                        self.accumulators[remove].count;

                    // Compact: move last active slot into the removed slot.
                    let last = self.active - 1;
                    if remove != last {
                        // Copy fields manually for no_std.
                        let mut k = 0;
                        while k < W {
                            self.accumulators[remove].sum[k] =
                                self.accumulators[last].sum[k];
                            k += 1;
                        }
                        self.accumulators[remove].count =
                            self.accumulators[last].count;
                        self.accumulators[remove].cluster_id =
                            self.accumulators[last].cluster_id;
                    }
                    self.accumulators[last].reset();
                    self.accumulators[last].cluster_id = 0;
                    self.active -= 1;

                    // Re-check position j since it may now hold a different entry.
                    continue;
                }
                j += 1;
            }
            i += 1;
        }
    }

    /// Returns the number of active template slots.
    pub fn active_count(&self) -> usize {
        self.active
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_basic() {
        let mut acc = TemplateAccumulator::<4>::new(1);
        let wf = [1.0, -0.5, 0.3, -0.8];
        acc.accumulate(&wf);
        acc.accumulate(&wf);
        acc.accumulate(&wf);
        assert_eq!(acc.count(), 3);
        let tmpl = acc.template();
        for i in 0..4 {
            assert!(
                (tmpl[i] - wf[i]).abs() < 1e-6,
                "sample {i}: got {}, expected {}",
                tmpl[i],
                wf[i]
            );
        }
    }

    #[test]
    fn accumulator_averaging() {
        let mut acc = TemplateAccumulator::<4>::new(1);
        let wf1 = [1.0, 0.0, 0.0, 0.0];
        let wf2 = [0.0, 1.0, 0.0, 0.0];
        acc.accumulate(&wf1);
        acc.accumulate(&wf2);
        let tmpl = acc.template();
        let expected = [0.5, 0.5, 0.0, 0.0];
        for i in 0..4 {
            assert!(
                (tmpl[i] - expected[i]).abs() < 1e-6,
                "sample {i}: got {}, expected {}",
                tmpl[i],
                expected[i]
            );
        }
    }

    #[test]
    fn accumulator_reset() {
        let mut acc = TemplateAccumulator::<4>::new(1);
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(acc.count(), 1);
        acc.reset();
        assert_eq!(acc.count(), 0);
        let tmpl = acc.template();
        assert_eq!(tmpl, [0.0; 4]);
    }

    #[test]
    fn learner_cold_start() {
        let mut learner = OnlineLearner::<4, 4>::new(1, 0.95);
        let wf = [1.0, -0.5, 0.3, -0.8];
        // Feed unclassified (cluster_id = 0).
        learner.learn(&wf, 0);
        assert_eq!(learner.active_count(), 1);
        let (templates, count) = learner.get_templates();
        assert_eq!(count, 1);
        // Template should match the input waveform.
        for i in 0..4 {
            assert!(
                (templates[0].0[i] - wf[i]).abs() < 1e-6,
                "sample {i}: got {}, expected {}",
                templates[0].0[i],
                wf[i]
            );
        }
        // Cluster ID should be non-zero.
        assert_ne!(templates[0].1, 0);
    }

    #[test]
    fn learner_accumulate_existing() {
        let mut learner = OnlineLearner::<4, 4>::new(1, 0.95);
        let wf = [1.0, -0.5, 0.3, -0.8];
        // Create a template via cold start.
        learner.learn(&wf, 0);
        let (templates, _) = learner.get_templates();
        let assigned_id = templates[0].1;

        // Feed more waveforms with the assigned cluster ID.
        learner.learn(&wf, assigned_id);
        learner.learn(&wf, assigned_id);

        // Should still be 1 active template, with count 3.
        assert_eq!(learner.active_count(), 1);
        let (templates, count) = learner.get_templates();
        assert_eq!(count, 1);
        // Template mean should still match (identical waveforms).
        for i in 0..4 {
            assert!((templates[0].0[i] - wf[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn learner_merge_similar() {
        let mut learner = OnlineLearner::<4, 4>::new(1, 0.90);

        // Create two very similar templates.
        let wf1 = [1.0, -0.5, 0.3, -0.8];
        let wf2 = [1.01, -0.49, 0.31, -0.79]; // nearly identical

        learner.learn(&wf1, 0);
        learner.learn(&wf2, 0);
        assert_eq!(learner.active_count(), 2);

        // NCC of wf1 and wf2 should be > 0.90.
        let ncc = ncc_f32(&wf1, &wf2);
        assert!(ncc > 0.90, "NCC = {ncc}, expected > 0.90");

        learner.try_merge();
        assert_eq!(learner.active_count(), 1);
    }

    #[test]
    fn learner_max_templates() {
        let mut learner = OnlineLearner::<4, 2>::new(1, 0.95);
        // Fill both slots.
        learner.learn(&[1.0, 0.0, 0.0, 0.0], 0);
        learner.learn(&[0.0, 1.0, 0.0, 0.0], 0);
        assert_eq!(learner.active_count(), 2);

        // Try to add a third -- should be gracefully ignored.
        learner.learn(&[0.0, 0.0, 1.0, 0.0], 0);
        assert_eq!(learner.active_count(), 2);
    }

    #[test]
    fn ncc_identical() {
        let a = [1.0, -0.5, 0.3, -0.8];
        let ncc = ncc_f32(&a, &a);
        assert!(
            (ncc - 1.0).abs() < 1e-5,
            "NCC of identical vectors = {ncc}, expected 1.0"
        );
    }

    #[test]
    fn ncc_orthogonal() {
        let a: [Float; 4] = [1.0, 0.0, 0.0, 0.0];
        let b: [Float; 4] = [0.0, 1.0, 0.0, 0.0];
        let ncc = ncc_f32(&a, &b);
        assert!(
            ncc.abs() < 1e-5,
            "NCC of orthogonal vectors = {ncc}, expected ~0.0"
        );
    }

    #[test]
    fn get_templates_respects_min_spikes() {
        let mut learner = OnlineLearner::<4, 4>::new(3, 0.95);
        let wf = [1.0, -0.5, 0.3, -0.8];

        // Add 1 waveform (below min_spikes_init = 3). Cold start assigns ID.
        learner.learn(&wf, 0);
        assert_eq!(learner.active_count(), 1);
        // Read the assigned cluster_id directly from the accumulator.
        let assigned_id = learner.accumulators[0].cluster_id;
        assert_ne!(assigned_id, 0);

        // Add second waveform with the known cluster ID.
        learner.learn(&wf, assigned_id);

        let (_, count) = learner.get_templates();
        assert_eq!(count, 0, "template with 2 spikes should not be returned");

        // Add one more to reach 3.
        learner.learn(&wf, assigned_id);
        let (_, count) = learner.get_templates();
        assert_eq!(count, 1, "template with 3 spikes should be returned");
    }
}
