//! Template subtraction for resolving overlapping spikes.
//!
//! Implements a SpyKING CIRCUS-style greedy matching pursuit (Yger et al. 2018)
//! that iteratively subtracts the best-matching template from the data to
//! resolve temporally overlapping spikes on the same channel.
//!
//! # Algorithm
//!
//! For each candidate spike time, the algorithm:
//! 1. Extracts the data window around the spike
//! 2. Computes the dot product with every template
//! 3. Selects the best-matching template and computes an amplitude scalar
//! 4. If the amplitude is within bounds, subtracts `amplitude * template`
//!    from the data in-place and records the result
//! 5. Repeats until convergence (no new subtractions in a full pass)
//!
//! # References
//!
//! - Yger, Spampinato, Esposito et al. (2018). A spike sorting toolbox for
//!   up to thousands of electrodes validated with ground truth recordings
//!   in vitro and in vivo. eLife 7:e34518.
//!
//! # Example
//!
//! ```
//! use zerostone::template_subtract::{TemplateSubtractor, PeelResult};
//!
//! let mut sub = TemplateSubtractor::<8, 4>::new(50);
//! let template = [-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3];
//! sub.add_template(&template, 0.5, 2.0).unwrap();
//!
//! // Inject a scaled spike into data
//! let mut data = [0.0; 20];
//! for i in 0..8 {
//!     data[4 + i] = 1.5 * template[i]; // amplitude 1.5
//! }
//! let spike_times = [6usize]; // peak near sample 6
//! let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
//! let n = sub.peel(&mut data, &spike_times, 1, 2, &mut results);
//! assert_eq!(n, 1);
//! assert_eq!(results[0].template_id, 0);
//! assert!((results[0].amplitude - 1.5).abs() < 0.1);
//! ```

use crate::float::Float;
use crate::spike_sort::SortError;

/// Result of a single template subtraction (one resolved spike).
///
/// # Example
///
/// ```
/// use zerostone::template_subtract::PeelResult;
///
/// let r = PeelResult { sample: 100, template_id: 2, amplitude: 1.3 };
/// assert_eq!(r.sample, 100);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PeelResult {
    /// Sample index of the resolved spike.
    pub sample: usize,
    /// Index of the matched template.
    pub template_id: usize,
    /// Amplitude scaling factor applied to the template.
    pub amplitude: Float,
}

/// Greedy matching pursuit for resolving overlapping spikes.
///
/// Stores templates and iteratively subtracts the best match from the data,
/// following SpyKING CIRCUS's "peeling" approach.
///
/// # Type Parameters
///
/// * `W` - Template/window length in samples
/// * `N` - Maximum number of templates
///
/// # Example
///
/// ```
/// use zerostone::template_subtract::TemplateSubtractor;
///
/// let mut sub = TemplateSubtractor::<16, 8>::new(50);
/// assert_eq!(sub.n_templates(), 0);
/// ```
pub struct TemplateSubtractor<const W: usize, const N: usize> {
    templates: [[Float; W]; N],
    norms_sq: [Float; N],
    amp_min: [Float; N],
    amp_max: [Float; N],
    n_templates: usize,
    max_iter: usize,
    max_failures: usize,
}

impl<const W: usize, const N: usize> TemplateSubtractor<W, N> {
    /// Create a new template subtractor.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum peeling passes over all candidates
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::template_subtract::TemplateSubtractor;
    ///
    /// let sub = TemplateSubtractor::<8, 4>::new(50);
    /// assert_eq!(sub.n_templates(), 0);
    /// ```
    pub fn new(max_iter: usize) -> Self {
        Self {
            templates: [[0.0; W]; N],
            norms_sq: [0.0; N],
            amp_min: [0.0; N],
            amp_max: [0.0; N],
            n_templates: 0,
            max_iter: if max_iter > 0 { max_iter } else { 1 },
            max_failures: 3,
        }
    }

    /// Add a template with amplitude bounds.
    ///
    /// The amplitude bounds define the acceptable range for the scaling
    /// factor when matching this template. SpyKING CIRCUS typically uses
    /// median +/- 5*MAD of observed amplitudes.
    ///
    /// # Arguments
    ///
    /// * `template` - The waveform template (W samples)
    /// * `amp_min` - Minimum acceptable amplitude scalar
    /// * `amp_max` - Maximum acceptable amplitude scalar
    ///
    /// # Returns
    ///
    /// The template index, or `SortError::TemplateFull` if all N slots are used.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::template_subtract::TemplateSubtractor;
    ///
    /// let mut sub = TemplateSubtractor::<4, 2>::new(10);
    /// let idx = sub.add_template(&[-1.0, -3.0, -2.0, 0.0], 0.5, 2.0).unwrap();
    /// assert_eq!(idx, 0);
    /// ```
    pub fn add_template(
        &mut self,
        template: &[Float; W],
        amp_min: Float,
        amp_max: Float,
    ) -> Result<usize, SortError> {
        if self.n_templates >= N {
            return Err(SortError::TemplateFull);
        }
        let idx = self.n_templates;
        self.templates[idx] = *template;

        // Precompute ||template||^2
        let mut norm_sq = 0.0;
        for &v in template.iter() {
            norm_sq += v * v;
        }
        self.norms_sq[idx] = norm_sq;
        self.amp_min[idx] = amp_min;
        self.amp_max[idx] = amp_max;
        self.n_templates += 1;
        Ok(idx)
    }

    /// Set amplitude bounds for an existing template.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::template_subtract::TemplateSubtractor;
    ///
    /// let mut sub = TemplateSubtractor::<4, 2>::new(10);
    /// sub.add_template(&[-1.0, -3.0, -2.0, 0.0], 0.5, 2.0).unwrap();
    /// sub.set_amplitude_bounds(0, 0.3, 3.0);
    /// ```
    pub fn set_amplitude_bounds(&mut self, idx: usize, amp_min: Float, amp_max: Float) {
        if idx < self.n_templates {
            self.amp_min[idx] = amp_min;
            self.amp_max[idx] = amp_max;
        }
    }

    /// Number of templates currently stored.
    pub fn n_templates(&self) -> usize {
        self.n_templates
    }

    /// Set the maximum number of failures per spike time before giving up.
    pub fn set_max_failures(&mut self, max_failures: usize) {
        self.max_failures = if max_failures > 0 { max_failures } else { 1 };
    }

    /// Greedy matching pursuit: peel overlapping spikes from data.
    ///
    /// For each candidate spike time, finds the best template match, checks
    /// amplitude bounds, and subtracts the scaled template from `data` in-place.
    /// Repeats for `max_iter` passes until convergence.
    ///
    /// # Arguments
    ///
    /// * `data` - Mutable data buffer (modified in-place by subtraction)
    /// * `spike_times` - Candidate spike sample indices
    /// * `n_times` - Number of valid entries in `spike_times`
    /// * `pre_samples` - Samples before the spike time in the extraction window
    /// * `output` - Buffer for recording resolved spikes
    ///
    /// # Returns
    ///
    /// Number of resolved spikes written to `output`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::template_subtract::{TemplateSubtractor, PeelResult};
    ///
    /// let mut sub = TemplateSubtractor::<4, 2>::new(10);
    /// sub.add_template(&[-1.0, -3.0, -2.0, 0.0], 0.5, 2.0).unwrap();
    ///
    /// let mut data = [0.0, -1.5, -4.5, -3.0, -1.5, 0.0, 0.0, 0.0];
    /// let times = [2usize];
    /// let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
    /// let n = sub.peel(&mut data, &times, 1, 1, &mut results);
    /// assert!(n >= 1);
    /// ```
    pub fn peel(
        &self,
        data: &mut [Float],
        spike_times: &[usize],
        n_times: usize,
        pre_samples: usize,
        output: &mut [PeelResult],
    ) -> usize {
        let data_len = data.len();
        if self.n_templates == 0 || n_times == 0 || data_len < W || output.is_empty() {
            return 0;
        }

        let nt = if n_times < spike_times.len() {
            n_times
        } else {
            spike_times.len()
        };

        // Track failures per spike time (using output buffer length as proxy cap)
        // We use a simple approach: track active spike times with failure counts
        // in small fixed arrays. For simplicity, use up to 256 active candidates.
        const MAX_CANDIDATES: usize = 256;
        let n_candidates = if nt < MAX_CANDIDATES {
            nt
        } else {
            MAX_CANDIDATES
        };
        let mut active = [true; MAX_CANDIDATES];
        let mut failures = [0usize; MAX_CANDIDATES];
        let mut result_count = 0;

        for _iter in 0..self.max_iter {
            let mut any_subtracted = false;

            for ci in 0..n_candidates {
                if !active[ci] || result_count >= output.len() {
                    continue;
                }

                let t = spike_times[ci];
                let start = t.saturating_sub(pre_samples);
                let end = start + W;
                if end > data_len || start + pre_samples < t {
                    active[ci] = false;
                    continue;
                }

                // Find best template match
                let mut best_template = 0;
                let mut best_dot: Float = 0.0;
                let mut best_norm_sq: Float = 1.0;

                for ti in 0..self.n_templates {
                    let mut dot = 0.0;
                    for w in 0..W {
                        dot += data[start + w] * self.templates[ti][w];
                    }
                    // Best = highest absolute dot product (most energy explained)
                    let abs_dot = if dot >= 0.0 { dot } else { -dot };
                    let abs_best = if best_dot >= 0.0 { best_dot } else { -best_dot };
                    if ti == 0 || abs_dot > abs_best {
                        best_dot = dot;
                        best_norm_sq = self.norms_sq[ti];
                        best_template = ti;
                    }
                }

                // Compute amplitude: a = dot(data, template) / ||template||^2
                if best_norm_sq < 1e-30 {
                    active[ci] = false;
                    continue;
                }
                let amplitude = best_dot / best_norm_sq;

                // Check amplitude bounds
                if amplitude >= self.amp_min[best_template]
                    && amplitude <= self.amp_max[best_template]
                {
                    // Subtract scaled template from data
                    for w in 0..W {
                        data[start + w] -= amplitude * self.templates[best_template][w];
                    }

                    output[result_count] = PeelResult {
                        sample: t,
                        template_id: best_template,
                        amplitude,
                    };
                    result_count += 1;
                    any_subtracted = true;
                    // Don't deactivate -- same time could have another overlapping spike
                } else {
                    failures[ci] += 1;
                    if failures[ci] >= self.max_failures {
                        active[ci] = false;
                    }
                }
            }

            if !any_subtracted {
                break;
            }
        }

        result_count
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `peel` does not panic for small N=2, W=4 inputs.
    #[kani::proof]
    #[kani::unwind(12)]
    fn peel_no_panic() {
        let mut sub = TemplateSubtractor::<4, 2>::new(3);

        let t0: Float = kani::any();
        let t1: Float = kani::any();
        let t2: Float = kani::any();
        let t3: Float = kani::any();

        kani::assume(t0.is_finite() && t0 >= -10.0 && t0 <= 10.0);
        kani::assume(t1.is_finite() && t1 >= -10.0 && t1 <= 10.0);
        kani::assume(t2.is_finite() && t2 >= -10.0 && t2 <= 10.0);
        kani::assume(t3.is_finite() && t3 >= -10.0 && t3 <= 10.0);

        let template = [t0, t1, t2, t3];
        let _ = sub.add_template(&template, 0.5, 2.0);

        let d0: Float = kani::any();
        let d1: Float = kani::any();
        let d2: Float = kani::any();
        let d3: Float = kani::any();
        let d4: Float = kani::any();
        let d5: Float = kani::any();
        let d6: Float = kani::any();
        let d7: Float = kani::any();

        kani::assume(d0.is_finite() && d0 >= -100.0 && d0 <= 100.0);
        kani::assume(d1.is_finite() && d1 >= -100.0 && d1 <= 100.0);
        kani::assume(d2.is_finite() && d2 >= -100.0 && d2 <= 100.0);
        kani::assume(d3.is_finite() && d3 >= -100.0 && d3 <= 100.0);
        kani::assume(d4.is_finite() && d4 >= -100.0 && d4 <= 100.0);
        kani::assume(d5.is_finite() && d5 >= -100.0 && d5 <= 100.0);
        kani::assume(d6.is_finite() && d6 >= -100.0 && d6 <= 100.0);
        kani::assume(d7.is_finite() && d7 >= -100.0 && d7 <= 100.0);

        let mut data = [d0, d1, d2, d3, d4, d5, d6, d7];
        let spike_time: usize = kani::any();
        kani::assume(spike_time <= 7);
        let times = [spike_time];
        let pre: usize = kani::any();
        kani::assume(pre <= 3);

        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 4];
        let n = sub.peel(&mut data, &times, 1, pre, &mut results);
        assert!(n <= 4, "result count must not exceed buffer");
    }

    /// Prove that `add_template` never panics and returns valid index or error.
    #[kani::proof]
    #[kani::unwind(6)]
    fn add_template_no_panic() {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);

        let t0: Float = kani::any();
        let t1: Float = kani::any();
        let t2: Float = kani::any();
        let t3: Float = kani::any();
        let amp_min: Float = kani::any();
        let amp_max: Float = kani::any();

        kani::assume(t0.is_finite() && t0 >= -10.0 && t0 <= 10.0);
        kani::assume(t1.is_finite() && t1 >= -10.0 && t1 <= 10.0);
        kani::assume(t2.is_finite() && t2 >= -10.0 && t2 <= 10.0);
        kani::assume(t3.is_finite() && t3 >= -10.0 && t3 <= 10.0);
        kani::assume(amp_min.is_finite() && amp_min >= 0.0 && amp_min <= 10.0);
        kani::assume(amp_max.is_finite() && amp_max >= amp_min && amp_max <= 20.0);

        let template = [t0, t1, t2, t3];
        match sub.add_template(&template, amp_min, amp_max) {
            Ok(idx) => assert!(idx < 2, "index must be within N"),
            Err(_) => {} // TemplateFull is valid after N adds
        }
    }

    /// Prove that `peel` output count never exceeds the output buffer length.
    #[kani::proof]
    #[kani::unwind(10)]
    fn peel_output_bounded() {
        let mut sub = TemplateSubtractor::<4, 2>::new(3);
        let _ = sub.add_template(&[-1.0, -3.0, -2.0, 0.0], 0.1, 5.0);

        let d0: Float = kani::any();
        let d1: Float = kani::any();
        let d2: Float = kani::any();
        let d3: Float = kani::any();
        let d4: Float = kani::any();
        let d5: Float = kani::any();

        kani::assume(d0.is_finite() && d0 >= -100.0 && d0 <= 100.0);
        kani::assume(d1.is_finite() && d1 >= -100.0 && d1 <= 100.0);
        kani::assume(d2.is_finite() && d2 >= -100.0 && d2 <= 100.0);
        kani::assume(d3.is_finite() && d3 >= -100.0 && d3 <= 100.0);
        kani::assume(d4.is_finite() && d4 >= -100.0 && d4 <= 100.0);
        kani::assume(d5.is_finite() && d5 >= -100.0 && d5 <= 100.0);

        let mut data = [d0, d1, d2, d3, d4, d5];
        let times = [2usize];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 2];
        let n = sub.peel(&mut data, &times, 1, 1, &mut results);
        assert!(n <= 2, "output count must not exceed buffer length");
    }

    /// Prove that creating, adding a template, and mutating config never panics.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_template_subtractor_lifecycle_no_panic() {
        let max_iter: usize = kani::any();
        kani::assume(max_iter <= 100);
        let mut sub = TemplateSubtractor::<4, 2>::new(max_iter);
        assert_eq!(sub.n_templates(), 0);

        let t0: Float = kani::any();
        let t1: Float = kani::any();
        let t2: Float = kani::any();
        let t3: Float = kani::any();
        kani::assume(t0.is_finite() && t0 >= -10.0 && t0 <= 10.0);
        kani::assume(t1.is_finite() && t1 >= -10.0 && t1 <= 10.0);
        kani::assume(t2.is_finite() && t2 >= -10.0 && t2 <= 10.0);
        kani::assume(t3.is_finite() && t3 >= -10.0 && t3 <= 10.0);

        let _ = sub.add_template(&[t0, t1, t2, t3], 0.5, 2.0);
        assert_eq!(sub.n_templates(), 1);

        // Mutate bounds and max_failures -- must not panic
        sub.set_amplitude_bounds(0, 0.1, 5.0);
        let mf: usize = kani::any();
        kani::assume(mf <= 50);
        sub.set_max_failures(mf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;

    #[test]
    fn test_add_template() {
        let mut sub = TemplateSubtractor::<8, 4>::new(50);
        let t = [-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3];
        let idx = sub.add_template(&t, 0.5, 2.0).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sub.n_templates(), 1);

        let idx2 = sub.add_template(&t, 0.3, 3.0).unwrap();
        assert_eq!(idx2, 1);
        assert_eq!(sub.n_templates(), 2);
    }

    #[test]
    fn test_template_full() {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        sub.add_template(&[1.0; 4], 0.5, 2.0).unwrap();
        sub.add_template(&[2.0; 4], 0.5, 2.0).unwrap();
        assert_eq!(
            sub.add_template(&[3.0; 4], 0.5, 2.0),
            Err(SortError::TemplateFull)
        );
    }

    #[test]
    fn test_basic_single_template_peel() {
        let mut sub = TemplateSubtractor::<8, 4>::new(50);
        let template = [-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3];
        sub.add_template(&template, 0.5, 2.0).unwrap();

        // Inject a spike at amplitude 1.5 starting at sample 4
        let mut data = [0.0; 24];
        let amp = 1.5;
        for i in 0..8 {
            data[4 + i] = amp * template[i];
        }

        let spike_times = [6usize]; // peak is near sample 6 (where template is -5.0)
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 8];
        let n = sub.peel(&mut data, &spike_times, 1, 2, &mut results);
        assert_eq!(n, 1);
        assert_eq!(results[0].template_id, 0);
        assert!(
            (results[0].amplitude - amp).abs() < 0.01,
            "Expected amplitude {}, got {}",
            amp,
            results[0].amplitude
        );

        // Data should be near-zero after subtraction
        let residual: Float = data.iter().map(|x| x * x).sum();
        assert!(
            residual < 0.01,
            "Residual energy should be near zero, got {}",
            residual
        );
    }

    #[test]
    fn test_two_template_overlap() {
        let mut sub = TemplateSubtractor::<4, 4>::new(50);
        let t1 = [-1.0, -3.0, -2.0, 0.0];
        let t2 = [0.0, -2.0, -4.0, -1.0];
        sub.add_template(&t1, 0.5, 2.0).unwrap();
        sub.add_template(&t2, 0.5, 2.0).unwrap();

        // Two overlapping spikes: t1 at time 2, t2 at time 3
        let mut data = [0.0; 12];
        // t1 with amp=1.0 starting at sample 1 (peak at 2, pre=1)
        for i in 0..4 {
            data[1 + i] += 1.0 * t1[i];
        }
        // t2 with amp=1.0 starting at sample 2 (peak at 3, pre=1)
        for i in 0..4 {
            data[2 + i] += 1.0 * t2[i];
        }

        let spike_times = [2usize, 3];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 8];
        let n = sub.peel(&mut data, &spike_times, 2, 1, &mut results);
        // Should resolve both spikes (possibly in multiple iterations)
        assert!(n >= 1, "Should resolve at least 1 spike, got {}", n);
    }

    #[test]
    fn test_amplitude_bounds_rejection() {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        let t = [-1.0, -3.0, -2.0, 0.0];
        sub.add_template(&t, 0.8, 1.2).unwrap(); // tight bounds

        // Inject spike with amplitude 2.0 (outside [0.8, 1.2])
        let mut data = [0.0; 12];
        for i in 0..4 {
            data[2 + i] = 2.0 * t[i];
        }

        let spike_times = [3usize];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 4];
        let n = sub.peel(&mut data, &spike_times, 1, 1, &mut results);
        assert_eq!(n, 0, "Should reject spike outside amplitude bounds");
    }

    #[test]
    fn test_convergence_empty_input() {
        let sub = TemplateSubtractor::<4, 2>::new(10);
        let mut data = [0.0; 8];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 4];

        // No templates
        let n = sub.peel(&mut data, &[2], 1, 1, &mut results);
        assert_eq!(n, 0);

        // No spike times
        let mut sub2 = TemplateSubtractor::<4, 2>::new(10);
        sub2.add_template(&[1.0; 4], 0.5, 2.0).unwrap();
        let n = sub2.peel(&mut data, &[], 0, 1, &mut results);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_empty_data() {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        sub.add_template(&[1.0; 4], 0.5, 2.0).unwrap();
        let mut data = [0.0; 2]; // too short for W=4
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 4];
        let n = sub.peel(&mut data, &[0], 1, 0, &mut results);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_set_amplitude_bounds() {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        let t = [-1.0, -3.0, -2.0, 0.0];
        sub.add_template(&t, 0.8, 1.2).unwrap();

        // Initially rejects amplitude 1.5
        let mut data = [0.0; 12];
        for i in 0..4 {
            data[2 + i] = 1.5 * t[i];
        }
        let spike_times = [3usize];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 4];
        let n = sub.peel(&mut data, &spike_times, 1, 1, &mut results);
        assert_eq!(n, 0);

        // Widen bounds and retry
        sub.set_amplitude_bounds(0, 0.5, 2.0);
        // Re-create data since peel doesn't modify if rejected
        for i in 0..4 {
            data[2 + i] = 1.5 * t[i];
        }
        let n = sub.peel(&mut data, &spike_times, 1, 1, &mut results);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_pure_noise_convergence() {
        // If data doesn't match any template well, peel should converge quickly
        let mut sub = TemplateSubtractor::<8, 2>::new(50);
        sub.add_template(&[-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3], 0.8, 1.2)
            .unwrap();

        // Random-ish data that doesn't match the template
        let mut data = [
            0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1,
        ];
        let spike_times = [4usize, 8];
        let mut results = [PeelResult {
            sample: 0,
            template_id: 0,
            amplitude: 0.0,
        }; 8];
        let n = sub.peel(&mut data, &spike_times, 2, 2, &mut results);
        // Should not find valid matches due to amplitude bounds
        assert_eq!(n, 0, "Noise should not produce valid matches");
    }
}
