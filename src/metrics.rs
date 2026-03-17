//! Spike sorting comparison metrics for ground-truth validation.
//!
//! Compares a sorter's output against ground-truth spike trains using the
//! standard metrics from SpikeInterface and SpikeForest:
//!
//! - **Accuracy** = TP / (TP + FN + FP)
//! - **Precision** = TP / (TP + FP)
//! - **Recall** = TP / (TP + FN)
//!
//! A spike is a *true positive* when a ground-truth spike and a sorted spike
//! fall within a tolerance window (typically 0.4 ms = 12 samples at 30 kHz).
//! Greedy matching assigns each ground-truth unit to the sorted unit that
//! maximises accuracy.
//!
//! # References
//!
//! - Buccino et al. (2020). SpikeInterface, a unified framework for spike
//!   sorting. eLife 9:e61834.
//! - Magland et al. (2020). SpikeForest, reproducible web-facing ground-truth
//!   validation of automated neural spike sorters. eLife 9:e55167.
//!
//! # Example
//!
//! ```
//! use zerostone::metrics::{compare_spike_trains, compare_sorting, UnitMatch};
//!
//! // Perfect match within tolerance of 2 samples
//! let gt = [10, 50, 100];
//! let sorted = [11, 49, 101];
//! let m = compare_spike_trains(&gt, &sorted, 2);
//! assert_eq!(m.true_positives, 3);
//! assert!((m.accuracy - 1.0).abs() < 1e-10);
//! ```

/// Result of comparing one ground-truth spike train to one sorted spike train.
#[derive(Debug, Clone, Copy)]
pub struct UnitMatch {
    /// Number of spikes matched within tolerance (found in both GT and sorted).
    pub true_positives: usize,
    /// Number of sorted spikes with no matching GT spike.
    pub false_positives: usize,
    /// Number of GT spikes with no matching sorted spike.
    pub false_negatives: usize,
    /// TP / (TP + FN + FP). 0.0 when all counts are zero.
    pub accuracy: f64,
    /// TP / (TP + FP). 0.0 when TP + FP == 0.
    pub precision: f64,
    /// TP / (TP + FN). 0.0 when TP + FN == 0.
    pub recall: f64,
}

impl UnitMatch {
    /// Create a `UnitMatch` with all zeros.
    pub fn empty() -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
        }
    }
}

/// Compare two spike trains within a sample-level tolerance window.
///
/// Both `gt_times` and `sorted_times` must be sorted in ascending order.
/// `tolerance` is the maximum absolute difference (in samples) for two spikes
/// to be considered a match.
///
/// Each ground-truth spike is matched to at most one sorted spike and vice
/// versa. Matching is greedy left-to-right: the earliest unmatched sorted
/// spike within tolerance is paired with each GT spike.
///
/// # Arguments
///
/// * `gt_times` - Ground-truth spike times (sorted ascending)
/// * `sorted_times` - Sorted spike times (sorted ascending)
/// * `tolerance` - Maximum sample distance for a match
///
/// # Returns
///
/// A [`UnitMatch`] with TP, FP, FN counts and derived metrics.
///
/// # Example
///
/// ```
/// use zerostone::metrics::compare_spike_trains;
///
/// let gt = [100, 200, 300, 400];
/// let sorted = [101, 199, 350, 401];
/// let m = compare_spike_trains(&gt, &sorted, 5);
/// assert_eq!(m.true_positives, 3); // 100-101, 200-199, 400-401 match
/// assert_eq!(m.false_negatives, 1); // 300 missed (350 too far)
/// assert_eq!(m.false_positives, 1); // 350 unmatched
/// ```
pub fn compare_spike_trains(
    gt_times: &[usize],
    sorted_times: &[usize],
    tolerance: usize,
) -> UnitMatch {
    if gt_times.is_empty() && sorted_times.is_empty() {
        return UnitMatch::empty();
    }
    if gt_times.is_empty() {
        return UnitMatch {
            true_positives: 0,
            false_positives: sorted_times.len(),
            false_negatives: 0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
        };
    }
    if sorted_times.is_empty() {
        return UnitMatch {
            true_positives: 0,
            false_positives: 0,
            false_negatives: gt_times.len(),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
        };
    }

    let mut tp = 0usize;
    // Track which sorted spikes have been consumed.
    // We use a scan pointer: for each GT spike, search forward from the
    // last matched sorted index. Because both arrays are sorted, a matched
    // sorted spike never needs to be revisited.
    let mut s_start = 0usize;

    for &gt_t in gt_times {
        // Advance s_start past sorted spikes that are before (gt_t - tolerance)
        // and already behind the window.
        while s_start < sorted_times.len() {
            if sorted_times[s_start] + tolerance < gt_t {
                s_start += 1;
            } else {
                break;
            }
        }

        // Search from s_start for the first unmatched sorted spike within tolerance.
        // Because we consume matches by advancing s_start, we need a local scan.
        let mut j = s_start;
        while j < sorted_times.len() {
            let st = sorted_times[j];
            // If sorted spike is past the window, stop.
            if st > gt_t + tolerance {
                break;
            }
            // Check if within tolerance.
            if gt_t.abs_diff(st) <= tolerance {
                tp += 1;
                // Consume this sorted spike so it cannot match again.
                // All sorted spikes before j+1 are now consumed.
                s_start = j + 1;
                break;
            }
            j += 1;
        }
    }

    let fn_count = gt_times.len() - tp;
    let fp_count = sorted_times.len() - tp;
    let total = tp + fn_count + fp_count;

    let accuracy = if total > 0 {
        tp as f64 / total as f64
    } else {
        0.0
    };
    let precision = if tp + fp_count > 0 {
        tp as f64 / (tp + fp_count) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    UnitMatch {
        true_positives: tp,
        false_positives: fp_count,
        false_negatives: fn_count,
        accuracy,
        precision,
        recall,
    }
}

/// Compare all ground-truth units against all sorted units via greedy matching.
///
/// For each ground-truth unit, computes accuracy against every sorted unit and
/// picks the best match. Each sorted unit can be assigned to at most one GT
/// unit (greedy: highest accuracy wins first). This is O(G * S * max(N))
/// where G and S are unit counts and N is max spike train length.
///
/// # Arguments
///
/// * `gt_trains` - Slice of ground-truth spike trains (each sorted ascending)
/// * `sorted_trains` - Slice of sorted spike trains (each sorted ascending)
/// * `tolerance` - Maximum sample distance for a spike match
/// * `output` - Caller-provided buffer; must have length >= `gt_trains.len()`.
///   On return, `output[i]` contains the best match for GT unit `i`.
///
/// # Returns
///
/// Number of GT units that were matched (had accuracy > 0). Returns 0 if
/// `output` is too small or inputs are empty.
///
/// # Example
///
/// ```
/// use zerostone::metrics::{compare_sorting, UnitMatch};
///
/// let gt0: &[usize] = &[100, 200, 300];
/// let gt1: &[usize] = &[150, 250, 350];
/// let s0: &[usize] = &[101, 201, 301];  // matches gt0
/// let s1: &[usize] = &[149, 251, 349];  // matches gt1
///
/// let gt_trains = [gt0, gt1];
/// let sorted_trains = [s0, s1];
/// let mut out = [UnitMatch::empty(); 2];
/// let matched = compare_sorting(&gt_trains, &sorted_trains, 5, &mut out);
/// assert_eq!(matched, 2);
/// assert_eq!(out[0].true_positives, 3);
/// assert_eq!(out[1].true_positives, 3);
/// ```
pub fn compare_sorting(
    gt_trains: &[&[usize]],
    sorted_trains: &[&[usize]],
    tolerance: usize,
    output: &mut [UnitMatch],
) -> usize {
    let n_gt = gt_trains.len();
    if n_gt == 0 || sorted_trains.is_empty() || output.len() < n_gt {
        // Fill output with empty matches for available slots.
        for o in output.iter_mut() {
            *o = UnitMatch::empty();
        }
        return 0;
    }

    // Track which sorted units have been claimed.
    // We support up to 256 sorted units without heap allocation.
    const MAX_SORTED: usize = 256;
    let n_sorted = if sorted_trains.len() <= MAX_SORTED {
        sorted_trains.len()
    } else {
        MAX_SORTED
    };
    let mut claimed = [false; MAX_SORTED];

    // Build a priority list: for each GT unit, find its best sorted unit.
    // We do multiple passes -- first, compute all pairwise accuracies, then
    // greedily assign in order of descending best accuracy.

    // Store best match index and accuracy for each GT unit.
    const MAX_GT: usize = 256;
    let n_gt_capped = if n_gt <= MAX_GT { n_gt } else { MAX_GT };
    let mut best_idx = [0usize; MAX_GT];
    let mut best_acc = [0.0f64; MAX_GT];
    let mut best_match = [UnitMatch::empty(); MAX_GT];

    // For greedy assignment: iteratively find the (gt, sorted) pair with
    // highest accuracy among unclaimed sorted units.
    let mut assigned_gt = [false; MAX_GT];
    let mut matched_count = 0usize;

    for _round in 0..n_gt_capped {
        // Find the GT unit (not yet assigned) with the highest best-accuracy
        // against any unclaimed sorted unit.
        let mut round_best_gt = 0;
        let mut round_best_sorted = 0;
        let mut round_best_acc = -1.0f64;
        let mut round_best_um = UnitMatch::empty();

        for gi in 0..n_gt_capped {
            if assigned_gt[gi] {
                continue;
            }
            for si in 0..n_sorted {
                if claimed[si] {
                    continue;
                }
                let um = compare_spike_trains(gt_trains[gi], sorted_trains[si], tolerance);
                if um.accuracy > round_best_acc {
                    round_best_acc = um.accuracy;
                    round_best_gt = gi;
                    round_best_sorted = si;
                    round_best_um = um;
                }
            }
        }

        if round_best_acc <= 0.0 {
            break;
        }

        assigned_gt[round_best_gt] = true;
        claimed[round_best_sorted] = true;
        best_idx[round_best_gt] = round_best_sorted;
        best_acc[round_best_gt] = round_best_acc;
        best_match[round_best_gt] = round_best_um;
        matched_count += 1;
    }

    // Write results to output.
    for gi in 0..n_gt_capped {
        if assigned_gt[gi] {
            output[gi] = best_match[gi];
        } else {
            // Unmatched GT unit: all spikes are false negatives.
            let fn_count = gt_trains[gi].len();
            output[gi] = UnitMatch {
                true_positives: 0,
                false_positives: 0,
                false_negatives: fn_count,
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
            };
        }
    }

    matched_count
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // compare_spike_trains tests
    // =========================================================================

    #[test]
    fn test_perfect_match() {
        let gt = [100, 200, 300];
        let sorted = [100, 200, 300];
        let m = compare_spike_trains(&gt, &sorted, 0);
        assert_eq!(m.true_positives, 3);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
        assert!((m.accuracy - 1.0).abs() < 1e-10);
        assert!((m.precision - 1.0).abs() < 1e-10);
        assert!((m.recall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_within_tolerance() {
        let gt = [100, 200, 300];
        let sorted = [101, 199, 302];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 3);
        assert!((m.accuracy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_outside_tolerance() {
        let gt = [100, 200, 300];
        let sorted = [110, 220, 330];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_negatives, 3);
        assert_eq!(m.false_positives, 3);
        assert!(m.accuracy < 1e-10);
    }

    #[test]
    fn test_partial_overlap() {
        let gt = [100, 200, 300, 400];
        let sorted = [101, 199, 350, 401];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 3);
        assert_eq!(m.false_negatives, 1); // 300 unmatched
        assert_eq!(m.false_positives, 1); // 350 unmatched
                                          // accuracy = 3 / (3 + 1 + 1) = 0.6
        assert!((m.accuracy - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_tolerance_boundary_exact() {
        // Exactly at tolerance boundary -- should match
        let gt = [100];
        let sorted = [105];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 1);

        // One sample beyond tolerance -- should not match
        let m2 = compare_spike_trains(&gt, &sorted, 4);
        assert_eq!(m2.true_positives, 0);
    }

    #[test]
    fn test_tolerance_boundary_left() {
        // Sorted spike before GT spike at exact boundary
        let gt = [200];
        let sorted = [195];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 1);

        let m2 = compare_spike_trains(&gt, &sorted, 4);
        assert_eq!(m2.true_positives, 0);
    }

    #[test]
    fn test_empty_gt() {
        let gt: [usize; 0] = [];
        let sorted = [100, 200];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_positives, 2);
        assert_eq!(m.false_negatives, 0);
    }

    #[test]
    fn test_empty_sorted() {
        let gt = [100, 200];
        let sorted: [usize; 0] = [];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 2);
    }

    #[test]
    fn test_both_empty() {
        let gt: [usize; 0] = [];
        let sorted: [usize; 0] = [];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
    }

    #[test]
    fn test_one_to_one_no_double_match() {
        // Two GT spikes close together, one sorted spike -- only one match
        let gt = [100, 102];
        let sorted = [101];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 1);
        assert_eq!(m.false_negatives, 1);
        assert_eq!(m.false_positives, 0);
    }

    #[test]
    fn test_many_sorted_one_gt() {
        // One GT spike, multiple sorted spikes nearby -- only one match
        let gt = [100];
        let sorted = [98, 99, 100, 101, 102];
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 1);
        assert_eq!(m.false_positives, 4);
    }

    #[test]
    fn test_zero_tolerance() {
        let gt = [100, 200, 300];
        let sorted = [100, 201, 300];
        let m = compare_spike_trains(&gt, &sorted, 0);
        assert_eq!(m.true_positives, 2);
        assert_eq!(m.false_negatives, 1);
        assert_eq!(m.false_positives, 1);
    }

    #[test]
    fn test_precision_recall_formulas() {
        let gt = [100, 200, 300]; // 3 GT spikes
        let sorted = [101, 201, 500, 600]; // 4 sorted, 2 match
        let m = compare_spike_trains(&gt, &sorted, 5);
        assert_eq!(m.true_positives, 2);
        assert_eq!(m.false_negatives, 1);
        assert_eq!(m.false_positives, 2);
        // precision = 2 / (2 + 2) = 0.5
        assert!((m.precision - 0.5).abs() < 1e-10);
        // recall = 2 / (2 + 1) = 2/3
        assert!((m.recall - 2.0 / 3.0).abs() < 1e-10);
        // accuracy = 2 / (2 + 1 + 2) = 0.4
        assert!((m.accuracy - 0.4).abs() < 1e-10);
    }

    // =========================================================================
    // compare_sorting tests
    // =========================================================================

    #[test]
    fn test_sorting_perfect_two_units() {
        let gt0: &[usize] = &[100, 200, 300];
        let gt1: &[usize] = &[150, 250, 350];
        let s0: &[usize] = &[101, 201, 301];
        let s1: &[usize] = &[149, 251, 349];

        let gt_trains = [gt0, gt1];
        let sorted_trains = [s0, s1];
        let mut out = [UnitMatch::empty(); 2];
        let matched = compare_sorting(&gt_trains, &sorted_trains, 5, &mut out);
        assert_eq!(matched, 2);
        assert_eq!(out[0].true_positives, 3);
        assert_eq!(out[1].true_positives, 3);
    }

    #[test]
    fn test_sorting_swapped_order() {
        // Sorted units are in reverse order relative to GT -- greedy should
        // still find the right assignment.
        let gt0: &[usize] = &[100, 200, 300];
        let gt1: &[usize] = &[1000, 2000, 3000];
        let s0: &[usize] = &[1001, 2001, 3001]; // matches gt1
        let s1: &[usize] = &[101, 201, 301]; // matches gt0

        let gt_trains = [gt0, gt1];
        let sorted_trains = [s0, s1];
        let mut out = [UnitMatch::empty(); 2];
        let matched = compare_sorting(&gt_trains, &sorted_trains, 5, &mut out);
        assert_eq!(matched, 2);
        assert_eq!(out[0].true_positives, 3);
        assert_eq!(out[1].true_positives, 3);
    }

    #[test]
    fn test_sorting_unmatched_gt() {
        // One GT unit, no sorted units match it
        let gt0: &[usize] = &[100, 200, 300];
        let s0: &[usize] = &[1000, 2000, 3000];

        let gt_trains: &[&[usize]] = &[gt0];
        let sorted_trains: &[&[usize]] = &[s0];
        let mut out = [UnitMatch::empty(); 1];
        let matched = compare_sorting(gt_trains, sorted_trains, 5, &mut out);
        assert_eq!(matched, 0);
        assert_eq!(out[0].false_negatives, 3);
    }

    #[test]
    fn test_sorting_empty_gt() {
        let sorted: &[&[usize]] = &[&[100, 200]];
        let gt: &[&[usize]] = &[];
        let mut out: [UnitMatch; 0] = [];
        let matched = compare_sorting(gt, sorted, 5, &mut out);
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_sorting_empty_sorted() {
        let gt: &[&[usize]] = &[&[100, 200]];
        let sorted: &[&[usize]] = &[];
        let mut out = [UnitMatch::empty(); 1];
        let matched = compare_sorting(gt, sorted, 5, &mut out);
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_sorting_more_sorted_than_gt() {
        // 1 GT, 3 sorted -- only one sorted should be claimed
        let gt0: &[usize] = &[100, 200, 300];
        let s0: &[usize] = &[101, 201, 301]; // best match
        let s1: &[usize] = &[105, 205, 305];
        let s2: &[usize] = &[1000, 2000, 3000];

        let gt_trains: &[&[usize]] = &[gt0];
        let sorted_trains: &[&[usize]] = &[s0, s1, s2];
        let mut out = [UnitMatch::empty(); 1];
        let matched = compare_sorting(gt_trains, sorted_trains, 5, &mut out);
        assert_eq!(matched, 1);
        assert_eq!(out[0].true_positives, 3);
    }

    #[test]
    fn test_unit_match_empty() {
        let m = UnitMatch::empty();
        assert_eq!(m.true_positives, 0);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
        assert!(m.accuracy < 1e-10);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `compare_spike_trains` never panics for sorted inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn compare_spike_trains_no_panic() {
        let t0: usize = kani::any();
        let t1: usize = kani::any();
        let s0: usize = kani::any();
        let s1: usize = kani::any();
        let tol: usize = kani::any();

        kani::assume(t0 <= 10000);
        kani::assume(t1 >= t0 && t1 <= 10000);
        kani::assume(s0 <= 10000);
        kani::assume(s1 >= s0 && s1 <= 10000);
        kani::assume(tol <= 1000);

        let gt = [t0, t1];
        let sorted = [s0, s1];
        let m = compare_spike_trains(&gt, &sorted, tol);

        // TP + FN == gt.len()
        assert!(m.true_positives + m.false_negatives == 2);
        // TP + FP == sorted.len()
        assert!(m.true_positives + m.false_positives == 2);
        // Accuracy in [0, 1]
        assert!(m.accuracy >= 0.0 && m.accuracy <= 1.0);
        assert!(m.precision >= 0.0 && m.precision <= 1.0);
        assert!(m.recall >= 0.0 && m.recall <= 1.0);
    }
}
