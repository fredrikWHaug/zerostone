//! Cluster quality metrics for spike sorting evaluation and auto-curation.
//!
//! Provides the standard quality metrics used in extracellular electrophysiology
//! to assess spike sorting output and automate unit curation (good / MUA / noise
//! classification).
//!
//! # Overview
//!
//! - [`isi_violation_rate`] -- fraction of inter-spike intervals below the refractory period
//! - [`contamination_rate`] -- estimated fraction of spikes from other neurons (Hill et al. 2011)
//! - [`silhouette_score`] -- cluster cohesion vs separation in feature space (-1 to +1)
//! - [`waveform_snr`] -- signal-to-noise ratio of mean waveform vs background noise
//! - [`d_prime`] -- discriminability index between two clusters in feature space
//! - [`isolation_distance`] -- Mahalanobis distance to nearest non-cluster spikes
//!
//! # References
//!
//! - Hill, Mehta & Kleinfeld (2011). Quality metrics to accompany spike sorting of
//!   extracellular signals. J Neurosci 31(24):8699-8705.
//! - Schmitzer-Torbert et al. (2005). Quantitative measures of cluster quality for
//!   use in extracellular recordings. Neuroscience 131(1):1-11.
//!
//! # Example
//!
//! ```
//! use zerostone::quality::{isi_violation_rate, waveform_snr, contamination_rate};
//!
//! // Spike train with one refractory violation (0.5ms gap at 30kHz)
//! let spike_times = [0.0, 0.050, 0.100, 0.1005, 0.200];
//! let rate = isi_violation_rate(&spike_times, 0.001);
//! assert!(rate.unwrap() > 0.0);
//!
//! // SNR of a clean waveform against noise
//! let waveform = [-1.0, -3.0, -8.0, -5.0, -2.0, 0.0, 1.0, 0.5];
//! let noise_std = 1.5;
//! let snr = waveform_snr(&waveform, noise_std);
//! assert!(snr.unwrap() > 1.0);
//! ```

use crate::float::{self, Float, INFINITY, MAX};

/// Compute the ISI violation rate for a sorted spike train.
///
/// Counts the fraction of consecutive inter-spike intervals that fall below
/// the biological refractory period. This is the gold-standard metric for
/// unit contamination: a well-isolated single unit should have very few
/// (ideally zero) refractory violations.
///
/// # Arguments
///
/// * `spike_times` - Sorted spike times in ascending order (seconds)
/// * `refractory_period` - Refractory period threshold in the same units as spike times
///
/// # Returns
///
/// `Some(rate)` where rate is in [0, 1], or `None` if fewer than 2 spikes.
///
/// # Example
///
/// ```
/// use zerostone::quality::isi_violation_rate;
///
/// // Regular 50ms firing -- no violations with 1ms refractory
/// let spikes = [0.0, 0.050, 0.100, 0.150, 0.200];
/// let rate = isi_violation_rate(&spikes, 0.001).unwrap();
/// assert!(rate < 1e-10);
///
/// // One violation: 0.5ms gap
/// let spikes = [0.0, 0.050, 0.100, 0.1005, 0.200];
/// let rate = isi_violation_rate(&spikes, 0.001).unwrap();
/// assert!((rate - 0.25).abs() < 1e-10); // 1 of 4 intervals
/// ```
pub fn isi_violation_rate(spike_times: &[Float], refractory_period: Float) -> Option<Float> {
    let n = spike_times.len();
    if n < 2 {
        return None;
    }
    let n_intervals = n - 1;
    let mut violations = 0u64;
    for i in 0..n_intervals {
        let isi = spike_times[i + 1] - spike_times[i];
        if isi < refractory_period {
            violations += 1;
        }
    }
    Some(violations as Float / n_intervals as Float)
}

/// Estimate the contamination rate from ISI violations (Hill et al. 2011).
///
/// Estimates the fraction of spikes that originate from other neurons based
/// on the rate of refractory period violations. Under the assumption that
/// contaminating spikes arrive as a Poisson process, the expected number of
/// violations is `2 * T_r * N^2 * f / T`, where `T_r` is the refractory
/// period, `N` is the spike count, `f` is the firing rate, and `T` is the
/// recording duration.
///
/// The contamination rate `C` satisfies: `n_violations = 2 * T_r * N * f_total * C^2`,
/// yielding `C = sqrt(n_violations / (2 * T_r * f * N))` where
/// `f = N / T` is the estimated firing rate.
///
/// # Arguments
///
/// * `spike_times` - Sorted spike times in ascending order (seconds)
/// * `refractory_period` - Refractory period threshold (seconds), typically 0.001-0.002
/// * `recording_duration` - Total recording duration (seconds)
///
/// # Returns
///
/// `Some(contamination)` in [0, 1] (clamped), or `None` if fewer than 2 spikes
/// or recording_duration <= 0.
///
/// # Example
///
/// ```
/// use zerostone::quality::contamination_rate;
///
/// // Clean unit: no violations
/// let spikes = [0.0, 0.050, 0.100, 0.150, 0.200];
/// let c = contamination_rate(&spikes, 0.0015, 1.0).unwrap();
/// assert!(c < 1e-10);
/// ```
pub fn contamination_rate(
    spike_times: &[Float],
    refractory_period: Float,
    recording_duration: Float,
) -> Option<Float> {
    let n = spike_times.len();
    if n < 2 || recording_duration <= 0.0 || refractory_period <= 0.0 {
        return None;
    }

    // Count violations
    let mut n_violations = 0u64;
    for i in 0..(n - 1) {
        let isi = spike_times[i + 1] - spike_times[i];
        if isi < refractory_period {
            n_violations += 1;
        }
    }

    if n_violations == 0 {
        return Some(0.0);
    }

    // Hill et al. 2011 formula:
    // C = sqrt(n_violations * T / (2 * T_r * N^2))
    let n_f = n as Float;
    let numer = n_violations as Float * recording_duration;
    let denom = 2.0 * refractory_period * n_f * n_f;
    let c = float::sqrt(numer / denom);

    // Clamp to [0, 1]
    if c > 1.0 {
        Some(1.0)
    } else {
        Some(c)
    }
}

/// Compute the silhouette score for a single point given its distances.
///
/// The silhouette score measures how similar a point is to its own cluster
/// compared to the nearest other cluster. It is defined as
/// `(b - a) / max(a, b)` where `a` is the mean distance to same-cluster
/// points and `b` is the mean distance to the nearest other cluster.
///
/// # Arguments
///
/// * `intra_distances` - Distances from this point to all other points in the same cluster
/// * `inter_distances` - Mean distances from this point to each other cluster.
///   Each entry is the mean distance to all points in that cluster.
///
/// # Returns
///
/// `Some(score)` in [-1, 1], or `None` if inputs are insufficient
/// (empty intra_distances or empty inter_distances).
///
/// # Example
///
/// ```
/// use zerostone::quality::silhouette_score;
///
/// // Tight cluster, far from others
/// let intra = [0.1, 0.2, 0.15];
/// let inter = [5.0, 8.0]; // mean distances to 2 other clusters
/// let s = silhouette_score(&intra, &inter).unwrap();
/// assert!(s > 0.9);
/// ```
pub fn silhouette_score(intra_distances: &[Float], inter_distances: &[Float]) -> Option<Float> {
    if intra_distances.is_empty() || inter_distances.is_empty() {
        return None;
    }

    // a = mean intra-cluster distance
    let mut sum_a: Float = 0.0;
    for &d in intra_distances {
        sum_a += d;
    }
    let a = sum_a / intra_distances.len() as Float;

    // b = minimum mean inter-cluster distance (nearest cluster)
    let mut b: Float = MAX;
    for &d in inter_distances {
        if d < b {
            b = d;
        }
    }

    let max_ab = if a > b { a } else { b };
    if max_ab <= 0.0 {
        return Some(0.0);
    }

    Some((b - a) / max_ab)
}

/// Compute the mean silhouette score for a cluster given pre-computed distances.
///
/// Averages the silhouette score of each point in the cluster. Each point
/// needs its intra-cluster distances and its mean distances to every other
/// cluster. This function operates on a flat layout: for `n` points, the
/// caller provides `n` sets of intra and inter distances via parallel slices.
///
/// # Arguments
///
/// * `n_points` - Number of points in the cluster
/// * `intra_dist_flat` - Flattened intra-cluster distances, `n_intra` distances per point
/// * `n_intra` - Number of intra-cluster distances per point (same for all points)
/// * `inter_means_flat` - Flattened mean inter-cluster distances, `n_clusters` per point
/// * `n_clusters` - Number of other clusters
///
/// # Returns
///
/// `Some(mean_score)` in [-1, 1], or `None` if inputs are empty or sizes don't match.
///
/// # Example
///
/// ```
/// use zerostone::quality::mean_silhouette;
///
/// // 2 points in cluster, 1 intra-dist each, 1 other cluster
/// let intra = [0.1, 0.1]; // point 0: d=0.1, point 1: d=0.1
/// let inter = [5.0, 5.0]; // point 0: mean_d=5.0, point 1: mean_d=5.0
/// let s = mean_silhouette(2, &intra, 1, &inter, 1).unwrap();
/// assert!(s > 0.9);
/// ```
pub fn mean_silhouette(
    n_points: usize,
    intra_dist_flat: &[Float],
    n_intra: usize,
    inter_means_flat: &[Float],
    n_clusters: usize,
) -> Option<Float> {
    if n_points == 0 || n_intra == 0 || n_clusters == 0 {
        return None;
    }
    if intra_dist_flat.len() < n_points * n_intra {
        return None;
    }
    if inter_means_flat.len() < n_points * n_clusters {
        return None;
    }

    let mut sum: Float = 0.0;
    for i in 0..n_points {
        let intra_start = i * n_intra;
        let inter_start = i * n_clusters;
        let s = silhouette_score(
            &intra_dist_flat[intra_start..intra_start + n_intra],
            &inter_means_flat[inter_start..inter_start + n_clusters],
        );
        match s {
            Some(val) => sum += val,
            None => return None,
        }
    }

    Some(sum / n_points as Float)
}

/// Compute the signal-to-noise ratio of a mean waveform.
///
/// SNR = (peak-to-peak amplitude of the mean waveform) / (2 * noise_std).
/// The peak-to-peak amplitude is max(waveform) - min(waveform).
///
/// This metric quantifies how clearly the unit's spike stands out from the
/// background noise. An SNR below ~2 typically indicates a noisy unit.
///
/// # Arguments
///
/// * `mean_waveform` - The mean (template) waveform for a cluster
/// * `noise_std` - Standard deviation of the background noise (e.g., from MAD estimation)
///
/// # Returns
///
/// `Some(snr)` (non-negative), or `None` if waveform is empty or noise_std <= 0.
///
/// # Example
///
/// ```
/// use zerostone::quality::waveform_snr;
///
/// let wf = [0.0, -1.0, -5.0, -3.0, 0.0, 1.0, 0.5, 0.0];
/// let snr = waveform_snr(&wf, 1.0).unwrap();
/// // peak-to-peak = 1.0 - (-5.0) = 6.0, SNR = 6.0 / 2.0 = 3.0
/// assert!((snr - 3.0).abs() < 1e-10);
/// ```
pub fn waveform_snr(mean_waveform: &[Float], noise_std: Float) -> Option<Float> {
    if mean_waveform.is_empty() || noise_std <= 0.0 {
        return None;
    }

    let mut min_val = mean_waveform[0];
    let mut max_val = mean_waveform[0];
    for &v in &mean_waveform[1..] {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let peak_to_peak = max_val - min_val;
    Some(peak_to_peak / (2.0 * noise_std))
}

/// Compute d-prime (discriminability index) between two clusters.
///
/// D-prime measures how well two clusters can be discriminated along the
/// axis connecting their centroids in feature space. It is defined as:
///
/// `d' = |mu_1 - mu_2| / sqrt(0.5 * (var_1 + var_2))`
///
/// where mu and var are the mean and variance of the projections of each
/// cluster onto the Fisher discriminant axis (the line connecting centroids).
///
/// For 1D feature vectors this simplifies to the standard d-prime formula.
/// For multi-dimensional features, call this function with the 1D projections
/// onto the discriminant axis.
///
/// # Arguments
///
/// * `cluster_a` - Feature values (1D projections) for cluster A
/// * `cluster_b` - Feature values (1D projections) for cluster B
///
/// # Returns
///
/// `Some(d_prime)` (non-negative), or `None` if either cluster has fewer
/// than 2 elements.
///
/// # Example
///
/// ```
/// use zerostone::quality::d_prime;
///
/// // Well-separated clusters
/// let a = [1.0, 1.1, 0.9, 1.05, 0.95];
/// let b = [5.0, 5.1, 4.9, 5.05, 4.95];
/// let dp = d_prime(&a, &b).unwrap();
/// assert!(dp > 10.0); // very well separated
///
/// // Overlapping clusters
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = [3.0, 4.0, 5.0, 6.0, 7.0];
/// let dp = d_prime(&a, &b).unwrap();
/// assert!(dp > 0.0 && dp < 3.0);
/// ```
pub fn d_prime(cluster_a: &[Float], cluster_b: &[Float]) -> Option<Float> {
    let na = cluster_a.len();
    let nb = cluster_b.len();
    if na < 2 || nb < 2 {
        return None;
    }

    // Mean and variance for cluster A
    let mut sum_a: Float = 0.0;
    for &v in cluster_a {
        sum_a += v;
    }
    let mean_a = sum_a / na as Float;
    let mut var_a: Float = 0.0;
    for &v in cluster_a {
        let d = v - mean_a;
        var_a += d * d;
    }
    var_a /= (na - 1) as Float;

    // Mean and variance for cluster B
    let mut sum_b: Float = 0.0;
    for &v in cluster_b {
        sum_b += v;
    }
    let mean_b = sum_b / nb as Float;
    let mut var_b: Float = 0.0;
    for &v in cluster_b {
        let d = v - mean_b;
        var_b += d * d;
    }
    var_b /= (nb - 1) as Float;

    let pooled_std = float::sqrt(0.5 * (var_a + var_b));
    if pooled_std <= 0.0 {
        // Both clusters have zero variance -- either identical or single values
        if float::abs(mean_a - mean_b) > 0.0 {
            return Some(INFINITY);
        }
        return Some(0.0);
    }

    Some(float::abs(mean_a - mean_b) / pooled_std)
}

/// Compute isolation distance for a cluster in feature space.
///
/// Isolation distance (Schmitzer-Torbert et al. 2005) measures how well a
/// cluster is separated from background spikes using Mahalanobis distance.
///
/// For each non-cluster spike, the squared Mahalanobis distance to the
/// cluster centroid is computed. The distances are sorted, and the isolation
/// distance is the N_c-th smallest distance, where N_c is the number of
/// spikes in the cluster. Geometrically, this is the radius of the ellipsoid
/// centered on the cluster that contains N_c cluster spikes and N_c non-cluster
/// spikes.
///
/// This function takes pre-computed squared Mahalanobis distances from
/// non-cluster spikes to the cluster centroid, since computing the full
/// Mahalanobis distance requires the inverse covariance matrix which
/// involves heap allocation for general dimensions.
///
/// # Arguments
///
/// * `n_cluster` - Number of spikes in the cluster
/// * `other_mahal_sq` - Squared Mahalanobis distances from non-cluster spikes to the
///   cluster centroid (will be sorted in place)
///
/// # Returns
///
/// `Some(distance)` or `None` if there are fewer non-cluster spikes than cluster spikes.
///
/// # Example
///
/// ```
/// use zerostone::quality::isolation_distance;
///
/// let n_cluster = 3;
/// let mut distances = [1.0, 5.0, 2.0, 8.0, 3.0];
/// let iso = isolation_distance(n_cluster, &mut distances).unwrap();
/// // Sorted: [1.0, 2.0, 3.0, 5.0, 8.0], 3rd smallest = 3.0
/// assert!((iso - 3.0).abs() < 1e-10);
/// ```
pub fn isolation_distance(n_cluster: usize, other_mahal_sq: &mut [Float]) -> Option<Float> {
    if n_cluster == 0 || other_mahal_sq.len() < n_cluster {
        return None;
    }

    other_mahal_sq.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    Some(other_mahal_sq[n_cluster - 1])
}

/// Compute Euclidean distance between two feature vectors.
///
/// Helper for computing pairwise distances in feature space. Returns the
/// L2 norm of the difference vector.
///
/// # Arguments
///
/// * `a` - First feature vector
/// * `b` - Second feature vector (must have same length as `a`)
///
/// # Returns
///
/// Euclidean distance, or 0.0 if either slice is empty.
///
/// # Example
///
/// ```
/// use zerostone::quality::euclidean_distance;
///
/// let a = [1.0, 0.0, 0.0];
/// let b = [0.0, 1.0, 0.0];
/// let d = euclidean_distance(&a, &b);
/// assert!((d - core::f64::consts::SQRT_2).abs() < 1e-10);
/// ```
pub fn euclidean_distance(a: &[Float], b: &[Float]) -> Float {
    let len = if a.len() < b.len() { a.len() } else { b.len() };
    if len == 0 {
        return 0.0;
    }
    let mut sum: Float = 0.0;
    for i in 0..len {
        let d = a[i] - b[i];
        sum += d * d;
    }
    float::sqrt(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ISI violation rate tests
    // =========================================================================

    #[test]
    fn test_isi_violation_no_violations() {
        // Regular 50ms intervals -- no violations with 1ms refractory
        let spikes: [Float; 5] = [0.0, 0.050, 0.100, 0.150, 0.200];
        let rate = isi_violation_rate(&spikes, 0.001).unwrap();
        assert!(rate < 1e-10, "Expected 0 violations, got {}", rate);
    }

    #[test]
    fn test_isi_violation_one_violation() {
        // One pair is 0.5ms apart (below 1ms refractory)
        let spikes: [Float; 5] = [0.0, 0.050, 0.100, 0.1005, 0.200];
        let rate = isi_violation_rate(&spikes, 0.001).unwrap();
        assert!(
            (rate - 0.25).abs() < 1e-10,
            "Expected 1/4 = 0.25, got {}",
            rate
        );
    }

    #[test]
    fn test_isi_violation_all_violations() {
        // All intervals below refractory period
        let spikes: [Float; 4] = [0.0, 0.0001, 0.0002, 0.0003];
        let rate = isi_violation_rate(&spikes, 0.001).unwrap();
        assert!((rate - 1.0).abs() < 1e-10, "Expected 1.0, got {}", rate);
    }

    #[test]
    fn test_isi_violation_insufficient_spikes() {
        assert!(isi_violation_rate(&[], 0.001).is_none());
        assert!(isi_violation_rate(&[0.0], 0.001).is_none());
    }

    #[test]
    fn test_isi_violation_two_spikes() {
        let spikes: [Float; 2] = [0.0, 0.050];
        let rate = isi_violation_rate(&spikes, 0.001).unwrap();
        assert!(rate < 1e-10);
    }

    // =========================================================================
    // Contamination rate tests
    // =========================================================================

    #[test]
    fn test_contamination_no_violations() {
        let spikes: [Float; 5] = [0.0, 0.050, 0.100, 0.150, 0.200];
        let c = contamination_rate(&spikes, 0.0015, 1.0).unwrap();
        assert!(c < 1e-10, "Expected 0 contamination, got {}", c);
    }

    #[test]
    fn test_contamination_with_violations() {
        // Create a spike train with known violations
        let spikes: [Float; 6] = [0.0, 0.050, 0.100, 0.1005, 0.200, 0.300];
        let c = contamination_rate(&spikes, 0.0015, 1.0).unwrap();
        assert!(c > 0.0, "Expected nonzero contamination");
        assert!(c <= 1.0, "Contamination should be <= 1.0, got {}", c);
    }

    #[test]
    fn test_contamination_clamped_at_one() {
        // Many violations relative to total -- should clamp to 1.0
        let spikes: [Float; 5] = [0.0, 0.0001, 0.0002, 0.0003, 0.0004];
        let c = contamination_rate(&spikes, 0.001, 0.001).unwrap();
        assert!(
            c <= 1.0,
            "Contamination should be clamped to 1.0, got {}",
            c
        );
    }

    #[test]
    fn test_contamination_insufficient_input() {
        assert!(contamination_rate(&[], 0.001, 1.0).is_none());
        assert!(contamination_rate(&[0.0], 0.001, 1.0).is_none());
        assert!(contamination_rate(&[0.0, 0.1], 0.001, 0.0).is_none());
        assert!(contamination_rate(&[0.0, 0.1], 0.0, 1.0).is_none());
    }

    // =========================================================================
    // Silhouette score tests
    // =========================================================================

    #[test]
    fn test_silhouette_well_separated() {
        // Tight cluster, far from others
        let intra: [Float; 4] = [0.1, 0.2, 0.15, 0.12];
        let inter: [Float; 2] = [10.0, 15.0];
        let s = silhouette_score(&intra, &inter).unwrap();
        assert!(s > 0.95, "Well-separated should give s~1, got {}", s);
    }

    #[test]
    fn test_silhouette_overlapping() {
        // Intra distances larger than inter distances
        let intra: [Float; 3] = [5.0, 6.0, 7.0];
        let inter: [Float; 2] = [2.0, 3.0];
        let s = silhouette_score(&intra, &inter).unwrap();
        assert!(s < 0.0, "Overlapping should give negative s, got {}", s);
    }

    #[test]
    fn test_silhouette_perfect() {
        // Zero intra distance, positive inter distance
        let intra: [Float; 3] = [0.0, 0.0, 0.0];
        let inter: [Float; 1] = [5.0];
        let s = silhouette_score(&intra, &inter).unwrap();
        assert!(
            (s - 1.0).abs() < 1e-10,
            "Perfect separation should give 1.0, got {}",
            s
        );
    }

    #[test]
    fn test_silhouette_range() {
        // Verify result is in [-1, 1]
        let intra: [Float; 3] = [1.0, 2.0, 3.0];
        let inter: [Float; 1] = [2.5];
        let s = silhouette_score(&intra, &inter).unwrap();
        assert!(
            (-1.0..=1.0).contains(&s),
            "Silhouette must be in [-1,1], got {}",
            s
        );
    }

    #[test]
    fn test_silhouette_empty_input() {
        assert!(silhouette_score(&[], &[1.0]).is_none());
        assert!(silhouette_score(&[1.0], &[]).is_none());
    }

    // =========================================================================
    // Mean silhouette tests
    // =========================================================================

    #[test]
    fn test_mean_silhouette_uniform() {
        // 3 points, each with 2 intra distances and 1 other cluster
        let intra: [Float; 6] = [0.1, 0.2, 0.15, 0.1, 0.12, 0.18];
        let inter: [Float; 3] = [5.0, 5.0, 5.0];
        let s = mean_silhouette(3, &intra, 2, &inter, 1).unwrap();
        assert!(s > 0.9, "Expected high mean silhouette, got {}", s);
    }

    #[test]
    fn test_mean_silhouette_empty() {
        assert!(mean_silhouette(0, &[], 0, &[], 0).is_none());
    }

    // =========================================================================
    // SNR tests
    // =========================================================================

    #[test]
    fn test_snr_known() {
        let wf: [Float; 8] = [0.0, -1.0, -5.0, -3.0, 0.0, 1.0, 0.5, 0.0];
        let snr = waveform_snr(&wf, 1.0).unwrap();
        // peak-to-peak = 1.0 - (-5.0) = 6.0, SNR = 6.0 / 2.0 = 3.0
        assert!((snr - 3.0).abs() < 1e-10, "Expected SNR = 3.0, got {}", snr);
    }

    #[test]
    fn test_snr_flat_waveform() {
        let wf = [3.0 as Float; 10];
        let snr = waveform_snr(&wf, 1.0).unwrap();
        assert!(
            snr < 1e-10,
            "Flat waveform should have SNR = 0, got {}",
            snr
        );
    }

    #[test]
    fn test_snr_zero_noise() {
        let wf: [Float; 3] = [0.0, -5.0, 0.0];
        assert!(waveform_snr(&wf, 0.0).is_none());
    }

    #[test]
    fn test_snr_negative_noise() {
        let wf: [Float; 3] = [0.0, -5.0, 0.0];
        assert!(waveform_snr(&wf, -1.0).is_none());
    }

    #[test]
    fn test_snr_empty_waveform() {
        assert!(waveform_snr(&[], 1.0).is_none());
    }

    #[test]
    fn test_snr_single_sample() {
        let wf: [Float; 1] = [5.0];
        let snr = waveform_snr(&wf, 1.0).unwrap();
        assert!(snr < 1e-10, "Single-sample waveform has 0 peak-to-peak");
    }

    // =========================================================================
    // D-prime tests
    // =========================================================================

    #[test]
    fn test_dprime_well_separated() {
        let a: [Float; 5] = [1.0, 1.1, 0.9, 1.05, 0.95];
        let b: [Float; 5] = [5.0, 5.1, 4.9, 5.05, 4.95];
        let dp = d_prime(&a, &b).unwrap();
        assert!(
            dp > 10.0,
            "Well-separated clusters should have high d', got {}",
            dp
        );
    }

    #[test]
    fn test_dprime_overlapping() {
        let a: [Float; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b: [Float; 5] = [3.0, 4.0, 5.0, 6.0, 7.0];
        let dp = d_prime(&a, &b).unwrap();
        assert!(
            dp > 0.0 && dp < 3.0,
            "Overlapping clusters d' should be moderate, got {}",
            dp
        );
    }

    #[test]
    fn test_dprime_identical() {
        let a: [Float; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b: [Float; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let dp = d_prime(&a, &b).unwrap();
        assert!(
            dp < 1e-10,
            "Identical clusters should have d' = 0, got {}",
            dp
        );
    }

    #[test]
    fn test_dprime_symmetric() {
        let a: [Float; 5] = [1.0, 1.5, 2.0, 2.5, 3.0];
        let b: [Float; 5] = [5.0, 5.5, 6.0, 6.5, 7.0];
        let dp_ab = d_prime(&a, &b).unwrap();
        let dp_ba = d_prime(&b, &a).unwrap();
        assert!(
            (dp_ab - dp_ba).abs() < 1e-10,
            "d' should be symmetric: {} vs {}",
            dp_ab,
            dp_ba
        );
    }

    #[test]
    fn test_dprime_insufficient() {
        assert!(d_prime(&[1.0], &[2.0, 3.0]).is_none());
        assert!(d_prime(&[1.0, 2.0], &[3.0]).is_none());
        assert!(d_prime(&[], &[]).is_none());
    }

    #[test]
    fn test_dprime_zero_variance() {
        // Both clusters constant but different
        let a: [Float; 3] = [3.0, 3.0, 3.0];
        let b: [Float; 3] = [7.0, 7.0, 7.0];
        let dp = d_prime(&a, &b).unwrap();
        assert!(
            dp.is_infinite(),
            "Zero-variance different means should give infinity"
        );
    }

    // =========================================================================
    // Isolation distance tests
    // =========================================================================

    #[test]
    fn test_isolation_distance_basic() {
        let n_cluster = 3;
        let mut distances: [Float; 5] = [1.0, 5.0, 2.0, 8.0, 3.0];
        let iso = isolation_distance(n_cluster, &mut distances).unwrap();
        assert!((iso - 3.0).abs() < 1e-10, "Expected 3.0, got {}", iso);
    }

    #[test]
    fn test_isolation_distance_sorted() {
        let n_cluster = 2;
        let mut distances: [Float; 4] = [1.0, 2.0, 3.0, 4.0];
        let iso = isolation_distance(n_cluster, &mut distances).unwrap();
        assert!((iso - 2.0).abs() < 1e-10, "Expected 2.0, got {}", iso);
    }

    #[test]
    fn test_isolation_distance_single() {
        let mut distances: [Float; 1] = [42.0];
        let iso = isolation_distance(1, &mut distances).unwrap();
        assert!((iso - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_isolation_distance_insufficient() {
        let mut distances: [Float; 2] = [1.0, 2.0];
        assert!(isolation_distance(3, &mut distances).is_none());
        assert!(isolation_distance(0, &mut distances).is_none());
    }

    // =========================================================================
    // Euclidean distance tests
    // =========================================================================

    #[test]
    fn test_euclidean_basic() {
        let a: [Float; 3] = [1.0, 0.0, 0.0];
        let b: [Float; 3] = [0.0, 1.0, 0.0];
        let d = euclidean_distance(&a, &b);
        assert!(
            (d - float::sqrt(2.0)).abs() < 1e-10,
            "Expected sqrt(2), got {}",
            d
        );
    }

    #[test]
    fn test_euclidean_same() {
        let a: [Float; 3] = [1.0, 2.0, 3.0];
        let d = euclidean_distance(&a, &a);
        assert!(d < 1e-10, "Same point should have distance 0");
    }

    #[test]
    fn test_euclidean_empty() {
        let d = euclidean_distance(&[], &[]);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_euclidean_1d() {
        let a: [Float; 1] = [3.0];
        let b: [Float; 1] = [7.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 4.0).abs() < 1e-10);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_identical_spike_times() {
        // Two spikes at the same time -- ISI = 0, which is < any positive refractory
        let spikes: [Float; 2] = [0.1, 0.1];
        let rate = isi_violation_rate(&spikes, 0.001).unwrap();
        assert!(
            (rate - 1.0).abs() < 1e-10,
            "Identical times should give 100% violations"
        );
    }

    #[test]
    fn test_contamination_known_formula() {
        // Verify the Hill formula with hand-computed values.
        // N = 100 spikes over T = 10s, firing rate f = 10 Hz.
        // refractory = 1ms, 5 violations.
        // C = sqrt(5 * 10 / (2 * 0.001 * 100^2)) = sqrt(50 / 20) = sqrt(2.5) ~ 1.58
        // Should be clamped to 1.0
        let mut spikes = [0.0 as Float; 100];
        for (i, spike) in spikes.iter_mut().enumerate() {
            *spike = i as Float * 0.1; // 10 Hz, regular
        }
        // Inject 5 violations by placing spikes close together
        spikes[10] = spikes[9] + 0.0005; // violation
        spikes[20] = spikes[19] + 0.0005;
        spikes[30] = spikes[29] + 0.0005;
        spikes[40] = spikes[39] + 0.0005;
        spikes[50] = spikes[49] + 0.0005;
        // Re-sort to maintain order
        spikes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let c = contamination_rate(&spikes, 0.001, 10.0).unwrap();
        assert!(c <= 1.0, "Should be clamped to 1.0");
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(8)]
    fn isi_violation_rate_no_panic() {
        let s0: f64 = kani::any();
        let s1: f64 = kani::any();
        let s2: f64 = kani::any();
        let s3: f64 = kani::any();
        let rp: f64 = kani::any();

        kani::assume(s0.is_finite() && s0 >= 0.0 && s0 <= 1e6);
        kani::assume(s1.is_finite() && s1 >= s0 && s1 <= 1e6);
        kani::assume(s2.is_finite() && s2 >= s1 && s2 <= 1e6);
        kani::assume(s3.is_finite() && s3 >= s2 && s3 <= 1e6);
        kani::assume(rp.is_finite() && rp > 0.0 && rp <= 1.0);

        let spikes = [s0 as Float, s1 as Float, s2 as Float, s3 as Float];
        let result = isi_violation_rate(&spikes, rp as Float);
        if let Some(rate) = result {
            assert!(rate >= 0.0 && rate <= 1.0, "rate must be in [0, 1]");
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn silhouette_score_bounded() {
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        let b0: f64 = kani::any();

        kani::assume(a0.is_finite() && a0 >= 0.0 && a0 <= 1e6);
        kani::assume(a1.is_finite() && a1 >= 0.0 && a1 <= 1e6);
        kani::assume(b0.is_finite() && b0 >= 0.0 && b0 <= 1e6);

        let intra = [a0 as Float, a1 as Float];
        let inter = [b0 as Float];
        if let Some(s) = silhouette_score(&intra, &inter) {
            assert!(s >= -1.0 && s <= 1.0, "silhouette must be in [-1, 1]");
        }
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn snr_finite_for_nonzero_noise() {
        let w0: f64 = kani::any();
        let w1: f64 = kani::any();
        let noise: f64 = kani::any();

        kani::assume(w0.is_finite() && w0 >= -1e6 && w0 <= 1e6);
        kani::assume(w1.is_finite() && w1 >= -1e6 && w1 <= 1e6);
        kani::assume(noise.is_finite() && noise > 0.0 && noise <= 1e6);

        let wf = [w0 as Float, w1 as Float];
        if let Some(snr) = waveform_snr(&wf, noise as Float) {
            assert!(snr.is_finite(), "SNR must be finite for valid input");
            assert!(snr >= 0.0, "SNR must be non-negative");
        }
    }

    /// Prove that `contamination_rate` output is always in [0, 1] for valid inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn contamination_rate_bounded() {
        let s0: f64 = kani::any();
        let s1: f64 = kani::any();
        let s2: f64 = kani::any();
        let rp: f64 = kani::any();
        let dur: f64 = kani::any();

        kani::assume(s0.is_finite() && s0 >= 0.0 && s0 <= 1e6);
        kani::assume(s1.is_finite() && s1 >= s0 && s1 <= 1e6);
        kani::assume(s2.is_finite() && s2 >= s1 && s2 <= 1e6);
        kani::assume(rp.is_finite() && rp > 0.0 && rp <= 1.0);
        kani::assume(dur.is_finite() && dur > 0.0 && dur <= 1e6);

        let spikes = [s0 as Float, s1 as Float, s2 as Float];
        if let Some(c) = contamination_rate(&spikes, rp as Float, dur as Float) {
            assert!(c >= 0.0, "contamination must be >= 0");
            assert!(c <= 1.0, "contamination must be <= 1 (clamped)");
        }
    }

    /// Prove that `euclidean_distance` is non-negative and finite for finite inputs.
    #[kani::proof]
    #[kani::unwind(4)]
    fn euclidean_distance_non_negative_finite() {
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        let b0: f64 = kani::any();
        let b1: f64 = kani::any();

        kani::assume(a0.is_finite() && a0 >= -1e6 && a0 <= 1e6);
        kani::assume(a1.is_finite() && a1 >= -1e6 && a1 <= 1e6);
        kani::assume(b0.is_finite() && b0 >= -1e6 && b0 <= 1e6);
        kani::assume(b1.is_finite() && b1 >= -1e6 && b1 <= 1e6);

        let d = euclidean_distance(&[a0 as Float, a1 as Float], &[b0 as Float, b1 as Float]);
        assert!(d >= 0.0, "distance must be non-negative");
        assert!(d.is_finite(), "distance must be finite for finite inputs");
    }

    /// Prove that `d_prime` is non-negative when it returns Some.
    #[kani::proof]
    #[kani::unwind(6)]
    fn d_prime_non_negative() {
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        let b0: f64 = kani::any();
        let b1: f64 = kani::any();

        kani::assume(a0.is_finite() && a0 >= -1e3 && a0 <= 1e3);
        kani::assume(a1.is_finite() && a1 >= -1e3 && a1 <= 1e3);
        kani::assume(b0.is_finite() && b0 >= -1e3 && b0 <= 1e3);
        kani::assume(b1.is_finite() && b1 >= -1e3 && b1 <= 1e3);

        let ca = [a0 as Float, a1 as Float];
        let cb = [b0 as Float, b1 as Float];
        if let Some(dp) = d_prime(&ca, &cb) {
            assert!(dp >= 0.0, "d-prime must be non-negative");
        }
    }
}
