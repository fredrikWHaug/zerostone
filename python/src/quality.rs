//! Python bindings for cluster quality metrics.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use zerostone::quality;

/// Compute the ISI violation rate for a sorted spike train.
///
/// Counts the fraction of consecutive inter-spike intervals that fall below
/// the biological refractory period.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times (seconds).
///     refractory_period (float): Refractory period threshold (same units).
///
/// Returns:
///     float or None: Violation rate in [0, 1], or None if fewer than 2 spikes.
#[pyfunction]
fn isi_violation_rate(
    spike_times: PyReadonlyArray1<f64>,
    refractory_period: f64,
) -> PyResult<Option<f64>> {
    let times = spike_times.as_slice()?;
    Ok(quality::isi_violation_rate(times, refractory_period))
}

/// Estimate contamination rate from ISI violations (Hill et al. 2011).
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times (seconds).
///     refractory_period (float): Refractory period threshold (seconds).
///     recording_duration (float): Total recording duration (seconds).
///
/// Returns:
///     float or None: Contamination rate in [0, 1], or None if insufficient data.
#[pyfunction]
fn contamination_rate(
    spike_times: PyReadonlyArray1<f64>,
    refractory_period: f64,
    recording_duration: f64,
) -> PyResult<Option<f64>> {
    let times = spike_times.as_slice()?;
    Ok(quality::contamination_rate(
        times,
        refractory_period,
        recording_duration,
    ))
}

/// Compute the silhouette score for a single point.
///
/// Measures how similar a point is to its own cluster compared to the nearest
/// other cluster: (b - a) / max(a, b).
///
/// Args:
///     intra_distances (np.ndarray): 1D float64 distances to same-cluster points.
///     inter_distances (np.ndarray): 1D float64 mean distances to each other cluster.
///
/// Returns:
///     float or None: Score in [-1, 1], or None if inputs are empty.
#[pyfunction]
fn silhouette_score(
    intra_distances: PyReadonlyArray1<f64>,
    inter_distances: PyReadonlyArray1<f64>,
) -> PyResult<Option<f64>> {
    let intra = intra_distances.as_slice()?;
    let inter = inter_distances.as_slice()?;
    Ok(quality::silhouette_score(intra, inter))
}

/// Compute the mean silhouette score for a cluster.
///
/// Args:
///     n_points (int): Number of points in the cluster.
///     intra_dist_flat (np.ndarray): 1D float64 flattened intra-cluster distances.
///     n_intra (int): Number of intra-cluster distances per point.
///     inter_means_flat (np.ndarray): 1D float64 flattened mean inter-cluster distances.
///     n_clusters (int): Number of other clusters.
///
/// Returns:
///     float or None: Mean score in [-1, 1], or None if inputs are invalid.
#[pyfunction]
fn mean_silhouette(
    n_points: usize,
    intra_dist_flat: PyReadonlyArray1<f64>,
    n_intra: usize,
    inter_means_flat: PyReadonlyArray1<f64>,
    n_clusters: usize,
) -> PyResult<Option<f64>> {
    let intra = intra_dist_flat.as_slice()?;
    let inter = inter_means_flat.as_slice()?;
    Ok(quality::mean_silhouette(
        n_points, intra, n_intra, inter, n_clusters,
    ))
}

/// Compute the signal-to-noise ratio of a mean waveform.
///
/// SNR = (peak-to-peak amplitude) / (2 * noise_std).
///
/// Args:
///     mean_waveform (np.ndarray): 1D float64 mean waveform.
///     noise_std (float): Background noise standard deviation.
///
/// Returns:
///     float or None: SNR (non-negative), or None if waveform is empty or noise_std <= 0.
#[pyfunction]
fn waveform_snr(mean_waveform: PyReadonlyArray1<f64>, noise_std: f64) -> PyResult<Option<f64>> {
    let wf = mean_waveform.as_slice()?;
    Ok(quality::waveform_snr(wf, noise_std))
}

/// Compute d-prime (discriminability index) between two clusters.
///
/// d' = |mu_a - mu_b| / sqrt(0.5 * (var_a + var_b))
///
/// Args:
///     cluster_a (np.ndarray): 1D float64 feature values for cluster A.
///     cluster_b (np.ndarray): 1D float64 feature values for cluster B.
///
/// Returns:
///     float or None: d-prime (non-negative), or None if either cluster has < 2 elements.
#[pyfunction]
fn d_prime(
    cluster_a: PyReadonlyArray1<f64>,
    cluster_b: PyReadonlyArray1<f64>,
) -> PyResult<Option<f64>> {
    let a = cluster_a.as_slice()?;
    let b = cluster_b.as_slice()?;
    Ok(quality::d_prime(a, b))
}

/// Compute isolation distance for a cluster in feature space.
///
/// The N_c-th smallest squared Mahalanobis distance from non-cluster spikes
/// to the cluster centroid, where N_c is the cluster size.
///
/// Args:
///     n_cluster (int): Number of spikes in the cluster.
///     other_mahal_sq (np.ndarray): 1D float64 squared Mahalanobis distances
///         from non-cluster spikes (will be copied and sorted internally).
///
/// Returns:
///     float or None: Isolation distance, or None if insufficient non-cluster spikes.
#[pyfunction]
fn isolation_distance(
    n_cluster: usize,
    other_mahal_sq: PyReadonlyArray1<f64>,
) -> PyResult<Option<f64>> {
    let slice = other_mahal_sq.as_slice()?;
    let mut owned = slice.to_vec();
    Ok(quality::isolation_distance(n_cluster, &mut owned))
}

/// Compute Euclidean distance between two feature vectors.
///
/// Args:
///     a (np.ndarray): 1D float64 first feature vector.
///     b (np.ndarray): 1D float64 second feature vector.
///
/// Returns:
///     float: Euclidean distance.
#[pyfunction]
fn euclidean_distance(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    Ok(quality::euclidean_distance(a_slice, b_slice))
}

/// Register quality metrics functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(isi_violation_rate, m)?)?;
    m.add_function(wrap_pyfunction!(contamination_rate, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(mean_silhouette, m)?)?;
    m.add_function(wrap_pyfunction!(waveform_snr, m)?)?;
    m.add_function(wrap_pyfunction!(d_prime, m)?)?;
    m.add_function(wrap_pyfunction!(isolation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    Ok(())
}
