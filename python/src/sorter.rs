//! Python bindings for the multi-channel sorting pipeline.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::sorter::{sort_multichannel as zs_sort, SortConfig};
use zerostone::spike_sort::{MultiChannelEvent, SortError};

fn sort_error_to_py(e: SortError) -> PyErr {
    let msg = match e {
        SortError::InsufficientData => "Insufficient data for sorting",
        SortError::NotFitted => "Model has not been fitted yet",
        SortError::EigenFailed => "Eigendecomposition failed (whitening or PCA)",
        SortError::TemplateFull => "Maximum number of templates reached",
        SortError::InvalidInput => "Invalid input parameters",
    };
    PyValueError::new_err(msg)
}

/// Full multi-channel sorting pipeline.
///
/// Runs the complete spike sorting pipeline on multi-channel data:
/// noise estimation, spatial whitening, threshold detection, deduplication,
/// peak alignment, waveform extraction, PCA, online k-means clustering,
/// and quality metrics.
///
/// Args:
///     data (np.ndarray): 2D float64 array of shape ``(n_samples, n_channels)``.
///         Supported channel counts: 2, 4, 8, 16, 32, 64.
///     probe (ProbeLayout): Probe geometry for spatial deduplication.
///     threshold (float): Detection threshold in MAD units. Default: 5.0.
///     refractory (int): Minimum samples between detections per channel. Default: 15.
///     spatial_radius (float): Deduplication radius in micrometers. Default: 75.0.
///     temporal_radius (int): Deduplication radius in samples. Default: 5.
///     align_half_window (int): Half-window for fine peak alignment. Default: 5.
///     pre_samples (int): Samples before peak in extracted waveforms. Default: 16.
///     cluster_threshold (float): Distance threshold for creating new clusters. Default: 3.0.
///     cluster_max_count (int): Maximum observation count per cluster centroid. Default: 1000.
///     whitening_epsilon (float): Regularization for whitening eigenvalues. Default: 1e-6.
///     merge_dprime_threshold (float): D-prime threshold for cluster merging. Default: 1.5.
///     merge_isi_threshold (float): ISI violation threshold for cluster merging. Default: 0.05.
///     split_min_cluster_size (int): Minimum spikes per cluster to attempt splitting. Default: 10.
///     split_bimodality_threshold (float): Gap/std threshold for cluster splitting. Default: 2.0.
///
/// Returns:
///     dict: Sorting results with keys:
///         - ``"n_spikes"`` (int): Total spikes detected.
///         - ``"n_clusters"`` (int): Number of clusters found.
///         - ``"labels"`` (np.ndarray): 1D int array of cluster labels per spike.
///         - ``"spike_times"`` (np.ndarray): 1D int64 array of sample indices per spike.
///         - ``"spike_channels"`` (np.ndarray): 1D int64 array of peak channel per spike.
///         - ``"clusters"`` (list[dict]): Per-cluster metrics, each with
///           ``"count"`` (int), ``"snr"`` (float), ``"isi_violation_rate"`` (float).
///
/// Example:
///     >>> import numpy as np
///     >>> import zpybci as zbci
///     >>> probe = zbci.ProbeLayout.linear(4, 25.0)
///     >>> data = np.random.randn(5000, 4)
///     >>> result = zbci.sort_multichannel(data, probe, threshold=4.0)
///     >>> result["n_spikes"]
///     0
#[pyfunction]
#[pyo3(signature = (
    data,
    probe,
    threshold = 5.0,
    refractory = 15,
    spatial_radius = 75.0,
    temporal_radius = 5,
    align_half_window = 5,
    pre_samples = 16,
    cluster_threshold = 3.0,
    cluster_max_count = 1000,
    whitening_epsilon = 1e-6,
    merge_dprime_threshold = 1.5,
    merge_isi_threshold = 0.05,
    split_min_cluster_size = 10,
    split_bimodality_threshold = 2.0,
))]
#[allow(clippy::too_many_arguments)]
fn sort_multichannel<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    probe: &super::probe::ProbeLayout,
    threshold: f64,
    refractory: usize,
    spatial_radius: f64,
    temporal_radius: usize,
    align_half_window: usize,
    pre_samples: usize,
    cluster_threshold: f64,
    cluster_max_count: u32,
    whitening_epsilon: f64,
    merge_dprime_threshold: f64,
    merge_isi_threshold: f64,
    split_min_cluster_size: usize,
    split_bimodality_threshold: f64,
) -> PyResult<PyObject> {
    let shape = data.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let data_slice = data.as_slice()?;

    let config = SortConfig {
        threshold_multiplier: threshold,
        refractory_samples: refractory,
        spatial_radius_um: spatial_radius,
        temporal_radius,
        align_half_window,
        pre_samples,
        cluster_threshold,
        cluster_max_count,
        whitening_epsilon,
        merge_dprime_threshold,
        merge_isi_threshold,
        split_min_cluster_size,
        split_bimodality_threshold,
    };

    // W=32, K=3, WM=1024, N=16
    const W: usize = 32;
    const K: usize = 3;
    const N: usize = 16;

    // Upper bound on events
    let max_events = n_samples / refractory.max(1) + n_channels;

    macro_rules! do_sort {
        ($c:expr, $cm:expr) => {{
            // Safety: reinterpret flat row-major slice as &mut [[f64; $c]]
            assert!(data_slice.len() == n_samples * $c);
            let mut data_owned: Vec<[f64; $c]> = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut row = [0.0f64; $c];
                row.copy_from_slice(&data_slice[i * $c..(i + 1) * $c]);
                data_owned.push(row);
            }

            let mut scratch = vec![0.0f64; n_samples];
            let mut event_buf = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0,
                };
                max_events
            ];
            let mut waveform_buf = vec![[0.0f64; W]; max_events];
            let mut feature_buf = vec![[0.0f64; K]; max_events];
            let mut labels = vec![0usize; max_events];

            let result = super::probe::with_probe_ref::<$c, _, _>(&probe, |zs_probe| {
                zs_sort::<$c, $cm, W, K, 1024, N>(
                    &config,
                    zs_probe,
                    &mut data_owned,
                    &mut scratch,
                    &mut event_buf,
                    &mut waveform_buf,
                    &mut feature_buf,
                    &mut labels,
                )
            })?;

            let sr = result.map_err(sort_error_to_py)?;
            build_result_dict(py, &sr, &labels, &event_buf[..sr.n_spikes])
        }};
    }

    match n_channels {
        2 => do_sort!(2, 4),
        4 => do_sort!(4, 16),
        8 => do_sort!(8, 64),
        16 => do_sort!(16, 256),
        32 => do_sort!(32, 1024),
        64 => do_sort!(64, 4096),
        _ => Err(PyValueError::new_err(
            "n_channels must be 2, 4, 8, 16, 32, or 64",
        )),
    }
}

/// Build the Python result dict from a SortResult.
fn build_result_dict<const N: usize>(
    py: Python<'_>,
    sr: &zerostone::sorter::SortResult<N>,
    labels: &[usize],
    events: &[MultiChannelEvent],
) -> PyResult<PyObject> {
    use pyo3::types::{PyDict, PyList};

    let dict = PyDict::new(py);
    dict.set_item("n_spikes", sr.n_spikes)?;
    dict.set_item("n_clusters", sr.n_clusters)?;

    // Labels array (only the first n_spikes entries are meaningful)
    let label_data: Vec<i64> = labels[..sr.n_spikes].iter().map(|&l| l as i64).collect();
    let label_array = Array1::from_vec(label_data);
    let label_py = PyArray1::from_owned_array(py, label_array);
    dict.set_item("labels", label_py)?;

    // Spike times array (sample indices from event_buf)
    let times_data: Vec<i64> = events.iter().map(|e| e.sample as i64).collect();
    let times_array = Array1::from_vec(times_data);
    let times_py = PyArray1::from_owned_array(py, times_array);
    dict.set_item("spike_times", times_py)?;

    // Spike channels array (peak channel per spike)
    let ch_data: Vec<i64> = events.iter().map(|e| e.channel as i64).collect();
    let ch_array = Array1::from_vec(ch_data);
    let ch_py = PyArray1::from_owned_array(py, ch_array);
    dict.set_item("spike_channels", ch_py)?;

    // Clusters list of dicts
    let clusters_list = PyList::empty(py);
    for ci in 0..sr.n_clusters {
        let cl = PyDict::new(py);
        cl.set_item("count", sr.clusters[ci].count)?;
        cl.set_item("snr", sr.clusters[ci].snr)?;
        cl.set_item("isi_violation_rate", sr.clusters[ci].isi_violation_rate)?;
        clusters_list.append(cl)?;
    }
    dict.set_item("clusters", clusters_list)?;

    Ok(dict.into())
}

/// Online spike sorter for real-time template matching.
///
/// After learning templates from batch sorting, classifies new spikes
/// by nearest-centroid distance in feature space.
///
/// Example:
///     >>> sorter = zbci.OnlineSorter()
///     >>> sorter.add_template([1.0, 0.0, 0.0])
///     >>> sorter.add_template([0.0, 1.0, 0.0])
///     >>> label, dist = sorter.classify([0.9, 0.1, 0.0])
#[pyclass]
pub struct OnlineSorter {
    inner: zerostone::sorter::OnlineSorter<3, 16>,
}

#[pymethods]
impl OnlineSorter {
    #[new]
    fn new() -> Self {
        Self {
            inner: zerostone::sorter::OnlineSorter::<3, 16>::new(),
        }
    }

    /// Create from a list of centroid arrays.
    #[staticmethod]
    fn from_centroids(centroids: Vec<Vec<f64>>) -> PyResult<Self> {
        let mut fixed: Vec<[f64; 3]> = Vec::new();
        for c in &centroids {
            if c.len() != 3 {
                return Err(PyValueError::new_err("Each centroid must have 3 elements"));
            }
            fixed.push([c[0], c[1], c[2]]);
        }
        Ok(Self {
            inner: zerostone::sorter::OnlineSorter::<3, 16>::from_centroids(&fixed),
        })
    }

    /// Add a template centroid. Returns template index or None if full.
    fn add_template(&mut self, centroid: Vec<f64>) -> PyResult<Option<usize>> {
        if centroid.len() != 3 {
            return Err(PyValueError::new_err("Centroid must have 3 elements"));
        }
        let arr = [centroid[0], centroid[1], centroid[2]];
        Ok(self.inner.add_template(&arr))
    }

    /// Set max distance for rejection.
    fn set_max_distance(&mut self, max_dist: f64) {
        self.inner.set_max_distance(max_dist);
    }

    /// Classify a spike. Returns (label, distance).
    fn classify(&mut self, features: Vec<f64>) -> PyResult<(usize, f64)> {
        if features.len() != 3 {
            return Err(PyValueError::new_err("Features must have 3 elements"));
        }
        let arr = [features[0], features[1], features[2]];
        Ok(self.inner.classify(&arr))
    }

    /// Classify or reject. Returns (label, distance) or None.
    fn classify_or_reject(&mut self, features: Vec<f64>) -> PyResult<Option<(usize, f64)>> {
        if features.len() != 3 {
            return Err(PyValueError::new_err("Features must have 3 elements"));
        }
        let arr = [features[0], features[1], features[2]];
        Ok(self.inner.classify_or_reject(&arr))
    }

    #[getter]
    fn n_templates(&self) -> usize {
        self.inner.n_templates()
    }

    #[getter]
    fn n_classified(&self) -> usize {
        self.inner.n_classified()
    }

    #[getter]
    fn n_rejected(&self) -> usize {
        self.inner.n_rejected()
    }

    fn reset_counters(&mut self) {
        self.inner.reset_counters();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineSorter(n_templates={}, classified={}, rejected={})",
            self.inner.n_templates(),
            self.inner.n_classified(),
            self.inner.n_rejected()
        )
    }
}

/// Register sorting pipeline functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort_multichannel, m)?)?;
    m.add_class::<OnlineSorter>()?;
    Ok(())
}
