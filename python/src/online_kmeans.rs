//! Python bindings for online k-means clustering.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zerostone::online_kmeans::{KMeansError, OnlineKMeans as ZsOnlineKMeans};

fn kmeans_error_to_py(e: KMeansError) -> PyErr {
    let msg = match e {
        KMeansError::ClustersFull => "All cluster slots are full",
        KMeansError::InvalidInput => "Invalid input parameter",
    };
    PyValueError::new_err(msg)
}

// --- Macro for const-generic dispatch ---

macro_rules! make_kmeans_inner {
    ($name:ident, $d:expr, $k:expr) => {
        #[allow(non_camel_case_types)]
        struct $name(ZsOnlineKMeans<$d, $k>);

        #[allow(dead_code)]
        impl $name {
            fn new(max_count: u32) -> Self {
                Self(ZsOnlineKMeans::new(max_count))
            }

            fn set_create_threshold(&mut self, t: f64) {
                self.0.set_create_threshold(t);
            }

            fn set_merge_threshold(&mut self, t: f64) {
                self.0.set_merge_threshold(t);
            }

            fn update(&mut self, point: &[f64]) -> (usize, f64, bool) {
                let mut p = [0.0f64; $d];
                p.copy_from_slice(point);
                let r = self.0.update(&p);
                (r.cluster, r.distance, r.created)
            }

            fn predict(&self, point: &[f64]) -> (usize, f64) {
                let mut p = [0.0f64; $d];
                p.copy_from_slice(point);
                self.0.predict(&p)
            }

            fn merge_closest(&mut self) -> Option<(usize, usize)> {
                self.0.merge_closest()
            }

            fn remove_cluster(&mut self, idx: usize) {
                self.0.remove_cluster(idx);
            }

            fn seed_centroid(&mut self, centroid: &[f64]) -> Result<usize, KMeansError> {
                let mut c = [0.0f64; $d];
                c.copy_from_slice(centroid);
                self.0.seed_centroid(&c)
            }

            fn centroid(&self, idx: usize) -> Option<Vec<f64>> {
                self.0.centroid(idx).map(|c| c.to_vec())
            }

            fn cluster_variance(&self, idx: usize) -> Option<Vec<f64>> {
                self.0.cluster_variance(idx).map(|v| v.to_vec())
            }

            fn n_active(&self) -> usize {
                self.0.n_active()
            }

            fn count(&self, idx: usize) -> u32 {
                self.0.count(idx)
            }

            fn reset(&mut self) {
                self.0.reset();
            }

            fn dimensions(&self) -> usize {
                $d
            }

            fn max_clusters(&self) -> usize {
                $k
            }

            fn centroids_flat(&self) -> Vec<f64> {
                let na = self.0.n_active();
                let mut flat = Vec::with_capacity(na * $d);
                for i in 0..na {
                    if let Some(c) = self.0.centroid(i) {
                        flat.extend_from_slice(c);
                    }
                }
                flat
            }

            fn counts_vec(&self) -> Vec<u32> {
                let na = self.0.n_active();
                (0..na).map(|i| self.0.count(i)).collect()
            }
        }
    };
}

// D=2, K={4,8,16,32}
make_kmeans_inner!(KM_D2K4, 2, 4);
make_kmeans_inner!(KM_D2K8, 2, 8);
make_kmeans_inner!(KM_D2K16, 2, 16);
make_kmeans_inner!(KM_D2K32, 2, 32);
// D=3, K={4,8,16,32}
make_kmeans_inner!(KM_D3K4, 3, 4);
make_kmeans_inner!(KM_D3K8, 3, 8);
make_kmeans_inner!(KM_D3K16, 3, 16);
make_kmeans_inner!(KM_D3K32, 3, 32);
// D=5, K={4,8,16,32}
make_kmeans_inner!(KM_D5K4, 5, 4);
make_kmeans_inner!(KM_D5K8, 5, 8);
make_kmeans_inner!(KM_D5K16, 5, 16);
make_kmeans_inner!(KM_D5K32, 5, 32);
// D=8, K={4,8,16,32}
make_kmeans_inner!(KM_D8K4, 8, 4);
make_kmeans_inner!(KM_D8K8, 8, 8);
make_kmeans_inner!(KM_D8K16, 8, 16);
make_kmeans_inner!(KM_D8K32, 8, 32);

enum KMeansInner {
    D2K4(KM_D2K4),
    D2K8(KM_D2K8),
    D2K16(KM_D2K16),
    D2K32(KM_D2K32),
    D3K4(KM_D3K4),
    D3K8(KM_D3K8),
    D3K16(KM_D3K16),
    D3K32(KM_D3K32),
    D5K4(KM_D5K4),
    D5K8(KM_D5K8),
    D5K16(KM_D5K16),
    D5K32(KM_D5K32),
    D8K4(KM_D8K4),
    D8K8(KM_D8K8),
    D8K16(KM_D8K16),
    D8K32(KM_D8K32),
}

/// Online k-means clustering for spike sorting.
///
/// Implements MacQueen-style sequential online k-means with adaptive cluster
/// creation. Operates point-by-point, O(K*D) per update, deterministic.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// km = zbci.OnlineKMeans(dimensions=3, max_clusters=8, create_threshold=5.0)
/// result = km.update(np.array([1.0, 2.0, 3.0]))
/// print(result)  # {'cluster': 0, 'distance': 0.0, 'created': True}
/// print(km.n_active)  # 1
/// ```
#[pyclass]
pub struct OnlineKMeans {
    inner: KMeansInner,
    dims: usize,
    max_k: usize,
}

#[pymethods]
impl OnlineKMeans {
    /// Create a new OnlineKMeans clusterer.
    ///
    /// Args:
    ///     dimensions (int): Feature dimensionality (2, 3, 5, or 8).
    ///     max_clusters (int): Maximum number of clusters (4, 8, 16, or 32).
    ///     max_count (int): Count cap for drift adaptation. Default: 10000.
    ///     create_threshold (float): Distance above which a new cluster is created. Default: None (infinite).
    ///     merge_threshold (float): Distance below which clusters are merged. Default: None (0.0).
    #[new]
    #[pyo3(signature = (dimensions, max_clusters, max_count=10000, create_threshold=None, merge_threshold=None))]
    fn new(
        dimensions: usize,
        max_clusters: usize,
        max_count: u32,
        create_threshold: Option<f64>,
        merge_threshold: Option<f64>,
    ) -> PyResult<Self> {
        macro_rules! make_inner {
            ($variant:ident, $ty:ident) => {{
                let mut km = $ty::new(max_count);
                if let Some(t) = create_threshold {
                    km.set_create_threshold(t);
                }
                if let Some(t) = merge_threshold {
                    km.set_merge_threshold(t);
                }
                KMeansInner::$variant(km)
            }};
        }

        let inner = match (dimensions, max_clusters) {
            (2, 4) => make_inner!(D2K4, KM_D2K4),
            (2, 8) => make_inner!(D2K8, KM_D2K8),
            (2, 16) => make_inner!(D2K16, KM_D2K16),
            (2, 32) => make_inner!(D2K32, KM_D2K32),
            (3, 4) => make_inner!(D3K4, KM_D3K4),
            (3, 8) => make_inner!(D3K8, KM_D3K8),
            (3, 16) => make_inner!(D3K16, KM_D3K16),
            (3, 32) => make_inner!(D3K32, KM_D3K32),
            (5, 4) => make_inner!(D5K4, KM_D5K4),
            (5, 8) => make_inner!(D5K8, KM_D5K8),
            (5, 16) => make_inner!(D5K16, KM_D5K16),
            (5, 32) => make_inner!(D5K32, KM_D5K32),
            (8, 4) => make_inner!(D8K4, KM_D8K4),
            (8, 8) => make_inner!(D8K8, KM_D8K8),
            (8, 16) => make_inner!(D8K16, KM_D8K16),
            (8, 32) => make_inner!(D8K32, KM_D8K32),
            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported (dimensions, max_clusters) combination. \
                     dimensions: 2, 3, 5, 8; max_clusters: 4, 8, 16, 32",
                ));
            }
        };

        Ok(Self {
            inner,
            dims: dimensions,
            max_k: max_clusters,
        })
    }

    /// Update with a single point.
    ///
    /// Args:
    ///     point (np.ndarray): 1D float64 array of length `dimensions`.
    ///
    /// Returns:
    ///     dict: {"cluster": int, "distance": float, "created": bool}
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = point.as_slice()?;
        if p.len() != self.dims {
            return Err(PyValueError::new_err(format!(
                "point has {} elements, expected {}",
                p.len(),
                self.dims
            )));
        }

        macro_rules! do_update {
            ($km:expr) => {{
                $km.update(p)
            }};
        }

        let (cluster, distance, created) = match &mut self.inner {
            KMeansInner::D2K4(km) => do_update!(km),
            KMeansInner::D2K8(km) => do_update!(km),
            KMeansInner::D2K16(km) => do_update!(km),
            KMeansInner::D2K32(km) => do_update!(km),
            KMeansInner::D3K4(km) => do_update!(km),
            KMeansInner::D3K8(km) => do_update!(km),
            KMeansInner::D3K16(km) => do_update!(km),
            KMeansInner::D3K32(km) => do_update!(km),
            KMeansInner::D5K4(km) => do_update!(km),
            KMeansInner::D5K8(km) => do_update!(km),
            KMeansInner::D5K16(km) => do_update!(km),
            KMeansInner::D5K32(km) => do_update!(km),
            KMeansInner::D8K4(km) => do_update!(km),
            KMeansInner::D8K8(km) => do_update!(km),
            KMeansInner::D8K16(km) => do_update!(km),
            KMeansInner::D8K32(km) => do_update!(km),
        };

        let dict = PyDict::new(py);
        dict.set_item("cluster", cluster)?;
        dict.set_item("distance", distance)?;
        dict.set_item("created", created)?;
        Ok(dict)
    }

    /// Update with a batch of points.
    ///
    /// Args:
    ///     points (np.ndarray): 2D float64 array of shape (n_points, dimensions).
    ///
    /// Returns:
    ///     tuple: (labels, distances) - 1D int and 1D float arrays of length n_points.
    fn update_batch<'py>(
        &mut self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let shape = points.shape();
        if shape[1] != self.dims {
            return Err(PyValueError::new_err(format!(
                "points have {} columns, expected {}",
                shape[1], self.dims
            )));
        }
        let flat = points.as_slice()?;
        let n = shape[0];
        let d = self.dims;
        let mut labels = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);

        macro_rules! do_batch {
            ($km:expr) => {{
                for i in 0..n {
                    let (cluster, dist, _) = $km.update(&flat[i * d..(i + 1) * d]);
                    labels.push(cluster as i64);
                    distances.push(dist);
                }
            }};
        }

        match &mut self.inner {
            KMeansInner::D2K4(km) => do_batch!(km),
            KMeansInner::D2K8(km) => do_batch!(km),
            KMeansInner::D2K16(km) => do_batch!(km),
            KMeansInner::D2K32(km) => do_batch!(km),
            KMeansInner::D3K4(km) => do_batch!(km),
            KMeansInner::D3K8(km) => do_batch!(km),
            KMeansInner::D3K16(km) => do_batch!(km),
            KMeansInner::D3K32(km) => do_batch!(km),
            KMeansInner::D5K4(km) => do_batch!(km),
            KMeansInner::D5K8(km) => do_batch!(km),
            KMeansInner::D5K16(km) => do_batch!(km),
            KMeansInner::D5K32(km) => do_batch!(km),
            KMeansInner::D8K4(km) => do_batch!(km),
            KMeansInner::D8K8(km) => do_batch!(km),
            KMeansInner::D8K16(km) => do_batch!(km),
            KMeansInner::D8K32(km) => do_batch!(km),
        }

        let label_arr = PyArray1::from_owned_array(py, Array1::from_vec(labels));
        let dist_arr = PyArray1::from_owned_array(py, Array1::from_vec(distances));
        Ok((label_arr, dist_arr))
    }

    /// Predict cluster assignments without modifying state.
    ///
    /// Args:
    ///     points (np.ndarray): 2D float64 array of shape (n_points, dimensions).
    ///
    /// Returns:
    ///     tuple: (labels, distances) - 1D int and 1D float arrays of length n_points.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let shape = points.shape();
        if shape[1] != self.dims {
            return Err(PyValueError::new_err(format!(
                "points have {} columns, expected {}",
                shape[1], self.dims
            )));
        }
        let flat = points.as_slice()?;
        let n = shape[0];
        let d = self.dims;
        let mut labels = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);

        macro_rules! do_predict {
            ($km:expr) => {{
                for i in 0..n {
                    let (cluster, dist) = $km.predict(&flat[i * d..(i + 1) * d]);
                    labels.push(cluster as i64);
                    distances.push(dist);
                }
            }};
        }

        match &self.inner {
            KMeansInner::D2K4(km) => do_predict!(km),
            KMeansInner::D2K8(km) => do_predict!(km),
            KMeansInner::D2K16(km) => do_predict!(km),
            KMeansInner::D2K32(km) => do_predict!(km),
            KMeansInner::D3K4(km) => do_predict!(km),
            KMeansInner::D3K8(km) => do_predict!(km),
            KMeansInner::D3K16(km) => do_predict!(km),
            KMeansInner::D3K32(km) => do_predict!(km),
            KMeansInner::D5K4(km) => do_predict!(km),
            KMeansInner::D5K8(km) => do_predict!(km),
            KMeansInner::D5K16(km) => do_predict!(km),
            KMeansInner::D5K32(km) => do_predict!(km),
            KMeansInner::D8K4(km) => do_predict!(km),
            KMeansInner::D8K8(km) => do_predict!(km),
            KMeansInner::D8K16(km) => do_predict!(km),
            KMeansInner::D8K32(km) => do_predict!(km),
        }

        let label_arr = PyArray1::from_owned_array(py, Array1::from_vec(labels));
        let dist_arr = PyArray1::from_owned_array(py, Array1::from_vec(distances));
        Ok((label_arr, dist_arr))
    }

    /// Merge the two closest clusters if below merge threshold.
    ///
    /// Returns:
    ///     tuple or None: (kept_index, removed_index) if merge occurred.
    fn merge_closest(&mut self) -> Option<(usize, usize)> {
        macro_rules! do_merge {
            ($km:expr) => {{
                $km.merge_closest()
            }};
        }

        match &mut self.inner {
            KMeansInner::D2K4(km) => do_merge!(km),
            KMeansInner::D2K8(km) => do_merge!(km),
            KMeansInner::D2K16(km) => do_merge!(km),
            KMeansInner::D2K32(km) => do_merge!(km),
            KMeansInner::D3K4(km) => do_merge!(km),
            KMeansInner::D3K8(km) => do_merge!(km),
            KMeansInner::D3K16(km) => do_merge!(km),
            KMeansInner::D3K32(km) => do_merge!(km),
            KMeansInner::D5K4(km) => do_merge!(km),
            KMeansInner::D5K8(km) => do_merge!(km),
            KMeansInner::D5K16(km) => do_merge!(km),
            KMeansInner::D5K32(km) => do_merge!(km),
            KMeansInner::D8K4(km) => do_merge!(km),
            KMeansInner::D8K8(km) => do_merge!(km),
            KMeansInner::D8K16(km) => do_merge!(km),
            KMeansInner::D8K32(km) => do_merge!(km),
        }
    }

    /// Remove a cluster by index.
    ///
    /// Args:
    ///     idx (int): Cluster index to remove.
    fn remove_cluster(&mut self, idx: usize) -> PyResult<()> {
        macro_rules! do_remove {
            ($km:expr) => {{
                if idx >= $km.n_active() {
                    return Err(PyValueError::new_err("cluster index out of range"));
                }
                $km.remove_cluster(idx);
            }};
        }

        match &mut self.inner {
            KMeansInner::D2K4(km) => do_remove!(km),
            KMeansInner::D2K8(km) => do_remove!(km),
            KMeansInner::D2K16(km) => do_remove!(km),
            KMeansInner::D2K32(km) => do_remove!(km),
            KMeansInner::D3K4(km) => do_remove!(km),
            KMeansInner::D3K8(km) => do_remove!(km),
            KMeansInner::D3K16(km) => do_remove!(km),
            KMeansInner::D3K32(km) => do_remove!(km),
            KMeansInner::D5K4(km) => do_remove!(km),
            KMeansInner::D5K8(km) => do_remove!(km),
            KMeansInner::D5K16(km) => do_remove!(km),
            KMeansInner::D5K32(km) => do_remove!(km),
            KMeansInner::D8K4(km) => do_remove!(km),
            KMeansInner::D8K8(km) => do_remove!(km),
            KMeansInner::D8K16(km) => do_remove!(km),
            KMeansInner::D8K32(km) => do_remove!(km),
        }
        Ok(())
    }

    /// Manually seed a centroid.
    ///
    /// Args:
    ///     centroid (np.ndarray): 1D float64 array of length `dimensions`.
    ///
    /// Returns:
    ///     int: Index of the new cluster.
    fn seed_centroid(&mut self, centroid: PyReadonlyArray1<f64>) -> PyResult<usize> {
        let c = centroid.as_slice()?;
        if c.len() != self.dims {
            return Err(PyValueError::new_err(format!(
                "centroid has {} elements, expected {}",
                c.len(),
                self.dims
            )));
        }

        macro_rules! do_seed {
            ($km:expr) => {{
                $km.seed_centroid(c).map_err(kmeans_error_to_py)
            }};
        }

        match &mut self.inner {
            KMeansInner::D2K4(km) => do_seed!(km),
            KMeansInner::D2K8(km) => do_seed!(km),
            KMeansInner::D2K16(km) => do_seed!(km),
            KMeansInner::D2K32(km) => do_seed!(km),
            KMeansInner::D3K4(km) => do_seed!(km),
            KMeansInner::D3K8(km) => do_seed!(km),
            KMeansInner::D3K16(km) => do_seed!(km),
            KMeansInner::D3K32(km) => do_seed!(km),
            KMeansInner::D5K4(km) => do_seed!(km),
            KMeansInner::D5K8(km) => do_seed!(km),
            KMeansInner::D5K16(km) => do_seed!(km),
            KMeansInner::D5K32(km) => do_seed!(km),
            KMeansInner::D8K4(km) => do_seed!(km),
            KMeansInner::D8K8(km) => do_seed!(km),
            KMeansInner::D8K16(km) => do_seed!(km),
            KMeansInner::D8K32(km) => do_seed!(km),
        }
    }

    /// Active centroids as 2D array of shape (n_active, dimensions).
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        macro_rules! get_centroids {
            ($km:expr) => {{
                $km.centroids_flat()
            }};
        }

        let flat = match &self.inner {
            KMeansInner::D2K4(km) => get_centroids!(km),
            KMeansInner::D2K8(km) => get_centroids!(km),
            KMeansInner::D2K16(km) => get_centroids!(km),
            KMeansInner::D2K32(km) => get_centroids!(km),
            KMeansInner::D3K4(km) => get_centroids!(km),
            KMeansInner::D3K8(km) => get_centroids!(km),
            KMeansInner::D3K16(km) => get_centroids!(km),
            KMeansInner::D3K32(km) => get_centroids!(km),
            KMeansInner::D5K4(km) => get_centroids!(km),
            KMeansInner::D5K8(km) => get_centroids!(km),
            KMeansInner::D5K16(km) => get_centroids!(km),
            KMeansInner::D5K32(km) => get_centroids!(km),
            KMeansInner::D8K4(km) => get_centroids!(km),
            KMeansInner::D8K8(km) => get_centroids!(km),
            KMeansInner::D8K16(km) => get_centroids!(km),
            KMeansInner::D8K32(km) => get_centroids!(km),
        };

        let na = self.n_active_inner();
        if na == 0 {
            let arr = Array2::<f64>::zeros((0, self.dims));
            return Ok(PyArray2::from_owned_array(py, arr));
        }
        let arr = Array2::from_shape_vec((na, self.dims), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Number of active clusters.
    #[getter]
    fn n_active(&self) -> usize {
        self.n_active_inner()
    }

    /// Observation counts as 1D array of length n_active.
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        macro_rules! get_counts {
            ($km:expr) => {{
                $km.counts_vec()
            }};
        }

        let counts = match &self.inner {
            KMeansInner::D2K4(km) => get_counts!(km),
            KMeansInner::D2K8(km) => get_counts!(km),
            KMeansInner::D2K16(km) => get_counts!(km),
            KMeansInner::D2K32(km) => get_counts!(km),
            KMeansInner::D3K4(km) => get_counts!(km),
            KMeansInner::D3K8(km) => get_counts!(km),
            KMeansInner::D3K16(km) => get_counts!(km),
            KMeansInner::D3K32(km) => get_counts!(km),
            KMeansInner::D5K4(km) => get_counts!(km),
            KMeansInner::D5K8(km) => get_counts!(km),
            KMeansInner::D5K16(km) => get_counts!(km),
            KMeansInner::D5K32(km) => get_counts!(km),
            KMeansInner::D8K4(km) => get_counts!(km),
            KMeansInner::D8K8(km) => get_counts!(km),
            KMeansInner::D8K16(km) => get_counts!(km),
            KMeansInner::D8K32(km) => get_counts!(km),
        };
        PyArray1::from_owned_array(py, Array1::from_vec(counts))
    }

    /// Per-dimension variance for a cluster.
    ///
    /// Args:
    ///     idx (int): Cluster index.
    ///
    /// Returns:
    ///     np.ndarray or None: 1D float64 array of length `dimensions`, or None if unavailable.
    fn cluster_variance<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        macro_rules! get_var {
            ($km:expr) => {{
                $km.cluster_variance(idx)
            }};
        }

        let var = match &self.inner {
            KMeansInner::D2K4(km) => get_var!(km),
            KMeansInner::D2K8(km) => get_var!(km),
            KMeansInner::D2K16(km) => get_var!(km),
            KMeansInner::D2K32(km) => get_var!(km),
            KMeansInner::D3K4(km) => get_var!(km),
            KMeansInner::D3K8(km) => get_var!(km),
            KMeansInner::D3K16(km) => get_var!(km),
            KMeansInner::D3K32(km) => get_var!(km),
            KMeansInner::D5K4(km) => get_var!(km),
            KMeansInner::D5K8(km) => get_var!(km),
            KMeansInner::D5K16(km) => get_var!(km),
            KMeansInner::D5K32(km) => get_var!(km),
            KMeansInner::D8K4(km) => get_var!(km),
            KMeansInner::D8K8(km) => get_var!(km),
            KMeansInner::D8K16(km) => get_var!(km),
            KMeansInner::D8K32(km) => get_var!(km),
        };

        Ok(var.map(|v| PyArray1::from_owned_array(py, Array1::from_vec(v))))
    }

    /// Feature dimensionality.
    #[getter]
    fn dimensions(&self) -> usize {
        self.dims
    }

    /// Maximum number of clusters.
    #[getter]
    fn max_clusters(&self) -> usize {
        self.max_k
    }

    /// Reset all clusters and state.
    fn reset(&mut self) {
        macro_rules! do_reset {
            ($km:expr) => {{
                $km.reset();
            }};
        }

        match &mut self.inner {
            KMeansInner::D2K4(km) => do_reset!(km),
            KMeansInner::D2K8(km) => do_reset!(km),
            KMeansInner::D2K16(km) => do_reset!(km),
            KMeansInner::D2K32(km) => do_reset!(km),
            KMeansInner::D3K4(km) => do_reset!(km),
            KMeansInner::D3K8(km) => do_reset!(km),
            KMeansInner::D3K16(km) => do_reset!(km),
            KMeansInner::D3K32(km) => do_reset!(km),
            KMeansInner::D5K4(km) => do_reset!(km),
            KMeansInner::D5K8(km) => do_reset!(km),
            KMeansInner::D5K16(km) => do_reset!(km),
            KMeansInner::D5K32(km) => do_reset!(km),
            KMeansInner::D8K4(km) => do_reset!(km),
            KMeansInner::D8K8(km) => do_reset!(km),
            KMeansInner::D8K16(km) => do_reset!(km),
            KMeansInner::D8K32(km) => do_reset!(km),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OnlineKMeans(dimensions={}, max_clusters={}, n_active={})",
            self.dims,
            self.max_k,
            self.n_active_inner()
        )
    }
}

impl OnlineKMeans {
    fn n_active_inner(&self) -> usize {
        macro_rules! get_n {
            ($km:expr) => {{
                $km.n_active()
            }};
        }

        match &self.inner {
            KMeansInner::D2K4(km) => get_n!(km),
            KMeansInner::D2K8(km) => get_n!(km),
            KMeansInner::D2K16(km) => get_n!(km),
            KMeansInner::D2K32(km) => get_n!(km),
            KMeansInner::D3K4(km) => get_n!(km),
            KMeansInner::D3K8(km) => get_n!(km),
            KMeansInner::D3K16(km) => get_n!(km),
            KMeansInner::D3K32(km) => get_n!(km),
            KMeansInner::D5K4(km) => get_n!(km),
            KMeansInner::D5K8(km) => get_n!(km),
            KMeansInner::D5K16(km) => get_n!(km),
            KMeansInner::D5K32(km) => get_n!(km),
            KMeansInner::D8K4(km) => get_n!(km),
            KMeansInner::D8K8(km) => get_n!(km),
            KMeansInner::D8K16(km) => get_n!(km),
            KMeansInner::D8K32(km) => get_n!(km),
        }
    }
}

/// Register online k-means classes and functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OnlineKMeans>()?;
    Ok(())
}
