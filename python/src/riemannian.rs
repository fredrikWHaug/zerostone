//! Python bindings for Riemannian geometry (tangent space projection, MDM classifier).

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::linalg::Matrix;
use zerostone::TangentSpace as ZsTangentSpace;

enum TangentSpaceInner {
    C4(ZsTangentSpace<4, 16, 10>),
    C8(ZsTangentSpace<8, 64, 36>),
    C16(ZsTangentSpace<16, 256, 136>),
    C32(ZsTangentSpace<32, 1024, 528>),
}

/// Tangent space projection for symmetric positive definite matrices.
///
/// Projects SPD matrices (e.g. covariance matrices) to the tangent space
/// at a reference point, enabling use of Euclidean classifiers on
/// Riemannian manifold data.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// ts = zbci.TangentSpace(channels=4)
/// ref_matrix = np.eye(4, dtype=np.float64)
/// ts.fit(ref_matrix)
/// vec = ts.transform(ref_matrix)  # shape (10,) for 4 channels
/// ```
#[pyclass]
pub struct TangentSpace {
    inner: TangentSpaceInner,
    channels: usize,
}

#[pymethods]
impl TangentSpace {
    /// Create a new tangent space projector.
    ///
    /// Args:
    ///     channels (int): Matrix dimension. Must be 4, 8, 16, or 32.
    #[new]
    fn new(channels: usize) -> PyResult<Self> {
        let inner = match channels {
            4 => TangentSpaceInner::C4(ZsTangentSpace::new()),
            8 => TangentSpaceInner::C8(ZsTangentSpace::new()),
            16 => TangentSpaceInner::C16(ZsTangentSpace::new()),
            32 => TangentSpaceInner::C32(ZsTangentSpace::new()),
            _ => return Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        };
        Ok(Self { inner, channels })
    }

    /// Fit the reference point (SPD matrix).
    ///
    /// Args:
    ///     reference (np.ndarray): 2D float64 array of shape (C, C).
    fn fit(&mut self, reference: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = reference.shape();
        if shape[0] != self.channels || shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrix, got ({}, {})",
                self.channels, self.channels, shape[0], shape[1]
            )));
        }
        let data = reference.as_slice()?;

        macro_rules! do_fit {
            ($ts:expr, $m:expr) => {{
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(data);
                $ts.fit(&Matrix::new(arr))
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))
            }};
        }
        match &mut self.inner {
            TangentSpaceInner::C4(ts) => do_fit!(ts, 16),
            TangentSpaceInner::C8(ts) => do_fit!(ts, 64),
            TangentSpaceInner::C16(ts) => do_fit!(ts, 256),
            TangentSpaceInner::C32(ts) => do_fit!(ts, 1024),
        }
    }

    /// Project an SPD matrix to the tangent space.
    ///
    /// Args:
    ///     matrix (np.ndarray): 2D float64 array of shape (C, C).
    ///
    /// Returns:
    ///     np.ndarray: 1D float64 vector of length C*(C+1)/2.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.channels || shape[1] != self.channels {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrix, got ({}, {})",
                self.channels, self.channels, shape[0], shape[1]
            )));
        }
        let data = matrix.as_slice()?;

        macro_rules! do_transform {
            ($ts:expr, $m:expr) => {{
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(data);
                let vec = $ts
                    .transform(&Matrix::new(arr))
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                Ok(PyArray1::from_vec(py, vec.to_vec()))
            }};
        }
        match &self.inner {
            TangentSpaceInner::C4(ts) => do_transform!(ts, 16),
            TangentSpaceInner::C8(ts) => do_transform!(ts, 64),
            TangentSpaceInner::C16(ts) => do_transform!(ts, 256),
            TangentSpaceInner::C32(ts) => do_transform!(ts, 1024),
        }
    }

    /// Reconstruct an SPD matrix from a tangent space vector.
    ///
    /// Args:
    ///     vector (np.ndarray): 1D float64 array of length C*(C+1)/2.
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (C, C).
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        vector: numpy::PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = vector.as_slice()?;

        macro_rules! do_inverse {
            ($ts:expr, $c:expr, $m:expr, $v:expr) => {{
                if data.len() != $v {
                    return Err(PyValueError::new_err(format!(
                        "Expected vector of length {}, got {}",
                        $v,
                        data.len()
                    )));
                }
                let mut arr = [0.0f64; $v];
                arr.copy_from_slice(data);
                let mat = $ts
                    .inverse_transform(&arr)
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                let flat = mat.data().to_vec();
                let array = Array2::from_shape_vec(($c, $c), flat).unwrap();
                Ok(PyArray2::from_owned_array(py, array))
            }};
        }
        match &self.inner {
            TangentSpaceInner::C4(ts) => do_inverse!(ts, 4, 16, 10),
            TangentSpaceInner::C8(ts) => do_inverse!(ts, 8, 64, 36),
            TangentSpaceInner::C16(ts) => do_inverse!(ts, 16, 256, 136),
            TangentSpaceInner::C32(ts) => do_inverse!(ts, 32, 1024, 528),
        }
    }

    /// Number of channels (matrix dimension).
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Length of the tangent space vector: C*(C+1)/2.
    #[getter]
    fn vector_length(&self) -> usize {
        self.channels * (self.channels + 1) / 2
    }

    fn __repr__(&self) -> String {
        format!(
            "TangentSpace(channels={}, vector_length={})",
            self.channels,
            self.vector_length()
        )
    }
}

// --- Free functions ---

/// Compute the Frechet (geometric) mean of a set of SPD matrices.
///
/// Args:
///     matrices (np.ndarray): 3D float64 array of shape (N, C, C).
///     max_iters (int): Maximum iterations for Karcher flow. Default: 20.
///     tol (float): Convergence tolerance. Default: 1e-8.
///
/// Returns:
///     np.ndarray: 2D float64 array of shape (C, C).
#[pyfunction]
#[pyo3(signature = (matrices, max_iters=20, tol=1e-8))]
fn frechet_mean<'py>(
    py: Python<'py>,
    matrices: PyReadonlyArray3<f64>,
    max_iters: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = matrices.shape();
    let n = shape[0];
    let c = shape[1];
    if shape[2] != c {
        return Err(PyValueError::new_err(format!(
            "Expected square matrices, got shape ({}, {}, {})",
            n, c, shape[2]
        )));
    }
    if n == 0 {
        return Err(PyValueError::new_err("Need at least one matrix"));
    }
    let data = matrices.as_slice()?;

    macro_rules! do_frechet {
        ($c:expr, $m:expr) => {{
            let mut mats: Vec<Matrix<$c, $m>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(&data[i * $m..(i + 1) * $m]);
                mats.push(Matrix::new(arr));
            }
            let refs: Vec<&Matrix<$c, $m>> = mats.iter().collect();
            let mut output: Matrix<$c, $m> = Matrix::zeros();
            zerostone::frechet_mean(&refs, &mut output, max_iters, tol)
                .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
            let flat = output.data().to_vec();
            let array = Array2::from_shape_vec(($c, $c), flat).unwrap();
            Ok(PyArray2::from_owned_array(py, array))
        }};
    }

    match c {
        4 => do_frechet!(4, 16),
        8 => do_frechet!(8, 64),
        16 => do_frechet!(16, 256),
        32 => do_frechet!(32, 1024),
        _ => Err(PyValueError::new_err(
            "channels must be 4, 8, 16, or 32",
        )),
    }
}

/// Compute the Riemannian (affine-invariant) distance between two SPD matrices.
///
/// Args:
///     a (np.ndarray): 2D float64 array of shape (C, C).
///     b (np.ndarray): 2D float64 array of shape (C, C).
///
/// Returns:
///     float: The Riemannian distance.
#[pyfunction]
fn riemannian_distance(a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let sa = a.shape();
    let sb = b.shape();
    let c = sa[0];
    if sa[1] != c || sb[0] != c || sb[1] != c {
        return Err(PyValueError::new_err(format!(
            "Expected two ({0}, {0}) matrices, got ({1}, {2}) and ({3}, {4})",
            c, sa[0], sa[1], sb[0], sb[1]
        )));
    }
    let da = a.as_slice()?;
    let db = b.as_slice()?;

    macro_rules! do_dist {
        ($c:expr, $m:expr) => {{
            let mut arr_a = [0.0f64; $m];
            let mut arr_b = [0.0f64; $m];
            arr_a.copy_from_slice(da);
            arr_b.copy_from_slice(db);
            let ma = Matrix::<$c, $m>::new(arr_a);
            let mb = Matrix::<$c, $m>::new(arr_b);
            zerostone::riemannian_distance(&ma, &mb)
                .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))
        }};
    }

    match c {
        4 => do_dist!(4, 16),
        8 => do_dist!(8, 64),
        16 => do_dist!(16, 256),
        32 => do_dist!(32, 1024),
        _ => Err(PyValueError::new_err(
            "channels must be 4, 8, 16, or 32",
        )),
    }
}

/// Recenter SPD matrices around the identity by removing a reference mean.
///
/// Computes ``M^{-1/2} X_i M^{-1/2}`` for each matrix ``X_i``, where ``M``
/// is the given reference mean. After recentering, the Frechet mean of the
/// transformed matrices is close to the identity matrix.
///
/// This is the core operation for Riemannian transfer learning.
///
/// Args:
///     matrices (np.ndarray): 3D float64 array of shape (N, C, C).
///     mean (np.ndarray): 2D float64 array of shape (C, C) — the reference mean.
///
/// Returns:
///     np.ndarray: 3D float64 array of shape (N, C, C) — recentered matrices.
#[pyfunction]
fn recenter<'py>(
    py: Python<'py>,
    matrices: PyReadonlyArray3<f64>,
    mean: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let shape = matrices.shape();
    let n = shape[0];
    let c = shape[1];
    if shape[2] != c {
        return Err(PyValueError::new_err(format!(
            "Expected square matrices, got shape ({}, {}, {})",
            n, c, shape[2]
        )));
    }
    let mean_shape = mean.shape();
    if mean_shape[0] != c || mean_shape[1] != c {
        return Err(PyValueError::new_err(format!(
            "Mean shape ({}, {}) doesn't match matrix dimension {}",
            mean_shape[0], mean_shape[1], c
        )));
    }
    let mat_data = matrices.as_slice()?;
    let mean_data = mean.as_slice()?;

    macro_rules! do_recenter {
        ($c:expr, $m:expr) => {{
            // Compute mean^{-1/2}
            let mut mean_arr = [0.0f64; $m];
            mean_arr.copy_from_slice(mean_data);
            let mean_mat = Matrix::<$c, $m>::new(mean_arr);
            let inv_sqrt = zerostone::matrix_inv_sqrt(&mean_mat)
                .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;

            // Recenter each matrix
            let mut result = Vec::with_capacity(n * $m);
            for i in 0..n {
                let mut arr = [0.0f64; $m];
                arr.copy_from_slice(&mat_data[i * $m..(i + 1) * $m]);
                let mat = Matrix::<$c, $m>::new(arr);
                let recentered = zerostone::recenter(&mat, &inv_sqrt);
                result.extend_from_slice(recentered.data());
            }

            let array = numpy::ndarray::Array3::from_shape_vec((n, $c, $c), result).unwrap();
            Ok(numpy::PyArray3::from_owned_array(py, array))
        }};
    }

    match c {
        4 => do_recenter!(4, 16),
        8 => do_recenter!(8, 64),
        16 => do_recenter!(16, 256),
        32 => do_recenter!(32, 1024),
        _ => Err(PyValueError::new_err(
            "channels must be 4, 8, 16, or 32",
        )),
    }
}

// --- MdmClassifier ---

/// Minimum Distance to Mean (MDM) classifier on the SPD manifold.
///
/// Classifies SPD matrices (e.g. covariance matrices) by computing the
/// Riemannian distance to each class mean, assigning the label of the
/// nearest mean.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// mdm = zbci.MdmClassifier(channels=4)
/// X = np.stack([np.eye(4) * (1 + 0.1*i) for i in range(20)])
/// y = np.array([0]*10 + [1]*10, dtype=np.int64)
/// mdm.fit(X, y)
/// preds = mdm.predict(X)
/// ```
#[pyclass]
pub struct MdmClassifier {
    channels: usize,
    /// Flattened class means, each of length C*C
    class_means: Vec<Vec<f64>>,
    /// Sorted unique class labels
    class_labels: Vec<i64>,
}

#[pymethods]
impl MdmClassifier {
    /// Create a new MDM classifier.
    ///
    /// Args:
    ///     channels (int): Matrix dimension. Must be 4, 8, 16, or 32.
    #[new]
    fn new(channels: usize) -> PyResult<Self> {
        match channels {
            4 | 8 | 16 | 32 => {}
            _ => return Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        }
        Ok(Self {
            channels,
            class_means: Vec::new(),
            class_labels: Vec::new(),
        })
    }

    /// Fit the classifier by computing Frechet means per class.
    ///
    /// Args:
    ///     x (np.ndarray): 3D float64 array of shape (n_trials, C, C).
    ///     y (np.ndarray): 1D int64 array of class labels, length n_trials.
    fn fit(&mut self, x: PyReadonlyArray3<f64>, y: numpy::PyReadonlyArray1<i64>) -> PyResult<()> {
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        if c != self.channels || shape[2] != c {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrices, got shape ({}, {}, {})",
                self.channels, self.channels, n, c, shape[2]
            )));
        }
        let y_data = y.as_slice()?;
        if y_data.len() != n {
            return Err(PyValueError::new_err(format!(
                "X has {} trials but y has {} labels",
                n,
                y_data.len()
            )));
        }
        let x_data = x.as_slice()?;
        let m = c * c;

        // Find unique labels
        let mut labels: Vec<i64> = y_data.to_vec();
        labels.sort();
        labels.dedup();

        let mut means = Vec::with_capacity(labels.len());

        for &label in &labels {
            // Collect indices for this class
            let indices: Vec<usize> = y_data
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == label)
                .map(|(i, _)| i)
                .collect();

            if indices.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "No trials for class {}",
                    label
                )));
            }

            // Compute Frechet mean for this class
            let mean_flat = self.compute_frechet_mean(x_data, m, &indices)?;
            means.push(mean_flat);
        }

        self.class_labels = labels;
        self.class_means = means;
        Ok(())
    }

    /// Predict class labels for SPD matrices.
    ///
    /// Args:
    ///     x (np.ndarray): 3D float64 array of shape (n_trials, C, C).
    ///
    /// Returns:
    ///     list[int]: Predicted class labels.
    fn predict(&self, x: PyReadonlyArray3<f64>) -> PyResult<Vec<i64>> {
        if self.class_means.is_empty() {
            return Err(PyValueError::new_err("Classifier not fitted"));
        }
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        if c != self.channels || shape[2] != c {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrices, got shape ({}, {}, {})",
                self.channels, self.channels, n, c, shape[2]
            )));
        }
        let x_data = x.as_slice()?;
        let m = c * c;

        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let trial_data = &x_data[i * m..(i + 1) * m];
            let (cls_idx, _) = self.classify_trial(trial_data)?;
            predictions.push(self.class_labels[cls_idx]);
        }
        Ok(predictions)
    }

    /// Compute distance matrix from trials to class means.
    ///
    /// Args:
    ///     x (np.ndarray): 3D float64 array of shape (n_trials, C, C).
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (n_trials, n_classes).
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.class_means.is_empty() {
            return Err(PyValueError::new_err("Classifier not fitted"));
        }
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        if c != self.channels || shape[2] != c {
            return Err(PyValueError::new_err(format!(
                "Expected ({}, {}) matrices, got shape ({}, {}, {})",
                self.channels, self.channels, n, c, shape[2]
            )));
        }
        let x_data = x.as_slice()?;
        let m = c * c;
        let n_classes = self.class_means.len();

        let mut distances = Vec::with_capacity(n * n_classes);
        for i in 0..n {
            let trial_data = &x_data[i * m..(i + 1) * m];
            let dists = self.distances_to_means(trial_data)?;
            distances.extend_from_slice(&dists);
        }

        let array = Array2::from_shape_vec((n, n_classes), distances).unwrap();
        Ok(PyArray2::from_owned_array(py, array))
    }

    /// Compute classification accuracy.
    ///
    /// Args:
    ///     x (np.ndarray): 3D float64 array of shape (n_trials, C, C).
    ///     y (np.ndarray): 1D int64 array of true class labels.
    ///
    /// Returns:
    ///     float: Classification accuracy (0.0 to 1.0).
    fn score(&self, x: PyReadonlyArray3<f64>, y: numpy::PyReadonlyArray1<i64>) -> PyResult<f64> {
        let predictions = self.predict(x)?;
        let y_data = y.as_slice()?;
        if predictions.len() != y_data.len() {
            return Err(PyValueError::new_err("Length mismatch"));
        }
        let correct = predictions
            .iter()
            .zip(y_data.iter())
            .filter(|(&p, &t)| p == t)
            .count();
        Ok(correct as f64 / predictions.len() as f64)
    }

    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.class_labels.len()
    }

    fn __repr__(&self) -> String {
        if self.class_means.is_empty() {
            format!("MdmClassifier(channels={}, fitted=False)", self.channels)
        } else {
            format!(
                "MdmClassifier(channels={}, n_classes={})",
                self.channels,
                self.class_labels.len()
            )
        }
    }
}

impl MdmClassifier {
    fn compute_frechet_mean(
        &self,
        x_data: &[f64],
        _m: usize,
        indices: &[usize],
    ) -> PyResult<Vec<f64>> {
        macro_rules! do_mean {
            ($c:expr, $m:expr) => {{
                let mut mats: Vec<Matrix<$c, $m>> = Vec::with_capacity(indices.len());
                for &idx in indices {
                    let mut arr = [0.0f64; $m];
                    arr.copy_from_slice(&x_data[idx * $m..(idx + 1) * $m]);
                    mats.push(Matrix::new(arr));
                }
                let refs: Vec<&Matrix<$c, $m>> = mats.iter().collect();
                let mut output: Matrix<$c, $m> = Matrix::zeros();
                zerostone::frechet_mean(&refs, &mut output, 20, 1e-8)
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                Ok(output.data().to_vec())
            }};
        }

        match self.channels {
            4 => do_mean!(4, 16),
            8 => do_mean!(8, 64),
            16 => do_mean!(16, 256),
            32 => do_mean!(32, 1024),
            _ => Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        }
    }

    fn classify_trial(&self, trial_data: &[f64]) -> PyResult<(usize, f64)> {
        macro_rules! do_classify {
            ($c:expr, $m:expr) => {{
                let mut trial_arr = [0.0f64; $m];
                trial_arr.copy_from_slice(trial_data);
                let trial_mat = Matrix::<$c, $m>::new(trial_arr);

                let mut mean_mats: Vec<Matrix<$c, $m>> = Vec::with_capacity(self.class_means.len());
                for mean_data in &self.class_means {
                    let mut arr = [0.0f64; $m];
                    arr.copy_from_slice(mean_data);
                    mean_mats.push(Matrix::new(arr));
                }
                let refs: Vec<&Matrix<$c, $m>> = mean_mats.iter().collect();
                zerostone::mdm_classify(&trial_mat, &refs)
                    .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))
            }};
        }

        match self.channels {
            4 => do_classify!(4, 16),
            8 => do_classify!(8, 64),
            16 => do_classify!(16, 256),
            32 => do_classify!(32, 1024),
            _ => Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        }
    }

    fn distances_to_means(&self, trial_data: &[f64]) -> PyResult<Vec<f64>> {
        macro_rules! do_distances {
            ($c:expr, $m:expr) => {{
                let mut trial_arr = [0.0f64; $m];
                trial_arr.copy_from_slice(trial_data);
                let trial_mat = Matrix::<$c, $m>::new(trial_arr);

                let mut dists = Vec::with_capacity(self.class_means.len());
                for mean_data in &self.class_means {
                    let mut arr = [0.0f64; $m];
                    arr.copy_from_slice(mean_data);
                    let mean_mat = Matrix::<$c, $m>::new(arr);
                    let d = zerostone::riemannian_distance(&trial_mat, &mean_mat)
                        .map_err(|e| PyValueError::new_err(format!("LinalgError: {:?}", e)))?;
                    dists.push(d);
                }
                Ok(dists)
            }};
        }

        match self.channels {
            4 => do_distances!(4, 16),
            8 => do_distances!(8, 64),
            16 => do_distances!(16, 256),
            32 => do_distances!(32, 1024),
            _ => Err(PyValueError::new_err("channels must be 4, 8, 16, or 32")),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(frechet_mean, m)?)?;
    m.add_function(wrap_pyfunction!(riemannian_distance, m)?)?;
    m.add_function(wrap_pyfunction!(recenter, m)?)?;
    Ok(())
}
