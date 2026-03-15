//! Python bindings for Linear Discriminant Analysis.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::lda::{Lda as ZsLda, LdaError};

fn lda_error_to_py(e: LdaError) -> PyErr {
    let msg = match e {
        LdaError::InsufficientData => "Insufficient data: need at least 2 samples per class",
        LdaError::NotFitted => "LDA model has not been fitted yet",
        LdaError::SingularScatter => {
            "Within-class scatter matrix is singular even after regularization"
        }
    };
    PyValueError::new_err(msg)
}

macro_rules! make_lda_inner {
    ($name:ident, $c:expr, $m:expr) => {
        struct $name(ZsLda<$c, $m>);

        impl $name {
            fn new(shrinkage: f64) -> Self {
                Self(ZsLda::new(shrinkage))
            }

            fn fit(&mut self, x: &[f64], y: &[i64], n_samples: usize) -> Result<(), LdaError> {
                let mut c0 = Vec::new();
                let mut c1 = Vec::new();
                for i in 0..n_samples {
                    let mut sample = [0.0f64; $c];
                    sample.copy_from_slice(&x[i * $c..(i + 1) * $c]);
                    if y[i] == 0 {
                        c0.push(sample);
                    } else {
                        c1.push(sample);
                    }
                }
                self.0.fit(&c0, &c1)
            }

            fn predict(&self, x: &[f64]) -> Result<usize, LdaError> {
                let mut sample = [0.0f64; $c];
                sample.copy_from_slice(x);
                self.0.predict(&sample)
            }

            fn predict_proba(&self, x: &[f64]) -> Result<f64, LdaError> {
                let mut sample = [0.0f64; $c];
                sample.copy_from_slice(x);
                self.0.predict_proba(&sample)
            }

            fn decision_function(&self, x: &[f64]) -> Result<f64, LdaError> {
                let mut sample = [0.0f64; $c];
                sample.copy_from_slice(x);
                self.0.decision_function(&sample)
            }

            fn weights(&self) -> Option<Vec<f64>> {
                self.0.weights().map(|w| w.to_vec())
            }

            fn threshold(&self) -> f64 {
                self.0.threshold()
            }

            fn is_fitted(&self) -> bool {
                self.0.is_fitted()
            }
        }
    };
}

use std::vec::Vec;

make_lda_inner!(LdaC2, 2, 4);
make_lda_inner!(LdaC4, 4, 16);
make_lda_inner!(LdaC6, 6, 36);
make_lda_inner!(LdaC8, 8, 64);
make_lda_inner!(LdaC12, 12, 144);
make_lda_inner!(LdaC16, 16, 256);
make_lda_inner!(LdaC32, 32, 1024);
make_lda_inner!(LdaC64, 64, 4096);

#[allow(clippy::large_enum_variant)] // PyO3 dispatch enum, lives on heap via #[pyclass]
enum LdaInner {
    C2(LdaC2),
    C4(LdaC4),
    C6(LdaC6),
    C8(LdaC8),
    C12(LdaC12),
    C16(LdaC16),
    C32(LdaC32),
    C64(Box<LdaC64>),
}

/// Fisher's Linear Discriminant Analysis for binary classification.
///
/// The standard classifier paired with CSP for motor imagery BCI.
/// Computes the optimal linear projection that maximizes class separability.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// lda = zbci.Lda(features=4, shrinkage=0.01)
///
/// # X shape: (n_samples, n_features), y: 1D int array with 0/1 labels
/// X = np.random.randn(100, 4)
/// y = np.array([0]*50 + [1]*50)
/// lda.fit(X, y)
///
/// predictions = lda.predict(X)
/// probabilities = lda.predict_proba(X)
/// accuracy = lda.score(X, y)
/// ```
#[pyclass]
pub struct Lda {
    inner: LdaInner,
    features: usize,
}

#[pymethods]
impl Lda {
    /// Create a new LDA classifier.
    ///
    /// Args:
    ///     features (int): Number of features (2, 4, 6, 8, 12, 16, 32, or 64).
    ///     shrinkage (float): Regularization parameter in [0, 1]. Default: 0.01.
    #[new]
    #[pyo3(signature = (features, shrinkage=0.01))]
    fn new(features: usize, shrinkage: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&shrinkage) {
            return Err(PyValueError::new_err("shrinkage must be in [0, 1]"));
        }

        let inner = match features {
            2 => LdaInner::C2(LdaC2::new(shrinkage)),
            4 => LdaInner::C4(LdaC4::new(shrinkage)),
            6 => LdaInner::C6(LdaC6::new(shrinkage)),
            8 => LdaInner::C8(LdaC8::new(shrinkage)),
            12 => LdaInner::C12(LdaC12::new(shrinkage)),
            16 => LdaInner::C16(LdaC16::new(shrinkage)),
            32 => LdaInner::C32(LdaC32::new(shrinkage)),
            64 => LdaInner::C64(Box::new(LdaC64::new(shrinkage))),
            _ => {
                return Err(PyValueError::new_err(
                    "features must be 2, 4, 6, 8, 12, 16, 32, or 64",
                ));
            }
        };

        Ok(Self { inner, features })
    }

    /// Fit the LDA model to labeled data.
    ///
    /// Args:
    ///     X (np.ndarray): 2D float64 array of shape (n_samples, n_features).
    ///     y (np.ndarray): 1D int array of labels (0 or 1).
    #[allow(non_snake_case)]
    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let x_shape = X.shape();
        if x_shape[1] != self.features {
            return Err(PyValueError::new_err(format!(
                "X has {} features, expected {}",
                x_shape[1], self.features
            )));
        }
        let y_slice = y.as_slice()?;
        if y_slice.len() != x_shape[0] {
            return Err(PyValueError::new_err(format!(
                "X has {} samples but y has {} labels",
                x_shape[0],
                y_slice.len()
            )));
        }
        let x_flat = X.as_slice()?;
        let n = x_shape[0];

        macro_rules! do_fit {
            ($lda:expr) => {{
                $lda.fit(x_flat, y_slice, n).map_err(lda_error_to_py)
            }};
        }

        match &mut self.inner {
            LdaInner::C2(lda) => do_fit!(lda),
            LdaInner::C4(lda) => do_fit!(lda),
            LdaInner::C6(lda) => do_fit!(lda),
            LdaInner::C8(lda) => do_fit!(lda),
            LdaInner::C12(lda) => do_fit!(lda),
            LdaInner::C16(lda) => do_fit!(lda),
            LdaInner::C32(lda) => do_fit!(lda),
            LdaInner::C64(lda) => do_fit!(lda),
        }
    }

    /// Predict class labels for samples.
    ///
    /// Args:
    ///     X (np.ndarray): 2D float64 array of shape (n_samples, n_features).
    ///
    /// Returns:
    ///     np.ndarray: 1D int array of predictions (0 or 1).
    #[allow(non_snake_case)]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let x_shape = X.shape();
        if x_shape[1] != self.features {
            return Err(PyValueError::new_err(format!(
                "X has {} features, expected {}",
                x_shape[1], self.features
            )));
        }
        let x_flat = X.as_slice()?;
        let n = x_shape[0];
        let c = self.features;

        macro_rules! do_predict {
            ($lda:expr) => {{
                let mut preds = Vec::with_capacity(n);
                for i in 0..n {
                    let p = $lda
                        .predict(&x_flat[i * c..(i + 1) * c])
                        .map_err(lda_error_to_py)?;
                    preds.push(p as i64);
                }
                let arr = Array1::from_vec(preds);
                Ok(PyArray1::from_owned_array(py, arr))
            }};
        }

        match &self.inner {
            LdaInner::C2(lda) => do_predict!(lda),
            LdaInner::C4(lda) => do_predict!(lda),
            LdaInner::C6(lda) => do_predict!(lda),
            LdaInner::C8(lda) => do_predict!(lda),
            LdaInner::C12(lda) => do_predict!(lda),
            LdaInner::C16(lda) => do_predict!(lda),
            LdaInner::C32(lda) => do_predict!(lda),
            LdaInner::C64(lda) => do_predict!(lda),
        }
    }

    /// Predict probability of class 0 for samples.
    ///
    /// Args:
    ///     X (np.ndarray): 2D float64 array of shape (n_samples, n_features).
    ///
    /// Returns:
    ///     np.ndarray: 1D float array of P(class 0) for each sample.
    #[allow(non_snake_case)]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_shape = X.shape();
        if x_shape[1] != self.features {
            return Err(PyValueError::new_err(format!(
                "X has {} features, expected {}",
                x_shape[1], self.features
            )));
        }
        let x_flat = X.as_slice()?;
        let n = x_shape[0];
        let c = self.features;

        macro_rules! do_proba {
            ($lda:expr) => {{
                let mut probs = Vec::with_capacity(n);
                for i in 0..n {
                    let p = $lda
                        .predict_proba(&x_flat[i * c..(i + 1) * c])
                        .map_err(lda_error_to_py)?;
                    probs.push(p);
                }
                let arr = Array1::from_vec(probs);
                Ok(PyArray1::from_owned_array(py, arr))
            }};
        }

        match &self.inner {
            LdaInner::C2(lda) => do_proba!(lda),
            LdaInner::C4(lda) => do_proba!(lda),
            LdaInner::C6(lda) => do_proba!(lda),
            LdaInner::C8(lda) => do_proba!(lda),
            LdaInner::C12(lda) => do_proba!(lda),
            LdaInner::C16(lda) => do_proba!(lda),
            LdaInner::C32(lda) => do_proba!(lda),
            LdaInner::C64(lda) => do_proba!(lda),
        }
    }

    /// Compute signed distance from decision boundary for samples.
    ///
    /// Args:
    ///     X (np.ndarray): 2D float64 array of shape (n_samples, n_features).
    ///
    /// Returns:
    ///     np.ndarray: 1D float array. Positive = class 0, negative = class 1.
    #[allow(non_snake_case)]
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_shape = X.shape();
        if x_shape[1] != self.features {
            return Err(PyValueError::new_err(format!(
                "X has {} features, expected {}",
                x_shape[1], self.features
            )));
        }
        let x_flat = X.as_slice()?;
        let n = x_shape[0];
        let c = self.features;

        macro_rules! do_decision {
            ($lda:expr) => {{
                let mut vals = Vec::with_capacity(n);
                for i in 0..n {
                    let d = $lda
                        .decision_function(&x_flat[i * c..(i + 1) * c])
                        .map_err(lda_error_to_py)?;
                    vals.push(d);
                }
                let arr = Array1::from_vec(vals);
                Ok(PyArray1::from_owned_array(py, arr))
            }};
        }

        match &self.inner {
            LdaInner::C2(lda) => do_decision!(lda),
            LdaInner::C4(lda) => do_decision!(lda),
            LdaInner::C6(lda) => do_decision!(lda),
            LdaInner::C8(lda) => do_decision!(lda),
            LdaInner::C12(lda) => do_decision!(lda),
            LdaInner::C16(lda) => do_decision!(lda),
            LdaInner::C32(lda) => do_decision!(lda),
            LdaInner::C64(lda) => do_decision!(lda),
        }
    }

    /// Compute classification accuracy.
    ///
    /// Args:
    ///     X (np.ndarray): 2D float64 array of shape (n_samples, n_features).
    ///     y (np.ndarray): 1D int array of true labels (0 or 1).
    ///
    /// Returns:
    ///     float: Accuracy in [0, 1].
    #[allow(non_snake_case)]
    fn score(&self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i64>) -> PyResult<f64> {
        let x_shape = X.shape();
        if x_shape[1] != self.features {
            return Err(PyValueError::new_err(format!(
                "X has {} features, expected {}",
                x_shape[1], self.features
            )));
        }
        let y_slice = y.as_slice()?;
        if y_slice.len() != x_shape[0] {
            return Err(PyValueError::new_err(format!(
                "X has {} samples but y has {} labels",
                x_shape[0],
                y_slice.len()
            )));
        }
        let x_flat = X.as_slice()?;
        let n = x_shape[0];
        let c = self.features;

        macro_rules! do_score {
            ($lda:expr) => {{
                let mut correct = 0usize;
                for i in 0..n {
                    let pred = $lda
                        .predict(&x_flat[i * c..(i + 1) * c])
                        .map_err(lda_error_to_py)?;
                    if pred as i64 == y_slice[i] {
                        correct += 1;
                    }
                }
                Ok(correct as f64 / n as f64)
            }};
        }

        match &self.inner {
            LdaInner::C2(lda) => do_score!(lda),
            LdaInner::C4(lda) => do_score!(lda),
            LdaInner::C6(lda) => do_score!(lda),
            LdaInner::C8(lda) => do_score!(lda),
            LdaInner::C12(lda) => do_score!(lda),
            LdaInner::C16(lda) => do_score!(lda),
            LdaInner::C32(lda) => do_score!(lda),
            LdaInner::C64(lda) => do_score!(lda),
        }
    }

    /// Discriminant weights (unit norm direction).
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        macro_rules! get_weights {
            ($lda:expr) => {{
                match $lda.weights() {
                    Some(w) => {
                        let arr = Array1::from_vec(w);
                        Ok(PyArray1::from_owned_array(py, arr))
                    }
                    None => Err(PyValueError::new_err("LDA model has not been fitted yet")),
                }
            }};
        }

        match &self.inner {
            LdaInner::C2(lda) => get_weights!(lda),
            LdaInner::C4(lda) => get_weights!(lda),
            LdaInner::C6(lda) => get_weights!(lda),
            LdaInner::C8(lda) => get_weights!(lda),
            LdaInner::C12(lda) => get_weights!(lda),
            LdaInner::C16(lda) => get_weights!(lda),
            LdaInner::C32(lda) => get_weights!(lda),
            LdaInner::C64(lda) => get_weights!(lda),
        }
    }

    /// Decision threshold.
    #[getter]
    fn threshold(&self) -> f64 {
        match &self.inner {
            LdaInner::C2(lda) => lda.threshold(),
            LdaInner::C4(lda) => lda.threshold(),
            LdaInner::C6(lda) => lda.threshold(),
            LdaInner::C8(lda) => lda.threshold(),
            LdaInner::C12(lda) => lda.threshold(),
            LdaInner::C16(lda) => lda.threshold(),
            LdaInner::C32(lda) => lda.threshold(),
            LdaInner::C64(lda) => lda.threshold(),
        }
    }

    /// Whether the model has been fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        match &self.inner {
            LdaInner::C2(lda) => lda.is_fitted(),
            LdaInner::C4(lda) => lda.is_fitted(),
            LdaInner::C6(lda) => lda.is_fitted(),
            LdaInner::C8(lda) => lda.is_fitted(),
            LdaInner::C12(lda) => lda.is_fitted(),
            LdaInner::C16(lda) => lda.is_fitted(),
            LdaInner::C32(lda) => lda.is_fitted(),
            LdaInner::C64(lda) => lda.is_fitted(),
        }
    }

    /// Number of features.
    #[getter]
    fn features(&self) -> usize {
        self.features
    }

    fn __repr__(&self) -> String {
        format!(
            "Lda(features={}, fitted={})",
            self.features,
            self.is_fitted()
        )
    }
}
