//! Python bindings for Kalman filter.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::kalman::{KalmanError, KalmanFilter as ZsKalmanFilter};
use zerostone::linalg::Matrix;

fn kalman_error_to_py(e: KalmanError) -> PyErr {
    match e {
        KalmanError::InnovationNotPositiveDefinite => {
            PyValueError::new_err("Innovation covariance is not positive definite")
        }
    }
}

macro_rules! make_kalman_inner {
    ($name:ident, $s:expr, $o:expr, $sm:expr, $om:expr, $som:expr) => {
        #[allow(dead_code)]
        struct $name(ZsKalmanFilter<$s, $o, $sm, $om, $som>);

        impl $name {
            fn new(
                f_flat: &[f64],
                h_flat: &[f64],
                q_flat: &[f64],
                r_flat: &[f64],
                x0_flat: &[f64],
                p0_flat: &[f64],
            ) -> Self {
                let mut f_data = [0.0f64; $sm];
                f_data.copy_from_slice(f_flat);
                let mut h_data = [0.0f64; $som];
                h_data.copy_from_slice(h_flat);
                let mut q_data = [0.0f64; $sm];
                q_data.copy_from_slice(q_flat);
                let mut r_data = [0.0f64; $om];
                r_data.copy_from_slice(r_flat);
                let mut x0_data = [0.0f64; $s];
                x0_data.copy_from_slice(x0_flat);
                let mut p0_data = [0.0f64; $sm];
                p0_data.copy_from_slice(p0_flat);

                Self(ZsKalmanFilter::new(
                    Matrix::new(f_data),
                    h_data,
                    Matrix::new(q_data),
                    Matrix::new(r_data),
                    x0_data,
                    Matrix::new(p0_data),
                ))
            }

            fn predict(&mut self) {
                self.0.predict();
            }

            fn update(&mut self, z: &[f64]) -> Result<Vec<f64>, KalmanError> {
                let mut z_arr = [0.0f64; $o];
                z_arr.copy_from_slice(z);
                let inn = self.0.update(&z_arr)?;
                Ok(inn.to_vec())
            }

            fn state(&self) -> Vec<f64> {
                self.0.state().to_vec()
            }

            fn covariance(&self) -> (Vec<f64>, usize) {
                let p = self.0.covariance();
                (p.data().to_vec(), $s)
            }

            fn reset(&mut self, x0: &[f64], p0: &[f64]) {
                let mut x0_data = [0.0f64; $s];
                x0_data.copy_from_slice(x0);
                let mut p0_data = [0.0f64; $sm];
                p0_data.copy_from_slice(p0);
                self.0.reset(x0_data, Matrix::new(p0_data));
            }

            #[allow(dead_code)]
            fn innovation(&self) -> Option<Vec<f64>> {
                self.0.innovation().map(|i| i.to_vec())
            }
        }
    };
}

use std::vec::Vec;

make_kalman_inner!(KalmanS2O1, 2, 1, 4, 1, 2);
make_kalman_inner!(KalmanS2O2, 2, 2, 4, 4, 4);
make_kalman_inner!(KalmanS4O2, 4, 2, 16, 4, 8);
make_kalman_inner!(KalmanS4O4, 4, 4, 16, 16, 16);
make_kalman_inner!(KalmanS6O3, 6, 3, 36, 9, 18);
make_kalman_inner!(KalmanS6O6, 6, 6, 36, 36, 36);
make_kalman_inner!(KalmanS8O4, 8, 4, 64, 16, 32);
make_kalman_inner!(KalmanS8O8, 8, 8, 64, 64, 64);

enum KalmanInner {
    S2O1(KalmanS2O1),
    S2O2(KalmanS2O2),
    S4O2(KalmanS4O2),
    S4O4(KalmanS4O4),
    S6O3(KalmanS6O3),
    S6O6(KalmanS6O6),
    S8O4(KalmanS8O4),
    S8O8(KalmanS8O8),
}

/// Kalman filter for state estimation and decoder output smoothing.
///
/// Standard linear Kalman filter used for velocity decoding in closed-loop BCI.
/// Uses Joseph form for numerically stable covariance updates.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// dt = 0.1
/// # State: [position, velocity], Observe: [position]
/// F = np.array([[1, dt], [0, 1]])
/// H = np.array([[1, 0]])
/// Q = np.eye(2) * 0.01
/// R = np.array([[1.0]])
///
/// kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)
/// kf.predict()
/// innovation = kf.update(np.array([3.5]))
/// print(kf.state)
/// ```
#[pyclass]
pub struct KalmanFilter {
    inner: KalmanInner,
    state_dim: usize,
    obs_dim: usize,
}

#[pymethods]
impl KalmanFilter {
    /// Create a new Kalman filter.
    ///
    /// Args:
    ///     state_dim (int): State dimension S.
    ///     obs_dim (int): Observation dimension O.
    ///     F (np.ndarray): State transition matrix (S x S).
    ///     H (np.ndarray): Observation matrix (O x S).
    ///     Q (np.ndarray): Process noise covariance (S x S).
    ///     R (np.ndarray): Measurement noise covariance (O x O).
    ///     x0 (np.ndarray, optional): Initial state (S,). Defaults to zeros.
    ///     P0 (np.ndarray, optional): Initial covariance (S x S). Defaults to identity.
    #[new]
    #[pyo3(signature = (state_dim, obs_dim, F, H, Q, R, x0=None, P0=None))]
    #[allow(non_snake_case)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_dim: usize,
        obs_dim: usize,
        F: PyReadonlyArray2<f64>,
        H: PyReadonlyArray2<f64>,
        Q: PyReadonlyArray2<f64>,
        R: PyReadonlyArray2<f64>,
        x0: Option<numpy::PyReadonlyArray1<f64>>,
        P0: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        // Validate shapes
        let f_shape = F.shape();
        if f_shape[0] != state_dim || f_shape[1] != state_dim {
            return Err(PyValueError::new_err(format!(
                "F must be ({}, {}), got ({}, {})",
                state_dim, state_dim, f_shape[0], f_shape[1]
            )));
        }
        let h_shape = H.shape();
        if h_shape[0] != obs_dim || h_shape[1] != state_dim {
            return Err(PyValueError::new_err(format!(
                "H must be ({}, {}), got ({}, {})",
                obs_dim, state_dim, h_shape[0], h_shape[1]
            )));
        }
        let q_shape = Q.shape();
        if q_shape[0] != state_dim || q_shape[1] != state_dim {
            return Err(PyValueError::new_err(format!(
                "Q must be ({}, {}), got ({}, {})",
                state_dim, state_dim, q_shape[0], q_shape[1]
            )));
        }
        let r_shape = R.shape();
        if r_shape[0] != obs_dim || r_shape[1] != obs_dim {
            return Err(PyValueError::new_err(format!(
                "R must be ({}, {}), got ({}, {})",
                obs_dim, obs_dim, r_shape[0], r_shape[1]
            )));
        }

        let f_flat = F.as_slice()?;
        let h_flat = H.as_slice()?;
        let q_flat = Q.as_slice()?;
        let r_flat = R.as_slice()?;

        // Build x0
        let x0_vec: Vec<f64> = match x0 {
            Some(arr) => {
                let s = arr.as_slice()?;
                if s.len() != state_dim {
                    return Err(PyValueError::new_err(format!(
                        "x0 must have length {}, got {}",
                        state_dim,
                        s.len()
                    )));
                }
                s.to_vec()
            }
            None => vec![0.0; state_dim],
        };

        // Build P0
        let p0_vec: Vec<f64> = match P0 {
            Some(arr) => {
                let s = arr.as_slice()?;
                if s.len() != state_dim * state_dim {
                    return Err(PyValueError::new_err(format!(
                        "P0 must be ({}, {})",
                        state_dim, state_dim
                    )));
                }
                s.to_vec()
            }
            None => {
                let mut p = vec![0.0; state_dim * state_dim];
                for i in 0..state_dim {
                    p[i * state_dim + i] = 1.0;
                }
                p
            }
        };

        let inner = match (state_dim, obs_dim) {
            (2, 1) => KalmanInner::S2O1(KalmanS2O1::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (2, 2) => KalmanInner::S2O2(KalmanS2O2::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (4, 2) => KalmanInner::S4O2(KalmanS4O2::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (4, 4) => KalmanInner::S4O4(KalmanS4O4::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (6, 3) => KalmanInner::S6O3(KalmanS6O3::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (6, 6) => KalmanInner::S6O6(KalmanS6O6::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (8, 4) => KalmanInner::S8O4(KalmanS8O4::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            (8, 8) => KalmanInner::S8O8(KalmanS8O8::new(
                f_flat, h_flat, q_flat, r_flat, &x0_vec, &p0_vec,
            )),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported (state_dim, obs_dim) = ({}, {}). Supported: (2,1), (2,2), (4,2), (4,4), (6,3), (6,6), (8,4), (8,8)",
                    state_dim, obs_dim
                )));
            }
        };

        Ok(Self {
            inner,
            state_dim,
            obs_dim,
        })
    }

    /// Predict step: propagate state and covariance forward.
    fn predict(&mut self) {
        match &mut self.inner {
            KalmanInner::S2O1(k) => k.predict(),
            KalmanInner::S2O2(k) => k.predict(),
            KalmanInner::S4O2(k) => k.predict(),
            KalmanInner::S4O4(k) => k.predict(),
            KalmanInner::S6O3(k) => k.predict(),
            KalmanInner::S6O6(k) => k.predict(),
            KalmanInner::S8O4(k) => k.predict(),
            KalmanInner::S8O8(k) => k.predict(),
        }
    }

    /// Update step: incorporate a new measurement.
    ///
    /// Args:
    ///     z (np.ndarray): Observation vector of length obs_dim.
    ///
    /// Returns:
    ///     np.ndarray: Innovation vector (z - H*x_predicted).
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        z: numpy::PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let z_slice = z.as_slice()?;
        if z_slice.len() != self.obs_dim {
            return Err(PyValueError::new_err(format!(
                "z must have length {}, got {}",
                self.obs_dim,
                z_slice.len()
            )));
        }

        macro_rules! do_update {
            ($k:expr) => {{
                let inn = $k.update(z_slice).map_err(kalman_error_to_py)?;
                let arr = Array1::from_vec(inn);
                Ok(PyArray1::from_owned_array(py, arr))
            }};
        }

        match &mut self.inner {
            KalmanInner::S2O1(k) => do_update!(k),
            KalmanInner::S2O2(k) => do_update!(k),
            KalmanInner::S4O2(k) => do_update!(k),
            KalmanInner::S4O4(k) => do_update!(k),
            KalmanInner::S6O3(k) => do_update!(k),
            KalmanInner::S6O6(k) => do_update!(k),
            KalmanInner::S8O4(k) => do_update!(k),
            KalmanInner::S8O8(k) => do_update!(k),
        }
    }

    /// Current state estimate.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        macro_rules! get_state {
            ($k:expr) => {{
                let s = $k.state();
                let arr = Array1::from_vec(s);
                PyArray1::from_owned_array(py, arr)
            }};
        }

        match &self.inner {
            KalmanInner::S2O1(k) => get_state!(k),
            KalmanInner::S2O2(k) => get_state!(k),
            KalmanInner::S4O2(k) => get_state!(k),
            KalmanInner::S4O4(k) => get_state!(k),
            KalmanInner::S6O3(k) => get_state!(k),
            KalmanInner::S6O6(k) => get_state!(k),
            KalmanInner::S8O4(k) => get_state!(k),
            KalmanInner::S8O8(k) => get_state!(k),
        }
    }

    /// Current state covariance matrix.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        macro_rules! get_cov {
            ($k:expr) => {{
                let (flat, dim) = $k.covariance();
                let arr = Array2::from_shape_vec((dim, dim), flat).unwrap();
                PyArray2::from_owned_array(py, arr)
            }};
        }

        match &self.inner {
            KalmanInner::S2O1(k) => get_cov!(k),
            KalmanInner::S2O2(k) => get_cov!(k),
            KalmanInner::S4O2(k) => get_cov!(k),
            KalmanInner::S4O4(k) => get_cov!(k),
            KalmanInner::S6O3(k) => get_cov!(k),
            KalmanInner::S6O6(k) => get_cov!(k),
            KalmanInner::S8O4(k) => get_cov!(k),
            KalmanInner::S8O8(k) => get_cov!(k),
        }
    }

    /// Reset the filter state.
    ///
    /// Args:
    ///     x0 (np.ndarray, optional): New initial state. Defaults to zeros.
    ///     P0 (np.ndarray, optional): New initial covariance. Defaults to identity.
    #[pyo3(signature = (x0=None, P0=None))]
    #[allow(non_snake_case)]
    fn reset(
        &mut self,
        x0: Option<numpy::PyReadonlyArray1<f64>>,
        P0: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x0_vec: Vec<f64> = match x0 {
            Some(arr) => {
                let s = arr.as_slice()?;
                if s.len() != self.state_dim {
                    return Err(PyValueError::new_err(format!(
                        "x0 must have length {}, got {}",
                        self.state_dim,
                        s.len()
                    )));
                }
                s.to_vec()
            }
            None => vec![0.0; self.state_dim],
        };

        let p0_vec: Vec<f64> = match P0 {
            Some(arr) => {
                let s = arr.as_slice()?;
                if s.len() != self.state_dim * self.state_dim {
                    return Err(PyValueError::new_err(format!(
                        "P0 must be ({}, {})",
                        self.state_dim, self.state_dim
                    )));
                }
                s.to_vec()
            }
            None => {
                let mut p = vec![0.0; self.state_dim * self.state_dim];
                for i in 0..self.state_dim {
                    p[i * self.state_dim + i] = 1.0;
                }
                p
            }
        };

        macro_rules! do_reset {
            ($k:expr) => {{
                $k.reset(&x0_vec, &p0_vec);
            }};
        }

        match &mut self.inner {
            KalmanInner::S2O1(k) => do_reset!(k),
            KalmanInner::S2O2(k) => do_reset!(k),
            KalmanInner::S4O2(k) => do_reset!(k),
            KalmanInner::S4O4(k) => do_reset!(k),
            KalmanInner::S6O3(k) => do_reset!(k),
            KalmanInner::S6O6(k) => do_reset!(k),
            KalmanInner::S8O4(k) => do_reset!(k),
            KalmanInner::S8O8(k) => do_reset!(k),
        }
        Ok(())
    }

    /// State dimension.
    #[getter]
    fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Observation dimension.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn __repr__(&self) -> String {
        format!(
            "KalmanFilter(state_dim={}, obs_dim={})",
            self.state_dim, self.obs_dim
        )
    }
}
