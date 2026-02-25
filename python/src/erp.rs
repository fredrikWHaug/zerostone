//! Python bindings for Event-Related Potential (ERP) analysis.
//!
//! Provides epoch averaging and xDAWN spatial filtering for P300 and other ERP-based BCIs.

use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::erp;

/// Compute average ERP across multiple epochs.
///
/// Args:
///     epochs (np.ndarray): 3D array of shape (n_trials, n_samples, n_channels).
///
/// Returns:
///     np.ndarray: Average epoch with shape (n_samples, n_channels).
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> epochs = np.random.randn(10, 100, 8).astype(np.float64)
///     >>> avg = zbci.epoch_average(epochs)
///     >>> avg.shape
///     (100, 8)
#[pyfunction]
fn epoch_average<'py>(
    py: Python<'py>,
    epochs: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = epochs.shape();
    let (n_trials, n_samples, n_channels) = (shape[0], shape[1], shape[2]);

    if n_trials == 0 || n_samples == 0 || n_channels == 0 {
        return Err(PyValueError::new_err("Empty epochs array"));
    }

    let data = epochs.as_slice()?;

    // Dispatch based on number of channels
    macro_rules! average_epochs {
        ($c:expr) => {{
            let mut output = vec![[0.0; $c]; n_samples];
            let mut trial_refs: Vec<&[[f64; $c]]> = Vec::with_capacity(n_trials);

            for trial_idx in 0..n_trials {
                let trial_offset = trial_idx * n_samples * $c;
                let mut trial_data = vec![[0.0; $c]; n_samples];

                for sample_idx in 0..n_samples {
                    for ch_idx in 0..$c {
                        trial_data[sample_idx][ch_idx] =
                            data[trial_offset + sample_idx * $c + ch_idx];
                    }
                }
                trial_refs.push(Box::leak(trial_data.into_boxed_slice()));
            }

            erp::epoch_average(&trial_refs, &mut output);

            // Clean up leaked memory
            for trial_ref in trial_refs {
                unsafe {
                    let _ = Box::from_raw(trial_ref as *const [[f64; $c]] as *mut [[f64; $c]]);
                }
            }

            // Convert to numpy - shape (n_samples, n_channels)
            let flat: Vec<Vec<f64>> = output.iter().map(|s| s.to_vec()).collect();
            Ok(PyArray2::from_vec2(py, &flat)?)
        }};
    }

    match n_channels {
        2 => average_epochs!(2),
        4 => average_epochs!(4),
        8 => average_epochs!(8),
        16 => average_epochs!(16),
        32 => average_epochs!(32),
        64 => average_epochs!(64),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported number of channels: {}. Supported: 2, 4, 8, 16, 32, 64",
            n_channels
        ))),
    }
}

/// sklearn-compatible xDAWN spatial filter transformer for ERP analysis.
///
/// Learns spatial filters that maximize the signal-to-signal-plus-noise ratio
/// of event-related potentials.
///
/// Parameters:
///     channels (int): Number of EEG channels (2, 4, 8, 16, 32, or 64).
///     filters (int): Number of spatial filters to extract (2, 4, or 6).
///     regularization (float): Regularization for covariance matrices. Default: 1e-6.
///     max_iters (int): Max iterations for eigenvalue decomposition. Default: 30.
///     tol (float): Convergence tolerance. Default: 1e-10.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> from sklearn.pipeline import Pipeline
///     >>>
///     >>> # Create transformer
///     >>> xdawn = zbci.XDawnTransformer(channels=8, filters=2)
///     >>>
///     >>> # Training data: (n_trials, n_samples, n_channels)
///     >>> X = np.random.randn(100, 200, 8).astype(np.float64)
///     >>> y = np.random.randint(0, 2, 100)  # Binary labels (target/nontarget)
///     >>>
///     >>> # Fit and transform
///     >>> xdawn.fit(X, y)
///     >>> X_filtered = xdawn.transform(X)
///     >>> X_filtered.shape
///     (100, 200, 2)
#[pyclass]
pub struct XDawnTransformer {
    channels: usize,
    num_filters: usize,
    regularization: f64,
    max_iters: usize,
    tol: f64,
    filters: Option<Vec<Vec<f64>>>,  // F × C stored as Vec<Vec>
}

#[pymethods]
impl XDawnTransformer {
    #[new]
    #[pyo3(signature = (channels, filters, regularization=1e-6, max_iters=30, tol=1e-10))]
    fn new(
        channels: usize,
        filters: usize,
        regularization: f64,
        max_iters: usize,
        tol: f64,
    ) -> PyResult<Self> {
        // Validate channels and filters
        if !matches!(channels, 2 | 4 | 8 | 16 | 32 | 64) {
            return Err(PyValueError::new_err(format!(
                "Unsupported channels: {}. Supported: 2, 4, 8, 16, 32, 64",
                channels
            )));
        }

        if !matches!(filters, 1 | 2 | 4 | 6) {
            return Err(PyValueError::new_err(format!(
                "Unsupported filters: {}. Supported: 1, 2, 4, 6",
                filters
            )));
        }

        if filters > channels {
            return Err(PyValueError::new_err(format!(
                "Number of filters ({}) cannot exceed channels ({})",
                filters, channels
            )));
        }

        Ok(Self {
            channels,
            num_filters: filters,
            regularization,
            max_iters,
            tol,
            filters: None,
        })
    }

    /// Fit xDAWN spatial filters from labeled epochs.
    ///
    /// Args:
    ///     X (np.ndarray): Epochs with shape (n_trials, n_samples, n_channels).
    ///     y (np.ndarray): Labels with shape (n_trials,). 1 for target, 0 for nontarget.
    ///
    /// Returns:
    ///     self: Fitted transformer.
    fn fit(&mut self, X: PyReadonlyArray3<f64>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let shape = X.shape();
        let (n_trials, n_samples, n_channels) = (shape[0], shape[1], shape[2]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "X has {} channels, expected {}",
                n_channels, self.channels
            )));
        }

        let y_data = y.as_slice()?;
        if y_data.len() != n_trials {
            return Err(PyValueError::new_err(format!(
                "y has {} elements, expected {}",
                y_data.len(),
                n_trials
            )));
        }

        let X_data = X.as_slice()?;

        // Dispatch based on channels and filters
        macro_rules! fit_xdawn {
            ($c:expr, $f:expr) => {{
                const M: usize = $c * $c;

                // Separate target and nontarget epochs
                let mut target_epochs: Vec<Vec<[f64; $c]>> = Vec::new();
                let mut nontarget_epochs: Vec<Vec<[f64; $c]>> = Vec::new();

                for trial_idx in 0..n_trials {
                    let trial_offset = trial_idx * n_samples * $c;
                    let mut trial = vec![[0.0; $c]; n_samples];

                    for sample_idx in 0..n_samples {
                        for ch_idx in 0..$c {
                            trial[sample_idx][ch_idx] =
                                X_data[trial_offset + sample_idx * $c + ch_idx];
                        }
                    }

                    if y_data[trial_idx] == 1 {
                        target_epochs.push(trial);
                    } else {
                        nontarget_epochs.push(trial);
                    }
                }

                if target_epochs.is_empty() || nontarget_epochs.is_empty() {
                    return Err(PyValueError::new_err(
                        "Need at least one target and one nontarget epoch",
                    ));
                }

                // Convert to references
                let target_refs: Vec<&[[f64; $c]]> =
                    target_epochs.iter().map(|e| &e[..]).collect();
                let nontarget_refs: Vec<&[[f64; $c]]> =
                    nontarget_epochs.iter().map(|e| &e[..]).collect();

                // Learn filters
                let mut evoked_workspace = vec![[0.0; $c]; n_samples];
                let mut filters_array = [[0.0; $c]; $f];
                erp::xdawn_filters::<$c, M, $f>(
                    &target_refs,
                    &nontarget_refs,
                    &mut evoked_workspace,
                    &mut filters_array,
                    self.regularization,
                    self.max_iters,
                    self.tol,
                )
                .map_err(|e| PyValueError::new_err(format!("xDAWN failed: {:?}", e)))?;

                // Store filters as Vec<Vec<f64>> for later use
                self.filters = Some(
                    filters_array
                        .iter()
                        .map(|f| f.to_vec())
                        .collect()
                );

                Ok(())
            }};
        }

        match (self.channels, self.num_filters) {
            (2, 1) => fit_xdawn!(2, 1),
            (2, 2) => fit_xdawn!(2, 2),
            (4, 1) => fit_xdawn!(4, 1),
            (4, 2) => fit_xdawn!(4, 2),
            (4, 4) => fit_xdawn!(4, 4),
            (8, 1) => fit_xdawn!(8, 1),
            (8, 2) => fit_xdawn!(8, 2),
            (8, 4) => fit_xdawn!(8, 4),
            (8, 6) => fit_xdawn!(8, 6),
            (16, 1) => fit_xdawn!(16, 1),
            (16, 2) => fit_xdawn!(16, 2),
            (16, 4) => fit_xdawn!(16, 4),
            (16, 6) => fit_xdawn!(16, 6),
            (32, 1) => fit_xdawn!(32, 1),
            (32, 2) => fit_xdawn!(32, 2),
            (32, 4) => fit_xdawn!(32, 4),
            (32, 6) => fit_xdawn!(32, 6),
            (64, 1) => fit_xdawn!(64, 1),
            (64, 2) => fit_xdawn!(64, 2),
            (64, 4) => fit_xdawn!(64, 4),
            (64, 6) => fit_xdawn!(64, 6),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported channels/filters combination: {}/{}",
                self.channels, self.num_filters
            ))),
        }
    }

    /// Transform epochs using learned spatial filters.
    ///
    /// Args:
    ///     X (np.ndarray): Epochs with shape (n_trials, n_samples, n_channels).
    ///
    /// Returns:
    ///     np.ndarray: Filtered epochs with shape (n_trials, n_samples, n_filters).
    fn transform<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let filters = self.filters.as_ref().ok_or_else(|| {
            PyValueError::new_err("Transformer not fitted. Call fit() first.")
        })?;

        let shape = X.shape();
        let (n_trials, n_samples, n_channels) = (shape[0], shape[1], shape[2]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "X has {} channels, expected {}",
                n_channels, self.channels
            )));
        }

        let X_data = X.as_slice()?;
        let mut output = vec![0.0; n_trials * n_samples * self.num_filters];

        // Apply filters
        macro_rules! apply_filters {
            ($c:expr, $f:expr) => {{
                let mut filters_array = [[0.0; $c]; $f];
                for (i, filter_vec) in filters.iter().enumerate() {
                    for (j, &val) in filter_vec.iter().enumerate() {
                        filters_array[i][j] = val;
                    }
                }

                for trial_idx in 0..n_trials {
                    let trial_offset = trial_idx * n_samples * $c;
                    let mut epoch = vec![[0.0; $c]; n_samples];

                    for sample_idx in 0..n_samples {
                        for ch_idx in 0..$c {
                            epoch[sample_idx][ch_idx] =
                                X_data[trial_offset + sample_idx * $c + ch_idx];
                        }
                    }

                    let mut filtered = vec![[0.0; $f]; n_samples];
                    erp::apply_spatial_filter(&epoch, &filters_array, &mut filtered);

                    // Copy to output
                    let out_offset = trial_idx * n_samples * $f;
                    for sample_idx in 0..n_samples {
                        for f_idx in 0..$f {
                            output[out_offset + sample_idx * $f + f_idx] =
                                filtered[sample_idx][f_idx];
                        }
                    }
                }

                Ok::<(), PyErr>(())
            }};
        }

        match (self.channels, self.num_filters) {
            (2, 1) => apply_filters!(2, 1)?,
            (2, 2) => apply_filters!(2, 2)?,
            (4, 1) => apply_filters!(4, 1)?,
            (4, 2) => apply_filters!(4, 2)?,
            (4, 4) => apply_filters!(4, 4)?,
            (8, 1) => apply_filters!(8, 1)?,
            (8, 2) => apply_filters!(8, 2)?,
            (8, 4) => apply_filters!(8, 4)?,
            (8, 6) => apply_filters!(8, 6)?,
            (16, 1) => apply_filters!(16, 1)?,
            (16, 2) => apply_filters!(16, 2)?,
            (16, 4) => apply_filters!(16, 4)?,
            (16, 6) => apply_filters!(16, 6)?,
            (32, 1) => apply_filters!(32, 1)?,
            (32, 2) => apply_filters!(32, 2)?,
            (32, 4) => apply_filters!(32, 4)?,
            (32, 6) => apply_filters!(32, 6)?,
            (64, 1) => apply_filters!(64, 1)?,
            (64, 2) => apply_filters!(64, 2)?,
            (64, 4) => apply_filters!(64, 4)?,
            (64, 6) => apply_filters!(64, 6)?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported channels/filters: {}/{}",
                    self.channels, self.num_filters
                )))
            }
        }

        // Reshape to 3D array
        Ok(PyArray3::from_vec3(
            py,
            &(0..n_trials)
                .map(|trial_idx| {
                    (0..n_samples)
                        .map(|sample_idx| {
                            let offset = trial_idx * n_samples * self.num_filters
                                + sample_idx * self.num_filters;
                            output[offset..offset + self.num_filters].to_vec()
                        })
                        .collect()
                })
                .collect::<Vec<_>>(),
        )?)
    }

    fn __repr__(&self) -> String {
        let fitted = if self.filters.is_some() { "True" } else { "False" };
        format!(
            "XDawnTransformer(channels={}, filters={}, regularization={}, fitted={})",
            self.channels,
            self.num_filters,
            self.regularization,
            fitted
        )
    }
}

/// Register ERP module functions and classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(epoch_average, m)?)?;
    m.add_class::<XDawnTransformer>()?;
    Ok(())
}
