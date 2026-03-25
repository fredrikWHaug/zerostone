//! Python bindings for spike sorting primitives.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::vec::Vec;
use zerostone::online_kmeans::OnlineKMeans as ZsOnlineKMeans;
use zerostone::spike_sort::{
    self as zs, SortError, TemplateMatch as ZsTemplateMatch, WaveformPca as ZsWaveformPca,
};

fn sort_error_to_py(e: SortError) -> PyErr {
    let msg = match e {
        SortError::InsufficientData => "Insufficient data for operation",
        SortError::NotFitted => "Model has not been fitted yet",
        SortError::EigenFailed => "Eigendecomposition failed during PCA",
        SortError::TemplateFull => "Maximum number of templates reached",
        SortError::InvalidInput => "Invalid input parameters",
    };
    PyValueError::new_err(msg)
}

// --- WaveformPca bindings ---

macro_rules! make_pca_inner {
    ($name:ident, $w:expr, $k:expr, $wm:expr) => {
        struct $name(ZsWaveformPca<$w, $k, $wm>);

        impl $name {
            fn new() -> Self {
                Self(ZsWaveformPca::new())
            }

            fn fit(&mut self, waveforms_flat: &[f64], n: usize) -> Result<(), SortError> {
                let mut wfs = Vec::with_capacity(n);
                for i in 0..n {
                    let mut wf = [0.0f64; $w];
                    wf.copy_from_slice(&waveforms_flat[i * $w..(i + 1) * $w]);
                    wfs.push(wf);
                }
                self.0.fit(&wfs)
            }

            fn transform(&self, waveform: &[f64]) -> Result<Vec<f64>, SortError> {
                let mut wf = [0.0f64; $w];
                wf.copy_from_slice(waveform);
                let mut out = [0.0f64; $k];
                self.0.transform(&wf, &mut out)?;
                Ok(out.to_vec())
            }

            fn explained_variance_ratio(&self) -> Vec<f64> {
                self.0.explained_variance_ratio().to_vec()
            }

            fn components(&self) -> Vec<f64> {
                let comps = self.0.components();
                let mut flat = Vec::with_capacity($k * $w);
                for k in 0..$k {
                    flat.extend_from_slice(&comps[k]);
                }
                flat
            }

            fn is_fitted(&self) -> bool {
                self.0.is_fitted()
            }
        }
    };
}

make_pca_inner!(PcaW32K3, 32, 3, 1024);
make_pca_inner!(PcaW32K5, 32, 5, 1024);
make_pca_inner!(PcaW48K3, 48, 3, 2304);
make_pca_inner!(PcaW48K5, 48, 5, 2304);
make_pca_inner!(PcaW64K3, 64, 3, 4096);
make_pca_inner!(PcaW64K5, 64, 5, 4096);

#[allow(clippy::large_enum_variant)] // PyO3 dispatch enum, lives on heap via #[pyclass]
enum PcaInner {
    W32K3(PcaW32K3),
    W32K5(PcaW32K5),
    W48K3(PcaW48K3),
    W48K5(Box<PcaW48K5>),
    W64K3(PcaW64K3),
    W64K5(Box<PcaW64K5>),
}

/// PCA for spike waveform dimensionality reduction.
///
/// Computes principal components of spike waveforms for feature extraction
/// before clustering or classification.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// pca = zbci.WaveformPca(window=64, n_components=3)
/// waveforms = np.random.randn(100, 64)
/// pca.fit(waveforms)
/// features = pca.transform(waveforms)  # shape (100, 3)
/// print(pca.explained_variance_ratio)
/// ```
#[pyclass]
pub struct WaveformPca {
    inner: PcaInner,
    window: usize,
    n_components: usize,
}

#[pymethods]
impl WaveformPca {
    /// Create a new WaveformPca.
    ///
    /// Args:
    ///     window (int): Waveform length (32, 48, or 64).
    ///     n_components (int): Number of PCA components (3 or 5). Default: 3.
    #[new]
    #[pyo3(signature = (window, n_components=3))]
    fn new(window: usize, n_components: usize) -> PyResult<Self> {
        let inner = match (window, n_components) {
            (32, 3) => PcaInner::W32K3(PcaW32K3::new()),
            (32, 5) => PcaInner::W32K5(PcaW32K5::new()),
            (48, 3) => PcaInner::W48K3(PcaW48K3::new()),
            (48, 5) => PcaInner::W48K5(Box::new(PcaW48K5::new())),
            (64, 3) => PcaInner::W64K3(PcaW64K3::new()),
            (64, 5) => PcaInner::W64K5(Box::new(PcaW64K5::new())),
            _ => {
                return Err(PyValueError::new_err(
                    "window must be 32, 48, or 64; n_components must be 3 or 5",
                ));
            }
        };
        Ok(Self {
            inner,
            window,
            n_components,
        })
    }

    /// Fit PCA on waveforms.
    ///
    /// Args:
    ///     waveforms (np.ndarray): 2D float64 array of shape (n_spikes, window).
    fn fit(&mut self, waveforms: PyReadonlyArray2<f64>) -> PyResult<()> {
        let shape = waveforms.shape();
        if shape[1] != self.window {
            return Err(PyValueError::new_err(format!(
                "waveforms have {} samples, expected {}",
                shape[1], self.window
            )));
        }
        let flat = waveforms.as_slice()?;
        let n = shape[0];

        macro_rules! do_fit {
            ($pca:expr) => {{
                $pca.fit(flat, n).map_err(sort_error_to_py)
            }};
        }

        match &mut self.inner {
            PcaInner::W32K3(p) => do_fit!(p),
            PcaInner::W32K5(p) => do_fit!(p),
            PcaInner::W48K3(p) => do_fit!(p),
            PcaInner::W48K5(p) => do_fit!(p),
            PcaInner::W64K3(p) => do_fit!(p),
            PcaInner::W64K5(p) => do_fit!(p),
        }
    }

    /// Project waveforms onto principal components.
    ///
    /// Args:
    ///     waveforms (np.ndarray): 2D float64 array of shape (n_spikes, window).
    ///
    /// Returns:
    ///     np.ndarray: 2D float64 array of shape (n_spikes, n_components).
    fn transform<'py>(
        &self,
        py: Python<'py>,
        waveforms: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = waveforms.shape();
        if shape[1] != self.window {
            return Err(PyValueError::new_err(format!(
                "waveforms have {} samples, expected {}",
                shape[1], self.window
            )));
        }
        let flat = waveforms.as_slice()?;
        let n = shape[0];
        let w = self.window;
        let k = self.n_components;

        macro_rules! do_transform {
            ($pca:expr) => {{
                let mut result = Vec::with_capacity(n * k);
                for i in 0..n {
                    let projected = $pca
                        .transform(&flat[i * w..(i + 1) * w])
                        .map_err(sort_error_to_py)?;
                    result.extend_from_slice(&projected);
                }
                let arr = Array2::from_shape_vec((n, k), result)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr))
            }};
        }

        match &self.inner {
            PcaInner::W32K3(p) => do_transform!(p),
            PcaInner::W32K5(p) => do_transform!(p),
            PcaInner::W48K3(p) => do_transform!(p),
            PcaInner::W48K5(p) => do_transform!(p),
            PcaInner::W64K3(p) => do_transform!(p),
            PcaInner::W64K5(p) => do_transform!(p),
        }
    }

    /// Fraction of variance explained by each component.
    #[getter]
    fn explained_variance_ratio<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        macro_rules! get_evr {
            ($pca:expr) => {{
                let evr = $pca.explained_variance_ratio();
                Ok(PyArray1::from_owned_array(py, Array1::from_vec(evr)))
            }};
        }

        match &self.inner {
            PcaInner::W32K3(p) => get_evr!(p),
            PcaInner::W32K5(p) => get_evr!(p),
            PcaInner::W48K3(p) => get_evr!(p),
            PcaInner::W48K5(p) => get_evr!(p),
            PcaInner::W64K3(p) => get_evr!(p),
            PcaInner::W64K5(p) => get_evr!(p),
        }
    }

    /// Principal components as 2D array of shape (n_components, window).
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let k = self.n_components;
        let w = self.window;

        macro_rules! get_comps {
            ($pca:expr) => {{
                let flat = $pca.components();
                let arr = Array2::from_shape_vec((k, w), flat)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr))
            }};
        }

        match &self.inner {
            PcaInner::W32K3(p) => get_comps!(p),
            PcaInner::W32K5(p) => get_comps!(p),
            PcaInner::W48K3(p) => get_comps!(p),
            PcaInner::W48K5(p) => get_comps!(p),
            PcaInner::W64K3(p) => get_comps!(p),
            PcaInner::W64K5(p) => get_comps!(p),
        }
    }

    /// Whether the PCA has been fitted.
    #[getter]
    fn is_fitted(&self) -> bool {
        match &self.inner {
            PcaInner::W32K3(p) => p.is_fitted(),
            PcaInner::W32K5(p) => p.is_fitted(),
            PcaInner::W48K3(p) => p.is_fitted(),
            PcaInner::W48K5(p) => p.is_fitted(),
            PcaInner::W64K3(p) => p.is_fitted(),
            PcaInner::W64K5(p) => p.is_fitted(),
        }
    }

    /// Waveform window length.
    #[getter]
    fn window(&self) -> usize {
        self.window
    }

    /// Number of components.
    #[getter]
    fn n_components(&self) -> usize {
        self.n_components
    }

    fn __repr__(&self) -> String {
        format!(
            "WaveformPca(window={}, n_components={}, fitted={})",
            self.window,
            self.n_components,
            self.is_fitted()
        )
    }
}

// --- TemplateMatcher bindings ---

macro_rules! make_matcher_inner {
    ($name:ident, $c:expr, $w:expr, $n:expr) => {
        struct $name(ZsTemplateMatch<$c, $w, $n>);

        impl $name {
            fn new() -> Self {
                Self(ZsTemplateMatch::new())
            }

            fn add_template(&mut self, template: &[f64]) -> Result<usize, SortError> {
                let mut t = [0.0f64; $w];
                t.copy_from_slice(template);
                self.0.add_template(&t)
            }

            fn match_euclidean(&self, waveform: &[f64]) -> Option<(usize, f64)> {
                let mut wf = [0.0f64; $w];
                wf.copy_from_slice(waveform);
                self.0.match_waveform(&wf)
            }

            fn match_ncc(&self, waveform: &[f64]) -> Option<(usize, f64)> {
                let mut wf = [0.0f64; $w];
                wf.copy_from_slice(waveform);
                self.0.match_waveform_ncc(&wf)
            }

            fn update_template(&mut self, idx: usize, waveform: &[f64]) {
                let mut wf = [0.0f64; $w];
                wf.copy_from_slice(waveform);
                self.0.update_template(idx, &wf);
            }

            fn n_templates(&self) -> usize {
                self.0.n_templates()
            }

            fn template(&self, idx: usize) -> Option<Vec<f64>> {
                self.0.template(idx).map(|t| t.to_vec())
            }
        }
    };
}

// C=1, W=32, N={4,8,16}
make_matcher_inner!(MatchC1W32N4, 1, 32, 4);
make_matcher_inner!(MatchC1W32N8, 1, 32, 8);
make_matcher_inner!(MatchC1W32N16, 1, 32, 16);
// C=1, W=64, N={4,8,16}
make_matcher_inner!(MatchC1W64N4, 1, 64, 4);
make_matcher_inner!(MatchC1W64N8, 1, 64, 8);
make_matcher_inner!(MatchC1W64N16, 1, 64, 16);
// C=4, W=32, N={4,8,16}
make_matcher_inner!(MatchC4W32N8, 4, 32, 8);
make_matcher_inner!(MatchC4W64N8, 4, 64, 8);
// C=8, W=32, N={4,8,16}
make_matcher_inner!(MatchC8W32N8, 8, 32, 8);
make_matcher_inner!(MatchC8W64N8, 8, 64, 8);

enum MatcherInner {
    C1W32N4(MatchC1W32N4),
    C1W32N8(MatchC1W32N8),
    C1W32N16(MatchC1W32N16),
    C1W64N4(MatchC1W64N4),
    C1W64N8(MatchC1W64N8),
    C1W64N16(Box<MatchC1W64N16>),
    C4W32N8(MatchC4W32N8),
    C4W64N8(MatchC4W64N8),
    C8W32N8(MatchC8W32N8),
    C8W64N8(MatchC8W64N8),
}

/// Template-based spike classification.
///
/// Classifies spike waveforms by matching them to stored templates
/// using Euclidean distance or normalized cross-correlation.
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// matcher = zbci.TemplateMatcher(channels=1, window=64, max_templates=8)
/// template = np.random.randn(64)
/// idx = matcher.add_template(template)
/// labels, distances = matcher.match_waveforms(waveforms)
/// ```
#[pyclass]
pub struct TemplateMatcher {
    inner: MatcherInner,
    channels: usize,
    window: usize,
    max_templates: usize,
}

#[pymethods]
impl TemplateMatcher {
    /// Create a new TemplateMatcher.
    ///
    /// Args:
    ///     channels (int): Number of channels (1, 4, or 8). Default: 1.
    ///     window (int): Waveform length (32 or 64). Default: 64.
    ///     max_templates (int): Maximum templates (4, 8, or 16). Default: 16.
    #[new]
    #[pyo3(signature = (channels=1, window=64, max_templates=16))]
    fn new(channels: usize, window: usize, max_templates: usize) -> PyResult<Self> {
        let inner = match (channels, window, max_templates) {
            (1, 32, 4) => MatcherInner::C1W32N4(MatchC1W32N4::new()),
            (1, 32, 8) => MatcherInner::C1W32N8(MatchC1W32N8::new()),
            (1, 32, 16) => MatcherInner::C1W32N16(MatchC1W32N16::new()),
            (1, 64, 4) => MatcherInner::C1W64N4(MatchC1W64N4::new()),
            (1, 64, 8) => MatcherInner::C1W64N8(MatchC1W64N8::new()),
            (1, 64, 16) => MatcherInner::C1W64N16(Box::new(MatchC1W64N16::new())),
            (4, 32, 8) => MatcherInner::C4W32N8(MatchC4W32N8::new()),
            (4, 64, 8) => MatcherInner::C4W64N8(MatchC4W64N8::new()),
            (8, 32, 8) => MatcherInner::C8W32N8(MatchC8W32N8::new()),
            (8, 64, 8) => MatcherInner::C8W64N8(MatchC8W64N8::new()),
            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported (channels, window, max_templates) combination. \
                     channels: 1,4,8; window: 32,64; max_templates: 4,8,16 (not all combos)",
                ));
            }
        };
        Ok(Self {
            inner,
            channels,
            window,
            max_templates,
        })
    }

    /// Add a template waveform.
    ///
    /// Args:
    ///     template (np.ndarray): 1D float64 array of length `window`.
    ///
    /// Returns:
    ///     int: Index of the added template.
    fn add_template(&mut self, template: PyReadonlyArray1<f64>) -> PyResult<usize> {
        let t = template.as_slice()?;
        if t.len() != self.window {
            return Err(PyValueError::new_err(format!(
                "template has {} samples, expected {}",
                t.len(),
                self.window
            )));
        }

        macro_rules! do_add {
            ($m:expr) => {{
                $m.add_template(t).map_err(sort_error_to_py)
            }};
        }

        match &mut self.inner {
            MatcherInner::C1W32N4(m) => do_add!(m),
            MatcherInner::C1W32N8(m) => do_add!(m),
            MatcherInner::C1W32N16(m) => do_add!(m),
            MatcherInner::C1W64N4(m) => do_add!(m),
            MatcherInner::C1W64N8(m) => do_add!(m),
            MatcherInner::C1W64N16(m) => do_add!(m),
            MatcherInner::C4W32N8(m) => do_add!(m),
            MatcherInner::C4W64N8(m) => do_add!(m),
            MatcherInner::C8W32N8(m) => do_add!(m),
            MatcherInner::C8W64N8(m) => do_add!(m),
        }
    }

    /// Match waveforms to templates.
    ///
    /// Args:
    ///     waveforms (np.ndarray): 2D float64 array of shape (n_spikes, window).
    ///     method (str): "euclidean" or "ncc". Default: "euclidean".
    ///
    /// Returns:
    ///     tuple: (labels, distances) - 1D int and 1D float arrays of length n_spikes.
    #[pyo3(signature = (waveforms, method="euclidean"))]
    #[allow(clippy::type_complexity)] // PyO3 return type with two numpy arrays
    fn match_waveforms<'py>(
        &self,
        py: Python<'py>,
        waveforms: PyReadonlyArray2<f64>,
        method: &str,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
        let shape = waveforms.shape();
        if shape[1] != self.window {
            return Err(PyValueError::new_err(format!(
                "waveforms have {} samples, expected {}",
                shape[1], self.window
            )));
        }

        let use_ncc = match method {
            "euclidean" => false,
            "ncc" => true,
            _ => return Err(PyValueError::new_err("method must be 'euclidean' or 'ncc'")),
        };

        let flat = waveforms.as_slice()?;
        let n = shape[0];
        let w = self.window;
        let mut labels = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);

        macro_rules! do_match {
            ($m:expr) => {{
                for i in 0..n {
                    let wf = &flat[i * w..(i + 1) * w];
                    let result = if use_ncc {
                        $m.match_ncc(wf)
                    } else {
                        $m.match_euclidean(wf)
                    };
                    match result {
                        Some((idx, dist)) => {
                            labels.push(idx as i64);
                            distances.push(dist);
                        }
                        None => {
                            labels.push(-1);
                            distances.push(f64::NAN);
                        }
                    }
                }
            }};
        }

        match &self.inner {
            MatcherInner::C1W32N4(m) => do_match!(m),
            MatcherInner::C1W32N8(m) => do_match!(m),
            MatcherInner::C1W32N16(m) => do_match!(m),
            MatcherInner::C1W64N4(m) => do_match!(m),
            MatcherInner::C1W64N8(m) => do_match!(m),
            MatcherInner::C1W64N16(m) => do_match!(m),
            MatcherInner::C4W32N8(m) => do_match!(m),
            MatcherInner::C4W64N8(m) => do_match!(m),
            MatcherInner::C8W32N8(m) => do_match!(m),
            MatcherInner::C8W64N8(m) => do_match!(m),
        }

        let label_arr = PyArray1::from_owned_array(py, Array1::from_vec(labels));
        let dist_arr = PyArray1::from_owned_array(py, Array1::from_vec(distances));
        Ok((label_arr, dist_arr))
    }

    /// Update a template with a running average.
    ///
    /// Args:
    ///     idx (int): Template index.
    ///     waveform (np.ndarray): 1D float64 array of length `window`.
    fn update_template(&mut self, idx: usize, waveform: PyReadonlyArray1<f64>) -> PyResult<()> {
        let wf = waveform.as_slice()?;
        if wf.len() != self.window {
            return Err(PyValueError::new_err(format!(
                "waveform has {} samples, expected {}",
                wf.len(),
                self.window
            )));
        }

        macro_rules! do_update {
            ($m:expr) => {{
                $m.update_template(idx, wf);
            }};
        }

        match &mut self.inner {
            MatcherInner::C1W32N4(m) => do_update!(m),
            MatcherInner::C1W32N8(m) => do_update!(m),
            MatcherInner::C1W32N16(m) => do_update!(m),
            MatcherInner::C1W64N4(m) => do_update!(m),
            MatcherInner::C1W64N8(m) => do_update!(m),
            MatcherInner::C1W64N16(m) => do_update!(m),
            MatcherInner::C4W32N8(m) => do_update!(m),
            MatcherInner::C4W64N8(m) => do_update!(m),
            MatcherInner::C8W32N8(m) => do_update!(m),
            MatcherInner::C8W64N8(m) => do_update!(m),
        }
        Ok(())
    }

    /// Number of stored templates.
    #[getter]
    fn n_templates(&self) -> usize {
        match &self.inner {
            MatcherInner::C1W32N4(m) => m.n_templates(),
            MatcherInner::C1W32N8(m) => m.n_templates(),
            MatcherInner::C1W32N16(m) => m.n_templates(),
            MatcherInner::C1W64N4(m) => m.n_templates(),
            MatcherInner::C1W64N8(m) => m.n_templates(),
            MatcherInner::C1W64N16(m) => m.n_templates(),
            MatcherInner::C4W32N8(m) => m.n_templates(),
            MatcherInner::C4W64N8(m) => m.n_templates(),
            MatcherInner::C8W32N8(m) => m.n_templates(),
            MatcherInner::C8W64N8(m) => m.n_templates(),
        }
    }

    /// All stored templates as 2D array of shape (n_templates, window).
    #[getter]
    fn templates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let nt = self.n_templates();
        let w = self.window;
        let mut flat = Vec::with_capacity(nt * w);

        macro_rules! get_templates {
            ($m:expr) => {{
                for i in 0..nt {
                    if let Some(t) = $m.template(i) {
                        flat.extend_from_slice(&t);
                    }
                }
            }};
        }

        match &self.inner {
            MatcherInner::C1W32N4(m) => get_templates!(m),
            MatcherInner::C1W32N8(m) => get_templates!(m),
            MatcherInner::C1W32N16(m) => get_templates!(m),
            MatcherInner::C1W64N4(m) => get_templates!(m),
            MatcherInner::C1W64N8(m) => get_templates!(m),
            MatcherInner::C1W64N16(m) => get_templates!(m),
            MatcherInner::C4W32N8(m) => get_templates!(m),
            MatcherInner::C4W64N8(m) => get_templates!(m),
            MatcherInner::C8W32N8(m) => get_templates!(m),
            MatcherInner::C8W64N8(m) => get_templates!(m),
        }

        if nt == 0 {
            let arr = Array2::<f64>::zeros((0, w));
            return Ok(PyArray2::from_owned_array(py, arr));
        }
        let arr = Array2::from_shape_vec((nt, w), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_owned_array(py, arr))
    }

    /// Number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    /// Waveform window length.
    #[getter]
    fn window(&self) -> usize {
        self.window
    }

    /// Maximum number of templates.
    #[getter]
    fn max_templates(&self) -> usize {
        self.max_templates
    }

    fn __repr__(&self) -> String {
        format!(
            "TemplateMatcher(channels={}, window={}, max_templates={}, n_templates={})",
            self.channels,
            self.window,
            self.max_templates,
            self.n_templates()
        )
    }
}

// --- Standalone functions ---

/// Extract waveform windows around spike times.
///
/// Args:
///     data (np.ndarray): 1D float64 signal.
///     spike_times (np.ndarray): 1D int array of spike sample indices.
///     window (int): Window length in samples.
///     pre_samples (int, optional): Samples before spike time. Default: window // 4.
///
/// Returns:
///     np.ndarray: 2D float64 array of shape (n_extracted, window).
#[pyfunction]
#[pyo3(signature = (data, spike_times, window, pre_samples=None))]
fn extract_waveforms<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    spike_times: PyReadonlyArray1<i64>,
    window: usize,
    pre_samples: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let data_slice = data.as_slice()?;
    let times = spike_times.as_slice()?;
    let pre = pre_samples.unwrap_or(window / 4);
    let n = data_slice.len();

    if pre >= window {
        return Err(PyValueError::new_err(
            "pre_samples must be less than window",
        ));
    }

    let mut waveforms = Vec::new();
    for &t in times {
        let t = t as usize;
        if t < pre {
            continue;
        }
        let start = t - pre;
        let end = start + window;
        if end > n {
            continue;
        }
        waveforms.extend_from_slice(&data_slice[start..end]);
    }

    let n_extracted = waveforms.len() / window;
    if n_extracted == 0 {
        let arr = Array2::<f64>::zeros((0, window));
        return Ok(PyArray2::from_owned_array(py, arr));
    }
    let arr = Array2::from_shape_vec((n_extracted, window), waveforms)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Estimate noise standard deviation using Median Absolute Deviation.
///
/// sigma = median(|x|) / 0.6745
///
/// Args:
///     data (np.ndarray): 1D float64 signal.
///
/// Returns:
///     float: Estimated noise standard deviation.
#[pyfunction]
fn estimate_noise_mad(data: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let data_slice = data.as_slice()?;
    let mut scratch = vec![0.0f64; data_slice.len()];
    Ok(zs::estimate_noise_mad(data_slice, &mut scratch))
}

/// Detect spikes via negative-threshold crossing with refractory period.
///
/// Args:
///     data (np.ndarray): 1D float64 signal.
///     threshold (float): Absolute threshold (spikes where data < -threshold).
///     refractory (int): Refractory period in samples. Default: 30.
///
/// Returns:
///     np.ndarray: 1D int array of spike times (sample indices).
#[pyfunction]
#[pyo3(signature = (data, threshold, refractory=30))]
fn detect_spikes<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    threshold: f64,
    refractory: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let data_slice = data.as_slice()?;
    // Generous upper bound on spike count
    let max_spikes = data_slice.len() / refractory.max(1) + 1;
    let mut spike_times = vec![0usize; max_spikes];
    let count = zs::detect_spikes(data_slice, threshold, refractory, &mut spike_times);
    let result: Vec<i64> = spike_times[..count].iter().map(|&t| t as i64).collect();
    Ok(PyArray1::from_owned_array(py, Array1::from_vec(result)))
}

/// One-shot spike sorting pipeline.
///
/// Runs the full spike sorting pipeline: MAD noise estimation -> threshold detection
/// -> waveform extraction -> PCA -> k-means clustering.
///
/// Args:
///     data (np.ndarray): 1D float64 raw signal.
///     sample_rate (float): Sampling rate in Hz.
///     threshold_std (float): Threshold multiplier for MAD. Default: 4.0.
///     window (int): Waveform window length (32, 48, or 64). Default: 64.
///     n_clusters (int): Number of clusters for k-means. Default: 3.
///
/// Returns:
///     dict: {"spike_times": 1D int, "waveforms": 2D, "labels": 1D int,
///            "templates": 2D, "pca_features": 2D}
#[pyfunction]
#[pyo3(signature = (data, sample_rate, threshold_std=4.0, window=64, n_clusters=3))]
fn spike_sort<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    sample_rate: f64,
    threshold_std: f64,
    window: usize,
    n_clusters: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    if window != 32 && window != 48 && window != 64 {
        return Err(PyValueError::new_err("window must be 32, 48, or 64"));
    }
    if n_clusters == 0 {
        return Err(PyValueError::new_err("n_clusters must be > 0"));
    }

    let data_slice = data.as_slice()?;
    let n = data_slice.len();

    // Step 1: MAD noise estimation
    let mut scratch = vec![0.0f64; n];
    let noise_std = zs::estimate_noise_mad(data_slice, &mut scratch);
    let threshold = threshold_std * noise_std;

    // Step 2: Detect spikes
    // Refractory: ~1ms
    let refractory = (sample_rate * 0.001) as usize;
    let refractory = refractory.max(10);
    let max_spikes = n / refractory + 1;
    let mut spike_buf = vec![0usize; max_spikes];
    let n_spikes = zs::detect_spikes(data_slice, threshold, refractory, &mut spike_buf);

    if n_spikes < n_clusters {
        return Err(PyValueError::new_err(format!(
            "Only {} spikes detected, need at least {}",
            n_spikes, n_clusters
        )));
    }

    // Step 3: Extract waveforms
    let pre = window / 4;
    let mut waveforms_flat = Vec::new();
    let mut valid_times = Vec::new();
    for &t in &spike_buf[..n_spikes] {
        if t < pre {
            continue;
        }
        let start = t - pre;
        let end = start + window;
        if end > n {
            continue;
        }
        waveforms_flat.extend_from_slice(&data_slice[start..end]);
        valid_times.push(t as i64);
    }
    let n_valid = valid_times.len();
    if n_valid < n_clusters {
        return Err(PyValueError::new_err(format!(
            "Only {} valid waveforms extracted, need at least {}",
            n_valid, n_clusters
        )));
    }

    // Step 4: PCA (3 components)
    let n_components = 3;
    let mut pca_features = Vec::with_capacity(n_valid * n_components);

    // Use window-specific PCA
    match window {
        32 => {
            let mut pca = ZsWaveformPca::<32, 3, 1024>::new();
            let mut wfs = Vec::with_capacity(n_valid);
            for i in 0..n_valid {
                let mut wf = [0.0f64; 32];
                wf.copy_from_slice(&waveforms_flat[i * 32..(i + 1) * 32]);
                wfs.push(wf);
            }
            pca.fit(&wfs).map_err(sort_error_to_py)?;
            for wf in &wfs {
                let mut out = [0.0f64; 3];
                pca.transform(wf, &mut out).map_err(sort_error_to_py)?;
                pca_features.extend_from_slice(&out);
            }
        }
        48 => {
            let mut pca = ZsWaveformPca::<48, 3, 2304>::new();
            let mut wfs = Vec::with_capacity(n_valid);
            for i in 0..n_valid {
                let mut wf = [0.0f64; 48];
                wf.copy_from_slice(&waveforms_flat[i * 48..(i + 1) * 48]);
                wfs.push(wf);
            }
            pca.fit(&wfs).map_err(sort_error_to_py)?;
            for wf in &wfs {
                let mut out = [0.0f64; 3];
                pca.transform(wf, &mut out).map_err(sort_error_to_py)?;
                pca_features.extend_from_slice(&out);
            }
        }
        64 => {
            let mut pca = ZsWaveformPca::<64, 3, 4096>::new();
            let mut wfs = Vec::with_capacity(n_valid);
            for i in 0..n_valid {
                let mut wf = [0.0f64; 64];
                wf.copy_from_slice(&waveforms_flat[i * 64..(i + 1) * 64]);
                wfs.push(wf);
            }
            pca.fit(&wfs).map_err(sort_error_to_py)?;
            for wf in &wfs {
                let mut out = [0.0f64; 3];
                pca.transform(wf, &mut out).map_err(sort_error_to_py)?;
                pca_features.extend_from_slice(&out);
            }
        }
        _ => unreachable!(),
    }

    // Step 5: K-means clustering in PCA space using OnlineKMeans
    let k = n_clusters;

    // Seed centroids from evenly-spaced waveforms, then assign all points
    let mut km = ZsOnlineKMeans::<3, 32>::new(10000);
    let step = n_valid / k;
    for c in 0..k {
        let idx = c * step;
        let mut centroid = [0.0f64; 3];
        centroid.copy_from_slice(&pca_features[idx * 3..(idx + 1) * 3]);
        let _ = km.seed_centroid(&centroid);
    }

    // Run two passes: first pass assigns all points, second pass refines
    let mut labels = vec![0usize; n_valid];
    for _pass in 0..2 {
        for i in 0..n_valid {
            let mut feat = [0.0f64; 3];
            feat.copy_from_slice(&pca_features[i * 3..(i + 1) * 3]);
            let result = km.update(&feat);
            labels[i] = result.cluster;
        }
    }

    // Compute mean template per cluster
    let mut templates_flat = vec![0.0f64; k * window];
    let mut cluster_counts = vec![0usize; k];
    for i in 0..n_valid {
        let c = labels[i];
        cluster_counts[c] += 1;
        for j in 0..window {
            templates_flat[c * window + j] += waveforms_flat[i * window + j];
        }
    }
    for c in 0..k {
        if cluster_counts[c] > 0 {
            for j in 0..window {
                templates_flat[c * window + j] /= cluster_counts[c] as f64;
            }
        }
    }

    // Build result dict
    let dict = pyo3::types::PyDict::new(py);

    let spike_times_arr = PyArray1::from_owned_array(py, Array1::from_vec(valid_times));
    dict.set_item("spike_times", spike_times_arr)?;

    let wf_arr = Array2::from_shape_vec((n_valid, window), waveforms_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    dict.set_item("waveforms", PyArray2::from_owned_array(py, wf_arr))?;

    let labels_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    dict.set_item(
        "labels",
        PyArray1::from_owned_array(py, Array1::from_vec(labels_i64)),
    )?;

    let templates_arr = Array2::from_shape_vec((k, window), templates_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    dict.set_item("templates", PyArray2::from_owned_array(py, templates_arr))?;

    let pca_arr = Array2::from_shape_vec((n_valid, n_components), pca_features)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    dict.set_item("pca_features", PyArray2::from_owned_array(py, pca_arr))?;

    Ok(dict)
}

// --- Multi-channel detection bindings ---

use zerostone::probe::ProbeLayout as ZsProbeLayout;
use zerostone::spike_sort::MultiChannelEvent;

/// Helper: convert a list of Python dicts to a Vec of MultiChannelEvent.
fn dicts_to_events(
    _py: Python<'_>,
    events: &Bound<'_, pyo3::types::PyList>,
) -> PyResult<Vec<MultiChannelEvent>> {
    let mut result = Vec::with_capacity(events.len());
    for item in events.iter() {
        let dict = item.downcast::<pyo3::types::PyDict>()?;
        let sample: usize = dict
            .get_item("sample")?
            .ok_or_else(|| PyValueError::new_err("event missing 'sample' key"))?
            .extract()?;
        let channel: usize = dict
            .get_item("channel")?
            .ok_or_else(|| PyValueError::new_err("event missing 'channel' key"))?
            .extract()?;
        let amplitude: f64 = dict
            .get_item("amplitude")?
            .ok_or_else(|| PyValueError::new_err("event missing 'amplitude' key"))?
            .extract()?;
        result.push(MultiChannelEvent {
            sample,
            channel,
            amplitude,
        });
    }
    Ok(result)
}

/// Helper: convert a slice of MultiChannelEvent to a list of Python dicts.
fn events_to_dicts<'py>(
    py: Python<'py>,
    events: &[MultiChannelEvent],
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let list = pyo3::types::PyList::empty(py);
    for e in events {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("sample", e.sample)?;
        dict.set_item("channel", e.channel)?;
        dict.set_item("amplitude", e.amplitude)?;
        list.append(dict)?;
    }
    Ok(list)
}

/// Detect spikes across multiple channels with per-channel noise thresholds.
///
/// Each channel is thresholded independently at `threshold * noise[ch]`.
/// Within the refractory window, only the sample with the largest negative
/// deflection is kept. Events are returned sorted by sample time.
///
/// Args:
///     data (np.ndarray): 2D float64 array of shape (n_samples, n_channels).
///         n_channels must be 2, 4, 8, 16, 32, or 64.
///     threshold (float): Threshold multiplier applied to per-channel noise.
///     noise (np.ndarray): 1D float64 array of per-channel noise estimates.
///     refractory (int): Refractory period in samples. Default: 30.
///
/// Returns:
///     list[dict]: List of ``{"sample": int, "channel": int, "amplitude": float}``.
#[pyfunction]
#[pyo3(signature = (data, threshold, noise, refractory=30))]
fn detect_spikes_multichannel<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    threshold: f64,
    noise: PyReadonlyArray1<f64>,
    refractory: usize,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let shape = data.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let noise_slice = noise.as_slice()?;
    if noise_slice.len() != n_channels {
        return Err(PyValueError::new_err(format!(
            "noise has {} elements, expected {}",
            noise_slice.len(),
            n_channels
        )));
    }
    let data_slice = data.as_slice()?;

    // Allocate output buffer: generous upper bound
    let max_events = n_samples * n_channels / refractory.max(1) + n_channels;

    macro_rules! do_detect {
        ($c:expr) => {{
            // Convert flat row-major [n_samples * C] to &[[f64; C]]
            let mut noise_arr = [0.0f64; $c];
            noise_arr.copy_from_slice(noise_slice);
            // Safety: reinterpret flat slice as &[[f64; C]]
            // The data is n_samples rows of C f64s each, contiguous in memory
            let typed_data: &[[f64; $c]] = {
                assert!(data_slice.len() == n_samples * $c);
                // SAFETY: [f64; C] has the same layout as C contiguous f64s,
                // and data_slice has exactly n_samples * C elements.
                unsafe {
                    core::slice::from_raw_parts(data_slice.as_ptr() as *const [f64; $c], n_samples)
                }
            };
            let mut events = vec![
                MultiChannelEvent {
                    sample: 0,
                    channel: 0,
                    amplitude: 0.0
                };
                max_events
            ];
            let n = zs::detect_spikes_multichannel::<$c>(
                typed_data,
                threshold,
                &noise_arr,
                refractory,
                &mut events,
            );
            events_to_dicts(py, &events[..n])
        }};
    }

    match n_channels {
        2 => do_detect!(2),
        4 => do_detect!(4),
        8 => do_detect!(8),
        16 => do_detect!(16),
        32 => do_detect!(32),
        64 => do_detect!(64),
        _ => Err(PyValueError::new_err(
            "n_channels must be 2, 4, 8, 16, 32, or 64",
        )),
    }
}

/// Remove duplicate spike detections across neighboring channels.
///
/// When a neuron fires, its waveform appears on multiple nearby channels.
/// This function keeps only the event with the largest amplitude within
/// each spatiotemporal neighborhood.
///
/// Args:
///     events (list[dict]): List of ``{"sample": int, "channel": int, "amplitude": float}``.
///     probe (ProbeLayout): Probe geometry for spatial neighbor queries.
///     temporal_radius (int): Maximum sample offset for duplicates. Default: 5.
///     spatial_radius (float): Maximum channel distance in micrometers. Default: 30.0.
///
/// Returns:
///     list[dict]: Deduplicated events.
#[pyfunction]
#[pyo3(signature = (events, probe, temporal_radius=5, spatial_radius=30.0))]
fn deduplicate_events<'py>(
    py: Python<'py>,
    events: Bound<'py, pyo3::types::PyList>,
    probe: &super::probe::ProbeLayout,
    temporal_radius: usize,
    spatial_radius: f64,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let mut ev = dicts_to_events(py, &events)?;
    let n_events = ev.len();
    let n_channels = probe.n_channels_inner();

    macro_rules! do_dedup {
        ($c:expr) => {{
            let probe_inner = ZsProbeLayout::<$c>::linear(1.0); // placeholder
            // We need the actual probe. Extract from the binding.
            // The probe binding stores a ProbeInner enum; we can use
            // its channel_distance method through a constructed local probe.
            // Instead, build a ProbeLayout from the Python probe's distances.
            // Actually, the cleanest approach: construct from positions.
            // We'll build a local probe by querying all channel distances.
            let _ = probe_inner; // suppress warning
            // Build positions by querying the probe -- unfortunately the Python
            // ProbeLayout doesn't expose positions. Use a different approach:
            // call deduplicate on each pair using the Python probe's distance.
            //
            // Better approach: reconstruct a Rust probe from the Python binding.
            // The probe binding stores ProbeInner which wraps ZsProbeLayout.
            // We need to extract the inner. Let's use a helper on ProbeLayout.
            //
            // Actually the simplest approach: implement dedup manually using
            // the Python probe's channel_distance method. But that's wasteful.
            //
            // The real simplest approach: accept that we need to build positions.
            // Use the Python probe to query distances and reconstruct positions.
            // For a linear probe, position[i] = (0, i * pitch).
            //
            // Wait -- we can just use the dedup function with a locally-constructed
            // probe by querying all pairwise distances from the Python probe.
            // But that defeats the purpose.
            //
            // Actually, let's add a helper method to the probe binding that
            // returns the inner reference. Or better: duplicate the approach used
            // in the Rust tests -- just dispatch on the ProbeInner enum variant.
            let n = super::probe::with_probe_ref::<$c, _, _>(&probe, |zs_probe| {
                zs::deduplicate_events::<$c>(&mut ev, n_events, zs_probe, spatial_radius, temporal_radius)
            })?;
            events_to_dicts(py, &ev[..n])
        }};
    }

    match n_channels {
        4 => do_dedup!(4),
        8 => do_dedup!(8),
        16 => do_dedup!(16),
        32 => do_dedup!(32),
        64 => do_dedup!(64),
        128 => do_dedup!(128),
        _ => Err(PyValueError::new_err(
            "probe n_channels must be 4, 8, 16, 32, 64, or 128",
        )),
    }
}

/// Fine-align spike event times to the nearest negative peak.
///
/// For each event, searches the data on the event's channel within
/// ``+/- half_window`` samples for the most negative value, then
/// updates the event's sample and amplitude.
///
/// Args:
///     events (list[dict]): List of ``{"sample": int, "channel": int, "amplitude": float}``.
///     data (np.ndarray): 2D float64 array of shape (n_samples, n_channels).
///     half_window (int): Search radius in samples. Default: 5.
///
/// Returns:
///     list[dict]: Events with aligned sample times and updated amplitudes.
#[pyfunction]
#[pyo3(signature = (events, data, half_window=5))]
fn align_to_peak<'py>(
    py: Python<'py>,
    events: Bound<'py, pyo3::types::PyList>,
    data: PyReadonlyArray2<f64>,
    half_window: usize,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let mut ev = dicts_to_events(py, &events)?;
    let n_events = ev.len();
    let shape = data.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let data_slice = data.as_slice()?;

    macro_rules! do_align {
        ($c:expr) => {{
            let typed_data: &[[f64; $c]] = {
                assert!(data_slice.len() == n_samples * $c);
                unsafe {
                    core::slice::from_raw_parts(data_slice.as_ptr() as *const [f64; $c], n_samples)
                }
            };
            zs::align_to_peak::<$c>(typed_data, &mut ev, n_events, half_window);
            events_to_dicts(py, &ev[..n_events])
        }};
    }

    match n_channels {
        1 => do_align!(1),
        2 => do_align!(2),
        4 => do_align!(4),
        8 => do_align!(8),
        16 => do_align!(16),
        32 => do_align!(32),
        64 => do_align!(64),
        _ => Err(PyValueError::new_err(
            "n_channels must be 1, 2, 4, 8, 16, 32, or 64",
        )),
    }
}

/// Combine PCA features with spatial features into a single feature vector.
///
/// Concatenates PCA features (unscaled) with spatial features (scaled by
/// ``spatial_weight``) to produce a combined feature array suitable for
/// clustering.
///
/// Args:
///     pca_features (np.ndarray): 2D float64 array of shape ``(n, k)`` -- PCA features.
///     spatial_features (np.ndarray): 2D float64 array of shape ``(n, s)`` -- spatial features.
///     spatial_weight (float): Scaling factor applied to spatial features.
///
/// Returns:
///     np.ndarray: 2D float64 array of shape ``(n, k + s)`` with combined features.
#[pyfunction]
fn combine_features<'py>(
    py: Python<'py>,
    pca_features: PyReadonlyArray2<f64>,
    spatial_features: PyReadonlyArray2<f64>,
    spatial_weight: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pca_shape = pca_features.shape();
    let spa_shape = spatial_features.shape();
    let n = pca_shape[0];
    let k = pca_shape[1];
    let s = spa_shape[1];

    if spa_shape[0] != n {
        return Err(PyValueError::new_err(format!(
            "pca_features has {} rows but spatial_features has {} rows",
            n, spa_shape[0]
        )));
    }

    let pca_slice = pca_features.as_slice()?;
    let spa_slice = spatial_features.as_slice()?;
    let t = k + s;

    let mut output = vec![0.0f64; n * t];
    for i in 0..n {
        for j in 0..k {
            output[i * t + j] = pca_slice[i * k + j];
        }
        for j in 0..s {
            output[i * t + k + j] = spa_slice[i * s + j] * spatial_weight;
        }
    }

    let arr =
        Array2::from_shape_vec((n, t), output).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Extract spatial position features for each spike from multi-channel data.
///
/// For each detected event, extracts peak amplitudes across channels in a
/// small window and computes center-of-mass position using probe geometry.
///
/// Args:
///     data (np.ndarray): 2D float64 array of shape ``(n_samples, n_channels)``.
///         n_channels must be 2, 4, 8, 16, 32, or 64.
///     events (list[dict]): List of ``{"sample": int, "channel": int, "amplitude": float}``.
///     positions (np.ndarray): 2D float64 array of shape ``(n_channels, 2)`` -- channel positions.
///
/// Returns:
///     np.ndarray: 2D float64 array of shape ``(n_events, 2)`` with (x, y) positions.
#[pyfunction]
fn extract_spatial_features<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f64>,
    events: Bound<'py, pyo3::types::PyList>,
    positions: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = data.shape();
    let n_samples = shape[0];
    let n_channels = shape[1];
    let data_slice = data.as_slice()?;

    let pos_shape = positions.shape();
    if pos_shape[0] != n_channels || pos_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape ({}, 2), got ({}, {})",
            n_channels, pos_shape[0], pos_shape[1]
        )));
    }
    let pos_slice = positions.as_slice()?;

    let ev = dicts_to_events(py, &events)?;
    let n_events = ev.len();

    macro_rules! do_extract {
        ($c:expr) => {{
            let typed_data: &[[f64; $c]] = {
                assert!(data_slice.len() == n_samples * $c);
                unsafe {
                    core::slice::from_raw_parts(data_slice.as_ptr() as *const [f64; $c], n_samples)
                }
            };
            let mut pos_arr = [[0.0f64; 2]; $c];
            for i in 0..$c {
                pos_arr[i][0] = pos_slice[i * 2];
                pos_arr[i][1] = pos_slice[i * 2 + 1];
            }
            let mut output = vec![[0.0f64; 2]; n_events];
            let n = zs::extract_spatial_features::<$c>(
                typed_data,
                &ev,
                n_events,
                &pos_arr,
                &mut output,
            );
            let mut flat = Vec::with_capacity(n * 2);
            for i in 0..n {
                flat.push(output[i][0]);
                flat.push(output[i][1]);
            }
            let arr = Array2::from_shape_vec((n, 2), flat)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }};
    }

    match n_channels {
        2 => do_extract!(2),
        4 => do_extract!(4),
        8 => do_extract!(8),
        16 => do_extract!(16),
        32 => do_extract!(32),
        64 => do_extract!(64),
        _ => Err(PyValueError::new_err(
            "n_channels must be 2, 4, 8, 16, 32, or 64",
        )),
    }
}

/// Register spike sorting functions and classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_waveforms, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise_mad, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(spike_sort, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes_multichannel, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_events, m)?)?;
    m.add_function(wrap_pyfunction!(align_to_peak, m)?)?;
    m.add_function(wrap_pyfunction!(combine_features, m)?)?;
    m.add_function(wrap_pyfunction!(extract_spatial_features, m)?)?;
    Ok(())
}
