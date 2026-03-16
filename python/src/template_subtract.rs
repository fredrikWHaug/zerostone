//! Python bindings for template subtraction.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::template_subtract::{
    PeelResult as ZsPeelResult, TemplateSubtractor as ZsTemplateSubtractor,
};

// --- Macro for const-generic dispatch ---

macro_rules! make_subtractor_inner {
    ($name:ident, $w:expr, $n:expr) => {
        struct $name(ZsTemplateSubtractor<$w, $n>);

        impl $name {
            fn new(max_iter: usize) -> Self {
                Self(ZsTemplateSubtractor::<$w, $n>::new(max_iter))
            }

            fn add_template(
                &mut self,
                template: &[f64],
                amp_min: f64,
                amp_max: f64,
            ) -> Result<usize, String> {
                if template.len() != $w {
                    return Err(format!(
                        "template has {} elements, expected {}",
                        template.len(),
                        $w
                    ));
                }
                let mut t = [0.0f64; $w];
                t.copy_from_slice(template);
                self.0
                    .add_template(&t, amp_min, amp_max)
                    .map_err(|e| format!("{:?}", e))
            }

            fn set_amplitude_bounds(&mut self, idx: usize, amp_min: f64, amp_max: f64) {
                self.0.set_amplitude_bounds(idx, amp_min, amp_max);
            }

            fn n_templates(&self) -> usize {
                self.0.n_templates()
            }

            fn peel(
                &self,
                data: &mut [f64],
                spike_times: &[usize],
                pre_samples: usize,
                max_results: usize,
            ) -> Vec<(usize, usize, f64)> {
                let n_times = spike_times.len();
                let cap = if max_results < 256 { max_results } else { 256 };
                let mut results = vec![
                    ZsPeelResult {
                        sample: 0,
                        template_id: 0,
                        amplitude: 0.0,
                    };
                    cap
                ];
                let n = self
                    .0
                    .peel(data, spike_times, n_times, pre_samples, &mut results);
                results
                    .iter()
                    .take(n)
                    .map(|r| (r.sample, r.template_id, r.amplitude))
                    .collect()
            }

            fn window_len(&self) -> usize {
                $w
            }
        }
    };
}

make_subtractor_inner!(Sub8x4, 8, 4);
make_subtractor_inner!(Sub8x8, 8, 8);
make_subtractor_inner!(Sub16x4, 16, 4);
make_subtractor_inner!(Sub16x8, 16, 8);
make_subtractor_inner!(Sub16x16, 16, 16);
make_subtractor_inner!(Sub32x8, 32, 8);
make_subtractor_inner!(Sub32x16, 32, 16);
make_subtractor_inner!(Sub64x16, 64, 16);
make_subtractor_inner!(Sub64x32, 64, 32);

enum SubtractorInner {
    W8N4(Box<Sub8x4>),
    W8N8(Box<Sub8x8>),
    W16N4(Box<Sub16x4>),
    W16N8(Box<Sub16x8>),
    W16N16(Box<Sub16x16>),
    W32N8(Box<Sub32x8>),
    W32N16(Box<Sub32x16>),
    W64N16(Box<Sub64x16>),
    W64N32(Box<Sub64x32>),
}

/// Greedy matching pursuit for resolving overlapping spikes.
///
/// Stores templates and iteratively subtracts the best match from data
/// following SpyKING CIRCUS's "peeling" approach.
///
/// Supported (window_len, max_templates) pairs:
/// (8, 4), (8, 8), (16, 4), (16, 8), (16, 16), (32, 8), (32, 16), (64, 16), (64, 32).
///
/// # Example
/// ```python
/// import zpybci as zbci
/// import numpy as np
///
/// sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
/// template = np.array([-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3])
/// idx = sub.add_template(template, 0.5, 2.0)
/// ```
#[pyclass]
pub struct TemplateSubtractor {
    inner: SubtractorInner,
}

macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match &$self.inner {
            SubtractorInner::W8N4(s) => s.$method($($arg),*),
            SubtractorInner::W8N8(s) => s.$method($($arg),*),
            SubtractorInner::W16N4(s) => s.$method($($arg),*),
            SubtractorInner::W16N8(s) => s.$method($($arg),*),
            SubtractorInner::W16N16(s) => s.$method($($arg),*),
            SubtractorInner::W32N8(s) => s.$method($($arg),*),
            SubtractorInner::W32N16(s) => s.$method($($arg),*),
            SubtractorInner::W64N16(s) => s.$method($($arg),*),
            SubtractorInner::W64N32(s) => s.$method($($arg),*),
        }
    };
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match &mut $self.inner {
            SubtractorInner::W8N4(s) => s.$method($($arg),*),
            SubtractorInner::W8N8(s) => s.$method($($arg),*),
            SubtractorInner::W16N4(s) => s.$method($($arg),*),
            SubtractorInner::W16N8(s) => s.$method($($arg),*),
            SubtractorInner::W16N16(s) => s.$method($($arg),*),
            SubtractorInner::W32N8(s) => s.$method($($arg),*),
            SubtractorInner::W32N16(s) => s.$method($($arg),*),
            SubtractorInner::W64N16(s) => s.$method($($arg),*),
            SubtractorInner::W64N32(s) => s.$method($($arg),*),
        }
    };
}

#[pymethods]
impl TemplateSubtractor {
    /// Create a new template subtractor.
    ///
    /// Args:
    ///     window_len (int): Template length in samples (8, 16, 32, or 64).
    ///     max_templates (int): Maximum number of templates (4, 8, 16, or 32).
    ///     max_iter (int): Maximum peeling passes. Default: 50.
    #[new]
    #[pyo3(signature = (window_len, max_templates, max_iter=50))]
    fn new(window_len: usize, max_templates: usize, max_iter: usize) -> PyResult<Self> {
        let inner = match (window_len, max_templates) {
            (8, 4) => SubtractorInner::W8N4(Box::new(Sub8x4::new(max_iter))),
            (8, 8) => SubtractorInner::W8N8(Box::new(Sub8x8::new(max_iter))),
            (16, 4) => SubtractorInner::W16N4(Box::new(Sub16x4::new(max_iter))),
            (16, 8) => SubtractorInner::W16N8(Box::new(Sub16x8::new(max_iter))),
            (16, 16) => SubtractorInner::W16N16(Box::new(Sub16x16::new(max_iter))),
            (32, 8) => SubtractorInner::W32N8(Box::new(Sub32x8::new(max_iter))),
            (32, 16) => SubtractorInner::W32N16(Box::new(Sub32x16::new(max_iter))),
            (64, 16) => SubtractorInner::W64N16(Box::new(Sub64x16::new(max_iter))),
            (64, 32) => SubtractorInner::W64N32(Box::new(Sub64x32::new(max_iter))),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported (window_len={}, max_templates={}). \
                     Supported: (8,4), (8,8), (16,4), (16,8), (16,16), \
                     (32,8), (32,16), (64,16), (64,32)",
                    window_len, max_templates
                )));
            }
        };
        Ok(Self { inner })
    }

    /// Add a template with amplitude bounds.
    ///
    /// Args:
    ///     template (np.ndarray): 1D float64 waveform of length window_len.
    ///     amp_min (float): Minimum acceptable amplitude scalar.
    ///     amp_max (float): Maximum acceptable amplitude scalar.
    ///
    /// Returns:
    ///     int: Template index.
    fn add_template(
        &mut self,
        template: PyReadonlyArray1<f64>,
        amp_min: f64,
        amp_max: f64,
    ) -> PyResult<usize> {
        let t = template.as_slice()?;
        dispatch_mut!(self, add_template, t, amp_min, amp_max).map_err(PyValueError::new_err)
    }

    /// Set amplitude bounds for an existing template.
    ///
    /// Args:
    ///     idx (int): Template index.
    ///     amp_min (float): New minimum amplitude.
    ///     amp_max (float): New maximum amplitude.
    fn set_amplitude_bounds(&mut self, idx: usize, amp_min: f64, amp_max: f64) {
        dispatch_mut!(self, set_amplitude_bounds, idx, amp_min, amp_max);
    }

    /// Number of templates currently stored.
    #[getter]
    fn n_templates(&self) -> usize {
        dispatch!(self, n_templates)
    }

    /// Greedy matching pursuit: peel overlapping spikes from data.
    ///
    /// Args:
    ///     data (np.ndarray): 1D float64 data buffer (modified in-place by subtraction).
    ///     spike_times (np.ndarray): 1D int64 candidate spike sample indices.
    ///     pre_samples (int): Samples before the spike time in the extraction window.
    ///     max_results (int): Maximum number of results to return. Default: 256.
    ///
    /// Returns:
    ///     list[dict]: List of resolved spikes, each with keys
    ///         "sample" (int), "template_id" (int), "amplitude" (float).
    #[pyo3(signature = (data, spike_times, pre_samples, max_results=256))]
    fn peel(
        &self,
        data: PyReadonlyArray1<f64>,
        spike_times: PyReadonlyArray1<i64>,
        pre_samples: usize,
        max_results: usize,
    ) -> PyResult<Vec<PyObject>> {
        // Copy data to mutable buffer
        let data_slice = data.as_slice()?;
        let mut data_buf = data_slice.to_vec();

        // Convert spike times to usize
        let times_slice = spike_times.as_slice()?;
        let times: Vec<usize> = times_slice.iter().map(|&t| t as usize).collect();

        let results = dispatch!(self, peel, &mut data_buf, &times, pre_samples, max_results);

        Python::with_gil(|py| {
            let out: Vec<PyObject> = results
                .iter()
                .map(|&(sample, template_id, amplitude)| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("sample", sample).unwrap();
                    dict.set_item("template_id", template_id).unwrap();
                    dict.set_item("amplitude", amplitude).unwrap();
                    dict.into_any().unbind()
                })
                .collect();
            Ok(out)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TemplateSubtractor(window_len={}, n_templates={})",
            dispatch!(self, window_len),
            dispatch!(self, n_templates)
        )
    }
}

/// Register template subtraction classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TemplateSubtractor>()?;
    Ok(())
}
