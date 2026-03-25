//! Python bindings for the streaming segment-based sorter.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use zerostone::sorter::SortConfig;
use zerostone::spike_sort::MultiChannelEvent;
use zerostone::streaming::StreamingSorter as ZsStreamingSorter;

/// Streaming segment-based spike sorter.
///
/// Processes continuous recordings in fixed-length segments, maintaining
/// a template library across segments for consistent label assignment.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> probe = zbci.ProbeLayout.linear(4, 25.0)
///     >>> sorter = zbci.StreamingSorter(4)
///     >>> data = np.random.randn(5000, 4)
///     >>> result = sorter.feed(data, probe)
///     >>> sorter.n_templates
///     0
#[pyclass]
pub struct StreamingSorter {
    inner_4: Option<ZsStreamingSorter<4, 16, 48, 4, 2304, 32>>,
    inner_8: Option<ZsStreamingSorter<8, 64, 48, 4, 2304, 32>>,
    inner_16: Option<ZsStreamingSorter<16, 256, 48, 4, 2304, 32>>,
    inner_32: Option<ZsStreamingSorter<32, 1024, 48, 4, 2304, 32>>,
    n_channels: usize,
}

fn make_config(
    threshold: f64,
    detection_mode: &str,
    sneo_smooth_window: usize,
    ccg_merge: bool,
) -> PyResult<SortConfig> {
    let det_mode = match detection_mode {
        "amplitude" => zerostone::sorter::DetectionMode::Amplitude,
        "neo" => zerostone::sorter::DetectionMode::Neo,
        "sneo" => zerostone::sorter::DetectionMode::Sneo {
            smooth_window: sneo_smooth_window,
        },
        _ => {
            return Err(PyValueError::new_err(
                "detection_mode must be 'amplitude', 'neo', or 'sneo'",
            ))
        }
    };

    Ok(SortConfig {
        threshold_multiplier: threshold,
        detection_mode: det_mode,
        ccg_merge,
        ..SortConfig::default()
    })
}

#[pymethods]
impl StreamingSorter {
    /// Create a new streaming sorter.
    ///
    /// Args:
    ///     n_channels (int): Number of channels (4, 8, 16, or 32).
    ///     decay (float): EMA decay for template updates. Default: 0.95.
    ///     threshold (float): Detection threshold. Default: 5.0.
    ///     detection_mode (str): "amplitude", "neo", or "sneo". Default: "amplitude".
    ///     sneo_smooth_window (int): SNEO smooth window. Default: 3.
    ///     ccg_merge (bool): Enable CCG merge. Default: False.
    #[new]
    #[pyo3(signature = (
        n_channels,
        decay = 0.95,
        threshold = 5.0,
        detection_mode = "amplitude",
        sneo_smooth_window = 3,
        ccg_merge = false,
    ))]
    fn new(
        n_channels: usize,
        decay: f64,
        threshold: f64,
        detection_mode: &str,
        sneo_smooth_window: usize,
        ccg_merge: bool,
    ) -> PyResult<Self> {
        let config = make_config(threshold, detection_mode, sneo_smooth_window, ccg_merge)?;

        let mut s = Self {
            inner_4: None,
            inner_8: None,
            inner_16: None,
            inner_32: None,
            n_channels,
        };

        match n_channels {
            4 => s.inner_4 = Some(ZsStreamingSorter::new(config, decay)),
            8 => s.inner_8 = Some(ZsStreamingSorter::new(config, decay)),
            16 => s.inner_16 = Some(ZsStreamingSorter::new(config, decay)),
            32 => s.inner_32 = Some(ZsStreamingSorter::new(config, decay)),
            _ => return Err(PyValueError::new_err("n_channels must be 4, 8, 16, or 32")),
        }

        Ok(s)
    }

    /// Feed a segment of data to the sorter.
    ///
    /// Args:
    ///     data (np.ndarray): 2D float64 array of shape ``(n_samples, n_channels)``.
    ///     probe (ProbeLayout): Probe geometry for spatial deduplication.
    ///
    /// Returns:
    ///     dict: Same format as ``sort_multichannel`` output.
    #[pyo3(signature = (data, probe))]
    fn feed<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        probe: &super::probe::ProbeLayout,
    ) -> PyResult<PyObject> {
        let shape = data.shape();
        let n_samples = shape[0];
        let n_channels = shape[1];
        if n_channels != self.n_channels {
            return Err(PyValueError::new_err(format!(
                "expected {} channels, got {}",
                self.n_channels, n_channels
            )));
        }
        let data_slice = data.as_slice()?;
        let refractory = 15usize;
        let max_events = n_samples / refractory.max(1) + n_channels;

        macro_rules! do_feed {
            ($inner:expr, $c:expr) => {{
                let inner = $inner.as_mut().unwrap();
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
                let mut waveform_buf = vec![[0.0f64; 48]; max_events];
                let mut feature_buf = vec![[0.0f64; 4]; max_events];
                let mut labels = vec![0usize; max_events];

                let result = super::probe::with_probe_ref::<$c, _, _>(probe, |zs_probe| {
                    inner.process_segment(
                        zs_probe,
                        &mut data_owned,
                        &mut scratch,
                        &mut event_buf,
                        &mut waveform_buf,
                        &mut feature_buf,
                        &mut labels,
                    )
                })?;

                let sr = result.map_err(|e| {
                    let msg = match e {
                        zerostone::spike_sort::SortError::InsufficientData => "Insufficient data",
                        zerostone::spike_sort::SortError::EigenFailed => {
                            "Eigendecomposition failed"
                        }
                        _ => "Sort error",
                    };
                    PyValueError::new_err(msg)
                })?;

                build_streaming_result(py, &sr, &labels, &event_buf[..sr.n_spikes])
            }};
        }

        match self.n_channels {
            4 => do_feed!(self.inner_4, 4),
            8 => do_feed!(self.inner_8, 8),
            16 => do_feed!(self.inner_16, 16),
            32 => do_feed!(self.inner_32, 32),
            _ => Err(PyValueError::new_err("unsupported channel count")),
        }
    }

    /// Number of active templates in the library.
    #[getter]
    fn n_templates(&self) -> usize {
        match self.n_channels {
            4 => self.inner_4.as_ref().map_or(0, |s| s.n_templates()),
            8 => self.inner_8.as_ref().map_or(0, |s| s.n_templates()),
            16 => self.inner_16.as_ref().map_or(0, |s| s.n_templates()),
            32 => self.inner_32.as_ref().map_or(0, |s| s.n_templates()),
            _ => 0,
        }
    }

    /// Number of segments processed.
    #[getter]
    fn segment_count(&self) -> usize {
        match self.n_channels {
            4 => self.inner_4.as_ref().map_or(0, |s| s.segment_count()),
            8 => self.inner_8.as_ref().map_or(0, |s| s.segment_count()),
            16 => self.inner_16.as_ref().map_or(0, |s| s.segment_count()),
            32 => self.inner_32.as_ref().map_or(0, |s| s.segment_count()),
            _ => 0,
        }
    }

    /// Reset the sorter state (clear templates, reset segment count).
    fn reset(&mut self) {
        match self.n_channels {
            4 => {
                if let Some(s) = self.inner_4.as_mut() {
                    s.reset();
                }
            }
            8 => {
                if let Some(s) = self.inner_8.as_mut() {
                    s.reset();
                }
            }
            16 => {
                if let Some(s) = self.inner_16.as_mut() {
                    s.reset();
                }
            }
            32 => {
                if let Some(s) = self.inner_32.as_mut() {
                    s.reset();
                }
            }
            _ => {}
        }
    }
}

fn build_streaming_result<const N: usize>(
    py: Python<'_>,
    sr: &zerostone::sorter::SortResult<N>,
    labels: &[usize],
    events: &[MultiChannelEvent],
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("n_spikes", sr.n_spikes)?;
    dict.set_item("n_clusters", sr.n_clusters)?;

    let label_data: Vec<i64> = labels[..sr.n_spikes].iter().map(|&l| l as i64).collect();
    let label_py = PyArray1::from_owned_array(py, Array1::from_vec(label_data));
    dict.set_item("labels", label_py)?;

    let times_data: Vec<i64> = events.iter().map(|e| e.sample as i64).collect();
    let times_py = PyArray1::from_owned_array(py, Array1::from_vec(times_data));
    dict.set_item("spike_times", times_py)?;

    let ch_data: Vec<i64> = events.iter().map(|e| e.channel as i64).collect();
    let ch_py = PyArray1::from_owned_array(py, Array1::from_vec(ch_data));
    dict.set_item("spike_channels", ch_py)?;

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

/// Register streaming sorter.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamingSorter>()?;
    Ok(())
}
