//! Python bindings for EDF/EDF+ file reading.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs;
use zerostone::edf;

/// An EDF/EDF+ recording loaded from a file.
///
/// Provides access to header metadata, signal information, and channel data
/// as numpy arrays.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// rec = zbci.read_edf("recording.edf")
/// print(rec.duration)       # total duration in seconds
/// print(rec.sample_rate)    # sample rate of first signal
/// ch0 = rec.get_channel(0)  # first channel as numpy array
/// ch1 = rec.get_channel("Fp1")  # channel by label
/// all_data = rec.get_all_channels()  # 2D array (channels x samples)
/// ```
#[pyclass]
pub struct EdfRecording {
    bytes: Vec<u8>,
    header: edf::EdfHeader,
    signals: Vec<edf::EdfSignalHeader>,
}

#[pymethods]
impl EdfRecording {
    /// Header metadata as a dictionary.
    ///
    /// Keys: patient_id, recording_id, start_date, start_time, n_records,
    /// record_duration, n_signals, is_edf_plus, is_continuous, header_bytes.
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("patient_id", self.header.patient_id_str())?;
        d.set_item("recording_id", self.header.recording_id_str())?;
        d.set_item(
            "start_date",
            core::str::from_utf8(&self.header.start_date).unwrap_or(""),
        )?;
        d.set_item(
            "start_time",
            core::str::from_utf8(&self.header.start_time).unwrap_or(""),
        )?;
        d.set_item("n_records", self.header.n_records)?;
        d.set_item("record_duration", self.header.record_duration)?;
        d.set_item("n_signals", self.header.n_signals)?;
        d.set_item("is_edf_plus", self.header.is_edf_plus)?;
        d.set_item("is_continuous", self.header.is_continuous)?;
        d.set_item("header_bytes", self.header.header_bytes)?;
        Ok(d)
    }

    /// List of signal info dictionaries.
    ///
    /// Each dict has keys: label, transducer, physical_dimension,
    /// physical_min, physical_max, digital_min, digital_max,
    /// n_samples_per_record, sample_rate.
    #[getter]
    fn signals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for sig in &self.signals {
            let d = PyDict::new(py);
            d.set_item("label", sig.label_str())?;
            d.set_item(
                "transducer",
                core::str::from_utf8(edf_trim_ascii(&sig.transducer_type)).unwrap_or(""),
            )?;
            d.set_item("physical_dimension", sig.physical_dimension_str())?;
            d.set_item("physical_min", sig.physical_min)?;
            d.set_item("physical_max", sig.physical_max)?;
            d.set_item("digital_min", sig.digital_min)?;
            d.set_item("digital_max", sig.digital_max)?;
            d.set_item("n_samples_per_record", sig.n_samples_per_record)?;
            d.set_item("sample_rate", sig.sample_rate(self.header.record_duration))?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Total recording duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.header.duration()
    }

    /// Sample rate of the first signal in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        if self.signals.is_empty() {
            0.0
        } else {
            self.signals[0].sample_rate(self.header.record_duration)
        }
    }

    /// Number of signals (channels).
    #[getter]
    fn n_signals(&self) -> usize {
        self.header.n_signals
    }

    /// Get one channel's data as a 1D numpy array (f64).
    ///
    /// Args:
    ///     channel: Channel index (int) or label (str).
    ///
    /// Returns:
    ///     numpy array of physical values.
    fn get_channel<'py>(
        &self,
        py: Python<'py>,
        channel: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let idx = self.resolve_channel(channel)?;
        let sig = &self.signals[idx];
        let n_records = if self.header.n_records >= 0 {
            self.header.n_records as usize
        } else {
            return Err(PyValueError::new_err("Unknown number of records"));
        };
        let total_samples = n_records * sig.n_samples_per_record;
        let mut output = vec![0.0f64; total_samples];

        edf::read_channel(&self.bytes, &self.header, &self.signals, idx, &mut output)
            .map_err(|e| PyValueError::new_err(format!("EDF read error: {:?}", e)))?;

        Ok(PyArray1::from_vec(py, output))
    }

    /// Get all channels as a 2D numpy array (n_signals x max_samples).
    ///
    /// Channels with fewer samples per record are zero-padded on the right.
    ///
    /// Returns:
    ///     2D numpy array of shape (n_signals, max_total_samples).
    fn get_all_channels<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n_records = if self.header.n_records >= 0 {
            self.header.n_records as usize
        } else {
            return Err(PyValueError::new_err("Unknown number of records"));
        };

        // Find max total samples across channels
        let max_samples = self
            .signals
            .iter()
            .map(|s| n_records * s.n_samples_per_record)
            .max()
            .unwrap_or(0);

        let n_ch = self.signals.len();
        let mut data = Array2::<f64>::zeros((n_ch, max_samples));

        for ch in 0..n_ch {
            let total = n_records * self.signals[ch].n_samples_per_record;
            let mut buf = vec![0.0f64; total];
            edf::read_channel(&self.bytes, &self.header, &self.signals, ch, &mut buf)
                .map_err(|e| PyValueError::new_err(format!("EDF read error: {:?}", e)))?;
            for (i, &v) in buf.iter().enumerate() {
                data[[ch, i]] = v;
            }
        }

        Ok(PyArray2::from_owned_array(py, data))
    }

    fn __repr__(&self) -> String {
        let labels: Vec<&str> = self.signals.iter().map(|s| s.label_str()).collect();
        format!(
            "EdfRecording(signals={}, duration={:.1}s, channels={:?})",
            self.header.n_signals,
            self.header.duration(),
            labels
        )
    }
}

impl EdfRecording {
    fn resolve_channel(&self, channel: &Bound<'_, PyAny>) -> PyResult<usize> {
        // Try integer index first
        if let Ok(idx) = channel.extract::<usize>() {
            if idx >= self.signals.len() {
                return Err(PyValueError::new_err(format!(
                    "Channel index {} out of range (0..{})",
                    idx,
                    self.signals.len()
                )));
            }
            return Ok(idx);
        }

        // Try string label
        if let Ok(label) = channel.extract::<String>() {
            for (i, sig) in self.signals.iter().enumerate() {
                if sig.label_str() == label {
                    return Ok(i);
                }
            }
            return Err(PyValueError::new_err(format!(
                "No signal with label '{}'",
                label
            )));
        }

        Err(PyValueError::new_err(
            "Channel must be an integer index or string label",
        ))
    }
}

/// Read an EDF/EDF+ file from disk.
///
/// Args:
///     filepath (str): Path to the EDF file.
///
/// Returns:
///     EdfRecording: Parsed recording with header, signals, and channel data.
///
/// Example:
///     >>> rec = zbci.read_edf("my_recording.edf")
///     >>> print(rec.header)
///     >>> ch = rec.get_channel("Fp1")
#[pyfunction]
fn read_edf(filepath: &str) -> PyResult<EdfRecording> {
    let bytes =
        fs::read(filepath).map_err(|e| PyValueError::new_err(format!("Cannot read file: {}", e)))?;

    let header = edf::parse_header(&bytes)
        .map_err(|e| PyValueError::new_err(format!("Invalid EDF header: {:?}", e)))?;

    let mut signals = Vec::with_capacity(header.n_signals);
    for i in 0..header.n_signals {
        let sig = edf::parse_signal_header(&bytes, header.n_signals, i)
            .map_err(|e| PyValueError::new_err(format!("Invalid signal header {}: {:?}", i, e)))?;
        signals.push(sig);
    }

    Ok(EdfRecording {
        bytes,
        header,
        signals,
    })
}

/// Register EDF functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_edf, m)?)?;
    Ok(())
}

/// Trim ASCII spaces (mirrors the private helper in zerostone::edf).
fn edf_trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|&b| b != b' ')
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|&b| b != b' ')
        .map(|p| p + 1)
        .unwrap_or(start);
    &bytes[start..end]
}
