//! Python bindings for XDF file reading.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs;
use zerostone::xdf;

/// A single stream from an XDF recording.
///
/// Contains the stream's metadata, data (as a numpy array or list of string lists),
/// timestamps, and clock offset measurements.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// rec = zbci.read_xdf("recording.xdf")
/// stream = rec.streams[0]
/// print(stream.name)           # stream name
/// print(stream.data.shape)     # (n_samples, n_channels)
/// print(stream.timestamps)     # array of timestamps
/// ```
#[pyclass]
pub struct XdfStream {
    name: String,
    stream_type: String,
    channel_count: u32,
    sample_rate: f64,
    channel_format: String,
    stream_id: u32,
    /// Numeric data: n_samples x n_channels (None for string streams)
    numeric_data: Option<Vec<f64>>,
    n_samples: usize,
    /// String data: [sample_idx][channel_idx] (None for numeric streams)
    string_data: Option<Vec<Vec<String>>>,
    timestamps: Vec<f64>,
    clock_offsets: Vec<(f64, f64)>,
}

#[pymethods]
impl XdfStream {
    /// Stream name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Stream type (e.g. "EEG", "Markers").
    #[getter]
    fn stream_type(&self) -> &str {
        &self.stream_type
    }

    /// Number of channels.
    #[getter]
    fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// Nominal sample rate in Hz (0 for irregular streams).
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Channel format string (e.g. "float32", "string").
    #[getter]
    fn channel_format(&self) -> &str {
        &self.channel_format
    }

    /// XDF stream ID.
    #[getter]
    fn stream_id(&self) -> u32 {
        self.stream_id
    }

    /// Stream data as a 2D numpy array (n_samples x n_channels) for numeric streams,
    /// or a list of lists of strings for string streams.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        if let Some(ref data) = self.numeric_data {
            let ch = self.channel_count as usize;
            let n = self.n_samples;
            let arr = Array2::from_shape_fn((n, ch), |(r, c)| data[r * ch + c]);
            Ok(PyArray2::from_owned_array(py, arr).into_any().unbind())
        } else if let Some(ref data) = self.string_data {
            let outer = PyList::empty(py);
            for sample in data {
                let inner = PyList::empty(py);
                for s in sample {
                    inner.append(s)?;
                }
                outer.append(inner)?;
            }
            Ok(outer.into_any().unbind())
        } else {
            // Empty stream
            let arr = Array2::<f64>::zeros((0, self.channel_count as usize));
            Ok(PyArray2::from_owned_array(py, arr).into_any().unbind())
        }
    }

    /// Timestamps as a 1D numpy array of f64.
    #[getter]
    fn timestamps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.timestamps.clone())
    }

    /// Clock offset measurements as a list of (collection_time, offset_value) tuples.
    #[getter]
    fn clock_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for &(ct, ov) in &self.clock_offsets {
            list.append((ct, ov))?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "XdfStream(name='{}', type='{}', channels={}, samples={}, rate={} Hz, format='{}')",
            self.name,
            self.stream_type,
            self.channel_count,
            self.n_samples,
            self.sample_rate,
            self.channel_format
        )
    }
}

/// An XDF recording loaded from a file.
///
/// Contains all streams with their data, timestamps, and metadata.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// rec = zbci.read_xdf("recording.xdf")
/// print(rec.stream_count)
/// eeg = rec.get_stream("EEG")
/// markers = rec.get_stream(1)  # by index
/// ```
#[pyclass]
pub struct XdfRecording {
    header_xml: String,
    streams: Vec<Py<XdfStream>>,
    stream_names: Vec<String>,
}

#[pymethods]
impl XdfRecording {
    /// File header metadata as a dictionary.
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        // Extract version from header XML
        let version = xdf::find_tag_value(self.header_xml.as_bytes(), b"version")
            .and_then(|v| core::str::from_utf8(v).ok())
            .unwrap_or("unknown");
        d.set_item("version", version)?;
        Ok(d)
    }

    /// List of XdfStream objects.
    #[getter]
    fn streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for s in &self.streams {
            list.append(s.clone_ref(py))?;
        }
        Ok(list)
    }

    /// Number of streams.
    #[getter]
    fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Get a stream by name (str) or index (int).
    fn get_stream<'py>(
        &self,
        py: Python<'py>,
        name_or_index: &Bound<'py, PyAny>,
    ) -> PyResult<Py<XdfStream>> {
        if let Ok(idx) = name_or_index.extract::<usize>() {
            if idx >= self.streams.len() {
                return Err(PyValueError::new_err(format!(
                    "Stream index {} out of range (0..{})",
                    idx,
                    self.streams.len()
                )));
            }
            return Ok(self.streams[idx].clone_ref(py));
        }

        if let Ok(name) = name_or_index.extract::<String>() {
            for (i, sn) in self.stream_names.iter().enumerate() {
                if sn == &name {
                    return Ok(self.streams[i].clone_ref(py));
                }
            }
            return Err(PyValueError::new_err(format!(
                "No stream with name '{}'",
                name
            )));
        }

        Err(PyValueError::new_err(
            "Argument must be a string name or integer index",
        ))
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.stream_names.iter().map(|s| s.as_str()).collect();
        format!(
            "XdfRecording(streams={}, names={:?})",
            self.streams.len(),
            names
        )
    }
}

/// Collected info per stream during first pass.
struct StreamCollect {
    info: StreamInfo,
    sample_count: usize,
    /// Byte ranges of Samples chunks for this stream: (content_after_stream_id_offset, content_len)
    sample_chunks: Vec<(usize, usize)>,
    clock_offsets: Vec<(f64, f64)>,
}

struct StreamInfo {
    stream_id: u32,
    name: String,
    stream_type: String,
    channel_count: u32,
    nominal_srate: f64,
    channel_format: xdf::XdfChannelFormat,
    channel_format_str: String,
}

/// Read an XDF file from disk.
///
/// Args:
///     filepath (str): Path to the XDF file.
///
/// Returns:
///     XdfRecording: Parsed recording with streams, data, and timestamps.
///
/// Example:
///     >>> rec = zbci.read_xdf("recording.xdf")
///     >>> print(rec.stream_count)
///     >>> eeg = rec.get_stream("EEG")
///     >>> print(eeg.data.shape)
#[pyfunction]
fn read_xdf(py: Python<'_>, filepath: &str) -> PyResult<XdfRecording> {
    let bytes = fs::read(filepath)
        .map_err(|e| PyValueError::new_err(format!("Cannot read file: {}", e)))?;

    xdf::validate_magic(&bytes)
        .map_err(|e| PyValueError::new_err(format!("Invalid XDF file: {:?}", e)))?;

    // First pass: collect stream headers, count samples, collect clock offsets
    let mut streams: HashMap<u32, StreamCollect> = HashMap::new();
    let mut stream_order: Vec<u32> = Vec::new();
    let mut header_xml = String::new();
    let mut pos = 4; // skip magic

    while let Some(chunk) = xdf::next_chunk(&bytes, &mut pos)
        .map_err(|e| PyValueError::new_err(format!("XDF parse error: {:?}", e)))?
    {
        match chunk.tag {
            xdf::TAG_FILE_HEADER => {
                header_xml = String::from_utf8_lossy(chunk.content).into_owned();
            }
            xdf::TAG_STREAM_HEADER => {
                let stream_id = chunk
                    .stream_id
                    .ok_or_else(|| PyValueError::new_err("StreamHeader missing stream_id"))?;
                // XML is content after the 4-byte stream_id
                let xml = &chunk.content[4..];
                let info = xdf::parse_stream_info(xml, stream_id).map_err(|e| {
                    PyValueError::new_err(format!("Invalid stream header: {:?}", e))
                })?;

                let format_str = match info.channel_format {
                    xdf::XdfChannelFormat::Float32 => "float32",
                    xdf::XdfChannelFormat::Double64 => "double64",
                    xdf::XdfChannelFormat::Int8 => "int8",
                    xdf::XdfChannelFormat::Int16 => "int16",
                    xdf::XdfChannelFormat::Int32 => "int32",
                    xdf::XdfChannelFormat::Int64 => "int64",
                    xdf::XdfChannelFormat::StringFormat => "string",
                };

                let sc = StreamCollect {
                    info: StreamInfo {
                        stream_id,
                        name: String::from_utf8_lossy(info.name).into_owned(),
                        stream_type: String::from_utf8_lossy(info.stream_type).into_owned(),
                        channel_count: info.channel_count,
                        nominal_srate: info.nominal_srate,
                        channel_format: info.channel_format,
                        channel_format_str: format_str.to_string(),
                    },
                    sample_count: 0,
                    sample_chunks: Vec::new(),
                    clock_offsets: Vec::new(),
                };
                stream_order.push(stream_id);
                streams.insert(stream_id, sc);
            }
            xdf::TAG_SAMPLES => {
                let stream_id = chunk
                    .stream_id
                    .ok_or_else(|| PyValueError::new_err("Samples chunk missing stream_id"))?;
                if let Some(sc) = streams.get_mut(&stream_id) {
                    let sample_content = &chunk.content[4..];
                    let n = xdf::count_chunk_samples(
                        sample_content,
                        sc.info.channel_count,
                        &sc.info.channel_format,
                    )
                    .map_err(|e| PyValueError::new_err(format!("Sample count error: {:?}", e)))?;
                    sc.sample_count += n;
                    // Store the byte offset into the original file for the content after stream_id
                    // chunk.content starts at some offset in bytes, and we need content[4..]
                    // We can recover the offset: chunk.content is a subslice of bytes
                    let content_ptr = chunk.content.as_ptr() as usize;
                    let bytes_ptr = bytes.as_ptr() as usize;
                    let content_offset = content_ptr - bytes_ptr + 4; // +4 to skip stream_id
                    let content_len = chunk.content.len() - 4;
                    sc.sample_chunks.push((content_offset, content_len));
                }
            }
            xdf::TAG_CLOCK_OFFSET => {
                let stream_id = chunk
                    .stream_id
                    .ok_or_else(|| PyValueError::new_err("ClockOffset missing stream_id"))?;
                if let Some(sc) = streams.get_mut(&stream_id) {
                    let co_content = &chunk.content[4..];
                    let pair = xdf::parse_clock_offset(co_content).map_err(|e| {
                        PyValueError::new_err(format!("Clock offset error: {:?}", e))
                    })?;
                    sc.clock_offsets
                        .push((pair.collection_time, pair.offset_value));
                }
            }
            _ => {} // Skip Boundary, StreamFooter
        }
    }

    // Second pass: decode samples into arrays
    let mut py_streams: Vec<Py<XdfStream>> = Vec::new();
    let mut stream_names: Vec<String> = Vec::new();

    for &sid in &stream_order {
        let sc = streams.remove(&sid).unwrap();
        let ch = sc.info.channel_count as usize;
        let n = sc.sample_count;

        if matches!(sc.info.channel_format, xdf::XdfChannelFormat::StringFormat) {
            // String stream
            let mut timestamps = vec![0.0f64; n];
            let mut string_data: Vec<Vec<String>> = Vec::with_capacity(n);
            for _ in 0..n {
                string_data.push(vec![String::new(); ch]);
            }

            let mut sample_offset = 0usize;
            for &(content_offset, content_len) in &sc.sample_chunks {
                let content = &bytes[content_offset..content_offset + content_len];
                let ts_slice = &mut timestamps[sample_offset..];
                let base = sample_offset;
                let decoded = xdf::decode_samples_string(
                    content,
                    sc.info.channel_count,
                    ts_slice,
                    |sample_idx, ch_idx, str_bytes| {
                        let s = String::from_utf8_lossy(str_bytes).into_owned();
                        string_data[base + sample_idx][ch_idx] = s;
                    },
                )
                .map_err(|e| PyValueError::new_err(format!("String decode error: {:?}", e)))?;
                sample_offset += decoded;
            }

            let xs = XdfStream {
                name: sc.info.name.clone(),
                stream_type: sc.info.stream_type.clone(),
                channel_count: sc.info.channel_count,
                sample_rate: sc.info.nominal_srate,
                channel_format: sc.info.channel_format_str,
                stream_id: sc.info.stream_id,
                numeric_data: None,
                n_samples: n,
                string_data: Some(string_data),
                timestamps,
                clock_offsets: sc.clock_offsets,
            };
            stream_names.push(sc.info.name);
            py_streams.push(Py::new(py, xs)?);
        } else {
            // Numeric stream
            let mut values = vec![0.0f64; n * ch];
            let mut timestamps = vec![0.0f64; n];

            let mut sample_offset = 0usize;
            for &(content_offset, content_len) in &sc.sample_chunks {
                let content = &bytes[content_offset..content_offset + content_len];
                let val_start = sample_offset * ch;
                let decoded = xdf::decode_samples_f64(
                    content,
                    sc.info.channel_count,
                    &sc.info.channel_format,
                    &mut values[val_start..],
                    &mut timestamps[sample_offset..],
                )
                .map_err(|e| PyValueError::new_err(format!("Sample decode error: {:?}", e)))?;
                sample_offset += decoded;
            }

            let xs = XdfStream {
                name: sc.info.name.clone(),
                stream_type: sc.info.stream_type.clone(),
                channel_count: sc.info.channel_count,
                sample_rate: sc.info.nominal_srate,
                channel_format: sc.info.channel_format_str,
                stream_id: sc.info.stream_id,
                numeric_data: Some(values),
                n_samples: n,
                string_data: None,
                timestamps,
                clock_offsets: sc.clock_offsets,
            };
            stream_names.push(sc.info.name);
            py_streams.push(Py::new(py, xs)?);
        }
    }

    Ok(XdfRecording {
        header_xml,
        streams: py_streams,
        stream_names,
    })
}

/// Register XDF functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_xdf, m)?)?;
    Ok(())
}
