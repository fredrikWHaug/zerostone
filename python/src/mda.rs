//! Python bindings for MDA file reading.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zerostone::mda;

/// Read an MDA (MountainSort Data Array) file from disk.
///
/// Parses the binary MDA format used by SpikeForest and MountainSort.
/// All element types are converted to float64.
///
/// Args:
///     path (str): Path to the .mda file.
///
/// Returns:
///     dict: A dictionary with keys:
///         - "data" (numpy.ndarray): 1D float64 array of all elements (column-major order)
///         - "shape" (tuple): Dimension sizes, e.g. (32, 1000) for 32 channels x 1000 samples
///         - "dtype" (str): Original data type string, e.g. "float32", "int16", "float64"
///
/// Example:
///     >>> result = zbci.read_mda("raw.mda")
///     >>> data = result["data"].reshape(result["shape"], order="F")
///     >>> print(data.shape)  # (32, 1000)
#[pyfunction]
fn read_mda(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let bytes = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to read {}: {}", path, e)))?;

    let header =
        mda::parse_mda_header(&bytes).ok_or_else(|| PyValueError::new_err("Invalid MDA header"))?;

    let n = mda::mda_num_elements(&header);
    let mut data = vec![0.0f64; n];
    mda::read_mda_f64(&bytes, &mut data)
        .ok_or_else(|| PyValueError::new_err("Failed to read MDA data"))?;

    let dict = PyDict::new(py);
    dict.set_item("data", PyArray1::from_vec(py, data))?;

    let shape: Vec<usize> = header.dims[..header.ndims].to_vec();
    dict.set_item("shape", shape)?;

    let dtype_str = match header.data_type {
        mda::MdaDataType::Uint8 => "uint8",
        mda::MdaDataType::Float32 => "float32",
        mda::MdaDataType::Int16 => "int16",
        mda::MdaDataType::Int32 => "int32",
        mda::MdaDataType::Uint16 => "uint16",
        mda::MdaDataType::Float64 => "float64",
        mda::MdaDataType::Uint32 => "uint32",
    };
    dict.set_item("dtype", dtype_str)?;

    Ok(dict.into())
}

/// Register MDA functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_mda, m)?)?;
    Ok(())
}
