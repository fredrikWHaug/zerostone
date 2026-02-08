use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Convert a Python numpy array to a Rust Vec<f32>
#[allow(dead_code)]
pub fn numpy_to_vec_f32(_py: Python, array: PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
    Ok(array.as_slice()?.to_vec())
}

/// Convert a Rust Vec<f32> to a Python numpy array
#[allow(dead_code)]
pub fn vec_to_numpy_f32(py: Python, vec: Vec<f32>) -> Py<PyArray1<f32>> {
    PyArray1::from_vec(py, vec).unbind()
}
