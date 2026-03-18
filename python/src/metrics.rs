//! Python bindings for spike sorting comparison metrics.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use zerostone::metrics;

/// Compare two spike trains within a sample-level tolerance window.
///
/// Both arrays must be sorted in ascending order. Each ground-truth spike
/// is matched to at most one sorted spike and vice versa (greedy left-to-right).
///
/// Args:
///     gt_times (np.ndarray): 1D int64 array of ground-truth spike sample indices (sorted).
///     sorted_times (np.ndarray): 1D int64 array of sorted spike sample indices (sorted).
///     tolerance (int): Maximum sample distance for a match (default 12, i.e. 0.4 ms at 30 kHz).
///
/// Returns:
///     dict: Keys "true_positives", "false_positives", "false_negatives",
///           "accuracy", "precision", "recall".
///
/// Example:
///     >>> m = zbci.compare_spike_trains(np.array([100, 200, 300]), np.array([101, 199, 301]), 5)
///     >>> m["true_positives"]
///     3
#[pyfunction]
#[pyo3(signature = (gt_times, sorted_times, tolerance=12))]
fn compare_spike_trains<'py>(
    py: Python<'py>,
    gt_times: PyReadonlyArray1<i64>,
    sorted_times: PyReadonlyArray1<i64>,
    tolerance: usize,
) -> PyResult<PyObject> {
    let gt_slice = gt_times.as_slice()?;
    let sorted_slice = sorted_times.as_slice()?;

    // Convert i64 to usize
    let gt_usize: Vec<usize> = gt_slice
        .iter()
        .map(|&t| {
            if t < 0 {
                Err(PyValueError::new_err("gt_times contains negative values"))
            } else {
                Ok(t as usize)
            }
        })
        .collect::<PyResult<_>>()?;

    let sorted_usize: Vec<usize> = sorted_slice
        .iter()
        .map(|&t| {
            if t < 0 {
                Err(PyValueError::new_err(
                    "sorted_times contains negative values",
                ))
            } else {
                Ok(t as usize)
            }
        })
        .collect::<PyResult<_>>()?;

    let m = metrics::compare_spike_trains(&gt_usize, &sorted_usize, tolerance);

    let dict = PyDict::new(py);
    dict.set_item("true_positives", m.true_positives)?;
    dict.set_item("false_positives", m.false_positives)?;
    dict.set_item("false_negatives", m.false_negatives)?;
    dict.set_item("accuracy", m.accuracy)?;
    dict.set_item("precision", m.precision)?;
    dict.set_item("recall", m.recall)?;

    Ok(dict.into())
}

/// Compare all ground-truth units against all sorted units via greedy matching.
///
/// For each ground-truth unit, finds the sorted unit that maximises accuracy.
/// Each sorted unit can be assigned to at most one GT unit.
///
/// Args:
///     gt_trains (list[np.ndarray]): List of 1D int64 arrays, one per GT unit (sorted).
///     sorted_trains (list[np.ndarray]): List of 1D int64 arrays, one per sorted unit (sorted).
///     tolerance (int): Maximum sample distance for a match (default 12).
///
/// Returns:
///     list[dict]: One dict per GT unit with keys "true_positives", "false_positives",
///                 "false_negatives", "accuracy", "precision", "recall".
///
/// Example:
///     >>> gt = [np.array([100, 200, 300]), np.array([150, 250, 350])]
///     >>> s = [np.array([101, 201, 301]), np.array([149, 251, 349])]
///     >>> results = zbci.compare_sorting(gt, s, 5)
///     >>> results[0]["true_positives"]
///     3
#[pyfunction]
#[pyo3(signature = (gt_trains, sorted_trains, tolerance=12))]
fn compare_sorting<'py>(
    py: Python<'py>,
    gt_trains: Vec<PyReadonlyArray1<i64>>,
    sorted_trains: Vec<PyReadonlyArray1<i64>>,
    tolerance: usize,
) -> PyResult<Bound<'py, PyList>> {
    // Convert each train from i64 to Vec<usize>
    let gt_vecs: Vec<Vec<usize>> = gt_trains
        .iter()
        .map(|arr| {
            arr.as_slice()?
                .iter()
                .map(|&t| {
                    if t < 0 {
                        Err(PyValueError::new_err("gt_trains contains negative values"))
                    } else {
                        Ok(t as usize)
                    }
                })
                .collect::<PyResult<Vec<usize>>>()
        })
        .collect::<PyResult<_>>()?;

    let sorted_vecs: Vec<Vec<usize>> = sorted_trains
        .iter()
        .map(|arr| {
            arr.as_slice()?
                .iter()
                .map(|&t| {
                    if t < 0 {
                        Err(PyValueError::new_err(
                            "sorted_trains contains negative values",
                        ))
                    } else {
                        Ok(t as usize)
                    }
                })
                .collect::<PyResult<Vec<usize>>>()
        })
        .collect::<PyResult<_>>()?;

    // Build slice-of-slices for the Rust API
    let gt_refs: Vec<&[usize]> = gt_vecs.iter().map(|v| v.as_slice()).collect();
    let sorted_refs: Vec<&[usize]> = sorted_vecs.iter().map(|v| v.as_slice()).collect();

    let n_gt = gt_refs.len();
    let mut output = vec![metrics::UnitMatch::empty(); n_gt];
    metrics::compare_sorting(&gt_refs, &sorted_refs, tolerance, &mut output);

    // Convert to Python list of dicts
    let result = PyList::empty(py);
    for um in &output {
        let dict = PyDict::new(py);
        dict.set_item("true_positives", um.true_positives)?;
        dict.set_item("false_positives", um.false_positives)?;
        dict.set_item("false_negatives", um.false_negatives)?;
        dict.set_item("accuracy", um.accuracy)?;
        dict.set_item("precision", um.precision)?;
        dict.set_item("recall", um.recall)?;
        result.append(dict)?;
    }

    Ok(result)
}

/// Register metrics functions on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_spike_trains, m)?)?;
    m.add_function(wrap_pyfunction!(compare_sorting, m)?)?;
    Ok(())
}
