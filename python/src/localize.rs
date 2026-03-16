//! Python bindings for spike localization.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::localize;

macro_rules! dispatch_com {
    ($amps:expr, $pos_flat:expr, $c:expr, $($C:expr),+) => {
        match $c {
            $($C => {
                let mut a = [0.0f64; $C];
                a.copy_from_slice($amps);
                let mut p = [[0.0f64; 2]; $C];
                for i in 0..$C {
                    p[i][0] = $pos_flat[i * 2];
                    p[i][1] = $pos_flat[i * 2 + 1];
                }
                localize::center_of_mass(&a, &p)
            }),+,
            _ => return Err(PyValueError::new_err(
                "n_channels must be 4, 8, 16, 32, or 64",
            )),
        }
    };
}

macro_rules! dispatch_com_threshold {
    ($amps:expr, $pos_flat:expr, $threshold:expr, $c:expr, $($C:expr),+) => {
        match $c {
            $($C => {
                let mut a = [0.0f64; $C];
                a.copy_from_slice($amps);
                let mut p = [[0.0f64; 2]; $C];
                for i in 0..$C {
                    p[i][0] = $pos_flat[i * 2];
                    p[i][1] = $pos_flat[i * 2 + 1];
                }
                localize::center_of_mass_threshold(&a, &p, $threshold)
            }),+,
            _ => return Err(PyValueError::new_err(
                "n_channels must be 4, 8, 16, 32, or 64",
            )),
        }
    };
}

macro_rules! dispatch_monopole {
    ($amps:expr, $pos_flat:expr, $z:expr, $n_iter:expr, $c:expr, $($C:expr),+) => {
        match $c {
            $($C => {
                let mut a = [0.0f64; $C];
                a.copy_from_slice($amps);
                let mut p = [[0.0f64; 2]; $C];
                for i in 0..$C {
                    p[i][0] = $pos_flat[i * 2];
                    p[i][1] = $pos_flat[i * 2 + 1];
                }
                localize::monopole_localize(&a, &p, $z, $n_iter)
            }),+,
            _ => return Err(PyValueError::new_err(
                "n_channels must be 4, 8, 16, 32, or 64",
            )),
        }
    };
}

/// Compute the center-of-mass position of a spike from per-channel amplitudes.
///
/// Args:
///     amplitudes (np.ndarray): 1D float64 array of peak amplitudes (length C).
///     positions (np.ndarray): 2D float64 array of shape (C, 2) with channel positions.
///
/// Returns:
///     np.ndarray: 1D float64 array [x, y] of estimated spike position.
#[pyfunction]
fn center_of_mass<'py>(
    py: Python<'py>,
    amplitudes: PyReadonlyArray1<f64>,
    positions: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let amps = amplitudes.as_slice()?;
    let pos = positions.as_slice()?;
    let c = amps.len();

    if positions.shape()[0] != c || positions.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape ({}, 2), got ({}, {})",
            c,
            positions.shape()[0],
            positions.shape()[1]
        )));
    }

    let result = dispatch_com!(amps, pos, c, 4, 8, 16, 32, 64);
    Ok(PyArray1::from_owned_array(
        py,
        Array1::from_vec(result.to_vec()),
    ))
}

/// Center-of-mass localization using only channels above an amplitude threshold.
///
/// Args:
///     amplitudes (np.ndarray): 1D float64 peak amplitudes (length C).
///     positions (np.ndarray): 2D float64 positions of shape (C, 2).
///     threshold (float): Minimum absolute amplitude to include a channel.
///
/// Returns:
///     np.ndarray or None: [x, y] position, or None if no channel meets threshold.
#[pyfunction]
fn center_of_mass_threshold<'py>(
    py: Python<'py>,
    amplitudes: PyReadonlyArray1<f64>,
    positions: PyReadonlyArray2<f64>,
    threshold: f64,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let amps = amplitudes.as_slice()?;
    let pos = positions.as_slice()?;
    let c = amps.len();

    if positions.shape()[0] != c || positions.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape ({}, 2), got ({}, {})",
            c,
            positions.shape()[0],
            positions.shape()[1]
        )));
    }

    let result = dispatch_com_threshold!(amps, pos, threshold, c, 4, 8, 16, 32, 64);
    Ok(result.map(|loc| PyArray1::from_owned_array(py, Array1::from_vec(loc.to_vec()))))
}

/// Fit a monopole source model to estimate spike position.
///
/// Args:
///     amplitudes (np.ndarray): 1D float64 absolute peak amplitudes (must be >= 0).
///     positions (np.ndarray): 2D float64 positions of shape (C, 2).
///     z_offset (float): Perpendicular distance from probe plane (um).
///     n_iter (int): Number of refinement iterations. Default: 10.
///
/// Returns:
///     np.ndarray or None: [x, y] position, or None if all amplitudes are zero.
#[pyfunction]
#[pyo3(signature = (amplitudes, positions, z_offset, n_iter=10))]
fn monopole_localize<'py>(
    py: Python<'py>,
    amplitudes: PyReadonlyArray1<f64>,
    positions: PyReadonlyArray2<f64>,
    z_offset: f64,
    n_iter: usize,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let amps = amplitudes.as_slice()?;
    let pos = positions.as_slice()?;
    let c = amps.len();

    if positions.shape()[0] != c || positions.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "positions must have shape ({}, 2), got ({}, {})",
            c,
            positions.shape()[0],
            positions.shape()[1]
        )));
    }

    let result = dispatch_monopole!(amps, pos, z_offset, n_iter, c, 4, 8, 16, 32, 64);
    Ok(result.map(|loc| PyArray1::from_owned_array(py, Array1::from_vec(loc.to_vec()))))
}

/// Register localization functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(center_of_mass, m)?)?;
    m.add_function(wrap_pyfunction!(center_of_mass_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(monopole_localize, m)?)?;
    Ok(())
}
