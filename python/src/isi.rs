//! Python bindings for ISI analysis.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::isi;

/// Compute the coefficient of variation of inter-spike intervals.
///
/// CV = std(ISI) / mean(ISI). Returns 0 for perfectly regular firing,
/// ~1 for Poisson, >1 for bursty firing.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times.
///
/// Returns:
///     float: Coefficient of variation.
#[pyfunction]
fn isi_cv(spike_times: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let times = spike_times.as_slice()?;
    Ok(isi::isi_cv(times))
}

/// Compute the local variation (Lv) of inter-spike intervals.
///
/// Lv compares adjacent ISIs rather than global statistics, making it
/// more robust to slow rate changes. Lv = 1 for Poisson, <1 for regular,
/// >1 for bursty firing.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times.
///
/// Returns:
///     float: Local variation statistic.
#[pyfunction]
fn local_variation(spike_times: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let times = spike_times.as_slice()?;
    Ok(isi::local_variation(times))
}

/// Compute an ISI histogram from spike times.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times.
///     bin_width (float): Width of each histogram bin (same units as spike times).
///     n_bins (int): Number of histogram bins. Default: 100.
///
/// Returns:
///     tuple: (bins, overflow, mean, std, cv) where bins is a 1D uint64 array.
#[pyfunction]
#[pyo3(signature = (spike_times, bin_width, n_bins=100))]
fn isi_histogram<'py>(
    py: Python<'py>,
    spike_times: PyReadonlyArray1<f64>,
    bin_width: f64,
    n_bins: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, u64, f64, f64, f64)> {
    if bin_width <= 0.0 {
        return Err(PyValueError::new_err("bin_width must be positive"));
    }
    let times = spike_times.as_slice()?;

    // Use dynamic dispatch since we can't use const generics at runtime.
    // Compute ISIs and bin them manually.
    let mut bins = vec![0u64; n_bins];
    let mut overflow = 0u64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u64;

    if times.len() >= 2 {
        for i in 1..times.len() {
            let isi = times[i] - times[i - 1];
            if isi < 0.0 {
                continue;
            }
            sum += isi;
            sum_sq += isi * isi;
            count += 1;
            let bin = (isi / bin_width) as usize;
            if bin < n_bins {
                bins[bin] += 1;
            } else {
                overflow += 1;
            }
        }
    }

    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let variance = if count >= 2 {
        let n = count as f64;
        sum_sq / n - mean * mean
    } else {
        0.0
    };
    let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

    let bins_arr = PyArray1::from_owned_array(py, Array1::from_vec(bins));
    Ok((bins_arr, overflow, mean, std_dev, cv))
}

/// Compute the burst index from spike times.
///
/// Burst index = fraction of ISIs below the threshold.
/// Returns value in [0, 1]. Higher values indicate more bursty firing.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times.
///     threshold (float): ISI threshold for bursts (same units as spike times).
///
/// Returns:
///     float: Burst index in [0, 1].
#[pyfunction]
fn burst_index(spike_times: PyReadonlyArray1<f64>, threshold: f64) -> PyResult<f64> {
    let times = spike_times.as_slice()?;
    if times.len() < 2 {
        return Ok(0.0);
    }
    let mut count = 0u64;
    let mut burst_count = 0u64;
    for i in 1..times.len() {
        let isi = times[i] - times[i - 1];
        if isi < 0.0 {
            continue;
        }
        count += 1;
        if isi < threshold {
            burst_count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(burst_count as f64 / count as f64)
}

/// Compute an autocorrelogram from spike times.
///
/// For each spike, counts how many other spikes fall into each time lag bin.
/// Reveals refractory periods and rhythmic firing patterns.
///
/// Args:
///     spike_times (np.ndarray): 1D float64 array of sorted spike times.
///     bin_width (float): Width of each lag bin (same units as spike times).
///     max_lag (float): Maximum lag to compute.
///
/// Returns:
///     np.ndarray: 1D uint64 array of autocorrelogram counts.
#[pyfunction]
fn autocorrelogram<'py>(
    py: Python<'py>,
    spike_times: PyReadonlyArray1<f64>,
    bin_width: f64,
    max_lag: f64,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if bin_width <= 0.0 {
        return Err(PyValueError::new_err("bin_width must be positive"));
    }
    if max_lag <= 0.0 {
        return Err(PyValueError::new_err("max_lag must be positive"));
    }
    let times = spike_times.as_slice()?;
    let n_bins = (max_lag / bin_width) as usize;
    let mut output = vec![0u64; n_bins];
    isi::autocorrelogram(times, bin_width, max_lag, &mut output);
    Ok(PyArray1::from_owned_array(py, Array1::from_vec(output)))
}

/// Register ISI analysis functions.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(isi_cv, m)?)?;
    m.add_function(wrap_pyfunction!(local_variation, m)?)?;
    m.add_function(wrap_pyfunction!(isi_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(burst_index, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelogram, m)?)?;
    Ok(())
}
