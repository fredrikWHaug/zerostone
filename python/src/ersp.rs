//! Python bindings for ERSP (Event-Related Spectral Perturbation).
//!
//! Provides baseline normalization and high-level ERSP computation
//! from epoched signals using the existing STFT machinery.

use numpy::ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::ersp as zs_ersp;
use zerostone::{Complex, Fft as ZsFft};

use crate::spectral::parse_window_type;

/// Compute Event-Related Spectral Perturbation (ERSP).
///
/// Computes time-frequency decomposition via STFT for each epoch, averages
/// across epochs (unless single_trial=True), and applies baseline normalization.
///
/// Args:
///     epochs (np.ndarray): 2D array of shape (n_epochs, n_samples), float32.
///     sample_rate (float): Sampling frequency in Hz.
///     baseline_window (tuple[float, float]): (start_sec, end_sec) relative to epoch onset.
///     fft_size (int): FFT window size. Must be 64, 128, 256, 512, 1024, or 2048. Default: 256.
///     hop_size (int or None): Samples between frames. Default: fft_size // 4.
///     window (str): Window type. Default: 'hann'.
///     mode (str): Normalization mode - 'db', 'zscore', 'percentage', 'logratio'. Default: 'db'.
///     single_trial (bool): If True, return per-epoch ERSP. Default: False.
///
/// Returns:
///     np.ndarray: (n_freqs, n_frames) float64 if averaged, or
///                 (n_epochs, n_freqs, n_frames) float64 if single_trial=True.
///     n_freqs = fft_size / 2 + 1 (one-sided spectrum).
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> rng = np.random.default_rng(42)
///     >>> epochs = rng.standard_normal((10, 1024)).astype(np.float32)
///     >>> ersp = zbci.compute_ersp(epochs, 256.0, (-0.5, 0.0))
///     >>> assert ersp.shape[0] == 129  # n_freqs = 256/2 + 1
#[pyfunction]
#[pyo3(signature = (epochs, sample_rate, baseline_window, fft_size=256, hop_size=None, window="hann", mode="db", single_trial=false))]
fn compute_ersp<'py>(
    py: Python<'py>,
    epochs: PyReadonlyArray2<f32>,
    sample_rate: f64,
    baseline_window: (f64, f64),
    fft_size: usize,
    hop_size: Option<usize>,
    window: &str,
    mode: &str,
    single_trial: bool,
) -> PyResult<PyObject> {
    let shape = epochs.shape();
    let n_epochs = shape[0];
    let n_samples = shape[1];

    if n_epochs == 0 {
        return Err(PyValueError::new_err("epochs must not be empty"));
    }
    if sample_rate <= 0.0 {
        return Err(PyValueError::new_err("sample_rate must be positive"));
    }

    let hop = hop_size.unwrap_or(fft_size / 4);
    if hop == 0 {
        return Err(PyValueError::new_err("hop_size must be positive"));
    }

    let _window_type = parse_window_type(window)?;
    let bl_mode = parse_baseline_mode(mode)?;

    // Validate FFT size
    if !matches!(fft_size, 64 | 128 | 256 | 512 | 1024 | 2048) {
        return Err(PyValueError::new_err(
            "fft_size must be 64, 128, 256, 512, 1024, or 2048",
        ));
    }

    if n_samples < fft_size {
        return Err(PyValueError::new_err(format!(
            "Epoch length {} is shorter than fft_size {}",
            n_samples, fft_size
        )));
    }

    let n_frames = (n_samples - fft_size) / hop + 1;
    let n_freqs = fft_size / 2 + 1;

    // Convert baseline window (seconds) to frame indices
    let (bl_start_sec, bl_end_sec) = baseline_window;
    if bl_start_sec >= bl_end_sec {
        return Err(PyValueError::new_err(
            "baseline_window start must be less than end",
        ));
    }

    // Frame time = (frame_idx * hop) / sample_rate seconds from epoch onset
    // Find frame indices corresponding to baseline window
    let bl_start_frame = sec_to_frame(bl_start_sec, sample_rate, hop, fft_size, n_frames)?;
    let bl_end_frame = sec_to_frame(bl_end_sec, sample_rate, hop, fft_size, n_frames)?;

    if bl_start_frame >= bl_end_frame {
        return Err(PyValueError::new_err(
            "baseline_window maps to zero-length frame range",
        ));
    }

    let epochs_array = epochs.as_array();

    // Compute STFT power for each epoch
    // Each epoch produces (n_frames, n_freqs) power matrix in f64
    let mut all_power = vec![0.0f64; n_epochs * n_frames * n_freqs];

    macro_rules! compute_stft_power {
        ($N:expr) => {{
            let fft = ZsFft::<$N>::new();
            for e in 0..n_epochs {
                let epoch_row = epochs_array.row(e);
                let epoch_slice = epoch_row
                    .as_slice()
                    .ok_or_else(|| PyValueError::new_err("epochs array must be contiguous"))?;
                let epoch_offset = e * n_frames * n_freqs;

                for frame_idx in 0..n_frames {
                    let start = frame_idx * hop;
                    let mut temp = [Complex::new(0.0, 0.0); $N];

                    for (i, &sample) in epoch_slice[start..start + $N].iter().enumerate() {
                        let wc = zerostone::window_coefficient(_window_type, i, $N);
                        temp[i] = Complex::from_real(sample * wc);
                    }

                    fft.forward(&mut temp);

                    // One-sided power spectrum as f64
                    for f in 0..n_freqs {
                        let mag_sq = (temp[f].re as f64) * (temp[f].re as f64)
                            + (temp[f].im as f64) * (temp[f].im as f64);
                        all_power[epoch_offset + frame_idx * n_freqs + f] = mag_sq;
                    }
                }
            }
        }};
    }

    match fft_size {
        64 => compute_stft_power!(64),
        128 => compute_stft_power!(128),
        256 => compute_stft_power!(256),
        512 => compute_stft_power!(512),
        1024 => compute_stft_power!(1024),
        2048 => compute_stft_power!(2048),
        _ => unreachable!(),
    }

    if single_trial {
        // Normalize each epoch independently, return (n_epochs, n_freqs, n_frames)
        for e in 0..n_epochs {
            let offset = e * n_frames * n_freqs;
            let epoch_power = &mut all_power[offset..offset + n_frames * n_freqs];
            zs_ersp::baseline_normalize(
                epoch_power,
                n_frames,
                n_freqs,
                bl_start_frame,
                bl_end_frame,
                bl_mode,
            )
            .map_err(|e| ersp_error_to_py(e))?;
        }

        // Reshape to (n_epochs, n_freqs, n_frames) -- transpose each epoch from (frames, freqs)
        let mut result = vec![0.0f64; n_epochs * n_freqs * n_frames];
        for e in 0..n_epochs {
            let src_offset = e * n_frames * n_freqs;
            let dst_offset = e * n_freqs * n_frames;
            for t in 0..n_frames {
                for f in 0..n_freqs {
                    result[dst_offset + f * n_frames + t] = all_power[src_offset + t * n_freqs + f];
                }
            }
        }

        let arr = Array3::from_shape_vec((n_epochs, n_freqs, n_frames), result)
            .map_err(|e| PyValueError::new_err(format!("reshape error: {}", e)))?;
        Ok(PyArray3::from_owned_array(py, arr).into_any().unbind())
    } else {
        // Average across epochs
        let mut avg_power = vec![0.0f64; n_frames * n_freqs];
        zs_ersp::epoch_average(&all_power, n_epochs, n_frames, n_freqs, &mut avg_power)
            .map_err(|e| ersp_error_to_py(e))?;

        // Baseline normalize
        zs_ersp::baseline_normalize(
            &mut avg_power,
            n_frames,
            n_freqs,
            bl_start_frame,
            bl_end_frame,
            bl_mode,
        )
        .map_err(|e| ersp_error_to_py(e))?;

        // Transpose from (n_frames, n_freqs) to (n_freqs, n_frames) for output
        let mut result = vec![0.0f64; n_freqs * n_frames];
        for t in 0..n_frames {
            for f in 0..n_freqs {
                result[f * n_frames + t] = avg_power[t * n_freqs + f];
            }
        }

        let arr = Array2::from_shape_vec((n_freqs, n_frames), result)
            .map_err(|e| PyValueError::new_err(format!("reshape error: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, arr).into_any().unbind())
    }
}

/// Apply baseline normalization to a time-frequency power matrix.
///
/// Args:
///     power (np.ndarray): 2D array of shape (n_frames, n_freqs), float64.
///     baseline_start_frame (int): Start frame of baseline (inclusive).
///     baseline_end_frame (int): End frame of baseline (exclusive).
///     mode (str): Normalization mode - 'db', 'zscore', 'percentage', 'logratio'. Default: 'db'.
///
/// Returns:
///     np.ndarray: Normalized copy as (n_frames, n_freqs) float64.
///
/// Example:
///     >>> import zpybci as zbci
///     >>> import numpy as np
///     >>> power = np.ones((10, 5), dtype=np.float64)
///     >>> power[5:] = 2.0
///     >>> norm = zbci.baseline_normalize(power, 0, 5, mode='db')
///     >>> assert abs(norm[0, 0]) < 1e-10
#[pyfunction]
#[pyo3(signature = (power, baseline_start_frame, baseline_end_frame, mode="db"))]
fn baseline_normalize<'py>(
    py: Python<'py>,
    power: PyReadonlyArray2<f64>,
    baseline_start_frame: usize,
    baseline_end_frame: usize,
    mode: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = power.shape();
    let n_frames = shape[0];
    let n_freqs = shape[1];

    if n_frames == 0 || n_freqs == 0 {
        return Err(PyValueError::new_err("power must not be empty"));
    }

    let bl_mode = parse_baseline_mode(mode)?;

    // Copy to mutable vec
    let arr = power.as_array();
    let mut data: Vec<f64> = Vec::with_capacity(n_frames * n_freqs);
    for row in arr.rows() {
        for &val in row.iter() {
            data.push(val);
        }
    }

    zs_ersp::baseline_normalize(
        &mut data,
        n_frames,
        n_freqs,
        baseline_start_frame,
        baseline_end_frame,
        bl_mode,
    )
    .map_err(|e| ersp_error_to_py(e))?;

    let result = Array2::from_shape_vec((n_frames, n_freqs), data)
        .map_err(|e| PyValueError::new_err(format!("reshape error: {}", e)))?;
    Ok(PyArray2::from_owned_array(py, result))
}

fn parse_baseline_mode(mode: &str) -> PyResult<zs_ersp::BaselineMode> {
    match mode.to_lowercase().as_str() {
        "db" => Ok(zs_ersp::BaselineMode::Db),
        "zscore" | "z-score" | "z_score" => Ok(zs_ersp::BaselineMode::Zscore),
        "percentage" | "percent" => Ok(zs_ersp::BaselineMode::Percentage),
        "logratio" | "log-ratio" | "log_ratio" => Ok(zs_ersp::BaselineMode::LogRatio),
        _ => Err(PyValueError::new_err(
            "mode must be 'db', 'zscore', 'percentage', or 'logratio'",
        )),
    }
}

fn sec_to_frame(
    sec: f64,
    sample_rate: f64,
    hop: usize,
    _fft_size: usize,
    n_frames: usize,
) -> PyResult<usize> {
    // Frame center time = (frame_idx * hop + fft_size / 2) / sample_rate
    // But for ERSP, convention is frame start time = frame_idx * hop / sample_rate
    let sample_idx = sec * sample_rate;
    let frame_f = sample_idx / hop as f64;
    let frame = if frame_f < 0.0 { 0 } else { frame_f as usize };
    Ok(frame.min(n_frames))
}

fn ersp_error_to_py(e: zs_ersp::ErspError) -> PyErr {
    let msg = match e {
        zs_ersp::ErspError::InvalidBaselineRange => "baseline start must be less than end",
        zs_ersp::ErspError::BaselineOutOfBounds => "baseline end exceeds number of frames",
        zs_ersp::ErspError::DimensionMismatch => "power dimensions do not match",
        zs_ersp::ErspError::EmptyPower => "power is empty",
        zs_ersp::ErspError::ZeroBaselineMean => "baseline mean is zero (cannot normalize)",
        zs_ersp::ErspError::ZeroBaselineStd => "baseline std is zero (cannot compute z-score)",
        zs_ersp::ErspError::OutputDimensionMismatch => "output dimensions do not match",
        zs_ersp::ErspError::NoEpochs => "no epochs provided",
        zs_ersp::ErspError::EpochDimensionMismatch => "epoch dimensions do not match",
    };
    PyValueError::new_err(msg)
}

/// Register ERSP functions with the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ersp, m)?)?;
    m.add_function(wrap_pyfunction!(baseline_normalize, m)?)?;
    Ok(())
}
