#![allow(
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::manual_saturating_arithmetic,
    clippy::implicit_saturating_sub
)]
//! Heap-allocated sorting pipeline for >128 channels.
//!
//! Provides `sort_heap()` for channel counts not supported by the const-generic
//! `sort_multichannel` (e.g. 256, 384 for Neuropixels). Uses `Vec<Float>` for all
//! channel-dimension arrays while keeping W, K, N as const generics.
//!
//! Data layout: row-major flat `&mut [Float]` of shape `(n_samples, n_channels)`.
//!
//! # Example
//!
//! ```
//! use zerostone::float::{self, Float};
//! use zerostone::sorter::SortConfig;
//! use zerostone::sorter_heap::sort_heap;
//!
//! let n_ch = 256;
//! let n_samples = 3000;
//! let mut data = vec![0.0 as Float; n_samples * n_ch];
//! // Fill with small noise
//! for (i, d) in data.iter_mut().enumerate() {
//!     *d = ((i * 17 + 3) % 100) as Float * 0.001 - 0.05;
//! }
//! let positions: Vec<[Float; 2]> = (0..n_ch).map(|i| [0.0, i as Float * 25.0]).collect();
//! let config = SortConfig::default();
//! let result = sort_heap(&config, &positions, &mut data, n_samples, n_ch);
//! assert!(result.is_ok());
//! assert_eq!(result.unwrap().n_spikes, 0);
//! ```

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::float::{self, Float};
use crate::linalg::LinalgError;
use crate::online_kmeans::OnlineKMeans;
use crate::sorter::{amplitude_bimodality_split, merge_clusters, split_clusters, SortConfig};
use crate::sorter_dyn::{DynClusterInfo, DynSortResult};
use crate::spike_sort::{MultiChannelEvent, SortError, WaveformPca};

// Fixed pipeline constants (same as const-generic path)
const W: usize = 48; // waveform window
const K: usize = 4; // PCA features
const WM: usize = W * W; // waveform matrix size
const N: usize = 32; // max clusters

// ---------------------------------------------------------------------------
// HeapMatrix: dynamic-size square matrix
// ---------------------------------------------------------------------------

/// Heap-allocated square matrix with row-major storage.
struct HeapMatrix {
    data: Vec<Float>,
    dim: usize,
}

impl HeapMatrix {
    fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim * dim],
            dim,
        }
    }

    fn identity(dim: usize) -> Self {
        let mut m = Self::zeros(dim);
        for i in 0..dim {
            m.data[i * dim + i] = 1.0;
        }
        m
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> Float {
        self.data[i * self.dim + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: Float) {
        self.data[i * self.dim + j] = v;
    }

    /// Jacobi eigendecomposition for symmetric matrices.
    /// Returns (eigenvalues, eigenvectors) sorted descending.
    fn eigen_symmetric(
        &self,
        max_iters: usize,
        tol: Float,
    ) -> Result<(Vec<Float>, HeapMatrix), LinalgError> {
        let c = self.dim;
        let mut a = Self {
            data: self.data.clone(),
            dim: c,
        };
        let mut v = Self::identity(c);

        let mut matrix_norm = 0.0;
        for val in &a.data {
            matrix_norm += val * val;
        }
        matrix_norm = float::sqrt(matrix_norm);
        if matrix_norm < 1e-300 {
            matrix_norm = 1.0;
        }
        let abs_tol = tol * matrix_norm;

        for sweep in 0..max_iters {
            let mut off_diag_norm_sq = 0.0;
            for i in 0..c {
                for j in (i + 1)..c {
                    let val = a.get(i, j);
                    off_diag_norm_sq += val * val;
                }
            }
            let off_diag_norm = float::sqrt(2.0 * off_diag_norm_sq);

            if off_diag_norm < abs_tol {
                let mut eigenvalues: Vec<Float> = (0..c).map(|i| a.get(i, i)).collect();
                sort_eigen_heap(&mut eigenvalues, &mut v);
                return Ok((eigenvalues, v));
            }

            let threshold = if sweep < 3 {
                0.2 * off_diag_norm_sq / ((c * c) as Float)
            } else {
                0.0
            };

            for i in 0..c {
                for j in (i + 1)..c {
                    let a_ij = a.get(i, j);
                    let abs_a_ij = float::abs(a_ij);

                    if sweep < 3 && abs_a_ij * abs_a_ij < threshold {
                        continue;
                    }
                    if sweep >= 3 {
                        let a_ii = float::abs(a.get(i, i));
                        let a_jj = float::abs(a.get(j, j));
                        if abs_a_ij < 1e-15 * (a_ii + a_jj) {
                            continue;
                        }
                    }
                    if abs_a_ij < 1e-300 {
                        continue;
                    }

                    let (cos_t, sin_t) = jacobi_rotation(&a, i, j);
                    apply_rotation(&mut a, i, j, cos_t, sin_t);
                    apply_rotation_vectors(&mut v, i, j, cos_t, sin_t);
                }
            }
        }

        Err(LinalgError::ConvergenceFailed)
    }
}

fn jacobi_rotation(a: &HeapMatrix, i: usize, j: usize) -> (Float, Float) {
    let a_ii = a.get(i, i);
    let a_jj = a.get(j, j);
    let a_ij = a.get(i, j);
    if a_ii == a_jj {
        let c = float::cos(float::PI / 4.0);
        let s = float::sin(float::PI / 4.0);
        return (c, s);
    }
    let tau = (a_jj - a_ii) / (2.0 * a_ij);
    let t = if tau >= 0.0 {
        1.0 / (tau + float::sqrt(1.0 + tau * tau))
    } else {
        -1.0 / (-tau + float::sqrt(1.0 + tau * tau))
    };
    let cos_t = 1.0 / float::sqrt(1.0 + t * t);
    let sin_t = t * cos_t;
    (cos_t, sin_t)
}

fn apply_rotation(a: &mut HeapMatrix, i: usize, j: usize, c: Float, s: Float) {
    let dim = a.dim;
    let a_ii = a.get(i, i);
    let a_jj = a.get(j, j);
    let a_ij = a.get(i, j);
    a.set(i, i, c * c * a_ii - 2.0 * c * s * a_ij + s * s * a_jj);
    a.set(j, j, s * s * a_ii + 2.0 * c * s * a_ij + c * c * a_jj);
    a.set(i, j, 0.0);
    a.set(j, i, 0.0);
    for k in 0..dim {
        if k != i && k != j {
            let a_ki = a.get(k, i);
            let a_kj = a.get(k, j);
            let new_ki = c * a_ki - s * a_kj;
            let new_kj = s * a_ki + c * a_kj;
            a.set(k, i, new_ki);
            a.set(i, k, new_ki);
            a.set(k, j, new_kj);
            a.set(j, k, new_kj);
        }
    }
}

fn apply_rotation_vectors(v: &mut HeapMatrix, i: usize, j: usize, c: Float, s: Float) {
    let dim = v.dim;
    for k in 0..dim {
        let v_ki = v.get(k, i);
        let v_kj = v.get(k, j);
        v.set(k, i, c * v_ki - s * v_kj);
        v.set(k, j, s * v_ki + c * v_kj);
    }
}

fn sort_eigen_heap(eigenvalues: &mut [Float], eigenvectors: &mut HeapMatrix) {
    let c = eigenvalues.len();
    for i in 0..c {
        let mut max_idx = i;
        for j in (i + 1)..c {
            if eigenvalues[j] > eigenvalues[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            eigenvalues.swap(i, max_idx);
            for k in 0..c {
                let tmp = eigenvectors.get(k, i);
                eigenvectors.set(k, i, eigenvectors.get(k, max_idx));
                eigenvectors.set(k, max_idx, tmp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Heap pipeline helpers
// ---------------------------------------------------------------------------

/// Estimate per-channel noise via MAD on flat row-major data.
fn estimate_noise_heap(data: &[Float], n_samples: usize, n_ch: usize) -> Vec<Float> {
    let mut noise = vec![1.0 as Float; n_ch];
    let cal_n = n_samples.min(2000);
    let mut buf = vec![0.0 as Float; cal_n];
    for ch in 0..n_ch {
        for t in 0..cal_n {
            buf[t] = data[t * n_ch + ch];
        }
        buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = if cal_n % 2 == 1 {
            buf[cal_n / 2]
        } else {
            (buf[cal_n / 2 - 1] + buf[cal_n / 2]) * 0.5
        };
        for t in 0..cal_n {
            buf[t] = float::abs(buf[t] - median);
        }
        buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let mad = if cal_n % 2 == 1 {
            buf[cal_n / 2]
        } else {
            (buf[cal_n / 2 - 1] + buf[cal_n / 2]) * 0.5
        };
        let sigma = mad / 0.6745;
        if sigma > 0.0 {
            noise[ch] = sigma;
        }
    }
    noise
}

/// Compute sample covariance matrix (flat C*C row-major).
fn compute_covariance_heap(data: &[Float], n_samples: usize, n_ch: usize) -> Vec<Float> {
    let mut cov = vec![0.0 as Float; n_ch * n_ch];
    let n = n_samples as Float;
    // Compute means
    let mut means = vec![0.0 as Float; n_ch];
    for t in 0..n_samples {
        let row = t * n_ch;
        for ch in 0..n_ch {
            means[ch] += data[row + ch];
        }
    }
    for ch in 0..n_ch {
        means[ch] /= n;
    }
    // Covariance accumulation
    for t in 0..n_samples {
        let row = t * n_ch;
        for i in 0..n_ch {
            let di = data[row + i] - means[i];
            for j in i..n_ch {
                let dj = data[row + j] - means[j];
                cov[i * n_ch + j] += di * dj;
            }
        }
    }
    // Normalize and symmetrize
    let scale = if n_samples > 1 { 1.0 / (n - 1.0) } else { 1.0 };
    for i in 0..n_ch {
        for j in i..n_ch {
            let v = cov[i * n_ch + j] * scale;
            cov[i * n_ch + j] = v;
            cov[j * n_ch + i] = v;
        }
    }
    cov
}

/// Build ZCA whitening matrix from covariance. Returns flat C*C matrix.
fn build_whitening_heap(
    cov: &[Float],
    n_ch: usize,
    epsilon: Float,
) -> Result<Vec<Float>, LinalgError> {
    let mut mat = HeapMatrix::zeros(n_ch);
    mat.data.copy_from_slice(&cov[..n_ch * n_ch]);

    let eigen_tol = if cfg!(feature = "f32") { 1e-6 } else { 1e-12 };
    let (eigenvalues, eigenvectors) = mat.eigen_symmetric(50, eigen_tol)?;

    // ZCA: W = E * diag(1/sqrt(lambda + eps)) * E^T
    let mut w = vec![0.0 as Float; n_ch * n_ch];
    for i in 0..n_ch {
        for j in 0..n_ch {
            let mut sum = 0.0;
            for k in 0..n_ch {
                let lambda = eigenvalues[k] + epsilon;
                let clamped = if lambda > epsilon { lambda } else { epsilon };
                let inv_sqrt = 1.0 / float::sqrt(clamped);
                sum += eigenvectors.get(i, k) * inv_sqrt * eigenvectors.get(j, k);
            }
            w[i * n_ch + j] = sum;
        }
    }
    Ok(w)
}

/// Apply whitening matrix to data in-place.
fn apply_whitening_heap(data: &mut [Float], n_samples: usize, n_ch: usize, w: &[Float]) {
    let mut tmp = vec![0.0 as Float; n_ch];
    for t in 0..n_samples {
        let row = t * n_ch;
        for i in 0..n_ch {
            let mut sum = 0.0;
            for j in 0..n_ch {
                sum += w[i * n_ch + j] * data[row + j];
            }
            tmp[i] = sum;
        }
        data[row..row + n_ch].copy_from_slice(&tmp);
    }
}

/// Detect spikes on flat row-major whitened data (amplitude mode).
fn detect_spikes_heap(
    data: &[Float],
    n_samples: usize,
    n_ch: usize,
    threshold: Float,
    refractory: usize,
    events: &mut [MultiChannelEvent],
) -> usize {
    // After whitening, noise ~ 1.0 per channel. Threshold in sigma units.
    let thresh = -threshold; // detect negative peaks
    let mut n = 0;
    let mut last_spike = vec![0usize; n_ch]; // last spike sample per channel
                                             // Initialize to allow detection from sample 0
    for ls in last_spike.iter_mut() {
        *ls = 0;
    }

    for t in 1..n_samples.saturating_sub(1) {
        let row = t * n_ch;
        for ch in 0..n_ch {
            let val = data[row + ch];
            if val < thresh {
                // Check refractory
                if t < last_spike[ch] + refractory {
                    continue;
                }
                // Check local minimum (negative peak)
                let prev = data[(t - 1) * n_ch + ch];
                let next = data[(t + 1) * n_ch + ch];
                if val <= prev && val <= next {
                    if n < events.len() {
                        events[n] = MultiChannelEvent {
                            sample: t,
                            channel: ch,
                            amplitude: val,
                        };
                        n += 1;
                        last_spike[ch] = t;
                    }
                }
            }
        }
    }
    n
}

/// Compute per-channel adaptive thresholds for heap-allocated data.
/// Returns absolute thresholds (positive values) for each channel.
fn compute_adaptive_thresholds_heap(
    data: &[Float],
    n_samples: usize,
    n_ch: usize,
    base_multiplier: Float,
    min_threshold: Float,
    max_rate_hz: Float,
    sample_rate: Float,
) -> Vec<Float> {
    let mut thresholds = vec![min_threshold; n_ch];
    if n_samples == 0 {
        return thresholds;
    }

    // Per-channel MAD noise estimation
    let mut scratch = vec![0.0 as Float; n_samples];
    for ch in 0..n_ch {
        for t in 0..n_samples {
            scratch[t] = float::abs(data[t * n_ch + ch]);
        }
        scratch[..n_samples]
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = if n_samples % 2 == 1 {
            scratch[n_samples / 2]
        } else {
            (scratch[n_samples / 2 - 1] + scratch[n_samples / 2]) * 0.5
        };
        let noise = median / 0.6745;
        let base = base_multiplier * noise;
        thresholds[ch] = if base > min_threshold {
            base
        } else {
            min_threshold
        };
    }

    // Activity check: scale up overactive channels
    let duration_s = n_samples as Float / sample_rate;
    if duration_s > 0.0 && max_rate_hz > 0.0 {
        for ch in 0..n_ch {
            let thresh = thresholds[ch];
            let mut crossings = 0usize;
            for t in 1..n_samples {
                if data[t * n_ch + ch] < -thresh && data[(t - 1) * n_ch + ch] >= -thresh {
                    crossings += 1;
                }
            }
            let rate = crossings as Float / duration_s;
            if rate > max_rate_hz {
                let scale = float::sqrt(rate / max_rate_hz);
                thresholds[ch] *= scale;
                if thresholds[ch] < min_threshold {
                    thresholds[ch] = min_threshold;
                }
            }
        }
    }

    thresholds
}

/// Detect spikes with per-channel adaptive thresholds on heap data.
fn detect_spikes_heap_adaptive(
    data: &[Float],
    n_samples: usize,
    n_ch: usize,
    thresholds: &[Float],
    refractory: usize,
    events: &mut [MultiChannelEvent],
) -> usize {
    let mut n = 0;
    let mut last_spike = vec![0usize; n_ch];

    for t in 1..n_samples.saturating_sub(1) {
        let row = t * n_ch;
        for ch in 0..n_ch {
            let val = data[row + ch];
            if val < -thresholds[ch] {
                if t < last_spike[ch] + refractory {
                    continue;
                }
                let prev = data[(t - 1) * n_ch + ch];
                let next = data[(t + 1) * n_ch + ch];
                if val <= prev && val <= next {
                    if n < events.len() {
                        events[n] = MultiChannelEvent {
                            sample: t,
                            channel: ch,
                            amplitude: val,
                        };
                        n += 1;
                        last_spike[ch] = t;
                    }
                }
            }
        }
    }
    n
}

/// Spatial deduplication of events using electrode positions.
fn deduplicate_events_heap(
    events: &mut [MultiChannelEvent],
    n: usize,
    positions: &[[Float; 2]],
    spatial_radius: Float,
    temporal_radius: usize,
) -> usize {
    if n == 0 {
        return 0;
    }
    // Sort by sample time, then by amplitude (most negative first)
    events[..n].sort_unstable_by(|a, b| {
        a.sample.cmp(&b.sample).then(
            a.amplitude
                .partial_cmp(&b.amplitude)
                .unwrap_or(core::cmp::Ordering::Equal),
        )
    });

    let mut keep = vec![true; n];
    let r2 = spatial_radius * spatial_radius;

    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let ei = &events[i];
        // Mark later events within spatial+temporal window as duplicates
        for j in (i + 1)..n {
            if events[j].sample > ei.sample + temporal_radius {
                break;
            }
            if !keep[j] {
                continue;
            }
            let ej = &events[j];
            let dx = positions[ei.channel][0] - positions[ej.channel][0];
            let dy = positions[ei.channel][1] - positions[ej.channel][1];
            if dx * dx + dy * dy <= r2 {
                keep[j] = false;
            }
        }
    }

    // Compact
    let mut out = 0;
    for i in 0..n {
        if keep[i] {
            events[out] = events[i];
            out += 1;
        }
    }
    out
}

/// Align events to negative peak within half_window on flat data.
fn align_to_peak_heap(
    data: &[Float],
    n_samples: usize,
    n_ch: usize,
    events: &mut [MultiChannelEvent],
    n: usize,
    half_window: usize,
) {
    for i in 0..n {
        let ev = &events[i];
        let ch = ev.channel;
        let t0 = ev.sample;
        let start = if t0 > half_window {
            t0 - half_window
        } else {
            0
        };
        let end = (t0 + half_window + 1).min(n_samples);
        let mut best_t = t0;
        let mut best_val = data[t0 * n_ch + ch];
        for t in start..end {
            let v = data[t * n_ch + ch];
            if v < best_val {
                best_val = v;
                best_t = t;
            }
        }
        events[i].sample = best_t;
        events[i].amplitude = best_val;
    }
}

/// Extract peak-channel waveforms from flat data into [Float; W] buffers.
fn extract_peak_channel_heap(
    data: &[Float],
    n_samples: usize,
    n_ch: usize,
    events: &[MultiChannelEvent],
    n: usize,
    pre_samples: usize,
    waveforms: &mut [[Float; W]],
) -> usize {
    let post = W - pre_samples;
    let mut out = 0;
    for i in 0..n {
        let ev = &events[i];
        let ch = ev.channel;
        let t0 = ev.sample;
        if t0 < pre_samples || t0 + post > n_samples {
            continue;
        }
        for w_idx in 0..W {
            let t = t0 - pre_samples + w_idx;
            waveforms[out][w_idx] = data[t * n_ch + ch];
        }
        // Copy event to output position if compacting
        if out != i {
            // Caller must handle event compaction
        }
        out += 1;
    }
    out
}

/// Common median reference on flat data in-place.
fn cmr_heap(data: &mut [Float], n_samples: usize, n_ch: usize) {
    let mut buf = vec![0.0 as Float; n_ch];
    for t in 0..n_samples {
        let row = t * n_ch;
        buf.copy_from_slice(&data[row..row + n_ch]);
        buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        #[allow(clippy::manual_is_multiple_of)]
        let median = if n_ch % 2 == 0 {
            (buf[n_ch / 2 - 1] + buf[n_ch / 2]) * 0.5
        } else {
            buf[n_ch / 2]
        };
        for ch in 0..n_ch {
            data[row + ch] -= median;
        }
    }
}

/// Bandpass filter on flat data in-place (4th order Butterworth, 2 biquad passes).
fn bandpass_heap(
    data: &mut [Float],
    n_samples: usize,
    n_ch: usize,
    sample_rate: Float,
    low: Float,
    high: Float,
) {
    if n_samples < 4 || low >= high || sample_rate <= 0.0 {
        return;
    }
    let w_low = float::tan(float::PI * low / sample_rate);
    let w_high = float::tan(float::PI * high / sample_rate);
    let bw = w_high - w_low;
    let w0_sq = w_low * w_high;
    let alpha = bw;
    let a0 = 1.0 + alpha + w0_sq;
    let a0_inv = 1.0 / a0;
    let b0 = alpha * a0_inv;
    let b2 = -alpha * a0_inv;
    let a1 = 2.0 * (w0_sq - 1.0) * a0_inv;
    let a2 = (1.0 - alpha + w0_sq) * a0_inv;

    for ch in 0..n_ch {
        for _pass in 0..2 {
            let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
            for t in 0..n_samples {
                let idx = t * n_ch + ch;
                let x = data[idx];
                let y = b0 * x + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = x;
                y2 = y1;
                y1 = y;
                data[idx] = y;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sort_heap: full pipeline
// ---------------------------------------------------------------------------

/// Sort multi-channel data with arbitrary channel count (heap-allocated).
///
/// This is the >128ch equivalent of `sort_multichannel`. All channel-dimension
/// arrays are heap-allocated. Waveform processing (PCA, k-means, merge/split)
/// reuses the existing const-generic code since W, K, N are fixed.
///
/// # Parameters
///
/// * `config` - Sorting configuration.
/// * `positions` - 2D electrode positions, one `[x, y]` per channel.
/// * `data` - Row-major flat array of shape `(n_samples, n_channels)`.
/// * `n_samples` - Number of time samples.
/// * `n_channels` - Number of channels.
pub fn sort_heap(
    config: &SortConfig,
    positions: &[[Float; 2]],
    data: &mut [Float],
    n_samples: usize,
    n_channels: usize,
) -> Result<DynSortResult, SortError> {
    if data.len() != n_samples * n_channels {
        return Err(SortError::InvalidInput);
    }
    if positions.len() != n_channels {
        return Err(SortError::InvalidInput);
    }
    if n_samples < W {
        return Err(SortError::InsufficientData);
    }

    // 0a. Common Median Reference
    if config.common_median_ref && n_channels > 2 {
        cmr_heap(data, n_samples, n_channels);
    }

    // 0b. Bandpass filter
    if config.bandpass_low > 0.0
        && config.bandpass_high > config.bandpass_low
        && config.sample_rate > 0.0
    {
        bandpass_heap(
            data,
            n_samples,
            n_channels,
            config.sample_rate,
            config.bandpass_low,
            config.bandpass_high,
        );
    }

    // 1. Noise estimation
    let pre_noise = estimate_noise_heap(data, n_samples, n_channels);
    let mut noise_mean: Float = 0.0;
    for v in &pre_noise {
        noise_mean += v;
    }
    noise_mean /= n_channels as Float;
    if noise_mean <= 0.0 {
        noise_mean = 1.0;
    }

    // 2. Covariance
    let cov = compute_covariance_heap(data, n_samples, n_channels);

    // 3. Whitening (ZCA, in-place)
    let w_matrix = build_whitening_heap(&cov, n_channels, config.whitening_epsilon)
        .map_err(|_| SortError::EigenFailed)?;
    apply_whitening_heap(data, n_samples, n_channels, &w_matrix);

    // 4. Detection (amplitude mode on whitened data)
    let max_events = n_samples / config.refractory_samples.max(1) + n_channels;
    let mut event_buf = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        max_events
    ];

    let n_detected = if config.adaptive_threshold {
        let thresholds = compute_adaptive_thresholds_heap(
            data,
            n_samples,
            n_channels,
            config.threshold_multiplier,
            config.adaptive_min_threshold,
            config.adaptive_max_rate_hz,
            config.sample_rate,
        );
        detect_spikes_heap_adaptive(
            data,
            n_samples,
            n_channels,
            &thresholds,
            config.refractory_samples,
            &mut event_buf,
        )
    } else {
        detect_spikes_heap(
            data,
            n_samples,
            n_channels,
            config.threshold_multiplier,
            config.refractory_samples,
            &mut event_buf,
        )
    };

    // 5. Deduplication
    let n_dedup = deduplicate_events_heap(
        &mut event_buf,
        n_detected,
        positions,
        config.spatial_radius_um,
        config.temporal_radius,
    );

    // 6. Alignment
    align_to_peak_heap(
        data,
        n_samples,
        n_channels,
        &mut event_buf,
        n_dedup,
        config.align_half_window,
    );

    // 7. Extraction (peak-channel waveforms -> [Float; W])
    let mut waveform_buf = vec![[0.0 as Float; W]; n_dedup.max(1)];
    let n_extracted = extract_peak_channel_heap(
        data,
        n_samples,
        n_channels,
        &event_buf,
        n_dedup,
        config.pre_samples,
        &mut waveform_buf,
    );

    // Compact events to match extracted waveforms
    // (events that couldn't be extracted due to boundary are dropped)
    let mut compact_events = Vec::with_capacity(n_extracted);
    let pre = config.pre_samples;
    let post = W - pre;
    for i in 0..n_dedup {
        let t0 = event_buf[i].sample;
        if t0 >= pre && t0 + post <= n_samples {
            compact_events.push(event_buf[i]);
        }
    }
    // Copy back
    for (i, ev) in compact_events.iter().enumerate() {
        event_buf[i] = *ev;
    }

    if n_extracted == 0 {
        return Ok(DynSortResult {
            n_spikes: 0,
            n_clusters: 0,
            labels: Vec::new(),
            spike_times: Vec::new(),
            spike_channels: Vec::new(),
            clusters: Vec::new(),
        });
    }

    // 8. PCA (uses const-generic W, K, WM -- no channel dependency)
    let mut feature_buf = vec![[0.0 as Float; K]; n_extracted];
    let mut pca = WaveformPca::<W, K, WM>::new();
    if pca.fit(&waveform_buf[..n_extracted]).is_ok() {
        for i in 0..n_extracted {
            let _ = pca.transform(&waveform_buf[i], &mut feature_buf[i]);
        }
    } else {
        // Fallback: use first K samples as features
        for i in 0..n_extracted {
            for k in 0..K {
                feature_buf[i][k] = waveform_buf[i][k.min(W - 1)];
            }
        }
    }

    // 8b. Encode channel index as last feature dimension
    if n_channels > 1 {
        let ch_scale = 1.0 / (n_channels as Float);
        for i in 0..n_extracted {
            feature_buf[i][K - 1] = event_buf[i].channel as Float * ch_scale;
        }
    }

    // 9. Clustering (OnlineKMeans<K, N> -- const generic, no channel dep)
    let mut labels = vec![0usize; n_extracted];
    let mut kmeans = OnlineKMeans::<K, N>::new(config.cluster_max_count);
    kmeans.set_create_threshold(config.cluster_threshold);

    // SVD init or farthest-point init
    if config.svd_init && n_extracted > 2 {
        let (seeds, n_seeds) =
            crate::sorter::svd_init_centroids::<K, N>(&feature_buf, n_extracted, N);
        for s in 0..n_seeds {
            let _ = kmeans.seed_centroid(&seeds[s]);
        }
    } else if n_extracted > 0 {
        kmeans.init_farthest_point(&feature_buf[..n_extracted], N.min(n_extracted));
    }

    for i in 0..n_extracted {
        let result = kmeans.update(&feature_buf[i]);
        labels[i] = result.cluster;
    }
    let mut n_clusters = kmeans.n_active();

    // 9b. Merge clusters (d-prime based)
    let mut merge_scratch = vec![0.0 as Float; n_extracted];
    n_clusters = merge_clusters::<K>(
        n_extracted,
        &mut labels,
        &feature_buf,
        &event_buf,
        n_clusters,
        config.merge_dprime_threshold,
        config.merge_isi_threshold,
        config.refractory_samples,
        &mut merge_scratch,
        K,
    );

    // 9c. Split bimodal clusters
    n_clusters = split_clusters::<K>(
        n_extracted,
        &mut labels,
        &feature_buf,
        n_clusters,
        config.split_min_cluster_size,
        config.split_bimodality_threshold,
    );

    // 9c3b. Amplitude bimodality split
    if config.split_bimodality_threshold > 0.0 && n_clusters > 0 {
        n_clusters = amplitude_bimodality_split(
            n_extracted,
            &mut labels,
            &event_buf,
            n_clusters,
            config.split_bimodality_threshold,
            config.split_min_cluster_size,
            N,
        );
    }

    // 10. Quality metrics + SNR curation
    let mut clusters_out = Vec::with_capacity(n_clusters);
    let mut label_map = vec![usize::MAX; N];
    let mut new_id = 0usize;
    for cl in 0..n_clusters {
        let mut count = 0usize;
        let mut amp_sum: Float = 0.0;
        for i in 0..n_extracted {
            if labels[i] == cl {
                count += 1;
                let a = float::abs(event_buf[i].amplitude);
                amp_sum += a;
            }
        }
        if count == 0 {
            continue;
        }
        let mean_amp = amp_sum / count as Float;
        let snr = mean_amp / noise_mean;

        if snr < config.min_cluster_snr {
            continue;
        }

        // ISI violation rate
        let mut spike_times_cl: Vec<usize> = Vec::new();
        for i in 0..n_extracted {
            if labels[i] == cl {
                spike_times_cl.push(event_buf[i].sample);
            }
        }
        spike_times_cl.sort_unstable();
        let mut isi_violations = 0usize;
        for w in spike_times_cl.windows(2) {
            if w[1] - w[0] < config.refractory_samples {
                isi_violations += 1;
            }
        }
        let isi_rate = if spike_times_cl.len() > 1 {
            isi_violations as Float / (spike_times_cl.len() - 1) as Float
        } else {
            0.0
        };

        label_map[cl] = new_id;
        new_id += 1;
        clusters_out.push(DynClusterInfo {
            count,
            snr,
            isi_violation_rate: isi_rate,
        });
    }

    // Remap labels and collect output
    let mut out_labels = Vec::with_capacity(n_extracted);
    let mut out_times = Vec::with_capacity(n_extracted);
    let mut out_channels = Vec::with_capacity(n_extracted);
    let mut out_count = 0;

    for i in 0..n_extracted {
        let cl = labels[i];
        if cl < N && label_map[cl] != usize::MAX {
            out_labels.push(label_map[cl]);
            out_times.push(event_buf[i].sample);
            out_channels.push(event_buf[i].channel);
            out_count += 1;
        }
    }

    Ok(DynSortResult {
        n_spikes: out_count,
        n_clusters: clusters_out.len(),
        labels: out_labels,
        spike_times: out_times,
        spike_channels: out_channels,
        clusters: clusters_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorter::SortConfig;

    #[test]
    fn test_sort_heap_256ch_noise() {
        let n_ch = 256;
        let n_samples = 3000;
        let mut data = vec![0.0 as Float; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 17 + 3) % 100) as Float * 0.001 - 0.05;
        }
        let positions: Vec<[Float; 2]> = (0..n_ch).map(|i| [0.0, i as Float * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_heap(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert_eq!(result.n_spikes, 0);
    }

    #[test]
    fn test_sort_heap_384ch_noise() {
        let n_ch = 384;
        let n_samples = 3000;
        let mut data = vec![0.0 as Float; n_samples * n_ch];
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 13 + 7) % 100) as Float * 0.001 - 0.05;
        }
        let positions: Vec<[Float; 2]> = (0..n_ch).map(|i| [0.0, i as Float * 25.0]).collect();
        let config = SortConfig::default();
        let result = sort_heap(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        assert_eq!(result.n_spikes, 0);
    }

    #[test]
    fn test_sort_heap_256ch_with_spikes() {
        let n_ch = 256;
        let n_samples = 5000;
        let mut data = vec![0.0 as Float; n_samples * n_ch];
        // Small noise
        for (i, d) in data.iter_mut().enumerate() {
            *d = ((i * 7 + 11) % 50) as Float * 0.001 - 0.025;
        }
        // Inject spikes on channel 10 at sample 500, 1500, 2500, 3500
        for &spike_t in &[500usize, 1500, 2500, 3500] {
            for offset in 0..20 {
                let t = spike_t + offset;
                if t < n_samples {
                    let amp = if offset < 10 {
                        -(offset as Float) * 2.0
                    } else {
                        -((20 - offset) as Float) * 2.0
                    };
                    data[t * n_ch + 10] = amp;
                }
            }
        }
        let positions: Vec<[Float; 2]> = (0..n_ch).map(|i| [0.0, i as Float * 25.0]).collect();
        let config = SortConfig {
            threshold_multiplier: 4.0,
            ..SortConfig::default()
        };
        let result = sort_heap(&config, &positions, &mut data, n_samples, n_ch).unwrap();
        // Should detect some spikes (at least 1)
        assert!(result.n_spikes > 0, "Expected spikes, got 0");
    }

    #[test]
    fn test_sort_heap_invalid_input() {
        let mut data = vec![0.0 as Float; 100];
        let positions: Vec<[Float; 2]> = (0..4).map(|i| [0.0, i as Float * 25.0]).collect();
        let config = SortConfig::default();
        // data length doesn't match
        assert!(sort_heap(&config, &positions, &mut data, 50, 4).is_err());
    }

    #[test]
    fn test_heap_matrix_eigen_identity() {
        let mat = HeapMatrix::identity(3);
        let eigen_tol = if cfg!(feature = "f32") { 1e-6 } else { 1e-12 };
        let test_tol = if cfg!(feature = "f32") { 1e-4 } else { 1e-10 };
        let (evals, _) = mat.eigen_symmetric(50, eigen_tol).unwrap();
        for &ev in &evals {
            assert!((ev - 1.0).abs() < test_tol);
        }
    }

    #[test]
    fn test_heap_whitening_identity_cov() {
        // Identity covariance -> whitening matrix should be ~identity
        let n_ch = 4;
        let mut cov = vec![0.0 as Float; n_ch * n_ch];
        for i in 0..n_ch {
            cov[i * n_ch + i] = 1.0;
        }
        let w = build_whitening_heap(&cov, n_ch, 1e-6).unwrap();
        // Check diagonal elements are ~1
        for i in 0..n_ch {
            assert!(
                (w[i * n_ch + i] - 1.0).abs() < 0.01,
                "Expected ~1.0, got {}",
                w[i * n_ch + i]
            );
        }
    }
}
