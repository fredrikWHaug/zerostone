//! Connectivity metrics for brain signal analysis.
//!
//! This module provides coherence, phase locking value (PLV), and Granger
//! causality for measuring synchronization and directional information flow
//! between brain regions. These are fundamental metrics in BCI research for
//! connectivity analysis.
//!
//! # Overview
//!
//! - [`coherence`] — Single-window magnitude-squared coherence via FFT
//! - [`spectral_coherence`] — Welch-style averaged coherence (more robust)
//! - [`phase_locking_value`] — Phase synchronization metric from instantaneous phases
//! - [`coherence_frequencies`] — Frequency bin centers for coherence output
//! - [`granger_causality`] — Test if one signal Granger-causes another
//! - [`conditional_granger`] — Granger causality conditioned on a confound signal
//! - [`granger_significance`] — P-value from F-statistic for Granger test
//!
//! # Coherence
//!
//! Magnitude-squared coherence measures the linear relationship between two
//! signals at each frequency:
//!
//! ```text
//! Cxy(f) = |Sxy(f)|² / (Sxx(f) · Syy(f))
//! ```
//!
//! where Sxy is the cross-spectral density and Sxx, Syy are the auto-spectral
//! densities. Values range from 0 (no linear relationship) to 1 (perfect
//! linear relationship).
//!
//! # Phase Locking Value
//!
//! PLV measures phase synchronization independent of amplitude:
//!
//! ```text
//! PLV = |mean(exp(j · (φ_a - φ_b)))|
//! ```
//!
//! Values range from 0 (random phase relationship) to 1 (constant phase
//! difference). Requires narrowband-filtered signals and Hilbert transform
//! for instantaneous phase extraction.
//!
//! # Example
//!
//! ```
//! use zerostone::connectivity::{coherence, phase_locking_value};
//! use zerostone::{WindowType, Float};
//!
//! // Two identical signals have coherence = 1.0
//! let signal = [0.0 as Float; 256];
//! let mut coh = [0.0 as Float; 129]; // N/2 + 1
//! coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);
//! // coh[k] ≈ 1.0 for all bins with signal energy
//!
//! // Phase-locked signals have PLV ≈ 1.0
//! let phases_a = [0.0 as Float; 100];
//! let phases_b = [0.5 as Float; 100]; // constant offset
//! let plv = phase_locking_value(&phases_a, &phases_b);
//! assert!((plv - 1.0).abs() < 1e-6); // perfect phase locking
//! ```

use crate::fft::{Complex, Fft};
use crate::float::Float;
use crate::window::{window_coefficient, WindowType};

/// Compute magnitude-squared coherence between two signals.
///
/// Uses a single FFT window. For more robust estimates with lower variance,
/// use [`spectral_coherence`] which averages over multiple overlapping segments.
///
/// # Arguments
///
/// * `signal_a` - First signal (length >= N)
/// * `signal_b` - Second signal (length >= N)
/// * `window` - Window function to apply before FFT
/// * `output` - Output buffer for coherence values (length >= N/2 + 1)
///
/// # Output
///
/// Writes N/2 + 1 coherence values in \[0, 1\] to `output`.
///
/// # Panics
///
/// Panics if signals are shorter than N or output is too small.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::coherence;
/// use zerostone::float::{self, Float, PI};
/// use zerostone::WindowType;
///
/// let mut signal = [0.0 as Float; 256];
/// for (i, s) in signal.iter_mut().enumerate() {
///     let t = i as Float / 256.0;
///     *s = float::sin(2.0 * PI * 10.0 * t);
/// }
///
/// let mut coh = [0.0 as Float; 129];
/// coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);
///
/// // Identical signals: coherence = 1.0 at the signal frequency
/// assert!(coh[10] > 0.99);
/// ```
pub fn coherence<const N: usize>(
    signal_a: &[Float],
    signal_b: &[Float],
    window: WindowType,
    output: &mut [Float],
) {
    assert!(
        signal_a.len() >= N,
        "signal_a length {} must be >= {}",
        signal_a.len(),
        N
    );
    assert!(
        signal_b.len() >= N,
        "signal_b length {} must be >= {}",
        signal_b.len(),
        N
    );
    let bins = N / 2 + 1;
    assert!(
        output.len() >= bins,
        "output length {} must be >= {}",
        output.len(),
        bins
    );

    let fft = Fft::<N>::new();

    // Window and FFT signal A
    let mut data_a: [Complex; N] = core::array::from_fn(|i| {
        let w = window_coefficient(window, i, N);
        Complex::from_real(signal_a[i] * w)
    });
    fft.forward(&mut data_a);

    // Window and FFT signal B
    let mut data_b: [Complex; N] = core::array::from_fn(|i| {
        let w = window_coefficient(window, i, N);
        Complex::from_real(signal_b[i] * w)
    });
    fft.forward(&mut data_b);

    // Compute coherence: |Sxy|² / (Sxx · Syy)
    for k in 0..bins {
        let sxy = data_a[k].cmul(data_b[k].conj());
        let sxx = data_a[k].magnitude_squared();
        let syy = data_b[k].magnitude_squared();

        let denom = sxx * syy;
        output[k] = if denom > 1e-20 {
            sxy.magnitude_squared() / denom
        } else {
            0.0
        };
    }
}

/// Compute Welch-style averaged coherence between two signals.
///
/// More robust than single-window [`coherence`]. Divides signals into
/// overlapping segments, computes cross- and auto-spectral densities for
/// each segment, averages them, then computes coherence from the averages.
///
/// # Arguments
///
/// * `signal_a` - First signal (length >= N, must equal signal_b length)
/// * `signal_b` - Second signal (length >= N, must equal signal_a length)
/// * `overlap_frac` - Overlap fraction in \[0.0, 1.0). 0.5 = 50% overlap (recommended).
/// * `window` - Window function to apply to each segment
/// * `output_coh` - Output buffer for coherence values (length >= N/2 + 1)
///
/// # Returns
///
/// Number of segments averaged.
///
/// # Panics
///
/// Panics if signals have different lengths, are shorter than N, or output is too small.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::spectral_coherence;
/// use zerostone::float::{self, Float, PI};
/// use zerostone::WindowType;
///
/// let mut sig_a = [0.0 as Float; 1024];
/// let mut sig_b = [0.0 as Float; 1024];
/// for (i, (a, b)) in sig_a.iter_mut().zip(sig_b.iter_mut()).enumerate() {
///     let t = i as Float / 256.0;
///     *a = float::sin(2.0 * PI * 10.0 * t);
///     *b = float::sin(2.0 * PI * 10.0 * t);
/// }
///
/// let mut coh = [0.0 as Float; 129];
/// let segments = spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);
/// assert!(segments > 1);
/// assert!(coh[10] > 0.99); // 10 Hz bin
/// ```
pub fn spectral_coherence<const N: usize>(
    signal_a: &[Float],
    signal_b: &[Float],
    overlap_frac: Float,
    window: WindowType,
    output_coh: &mut [Float],
) -> usize {
    assert!(
        signal_a.len() >= N,
        "signal_a length {} must be >= {}",
        signal_a.len(),
        N
    );
    assert!(
        signal_b.len() >= N,
        "signal_b length {} must be >= {}",
        signal_b.len(),
        N
    );
    assert_eq!(
        signal_a.len(),
        signal_b.len(),
        "Signals must have equal length"
    );
    let bins = N / 2 + 1;
    assert!(
        output_coh.len() >= bins,
        "output length {} must be >= {}",
        output_coh.len(),
        bins
    );
    assert!(
        (0.0..1.0).contains(&overlap_frac),
        "overlap_frac must be in [0.0, 1.0)"
    );

    let fft = Fft::<N>::new();
    let overlap = (N as Float * overlap_frac) as usize;
    let hop = N - overlap;
    let signal_len = signal_a.len();
    let num_segments = (signal_len - N) / hop + 1;

    // Accumulators for averaged spectra (use full N arrays, only first `bins` used)
    let mut sxx_acc = [0.0 as Float; N];
    let mut syy_acc = [0.0 as Float; N];
    let mut sxy_re_acc = [0.0 as Float; N];
    let mut sxy_im_acc = [0.0 as Float; N];

    for seg in 0..num_segments {
        let start = seg * hop;

        let mut data_a: [Complex; N] = core::array::from_fn(|i| {
            let w = window_coefficient(window, i, N);
            Complex::from_real(signal_a[start + i] * w)
        });
        fft.forward(&mut data_a);

        let mut data_b: [Complex; N] = core::array::from_fn(|i| {
            let w = window_coefficient(window, i, N);
            Complex::from_real(signal_b[start + i] * w)
        });
        fft.forward(&mut data_b);

        for k in 0..bins {
            sxx_acc[k] += data_a[k].magnitude_squared();
            syy_acc[k] += data_b[k].magnitude_squared();
            let cross = data_a[k].cmul(data_b[k].conj());
            sxy_re_acc[k] += cross.re;
            sxy_im_acc[k] += cross.im;
        }
    }

    // Compute coherence from averaged spectra
    for k in 0..bins {
        let sxy_mag_sq = sxy_re_acc[k] * sxy_re_acc[k] + sxy_im_acc[k] * sxy_im_acc[k];
        let denom = sxx_acc[k] * syy_acc[k];
        output_coh[k] = if denom > 1e-20 {
            sxy_mag_sq / denom
        } else {
            0.0
        };
    }

    num_segments
}

/// Compute Phase Locking Value between two instantaneous phase arrays.
///
/// PLV measures the consistency of the phase difference between two signals:
///
/// ```text
/// PLV = |mean(exp(j · (φ_a - φ_b)))|
/// ```
///
/// # Arguments
///
/// * `phases_a` - Instantaneous phases of signal A (radians, from Hilbert transform)
/// * `phases_b` - Instantaneous phases of signal B (radians, from Hilbert transform)
///
/// # Returns
///
/// PLV in \[0, 1\] where:
/// - 1.0 = constant phase difference (perfect synchronization)
/// - 0.0 = random phase relationship (no synchronization)
///
/// # Panics
///
/// Panics if phase arrays have different lengths or are empty.
///
/// # Example
///
/// ```
/// use zerostone::connectivity::phase_locking_value;
/// use zerostone::Float;
///
/// // Constant phase difference → PLV = 1.0
/// let phases_a: [Float; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
/// let phases_b: [Float; 5] = [0.3, 0.8, 1.3, 1.8, 2.3]; // constant 0.3 rad offset
/// let plv = phase_locking_value(&phases_a, &phases_b);
/// assert!((plv - 1.0).abs() < 1e-6);
/// ```
#[allow(clippy::unnecessary_cast)]
pub fn phase_locking_value(phases_a: &[Float], phases_b: &[Float]) -> Float {
    assert_eq!(
        phases_a.len(),
        phases_b.len(),
        "Phase arrays must have equal length"
    );
    assert!(!phases_a.is_empty(), "Phase arrays must not be empty");

    let n = phases_a.len();
    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;

    for i in 0..n {
        let diff = (phases_a[i] - phases_b[i]) as f64;
        sum_re += libm::cos(diff);
        sum_im += libm::sin(diff);
    }

    sum_re /= n as f64;
    sum_im /= n as f64;

    libm::sqrt(sum_re * sum_re + sum_im * sum_im) as Float
}

/// Compute frequency bin centers for coherence output.
///
/// # Arguments
///
/// * `sample_rate` - Sample rate in Hz
/// * `output` - Output buffer (length >= N/2 + 1)
///
/// # Example
///
/// ```
/// use zerostone::connectivity::coherence_frequencies;
/// use zerostone::Float;
///
/// let mut freqs = [0.0 as Float; 129];
/// coherence_frequencies::<256>(256.0, &mut freqs);
/// assert!((freqs[0] - 0.0).abs() < 1e-6);
/// assert!((freqs[1] - 1.0).abs() < 1e-6);
/// assert!((freqs[128] - 128.0).abs() < 1e-6);
/// ```
pub fn coherence_frequencies<const N: usize>(sample_rate: Float, output: &mut [Float]) {
    let bins = N / 2 + 1;
    assert!(
        output.len() >= bins,
        "output length {} must be >= {}",
        output.len(),
        bins
    );
    let freq_res = sample_rate / N as Float;
    for (k, val) in output[..bins].iter_mut().enumerate() {
        *val = k as Float * freq_res;
    }
}

// --- Granger Causality ---

/// Maximum AR model order for Granger causality.
const GRANGER_MAX_ORDER: usize = 20;

/// Maximum number of regression parameters (3 * MAX_ORDER for conditional).
const GRANGER_MAX_DIM: usize = 60;

/// Result of a Granger causality test.
///
/// Tests whether one time series (x) provides statistically significant
/// information for predicting another (y) beyond what y's own past provides.
///
/// The test compares two models:
/// - Restricted: y predicted from its own lagged values only
/// - Unrestricted: y predicted from lagged values of both y and x
///
/// If the unrestricted model significantly reduces prediction error, x is said
/// to Granger-cause y.
#[derive(Debug, Clone, Copy)]
pub struct GrangerResult {
    /// F-statistic for the Granger causality test.
    pub f_statistic: f64,
    /// p-value (probability of observing this F-statistic under the null).
    pub p_value: f64,
    /// Residual variance of the restricted model.
    pub restricted_variance: f64,
    /// Residual variance of the unrestricted model.
    pub unrestricted_variance: f64,
}

/// Continued fraction evaluation for the regularized incomplete beta function.
///
/// Uses the modified Lentz algorithm (Numerical Recipes, Press et al.).
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-12;
    const FPMIN: f64 = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if libm::fabs(d) < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        // Even step
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if libm::fabs(d) < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if libm::fabs(c) < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if libm::fabs(d) < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if libm::fabs(c) < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if libm::fabs(del - 1.0) <= EPS {
            break;
        }
    }

    h
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction representation with symmetry transform
/// for convergence.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let (lga, _) = libm::lgamma_r(a);
    let (lgb, _) = libm::lgamma_r(b);
    let (lgab, _) = libm::lgamma_r(a + b);

    let bt = libm::exp(a * libm::log(x) + b * libm::log(1.0 - x) + lgab - lga - lgb);

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Survival function of the F-distribution: P(F > f | df1, df2).
fn f_distribution_sf(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    let x = df1 * f / (df1 * f + df2);
    1.0 - regularized_incomplete_beta(x, df1 / 2.0, df2 / 2.0)
}

/// Cholesky decomposition of a dim x dim matrix stored as flat row-major array.
///
/// Overwrites the lower triangle of `a` with L such that A = L * L^T.
/// Returns false if matrix is not positive definite.
fn cholesky_inplace(a: &mut [f64], dim: usize) -> bool {
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += a[i * dim + k] * a[j * dim + k];
            }
            if i == j {
                let val = a[i * dim + i] - sum;
                if val <= 1e-14 {
                    return false;
                }
                a[i * dim + j] = libm::sqrt(val);
            } else {
                a[i * dim + j] = (a[i * dim + j] - sum) / a[j * dim + j];
            }
        }
    }
    true
}

/// Solve L * L^T * x = b given Cholesky factor L (lower triangular, flat row-major).
fn cholesky_solve(l: &[f64], b: &[f64], dim: usize, x: &mut [f64]) {
    // Forward substitution: L * z = b
    let mut z = [0.0f64; GRANGER_MAX_DIM];
    for i in 0..dim {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * dim + j] * z[j];
        }
        z[i] = sum / l[i * dim + i];
    }
    // Backward substitution: L^T * x = z
    for i in (0..dim).rev() {
        let mut sum = z[i];
        for j in (i + 1)..dim {
            sum -= l[j * dim + i] * x[j];
        }
        x[i] = sum / l[i * dim + i];
    }
}

/// Compute residual sum of squares from OLS fit of target on lagged signals.
///
/// Fits the model: target(t) = sum_{s,k} beta_{s,k} * signals[s][t-k-1] + e(t)
/// for t in [order, n), and returns the residual sum of squares.
///
/// Each signal contributes `order` lagged regressors (lags 1 through order).
fn ols_residual_sum_of_squares(signals: &[&[f64]], target: &[f64], n: usize, order: usize) -> f64 {
    let n_signals = signals.len();
    let dim = n_signals * order;

    // Build X'X (upper triangle) and X'y
    let mut xtx = [0.0f64; GRANGER_MAX_DIM * GRANGER_MAX_DIM];
    let mut xty = [0.0f64; GRANGER_MAX_DIM];

    for t in order..n {
        let yt = target[t];
        for i in 0..dim {
            let si = i / order;
            let ki = i % order;
            let ri = signals[si][t - ki - 1];
            xty[i] += ri * yt;
            for j in i..dim {
                let sj = j / order;
                let kj = j % order;
                let rj = signals[sj][t - kj - 1];
                xtx[i * dim + j] += ri * rj;
            }
        }
    }

    // Mirror upper to lower triangle
    for i in 0..dim {
        for j in (i + 1)..dim {
            xtx[j * dim + i] = xtx[i * dim + j];
        }
    }

    // Tikhonov regularization for numerical stability
    let mut trace = 0.0f64;
    for i in 0..dim {
        trace += xtx[i * dim + i];
    }
    if trace > 0.0 {
        let reg = 1e-10 * trace / dim as f64;
        for i in 0..dim {
            xtx[i * dim + i] += reg;
        }
    }

    // Cholesky solve
    assert!(
        cholesky_inplace(&mut xtx, dim),
        "Normal equations matrix is not positive definite (degenerate signal)"
    );
    let mut beta = [0.0f64; GRANGER_MAX_DIM];
    cholesky_solve(&xtx, &xty, dim, &mut beta);

    // Compute RSS
    let mut rss = 0.0f64;
    for t in order..n {
        let mut pred = 0.0;
        for (i, &bi) in beta[..dim].iter().enumerate() {
            let si = i / order;
            let ki = i % order;
            pred += bi * signals[si][t - ki - 1];
        }
        let residual = target[t] - pred;
        rss += residual * residual;
    }

    rss
}

/// Test whether x Granger-causes y at the given model order.
///
/// Fits two models and compares their prediction errors:
/// - Restricted: y(t) predicted from its own lags only
/// - Unrestricted: y(t) predicted from lags of both y and x
///
/// Uses OLS via normal equations (Cholesky decomposition) for both models.
/// The restricted model's normal equations have Toeplitz structure, equivalent
/// to Levinson-Durbin recursion on the sample autocovariance.
///
/// # Arguments
///
/// * `x` - Potential cause signal (f64 slice)
/// * `y` - Effect signal (f64 slice, same length as x)
/// * `order` - Model order (number of lags, 1..=20)
///
/// # Returns
///
/// [`GrangerResult`] with F-statistic, p-value, and residual variances.
///
/// # Panics
///
/// Panics if signals have different lengths, order is 0 or > 20,
/// or signals are too short (need length > 3 * order).
///
/// # Example
///
/// ```
/// use zerostone::connectivity::granger_causality;
///
/// let n = 300;
/// let mut x = [0.0f64; 300];
/// let mut y = [0.0f64; 300];
/// let mut state: u64 = 42;
/// for t in 0..n {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     x[t] = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
/// }
/// for t in 1..n {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
///     y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + noise * 0.1;
/// }
///
/// let result = granger_causality(&x, &y, 1);
/// assert!(result.p_value < 0.05);
/// ```
pub fn granger_causality(x: &[f64], y: &[f64], order: usize) -> GrangerResult {
    assert_eq!(x.len(), y.len(), "Signals must have equal length");
    assert!(order > 0, "Order must be > 0");
    assert!(
        order <= GRANGER_MAX_ORDER,
        "Order {} exceeds maximum {}",
        order,
        GRANGER_MAX_ORDER
    );
    let n = x.len();
    assert!(
        n > 3 * order,
        "Signal length {} too short for order {} (need > {})",
        n,
        order,
        3 * order
    );

    let n_eff = n - order;
    let p = order;

    // Restricted model: y on its own lags
    let rss_r = ols_residual_sum_of_squares(&[y], y, n, order);

    // Unrestricted model: y on lags of y and x
    let rss_u = ols_residual_sum_of_squares(&[y, x], y, n, order);

    // F-test: df1 = p (extra parameters), df2 = n_eff - 2p (residual df of unrestricted)
    let df1 = p as f64;
    let df2 = (n_eff - 2 * p) as f64;
    assert!(
        df2 > 0.0,
        "Not enough observations for F-test (n={}, order={})",
        n,
        order
    );

    let f_stat = if rss_u > 0.0 {
        ((rss_r - rss_u) / df1) / (rss_u / df2)
    } else {
        0.0
    };
    // Clamp to non-negative (rounding could make it slightly negative)
    let f_stat = if f_stat < 0.0 { 0.0 } else { f_stat };

    let p_value = f_distribution_sf(f_stat, df1, df2);

    GrangerResult {
        f_statistic: f_stat,
        p_value,
        restricted_variance: rss_r / n_eff as f64,
        unrestricted_variance: rss_u / n_eff as f64,
    }
}

/// Test whether x Granger-causes y, conditioned on confound signal z.
///
/// Controls for the effect of z by including its lags in both models:
/// - Restricted: y(t) predicted from lags of y and z
/// - Unrestricted: y(t) predicted from lags of y, x, and z
///
/// If x provides additional predictive information beyond what y and z
/// already provide, the test will be significant.
///
/// # Arguments
///
/// * `x` - Potential cause signal
/// * `y` - Effect signal (same length as x)
/// * `z` - Confound signal to control for (same length as x)
/// * `order` - Model order (1..=20)
///
/// # Panics
///
/// Panics if signals have different lengths, order is 0 or > 20,
/// or signals are too short (need length > 4 * order).
///
/// # Example
///
/// ```
/// use zerostone::connectivity::conditional_granger;
///
/// let n = 300;
/// let mut x = [0.0f64; 300];
/// let mut y = [0.0f64; 300];
/// let mut z = [0.0f64; 300];
/// let mut state: u64 = 42;
/// for t in 0..n {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     x[t] = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     z[t] = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
/// }
/// for t in 1..n {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
///     let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
///     y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + noise * 0.1;
/// }
///
/// let result = conditional_granger(&x, &y, &z, 1);
/// assert!(result.p_value < 0.05); // x still causes y even controlling for z
/// ```
pub fn conditional_granger(x: &[f64], y: &[f64], z: &[f64], order: usize) -> GrangerResult {
    assert_eq!(x.len(), y.len(), "Signals must have equal length");
    assert_eq!(y.len(), z.len(), "Signals must have equal length");
    assert!(order > 0, "Order must be > 0");
    assert!(
        order <= GRANGER_MAX_ORDER,
        "Order {} exceeds maximum {}",
        order,
        GRANGER_MAX_ORDER
    );
    let n = x.len();
    assert!(
        n > 4 * order,
        "Signal length {} too short for conditional Granger with order {} (need > {})",
        n,
        order,
        4 * order
    );

    let n_eff = n - order;

    // Restricted: y on lags of y and z (2p parameters)
    let rss_r = ols_residual_sum_of_squares(&[y, z], y, n, order);

    // Unrestricted: y on lags of y, x, and z (3p parameters)
    let rss_u = ols_residual_sum_of_squares(&[y, x, z], y, n, order);

    // F-test: df1 = p, df2 = n_eff - 3p
    let df1 = order as f64;
    let df2 = (n_eff - 3 * order) as f64;
    assert!(
        df2 > 0.0,
        "Not enough observations for conditional F-test (n={}, order={})",
        n,
        order
    );

    let f_stat = if rss_u > 0.0 {
        ((rss_r - rss_u) / df1) / (rss_u / df2)
    } else {
        0.0
    };
    let f_stat = if f_stat < 0.0 { 0.0 } else { f_stat };

    let p_value = f_distribution_sf(f_stat, df1, df2);

    GrangerResult {
        f_statistic: f_stat,
        p_value,
        restricted_variance: rss_r / n_eff as f64,
        unrestricted_variance: rss_u / n_eff as f64,
    }
}

/// Compute p-value for a Granger causality F-statistic.
///
/// Standalone function for computing the p-value when you have the F-statistic
/// from a previous test. Uses the F-distribution with df1 = order and
/// df2 = n_obs - 3 * order.
///
/// # Arguments
///
/// * `f_statistic` - The F-statistic from a Granger causality test
/// * `n_obs` - Number of observations in the original signals
/// * `order` - Model order used in the test
///
/// # Returns
///
/// p-value in \[0, 1\].
///
/// # Panics
///
/// Panics if there are not enough observations (need n_obs > 3 * order).
///
/// # Example
///
/// ```
/// use zerostone::connectivity::granger_significance;
///
/// // F(5, 100) = 3.0 should give a small p-value
/// let p = granger_significance(3.0, 120, 5);
/// assert!(p < 0.05);
/// ```
pub fn granger_significance(f_statistic: f64, n_obs: usize, order: usize) -> f64 {
    let n_eff = n_obs - order;
    let df1 = order as f64;
    let df2 = (n_eff - 2 * order) as f64;
    assert!(
        df2 > 0.0,
        "Not enough observations (n={}, order={})",
        n_obs,
        order
    );
    f_distribution_sf(f_statistic, df1, df2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float;
    use crate::float::{Float, PI};

    #[test]
    fn test_coherence_identical_signals() {
        let mut signal = [0.0 as Float; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as Float / 256.0;
            *s = float::sin(2.0 * PI * 10.0 * t);
        }

        let mut coh = [0.0 as Float; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);

        // At the signal frequency (bin 10), coherence should be 1.0
        assert!(
            coh[10] > 0.99,
            "Coherence at signal frequency should be ~1.0, got {}",
            coh[10]
        );
    }

    #[test]
    fn test_coherence_identical_all_bins() {
        // For identical signals, all bins with energy should have coherence 1.0
        let mut signal = [0.0 as Float; 256];
        let mut state: u32 = 42;
        for s in signal.iter_mut() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *s = (state as Float / u32::MAX as Float) * 2.0 - 1.0;
        }

        let mut coh = [0.0 as Float; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh);

        // All bins should be ~1.0 (identical signals)
        for (k, &c) in coh[1..128].iter().enumerate() {
            assert!(
                c > 0.99,
                "Coherence at bin {} should be ~1.0, got {}",
                k + 1,
                c
            );
        }
    }

    #[test]
    fn test_coherence_single_window_bias() {
        // Single-window magnitude-squared coherence is always 1.0 for any two
        // non-zero signals (known property). This is why spectral_coherence
        // with Welch averaging is needed for meaningful estimates.
        let mut sig_a = [0.0 as Float; 256];
        let mut sig_b = [0.0 as Float; 256];
        let mut state_a: u32 = 42;
        let mut state_b: u32 = 99999;
        for i in 0..256 {
            state_a = state_a.wrapping_mul(1103515245).wrapping_add(12345);
            state_b = state_b.wrapping_mul(1103515245).wrapping_add(12345);
            sig_a[i] = (state_a as Float / u32::MAX as Float) * 2.0 - 1.0;
            sig_b[i] = (state_b as Float / u32::MAX as Float) * 2.0 - 1.0;
        }

        let mut coh = [0.0 as Float; 129];
        coherence::<256>(&sig_a, &sig_b, WindowType::Hann, &mut coh);

        // Single-window coherence is biased to 1.0 — this is expected behavior
        let mean_coh: Float = coh[1..128].iter().sum::<Float>() / 127.0;
        assert!(
            mean_coh > 0.99,
            "Single-window coherence should be ~1.0 (known bias), got {}",
            mean_coh
        );
    }

    #[test]
    fn test_coherence_range() {
        // Coherence values must be in [0, 1]
        let mut sig_a = [0.0 as Float; 256];
        let mut sig_b = [0.0 as Float; 256];
        for (i, (a, b)) in sig_a.iter_mut().zip(sig_b.iter_mut()).enumerate() {
            let t = i as Float / 256.0;
            *a = float::sin(2.0 * PI * 10.0 * t) + float::sin(2.0 * PI * 30.0 * t);
            *b = float::sin(2.0 * PI * 10.0 * t) + float::cos(2.0 * PI * 50.0 * t);
        }

        let mut coh = [0.0 as Float; 129];
        coherence::<256>(&sig_a, &sig_b, WindowType::Hann, &mut coh);

        for (k, &c) in coh.iter().enumerate() {
            assert!(c >= 0.0, "Coherence at bin {} is negative: {}", k, c);
            assert!(c <= 1.0 + 1e-6, "Coherence at bin {} exceeds 1.0: {}", k, c);
        }
    }

    #[test]
    fn test_spectral_coherence_identical_signals() {
        let mut signal = [0.0 as Float; 1024];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as Float / 256.0;
            *s = float::sin(2.0 * PI * 10.0 * t);
        }

        let mut coh = [0.0 as Float; 129];
        let segments = spectral_coherence::<256>(&signal, &signal, 0.5, WindowType::Hann, &mut coh);

        assert!(segments > 1, "Should have multiple segments");
        assert!(
            coh[10] > 0.99,
            "Spectral coherence at 10 Hz should be ~1.0, got {}",
            coh[10]
        );
    }

    #[test]
    fn test_spectral_coherence_independent_noise() {
        let mut sig_a = [0.0 as Float; 2048];
        let mut sig_b = [0.0 as Float; 2048];
        let mut state_a: u32 = 42;
        let mut state_b: u32 = 99999;
        for i in 0..2048 {
            state_a = state_a.wrapping_mul(1103515245).wrapping_add(12345);
            state_b = state_b.wrapping_mul(1103515245).wrapping_add(12345);
            sig_a[i] = (state_a as Float / u32::MAX as Float) * 2.0 - 1.0;
            sig_b[i] = (state_b as Float / u32::MAX as Float) * 2.0 - 1.0;
        }

        let mut coh = [0.0 as Float; 129];
        let segments = spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        assert!(segments > 3);

        // With multiple segments, averaged coherence of independent noise should be low
        let mean_coh: Float = coh[1..128].iter().sum::<Float>() / 127.0;
        assert!(
            mean_coh < 0.3,
            "Mean spectral coherence of independent noise should be low, got {}",
            mean_coh
        );
    }

    #[test]
    fn test_spectral_coherence_range() {
        let mut sig_a = [0.0 as Float; 1024];
        let mut sig_b = [0.0 as Float; 1024];
        let mut state: u32 = 42;
        for i in 0..1024 {
            let t = i as Float / 256.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (state as Float / u32::MAX as Float) * 2.0 - 1.0;
            sig_a[i] = float::sin(2.0 * PI * 10.0 * t) + noise * 0.3;
            sig_b[i] = float::sin(2.0 * PI * 10.0 * t) + noise * 0.5;
        }

        let mut coh = [0.0 as Float; 129];
        spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        for (k, &c) in coh.iter().enumerate() {
            assert!(c >= 0.0, "Coherence at bin {} is negative: {}", k, c);
            assert!(c <= 1.0 + 1e-6, "Coherence at bin {} exceeds 1.0: {}", k, c);
        }
    }

    #[test]
    fn test_plv_constant_phase_difference() {
        // Constant phase difference → PLV = 1.0
        let phases_a: [Float; 100] = core::array::from_fn(|i| i as Float * 0.1);
        let phases_b: [Float; 100] = core::array::from_fn(|i| i as Float * 0.1 + 0.5);

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(
            (plv - 1.0).abs() < 1e-6,
            "PLV with constant phase difference should be 1.0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_zero_phase_difference() {
        let phases: [Float; 100] = core::array::from_fn(|i| i as Float * 0.1);
        let plv = phase_locking_value(&phases, &phases);
        assert!(
            (plv - 1.0).abs() < 1e-6,
            "PLV with zero phase difference should be 1.0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_random_phases() {
        // Uniformly distributed phase differences → PLV ≈ 0
        let mut phases_a = [0.0 as Float; 1000];
        let mut phases_b = [0.0 as Float; 1000];
        let mut state: u32 = 42;
        for i in 0..1000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            phases_a[i] = (state as Float / u32::MAX as Float) * 2.0 * PI;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            phases_b[i] = (state as Float / u32::MAX as Float) * 2.0 * PI;
        }

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(
            plv < 0.15,
            "PLV with random phases should be ~0, got {}",
            plv
        );
    }

    #[test]
    fn test_plv_range() {
        // PLV should always be in [0, 1]
        let phases_a: [Float; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        let phases_b: [Float; 5] = [0.5, 1.5, 0.0, 3.0, 5.0];

        let plv = phase_locking_value(&phases_a, &phases_b);
        assert!(plv >= 0.0, "PLV must be >= 0, got {}", plv);
        assert!(plv <= 1.0 + 1e-6, "PLV must be <= 1, got {}", plv);
    }

    #[test]
    #[should_panic(expected = "Phase arrays must have equal length")]
    fn test_plv_unequal_lengths() {
        let a = [0.0 as Float; 5];
        let b = [0.0 as Float; 3];
        phase_locking_value(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Phase arrays must not be empty")]
    fn test_plv_empty() {
        let a: [Float; 0] = [];
        let b: [Float; 0] = [];
        phase_locking_value(&a, &b);
    }

    #[test]
    fn test_coherence_frequencies() {
        let mut freqs = [0.0 as Float; 129];
        coherence_frequencies::<256>(256.0, &mut freqs);

        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - 1.0).abs() < 1e-6);
        assert!((freqs[128] - 128.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_frequencies_250hz() {
        let mut freqs = [0.0 as Float; 129];
        coherence_frequencies::<256>(250.0, &mut freqs);

        let freq_res = 250.0 / 256.0;
        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - freq_res).abs() < 1e-4);
        assert!((freqs[128] - 125.0).abs() < 1e-3);
    }

    #[test]
    fn test_spectral_coherence_single_segment() {
        // With exactly N samples, spectral coherence should equal single-window coherence
        let mut signal = [0.0 as Float; 256];
        for (i, s) in signal.iter_mut().enumerate() {
            let t = i as Float / 256.0;
            *s = float::sin(2.0 * PI * 10.0 * t);
        }

        let mut coh_single = [0.0 as Float; 129];
        coherence::<256>(&signal, &signal, WindowType::Hann, &mut coh_single);

        let mut coh_welch = [0.0 as Float; 129];
        let segments =
            spectral_coherence::<256>(&signal, &signal, 0.5, WindowType::Hann, &mut coh_welch);

        assert_eq!(segments, 1);

        for k in 0..129 {
            assert!(
                (coh_single[k] - coh_welch[k]).abs() < 1e-5,
                "Bin {}: single={} vs welch={}",
                k,
                coh_single[k],
                coh_welch[k]
            );
        }
    }

    #[test]
    fn test_coherence_shared_component() {
        // Two signals sharing a common component should have high coherence at that frequency
        let mut sig_a = [0.0 as Float; 1024];
        let mut sig_b = [0.0 as Float; 1024];
        let mut state: u32 = 42;
        for i in 0..1024 {
            let t = i as Float / 256.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise_a = (state as Float / u32::MAX as Float) * 2.0 - 1.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise_b = (state as Float / u32::MAX as Float) * 2.0 - 1.0;

            let shared = float::sin(2.0 * PI * 10.0 * t);
            sig_a[i] = shared + noise_a * 0.1;
            sig_b[i] = shared + noise_b * 0.1;
        }

        let mut coh = [0.0 as Float; 129];
        spectral_coherence::<256>(&sig_a, &sig_b, 0.5, WindowType::Hann, &mut coh);

        // High coherence at 10 Hz (bin 10)
        assert!(
            coh[10] > 0.8,
            "Coherence at shared frequency should be high, got {}",
            coh[10]
        );
    }

    // --- Granger Causality Tests ---

    /// Deterministic PRNG for test reproducibility (LCG).
    fn lcg_next(state: &mut u32) -> f64 {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        (*state as f64 / u32::MAX as f64) * 2.0 - 1.0
    }

    #[test]
    fn test_granger_known_causality() {
        // x is white noise, y(t) = 0.5*y(t-1) + 0.3*x(t-1) + small noise
        // x should Granger-cause y.
        let mut x = [0.0f64; 500];
        let mut y = [0.0f64; 500];
        let mut state: u32 = 42;

        for xi in x.iter_mut() {
            *xi = lcg_next(&mut state);
        }
        for t in 1..x.len() {
            let noise = lcg_next(&mut state);
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + noise * 0.1;
        }

        let result = granger_causality(&x, &y, 1);
        assert!(
            result.p_value < 0.01,
            "x should Granger-cause y, p={}",
            result.p_value
        );
        assert!(result.f_statistic > 0.0);
        assert!(result.unrestricted_variance < result.restricted_variance);
    }

    #[test]
    fn test_granger_no_causality() {
        // Independent AR(1) processes — should not show Granger causality.
        let mut x = [0.0f64; 1000];
        let mut y = [0.0f64; 1000];
        let mut state: u32 = 42;

        for t in 1..x.len() {
            let nx = lcg_next(&mut state);
            x[t] = 0.5 * x[t - 1] + nx * 0.5;
            let ny = lcg_next(&mut state);
            y[t] = 0.5 * y[t - 1] + ny * 0.5;
        }

        let result = granger_causality(&x, &y, 1);
        assert!(
            result.p_value > 0.01,
            "Independent signals should not show causality, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_granger_reverse_direction() {
        // If x causes y, then y→x should be much weaker than x→y.
        let mut x = [0.0f64; 500];
        let mut y = [0.0f64; 500];
        let mut state: u32 = 42;

        for xi in x.iter_mut() {
            *xi = lcg_next(&mut state);
        }
        for t in 1..x.len() {
            let noise = lcg_next(&mut state);
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + noise * 0.1;
        }

        let result_xy = granger_causality(&x, &y, 1);
        let result_yx = granger_causality(&y, &x, 1);

        // x→y should be far more significant than y→x
        assert!(
            result_xy.p_value < result_yx.p_value,
            "x→y (p={}) should be more significant than y→x (p={})",
            result_xy.p_value,
            result_yx.p_value
        );
    }

    #[test]
    fn test_granger_higher_order() {
        // x causes y with lag 3
        let mut x = [0.0f64; 600];
        let mut y = [0.0f64; 600];
        let mut state: u32 = 77;

        for xi in x.iter_mut() {
            *xi = lcg_next(&mut state);
        }
        for t in 3..x.len() {
            let noise = lcg_next(&mut state);
            y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 3] + noise * 0.1;
        }

        // Order 1 should miss the lag-3 effect
        let result_low = granger_causality(&x, &y, 1);
        // Order 3+ should capture it
        let result_high = granger_causality(&x, &y, 3);

        assert!(
            result_high.f_statistic > result_low.f_statistic,
            "Higher order should capture lag-3 effect: F(3)={} vs F(1)={}",
            result_high.f_statistic,
            result_low.f_statistic
        );
    }

    #[test]
    fn test_granger_p_value_range() {
        let mut x = [0.0f64; 200];
        let mut y = [0.0f64; 200];
        let mut state: u32 = 42;
        for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
            *xi = lcg_next(&mut state);
            *yi = lcg_next(&mut state);
        }

        let result = granger_causality(&x, &y, 5);
        assert!(result.p_value >= 0.0, "p-value must be >= 0");
        assert!(result.p_value <= 1.0, "p-value must be <= 1");
        assert!(result.f_statistic >= 0.0, "F-statistic must be >= 0");
    }

    #[test]
    #[should_panic(expected = "Order must be > 0")]
    fn test_granger_order_zero() {
        let x = [0.0f64; 100];
        let y = [0.0f64; 100];
        granger_causality(&x, &y, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds maximum")]
    fn test_granger_order_too_large() {
        let x = [0.0f64; 100];
        let y = [0.0f64; 100];
        granger_causality(&x, &y, 21);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn test_granger_signal_too_short() {
        let x = [0.0f64; 10];
        let y = [0.0f64; 10];
        granger_causality(&x, &y, 5);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_granger_unequal_lengths() {
        let x = [0.0f64; 100];
        let y = [0.0f64; 50];
        granger_causality(&x, &y, 1);
    }

    #[test]
    fn test_granger_significance_standalone() {
        // F(1, 100) critical value at alpha=0.05 is ~3.94
        let p = granger_significance(5.0, 106, 1);
        assert!(
            p < 0.05,
            "F=5.0 with df(1,100) should be significant, p={}",
            p
        );

        let p2 = granger_significance(1.0, 106, 1);
        assert!(
            p2 > 0.1,
            "F=1.0 with df(1,100) should not be significant, p={}",
            p2
        );
    }

    #[test]
    fn test_conditional_granger_direct_cause() {
        // x directly causes y, z is independent noise.
        // Conditional Granger (controlling for z) should still detect x→y.
        let n = 500;
        let mut x = [0.0f64; 500];
        let mut y = [0.0f64; 500];
        let mut z = [0.0f64; 500];
        let mut state: u32 = 42;

        for i in 0..n {
            x[i] = lcg_next(&mut state);
            z[i] = lcg_next(&mut state);
        }
        for t in 1..n {
            let noise = lcg_next(&mut state);
            y[t] = 0.5 * y[t - 1] + 0.3 * x[t - 1] + noise * 0.1;
        }

        let result = conditional_granger(&x, &y, &z, 1);
        assert!(
            result.p_value < 0.01,
            "x→y should still be significant controlling for z, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_conditional_granger_mediated() {
        // x→z→y: x causes z, z causes y. Conditioning on z should reduce
        // the apparent x→y causality.
        let mut x = [0.0f64; 500];
        let mut y = [0.0f64; 500];
        let mut z = [0.0f64; 500];
        let mut state: u32 = 42;

        for xi in x.iter_mut() {
            *xi = lcg_next(&mut state);
        }
        for t in 1..x.len() {
            let nz = lcg_next(&mut state);
            z[t] = 0.3 * z[t - 1] + 0.5 * x[t - 1] + nz * 0.1;
            let ny = lcg_next(&mut state);
            y[t] = 0.3 * y[t - 1] + 0.5 * z[t - 1] + ny * 0.1;
        }

        // Unconditional: x→y should be significant (indirect effect)
        let result_uncond = granger_causality(&x, &y, 2);
        assert!(
            result_uncond.p_value < 0.05,
            "Unconditional x→y should be significant, p={}",
            result_uncond.p_value
        );

        // Conditional on z: x→y should be weaker (z mediates the effect)
        let result_cond = conditional_granger(&x, &y, &z, 2);
        assert!(
            result_cond.p_value > result_uncond.p_value,
            "Conditional x→y|z (p={}) should be less significant than unconditional (p={})",
            result_cond.p_value,
            result_uncond.p_value
        );
    }

    #[test]
    fn test_incomplete_beta_known_values() {
        // I_0.5(1, 1) = 0.5 (uniform distribution)
        let val = regularized_incomplete_beta(0.5, 1.0, 1.0);
        assert!(
            libm::fabs(val - 0.5) < 1e-10,
            "I_0.5(1,1) should be 0.5, got {}",
            val
        );

        // I_0(a, b) = 0 for any a, b
        let val = regularized_incomplete_beta(0.0, 2.0, 3.0);
        assert!(libm::fabs(val) < 1e-10, "I_0(a,b) should be 0, got {}", val);

        // I_1(a, b) = 1 for any a, b
        let val = regularized_incomplete_beta(1.0, 2.0, 3.0);
        assert!(
            libm::fabs(val - 1.0) < 1e-10,
            "I_1(a,b) should be 1, got {}",
            val
        );

        // I_0.5(2, 2) = 0.5 (symmetric beta distribution)
        let val = regularized_incomplete_beta(0.5, 2.0, 2.0);
        assert!(
            libm::fabs(val - 0.5) < 1e-8,
            "I_0.5(2,2) should be 0.5, got {}",
            val
        );
    }

    #[test]
    fn test_f_distribution_sf_known_values() {
        // P(F > 0 | df1, df2) = 1.0 for any df
        let p = f_distribution_sf(0.0, 5.0, 100.0);
        assert!(
            libm::fabs(p - 1.0) < 1e-10,
            "P(F>0) should be 1.0, got {}",
            p
        );

        // F(1, inf) ~ chi-squared(1). P(chi2 > 3.84) ≈ 0.05
        // For F(1, 1000), P(F > 3.84) should be close to 0.05
        let p = f_distribution_sf(3.84, 1.0, 1000.0);
        assert!(
            libm::fabs(p - 0.05) < 0.01,
            "P(F(1,1000) > 3.84) should be ~0.05, got {}",
            p
        );

        // Very large F → p ≈ 0
        let p = f_distribution_sf(100.0, 5.0, 100.0);
        assert!(p < 1e-10, "Very large F should give tiny p, got {}", p);
    }

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        let mut a = [4.0, 2.0, 2.0, 3.0];
        assert!(cholesky_inplace(&mut a, 2));
        assert!(libm::fabs(a[0] - 2.0) < 1e-10);
        assert!(libm::fabs(a[2] - 1.0) < 1e-10);
        assert!(libm::fabs(a[3] - libm::sqrt(2.0)) < 1e-10);
    }

    #[test]
    fn test_cholesky_solve_identity() {
        // Solve I * x = b → x = b
        let l = [1.0, 0.0, 0.0, 1.0]; // Identity (already Cholesky factor)
        let b = [3.0, 7.0];
        let mut x = [0.0; GRANGER_MAX_DIM];
        cholesky_solve(&l, &b, 2, &mut x);
        assert!(libm::fabs(x[0] - 3.0) < 1e-10);
        assert!(libm::fabs(x[1] - 7.0) < 1e-10);
    }
}
