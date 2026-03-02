//! Independent Component Analysis (FastICA) for artifact removal.
//!
//! Implements the symmetric (parallel) FastICA algorithm (Hyvarinen 1999) for
//! blind source separation. Finds statistically independent source signals from
//! mixed observations by maximizing non-Gaussianity via fixed-point iteration.
//!
//! The symmetric variant extracts all components simultaneously, avoiding error
//! accumulation that occurs with deflation approaches.
//!
//! # Typical BCI Pipeline
//!
//! 1. Record multi-channel EEG
//! 2. Fit ICA to decompose into independent components
//! 3. Identify artifact components (eye blinks, muscle, heartbeat)
//! 4. Remove artifact components via `remove_components`
//! 5. Continue with clean signal
//!
//! # Example
//!
//! ```
//! use zerostone::ica::{ContrastFunction, Ica};
//!
//! // 4-channel ICA
//! let mut ica: Ica<4, 16> = Ica::new(ContrastFunction::LogCosh);
//!
//! // Generate some mixed data (4 channels, 1000 samples)
//! let data: Vec<[f64; 4]> = (0..1000)
//!     .map(|t| {
//!         let t = t as f64 / 1000.0;
//!         [
//!             libm::sin(2.0 * core::f64::consts::PI * 5.0 * t) + 0.5 * libm::cos(2.0 * core::f64::consts::PI * 11.0 * t),
//!             0.7 * libm::sin(2.0 * core::f64::consts::PI * 5.0 * t) + libm::cos(2.0 * core::f64::consts::PI * 11.0 * t),
//!             libm::sin(2.0 * core::f64::consts::PI * 5.0 * t) + 0.3 * libm::cos(2.0 * core::f64::consts::PI * 11.0 * t),
//!             0.4 * libm::sin(2.0 * core::f64::consts::PI * 5.0 * t) + 0.9 * libm::cos(2.0 * core::f64::consts::PI * 11.0 * t),
//!         ]
//!     })
//!     .collect();
//!
//! ica.fit(&data, 200, 1e-4).unwrap();
//!
//! // Transform to independent components
//! let mut sources = vec![[0.0; 4]; data.len()];
//! ica.transform(&data, &mut sources).unwrap();
//!
//! // Remove component 0 (e.g., identified as eye blink artifact)
//! let mut cleaned = vec![[0.0; 4]; data.len()];
//! ica.remove_components(&data, &[0], &mut cleaned).unwrap();
//! ```

use crate::linalg::Matrix;
use crate::riemannian::reconstruct_from_eigen;
use libm;

/// Errors specific to ICA operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IcaError {
    /// Not enough data samples (need >= 2 * C)
    InsufficientData,
    /// ICA model has not been fitted yet
    NotFitted,
    /// Fixed-point iteration did not converge
    ConvergenceFailed,
    /// Eigenvalue decomposition failed
    EigenFailed,
    /// Numerical instability (e.g., NaN or Inf detected)
    NumericalInstability,
}

/// Contrast (nonlinearity) function for FastICA.
///
/// Controls how non-Gaussianity is measured. LogCosh is the default and
/// provides a good balance of robustness and performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContrastFunction {
    /// g(u) = tanh(u), robust and general-purpose (default)
    LogCosh,
    /// g(u) = u * exp(-u^2/2), good for super-Gaussian sources
    Exp,
    /// g(u) = u^3, fast but less robust to outliers
    Cube,
}

/// Apply contrast function, returning (g(u), g'(u)).
#[inline]
fn apply_contrast(u: f64, contrast: ContrastFunction) -> (f64, f64) {
    match contrast {
        ContrastFunction::LogCosh => {
            let t = libm::tanh(u);
            (t, 1.0 - t * t)
        }
        ContrastFunction::Exp => {
            let e = libm::exp(-u * u / 2.0);
            (u * e, (1.0 - u * u) * e)
        }
        ContrastFunction::Cube => (u * u * u, 3.0 * u * u),
    }
}

/// Matrix-vector multiply: result = mat * v.
#[inline]
#[allow(clippy::needless_range_loop)]
fn matvec<const C: usize, const M: usize>(mat: &Matrix<C, M>, v: &[f64; C]) -> [f64; C] {
    let mut result = [0.0f64; C];
    for i in 0..C {
        let mut sum = 0.0;
        for j in 0..C {
            sum += mat.get(i, j) * v[j];
        }
        result[i] = sum;
    }
    result
}

/// Symmetric decorrelation: W_new = (W * W^T)^{-1/2} * W.
fn sym_decorrelate<const C: usize, const M: usize>(
    w: &Matrix<C, M>,
) -> Result<Matrix<C, M>, IcaError> {
    // Compute W * W^T
    let wt = w.transpose();
    let wwt = w.matmul(&wt);

    // Eigendecompose W * W^T
    let eigen = wwt
        .eigen_symmetric(50, 1e-12)
        .map_err(|_| IcaError::EigenFailed)?;

    // Compute (W * W^T)^{-1/2}: eigenvalues -> 1/sqrt(eigenvalue)
    let mut inv_sqrt_eigs = eigen.eigenvalues;
    for val in &mut inv_sqrt_eigs {
        if *val <= 1e-15 {
            return Err(IcaError::NumericalInstability);
        }
        *val = 1.0 / libm::sqrt(*val);
    }

    // Reconstruct (W * W^T)^{-1/2}
    let inv_sqrt_wwt = reconstruct_from_eigen(&eigen.eigenvectors, &inv_sqrt_eigs)
        .map_err(|_| IcaError::EigenFailed)?;

    // Return (W * W^T)^{-1/2} * W
    Ok(inv_sqrt_wwt.matmul(w))
}

/// Check convergence: max |abs(diag(W_new * W_old^T)) - 1| < tol.
fn check_convergence<const C: usize, const M: usize>(
    w_new: &Matrix<C, M>,
    w_old: &Matrix<C, M>,
    tol: f64,
) -> bool {
    let w_old_t = w_old.transpose();
    let product = w_new.matmul(&w_old_t);

    let mut max_diff = 0.0f64;
    for i in 0..C {
        let diag_val = product.get(i, i);
        let diff = libm::fabs(libm::fabs(diag_val) - 1.0);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff < tol
}

/// Independent Component Analysis (FastICA, symmetric/parallel).
///
/// Decomposes multi-channel signals into statistically independent sources.
/// The primary use case is artifact removal in EEG: decompose, identify
/// artifact components, then reconstruct without them.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Total matrix elements (M = C * C)
pub struct Ica<const C: usize, const M: usize> {
    /// Unmixing matrix in whitened space (C x C)
    w: Matrix<C, M>,
    /// Whitening matrix K (C x C): z = K * (x - mean)
    k: Matrix<C, M>,
    /// Full unmixing matrix (C x C): s = unmixing * (x - mean)
    unmixing: Matrix<C, M>,
    /// Mixing matrix (C x C): x = mixing * s + mean
    mixing: Matrix<C, M>,
    /// Per-channel mean
    mean: [f64; C],
    /// Whether the model has been fitted
    fitted: bool,
    /// Contrast function
    contrast: ContrastFunction,
}

impl<const C: usize, const M: usize> Ica<C, M> {
    /// Compile-time assertion that M = C * C
    const _ASSERT_M: () = assert!(M == C * C, "M must equal C * C");

    /// Create a new ICA decomposer.
    ///
    /// # Arguments
    ///
    /// * `contrast` - Contrast function for measuring non-Gaussianity
    pub fn new(contrast: ContrastFunction) -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_M;

        Self {
            w: Matrix::zeros(),
            k: Matrix::zeros(),
            unmixing: Matrix::zeros(),
            mixing: Matrix::zeros(),
            mean: [0.0; C],
            fitted: false,
            contrast,
        }
    }

    /// Fit the ICA model to data.
    ///
    /// Extracts independent components from multi-channel observations using
    /// symmetric FastICA with the configured contrast function.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of samples, each with C channels. Need >= 2*C samples.
    /// * `max_iter` - Maximum number of fixed-point iterations (typical: 200)
    /// * `tolerance` - Convergence tolerance (typical: 1e-4)
    ///
    /// # Errors
    ///
    /// Returns `IcaError::InsufficientData` if `data.len() < 2 * C`.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(
        &mut self,
        data: &[[f64; C]],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<(), IcaError> {
        let n = data.len();
        if n < 2 * C {
            return Err(IcaError::InsufficientData);
        }

        // Step 1: Compute mean
        let mut mean = [0.0f64; C];
        for sample in data {
            for j in 0..C {
                mean[j] += sample[j];
            }
        }
        let inv_n = 1.0 / n as f64;
        for j in 0..C {
            mean[j] *= inv_n;
        }

        // Step 2: Compute covariance matrix
        let mut cov = Matrix::<C, M>::zeros();
        for sample in data {
            for i in 0..C {
                let xi = sample[i] - mean[i];
                for j in i..C {
                    let xj = sample[j] - mean[j];
                    let old = cov.get(i, j);
                    cov.set(i, j, old + xi * xj);
                }
            }
        }
        // Normalize and symmetrize
        for i in 0..C {
            for j in i..C {
                let val = cov.get(i, j) * inv_n;
                cov.set(i, j, val);
                if i != j {
                    cov.set(j, i, val);
                }
            }
        }
        // Add regularization to diagonal
        for i in 0..C {
            let val = cov.get(i, i);
            cov.set(i, i, val + 1e-8);
        }

        // Step 3: Eigendecompose covariance for whitening
        let eigen = cov
            .eigen_symmetric(50, 1e-12)
            .map_err(|_| IcaError::EigenFailed)?;

        // Find max eigenvalue for clipping threshold
        let mut max_eig = 0.0f64;
        for i in 0..C {
            if eigen.eigenvalues[i] > max_eig {
                max_eig = eigen.eigenvalues[i];
            }
        }
        let eig_floor = max_eig * 1e-12;

        // Compute whitening K = D^{-1/2} * E^T and de-whitening K_inv = E * D^{1/2}
        let mut d_inv_sqrt = [0.0f64; C];
        let mut d_sqrt = [0.0f64; C];
        for i in 0..C {
            let eig = if eigen.eigenvalues[i] < eig_floor {
                eig_floor
            } else {
                eigen.eigenvalues[i]
            };
            d_inv_sqrt[i] = 1.0 / libm::sqrt(eig);
            d_sqrt[i] = libm::sqrt(eig);
        }

        // K = D^{-1/2} * E^T  (whitening matrix)
        // K[i,j] = d_inv_sqrt[i] * E[j,i]  (E^T means we swap indices)
        let mut k = Matrix::<C, M>::zeros();
        for i in 0..C {
            for j in 0..C {
                k.set(i, j, d_inv_sqrt[i] * eigen.eigenvectors.get(j, i));
            }
        }

        // K_inv = E * D^{1/2}  (de-whitening matrix)
        // K_inv[i,j] = E[i,j] * d_sqrt[j]
        let mut k_inv = Matrix::<C, M>::zeros();
        for i in 0..C {
            for j in 0..C {
                k_inv.set(i, j, eigen.eigenvectors.get(i, j) * d_sqrt[j]);
            }
        }

        // Step 4: Initialize W = Identity (PCA starting point)
        let mut w = Matrix::<C, M>::identity();

        // Step 5: Fixed-point iteration
        let contrast = self.contrast;
        for _iter in 0..max_iter {
            let w_old = w;

            // Accumulate G and gp across all samples
            // G[p][j] = (1/T) * sum_t g(y_p_t) * z_j_t
            // gp[p]   = (1/T) * sum_t g'(y_p_t)
            let mut g_mat = Matrix::<C, M>::zeros();
            let mut gp = [0.0f64; C];

            for sample in data.iter() {
                // Whiten on-the-fly: z = K * (x - mean)
                let mut centered = [0.0f64; C];
                for j in 0..C {
                    centered[j] = sample[j] - mean[j];
                }
                let z = matvec(&k, &centered);

                // y = W * z
                let y = matvec(&w_old, &z);

                // Apply contrast and accumulate
                for p in 0..C {
                    let (g_val, gp_val) = apply_contrast(y[p], contrast);
                    gp[p] += gp_val;
                    for j in 0..C {
                        let old = g_mat.get(p, j);
                        g_mat.set(p, j, old + g_val * z[j]);
                    }
                }
            }

            // Normalize: W_new = G/T - diag(gp/T) * W_old
            let mut w_new = Matrix::<C, M>::zeros();
            for p in 0..C {
                let gp_avg = gp[p] * inv_n;
                for j in 0..C {
                    let g_avg = g_mat.get(p, j) * inv_n;
                    let w_old_val = w_old.get(p, j);
                    w_new.set(p, j, g_avg - gp_avg * w_old_val);
                }
            }

            // Symmetric decorrelation
            w = sym_decorrelate(&w_new)?;

            // Check for NaN
            if w.get(0, 0).is_nan() {
                return Err(IcaError::NumericalInstability);
            }

            // Check convergence
            if check_convergence(&w, &w_old, tolerance) {
                // Store results
                self.w = w;
                self.k = k;
                self.mean = mean;

                // unmixing = W * K
                self.unmixing = w.matmul(&k);

                // mixing = K_inv * W^T
                let wt = w.transpose();
                self.mixing = k_inv.matmul(&wt);

                self.fitted = true;
                return Ok(());
            }
        }

        // Did not converge within max_iter, but store best result anyway
        self.w = w;
        self.k = k;
        self.mean = mean;
        self.unmixing = w.matmul(&k);
        let wt = w.transpose();
        self.mixing = k_inv.matmul(&wt);
        self.fitted = true;

        Err(IcaError::ConvergenceFailed)
    }

    /// Transform data to independent components.
    ///
    /// Computes s = unmixing * (x - mean) for each sample.
    ///
    /// # Arguments
    ///
    /// * `data` - Input samples (T x C)
    /// * `output` - Output buffer for independent components (T x C)
    pub fn transform(&self, data: &[[f64; C]], output: &mut [[f64; C]]) -> Result<(), IcaError> {
        if !self.fitted {
            return Err(IcaError::NotFitted);
        }
        for (i, sample) in data.iter().enumerate() {
            let mut centered = [0.0f64; C];
            for j in 0..C {
                centered[j] = sample[j] - self.mean[j];
            }
            output[i] = matvec(&self.unmixing, &centered);
        }
        Ok(())
    }

    /// Reconstruct data from independent components.
    ///
    /// Computes x = mixing * s + mean for each sample.
    ///
    /// # Arguments
    ///
    /// * `sources` - Independent component samples (T x C)
    /// * `output` - Output buffer for reconstructed data (T x C)
    pub fn inverse_transform(
        &self,
        sources: &[[f64; C]],
        output: &mut [[f64; C]],
    ) -> Result<(), IcaError> {
        if !self.fitted {
            return Err(IcaError::NotFitted);
        }
        for (i, source) in sources.iter().enumerate() {
            let reconstructed = matvec(&self.mixing, source);
            for j in 0..C {
                output[i][j] = reconstructed[j] + self.mean[j];
            }
        }
        Ok(())
    }

    /// Remove specified components and reconstruct.
    ///
    /// Transform to IC space, zero out excluded components, then inverse transform.
    ///
    /// # Arguments
    ///
    /// * `data` - Input samples (T x C)
    /// * `exclude` - Indices of components to remove
    /// * `output` - Output buffer for cleaned data (T x C)
    pub fn remove_components(
        &self,
        data: &[[f64; C]],
        exclude: &[usize],
        output: &mut [[f64; C]],
    ) -> Result<(), IcaError> {
        if !self.fitted {
            return Err(IcaError::NotFitted);
        }
        for (i, sample) in data.iter().enumerate() {
            // Transform to IC space
            let mut centered = [0.0f64; C];
            for j in 0..C {
                centered[j] = sample[j] - self.mean[j];
            }
            let mut sources = matvec(&self.unmixing, &centered);

            // Zero excluded components
            for &idx in exclude {
                if idx < C {
                    sources[idx] = 0.0;
                }
            }

            // Inverse transform
            let reconstructed = matvec(&self.mixing, &sources);
            for j in 0..C {
                output[i][j] = reconstructed[j] + self.mean[j];
            }
        }
        Ok(())
    }

    /// Get the mixing matrix A (C x C).
    ///
    /// x = A * s + mean
    pub fn mixing_matrix(&self) -> &Matrix<C, M> {
        &self.mixing
    }

    /// Get the unmixing matrix W_full (C x C).
    ///
    /// s = W_full * (x - mean)
    pub fn unmixing_matrix(&self) -> &Matrix<C, M> {
        &self.unmixing
    }

    /// Get the per-channel mean.
    pub fn mean(&self) -> &[f64; C] {
        &self.mean
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f64::consts::PI;

    /// Simple LCG pseudo-random number generator for no_std tests.
    struct Rng {
        state: u64,
    }

    impl Rng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            // LCG parameters from Numerical Recipes
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [0, 1)
            (self.state >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Approximate normal via Box-Muller
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-15);
            let u2 = self.next_f64();
            libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
        }
    }

    /// Generate a simple signal: sine wave
    fn sine_wave(n: usize, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|t| libm::sin(2.0 * PI * freq * t as f64 / n as f64))
            .collect()
    }

    /// Generate a square wave
    fn square_wave(n: usize, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let phase = freq * t as f64 / n as f64;
                if phase - libm::floor(phase) < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    /// Generate a sawtooth wave
    fn sawtooth_wave(n: usize, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let phase = freq * t as f64 / n as f64;
                2.0 * (phase - libm::floor(phase)) - 1.0
            })
            .collect()
    }

    /// Compute Pearson correlation between two signals (absolute value).
    fn abs_correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let mean_a: f64 = a.iter().sum::<f64>() / n;
        let mean_b: f64 = b.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        for i in 0..a.len() {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        if var_a < 1e-15 || var_b < 1e-15 {
            return 0.0;
        }
        libm::fabs(cov / libm::sqrt(var_a * var_b))
    }

    /// Find the best match correlation between sources and recovered signals.
    /// Returns the minimum best-match correlation.
    fn best_match_correlation(sources: &[Vec<f64>], recovered: &[Vec<f64>]) -> f64 {
        let n_sources = sources.len();
        let mut min_corr = f64::MAX;

        // For each source, find the recovered signal with highest correlation
        for src in sources {
            let mut best = 0.0f64;
            for rec in recovered.iter().take(n_sources) {
                let c = abs_correlation(src, rec);
                if c > best {
                    best = c;
                }
            }
            if best < min_corr {
                min_corr = best;
            }
        }
        min_corr
    }

    #[test]
    fn test_2_component_cocktail_party() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);

        // Mixing matrix
        let a = [[0.8, 0.6], [0.3, 0.9]];

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| {
                [
                    a[0][0] * s1[t] + a[0][1] * s2[t],
                    a[1][0] * s1[t] + a[1][1] * s2[t],
                ]
            })
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut sources = vec![[0.0; 2]; n];
        ica.transform(&data, &mut sources).unwrap();

        // Extract recovered signals
        let rec0: Vec<f64> = sources.iter().map(|s| s[0]).collect();
        let rec1: Vec<f64> = sources.iter().map(|s| s[1]).collect();

        let min_corr = best_match_correlation(&[s1, s2], &[rec0, rec1]);
        assert!(
            min_corr > 0.9,
            "2-component separation: min correlation = {}, expected > 0.9",
            min_corr
        );
    }

    #[test]
    fn test_4_component_separation() {
        let n = 5000;
        let s1 = sine_wave(n, 2.0);
        let s2 = square_wave(n, 5.0);
        let s3 = sawtooth_wave(n, 3.0);
        let s4 = sine_wave(n, 11.0); // Use another non-Gaussian source instead of noise

        // 4x4 mixing matrix (well-conditioned)
        let a = [
            [0.8, 0.2, 0.4, 0.1],
            [0.1, 0.9, 0.1, 0.3],
            [0.3, 0.1, 0.8, 0.2],
            [0.2, 0.4, 0.2, 0.9],
        ];

        let data: Vec<[f64; 4]> = (0..n)
            .map(|t| {
                [
                    a[0][0] * s1[t] + a[0][1] * s2[t] + a[0][2] * s3[t] + a[0][3] * s4[t],
                    a[1][0] * s1[t] + a[1][1] * s2[t] + a[1][2] * s3[t] + a[1][3] * s4[t],
                    a[2][0] * s1[t] + a[2][1] * s2[t] + a[2][2] * s3[t] + a[2][3] * s4[t],
                    a[3][0] * s1[t] + a[3][1] * s2[t] + a[3][2] * s3[t] + a[3][3] * s4[t],
                ]
            })
            .collect();

        let mut ica: Ica<4, 16> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut sources = vec![[0.0; 4]; n];
        ica.transform(&data, &mut sources).unwrap();

        let recovered: Vec<Vec<f64>> = (0..4)
            .map(|c| sources.iter().map(|s| s[c]).collect())
            .collect();

        let min_corr = best_match_correlation(&[s1, s2, s3, s4], &recovered);
        assert!(
            min_corr > 0.85,
            "4-component separation: min correlation = {}, expected > 0.85",
            min_corr
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_whitening_produces_identity_covariance() {
        let n = 2000;
        let mut rng = Rng::new(123);

        // Generate correlated data
        let data: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let y = rng.next_normal();
                let z = rng.next_normal();
                [x + 0.5 * y, y + 0.3 * z, z + 0.2 * x]
            })
            .collect();

        let mut ica: Ica<3, 9> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        // Whiten all data using K * (x - mean)
        let whitened: Vec<[f64; 3]> = data
            .iter()
            .map(|x| {
                let mut centered = [0.0; 3];
                for j in 0..3 {
                    centered[j] = x[j] - ica.mean()[j];
                }
                matvec(&ica.k, &centered)
            })
            .collect();

        // Compute covariance of whitened data
        let mut cov = [[0.0f64; 3]; 3];
        let inv_n = 1.0 / n as f64;
        for z in &whitened {
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += z[i] * z[j];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] *= inv_n;
            }
        }

        // Should be close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    libm::fabs(cov[i][j] - expected) < 0.1,
                    "Whitened covariance [{},{}] = {}, expected {}",
                    i,
                    j,
                    cov[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_w_is_orthonormal() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);
        let s3 = sawtooth_wave(n, 5.0);

        let a = [[0.7, 0.3, 0.5], [0.2, 0.8, 0.1], [0.4, 0.1, 0.7]];

        let data: Vec<[f64; 3]> = (0..n)
            .map(|t| {
                [
                    a[0][0] * s1[t] + a[0][1] * s2[t] + a[0][2] * s3[t],
                    a[1][0] * s1[t] + a[1][1] * s2[t] + a[1][2] * s3[t],
                    a[2][0] * s1[t] + a[2][1] * s2[t] + a[2][2] * s3[t],
                ]
            })
            .collect();

        let mut ica: Ica<3, 9> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        // Check W * W^T ~ I
        let wt = ica.w.transpose();
        let wwt = ica.w.matmul(&wt);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    libm::fabs(wwt.get(i, j) - expected) < 1e-6,
                    "W*W^T [{},{}] = {}, expected {}",
                    i,
                    j,
                    wwt.get(i, j),
                    expected
                );
            }
        }
    }

    #[test]
    fn test_round_trip_transform() {
        let n = 1000;
        let mut rng = Rng::new(99);

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| {
                let t_f = t as f64 / n as f64;
                [
                    libm::sin(2.0 * PI * 5.0 * t_f) + 0.1 * rng.next_normal(),
                    libm::cos(2.0 * PI * 3.0 * t_f) + 0.1 * rng.next_normal(),
                ]
            })
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut sources = vec![[0.0; 2]; n];
        ica.transform(&data, &mut sources).unwrap();

        let mut reconstructed = vec![[0.0; 2]; n];
        ica.inverse_transform(&sources, &mut reconstructed).unwrap();

        for t in 0..n {
            for j in 0..2 {
                assert!(
                    libm::fabs(reconstructed[t][j] - data[t][j]) < 1e-10,
                    "Round trip error at t={}, ch={}: got {}, expected {}",
                    t,
                    j,
                    reconstructed[t][j],
                    data[t][j]
                );
            }
        }
    }

    #[test]
    fn test_insufficient_data_error() {
        let data = [[1.0, 2.0, 3.0, 4.0]; 5]; // 5 < 2*4 = 8
        let mut ica: Ica<4, 16> = Ica::new(ContrastFunction::LogCosh);
        let result = ica.fit(&data, 200, 1e-4);
        assert_eq!(result, Err(IcaError::InsufficientData));
    }

    #[test]
    fn test_not_fitted_error() {
        let ica: Ica<2, 4> = Ica::new(ContrastFunction::LogCosh);
        let data = [[1.0, 2.0]; 10];
        let mut output = [[0.0; 2]; 10];
        assert_eq!(ica.transform(&data, &mut output), Err(IcaError::NotFitted));
    }

    #[test]
    fn test_convergence_within_100_iterations() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| [0.8 * s1[t] + 0.6 * s2[t], 0.3 * s1[t] + 0.9 * s2[t]])
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::LogCosh);
        // Use 100 as max_iter -- should converge
        let result = ica.fit(&data, 100, 1e-4);
        assert!(result.is_ok(), "Should converge within 100 iterations");
        assert!(ica.is_fitted());
    }

    #[test]
    fn test_contrast_exp() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| [0.8 * s1[t] + 0.6 * s2[t], 0.3 * s1[t] + 0.9 * s2[t]])
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::Exp);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut sources = vec![[0.0; 2]; n];
        ica.transform(&data, &mut sources).unwrap();

        let rec0: Vec<f64> = sources.iter().map(|s| s[0]).collect();
        let rec1: Vec<f64> = sources.iter().map(|s| s[1]).collect();

        let min_corr = best_match_correlation(&[s1, s2], &[rec0, rec1]);
        assert!(
            min_corr > 0.85,
            "Exp contrast: min correlation = {}",
            min_corr
        );
    }

    #[test]
    fn test_contrast_cube() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| [0.8 * s1[t] + 0.6 * s2[t], 0.3 * s1[t] + 0.9 * s2[t]])
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::Cube);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut sources = vec![[0.0; 2]; n];
        ica.transform(&data, &mut sources).unwrap();

        let rec0: Vec<f64> = sources.iter().map(|s| s[0]).collect();
        let rec1: Vec<f64> = sources.iter().map(|s| s[1]).collect();

        let min_corr = best_match_correlation(&[s1, s2], &[rec0, rec1]);
        assert!(
            min_corr > 0.85,
            "Cube contrast: min correlation = {}",
            min_corr
        );
    }

    #[test]
    fn test_remove_components() {
        let n = 2000;
        let s1 = sine_wave(n, 3.0);
        let s2 = square_wave(n, 7.0);

        let data: Vec<[f64; 2]> = (0..n)
            .map(|t| [0.8 * s1[t] + 0.6 * s2[t], 0.3 * s1[t] + 0.9 * s2[t]])
            .collect();

        let mut ica: Ica<2, 4> = Ica::new(ContrastFunction::LogCosh);
        ica.fit(&data, 200, 1e-4).unwrap();

        let mut cleaned = vec![[0.0; 2]; n];
        ica.remove_components(&data, &[0], &mut cleaned).unwrap();

        // Cleaned output should differ from original
        let mut total_diff = 0.0;
        for t in 0..n {
            for j in 0..2 {
                total_diff += libm::fabs(cleaned[t][j] - data[t][j]);
            }
        }
        assert!(
            total_diff > 1.0,
            "Removing a component should change the signal, diff = {}",
            total_diff
        );
    }
}
