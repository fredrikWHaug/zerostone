//! Spatial whitening (ZCA/PCA) for multi-channel neural recordings.
//!
//! Whitening (or sphering) decorrelates multi-channel signals and normalizes
//! their variance, producing data with identity covariance. This is a critical
//! preprocessing step for spike sorting pipelines (Kilosort, MountainSort) and
//! ICA-based source separation.
//!
//! Two whitening modes are supported:
//!
//! - **ZCA** (Zero-phase Component Analysis): `W = C^(-1/2)`. Preserves the
//!   spatial orientation of the original data -- whitened signals remain
//!   interpretable as channels. Preferred when channel identity matters.
//! - **PCA**: `W = D^(-1/2) E^T`. Rotates data into the principal component
//!   basis before scaling. Useful when dimensionality reduction follows.
//!
//! Both modes add a regularization term `epsilon` to eigenvalues before
//! inversion, ensuring numerical stability for ill-conditioned or singular
//! covariance matrices.
//!
//! # Algorithm
//!
//! Given a C x C covariance matrix with eigendecomposition `Cov = E D E^T`:
//! - ZCA: `W = E D_reg^(-1/2) E^T`   (symmetric, preserves channel orientation)
//! - PCA: `W = D_reg^(-1/2) E^T`      (rotates into eigenbasis)
//!
//! where `D_reg = D + epsilon * I`.
//!
//! # References
//!
//! - Kilosort (Pachitariu et al.): local ZCA whitening with 32-channel
//!   neighborhoods, `epsilon` added to eigenvalue diagonal.
//! - SpikeInterface: `spre.whiten()` for full or local whitening.
//! - Kessy, Lewin & Strimmer (2018): "Optimal Whitening and Decorrelation",
//!   *The American Statistician*.
//!
//! # Example
//!
//! ```
//! use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
//!
//! // 2-channel covariance matrix: correlated channels
//! let cov = [[4.0, 2.0],
//!            [2.0, 3.0]];
//!
//! let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6).unwrap();
//! let sample = [1.0, 0.5];
//! let whitened = wm.apply(&sample);
//!
//! // Whitened output is decorrelated and variance-normalized
//! assert!(whitened[0].is_finite());
//! assert!(whitened[1].is_finite());
//! ```

use crate::linalg::{LinalgError, Matrix};

/// Whitening mode selection.
///
/// # Examples
///
/// ```
/// use zerostone::whitening::WhiteningMode;
///
/// let mode = WhiteningMode::Zca;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhiteningMode {
    /// Zero-phase Component Analysis whitening.
    ///
    /// `W = E D^(-1/2) E^T` -- the whitened data stays as close as possible
    /// (in Frobenius norm) to the original data. Channel identity is preserved.
    Zca,

    /// PCA whitening.
    ///
    /// `W = D^(-1/2) E^T` -- the data is rotated into the principal component
    /// basis and scaled. Useful as a precursor to dimensionality reduction.
    Pca,
}

/// Precomputed whitening matrix for C-channel data.
///
/// Stores a C x C transformation matrix computed from the data covariance.
/// Apply it to each multi-channel sample via matrix-vector multiplication.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Total matrix elements (must equal C * C)
///
/// # Examples
///
/// ```
/// use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
///
/// // Identity covariance -- whitening should be near-identity
/// let cov = [[1.0, 0.0],
///            [0.0, 1.0]];
/// let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6).unwrap();
/// let out = wm.apply(&[3.0, 7.0]);
/// assert!((out[0] - 3.0).abs() < 0.01);
/// assert!((out[1] - 7.0).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WhiteningMatrix<const C: usize, const M: usize> {
    /// The C x C whitening transform matrix
    matrix: Matrix<C, M>,
}

/// Compile-time assertion helper.
const fn assert_m_eq_c_squared(c: usize, m: usize) {
    assert!(m == c * c, "M must equal C * C");
}

impl<const C: usize, const M: usize> WhiteningMatrix<C, M> {
    /// Compile-time check that M = C * C.
    const _ASSERT_M: () = assert_m_eq_c_squared(C, M);

    /// Compute a whitening matrix from a C x C covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `cov` - Symmetric positive semi-definite covariance matrix, stored
    ///   as `[[f64; C]; C]` in row-major order.
    /// * `mode` - [`WhiteningMode::Zca`] or [`WhiteningMode::Pca`].
    /// * `epsilon` - Regularization added to each eigenvalue before inversion.
    ///   Prevents division by zero for rank-deficient matrices. Typical values:
    ///   `1e-6` for well-conditioned data, `1e-3` for noisy recordings.
    ///
    /// # Returns
    ///
    /// A `WhiteningMatrix` that can be applied to individual samples.
    ///
    /// # Errors
    ///
    /// Returns [`LinalgError::ConvergenceFailed`] if the Jacobi eigensolver
    /// does not converge within 50 sweeps.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
    ///
    /// let cov = [[2.0, 1.0, 0.0],
    ///            [1.0, 2.0, 1.0],
    ///            [0.0, 1.0, 2.0]];
    /// let wm = WhiteningMatrix::<3, 9>::from_covariance(
    ///     &cov, WhiteningMode::Zca, 1e-6,
    /// ).unwrap();
    /// let out = wm.apply(&[1.0, 2.0, 3.0]);
    /// assert!(out[0].is_finite());
    /// ```
    pub fn from_covariance(
        cov: &[[f64; C]; C],
        mode: WhiteningMode,
        epsilon: f64,
    ) -> Result<Self, LinalgError> {
        // Trigger compile-time assertion
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_M;

        // Convert [[f64; C]; C] to Matrix<C, M>
        let mut mat = Matrix::<C, M>::zeros();
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            for j in 0..C {
                mat.set(i, j, cov[i][j]);
            }
        }

        // Eigendecomposition: Cov = E D E^T
        let eigen = mat.eigen_symmetric(50, 1e-12)?;

        // Build regularized inverse-square-root eigenvalues
        let mut inv_sqrt_eig = [0.0f64; C];
        #[allow(clippy::needless_range_loop)]
        for k in 0..C {
            let lambda = eigen.eigenvalues[k] + epsilon;
            // After regularization, lambda should be > 0 since epsilon > 0
            // but guard against pathological negative eigenvalues from numerical noise
            let clamped = if lambda > epsilon { lambda } else { epsilon };
            inv_sqrt_eig[k] = 1.0 / libm::sqrt(clamped);
        }

        let w = match mode {
            WhiteningMode::Zca => {
                // W = E * diag(inv_sqrt_eig) * E^T
                Self::reconstruct_symmetric(&eigen.eigenvectors, &inv_sqrt_eig)
            }
            WhiteningMode::Pca => {
                // W = diag(inv_sqrt_eig) * E^T
                Self::build_pca_matrix(&eigen.eigenvectors, &inv_sqrt_eig)
            }
        };

        Ok(Self { matrix: w })
    }

    /// Apply the whitening transform to a single multi-channel sample.
    ///
    /// Computes `y = W * x` where `W` is the precomputed whitening matrix.
    ///
    /// # Arguments
    ///
    /// * `sample` - Input sample with one value per channel.
    ///
    /// # Returns
    ///
    /// Whitened sample as `[f64; C]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
    ///
    /// let cov = [[1.0, 0.0], [0.0, 1.0]];
    /// let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Pca, 1e-6).unwrap();
    /// let y = wm.apply(&[5.0, 3.0]);
    /// assert!(y[0].is_finite());
    /// ```
    #[inline]
    pub fn apply(&self, sample: &[f64; C]) -> [f64; C] {
        let mut out = [0.0f64; C];
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            let mut sum = 0.0;
            for j in 0..C {
                sum += self.matrix.get(i, j) * sample[j];
            }
            out[i] = sum;
        }
        out
    }

    /// Get a reference to the underlying whitening matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
    /// use zerostone::linalg::Matrix;
    ///
    /// let cov = [[1.0, 0.0], [0.0, 1.0]];
    /// let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6).unwrap();
    /// let mat: &Matrix<2, 4> = wm.matrix();
    /// assert!((mat.get(0, 0) - 1.0).abs() < 0.01);
    /// ```
    pub fn matrix(&self) -> &Matrix<C, M> {
        &self.matrix
    }

    /// Reconstruct a symmetric matrix: `E * diag(d) * E^T`.
    fn reconstruct_symmetric(eigvecs: &Matrix<C, M>, d: &[f64; C]) -> Matrix<C, M> {
        let mut result = Matrix::zeros();
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            for j in 0..C {
                let mut sum = 0.0;
                for k in 0..C {
                    sum += eigvecs.get(i, k) * d[k] * eigvecs.get(j, k);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Build PCA whitening matrix: `diag(d) * E^T`.
    fn build_pca_matrix(eigvecs: &Matrix<C, M>, d: &[f64; C]) -> Matrix<C, M> {
        let mut result = Matrix::zeros();
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            for j in 0..C {
                // row i of W = d[i] * row i of E^T = d[i] * column i of E
                // W[i,j] = d[i] * E^T[i,j] = d[i] * E[j,i]
                result.set(i, j, d[i] * eigvecs.get(j, i));
            }
        }
        result
    }
}

/// Apply a precomputed whitening transform to a multi-channel sample.
///
/// This is a convenience free function equivalent to
/// [`WhiteningMatrix::apply`].
///
/// # Examples
///
/// ```
/// use zerostone::whitening::{WhiteningMatrix, WhiteningMode, apply_whitening};
///
/// let cov = [[4.0, 0.0], [0.0, 9.0]];
/// let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6).unwrap();
/// let y = apply_whitening(&wm, &[2.0, 3.0]);
/// // After ZCA whitening of diagonal cov [4,9]: W ≈ diag(1/2, 1/3)
/// assert!((y[0] - 1.0).abs() < 0.01);
/// assert!((y[1] - 1.0).abs() < 0.01);
/// ```
pub fn apply_whitening<const C: usize, const M: usize>(
    wm: &WhiteningMatrix<C, M>,
    sample: &[f64; C],
) -> [f64; C] {
    wm.apply(sample)
}

/// Estimate noise covariance from spike-free segments of multi-channel data.
///
/// Identifies quiet periods (where all channels are below `threshold * noise_std`)
/// and computes covariance only from those samples. This produces a cleaner
/// whitening matrix that doesn't overfit to spike waveform structure.
///
/// Falls back to full-data covariance if fewer than `min_samples` quiet
/// samples are found.
///
/// # Type Parameters
/// * `C` - Number of channels
///
/// # Arguments
/// * `data` - Multi-channel recording data, one `[f64; C]` per time sample
/// * `noise_std` - Per-channel noise standard deviation (e.g., from MAD estimation)
/// * `threshold` - Amplitude threshold in noise units (typically 2.0-3.0)
/// * `min_samples` - Minimum quiet samples required (fallback to full data if fewer)
///
/// # Returns
/// Covariance matrix `[[f64; C]; C]`
///
/// # Examples
///
/// ```
/// use zerostone::whitening::estimate_noise_covariance;
///
/// // 3 channels, 6 quiet samples
/// let data = [
///     [0.1, -0.2,  0.05],
///     [0.3,  0.1, -0.1],
///     [0.0,  0.15, 0.2],
///     [-0.1, 0.0,  0.1],
///     [0.2, -0.1,  0.0],
///     [-0.2, 0.05, 0.15],
/// ];
/// let noise_std = [1.0, 1.0, 1.0];
/// let cov = estimate_noise_covariance(&data, &noise_std, 3.0, 2);
/// // Result is a 3x3 symmetric covariance matrix
/// assert!((cov[0][1] - cov[1][0]).abs() < 1e-12);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn estimate_noise_covariance<const C: usize>(
    data: &[[f64; C]],
    noise_std: &[f64; C],
    threshold: f64,
    min_samples: usize,
) -> [[f64; C]; C] {
    let n = data.len();
    if n <= 1 {
        return [[0.0; C]; C];
    }

    // Count quiet samples: all channels below threshold * noise_std[ch]
    let mut quiet_count: usize = 0;
    for sample in data.iter() {
        let mut all_quiet = true;
        for ch in 0..C {
            if libm::fabs(sample[ch]) >= threshold * noise_std[ch] {
                all_quiet = false;
                break;
            }
        }
        if all_quiet {
            quiet_count += 1;
        }
    }

    // Decide whether to use quiet-only or full data
    let use_quiet = quiet_count >= min_samples && min_samples > 0;

    // First pass: compute means
    let mut mean = [0.0f64; C];
    let mut count: usize = 0;
    for sample in data.iter() {
        if use_quiet {
            let mut all_quiet = true;
            for ch in 0..C {
                if libm::fabs(sample[ch]) >= threshold * noise_std[ch] {
                    all_quiet = false;
                    break;
                }
            }
            if !all_quiet {
                continue;
            }
        }
        for ch in 0..C {
            mean[ch] += sample[ch];
        }
        count += 1;
    }

    if count <= 1 {
        return [[0.0; C]; C];
    }

    for ch in 0..C {
        mean[ch] /= count as f64;
    }

    // Second pass: compute covariance (centered outer products)
    let mut cov = [[0.0f64; C]; C];
    for sample in data.iter() {
        if use_quiet {
            let mut all_quiet = true;
            for ch in 0..C {
                if libm::fabs(sample[ch]) >= threshold * noise_std[ch] {
                    all_quiet = false;
                    break;
                }
            }
            if !all_quiet {
                continue;
            }
        }
        for i in 0..C {
            for j in 0..C {
                cov[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
            }
        }
    }

    let denom = (count - 1) as f64;
    for i in 0..C {
        for j in 0..C {
            cov[i][j] /= denom;
        }
    }

    cov
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `apply` does not panic for arbitrary finite inputs
    /// when the whitening matrix contains finite values.
    #[kani::proof]
    #[kani::unwind(4)]
    fn apply_whitening_no_panic() {
        let w00: f64 = kani::any();
        let w01: f64 = kani::any();
        let w10: f64 = kani::any();
        let w11: f64 = kani::any();
        let x0: f64 = kani::any();
        let x1: f64 = kani::any();

        kani::assume(w00.is_finite() && w00 >= -1e6 && w00 <= 1e6);
        kani::assume(w01.is_finite() && w01 >= -1e6 && w01 <= 1e6);
        kani::assume(w10.is_finite() && w10 >= -1e6 && w10 <= 1e6);
        kani::assume(w11.is_finite() && w11 >= -1e6 && w11 <= 1e6);
        kani::assume(x0.is_finite() && x0 >= -1e6 && x0 <= 1e6);
        kani::assume(x1.is_finite() && x1 >= -1e6 && x1 <= 1e6);

        let mat = Matrix::<2, 4>::new([w00, w01, w10, w11]);
        let wm = WhiteningMatrix { matrix: mat };
        let sample = [x0, x1];

        let result = wm.apply(&sample);
        assert!(result[0].is_finite(), "output channel 0 must be finite");
        assert!(result[1].is_finite(), "output channel 1 must be finite");
    }

    /// Prove that ZCA `reconstruct_symmetric` produces a symmetric matrix.
    /// The ZCA whitening matrix W = E * diag(d) * E^T is symmetric by construction.
    #[kani::proof]
    #[kani::unwind(4)]
    fn zca_matrix_is_symmetric() {
        let e00: f64 = kani::any();
        let e01: f64 = kani::any();
        let e10: f64 = kani::any();
        let e11: f64 = kani::any();
        let d0: f64 = kani::any();
        let d1: f64 = kani::any();

        kani::assume(e00.is_finite() && e00 >= -10.0 && e00 <= 10.0);
        kani::assume(e01.is_finite() && e01 >= -10.0 && e01 <= 10.0);
        kani::assume(e10.is_finite() && e10 >= -10.0 && e10 <= 10.0);
        kani::assume(e11.is_finite() && e11 >= -10.0 && e11 <= 10.0);
        kani::assume(d0.is_finite() && d0 >= -10.0 && d0 <= 10.0);
        kani::assume(d1.is_finite() && d1 >= -10.0 && d1 <= 10.0);

        let eigvecs = Matrix::<2, 4>::new([e00, e01, e10, e11]);
        let d = [d0, d1];
        let result = WhiteningMatrix::<2, 4>::reconstruct_symmetric(&eigvecs, &d);

        // W[0,1] must equal W[1,0] (symmetry)
        let w01 = result.get(0, 1);
        let w10 = result.get(1, 0);
        let diff = if w01 >= w10 { w01 - w10 } else { w10 - w01 };
        assert!(diff < 1e-10, "ZCA matrix must be symmetric");
    }

    /// Prove that `apply_whitening` free function produces the same result
    /// as the method call (structural equivalence, not just a trivial wrapper
    /// at runtime but verified at the verification level).
    #[kani::proof]
    #[kani::unwind(4)]
    fn apply_free_fn_matches_method() {
        let w00: f64 = kani::any();
        let w01: f64 = kani::any();
        let w10: f64 = kani::any();
        let w11: f64 = kani::any();
        let x0: f64 = kani::any();
        let x1: f64 = kani::any();

        kani::assume(w00.is_finite() && w00 >= -1e3 && w00 <= 1e3);
        kani::assume(w01.is_finite() && w01 >= -1e3 && w01 <= 1e3);
        kani::assume(w10.is_finite() && w10 >= -1e3 && w10 <= 1e3);
        kani::assume(w11.is_finite() && w11 >= -1e3 && w11 <= 1e3);
        kani::assume(x0.is_finite() && x0 >= -1e3 && x0 <= 1e3);
        kani::assume(x1.is_finite() && x1 >= -1e3 && x1 <= 1e3);

        let mat = Matrix::<2, 4>::new([w00, w01, w10, w11]);
        let wm = WhiteningMatrix { matrix: mat };
        let sample = [x0, x1];

        let r1 = wm.apply(&sample);
        let r2 = apply_whitening(&wm, &sample);

        assert!(r1[0] == r2[0], "free fn must match method for ch 0");
        assert!(r1[1] == r2[1], "free fn must match method for ch 1");
    }

    /// Prove that `estimate_noise_covariance` does not panic for small inputs.
    #[kani::proof]
    #[kani::unwind(6)]
    fn estimate_noise_covariance_no_panic() {
        let s0: [f64; 2] = [kani::any(), kani::any()];
        let s1: [f64; 2] = [kani::any(), kani::any()];
        let s2: [f64; 2] = [kani::any(), kani::any()];
        let noise_std: [f64; 2] = [kani::any(), kani::any()];
        let threshold: f64 = kani::any();

        kani::assume(s0[0].is_finite() && s0[0] >= -100.0 && s0[0] <= 100.0);
        kani::assume(s0[1].is_finite() && s0[1] >= -100.0 && s0[1] <= 100.0);
        kani::assume(s1[0].is_finite() && s1[0] >= -100.0 && s1[0] <= 100.0);
        kani::assume(s1[1].is_finite() && s1[1] >= -100.0 && s1[1] <= 100.0);
        kani::assume(s2[0].is_finite() && s2[0] >= -100.0 && s2[0] <= 100.0);
        kani::assume(s2[1].is_finite() && s2[1] >= -100.0 && s2[1] <= 100.0);
        kani::assume(noise_std[0].is_finite() && noise_std[0] >= 0.01 && noise_std[0] <= 100.0);
        kani::assume(noise_std[1].is_finite() && noise_std[1] >= 0.01 && noise_std[1] <= 100.0);
        kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 100.0);

        let data = [s0, s1, s2];
        let result = estimate_noise_covariance(&data, &noise_std, threshold, 2);

        // Result should contain finite values
        for i in 0..2 {
            for j in 0..2 {
                assert!(result[i][j].is_finite(), "covariance must be finite");
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Helper: compute sample covariance from a set of multi-channel samples.
    fn sample_covariance<const C: usize>(data: &[[f64; C]]) -> [[f64; C]; C] {
        let n = data.len();
        assert!(n > 1);

        // Compute means
        let mut mean = [0.0f64; C];
        for sample in data.iter() {
            for c in 0..C {
                mean[c] += sample[c];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }

        // Compute covariance
        let mut cov = [[0.0f64; C]; C];
        for sample in data.iter() {
            for (i, row) in cov.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val += (sample[i] - mean[i]) * (sample[j] - mean[j]);
                }
            }
        }
        for row in cov.iter_mut() {
            for val in row.iter_mut() {
                *val /= (n - 1) as f64;
            }
        }
        cov
    }

    // Simple pseudo-RNG (xorshift64) for deterministic tests
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn gaussian(&mut self, mean: f64, std: f64) -> f64 {
            let u1 = (self.next_u64() % 1_000_000 + 1) as f64 / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as f64 / 1_000_000.0;
            let z = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2);
            mean + z * std
        }
    }

    #[test]
    fn test_zca_identity_covariance() {
        // Identity covariance -> whitening matrix should be near identity
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let m = wm.matrix();
        assert!((m.get(0, 0) - 1.0).abs() < 1e-4);
        assert!((m.get(0, 1)).abs() < 1e-4);
        assert!((m.get(1, 0)).abs() < 1e-4);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_pca_identity_covariance() {
        // Identity covariance -> PCA whitening should also be near identity
        // (eigenvectors of I are the standard basis)
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Pca, 1e-10).unwrap();

        let sample = [3.0, 7.0];
        let out = wm.apply(&sample);
        // The output may be permuted/sign-flipped since eigenvectors of I
        // are arbitrary, but each output component should have magnitude
        // matching one input component.
        let mut mags: [f64; 2] = [libm::fabs(out[0]), libm::fabs(out[1])];
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((mags[0] - 3.0).abs() < 0.1);
        assert!((mags[1] - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_zca_known_2x2() {
        // Diagonal covariance [4, 0; 0, 9]
        // ZCA whitening: W = diag(1/sqrt(4), 1/sqrt(9)) = diag(0.5, 1/3)
        let cov = [[4.0, 0.0], [0.0, 9.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let out = wm.apply(&[2.0, 3.0]);
        assert!((out[0] - 1.0).abs() < 0.01, "Expected ~1.0, got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 0.01, "Expected ~1.0, got {}", out[1]);
    }

    #[test]
    fn test_zca_produces_identity_covariance() {
        // Generate correlated 2-channel data, whiten, verify output covariance ~ I
        let mut rng = Rng::new(42);
        let n = 2000;
        let cov_true = [[4.0, 2.0], [2.0, 3.0]];

        // Generate samples with known covariance using Cholesky factor
        // L = chol([[4,2],[2,3]]) => L[0,0]=2, L[1,0]=1, L[1,1]=sqrt(2)
        let l00 = 2.0;
        let l10 = 1.0;
        let l11 = libm::sqrt(2.0);

        let mut data = [[0.0f64; 2]; 2000];
        for i in 0..n {
            let z0 = rng.gaussian(0.0, 1.0);
            let z1 = rng.gaussian(0.0, 1.0);
            data[i] = [l00 * z0, l10 * z0 + l11 * z1];
        }

        // Compute empirical covariance and whiten
        let emp_cov = sample_covariance(&data);
        let wm =
            WhiteningMatrix::<2, 4>::from_covariance(&emp_cov, WhiteningMode::Zca, 1e-10).unwrap();

        // Whiten all samples
        let mut whitened = [[0.0f64; 2]; 2000];
        for i in 0..n {
            whitened[i] = wm.apply(&data[i]);
        }

        // Covariance of whitened data should be near identity
        let white_cov = sample_covariance(&whitened);
        assert!(
            (white_cov[0][0] - 1.0).abs() < 0.1,
            "Var(ch0)={}, expected ~1",
            white_cov[0][0]
        );
        assert!(
            (white_cov[1][1] - 1.0).abs() < 0.1,
            "Var(ch1)={}, expected ~1",
            white_cov[1][1]
        );
        assert!(
            white_cov[0][1].abs() < 0.1,
            "Cov(0,1)={}, expected ~0",
            white_cov[0][1]
        );
        // Suppress unused warning on cov_true
        let _ = cov_true;
    }

    #[test]
    fn test_pca_decorrelates() {
        // PCA whitening should produce uncorrelated unit-variance output
        let mut rng = Rng::new(77);
        let n = 2000;

        let mut data = [[0.0f64; 2]; 2000];
        for i in 0..n {
            let z0 = rng.gaussian(0.0, 1.0);
            let z1 = rng.gaussian(0.0, 1.0);
            data[i] = [3.0 * z0 + z1, z0 + 2.0 * z1];
        }

        let emp_cov = sample_covariance(&data);
        let wm =
            WhiteningMatrix::<2, 4>::from_covariance(&emp_cov, WhiteningMode::Pca, 1e-10).unwrap();

        let mut whitened = [[0.0f64; 2]; 2000];
        for i in 0..n {
            whitened[i] = wm.apply(&data[i]);
        }

        let white_cov = sample_covariance(&whitened);
        assert!(
            (white_cov[0][0] - 1.0).abs() < 0.1,
            "Var(pc0)={}, expected ~1",
            white_cov[0][0]
        );
        assert!(
            (white_cov[1][1] - 1.0).abs() < 0.1,
            "Var(pc1)={}, expected ~1",
            white_cov[1][1]
        );
        assert!(
            white_cov[0][1].abs() < 0.1,
            "Cov(pc0,pc1)={}, expected ~0",
            white_cov[0][1]
        );
    }

    #[test]
    fn test_regularization_singular_covariance() {
        // Rank-1 covariance: all channels identical
        // Without regularization this would be singular
        let cov = [[1.0, 1.0], [1.0, 1.0]];
        let result = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-3);
        assert!(result.is_ok(), "Regularization should handle singular cov");

        let wm = result.unwrap();
        let out = wm.apply(&[1.0, 1.0]);
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
    }

    #[test]
    fn test_already_white_is_near_identity() {
        // If data is already white (identity covariance), whitening
        // should barely change the signal
        let cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let wm = WhiteningMatrix::<3, 9>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let sample = [1.5, -2.3, 0.7];
        let out = wm.apply(&sample);
        for c in 0..3 {
            assert!(
                (out[c] - sample[c]).abs() < 0.01,
                "Channel {} changed: {} -> {}",
                c,
                sample[c],
                out[c]
            );
        }
    }

    #[test]
    fn test_known_4x4_diagonal() {
        // Diagonal covariance with known eigenvalues
        let cov = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 16.0],
        ];
        let wm =
            WhiteningMatrix::<4, 16>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let sample = [1.0, 2.0, 3.0, 4.0];
        let out = wm.apply(&sample);

        // W = diag(1, 1/2, 1/3, 1/4)
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 1.0).abs() < 0.01);
        assert!((out[2] - 1.0).abs() < 0.01);
        assert!((out[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_known_4x4_correlated() {
        // 4x4 correlated covariance, ZCA whitening round-trip
        let mut rng = Rng::new(314);
        let n = 3000;

        // Generate correlated 4-channel data
        let mut data = [[0.0f64; 4]; 3000];
        for i in 0..n {
            let z0 = rng.gaussian(0.0, 1.0);
            let z1 = rng.gaussian(0.0, 1.0);
            let z2 = rng.gaussian(0.0, 1.0);
            let z3 = rng.gaussian(0.0, 1.0);
            data[i] = [
                2.0 * z0 + 0.5 * z1,
                0.5 * z0 + 2.0 * z1 + 0.3 * z2,
                0.3 * z1 + 2.0 * z2 + 0.5 * z3,
                0.5 * z2 + 2.0 * z3,
            ];
        }

        let emp_cov = sample_covariance(&data);
        let wm =
            WhiteningMatrix::<4, 16>::from_covariance(&emp_cov, WhiteningMode::Zca, 1e-10).unwrap();

        let mut whitened = [[0.0f64; 4]; 3000];
        for i in 0..n {
            whitened[i] = wm.apply(&data[i]);
        }

        let white_cov = sample_covariance(&whitened);

        // Diagonal should be ~1, off-diagonal ~0
        for i in 0..4 {
            assert!(
                (white_cov[i][i] - 1.0).abs() < 0.15,
                "Var(ch{})={}, expected ~1",
                i,
                white_cov[i][i]
            );
            for j in 0..4 {
                if i != j {
                    assert!(
                        white_cov[i][j].abs() < 0.15,
                        "Cov({},{})={}, expected ~0",
                        i,
                        j,
                        white_cov[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_roundtrip_whitened_data_unit_covariance() {
        // End-to-end: generate data, compute covariance, whiten, verify
        let mut rng = Rng::new(999);
        let n = 5000;

        let mut data = [[0.0f64; 3]; 5000];
        for i in 0..n {
            let z0 = rng.gaussian(0.0, 1.0);
            let z1 = rng.gaussian(0.0, 1.0);
            let z2 = rng.gaussian(0.0, 1.0);
            data[i] = [5.0 * z0 + z1, z0 + 3.0 * z1 + 0.5 * z2, 0.5 * z1 + 2.0 * z2];
        }

        let emp_cov = sample_covariance(&data);
        let wm =
            WhiteningMatrix::<3, 9>::from_covariance(&emp_cov, WhiteningMode::Zca, 1e-10).unwrap();

        let mut whitened = [[0.0f64; 3]; 5000];
        for i in 0..n {
            whitened[i] = wm.apply(&data[i]);
        }

        let white_cov = sample_covariance(&whitened);
        for i in 0..3 {
            assert!(
                (white_cov[i][i] - 1.0).abs() < 0.1,
                "Var(ch{})={}, expected ~1",
                i,
                white_cov[i][i]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        white_cov[i][j].abs() < 0.1,
                        "Cov({},{})={}, expected ~0",
                        i,
                        j,
                        white_cov[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_zca_symmetry() {
        // ZCA whitening matrix should be symmetric
        let cov = [[4.0, 2.0], [2.0, 3.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let m = wm.matrix();
        assert!(
            (m.get(0, 1) - m.get(1, 0)).abs() < 1e-10,
            "ZCA matrix should be symmetric"
        );
    }

    #[test]
    fn test_apply_whitening_free_function() {
        let cov = [[4.0, 0.0], [0.0, 9.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();

        let out1 = wm.apply(&[2.0, 3.0]);
        let out2 = apply_whitening(&wm, &[2.0, 3.0]);
        assert!((out1[0] - out2[0]).abs() < 1e-15);
        assert!((out1[1] - out2[1]).abs() < 1e-15);
    }

    #[test]
    fn test_noise_cov_all_quiet() {
        // When all data is below threshold, noise cov should equal full cov
        let data = [
            [0.1, -0.2],
            [0.3, 0.1],
            [-0.1, 0.15],
            [0.2, -0.1],
            [-0.2, 0.05],
        ];
        let noise_std = [1.0, 1.0];
        let noise_cov = estimate_noise_covariance(&data, &noise_std, 3.0, 2);
        let full_cov = sample_covariance(&data);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (noise_cov[i][j] - full_cov[i][j]).abs() < 1e-12,
                    "All-quiet noise cov should match full cov at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_noise_cov_with_spikes() {
        // Inject large spikes: noise cov should differ from full cov
        let mut data = [[0.0f64; 2]; 100];
        let mut rng = Rng::new(123);
        for i in 0..100 {
            data[i] = [rng.gaussian(0.0, 1.0), rng.gaussian(0.0, 1.0)];
        }
        // Inject spikes at known positions
        data[10] = [50.0, 50.0];
        data[20] = [-40.0, 60.0];
        data[30] = [45.0, -55.0];

        let noise_std = [1.0, 1.0];
        let noise_cov = estimate_noise_covariance(&data, &noise_std, 5.0, 10);
        let full_cov = sample_covariance(&data);

        // Full covariance should be inflated by spikes
        // Noise covariance should have smaller diagonal entries
        assert!(
            noise_cov[0][0] < full_cov[0][0],
            "Noise var should be smaller than full var: {} vs {}",
            noise_cov[0][0],
            full_cov[0][0]
        );
        assert!(
            noise_cov[1][1] < full_cov[1][1],
            "Noise var should be smaller than full var: {} vs {}",
            noise_cov[1][1],
            full_cov[1][1]
        );
    }

    #[test]
    fn test_noise_cov_fallback() {
        // All samples exceed threshold -> falls back to full covariance
        let data = [
            [10.0, 10.0],
            [11.0, 9.0],
            [-10.0, 10.0],
            [9.0, -11.0],
            [-11.0, -9.0],
        ];
        let noise_std = [1.0, 1.0];
        // threshold=3.0: all samples have |val| > 3.0, so 0 quiet samples
        let noise_cov = estimate_noise_covariance(&data, &noise_std, 3.0, 2);
        let full_cov = sample_covariance(&data);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (noise_cov[i][j] - full_cov[i][j]).abs() < 1e-12,
                    "Fallback should match full cov at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn test_noise_cov_empty_data() {
        let data: &[[f64; 2]] = &[];
        let noise_std = [1.0, 1.0];
        let cov = estimate_noise_covariance(data, &noise_std, 3.0, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov[i][j]).abs() < 1e-15, "Empty data should give zero cov");
            }
        }
    }

    #[test]
    fn test_noise_cov_symmetry() {
        let mut rng = Rng::new(555);
        let mut data = [[0.0f64; 3]; 200];
        for i in 0..200 {
            data[i] = [
                rng.gaussian(0.0, 1.0),
                rng.gaussian(0.0, 2.0),
                rng.gaussian(0.0, 1.5),
            ];
        }
        // Inject a few spikes
        data[50] = [20.0, 30.0, 25.0];
        data[100] = [-15.0, 20.0, -30.0];

        let noise_std = [1.0, 2.0, 1.5];
        let cov = estimate_noise_covariance(&data, &noise_std, 4.0, 10);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[i][j] - cov[j][i]).abs() < 1e-12,
                    "Covariance must be symmetric: cov[{i}][{j}]={} vs cov[{j}][{i}]={}",
                    cov[i][j],
                    cov[j][i]
                );
            }
        }
    }

    #[test]
    fn test_noise_cov_single_sample() {
        // Single sample: not enough for covariance, should return zeros
        let data = [[1.0, 2.0]];
        let noise_std = [1.0, 1.0];
        let cov = estimate_noise_covariance(&data, &noise_std, 3.0, 1);
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov[i][j]).abs() < 1e-15, "Single sample should give zero cov");
            }
        }
    }

    #[test]
    fn test_large_epsilon_shrinks_toward_identity() {
        // With very large epsilon, all eigenvalues become ~epsilon,
        // so W ≈ (1/sqrt(eps)) * I
        let cov = [[100.0, 50.0], [50.0, 100.0]];
        let eps = 1e6;
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, eps).unwrap();

        let m = wm.matrix();
        let expected = 1.0 / libm::sqrt(eps);
        assert!(
            (m.get(0, 0) - expected).abs() / expected < 0.1,
            "Diagonal should be ~1/sqrt(eps)"
        );
        assert!(
            m.get(0, 1).abs() / expected < 0.1,
            "Off-diagonal should be small relative to diagonal"
        );
    }
}
