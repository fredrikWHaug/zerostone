//! Riemannian geometry primitives for SPD matrices in BCI applications.
//!
//! Provides tangent space projections for symmetric positive-definite (SPD) covariance
//! matrices, enabling the use of Euclidean machine learning classifiers on manifold data.
//! Widely used in motor imagery BCI for feature extraction and transfer learning.
//!
//! # Background
//!
//! Covariance matrices from multi-channel neural recordings lie on the Riemannian manifold
//! of SPD matrices. Operating directly on this manifold requires specialized algorithms.
//! Tangent space projection maps SPD matrices to a local Euclidean space where standard
//! classifiers (SVM, LDA) can operate.
//!
//! # Typical BCI Pipeline
//!
//! 1. Compute covariance matrices from multi-channel signals
//! 2. Calculate geometric mean as reference point
//! 3. Project all matrices to tangent space at reference
//! 4. Vectorize symmetric matrices (upper triangular)
//! 5. Train Euclidean classifier (SVM, LDA, etc.)
//!
//! # References
//!
//! - Barachant et al. (2012): "Multiclass Brain-Computer Interface Classification by
//!   Riemannian Geometry"
//! - Congedo et al. (2017): "Riemannian geometry for EEG-based brain-computer interfaces"
//! - pyRiemann library: Standard implementation for BCI research
//!
//! # Example
//!
//! ```
//! use zerostone::riemannian::TangentSpace;
//! use zerostone::linalg::Matrix;
//!
//! // Two 3×3 covariance matrices from two trials
//! let cov1: Matrix<3, 9> = Matrix::new([
//!     2.0, 0.5, 0.3,
//!     0.5, 1.8, 0.2,
//!     0.3, 0.2, 1.5,
//! ]);
//! let cov2: Matrix<3, 9> = Matrix::new([
//!     1.9, 0.4, 0.2,
//!     0.4, 2.1, 0.3,
//!     0.2, 0.3, 1.6,
//! ]);
//!
//! // Create tangent space projector
//! let mut ts: TangentSpace<3, 9, 6> = TangentSpace::new();
//!
//! // Fit to reference (uses first matrix as reference for this example)
//! ts.fit(&cov1).unwrap();
//!
//! // Project matrices to tangent space and vectorize
//! let vec1 = ts.transform(&cov1).unwrap();
//! let vec2 = ts.transform(&cov2).unwrap();
//!
//! // vec1 and vec2 are now 6-dimensional vectors (3×4/2 = 6 upper triangular elements)
//! // Ready for Euclidean classifiers
//! assert_eq!(vec1.len(), 6);
//! ```

use crate::linalg::{LinalgError, Matrix};
use libm;

/// Tangent space projection for SPD matrices.
///
/// Maps covariance matrices from the Riemannian manifold to a tangent space
/// at a reference point, then vectorizes for use with Euclidean classifiers.
///
/// # Type Parameters
///
/// * `C` - Matrix dimension (C×C covariance matrix)
/// * `M` - Total matrix elements (M = C×C)
/// * `V` - Vector dimension (V = C×(C+1)/2, upper triangular elements)
///
/// # Memory Layout
///
/// All operations are zero-allocation after initialization. Reference matrix
/// and its square root are pre-computed during `fit()`.
pub struct TangentSpace<const C: usize, const M: usize, const V: usize> {
    /// Reference matrix (typically geometric mean of training set)
    reference: Option<Matrix<C, M>>,

    /// Square root of reference matrix: C^(1/2)
    reference_sqrt: Option<Matrix<C, M>>,

    /// Inverse square root of reference matrix: C^(-1/2)
    reference_inv_sqrt: Option<Matrix<C, M>>,
}

impl<const C: usize, const M: usize, const V: usize> TangentSpace<C, M, V> {
    /// Compile-time assertion that M = C * C
    const _ASSERT_M: () = assert!(M == C * C, "M must equal C * C");

    /// Compile-time assertion that V = C * (C + 1) / 2
    const _ASSERT_V: () = assert!(V == C * (C + 1) / 2, "V must equal C * (C + 1) / 2");

    /// Create a new tangent space projector.
    ///
    /// Call `fit()` to set the reference point before using `transform()`.
    pub fn new() -> Self {
        // Trigger compile-time assertions
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_M;
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_V;

        Self {
            reference: None,
            reference_sqrt: None,
            reference_inv_sqrt: None,
        }
    }

    /// Fit the tangent space to a reference matrix.
    ///
    /// The reference matrix is typically the geometric mean of a set of SPD matrices.
    /// For simplicity, you can also use a single representative matrix from your dataset.
    ///
    /// # Arguments
    ///
    /// * `reference` - SPD matrix to use as the tangent space reference point
    ///
    /// # Errors
    ///
    /// Returns error if eigenvalue decomposition fails or matrix is not positive definite.
    pub fn fit(&mut self, reference: &Matrix<C, M>) -> Result<(), LinalgError> {
        // Compute square root and inverse square root of reference
        let sqrt = matrix_sqrt(reference)?;
        let inv_sqrt = matrix_inv_sqrt(reference)?;

        self.reference = Some(*reference);
        self.reference_sqrt = Some(sqrt);
        self.reference_inv_sqrt = Some(inv_sqrt);

        Ok(())
    }

    /// Project an SPD matrix to tangent space and vectorize.
    ///
    /// Computes the logarithmic map: log_C(P) = C^(1/2) log(C^(-1/2) P C^(-1/2)) C^(1/2),
    /// then extracts the upper triangular elements as a vector.
    ///
    /// # Arguments
    ///
    /// * `matrix` - SPD matrix to project (e.g., covariance matrix)
    ///
    /// # Returns
    ///
    /// Vector of length V = C×(C+1)/2 containing the upper triangular elements
    /// of the tangent space projection.
    ///
    /// # Errors
    ///
    /// Returns error if `fit()` has not been called or if matrix logarithm fails.
    pub fn transform(&self, matrix: &Matrix<C, M>) -> Result<[f64; V], LinalgError> {
        let inv_sqrt = self
            .reference_inv_sqrt
            .as_ref()
            .ok_or(LinalgError::NumericalInstability)?;
        let sqrt = self
            .reference_sqrt
            .as_ref()
            .ok_or(LinalgError::NumericalInstability)?;

        // Compute C^(-1/2) * P * C^(-1/2)
        let temp1 = inv_sqrt.matmul(matrix);
        let temp2 = temp1.matmul(inv_sqrt);

        // Compute log(C^(-1/2) * P * C^(-1/2))
        let log_temp = matrix_log(&temp2)?;

        // Compute C^(1/2) * log(...) * C^(1/2)
        let temp3 = sqrt.matmul(&log_temp);
        let tangent = temp3.matmul(sqrt);

        // Vectorize upper triangular
        Ok(vectorize_upper_triangular(&tangent))
    }

    /// Project from tangent space back to the manifold.
    ///
    /// Computes the exponential map: exp_C(S) = C^(1/2) exp(C^(-1/2) S C^(-1/2)) C^(1/2),
    /// after reconstructing the symmetric matrix from its vector representation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Tangent space vector (length V = C×(C+1)/2)
    ///
    /// # Returns
    ///
    /// SPD matrix on the manifold.
    ///
    /// # Errors
    ///
    /// Returns error if `fit()` has not been called or if matrix exponential fails.
    pub fn inverse_transform(&self, vector: &[f64; V]) -> Result<Matrix<C, M>, LinalgError> {
        let inv_sqrt = self
            .reference_inv_sqrt
            .as_ref()
            .ok_or(LinalgError::NumericalInstability)?;
        let sqrt = self
            .reference_sqrt
            .as_ref()
            .ok_or(LinalgError::NumericalInstability)?;

        // Reconstruct symmetric matrix from vector
        let tangent = unvectorize_upper_triangular(vector);

        // Compute C^(-1/2) * S * C^(-1/2)
        let temp1 = inv_sqrt.matmul(&tangent);
        let temp2 = temp1.matmul(inv_sqrt);

        // Compute exp(C^(-1/2) * S * C^(-1/2))
        let exp_temp = matrix_exp(&temp2)?;

        // Compute C^(1/2) * exp(...) * C^(1/2)
        let temp3 = sqrt.matmul(&exp_temp);
        Ok(temp3.matmul(sqrt))
    }

    /// Get the reference matrix.
    pub fn reference(&self) -> Option<&Matrix<C, M>> {
        self.reference.as_ref()
    }
}

impl<const C: usize, const M: usize, const V: usize> Default for TangentSpace<C, M, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute matrix logarithm of an SPD matrix via eigenvalue decomposition.
///
/// For SPD matrix P with eigendecomposition P = U Λ U^T:
/// log(P) = U log(Λ) U^T
///
/// # Arguments
///
/// * `matrix` - Symmetric positive-definite matrix
///
/// # Returns
///
/// Symmetric matrix logarithm
///
/// # Errors
///
/// Returns error if eigenvalue decomposition fails or eigenvalues are non-positive.
pub fn matrix_log<const C: usize, const M: usize>(
    matrix: &Matrix<C, M>,
) -> Result<Matrix<C, M>, LinalgError> {
    // Compute eigenvalue decomposition
    let eigen = matrix.eigen_symmetric(50, 1e-12)?;

    // Take log of eigenvalues
    let mut log_eigenvalues = eigen.eigenvalues;
    for val in &mut log_eigenvalues {
        if *val <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite);
        }
        *val = libm::log(*val);
    }

    // Reconstruct: log(P) = U * log(Λ) * U^T
    reconstruct_from_eigen(&eigen.eigenvectors, &log_eigenvalues)
}

/// Compute matrix exponential of a symmetric matrix via eigenvalue decomposition.
///
/// For symmetric matrix S with eigendecomposition S = U Λ U^T:
/// exp(S) = U exp(Λ) U^T
///
/// # Arguments
///
/// * `matrix` - Symmetric matrix
///
/// # Returns
///
/// SPD matrix exponential
///
/// # Errors
///
/// Returns error if eigenvalue decomposition fails.
pub fn matrix_exp<const C: usize, const M: usize>(
    matrix: &Matrix<C, M>,
) -> Result<Matrix<C, M>, LinalgError> {
    // Compute eigenvalue decomposition
    let eigen = matrix.eigen_symmetric(50, 1e-12)?;

    // Take exp of eigenvalues
    let mut exp_eigenvalues = eigen.eigenvalues;
    for val in &mut exp_eigenvalues {
        *val = libm::exp(*val);
    }

    // Reconstruct: exp(S) = U * exp(Λ) * U^T
    reconstruct_from_eigen(&eigen.eigenvectors, &exp_eigenvalues)
}

/// Compute matrix square root of an SPD matrix via eigenvalue decomposition.
///
/// For SPD matrix P with eigendecomposition P = U Λ U^T:
/// sqrt(P) = U sqrt(Λ) U^T
///
/// # Arguments
///
/// * `matrix` - Symmetric positive-definite matrix
///
/// # Returns
///
/// Symmetric matrix square root
///
/// # Errors
///
/// Returns error if eigenvalue decomposition fails or eigenvalues are non-positive.
pub fn matrix_sqrt<const C: usize, const M: usize>(
    matrix: &Matrix<C, M>,
) -> Result<Matrix<C, M>, LinalgError> {
    let eigen = matrix.eigen_symmetric(50, 1e-12)?;

    let mut sqrt_eigenvalues = eigen.eigenvalues;
    for val in &mut sqrt_eigenvalues {
        if *val < 0.0 {
            return Err(LinalgError::NotPositiveDefinite);
        }
        *val = libm::sqrt(*val);
    }

    reconstruct_from_eigen(&eigen.eigenvectors, &sqrt_eigenvalues)
}

/// Compute inverse matrix square root of an SPD matrix via eigenvalue decomposition.
///
/// For SPD matrix P with eigendecomposition P = U Λ U^T:
/// P^(-1/2) = U Λ^(-1/2) U^T
///
/// # Arguments
///
/// * `matrix` - Symmetric positive-definite matrix
///
/// # Returns
///
/// Symmetric inverse matrix square root
///
/// # Errors
///
/// Returns error if eigenvalue decomposition fails or eigenvalues are non-positive.
pub fn matrix_inv_sqrt<const C: usize, const M: usize>(
    matrix: &Matrix<C, M>,
) -> Result<Matrix<C, M>, LinalgError> {
    let eigen = matrix.eigen_symmetric(50, 1e-12)?;

    let mut inv_sqrt_eigenvalues = eigen.eigenvalues;
    for val in &mut inv_sqrt_eigenvalues {
        if *val <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite);
        }
        *val = 1.0 / libm::sqrt(*val);
    }

    reconstruct_from_eigen(&eigen.eigenvectors, &inv_sqrt_eigenvalues)
}

/// Reconstruct matrix from eigenvectors and modified eigenvalues.
///
/// Computes: U * diag(eigenvalues) * U^T
fn reconstruct_from_eigen<const C: usize, const M: usize>(
    eigenvectors: &Matrix<C, M>,
    eigenvalues: &[f64; C],
) -> Result<Matrix<C, M>, LinalgError> {
    let mut result = Matrix::zeros();

    // result = U * Λ * U^T
    // Compute element-by-element: result[i,j] = sum_k U[i,k] * λ[k] * U[j,k]
    #[allow(clippy::needless_range_loop)]
    for i in 0..C {
        for j in 0..C {
            let mut sum = 0.0;
            for k in 0..C {
                let u_ik = eigenvectors.get(i, k);
                let u_jk = eigenvectors.get(j, k);
                sum += u_ik * eigenvalues[k] * u_jk;
            }
            result.set(i, j, sum);
        }
    }

    Ok(result)
}

/// Extract upper triangular elements from a symmetric matrix.
///
/// For a C×C matrix, extracts C×(C+1)/2 elements in row-major order.
/// Includes diagonal elements.
///
/// # Example
///
/// For 3×3 matrix:
/// ```text
/// [a b c]
/// [b d e]  ->  [a, b, c, d, e, f]
/// [c e f]
/// ```
fn vectorize_upper_triangular<const C: usize, const M: usize, const V: usize>(
    matrix: &Matrix<C, M>,
) -> [f64; V] {
    let mut vec = [0.0; V];
    let mut idx = 0;

    for i in 0..C {
        for j in i..C {
            vec[idx] = matrix.get(i, j);
            idx += 1;
        }
    }

    vec
}

/// Reconstruct a symmetric matrix from upper triangular elements.
///
/// Inverse of `vectorize_upper_triangular`. Fills both upper and lower
/// triangular parts to maintain symmetry.
fn unvectorize_upper_triangular<const C: usize, const M: usize, const V: usize>(
    vector: &[f64; V],
) -> Matrix<C, M> {
    let mut matrix = Matrix::zeros();
    let mut idx = 0;

    for i in 0..C {
        for j in i..C {
            let val = vector[idx];
            matrix.set(i, j, val);
            if i != j {
                matrix.set(j, i, val); // Symmetric
            }
            idx += 1;
        }
    }

    matrix
}

/// Compute the Riemannian (affine-invariant) distance between two SPD matrices.
///
/// d(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F
///         = sqrt(sum(log(lambda_i)^2))
///
/// where lambda_i are the eigenvalues of A^{-1/2} B A^{-1/2}.
///
/// # Errors
///
/// Returns error if eigenvalue decomposition fails or matrices are not SPD.
pub fn riemannian_distance<const C: usize, const M: usize>(
    a: &Matrix<C, M>,
    b: &Matrix<C, M>,
) -> Result<f64, LinalgError> {
    let a_inv_sqrt = matrix_inv_sqrt(a)?;

    // Compute A^{-1/2} B A^{-1/2}
    let temp = a_inv_sqrt.matmul(b).matmul(&a_inv_sqrt);

    // Eigendecompose
    let eigen = temp.eigen_symmetric(50, 1e-12)?;

    // d = sqrt(sum(log(lambda_i)^2))
    let mut sum_sq = 0.0;
    for i in 0..C {
        if eigen.eigenvalues[i] <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let log_val = libm::log(eigen.eigenvalues[i]);
        sum_sq += log_val * log_val;
    }

    Ok(libm::sqrt(sum_sq))
}

/// Compute the Frechet (geometric/Karcher) mean of a set of SPD matrices.
///
/// Uses Karcher flow iteration with adaptive step size.
///
/// # Arguments
///
/// * `matrices` - Slice of references to SPD matrices
/// * `output` - Output matrix for the result
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance on ||J||_F
///
/// # Returns
///
/// Number of iterations used.
///
/// # Errors
///
/// Returns error if any matrix operation fails or convergence is not reached.
pub fn frechet_mean<const C: usize, const M: usize>(
    matrices: &[&Matrix<C, M>],
    output: &mut Matrix<C, M>,
    max_iter: usize,
    tolerance: f64,
) -> Result<usize, LinalgError> {
    let n = matrices.len();
    if n == 0 {
        return Err(LinalgError::DimensionMismatch);
    }
    if n == 1 {
        *output = *matrices[0];
        return Ok(0);
    }

    // Initialize with arithmetic mean
    let mut g = Matrix::<C, M>::zeros();
    for mat in matrices {
        g = g.add(mat);
    }
    g = g.scale(1.0 / n as f64);

    let mut nu = 1.0; // adaptive step size
    let mut prev_norm = f64::MAX;

    for iter in 0..max_iter {
        let g_sqrt = matrix_sqrt(&g)?;
        let g_inv_sqrt = matrix_inv_sqrt(&g)?;

        // Average tangent vectors at G
        let mut j = Matrix::<C, M>::zeros();
        for mat in matrices {
            // logm(G^{-1/2} * C_i * G^{-1/2})
            let whitened = g_inv_sqrt.matmul(mat).matmul(&g_inv_sqrt);
            let log_whitened = matrix_log(&whitened)?;
            j = j.add(&log_whitened);
        }
        j = j.scale(1.0 / n as f64);

        // Check convergence
        let j_norm = j.frobenius_norm();
        if j_norm < tolerance {
            *output = g;
            return Ok(iter);
        }

        // Adaptive step size
        if j_norm < prev_norm {
            nu *= 0.95;
        } else {
            nu *= 0.5;
        }
        prev_norm = j_norm;

        // Update: G = G^{1/2} * expm(nu * J) * G^{1/2}
        let scaled_j = j.scale(nu);
        let exp_j = matrix_exp(&scaled_j)?;
        g = g_sqrt.matmul(&exp_j).matmul(&g_sqrt);

        // Force symmetry
        let g_t = g.transpose();
        g = g.add(&g_t).scale(0.5);
    }

    // Return best result even if not fully converged
    *output = g;
    Ok(max_iter)
}

/// Classify an SPD matrix trial by minimum Riemannian distance to class means.
///
/// # Arguments
///
/// * `trial` - SPD matrix to classify
/// * `class_means` - Slice of references to class mean SPD matrices
///
/// # Returns
///
/// Tuple of (class_index, min_distance).
///
/// # Errors
///
/// Returns error if distance computation fails or class_means is empty.
pub fn mdm_classify<const C: usize, const M: usize>(
    trial: &Matrix<C, M>,
    class_means: &[&Matrix<C, M>],
) -> Result<(usize, f64), LinalgError> {
    if class_means.is_empty() {
        return Err(LinalgError::DimensionMismatch);
    }

    let mut best_class = 0;
    let mut best_dist = f64::MAX;

    for (i, mean) in class_means.iter().enumerate() {
        let dist = riemannian_distance(trial, mean)?;
        if dist < best_dist {
            best_dist = dist;
            best_class = i;
        }
    }

    Ok((best_class, best_dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_log_identity() {
        // log(I) = 0
        let eye: Matrix<3, 9> = Matrix::identity();
        let log_eye = matrix_log(&eye).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    log_eye.get(i, j).abs() < 1e-10,
                    "log(I) should be zero matrix"
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_zero() {
        // exp(0) = I
        let zero: Matrix<3, 9> = Matrix::zeros();
        let exp_zero = matrix_exp(&zero).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (exp_zero.get(i, j) - expected).abs() < 1e-10,
                    "exp(0) should be identity"
                );
            }
        }
    }

    #[test]
    fn test_log_exp_inverse() {
        // exp(log(P)) = P for SPD matrix P
        let mut p: Matrix<2, 4> = Matrix::identity();
        p.set(0, 0, 2.0);
        p.set(1, 1, 3.0);
        p.set(0, 1, 0.5);
        p.set(1, 0, 0.5);

        let log_p = matrix_log(&p).unwrap();
        let reconstructed = matrix_exp(&log_p).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed.get(i, j) - p.get(i, j)).abs() < 1e-9,
                    "exp(log(P)) should equal P"
                );
            }
        }
    }

    #[test]
    fn test_matrix_sqrt() {
        // sqrt(P)^2 = P
        let mut p: Matrix<2, 4> = Matrix::identity();
        p.set(0, 0, 4.0);
        p.set(1, 1, 9.0);

        let sqrt_p = matrix_sqrt(&p).unwrap();
        let reconstructed = sqrt_p.matmul(&sqrt_p);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed.get(i, j) - p.get(i, j)).abs() < 1e-9,
                    "sqrt(P)^2 should equal P"
                );
            }
        }
    }

    #[test]
    fn test_matrix_inv_sqrt() {
        // P^(-1/2) * P^(1/2) = I
        let mut p: Matrix<2, 4> = Matrix::identity();
        p.set(0, 0, 4.0);
        p.set(1, 1, 9.0);
        p.set(0, 1, 1.0);
        p.set(1, 0, 1.0);

        let sqrt_p = matrix_sqrt(&p).unwrap();
        let inv_sqrt_p = matrix_inv_sqrt(&p).unwrap();
        let result = inv_sqrt_p.matmul(&sqrt_p);

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.get(i, j) - expected).abs() < 1e-9,
                    "P^(-1/2) * P^(1/2) should be identity"
                );
            }
        }
    }

    #[test]
    fn test_vectorize_upper_triangular() {
        let mut m: Matrix<3, 9> = Matrix::zeros();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 1, 4.0);
        m.set(1, 2, 5.0);
        m.set(2, 2, 6.0);

        let vec: [f64; 6] = vectorize_upper_triangular(&m);

        assert_eq!(vec, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unvectorize_upper_triangular() {
        let vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m: Matrix<3, 9> = unvectorize_upper_triangular(&vec);

        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
        assert_eq!(m.get(1, 2), 5.0);
        assert_eq!(m.get(2, 2), 6.0);

        // Check symmetry
        assert_eq!(m.get(1, 0), 2.0);
        assert_eq!(m.get(2, 0), 3.0);
        assert_eq!(m.get(2, 1), 5.0);
    }

    #[test]
    fn test_tangent_space_identity_reference() {
        // Using identity as reference simplifies to log map
        let mut ts: TangentSpace<2, 4, 3> = TangentSpace::new();
        let eye: Matrix<2, 4> = Matrix::identity();
        ts.fit(&eye).unwrap();

        // Transform diagonal matrix
        let mut p: Matrix<2, 4> = Matrix::identity();
        p.set(0, 0, 2.0);
        p.set(1, 1, 3.0);

        let vec = ts.transform(&p).unwrap();

        // Should be close to log([2, 0, 0, 3]) = [log(2), 0, log(3)]
        assert!((vec[0] - libm::log(2.0)).abs() < 1e-9, "Diagonal element");
        assert!(vec[1].abs() < 1e-9, "Off-diagonal should be zero");
        assert!((vec[2] - libm::log(3.0)).abs() < 1e-9, "Diagonal element");
    }

    #[test]
    fn test_tangent_space_round_trip() {
        // transform() then inverse_transform() should recover original
        let mut ts: TangentSpace<2, 4, 3> = TangentSpace::new();

        let reference: Matrix<2, 4> = Matrix::identity();
        ts.fit(&reference).unwrap();

        let mut p: Matrix<2, 4> = Matrix::identity();
        p.set(0, 0, 2.5);
        p.set(1, 1, 1.8);
        p.set(0, 1, 0.3);
        p.set(1, 0, 0.3);

        let vec = ts.transform(&p).unwrap();
        let reconstructed = ts.inverse_transform(&vec).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed.get(i, j) - p.get(i, j)).abs() < 1e-8,
                    "Round trip should preserve matrix"
                );
            }
        }
    }

    #[test]
    fn test_tangent_space_multiple_matrices() {
        // Project multiple covariance matrices
        let mut ts: TangentSpace<3, 9, 6> = TangentSpace::new();

        let cov1: Matrix<3, 9> = Matrix::new([2.0, 0.5, 0.3, 0.5, 1.8, 0.2, 0.3, 0.2, 1.5]);

        let cov2: Matrix<3, 9> = Matrix::new([1.9, 0.4, 0.2, 0.4, 2.1, 0.3, 0.2, 0.3, 1.6]);

        ts.fit(&cov1).unwrap();

        let vec1 = ts.transform(&cov1).unwrap();
        let vec2 = ts.transform(&cov2).unwrap();

        // Both should be 6-dimensional vectors
        assert_eq!(vec1.len(), 6);
        assert_eq!(vec2.len(), 6);

        // Vectors should be different
        let mut different = false;
        for i in 0..6 {
            if (vec1[i] - vec2[i]).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Different matrices should produce different vectors"
        );
    }

    // --- Riemannian distance tests ---

    #[test]
    fn test_riemannian_distance_identity_to_diagonal() {
        // d(I, diag(a,b)) = sqrt(log(a)^2 + log(b)^2)
        let eye: Matrix<2, 4> = Matrix::identity();
        let mut diag: Matrix<2, 4> = Matrix::zeros();
        diag.set(0, 0, 2.0);
        diag.set(1, 1, 3.0);

        let d = riemannian_distance(&eye, &diag).unwrap();
        let expected =
            libm::sqrt(libm::log(2.0) * libm::log(2.0) + libm::log(3.0) * libm::log(3.0));
        assert!(
            (d - expected).abs() < 1e-9,
            "d={}, expected={}",
            d,
            expected
        );
    }

    #[test]
    fn test_riemannian_distance_self_is_zero() {
        let cov: Matrix<3, 9> = Matrix::new([2.0, 0.5, 0.3, 0.5, 1.8, 0.2, 0.3, 0.2, 1.5]);
        let d = riemannian_distance(&cov, &cov).unwrap();
        assert!(d < 1e-9, "d(A,A) should be 0, got {}", d);
    }

    #[test]
    fn test_riemannian_distance_symmetry() {
        let a: Matrix<2, 4> = Matrix::new([2.0, 0.3, 0.3, 1.5]);
        let b: Matrix<2, 4> = Matrix::new([1.5, 0.2, 0.2, 2.5]);

        let d_ab = riemannian_distance(&a, &b).unwrap();
        let d_ba = riemannian_distance(&b, &a).unwrap();
        assert!(
            (d_ab - d_ba).abs() < 1e-9,
            "d(A,B)={} != d(B,A)={}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_riemannian_distance_triangle_inequality() {
        let a: Matrix<2, 4> = Matrix::new([2.0, 0.3, 0.3, 1.5]);
        let b: Matrix<2, 4> = Matrix::new([1.5, 0.2, 0.2, 2.5]);
        let c: Matrix<2, 4> = Matrix::new([3.0, 0.1, 0.1, 1.2]);

        let d_ab = riemannian_distance(&a, &b).unwrap();
        let d_bc = riemannian_distance(&b, &c).unwrap();
        let d_ac = riemannian_distance(&a, &c).unwrap();

        assert!(
            d_ac <= d_ab + d_bc + 1e-9,
            "Triangle inequality violated: d(A,C)={} > d(A,B)+d(B,C)={}",
            d_ac,
            d_ab + d_bc
        );
    }

    // --- Frechet mean tests ---

    #[test]
    fn test_frechet_mean_identical_matrices() {
        let cov: Matrix<3, 9> = Matrix::new([2.0, 0.5, 0.3, 0.5, 1.8, 0.2, 0.3, 0.2, 1.5]);
        let matrices = [&cov, &cov, &cov];
        let mut output = Matrix::zeros();

        let iters = frechet_mean(&matrices, &mut output, 20, 1e-8).unwrap();
        assert!(iters < 5, "Should converge quickly for identical matrices");

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (output.get(i, j) - cov.get(i, j)).abs() < 1e-6,
                    "Mean of identical matrices should equal the matrix"
                );
            }
        }
    }

    #[test]
    fn test_frechet_mean_identity_matrices() {
        let eye: Matrix<3, 9> = Matrix::identity();
        let matrices = [&eye, &eye, &eye, &eye];
        let mut output = Matrix::zeros();

        frechet_mean(&matrices, &mut output, 20, 1e-8).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (output.get(i, j) - expected).abs() < 1e-6,
                    "Mean of identity matrices should be identity"
                );
            }
        }
    }

    #[test]
    fn test_frechet_mean_convergence_8x8() {
        // Build distinct SPD matrices
        let mut mats: [Matrix<8, 64>; 4] = [Matrix::zeros(); 4];
        for (idx, mat) in mats.iter_mut().enumerate() {
            *mat = Matrix::identity();
            for i in 0..8 {
                mat.set(i, i, 1.0 + 0.5 * ((i + idx) as f64));
            }
            // Add small off-diagonal
            for i in 0..7 {
                let off = 0.1 * libm::sin((i + idx + 1) as f64);
                mat.set(i, i + 1, off);
                mat.set(i + 1, i, off);
            }
        }

        let refs = [&mats[0], &mats[1], &mats[2], &mats[3]];
        let mut output = Matrix::zeros();

        let iters = frechet_mean(&refs, &mut output, 50, 1e-8).unwrap();
        assert!(iters <= 50, "Should converge within 50 iterations");

        // Verify output is SPD: eigenvalues all positive
        let eigen = output.eigen_symmetric(30, 1e-10).unwrap();
        for i in 0..8 {
            assert!(
                eigen.eigenvalues[i] > 0.0,
                "Mean should be SPD, eigenvalue {} = {}",
                i,
                eigen.eigenvalues[i]
            );
        }
    }

    // --- MDM classify tests ---

    #[test]
    fn test_mdm_classify_2class() {
        // Class 0: high variance in channel 0
        let class0: Matrix<2, 4> = Matrix::new([3.0, 0.0, 0.0, 1.0]);
        // Class 1: high variance in channel 1
        let class1: Matrix<2, 4> = Matrix::new([1.0, 0.0, 0.0, 3.0]);

        let means = [&class0, &class1];

        // Trial close to class 0
        let trial0: Matrix<2, 4> = Matrix::new([2.8, 0.0, 0.0, 1.1]);
        let (cls, _) = mdm_classify(&trial0, &means).unwrap();
        assert_eq!(cls, 0, "Should classify as class 0");

        // Trial close to class 1
        let trial1: Matrix<2, 4> = Matrix::new([1.1, 0.0, 0.0, 2.8]);
        let (cls, _) = mdm_classify(&trial1, &means).unwrap();
        assert_eq!(cls, 1, "Should classify as class 1");
    }

    #[test]
    fn test_mdm_classify_exact_match() {
        let class0: Matrix<2, 4> = Matrix::new([2.0, 0.0, 0.0, 1.0]);
        let class1: Matrix<2, 4> = Matrix::new([1.0, 0.0, 0.0, 2.0]);
        let means = [&class0, &class1];

        let (cls, dist) = mdm_classify(&class0, &means).unwrap();
        assert_eq!(cls, 0);
        assert!(dist < 1e-9, "Distance to exact match should be ~0");
    }
}
