//! Linear algebra primitives for BCI applications.
//!
//! Provides matrix operations and decompositions needed for Common Spatial Patterns (CSP)
//! and other multivariate BCI methods. All operations are no_std compatible with
//! zero heap allocation.
//!
//! # Matrix Storage
//!
//! Matrices are stored in row-major order as flat arrays. For a C×C matrix:
//! - Element at row i, column j is stored at index `i * C + j`
//! - This layout is cache-friendly for row-wise operations
//!
//! # Numerical Stability
//!
//! All decompositions include regularization options and error handling for
//! ill-conditioned matrices. Use appropriate regularization (e.g., 1e-6) when
//! working with covariance matrices.

#![allow(dead_code)] // Remove once CSP module uses these

use core::f64;
use libm;

/// Errors that can occur during matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinalgError {
    /// Matrix is not positive definite (Cholesky failed)
    NotPositiveDefinite,

    /// Eigenvalue decomposition did not converge
    ConvergenceFailed,

    /// Numerical instability detected
    NumericalInstability,

    /// Matrix dimensions incompatible
    DimensionMismatch,
}

/// A square matrix stored in row-major order.
///
/// # Type Parameters
///
/// * `C` - Matrix dimension (C×C matrix)
/// * `M` - Total number of elements (must equal C×C)
///
/// # Example
///
/// ```
/// use zerostone::linalg::Matrix;
///
/// // Create 2×2 identity matrix
/// let eye: Matrix<2, 4> = Matrix::identity();
/// assert_eq!(eye.get(0, 0), 1.0);
/// assert_eq!(eye.get(0, 1), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<const C: usize, const M: usize> {
    /// Matrix data in row-major order
    data: [f64; M],
}

impl<const C: usize, const M: usize> Matrix<C, M> {
    /// Create a new matrix from a flat array in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if M ≠ C×C.
    pub const fn new(data: [f64; M]) -> Self {
        assert!(M == C * C, "Matrix size M must equal C * C");
        Self { data }
    }

    /// Create a zero matrix.
    pub const fn zeros() -> Self {
        Self { data: [0.0; M] }
    }

    /// Create an identity matrix (ones on diagonal, zeros elsewhere).
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..C {
            m.data[i * C + i] = 1.0;
        }
        m
    }

    /// Get element at row i, column j.
    ///
    /// # Panics
    ///
    /// Panics if i >= C or j >= C.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < C && j < C, "Index out of bounds");
        self.data[i * C + j]
    }

    /// Set element at row i, column j.
    ///
    /// # Panics
    ///
    /// Panics if i >= C or j >= C.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        assert!(i < C && j < C, "Index out of bounds");
        self.data[i * C + j] = value;
    }

    /// Get reference to the underlying data array.
    pub fn data(&self) -> &[f64; M] {
        &self.data
    }

    /// Create matrix from covariance array (same format).
    pub fn from_array(data: [f64; M]) -> Self {
        Self::new(data)
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros();
        for i in 0..C {
            for j in 0..C {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Add a constant to the diagonal (for regularization).
    ///
    /// This is used to improve numerical stability: A + λI
    pub fn add_diagonal(&mut self, lambda: f64) {
        for i in 0..C {
            self.data[i * C + i] += lambda;
        }
    }

    /// Compute matrix multiplication: self × other.
    ///
    /// # Performance
    ///
    /// O(C³) - standard triple nested loop.
    pub fn matmul(&self, other: &Self) -> Self {
        let mut result = Self::zeros();

        for i in 0..C {
            for j in 0..C {
                let mut sum = 0.0;
                for k in 0..C {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }

    /// Add two matrices element-wise.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..M {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Self {
        let mut result = *self;
        for i in 0..M {
            result.data[i] *= scalar;
        }
        result
    }

    /// Cholesky decomposition: A = LL^T where L is lower triangular.
    ///
    /// Uses the Cholesky-Banachiewicz algorithm. Matrix must be symmetric
    /// and positive definite.
    ///
    /// # Returns
    ///
    /// Lower triangular matrix L such that A = LL^T.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::NotPositiveDefinite` if matrix is not positive definite.
    ///
    /// # Performance
    ///
    /// O(C³/3) operations.
    pub fn cholesky(&self) -> Result<Self, LinalgError> {
        let mut l = Self::zeros();

        for j in 0..C {
            for i in j..C {
                let mut sum = self.get(i, j);

                // Subtract sum of L[i,k] * L[j,k] for k < j
                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }

                if i == j {
                    // Diagonal element
                    if sum <= 0.0 {
                        return Err(LinalgError::NotPositiveDefinite);
                    }
                    l.set(i, j, libm::sqrt(sum));
                } else {
                    // Off-diagonal element
                    let l_jj = l.get(j, j);
                    if libm::fabs(l_jj) < 1e-15 {
                        return Err(LinalgError::NotPositiveDefinite);
                    }
                    l.set(i, j, sum / l_jj);
                }
            }
        }

        Ok(l)
    }

    /// Forward substitution: solve Lx = b where L is lower triangular.
    ///
    /// # Panics
    ///
    /// Panics if L has zero diagonal elements.
    pub fn forward_substitute(&self, b: &[f64; C]) -> [f64; C] {
        let mut x = [0.0; C];

        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.get(i, j) * x[j];
            }
            let l_ii = self.get(i, i);
            assert!(libm::fabs(l_ii) > 1e-15, "Singular matrix in forward substitution");
            x[i] = sum / l_ii;
        }

        x
    }

    /// Backward substitution: solve L^T x = b where L is lower triangular.
    ///
    /// # Panics
    ///
    /// Panics if L has zero diagonal elements.
    pub fn backward_substitute(&self, b: &[f64; C]) -> [f64; C] {
        let mut x = [0.0; C];

        #[allow(clippy::needless_range_loop)]
        for i in (0..C).rev() {
            let mut sum = b[i];
            for j in (i + 1)..C {
                sum -= self.get(j, i) * x[j];  // Note: transpose indexing
            }
            let l_ii = self.get(i, i);
            assert!(libm::fabs(l_ii) > 1e-15, "Singular matrix in backward substitution");
            x[i] = sum / l_ii;
        }

        x
    }

    /// Solve Ax = b using Cholesky decomposition.
    ///
    /// More efficient than direct inversion when A is positive definite.
    ///
    /// # Errors
    ///
    /// Returns error if Cholesky decomposition fails.
    pub fn solve(&self, b: &[f64; C]) -> Result<[f64; C], LinalgError> {
        let l = self.cholesky()?;
        let y = l.forward_substitute(b);
        Ok(l.backward_substitute(&y))
    }

    /// Eigenvalue decomposition using Jacobi iteration for symmetric matrices.
    ///
    /// Computes eigenvalues and eigenvectors such that A = V Λ V^T.
    ///
    /// # Arguments
    ///
    /// * `max_iters` - Maximum number of Jacobi sweeps (recommended: 30 for C=32)
    /// * `tol` - Convergence tolerance for off-diagonal elements (recommended: 1e-10)
    ///
    /// # Returns
    ///
    /// Eigenvalues in descending order with corresponding eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns `LinalgError::ConvergenceFailed` if convergence not reached within max_iters.
    ///
    /// # Performance
    ///
    /// O(C³) per sweep, typically 10-30 sweeps needed.
    pub fn eigen_symmetric(
        &self,
        max_iters: usize,
        tol: f64,
    ) -> Result<EigenDecomposition<C, M>, LinalgError> {
        // Copy matrix (will be modified to diagonal form)
        let mut a = *self;

        // Initialize eigenvectors to identity
        let mut v = Self::identity();

        for _sweep in 0..max_iters {
            // Find maximum off-diagonal element
            let (max_i, max_j, max_val) = find_max_off_diagonal(&a);

            // Check convergence
            if max_val < tol {
                // Extract eigenvalues and eigenvectors
                let mut eigenvalues = [0.0; C];
                #[allow(clippy::needless_range_loop)]
                for i in 0..C {
                    eigenvalues[i] = a.get(i, i);
                }

                // Sort eigenvalues and eigenvectors in descending order
                sort_eigen(&mut eigenvalues, &mut v);

                return Ok(EigenDecomposition {
                    eigenvalues,
                    eigenvectors: v,
                });
            }

            // Compute Jacobi rotation to annihilate A[max_i, max_j]
            let (cos_theta, sin_theta) = compute_jacobi_rotation(&a, max_i, max_j);

            // Apply rotation to A: A' = R^T A R
            apply_jacobi_rotation(&mut a, max_i, max_j, cos_theta, sin_theta);

            // Update eigenvectors: V' = V R
            apply_rotation_to_vectors(&mut v, max_i, max_j, cos_theta, sin_theta);
        }

        // Did not converge
        Err(LinalgError::ConvergenceFailed)
    }
}

/// Result of eigenvalue decomposition.
#[derive(Debug, Clone, Copy)]
pub struct EigenDecomposition<const C: usize, const M: usize> {
    /// Eigenvalues in descending order
    pub eigenvalues: [f64; C],

    /// Eigenvectors as columns (eigenvectors.get(row, col) = col-th eigenvector, row-th element)
    pub eigenvectors: Matrix<C, M>,
}

/// Find the maximum off-diagonal element in a symmetric matrix.
fn find_max_off_diagonal<const C: usize, const M: usize>(
    a: &Matrix<C, M>,
) -> (usize, usize, f64) {
    let mut max_i = 0;
    let mut max_j = 1;
    let mut max_val = a.get(0, 1).abs();

    for i in 0..C {
        for j in (i + 1)..C {
            let val = a.get(i, j).abs();
            if val > max_val {
                max_val = val;
                max_i = i;
                max_j = j;
            }
        }
    }

    (max_i, max_j, max_val)
}

/// Compute Jacobi rotation parameters to annihilate A[i,j].
fn compute_jacobi_rotation<const C: usize, const M: usize>(
    a: &Matrix<C, M>,
    i: usize,
    j: usize,
) -> (f64, f64) {
    let a_ii = a.get(i, i);
    let a_jj = a.get(j, j);
    let a_ij = a.get(i, j);

    if a_ii == a_jj {
        // Special case: diagonal elements equal
        let cos_theta = libm::cos(f64::consts::PI / 4.0);
        let sin_theta = libm::sin(f64::consts::PI / 4.0);
        return (cos_theta, sin_theta);
    }

    // Compute rotation angle
    let tau = (a_jj - a_ii) / (2.0 * a_ij);
    let t = if tau >= 0.0 {
        1.0 / (tau + libm::sqrt(1.0 + tau * tau))
    } else {
        -1.0 / (-tau + libm::sqrt(1.0 + tau * tau))
    };

    let cos_theta = 1.0 / libm::sqrt(1.0 + t * t);
    let sin_theta = t * cos_theta;

    (cos_theta, sin_theta)
}

/// Apply Jacobi rotation to matrix A.
fn apply_jacobi_rotation<const C: usize, const M: usize>(
    a: &mut Matrix<C, M>,
    i: usize,
    j: usize,
    cos_theta: f64,
    sin_theta: f64,
) {
    // Update diagonal elements
    let a_ii = a.get(i, i);
    let a_jj = a.get(j, j);
    let a_ij = a.get(i, j);

    let new_a_ii = cos_theta * cos_theta * a_ii - 2.0 * cos_theta * sin_theta * a_ij
        + sin_theta * sin_theta * a_jj;
    let new_a_jj = sin_theta * sin_theta * a_ii + 2.0 * cos_theta * sin_theta * a_ij
        + cos_theta * cos_theta * a_jj;

    a.set(i, i, new_a_ii);
    a.set(j, j, new_a_jj);
    a.set(i, j, 0.0);
    a.set(j, i, 0.0);

    // Update off-diagonal elements in rows/columns i and j
    for k in 0..C {
        if k != i && k != j {
            let a_ki = a.get(k, i);
            let a_kj = a.get(k, j);

            let new_a_ki = cos_theta * a_ki - sin_theta * a_kj;
            let new_a_kj = sin_theta * a_ki + cos_theta * a_kj;

            a.set(k, i, new_a_ki);
            a.set(i, k, new_a_ki); // Symmetric
            a.set(k, j, new_a_kj);
            a.set(j, k, new_a_kj); // Symmetric
        }
    }
}

/// Apply rotation to eigenvector matrix.
fn apply_rotation_to_vectors<const C: usize, const M: usize>(
    v: &mut Matrix<C, M>,
    i: usize,
    j: usize,
    cos_theta: f64,
    sin_theta: f64,
) {
    for k in 0..C {
        let v_ki = v.get(k, i);
        let v_kj = v.get(k, j);

        let new_v_ki = cos_theta * v_ki - sin_theta * v_kj;
        let new_v_kj = sin_theta * v_ki + cos_theta * v_kj;

        v.set(k, i, new_v_ki);
        v.set(k, j, new_v_kj);
    }
}

/// Sort eigenvalues in descending order and reorder corresponding eigenvectors.
fn sort_eigen<const C: usize, const M: usize>(
    eigenvalues: &mut [f64; C],
    eigenvectors: &mut Matrix<C, M>,
) {
    // Simple selection sort (adequate for small C)
    for i in 0..C {
        let mut max_idx = i;
        for j in (i + 1)..C {
            if eigenvalues[j] > eigenvalues[max_idx] {
                max_idx = j;
            }
        }

        if max_idx != i {
            // Swap eigenvalues
            eigenvalues.swap(i, max_idx);

            // Swap eigenvector columns
            for k in 0..C {
                let tmp = eigenvectors.get(k, i);
                eigenvectors.set(k, i, eigenvectors.get(k, max_idx));
                eigenvectors.set(k, max_idx, tmp);
            }
        }
    }
}

impl<const C: usize, const M: usize> EigenDecomposition<C, M> {
    /// Get the k-th eigenvector as an array.
    pub fn eigenvector(&self, k: usize) -> [f64; C] {
        let mut vec = [0.0; C];
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            vec[i] = self.eigenvectors.get(i, k);
        }
        vec
    }
}

/// Solve generalized eigenvalue problem: A w = λ B w.
///
/// Uses Cholesky whitening transformation to convert to standard eigenvalue problem.
///
/// # Arguments
///
/// * `a` - First symmetric positive semi-definite matrix
/// * `b` - Second symmetric positive definite matrix
/// * `regularization` - Added to B diagonal for numerical stability (e.g., 1e-6)
/// * `max_iters` - Maximum Jacobi iterations
/// * `tol` - Convergence tolerance
///
/// # Algorithm
///
/// 1. Compute B_reg = B + λI (regularized)
/// 2. Cholesky: B_reg = LL^T
/// 3. Compute L^{-1} via triangular solve
/// 4. Transform: P = L^{-1} A L^{-T}
/// 5. Standard EVD: P v = μ v
/// 6. Transform back: w = L^{-T} v
///
/// # Errors
///
/// Returns error if Cholesky fails or eigenvalue decomposition fails.
pub fn generalized_eigen<const C: usize, const M: usize>(
    a: &Matrix<C, M>,
    b: &Matrix<C, M>,
    regularization: f64,
    max_iters: usize,
    tol: f64,
) -> Result<EigenDecomposition<C, M>, LinalgError> {
    // Step 1: Regularize B
    let mut b_reg = *b;
    b_reg.add_diagonal(regularization);

    // Step 2: Cholesky decomposition of B
    let l = b_reg.cholesky()?;

    // Step 3: Compute L^{-1} by solving L × L_inv = I
    let mut l_inv = Matrix::zeros();
    for i in 0..C {
        let mut e_i = [0.0; C];
        e_i[i] = 1.0;
        let col_i = l.forward_substitute(&e_i);
        #[allow(clippy::needless_range_loop)]
        for j in 0..C {
            l_inv.set(j, i, col_i[j]);
        }
    }

    // Step 4: Compute L^{-T}
    let l_inv_t = l_inv.transpose();

    // Step 5: Transform A: P = L^{-1} × A × L^{-T}
    let temp = l_inv.matmul(a);
    let p = temp.matmul(&l_inv_t);

    // Step 6: Standard eigenvalue decomposition of P
    let mut eigen_p = p.eigen_symmetric(max_iters, tol)?;

    // Step 7: Transform eigenvectors back: w = L^{-T} × v
    let w = l_inv_t.matmul(&eigen_p.eigenvectors);

    // Normalize eigenvectors
    let mut w_normalized = w;
    for j in 0..C {
        let mut norm_sq = 0.0;
        for i in 0..C {
            let val = w_normalized.get(i, j);
            norm_sq += val * val;
        }
        let norm = libm::sqrt(norm_sq);
        if norm > 1e-15 {
            for i in 0..C {
                let val = w_normalized.get(i, j);
                w_normalized.set(i, j, val / norm);
            }
        }
    }

    eigen_p.eigenvectors = w_normalized;
    Ok(eigen_p)
}

impl<const C: usize, const M: usize> Default for Matrix<C, M> {
    fn default() -> Self {
        Self::zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_identity() {
        let eye: Matrix<3, 9> = Matrix::identity();

        // Check diagonal
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);

        // Check off-diagonal
        assert_eq!(eye.get(0, 1), 0.0);
        assert_eq!(eye.get(1, 0), 0.0);
        assert_eq!(eye.get(1, 2), 0.0);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m: Matrix<2, 4> = Matrix::zeros();
        m.set(0, 1, 5.0);
        m.set(1, 0, 3.0);

        assert_eq!(m.get(0, 1), 5.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m: Matrix<2, 4> = Matrix::zeros();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);

        let t = m.transpose();

        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 3.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_add_diagonal() {
        let mut m: Matrix<3, 9> = Matrix::identity();
        m.add_diagonal(0.5);

        assert_eq!(m.get(0, 0), 1.5);
        assert_eq!(m.get(1, 1), 1.5);
        assert_eq!(m.get(2, 2), 1.5);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        let m: Matrix<2, 4> = Matrix::new([1.0, 2.0, 3.0, 4.0]);
        let eye: Matrix<2, 4> = Matrix::identity();

        let result = m.matmul(&eye);

        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 2.0);
        assert_eq!(result.get(1, 0), 3.0);
        assert_eq!(result.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_multiply() {
        // | 1 2 |   | 5 6 |   | 19 22 |
        // | 3 4 | × | 7 8 | = | 43 50 |
        let a: Matrix<2, 4> = Matrix::new([1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<2, 4> = Matrix::new([5.0, 6.0, 7.0, 8.0]);

        let c = a.matmul(&b);

        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_matrix_add() {
        let a: Matrix<2, 4> = Matrix::new([1.0, 2.0, 3.0, 4.0]);
        let b: Matrix<2, 4> = Matrix::new([5.0, 6.0, 7.0, 8.0]);

        let c = a.add(&b);

        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(0, 1), 8.0);
        assert_eq!(c.get(1, 0), 10.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_matrix_scale() {
        let a: Matrix<2, 4> = Matrix::new([1.0, 2.0, 3.0, 4.0]);
        let b = a.scale(2.0);

        assert_eq!(b.get(0, 0), 2.0);
        assert_eq!(b.get(0, 1), 4.0);
        assert_eq!(b.get(1, 0), 6.0);
        assert_eq!(b.get(1, 1), 8.0);
    }

    #[test]
    fn test_cholesky_identity() {
        let eye: Matrix<3, 9> = Matrix::identity();
        let l = eye.cholesky().unwrap();

        // Cholesky of identity is identity
        assert!((l.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((l.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((l.get(2, 2) - 1.0).abs() < 1e-10);
        assert!(l.get(0, 1).abs() < 1e-10);
        assert!(l.get(0, 2).abs() < 1e-10);
        assert!(l.get(1, 2).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_diagonal() {
        // Diagonal matrix [4, 0; 0, 9]
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, 4.0);
        a.set(1, 1, 9.0);

        let l = a.cholesky().unwrap();

        // L should be [2, 0; 0, 3]
        assert!((l.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((l.get(1, 1) - 3.0).abs() < 1e-10);
        assert!(l.get(0, 1).abs() < 1e-10);
        assert!(l.get(1, 0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_symmetric() {
        // Symmetric positive definite matrix
        // [4, 2]
        // [2, 3]
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, 4.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 3.0);

        let l = a.cholesky().unwrap();

        // Verify LL^T = A
        let lt = l.transpose();
        let reconstructed = l.matmul(&lt);

        assert!((reconstructed.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((reconstructed.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((reconstructed.get(1, 0) - 2.0).abs() < 1e-10);
        assert!((reconstructed.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Negative diagonal - not positive definite
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, -1.0);
        a.set(1, 1, 1.0);

        let result = a.cholesky();
        assert!(matches!(result, Err(LinalgError::NotPositiveDefinite)));
    }

    #[test]
    fn test_forward_substitute() {
        // Lower triangular matrix L
        // [2, 0]
        // [3, 4]
        let mut l: Matrix<2, 4> = Matrix::zeros();
        l.set(0, 0, 2.0);
        l.set(1, 0, 3.0);
        l.set(1, 1, 4.0);

        // Solve Lx = [2, 5]
        let b = [2.0, 5.0];
        let x = l.forward_substitute(&b);

        // x should be [1, 0.5]
        // L[0,0]*x[0] = 2 -> x[0] = 1
        // L[1,0]*x[0] + L[1,1]*x[1] = 5 -> 3*1 + 4*x[1] = 5 -> x[1] = 0.5
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_backward_substitute() {
        // Lower triangular matrix L (will use L^T for backward)
        // [2, 0]
        // [3, 4]
        let mut l: Matrix<2, 4> = Matrix::zeros();
        l.set(0, 0, 2.0);
        l.set(1, 0, 3.0);
        l.set(1, 1, 4.0);

        // Solve L^T x = [10, 4]
        // L^T = [2, 3]
        //       [0, 4]
        let b = [10.0, 4.0];
        let x = l.backward_substitute(&b);

        // Working backward:
        // L^T[1,1]*x[1] = 4 -> 4*x[1] = 4 -> x[1] = 1
        // L^T[0,0]*x[0] + L^T[0,1]*x[1] = 10 -> 2*x[0] + 3*1 = 10 -> x[0] = 3.5
        assert!((x[0] - 3.5).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve() {
        // Solve Ax = b for symmetric positive definite A
        // A = [4, 2]
        //     [2, 3]
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, 4.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 3.0);

        let b = [8.0, 7.0];
        let x = a.solve(&b).unwrap();

        // Verify Ax = b
        let ax = [
            a.get(0, 0) * x[0] + a.get(0, 1) * x[1],
            a.get(1, 0) * x[0] + a.get(1, 1) * x[1],
        ];

        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_eigen_identity() {
        let eye: Matrix<3, 9> = Matrix::identity();
        let eigen = eye.eigen_symmetric(30, 1e-10).unwrap();

        // All eigenvalues should be 1
        for i in 0..3 {
            assert!((eigen.eigenvalues[i] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_eigen_diagonal() {
        // Diagonal matrix with eigenvalues [5, 3, 1]
        let mut a: Matrix<3, 9> = Matrix::zeros();
        a.set(0, 0, 5.0);
        a.set(1, 1, 3.0);
        a.set(2, 2, 1.0);

        let eigen = a.eigen_symmetric(30, 1e-10).unwrap();

        // Eigenvalues should be [5, 3, 1] in descending order
        assert!((eigen.eigenvalues[0] - 5.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[1] - 3.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[2] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_eigen_symmetric_2x2() {
        // Symmetric matrix [3, 1]
        //                  [1, 3]
        // Eigenvalues: 4 and 2
        // Eigenvectors: [1/√2, 1/√2] and [1/√2, -1/√2]
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, 3.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);

        let eigen = a.eigen_symmetric(30, 1e-10).unwrap();

        // Check eigenvalues
        assert!((eigen.eigenvalues[0] - 4.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[1] - 2.0).abs() < 1e-8);

        // Verify Av = λv for first eigenvector
        let v0 = eigen.eigenvector(0);
        let av0 = [
            a.get(0, 0) * v0[0] + a.get(0, 1) * v0[1],
            a.get(1, 0) * v0[0] + a.get(1, 1) * v0[1],
        ];
        let lambda_v0 = [
            eigen.eigenvalues[0] * v0[0],
            eigen.eigenvalues[0] * v0[1],
        ];

        assert!((av0[0] - lambda_v0[0]).abs() < 1e-8);
        assert!((av0[1] - lambda_v0[1]).abs() < 1e-8);
    }

    #[test]
    fn test_eigen_sorting() {
        // Matrix with eigenvalues in non-descending order
        let mut a: Matrix<4, 16> = Matrix::zeros();
        a.set(0, 0, 1.0);
        a.set(1, 1, 4.0);
        a.set(2, 2, 2.0);
        a.set(3, 3, 3.0);

        let eigen = a.eigen_symmetric(30, 1e-10).unwrap();

        // Should be sorted descending: [4, 3, 2, 1]
        assert!((eigen.eigenvalues[0] - 4.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[1] - 3.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[2] - 2.0).abs() < 1e-8);
        assert!((eigen.eigenvalues[3] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_eigen_orthogonal_eigenvectors() {
        // Symmetric matrix
        let mut a: Matrix<3, 9> = Matrix::zeros();
        a.set(0, 0, 4.0);
        a.set(0, 1, 1.0);
        a.set(0, 2, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 4.0);
        a.set(1, 2, 1.0);
        a.set(2, 0, 1.0);
        a.set(2, 1, 1.0);
        a.set(2, 2, 4.0);

        let eigen = a.eigen_symmetric(30, 1e-10).unwrap();

        // Check orthogonality: v_i · v_j = 0 for i ≠ j
        for i in 0..3 {
            for j in (i + 1)..3 {
                let vi = eigen.eigenvector(i);
                let vj = eigen.eigenvector(j);
                let dot = vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2];
                assert!(libm::fabs(dot) < 1e-8, "Eigenvectors not orthogonal");
            }
        }

        // Check normalization: ||v_i|| = 1
        for i in 0..3 {
            let vi = eigen.eigenvector(i);
            let norm_sq = vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2];
            assert!((norm_sq - 1.0).abs() < 1e-8, "Eigenvector not normalized");
        }
    }

    #[test]
    fn test_generalized_eigen_simple() {
        // Simple 2x2 case: A = [2, 0; 0, 1], B = [1, 0; 0, 1] (identity)
        // Generalized eigenvalue problem reduces to standard: Aw = λw
        let mut a: Matrix<2, 4> = Matrix::zeros();
        a.set(0, 0, 2.0);
        a.set(1, 1, 1.0);

        let b: Matrix<2, 4> = Matrix::identity();

        let eigen = generalized_eigen(&a, &b, 1e-8, 30, 1e-10).unwrap();

        // Eigenvalues should be [2, 1]
        assert!((eigen.eigenvalues[0] - 2.0).abs() < 1e-6);
        assert!((eigen.eigenvalues[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generalized_eigen_csp_like() {
        // CSP-like scenario: two class covariance matrices
        // C1 has variance in first channel
        let mut c1: Matrix<2, 4> = Matrix::identity();
        c1.set(0, 0, 3.0); // Higher variance in channel 0

        // C2 has variance in second channel
        let mut c2: Matrix<2, 4> = Matrix::identity();
        c2.set(1, 1, 3.0); // Higher variance in channel 1

        // Solve C1 w = λ (C1 + C2) w
        let c_sum = c1.add(&c2);
        let eigen = generalized_eigen(&c1, &c_sum, 1e-6, 30, 1e-10).unwrap();

        // Top eigenvector should emphasize channel 0
        // Bottom eigenvector should emphasize channel 1
        let v0 = eigen.eigenvector(0);
        let v1 = eigen.eigenvector(1);

        // Check that v0 has larger weight on channel 0
        assert!(v0[0].abs() > v0[1].abs(), "Top eigenvector should favor channel 0");

        // Check that v1 has larger weight on channel 1
        assert!(v1[1].abs() > v1[0].abs(), "Bottom eigenvector should favor channel 1");
    }

    #[test]
    fn test_generalized_eigen_orthogonality() {
        // Test that generalized eigenvectors are B-orthogonal
        let mut a: Matrix<3, 9> = Matrix::identity();
        a.set(0, 0, 4.0);
        a.set(1, 1, 2.0);
        a.set(2, 2, 1.0);

        let b: Matrix<3, 9> = Matrix::identity();

        let eigen = generalized_eigen(&a, &b, 1e-8, 30, 1e-10).unwrap();

        // Eigenvectors should be orthonormal (since B = I)
        for i in 0..3 {
            let vi = eigen.eigenvector(i);
            let norm_sq = vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2];
            assert!((norm_sq - 1.0).abs() < 1e-6, "Eigenvector not normalized");
        }
    }
}
