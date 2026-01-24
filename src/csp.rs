//! Common Spatial Patterns (CSP) for motor imagery BCI.
//!
//! CSP is a supervised learning algorithm that finds spatial filters maximizing
//! the variance ratio between two classes of multi-channel signals. Essential for
//! motor imagery BCIs (e.g., left vs right hand imagination).
//!
//! # Algorithm Overview
//!
//! Given two classes of trials (C₁ and C₂), CSP solves the generalized eigenvalue problem:
//! ```text
//! C₁w = λ(C₁ + C₂)w
//! ```
//! The top eigenvectors maximize class 1 variance, bottom eigenvectors maximize class 2 variance.
//!
//! # Example
//!
//! ```
//! use zerostone::{AdaptiveCsp, UpdateConfig};
//!
//! // Create CSP with 4 channels, extract 2 filters (2×4 = 8 coefficients)
//! let mut csp: AdaptiveCsp<4, 16, 2, 8> = AdaptiveCsp::new(UpdateConfig::default());
//!
//! // Training: accumulate class 1 and class 2 data
//! # let left_hand_trial = [[0.0; 4]; 100];
//! # let right_hand_trial = [[0.0; 4]; 100];
//! csp.update_class1(&left_hand_trial);
//! csp.update_class2(&right_hand_trial);
//!
//! // Compute filters
//! csp.recompute_filters().unwrap();
//!
//! // Apply to new sample
//! let sample = [1.0, 2.0, 3.0, 4.0];
//! let features = csp.apply(&sample).unwrap();
//! ```
//!
//! # References
//!
//! - Ramoser et al. (2000): "Optimal spatial filtering of single trial EEG during imagined hand movement"
//! - Blankertz et al. (2008): "Optimizing spatial filters for robust EEG single-trial analysis"

use crate::linalg::{generalized_eigen, Matrix};
use crate::stats::OnlineCov;

/// Errors that can occur during CSP operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CspError {
    /// Not enough data to compute filters
    InsufficientData,

    /// Filters not yet computed
    FiltersNotReady,

    /// Eigenvalue decomposition failed
    EigenDecompositionFailed,

    /// Matrix is not positive definite
    NotPositiveDefinite,

    /// Numerical instability detected
    NumericalInstability,
}

/// Configuration for CSP filter updates.
#[derive(Debug, Clone, Copy)]
pub struct UpdateConfig {
    /// Minimum samples per class before first computation
    pub min_samples: u64,

    /// Trigger update every N samples (0 = never auto-update)
    pub update_interval: u64,

    /// Regularization parameter for covariance matrices
    pub regularization: f64,

    /// Max iterations for eigenvalue decomposition
    pub max_eigen_iters: usize,

    /// Convergence tolerance for eigenvalue decomposition
    pub eigen_tol: f64,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,     // 100 trials minimum
            update_interval: 50,  // Update every 50 new trials
            regularization: 1e-6, // Tikhonov regularization
            max_eigen_iters: 30,  // 30 Jacobi sweeps
            eigen_tol: 1e-10,     // Convergence threshold
        }
    }
}

/// Online adaptive Common Spatial Patterns filter.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Matrix size (must equal C × C)
/// * `K` - Number of spatial filters to extract
/// * `F` - Filter storage size (must equal K × C)
///
/// # Example
///
/// ```
/// use zerostone::{AdaptiveCsp, UpdateConfig};
///
/// // 8 channels, extract 4 CSP filters (4 × 8 = 32 filter coefficients)
/// let mut csp: AdaptiveCsp<8, 64, 4, 32> = AdaptiveCsp::new(UpdateConfig {
///     min_samples: 50,
///     update_interval: 0,  // Manual updates only
///     ..Default::default()
/// });
/// ```
pub struct AdaptiveCsp<const C: usize, const M: usize, const K: usize, const F: usize> {
    /// Covariance estimator for class 1 (e.g., left hand MI)
    cov1: OnlineCov<C, M>,

    /// Covariance estimator for class 2 (e.g., right hand MI)
    cov2: OnlineCov<C, M>,

    /// Current spatial filters (K filters × C channels)
    /// Stored row-major: filters[k*C + c] is k-th filter, c-th channel
    filters: [f64; F],

    /// Update trigger strategy
    update_config: UpdateConfig,

    /// Samples accumulated since last filter update
    samples_since_update: u64,

    /// Flag indicating if filters have been computed
    filters_ready: bool,
}

impl<const C: usize, const M: usize, const K: usize, const F: usize> AdaptiveCsp<C, M, K, F> {
    /// Create new adaptive CSP filter.
    ///
    /// # Panics
    ///
    /// Panics if M ≠ C×C or F ≠ K×C or K > C.
    pub const fn new(config: UpdateConfig) -> Self {
        assert!(M == C * C, "M must equal C * C");
        assert!(F == K * C, "F must equal K * C");
        assert!(K <= C, "Cannot extract more filters than channels");
        assert!(K > 0, "Must extract at least one filter");

        Self {
            cov1: OnlineCov::new(),
            cov2: OnlineCov::new(),
            filters: [0.0; F],
            update_config: config,
            samples_since_update: 0,
            filters_ready: false,
        }
    }

    /// Update with class 1 trial.
    ///
    /// # Arguments
    ///
    /// * `trial` - Multi-channel time series \[samples\]\[channels\]
    pub fn update_class1(&mut self, trial: &[[f64; C]]) {
        for sample in trial {
            self.cov1.update(sample);
        }
        self.samples_since_update += trial.len() as u64;
        self.check_and_update();
    }

    /// Update with class 2 trial.
    pub fn update_class2(&mut self, trial: &[[f64; C]]) {
        for sample in trial {
            self.cov2.update(sample);
        }
        self.samples_since_update += trial.len() as u64;
        self.check_and_update();
    }

    /// Manually trigger filter recomputation.
    ///
    /// # Errors
    ///
    /// Returns error if insufficient data or eigendecomposition fails.
    pub fn recompute_filters(&mut self) -> Result<(), CspError> {
        if self.cov1.count() < self.update_config.min_samples {
            return Err(CspError::InsufficientData);
        }
        if self.cov2.count() < self.update_config.min_samples {
            return Err(CspError::InsufficientData);
        }

        // Get covariance matrices
        let c1 = Matrix::from_array(self.cov1.covariance());
        let c2 = Matrix::from_array(self.cov2.covariance());

        // Compute C_sum = C1 + C2
        let c_sum = c1.add(&c2);

        // Solve generalized eigenvalue problem: C1 w = λ (C1 + C2) w
        let eigen = generalized_eigen(
            &c1,
            &c_sum,
            self.update_config.regularization,
            self.update_config.max_eigen_iters,
            self.update_config.eigen_tol,
        )
        .map_err(|_| CspError::EigenDecompositionFailed)?;

        // Extract top K/2 and bottom K/2 eigenvectors
        self.extract_filters(&eigen)?;

        self.filters_ready = true;
        self.samples_since_update = 0;
        Ok(())
    }

    /// Apply spatial filters to a single sample.
    ///
    /// Returns K-dimensional feature vector.
    ///
    /// # Errors
    ///
    /// Returns error if filters not yet computed.
    pub fn apply(&self, sample: &[f64; C]) -> Result<[f64; K], CspError> {
        if !self.filters_ready {
            return Err(CspError::FiltersNotReady);
        }

        let mut output = [0.0; K];
        // Note: need indices here to compute filter offset k*C + c
        #[allow(clippy::needless_range_loop)]
        for k in 0..K {
            let mut sum = 0.0;
            for (c, &s) in sample.iter().enumerate() {
                sum += self.filters[k * C + c] * s;
            }
            output[k] = sum;
        }
        Ok(output)
    }

    /// Get current filter matrix (for inspection).
    ///
    /// Returns None if filters not yet computed.
    pub fn filters(&self) -> Option<&[f64; F]> {
        if self.filters_ready {
            Some(&self.filters)
        } else {
            None
        }
    }

    /// Get number of samples in class 1.
    pub fn class1_count(&self) -> u64 {
        self.cov1.count()
    }

    /// Get number of samples in class 2.
    pub fn class2_count(&self) -> u64 {
        self.cov2.count()
    }

    /// Check if filters are ready.
    pub fn is_ready(&self) -> bool {
        self.filters_ready
    }

    /// Reset all accumulators (clear training data).
    pub fn reset(&mut self) {
        self.cov1.reset();
        self.cov2.reset();
        self.samples_since_update = 0;
        self.filters_ready = false;
    }

    /// Check if automatic update should trigger.
    fn check_and_update(&mut self) {
        if self.update_config.update_interval == 0 {
            return; // Auto-update disabled
        }

        if self.samples_since_update >= self.update_config.update_interval {
            let _ = self.recompute_filters(); // Ignore errors in auto-update
        }
    }

    /// Extract CSP filters from eigendecomposition.
    fn extract_filters(
        &mut self,
        eigen: &crate::linalg::EigenDecomposition<C, M>,
    ) -> Result<(), CspError> {
        // Strategy: Extract top K/2 and bottom K/2 eigenvectors
        // Top eigenvectors maximize variance ratio for class 1
        // Bottom eigenvectors maximize variance ratio for class 2

        let half_k = K / 2;
        let remainder = K % 2;

        // Copy top eigenvectors
        for k in 0..(half_k + remainder) {
            for c in 0..C {
                self.filters[k * C + c] = eigen.eigenvectors.get(c, k);
            }
        }

        // Copy bottom eigenvectors
        for k in 0..half_k {
            let src_idx = C - half_k + k;
            let dst_idx = half_k + remainder + k;
            for c in 0..C {
                self.filters[dst_idx * C + c] = eigen.eigenvectors.get(c, src_idx);
            }
        }

        Ok(())
    }
}

impl<const C: usize, const M: usize, const K: usize, const F: usize> Default
    for AdaptiveCsp<C, M, K, F>
{
    fn default() -> Self {
        Self::new(UpdateConfig::default())
    }
}

/// Multi-class CSP using One-vs-Rest (OVR) decomposition.
///
/// For N classes, trains N sets of CSP filters where each set maximizes the
/// variance ratio between one class and all others combined. This extends
/// binary CSP to handle motor imagery tasks with more than 2 classes
/// (e.g., left hand, right hand, feet, tongue).
///
/// # Algorithm
///
/// For each class i:
/// 1. Compute covariance C_i from class i trials
/// 2. Compute "rest" covariance C_rest = Σ_{j≠i} C_j
/// 3. Solve generalized eigenvalue problem: C_i w = λ (C_i + C_rest) w
/// 4. Extract top K/2 and bottom K/2 eigenvectors as spatial filters
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Matrix size (must equal C × C)
/// * `K` - Number of spatial filters per class
/// * `F` - Filter storage per class (must equal K × C)
/// * `N` - Number of classes
/// * `T` - Total filter storage (must equal N × F = N × K × C)
/// * `O` - Output feature size (must equal N × K)
///
/// # Example
///
/// ```
/// use zerostone::{MulticlassCsp, UpdateConfig};
///
/// // 4-class motor imagery: 8 channels, 4 filters per class
/// // M = 8×8 = 64, F = 4×8 = 32, T = 4×32 = 128, O = 4×4 = 16
/// let mut csp: MulticlassCsp<8, 64, 4, 32, 4, 128, 16> = MulticlassCsp::new(UpdateConfig {
///     min_samples: 50,
///     update_interval: 0,
///     ..Default::default()
/// });
///
/// // Training: update each class with its trials
/// # let left_trial = [[0.0f64; 8]; 100];
/// # let right_trial = [[0.0f64; 8]; 100];
/// # let feet_trial = [[0.0f64; 8]; 100];
/// # let tongue_trial = [[0.0f64; 8]; 100];
/// csp.update_class(0, &left_trial);   // Left hand
/// csp.update_class(1, &right_trial);  // Right hand
/// csp.update_class(2, &feet_trial);   // Feet
/// csp.update_class(3, &tongue_trial); // Tongue
///
/// // Compute all OVR filters
/// csp.recompute_filters().unwrap();
///
/// // Apply to new sample - returns N×K = 16 features
/// let sample = [0.0f64; 8];
/// let features = csp.apply(&sample).unwrap();
/// assert_eq!(features.len(), 16);
/// ```
///
/// # References
///
/// - Ang et al. (2008): "Filter Bank Common Spatial Pattern (FBCSP) in BCI"
/// - Grosse-Wentrup & Buss (2008): "Multiclass CSP and SVM for EEG-based BCI"
pub struct MulticlassCsp<
    const C: usize,
    const M: usize,
    const K: usize,
    const F: usize,
    const N: usize,
    const T: usize,
    const O: usize,
> {
    /// Covariance estimator for each class
    class_cov: [OnlineCov<C, M>; N],

    /// CSP filters for all classes
    /// Indexed as: filters[class * F + filter * C + channel]
    /// Total size: N × K × C = T
    filters: [f64; T],

    /// Configuration for filter computation
    config: UpdateConfig,

    /// Samples accumulated since last filter update
    samples_since_update: u64,

    /// Whether filters have been computed
    filters_ready: bool,

    /// Number of samples per class (for tracking)
    class_counts: [u64; N],
}

impl<
        const C: usize,
        const M: usize,
        const K: usize,
        const F: usize,
        const N: usize,
        const T: usize,
        const O: usize,
    > MulticlassCsp<C, M, K, F, N, T, O>
{
    /// Create a new multi-class CSP filter.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - M ≠ C × C
    /// - F ≠ K × C
    /// - T ≠ N × F
    /// - O ≠ N × K
    /// - K > C
    /// - N < 2
    pub fn new(config: UpdateConfig) -> Self {
        assert!(M == C * C, "M must equal C * C");
        assert!(F == K * C, "F must equal K * C");
        assert!(T == N * F, "T must equal N * F");
        assert!(O == N * K, "O must equal N * K");
        assert!(K <= C, "Cannot extract more filters than channels");
        assert!(K > 0, "Must extract at least one filter");
        assert!(N >= 2, "Need at least 2 classes for multi-class CSP");

        Self {
            class_cov: core::array::from_fn(|_| OnlineCov::new()),
            filters: [0.0; T],
            config,
            samples_since_update: 0,
            filters_ready: false,
            class_counts: [0; N],
        }
    }

    /// Update with a trial from a specific class.
    ///
    /// # Arguments
    ///
    /// * `class_idx` - Class index (0 to N-1)
    /// * `trial` - Multi-channel time series \[samples\]\[channels\]
    ///
    /// # Panics
    ///
    /// Panics if `class_idx >= N`.
    pub fn update_class(&mut self, class_idx: usize, trial: &[[f64; C]]) {
        assert!(class_idx < N, "Class index out of bounds");

        for sample in trial {
            self.class_cov[class_idx].update(sample);
        }
        self.class_counts[class_idx] += trial.len() as u64;
        self.samples_since_update += trial.len() as u64;
        self.check_and_update();
    }

    /// Manually trigger filter recomputation for all classes.
    ///
    /// Computes One-vs-Rest CSP filters for each class.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any class has insufficient data
    /// - Eigendecomposition fails for any class
    pub fn recompute_filters(&mut self) -> Result<(), CspError> {
        // Check all classes have enough data
        for class_idx in 0..N {
            if self.class_cov[class_idx].count() < self.config.min_samples {
                return Err(CspError::InsufficientData);
            }
        }

        // Compute OVR filters for each class
        for class_idx in 0..N {
            self.compute_ovr_filters(class_idx)?;
        }

        self.filters_ready = true;
        self.samples_since_update = 0;
        Ok(())
    }

    /// Apply all spatial filters to a single sample.
    ///
    /// Returns N×K dimensional feature vector, organized as:
    /// \[class0_filter0, class0_filter1, ..., class0_filterK-1,
    ///  class1_filter0, ..., classN-1_filterK-1\]
    ///
    /// # Errors
    ///
    /// Returns error if filters not yet computed.
    pub fn apply(&self, sample: &[f64; C]) -> Result<[f64; O], CspError> {
        if !self.filters_ready {
            return Err(CspError::FiltersNotReady);
        }

        let mut output = [0.0; O];

        // Apply all N×K filters
        // output[n*K + k] = sum_c(filters[n*F + k*C + c] * sample[c])
        for n in 0..N {
            for k in 0..K {
                let mut sum = 0.0;
                for (c, &s) in sample.iter().enumerate() {
                    sum += self.filters[n * F + k * C + c] * s;
                }
                output[n * K + k] = sum;
            }
        }

        Ok(output)
    }

    /// Apply filters for a specific class only.
    ///
    /// Returns K-dimensional feature vector for the specified class.
    ///
    /// # Arguments
    ///
    /// * `class_idx` - Class index (0 to N-1)
    /// * `sample` - Input sample
    ///
    /// # Errors
    ///
    /// Returns error if filters not yet computed.
    ///
    /// # Panics
    ///
    /// Panics if `class_idx >= N`.
    pub fn apply_class(&self, class_idx: usize, sample: &[f64; C]) -> Result<[f64; K], CspError> {
        assert!(class_idx < N, "Class index out of bounds");

        if !self.filters_ready {
            return Err(CspError::FiltersNotReady);
        }

        let mut output = [0.0; K];

        for (k, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (c, &s) in sample.iter().enumerate() {
                sum += self.filters[class_idx * F + k * C + c] * s;
            }
            *out = sum;
        }

        Ok(output)
    }

    /// Get all filter coefficients.
    ///
    /// Returns None if filters not yet computed.
    ///
    /// Filters are organized as: `filters[class * F + filter * C + channel]`
    pub fn filters(&self) -> Option<&[f64; T]> {
        if self.filters_ready {
            Some(&self.filters)
        } else {
            None
        }
    }

    /// Get filter coefficients for a specific class.
    ///
    /// Returns None if filters not yet computed.
    ///
    /// # Panics
    ///
    /// Panics if `class_idx >= N`.
    pub fn class_filters(&self, class_idx: usize) -> Option<&[f64]> {
        assert!(class_idx < N, "Class index out of bounds");

        if self.filters_ready {
            let start = class_idx * F;
            Some(&self.filters[start..start + F])
        } else {
            None
        }
    }

    /// Get number of samples for a specific class.
    pub fn class_count(&self, class_idx: usize) -> u64 {
        if class_idx < N {
            self.class_counts[class_idx]
        } else {
            0
        }
    }

    /// Get total number of classes.
    pub const fn num_classes(&self) -> usize {
        N
    }

    /// Get number of filters per class.
    pub const fn filters_per_class(&self) -> usize {
        K
    }

    /// Check if filters are ready.
    pub fn is_ready(&self) -> bool {
        self.filters_ready
    }

    /// Reset all accumulators and filters.
    pub fn reset(&mut self) {
        for cov in &mut self.class_cov {
            cov.reset();
        }
        self.class_counts = [0; N];
        self.samples_since_update = 0;
        self.filters_ready = false;
    }

    /// Check if automatic update should trigger.
    fn check_and_update(&mut self) {
        if self.config.update_interval == 0 {
            return;
        }

        if self.samples_since_update >= self.config.update_interval {
            let _ = self.recompute_filters();
        }
    }

    /// Compute OVR filters for a single class.
    fn compute_ovr_filters(&mut self, class_idx: usize) -> Result<(), CspError> {
        // Get target class covariance
        let c_target: Matrix<C, M> = Matrix::from_array(self.class_cov[class_idx].covariance());

        // Compute "rest" covariance (sum of all other classes)
        let mut c_rest_data = [0.0; M];
        for (other_idx, cov) in self.class_cov.iter().enumerate() {
            if other_idx != class_idx {
                let other_cov = cov.covariance();
                for (r, &o) in c_rest_data.iter_mut().zip(other_cov.iter()) {
                    *r += o;
                }
            }
        }
        let c_rest: Matrix<C, M> = Matrix::from_array(c_rest_data);

        // Compute C_sum = C_target + C_rest
        let c_sum: Matrix<C, M> = c_target.add(&c_rest);

        // Solve generalized eigenvalue problem: C_target w = λ (C_target + C_rest) w
        let eigen = generalized_eigen(
            &c_target,
            &c_sum,
            self.config.regularization,
            self.config.max_eigen_iters,
            self.config.eigen_tol,
        )
        .map_err(|_| CspError::EigenDecompositionFailed)?;

        // Extract filters: top K/2 for target class, bottom K/2 for rest
        let half_k = K / 2;
        let remainder = K % 2;
        let filter_offset = class_idx * F;

        // Top eigenvectors (maximize target class variance)
        for k in 0..(half_k + remainder) {
            for c in 0..C {
                self.filters[filter_offset + k * C + c] = eigen.eigenvectors.get(c, k);
            }
        }

        // Bottom eigenvectors (maximize rest class variance)
        for k in 0..half_k {
            let src_idx = C - half_k + k;
            let dst_idx = half_k + remainder + k;
            for c in 0..C {
                self.filters[filter_offset + dst_idx * C + c] = eigen.eigenvectors.get(c, src_idx);
            }
        }

        Ok(())
    }
}

impl<
        const C: usize,
        const M: usize,
        const K: usize,
        const F: usize,
        const N: usize,
        const T: usize,
        const O: usize,
    > Default for MulticlassCsp<C, M, K, F, N, T, O>
{
    fn default() -> Self {
        Self::new(UpdateConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csp_initialization() {
        let csp: AdaptiveCsp<4, 16, 2, 8> = AdaptiveCsp::new(UpdateConfig::default());

        assert!(!csp.is_ready());
        assert_eq!(csp.class1_count(), 0);
        assert_eq!(csp.class2_count(), 0);
        assert!(csp.filters().is_none());
    }

    #[test]
    fn test_csp_insufficient_data() {
        let mut csp: AdaptiveCsp<2, 4, 2, 4> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 10,
            ..Default::default()
        });

        // Add just 5 samples
        let trial = [[1.0, 2.0]; 5];
        csp.update_class1(&trial);
        csp.update_class2(&trial);

        // Should fail due to insufficient data
        let result = csp.recompute_filters();
        assert_eq!(result, Err(CspError::InsufficientData));
    }

    #[test]
    fn test_csp_apply_before_ready() {
        let csp: AdaptiveCsp<2, 4, 2, 4> = AdaptiveCsp::new(UpdateConfig::default());
        let sample = [1.0, 2.0];

        let result = csp.apply(&sample);
        assert_eq!(result, Err(CspError::FiltersNotReady));
    }

    #[test]
    fn test_csp_two_class_separation() {
        // Create CSP for 2-channel, 2-filter setup
        let mut csp: AdaptiveCsp<2, 4, 2, 4> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 20,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        // Class 1: signal in channel 0
        for _ in 0..50 {
            let mut trial = [[0.0; 2]; 10];
            for sample in &mut trial {
                sample[0] = 1.0 + (libm::sin(0.1) * 0.5); // Signal
                sample[1] = 0.1; // Noise
            }
            csp.update_class1(&trial);
        }

        // Class 2: signal in channel 1
        for _ in 0..50 {
            let mut trial = [[0.0; 2]; 10];
            for sample in &mut trial {
                sample[0] = 0.1; // Noise
                sample[1] = 1.0 + (libm::cos(0.1) * 0.5); // Signal
            }
            csp.update_class2(&trial);
        }

        // Compute filters
        csp.recompute_filters().unwrap();
        assert!(csp.is_ready());

        // Test filtering
        let sample1 = [1.0, 0.0]; // Class 1 like
        let features1 = csp.apply(&sample1).unwrap();

        let sample2 = [0.0, 1.0]; // Class 2 like
        let features2 = csp.apply(&sample2).unwrap();

        // Features should be different for different classes
        assert!(features1[0].abs() > 0.01 || features1[1].abs() > 0.01);
        assert!(features2[0].abs() > 0.01 || features2[1].abs() > 0.01);
    }

    #[test]
    fn test_csp_reset() {
        let mut csp: AdaptiveCsp<2, 4, 2, 4> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Add some data
        let trial = [[1.0, 2.0]; 10];
        csp.update_class1(&trial);
        csp.update_class2(&trial);

        assert_eq!(csp.class1_count(), 10);
        assert_eq!(csp.class2_count(), 10);

        // Reset
        csp.reset();

        assert_eq!(csp.class1_count(), 0);
        assert_eq!(csp.class2_count(), 0);
        assert!(!csp.is_ready());
    }

    #[test]
    fn test_csp_manual_update() {
        let mut csp: AdaptiveCsp<2, 4, 2, 4> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 10,
            update_interval: 0, // Manual updates only
            ..Default::default()
        });

        // Add data
        let trial = [[1.0, 2.0]; 15];
        csp.update_class1(&trial);
        csp.update_class2(&trial);

        // Should not auto-update
        assert!(!csp.is_ready());

        // Manual trigger
        csp.recompute_filters().unwrap();
        assert!(csp.is_ready());
    }

    #[test]
    fn test_csp_filter_extraction() {
        let mut csp: AdaptiveCsp<4, 16, 4, 16> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 10,
            ..Default::default()
        });

        // Generate synthetic data
        for _ in 0..20 {
            let mut trial = [[0.0; 4]; 10];
            for sample in &mut trial {
                sample[0] = 1.0;
                sample[1] = 0.5;
            }
            csp.update_class1(&trial);
        }

        for _ in 0..20 {
            let mut trial = [[0.0; 4]; 10];
            for sample in &mut trial {
                sample[2] = 1.0;
                sample[3] = 0.5;
            }
            csp.update_class2(&trial);
        }

        csp.recompute_filters().unwrap();

        // Filters should be available
        let filters = csp.filters().unwrap();
        assert_eq!(filters.len(), 16); // 4 filters × 4 channels
    }

    // ===========================================
    // Multi-class CSP tests
    // ===========================================

    #[test]
    fn test_multiclass_csp_initialization() {
        // 3-class CSP: 4 channels, 2 filters per class
        // M = 16, F = 8, T = 24, O = 6
        let csp: MulticlassCsp<4, 16, 2, 8, 3, 24, 6> = MulticlassCsp::new(UpdateConfig::default());

        assert!(!csp.is_ready());
        assert_eq!(csp.num_classes(), 3);
        assert_eq!(csp.filters_per_class(), 2);
        assert_eq!(csp.class_count(0), 0);
        assert_eq!(csp.class_count(1), 0);
        assert_eq!(csp.class_count(2), 0);
        assert!(csp.filters().is_none());
    }

    #[test]
    fn test_multiclass_csp_insufficient_data() {
        let mut csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> = MulticlassCsp::new(UpdateConfig {
            min_samples: 10,
            ..Default::default()
        });

        // Only add data to two classes
        let trial = [[1.0, 2.0]; 15];
        csp.update_class(0, &trial);
        csp.update_class(1, &trial);
        // Class 2 has no data

        let result = csp.recompute_filters();
        assert_eq!(result, Err(CspError::InsufficientData));
    }

    #[test]
    fn test_multiclass_csp_apply_before_ready() {
        let csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> = MulticlassCsp::new(UpdateConfig::default());
        let sample = [1.0, 2.0];

        let result = csp.apply(&sample);
        assert_eq!(result, Err(CspError::FiltersNotReady));

        let result_class = csp.apply_class(0, &sample);
        assert_eq!(result_class, Err(CspError::FiltersNotReady));
    }

    #[test]
    fn test_multiclass_csp_three_class() {
        // 3-class CSP: 3 channels, 2 filters per class
        // Each class has signal concentrated in a different channel
        // M = 9, F = 6, T = 18, O = 6
        let mut csp: MulticlassCsp<3, 9, 2, 6, 3, 18, 6> = MulticlassCsp::new(UpdateConfig {
            min_samples: 20,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        // Class 0: signal in channel 0
        for _ in 0..50 {
            let mut trial = [[0.0; 3]; 10];
            for sample in &mut trial {
                sample[0] = 1.0;
                sample[1] = 0.1;
                sample[2] = 0.1;
            }
            csp.update_class(0, &trial);
        }

        // Class 1: signal in channel 1
        for _ in 0..50 {
            let mut trial = [[0.0; 3]; 10];
            for sample in &mut trial {
                sample[0] = 0.1;
                sample[1] = 1.0;
                sample[2] = 0.1;
            }
            csp.update_class(1, &trial);
        }

        // Class 2: signal in channel 2
        for _ in 0..50 {
            let mut trial = [[0.0; 3]; 10];
            for sample in &mut trial {
                sample[0] = 0.1;
                sample[1] = 0.1;
                sample[2] = 1.0;
            }
            csp.update_class(2, &trial);
        }

        // Compute filters
        csp.recompute_filters().unwrap();
        assert!(csp.is_ready());

        // Test samples should produce different feature patterns
        let sample0 = [1.0, 0.0, 0.0]; // Class 0 like
        let sample1 = [0.0, 1.0, 0.0]; // Class 1 like
        let sample2 = [0.0, 0.0, 1.0]; // Class 2 like

        let features0 = csp.apply(&sample0).unwrap();
        let features1 = csp.apply(&sample1).unwrap();
        let features2 = csp.apply(&sample2).unwrap();

        // Features should be different for different classes
        assert_ne!(features0, features1);
        assert_ne!(features1, features2);
        assert_ne!(features0, features2);
    }

    #[test]
    fn test_multiclass_csp_apply_class() {
        let mut csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> = MulticlassCsp::new(UpdateConfig {
            min_samples: 10,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        // Add sufficient data to all classes
        for class_idx in 0..3 {
            for _ in 0..20 {
                let mut trial = [[0.0; 2]; 10];
                for sample in &mut trial {
                    sample[class_idx % 2] = 1.0;
                }
                csp.update_class(class_idx, &trial);
            }
        }

        csp.recompute_filters().unwrap();

        // Test apply_class for each class
        let sample = [1.0, 0.5];

        let features_class0 = csp.apply_class(0, &sample).unwrap();
        let features_class1 = csp.apply_class(1, &sample).unwrap();
        let features_class2 = csp.apply_class(2, &sample).unwrap();

        assert_eq!(features_class0.len(), 2);
        assert_eq!(features_class1.len(), 2);
        assert_eq!(features_class2.len(), 2);

        // Full apply should concatenate all class features
        let all_features = csp.apply(&sample).unwrap();
        assert_eq!(all_features.len(), 6); // 3 classes × 2 filters
    }

    #[test]
    fn test_multiclass_csp_reset() {
        let mut csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> = MulticlassCsp::new(UpdateConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Add some data
        let trial = [[1.0, 2.0]; 10];
        csp.update_class(0, &trial);
        csp.update_class(1, &trial);
        csp.update_class(2, &trial);

        assert_eq!(csp.class_count(0), 10);
        assert_eq!(csp.class_count(1), 10);
        assert_eq!(csp.class_count(2), 10);

        // Reset
        csp.reset();

        assert_eq!(csp.class_count(0), 0);
        assert_eq!(csp.class_count(1), 0);
        assert_eq!(csp.class_count(2), 0);
        assert!(!csp.is_ready());
    }

    #[test]
    fn test_multiclass_csp_class_filters() {
        let mut csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> = MulticlassCsp::new(UpdateConfig {
            min_samples: 10,
            update_interval: 0,
            ..Default::default()
        });

        // Add data to all classes
        for class_idx in 0..3 {
            let trial = [[1.0, 2.0]; 20];
            csp.update_class(class_idx, &trial);
        }

        csp.recompute_filters().unwrap();

        // Get class-specific filters
        let filters0 = csp.class_filters(0).unwrap();
        let filters1 = csp.class_filters(1).unwrap();
        let filters2 = csp.class_filters(2).unwrap();

        assert_eq!(filters0.len(), 4); // K × C = 2 × 2
        assert_eq!(filters1.len(), 4);
        assert_eq!(filters2.len(), 4);

        // All filters together
        let all_filters = csp.filters().unwrap();
        assert_eq!(all_filters.len(), 12); // N × K × C = 3 × 2 × 2
    }

    #[test]
    fn test_multiclass_csp_four_class() {
        // 4-class motor imagery scenario: 4 channels, 2 filters per class
        // M = 16, F = 8, T = 32, O = 8
        let mut csp: MulticlassCsp<4, 16, 2, 8, 4, 32, 8> = MulticlassCsp::new(UpdateConfig {
            min_samples: 20,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        // Each class has signal in a different channel
        for class_idx in 0..4 {
            for _ in 0..30 {
                let mut trial = [[0.1; 4]; 10];
                for sample in &mut trial {
                    sample[class_idx] = 1.0;
                }
                csp.update_class(class_idx, &trial);
            }
        }

        csp.recompute_filters().unwrap();
        assert!(csp.is_ready());

        // Total features: 4 classes × 2 filters = 8
        let sample = [1.0, 0.0, 0.0, 0.0];
        let features = csp.apply(&sample).unwrap();
        assert_eq!(features.len(), 8);
    }

    #[test]
    #[should_panic(expected = "Class index out of bounds")]
    fn test_multiclass_csp_invalid_class_index() {
        let mut csp: MulticlassCsp<2, 4, 2, 4, 3, 12, 6> =
            MulticlassCsp::new(UpdateConfig::default());
        let trial = [[1.0, 2.0]; 10];
        csp.update_class(5, &trial); // Invalid: only 3 classes
    }
}
