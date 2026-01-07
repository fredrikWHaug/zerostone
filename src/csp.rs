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
            min_samples: 100,       // 100 trials minimum
            update_interval: 50,    // Update every 50 new trials
            regularization: 1e-6,   // Tikhonov regularization
            max_eigen_iters: 30,    // 30 Jacobi sweeps
            eigen_tol: 1e-10,       // Convergence threshold
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
    /// * `trial` - Multi-channel time series [samples][channels]
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
}
