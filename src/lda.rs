//! Fisher's Linear Discriminant Analysis for binary classification.
//!
//! The standard classifier paired with CSP for motor imagery BCI.
//! Computes the optimal linear projection that maximizes class separability
//! under Gaussian assumptions.
//!
//! Features:
//! - Shrinkage regularization for ill-conditioned scatter matrices
//! - Calibrated probability output via sigmoid
//! - Cholesky-based solve (reuses existing linalg infrastructure)

use crate::linalg::Matrix;

/// Errors that can occur during LDA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdaError {
    /// Need at least 2 samples per class
    InsufficientData,
    /// Model has not been fitted yet
    NotFitted,
    /// Within-class scatter matrix is singular even after regularization
    SingularScatter,
}

/// Fisher's Linear Discriminant Analysis for binary classification.
///
/// # Type Parameters
///
/// * `C` - Number of features
/// * `M` - Must equal C*C
pub struct Lda<const C: usize, const M: usize> {
    /// Discriminant direction (unit norm)
    weights: [f64; C],
    /// Decision boundary threshold
    threshold: f64,
    /// Class 0 mean
    mean0: [f64; C],
    /// Class 1 mean
    mean1: [f64; C],
    /// Pooled within-class variance along w (for calibrated probabilities)
    pooled_var: f64,
    /// Regularization parameter [0, 1]
    shrinkage: f64,
    /// Whether the model has been fitted
    fitted: bool,
}

impl<const C: usize, const M: usize> Lda<C, M> {
    /// Create a new LDA classifier.
    ///
    /// # Arguments
    ///
    /// * `shrinkage` - Regularization parameter in [0, 1]. Default: 0.01.
    ///   Higher values help with ill-conditioned scatter matrices.
    pub fn new(shrinkage: f64) -> Self {
        assert!(M == C * C, "M must equal C * C");
        Self {
            weights: [0.0; C],
            threshold: 0.0,
            mean0: [0.0; C],
            mean1: [0.0; C],
            pooled_var: 1.0,
            shrinkage,
            fitted: false,
        }
    }

    /// Fit the LDA model to labeled data.
    ///
    /// # Arguments
    ///
    /// * `class0` - Samples belonging to class 0
    /// * `class1` - Samples belonging to class 1
    ///
    /// # Errors
    ///
    /// Returns `InsufficientData` if either class has fewer than 2 samples.
    /// Returns `SingularScatter` if the within-class scatter matrix is not
    /// positive definite even after regularization.
    pub fn fit(&mut self, class0: &[[f64; C]], class1: &[[f64; C]]) -> Result<(), LdaError> {
        let n0 = class0.len();
        let n1 = class1.len();
        if n0 < 2 || n1 < 2 {
            return Err(LdaError::InsufficientData);
        }

        // Compute class means
        let mut m0 = [0.0f64; C];
        let mut m1 = [0.0f64; C];
        for sample in class0 {
            for j in 0..C {
                m0[j] += sample[j];
            }
        }
        for sample in class1 {
            for j in 0..C {
                m1[j] += sample[j];
            }
        }
        for j in 0..C {
            m0[j] /= n0 as f64;
            m1[j] /= n1 as f64;
        }

        // Compute within-class scatter: Sw = S0 + S1
        let mut sw = Matrix::<C, M>::zeros();
        for sample in class0 {
            let mut diff = [0.0f64; C];
            for j in 0..C {
                diff[j] = sample[j] - m0[j];
            }
            // Add outer product diff * diff^T
            for i in 0..C {
                for j in 0..C {
                    let cur = sw.get(i, j);
                    sw.set(i, j, cur + diff[i] * diff[j]);
                }
            }
        }
        for sample in class1 {
            let mut diff = [0.0f64; C];
            for j in 0..C {
                diff[j] = sample[j] - m1[j];
            }
            for i in 0..C {
                for j in 0..C {
                    let cur = sw.get(i, j);
                    sw.set(i, j, cur + diff[i] * diff[j]);
                }
            }
        }

        // Apply shrinkage: Sw_reg = (1-alpha)*Sw + alpha*(trace(Sw)/C)*I
        let trace = {
            let mut t = 0.0;
            for i in 0..C {
                t += sw.get(i, i);
            }
            t
        };
        let alpha = self.shrinkage;
        let shrink_val = alpha * trace / C as f64;
        for i in 0..M {
            sw.data_mut()[i] *= 1.0 - alpha;
        }
        sw.add_diagonal(shrink_val);

        // Mean difference: m0 - m1
        let mut diff_mean = [0.0f64; C];
        for j in 0..C {
            diff_mean[j] = m0[j] - m1[j];
        }

        // Solve: w = Sw^{-1} * (m0 - m1)
        let w = sw
            .solve(&diff_mean)
            .map_err(|_| LdaError::SingularScatter)?;

        // Normalize w to unit length
        let mut norm = 0.0;
        for val in w.iter() {
            norm += val * val;
        }
        norm = libm::sqrt(norm);
        if norm < 1e-15 {
            return Err(LdaError::SingularScatter);
        }

        let mut w_unit = [0.0f64; C];
        for j in 0..C {
            w_unit[j] = w[j] / norm;
        }

        // Threshold: midpoint of projected means
        let dot0 = dot::<C>(&w_unit, &m0);
        let dot1 = dot::<C>(&w_unit, &m1);
        let threshold = (dot0 + dot1) * 0.5;

        // Compute pooled variance along w for calibrated probabilities
        let n_total = (n0 + n1) as f64;
        let mut var_sum = 0.0;
        for sample in class0 {
            let proj = dot::<C>(&w_unit, sample) - dot0;
            var_sum += proj * proj;
        }
        for sample in class1 {
            let proj = dot::<C>(&w_unit, sample) - dot1;
            var_sum += proj * proj;
        }
        let pooled_var = var_sum / (n_total - 2.0);
        let pooled_var = if pooled_var < 1e-15 { 1.0 } else { pooled_var };

        self.weights = w_unit;
        self.threshold = threshold;
        self.mean0 = m0;
        self.mean1 = m1;
        self.pooled_var = pooled_var;
        self.fitted = true;
        Ok(())
    }

    /// Predict class label (0 or 1) for a single sample.
    pub fn predict(&self, x: &[f64; C]) -> Result<usize, LdaError> {
        if !self.fitted {
            return Err(LdaError::NotFitted);
        }
        let d = self.decision_function_inner(x);
        Ok(if d >= 0.0 { 0 } else { 1 })
    }

    /// Compute probability of class 0 for a single sample.
    ///
    /// Uses sigmoid scaled by pooled within-class variance.
    pub fn predict_proba(&self, x: &[f64; C]) -> Result<f64, LdaError> {
        if !self.fitted {
            return Err(LdaError::NotFitted);
        }
        let d = self.decision_function_inner(x);
        // Scale by inverse of pooled standard deviation for calibrated sigmoid
        let scale = 1.0 / libm::sqrt(self.pooled_var);
        Ok(sigmoid(d * scale))
    }

    /// Compute signed distance from decision boundary.
    ///
    /// Positive values indicate class 0, negative indicate class 1.
    pub fn decision_function(&self, x: &[f64; C]) -> Result<f64, LdaError> {
        if !self.fitted {
            return Err(LdaError::NotFitted);
        }
        Ok(self.decision_function_inner(x))
    }

    /// Get the discriminant weights (unit norm direction).
    pub fn weights(&self) -> Option<&[f64; C]> {
        if self.fitted {
            Some(&self.weights)
        } else {
            None
        }
    }

    /// Get the decision threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Internal: compute w^T * x - threshold
    #[inline]
    fn decision_function_inner(&self, x: &[f64; C]) -> f64 {
        dot::<C>(&self.weights, x) - self.threshold
    }
}

/// Dot product of two vectors.
#[inline]
fn dot<const C: usize>(a: &[f64; C], b: &[f64; C]) -> f64 {
    let mut sum = 0.0;
    for i in 0..C {
        sum += a[i] * b[i];
    }
    sum
}

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    // Simple pseudo-RNG for tests (xorshift64)
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

    fn make_gaussian_clusters(
        rng: &mut Rng,
        mean0: &[f64],
        mean1: &[f64],
        std: f64,
        n_per_class: usize,
    ) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let mut c0 = Vec::with_capacity(n_per_class);
        let mut c1 = Vec::with_capacity(n_per_class);
        for _ in 0..n_per_class {
            c0.push([rng.gaussian(mean0[0], std), rng.gaussian(mean0[1], std)]);
        }
        for _ in 0..n_per_class {
            c1.push([rng.gaussian(mean1[0], std), rng.gaussian(mean1[1], std)]);
        }
        (c0, c1)
    }

    #[test]
    fn test_perfectly_separable() {
        let mut rng = Rng::new(42);
        let (c0, c1) = make_gaussian_clusters(&mut rng, &[0.0, 0.0], &[10.0, 10.0], 0.5, 50);

        let mut lda = Lda::<2, 4>::new(0.01);
        lda.fit(&c0, &c1).unwrap();

        let mut correct = 0;
        for sample in &c0 {
            if lda.predict(sample).unwrap() == 0 {
                correct += 1;
            }
        }
        for sample in &c1 {
            if lda.predict(sample).unwrap() == 1 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / 100.0;
        assert!(
            accuracy > 0.99,
            "Expected near-perfect accuracy, got {}",
            accuracy
        );
    }

    #[test]
    fn test_overlapping_gaussians() {
        let mut rng = Rng::new(55);
        let (c0, c1) = make_gaussian_clusters(&mut rng, &[0.0, 0.0], &[2.0, 2.0], 1.0, 100);

        let mut lda = Lda::<2, 4>::new(0.01);
        lda.fit(&c0, &c1).unwrap();

        let mut correct = 0;
        for sample in &c0 {
            if lda.predict(sample).unwrap() == 0 {
                correct += 1;
            }
        }
        for sample in &c1 {
            if lda.predict(sample).unwrap() == 1 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / 200.0;
        assert!(
            accuracy > 0.70,
            "Expected reasonable accuracy, got {}",
            accuracy
        );
    }

    #[test]
    fn test_predict_proba() {
        let mut rng = Rng::new(42);
        let (c0, c1) = make_gaussian_clusters(&mut rng, &[0.0, 0.0], &[5.0, 5.0], 0.5, 50);

        let mut lda = Lda::<2, 4>::new(0.01);
        lda.fit(&c0, &c1).unwrap();

        // Points far from boundary should have extreme probabilities
        let far_c0 = [-5.0, -5.0];
        let far_c1 = [10.0, 10.0];
        let p0 = lda.predict_proba(&far_c0).unwrap();
        let p1 = lda.predict_proba(&far_c1).unwrap();
        assert!(
            p0 > 0.95,
            "Far class 0 point should have P>0.95, got {}",
            p0
        );
        assert!(
            p1 < 0.05,
            "Far class 1 point should have P<0.05, got {}",
            p1
        );

        // Point near boundary should have probability near 0.5
        let w = lda.weights().unwrap();
        let t = lda.threshold();
        // Find a point on the decision boundary: w^T * x = t
        // Use x = t * w / (w^T * w) = t * w (since w is unit)
        let boundary = [w[0] * t, w[1] * t];
        let p_boundary = lda.predict_proba(&boundary).unwrap();
        assert!(
            libm::fabs(p_boundary - 0.5) < 0.1,
            "Boundary point should have P~0.5, got {}",
            p_boundary
        );
    }

    #[test]
    fn test_decision_function() {
        let mut rng = Rng::new(42);
        let (c0, c1) = make_gaussian_clusters(&mut rng, &[0.0, 0.0], &[5.0, 5.0], 0.5, 50);

        let mut lda = Lda::<2, 4>::new(0.01);
        lda.fit(&c0, &c1).unwrap();

        // Sign of decision function should match predict
        for sample in &c0 {
            let d = lda.decision_function(sample).unwrap();
            let pred = lda.predict(sample).unwrap();
            if pred == 0 {
                assert!(d >= 0.0);
            } else {
                assert!(d < 0.0);
            }
        }
        for sample in &c1 {
            let d = lda.decision_function(sample).unwrap();
            let pred = lda.predict(sample).unwrap();
            if pred == 0 {
                assert!(d >= 0.0);
            } else {
                assert!(d < 0.0);
            }
        }

        // Magnitude should increase with distance from boundary
        let near = [2.5, 2.5]; // near midpoint
        let far = [-5.0, -5.0]; // far from boundary
        let d_near = libm::fabs(lda.decision_function(&near).unwrap());
        let d_far = libm::fabs(lda.decision_function(&far).unwrap());
        assert!(
            d_far > d_near,
            "Far point should have larger |d|: {} vs {}",
            d_far,
            d_near
        );
    }

    #[test]
    fn test_shrinkage_effect() {
        // Create ill-conditioned data: features highly correlated
        let mut rng = Rng::new(88);
        let mut c0 = Vec::new();
        let mut c1 = Vec::new();
        for _ in 0..10 {
            let x = rng.gaussian(0.0, 1.0);
            c0.push([x, x + rng.gaussian(0.0, 0.001)]); // nearly duplicate features
        }
        for _ in 0..10 {
            let x = rng.gaussian(3.0, 1.0);
            c1.push([x, x + rng.gaussian(0.0, 0.001)]);
        }

        // Without regularization -> might fail
        let mut lda_no_reg = Lda::<2, 4>::new(0.0);
        // With shrinkage -> should succeed
        let mut lda_reg = Lda::<2, 4>::new(0.1);

        let result_reg = lda_reg.fit(&c0, &c1);
        assert!(result_reg.is_ok(), "Regularized LDA should succeed");

        // The unregularized version may or may not fail depending on exact data,
        // but the regularized version should always work
        let _ = lda_no_reg.fit(&c0, &c1); // don't assert, just show shrinkage helps
    }

    #[test]
    fn test_weight_direction_identity_scatter() {
        // When Sw ~ identity, w should be parallel to (m0 - m1)
        let mut rng = Rng::new(42);
        let mut c0 = Vec::new();
        let mut c1 = Vec::new();
        // Generate isotropic Gaussian clusters
        for _ in 0..200 {
            c0.push([rng.gaussian(0.0, 1.0), rng.gaussian(0.0, 1.0)]);
        }
        for _ in 0..200 {
            c1.push([rng.gaussian(3.0, 1.0), rng.gaussian(4.0, 1.0)]);
        }

        let mut lda = Lda::<2, 4>::new(0.01);
        lda.fit(&c0, &c1).unwrap();

        let w = lda.weights().unwrap();
        // Expected direction: (m0-m1) normalized
        let diff = [-3.0, -4.0];
        let diff_norm = libm::sqrt(diff[0] * diff[0] + diff[1] * diff[1]);
        let expected = [diff[0] / diff_norm, diff[1] / diff_norm];

        // |cos(angle)| should be close to 1
        let cos_angle = libm::fabs(w[0] * expected[0] + w[1] * expected[1]);
        assert!(
            cos_angle > 0.9,
            "Weight direction should be parallel to mean diff, cos={}",
            cos_angle
        );
    }

    #[test]
    fn test_insufficient_data() {
        let c0 = [[0.0, 0.0]]; // only 1 sample
        let c1 = [[1.0, 1.0], [2.0, 2.0]];

        let mut lda = Lda::<2, 4>::new(0.01);
        assert_eq!(lda.fit(&c0, &c1), Err(LdaError::InsufficientData));
    }

    #[test]
    fn test_not_fitted() {
        let lda = Lda::<2, 4>::new(0.01);
        let x = [0.0, 0.0];
        assert_eq!(lda.predict(&x), Err(LdaError::NotFitted));
        assert_eq!(lda.predict_proba(&x), Err(LdaError::NotFitted));
        assert_eq!(lda.decision_function(&x), Err(LdaError::NotFitted));
        assert!(lda.weights().is_none());
        assert!(!lda.is_fitted());
    }
}
