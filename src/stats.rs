pub struct OnlineStats<const C: usize> {
    count: u64,
    mean: [f64; C],
    m2: [f64; C],
}

impl<const C: usize> Default for OnlineStats<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize> OnlineStats<C> {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: [0.0; C],
            m2: [0.0; C],
        }
    }

    pub fn update(&mut self, sample: &[f64; C]) {
        self.count += 1;
        let n = self.count as f64;

        for (i, &s) in sample.iter().enumerate() {
            let delta = s - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = s - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    pub fn mean(&self) -> &[f64; C] {
        &self.mean
    }

    pub fn variance(&self) -> [f64; C] {
        if self.count < 2 {
            return [0.0; C];
        }

        let mut var = [0.0; C];
        for (v, &m) in var.iter_mut().zip(self.m2.iter()) {
            *v = m / (self.count - 1) as f64;
        }
        var
    }

    /// Returns the standard deviation for each dimension.
    ///
    /// This is the square root of the sample variance.
    pub fn std_dev(&self) -> [f64; C] {
        let var = self.variance();
        let mut std = [0.0; C];
        for (s, &v) in std.iter_mut().zip(var.iter()) {
            *s = libm::sqrt(v);
        }
        std
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = [0.0; C];
        self.m2 = [0.0; C];
    }
}

/// Online covariance matrix estimation using Welford's algorithm.
///
/// Computes the C×C covariance matrix incrementally without storing
/// the full sample history. Essential for Common Spatial Patterns (CSP)
/// and Riemannian geometry-based BCI methods.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Matrix size (must equal C × C)
///
/// # Example
///
/// ```
/// use zerostone::OnlineCov;
///
/// // Create covariance estimator for 4 channels (4×4 = 16 matrix elements)
/// let mut cov: OnlineCov<4, 16> = OnlineCov::new();
///
/// // Feed samples
/// for _ in 0..100 {
///     let sample = [1.0, 2.0, 3.0, 4.0];
///     cov.update(&sample);
/// }
///
/// // Get covariance matrix
/// let cov_matrix = cov.covariance();
/// // cov_matrix[i * 4 + j] is the covariance between channels i and j
/// ```
pub struct OnlineCov<const C: usize, const M: usize> {
    count: u64,
    mean: [f64; C],
    /// Covariance accumulator stored as row-major C×C matrix
    cov_accum: [f64; M],
}

impl<const C: usize, const M: usize> Default for OnlineCov<C, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const M: usize> OnlineCov<C, M> {
    /// Create a new online covariance estimator.
    ///
    /// # Panics
    ///
    /// Panics if M ≠ C × C.
    pub const fn new() -> Self {
        assert!(M == C * C, "Matrix size M must equal C * C");
        Self {
            count: 0,
            mean: [0.0; C],
            cov_accum: [0.0; M],
        }
    }

    /// Update with a new sample.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `sample` - Multi-channel sample (length C)
    pub fn update(&mut self, sample: &[f64; C]) {
        self.count += 1;
        let n = self.count as f64;

        // Compute delta before updating mean
        let mut delta_before = [0.0; C];
        for (i, &s) in sample.iter().enumerate() {
            delta_before[i] = s - self.mean[i];
        }

        // Update mean
        for (mean, &delta) in self.mean.iter_mut().zip(delta_before.iter()) {
            *mean += delta / n;
        }

        // Compute delta after updating mean
        let mut delta_after = [0.0; C];
        for ((delta, &s), &m) in delta_after.iter_mut().zip(sample.iter()).zip(self.mean.iter()) {
            *delta = s - m;
        }

        // Update covariance accumulator
        // Note: we need indices here to compute the matrix offset i * C + j
        #[allow(clippy::needless_range_loop)]
        for i in 0..C {
            for j in 0..C {
                self.cov_accum[i * C + j] += delta_before[i] * delta_after[j];
            }
        }
    }

    /// Get the current mean vector.
    pub fn mean(&self) -> &[f64; C] {
        &self.mean
    }

    /// Get the sample covariance matrix (normalized by n-1).
    ///
    /// Returns the C×C covariance matrix in row-major order.
    /// Element `[i * C + j]` is the covariance between channels i and j.
    ///
    /// Returns zero matrix if count < 2.
    pub fn covariance(&self) -> [f64; M] {
        if self.count < 2 {
            return [0.0; M];
        }

        let mut cov = [0.0; M];
        let divisor = (self.count - 1) as f64;

        for (c, &accum) in cov.iter_mut().zip(self.cov_accum.iter()) {
            *c = accum / divisor;
        }

        cov
    }

    /// Get a specific covariance element.
    ///
    /// # Arguments
    ///
    /// * `i` - Row index (channel i)
    /// * `j` - Column index (channel j)
    ///
    /// # Returns
    ///
    /// Covariance between channels i and j
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.cov_accum[i * C + j] / (self.count - 1) as f64
    }

    /// Get the correlation matrix (normalized covariances).
    ///
    /// Each element is cov(i,j) / sqrt(var(i) * var(j))
    pub fn correlation(&self) -> [f64; M] {
        if self.count < 2 {
            return [0.0; M];
        }

        let cov = self.covariance();
        let mut corr = [0.0; M];

        // Get diagonal (variances)
        let mut std_dev = [0.0; C];
        for i in 0..C {
            std_dev[i] = libm::sqrt(cov[i * C + i]);
        }

        // Compute correlation
        for i in 0..C {
            for j in 0..C {
                if std_dev[i] > 1e-10 && std_dev[j] > 1e-10 {
                    corr[i * C + j] = cov[i * C + j] / (std_dev[i] * std_dev[j]);
                } else {
                    corr[i * C + j] = 0.0;
                }
            }
        }

        corr
    }

    /// Get the number of samples processed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset the estimator to initial state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = [0.0; C];
        self.cov_accum = [0.0; M];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_cov_identity() {
        let mut cov: OnlineCov<2, 4> = OnlineCov::new();

        // Feed uncorrelated unit-variance data (identity covariance)
        // Using all combinations of {-1, 1} for both channels gives:
        // Mean = [0, 0], Var = [1, 1], Cov(0,1) = 0
        for _ in 0..1000 {
            cov.update(&[1.0, 1.0]);
            cov.update(&[1.0, -1.0]);
            cov.update(&[-1.0, 1.0]);
            cov.update(&[-1.0, -1.0]);
        }

        let cov_matrix = cov.covariance();

        // Variances should be ~1
        assert!((cov_matrix[0] - 1.0).abs() < 0.1); // var(x0)
        assert!((cov_matrix[3] - 1.0).abs() < 0.1); // var(x1)

        // Covariances should be ~0
        assert!(cov_matrix[1].abs() < 0.1); // cov(x0, x1)
        assert!(cov_matrix[2].abs() < 0.1); // cov(x1, x0)
    }

    #[test]
    fn test_online_cov_correlated() {
        let mut cov: OnlineCov<2, 4> = OnlineCov::new();

        // Feed perfectly correlated data
        for i in 0..100 {
            let val = i as f64;
            cov.update(&[val, val]); // x1 = x0
        }

        let cov_matrix = cov.covariance();

        // Variances should be equal
        assert!((cov_matrix[0] - cov_matrix[3]).abs() < 1e-6);

        // Covariance should equal variance (perfect correlation)
        assert!((cov_matrix[1] - cov_matrix[0]).abs() < 1e-6);
        assert!((cov_matrix[2] - cov_matrix[0]).abs() < 1e-6);
    }

    #[test]
    fn test_online_cov_get() {
        let mut cov: OnlineCov<3, 9> = OnlineCov::new();

        for i in 0..50 {
            cov.update(&[i as f64, (i * 2) as f64, (i * 3) as f64]);
        }

        // Test individual element access
        let cov_01 = cov.get(0, 1);
        let cov_matrix = cov.covariance();

        assert!((cov_01 - cov_matrix[1]).abs() < 1e-10);
    }

    #[test]
    fn test_online_cov_correlation() {
        let mut cov: OnlineCov<2, 4> = OnlineCov::new();

        // Feed perfectly correlated data
        for i in 0..100 {
            let val = i as f64;
            cov.update(&[val, val]);
        }

        let corr = cov.correlation();

        // Diagonal should be 1
        assert!((corr[0] - 1.0).abs() < 1e-6);
        assert!((corr[3] - 1.0).abs() < 1e-6);

        // Off-diagonal should be 1 (perfect correlation)
        assert!((corr[1] - 1.0).abs() < 1e-6);
        assert!((corr[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_online_cov_reset() {
        let mut cov: OnlineCov<2, 4> = OnlineCov::new();

        cov.update(&[1.0, 2.0]);
        cov.update(&[3.0, 4.0]);

        assert_eq!(cov.count(), 2);

        cov.reset();

        assert_eq!(cov.count(), 0);
        assert_eq!(cov.mean(), &[0.0, 0.0]);

        let cov_matrix = cov.covariance();
        for &v in &cov_matrix {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_online_cov_mean() {
        let mut cov: OnlineCov<3, 9> = OnlineCov::new();

        for _ in 0..100 {
            cov.update(&[1.0, 2.0, 3.0]);
        }

        let mean = cov.mean();
        assert!((mean[0] - 1.0).abs() < 1e-10);
        assert!((mean[1] - 2.0).abs() < 1e-10);
        assert!((mean[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_online_cov_symmetry() {
        let mut cov: OnlineCov<3, 9> = OnlineCov::new();

        for i in 0..100 {
            cov.update(&[i as f64, (i * 2) as f64, (i + 5) as f64]);
        }

        let cov_matrix = cov.covariance();

        // Covariance matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((cov_matrix[i * 3 + j] - cov_matrix[j * 3 + i]).abs() < 1e-10);
            }
        }
    }
}
