//! Gaussian Mixture Model for spike sorting cluster refinement.
//!
//! Implements batch Expectation-Maximization (EM) with full covariance
//! matrices. When initialized from k-means centroids, EM typically
//! converges in 5-10 iterations and produces significantly better
//! cluster assignments than k-means because it models cluster shape
//! (covariance) rather than assuming spherical clusters.
//!
//! # Theory
//!
//! Each component `k` is a multivariate Gaussian N(mu_k, Sigma_k) with
//! mixing weight pi_k. The EM algorithm alternates:
//!
//! **E-step**: Compute responsibilities (posterior probability that point i
//! belongs to component k):
//!   `r_ik = pi_k * N(x_i | mu_k, Sigma_k) / sum_j pi_j * N(x_i | mu_j, Sigma_j)`
//!
//! **M-step**: Update parameters from weighted sufficient statistics:
//!   `pi_k = (1/N) * sum_i r_ik`
//!   `mu_k = sum_i r_ik * x_i / sum_i r_ik`
//!   `Sigma_k = sum_i r_ik * (x_i - mu_k)(x_i - mu_k)^T / sum_i r_ik`
//!
//! Convergence is monitored by the log-likelihood:
//!   `L = sum_i log(sum_k pi_k * N(x_i | mu_k, Sigma_k))`
//!
//! # Regularization
//!
//! To prevent singular covariance matrices, a regularization term is added:
//!   `Sigma_k += epsilon * I`
//!
//! This bounds the minimum eigenvalue of each covariance matrix, ensuring
//! numerical stability even for clusters with very few points.
//!
//! # Example
//!
//! ```
//! use zerostone::gmm::GaussianMixture;
//!
//! let mut gmm = GaussianMixture::<4, 8>::new(1e-4);
//! // Initialize from k-means centroids
//! let centroids = [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]];
//! gmm.init_from_centroids(&centroids, 2);
//!
//! // Fit with EM
//! let data = [
//!     [1.1, 0.1, -0.1, 0.0], [0.9, -0.1, 0.2, 0.1],
//!     [1.0, 0.0, 0.0, -0.1], [-1.1, 0.1, 0.0, 0.1],
//!     [-0.9, -0.1, 0.1, 0.0], [-1.0, 0.0, -0.1, 0.0],
//! ];
//! let ll = gmm.fit(&data, 10);
//! assert!(ll.is_finite());
//!
//! // Classify
//! let (label, _prob) = gmm.predict(&[1.0, 0.0, 0.0, 0.0]);
//! assert_eq!(label, 0);
//! let (label2, _prob2) = gmm.predict(&[-1.0, 0.0, 0.0, 0.0]);
//! assert_eq!(label2, 1);
//! ```

// Matrix index loops are intentional: d1, d2 index multiple arrays simultaneously.
// Clippy's iterator suggestions don't apply to coupled-index matrix math.
#![allow(clippy::needless_range_loop)]

/// Gaussian Mixture Model with full covariance matrices.
///
/// # Type Parameters
///
/// * `D` - Feature dimensionality (e.g. 4 for 3 PCA + 1 channel)
/// * `K` - Maximum number of components/clusters
///
/// # Example
///
/// ```
/// use zerostone::gmm::GaussianMixture;
///
/// let gmm = GaussianMixture::<4, 8>::new(1e-4);
/// assert_eq!(gmm.n_components(), 0);
/// ```
pub struct GaussianMixture<const D: usize, const K: usize> {
    /// Component means: mu_k[d]
    means: [[f64; D]; K],
    /// Component covariances: cov_k[d1][d2] (symmetric)
    covs: [[[f64; D]; D]; K],
    /// Cholesky factors of covariance matrices (lower triangular)
    cholesky: [[[f64; D]; D]; K],
    /// Log-determinants of covariance matrices (from Cholesky diagonal)
    log_dets: [f64; K],
    /// Mixing weights: pi_k
    weights: [f64; K],
    /// Number of active components
    n_components: usize,
    /// Regularization parameter (added to diagonal)
    epsilon: f64,
    /// Whether Cholesky factors are up to date
    cholesky_valid: [bool; K],
}

impl<const D: usize, const K: usize> GaussianMixture<D, K> {
    /// Create a new GMM with no components.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Covariance regularization (added to diagonal). Prevents
    ///   singular matrices. Typical value: 1e-4 for normalized features.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let gmm = GaussianMixture::<4, 8>::new(1e-4);
    /// assert_eq!(gmm.n_components(), 0);
    /// ```
    pub fn new(epsilon: f64) -> Self {
        Self {
            means: [[0.0; D]; K],
            covs: [[[0.0; D]; D]; K],
            cholesky: [[[0.0; D]; D]; K],
            log_dets: [0.0; K],
            weights: [0.0; K],
            n_components: 0,
            epsilon: if epsilon > 0.0 { epsilon } else { 1e-6 },
            cholesky_valid: [false; K],
        }
    }

    /// Number of active components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Initialize from k-means centroids with identity covariance.
    ///
    /// Sets each component's mean to the corresponding centroid,
    /// covariance to epsilon * I, and uniform mixing weights.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
    /// let centroids = [[1.0, 0.0], [-1.0, 0.0]];
    /// gmm.init_from_centroids(&centroids, 2);
    /// assert_eq!(gmm.n_components(), 2);
    /// ```
    pub fn init_from_centroids(&mut self, centroids: &[[f64; D]], n: usize) {
        let n = n.min(K).min(centroids.len());
        self.n_components = n;
        let w = if n > 0 { 1.0 / n as f64 } else { 0.0 };

        for (k, centroid) in centroids.iter().enumerate().take(n) {
            self.means[k] = *centroid;
            self.covs[k] = [[0.0; D]; D];
            for d in 0..D {
                self.covs[k][d][d] = self.epsilon;
            }
            self.weights[k] = w;
            self.cholesky_valid[k] = false;
        }
    }

    /// Initialize from k-means centroids and data (estimates initial covariance).
    ///
    /// Uses the k-means labels to compute per-cluster covariance from the data,
    /// giving EM a warm start that's much better than identity covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
    /// let data = [[1.0, 0.1], [0.9, -0.1], [-1.0, 0.1], [-0.9, -0.1]];
    /// let labels = [0, 0, 1, 1];
    /// gmm.init_from_labels(&data, &labels, 2);
    /// assert_eq!(gmm.n_components(), 2);
    /// ```
    pub fn init_from_labels(&mut self, data: &[[f64; D]], labels: &[usize], n_clusters: usize) {
        let n = n_clusters.min(K);
        self.n_components = n;
        let n_data = data.len().min(labels.len());
        if n == 0 || n_data == 0 {
            return;
        }

        // Compute per-cluster means
        let mut counts = [0usize; K];
        self.means = [[0.0; D]; K];
        for (xi, &lab) in data.iter().zip(labels.iter()).take(n_data) {
            if lab < n {
                for d in 0..D {
                    self.means[lab][d] += xi[d];
                }
                counts[lab] += 1;
            }
        }
        for k in 0..n {
            if counts[k] > 0 {
                let inv = 1.0 / counts[k] as f64;
                for d in 0..D {
                    self.means[k][d] *= inv;
                }
            }
        }

        // Compute per-cluster covariance
        self.covs = [[[0.0; D]; D]; K];
        for (xi, &lab) in data.iter().zip(labels.iter()).take(n_data) {
            if lab < n {
                let mut diff = [0.0f64; D];
                for d in 0..D {
                    diff[d] = xi[d] - self.means[lab][d];
                }
                for d1 in 0..D {
                    for d2 in d1..D {
                        let v = diff[d1] * diff[d2];
                        self.covs[lab][d1][d2] += v;
                        if d1 != d2 {
                            self.covs[lab][d2][d1] += v;
                        }
                    }
                }
            }
        }

        // Normalize and regularize
        for k in 0..n {
            if counts[k] > 1 {
                let inv = 1.0 / (counts[k] - 1) as f64; // unbiased
                for d1 in 0..D {
                    for d2 in 0..D {
                        self.covs[k][d1][d2] *= inv;
                    }
                }
            }
            for d in 0..D {
                self.covs[k][d][d] += self.epsilon;
            }
            self.weights[k] = counts[k] as f64 / n_data as f64;
            self.cholesky_valid[k] = false;
        }
    }

    /// Compute Cholesky decomposition and log-determinant for component k.
    fn update_cholesky(&mut self, k: usize) -> bool {
        if k >= self.n_components {
            return false;
        }
        if self.cholesky_valid[k] {
            return true;
        }

        // Cholesky decomposition: L such that Sigma = L * L^T
        let cov = &self.covs[k];
        let l = &mut self.cholesky[k];
        *l = [[0.0; D]; D];

        for j in 0..D {
            for i in j..D {
                let mut sum = cov[i][j];
                for p in 0..j {
                    sum -= l[i][p] * l[j][p];
                }
                if i == j {
                    if sum <= 0.0 {
                        self.covs[k][j][j] += self.epsilon * 10.0;
                        self.cholesky_valid[k] = false;
                        return false;
                    }
                    l[i][j] = libm::sqrt(sum);
                } else {
                    let l_jj = l[j][j];
                    if libm::fabs(l_jj) < 1e-15 {
                        return false;
                    }
                    l[i][j] = sum / l_jj;
                }
            }
        }

        // Log-determinant = 2 * sum(log(L_ii))
        let mut log_det = 0.0;
        for d in 0..D {
            let diag = l[d][d];
            if diag <= 0.0 {
                return false;
            }
            log_det += libm::log(diag);
        }
        self.log_dets[k] = 2.0 * log_det;
        self.cholesky_valid[k] = true;
        true
    }

    /// Solve L * x = b (forward substitution) for component k.
    fn forward_solve(&self, k: usize, b: &[f64; D]) -> [f64; D] {
        let l = &self.cholesky[k];
        let mut x = [0.0f64; D];
        for i in 0..D {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i][j] * x[j];
            }
            let diag = l[i][i];
            x[i] = if libm::fabs(diag) > 1e-15 {
                sum / diag
            } else {
                0.0
            };
        }
        x
    }

    /// Log-probability of x under component k (up to additive constant).
    ///
    /// Returns `log N(x | mu_k, Sigma_k)` = -0.5 * [mahalanobis^2 + log|Sigma| + D*log(2*pi)]
    fn log_prob(&self, x: &[f64; D], k: usize) -> f64 {
        if k >= self.n_components || !self.cholesky_valid[k] {
            return f64::NEG_INFINITY;
        }

        // Mahalanobis distance via Cholesky: solve L*y = (x - mu), then ||y||^2
        let mut diff = [0.0f64; D];
        for d in 0..D {
            diff[d] = x[d] - self.means[k][d];
        }
        let y = self.forward_solve(k, &diff);
        let mut maha_sq = 0.0;
        for yv in y.iter() {
            maha_sq += yv * yv;
        }

        const LOG_2PI: f64 = 1.8378770664093453; // log(2*pi)
        -0.5 * (maha_sq + self.log_dets[k] + D as f64 * LOG_2PI)
    }

    /// Compute responsibilities for a single point.
    ///
    /// Returns log-responsibilities (unnormalized) and the log-sum-exp.
    fn responsibilities(&self, x: &[f64; D], resp: &mut [f64; K]) -> f64 {
        let n = self.n_components;
        if n == 0 {
            return f64::NEG_INFINITY;
        }

        // Compute log(pi_k) + log N(x | mu_k, Sigma_k) for each k
        let mut max_log = f64::NEG_INFINITY;
        for (k, rk) in resp.iter_mut().enumerate().take(n) {
            let log_w = if self.weights[k] > 0.0 {
                libm::log(self.weights[k])
            } else {
                f64::NEG_INFINITY
            };
            *rk = log_w + self.log_prob(x, k);
            if *rk > max_log {
                max_log = *rk;
            }
        }
        // Zero out inactive components
        for rk in resp.iter_mut().skip(n) {
            *rk = f64::NEG_INFINITY;
        }

        if !max_log.is_finite() {
            return f64::NEG_INFINITY;
        }

        // Log-sum-exp for normalization
        let mut sum_exp = 0.0;
        for rk in resp.iter().take(n) {
            if rk.is_finite() {
                sum_exp += libm::exp(rk - max_log);
            }
        }
        let log_sum = max_log + libm::log(sum_exp);

        // Normalize to probabilities
        for rk in resp.iter_mut().take(n) {
            if rk.is_finite() {
                *rk = libm::exp(*rk - log_sum);
            } else {
                *rk = 0.0;
            }
        }

        log_sum
    }

    /// Run batch EM for the given number of iterations.
    ///
    /// Returns the final log-likelihood. Convergence is checked by monitoring
    /// the log-likelihood increase; EM stops early if the improvement is
    /// below 1e-6 per data point.
    ///
    /// # Arguments
    ///
    /// * `data` - Feature vectors (one per spike)
    /// * `max_iter` - Maximum number of EM iterations
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
    /// let data = [[1.0, 0.0], [1.1, 0.1], [-1.0, 0.0], [-1.1, -0.1]];
    /// let labels = [0, 0, 1, 1];
    /// gmm.init_from_labels(&data, &labels, 2);
    /// let ll = gmm.fit(&data, 10);
    /// assert!(ll.is_finite());
    /// ```
    pub fn fit(&mut self, data: &[[f64; D]], max_iter: usize) -> f64 {
        let n_data = data.len();
        let n = self.n_components;
        if n == 0 || n_data == 0 || D == 0 {
            return f64::NEG_INFINITY;
        }

        // Ensure all Cholesky factors are valid
        for k in 0..n {
            if !self.update_cholesky(k) {
                for d in 0..D {
                    self.covs[k][d][d] += self.epsilon * 100.0;
                }
                self.cholesky_valid[k] = false;
                self.update_cholesky(k);
            }
        }

        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // --- E-step: compute responsibilities ---
            let mut total_ll = 0.0;
            let mut new_means = [[0.0f64; D]; K];
            let mut new_covs = [[[0.0f64; D]; D]; K];
            let mut new_weights = [0.0f64; K];

            for xi in data.iter().take(n_data) {
                let mut resp = [0.0f64; K];
                let ll_i = self.responsibilities(xi, &mut resp);
                if ll_i.is_finite() {
                    total_ll += ll_i;
                }

                // Accumulate sufficient statistics (M-step accumulators)
                for k in 0..n {
                    let r = resp[k];
                    if r < 1e-300 {
                        continue;
                    }
                    new_weights[k] += r;
                    for d in 0..D {
                        new_means[k][d] += r * xi[d];
                    }
                    // Outer product accumulation for covariance
                    for d1 in 0..D {
                        let diff1 = xi[d1] - self.means[k][d1];
                        for d2 in d1..D {
                            let diff2 = xi[d2] - self.means[k][d2];
                            let v = r * diff1 * diff2;
                            new_covs[k][d1][d2] += v;
                            if d1 != d2 {
                                new_covs[k][d2][d1] += v;
                            }
                        }
                    }
                }
            }

            // --- M-step: update parameters ---
            for k in 0..n {
                let nk = new_weights[k];
                if nk < 1e-10 {
                    continue;
                }
                let inv_nk = 1.0 / nk;

                // Compute new mean and the shift from old mean
                let mut delta = [0.0f64; D];
                let new_mean_k = {
                    let mut m = [0.0f64; D];
                    for d in 0..D {
                        m[d] = new_means[k][d] * inv_nk;
                        delta[d] = m[d] - self.means[k][d];
                    }
                    m
                };

                // Update covariance using the parallel axis theorem.
                //
                // We accumulated S_k = sum_i r_ik * (x_i - mu_old)(x_i - mu_old)^T
                // in new_covs[k]. The covariance around the new mean is:
                //
                //   Sigma_k = (1/N_k) * S_k - delta * delta^T
                //
                // where delta = mu_new - mu_old. This is exact and avoids a
                // second O(N*K*D^2) pass through the data.
                for d1 in 0..D {
                    for d2 in d1..D {
                        let v = new_covs[k][d1][d2] * inv_nk - delta[d1] * delta[d2];
                        self.covs[k][d1][d2] = v;
                        if d1 != d2 {
                            self.covs[k][d2][d1] = v;
                        }
                    }
                }

                // Regularize
                for d in 0..D {
                    self.covs[k][d][d] += self.epsilon;
                }

                // Update mean and weight
                self.means[k] = new_mean_k;
                self.weights[k] = nk / n_data as f64;
                self.cholesky_valid[k] = false;
            }

            // Update Cholesky for next iteration
            for k in 0..n {
                self.update_cholesky(k);
            }

            // Check convergence
            let ll_per_point = total_ll / n_data as f64;
            let prev_per_point = prev_ll / n_data as f64;
            if _iter > 0 && libm::fabs(ll_per_point - prev_per_point) < 1e-6 {
                return total_ll;
            }
            prev_ll = total_ll;
        }

        prev_ll
    }

    /// Predict the most likely component for a point.
    ///
    /// Returns `(component_index, posterior_probability)`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
    /// let data = [[1.0, 0.0], [1.1, 0.1], [-1.0, 0.0], [-1.1, -0.1]];
    /// let labels = [0, 0, 1, 1];
    /// gmm.init_from_labels(&data, &labels, 2);
    /// gmm.fit(&data, 10);
    /// let (label, prob) = gmm.predict(&[1.0, 0.0]);
    /// assert_eq!(label, 0);
    /// assert!(prob > 0.5);
    /// ```
    pub fn predict(&self, x: &[f64; D]) -> (usize, f64) {
        let mut resp = [0.0f64; K];
        let _ = self.responsibilities(x, &mut resp);

        let mut best_k = 0;
        let mut best_r = 0.0;
        for (k, &rk) in resp.iter().enumerate().take(self.n_components) {
            if rk > best_r {
                best_r = rk;
                best_k = k;
            }
        }
        (best_k, best_r)
    }

    /// Relabel all data points using GMM posteriors.
    ///
    /// Returns the number of points whose label changed from the input.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::gmm::GaussianMixture;
    ///
    /// let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
    /// let data = [[1.0, 0.0], [1.1, 0.1], [-1.0, 0.0], [-1.1, -0.1]];
    /// let mut labels = [0, 0, 1, 1];
    /// gmm.init_from_labels(&data, &labels, 2);
    /// gmm.fit(&data, 10);
    /// let changed = gmm.relabel(&data, &mut labels);
    /// // Labels should stay the same for well-separated clusters
    /// assert_eq!(changed, 0);
    /// ```
    pub fn relabel(&self, data: &[[f64; D]], labels: &mut [usize]) -> usize {
        let mut changed = 0usize;
        for (xi, lab) in data.iter().zip(labels.iter_mut()) {
            let (best_k, _) = self.predict(xi);
            if *lab != best_k {
                *lab = best_k;
                changed += 1;
            }
        }
        changed
    }

    /// Bayesian Information Criterion for model selection.
    ///
    /// `BIC = -2 * log_likelihood + p * log(N)`
    ///
    /// where p is the number of free parameters:
    /// - K means (K * D)
    /// - K covariances (K * D * (D+1) / 2)
    /// - K - 1 mixing weights
    ///
    /// Lower BIC indicates a better model (balances fit vs complexity).
    pub fn bic(&self, data: &[[f64; D]]) -> f64 {
        let n_data = data.len();
        if n_data == 0 || self.n_components == 0 {
            return f64::INFINITY;
        }

        // Compute log-likelihood
        let mut ll = 0.0;
        let mut resp = [0.0f64; K];
        for x in data {
            let ll_i = self.responsibilities(x, &mut resp);
            if ll_i.is_finite() {
                ll += ll_i;
            }
        }

        // Number of free parameters
        let k = self.n_components;
        let n_params = k * D + k * D * (D + 1) / 2 + (k - 1);

        -2.0 * ll + n_params as f64 * libm::log(n_data as f64)
    }

    /// Get the mean of component k.
    pub fn mean(&self, k: usize) -> &[f64; D] {
        &self.means[k.min(K - 1)]
    }

    /// Get the weight of component k.
    pub fn weight(&self, k: usize) -> f64 {
        if k < self.n_components {
            self.weights[k]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let gmm = GaussianMixture::<4, 8>::new(1e-4);
        assert_eq!(gmm.n_components(), 0);
    }

    #[test]
    fn test_init_from_centroids() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let centroids = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]];
        gmm.init_from_centroids(&centroids, 3);
        assert_eq!(gmm.n_components(), 3);
        assert!((gmm.weight(0) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_init_from_labels() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [[1.0, 0.0], [1.2, 0.1], [-1.0, 0.0], [-1.2, -0.1]];
        let labels = [0, 0, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        assert_eq!(gmm.n_components(), 2);
        assert!((gmm.mean(0)[0] - 1.1).abs() < 1e-10);
        assert!((gmm.mean(0)[1] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_two_clusters_well_separated() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [
            [5.0, 0.0],
            [5.1, 0.1],
            [4.9, -0.1],
            [5.0, 0.2],
            [-5.0, 0.0],
            [-5.1, -0.1],
            [-4.9, 0.1],
            [-5.0, -0.2],
        ];
        let labels = [0, 0, 0, 0, 1, 1, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        let ll = gmm.fit(&data, 20);
        assert!(ll.is_finite(), "log-likelihood should be finite");

        let (l0, p0) = gmm.predict(&[5.0, 0.0]);
        assert_eq!(l0, 0);
        assert!(p0 > 0.99);

        let (l1, p1) = gmm.predict(&[-5.0, 0.0]);
        assert_eq!(l1, 1);
        assert!(p1 > 0.99);
    }

    #[test]
    fn test_relabel_well_separated() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [
            [5.0, 0.0],
            [5.1, 0.1],
            [4.9, -0.1],
            [-5.0, 0.0],
            [-5.1, -0.1],
            [-4.9, 0.1],
        ];
        let mut labels = [0, 0, 0, 1, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        gmm.fit(&data, 10);
        let changed = gmm.relabel(&data, &mut labels);
        assert_eq!(changed, 0, "well-separated clusters should not change");
    }

    #[test]
    fn test_relabel_corrects_mislabeled() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [
            [5.0, 0.0],
            [5.1, 0.1],
            [4.9, -0.1],
            [5.0, 0.2],
            [-5.0, 0.0],
            [-5.1, -0.1],
            [-4.9, 0.1],
        ];
        let init_labels = [0, 0, 0, 0, 1, 1, 1];
        gmm.init_from_labels(&data, &init_labels, 2);
        gmm.fit(&data, 10);

        let mut labels = [0, 0, 0, 1, 1, 1, 1];
        let changed = gmm.relabel(&data, &mut labels);
        assert!(changed >= 1, "GMM should correct mislabeled point");
        assert_eq!(labels[3], 0, "point 3 should be relabeled to cluster 0");
    }

    #[test]
    fn test_four_dimensional() {
        let mut gmm = GaussianMixture::<4, 8>::new(1e-3);
        let data = [
            [3.0, 0.1, -0.2, 0.0],
            [3.1, -0.1, 0.1, 0.2],
            [2.9, 0.0, 0.0, -0.1],
            [-3.0, 0.1, 0.2, 0.0],
            [-3.1, -0.1, -0.1, 0.2],
            [-2.9, 0.0, 0.0, -0.1],
        ];
        let labels = [0, 0, 0, 1, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        let ll = gmm.fit(&data, 10);
        assert!(ll.is_finite());

        let (l, p) = gmm.predict(&[3.0, 0.0, 0.0, 0.0]);
        assert_eq!(l, 0);
        assert!(p > 0.9);
    }

    #[test]
    fn test_bic() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [[5.0, 0.0], [5.1, 0.1], [-5.0, 0.0], [-5.1, -0.1]];
        let labels = [0, 0, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        gmm.fit(&data, 10);
        let bic = gmm.bic(&data);
        assert!(bic.is_finite(), "BIC should be finite");
    }

    #[test]
    fn test_convergence_monotonic() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [
            [2.0, 1.0],
            [2.1, 0.9],
            [1.9, 1.1],
            [-2.0, -1.0],
            [-2.1, -0.9],
            [-1.9, -1.1],
        ];
        let labels = [0, 0, 0, 1, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);

        let mut prev_ll = f64::NEG_INFINITY;
        for _ in 0..10 {
            let ll = gmm.fit(&data, 1);
            if ll.is_finite() && prev_ll.is_finite() {
                assert!(
                    ll >= prev_ll - 1e-10,
                    "EM should be monotonic: {} >= {}",
                    ll,
                    prev_ll
                );
            }
            prev_ll = ll;
        }
    }

    #[test]
    fn test_single_component() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]];
        let labels = [0, 0, 0];
        gmm.init_from_labels(&data, &labels, 1);
        let ll = gmm.fit(&data, 5);
        assert!(ll.is_finite());
        let (l, p) = gmm.predict(&[0.0, 0.0]);
        assert_eq!(l, 0);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_elongated_cluster() {
        let mut gmm = GaussianMixture::<2, 4>::new(1e-4);
        let data = [
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
            [4.0, 0.0],
            [0.0, 5.0],
            [0.0, 6.0],
            [0.0, 7.0],
            [0.0, 4.0],
        ];
        let labels = [0, 0, 0, 0, 1, 1, 1, 1];
        gmm.init_from_labels(&data, &labels, 2);
        gmm.fit(&data, 10);

        let (l, _) = gmm.predict(&[3.0, 0.0]);
        assert_eq!(l, 0, "point along x-axis should be cluster 0");

        let (l2, _) = gmm.predict(&[0.0, 3.0]);
        assert_eq!(l2, 1, "point along y-axis should be cluster 1");
    }
}

// --- Kani formal verification ---

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove: init_from_centroids never panics for valid inputs.
    #[kani::proof]
    fn init_from_centroids_panic_free() {
        let mut gmm = GaussianMixture::<2, 2>::new(1e-4);
        let c0: [f64; 2] = [kani::any(), kani::any()];
        let c1: [f64; 2] = [kani::any(), kani::any()];
        for v in c0.iter().chain(c1.iter()) {
            kani::assume(v.is_finite());
            kani::assume(v.abs() < 100.0);
        }
        let n: usize = kani::any();
        kani::assume(n <= 2);
        gmm.init_from_centroids(&[c0, c1], n);
        assert!(gmm.n_components() <= 2);
    }

    /// Prove: predict never panics after init.
    #[kani::proof]
    fn predict_panic_free() {
        let mut gmm = GaussianMixture::<2, 2>::new(1e-2);
        let centroids = [[1.0, 0.0], [-1.0, 0.0]];
        gmm.init_from_centroids(&centroids, 2);
        gmm.update_cholesky(0);
        gmm.update_cholesky(1);
        let x: [f64; 2] = [kani::any(), kani::any()];
        kani::assume(x[0].is_finite() && x[1].is_finite());
        kani::assume(x[0].abs() < 100.0 && x[1].abs() < 100.0);
        let (k, p) = gmm.predict(&x);
        assert!(k < 2);
        assert!(p >= 0.0);
    }

    /// Prove: update_cholesky returns bool, never panics.
    #[kani::proof]
    fn cholesky_update_panic_free() {
        let mut gmm = GaussianMixture::<2, 2>::new(1e-2);
        let centroids = [[0.0, 0.0]];
        gmm.init_from_centroids(&centroids, 1);
        let result = gmm.update_cholesky(0);
        assert!(result);
    }
}
