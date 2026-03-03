//! Kalman filter for state estimation and decoder output smoothing.
//!
//! Standard linear Kalman filter used for velocity decoding in closed-loop BCI.
//! Supports rectangular observation matrices (O x S) for mapping between
//! state space and observation space.
//!
//! Uses Joseph form for covariance update to preserve positive definiteness,
//! and Cholesky-based solve for the Kalman gain to avoid explicit matrix inversion.
//!
//! # Type Parameters
//!
//! * `S` - State dimension
//! * `O` - Observation dimension
//! * `SM` - S*S (state covariance size)
//! * `OM` - O*O (observation covariance size)
//! * `SOM` - S*O (cross-dimension matrix size)

use crate::linalg::Matrix;

/// Errors that can occur during Kalman filter operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KalmanError {
    /// Innovation covariance is not positive definite (Cholesky failed)
    InnovationNotPositiveDefinite,
}

/// Linear Kalman filter for state estimation.
///
/// # Type Parameters
///
/// * `S` - State dimension (e.g., 4 for [px, vx, py, vy])
/// * `O` - Observation dimension (e.g., 2 for [px, py])
/// * `SM` - Must equal S*S
/// * `OM` - Must equal O*O
/// * `SOM` - Must equal S*O
pub struct KalmanFilter<
    const S: usize,
    const O: usize,
    const SM: usize,
    const OM: usize,
    const SOM: usize,
> {
    /// State estimate
    x: [f64; S],
    /// State covariance (S x S)
    p: Matrix<S, SM>,
    /// State transition matrix (S x S)
    f: Matrix<S, SM>,
    /// Observation matrix (O x S), row-major
    h: [f64; SOM],
    /// Process noise covariance (S x S)
    q: Matrix<S, SM>,
    /// Measurement noise covariance (O x O)
    r: Matrix<O, OM>,
    /// Last innovation (for diagnostics)
    last_innovation: Option<[f64; O]>,
}

impl<const S: usize, const O: usize, const SM: usize, const OM: usize, const SOM: usize>
    KalmanFilter<S, O, SM, OM, SOM>
{
    /// Create a new Kalman filter.
    ///
    /// # Arguments
    ///
    /// * `f` - State transition matrix (S x S)
    /// * `h` - Observation matrix (O x S), stored row-major in flat array
    /// * `q` - Process noise covariance (S x S)
    /// * `r` - Measurement noise covariance (O x O)
    /// * `x0` - Initial state estimate
    /// * `p0` - Initial state covariance (S x S)
    pub fn new(
        f: Matrix<S, SM>,
        h: [f64; SOM],
        q: Matrix<S, SM>,
        r: Matrix<O, OM>,
        x0: [f64; S],
        p0: Matrix<S, SM>,
    ) -> Self {
        assert!(SM == S * S, "SM must equal S * S");
        assert!(OM == O * O, "OM must equal O * O");
        assert!(SOM == S * O, "SOM must equal S * O");

        Self {
            x: x0,
            p: p0,
            f,
            h,
            q,
            r,
            last_innovation: None,
        }
    }

    /// Predict step: propagate state and covariance forward.
    ///
    /// x = F * x
    /// P = F * P * F^T + Q
    pub fn predict(&mut self) {
        // x = F * x
        self.x = sq_matvec::<S, SM>(&self.f, &self.x);

        // P = F * P * F^T + Q
        let ft = self.f.transpose();
        self.p = self.f.matmul(&self.p).matmul(&ft).add(&self.q);

        // Enforce symmetry
        symmetrize::<S, SM>(&mut self.p);
    }

    /// Update step: incorporate a new measurement using Joseph form.
    ///
    /// Returns the innovation vector (z - H*x_predicted).
    pub fn update(&mut self, z: &[f64; O]) -> Result<[f64; O], KalmanError> {
        // Innovation: y = z - H * x
        let hx = h_mul_x::<S, O, SOM>(&self.h, &self.x);
        let mut y = [0.0f64; O];
        for i in 0..O {
            y[i] = z[i] - hx[i];
        }

        // Innovation covariance: S_inn = H * P * H^T + R (O x O)
        let s_inn = h_p_ht::<S, O, SM, OM, SOM>(&self.h, &self.p).add(&self.r);

        // Compute P * H^T (S x O)
        let pht = p_times_ht::<S, O, SM, SOM>(&self.p, &self.h);

        // Cholesky decompose S_inn
        let l = s_inn
            .cholesky()
            .map_err(|_| KalmanError::InnovationNotPositiveDefinite)?;

        // Compute Kalman gain K (S x O) row-by-row using Cholesky solve
        // For each row i of K: solve S_inn * K[i]^T = PHt[i]^T
        let mut k = [0.0f64; SOM];
        for i in 0..S {
            // Extract row i of PHt (length O)
            let mut rhs = [0.0f64; O];
            for j in 0..O {
                rhs[j] = pht[i * O + j];
            }
            // Solve L * L^T * x = rhs
            let fwd = l.forward_substitute(&rhs);
            let sol = l.backward_substitute(&fwd);
            // Store as row i of K
            for j in 0..O {
                k[i * O + j] = sol[j];
            }
        }

        // x = x + K * y
        let ky = k_mul_y::<S, O, SOM>(&k, &y);
        #[allow(clippy::needless_range_loop)]
        for i in 0..S {
            self.x[i] += ky[i];
        }

        // Joseph form: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        let kh = k_mul_h::<S, O, SM, SOM>(&k, &self.h);
        let mut i_kh = Matrix::<S, SM>::identity();
        for idx in 0..SM {
            i_kh.data_mut()[idx] -= kh.data()[idx];
        }
        let i_kh_t = i_kh.transpose();
        let krkt = k_r_kt::<S, O, SM, OM, SOM>(&k, &self.r);
        self.p = i_kh.matmul(&self.p).matmul(&i_kh_t).add(&krkt);

        // Enforce symmetry
        symmetrize::<S, SM>(&mut self.p);

        self.last_innovation = Some(y);
        Ok(y)
    }

    /// Get the current state estimate.
    pub fn state(&self) -> &[f64; S] {
        &self.x
    }

    /// Get the current state covariance.
    pub fn covariance(&self) -> &Matrix<S, SM> {
        &self.p
    }

    /// Get the last innovation vector (z - H*x_predicted).
    pub fn innovation(&self) -> Option<&[f64; O]> {
        self.last_innovation.as_ref()
    }

    /// Reset the filter to a new initial state and covariance.
    pub fn reset(&mut self, x0: [f64; S], p0: Matrix<S, SM>) {
        self.x = x0;
        self.p = p0;
        self.last_innovation = None;
    }
}

// -- Rectangular matrix helpers (private) --

/// Multiply square matrix (S x S) by vector (S) -> (S)
#[allow(clippy::needless_range_loop)]
fn sq_matvec<const S: usize, const SM: usize>(mat: &Matrix<S, SM>, x: &[f64; S]) -> [f64; S] {
    let mut result = [0.0f64; S];
    for i in 0..S {
        let mut sum = 0.0;
        for j in 0..S {
            sum += mat.get(i, j) * x[j];
        }
        result[i] = sum;
    }
    result
}

/// Multiply H (O x S) by vector x (S) -> y (O)
fn h_mul_x<const S: usize, const O: usize, const SOM: usize>(
    h: &[f64; SOM],
    x: &[f64; S],
) -> [f64; O] {
    let mut result = [0.0f64; O];
    for i in 0..O {
        let mut sum = 0.0;
        for j in 0..S {
            sum += h[i * S + j] * x[j];
        }
        result[i] = sum;
    }
    result
}

/// Compute H * P * H^T where H is (O x S), P is (S x S), result is (O x O)
fn h_p_ht<const S: usize, const O: usize, const SM: usize, const OM: usize, const SOM: usize>(
    h: &[f64; SOM],
    p: &Matrix<S, SM>,
) -> Matrix<O, OM> {
    // First compute HP = H * P (O x S)
    let mut hp = [0.0f64; SOM];
    for i in 0..O {
        for j in 0..S {
            let mut sum = 0.0;
            for k in 0..S {
                sum += h[i * S + k] * p.get(k, j);
            }
            hp[i * S + j] = sum;
        }
    }
    // Then compute HP * H^T (O x O)
    let mut result = Matrix::<O, OM>::zeros();
    for i in 0..O {
        for j in 0..O {
            let mut sum = 0.0;
            for k in 0..S {
                sum += hp[i * S + k] * h[j * S + k]; // H^T[k][j] = H[j][k]
            }
            result.set(i, j, sum);
        }
    }
    result
}

/// Compute P * H^T where P is (S x S), H is (O x S), result is (S x O)
fn p_times_ht<const S: usize, const O: usize, const SM: usize, const SOM: usize>(
    p: &Matrix<S, SM>,
    h: &[f64; SOM],
) -> [f64; SOM] {
    let mut result = [0.0f64; SOM];
    for i in 0..S {
        for j in 0..O {
            let mut sum = 0.0;
            for k in 0..S {
                sum += p.get(i, k) * h[j * S + k]; // H^T[k][j] = H[j][k]
            }
            result[i * O + j] = sum;
        }
    }
    result
}

/// Multiply K (S x O) by vector y (O) -> result (S)
fn k_mul_y<const S: usize, const O: usize, const SOM: usize>(
    k: &[f64; SOM],
    y: &[f64; O],
) -> [f64; S] {
    let mut result = [0.0f64; S];
    for i in 0..S {
        let mut sum = 0.0;
        for j in 0..O {
            sum += k[i * O + j] * y[j];
        }
        result[i] = sum;
    }
    result
}

/// Multiply K (S x O) by H (O x S) -> result (S x S)
fn k_mul_h<const S: usize, const O: usize, const SM: usize, const SOM: usize>(
    k: &[f64; SOM],
    h: &[f64; SOM],
) -> Matrix<S, SM> {
    let mut result = Matrix::<S, SM>::zeros();
    for i in 0..S {
        for j in 0..S {
            let mut sum = 0.0;
            for m in 0..O {
                sum += k[i * O + m] * h[m * S + j];
            }
            result.set(i, j, sum);
        }
    }
    result
}

/// Compute K * R * K^T where K is (S x O), R is (O x O), result is (S x S)
fn k_r_kt<const S: usize, const O: usize, const SM: usize, const OM: usize, const SOM: usize>(
    k: &[f64; SOM],
    r: &Matrix<O, OM>,
) -> Matrix<S, SM> {
    // First compute KR = K * R (S x O)
    let mut kr = [0.0f64; SOM];
    for i in 0..S {
        for j in 0..O {
            let mut sum = 0.0;
            for m in 0..O {
                sum += k[i * O + m] * r.get(m, j);
            }
            kr[i * O + j] = sum;
        }
    }
    // Then compute KR * K^T (S x S)
    let mut result = Matrix::<S, SM>::zeros();
    for i in 0..S {
        for j in 0..S {
            let mut sum = 0.0;
            for m in 0..O {
                sum += kr[i * O + m] * k[j * O + m]; // K^T[m][j] = K[j][m]
            }
            result.set(i, j, sum);
        }
    }
    result
}

/// Enforce symmetry: P = (P + P^T) / 2
fn symmetrize<const C: usize, const M: usize>(p: &mut Matrix<C, M>) {
    for i in 0..C {
        for j in (i + 1)..C {
            let avg = (p.get(i, j) + p.get(j, i)) * 0.5;
            p.set(i, j, avg);
            p.set(j, i, avg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
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
        // Box-Muller for Gaussian
        fn gaussian(&mut self, mean: f64, std: f64) -> f64 {
            let u1 = (self.next_u64() % 1_000_000 + 1) as f64 / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as f64 / 1_000_000.0;
            let z = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2);
            mean + z * std
        }
    }

    #[test]
    fn test_1d_constant_velocity() {
        // State: [position, velocity], Observe: [position]
        // S=2, O=1, SM=4, OM=1, SOM=2
        let dt = 1.0;
        let f = Matrix::<2, 4>::new([1.0, dt, 0.0, 1.0]);
        let h = [1.0, 0.0]; // observe position only
        let q = Matrix::<2, 4>::new([0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::<1, 1>::new([1.0]);
        let x0 = [0.0, 1.0]; // start at 0, velocity=1
        let p0 = Matrix::<2, 4>::identity();

        let mut kf = KalmanFilter::<2, 1, 4, 1, 2>::new(f, h, q, r, x0, p0);

        let mut rng = Rng::new(42);
        let true_velocity = 1.0;
        let mut errors = Vec::new();

        for t in 0..50 {
            kf.predict();
            let true_pos = true_velocity * (t + 1) as f64;
            let noisy_pos = true_pos + rng.gaussian(0.0, 1.0);
            kf.update(&[noisy_pos]).unwrap();
            let err = libm::fabs(kf.state()[0] - true_pos);
            errors.push(err);
        }

        // Average error in last 20 steps should be small
        let late_avg: f64 = errors[30..].iter().sum::<f64>() / 20.0;
        assert!(late_avg < 2.0, "Late average error too large: {}", late_avg);

        // Velocity estimate should converge near true velocity
        let vel_err = libm::fabs(kf.state()[1] - true_velocity);
        assert!(vel_err < 1.0, "Velocity error too large: {}", vel_err);
    }

    #[test]
    fn test_2d_tracking() {
        // State: [px, vx, py, vy], Observe: [px, py]
        // S=4, O=2, SM=16, OM=4, SOM=8
        let dt = 0.1;
        #[rustfmt::skip]
        let f = Matrix::<4, 16>::new([
            1.0, dt,  0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, dt,
            0.0, 0.0, 0.0, 1.0,
        ]);
        #[rustfmt::skip]
        let h = [
            1.0, 0.0, 0.0, 0.0,  // observe px
            0.0, 0.0, 1.0, 0.0,  // observe py
        ];
        let mut q_data = [0.0f64; 16];
        for i in 0..4 {
            q_data[i * 4 + i] = 0.01;
        }
        let q = Matrix::<4, 16>::new(q_data);
        let r = Matrix::<2, 4>::new([0.5, 0.0, 0.0, 0.5]);
        let x0 = [0.0, 1.0, 0.0, 0.5]; // moving diagonally
        let p0 = Matrix::<4, 16>::identity();

        let mut kf = KalmanFilter::<4, 2, 16, 4, 8>::new(f, h, q, r, x0, p0);

        let mut rng = Rng::new(123);
        let mut raw_errors = Vec::new();
        let mut filtered_errors = Vec::new();

        for t in 0..100 {
            kf.predict();
            let true_px = 1.0 * (t + 1) as f64 * dt;
            let true_py = 0.5 * (t + 1) as f64 * dt;
            let noise_x = rng.gaussian(0.0, 0.7);
            let noise_y = rng.gaussian(0.0, 0.7);
            let z = [true_px + noise_x, true_py + noise_y];
            kf.update(&z).unwrap();

            let raw_err = libm::sqrt(noise_x * noise_x + noise_y * noise_y);
            let filt_err = libm::sqrt(
                (kf.state()[0] - true_px) * (kf.state()[0] - true_px)
                    + (kf.state()[2] - true_py) * (kf.state()[2] - true_py),
            );
            raw_errors.push(raw_err);
            filtered_errors.push(filt_err);
        }

        // Filtered RMS should be less than raw RMS in the second half
        let raw_rms: f64 = libm::sqrt(raw_errors[50..].iter().map(|e| e * e).sum::<f64>() / 50.0);
        let filt_rms: f64 =
            libm::sqrt(filtered_errors[50..].iter().map(|e| e * e).sum::<f64>() / 50.0);
        assert!(
            filt_rms < raw_rms,
            "Filter RMS {} not less than raw RMS {}",
            filt_rms,
            raw_rms
        );
    }

    #[test]
    fn test_steady_state_convergence() {
        // 1D position tracking: check gain stabilizes
        let f = Matrix::<2, 4>::new([1.0, 1.0, 0.0, 1.0]);
        let h = [1.0, 0.0];
        let q = Matrix::<2, 4>::new([0.1, 0.0, 0.0, 0.1]);
        let r = Matrix::<1, 1>::new([1.0]);
        let x0 = [0.0, 0.0];
        let p0 = Matrix::<2, 4>::new([10.0, 0.0, 0.0, 10.0]);

        let mut kf = KalmanFilter::<2, 1, 4, 1, 2>::new(f, h, q, r, x0, p0);

        let mut prev_p_diag = [10.0, 10.0];
        let mut converged_at = None;

        for t in 0..100 {
            kf.predict();
            kf.update(&[0.0]).unwrap();

            let p = kf.covariance();
            let p_diag = [p.get(0, 0), p.get(1, 1)];
            let max_change = libm::fmax(
                libm::fabs(p_diag[0] - prev_p_diag[0]),
                libm::fabs(p_diag[1] - prev_p_diag[1]),
            );

            if max_change < 1e-6 && converged_at.is_none() {
                converged_at = Some(t);
            }

            prev_p_diag = p_diag;
        }

        assert!(
            converged_at.is_some(),
            "Kalman gain did not converge in 100 iterations"
        );
        assert!(
            converged_at.unwrap() < 30,
            "Convergence too slow: {} iterations",
            converged_at.unwrap()
        );
    }

    #[test]
    fn test_heavy_smoothing() {
        // Large R relative to Q -> heavy smoothing
        let f = Matrix::<1, 1>::new([1.0]);
        let h = [1.0];
        let q = Matrix::<1, 1>::new([0.001]);
        let r = Matrix::<1, 1>::new([10.0]);
        let x0 = [0.0];
        let p0 = Matrix::<1, 1>::new([1.0]);

        let mut kf = KalmanFilter::<1, 1, 1, 1, 1>::new(f, h, q, r, x0, p0);

        let mut rng = Rng::new(99);
        let mut raw_var = 0.0;
        let mut filt_var = 0.0;
        let n = 200;

        let mut raw_vals = vec![0.0f64; n];
        let mut filt_vals = vec![0.0f64; n];

        for i in 0..n {
            kf.predict();
            let z = rng.gaussian(0.0, 3.0);
            kf.update(&[z]).unwrap();
            raw_vals[i] = z;
            filt_vals[i] = kf.state()[0];
        }

        // Compute variance of differences (smoothness proxy)
        for i in 1..n {
            let rd = raw_vals[i] - raw_vals[i - 1];
            let fd = filt_vals[i] - filt_vals[i - 1];
            raw_var += rd * rd;
            filt_var += fd * fd;
        }

        assert!(
            filt_var < raw_var * 0.5,
            "Filtered not smooth enough: filt_var={}, raw_var={}",
            filt_var,
            raw_var
        );
    }

    #[test]
    fn test_identity_observation() {
        // H=I, S=O=2. Filter should reduce noise on i.i.d. observations.
        let f = Matrix::<2, 4>::identity();
        let h = [1.0, 0.0, 0.0, 1.0]; // identity
        let q = Matrix::<2, 4>::new([0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::<2, 4>::new([1.0, 0.0, 0.0, 1.0]);
        let x0 = [5.0, 3.0];
        let p0 = Matrix::<2, 4>::identity();

        let mut kf = KalmanFilter::<2, 2, 4, 4, 4>::new(f, h, q, r, x0, p0);

        let true_state = [5.0, 3.0];
        let mut rng = Rng::new(77);
        let mut raw_err_sum = 0.0;
        let mut filt_err_sum = 0.0;
        let n = 100;

        for _ in 0..n {
            kf.predict();
            let z = [
                true_state[0] + rng.gaussian(0.0, 1.0),
                true_state[1] + rng.gaussian(0.0, 1.0),
            ];
            kf.update(&z).unwrap();

            let raw_err = libm::sqrt(
                (z[0] - true_state[0]) * (z[0] - true_state[0])
                    + (z[1] - true_state[1]) * (z[1] - true_state[1]),
            );
            let filt_err = libm::sqrt(
                (kf.state()[0] - true_state[0]) * (kf.state()[0] - true_state[0])
                    + (kf.state()[1] - true_state[1]) * (kf.state()[1] - true_state[1]),
            );

            raw_err_sum += raw_err;
            filt_err_sum += filt_err;
        }

        assert!(
            filt_err_sum < raw_err_sum,
            "Filter error {} not less than raw error {}",
            filt_err_sum,
            raw_err_sum
        );
    }

    #[test]
    fn test_predict_only() {
        // Repeated predict without update: state follows F dynamics, P grows
        let f = Matrix::<2, 4>::new([1.0, 1.0, 0.0, 1.0]);
        let h = [1.0, 0.0];
        let q = Matrix::<2, 4>::new([0.1, 0.0, 0.0, 0.1]);
        let r = Matrix::<1, 1>::new([1.0]);
        let x0 = [0.0, 1.0];
        let p0 = Matrix::<2, 4>::identity();

        let mut kf = KalmanFilter::<2, 1, 4, 1, 2>::new(f, h, q, r, x0, p0);

        let p0_trace = kf.covariance().get(0, 0) + kf.covariance().get(1, 1);

        for _ in 0..10 {
            kf.predict();
        }

        // State should follow constant velocity model
        // After 10 steps: pos = 0 + 1*10 = 10, vel = 1
        assert!(
            libm::fabs(kf.state()[0] - 10.0) < 1e-10,
            "Position should be 10, got {}",
            kf.state()[0]
        );
        assert!(
            libm::fabs(kf.state()[1] - 1.0) < 1e-10,
            "Velocity should be 1, got {}",
            kf.state()[1]
        );

        // P should have grown
        let p_trace = kf.covariance().get(0, 0) + kf.covariance().get(1, 1);
        assert!(
            p_trace > p0_trace,
            "Covariance trace should grow: {} vs {}",
            p_trace,
            p0_trace
        );
    }

    #[test]
    fn test_reset() {
        let f = Matrix::<2, 4>::new([1.0, 1.0, 0.0, 1.0]);
        let h = [1.0, 0.0];
        let q = Matrix::<2, 4>::new([0.1, 0.0, 0.0, 0.1]);
        let r = Matrix::<1, 1>::new([1.0]);
        let x0 = [0.0, 1.0];
        let p0 = Matrix::<2, 4>::identity();

        let mut kf = KalmanFilter::<2, 1, 4, 1, 2>::new(f, h, q, r, x0, p0);

        // Run a few steps
        for _ in 0..10 {
            kf.predict();
            kf.update(&[5.0]).unwrap();
        }
        assert!(kf.innovation().is_some());

        // Reset
        let new_x0 = [100.0, 0.0];
        let new_p0 = Matrix::<2, 4>::new([5.0, 0.0, 0.0, 5.0]);
        kf.reset(new_x0, new_p0);

        assert_eq!(kf.state()[0], 100.0);
        assert_eq!(kf.state()[1], 0.0);
        assert_eq!(kf.covariance().get(0, 0), 5.0);
        assert!(kf.innovation().is_none());
    }
}
