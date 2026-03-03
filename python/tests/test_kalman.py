"""Tests for Kalman filter Python bindings."""

import numpy as np
import pytest
import zpybci as zbci


class TestKalmanConstruction:
    def test_create_s2o1(self):
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])
        kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)
        assert kf.state_dim == 2
        assert kf.obs_dim == 1

    @pytest.mark.parametrize(
        "s,o",
        [(2, 1), (2, 2), (4, 2), (4, 4), (6, 3), (6, 6), (8, 4), (8, 8)],
    )
    def test_supported_dimensions(self, s, o):
        F = np.eye(s)
        H = np.eye(o, s)
        Q = np.eye(s) * 0.01
        R = np.eye(o) * 1.0
        kf = zbci.KalmanFilter(state_dim=s, obs_dim=o, F=F, H=H, Q=Q, R=R)
        assert kf.state_dim == s
        assert kf.obs_dim == o

    def test_unsupported_dimensions(self):
        with pytest.raises(ValueError, match="Unsupported"):
            zbci.KalmanFilter(
                state_dim=3,
                obs_dim=1,
                F=np.eye(3),
                H=np.eye(1, 3),
                Q=np.eye(3),
                R=np.eye(1),
            )

    def test_custom_initial_state(self):
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])
        x0 = np.array([5.0, 3.0])
        kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R, x0=x0)
        np.testing.assert_allclose(kf.state, x0)

    def test_repr(self):
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])
        kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)
        assert "KalmanFilter" in repr(kf)
        assert "state_dim=2" in repr(kf)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="F must be"):
            zbci.KalmanFilter(
                state_dim=2,
                obs_dim=1,
                F=np.eye(3),  # wrong shape
                H=np.array([[1.0, 0.0]]),
                Q=np.eye(2),
                R=np.array([[1.0]]),
            )


class TestKalman1DTracking:
    def make_1d_filter(self, dt=1.0, q=0.01, r=1.0):
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * q
        R = np.array([[r]])
        return zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)

    def test_constant_velocity_tracking(self):
        kf = self.make_1d_filter()
        rng = np.random.default_rng(42)

        true_vel = 1.0
        errors = []
        for t in range(50):
            kf.predict()
            true_pos = true_vel * (t + 1)
            noisy_pos = true_pos + rng.normal(0, 1.0)
            kf.update(np.array([noisy_pos]))
            errors.append(abs(kf.state[0] - true_pos))

        # Late errors should be small
        late_avg = np.mean(errors[30:])
        assert late_avg < 2.0, f"Late average error too large: {late_avg}"

        # Velocity should converge
        vel_err = abs(kf.state[1] - true_vel)
        assert vel_err < 1.0, f"Velocity error too large: {vel_err}"

    def test_update_returns_innovation(self):
        kf = self.make_1d_filter()
        kf.predict()
        inn = kf.update(np.array([5.0]))
        assert inn.shape == (1,)
        # Innovation should be z - H*x = 5.0 - 0.0 = 5.0
        assert abs(inn[0] - 5.0) < 0.01

    def test_heavy_smoothing(self):
        # Large R -> heavy smoothing -> less variation in output
        kf = self.make_1d_filter(q=0.001, r=10.0)
        rng = np.random.default_rng(99)

        raw = []
        filtered = []
        for _ in range(200):
            kf.predict()
            z = rng.normal(0, 3.0)
            kf.update(np.array([z]))
            raw.append(z)
            filtered.append(kf.state[0])

        raw_diff_var = np.var(np.diff(raw))
        filt_diff_var = np.var(np.diff(filtered))
        assert filt_diff_var < raw_diff_var * 0.5

    def test_predict_only_grows_covariance(self):
        kf = self.make_1d_filter()
        p0_trace = np.trace(kf.covariance)

        for _ in range(10):
            kf.predict()

        p_trace = np.trace(kf.covariance)
        assert p_trace > p0_trace

    def test_predict_only_follows_dynamics(self):
        x0 = np.array([0.0, 1.0])
        kf = self.make_1d_filter()
        kf.reset(x0=x0)

        for _ in range(10):
            kf.predict()

        # pos = 0 + 1*10 = 10, vel = 1
        np.testing.assert_allclose(kf.state[0], 10.0, atol=1e-10)
        np.testing.assert_allclose(kf.state[1], 1.0, atol=1e-10)


class TestKalman2DTracking:
    def make_2d_filter(self, dt=0.1):
        F = np.array(
            [
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        H = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        Q = np.eye(4) * 0.01
        R = np.eye(2) * 0.5
        x0 = np.array([0.0, 1.0, 0.0, 0.5])
        return zbci.KalmanFilter(
            state_dim=4, obs_dim=2, F=F, H=H, Q=Q, R=R, x0=x0
        )

    def test_reduces_noise(self):
        dt = 0.1
        kf = self.make_2d_filter(dt)
        rng = np.random.default_rng(123)

        raw_errors = []
        filt_errors = []
        for t in range(100):
            kf.predict()
            true_px = 1.0 * (t + 1) * dt
            true_py = 0.5 * (t + 1) * dt
            noise = rng.normal(0, 0.7, size=2)
            z = np.array([true_px + noise[0], true_py + noise[1]])
            kf.update(z)

            raw_err = np.linalg.norm(noise)
            filt_err = np.sqrt(
                (kf.state[0] - true_px) ** 2 + (kf.state[2] - true_py) ** 2
            )
            raw_errors.append(raw_err)
            filt_errors.append(filt_err)

        raw_rms = np.sqrt(np.mean(np.array(raw_errors[50:]) ** 2))
        filt_rms = np.sqrt(np.mean(np.array(filt_errors[50:]) ** 2))
        assert filt_rms < raw_rms, f"Filter RMS {filt_rms} >= raw RMS {raw_rms}"


class TestKalmanSteadyState:
    def test_gain_converges(self):
        F = np.array([[1, 1], [0, 1]], dtype=np.float64)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[1.0]])
        P0 = np.eye(2) * 10.0
        kf = zbci.KalmanFilter(
            state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R, P0=P0
        )

        prev_diag = np.array([10.0, 10.0])
        converged_at = None
        for t in range(100):
            kf.predict()
            kf.update(np.array([0.0]))
            cov = kf.covariance
            diag = np.diag(cov)
            max_change = np.max(np.abs(diag - prev_diag))
            if max_change < 1e-6 and converged_at is None:
                converged_at = t
            prev_diag = diag.copy()

        assert converged_at is not None, "Gain did not converge"
        assert converged_at < 30, f"Convergence too slow: {converged_at} iters"


class TestKalmanReset:
    def test_reset_restores_state(self):
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])
        kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)

        for _ in range(10):
            kf.predict()
            kf.update(np.array([5.0]))

        new_x0 = np.array([100.0, 0.0])
        new_P0 = np.eye(2) * 5.0
        kf.reset(x0=new_x0, P0=new_P0)

        np.testing.assert_allclose(kf.state, new_x0)
        np.testing.assert_allclose(kf.covariance, new_P0)

    def test_reset_defaults(self):
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[1.0]])
        kf = zbci.KalmanFilter(state_dim=2, obs_dim=1, F=F, H=H, Q=Q, R=R)

        for _ in range(10):
            kf.predict()
            kf.update(np.array([5.0]))

        kf.reset()
        np.testing.assert_allclose(kf.state, [0.0, 0.0])


class TestKalmanIdentityObservation:
    def test_noise_reduction(self):
        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 1.0
        x0 = np.array([5.0, 3.0])
        kf = zbci.KalmanFilter(
            state_dim=2, obs_dim=2, F=F, H=H, Q=Q, R=R, x0=x0
        )

        true_state = np.array([5.0, 3.0])
        rng = np.random.default_rng(77)
        raw_err_sum = 0.0
        filt_err_sum = 0.0

        for _ in range(100):
            kf.predict()
            z = true_state + rng.normal(0, 1.0, size=2)
            kf.update(z)
            raw_err_sum += np.linalg.norm(z - true_state)
            filt_err_sum += np.linalg.norm(kf.state - true_state)

        assert filt_err_sum < raw_err_sum
