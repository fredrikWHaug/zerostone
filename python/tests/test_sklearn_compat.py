"""Tests for sklearn-compatible npyci wrappers.

Verifies that CSPTransformer, TangentSpaceTransformer, BandPowerTransformer,
and CovarianceEstimator implement the sklearn interface correctly and compose
into sklearn pipelines and cross_val_score.
"""

import sys
import os

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Resolve the module from the examples directory without installing it
_COMPAT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "examples", "motor_imagery"
)
sys.path.insert(0, os.path.abspath(_COMPAT_DIR))

from sklearn_compat import (  # noqa: E402
    CSPTransformer,
    TangentSpaceTransformer,
    BandPowerTransformer,
    CovarianceEstimator,
)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TRIALS = 20
N_SAMPLES = 256

# CSP uses 4 channels: the Jacobi eigensolver (30 sweeps, hard-coded in the
# Rust binding) only converges reliably on 4×4 matrices with random data.
# Existing Python tests (test_csp.py) also use 4 channels for this reason.
CSP_CHANNELS = 4
CSP_FILTERS = 2

# TangentSpace supports 4, 8, 16, 32. Use 4 to keep matrices small and
# keep Jacobi convergence reliable.
TS_CHANNELS = 4

# CovarianceEstimator and BandPowerTransformer have no convergence dependency.
COV_CHANNELS = 8
BP_CHANNELS = 8


def _make_labels(n_trials=N_TRIALS):
    return np.array([i % 2 for i in range(n_trials)], dtype=int)


def _make_trials_csp(n_trials=N_TRIALS, n_samples=N_SAMPLES):
    """Generate (n_trials, n_samples, 4) trials for CSP.

    Class 0: signal in channels 0-1. Class 1: signal in channels 2-3.
    Mirrors the data pattern used in the existing test_csp.py tests.
    """
    y = _make_labels(n_trials)
    X = np.zeros((n_trials, n_samples, CSP_CHANNELS), dtype=np.float64)
    for i, label in enumerate(y):
        if label == 0:
            X[i, :, 0] = RNG.standard_normal(n_samples)
            X[i, :, 1] = RNG.standard_normal(n_samples)
        else:
            X[i, :, 2] = RNG.standard_normal(n_samples)
            X[i, :, 3] = RNG.standard_normal(n_samples)
    return X


def _make_trials(n_trials=N_TRIALS, n_samples=N_SAMPLES, channels=COV_CHANNELS,
                 dtype=np.float64):
    return RNG.standard_normal((n_trials, n_samples, channels)).astype(dtype)


def _make_covs(n_trials=N_TRIALS, channels=TS_CHANNELS):
    """Generate near-identity SPD matrices with small perturbations.
    Near-identity ensures Jacobi convergence in TangentSpace.fit().
    """
    covs = np.empty((n_trials, channels, channels), dtype=np.float64)
    for i in range(n_trials):
        A = RNG.standard_normal((channels, channels)) * 0.1
        covs[i] = A @ A.T + np.eye(channels)
    return covs


# ---------------------------------------------------------------------------
# CSPTransformer
# ---------------------------------------------------------------------------

class TestCSPTransformer:
    def test_fit_transform_shape(self):
        X = _make_trials_csp()
        y = _make_labels()
        t = CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)
        t.fit(X, y)
        out = t.transform(X)
        assert out.shape == (N_TRIALS, CSP_FILTERS)

    def test_output_dtype(self):
        X = _make_trials_csp()
        y = _make_labels()
        t = CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)
        t.fit(X, y)
        out = t.transform(X)
        assert out.dtype == np.float64

    def test_output_finite(self):
        X = _make_trials_csp()
        y = _make_labels()
        t = CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)
        t.fit(X, y)
        out = t.transform(X)
        assert np.isfinite(out).all()

    def test_fit_transform_convenience(self):
        X = _make_trials_csp()
        y = _make_labels()
        t = CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)
        out = t.fit_transform(X, y)
        assert out.shape == (N_TRIALS, CSP_FILTERS)

    def test_get_params(self):
        t = CSPTransformer(channels=4, filters=2, min_samples=50)
        params = t.get_params()
        assert params["channels"] == 4
        assert params["filters"] == 2
        assert params["min_samples"] == 50

    def test_different_filter_counts(self):
        X = _make_trials_csp()
        y = _make_labels()
        # channels=4 supports filters=2 and 4 only
        for filters in [2, 4]:
            t = CSPTransformer(channels=CSP_CHANNELS, filters=filters)
            t.fit(X, y)
            out = t.transform(X)
            assert out.shape == (N_TRIALS, filters)

    def test_sklearn_pipeline(self):
        X = _make_trials_csp()
        y = _make_labels()
        pipe = Pipeline([
            ("csp", CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_TRIALS,)

    def test_cross_val_score(self):
        X = _make_trials_csp()
        y = _make_labels()
        pipe = Pipeline([
            ("csp", CSPTransformer(channels=CSP_CHANNELS, filters=CSP_FILTERS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        scores = cross_val_score(pipe, X, y, cv=5)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
        assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ---------------------------------------------------------------------------
# TangentSpaceTransformer
# ---------------------------------------------------------------------------

class TestTangentSpaceTransformer:
    def test_fit_transform_shape(self):
        X = _make_covs()
        t = TangentSpaceTransformer(channels=TS_CHANNELS)
        t.fit(X)
        out = t.transform(X)
        expected_features = TS_CHANNELS * (TS_CHANNELS + 1) // 2
        assert out.shape == (N_TRIALS, expected_features)

    def test_output_dtype(self):
        X = _make_covs()
        t = TangentSpaceTransformer(channels=TS_CHANNELS)
        t.fit(X)
        out = t.transform(X)
        assert out.dtype == np.float64

    def test_output_finite(self):
        X = _make_covs()
        t = TangentSpaceTransformer(channels=TS_CHANNELS)
        t.fit(X)
        out = t.transform(X)
        assert np.isfinite(out).all()

    def test_fit_transform_convenience(self):
        X = _make_covs()
        t = TangentSpaceTransformer(channels=TS_CHANNELS)
        out = t.fit_transform(X)
        assert out.shape == (N_TRIALS, TS_CHANNELS * (TS_CHANNELS + 1) // 2)

    def test_get_params(self):
        t = TangentSpaceTransformer(channels=4)
        params = t.get_params()
        assert params["channels"] == 4

    def test_sklearn_pipeline_with_lda(self):
        X = _make_covs()
        y = _make_labels()
        pipe = Pipeline([
            ("ts", TangentSpaceTransformer(channels=TS_CHANNELS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_TRIALS,)

    def test_cross_val_score(self):
        X = _make_covs()
        y = _make_labels()
        pipe = Pipeline([
            ("ts", TangentSpaceTransformer(channels=TS_CHANNELS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        scores = cross_val_score(pipe, X, y, cv=5)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
        assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ---------------------------------------------------------------------------
# BandPowerTransformer
# ---------------------------------------------------------------------------

class TestBandPowerTransformer:
    def test_fit_transform_shape(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        t = BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0, fft_size=256)
        t.fit(X)
        out = t.transform(X)
        # 4 default bands x 8 channels = 32
        assert out.shape == (N_TRIALS, 4 * BP_CHANNELS)

    def test_custom_bands(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        bands = [(8.0, 13.0), (13.0, 30.0)]
        t = BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0, fft_size=256, bands=bands)
        out = t.fit_transform(X)
        assert out.shape == (N_TRIALS, len(bands) * BP_CHANNELS)

    def test_output_dtype(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        t = BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0)
        out = t.fit_transform(X)
        assert out.dtype == np.float32

    def test_output_finite(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        t = BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0)
        out = t.fit_transform(X)
        assert np.isfinite(out).all()

    def test_output_nonnegative(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        t = BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0)
        out = t.fit_transform(X)
        assert (out >= 0.0).all()

    def test_get_params(self):
        t = BandPowerTransformer(channels=8, sample_rate=160.0, fft_size=512)
        params = t.get_params()
        assert params["channels"] == 8
        assert params["sample_rate"] == 160.0
        assert params["fft_size"] == 512

    def test_sklearn_pipeline(self):
        X = _make_trials(channels=BP_CHANNELS, dtype=np.float32)
        y = _make_labels()
        pipe = Pipeline([
            ("bp", BandPowerTransformer(channels=BP_CHANNELS, sample_rate=160.0)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_TRIALS,)


# ---------------------------------------------------------------------------
# CovarianceEstimator
# ---------------------------------------------------------------------------

class TestCovarianceEstimator:
    def test_fit_transform_shape(self):
        X = _make_trials(channels=COV_CHANNELS)
        t = CovarianceEstimator(channels=COV_CHANNELS)
        out = t.fit_transform(X)
        assert out.shape == (N_TRIALS, COV_CHANNELS, COV_CHANNELS)

    def test_output_dtype(self):
        X = _make_trials(channels=COV_CHANNELS)
        t = CovarianceEstimator(channels=COV_CHANNELS)
        out = t.fit_transform(X)
        assert out.dtype == np.float64

    def test_output_finite(self):
        X = _make_trials(channels=COV_CHANNELS)
        t = CovarianceEstimator(channels=COV_CHANNELS)
        out = t.fit_transform(X)
        assert np.isfinite(out).all()

    def test_output_symmetric(self):
        X = _make_trials(channels=COV_CHANNELS)
        t = CovarianceEstimator(channels=COV_CHANNELS)
        out = t.fit_transform(X)
        for cov in out:
            assert np.allclose(cov, cov.T, atol=1e-10)

    def test_output_positive_semidefinite(self):
        X = _make_trials(n_samples=500, channels=COV_CHANNELS)
        t = CovarianceEstimator(channels=COV_CHANNELS)
        out = t.fit_transform(X)
        for cov in out:
            eigenvalues = np.linalg.eigvalsh(cov)
            assert (eigenvalues >= -1e-8).all()

    def test_get_params(self):
        t = CovarianceEstimator(channels=8)
        params = t.get_params()
        assert params["channels"] == 8

    def test_pipeline_cov_then_tangent(self):
        """Verify CovarianceEstimator -> TangentSpaceTransformer pipeline.
        Uses 4 channels so TangentSpace Jacobi converges reliably.
        """
        X = _make_trials(channels=TS_CHANNELS)
        y = _make_labels()
        pipe = Pipeline([
            ("cov", CovarianceEstimator(channels=TS_CHANNELS)),
            ("ts", TangentSpaceTransformer(channels=TS_CHANNELS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_TRIALS,)

    def test_cross_val_score_riemannian(self):
        """Full Riemannian pipeline: covariance -> tangent space -> LDA.
        Uses 4 channels so TangentSpace Jacobi converges reliably.
        """
        X = _make_trials(channels=TS_CHANNELS)
        y = _make_labels()
        pipe = Pipeline([
            ("cov", CovarianceEstimator(channels=TS_CHANNELS)),
            ("ts", TangentSpaceTransformer(channels=TS_CHANNELS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        scores = cross_val_score(pipe, X, y, cv=5)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
