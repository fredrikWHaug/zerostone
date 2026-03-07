"""Tests for sklearn-compatible zpybci wrappers.

Verifies that CSPTransformer, TangentSpaceTransformer, BandPowerTransformer,
CovarianceEstimator, LdaClassifier, and IcaTransformer implement the sklearn
interface correctly and compose into sklearn pipelines and cross_val_score.
"""

import numpy as np
import pytest

# Skip entire module if scikit-learn is not installed.
# sklearn is an optional dependency (not required by zpybci itself).
# Install with: pip install scikit-learn
pytest.importorskip("sklearn")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.pipeline import Pipeline                                  # noqa: E402
from sklearn.model_selection import cross_val_score                    # noqa: E402

from zpybci.sklearn import (  # noqa: E402
    CSPTransformer,
    TangentSpaceTransformer,
    BandPowerTransformer,
    CovarianceEstimator,
    LdaClassifier,
    IcaTransformer,
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


# ---------------------------------------------------------------------------
# LdaClassifier
# ---------------------------------------------------------------------------

LDA_FEATURES = 4
N_LDA_SAMPLES = 40


def _make_lda_data(n_samples=N_LDA_SAMPLES, n_features=LDA_FEATURES):
    """Generate linearly separable 2-class data."""
    rng = np.random.default_rng(99)
    X0 = rng.standard_normal((n_samples // 2, n_features)) - 2.0
    X1 = rng.standard_normal((n_samples // 2, n_features)) + 2.0
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=int)
    return X, y


class TestLdaClassifier:
    def test_fit_predict_shape(self):
        X, y = _make_lda_data()
        clf = LdaClassifier(features=LDA_FEATURES)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (N_LDA_SAMPLES,)

    def test_separable_accuracy(self):
        X, y = _make_lda_data()
        clf = LdaClassifier(features=LDA_FEATURES)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert acc > 0.9, f"Expected >90% on separable data, got {acc}"

    def test_predict_proba_shape(self):
        X, y = _make_lda_data()
        clf = LdaClassifier(features=LDA_FEATURES)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (N_LDA_SAMPLES, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_decision_function_shape(self):
        X, y = _make_lda_data()
        clf = LdaClassifier(features=LDA_FEATURES)
        clf.fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (N_LDA_SAMPLES,)

    def test_get_params(self):
        clf = LdaClassifier(features=8, shrinkage=0.05)
        params = clf.get_params()
        assert params["features"] == 8
        assert params["shrinkage"] == 0.05

    def test_cross_val_score(self):
        X, y = _make_lda_data(n_samples=50)
        clf = LdaClassifier(features=LDA_FEATURES)
        scores = cross_val_score(clf, X, y, cv=5)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()

    def test_pipeline_cov_ts_lda(self):
        """Full pipeline: raw epochs -> covariance -> tangent space -> LDA."""
        X = _make_trials(channels=TS_CHANNELS)
        y = _make_labels()
        n_features = TS_CHANNELS * (TS_CHANNELS + 1) // 2  # 10 for 4 channels
        # LDA needs a supported feature count -- 8 is closest below 10
        # Use sklearn LDA here since zbci.Lda needs exact feature match
        pipe = Pipeline([
            ("cov", CovarianceEstimator(channels=TS_CHANNELS)),
            ("ts", TangentSpaceTransformer(channels=TS_CHANNELS)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (N_TRIALS,)


# ---------------------------------------------------------------------------
# IcaTransformer
# ---------------------------------------------------------------------------

ICA_CHANNELS = 4
N_ICA_SAMPLES = 500


def _make_ica_data(n_samples=N_ICA_SAMPLES, channels=ICA_CHANNELS):
    """Generate mixed source data for ICA."""
    rng = np.random.default_rng(77)
    t = np.linspace(0, 1, n_samples)
    sources = np.zeros((n_samples, channels))
    sources[:, 0] = np.sin(2 * np.pi * 5 * t)
    sources[:, 1] = np.sign(np.sin(2 * np.pi * 11 * t))
    for c in range(2, channels):
        sources[:, c] = rng.standard_normal(n_samples)
    A = rng.standard_normal((channels, channels))
    return sources @ A.T


class TestIcaTransformer:
    def test_fit_transform_shape(self):
        X = _make_ica_data()
        ica = IcaTransformer(channels=ICA_CHANNELS)
        ica.fit(X)
        out = ica.transform(X)
        assert out.shape == (N_ICA_SAMPLES, ICA_CHANNELS)

    def test_output_finite(self):
        X = _make_ica_data()
        ica = IcaTransformer(channels=ICA_CHANNELS)
        ica.fit(X)
        out = ica.transform(X)
        assert np.isfinite(out).all()

    def test_fit_transform_convenience(self):
        X = _make_ica_data()
        ica = IcaTransformer(channels=ICA_CHANNELS)
        out = ica.fit_transform(X)
        assert out.shape == (N_ICA_SAMPLES, ICA_CHANNELS)

    def test_exclude_components(self):
        X = _make_ica_data()
        ica_no_exclude = IcaTransformer(channels=ICA_CHANNELS)
        ica_no_exclude.fit(X)
        out_full = ica_no_exclude.transform(X)

        ica_exclude = IcaTransformer(channels=ICA_CHANNELS, exclude=[0])
        ica_exclude.fit(X)
        out_cleaned = ica_exclude.transform(X)

        # Removing a component should change the output
        assert not np.allclose(out_full, out_cleaned, atol=1e-6)

    def test_get_params(self):
        ica = IcaTransformer(channels=8, contrast="exp", max_iter=100)
        params = ica.get_params()
        assert params["channels"] == 8
        assert params["contrast"] == "exp"
        assert params["max_iter"] == 100

    def test_different_contrasts(self):
        X = _make_ica_data()
        for contrast in ["logcosh", "exp", "cube"]:
            ica = IcaTransformer(channels=ICA_CHANNELS, contrast=contrast)
            out = ica.fit_transform(X)
            assert out.shape == (N_ICA_SAMPLES, ICA_CHANNELS)
            assert np.isfinite(out).all()
