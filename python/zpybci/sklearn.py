"""Sklearn-compatible wrappers for zpybci signal processing classes.

Thin wrappers that implement the sklearn BaseEstimator + TransformerMixin
interface, enabling zpybci to plug into sklearn pipelines, cross-validation,
and grid search.

Usage::

    from zpybci.sklearn import CSPTransformer, TangentSpaceTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    pipe = make_pipeline(
        CovarianceEstimator(channels=16),
        TangentSpaceTransformer(channels=16),
        LinearDiscriminantAnalysis(),
    )
    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

Supported transformers:
    CSPTransformer          -- wraps zbci.AdaptiveCsp
    TangentSpaceTransformer -- wraps zbci.TangentSpace
    BandPowerTransformer    -- wraps zbci.MultiBandPower
    CovarianceEstimator     -- wraps zbci.OnlineCov
    MdmWrapper              -- wraps zbci.MdmClassifier
    XDawnWrapper            -- wraps zbci.XDawnTransformer
    LdaClassifier           -- wraps zbci.Lda
    IcaTransformer          -- wraps zbci.Ica

Channel constraints (inherited from zpybci enum dispatch):
    AdaptiveCsp      : channels in {4, 8, 16, 32, 64}, filters in {2, 4, 6}
    TangentSpace     : channels in {4, 8, 16, 32}
    MultiBandPower   : channels in {1, 4, 8, 16, 32, 64}, fft_size in {256, 512, 1024}
    OnlineCov        : channels in {4, 8, 16, 32, 64}
    MdmClassifier    : channels in {4, 8, 16, 32}
    XDawnTransformer : channels in {2, 4, 8, 16, 32, 64}, filters in {1, 2, 4, 6}
    Lda              : features in {2, 4, 6, 8, 12, 16, 32, 64}
    Ica              : channels in {4, 8, 16, 32, 64}
"""

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
except ImportError:
    raise ImportError(
        "scikit-learn is required for zpybci.sklearn. "
        "Install it with: pip install scikit-learn"
    )

import zpybci as zbci


# ---------------------------------------------------------------------------
# CSPTransformer
# ---------------------------------------------------------------------------

class CSPTransformer(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns as a sklearn transformer.

    Wraps zbci.AdaptiveCsp. Expects binary class labels (0 and 1).

    Input:
        X : (n_trials, n_samples, n_channels) float64 array
        y : (n_trials,) int array, binary {0, 1}

    Output:
        (n_trials, n_filters) float64 -- log-variance of CSP-filtered trials

    Parameters
    ----------
    channels : int
        Number of EEG channels. Must be 4, 8, 16, 32, or 64.
    filters : int
        Number of spatial filters to extract. Must be 2, 4, or 6.
    min_samples : int
        Minimum samples per class required before filter computation.
    regularization : float
        Regularization for covariance estimation.
    """

    def __init__(self, channels, filters=4, min_samples=100, regularization=1e-6):
        self.channels = channels
        self.filters = filters
        self.min_samples = min_samples
        self.regularization = regularization

    def fit(self, X, y):
        self.csp_ = zbci.AdaptiveCsp(
            channels=self.channels,
            filters=self.filters,
            min_samples=self.min_samples,
            regularization=self.regularization,
        )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        for trial, label in zip(X, y):
            if label == 0:
                self.csp_.update_class1(trial)
            else:
                self.csp_.update_class2(trial)

        self.csp_.recompute_filters()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_trials = X.shape[0]
        features = np.empty((n_trials, self.filters), dtype=np.float64)

        for i, trial in enumerate(X):
            filtered = self.csp_.apply_block(trial)
            features[i] = np.log(np.var(filtered, axis=0) + 1e-10)

        return features


# ---------------------------------------------------------------------------
# TangentSpaceTransformer
# ---------------------------------------------------------------------------

class TangentSpaceTransformer(BaseEstimator, TransformerMixin):
    """Riemannian tangent space projection as a sklearn transformer.

    Wraps zbci.TangentSpace. Fits the reference point as the arithmetic mean
    of the training covariance matrices.

    Input:
        X : (n_trials, n_channels, n_channels) float64 -- covariance matrices

    Output:
        (n_trials, n_channels*(n_channels+1)//2) float64 -- tangent vectors

    Parameters
    ----------
    channels : int
        Matrix dimension. Must be 4, 8, 16, or 32.
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        self.ts_ = zbci.TangentSpace(channels=self.channels)
        X = np.asarray(X, dtype=np.float64)
        ref = np.mean(X, axis=0)
        self.ts_.fit(ref)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_trials = X.shape[0]
        n_features = self.channels * (self.channels + 1) // 2
        features = np.empty((n_trials, n_features), dtype=np.float64)

        for i, cov in enumerate(X):
            features[i] = self.ts_.transform(cov)

        return features


# ---------------------------------------------------------------------------
# BandPowerTransformer
# ---------------------------------------------------------------------------

DEFAULT_BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0)]


class BandPowerTransformer(BaseEstimator, TransformerMixin):
    """Per-channel frequency band power as a sklearn transformer.

    Wraps zbci.MultiBandPower. The last ``fft_size`` samples of each trial
    are used for the PSD estimate.

    Input:
        X : (n_trials, n_samples, n_channels) float32 array

    Output:
        (n_trials, n_channels * n_bands) float32 -- interleaved channel-then-band

    Parameters
    ----------
    channels : int
        Number of channels. Must be 1, 4, 8, 16, 32, or 64.
    sample_rate : float
        Sample rate in Hz.
    fft_size : int
        FFT window length. Must be 256, 512, or 1024.
    bands : list of (float, float), optional
        Frequency bands as (low_hz, high_hz). Defaults to theta/alpha/beta/gamma.
    """

    def __init__(self, channels, sample_rate, fft_size=256, bands=None):
        self.channels = channels
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bands = bands if bands is not None else DEFAULT_BANDS

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_trials = X.shape[0]
        n_bands = len(self.bands)
        n_features = self.channels * n_bands
        features = np.empty((n_trials, n_features), dtype=np.float32)

        bp = zbci.MultiBandPower(
            fft_size=self.fft_size,
            channels=self.channels,
            sample_rate=float(self.sample_rate),
        )

        for i, trial in enumerate(X):
            window = trial[-self.fft_size:, :]
            signals = np.ascontiguousarray(window.T)
            bp.compute(signals)
            band_powers = [bp.band_power(low, high) for low, high in self.bands]
            features[i] = np.array(band_powers).T.ravel()

        return features


# ---------------------------------------------------------------------------
# CovarianceEstimator
# ---------------------------------------------------------------------------

class CovarianceEstimator(BaseEstimator, TransformerMixin):
    """Per-trial covariance matrix estimation as a sklearn transformer.

    Wraps zbci.OnlineCov.

    Input:
        X : (n_trials, n_samples, n_channels) float64 array

    Output:
        (n_trials, n_channels, n_channels) float64 -- covariance matrices

    Parameters
    ----------
    channels : int
        Number of channels. Must be 4, 8, 16, 32, or 64.
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_trials = X.shape[0]
        covariances = np.empty(
            (n_trials, self.channels, self.channels), dtype=np.float64
        )

        cov = zbci.OnlineCov(channels=self.channels)

        for i, trial in enumerate(X):
            cov.reset()
            for sample in trial:
                cov.update(sample)
            covariances[i] = cov.covariance

        return covariances


# ---------------------------------------------------------------------------
# MdmWrapper
# ---------------------------------------------------------------------------

class MdmWrapper(BaseEstimator, ClassifierMixin):
    """Minimum Distance to Mean classifier as a sklearn classifier.

    Wraps zbci.MdmClassifier. Expects covariance matrices as input (3D array).

    Input:
        X : (n_trials, n_channels, n_channels) float64 -- covariance matrices
        y : (n_trials,) int array

    Parameters
    ----------
    channels : int
        Matrix dimension. Must be 4, 8, 16, or 32.
    """

    def __init__(self, channels):
        self.channels = channels

    def fit(self, X, y):
        self.mdm_ = zbci.MdmClassifier(channels=self.channels)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mdm_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array(self.mdm_.predict(X), dtype=np.int64)

    def score(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        return self.mdm_.score(X, y)


# ---------------------------------------------------------------------------
# XDawnWrapper
# ---------------------------------------------------------------------------

class XDawnWrapper(BaseEstimator, TransformerMixin):
    """xDAWN spatial filter as a sklearn transformer.

    Wraps zbci.XDawnTransformer. Learns spatial filters that maximize the
    signal-to-signal-plus-noise ratio of event-related potentials.

    Input:
        X : (n_trials, n_samples, n_channels) float64 array
        y : (n_trials,) int array, binary {0, 1}

    Output:
        (n_trials, n_samples, n_filters) float64 -- spatially filtered epochs

    Parameters
    ----------
    channels : int
        Number of channels. Must be 2, 4, 8, 16, 32, or 64.
    filters : int
        Number of spatial filters. Must be 1, 2, 4, or 6.
    regularization : float
        Regularization for covariance matrices.
    """

    def __init__(self, channels, filters=2, regularization=1e-6):
        self.channels = channels
        self.filters = filters
        self.regularization = regularization

    def fit(self, X, y):
        self.xdawn_ = zbci.XDawnTransformer(
            channels=self.channels,
            filters=self.filters,
            regularization=self.regularization,
        )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.xdawn_.fit(X, y)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self.xdawn_.transform(X))


# ---------------------------------------------------------------------------
# LdaClassifier
# ---------------------------------------------------------------------------

class LdaClassifier(BaseEstimator, ClassifierMixin):
    """Fisher's LDA as a sklearn classifier.

    Wraps zbci.Lda. Binary classification with shrinkage regularization.

    Input:
        X : (n_samples, n_features) float64 array
        y : (n_samples,) int array, binary {0, 1}

    Parameters
    ----------
    features : int
        Number of input features. Must be 2, 4, 6, 8, 12, 16, 32, or 64.
    shrinkage : float
        Shrinkage regularization parameter (0 to 1).
    """

    def __init__(self, features, shrinkage=0.01):
        self.features = features
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.lda_ = zbci.Lda(features=self.features, shrinkage=self.shrinkage)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.lda_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array(self.lda_.predict(X), dtype=np.int64)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        proba_1 = np.array(self.lda_.predict_proba(X), dtype=np.float64)
        return np.column_stack([1.0 - proba_1, proba_1])

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array(self.lda_.decision_function(X), dtype=np.float64)

    def score(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        return self.lda_.score(X, y)


# ---------------------------------------------------------------------------
# IcaTransformer
# ---------------------------------------------------------------------------

class IcaTransformer(BaseEstimator, TransformerMixin):
    """FastICA as a sklearn transformer.

    Wraps zbci.Ica. Fits ICA on training data and transforms new data
    into independent component space.

    Input:
        X : (n_samples, n_channels) float64 array

    Output:
        (n_samples, n_channels) float64 -- independent components

    Parameters
    ----------
    channels : int
        Number of channels. Must be 4, 8, 16, 32, or 64.
    contrast : str
        Contrast function: 'logcosh', 'exp', or 'cube'.
    max_iter : int
        Maximum iterations for convergence.
    tolerance : float
        Convergence tolerance.
    exclude : list of int, optional
        Component indices to zero out during inverse_transform.
        Useful for artifact removal (e.g., exclude=[0] to remove
        the component most correlated with blink artifacts).
    """

    def __init__(self, channels, contrast="logcosh", max_iter=200,
                 tolerance=1e-4, exclude=None):
        self.channels = channels
        self.contrast = contrast
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.exclude = exclude

    def fit(self, X, y=None):
        self.ica_ = zbci.Ica(channels=self.channels, contrast=self.contrast)
        X = np.asarray(X, dtype=np.float64)
        self.ica_.fit(X, max_iter=self.max_iter, tolerance=self.tolerance)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.exclude is not None:
            return self.ica_.remove_components(X, self.exclude)
        return self.ica_.transform(X)
