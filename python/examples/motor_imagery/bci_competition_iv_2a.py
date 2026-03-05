"""BCI Competition IV Dataset 2a Validation -- zpybci on standard 4-class motor imagery.

Evaluates 4 pipelines on the 9-subject, 4-class motor imagery dataset (BNCI2014_001):
  P1: CSP + LDA (One-vs-Rest)
  P2: Covariance + TangentSpace + LDA
  P3: Covariance + MDM
  P4: xDAWN Covariance + MDM

Protocol: session-to-session transfer (train on session T, evaluate on session E).
This is the standard BCI Competition IV protocol used in published papers.

Channels: 16 sensorimotor (FC3-FC4, C5-C6, CP3-CP2), selected from 22 EEG.
Bandpass: 8-30 Hz via zpybci Butterworth filter.
Epoch: 0.5-4.0s relative to cue onset (skip 500ms reaction time).

Dependencies (benchmark only, not zpybci):
    pip install moabb scikit-learn

Usage:
    python python/examples/motor_imagery/bci_competition_iv_2a.py
"""

import os
import sys
import math

import numpy as np
import mne
from moabb.datasets import BNCI2014_001
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score

import zpybci as zbci

# Resolve sibling modules relative to this file
sys.path.insert(0, os.path.dirname(__file__))
from sklearn_compat import (
    CSPTransformer,
    TangentSpaceTransformer,
    CovarianceEstimator,
    MdmWrapper,
    XDawnWrapper,
)

# Suppress MNE info messages
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 250.0  # Dataset sample rate (Hz)
N_SUBJECTS = 9

# 16 sensorimotor channels (skip Fz at idx 0 and parietal channels 17-21)
MOTOR_CH_NAMES = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2",
]
N_CH = 16

# Motor imagery class labels (MOABB/BNCI2014_001 convention)
EVENT_LABELS = ["left_hand", "right_hand", "feet", "tongue"]

# Epoch window relative to cue onset
TMIN = 0.5  # skip 500ms reaction time
TMAX = 4.0  # end of imagery period

CHANCE = 0.25

# Published baselines (4-class, session-to-session transfer)
# Competition winner (FBCSP): ~63%. These are for simple pipelines without
# artifact rejection, filter banks, or hyperparameter tuning.
PUBLISHED = {
    "CSP+LDA": "~40-50%",
    "TS+LDA": "~60-68%",
    "MDM": "~55-62%",
    "xDAWN+MDM": "~55-60%",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_epochs(session_runs):
    """Extract motor imagery epochs from a single MOABB session.

    Parameters
    ----------
    session_runs : dict
        {run_name: mne.io.Raw} from dataset.get_data()

    Returns
    -------
    X : ndarray, shape (n_trials, n_samples, 16)
    y : ndarray, shape (n_trials,) with values 0-3
    """
    raws = [session_runs[k] for k in sorted(session_runs.keys())]
    raw = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws)

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Keep only motor imagery events
    target_event_id = {k: v for k, v in event_id.items() if k in EVENT_LABELS}

    if not target_event_id:
        # Try without underscores (some MOABB versions use spaces)
        alt_labels = [l.replace("_", " ") for l in EVENT_LABELS]
        target_event_id = {k: v for k, v in event_id.items() if k in alt_labels}

    if not target_event_id:
        available = list(event_id.keys())
        raise ValueError(f"No MI events found. Available annotations: {available}")

    epochs = mne.Epochs(
        raw, events, event_id=target_event_id,
        tmin=TMIN, tmax=TMAX, baseline=None,
        preload=True, verbose=False,
        picks=MOTOR_CH_NAMES,
    )

    X = epochs.get_data(copy=False)          # (n_trials, n_channels, n_samples)
    X = np.transpose(X, (0, 2, 1))           # (n_trials, n_samples, n_channels)
    X = np.ascontiguousarray(X, dtype=np.float64)

    # Map event codes to class indices 0-3
    # Build mapping from whatever event_id values to our canonical ordering
    label_order = EVENT_LABELS
    alt_order = [l.replace("_", " ") for l in EVENT_LABELS]
    label_to_class = {}
    for i, (std, alt) in enumerate(zip(label_order, alt_order)):
        if std in target_event_id:
            label_to_class[target_event_id[std]] = i
        elif alt in target_event_id:
            label_to_class[target_event_id[alt]] = i

    y = np.array([label_to_class[ev] for ev in epochs.events[:, 2]], dtype=np.int32)

    return X, y


def load_subject(dataset, subject_id):
    """Load train and test epochs for one subject.

    Returns
    -------
    X_train : (n_trials, n_samples, 16)
    y_train : (n_trials,) int 0-3
    X_test  : (n_trials, n_samples, 16)
    y_test  : (n_trials,) int 0-3
    """
    data = dataset.get_data(subjects=[subject_id])
    subj_sessions = data[subject_id]

    session_names = sorted(subj_sessions.keys())
    if len(session_names) != 2:
        raise ValueError(f"Expected 2 sessions, got {session_names}")

    # Sorted order: '0train' < '1test' (MOABB convention)
    X_train, y_train = _extract_epochs(subj_sessions[session_names[0]])
    X_test, y_test = _extract_epochs(subj_sessions[session_names[1]])

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X):
    """Per-channel 8-30 Hz Butterworth bandpass + CAR via zpybci.

    Parameters
    ----------
    X : (n_trials, n_samples, n_channels) float64

    Returns
    -------
    X_filtered : same shape, float64
    """
    n_trials, n_samples, n_channels = X.shape
    X_filtered = np.empty_like(X)

    filters = [zbci.IirFilter.butterworth_bandpass(FS, 8.0, 30.0)
               for _ in range(n_channels)]
    car = zbci.CAR(channels=n_channels)

    for i in range(n_trials):
        for f in filters:
            f.reset()
        trial = np.empty((n_samples, n_channels), dtype=np.float32)
        for ch in range(n_channels):
            signal = np.ascontiguousarray(X[i, :, ch], dtype=np.float32)
            trial[:, ch] = filters[ch].process(signal)
        referenced = car.process(trial)
        X_filtered[i] = referenced.astype(np.float64)

    return X_filtered


# ---------------------------------------------------------------------------
# Pipeline P1: CSP + LDA (One-vs-Rest)
# ---------------------------------------------------------------------------

def _regularize_covariances(covs, reg=1e-7):
    """Add regularization to ensure positive definiteness."""
    n, c, _ = covs.shape
    out = covs.copy()
    eye = np.eye(c)
    for i in range(n):
        out[i] += reg * np.trace(out[i]) / c * eye
    return out


def run_csp_lda(X_train, y_train, X_test, y_test):
    pipe = OneVsRestClassifier(
        SklearnPipeline([
            ("csp", CSPTransformer(channels=N_CH, filters=6)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ---------------------------------------------------------------------------
# Pipeline P2: Covariance + TangentSpace + LDA
# ---------------------------------------------------------------------------

class RegCovarianceEstimator(CovarianceEstimator):
    """CovarianceEstimator with trace-normalized regularization."""

    def __init__(self, channels, reg=1e-6):
        super().__init__(channels)
        self.reg = reg

    def transform(self, X):
        covs = super().transform(X)
        return _regularize_covariances(covs, self.reg)


class FrechetTangentSpaceTransformer(TangentSpaceTransformer):
    """TangentSpaceTransformer using Frechet (geometric) mean as reference."""

    def fit(self, X, y=None):
        self.ts_ = zbci.TangentSpace(channels=self.channels)
        X = np.asarray(X, dtype=np.float64)
        ref = zbci.frechet_mean(X)
        self.ts_.fit(ref)
        return self


def run_ts_lda(X_train, y_train, X_test, y_test):
    pipe = SklearnPipeline([
        ("cov", RegCovarianceEstimator(channels=N_CH)),
        ("ts", FrechetTangentSpaceTransformer(channels=N_CH)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ---------------------------------------------------------------------------
# Pipeline P3: Covariance + MDM
# ---------------------------------------------------------------------------

def run_mdm(X_train, y_train, X_test, y_test):
    pipe = SklearnPipeline([
        ("cov", RegCovarianceEstimator(channels=N_CH)),
        ("mdm", MdmWrapper(channels=N_CH)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ---------------------------------------------------------------------------
# Pipeline P4: xDAWN Covariance + MDM
# ---------------------------------------------------------------------------

def run_xdawn_mdm(X_train, y_train, X_test, y_test):
    """Per-class xDAWN -> concatenate -> covariance -> MDM.

    For each class k: fit xDAWN with binary labels (class k vs rest).
    Each xDAWN extracts n_filters=4 spatial filters.
    Concatenate all 4 xDAWN outputs: 4 classes * 4 filters = 16 channels.
    Compute covariance on the 16-channel concatenated signal -> (16, 16).
    Classify with MDM(16).
    """
    n_classes = len(EVENT_LABELS)
    n_filters = 4

    # Fit one xDAWN per class (one-vs-rest binary)
    xdawns = []
    for k in range(n_classes):
        y_binary = (y_train == k).astype(np.int64)
        xd = XDawnWrapper(channels=N_CH, filters=n_filters)
        xd.fit(X_train, y_binary)
        xdawns.append(xd)

    def transform_xdawn(X):
        parts = [xd.transform(X) for xd in xdawns]
        # Each part: (n_trials, n_samples, n_filters=4)
        # Concatenate along channel axis -> (n_trials, n_samples, 16)
        return np.concatenate(parts, axis=2)

    X_train_xd = transform_xdawn(X_train)
    X_test_xd = transform_xdawn(X_test)

    n_ch_xd = n_classes * n_filters  # 16

    cov_est = RegCovarianceEstimator(channels=n_ch_xd)
    cov_train = cov_est.transform(X_train_xd)
    cov_test = cov_est.transform(X_test_xd)

    mdm = MdmWrapper(channels=n_ch_xd)
    mdm.fit(cov_train, y_train)
    y_pred = mdm.predict(cov_test)
    return accuracy_score(y_test, y_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("BCI Competition IV Dataset 2a -- zpybci Validation")
    print(f"Dataset: BNCI2014_001, {N_SUBJECTS} subjects, 4-class motor imagery")
    print(f"Protocol: train on session T, evaluate on session E")
    print(f"Channels: {N_CH} sensorimotor")
    print(f"Preprocessing: 8-30 Hz bandpass + CAR (zpybci)")
    print(f"Epoch: {TMIN}-{TMAX}s post-cue")
    print()
    sys.stdout.flush()

    dataset = BNCI2014_001()

    pipelines = ["CSP+LDA", "TS+LDA", "MDM", "xDAWN+MDM"]
    pipeline_funcs = {
        "CSP+LDA": run_csp_lda,
        "TS+LDA": run_ts_lda,
        "MDM": run_mdm,
        "xDAWN+MDM": run_xdawn_mdm,
    }

    results = {p: [] for p in pipelines}

    for sid in range(1, N_SUBJECTS + 1):
        tag = f"S{sid:02d}"
        print(f"{tag}: loading ...", end=" ", flush=True)

        try:
            X_train, y_train, X_test, y_test = load_subject(dataset, sid)
        except Exception as exc:
            print(f"LOAD ERROR: {exc}")
            for p in pipelines:
                results[p].append(float("nan"))
            continue

        # Preprocess: bandpass + CAR
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)

        n_train, n_test = X_train.shape[0], X_test.shape[0]
        _, counts = np.unique(y_train, return_counts=True)
        balance = "/".join(str(c) for c in counts)
        print(f"{n_train} train, {n_test} test [{balance}]")
        sys.stdout.flush()

        for p_name in pipelines:
            try:
                acc = pipeline_funcs[p_name](X_train, y_train, X_test, y_test)
                results[p_name].append(acc)
                print(f"  {p_name}: {acc * 100:.1f}%")
            except Exception as exc:
                results[p_name].append(float("nan"))
                print(f"  {p_name}: ERROR -- {exc}")
            sys.stdout.flush()

        print()

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("RESULTS: Session-to-Session Transfer (Train T -> Eval E)")
    print("=" * 72)
    print()

    # Header
    header = f"| {'Subject':^9} |"
    for p in pipelines:
        header += f" {p:^11} |"
    print(header)

    sep = "|" + "-" * 11 + "|"
    for _ in pipelines:
        sep += "-" * 13 + "|"
    print(sep)

    # Per-subject rows
    for i, sid in enumerate(range(1, N_SUBJECTS + 1)):
        tag = f"S{sid:02d}"
        row = f"| {tag:^9} |"
        for p in pipelines:
            acc = results[p][i]
            cell = f"{acc * 100:.1f}%" if not math.isnan(acc) else "NaN"
            row += f" {cell:^11} |"
        print(row)

    print(sep)

    # Mean
    row = f"| {'Mean':^9} |"
    for p in pipelines:
        valid = [a for a in results[p] if not math.isnan(a)]
        m = float(np.mean(valid)) if valid else float("nan")
        cell = f"{m * 100:.1f}%" if not math.isnan(m) else "NaN"
        row += f" {cell:^11} |"
    print(row)

    # Std
    row = f"| {'Std':^9} |"
    for p in pipelines:
        valid = [a for a in results[p] if not math.isnan(a)]
        s = float(np.std(valid)) if valid else float("nan")
        cell = f"{s * 100:.1f}%" if not math.isnan(s) else "NaN"
        row += f" {cell:^11} |"
    print(row)

    print()

    # Comparison with published baselines
    print("Published baselines (4-class, session-to-session):")
    for p in pipelines:
        print(f"  {p}: {PUBLISHED[p]}")
    print(f"  Chance: {CHANCE * 100:.0f}%")
    print()

    # Summary
    for p in pipelines:
        valid = [a for a in results[p] if not math.isnan(a)]
        if valid:
            mean_acc = float(np.mean(valid))
            n_above = sum(1 for a in valid if a > CHANCE)
            print(f"{p}: mean {mean_acc * 100:.1f}%, "
                  f"{n_above}/{len(valid)} subjects above chance")


if __name__ == "__main__":
    main()
