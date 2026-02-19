"""Motor Imagery Classification Benchmark — Week 6 Thursday deliverable.

Runs a binary left/right motor imagery classification benchmark on real
Physionet EEGMMIDB data. Reports per-subject and mean accuracy for two
pipelines:
  1. CSP + LDA
  2. Riemannian (covariance -> tangent space) + LDA

Preprocessing: bandpass (8-30 Hz) -> notch (60 Hz) -> CAR on all 64 channels,
then select 4 central motor cortex channels (C5, C3, C1, Cz) before classification.

Usage:
    python examples/motor_imagery/benchmark.py
"""
import os
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline

import zpybci as zbci

# Resolve sibling modules (load_data, sklearn_compat) relative to this file
sys.path.insert(0, os.path.dirname(__file__))
from load_data import stream_subject, CHANNELS, SAMPLES_PER_TRIAL
from sklearn_compat import CSPTransformer, TangentSpaceTransformer, CovarianceEstimator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 160.0
N_SUBJECTS = 9
MAX_WORKERS = 6

# Motor cortex channels: C5=7, C3=8, C1=9, Cz=10 (0-indexed in 64-ch cap)
MOTOR_CH = [7, 8, 9, 10]
N_MOTOR_CH = len(MOTOR_CH)

CHANCE = 0.50
ABOVE_CHANCE_LABEL = "**"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_filters():
    """Create one bandpass filter per channel plus a shared notch."""
    bandpass_filters = [
        zbci.IirFilter.butterworth_bandpass(FS, 8.0, 30.0) for _ in range(CHANNELS)
    ]
    notch = zbci.NotchFilter.powerline_60hz(FS, channels=CHANNELS)
    car = zbci.CAR(channels=CHANNELS)
    return bandpass_filters, notch, car


def _reset_filters(bandpass_filters, notch):
    for f in bandpass_filters:
        f.reset()
    notch.reset()


def _preprocess_trial(epoch, bandpass_filters, notch, car):
    """Run one epoch through bandpass -> notch -> CAR.

    epoch : (640, 64) float32
    Returns: (640, 64) float64
    """
    # IirFilter.process requires float32; keep as float32 through bandpass
    epoch = np.asarray(epoch, dtype=np.float32)

    bandpassed = np.empty_like(epoch)
    for ch in range(CHANNELS):
        col = np.ascontiguousarray(epoch[:, ch])
        bandpassed[:, ch] = bandpass_filters[ch].process(col)

    denoised = notch.process(bandpassed)
    referenced = car.process(denoised)
    return referenced.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def preprocess_subject(trials):
    """Apply the full pipeline to a list of (epoch, label) pairs.

    Returns:
        X_motor : (n_trials, 640, 4) float64  — motor cortex channels only
        y       : (n_trials,) int array
    """
    bandpass_filters, notch, car = _build_filters()

    X_list = []
    y_list = []

    for epoch, label in trials:
        _reset_filters(bandpass_filters, notch)
        processed = _preprocess_trial(epoch, bandpass_filters, notch, car)
        X_list.append(processed[:, MOTOR_CH])  # (640, 4)
        y_list.append(label)

    if not X_list:
        return (
            np.empty((0, SAMPLES_PER_TRIAL, N_MOTOR_CH), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
        )

    X_motor = np.stack(X_list, axis=0)  # (n_trials, 640, 4)
    y = np.array(y_list, dtype=np.int32)
    return X_motor, y


# ---------------------------------------------------------------------------
# Classification pipelines
# ---------------------------------------------------------------------------

def _csp_lda_pipeline():
    return SklearnPipeline([
        ("csp", CSPTransformer(channels=N_MOTOR_CH, filters=2)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _riem_lda_pipeline():
    return SklearnPipeline([
        ("cov", CovarianceEstimator(channels=N_MOTOR_CH)),
        ("ts",  TangentSpaceTransformer(channels=N_MOTOR_CH)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def run_cv(pipeline, X, y, n_splits=5):
    """5-fold stratified CV. Returns mean accuracy, or NaN on error."""
    if len(np.unique(y)) < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))
    except Exception as exc:
        print(f"    CV error: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    subject_ids = list(range(1, N_SUBJECTS + 1))

    print(f"Motor Imagery Benchmark — {N_SUBJECTS} subjects, 5-fold CV")
    print(f"Downloading data ({MAX_WORKERS} parallel workers) ...")
    sys.stdout.flush()

    # Phase 1: parallel download (I/O-bound)
    subject_trials = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(stream_subject, sid): sid for sid in subject_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                trials = future.result()
                subject_trials[sid] = trials
                print(f"  S{sid:03d}: {len(trials)} trials downloaded")
            except Exception as exc:
                print(f"  S{sid:03d}: download error — {exc}")
                subject_trials[sid] = []
            sys.stdout.flush()

    print()
    print("Processing and classifying ...")
    sys.stdout.flush()

    # Phase 2: sequential processing + classification
    results = []

    for sid in subject_ids:
        trials = subject_trials.get(sid, [])
        tag = f"S{sid:03d}"

        if not trials:
            results.append((tag, 0, float("nan"), float("nan")))
            continue

        X_motor, y = preprocess_subject(trials)
        n_trials = X_motor.shape[0]

        if n_trials == 0:
            results.append((tag, 0, float("nan"), float("nan")))
            continue

        print(f"  {tag}: {n_trials} trials, class balance {int(np.sum(y==0))}/{int(np.sum(y==1))}")
        sys.stdout.flush()

        acc_csp  = run_cv(_csp_lda_pipeline(),  X_motor, y)
        acc_riem = run_cv(_riem_lda_pipeline(), X_motor, y)

        results.append((tag, n_trials, acc_csp, acc_riem))

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    print()
    header = f"{'Subject':<10} {'Trials':<8} {'CSP+LDA':<13} {'Riem+LDA':<13} {'vs Chance'}"
    sep    = "-" * len(header)
    print(header)
    print(sep)

    csp_accs  = []
    riem_accs = []

    for tag, n_trials, acc_csp, acc_riem in results:
        def fmt(v):
            return f"{v*100:.1f}%" if not math.isnan(v) else "NaN"

        above_csp  = not math.isnan(acc_csp)  and acc_csp  > CHANCE
        above_riem = not math.isnan(acc_riem) and acc_riem > CHANCE
        marker = ABOVE_CHANCE_LABEL if (above_csp or above_riem) else ""

        print(f"{tag:<10} {n_trials:<8} {fmt(acc_csp):<13} {fmt(acc_riem):<13} {marker}")

        if not math.isnan(acc_csp):
            csp_accs.append(acc_csp)
        if not math.isnan(acc_riem):
            riem_accs.append(acc_riem)

    print(sep)
    mean_csp  = float(np.mean(csp_accs))  if csp_accs  else float("nan")
    mean_riem = float(np.mean(riem_accs)) if riem_accs else float("nan")

    def fmt(v):
        return f"{v*100:.1f}%" if not math.isnan(v) else "NaN"

    print(f"{'Mean':<10} {'':<8} {fmt(mean_csp):<13} {fmt(mean_riem):<13}")

    print()
    print(f"Chance level : {CHANCE*100:.0f}%")
    print(f"Literature   : CSP+LDA ~75-85%, Riemannian ~80-90% (binary L/R motor imagery)")

    n_csp_above  = sum(1 for _, _, acc, _    in results if not math.isnan(acc) and acc > CHANCE)
    n_riem_above = sum(1 for _, _, _,  acc   in results if not math.isnan(acc) and acc > CHANCE)
    print(f"CSP above chance : {n_csp_above}/{N_SUBJECTS} subjects")
    print(f"Riem above chance: {n_riem_above}/{N_SUBJECTS} subjects")


if __name__ == "__main__":
    main()
