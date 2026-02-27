"""Motor Imagery Classification Benchmark v2 — Multi-channel + Transfer Learning.

Extends v1 with:
  - Three channel configurations (4, 8, 16) showing accuracy scaling
  - Three pipelines: CSP+LDA, Cov+TangentSpace+LDA, Cov+MDM
  - Cross-subject transfer learning via Riemannian recentering

Usage:
    python examples/motor_imagery/benchmark_v2.py
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

sys.path.insert(0, os.path.dirname(__file__))
from load_data import stream_subject, CHANNELS, SAMPLES_PER_TRIAL
from sklearn_compat import (
    CSPTransformer,
    TangentSpaceTransformer,
    CovarianceEstimator,
    MdmWrapper,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 160.0
N_SUBJECTS = 9
MAX_WORKERS = 6

CHANNEL_CONFIGS = {
    "4ch": [7, 8, 9, 10],
    "8ch": [3, 4, 7, 8, 9, 10, 15, 16],
    "16ch": [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19],
}

PIPELINES = ["CSP+LDA", "TS+LDA", "MDM"]

CHANCE = 0.50


# ---------------------------------------------------------------------------
# Preprocessing (same as v1: bandpass -> notch -> CAR on all 64 channels)
# ---------------------------------------------------------------------------

def _build_filters():
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
    epoch = np.asarray(epoch, dtype=np.float32)
    bandpassed = np.empty_like(epoch)
    for ch in range(CHANNELS):
        col = np.ascontiguousarray(epoch[:, ch])
        bandpassed[:, ch] = bandpass_filters[ch].process(col)
    denoised = notch.process(bandpassed)
    referenced = car.process(denoised)
    return referenced.astype(np.float64)


def preprocess_subject(trials):
    """Preprocess all trials, returning full 64-channel data."""
    bandpass_filters, notch, car = _build_filters()
    X_list, y_list = [], []
    for epoch, label in trials:
        _reset_filters(bandpass_filters, notch)
        processed = _preprocess_trial(epoch, bandpass_filters, notch, car)
        X_list.append(processed)
        y_list.append(label)
    if not X_list:
        return np.empty((0, SAMPLES_PER_TRIAL, CHANNELS), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _csp_lda(n_ch):
    filters = min(6, n_ch)
    return SklearnPipeline([
        ("csp", CSPTransformer(channels=n_ch, filters=filters)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _ts_lda(n_ch):
    return SklearnPipeline([
        ("cov", CovarianceEstimator(channels=n_ch)),
        ("ts", TangentSpaceTransformer(channels=n_ch)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _mdm(n_ch):
    return SklearnPipeline([
        ("cov", CovarianceEstimator(channels=n_ch)),
        ("mdm", MdmWrapper(channels=n_ch)),
    ])


PIPELINE_BUILDERS = {
    "CSP+LDA": _csp_lda,
    "TS+LDA": _ts_lda,
    "MDM": _mdm,
}


def run_cv(pipeline, X, y, n_splits=5):
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
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    subject_ids = list(range(1, N_SUBJECTS + 1))

    print(f"Motor Imagery Benchmark v2 -- {N_SUBJECTS} subjects, 5-fold CV")
    print(f"Channel configs: {', '.join(f'{k}({len(v)})' for k,v in CHANNEL_CONFIGS.items())}")
    print(f"Pipelines: {', '.join(PIPELINES)}")
    print(f"Downloading data ({MAX_WORKERS} parallel workers) ...")
    sys.stdout.flush()

    # Phase 1: parallel download
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
                print(f"  S{sid:03d}: download error -- {exc}")
                subject_trials[sid] = []
            sys.stdout.flush()

    print()
    print("Processing and classifying ...")
    sys.stdout.flush()

    # Phase 2: preprocess all subjects (keep full 64 channels)
    subject_data = {}
    for sid in subject_ids:
        trials = subject_trials.get(sid, [])
        if not trials:
            continue
        X_full, y = preprocess_subject(trials)
        if X_full.shape[0] > 0:
            subject_data[sid] = (X_full, y)
            print(f"  S{sid:03d}: {X_full.shape[0]} trials, balance {int(np.sum(y==0))}/{int(np.sum(y==1))}")
            sys.stdout.flush()

    # Phase 3: run pipelines per channel config
    # results[ch_name][pipeline_name] = list of per-subject accuracies
    results = {ch: {p: [] for p in PIPELINES} for ch in CHANNEL_CONFIGS}

    for ch_name, ch_indices in CHANNEL_CONFIGS.items():
        n_ch = len(ch_indices)
        print(f"\n--- {ch_name} ({n_ch} channels) ---")
        sys.stdout.flush()

        for sid in subject_ids:
            if sid not in subject_data:
                for p in PIPELINES:
                    results[ch_name][p].append(float("nan"))
                continue

            X_full, y = subject_data[sid]
            X_sel = X_full[:, :, ch_indices]

            for p_name in PIPELINES:
                pipe = PIPELINE_BUILDERS[p_name](n_ch)
                acc = run_cv(pipe, X_sel, y)
                results[ch_name][p_name].append(acc)

            accs = [results[ch_name][p][-1] for p in PIPELINES]
            acc_strs = [f"{a*100:.1f}%" if not math.isnan(a) else "NaN" for a in accs]
            print(f"  S{sid:03d}: " + "  ".join(f"{p}={a}" for p, a in zip(PIPELINES, acc_strs)))
            sys.stdout.flush()

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Build header
    col_headers = []
    for ch_name in CHANNEL_CONFIGS:
        for p_name in PIPELINES:
            col_headers.append(f"{ch_name}/{p_name}")

    header = f"{'Subject':<10}"
    for h in col_headers:
        header += f" {h:>12}"
    print(header)
    print("-" * len(header))

    for i, sid in enumerate(subject_ids):
        row = f"S{sid:03d}      "
        for ch_name in CHANNEL_CONFIGS:
            for p_name in PIPELINES:
                acc = results[ch_name][p_name][i]
                cell = f"{acc*100:.1f}%" if not math.isnan(acc) else "NaN"
                row += f" {cell:>12}"
        print(row)

    print("-" * len(header))

    # Means
    row = f"{'Mean':<10}"
    for ch_name in CHANNEL_CONFIGS:
        for p_name in PIPELINES:
            valid = [a for a in results[ch_name][p_name] if not math.isnan(a)]
            mean_acc = float(np.mean(valid)) if valid else float("nan")
            cell = f"{mean_acc*100:.1f}%" if not math.isnan(mean_acc) else "NaN"
            row += f" {cell:>12}"
    print(row)

    print()
    print(f"Chance level: {CHANCE*100:.0f}%")

    # ---------------------------------------------------------------------------
    # Cross-subject transfer learning
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-SUBJECT TRANSFER LEARNING (Riemannian Recentering)")
    print("=" * 80)

    # Use 8-channel config for transfer learning (good balance of info vs speed)
    transfer_ch = CHANNEL_CONFIGS["8ch"]
    n_ch = len(transfer_ch)
    print(f"Channel config: 8ch ({n_ch} channels)")
    print(f"Method: Recenter covariance matrices to identity, then MDM")
    print()

    # Compute per-subject covariance matrices
    cov_est = CovarianceEstimator(channels=n_ch)
    subject_covs = {}
    subject_labels = {}

    for sid in subject_ids:
        if sid not in subject_data:
            continue
        X_full, y = subject_data[sid]
        X_sel = X_full[:, :, transfer_ch]
        covs = cov_est.transform(X_sel)
        subject_covs[sid] = covs
        subject_labels[sid] = y

    available_sids = sorted(subject_covs.keys())
    if len(available_sids) < 3:
        print("Not enough subjects for transfer learning experiment.")
        return

    # Leave-one-subject-out cross-validation with recentering
    print(f"{'Test Subj':<12} {'Within-CV':<12} {'Transfer':<12} {'Diff':>8}")
    print("-" * 48)

    within_accs = []
    transfer_accs = []

    for test_sid in available_sids:
        train_sids = [s for s in available_sids if s != test_sid]

        # Collect train data
        train_covs = np.concatenate([subject_covs[s] for s in train_sids])
        train_labels = np.concatenate([subject_labels[s] for s in train_sids]).astype(np.int64)

        test_covs = subject_covs[test_sid]
        test_labels = subject_labels[test_sid].astype(np.int64)

        # -- Within-subject CV (baseline) --
        mdm_within = MdmWrapper(channels=n_ch)
        within_acc = run_cv(
            SklearnPipeline([("cov", CovarianceEstimator(channels=n_ch)), ("mdm", MdmWrapper(channels=n_ch))]),
            subject_data[test_sid][0][:, :, transfer_ch],
            subject_labels[test_sid],
        )

        # -- Cross-subject transfer with recentering --
        # Recenter train data: compute per-source-subject mean, recenter each
        recentered_train_parts = []
        recentered_train_labels = []
        for s in train_sids:
            s_covs = subject_covs[s]
            s_mean = zbci.frechet_mean(s_covs)
            s_recentered = zbci.recenter(s_covs, s_mean)
            recentered_train_parts.append(s_recentered)
            recentered_train_labels.append(subject_labels[s])

        recentered_train = np.concatenate(recentered_train_parts)
        recentered_train_y = np.concatenate(recentered_train_labels).astype(np.int64)

        # Recenter test data
        test_mean = zbci.frechet_mean(test_covs)
        recentered_test = zbci.recenter(test_covs, test_mean)

        # Train MDM on recentered train, predict on recentered test
        mdm_transfer = zbci.MdmClassifier(channels=n_ch)
        mdm_transfer.fit(recentered_train, recentered_train_y)
        transfer_acc = mdm_transfer.score(recentered_test, test_labels)

        within_accs.append(within_acc)
        transfer_accs.append(transfer_acc)

        def fmt(v):
            return f"{v*100:.1f}%" if not math.isnan(v) else "NaN"

        diff = transfer_acc - within_acc if not math.isnan(within_acc) else float("nan")
        diff_str = f"{diff*100:+.1f}pp" if not math.isnan(diff) else "NaN"
        print(f"S{test_sid:03d}        {fmt(within_acc):<12} {fmt(transfer_acc):<12} {diff_str:>8}")
        sys.stdout.flush()

    print("-" * 48)
    mean_within = float(np.nanmean(within_accs))
    mean_transfer = float(np.nanmean(transfer_accs))
    diff = mean_transfer - mean_within
    print(f"{'Mean':<12} {mean_within*100:.1f}%{'':>5} {mean_transfer*100:.1f}%{'':>5} {diff*100:+.1f}pp")

    print()
    print("Transfer learning uses Riemannian recentering (M^{-1/2} X M^{-1/2})")
    print("to align covariance geometry across subjects before MDM classification.")


if __name__ == "__main__":
    main()
