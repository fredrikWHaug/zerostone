# Motor Imagery Classification with zpybci

This example demonstrates a complete motor imagery classification pipeline using zpybci on real EEG data. It shows how to preprocess multi-channel EEG, extract features, and build sklearn-compatible classifiers for brain-computer interface research.

## Overview

Motor imagery classification is a fundamental BCI paradigm where users imagine moving different body parts (e.g., left hand vs. right hand) and the system decodes their intention from EEG signals. This is used in rehabilitation, assistive devices, and neuroscience research.

We use the **Physionet EEGMMIDB dataset** (109 subjects, 64 channels, 160 Hz) and show how to:
1. Load and epoch real EEG data with minimal disk footprint
2. Preprocess signals with bandpass filtering, notch filtering, and spatial referencing
3. Extract features using Common Spatial Patterns (CSP) and Riemannian geometry
4. Build sklearn pipelines for cross-validation and classification
5. Benchmark performance on binary left/right motor imagery

## Prerequisites

```bash
pip install zpybci scikit-learn wfdb pyedflib numpy
```

## Dataset

**Physionet EEGMMIDB** — EEG Motor Movement/Imagery Dataset
- 109 subjects performing motor imagery tasks
- 64-channel EEG at 160 Hz sampling rate
- Binary classification: left fist (T1) vs. right fist (T2)
- Trials extracted from runs 4, 8, 12 (imagined movements)
- 4-second epochs (640 samples @ 160 Hz)

The dataset is streamed on-demand via `wfdb` — files are downloaded to a temporary directory, processed, and immediately discarded. No persistent storage required.

## Quick Start

### 1. Load Data

```python
from load_data import stream_subject

# Download and epoch all motor imagery trials for one subject
trials = stream_subject(subject_id=1)

# trials is a list of (epoch, label) tuples
# epoch: (640, 64) float32 — 4 seconds of 64-channel EEG
# label: int — 0 = left fist, 1 = right fist

print(f"Loaded {len(trials)} trials")
```

### 2. Preprocess Signals

```python
import zpybci as zbci
import numpy as np

# Create preprocessing filters
bandpass_filters = [
    zbci.IirFilter.butterworth_bandpass(160.0, 8.0, 30.0)
    for _ in range(64)
]
notch = zbci.NotchFilter.powerline_60hz(160.0, channels=64)
car = zbci.CAR(channels=64)

# Process one trial
epoch, label = trials[0]
epoch = np.asarray(epoch, dtype=np.float32)

# Apply bandpass (8-30 Hz: mu + beta bands)
bandpassed = np.empty_like(epoch)
for ch in range(64):
    col = np.ascontiguousarray(epoch[:, ch])
    bandpassed[:, ch] = bandpass_filters[ch].process(col)

# Apply notch (remove 60 Hz powerline noise)
denoised = notch.process(bandpassed)

# Apply common average reference
referenced = car.process(denoised)
```

### 3. Feature Extraction

We demonstrate two feature extraction approaches:

#### Option A: Common Spatial Patterns (CSP)

CSP learns spatial filters that maximize the variance difference between two classes.

```python
from sklearn_compat import CSPTransformer

# X: (n_trials, n_samples, n_channels)
# y: (n_trials,) binary labels {0, 1}

csp = CSPTransformer(channels=4, filters=2)
csp.fit(X_train, y_train)
features = csp.transform(X_test)  # (n_trials, 2) log-variance features
```

#### Option B: Riemannian Geometry (Covariance + Tangent Space)

Riemannian methods operate on the manifold of symmetric positive-definite covariance matrices.

```python
from sklearn_compat import CovarianceEstimator, TangentSpaceTransformer

# Estimate covariance matrices for each trial
cov = CovarianceEstimator(channels=4)
covariances = cov.transform(X)  # (n_trials, 4, 4)

# Project to tangent space
ts = TangentSpaceTransformer(channels=4)
ts.fit(covariances)
features = ts.transform(covariances)  # (n_trials, 10) tangent vectors
```

### 4. Classification with sklearn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# Pipeline 1: CSP + LDA
pipe_csp = Pipeline([
    ('csp', CSPTransformer(channels=4, filters=2)),
    ('lda', LinearDiscriminantAnalysis()),
])

# Pipeline 2: Riemannian + LDA
pipe_riem = Pipeline([
    ('cov', CovarianceEstimator(channels=4)),
    ('ts', TangentSpaceTransformer(channels=4)),
    ('lda', LinearDiscriminantAnalysis()),
])

# 5-fold cross-validation
scores_csp = cross_val_score(pipe_csp, X, y, cv=5, scoring='accuracy')
scores_riem = cross_val_score(pipe_riem, X, y, cv=5, scoring='accuracy')

print(f"CSP+LDA accuracy: {scores_csp.mean():.1%}")
print(f"Riemannian+LDA accuracy: {scores_riem.mean():.1%}")
```

## Benchmark Results

The `benchmark.py` script runs the full pipeline on 9 subjects from the Physionet EEGMMIDB dataset. We preprocess all 64 channels (bandpass 8-30 Hz, notch 60 Hz, CAR), then select 4 motor cortex channels (C5, C3, C1, Cz) for classification.

**Preprocessing**: Bandpass (8-30 Hz) → Notch (60 Hz) → CAR → Select motor channels

**Results** (9 subjects, 5-fold CV, binary L/R imagery):

```
Subject    Trials   CSP+LDA       Riem+LDA      vs Chance
-----------------------------------------------------------
S001       45       68.9%         66.7%         **
S002       45       62.2%         77.8%         **
S003       45       33.3%         26.7%
S004       45       55.6%         68.9%         **
S005       45       46.7%         46.7%
S006       45       57.8%         33.3%         **
S007       45       64.4%         86.7%         **
S008       45       64.4%         53.3%         **
S009       45       60.0%         46.7%         **
-----------------------------------------------------------
Mean                57.0%         56.3%
```

- **CSP above chance**: 7/9 subjects
- **Riem above chance**: 5/9 subjects
- **Chance level**: 50%
- **Literature baselines** (binary L/R motor imagery): CSP+LDA ~75-85%, Riemannian ~80-90%

The gap vs. literature is expected:
- Only 4 of 64 channels used (eigensolver convergence constraint)
- Small trial count (45 trials per subject after preprocessing)
- No per-subject hyperparameter tuning
- Simple preprocessing (no artifact rejection, no ICA)

This demonstrates that zpybci produces reasonable results on real neural data and integrates seamlessly with the sklearn ecosystem.

## Files in This Example

- **load_data.py** — Download and epoch Physionet EEGMMIDB data via wfdb
- **sklearn_compat.py** — Sklearn-compatible transformer wrappers for zpybci classes
- **validate_pipeline.py** — Sanity-check preprocessing on real EEG
- **benchmark.py** — Full classification benchmark on 9 subjects
- **README.md** — This tutorial

## Sklearn-Compatible Transformers

The `sklearn_compat.py` module provides four transformers:

### CSPTransformer
Wraps `zbci.AdaptiveCsp` for Common Spatial Patterns feature extraction.
- **Input**: `(n_trials, n_samples, n_channels)` EEG trials
- **Output**: `(n_trials, n_filters)` log-variance features
- **Parameters**: `channels`, `filters`, `min_samples`, `regularization`

### TangentSpaceTransformer
Wraps `zbci.TangentSpace` for Riemannian tangent space projection.
- **Input**: `(n_trials, n_channels, n_channels)` covariance matrices
- **Output**: `(n_trials, n_features)` tangent vectors
- **Parameters**: `channels`, `metric` (default: "riemann")

### CovarianceEstimator
Wraps `zbci.OnlineCov` for trial-wise covariance matrix estimation.
- **Input**: `(n_trials, n_samples, n_channels)` EEG trials
- **Output**: `(n_trials, n_channels, n_channels)` covariance matrices
- **Parameters**: `channels`

### BandPowerTransformer
Wraps `zbci.MultiBandPower` for multi-band power feature extraction.
- **Input**: `(n_trials, n_samples, n_channels)` EEG trials
- **Output**: `(n_trials, n_bands * n_channels)` power features
- **Parameters**: `sample_rate`, `channels`, `fft_size`, `bands`

All transformers implement sklearn's `BaseEstimator` and `TransformerMixin`, making them compatible with `Pipeline`, `GridSearchCV`, `cross_val_score`, and other sklearn utilities.

## Channel Count Constraints

Due to enum dispatch in the Rust core, transformers have specific channel count requirements:

- `CSPTransformer`: 4, 8, 16, 32, or 64 channels
- `TangentSpaceTransformer`: 4, 8, 16, or 32 channels
- `CovarianceEstimator`: 4, 8, 16, 32, or 64 channels
- `BandPowerTransformer`: 1, 4, 8, 16, 32, or 64 channels

If your data has a different channel count, use channel selection or zero-padding to match a supported size.

## Running the Benchmark

To reproduce the results table above:

```bash
cd python/examples/motor_imagery
python benchmark.py
```

Expected runtime: 5-10 minutes (parallel download, sequential processing).

## What's Next

This example demonstrates offline analysis. For real-time BCI applications, consider:

1. **Real-time streaming** — Use Lab Streaming Layer (LSL) to acquire live EEG
2. **Other paradigms** — P300, SSVEP, error-related potentials
3. **Advanced preprocessing** — ICA for artifact removal, Laplacian spatial filters
4. **Deep learning** — Combine zpybci preprocessing with PyTorch/TensorFlow models
5. **Adaptive pipelines** — Update CSP filters online as new data arrives

zpybci provides the low-level primitives for all of these use cases. The sklearn-compatible interface lets you focus on research questions rather than implementation details.

## References

- **Physionet EEGMMIDB**: Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220.
- **Common Spatial Patterns**: Ramoser et al. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE Trans. Rehabil. Eng.
- **Riemannian Geometry for BCI**: Barachant et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans. Biomed. Eng.

## License

zpybci is licensed under AGPL-3.0. See the root repository for details.
