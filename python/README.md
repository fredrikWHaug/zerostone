# zpybci -- Zerostone Python for BCI

High-performance, real-time signal processing for brain-computer interface research. Powered by Rust with zero-copy NumPy integration.

## Installation

```bash
pip install zpybci
```

Wheels are available for Linux (x86_64, aarch64), macOS (Intel + Apple Silicon), and Windows (x86_64). Python 3.8+.

## Quick Start

```python
import numpy as np
import zpybci as zbci

# Bandpass filter for alpha band (8-12 Hz)
bpf = zbci.IirFilter.butterworth_bandpass(sample_rate=256.0, low_cutoff=8.0, high_cutoff=12.0)
signal = np.random.randn(1000).astype(np.float32)
filtered = bpf.process(signal)

# ICA for artifact removal
ica = zbci.Ica(channels=16, contrast="logcosh")
ica.fit(eeg_data, max_iter=200)
cleaned = ica.remove_components(eeg_data, exclude=[0, 2])  # remove blink/muscle artifacts

# Load an EDF file -- no MNE dependency needed
rec = zbci.read_edf("recording.edf")
ch1 = rec.get_channel("Fp1")
all_data = rec.get_all_channels()  # (n_channels, n_samples) numpy array
```

## Validated Results

Tested on the BCI Competition IV 2a motor imagery benchmark (9 subjects, 4-class, session-to-session transfer):

| Pipeline   | Mean Accuracy | Published Baseline |
|------------|---------------|--------------------|
| CSP+LDA    | 40.5%         | ~40-50%            |
| TS+LDA     | **64.4%**     | ~60-68%            |
| MDM        | 59.0%         | ~55-62%            |
| xDAWN+MDM  | 57.4%         | ~55-60%            |

TS+LDA exceeds the original competition winner (FBCSP, ~63%). All pipelines built entirely with zpybci primitives.

## Features

### Filters
- **IIR** -- Butterworth lowpass, highpass, bandpass (order 2/4/6/8, proper pole placement matching scipy)
- **FIR** -- arbitrary-length finite impulse response
- **AC coupling** -- DC removal for streaming data
- **Median** -- nonlinear smoothing
- **Adaptive** -- LMS and NLMS for noise cancellation
- **Notch** -- narrowband rejection (e.g., 50/60 Hz line noise)

### Spatial Filters
- **CAR** -- common average reference
- **Surface Laplacian** -- current source density approximation
- **Channel Router** -- flexible channel remapping
- **xDAWN** -- supervised spatial filters maximizing ERP signal-to-noise ratio

### Spectral Analysis
- **FFT** -- fast Fourier transform (magnitude/phase)
- **STFT** -- short-time Fourier transform
- **Multi-band power** -- concurrent power in multiple frequency bands
- **Welch PSD** -- power spectral density estimation
- **CWT** -- continuous wavelet transform (Morlet)

### Independent Component Analysis
- **FastICA** -- symmetric parallel extraction with LogCosh, Exp, and Cube contrast functions
- **Artifact removal** -- remove blink/muscle/cardiac components and reconstruct clean signal
- **Channel counts** -- 4, 8, 16, 32, or 64 channels

### Kalman Filter
- **State estimation** -- predict/update cycle for real-time decoder smoothing
- **Joseph form** -- numerically stable covariance update (preserves positive definiteness)
- **Flexible dimensions** -- 8 state/observation combinations from (2,1) to (8,8)

### Linear Discriminant Analysis
- **Fisher's LDA** -- binary classification with shrinkage regularization
- **Calibrated probabilities** -- `predict_proba()` via sigmoid scaling
- **Feature dimensions** -- 2, 4, 6, 8, 12, 16, 32, or 64

### Spike Sorting
- **Full pipeline** -- `sort_multichannel()` for end-to-end multi-channel spike sorting (noise estimation, spatial whitening, detection, deduplication, peak alignment, PCA, online k-means, cluster merge/split, template subtraction, NCC residual detection, SNR auto-curation)
- **Streaming API** -- `StreamingSorter` for segment-based processing with persistent template library, exponential moving average template updates, and online rigid drift estimation
- **Batch parallel** -- `sort_batch_parallel()` for rayon-based parallel sorting of long recordings
- **Detection modes** -- amplitude threshold, NEO (Nonlinear Energy Operator), SNEO (Smoothed NEO)
- **Probe geometry** -- `ProbeLayout` with linear, polytrode, tetrode, Neuropixels 1.0/2.0, Utah array presets
- **Template subtraction** -- multi-pass template subtraction with per-spike amplitude scaling and NCC residual detection
- **Cluster refinement** -- d-prime merge, spatial merge, CCG-based merge, ISI-violation split, amplitude bimodality split, GMM full-covariance EM
- **Quality metrics** -- per-cluster SNR, ISI violation rate, contamination rate, d-prime, silhouette score
- **SVD template initialization** -- power-iteration SVD for principled initial centroids (no random k-means init dependency)
- **Drift correction** -- rigid drift estimation per streaming segment via spatial cross-correlation of template positions
- **Wavelet denoising** -- Haar SWT with Donoho-Johnstone universal threshold for preprocessing
- **Supported channels** -- 4, 8, 16, 32, 64, 96, 128 (const-generic, no_std core)
- **Convenience pipeline** -- `spike_sort()` for simple single-channel sorting with online k-means

### EDF/EDF+ File Reader
- **read_edf()** -- load EDF/EDF+ files without MNE or pyedflib
- **Channel access** -- by index or label name
- **Physical units** -- automatic digital-to-physical conversion
- **Mixed sample rates** -- zero-padded multi-channel extraction

### XDF File Reader (Lab Streaming Layer)
- **read_xdf()** -- load XDF files from LSL recordings
- **Multi-stream** -- numeric and string streams with per-stream metadata
- **Clock offsets** -- collection time and offset value pairs
- **Sample formats** -- float32, float64, int8, int16, int32, int64, string

### Detection
- **Threshold** -- fixed-threshold event detection
- **Adaptive threshold** -- self-adjusting threshold based on signal statistics
- **Zero-crossing** -- rate estimation

### Artifact Handling
- **Amplitude-based** -- flag samples exceeding a threshold
- **Z-score** -- flag statistically outlying segments

### Analysis
- **Envelope follower** -- instantaneous amplitude via rectification + smoothing
- **Windowed RMS** -- streaming root-mean-square
- **Hilbert transform** -- analytic signal, instantaneous phase/frequency

### Phase-Amplitude Coupling
- **Modulation index** -- Tort et al. (2010), KL-divergence from uniform phase distribution
- **Mean vector length** -- Canolty et al. (2006), normalized complex coupling strength
- **Comodulogram** -- PAC across frequency pairs (bandpass + Hilbert + MI/MVL)
- **Phase-amplitude distribution** -- binned amplitude histogram for visualization

### Entropy Measures
- **Sample entropy** -- template-matching complexity measure (SampEn)
- **Approximate entropy** -- regularity statistic with self-matches (ApEn)
- **Spectral entropy** -- Shannon entropy of normalized PSD
- **Multiscale entropy** -- coarse-grained sample entropy across time scales

### Event-Related Spectral Perturbation (ERSP)
- **compute_ersp()** -- STFT-based time-frequency decomposition with epoch averaging
- **baseline_normalize()** -- dB, z-score, percentage, and log-ratio normalization modes
- **Single-trial mode** -- per-epoch time-frequency maps without averaging

### Statistics
- **Online mean/variance** -- Welford's algorithm, no buffer required
- **Online covariance** -- streaming covariance matrix
- **Connectivity** -- coherence, phase locking value (PLV), and Granger causality
- **Conditional Granger** -- Granger causality conditioned on confound variables

### BCI Paradigms
- **Motor imagery** -- CSP with online adaptation, sklearn-compatible transformer
- **SSVEP** -- CCA-based frequency detection, reference signal generation
- **P300/ERP** -- epoch averaging, xDAWN spatial filters

### Riemannian Geometry
- **Tangent space** -- SPD manifold projection for classification
- **MDM classifier** -- minimum distance to mean on SPD manifold
- **Frechet mean** -- geometric mean of SPD matrices
- **Riemannian distance** -- affine-invariant distance metric
- **Recentering** -- domain adaptation via Riemannian transport

### Sklearn Integration
- **CSPTransformer** -- common spatial patterns as sklearn transformer
- **TangentSpaceTransformer** -- Riemannian tangent space projection
- **BandPowerTransformer** -- multi-band power feature extraction
- **CovarianceEstimator** -- covariance matrix estimation
- **LdaClassifier** -- Fisher's LDA with predict_proba and score
- **IcaTransformer** -- ICA with component exclusion for artifact removal
- **MdmWrapper** -- minimum distance to mean classifier
- **XDawnWrapper** -- xDAWN spatial filter transformer
- Works with `make_pipeline()`, `cross_val_score()`, `GridSearchCV()`

### Advanced
- **OASIS deconvolution** -- calcium transient inference from fluorescence traces

### Utilities
- **Pipeline** -- declarative stage chaining with a single `process()` call
- **Resampling** -- integer decimation and interpolation
- **Streaming percentile** -- approximate quantiles on unbounded streams
- **Clock sync** -- offset estimation, linear drift correction, sample clock alignment
- **Cross-correlation** -- full, valid, and circular modes
- **Window functions** -- Hann, Hamming, Blackman, flat-top, Kaiser

## Version

0.8.0

## License

GPL-3.0
