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

# Chain multiple stages into a pipeline
pipe = zbci.Pipeline(sample_rate=256.0, stages=[
    ("highpass", {"cutoff": 1.0}),
    ("notch", {"freq": 50.0}),
    ("lowpass", {"cutoff": 40.0}),
])
cleaned = pipe.process(signal)
```

## Features

### Filters
- **IIR** -- 4th-order Butterworth (lowpass, highpass, bandpass)
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

### Statistics
- **Online mean/variance** -- Welford's algorithm, no buffer required
- **Online covariance** -- streaming covariance matrix
- **Connectivity** -- coherence and phase locking value (PLV)

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

0.3.0

## License

AGPL-3.0
