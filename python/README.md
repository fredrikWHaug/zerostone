# npyci - NumPy for BCI

High-performance signal processing for BCI research, powered by Rust.

## Installation

```bash
pip install npyci
```

## Quick Start

```python
import numpy as np
import npyci as npy

# Create a lowpass filter
lpf = npy.IirFilter.butterworth_lowpass(sample_rate=1000.0, cutoff=30.0)

# Generate and filter a signal
signal = np.random.randn(1000).astype(np.float32)
filtered = lpf.process(signal)
```

## Features

- Zero-allocation signal processing
- Real-time performance
- Butterworth IIR filters (lowpass, highpass, bandpass)
- More primitives coming soon!

## License

AGPL-3.0
