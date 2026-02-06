# Zerostone

Zero-allocation signal processing primitives for real-time neural data acquisition. Designed for invasive extracellular electrophysiological recordings with deterministic latency and embedded deployment (`no_std`).

**Core primitives:** Lock-free ring buffers, cascaded IIR/FIR filters, streaming covariance estimation, spatial filters (CAR, Surface Laplacian), adaptive filtering, FFT/wavelet transforms, spike detection.

**Target systems:** Microelectrode arrays, Neuropixels probes, ECoG, stereo-EEG, calcium imaging pipelines.

---

## Quick Start

The fastest way to see what Zerostone can do is to run the interactive **Explore Tool**:

```bash
# Clone the repository
git clone https://github.com/fredrikwhaug/zerostone.git
cd zerostone

# Run a built-in recipe
cargo run --example explore -- --recipe lowpass
```

This will generate a visualization showing a 10 Hz signal with 60 Hz interference being cleaned by a lowpass filter. Check `output/explore_lowpass.png` to see the results!

**Try other recipes:**
```bash
cargo run --example explore -- --recipe highpass   # Remove low-frequency drift
cargo run --example explore -- --recipe bandpass   # Isolate a frequency band
cargo run --example explore -- --recipe spatial    # Multi-channel spatial filtering
cargo run --example explore -- --recipe pipeline   # Multi-stage processing chain
```

---

## Explore Tool

The **Explore Tool** lets you experiment with different signal processing configurations without writing any code. It uses TOML files to configure signals and filter chains, making it perfect for:

- Learning how different filters work
- Prototyping signal processing pipelines
- Understanding multi-channel spatial filters
- Visualizing the effects of filter parameters

### Usage

```bash
# Use the default configuration (explore.toml)
cargo run --example explore

# Use a built-in recipe
cargo run --example explore -- --recipe RECIPE_NAME

# Use a custom TOML file
cargo run --example explore -- --config my_config.toml
```

### Built-in Recipes

#### `lowpass` - High-Frequency Noise Removal
**Use case:** Removing powerline interference and high-frequency noise from neural signals.

**Signal:**
- 10 Hz sine wave (1.0 amplitude) - the desired signal
- 60 Hz sine wave (0.3 amplitude) - powerline interference
- White noise (0.2 amplitude)

**Filter:** 4th-order Butterworth lowpass at 30 Hz

**Result:** Clean 10 Hz signal with 60 Hz interference removed.

---

#### `highpass` - Low-Frequency Drift Removal
**Use case:** Removing baseline drift and DC offsets from recordings.

**Signal:**
- 5 Hz low-frequency drift (0.5 amplitude)
- 60 Hz sine wave (1.0 amplitude) - the desired signal
- White noise (0.2 amplitude)

**Filter:** 4th-order Butterworth highpass at 20 Hz

**Result:** Clean 60 Hz signal with low-frequency drift removed.

---

#### `bandpass` - Frequency Band Isolation
**Use case:** Isolating specific frequency bands (e.g., alpha, beta, gamma rhythms in EEG).

**Signal:**
- 5 Hz component (0.3 amplitude) - delta/theta
- 20 Hz component (1.0 amplitude) - beta (target band)
- 60 Hz component (0.3 amplitude) - gamma
- White noise (0.2 amplitude)

**Filter:** 4th-order Butterworth bandpass at 15-25 Hz

**Result:** Isolated 20 Hz component, rejecting both lower and higher frequencies.

---

#### `spatial` - Common-Mode Noise Rejection
**Use case:** Removing artifacts that affect all channels equally (e.g., reference electrode noise, powerline interference in multi-channel recordings).

**Signal:**
- 8 channels, each with alpha (10 Hz) and beta (20 Hz) rhythms
- 50 Hz powerline noise (0.5 amplitude) added to ALL channels

**Filter:** Common Average Reference (CAR)

**Result:** Powerline noise removed from all channels by subtracting the channel average. Individual channel variations preserved.

---

#### `pipeline` - Multi-Stage Processing
**Use case:** Realistic preprocessing pipeline for motor imagery BCI.

**Signal:**
- 8 channels with multi-band EEG-like signals
- Delta (3 Hz), theta (7 Hz), alpha (10 Hz), beta (20 Hz), gamma (40 Hz)
- 50 Hz powerline noise affecting all channels
- White noise

**Filters:**
1. Bandpass 8-30 Hz (isolate alpha/beta for motor imagery)
2. Common Average Reference (remove common-mode artifacts)

**Result:** Clean alpha/beta signals suitable for motor imagery classification.

---

### Custom Configurations

Create your own TOML file to experiment with different signals and filters:

```toml
# my_experiment.toml

[signal]
duration = 1.0          # Signal length in seconds
sample_rate = 1000.0    # Sampling rate in Hz
channels = 1            # Number of channels (1, 4, or 8)

[signal.components]
# Add sine wave components: [frequency_hz, amplitude]
sine_waves = [
    [10.0, 1.0],    # 10 Hz alpha rhythm
    [60.0, 0.3]     # 60 Hz powerline interference
]
noise_amplitude = 0.2   # White noise level
noise_seed = 42         # Random seed for reproducibility

# Optional: Add common-mode noise (for multi-channel signals)
# [signal.common_noise]
# frequency = 50.0
# amplitude = 0.5

# Define filter chain (applied sequentially)
[[filters]]
type = "lowpass"
cutoff = 30.0          # Cutoff frequency in Hz
order = 4              # Filter order (2, 4, 6, or 8)

# Add more filter stages as needed
# [[filters]]
# type = "highpass"
# cutoff = 5.0
# order = 2

[output]
plot_path = "output/my_experiment.png"
csv_path = "output/my_experiment.csv"
layout = "stacked"     # "stacked" or "grid"
```

Run your custom configuration:
```bash
cargo run --example explore -- --config my_experiment.toml
```

### Filter Types

The explore tool supports the following filters:

**Temporal Filters (single or multi-channel):**
- `lowpass` - Removes frequencies above cutoff
  - Parameters: `cutoff` (Hz), `order` (2, 4, 6, or 8)
- `highpass` - Removes frequencies below cutoff
  - Parameters: `cutoff` (Hz), `order` (2, 4, 6, or 8)
- `bandpass` - Keeps frequencies between low and high cutoff
  - Parameters: `low` (Hz), `high` (Hz), `order` (2, 4, 6, or 8)

**Spatial Filters (multi-channel only, requires 4 or 8 channels):**
- `car` - Common Average Reference
  - Subtracts the average of all channels from each channel
  - Removes common-mode noise and reference electrode bias
- `laplacian` - Surface Laplacian (Current Source Density)
  - Spatial high-pass filter using neighboring electrodes
  - Improves spatial resolution and reduces volume conduction

### Output

The explore tool generates two files:

1. **PNG plot** - Visualization showing all processing stages
   - Time-domain plots (first 500ms)
   - Multi-channel signals are overlaid with different colors
   - Stacked layout: good for 2-4 stages
   - Grid layout: good for 4+ stages

2. **CSV data** - Complete numerical data for all stages
   - Includes sample number, time in milliseconds, and all channel values
   - Separated by stage with descriptive headers
   - Easy to import into Python, MATLAB, R, etc. for further analysis

---

## Examples

Zerostone includes several examples demonstrating different features:

### Interactive Examples

- **`explore`** - Runtime-configurable signal processing with TOML files (recommended starting point)
  - 5 built-in recipes covering common use cases
  - Custom configuration support
  - Multi-stage pipeline visualization

### Specialized Demonstrations

- **`filter_demo`** - IIR filter demonstration with frequency/time domain plots
  - Lowpass, highpass, and bandpass filters
  - Configurable cutoff frequencies via CLI
  - FFT-based power spectrum visualization
  - Filter frequency response (Bode plots)
  - Magnitude and phase analysis

- **`spatial_demo`** - Multi-channel spatial filtering (CAR and Laplacian)
  - 8-channel signal with common-mode noise
  - Before/after comparison plots
  - Channel-by-channel visualization

- **`pipeline_demo`** - Multi-stage processing pipeline
  - Bandpass filtering → CAR → output
  - Demonstrates filter chaining
  - Intermediate stage inspection

- **`benchmark_viz`** - Performance benchmarking and visualization
  - Throughput measurements for all filter types
  - Scaling analysis (1, 8, 32, 128 channels)
  - PNG output with performance plots

Run any example with:
```bash
cargo run --example EXAMPLE_NAME -- [OPTIONS]
```

---

## Contributing

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/zerostone.git
   cd zerostone
   ```

2. **Create a feature branch** with the `feature/` prefix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and test locally.

4. **Run CI checks** before committing:
   ```bash
   ./ci-local.sh
   ```
   This runs the exact same checks as GitHub CI/CD:
   - Code formatting (`cargo fmt`)
   - Linting (`cargo clippy`)
   - Documentation (`cargo doc`)
   - Compilation (`cargo check`)
   - Tests (`cargo test`)

5. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** from your fork's feature branch to the main repository.

---

## Building

```bash
cargo build --release
cargo test
cargo bench
```

## License

AGPL-3.0 (see `LICENSE` file)
