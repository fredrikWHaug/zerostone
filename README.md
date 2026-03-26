# Zerostone

Zero-allocation signal processing primitives for real-time neural data acquisition. Designed for invasive extracellular electrophysiological recordings with deterministic latency and embedded deployment.

---

## Python Package

```bash
pip install zpybci
```

```python
import zpybci as zbci

# Bandpass filter for alpha band
bpf = zbci.IirFilter.butterworth_bandpass(sample_rate=256.0, low_cutoff=8.0, high_cutoff=12.0)
filtered = bpf.process(signal)

# ICA artifact removal
ica = zbci.Ica(channels=16, contrast="logcosh")
ica.fit(eeg_data)
cleaned = ica.remove_components(eeg_data, exclude=[0])

# Kalman filter for decoder smoothing
kf = zbci.KalmanFilter(state_dim=4, obs_dim=2, F=F, H=H, Q=Q, R=R)
kf.predict()
innovation = kf.update(observation)

# Load EDF files without MNE
rec = zbci.read_edf("recording.edf")
data = rec.get_all_channels()

# Riemannian MDM classifier (sklearn-compatible)
mdm = zbci.MdmClassifier(channels=16)
mdm.fit(covariance_matrices, labels)
predictions = mdm.predict(test_covariances)
```

Supports all three major BCI paradigms: motor imagery (CSP), SSVEP (CCA), and P300/ERP (xDAWN). Includes sklearn-compatible wrappers, spike sorting with template subtraction, XDF/EDF file readers, phase-amplitude coupling, entropy measures, ERSP, and Granger causality. See [`python/README.md`](python/README.md) for the full feature list.

### Spike Sorting Accuracy

Synthetic benchmarks with known ground truth (seed=42, 60s recordings, 30 kHz):

| Preset | Ch | Units | Accuracy | Precision | Recall |
|--------|----|-------|----------|-----------|--------|
| easy   | 32 | 5     | **95.9%** | 98.7%    | 97.2%  |
| medium | 32 | 10    | **75.9%** | 91.5%    | 81.7%  |
| hard   | 64 | 20    | **50.4%** | 82.8%    | 56.3%  |

Pipeline: noise estimation, spatial whitening, threshold detection (amplitude/NEO/SNEO), deduplication, peak alignment, PCA, online k-means, cluster merge/split, template subtraction with per-spike amplitude scaling, NCC residual detection, matched filter second-pass detection (Neyman-Pearson optimal), CCG-based cluster merge, SNR auto-curation. Streaming segment API with persistent template library. See [`benchmarks/benchmark_results.md`](benchmarks/benchmark_results.md) for analysis.

### BCI Competition IV 2a Results

Validated on the standard 4-class motor imagery benchmark (9 subjects, session-to-session transfer):

| Pipeline   | Mean Accuracy | Published Baseline |
|------------|---------------|--------------------|
| TS+LDA     | **64.4%**     | ~60-68%            |
| MDM        | 59.0%         | ~55-62%            |
| xDAWN+MDM  | 57.4%         | ~55-60%            |
| CSP+LDA    | 40.5%         | ~40-50%            |

TS+LDA exceeds the original FBCSP competition winner (~63%). All pipelines built entirely with zpybci primitives.

---

## Rust Core

The Rust library is `no_std` compatible with zero heap allocation, suitable for embedded deployment on ARM Cortex-M and similar targets.

### Explore Tool

```bash
cargo run --example explore -- --recipe lowpass
```

This generates a visualization showing a 10 Hz signal with 60 Hz interference being cleaned by a lowpass filter. Check `output/explore_lowpass.png` to see the results.

**Other recipes:**
```bash
cargo run --example explore -- --recipe highpass   # Remove low-frequency drift
cargo run --example explore -- --recipe bandpass   # Isolate a frequency band
cargo run --example explore -- --recipe spatial    # Multi-channel spatial filtering
cargo run --example explore -- --recipe pipeline   # Multi-stage processing chain
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

GPL-3.0 (see `LICENSE` file)
