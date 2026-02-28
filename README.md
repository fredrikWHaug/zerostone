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

# SSVEP frequency detection
best_freq, correlation = zbci.ssvep_detect(eeg_data, sample_rate=250.0,
                                            target_frequencies=[8.0, 10.0, 12.0, 15.0])

# Riemannian MDM classifier (sklearn-compatible)
mdm = zbci.MdmClassifier(channels=16)
mdm.fit(covariance_matrices, labels)
predictions = mdm.predict(test_covariances)
```

Supports all three major BCI paradigms: motor imagery (CSP), SSVEP (CCA), and P300/ERP (xDAWN). See [`python/README.md`](python/README.md) for the full feature list.

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

AGPL-3.0 (see `LICENSE` file)
