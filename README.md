# Zerostone

Zero-allocation signal processing primitives for real-time neural data acquisition. Designed for invasive extracellular electrophysiological recordings with deterministic latency and embedded deployment.

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
