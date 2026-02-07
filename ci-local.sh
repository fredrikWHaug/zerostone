#!/bin/bash
# Run the same checks as CI/CD locally before pushing
#
# System dependencies (for plotters visualization in examples):
#   Linux: sudo apt-get install libfontconfig1-dev
#   macOS: (included by default)

set -e  # Exit on first error

echo "Running formatting check..."
cargo fmt --check

echo "Running clippy on all targets..."
cargo clippy --all-targets -- -D warnings

echo "Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

echo "Running cargo check..."
cargo check

echo "Checking MSRV (1.70.0) compatibility..."
# This matches the GitHub CI MSRV check - ensures code compiles on older Rust versions
if ! rustup toolchain list | grep -q "1.70.0"; then
    echo "Installing Rust 1.70.0 toolchain for MSRV check..."
    rustup toolchain install 1.70.0
fi
# Temporarily remove Cargo.lock to avoid lock file version conflicts with older Rust
mv Cargo.lock Cargo.lock.backup 2>/dev/null || true
cargo +1.70.0 check
mv Cargo.lock.backup Cargo.lock 2>/dev/null || true

echo "Running all tests (lib, integration, doc)..."
cargo test --all-targets
cargo test --doc

echo "Building Python bindings..."
cd python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release
cd ..

echo "Running Python tests..."
# Detect Python version that maturin used
WHEEL=$(ls python/target/wheels/npyci-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "Warning: No Python wheel found, skipping Python tests"
else
    # Extract Python version from wheel name (e.g., cp313 -> python3.13)
    PYVER=$(echo "$WHEEL" | grep -o 'cp[0-9]*' | head -1 | sed 's/cp//' | sed 's/\(.\)\(.\)/\1.\2/')
    PYTHON_CMD="python${PYVER}"

    # Check if this Python version has pytest
    if ! $PYTHON_CMD -m pytest --version &> /dev/null; then
        echo "Warning: pytest not found for $PYTHON_CMD, skipping Python tests"
        echo "Install with: $PYTHON_CMD -m pip install pytest"
    else
        # Install the wheel and run tests
        $PYTHON_CMD -m pip install "$WHEEL" --force-reinstall --break-system-packages --quiet
        $PYTHON_CMD -m pytest python/tests/ -v
        echo "Python tests passed!"
    fi
fi

echo "All CI/CD checks passed locally!"
