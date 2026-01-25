#!/bin/bash
# Run the same checks as CI/CD locally before pushing

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

echo "All CI/CD checks passed locally!"
