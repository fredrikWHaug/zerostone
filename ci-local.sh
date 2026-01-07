#!/bin/bash
# Run the same checks as CI/CD locally before pushing

set -e  # Exit on first error

echo "Running formatting check..."
cargo fmt --check

echo "Running clippy on all targets..."
cargo clippy --all-targets -- -D warnings

echo "Building documentation..."
cargo doc --no-deps

echo "Running cargo check..."
cargo check

echo "Running all tests..."
cargo test --all-targets

echo "All CI/CD checks passed locally!"
