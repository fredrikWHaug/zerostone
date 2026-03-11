#!/bin/bash
# Line count for zerostone

rust_core=$(find src -name '*.rs' | xargs wc -l | tail -1 | awk '{print $1}')
python_bind=$(find python/src -name '*.rs' | xargs wc -l | tail -1 | awk '{print $1}')
total=$((rust_core + python_bind))

printf "Rust core (src/):          %5d\n" "$rust_core"
printf "Python bindings (python/src/): %5d\n" "$python_bind"
printf "Total:                     %5d\n" "$total"
