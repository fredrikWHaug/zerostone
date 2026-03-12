#!/bin/bash
# Line count for zerostone
# Tracks non-test code lines against a 25K ceiling (tinygrad-inspired discipline).
# Test code (#[cfg(test)] blocks), blanks, and comments are excluded from the budget.

set -e

CEILING=25000
WARN_THRESHOLD=700

# Count non-test code lines in a Rust file (excludes #[cfg(test)] blocks, blanks, comments)
count_code() {
    python3 -c "
import sys
in_test = False
depth = 0
code = 0
for line in open(sys.argv[1]):
    stripped = line.strip()
    if '#[cfg(test)]' in stripped:
        in_test = True
        depth = 0
        continue
    if in_test:
        depth += line.count('{') - line.count('}')
        if depth <= 0 and '}' in line:
            in_test = False
        continue
    if not stripped or stripped.startswith('//'):
        continue
    code += 1
print(code)
" "$1"
}

# Per-module breakdown
if [ "$1" = "--detail" ]; then
    printf "%6s %6s  %s\n" "Code" "Total" "Module"
    printf "%s\n" "$(printf '%0.s-' {1..50})"
fi

rust_code=0
rust_total=0
over_threshold=""

for f in $(find src -name '*.rs' | sort); do
    total=$(wc -l < "$f")
    code=$(count_code "$f")
    rust_code=$((rust_code + code))
    rust_total=$((rust_total + total))
    if [ "$1" = "--detail" ]; then
        printf "%6d %6d  %s\n" "$code" "$total" "$f"
    fi
    if [ "$code" -gt "$WARN_THRESHOLD" ]; then
        over_threshold="$over_threshold  $code  $f\n"
    fi
done

py_code=0
py_total=0
for f in $(find python/src -name '*.rs' | sort); do
    total=$(wc -l < "$f")
    code=$(count_code "$f")
    py_code=$((py_code + code))
    py_total=$((py_total + total))
    if [ "$1" = "--detail" ]; then
        printf "%6d %6d  %s\n" "$code" "$total" "$f"
    fi
done

printf "\nRust core code:    %5d / %d ceiling\n" "$rust_code" "$CEILING"
printf "Python bindings:   %5d\n" "$py_code"
printf "Headroom:          %5d lines\n" "$((CEILING - rust_code))"

if [ -n "$over_threshold" ]; then
    printf "\nModules over %d code lines:\n" "$WARN_THRESHOLD"
    printf "$over_threshold"
fi

if [ "$rust_code" -gt "$CEILING" ]; then
    printf "\nERROR: Rust core (%d) exceeds %d line ceiling!\n" "$rust_code" "$CEILING"
    exit 1
fi
