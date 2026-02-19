#!/usr/bin/env python3
"""
Compare zerostone (zpybci) filters against scipy.signal.

This script generates visual comparisons showing:
1. Time-domain filter outputs (zerostone vs scipy overlay)
2. Frequency response comparisons
3. Numerical differences (error plots)

Usage:
    python compare_scipy.py [--output-dir OUTPUT_DIR]

Output:
    - PNG plots saved to output/scipy_validation/
    - Console summary of differences
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import zpybci as zbci
except ImportError:
    print("Error: zpybci not installed. Run 'maturin develop' in python/ directory first.")
    sys.exit(1)

try:
    from scipy import signal
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: scipy and matplotlib required. Install with: pip install scipy matplotlib")
    sys.exit(1)


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_lowpass_filter(output_dir: Path):
    """Compare lowpass filter outputs."""
    print("\n=== Lowpass Filter Comparison ===")

    sample_rate = 1000.0
    cutoff = 40.0
    duration = 1.0

    # Create test signal: 10 Hz (passband) + 100 Hz (stopband) + noise
    t = np.arange(0, duration, 1/sample_rate, dtype=np.float32)
    passband = np.sin(2 * np.pi * 10 * t)
    stopband = 0.5 * np.sin(2 * np.pi * 100 * t)
    noise = 0.1 * np.random.randn(len(t))
    test_signal = (passband + stopband + noise).astype(np.float32)

    # Create filters
    zs_filter = zbci.IirFilter.butterworth_lowpass(sample_rate, cutoff)
    sos = signal.butter(4, cutoff, btype='low', fs=sample_rate, output='sos')

    # Filter
    zs_output = zs_filter.process(test_signal)
    scipy_output = signal.sosfilt(sos, test_signal.astype(np.float64)).astype(np.float32)

    # Calculate difference
    diff = zs_output - scipy_output
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print(f"  Cutoff: {cutoff} Hz")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Plot 1: Original signal
    axes[0].plot(t[:500], test_signal[:500], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Original Signal (10 Hz + 100 Hz + noise)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered outputs overlay
    axes[1].plot(t[:500], zs_output[:500], 'b-', linewidth=1, label='zerostone', alpha=0.8)
    axes[1].plot(t[:500], scipy_output[:500], 'r--', linewidth=1, label='scipy', alpha=0.8)
    axes[1].set_title(f'Filtered Outputs (4th order Butterworth lowpass @ {cutoff} Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Difference
    axes[2].plot(t, diff, 'g-', linewidth=0.5)
    axes[2].set_title(f'Difference (zerostone - scipy), Max: {max_diff:.2e}, RMS: {rms_diff:.2e}')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Frequency response comparison
    freqs_zs = []
    freqs_scipy = []
    test_freqs = np.logspace(0.5, 2.5, 50)  # 3 Hz to 300 Hz

    w, h_scipy = signal.sosfreqz(sos, worN=2048, fs=sample_rate)

    for freq in test_freqs:
        # Measure zerostone response
        zs_filter.reset()
        t_test = np.arange(0, 2, 1/sample_rate, dtype=np.float32)
        test_sine = np.sin(2 * np.pi * freq * t_test).astype(np.float32)
        output = zs_filter.process(test_sine)
        rms_out = np.sqrt(np.mean(output[1000:] ** 2))
        rms_in = np.sqrt(np.mean(test_sine[1000:] ** 2))
        freqs_zs.append(rms_out / rms_in)

        # Get scipy response
        idx = np.argmin(np.abs(w - freq))
        freqs_scipy.append(np.abs(h_scipy[idx]))

    axes[3].semilogx(test_freqs, 20 * np.log10(np.array(freqs_zs) + 1e-10), 'b-', linewidth=2, label='zerostone')
    axes[3].semilogx(test_freqs, 20 * np.log10(np.array(freqs_scipy) + 1e-10), 'r--', linewidth=2, label='scipy')
    axes[3].axvline(x=cutoff, color='gray', linestyle=':', label=f'cutoff ({cutoff} Hz)')
    axes[3].set_title('Frequency Response Comparison')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude (dB)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-60, 5])

    plt.tight_layout()
    plt.savefig(output_dir / 'lowpass_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'lowpass_comparison.png'}")


def compare_highpass_filter(output_dir: Path):
    """Compare highpass filter outputs."""
    print("\n=== Highpass Filter Comparison ===")

    sample_rate = 1000.0
    cutoff = 10.0
    duration = 2.0

    # Create test signal: DC offset + 50 Hz signal
    t = np.arange(0, duration, 1/sample_rate, dtype=np.float32)
    dc_offset = 1.5
    ac_signal = np.sin(2 * np.pi * 50 * t)
    test_signal = (dc_offset + ac_signal).astype(np.float32)

    # Create filters
    zs_filter = zbci.IirFilter.butterworth_highpass(sample_rate, cutoff)
    sos = signal.butter(4, cutoff, btype='high', fs=sample_rate, output='sos')

    # Filter
    zs_output = zs_filter.process(test_signal)
    scipy_output = signal.sosfilt(sos, test_signal.astype(np.float64)).astype(np.float32)

    # Calculate difference
    diff = zs_output - scipy_output
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print(f"  Cutoff: {cutoff} Hz")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Original signal (zoom to show DC)
    axes[0].plot(t[:500], test_signal[:500], 'b-', linewidth=0.5)
    axes[0].axhline(y=dc_offset, color='r', linestyle='--', label=f'DC offset = {dc_offset}')
    axes[0].set_title('Original Signal (DC + 50 Hz sine)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered outputs
    axes[1].plot(t[:500], zs_output[:500], 'b-', linewidth=1, label='zerostone', alpha=0.8)
    axes[1].plot(t[:500], scipy_output[:500], 'r--', linewidth=1, label='scipy', alpha=0.8)
    axes[1].axhline(y=0, color='gray', linestyle=':')
    axes[1].set_title(f'Filtered Outputs (DC removed, 4th order Butterworth highpass @ {cutoff} Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Difference
    axes[2].plot(t, diff, 'g-', linewidth=0.5)
    axes[2].set_title(f'Difference (zerostone - scipy), Max: {max_diff:.2e}, RMS: {rms_diff:.2e}')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'highpass_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'highpass_comparison.png'}")


def compare_bandpass_filter(output_dir: Path):
    """Compare bandpass filter outputs."""
    print("\n=== Bandpass Filter Comparison ===")

    sample_rate = 1000.0
    low_cutoff = 8.0
    high_cutoff = 12.0
    duration = 3.0

    # Create test signal: multiple frequency components
    t = np.arange(0, duration, 1/sample_rate, dtype=np.float32)
    below_band = np.sin(2 * np.pi * 3 * t)  # 3 Hz (below passband)
    in_band = np.sin(2 * np.pi * 10 * t)     # 10 Hz (in passband, alpha rhythm)
    above_band = np.sin(2 * np.pi * 30 * t)  # 30 Hz (above passband)
    test_signal = (below_band + in_band + above_band).astype(np.float32)

    # Create filters
    zs_filter = zbci.IirFilter.butterworth_bandpass(sample_rate, low_cutoff, high_cutoff)
    sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', fs=sample_rate, output='sos')

    # Filter
    zs_output = zs_filter.process(test_signal)
    scipy_output = signal.sosfilt(sos, test_signal.astype(np.float64)).astype(np.float32)

    # Calculate difference
    diff = zs_output - scipy_output
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print(f"  Passband: {low_cutoff}-{high_cutoff} Hz")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Original signal
    axes[0].plot(t[:1000], test_signal[:1000], 'b-', linewidth=0.5)
    axes[0].set_title('Original Signal (3 Hz + 10 Hz + 30 Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered outputs (zoom to show 10 Hz)
    axes[1].plot(t[1000:1500], zs_output[1000:1500], 'b-', linewidth=1, label='zerostone', alpha=0.8)
    axes[1].plot(t[1000:1500], scipy_output[1000:1500], 'r--', linewidth=1, label='scipy', alpha=0.8)
    axes[1].plot(t[1000:1500], in_band[1000:1500], 'k:', linewidth=0.5, label='ideal (10 Hz only)', alpha=0.5)
    axes[1].set_title(f'Filtered Outputs ({low_cutoff}-{high_cutoff} Hz bandpass, alpha band)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Difference
    axes[2].plot(t, diff, 'g-', linewidth=0.5)
    axes[2].set_title(f'Difference (zerostone - scipy), Max: {max_diff:.2e}, RMS: {rms_diff:.2e}')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'bandpass_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'bandpass_comparison.png'}")


def compare_fir_filter(output_dir: Path):
    """Compare FIR filter outputs."""
    print("\n=== FIR Filter Comparison ===")

    # Test moving average
    window_size = 5

    # Create filters
    zs_filter = zbci.FirFilter.moving_average(window_size)
    scipy_coeffs = np.ones(window_size) / window_size

    # Test signal with noise
    np.random.seed(42)
    test_signal = np.random.randn(500).astype(np.float32)

    # Filter
    zs_output = zs_filter.process(test_signal)
    scipy_output = signal.lfilter(scipy_coeffs, 1.0, test_signal.astype(np.float64)).astype(np.float32)

    # Calculate difference
    diff = zs_output - scipy_output
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff ** 2))

    print(f"  Window size: {window_size}")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    t = np.arange(len(test_signal))

    # Plot 1: Original and filtered
    axes[0].plot(t[:200], test_signal[:200], 'b-', linewidth=0.5, alpha=0.5, label='original')
    axes[0].plot(t[:200], zs_output[:200], 'r-', linewidth=1, label='filtered')
    axes[0].set_title(f'Moving Average Filter (window={window_size})')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Comparison
    axes[1].plot(t[:200], zs_output[:200], 'b-', linewidth=1, label='zerostone', alpha=0.8)
    axes[1].plot(t[:200], scipy_output[:200], 'r--', linewidth=1, label='scipy', alpha=0.8)
    axes[1].set_title('Output Comparison (should be identical)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Difference (should be near zero)
    axes[2].plot(t, diff, 'g-', linewidth=0.5)
    axes[2].set_title(f'Difference (zerostone - scipy), Max: {max_diff:.2e}')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'fir_comparison.png'}")


def compare_frequency_responses(output_dir: Path):
    """Compare frequency responses of all filter types."""
    print("\n=== Frequency Response Comparison ===")

    sample_rate = 1000.0
    freqs = np.logspace(0, 2.7, 200)  # 1 Hz to 500 Hz

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Lowpass
    cutoff_lp = 50.0
    zs_lp = zbci.IirFilter.butterworth_lowpass(sample_rate, cutoff_lp)
    sos_lp = signal.butter(4, cutoff_lp, btype='low', fs=sample_rate, output='sos')
    w, h_scipy_lp = signal.sosfreqz(sos_lp, worN=2048, fs=sample_rate)

    zs_resp_lp = measure_frequency_response(zs_lp, freqs, sample_rate)
    scipy_resp_lp = np.interp(freqs, w, np.abs(h_scipy_lp))

    axes[0, 0].semilogx(freqs, 20 * np.log10(zs_resp_lp + 1e-10), 'b-', linewidth=2, label='zerostone')
    axes[0, 0].semilogx(freqs, 20 * np.log10(scipy_resp_lp + 1e-10), 'r--', linewidth=2, label='scipy')
    axes[0, 0].axvline(x=cutoff_lp, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_title(f'Lowpass (fc={cutoff_lp} Hz)')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-60, 5])

    # Highpass
    cutoff_hp = 20.0
    zs_hp = zbci.IirFilter.butterworth_highpass(sample_rate, cutoff_hp)
    sos_hp = signal.butter(4, cutoff_hp, btype='high', fs=sample_rate, output='sos')
    w, h_scipy_hp = signal.sosfreqz(sos_hp, worN=2048, fs=sample_rate)

    zs_resp_hp = measure_frequency_response(zs_hp, freqs, sample_rate)
    scipy_resp_hp = np.interp(freqs, w, np.abs(h_scipy_hp))

    axes[0, 1].semilogx(freqs, 20 * np.log10(zs_resp_hp + 1e-10), 'b-', linewidth=2, label='zerostone')
    axes[0, 1].semilogx(freqs, 20 * np.log10(scipy_resp_hp + 1e-10), 'r--', linewidth=2, label='scipy')
    axes[0, 1].axvline(x=cutoff_hp, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_title(f'Highpass (fc={cutoff_hp} Hz)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-60, 5])

    # Bandpass
    low_bp, high_bp = 20.0, 50.0
    zs_bp = zbci.IirFilter.butterworth_bandpass(sample_rate, low_bp, high_bp)
    sos_bp = signal.butter(4, [low_bp, high_bp], btype='band', fs=sample_rate, output='sos')
    w, h_scipy_bp = signal.sosfreqz(sos_bp, worN=2048, fs=sample_rate)

    zs_resp_bp = measure_frequency_response(zs_bp, freqs, sample_rate)
    scipy_resp_bp = np.interp(freqs, w, np.abs(h_scipy_bp))

    axes[1, 0].semilogx(freqs, 20 * np.log10(zs_resp_bp + 1e-10), 'b-', linewidth=2, label='zerostone')
    axes[1, 0].semilogx(freqs, 20 * np.log10(scipy_resp_bp + 1e-10), 'r--', linewidth=2, label='scipy')
    axes[1, 0].axvline(x=low_bp, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].axvline(x=high_bp, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_title(f'Bandpass ({low_bp}-{high_bp} Hz)')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-60, 5])

    # Response difference summary
    axes[1, 1].semilogx(freqs, 20 * np.log10(zs_resp_lp + 1e-10) - 20 * np.log10(scipy_resp_lp + 1e-10),
                        'b-', linewidth=1, label='lowpass', alpha=0.7)
    axes[1, 1].semilogx(freqs, 20 * np.log10(zs_resp_hp + 1e-10) - 20 * np.log10(scipy_resp_hp + 1e-10),
                        'r-', linewidth=1, label='highpass', alpha=0.7)
    axes[1, 1].semilogx(freqs, 20 * np.log10(zs_resp_bp + 1e-10) - 20 * np.log10(scipy_resp_bp + 1e-10),
                        'g-', linewidth=1, label='bandpass', alpha=0.7)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Response Difference (zerostone - scipy)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Difference (dB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-2, 2])

    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_response_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'frequency_response_comparison.png'}")


def measure_frequency_response(filter_obj, freqs: np.ndarray, sample_rate: float) -> np.ndarray:
    """Measure filter frequency response by filtering sinusoids."""
    response = []
    for freq in freqs:
        filter_obj.reset()
        t = np.arange(0, 2, 1/sample_rate, dtype=np.float32)
        test_sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
        output = filter_obj.process(test_sine)
        rms_out = np.sqrt(np.mean(output[1000:] ** 2))
        rms_in = np.sqrt(np.mean(test_sine[1000:] ** 2))
        response.append(rms_out / max(rms_in, 1e-10))
    return np.array(response)


def print_summary():
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("""
Key findings:

IIR FILTERS (lowpass/highpass/bandpass):
- Zerostone uses cascaded IDENTICAL biquad sections, while scipy uses
  proper Butterworth pole placement with different coefficients per section
- This causes a resonance peak near the cutoff frequency (gain ~2x at cutoff)
- Stopband attenuation is excellent (steep rolloff)
- For applications requiring flat passband, use scipy instead
- For applications needing strong stopband rejection, zerostone works well

FIR FILTERS:
- Exact match with scipy (difference ~1e-7)
- Same direct-form convolution algorithm

NUMERICAL PRECISION:
- Zerostone uses f32, scipy uses f64
- FIR differences are purely numerical precision (~1e-7)
- IIR differences are primarily from different filter designs

USE CASES:
- Zerostone IIR: Good for stopband rejection, avoid for flat passband needs
- Zerostone FIR: Identical to scipy, full interoperability
- For critical Butterworth response, design coefficients with scipy and use
  FirFilter with the resulting taps
""")


def main():
    parser = argparse.ArgumentParser(description='Compare zerostone filters against scipy')
    parser.add_argument('--output-dir', default='output/scipy_validation',
                        help='Directory for output plots')
    args = parser.parse_args()

    output_dir = setup_output_dir(args.output_dir)

    print("=" * 60)
    print("Zerostone vs SciPy Filter Validation")
    print("=" * 60)

    compare_lowpass_filter(output_dir)
    compare_highpass_filter(output_dir)
    compare_bandpass_filter(output_dir)
    compare_fir_filter(output_dir)
    compare_frequency_responses(output_dir)

    print_summary()

    print(f"\nAll plots saved to: {output_dir}/")
    print("Run 'pytest tests/test_scipy_validation.py -v' for automated tests")


if __name__ == '__main__':
    main()
