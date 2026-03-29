# Zerostone Firmware Power Budget

**Target**: nRF5340 DK (application core + network core)
**Date**: 2026-03-29
**Source**: Nordic nRF5340 Product Specification v1.3

## Component Power Draw

| Component | Mode | Current | Duty Cycle | Average Current |
|-----------|------|---------|------------|----------------|
| App core (128 MHz) | Active | 3.2 mA | 100% | 3.2 mA |
| App core (128 MHz) | WFI idle | 1.1 mA | - | - |
| Network core (64 MHz) | Active + BLE TX | 4.6 mA | ~5% | 0.23 mA |
| Network core (64 MHz) | Sleep | 0.002 mA | 95% | 0.002 mA |
| SPI (8 MHz SCLK) | Active | 0.3 mA | 64% | 0.19 mA |
| SPI | Idle | 0.0 mA | 36% | 0.0 mA |
| GPIO (LED heartbeat) | On | 2.0 mA | 50% | 1.0 mA |
| Intan RHD2132 | Active | 7.0 mA | 100% | 7.0 mA |
| Voltage regulator | Quiescent | 0.01 mA | 100% | 0.01 mA |
| **Total system** | | | | **~11.6 mA** |

## Without LED (Production)

| | Average Current |
|---|----------------|
| Total without LED | ~10.6 mA |
| Total with sleep optimization* | ~8.5 mA |

*Sleep optimization: clock-gate app core between 30 kHz samples (WFI for ~12 us per 33 us period). Reduces app core average from 3.2 mA to ~1.7 mA.

## Battery Life Estimates

| Battery | Capacity | Weight | Life (no opt) | Life (with sleep) |
|---------|----------|--------|---------------|-------------------|
| 30 mAh (1g LiPo) | 30 mAh | 1.0 g | 2.6 hrs | 3.5 hrs |
| 60 mAh (1.5g LiPo) | 60 mAh | 1.5 g | 5.2 hrs | 7.1 hrs |
| 100 mAh (2.5g LiPo) | 100 mAh | 2.5 g | 8.6 hrs | 11.8 hrs |
| 150 mAh (3g LiPo) | 150 mAh | 3.0 g | 12.9 hrs | 17.6 hrs |

## BLE Power Details

- Connection interval: 7.5 ms (minimum for low latency)
- Spike event size: 8 bytes per event
- Expected spike rate: ~10-50 Hz per channel, ~320-1600 Hz total (32ch)
- At 1600 Hz spike rate: ~12.8 KB/s -> fits in single BLE connection event
- BLE TX power: 0 dBm (default), range ~10m in lab

## Intan RHD2132 Power

The Intan chip dominates power at 7 mA. Options to reduce:
1. Disable unused channels (saves ~0.2 mA per disabled channel)
2. Lower sampling rate when not actively recording
3. Power-gate Intan between recording sessions (100 ms startup from power-on)

## Key Takeaway

With a 60 mAh battery (~1.5g), the system runs for 5-7 hours -- sufficient for a full mouse behavioral session. The Intan IC is the dominant power consumer; the nRF5340 itself draws only ~3.5 mA active.
