# Cortex-M33 DSP Assessment for Spike Classification

## Cortex-M33 DSP Capabilities

The nRF5340 application core is an ARM Cortex-M33 running at 128 MHz with:

- **Single-precision FPU (FPv5-SP)**: Hardware f32 add, multiply, and
  fused multiply-accumulate (FMA) in 1 cycle each. The FMA instruction
  (`VFMA.F32`) computes `a += b * c` atomically, which is the core operation
  in dot product / NCC computation.

- **DSP SIMD extensions**: Dual 16-bit multiply-accumulate (`SMLAD`, `SMLALD`)
  for integer paths. Not used in the current f32 pipeline but available for
  future int16 quantized paths.

- **No hardware VSQRT**: Square root is a multi-cycle iterative instruction
  (~14 cycles). We avoid it entirely by using Quake-style fast inverse square
  root (`fast_inv_sqrt`), which compiles to one integer subtract, one shift,
  and a few FMA iterations -- approximately 6-8 cycles total with <0.2%
  relative error.

## Cycle Count Estimates: NCC at 48 Samples x 8 Templates

### Per-template NCC breakdown

| Operation              | Cycles per sample | Samples | Total cycles |
|------------------------|:-----------------:|:-------:|:------------:|
| Waveform norm_sq       | 1 (FMA)           | 48      | 48           |
| Dot product            | 1 (FMA)           | 48      | 48           |
| fast_inv_sqrt          | --                | --      | ~8           |
| Multiply + compare     | --                | --      | ~4           |
| Loop overhead          | --                | --      | ~12          |
| **Per-template total** |                   |         | **~120**     |

The waveform `norm_sq` is computed once and reused across all templates.

### Full classification pass (8 templates)

| Component                  | Cycles       |
|----------------------------|:------------:|
| Waveform norm_sq (once)    | ~60          |
| 8x template NCC            | 8 x 72 = 576|
| Best-of-N compare + return | ~20          |
| **Total**                  | **~656**     |

At 128 MHz, 656 cycles = **5.1 microseconds**.

### Throughput headroom

At 30 kHz sampling with a typical firing rate of 50 Hz per channel across
32 channels, the worst-case spike rate is ~1,600 spikes/second. Each
classification takes ~5.1 us, so total classification compute is:

    1,600 spikes/s x 5.1 us = 8.2 ms/s = 0.82% CPU utilization

Even at 10x this rate (pathological burst), classification uses <10% of CPU.

## Conclusion: NPU Is Not Needed

The Cortex-M33 FPU handles the NCC compute budget with substantial margin.
At the target workload (48-sample waveforms, 8 templates, 30 kHz sampling),
classification consumes under 1% of available CPU cycles.

**The key bottleneck is SPI transfer time, not compute.** Reading 32 channels
from the Intan RHD2132 at 30 kHz over SPI requires:

    32 channels x 16 bits x 30,000 Hz = 15.36 Mbps sustained

At the nRF5340's maximum SPI clock of 32 MHz, this is ~48% bus utilization
before accounting for command overhead and pipeline fill. The SPI transfer
dominates the per-frame time budget (~33.3 us per frame, of which ~16 us is
SPI transfer vs ~5.1 us worst-case classification).

An NPU would add silicon cost, power draw, and firmware complexity for a
workload that fits comfortably in the existing FPU. The correct optimization
target is SPI DMA scheduling and pipeline overlap, not additional compute
hardware.
