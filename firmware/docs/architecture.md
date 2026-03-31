# Zerostone Firmware Architecture

**Target**: nRF5340 Application Core (Cortex-M33, 128 MHz, FPv5-SP FPU)
**Framework**: Embassy async executor (`no_std`, zero heap)
**Date**: 2026-03-31

## Data Flow

```
                        30 kHz Ticker
                             |
                             v
+------------------+    +----------+    +-----------+    +-----------+    +----------+
| Intan RHD2132    |--->| spi_task |--->| FRAME_CH  |--->| proc_task |--->| EVENT_CH |
| 32ch amplifier   | SPI| read     |    | 64 frames |    | detect +  |    | 32 events|
| + ADC (16-bit)   | 8M | _frame() |    | [i16; 32] |    | classify +|    | SpikeEvt |
+------------------+    +----------+    +-----------+    | learn     |    +----------+
                                                         +-----------+         |
                                                              |                v
                                                              |          +----------+
                                                         +----+----+     | ble_task |
                                                         |STATS_CH |     | BleServer|
                                                         | 4 x 16B |     | GATT     |
                                                         +----+----+     +----------+
                                                              |                |
                                                              v                v
                                                         +----------+    BLE GATT
                                                         |stats_task|   Notification
                                                         | defmt log|
                                                         +----------+
```

Detailed processing pipeline inside `processing_task`:

```
ADC frame [i16; 32]
    |
    v
WaveformExtractor.push_frame()    -- circular buffer, 48 samples/ch
    |
    v
Pipeline.process_frame()          -- per-channel EMA noise estimation
    |                                 threshold = 5.0 * MAD(noise)
    |                                 hysteresis (in_spike flag)
    v
SpikeEvent { sample_idx, channel, amplitude, cluster_id=0 }
    |
    v
WaveformExtractor.extract()       -- last 48 samples for spike channel
    |
    v
Classifier.classify()             -- NCC against 8 templates
    |                                 min_correlation = 0.7
    |                                 uses fast_inv_sqrt (no VSQRT)
    v
SpikeEvent { ..., cluster_id }    -- 0 = unclassified, 1..N = matched
    |
    +-> OnlineLearner.learn()     -- EMA accumulate waveform into template
    |   (every 10K frames: merge + reload classifier)
    |
    +-> RuntimeStats.record_spike()  -- track classified/unclassified counts
    |
    v
EVENT_CHANNEL.try_send()          -- non-blocking, drops if full
```

## Task Architecture

Four Embassy async tasks, cooperatively scheduled on a single core (no preemption, no RTOS threads). All inter-task communication uses `embassy_sync::Channel` with `NoopRawMutex` (safe because Embassy tasks on a single core never truly run in parallel).

| Task | Priority | Trigger | Period | Function |
|------|----------|---------|--------|----------|
| `spi_task` | Normal | `Ticker` (33 us) | 30 kHz | Read 32ch frame from Intan over SPI |
| `processing_task` | Normal | `frame_rx.receive()` | Event-driven | Detect + classify + learn + stats |
| `ble_task` | Normal | `event_rx.receive()` | Event-driven | BleServer GATT + spike serialization |
| `stats_task` | Normal | `stats_rx.receive()` | ~1 Hz | Deserialize + defmt log stats snapshot |
| `heartbeat_task` | Normal | `Timer::after_millis(500)` | 1 Hz | Toggle LED on P0.28 |

### Channel interconnects

```
spi_task ---[FRAME_CHANNEL]--> processing_task ---[EVENT_CHANNEL]--> ble_task
                                     |
                                     +---[STATS_CHANNEL]--> stats_task

FRAME_CHANNEL: Channel<NoopRawMutex, [i16; 32], 64>
  - 64 frames deep (2.1 ms buffer at 30 kHz)
  - try_send() in spi_task (non-blocking, drops frame if full)
  - receive().await in processing_task (blocks until frame available)

EVENT_CHANNEL: Channel<NoopRawMutex, SpikeEvent, 32>
  - 32 events deep
  - try_send() in processing_task (non-blocking, drops event if full)
  - receive().await in ble_task (blocks until event available)

STATS_CHANNEL: Channel<NoopRawMutex, [u8; 16], 4>
  - 4 snapshots deep (16 bytes each, RuntimeStats::serialize format)
  - try_send() every 30,000 frames (~1 Hz) in processing_task
  - receive().await in stats_task (logs via defmt)
```

### Static allocation

Both channels are allocated in `StaticCell` statics (zero heap, `'static` lifetime for task sharing).

## Timing Budget (per 33.3 us frame period)

| Phase | Duration | Source |
|-------|----------|--------|
| SPI frame read (34 transactions x 16 bits at 8 MHz) | ~16.0 us | 34 x 2B at 8 MHz SPI clock |
| Pipeline noise update (32 channels, EMA) | ~2.0 us | 32 x ~8 FPU ops |
| Threshold detection + hysteresis (32 ch) | ~1.0 us | Comparisons + branch |
| Waveform extraction (if spike) | ~0.5 us | Circular buffer copy, 48 samples |
| NCC classification (if spike, 8 templates) | ~5.1 us | 656 cycles at 128 MHz (see npu_assessment.md) |
| BLE serialization (if spike) | ~0.1 us | 8-byte pack |
| **Total (no spike)** | **~19.0 us** | **57% of frame budget** |
| **Total (with spike + classify)** | **~24.7 us** | **74% of frame budget** |
| **Headroom** | **~8.6 us** | **26% margin** |

Notes:
- Classification only runs when a spike is detected (~50 Hz per channel typical, not every frame).
- The SPI transfer is the dominant cost. DMA would recover ~16 us of CPU time but is not yet implemented.
- Worst case: all 32 channels spike simultaneously in one frame. This cannot happen in practice (refractory period).

## Memory Layout Summary

See `memory_map.md` for full details.

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| Flash (.text + .rodata) | ~23.6 KB | 1 MB | 2.3% |
| SRAM (static + stack) | ~26 KB | 256 KB | 10% |
| Heap | 0 | -- | `#![no_std]`, no alloc |

Key static allocations:
- `FRAME_CHANNEL`: 4.0 KB (64 frames x 32 ch x 2 B)
- `EVENT_CHANNEL`: 0.4 KB (32 events x ~12 B)
- `STATS_CHANNEL`: 0.1 KB (4 snapshots x 16 B)
- Processing task stack: ~14 KB (Pipeline + WaveformExtractor + Classifier + OnlineLearner + RuntimeStats)

## Module Dependency Graph

```
main.rs (binary entry point)
  |-- embassy_executor, embassy_nrf, embassy_sync, embassy_time
  |-- embedded_hal_bus::spi::ExclusiveDevice
  |-- static_cell, defmt_rtt, panic_probe
  |
  |-- intan.rs          (Intan RHD2132 SPI driver)
  |     \-- embedded_hal_async::spi::SpiDevice
  |
  |-- pipeline.rs       (spike detection + EventQueue)
  |     \-- zerostone::float::Float
  |
  |-- classifier.rs     (NCC template matching + WaveformExtractor)
  |     \-- dsp.rs      (dot_product, norm_sq, fast_inv_sqrt, ncc)
  |
  |-- ble.rs            (GATT UUIDs + spike/config serialization)
  |     \-- pipeline::SpikeEvent
  |
  |-- online_learn.rs   (TemplateAccumulator + OnlineLearner)
  |     \-- zerostone::float::{Float, sqrt}
  |
  |-- ble_server.rs     (BleServer<N> + GattTable + state machine)
  |     \-- pipeline::SpikeEvent, ble::serialize_spike_event
  |
  |-- stats.rs          (RuntimeStats, 16B BLE serialization)
  |
  |-- stim.rs           (StimConfig + StimState + StimDecision)
  |     \-- pipeline::SpikeEvent
  |
  |-- power.rs          (PowerConfig + PowerProfile + battery estimates)
  |
  |-- flash_log.rs      (FlashRegion + FlashLog<N> + LogEntryHeader)
  |
  |-- fault.rs          (FaultLog<N> + DfuConfig + ResetReason)
  |
  |-- watchdog.rs       (WatchdogConfig + WatchdogState)
  |
  \-- ring_buffer.rs    (FrameRingBuffer, available but Channel used instead)
```

External dependency: `zerostone` (parent crate) is used only for `zerostone::float::Float` (f32 type alias and `sqrt` wrapper). All signal processing is reimplemented in the firmware crate with fixed-size arrays for `no_std` compatibility.

## Key Design Decisions

### Why Embassy (not RTIC, FreeRTOS, or bare-metal)

- **Async/await maps naturally to the data flow**: SPI read -> channel -> process -> channel -> BLE. Each stage is an `async fn` that awaits data arrival.
- **No preemption complexity**: All tasks run cooperatively on one core. No priority inversion, no mutexes, no critical sections beyond what Embassy provides.
- **Zero-cost channels**: `NoopRawMutex` because single-core cooperative scheduling guarantees no concurrent access. No atomic overhead.
- **Mature nRF5340 HAL**: `embassy-nrf` provides async SPI, GPIO, and timer drivers with interrupt-driven wakeup.

### Why f32 (not i16 or fixed-point)

- **Cortex-M33 has a hardware FPU**: Single-cycle FMA via `VFMA.F32`. Using f32 is literally free compared to integer multiply-accumulate.
- **NCC requires normalization**: The `dot / sqrt(norm_a * norm_b)` formula needs division and square root. In fixed-point, these require multi-step iterative algorithms that are slower than the f32 FPU path.
- **Maintainability**: f32 code is readable. Fixed-point requires manual scaling at every operation boundary.
- **Precision is adequate**: 23-bit mantissa gives ~7 decimal digits, more than enough for spike waveforms normalized to [-1, 1].

### Why NCC (not Euclidean distance, PCA+GMM, or neural network)

- **Amplitude-invariant**: NCC matches waveform shape regardless of spike amplitude. A unit that fires weakly still matches its template. Euclidean distance would require amplitude normalization as a separate step.
- **Precomputable**: Template `norm_sq` is computed once at template load time. Per-spike cost is one dot product + one fast_inv_sqrt.
- **Deterministic and interpretable**: No iterative convergence, no random initialization. The same input always produces the same output. Critical for FDA-class closed-loop applications.
- **Bounded output**: NCC is in [-1, 1]. The threshold `min_correlation = 0.7` has a clear geometric interpretation (cosine similarity). Easy to tune.
- **Sufficient for 8 templates**: NCC scales linearly with template count. At 8 templates x 48 samples, total classification is ~5 us. PCA+GMM would require eigenvector projection (~similar cost) plus Mahalanobis distance (~more expensive) with no clear accuracy benefit for <10 units.

## Current Limitations

1. **BLE radio not connected**: `ble_task` uses BleServer with mock GATT table. Real `trouble` BLE stack integration is pending hardware bring-up.
2. **No DMA for SPI**: The SPI task uses polled `transfer_in_place`. DMA would free the CPU during the 16 us SPI transfer window.
3. **Single-channel classification**: Each channel is classified independently. No multi-channel deduplication or spatial localization.
4. **Stimulation not wired into main.rs**: `stim.rs` module is complete and tested but the GPIO trigger path is not in the main loop (needs hardware pin assignment).

## Future Work

- Add DMA-based SPI transfers with double-buffering for CPU overlap.
- Connect `trouble` BLE stack on the network core for actual wireless transmission.
- Wire `StimState` into processing_task with GPIO output pin for closed-loop stimulation.
- Add multi-channel template matching with probe geometry awareness.
- Flash logging integration: write stats/faults to NVMC during recording sessions.
