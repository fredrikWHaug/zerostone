# Zerostone Firmware Memory Map

**Target**: nRF5340 Application Core (Cortex-M33, 1 MB Flash, 256 KB SRAM)
**Date**: 2026-03-31
**Build**: `cargo build --target thumbv8m.main-none-eabihf --release`

## Flash Layout

| Section | Size | Address | Description |
|---------|------|---------|-------------|
| .vector_table | 340 B | 0x00000000 | Cortex-M exception vectors |
| .text | 20.0 KB | 0x00000154 | Code (LTO + opt-level=z) |
| .rodata | 3.2 KB | 0x00005154 | Constants, defmt strings |
| .gnu.sgstubs | 0 B | 0x00005E28 | TrustZone stubs (unused) |
| **Total Flash** | **~23.6 KB** | | **2.3% of 1 MB** |

## SRAM Layout

| Section | Size | Address | Description |
|---------|------|---------|-------------|
| .data | 80 B | 0x20000000 | Initialized globals |
| .bss | 9.0 KB | 0x20000050 | Zero-initialized globals |
| .uninit | 1.0 KB | 0x20002450 | Uninitialized (defmt buffer) |
| **Total Static** | **~10.3 KB** | | **4% of 256 KB** |

## Static Allocation Breakdown (.bss)

| Allocation | Size | Notes |
|------------|------|-------|
| FRAME_CHANNEL (64 frames x 32ch x 2B) | 4.0 KB | SPI -> Processing |
| EVENT_CHANNEL (32 events x 12B) | 0.4 KB | Processing -> BLE |
| STATS_CHANNEL (4 snapshots x 16B) | 0.1 KB | Processing -> Stats |
| Embassy executor + timer | ~3.0 KB | Runtime overhead |
| Misc (defmt, cortex-m) | ~1.5 KB | |

## Stack Budget (estimated)

| Task | Stack (est.) | Notes |
|------|-------------|-------|
| spi_task | ~1 KB | IntanDriver + SPI buffer |
| processing_task | ~14 KB | Pipeline + WaveformExtractor + Classifier + OnlineLearner + RuntimeStats |
| ble_task | ~1 KB | BleServer<8> + serialize buffer |
| stats_task | ~0.2 KB | Deserialize + defmt |
| heartbeat_task | ~0.1 KB | GPIO toggle |
| **Total Stack** | **~16 KB** | |

## Total SRAM Usage

| Component | Size |
|-----------|------|
| Static (.data + .bss + .uninit) | 10.3 KB |
| Stack (all tasks) | ~16 KB |
| **Total** | **~26 KB** |
| **Available** | **256 KB** |
| **Headroom** | **~91%** |

## Notes

- No heap allocation: `#![no_std]` with no `alloc` crate.
- All buffers are either stack-allocated (within tasks) or static (channels).
- Binary size is dominated by Embassy runtime and defmt formatting. The spike sorting pipeline itself is <4 KB .text.
- LTO and opt-level=z are enabled, stripping all unused Zerostone modules.
- The Zerostone library dependency pulls in only the modules actually used: `float` (f32 wrappers). All signal processing (pipeline, classifier) is reimplemented in the firmware crate for `no_std` with fixed-size arrays.
