# Zerostone Firmware Memory Map

**Target**: nRF5340 Application Core (Cortex-M33, 1 MB Flash, 256 KB SRAM)
**Date**: 2026-03-29
**Build**: `cargo build --target thumbv8m.main-none-eabihf --release`

## Flash Layout

| Section | Size | Address | Description |
|---------|------|---------|-------------|
| .vector_table | 340 B | 0x00000000 | Cortex-M exception vectors |
| .text | 16.5 KB | 0x00000154 | Code (LTO + opt-level=z) |
| .rodata | 3.1 KB | 0x00004388 | Constants, defmt strings |
| .gnu.sgstubs | 0 B | 0x00005060 | TrustZone stubs (unused) |
| **Total Flash** | **~20.5 KB** | | **2% of 1 MB** |

## SRAM Layout

| Section | Size | Address | Description |
|---------|------|---------|-------------|
| .data | 80 B | 0x20000000 | Initialized globals |
| .bss | 8.9 KB | 0x20000050 | Zero-initialized globals |
| .uninit | 1.0 KB | 0x20002404 | Uninitialized (defmt buffer) |
| **Total Static** | **~10.2 KB** | | **4% of 256 KB** |

## Static Allocation Breakdown (.bss)

| Allocation | Size | Notes |
|------------|------|-------|
| FRAME_CHANNEL (64 frames x 32ch x 2B) | 4.0 KB | SPI -> Processing |
| EVENT_CHANNEL (32 events x 12B) | 0.4 KB | Processing -> BLE |
| Embassy executor + timer | ~3.0 KB | Runtime overhead |
| Misc (defmt, cortex-m) | ~1.5 KB | |

## Stack Budget (estimated)

| Task | Stack (est.) | Notes |
|------|-------------|-------|
| spi_task | ~1 KB | IntanDriver + SPI buffer |
| processing_task | ~12 KB | Pipeline (128ch state) + WaveformExtractor (32ch x 48 x 4B = 6 KB) + Classifier templates |
| ble_task | ~0.5 KB | 8-byte serialize buffer |
| heartbeat_task | ~0.1 KB | GPIO toggle |
| **Total Stack** | **~14 KB** | |

## Total SRAM Usage

| Component | Size |
|-----------|------|
| Static (.data + .bss + .uninit) | 10.2 KB |
| Stack (all tasks) | ~14 KB |
| **Total** | **~24 KB** |
| **Available** | **256 KB** |
| **Headroom** | **~91%** |

## Notes

- No heap allocation: `#![no_std]` with no `alloc` crate.
- All buffers are either stack-allocated (within tasks) or static (channels).
- Binary size is dominated by Embassy runtime and defmt formatting. The spike sorting pipeline itself is <4 KB .text.
- LTO and opt-level=z are enabled, stripping all unused Zerostone modules.
- The Zerostone library dependency pulls in only the modules actually used: `float` (f32 wrappers). All signal processing (pipeline, classifier) is reimplemented in the firmware crate for `no_std` with fixed-size arrays.
