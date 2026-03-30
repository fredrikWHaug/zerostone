# Zerostone Firmware Pin Assignments

**Target**: nRF5340-DK / Custom headstage PCB
**Package**: aQFN94 (nRF5340)
**Date**: 2026-03-30

## Pin Assignment Table

### Application-Defined Pins (configurable in firmware)

These pins are assigned in `src/main.rs` and can be changed to any available GPIO.

| nRF5340 Pin | Function | Direction | Connected To | Speed | Notes |
|-------------|----------|-----------|--------------|-------|-------|
| P0.13 | SPI SCK | Output | Intan RHD2132 SCLK | 8 MHz | CPOL=0, CPHA=0 (MODE_0) |
| P0.14 | SPI MOSI | Output | Intan RHD2132 MOSI | 8 MHz | 16-bit commands |
| P0.15 | SPI MISO | Input | Intan RHD2132 MISO | 8 MHz | 16-bit ADC results, 2-deep pipeline |
| P0.16 | SPI CS | Output | Intan RHD2132 CS# | -- | Active low, managed by ExclusiveDevice |
| P0.28 | LED | Output | Status LED (anode) | -- | 1 Hz heartbeat blink, StandardDrive |

### SPI Configuration Details

| Parameter | Value | Source |
|-----------|-------|--------|
| Peripheral | SERIAL0 (SPIM0) | `peripherals::SERIAL0` in main.rs |
| Clock frequency | 8 MHz | `spim::Frequency::M8` |
| Mode | MODE_0 (CPOL=0, CPHA=0) | `spim::MODE_0` (Intan default) |
| Word size | 16-bit (2 x 8-bit transfers) | `transfer_in_place(&mut [u8; 2])` |
| CS polarity | Active low | `Output::new(..., Level::High, ...)` |
| Interrupt | SERIAL0 | Bound in `bind_interrupts!` |

### Fixed Hardware Pins (not reassignable)

These pins have dedicated functions determined by the nRF5340 silicon.

| nRF5340 Pin | Function | Direction | Notes |
|-------------|----------|-----------|-------|
| SWDIO | Debug data | Bidir | SWD debug interface, active during development |
| SWDCLK | Debug clock | Input | SWD debug interface |
| RESET | System reset | Input | Active low, internal pull-up. Optional external button. |
| XC1 / P0.00 | 32 MHz HFXO | Analog | External 32 MHz crystal, pin 1 |
| XC2 / P0.01 | 32 MHz HFXO | Analog | External 32 MHz crystal, pin 2 |
| P0.14 | 32.768 kHz LFXO | Analog | Optional: shares P0.14/P0.15 with LFXO. Firmware uses RTC1 timer driver which requires LFCLK. If LFXO is used, SPI pins must be reassigned. |
| P0.15 | 32.768 kHz LFXO | Analog | See note above |
| VDDH | High voltage supply | Power | 2.5V -- 5.5V input (from LiPo via LDO, or USB) |
| VDD | Core supply | Power | 1.8V internal regulator output (decoupled externally) |
| VDDH_RADIO | Radio supply | Power | Same rail as VDDH, separate decoupling |
| VSS | Ground | Power | All VSS pins must connect to ground plane |
| DEC1 -- DEC6 | Decoupling | Power | Internal regulator decoupling, 100 nF to GND each |
| ANT | BLE antenna | RF | 2.4 GHz antenna feed, 50 ohm |

**IMPORTANT -- LFXO Pin Conflict**: The nRF5340 LFXO pins are P0.14 and P0.15, which overlap with the current SPI MOSI and MISO assignments. Two options:

1. **Use LFRC** (internal 32.768 kHz RC oscillator) instead of LFXO. Accuracy is +/-250 ppm, adequate for Embassy tick timing but not for precision RTC. The current firmware uses `time-driver-rtc1` which works with either LFRC or LFXO.
2. **Reassign SPI pins** to avoid P0.14/P0.15. Any GPIO can serve as SPI since the nRF5340 SPIM peripheral is fully remappable.

Recommendation: Reassign SPI MOSI/MISO to P0.17/P0.18 (or similar) and use LFXO for better timing accuracy. Update `main.rs` pin assignments accordingly.

### Power Pins

| Pin | Voltage | Notes |
|-----|---------|-------|
| VDDH (pins 31, 48) | 3.0V -- 3.6V from LiPo | Primary supply input |
| VDD (pins 13, 36, 52) | 1.8V | Internal REG0 output, decouple with 1 uF + 100 nF |
| VDDH_RADIO (pin 53) | Same as VDDH | Separate decoupling, short trace to radio |
| DEC1 (pin 40) | -- | 100 nF to GND |
| DEC2 (pin 33) | -- | 100 nF to GND |
| DEC3 (pin 34) | -- | 100 nF to GND |
| DEC4 (pin 55) | -- | 100 nF to GND |
| DEC5 (pin 57) | -- | 100 nF to GND |
| DEC6 (pin 68) | -- | 100 nF to GND |
| VSS (multiple) | GND | All VSS pins to ground plane, short vias |

### Unused GPIO (available for future use)

All P0 and P1 GPIO pins not listed above are available. Candidates for future signals:

| Pin | Suggested Future Use |
|-----|---------------------|
| P0.02 -- P0.12 | Additional SPI CS lines (multi-Intan), aux ADC, digital I/O |
| P0.17 -- P0.27 | I2C (accelerometer, temp sensor), UART debug output |
| P0.29 -- P0.31 | Additional LEDs, interrupt inputs |
| P1.00 -- P1.15 | Available on nRF5340, useful for expanded I/O on larger boards |

### Test Point Recommendations

These signals should have accessible test points on the PCB for bring-up and debugging.

| Signal | Pin | Reason |
|--------|-----|--------|
| SPI SCK | P0.13 | Verify 8 MHz clock, signal integrity |
| SPI MISO | P0.15 | Verify Intan ADC data |
| SPI CS | P0.16 | Verify CS timing relative to SCK |
| SWDIO | SWDIO | Required for flash programming and debug |
| SWDCLK | SWDCLK | Required for flash programming and debug |
| VDD | VDD | Verify 1.8V rail |
| VDDH | VDDH | Verify 3.3V supply |
| GND | VSS | Probe ground reference |
| RESET | RESET | Manual reset during debug |
| LED | P0.28 | Verify firmware is running (heartbeat) |
