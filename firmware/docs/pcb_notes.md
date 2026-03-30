# PCB Design Notes for Zerostone Headstage

**Target**: Mouse headstage for chronic neural recording
**Board size**: < 15 mm x 15 mm (target 12 mm x 12 mm)
**Layer stackup**: 4-layer recommended
**Date**: 2026-03-30

## 1. Layer Stackup

Recommended 4-layer stackup for signal integrity and EMI control:

```
Layer 1 (Top):     Signal + components (SPI traces, nRF5340, Intan, antenna)
Layer 2 (Inner 1): Ground plane (continuous, unbroken under SPI and antenna)
Layer 3 (Inner 2): Power plane (3.3V, split if separate analog supply needed)
Layer 4 (Bottom):  Signal + components (passives, test points, debug header)
```

- Maintain continuous ground plane on Layer 2 with no splits under the SPI bus or antenna feed.
- If the board is too small for 4 layers, a 2-layer board is possible but requires careful ground pour management and wider trace spacing.

## 2. SPI Bus Routing (nRF5340 to Intan RHD2132)

The SPI bus runs at 8 MHz. At this frequency, transmission line effects are minimal (wavelength >> trace length), but good practice prevents noise coupling into the analog front end.

| Parameter | Recommendation |
|-----------|---------------|
| Trace width | 6 mil (150 um) for 50 ohm on 4-layer stackup with 8 mil dielectric |
| Trace spacing | Minimum 3x trace width (18 mil) between SPI signals |
| Length matching | Not required at 8 MHz (propagation delay differences are negligible at <15 mm) |
| Ground plane | Continuous copper pour on Layer 2 directly underneath all SPI traces |
| Vias | Avoid vias in SPI traces if possible. If layer transitions are needed, use two ground vias adjacent to each signal via |
| Routing | Keep SPI traces as short as possible. Place nRF5340 and Intan adjacent, ideally < 5 mm apart |
| Guard traces | Not required at 8 MHz, but keep SPI traces away from the antenna feed |

### SPI Signal Connections

| nRF5340 Pin | Signal | Intan RHD2132 Pin |
|-------------|--------|-------------------|
| P0.13 | SCK | SCLK |
| P0.14 | MOSI | MOSI |
| P0.15 | MISO | MISO |
| P0.16 | CS# | CS# |

See `pinout.md` for LFXO pin conflict note -- if using LFXO, reassign MOSI/MISO to P0.17/P0.18.

## 3. Decoupling Capacitors

### nRF5340 Decoupling

Follow Nordic's reference design (nRF5340 Product Specification Section 9.2). Every VDD and VDDH pin requires local decoupling.

| Pin | Capacitor | Placement |
|-----|-----------|-----------|
| Each VDD pin (3 pins) | 100 nF ceramic (0201) | < 2 mm from pin, via to GND plane |
| VDDH bulk | 10 uF ceramic (0402) | Near VDDH pin, shared across VDDH pins |
| Each VDDH pin (2 pins) | 100 nF ceramic (0201) | < 2 mm from pin |
| VDDH_RADIO | 100 nF + 10 uF | < 2 mm from pin, dedicated trace to LDO |
| DEC1 -- DEC6 | 100 nF ceramic (0201) each | As close to pin as physically possible |

Rules:
- Place 100 nF caps on the same side of the board as the nRF5340 (no vias in the cap-to-pin path).
- Use 0201 package for minimum inductance.
- Via-in-pad for ground connection where possible.

### Intan RHD2132 Decoupling

| Pin | Capacitor | Placement |
|-----|-----------|-----------|
| VDD (digital) | 100 nF + 10 uF | < 2 mm from pin |
| VDA (analog) | 100 nF + 10 uF | < 2 mm from pin, separate trace from VDD |
| VREF | 10 uF | Low-ESR, for ADC reference stability |

## 4. Intan RHD2132 Analog Considerations

The RHD2132 is a mixed-signal device with 32 low-noise amplifiers. Analog layout is critical for recording quality.

### Ground separation

- Use a **star ground** topology or carefully managed split ground plane:
  - Analog ground (AGND): Intan analog pins, electrode connector, reference caps.
  - Digital ground (DGND): nRF5340, SPI traces, LED, BLE antenna.
  - Connect AGND and DGND at a **single point** near the Intan's GND pin.
- Do not route digital signals (SPI, LED) over the analog ground region.

### Reference voltage

- The Intan ELEC_REF pin sets the electrode reference voltage. Route this trace away from all digital signals.
- Place a 10 uF + 100 nF filter cap on ELEC_REF, directly adjacent to the pin.
- If using an external reference electrode, route the trace as a guarded pair (ground traces on both sides).

### Input filtering

- The RHD2132 has on-chip bandpass filtering (configured via SPI registers 0-3 for 300 Hz -- 6 kHz in the current firmware).
- No external input filtering is required on the electrode traces. Keep electrode traces as short as possible to minimize pickup.
- Electrode connector should be placed as close to the Intan as possible (< 3 mm).

### Bandwidth configuration

The firmware configures the following analog bandwidth via registers:
- Upper cutoff: ~6 kHz (Rh1 DAC = 8, register 0)
- Lower cutoff: ~300 Hz (RL DAC = 4, register 3)
- This is appropriate for extracellular spike recording at 30 kHz sample rate.

## 5. Power Supply

### Architecture

```
LiPo (3.7V nom, 3.0-4.2V)
    |
    v
LDO regulator (3.3V output)
    |
    +---> VDDH (nRF5340, 3.3V)
    +---> VDDH_RADIO (nRF5340, 3.3V, separate trace)
    +---> VDD_INTAN (Intan digital, 3.3V)
    +---> VDA_INTAN (Intan analog, 3.3V, filtered)
    |
    nRF5340 internal regulator
    |
    +---> VDD (1.8V core, auto-regulated)
```

### LDO Selection Criteria

| Parameter | Requirement | Reason |
|-----------|-------------|--------|
| Output voltage | 3.3V +/- 2% | nRF5340 VDDH range: 2.5V -- 5.5V; Intan needs 3.3V nominal |
| Output current | >= 30 mA | System draws ~11.6 mA typical, 30 mA covers startup transients |
| Dropout | < 300 mV | LiPo end-of-discharge is ~3.0V, need 3.3V out |
| Quiescent current | < 10 uA | Battery life in sleep/standby |
| PSRR | > 60 dB at 1 kHz | Noise rejection for analog recording |
| Package | SOT-23 or smaller | Board space constraint |
| Input range | 3.0V -- 4.2V | Full LiPo voltage range |

Suggested parts: TPS7A02 (TI, 200 mA, 0.75 uA Iq, SOT-23), AP2112K (Diodes Inc, 600 mA, 55 uA Iq, SOT-23).

### Battery

| Battery | Capacity | Weight | Dimensions | Runtime (est.) |
|---------|----------|--------|------------|----------------|
| 30 mAh LiPo | 30 mAh | ~1.0 g | 10x10x3 mm | 2.6 -- 3.5 hrs |
| 60 mAh LiPo | 60 mAh | ~1.5 g | 12x12x3 mm | 5.2 -- 7.1 hrs |

Target: 60 mAh cell for a full behavioral session (5-7 hours). See `power_budget.md` for detailed current breakdown.

Total headstage weight target: < 3 g (PCB + battery + connector + enclosure).

## 6. BLE Antenna

### Option A: PCB trace antenna (recommended for headstage)

- Use Nordic's reference PCB antenna design (inverted-F or meander line).
- The antenna occupies one corner/edge of the board.
- **Keepout zone**: No ground plane, traces, or components within 3 mm of the antenna trace on the same layer and the layer directly below. The ground plane must be cut back in this region.
- Antenna feed impedance: 50 ohm matched.
- Place a pi-network matching circuit (2 caps + 1 inductor, 0201) between the nRF5340 ANT pin and the antenna trace for tuning.

### Option B: Chip antenna

- If board space allows, a chip antenna (e.g., Johanson 2450AT, 3.2x1.6 mm) is easier to tune.
- Still requires ground plane clearance per the antenna datasheet.

### Antenna placement rules

- Antenna at the board edge, as far as possible from the Intan and electrode connector.
- No ground pour under or adjacent to the antenna.
- No traces crossing the antenna keepout zone.
- The BLE range requirement is ~10 m (lab environment). 0 dBm TX power is sufficient.

## 7. Crystal Placement

### 32 MHz HFXO

- Required for the nRF5340 radio and accurate CPU clock.
- Place the crystal within 3 mm of the XC1/XC2 pins.
- Load capacitors (typically 12 pF, check crystal datasheet): place adjacent to crystal pads.
- Ground plane underneath the crystal -- no signal routing under or near the crystal.
- Guard ring (ground pour ring) around the crystal footprint.
- Use a 4-pin crystal package with grounded case if available.

### 32.768 kHz LFXO (optional)

- Provides accurate low-frequency clock for RTC and Embassy tick timer.
- If used, place within 5 mm of P0.14/P0.15 (LFXO pins). Note: this conflicts with current SPI pin assignments; see `pinout.md`.
- Load capacitors per crystal datasheet (typically 6.8 -- 15 pF).
- If LFXO is not populated, the firmware falls back to LFRC (internal RC oscillator, +/-250 ppm). This is acceptable for spike sorting timing but may cause BLE connection interval drift.

## 8. Test Points

Provide accessible test points for board bring-up. Minimum set:

| Test Point | Signal | Probe Type |
|------------|--------|------------|
| TP1 | VDD (1.8V) | Voltage |
| TP2 | VDDH (3.3V) | Voltage |
| TP3 | GND | Ground clip |
| TP4 | SPI SCK (P0.13) | Logic analyzer |
| TP5 | SPI MISO (P0.15) | Logic analyzer |
| TP6 | SPI CS (P0.16) | Logic analyzer |
| TP7 | SWDIO | SWD programmer |
| TP8 | SWDCLK | SWD programmer |
| TP9 | RESET | Manual reset |
| TP10 | VBAT (battery voltage) | Battery monitoring |

Use 0.5 mm pads or tag-connect footprint for SWD to minimize board area.

## 9. Electrode Connector

- The Intan RHD2132 has 32 input channels. Use a fine-pitch connector compatible with the probe/electrode array.
- Common options: Omnetics nano strip (A79 series, 36-pin), or flex cable to probe.
- Place the connector on the board edge closest to the Intan IC.
- Route electrode traces on the top layer with ground guard traces, no vias.
- Keep electrode traces < 5 mm long from connector pad to Intan input pin.

## 10. Mechanical Considerations

| Parameter | Target |
|-----------|--------|
| Board dimensions | 12 mm x 12 mm (max 15 mm x 15 mm) |
| Board thickness | 0.8 mm (standard 1.6 mm is too heavy) |
| Weight (bare PCB) | < 0.5 g |
| Weight (assembled, no battery) | < 1.0 g |
| Weight (with 60 mAh battery) | < 2.5 g |
| Total with enclosure | < 3.0 g |
| Mounting | Headstage clamp or dental cement anchor |

- Use 0201 passives wherever possible to save board area.
- Consider placing passives on both sides of the board to fit within 15x15 mm.
- Conformal coat the assembled board for moisture protection (mouse behavioral chambers are humid).
- No sharp edges or exposed components that could injure the animal.

## 11. Design Review Checklist

- [ ] Continuous ground plane on Layer 2 under all SPI traces
- [ ] Ground plane clearance under BLE antenna (>= 3 mm)
- [ ] All nRF5340 DEC pins have 100 nF caps within 2 mm
- [ ] All VDD/VDDH pins have 100 nF local decoupling
- [ ] VDDH has 10 uF bulk cap
- [ ] Analog/digital ground connected at single point near Intan
- [ ] 32 MHz crystal within 3 mm of XC1/XC2 with load caps
- [ ] SWD pins accessible (test points or tag-connect footprint)
- [ ] SPI test points accessible for logic analyzer
- [ ] Antenna at board edge, no traces in keepout zone
- [ ] Electrode connector close to Intan, short trace runs
- [ ] LDO output decoupled with 10 uF + 100 nF
- [ ] No digital signal routing over analog ground region
- [ ] LFXO/SPI pin conflict resolved (see pinout.md)
- [ ] Board thickness 0.8 mm specified in fab notes
