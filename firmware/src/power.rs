//! Power management configuration and battery life estimation.
//!
//! Provides duty cycle calculations and current draw models for the
//! Zerostone neural recording platform. Embassy handles WFI insertion
//! automatically; this module exists for configuration and power
//! analysis during system design.

/// Operating profile for the nRF5340 application core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerProfile {
    /// 128 MHz, no clock gating, maximum throughput.
    FullSpeed,
    /// 128 MHz with WFI between samples (Embassy default).
    Balanced,
    /// 64 MHz, aggressive WFI, longer BLE connection interval.
    LowPower,
}

impl PowerProfile {
    /// CPU clock frequency in MHz.
    pub fn clock_mhz(self) -> u32 {
        match self {
            Self::FullSpeed => 128,
            Self::Balanced => 128,
            Self::LowPower => 64,
        }
    }

    /// Active current draw in milliamps.
    pub fn active_current_ma(self) -> f32 {
        match self {
            Self::FullSpeed => 3.2,
            Self::Balanced => 3.2,
            Self::LowPower => 2.0,
        }
    }

    /// Idle (WFI) current draw in milliamps.
    pub fn idle_current_ma(self) -> f32 {
        match self {
            Self::FullSpeed => 3.2, // no WFI, always active
            Self::Balanced => 1.1,
            Self::LowPower => 1.1,
        }
    }
}

/// Power configuration for the recording system.
pub struct PowerConfig {
    /// Operating profile.
    pub profile: PowerProfile,
    /// BLE connection interval in milliseconds (7.5 to 4000).
    pub ble_connection_interval_ms: u16,
    /// BLE transmit power in dBm (-40 to +8).
    pub ble_tx_power_dbm: i8,
    /// ADC sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Estimated processing time per sample in microseconds.
    pub processing_us: u32,
}

impl PowerConfig {
    /// Create a new configuration with Balanced defaults.
    pub fn new() -> Self {
        Self {
            profile: PowerProfile::Balanced,
            ble_connection_interval_ms: 50,
            ble_tx_power_dbm: 0,
            sample_rate_hz: 30_000,
            processing_us: 19,
        }
    }

    /// Set the power profile.
    pub fn with_profile(mut self, profile: PowerProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the BLE connection interval in milliseconds.
    pub fn with_ble_connection_interval_ms(mut self, ms: u16) -> Self {
        self.ble_connection_interval_ms = ms;
        self
    }

    /// Set the BLE transmit power in dBm.
    pub fn with_ble_tx_power_dbm(mut self, dbm: i8) -> Self {
        self.ble_tx_power_dbm = dbm;
        self
    }

    /// Set the ADC sample rate in Hz.
    pub fn with_sample_rate_hz(mut self, hz: u32) -> Self {
        self.sample_rate_hz = hz;
        self
    }

    /// Set the estimated processing time per sample in microseconds.
    pub fn with_processing_us(mut self, us: u32) -> Self {
        self.processing_us = us;
        self
    }

    /// Fraction of time the CPU is active, in `[0.0, 1.0]`.
    ///
    /// Computed as `processing_us / sample_period_us`.
    pub fn duty_cycle(&self) -> f32 {
        let sample_period_us = 1_000_000.0 / self.sample_rate_hz as f32;
        let dc = self.processing_us as f32 / sample_period_us;
        if dc > 1.0 {
            1.0
        } else {
            dc
        }
    }

    /// Average MCU current draw in milliamps (excludes BLE and peripherals).
    pub fn average_current_ma(&self) -> f32 {
        let dc = self.duty_cycle();
        let active = self.profile.active_current_ma();
        let idle = self.profile.idle_current_ma();
        dc * active + (1.0 - dc) * idle
    }

    /// Average BLE current draw in milliamps.
    ///
    /// Rough model: 4.6 mA active for ~2 ms every connection interval,
    /// plus 0.002 mA quiescent.
    pub fn ble_average_current_ma(&self) -> f32 {
        let active_ma = 4.6_f32;
        let active_ms = 2.0_f32;
        let interval_ms = self.ble_connection_interval_ms as f32;
        let duty = active_ms / interval_ms;
        duty * active_ma + 0.002
    }

    /// Intan RHD2132 current draw in milliamps (constant during recording).
    pub fn intan_current_ma(&self) -> f32 {
        7.0
    }

    /// Total system current draw in milliamps.
    ///
    /// Includes MCU average, BLE average, Intan, and LDO quiescent (0.01 mA).
    pub fn total_system_current_ma(&self) -> f32 {
        self.average_current_ma()
            + self.ble_average_current_ma()
            + self.intan_current_ma()
            + 0.01
    }

    /// MCU-only battery life in hours.
    pub fn battery_life_hours(&self, capacity_mah: f32) -> f32 {
        capacity_mah / self.average_current_ma()
    }

    /// Full system battery life in hours.
    pub fn system_battery_life_hours(&self, capacity_mah: f32) -> f32 {
        capacity_mah / self.total_system_current_ma()
    }
}

/// Battery size estimate with predicted life.
#[derive(Debug, Clone, Copy)]
pub struct BatteryEstimate {
    /// Battery capacity in milliamp-hours.
    pub capacity_mah: f32,
    /// Battery weight in grams.
    pub weight_grams: f32,
    /// Predicted system battery life in hours.
    pub life_hours: f32,
}

/// Return battery life estimates for four standard LiPo cell sizes.
///
/// Sizes: 30 mAh (1 g), 60 mAh (1.5 g), 100 mAh (2.5 g), 150 mAh (3 g).
pub fn standard_battery_estimates(config: &PowerConfig) -> [BatteryEstimate; 4] {
    const CELLS: [(f32, f32); 4] = [
        (30.0, 1.0),
        (60.0, 1.5),
        (100.0, 2.5),
        (150.0, 3.0),
    ];
    let mut out = [BatteryEstimate {
        capacity_mah: 0.0,
        weight_grams: 0.0,
        life_hours: 0.0,
    }; 4];
    let mut i = 0;
    while i < 4 {
        let (cap, wt) = CELLS[i];
        out[i] = BatteryEstimate {
            capacity_mah: cap,
            weight_grams: wt,
            life_hours: config.system_battery_life_hours(cap),
        };
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let c = PowerConfig::new();
        assert_eq!(c.profile, PowerProfile::Balanced);
        assert_eq!(c.ble_connection_interval_ms, 50);
        assert_eq!(c.ble_tx_power_dbm, 0);
        assert_eq!(c.sample_rate_hz, 30_000);
        assert_eq!(c.processing_us, 19);
    }

    #[test]
    fn config_builder() {
        let c = PowerConfig::new()
            .with_profile(PowerProfile::LowPower)
            .with_ble_connection_interval_ms(100)
            .with_ble_tx_power_dbm(-20)
            .with_sample_rate_hz(20_000)
            .with_processing_us(25);
        assert_eq!(c.profile, PowerProfile::LowPower);
        assert_eq!(c.ble_connection_interval_ms, 100);
        assert_eq!(c.ble_tx_power_dbm, -20);
        assert_eq!(c.sample_rate_hz, 20_000);
        assert_eq!(c.processing_us, 25);
    }

    #[test]
    fn duty_cycle_30khz() {
        let c = PowerConfig::new(); // 19us processing, 30kHz
        let dc = c.duty_cycle();
        // sample period = 33.33us, duty = 19/33.33 = 0.57
        assert!((dc - 0.57).abs() < 0.01, "duty_cycle = {dc}");
    }

    #[test]
    fn duty_cycle_full_speed() {
        // 33us processing at 30kHz = 33/33.33 ~ 99%, but 34us should clamp to 1.0
        let c = PowerConfig::new().with_processing_us(34);
        let dc = c.duty_cycle();
        assert!((dc - 1.0).abs() < 0.001, "duty_cycle = {dc}");
    }

    #[test]
    fn average_current_balanced() {
        let c = PowerConfig::new();
        let dc = c.duty_cycle();
        let expected = dc * 3.2 + (1.0 - dc) * 1.1;
        let actual = c.average_current_ma();
        assert!(
            (actual - expected).abs() < 0.001,
            "avg current: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn battery_life_60mah() {
        let c = PowerConfig::new();
        let life = c.system_battery_life_hours(60.0);
        // total ~9-10 mA system draw -> ~6-7 hours
        assert!(life > 4.0 && life < 10.0, "life = {life} hours");
    }

    #[test]
    fn ble_current_scales_with_interval() {
        let short = PowerConfig::new().with_ble_connection_interval_ms(10);
        let long = PowerConfig::new().with_ble_connection_interval_ms(500);
        assert!(
            short.ble_average_current_ma() > long.ble_average_current_ma(),
            "short CI = {} > long CI = {}",
            short.ble_average_current_ma(),
            long.ble_average_current_ma()
        );
    }

    #[test]
    fn total_system_includes_intan() {
        let c = PowerConfig::new();
        let total = c.total_system_current_ma();
        // Must include at least 7.0 mA from Intan
        assert!(total > 7.0, "total = {total}");
        // Check decomposition adds up
        let expected = c.average_current_ma()
            + c.ble_average_current_ma()
            + c.intan_current_ma()
            + 0.01;
        assert!(
            (total - expected).abs() < 0.001,
            "total {total} != sum {expected}"
        );
    }

    #[test]
    fn low_power_profile() {
        let p = PowerProfile::LowPower;
        assert_eq!(p.clock_mhz(), 64);
        assert!((p.active_current_ma() - 2.0).abs() < 0.001);
        assert!((p.idle_current_ma() - 1.1).abs() < 0.001);
    }

    #[test]
    fn standard_battery_estimates_ordered() {
        let c = PowerConfig::new();
        let ests = standard_battery_estimates(&c);
        for i in 1..ests.len() {
            assert!(
                ests[i].life_hours > ests[i - 1].life_hours,
                "est[{}].life = {} should be > est[{}].life = {}",
                i,
                ests[i].life_hours,
                i - 1,
                ests[i - 1].life_hours
            );
        }
    }

    mod proptest_properties {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn total_system_current_gte_intan(
                sample_rate in 1u32..=100_000,
                processing_us in 0u32..=1_000_000,
                ble_interval in 8u16..=4000,
            ) {
                let c = PowerConfig::new()
                    .with_sample_rate_hz(sample_rate)
                    .with_processing_us(processing_us)
                    .with_ble_connection_interval_ms(ble_interval);
                let total = c.total_system_current_ma();
                let intan = c.intan_current_ma();
                prop_assert!(
                    total >= intan,
                    "total {} < intan {}",
                    total, intan
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kani proofs
// ---------------------------------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn duty_cycle_in_range() {
        let sample_rate: u32 = kani::any();
        kani::assume(sample_rate > 0 && sample_rate <= 100_000);
        let processing_us: u32 = kani::any();
        kani::assume(processing_us <= 1_000_000);

        let c = PowerConfig::new()
            .with_sample_rate_hz(sample_rate)
            .with_processing_us(processing_us);
        let dc = c.duty_cycle();
        assert!(dc >= 0.0, "duty_cycle {} < 0.0", dc);
        assert!(dc <= 1.0, "duty_cycle {} > 1.0", dc);
    }
}
