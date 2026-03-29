//! Watchdog configuration and state tracking for crash recovery.
//!
//! Provides a thin abstraction over the nRF5340 watchdog timer. The actual
//! hardware WDT is configured at startup; this module manages the
//! configuration parameters and tracks pet timing for staleness detection.

/// Configuration for the hardware watchdog timer.
pub struct WatchdogConfig {
    /// Watchdog timeout in milliseconds.
    pub timeout_ms: u32,
    /// Whether to pause the WDT during CPU sleep.
    pub pause_on_sleep: bool,
    /// Whether to pause the WDT during debug halt.
    pub pause_on_debug: bool,
}

impl WatchdogConfig {
    /// Create a new configuration with defaults:
    /// - timeout: 2000 ms
    /// - pause on sleep: true
    /// - pause on debug: true
    pub fn new() -> Self {
        Self {
            timeout_ms: 2000,
            pause_on_sleep: true,
            pause_on_debug: true,
        }
    }

    /// Set the watchdog timeout in milliseconds (builder pattern).
    pub fn with_timeout(mut self, ms: u32) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set whether to pause the WDT during CPU sleep (builder pattern).
    pub fn with_pause_on_sleep(mut self, pause: bool) -> Self {
        self.pause_on_sleep = pause;
        self
    }

    /// Set whether to pause the WDT during debug halt (builder pattern).
    pub fn with_pause_on_debug(mut self, pause: bool) -> Self {
        self.pause_on_debug = pause;
        self
    }
}

/// Runtime state for watchdog pet tracking.
pub struct WatchdogState {
    /// Last sample index when the watchdog was pet.
    last_pet_sample: u32,
    /// Total number of times the watchdog has been pet.
    pet_count: u32,
    /// The watchdog configuration.
    pub config: WatchdogConfig,
}

impl WatchdogState {
    /// Create a new watchdog state with the given configuration.
    pub fn new(config: WatchdogConfig) -> Self {
        Self {
            last_pet_sample: 0,
            pet_count: 0,
            config,
        }
    }

    /// Pet the watchdog at the given sample index.
    pub fn pet(&mut self, current_sample: u32) {
        self.last_pet_sample = current_sample;
        self.pet_count = self.pet_count.wrapping_add(1);
    }

    /// Check if the watchdog is stale (hasn't been pet within the timeout).
    ///
    /// Computes elapsed time as `(current_sample - last_pet_sample) / sample_rate * 1000`
    /// and returns `true` if it exceeds `config.timeout_ms`.
    ///
    /// Uses wrapping subtraction to handle sample counter wrap-around.
    pub fn is_stale(&self, current_sample: u32, sample_rate: u32) -> bool {
        if sample_rate == 0 {
            return true;
        }
        let elapsed_samples = current_sample.wrapping_sub(self.last_pet_sample);
        // Compute elapsed_ms = elapsed_samples * 1000 / sample_rate
        // Use u64 to avoid overflow in the multiplication.
        let elapsed_ms = (elapsed_samples as u64 * 1000) / sample_rate as u64;
        elapsed_ms > self.config.timeout_ms as u64
    }

    /// Return the total pet count.
    pub fn pet_count(&self) -> u32 {
        self.pet_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_config_defaults() {
        let cfg = WatchdogConfig::new();
        assert_eq!(cfg.timeout_ms, 2000);
        assert!(cfg.pause_on_sleep);
        assert!(cfg.pause_on_debug);
    }

    #[test]
    fn watchdog_config_builder() {
        let cfg = WatchdogConfig::new()
            .with_timeout(5000)
            .with_pause_on_sleep(false)
            .with_pause_on_debug(false);
        assert_eq!(cfg.timeout_ms, 5000);
        assert!(!cfg.pause_on_sleep);
        assert!(!cfg.pause_on_debug);
    }

    #[test]
    fn watchdog_pet_updates_state() {
        let cfg = WatchdogConfig::new();
        let mut wd = WatchdogState::new(cfg);
        assert_eq!(wd.pet_count(), 0);

        wd.pet(1000);
        assert_eq!(wd.pet_count(), 1);
        assert_eq!(wd.last_pet_sample, 1000);

        wd.pet(2000);
        assert_eq!(wd.pet_count(), 2);
        assert_eq!(wd.last_pet_sample, 2000);
    }

    #[test]
    fn watchdog_stale_detection() {
        let cfg = WatchdogConfig::new().with_timeout(2000);
        let mut wd = WatchdogState::new(cfg);

        let sample_rate = 30_000; // 30 kHz
        wd.pet(0);

        // After 1 second (30000 samples) -- not stale (1000ms < 2000ms)
        assert!(!wd.is_stale(30_000, sample_rate));

        // After 2 seconds (60000 samples) -- not stale (2000ms == 2000ms, need >)
        assert!(!wd.is_stale(60_000, sample_rate));

        // After 2.1 seconds (63000 samples) -- stale (2100ms > 2000ms)
        assert!(wd.is_stale(63_000, sample_rate));

        // Pet again, no longer stale
        wd.pet(63_000);
        assert!(!wd.is_stale(63_001, sample_rate));
    }
}
