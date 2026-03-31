//! Stimulation output driver for closed-loop electrophysiology.
//!
//! When the spike sorter identifies a target neuron (specific cluster ID),
//! the system triggers a stimulation pulse on a GPIO pin. This module
//! handles the decision logic: refractory period enforcement, rate
//! limiting, and safety checks. The actual GPIO toggling is left to the
//! caller so this module remains hardware-agnostic and testable on host.

use crate::pipeline::SpikeEvent;

// ---------------------------------------------------------------------------
// StimConfig
// ---------------------------------------------------------------------------

/// Configuration for the stimulation driver.
#[derive(Clone, Copy, Debug)]
pub struct StimConfig {
    /// Which cluster ID triggers stimulation. 0 = disabled.
    pub target_cluster_id: u8,
    /// Stimulation pulse width in microseconds.
    pub pulse_width_us: u32,
    /// Minimum interval between pulses in milliseconds.
    pub refractory_ms: u32,
    /// Maximum stimulation rate in Hz (safety limit).
    pub max_rate_hz: u16,
    /// Master enable switch.
    pub enabled: bool,
}

impl StimConfig {
    /// Creates a new configuration with sensible defaults.
    pub const fn new() -> Self {
        Self {
            target_cluster_id: 0,
            pulse_width_us: 200,
            refractory_ms: 5,
            max_rate_hz: 100,
            enabled: false,
        }
    }

    /// Set the target cluster ID.
    pub const fn with_target_cluster_id(mut self, id: u8) -> Self {
        self.target_cluster_id = id;
        self
    }

    /// Set the pulse width in microseconds.
    pub const fn with_pulse_width_us(mut self, us: u32) -> Self {
        self.pulse_width_us = us;
        self
    }

    /// Set the refractory period in milliseconds.
    pub const fn with_refractory_ms(mut self, ms: u32) -> Self {
        self.refractory_ms = ms;
        self
    }

    /// Set the maximum stimulation rate in Hz.
    pub const fn with_max_rate_hz(mut self, hz: u16) -> Self {
        self.max_rate_hz = hz;
        self
    }

    /// Set the master enable switch.
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// StimDecision / RejectReason
// ---------------------------------------------------------------------------

/// Decision returned by [`StimState::evaluate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StimDecision {
    /// Fire stimulation with the given pulse width.
    Trigger {
        /// Pulse width in microseconds.
        pulse_width_us: u32,
    },
    /// Stimulation was rejected for the given reason.
    Reject(RejectReason),
}

/// Why a spike event did not trigger stimulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Stimulation is disabled (master switch off or target_cluster_id == 0).
    Disabled,
    /// The spike's cluster_id does not match the target.
    WrongCluster,
    /// Too soon after the last trigger (refractory period).
    Refractory,
    /// Maximum stimulation rate for this second exceeded.
    RateLimit,
}

// ---------------------------------------------------------------------------
// StimStats
// ---------------------------------------------------------------------------

/// Snapshot of stimulation statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StimStats {
    /// Total triggers since boot.
    pub trigger_count: u32,
    /// Spikes rejected due to refractory period.
    pub refractory_rejects: u32,
    /// Spikes rejected due to wrong cluster ID.
    pub cluster_rejects: u32,
    /// Spikes rejected due to rate limit.
    pub rate_rejects: u32,
}

// ---------------------------------------------------------------------------
// StimState
// ---------------------------------------------------------------------------

/// Run-time state for the closed-loop stimulation driver.
pub struct StimState {
    /// Active configuration.
    pub config: StimConfig,
    /// Sample index of the last trigger.
    pub last_trigger_sample: u32,
    /// Total triggers since boot.
    pub trigger_count: u32,
    /// Spikes rejected due to refractory period.
    pub refractory_rejects: u32,
    /// Spikes rejected due to wrong cluster ID.
    pub cluster_rejects: u32,
    /// Spikes rejected due to rate limit.
    pub rate_rejects: u32,
    /// Consecutive triggers without a non-target spike.
    pub consecutive_triggers: u16,
    /// Triggers in the current 1-second window.
    pub current_second_triggers: u16,
    /// Sample index at the start of the current rate window.
    pub current_second_start: u32,
}

impl StimState {
    /// Creates a new stimulation state with the given configuration.
    pub const fn new(config: StimConfig) -> Self {
        Self {
            config,
            last_trigger_sample: 0,
            trigger_count: 0,
            refractory_rejects: 0,
            cluster_rejects: 0,
            rate_rejects: 0,
            consecutive_triggers: 0,
            current_second_triggers: 0,
            current_second_start: 0,
        }
    }

    /// Evaluate whether a spike event should trigger stimulation.
    ///
    /// The decision pipeline:
    /// 1. Check master enable and that target_cluster_id != 0.
    /// 2. Check cluster match.
    /// 3. Check refractory period.
    /// 4. Check rate limit.
    /// 5. If all pass, return [`StimDecision::Trigger`] and update state.
    pub fn evaluate(&mut self, event: &SpikeEvent, sample_rate: u32) -> StimDecision {
        // 1. Master enable + valid target.
        if !self.config.enabled || self.config.target_cluster_id == 0 {
            return StimDecision::Reject(RejectReason::Disabled);
        }

        // 2. Cluster match.
        if event.cluster_id != self.config.target_cluster_id {
            self.cluster_rejects += 1;
            self.consecutive_triggers = 0;
            return StimDecision::Reject(RejectReason::WrongCluster);
        }

        // 3. Refractory period.
        // Only enforce if we have triggered before (trigger_count > 0).
        if self.trigger_count > 0 {
            let elapsed_samples = event.sample_idx.wrapping_sub(self.last_trigger_sample);
            // Convert refractory_ms to samples: refractory_ms * sample_rate / 1000
            let refractory_samples = (self.config.refractory_ms as u64)
                .wrapping_mul(sample_rate as u64)
                / 1000;
            if (elapsed_samples as u64) < refractory_samples {
                self.refractory_rejects += 1;
                return StimDecision::Reject(RejectReason::Refractory);
            }
        }

        // 4. Rate limit.
        if self.current_second_triggers >= self.config.max_rate_hz {
            self.rate_rejects += 1;
            return StimDecision::Reject(RejectReason::RateLimit);
        }

        // All checks passed -- trigger.
        self.last_trigger_sample = event.sample_idx;
        self.trigger_count += 1;
        self.consecutive_triggers += 1;
        self.current_second_triggers += 1;

        StimDecision::Trigger {
            pulse_width_us: self.config.pulse_width_us,
        }
    }

    /// Reset the per-second rate window.
    ///
    /// Call this once per second (or when the sample counter crosses a
    /// 1-second boundary) to allow the next batch of stimulations.
    pub fn tick_second(&mut self, current_sample: u32) {
        self.current_second_start = current_sample;
        self.current_second_triggers = 0;
    }

    /// Return a snapshot of cumulative statistics.
    pub fn stats(&self) -> StimStats {
        StimStats {
            trigger_count: self.trigger_count,
            refractory_rejects: self.refractory_rejects,
            cluster_rejects: self.cluster_rejects,
            rate_rejects: self.rate_rejects,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a SpikeEvent with the given cluster and sample index.
    fn make_event(cluster_id: u8, sample_idx: u32) -> SpikeEvent {
        SpikeEvent {
            sample_idx,
            channel: 0,
            cluster_id,
            amplitude: -0.5,
        }
    }

    const SAMPLE_RATE: u32 = 30_000;

    #[test]
    fn config_defaults() {
        let cfg = StimConfig::new();
        assert_eq!(cfg.target_cluster_id, 0);
        assert_eq!(cfg.pulse_width_us, 200);
        assert_eq!(cfg.refractory_ms, 5);
        assert_eq!(cfg.max_rate_hz, 100);
        assert!(!cfg.enabled);
    }

    #[test]
    fn config_builder() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(3)
            .with_pulse_width_us(500)
            .with_refractory_ms(10)
            .with_max_rate_hz(50)
            .with_enabled(true);

        assert_eq!(cfg.target_cluster_id, 3);
        assert_eq!(cfg.pulse_width_us, 500);
        assert_eq!(cfg.refractory_ms, 10);
        assert_eq!(cfg.max_rate_hz, 50);
        assert!(cfg.enabled);
    }

    #[test]
    fn trigger_on_matching_cluster() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(2)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        let event = make_event(2, 1000);
        let decision = state.evaluate(&event, SAMPLE_RATE);

        assert_eq!(
            decision,
            StimDecision::Trigger { pulse_width_us: 200 }
        );
        assert_eq!(state.trigger_count, 1);
    }

    #[test]
    fn reject_wrong_cluster() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(2)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        let event = make_event(5, 1000);
        let decision = state.evaluate(&event, SAMPLE_RATE);

        assert_eq!(decision, StimDecision::Reject(RejectReason::WrongCluster));
        assert_eq!(state.cluster_rejects, 1);
        assert_eq!(state.trigger_count, 0);
    }

    #[test]
    fn reject_disabled() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(2)
            .with_enabled(false);
        let mut state = StimState::new(cfg);

        let event = make_event(2, 1000);
        let decision = state.evaluate(&event, SAMPLE_RATE);

        assert_eq!(decision, StimDecision::Reject(RejectReason::Disabled));
    }

    #[test]
    fn reject_refractory() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(5)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // First trigger succeeds.
        let ev1 = make_event(1, 0);
        assert_eq!(
            state.evaluate(&ev1, SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );

        // Second trigger 2 ms later (60 samples at 30 kHz) -- within refractory.
        let ev2 = make_event(1, 60);
        assert_eq!(
            state.evaluate(&ev2, SAMPLE_RATE),
            StimDecision::Reject(RejectReason::Refractory)
        );
        assert_eq!(state.refractory_rejects, 1);
    }

    #[test]
    fn refractory_allows_after_timeout() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(5)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // First trigger at sample 0.
        let ev1 = make_event(1, 0);
        assert_eq!(
            state.evaluate(&ev1, SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );

        // 5 ms later = 150 samples at 30 kHz. Should be allowed.
        let ev2 = make_event(1, 150);
        assert_eq!(
            state.evaluate(&ev2, SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );
        assert_eq!(state.trigger_count, 2);
    }

    #[test]
    fn rate_limit_enforced() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(0) // no refractory so we can hit rate limit
            .with_max_rate_hz(3)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // Fire 3 triggers (the limit).
        for i in 0..3u32 {
            let ev = make_event(1, i * 1000);
            assert_eq!(
                state.evaluate(&ev, SAMPLE_RATE),
                StimDecision::Trigger { pulse_width_us: 200 }
            );
        }

        // 4th should be rate-limited.
        let ev = make_event(1, 5000);
        assert_eq!(
            state.evaluate(&ev, SAMPLE_RATE),
            StimDecision::Reject(RejectReason::RateLimit)
        );
        assert_eq!(state.rate_rejects, 1);
    }

    #[test]
    fn rate_limit_resets_each_second() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(0)
            .with_max_rate_hz(2)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // Use up the limit.
        assert_eq!(
            state.evaluate(&make_event(1, 0), SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );
        assert_eq!(
            state.evaluate(&make_event(1, 100), SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );
        assert_eq!(
            state.evaluate(&make_event(1, 200), SAMPLE_RATE),
            StimDecision::Reject(RejectReason::RateLimit)
        );

        // Tick the second window.
        state.tick_second(SAMPLE_RATE);

        // Should be able to trigger again.
        assert_eq!(
            state.evaluate(&make_event(1, SAMPLE_RATE + 100), SAMPLE_RATE),
            StimDecision::Trigger { pulse_width_us: 200 }
        );
        assert_eq!(state.current_second_triggers, 1);
    }

    #[test]
    fn stats_tracking() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(0)
            .with_max_rate_hz(100)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // One trigger.
        state.evaluate(&make_event(1, 0), SAMPLE_RATE);
        // One wrong cluster.
        state.evaluate(&make_event(5, 100), SAMPLE_RATE);
        // Another trigger.
        state.evaluate(&make_event(1, 200), SAMPLE_RATE);

        let s = state.stats();
        assert_eq!(s.trigger_count, 2);
        assert_eq!(s.cluster_rejects, 1);
        assert_eq!(s.refractory_rejects, 0);
        assert_eq!(s.rate_rejects, 0);
    }

    #[test]
    fn zero_target_cluster_disables() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(0)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        let event = make_event(0, 1000);
        let decision = state.evaluate(&event, SAMPLE_RATE);

        assert_eq!(decision, StimDecision::Reject(RejectReason::Disabled));
    }

    #[test]
    fn consecutive_rapid_spikes() {
        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(2) // 2 ms = 60 samples at 30 kHz
            .with_max_rate_hz(100)
            .with_enabled(true);
        let mut state = StimState::new(cfg);

        // Rapid burst: 5 spikes 1 sample apart. Only the first should trigger.
        let d0 = state.evaluate(&make_event(1, 1000), SAMPLE_RATE);
        let d1 = state.evaluate(&make_event(1, 1001), SAMPLE_RATE);
        let d2 = state.evaluate(&make_event(1, 1002), SAMPLE_RATE);
        let d3 = state.evaluate(&make_event(1, 1003), SAMPLE_RATE);
        let d4 = state.evaluate(&make_event(1, 1004), SAMPLE_RATE);

        assert_eq!(d0, StimDecision::Trigger { pulse_width_us: 200 });
        assert_eq!(d1, StimDecision::Reject(RejectReason::Refractory));
        assert_eq!(d2, StimDecision::Reject(RejectReason::Refractory));
        assert_eq!(d3, StimDecision::Reject(RejectReason::Refractory));
        assert_eq!(d4, StimDecision::Reject(RejectReason::Refractory));

        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.refractory_rejects, 4);
        assert_eq!(state.consecutive_triggers, 1);
    }
}
