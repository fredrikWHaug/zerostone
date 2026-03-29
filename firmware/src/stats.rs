//! Runtime statistics for firmware health monitoring over BLE.

/// Lightweight statistics tracker for monitoring firmware health.
///
/// All fields are public for direct inspection. Use the provided methods
/// for atomic updates that maintain invariants (e.g., peak tracking).
pub struct RuntimeStats {
    /// Total ADC frames processed.
    pub total_frames: u32,
    /// Total spikes detected.
    pub total_spikes: u32,
    /// Spikes with cluster_id != 0.
    pub total_classified: u32,
    /// Frames dropped due to full channel.
    pub dropped_frames: u32,
    /// Events dropped due to full event queue.
    pub dropped_events: u32,
    /// Seconds since boot.
    pub uptime_seconds: u32,
    /// Highest spikes/second observed.
    pub peak_spike_rate: u16,
    /// Spikes detected in the current second.
    pub current_spike_rate: u16,
}

impl RuntimeStats {
    /// Create a new stats tracker with all fields zeroed.
    pub fn new() -> Self {
        Self {
            total_frames: 0,
            total_spikes: 0,
            total_classified: 0,
            dropped_frames: 0,
            dropped_events: 0,
            uptime_seconds: 0,
            peak_spike_rate: 0,
            current_spike_rate: 0,
        }
    }

    /// Record one ADC frame processed.
    pub fn record_frame(&mut self) {
        self.total_frames = self.total_frames.wrapping_add(1);
    }

    /// Record a detected spike. If `classified` is true, also count it
    /// as a classified spike (cluster_id != 0).
    pub fn record_spike(&mut self, classified: bool) {
        self.total_spikes = self.total_spikes.wrapping_add(1);
        if classified {
            self.total_classified = self.total_classified.wrapping_add(1);
        }
        self.current_spike_rate = self.current_spike_rate.saturating_add(1);
    }

    /// Record a dropped frame (channel full).
    pub fn record_dropped_frame(&mut self) {
        self.dropped_frames = self.dropped_frames.wrapping_add(1);
    }

    /// Record a dropped event (event queue full).
    pub fn record_dropped_event(&mut self) {
        self.dropped_events = self.dropped_events.wrapping_add(1);
    }

    /// Called once per second. Updates uptime, tracks peak spike rate,
    /// and resets the current-second counter.
    pub fn tick_second(&mut self) {
        self.uptime_seconds = self.uptime_seconds.wrapping_add(1);
        if self.current_spike_rate > self.peak_spike_rate {
            self.peak_spike_rate = self.current_spike_rate;
        }
        self.current_spike_rate = 0;
    }

    /// Pack statistics into a 16-byte little-endian buffer for BLE transmission.
    ///
    /// Layout:
    /// - bytes 0..4: total_frames (u32 LE)
    /// - bytes 4..8: total_spikes (u32 LE)
    /// - bytes 8..10: dropped_frames (u16 LE, saturating)
    /// - bytes 10..12: dropped_events (u16 LE, saturating)
    /// - bytes 12..14: peak_spike_rate (u16 LE)
    /// - bytes 14..16: uptime_seconds (u16 LE, saturating)
    ///
    /// Returns the number of bytes written (always 16).
    pub fn serialize(&self, buf: &mut [u8; 16]) -> usize {
        let frames = self.total_frames.to_le_bytes();
        let spikes = self.total_spikes.to_le_bytes();
        let dropped_f = saturate_u32_to_u16(self.dropped_frames).to_le_bytes();
        let dropped_e = saturate_u32_to_u16(self.dropped_events).to_le_bytes();
        let peak = self.peak_spike_rate.to_le_bytes();
        let uptime = saturate_u32_to_u16(self.uptime_seconds).to_le_bytes();

        buf[0] = frames[0];
        buf[1] = frames[1];
        buf[2] = frames[2];
        buf[3] = frames[3];
        buf[4] = spikes[0];
        buf[5] = spikes[1];
        buf[6] = spikes[2];
        buf[7] = spikes[3];
        buf[8] = dropped_f[0];
        buf[9] = dropped_f[1];
        buf[10] = dropped_e[0];
        buf[11] = dropped_e[1];
        buf[12] = peak[0];
        buf[13] = peak[1];
        buf[14] = uptime[0];
        buf[15] = uptime[1];

        16
    }
}

/// Saturate a u32 value to u16::MAX if it exceeds the u16 range.
fn saturate_u32_to_u16(v: u32) -> u16 {
    if v > u16::MAX as u32 {
        u16::MAX
    } else {
        v as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_new_all_zeros() {
        let s = RuntimeStats::new();
        assert_eq!(s.total_frames, 0);
        assert_eq!(s.total_spikes, 0);
        assert_eq!(s.total_classified, 0);
        assert_eq!(s.dropped_frames, 0);
        assert_eq!(s.dropped_events, 0);
        assert_eq!(s.uptime_seconds, 0);
        assert_eq!(s.peak_spike_rate, 0);
        assert_eq!(s.current_spike_rate, 0);
    }

    #[test]
    fn stats_record_frame_increments() {
        let mut s = RuntimeStats::new();
        s.record_frame();
        s.record_frame();
        s.record_frame();
        assert_eq!(s.total_frames, 3);
    }

    #[test]
    fn stats_record_spike_classified_and_unclassified() {
        let mut s = RuntimeStats::new();
        s.record_spike(true);
        s.record_spike(false);
        s.record_spike(true);
        assert_eq!(s.total_spikes, 3);
        assert_eq!(s.total_classified, 2);
        assert_eq!(s.current_spike_rate, 3);
    }

    #[test]
    fn stats_tick_second_updates_peak() {
        let mut s = RuntimeStats::new();
        // Record 5 spikes in first second
        for _ in 0..5 {
            s.record_spike(false);
        }
        s.tick_second();
        assert_eq!(s.peak_spike_rate, 5);
        assert_eq!(s.current_spike_rate, 0);
        assert_eq!(s.uptime_seconds, 1);

        // Record 3 spikes in second second -- peak stays at 5
        for _ in 0..3 {
            s.record_spike(false);
        }
        s.tick_second();
        assert_eq!(s.peak_spike_rate, 5);
        assert_eq!(s.uptime_seconds, 2);

        // Record 10 spikes in third second -- peak updates to 10
        for _ in 0..10 {
            s.record_spike(false);
        }
        s.tick_second();
        assert_eq!(s.peak_spike_rate, 10);
    }

    #[test]
    fn stats_serialize_round_trip() {
        let mut s = RuntimeStats::new();
        s.total_frames = 0x01020304;
        s.total_spikes = 0xAABBCCDD;
        s.dropped_frames = 1234;
        s.dropped_events = 5678;
        s.peak_spike_rate = 999;
        s.uptime_seconds = 3600;

        let mut buf = [0u8; 16];
        let n = s.serialize(&mut buf);
        assert_eq!(n, 16);

        assert_eq!(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]), 0x01020304);
        assert_eq!(u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]), 0xAABBCCDD);
        assert_eq!(u16::from_le_bytes([buf[8], buf[9]]), 1234);
        assert_eq!(u16::from_le_bytes([buf[10], buf[11]]), 5678);
        assert_eq!(u16::from_le_bytes([buf[12], buf[13]]), 999);
        assert_eq!(u16::from_le_bytes([buf[14], buf[15]]), 3600);
    }

    #[test]
    fn stats_saturating_dropped() {
        let mut s = RuntimeStats::new();
        s.dropped_frames = 100_000; // > u16::MAX (65535)
        s.dropped_events = 200_000;
        s.uptime_seconds = 70_000;

        let mut buf = [0u8; 16];
        s.serialize(&mut buf);

        assert_eq!(u16::from_le_bytes([buf[8], buf[9]]), u16::MAX);
        assert_eq!(u16::from_le_bytes([buf[10], buf[11]]), u16::MAX);
        assert_eq!(u16::from_le_bytes([buf[14], buf[15]]), u16::MAX);
    }
}
