//! BLE GATT service stub for Zerostone neural data streaming.
//!
//! Defines the service/characteristic UUIDs, spike event and configuration
//! serialization for BLE notifications. This is a pure data-format module
//! with no BLE radio dependencies — wire it up to the actual stack when
//! hardware arrives.

use crate::pipeline::SpikeEvent;

// ---------------------------------------------------------------------------
// Service and characteristic UUIDs (custom 128-bit)
// ---------------------------------------------------------------------------

/// UUID for the Zerostone neural data service.
///
/// `7a657273-746f-6e65-0001-000000000000`
/// (ASCII "zeRstone" in the first 8 bytes, sub-ID 0001)
pub const NEURAL_SERVICE_UUID: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x65, 0x6e,
    0x6f, 0x74, 0x73, 0x72, 0x65, 0x7a, 0x00, 0x00,
];

/// Spike event notification characteristic UUID.
///
/// `7a657273-746f-6e65-0002-000000000000`
pub const SPIKE_EVENT_CHAR_UUID: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x65, 0x6e,
    0x6f, 0x74, 0x73, 0x72, 0x65, 0x7a, 0x00, 0x00,
];

/// Configuration characteristic UUID.
///
/// `7a657273-746f-6e65-0003-000000000000`
pub const CONFIG_CHAR_UUID: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x65, 0x6e,
    0x6f, 0x74, 0x73, 0x72, 0x65, 0x7a, 0x00, 0x00,
];

// ---------------------------------------------------------------------------
// Spike event serialization (8 bytes)
// ---------------------------------------------------------------------------

/// Scale factor for converting f32 amplitude to i16.
///
/// Maps [-1.0, 1.0] to [-32767, 32767]. Values outside that range are
/// clamped.
const AMP_SCALE: f32 = 32767.0;

/// Serialize a [`SpikeEvent`] into 8 bytes (little-endian).
///
/// Layout:
/// - bytes 0..4: `sample_idx` as u32 LE
/// - byte 4: `channel`
/// - byte 5: `cluster_id`
/// - bytes 6..8: amplitude as i16 LE (scaled from f32, clamped to [-1, 1])
///
/// Returns the number of bytes written (always 8).
pub fn serialize_spike_event(event: &SpikeEvent, buf: &mut [u8; 8]) -> usize {
    let idx_bytes = event.sample_idx.to_le_bytes();
    buf[0] = idx_bytes[0];
    buf[1] = idx_bytes[1];
    buf[2] = idx_bytes[2];
    buf[3] = idx_bytes[3];
    buf[4] = event.channel;
    buf[5] = event.cluster_id;

    // Clamp amplitude to [-1.0, 1.0] then scale to i16.
    let clamped = clamp_f32(event.amplitude, -1.0, 1.0);
    let scaled = (clamped * AMP_SCALE) as i16;
    let amp_bytes = scaled.to_le_bytes();
    buf[6] = amp_bytes[0];
    buf[7] = amp_bytes[1];

    8
}

/// Deserialize a [`SpikeEvent`] from 8 bytes (little-endian).
///
/// Inverse of [`serialize_spike_event`].
pub fn deserialize_spike_event(buf: &[u8; 8]) -> SpikeEvent {
    let sample_idx = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let channel = buf[4];
    let cluster_id = buf[5];
    let raw_amp = i16::from_le_bytes([buf[6], buf[7]]);
    let amplitude = raw_amp as f32 / AMP_SCALE;

    SpikeEvent {
        sample_idx,
        channel,
        cluster_id,
        amplitude,
    }
}

// ---------------------------------------------------------------------------
// Configuration serialization (8 bytes)
// ---------------------------------------------------------------------------

/// Sorter configuration transmitted over BLE.
#[derive(Clone, Copy, Debug)]
pub struct SorterConfig {
    /// MAD multiplier for the detection threshold (e.g., 5.0).
    pub threshold_factor: f32,
    /// Number of templates for template matching.
    pub n_templates: u8,
    /// Sampling rate in Hz (e.g., 30000).
    pub sample_rate_hz: u16,
}

/// Serialize a [`SorterConfig`] into 8 bytes (little-endian).
///
/// Layout:
/// - bytes 0..4: `threshold_factor` as f32 LE
/// - byte 4: `n_templates`
/// - bytes 5..7: `sample_rate_hz` as u16 LE
/// - byte 7: reserved (zero)
pub fn serialize_config(config: &SorterConfig, buf: &mut [u8; 8]) {
    let tf_bytes = config.threshold_factor.to_le_bytes();
    buf[0] = tf_bytes[0];
    buf[1] = tf_bytes[1];
    buf[2] = tf_bytes[2];
    buf[3] = tf_bytes[3];
    buf[4] = config.n_templates;
    let sr_bytes = config.sample_rate_hz.to_le_bytes();
    buf[5] = sr_bytes[0];
    buf[6] = sr_bytes[1];
    buf[7] = 0; // reserved
}

/// Deserialize a [`SorterConfig`] from 8 bytes (little-endian).
///
/// Inverse of [`serialize_config`].
pub fn deserialize_config(buf: &[u8; 8]) -> SorterConfig {
    let threshold_factor = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let n_templates = buf[4];
    let sample_rate_hz = u16::from_le_bytes([buf[5], buf[6]]);

    SorterConfig {
        threshold_factor,
        n_templates,
        sample_rate_hz,
    }
}

// ---------------------------------------------------------------------------
// Batch notification
// ---------------------------------------------------------------------------

/// Serialize multiple [`SpikeEvent`]s into a contiguous buffer.
///
/// Packs up to `max_events` events (8 bytes each) into `buf`. Returns the
/// total number of bytes written. Stops early if `buf` cannot fit another
/// 8-byte event or if all events have been serialized.
///
/// This is MTU-aware: for a typical BLE 4.2 ATT MTU of 247 bytes (minus 3
/// bytes overhead), `max_events` should be `(mtu - 3) / 8`.
pub fn serialize_spike_batch(events: &[SpikeEvent], buf: &mut [u8], max_events: usize) -> usize {
    let count = events.len().min(max_events);
    let mut offset = 0usize;

    for event in events.iter().take(count) {
        if offset + 8 > buf.len() {
            break;
        }
        let mut tmp = [0u8; 8];
        serialize_spike_event(event, &mut tmp);
        buf[offset..offset + 8].copy_from_slice(&tmp);
        offset += 8;
    }

    offset
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Clamp an f32 to [min, max] without libm.
fn clamp_f32(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(sample_idx: u32, channel: u8, cluster_id: u8, amplitude: f32) -> SpikeEvent {
        SpikeEvent {
            sample_idx,
            channel,
            cluster_id,
            amplitude,
        }
    }

    // -- Spike event round-trip ---------------------------------------------

    #[test]
    fn spike_event_round_trip_basic() {
        let ev = make_event(12345, 7, 3, -0.25);
        let mut buf = [0u8; 8];
        let n = serialize_spike_event(&ev, &mut buf);
        assert_eq!(n, 8);

        let out = deserialize_spike_event(&buf);
        assert_eq!(out.sample_idx, 12345);
        assert_eq!(out.channel, 7);
        assert_eq!(out.cluster_id, 3);
        // Amplitude loses precision from f32 -> i16 -> f32.
        assert!((out.amplitude - (-0.25)).abs() < 0.001);
    }

    #[test]
    fn spike_event_round_trip_zero_amplitude() {
        let ev = make_event(0, 0, 0, 0.0);
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let out = deserialize_spike_event(&buf);
        assert_eq!(out.sample_idx, 0);
        assert_eq!(out.channel, 0);
        assert_eq!(out.cluster_id, 0);
        assert!((out.amplitude).abs() < 0.0001);
    }

    #[test]
    fn spike_event_round_trip_positive_amplitude() {
        let ev = make_event(100, 2, 1, 0.75);
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let out = deserialize_spike_event(&buf);
        assert!((out.amplitude - 0.75).abs() < 0.001);
    }

    #[test]
    fn spike_event_max_sample_idx() {
        let ev = make_event(u32::MAX, 255, 255, 0.5);
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let out = deserialize_spike_event(&buf);
        assert_eq!(out.sample_idx, u32::MAX);
        assert_eq!(out.channel, 255);
        assert_eq!(out.cluster_id, 255);
    }

    #[test]
    fn spike_event_amplitude_clamped_positive() {
        // Amplitude > 1.0 should clamp to 1.0.
        let ev = make_event(0, 0, 0, 2.5);
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let out = deserialize_spike_event(&buf);
        assert!((out.amplitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn spike_event_amplitude_clamped_negative() {
        // Amplitude < -1.0 should clamp to -1.0.
        let ev = make_event(0, 0, 0, -3.0);
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let out = deserialize_spike_event(&buf);
        assert!((out.amplitude - (-1.0)).abs() < 0.001);
    }

    // -- Config round-trip --------------------------------------------------

    #[test]
    fn config_round_trip_basic() {
        let cfg = SorterConfig {
            threshold_factor: 5.0,
            n_templates: 4,
            sample_rate_hz: 30000,
        };
        let mut buf = [0u8; 8];
        serialize_config(&cfg, &mut buf);
        let out = deserialize_config(&buf);
        assert!((out.threshold_factor - 5.0).abs() < f32::EPSILON);
        assert_eq!(out.n_templates, 4);
        assert_eq!(out.sample_rate_hz, 30000);
    }

    #[test]
    fn config_round_trip_edge_values() {
        let cfg = SorterConfig {
            threshold_factor: 0.0,
            n_templates: 255,
            sample_rate_hz: u16::MAX,
        };
        let mut buf = [0u8; 8];
        serialize_config(&cfg, &mut buf);
        let out = deserialize_config(&buf);
        assert!((out.threshold_factor).abs() < f32::EPSILON);
        assert_eq!(out.n_templates, 255);
        assert_eq!(out.sample_rate_hz, u16::MAX);
    }

    #[test]
    fn config_reserved_byte_is_zero() {
        let cfg = SorterConfig {
            threshold_factor: 1.0,
            n_templates: 1,
            sample_rate_hz: 20000,
        };
        let mut buf = [0xFFu8; 8];
        serialize_config(&cfg, &mut buf);
        assert_eq!(buf[7], 0, "reserved byte should be zero");
    }

    // -- Batch serialization ------------------------------------------------

    #[test]
    fn batch_serialize_multiple_events() {
        let events = [
            make_event(100, 0, 1, 0.5),
            make_event(200, 1, 2, -0.3),
            make_event(300, 2, 0, 0.1),
        ];
        let mut buf = [0u8; 64];
        let n = serialize_spike_batch(&events, &mut buf, 10);
        assert_eq!(n, 24); // 3 * 8

        // Verify each event can be deserialized.
        for (i, ev) in events.iter().enumerate() {
            let start = i * 8;
            let slice: &[u8; 8] = buf[start..start + 8].try_into().unwrap();
            let out = deserialize_spike_event(slice);
            assert_eq!(out.sample_idx, ev.sample_idx);
            assert_eq!(out.channel, ev.channel);
            assert_eq!(out.cluster_id, ev.cluster_id);
        }
    }

    #[test]
    fn batch_serialize_respects_max_events() {
        let events = [
            make_event(1, 0, 0, 0.0),
            make_event(2, 0, 0, 0.0),
            make_event(3, 0, 0, 0.0),
        ];
        let mut buf = [0u8; 64];
        let n = serialize_spike_batch(&events, &mut buf, 2);
        assert_eq!(n, 16); // only 2 events = 16 bytes
    }

    #[test]
    fn batch_serialize_respects_buffer_size() {
        let events = [
            make_event(1, 0, 0, 0.0),
            make_event(2, 0, 0, 0.0),
            make_event(3, 0, 0, 0.0),
        ];
        // Buffer only fits 2 events (16 bytes), even though max_events is 10.
        let mut buf = [0u8; 16];
        let n = serialize_spike_batch(&events, &mut buf, 10);
        assert_eq!(n, 16);
    }

    #[test]
    fn batch_serialize_empty() {
        let events: [SpikeEvent; 0] = [];
        let mut buf = [0u8; 64];
        let n = serialize_spike_batch(&events, &mut buf, 10);
        assert_eq!(n, 0);
    }

    #[test]
    fn batch_serialize_zero_max() {
        let events = [make_event(1, 0, 0, 0.0)];
        let mut buf = [0u8; 64];
        let n = serialize_spike_batch(&events, &mut buf, 0);
        assert_eq!(n, 0);
    }
}
