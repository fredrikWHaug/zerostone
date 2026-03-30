//! BLE GATT server abstraction for Zerostone.
//!
//! Provides a mock/abstraction layer for the BLE GATT server that can be
//! tested on host without any BLE radio dependencies. The real `trouble`
//! BLE stack can be plugged in later via the trait-based architecture.
//!
//! Builds on top of [`crate::ble`] which defines UUIDs and serialization.

use crate::ble::{
    self, CONFIG_CHAR_UUID, SPIKE_EVENT_CHAR_UUID,
};
use crate::pipeline::SpikeEvent;
use crate::stats::RuntimeStats;

// ---------------------------------------------------------------------------
// Permissions (bitflags via u8)
// ---------------------------------------------------------------------------

/// Attribute permission flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Permissions(u8);

impl Permissions {
    pub const READ: Permissions = Permissions(0x01);
    pub const WRITE: Permissions = Permissions(0x02);
    pub const NOTIFY: Permissions = Permissions(0x04);

    /// Combine two permission sets.
    pub const fn union(self, other: Permissions) -> Permissions {
        Permissions(self.0 | other.0)
    }

    /// Check whether `self` contains all bits in `other`.
    pub const fn contains(self, other: Permissions) -> bool {
        (self.0 & other.0) == other.0
    }
}

// ---------------------------------------------------------------------------
// GattAttribute
// ---------------------------------------------------------------------------

/// A single GATT attribute (characteristic value).
#[derive(Clone)]
pub struct GattAttribute {
    /// 128-bit UUID.
    pub uuid: [u8; 16],
    /// Attribute handle (assigned by the table).
    pub handle: u16,
    /// Permission flags.
    pub permissions: Permissions,
    /// Value buffer (max 32 bytes).
    pub value: [u8; 32],
    /// Actual length of valid data in `value`.
    pub value_len: usize,
}

impl GattAttribute {
    fn new(uuid: [u8; 16], handle: u16, permissions: Permissions) -> Self {
        Self {
            uuid,
            handle,
            permissions,
            value: [0u8; 32],
            value_len: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Stats characteristic UUID
// ---------------------------------------------------------------------------

/// Runtime stats characteristic UUID.
///
/// `7a657273-746f-6e65-0004-000000000000`
pub const STATS_CHAR_UUID: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x65, 0x6e,
    0x6f, 0x74, 0x73, 0x72, 0x65, 0x7a, 0x00, 0x00,
];

// ---------------------------------------------------------------------------
// GattTable
// ---------------------------------------------------------------------------

/// Fixed-capacity GATT attribute table.
///
/// `N` is the maximum number of attributes the table can hold.
pub struct GattTable<const N: usize> {
    attrs: [Option<GattAttribute>; N],
    count: usize,
    next_handle: u16,
}

// We need a helper because Option<GattAttribute> is not Copy.
impl<const N: usize> GattTable<N> {
    /// Create a new empty GATT table.
    pub fn new() -> Self {
        Self {
            attrs: core::array::from_fn(|_| None),
            count: 0,
            next_handle: 1, // handles start at 1 (0 is reserved in ATT)
        }
    }

    /// Add an attribute with the given UUID and permissions.
    ///
    /// Returns the assigned handle, or 0 if the table is full.
    pub fn add_attribute(&mut self, uuid: [u8; 16], permissions: Permissions) -> u16 {
        if self.count >= N {
            return 0;
        }
        let handle = self.next_handle;
        self.attrs[self.count] = Some(GattAttribute::new(uuid, handle, permissions));
        self.count += 1;
        self.next_handle += 1;
        handle
    }

    /// Read the value of an attribute by handle.
    pub fn read(&self, handle: u16) -> Option<&[u8]> {
        for i in 0..self.count {
            if let Some(ref attr) = self.attrs[i] {
                if attr.handle == handle {
                    return Some(&attr.value[..attr.value_len]);
                }
            }
        }
        None
    }

    /// Write data to an attribute by handle.
    ///
    /// Returns `true` on success, `false` if the handle is invalid or
    /// data exceeds the 32-byte buffer.
    pub fn write(&mut self, handle: u16, data: &[u8]) -> bool {
        if data.len() > 32 {
            return false;
        }
        for i in 0..self.count {
            if let Some(ref mut attr) = self.attrs[i] {
                if attr.handle == handle {
                    attr.value[..data.len()].copy_from_slice(data);
                    attr.value_len = data.len();
                    return true;
                }
            }
        }
        false
    }

    /// Number of attributes in the table.
    pub fn count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// BleState
// ---------------------------------------------------------------------------

/// BLE connection state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BleState {
    Idle,
    Advertising,
    Connected,
    Disconnected,
}

// ---------------------------------------------------------------------------
// BleServer
// ---------------------------------------------------------------------------

/// BLE GATT server managing the Zerostone neural data service.
///
/// `N` is the GATT table capacity (number of characteristics).
pub struct BleServer<const N: usize> {
    pub state: BleState,
    pub table: GattTable<N>,
    pub spike_event_handle: u16,
    pub config_handle: u16,
    pub stats_handle: u16,
    pub notify_enabled: bool,
    pub connection_count: u32,
}

impl<const N: usize> BleServer<N> {
    /// Initialize a new BLE server with the Zerostone GATT service.
    ///
    /// Registers the spike event, config, and stats characteristics
    /// using UUIDs from [`crate::ble`].
    pub fn new() -> Self {
        let mut table = GattTable::new();

        let spike_event_handle = table.add_attribute(
            SPIKE_EVENT_CHAR_UUID,
            Permissions::READ.union(Permissions::NOTIFY),
        );
        let config_handle = table.add_attribute(
            CONFIG_CHAR_UUID,
            Permissions::READ.union(Permissions::WRITE),
        );
        let stats_handle = table.add_attribute(
            STATS_CHAR_UUID,
            Permissions::READ.union(Permissions::NOTIFY),
        );

        Self {
            state: BleState::Idle,
            table,
            spike_event_handle,
            config_handle,
            stats_handle,
            notify_enabled: false,
            connection_count: 0,
        }
    }

    /// Handle a BLE connection event.
    pub fn on_connect(&mut self) {
        self.state = BleState::Connected;
        self.connection_count += 1;
    }

    /// Handle a BLE disconnection event.
    pub fn on_disconnect(&mut self) {
        self.state = BleState::Disconnected;
        self.notify_enabled = false;
    }

    /// Begin advertising.
    pub fn start_advertising(&mut self) {
        self.state = BleState::Advertising;
    }

    /// Enable notifications (client has written to the CCCD).
    pub fn enable_notify(&mut self) {
        self.notify_enabled = true;
    }

    /// Serialize and write a spike event to the spike event attribute.
    ///
    /// Returns `true` on success.
    pub fn push_spike_event(&mut self, event: &SpikeEvent) -> bool {
        let mut buf = [0u8; 8];
        ble::serialize_spike_event(event, &mut buf);
        self.table.write(self.spike_event_handle, &buf)
    }

    /// Serialize and write runtime stats to the stats attribute.
    ///
    /// Returns `true` on success.
    pub fn push_stats(&mut self, stats: &RuntimeStats) -> bool {
        let mut buf = [0u8; 16];
        stats.serialize(&mut buf);
        self.table.write(self.stats_handle, &buf)
    }

    /// Read and deserialize the config attribute.
    pub fn read_config(&self) -> Option<ble::SorterConfig> {
        let data = self.table.read(self.config_handle)?;
        if data.len() < 8 {
            return None;
        }
        let buf: [u8; 8] = [
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ];
        Some(ble::deserialize_config(&buf))
    }

    /// Serialize and write a config to the config attribute.
    pub fn write_config(&mut self, config: &ble::SorterConfig) -> bool {
        let mut buf = [0u8; 8];
        ble::serialize_config(config, &mut buf);
        self.table.write(self.config_handle, &buf)
    }

    /// Whether the server is in the Connected state.
    pub fn is_connected(&self) -> bool {
        self.state == BleState::Connected
    }

    /// Whether notifications are enabled.
    pub fn is_notify_enabled(&self) -> bool {
        self.notify_enabled
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GattTable tests ----------------------------------------------------

    #[test]
    fn gatt_table_add_and_read() {
        let mut table = GattTable::<4>::new();
        let handle = table.add_attribute([0u8; 16], Permissions::READ);
        assert_ne!(handle, 0);
        // Freshly added attribute has zero-length value.
        let data = table.read(handle).unwrap();
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn gatt_table_write() {
        let mut table = GattTable::<4>::new();
        let handle = table.add_attribute([0u8; 16], Permissions::READ.union(Permissions::WRITE));
        assert!(table.write(handle, &[0xAA, 0xBB, 0xCC]));
        let data = table.read(handle).unwrap();
        assert_eq!(data, &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn gatt_table_invalid_handle() {
        let mut table = GattTable::<4>::new();
        assert!(table.read(99).is_none());
        assert!(!table.write(99, &[0x01]));
    }

    #[test]
    fn gatt_table_full() {
        let mut table = GattTable::<2>::new();
        let h1 = table.add_attribute([0u8; 16], Permissions::READ);
        let h2 = table.add_attribute([1u8; 16], Permissions::READ);
        let h3 = table.add_attribute([2u8; 16], Permissions::READ);
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
        assert_eq!(h3, 0, "should return 0 when table is full");
        assert_eq!(table.count(), 2);
    }

    // -- BleServer tests ----------------------------------------------------

    #[test]
    fn server_init_has_three_characteristics() {
        let server = BleServer::<8>::new();
        assert_eq!(server.table.count(), 3);
        assert_ne!(server.spike_event_handle, 0);
        assert_ne!(server.config_handle, 0);
        assert_ne!(server.stats_handle, 0);
        // All handles are distinct.
        assert_ne!(server.spike_event_handle, server.config_handle);
        assert_ne!(server.spike_event_handle, server.stats_handle);
        assert_ne!(server.config_handle, server.stats_handle);
    }

    #[test]
    fn server_state_transitions() {
        let mut server = BleServer::<8>::new();
        assert_eq!(server.state, BleState::Idle);

        server.start_advertising();
        assert_eq!(server.state, BleState::Advertising);

        server.on_connect();
        assert_eq!(server.state, BleState::Connected);
        assert!(server.is_connected());
        assert_eq!(server.connection_count, 1);

        server.on_disconnect();
        assert_eq!(server.state, BleState::Disconnected);
        assert!(!server.is_connected());
    }

    #[test]
    fn server_push_spike_event() {
        let mut server = BleServer::<8>::new();
        let event = SpikeEvent {
            sample_idx: 42000,
            channel: 3,
            cluster_id: 2,
            amplitude: -0.5,
        };
        assert!(server.push_spike_event(&event));

        // Read back and verify via deserialization.
        let data = server.table.read(server.spike_event_handle).unwrap();
        assert_eq!(data.len(), 8);
        let buf: [u8; 8] = data.try_into().unwrap();
        let out = ble::deserialize_spike_event(&buf);
        assert_eq!(out.sample_idx, 42000);
        assert_eq!(out.channel, 3);
        assert_eq!(out.cluster_id, 2);
        assert!((out.amplitude - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn server_push_stats() {
        let mut server = BleServer::<8>::new();
        let mut stats = RuntimeStats::new();
        stats.total_frames = 1000;
        stats.total_spikes = 50;
        stats.peak_spike_rate = 12;

        assert!(server.push_stats(&stats));

        let data = server.table.read(server.stats_handle).unwrap();
        assert_eq!(data.len(), 16);
        // Verify total_frames at offset 0..4.
        let frames = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(frames, 1000);
        // Verify total_spikes at offset 4..8.
        let spikes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(spikes, 50);
    }

    #[test]
    fn server_config_round_trip() {
        let mut server = BleServer::<8>::new();
        let config = ble::SorterConfig {
            threshold_factor: 5.5,
            n_templates: 8,
            sample_rate_hz: 30000,
        };
        assert!(server.write_config(&config));

        let out = server.read_config().unwrap();
        assert!((out.threshold_factor - 5.5).abs() < f32::EPSILON);
        assert_eq!(out.n_templates, 8);
        assert_eq!(out.sample_rate_hz, 30000);
    }

    #[test]
    fn server_notify_disabled_by_default() {
        let server = BleServer::<8>::new();
        assert!(!server.is_notify_enabled());
    }

    #[test]
    fn server_disconnect_disables_notify() {
        let mut server = BleServer::<8>::new();
        server.on_connect();
        server.enable_notify();
        assert!(server.is_notify_enabled());

        server.on_disconnect();
        assert!(!server.is_notify_enabled());
    }
}
