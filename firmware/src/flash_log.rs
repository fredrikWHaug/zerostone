//! Flash-based session logging for post-session data retrieval.
//!
//! After a recording session, the researcher removes the headstage and
//! retrieves session metadata: runtime stats, faults, configuration,
//! and session timestamps. This module provides a RAM-side index that
//! tracks what has been written to flash, suitable for the nRF5340
//! (1 MB flash, 4 KB pages, ~10K erase cycles per page).
//!
//! All structures are `no_std` and zero-alloc.

/// Magic value for log entry headers: 0x5A4C ("ZL" for Zerostone Log).
const MAGIC: u16 = 0x5A4C;

/// A contiguous region of flash memory used for logging.
#[derive(Clone, Copy, Debug)]
pub struct FlashRegion {
    /// Start address of the logging region in flash.
    pub start_addr: u32,
    /// Total size of the region in bytes.
    pub size: u32,
    /// Flash page size in bytes (4096 for nRF5340).
    pub page_size: u32,
}

impl FlashRegion {
    /// Create a new flash region descriptor.
    pub fn new(start_addr: u32, size: u32, page_size: u32) -> Self {
        Self {
            start_addr,
            size,
            page_size,
        }
    }

    /// Number of complete pages in this region.
    pub fn num_pages(&self) -> u32 {
        if self.page_size == 0 {
            return 0;
        }
        self.size / self.page_size
    }

    /// Return the start address of page N, or `None` if out of bounds.
    pub fn page_addr(&self, page_index: u32) -> Option<u32> {
        if page_index >= self.num_pages() {
            return None;
        }
        Some(self.start_addr + page_index * self.page_size)
    }

    /// Check whether an address falls within this region.
    pub fn contains(&self, addr: u32) -> bool {
        addr >= self.start_addr && addr < self.end_addr()
    }

    /// One-past-the-end address of this region.
    pub fn end_addr(&self) -> u32 {
        self.start_addr + self.size
    }
}

/// Type tag for each log entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LogEntryType {
    /// Session started.
    SessionStart = 0x01,
    /// Session ended.
    SessionEnd = 0x02,
    /// Runtime statistics snapshot.
    Stats = 0x03,
    /// Fault event.
    Fault = 0x04,
    /// Configuration snapshot.
    Config = 0x05,
}

/// Packed 8-byte header for each log entry.
///
/// Layout (little-endian):
/// - bytes 0..2: magic (0x5A4C)
/// - byte 2: entry_type
/// - byte 3: payload length
/// - bytes 4..6: sequence number
/// - bytes 6..8: timestamp (seconds since session start)
#[derive(Clone, Copy, Debug)]
pub struct LogEntryHeader {
    /// Magic value, must be 0x5A4C.
    pub magic: u16,
    /// Entry type as raw u8.
    pub entry_type: u8,
    /// Payload length in bytes (max 255).
    pub length: u8,
    /// Monotonically increasing sequence number.
    pub sequence: u16,
    /// Seconds since session start (wraps at 65535).
    pub timestamp_s: u16,
}

impl LogEntryHeader {
    /// Create a new log entry header with the correct magic.
    pub fn new(entry_type: LogEntryType, length: u8, sequence: u16, timestamp_s: u16) -> Self {
        Self {
            magic: MAGIC,
            entry_type: entry_type as u8,
            length,
            sequence,
            timestamp_s,
        }
    }

    /// Serialize this header into an 8-byte little-endian buffer.
    pub fn serialize(&self, buf: &mut [u8; 8]) {
        let m = self.magic.to_le_bytes();
        buf[0] = m[0];
        buf[1] = m[1];
        buf[2] = self.entry_type;
        buf[3] = self.length;
        let s = self.sequence.to_le_bytes();
        buf[4] = s[0];
        buf[5] = s[1];
        let t = self.timestamp_s.to_le_bytes();
        buf[6] = t[0];
        buf[7] = t[1];
    }

    /// Deserialize from an 8-byte buffer. Returns `None` if magic is invalid.
    pub fn deserialize(buf: &[u8; 8]) -> Option<Self> {
        let magic = u16::from_le_bytes([buf[0], buf[1]]);
        if magic != MAGIC {
            return None;
        }
        Some(Self {
            magic,
            entry_type: buf[2],
            length: buf[3],
            sequence: u16::from_le_bytes([buf[4], buf[5]]),
            timestamp_s: u16::from_le_bytes([buf[6], buf[7]]),
        })
    }

    /// Check whether the magic field is valid.
    pub fn is_valid(&self) -> bool {
        self.magic == MAGIC
    }

    /// Total size of this entry in flash (header + payload).
    pub fn total_size(&self) -> usize {
        8 + self.length as usize
    }
}

/// RAM-side index tracking entries written to a flash logging region.
///
/// `MAX_ENTRIES` is the maximum number of entries tracked in the RAM index.
/// The struct does NOT perform actual flash writes -- it tracks offsets and
/// headers so the caller can issue flash writes and later retrieve entries.
pub struct FlashLog<const MAX_ENTRIES: usize> {
    /// Flash region descriptor.
    pub region: FlashRegion,
    /// Current write position (byte offset from region start).
    pub write_offset: u32,
    /// Total entries written this session.
    pub entry_count: u16,
    /// Next sequence number to assign.
    pub sequence: u16,
    /// Absolute seconds at session start (for computing relative timestamps).
    pub session_start_s: u32,
    /// Circular index of (flash_offset, header) for each entry.
    pub entries: [(u32, LogEntryHeader); MAX_ENTRIES],
}

impl<const MAX_ENTRIES: usize> FlashLog<MAX_ENTRIES> {
    /// Default (zeroed) header for array initialization.
    const EMPTY_HEADER: LogEntryHeader = LogEntryHeader {
        magic: 0,
        entry_type: 0,
        length: 0,
        sequence: 0,
        timestamp_s: 0,
    };

    /// Create a new flash log tracker for the given region.
    pub fn new(region: FlashRegion) -> Self {
        Self {
            region,
            write_offset: 0,
            entry_count: 0,
            sequence: 0,
            session_start_s: 0,
            entries: [(0u32, Self::EMPTY_HEADER); MAX_ENTRIES],
        }
    }

    /// Begin a new recording session. Resets write offset and records a
    /// SessionStart entry.
    pub fn begin_session(&mut self, timestamp_s: u32) {
        self.session_start_s = timestamp_s;
        self.write_offset = 0;
        self.entry_count = 0;
        self.sequence = 0;
        // Append a session-start marker (no payload).
        self.append_entry(LogEntryType::SessionStart, 0, timestamp_s);
    }

    /// Record a session-end marker.
    pub fn end_session(&mut self, timestamp_s: u32) {
        self.append_entry(LogEntryType::SessionEnd, 0, timestamp_s);
    }

    /// Append a runtime stats snapshot (16-byte payload).
    /// Returns `false` if the region is full.
    pub fn append_stats(&mut self, _stats_bytes: &[u8; 16], timestamp_s: u32) -> bool {
        self.append_entry(LogEntryType::Stats, 16, timestamp_s)
    }

    /// Append a fault entry (9-byte payload: 1 byte type + 4 byte PC + 4 byte LR).
    /// Returns `false` if the region is full.
    pub fn append_fault(
        &mut self,
        _fault_type: u8,
        _pc: u32,
        _lr: u32,
        timestamp_s: u32,
    ) -> bool {
        self.append_entry(LogEntryType::Fault, 9, timestamp_s)
    }

    /// Append a configuration snapshot.
    /// Returns `false` if the region is full.
    pub fn append_config(&mut self, config_bytes: &[u8], timestamp_s: u32) -> bool {
        if config_bytes.len() > 255 {
            return false;
        }
        self.append_entry(LogEntryType::Config, config_bytes.len() as u8, timestamp_s)
    }

    /// Number of entries written this session.
    pub fn entry_count(&self) -> u16 {
        self.entry_count
    }

    /// Look up an entry by index in the RAM index.
    pub fn get_entry(&self, index: usize) -> Option<&(u32, LogEntryHeader)> {
        if index >= self.entry_count as usize || index >= MAX_ENTRIES {
            return None;
        }
        Some(&self.entries[index % MAX_ENTRIES])
    }

    /// Check whether the next minimum-size entry (header only, 8 bytes)
    /// would exceed the region.
    pub fn is_full(&self) -> bool {
        self.write_offset + 8 > self.region.size
    }

    /// Bytes consumed so far in the flash region.
    pub fn bytes_used(&self) -> u32 {
        self.write_offset
    }

    /// Bytes remaining in the flash region.
    pub fn bytes_remaining(&self) -> u32 {
        self.region.size.saturating_sub(self.write_offset)
    }

    /// Internal: compute relative timestamp and append an entry to the index.
    fn append_entry(&mut self, entry_type: LogEntryType, payload_len: u8, timestamp_s: u32) -> bool {
        let total = 8u32 + payload_len as u32;
        if self.write_offset + total > self.region.size {
            return false;
        }

        let rel_ts = timestamp_s.saturating_sub(self.session_start_s);
        let rel_ts_u16 = if rel_ts > u16::MAX as u32 {
            u16::MAX
        } else {
            rel_ts as u16
        };

        let header = LogEntryHeader::new(entry_type, payload_len, self.sequence, rel_ts_u16);
        let idx = (self.entry_count as usize) % MAX_ENTRIES;
        self.entries[idx] = (self.write_offset, header);

        self.write_offset += total;
        self.entry_count = self.entry_count.wrapping_add(1);
        self.sequence = self.sequence.wrapping_add(1);

        true
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_region() -> FlashRegion {
        FlashRegion::new(0x000E_0000, 64 * 1024, 4096)
    }

    // -- FlashRegion --------------------------------------------------------

    #[test]
    fn flash_region_basic() {
        let r = default_region();
        assert_eq!(r.num_pages(), 16); // 64 KB / 4 KB
        assert_eq!(r.page_addr(0), Some(0x000E_0000));
        assert_eq!(r.page_addr(1), Some(0x000E_1000));
        assert_eq!(r.page_addr(15), Some(0x000E_F000));
        assert!(r.contains(0x000E_0000));
        assert!(r.contains(0x000E_FFFF));
        assert!(!r.contains(0x000F_0000)); // one-past-end
        assert!(!r.contains(0x000D_FFFF)); // before start
        assert_eq!(r.end_addr(), 0x000F_0000);
    }

    #[test]
    fn flash_region_page_addr_out_of_bounds() {
        let r = default_region();
        assert_eq!(r.page_addr(16), None);
        assert_eq!(r.page_addr(100), None);
        assert_eq!(r.page_addr(u32::MAX), None);
    }

    // -- LogEntryHeader -----------------------------------------------------

    #[test]
    fn header_serialize_deserialize() {
        let h = LogEntryHeader::new(LogEntryType::Stats, 16, 42, 1000);
        let mut buf = [0u8; 8];
        h.serialize(&mut buf);

        let h2 = LogEntryHeader::deserialize(&buf).unwrap();
        assert_eq!(h2.magic, MAGIC);
        assert_eq!(h2.entry_type, LogEntryType::Stats as u8);
        assert_eq!(h2.length, 16);
        assert_eq!(h2.sequence, 42);
        assert_eq!(h2.timestamp_s, 1000);
    }

    #[test]
    fn header_invalid_magic() {
        let mut buf = [0u8; 8];
        // Write wrong magic.
        buf[0] = 0xFF;
        buf[1] = 0xFF;
        assert!(LogEntryHeader::deserialize(&buf).is_none());
    }

    #[test]
    fn header_total_size() {
        let h = LogEntryHeader::new(LogEntryType::Config, 200, 0, 0);
        assert_eq!(h.total_size(), 208); // 8 + 200

        let h0 = LogEntryHeader::new(LogEntryType::SessionStart, 0, 0, 0);
        assert_eq!(h0.total_size(), 8);
    }

    #[test]
    fn header_is_valid() {
        let h = LogEntryHeader::new(LogEntryType::Fault, 9, 0, 0);
        assert!(h.is_valid());

        let bad = LogEntryHeader {
            magic: 0x0000,
            entry_type: 0,
            length: 0,
            sequence: 0,
            timestamp_s: 0,
        };
        assert!(!bad.is_valid());
    }

    // -- FlashLog -----------------------------------------------------------

    #[test]
    fn flash_log_new_empty() {
        let log: FlashLog<64> = FlashLog::new(default_region());
        assert_eq!(log.entry_count(), 0);
        assert_eq!(log.bytes_used(), 0);
        assert!(!log.is_full());
    }

    #[test]
    fn flash_log_begin_session() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(1000);
        assert_eq!(log.entry_count(), 1);
        assert_eq!(log.session_start_s, 1000);

        let (offset, header) = log.get_entry(0).unwrap();
        assert_eq!(*offset, 0);
        assert_eq!(header.entry_type, LogEntryType::SessionStart as u8);
        assert_eq!(header.length, 0);
        assert_eq!(header.timestamp_s, 0); // relative to session start
    }

    #[test]
    fn flash_log_append_stats() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(100);
        let stats = [0u8; 16];
        let ok = log.append_stats(&stats, 105);
        assert!(ok);
        assert_eq!(log.entry_count(), 2); // session_start + stats

        let (_, header) = log.get_entry(1).unwrap();
        assert_eq!(header.entry_type, LogEntryType::Stats as u8);
        assert_eq!(header.length, 16);
        assert_eq!(header.timestamp_s, 5); // 105 - 100
    }

    #[test]
    fn flash_log_append_fault() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(200);
        let ok = log.append_fault(0x01, 0x0800_1234, 0x0800_5678, 210);
        assert!(ok);
        assert_eq!(log.entry_count(), 2);

        let (_, header) = log.get_entry(1).unwrap();
        assert_eq!(header.entry_type, LogEntryType::Fault as u8);
        assert_eq!(header.length, 9);
        assert_eq!(header.timestamp_s, 10);
    }

    #[test]
    fn flash_log_full_rejects() {
        // Tiny region: 20 bytes. SessionStart header = 8 bytes.
        // Then 12 bytes remain, which is not enough for stats (8 + 16 = 24).
        let region = FlashRegion::new(0x1000, 20, 4096);
        let mut log: FlashLog<8> = FlashLog::new(region);
        log.begin_session(0); // uses 8 bytes, 12 remain
        let stats = [0u8; 16];
        let ok = log.append_stats(&stats, 1);
        assert!(!ok); // 8 + 16 = 24 > 12 remaining
    }

    #[test]
    fn flash_log_bytes_remaining() {
        let region = FlashRegion::new(0x1000, 100, 4096);
        let mut log: FlashLog<16> = FlashLog::new(region);
        assert_eq!(log.bytes_remaining(), 100);

        log.begin_session(0); // 8 bytes (header only, no payload)
        assert_eq!(log.bytes_used(), 8);
        assert_eq!(log.bytes_remaining(), 92);
    }

    #[test]
    fn flash_log_sequence_increments() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(0);

        let stats = [0u8; 16];
        log.append_stats(&stats, 1);
        log.append_fault(0, 0, 0, 2);

        // Entries: session_start(seq=0), stats(seq=1), fault(seq=2)
        assert_eq!(log.get_entry(0).unwrap().1.sequence, 0);
        assert_eq!(log.get_entry(1).unwrap().1.sequence, 1);
        assert_eq!(log.get_entry(2).unwrap().1.sequence, 2);
    }

    #[test]
    fn flash_log_end_session() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(500);
        log.end_session(600);
        assert_eq!(log.entry_count(), 2);

        let (_, header) = log.get_entry(1).unwrap();
        assert_eq!(header.entry_type, LogEntryType::SessionEnd as u8);
        assert_eq!(header.timestamp_s, 100); // 600 - 500
    }

    #[test]
    fn flash_log_append_config() {
        let mut log: FlashLog<64> = FlashLog::new(default_region());
        log.begin_session(0);
        let config = [0xAA; 8];
        let ok = log.append_config(&config, 0);
        assert!(ok);
        assert_eq!(log.entry_count(), 2);

        let (_, header) = log.get_entry(1).unwrap();
        assert_eq!(header.entry_type, LogEntryType::Config as u8);
        assert_eq!(header.length, 8);
    }

    #[test]
    fn flash_log_get_entry_out_of_bounds() {
        let log: FlashLog<64> = FlashLog::new(default_region());
        assert!(log.get_entry(0).is_none());
        assert!(log.get_entry(100).is_none());
    }

    #[test]
    fn flash_region_zero_page_size() {
        let r = FlashRegion::new(0, 4096, 0);
        assert_eq!(r.num_pages(), 0);
        assert_eq!(r.page_addr(0), None);
    }
}
