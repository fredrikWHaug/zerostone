//! Fault handling and DFU configuration for nRF5340.
//!
//! Provides fault logging (circular buffer of fault events), DFU flash layout
//! validation, and reset reason parsing for the nRF5340 RESETREAS register.

/// Classification of ARM Cortex-M fault types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultType {
    /// HardFault -- escalated or unhandled fault.
    HardFault,
    /// BusFault -- memory bus error during data/instruction fetch.
    BusFault,
    /// UsageFault -- undefined instruction, unaligned access, etc.
    UsageFault,
    /// MemManage -- MPU violation or invalid memory access.
    MemManage,
    /// Unknown or unclassified fault.
    Unknown,
}

/// Snapshot of processor state at the time of a fault.
#[derive(Debug, Clone, Copy)]
pub struct FaultInfo {
    /// Type of fault that occurred.
    pub fault_type: FaultType,
    /// Program counter at the point of fault.
    pub pc: u32,
    /// Link register value at the point of fault.
    pub lr: u32,
    /// Configurable Fault Status Register value (SCB->CFSR).
    pub cfsr: u32,
    /// Approximate milliseconds since boot when the fault occurred.
    pub timestamp_ms: u32,
}

/// Fixed-capacity circular buffer for recording fault events.
///
/// `N` is the maximum number of fault entries stored. When full, the oldest
/// entry is overwritten. No heap allocation -- all storage is inline.
pub struct FaultLog<const N: usize> {
    /// Backing storage.
    buf: [Option<FaultInfo>; N],
    /// Index of the next slot to write into.
    head: usize,
    /// Total number of faults ever recorded (including overwritten).
    total: usize,
}

impl<const N: usize> FaultLog<N> {
    /// Create a new empty fault log.
    pub fn new() -> Self {
        // SAFETY: None is valid for Option<FaultInfo> at any index.
        // We use a const-compatible initialization.
        Self {
            buf: [const { None }; N],
            head: 0,
            total: 0,
        }
    }

    /// Record a fault event. If the log is full, the oldest entry is
    /// overwritten.
    pub fn record(&mut self, info: FaultInfo) {
        if N == 0 {
            return;
        }
        self.buf[self.head] = Some(info);
        self.head = (self.head + 1) % N;
        self.total += 1;
    }

    /// Return the most recently recorded fault, if any.
    pub fn last(&self) -> Option<&FaultInfo> {
        if self.total == 0 || N == 0 {
            return None;
        }
        let idx = if self.head == 0 { N - 1 } else { self.head - 1 };
        self.buf[idx].as_ref()
    }

    /// Total number of faults ever recorded (including overwritten).
    pub fn count(&self) -> usize {
        self.total
    }

    /// Iterate over stored faults from oldest to newest.
    ///
    /// Returns at most `N` entries (the current buffer contents).
    pub fn iter(&self) -> FaultLogIter<'_, N> {
        let stored = if self.total < N { self.total } else { N };
        let start = if self.total < N { 0 } else { self.head };
        FaultLogIter {
            log: self,
            pos: start,
            remaining: stored,
        }
    }

    /// Clear all recorded faults.
    pub fn clear(&mut self) {
        self.buf = [const { None }; N];
        self.head = 0;
        self.total = 0;
    }
}

/// Iterator over fault log entries from oldest to newest.
pub struct FaultLogIter<'a, const N: usize> {
    log: &'a FaultLog<N>,
    pos: usize,
    remaining: usize,
}

impl<'a, const N: usize> Iterator for FaultLogIter<'a, N> {
    type Item = &'a FaultInfo;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 || N == 0 {
            return None;
        }
        let entry = self.log.buf[self.pos].as_ref();
        self.pos = (self.pos + 1) % N;
        self.remaining -= 1;
        entry
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// DFU (Device Firmware Update) flash layout configuration for nRF5340.
///
/// Describes where the bootloader and application live in flash, and the
/// retained-RAM address used to signal a DFU request across resets.
pub struct DfuConfig {
    /// Flash address where the bootloader starts.
    /// Default: 0x000F_0000 (last 64 KB of 1 MB flash).
    pub bootloader_start: u32,
    /// Flash address where the application starts.
    /// Default: 0x0000_0000.
    pub app_start: u32,
    /// Address in retained RAM for the DFU trigger flag.
    /// Default: 0x2003_FF00.
    pub dfu_trigger_flag_addr: u32,
}

impl DfuConfig {
    /// Total flash size of the nRF5340 application core.
    const FLASH_SIZE: u32 = 0x0010_0000; // 1 MB

    /// Create a new DFU configuration with nRF5340 defaults.
    pub fn new() -> Self {
        Self {
            bootloader_start: 0x000F_0000,
            app_start: 0x0000_0000,
            dfu_trigger_flag_addr: 0x2003_FF00,
        }
    }

    /// Verify the flash layout is valid:
    /// - bootloader must start after the application
    /// - both addresses must be aligned to 4 KB page boundaries
    pub fn flash_layout_valid(&self) -> bool {
        let page_aligned = |addr: u32| addr & 0xFFF == 0;
        self.bootloader_start > self.app_start
            && page_aligned(self.bootloader_start)
            && page_aligned(self.app_start)
    }

    /// Size of the bootloader region in bytes (from bootloader_start to end of flash).
    pub fn bootloader_size(&self) -> u32 {
        Self::FLASH_SIZE - self.bootloader_start
    }
}

/// Reason for the most recent MCU reset, parsed from nRF5340 RESETREAS register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetReason {
    /// Power-on reset or brownout.
    PowerOn,
    /// Reset from the external pin.
    Pin,
    /// Watchdog timer reset.
    Watchdog,
    /// Software-triggered reset (AIRCR.SYSRESETREQ).
    Software,
    /// CPU lockup reset.
    Lockup,
    /// Unknown or multiple reset sources.
    Unknown(u32),
}

impl ResetReason {
    /// Parse an nRF5340 RESETREAS register value into a `ResetReason`.
    ///
    /// Bit mapping (nRF5340 Product Specification):
    /// - Bit 0: RESETPIN -- reset from pin
    /// - Bit 1: DOG0 -- watchdog 0 reset
    /// - Bit 2: CTRLAP -- ctrl-AP reset
    /// - Bit 3: SREQ -- software reset (AIRCR)
    /// - Bit 4: LOCKUP -- CPU lockup
    /// - Bit 16: DOG1 -- watchdog 1 reset
    ///
    /// A raw value of 0 indicates power-on reset (no reset source flag set).
    pub fn from_raw(raw: u32) -> Self {
        if raw == 0 {
            return ResetReason::PowerOn;
        }
        // Check bits in priority order.
        if raw & (1 << 4) != 0 {
            ResetReason::Lockup
        } else if raw & (1 << 1) != 0 || raw & (1 << 16) != 0 {
            ResetReason::Watchdog
        } else if raw & (1 << 3) != 0 {
            ResetReason::Software
        } else if raw & (1 << 0) != 0 {
            ResetReason::Pin
        } else {
            ResetReason::Unknown(raw)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_log_empty() {
        let log: FaultLog<4> = FaultLog::new();
        assert_eq!(log.count(), 0);
        assert!(log.last().is_none());
        assert_eq!(log.iter().count(), 0);
    }

    #[test]
    fn fault_log_record_and_last() {
        let mut log: FaultLog<4> = FaultLog::new();
        let info = FaultInfo {
            fault_type: FaultType::HardFault,
            pc: 0x0800_1234,
            lr: 0x0800_5678,
            cfsr: 0x0000_0001,
            timestamp_ms: 1500,
        };
        log.record(info);
        assert_eq!(log.count(), 1);

        let last = log.last().unwrap();
        assert_eq!(last.fault_type, FaultType::HardFault);
        assert_eq!(last.pc, 0x0800_1234);
        assert_eq!(last.lr, 0x0800_5678);
        assert_eq!(last.cfsr, 0x0000_0001);
        assert_eq!(last.timestamp_ms, 1500);
    }

    #[test]
    fn fault_log_circular_overflow() {
        let mut log: FaultLog<2> = FaultLog::new();
        let mk = |ts: u32| FaultInfo {
            fault_type: FaultType::BusFault,
            pc: 0,
            lr: 0,
            cfsr: 0,
            timestamp_ms: ts,
        };

        log.record(mk(100)); // slot 0
        log.record(mk(200)); // slot 1
        log.record(mk(300)); // overwrites slot 0

        assert_eq!(log.count(), 3);

        // Last should be the most recent (300).
        assert_eq!(log.last().unwrap().timestamp_ms, 300);

        // Iterate oldest to newest: 200, 300 (100 was overwritten).
        let timestamps: Vec<u32> = log.iter().map(|f| f.timestamp_ms).collect();
        assert_eq!(timestamps, vec![200, 300]);
    }

    #[test]
    fn fault_log_count() {
        let mut log: FaultLog<8> = FaultLog::new();
        let info = FaultInfo {
            fault_type: FaultType::UsageFault,
            pc: 0,
            lr: 0,
            cfsr: 0,
            timestamp_ms: 0,
        };
        for _ in 0..5 {
            log.record(info);
        }
        assert_eq!(log.count(), 5);
    }

    #[test]
    fn fault_log_clear() {
        let mut log: FaultLog<4> = FaultLog::new();
        let info = FaultInfo {
            fault_type: FaultType::MemManage,
            pc: 0x1000,
            lr: 0x2000,
            cfsr: 0xFF,
            timestamp_ms: 42,
        };
        log.record(info);
        log.record(info);
        assert_eq!(log.count(), 2);

        log.clear();
        assert_eq!(log.count(), 0);
        assert!(log.last().is_none());
        assert_eq!(log.iter().count(), 0);
    }

    #[test]
    fn dfu_config_defaults() {
        let cfg = DfuConfig::new();
        assert_eq!(cfg.bootloader_start, 0x000F_0000);
        assert_eq!(cfg.app_start, 0x0000_0000);
        assert_eq!(cfg.dfu_trigger_flag_addr, 0x2003_FF00);
        assert_eq!(cfg.bootloader_size(), 0x0001_0000); // 64 KB
    }

    #[test]
    fn dfu_config_layout_valid() {
        // Default layout is valid.
        let cfg = DfuConfig::new();
        assert!(cfg.flash_layout_valid());

        // Bootloader before app -- invalid.
        let bad = DfuConfig {
            bootloader_start: 0x0000_0000,
            app_start: 0x0001_0000,
            dfu_trigger_flag_addr: 0x2003_FF00,
        };
        assert!(!bad.flash_layout_valid());

        // Misaligned bootloader -- invalid.
        let misaligned = DfuConfig {
            bootloader_start: 0x000F_0100,
            app_start: 0x0000_0000,
            dfu_trigger_flag_addr: 0x2003_FF00,
        };
        assert!(!misaligned.flash_layout_valid());

        // Misaligned app -- invalid.
        let misaligned_app = DfuConfig {
            bootloader_start: 0x000F_0000,
            app_start: 0x0000_0001,
            dfu_trigger_flag_addr: 0x2003_FF00,
        };
        assert!(!misaligned_app.flash_layout_valid());

        // Equal addresses -- invalid (not strictly greater).
        let equal = DfuConfig {
            bootloader_start: 0x0000_0000,
            app_start: 0x0000_0000,
            dfu_trigger_flag_addr: 0x2003_FF00,
        };
        assert!(!equal.flash_layout_valid());
    }

    #[test]
    fn reset_reason_parse() {
        // Power-on: raw = 0
        assert_eq!(ResetReason::from_raw(0), ResetReason::PowerOn);

        // Pin reset: bit 0
        assert_eq!(ResetReason::from_raw(1 << 0), ResetReason::Pin);

        // Watchdog 0: bit 1
        assert_eq!(ResetReason::from_raw(1 << 1), ResetReason::Watchdog);

        // Watchdog 1: bit 16
        assert_eq!(ResetReason::from_raw(1 << 16), ResetReason::Watchdog);

        // Software reset: bit 3
        assert_eq!(ResetReason::from_raw(1 << 3), ResetReason::Software);

        // Lockup: bit 4
        assert_eq!(ResetReason::from_raw(1 << 4), ResetReason::Lockup);

        // Unknown: some other bit
        assert_eq!(ResetReason::from_raw(1 << 2), ResetReason::Unknown(1 << 2));
    }

    #[test]
    fn fault_log_iter_order() {
        let mut log: FaultLog<4> = FaultLog::new();
        let mk = |ts: u32| FaultInfo {
            fault_type: FaultType::Unknown,
            pc: 0,
            lr: 0,
            cfsr: 0,
            timestamp_ms: ts,
        };

        log.record(mk(10));
        log.record(mk(20));
        log.record(mk(30));

        let timestamps: Vec<u32> = log.iter().map(|f| f.timestamp_ms).collect();
        assert_eq!(timestamps, vec![10, 20, 30]);
    }

    #[test]
    fn fault_type_variants() {
        // Ensure all FaultType variants are distinct.
        let types = [
            FaultType::HardFault,
            FaultType::BusFault,
            FaultType::UsageFault,
            FaultType::MemManage,
            FaultType::Unknown,
        ];
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(types[i], types[j]);
            }
        }
    }
}
