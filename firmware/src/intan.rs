//! Intan RHD2132 SPI driver for neural recording.
//!
//! The RHD2132 is a 32-channel neural recording amplifier with integrated ADCs,
//! communicating over a 16-bit SPI interface. Each SPI transaction sends a command
//! on MOSI and receives the result of a command issued **two transactions earlier**
//! (2-deep pipeline).
//!
//! This driver handles command encoding, pipeline tracking, register configuration,
//! and full 32-channel frame acquisition.

use embedded_hal_async::spi::SpiDevice;

// ---------------------------------------------------------------------------
// Command encoding
// ---------------------------------------------------------------------------

/// Number of recording channels on the RHD2132.
pub const NUM_CHANNELS: usize = 32;

/// Pipeline depth: results arrive 2 SPI transactions after the command is sent.
const PIPELINE_DEPTH: usize = 2;

/// Encode a CONVERT command for the given `channel` (0-31).
///
/// Bit layout: `[15:14]=0b00, [13:8]=channel, [7:0]=0`.
#[inline]
pub const fn cmd_convert(channel: u8) -> u16 {
    ((channel as u16) & 0x3F) << 8
}

/// Encode a READ register command.
///
/// Bit layout: `[15:14]=0b11, [13:8]=register, [7:0]=0`.
#[inline]
pub const fn cmd_read(register: u8) -> u16 {
    0xC000 | (((register as u16) & 0x3F) << 8)
}

/// Encode a WRITE register command.
///
/// Bit layout: `[15:14]=0b10, [13:8]=register, [7:0]=value`.
#[inline]
pub const fn cmd_write(register: u8, value: u8) -> u16 {
    0x8000 | (((register as u16) & 0x3F) << 8) | (value as u16)
}

/// ADC self-calibration command (`0x5500`).
pub const CMD_CALIBRATE: u16 = 0x5500;

/// Clear calibration command (`0x6A00`).
pub const CMD_CLEAR_CALIBRATE: u16 = 0x6A00;

/// Convert a raw 16-bit ADC value (offset binary) to signed `i16`.
///
/// The RHD2132 ADC produces unsigned 16-bit results centered at 0x8000.
/// Subtracting 0x8000 maps the range to `i16::MIN..=i16::MAX`.
#[inline]
pub const fn adc_to_signed(raw: u16) -> i16 {
    (raw as i16).wrapping_sub(i16::MIN) // equivalent to raw.wrapping_sub(0x8000) as i16
}

// ---------------------------------------------------------------------------
// Pipeline FIFO
// ---------------------------------------------------------------------------

/// Tracks the 2-deep SPI result pipeline.
///
/// Each slot records which command was in-flight so that the caller can
/// associate the received result with the correct request.
struct Pipeline {
    /// Ring buffer of pending command tags.
    slots: [u16; PIPELINE_DEPTH],
    /// Write index (next slot to fill).
    head: usize,
    /// Number of valid entries (0..=PIPELINE_DEPTH).
    len: usize,
}

impl Pipeline {
    const fn new() -> Self {
        Self {
            slots: [0; PIPELINE_DEPTH],
            head: 0,
            len: 0,
        }
    }

    /// Push a command into the pipeline. Returns the command that was
    /// previously in the oldest slot (the one whose result is now arriving),
    /// or `None` if the pipeline is still filling.
    fn push(&mut self, cmd: u16) -> Option<u16> {
        let popped = if self.len == PIPELINE_DEPTH {
            // Pipeline is full — the oldest entry corresponds to the result
            // we are receiving right now.
            let tail = (self.head + PIPELINE_DEPTH - self.len) % PIPELINE_DEPTH;
            let old = self.slots[tail];
            Some(old)
        } else {
            self.len += 1;
            None
        };
        self.slots[self.head] = cmd;
        self.head = (self.head + 1) % PIPELINE_DEPTH;
        popped
    }

    /// Reset the pipeline state.
    fn reset(&mut self) {
        self.slots = [0; PIPELINE_DEPTH];
        self.head = 0;
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Async driver for the Intan RHD2132 neural recording chip.
///
/// Generic over any SPI device implementing `embedded_hal_async::spi::SpiDevice`.
///
/// # Example (pseudo-code)
///
/// ```ignore
/// let mut intan = IntanDriver::new(spi);
/// intan.init().await?;
/// loop {
///     let frame = intan.read_frame().await?;
///     // frame contains 32 signed ADC samples
/// }
/// ```
pub struct IntanDriver<SPI> {
    spi: SPI,
    pipeline: Pipeline,
}

impl<SPI, E> IntanDriver<SPI>
where
    SPI: SpiDevice<u8, Error = E>,
{
    /// Create a new driver wrapping the given SPI device.
    pub fn new(spi: SPI) -> Self {
        Self {
            spi,
            pipeline: Pipeline::new(),
        }
    }

    /// Perform a single 16-bit SPI transaction.
    ///
    /// Sends `cmd` on MOSI and returns the simultaneously received 16-bit word
    /// from MISO.
    async fn transfer_word(&mut self, cmd: u16) -> Result<u16, E> {
        let mut buf = cmd.to_be_bytes();
        self.spi.transfer_in_place(&mut buf).await?;
        Ok(u16::from_be_bytes(buf))
    }

    /// Send a command and return the raw 16-bit MISO response.
    ///
    /// Note: due to the 2-deep pipeline, the returned value corresponds to
    /// a command sent **two transactions ago**, not the current one.
    async fn send_cmd(&mut self, cmd: u16) -> Result<u16, E> {
        let result = self.transfer_word(cmd).await?;
        self.pipeline.push(cmd);
        Ok(result)
    }

    /// Initialize the RHD2132: configure registers and run ADC calibration.
    ///
    /// Register configuration targets a 300 Hz -- 6 kHz bandpass, suitable for
    /// extracellular spike recording at 20--30 kHz sample rate.
    ///
    /// After writing registers, a CALIBRATE command is issued followed by
    /// 9 dummy transactions (the chip requires 9 cycles to complete calibration).
    pub async fn init(&mut self) -> Result<(), E> {
        self.pipeline.reset();

        // --- Bandwidth registers (0-3) ---
        // Register 0: ADC config / upper bandwidth resistor Rh1 DAC1
        //   For ~6 kHz upper cutoff at 30 kHz sampling: Rh1 DAC1 = 8
        self.send_cmd(cmd_write(0, 0x08)).await?;

        // Register 1: Rh1 DAC2 (upper bandwidth fine adjust)
        self.send_cmd(cmd_write(1, 0x00)).await?;

        // Register 2: Rh2 DAC1 (upper bandwidth, second stage)
        self.send_cmd(cmd_write(2, 0x08)).await?;

        // Register 3: Rh2 DAC2 + RL DAC1 for lower cutoff ~300 Hz
        self.send_cmd(cmd_write(3, 0x04)).await?;

        // --- ADC configuration (register 4) ---
        // Bit 7: weak MISO = 0, Bit 6: ADC reference BW = 1 (wide),
        // Bits 5-4: temp sensor / aux inputs, Bit 2: twoscomp = 0 (offset binary)
        self.send_cmd(cmd_write(4, 0x40)).await?;

        // --- Registers 5-7: impedance check / DAC (defaults = 0) ---
        self.send_cmd(cmd_write(5, 0x00)).await?;
        self.send_cmd(cmd_write(6, 0x00)).await?;
        self.send_cmd(cmd_write(7, 0x00)).await?;

        // --- Channel enable registers 14-17 ---
        // Each register enables 8 channels (bits 7:0). 0xFF = all on.
        self.send_cmd(cmd_write(14, 0xFF)).await?; // channels 0-7
        self.send_cmd(cmd_write(15, 0xFF)).await?; // channels 8-15
        self.send_cmd(cmd_write(16, 0xFF)).await?; // channels 16-23
        self.send_cmd(cmd_write(17, 0xFF)).await?; // channels 24-31

        // --- Run ADC calibration ---
        self.send_cmd(CMD_CALIBRATE).await?;

        // Calibration requires 9 SPI cycles to complete.
        for _ in 0..9 {
            self.send_cmd(cmd_convert(0)).await?;
        }

        // Reset pipeline state so read_frame starts clean.
        self.pipeline.reset();

        Ok(())
    }

    /// Read one full frame of 32 ADC channels.
    ///
    /// Sends 32 CONVERT commands (channels 0-31) plus 2 pipeline-flush
    /// dummy commands (34 total transactions). Returns the 32 signed samples
    /// in channel order.
    pub async fn read_frame(&mut self) -> Result<[i16; NUM_CHANNELS], E> {
        self.pipeline.reset();
        let mut frame = [0i16; NUM_CHANNELS];

        // Total SPI transactions: 32 CONVERT + 2 flush = 34.
        // The first 2 results are pipeline fill (discarded).
        // Results 2..33 correspond to channels 0..31.
        let total = NUM_CHANNELS + PIPELINE_DEPTH;

        for i in 0..total {
            let cmd = if i < NUM_CHANNELS {
                cmd_convert(i as u8)
            } else {
                // Pipeline flush: send dummy CONVERT(0)
                cmd_convert(0)
            };

            let raw = self.send_cmd(cmd).await?;

            // After the pipeline fills (i >= PIPELINE_DEPTH), each result
            // corresponds to channel (i - PIPELINE_DEPTH).
            if i >= PIPELINE_DEPTH {
                let ch = i - PIPELINE_DEPTH;
                frame[ch] = adc_to_signed(raw);
            }
        }

        Ok(frame)
    }

    /// Read a single register from the RHD2132.
    ///
    /// Sends a READ command followed by 2 pipeline-flush dummy commands,
    /// then returns the register value from the third response.
    pub async fn read_register(&mut self, reg: u8) -> Result<u16, E> {
        self.pipeline.reset();

        // Transaction 0: send READ(reg) — result is stale (pipeline fill)
        self.send_cmd(cmd_read(reg)).await?;
        // Transaction 1: flush — result is still stale
        self.send_cmd(cmd_convert(0)).await?;
        // Transaction 2: flush — THIS result corresponds to READ(reg)
        let result = self.send_cmd(cmd_convert(0)).await?;

        Ok(result)
    }

    /// Write a value to a register on the RHD2132.
    ///
    /// Sends the WRITE command and flushes the pipeline with 2 dummy
    /// transactions to ensure the write completes.
    pub async fn write_register(&mut self, reg: u8, value: u8) -> Result<(), E> {
        self.pipeline.reset();

        self.send_cmd(cmd_write(reg, value)).await?;
        // Flush pipeline to ensure the write is clocked through.
        self.send_cmd(cmd_convert(0)).await?;
        self.send_cmd(cmd_convert(0)).await?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use embedded_hal::spi::{self, ErrorType, Operation};

    // --- Mock SPI error type ---

    #[derive(Debug, Clone, Copy)]
    struct MockError;

    impl spi::Error for MockError {
        fn kind(&self) -> spi::ErrorKind {
            spi::ErrorKind::Other
        }
    }

    // --- Mock SPI device ---

    /// A mock SPI device that simulates the RHD2132 2-deep pipeline.
    ///
    /// On each `transfer_in_place` call it:
    /// 1. Records the incoming 16-bit command.
    /// 2. Returns the response from 2 transactions ago (pipeline delay).
    ///
    /// Synthetic ADC responses for CONVERT commands return `channel * 100 + 0x8000`
    /// (offset binary), so after `adc_to_signed` the value equals `channel * 100`.
    struct MockSpi {
        inner: RefCell<MockSpiInner>,
    }

    struct MockSpiInner {
        /// Pipeline buffer: last 2 commands' responses.
        pipe: [u16; PIPELINE_DEPTH],
        /// Number of transactions so far (used to fill the pipeline).
        count: usize,
    }

    impl MockSpi {
        fn new() -> Self {
            Self {
                inner: RefCell::new(MockSpiInner {
                    pipe: [0; PIPELINE_DEPTH],
                    count: 0,
                }),
            }
        }

        /// Generate the response that would be produced by `cmd`.
        fn response_for(cmd: u16) -> u16 {
            let opcode = (cmd >> 14) & 0x03;
            match opcode {
                // CONVERT: return channel * 100 + 0x8000 (offset binary)
                0b00 => {
                    let ch = ((cmd >> 8) & 0x3F) as u16;
                    ch * 100 + 0x8000
                }
                // READ: return the register number in the low byte
                0b11 => (cmd >> 8) & 0x3F,
                // WRITE: return 0 (no meaningful response)
                0b10 => 0,
                // CALIBRATE / CLEAR: return 0
                _ => 0,
            }
        }
    }

    impl ErrorType for MockSpi {
        type Error = MockError;
    }

    impl spi::SpiDevice<u8> for MockSpi {
        fn transaction(
            &mut self,
            operations: &mut [Operation<'_, u8>],
        ) -> Result<(), Self::Error> {
            for op in operations.iter_mut() {
                if let Operation::TransferInPlace(buf) = op {
                    if buf.len() == 2 {
                        let inner = self.inner.get_mut();
                        let cmd = u16::from_be_bytes([buf[0], buf[1]]);

                        // The response that will eventually come back for this cmd.
                        let resp = Self::response_for(cmd);

                        // What we return NOW is the response from 2 transactions ago.
                        let output = if inner.count >= PIPELINE_DEPTH {
                            inner.pipe[inner.count % PIPELINE_DEPTH]
                        } else {
                            0 // pipeline still filling
                        };

                        // Store this command's response for delivery later.
                        inner.pipe[inner.count % PIPELINE_DEPTH] = resp;
                        inner.count += 1;

                        let out_bytes = output.to_be_bytes();
                        buf[0] = out_bytes[0];
                        buf[1] = out_bytes[1];
                    }
                }
            }
            Ok(())
        }
    }

    // --- Async wrapper for blocking mock ---

    /// Wraps the blocking `MockSpi` to implement `embedded_hal_async::spi::SpiDevice`.
    struct AsyncMockSpi {
        mock: MockSpi,
    }

    impl AsyncMockSpi {
        fn new() -> Self {
            Self {
                mock: MockSpi::new(),
            }
        }
    }

    impl embedded_hal_async::spi::ErrorType for AsyncMockSpi {
        type Error = MockError;
    }

    impl embedded_hal_async::spi::SpiDevice<u8> for AsyncMockSpi {
        async fn transaction(
            &mut self,
            operations: &mut [embedded_hal_async::spi::Operation<'_, u8>],
        ) -> Result<(), Self::Error> {
            for op in operations.iter_mut() {
                match op {
                    embedded_hal_async::spi::Operation::TransferInPlace(buf) => {
                        let mut blocking_ops =
                            [spi::Operation::TransferInPlace(buf)];
                        spi::SpiDevice::transaction(&mut self.mock, &mut blocking_ops)?;
                    }
                    _ => {} // Other operations not used by this driver
                }
            }
            Ok(())
        }
    }

    // --- Command encoding tests ---

    #[test]
    fn test_cmd_convert() {
        // Channel 0
        assert_eq!(cmd_convert(0), 0x0000);
        // Channel 1
        assert_eq!(cmd_convert(1), 0x0100);
        // Channel 31
        assert_eq!(cmd_convert(31), 0x1F00);
        // Bits 15:14 must be 0b00
        assert_eq!(cmd_convert(31) & 0xC000, 0x0000);
    }

    #[test]
    fn test_cmd_read() {
        // Register 0
        assert_eq!(cmd_read(0), 0xC000);
        // Register 5
        assert_eq!(cmd_read(5), 0xC500);
        // Register 63
        assert_eq!(cmd_read(63), 0xFF00);
        // Bits 15:14 must be 0b11
        assert_eq!(cmd_read(0) & 0xC000, 0xC000);
    }

    #[test]
    fn test_cmd_write() {
        // Register 0, value 0
        assert_eq!(cmd_write(0, 0), 0x8000);
        // Register 4, value 0x40
        assert_eq!(cmd_write(4, 0x40), 0x8440);
        // Register 14, value 0xFF
        assert_eq!(cmd_write(14, 0xFF), 0x8EFF);
        // Bits 15:14 must be 0b10
        assert_eq!(cmd_write(0, 0) & 0xC000, 0x8000);
    }

    #[test]
    fn test_cmd_calibrate() {
        // 0b01_010101_00000000 = 0x5500
        assert_eq!(CMD_CALIBRATE, 0x5500);
        assert_eq!(CMD_CALIBRATE & 0xC000, 0x4000); // bits 15:14 = 0b01
    }

    #[test]
    fn test_cmd_clear_calibrate() {
        // 0b01_101010_00000000 = 0x6A00
        assert_eq!(CMD_CLEAR_CALIBRATE, 0x6A00);
        assert_eq!(CMD_CLEAR_CALIBRATE & 0xC000, 0x4000);
    }

    // --- ADC conversion test ---

    #[test]
    fn test_adc_to_signed() {
        assert_eq!(adc_to_signed(0x8000), 0);
        assert_eq!(adc_to_signed(0x8001), 1);
        assert_eq!(adc_to_signed(0x7FFF), -1);
        assert_eq!(adc_to_signed(0xFFFF), i16::MAX);
        assert_eq!(adc_to_signed(0x0000), i16::MIN);
    }

    // --- Pipeline state tests ---

    #[test]
    fn test_pipeline_fill() {
        let mut pipe = Pipeline::new();
        // First two pushes return None (pipeline filling).
        assert!(pipe.push(0xAAAA).is_none());
        assert!(pipe.push(0xBBBB).is_none());
        // Third push returns the first command.
        assert_eq!(pipe.push(0xCCCC), Some(0xAAAA));
        // Fourth push returns the second command.
        assert_eq!(pipe.push(0xDDDD), Some(0xBBBB));
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipe = Pipeline::new();
        pipe.push(0x1111);
        pipe.push(0x2222);
        pipe.reset();
        // After reset, pipeline must fill again.
        assert!(pipe.push(0x3333).is_none());
        assert!(pipe.push(0x4444).is_none());
        assert_eq!(pipe.push(0x5555), Some(0x3333));
    }

    // --- Frame reading test with mock SPI ---

    #[tokio::test]
    async fn test_read_frame() {
        let spi = AsyncMockSpi::new();
        let mut driver = IntanDriver::new(spi);

        // Init first (configures registers + calibration).
        driver.init().await.unwrap();

        // Read a frame.
        let frame = driver.read_frame().await.unwrap();

        // Each channel should equal channel_number * 100 (from the mock).
        for ch in 0..NUM_CHANNELS {
            assert_eq!(
                frame[ch], (ch as i16) * 100,
                "channel {} mismatch: got {}, expected {}",
                ch, frame[ch], (ch as i16) * 100
            );
        }
    }

    // --- Register read test ---

    #[tokio::test]
    async fn test_read_register() {
        let spi = AsyncMockSpi::new();
        let mut driver = IntanDriver::new(spi);

        // The mock returns the register number in the low byte for READ commands.
        let val = driver.read_register(5).await.unwrap();
        assert_eq!(val, 5);
    }

    // --- Register write test (should not panic) ---

    #[tokio::test]
    async fn test_write_register() {
        let spi = AsyncMockSpi::new();
        let mut driver = IntanDriver::new(spi);
        driver.write_register(14, 0xFF).await.unwrap();
    }

    // --- Blocking mock SPI pipeline delay test ---

    #[test]
    fn test_mock_spi_pipeline_delay() {
        let mut mock = MockSpi::new();

        // Helper to do one 16-bit transfer.
        let mut xfer = |cmd: u16| -> u16 {
            let mut buf = cmd.to_be_bytes();
            let mut ops = [spi::Operation::TransferInPlace(&mut buf)];
            spi::SpiDevice::transaction(&mut mock, &mut ops).unwrap();
            u16::from_be_bytes(buf)
        };

        // First 2 transfers return 0 (pipeline filling).
        let r0 = xfer(cmd_convert(10)); // pipeline slot 0
        assert_eq!(r0, 0);
        let r1 = xfer(cmd_convert(20)); // pipeline slot 1
        assert_eq!(r1, 0);

        // Third transfer returns the response for cmd_convert(10).
        let r2 = xfer(cmd_convert(0));
        assert_eq!(r2, 10 * 100 + 0x8000);

        // Fourth transfer returns the response for cmd_convert(20).
        let r3 = xfer(cmd_convert(0));
        assert_eq!(r3, 20 * 100 + 0x8000);
    }
}
