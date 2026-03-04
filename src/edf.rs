//! EDF/EDF+ file format parser.
//!
//! Parses European Data Format (EDF) and EDF+ files -- the most common
//! clinical EEG format. The parser is `no_std` compatible: it operates
//! on byte slices with caller-provided output buffers.
//!
//! EDF stores multi-channel physiological signals as 16-bit integers with
//! linear scaling to physical units. The format uses fixed-size ASCII headers
//! followed by sequential data records.

/// Errors during EDF parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdfError {
    /// Header is malformed or cannot be parsed
    InvalidHeader,
    /// File is shorter than expected
    TruncatedFile,
    /// Signal index out of range
    InvalidSignal,
    /// Output buffer too small
    BufferTooSmall,
    /// Invalid record index
    InvalidRecord,
}

/// Main EDF header (parsed from first 256 bytes).
#[derive(Debug, Clone)]
pub struct EdfHeader {
    /// Patient identification (80 bytes, space-padded ASCII)
    pub patient_id: [u8; 80],
    /// Recording identification (80 bytes, space-padded ASCII)
    pub recording_id: [u8; 80],
    /// Start date (dd.mm.yy)
    pub start_date: [u8; 8],
    /// Start time (hh.mm.ss)
    pub start_time: [u8; 8],
    /// Total number of header bytes (256 + 256 * n_signals)
    pub header_bytes: usize,
    /// Number of data records (-1 if unknown)
    pub n_records: i32,
    /// Duration of each data record in seconds
    pub record_duration: f64,
    /// Number of signals
    pub n_signals: usize,
    /// Whether this is an EDF+ file
    pub is_edf_plus: bool,
    /// Whether recording is continuous (true for EDF or EDF+C, false for EDF+D)
    pub is_continuous: bool,
}

impl EdfHeader {
    /// Patient ID as trimmed string.
    pub fn patient_id_str(&self) -> &str {
        let trimmed = trim_ascii(&self.patient_id);
        core::str::from_utf8(trimmed).unwrap_or("")
    }

    /// Recording ID as trimmed string.
    pub fn recording_id_str(&self) -> &str {
        let trimmed = trim_ascii(&self.recording_id);
        core::str::from_utf8(trimmed).unwrap_or("")
    }

    /// Total recording duration in seconds.
    pub fn duration(&self) -> f64 {
        if self.n_records >= 0 {
            self.n_records as f64 * self.record_duration
        } else {
            0.0
        }
    }
}

/// Signal header for one channel.
#[derive(Debug, Clone)]
pub struct EdfSignalHeader {
    /// Signal label (16 bytes, space-padded, e.g. "Fp1" or "EEG Fp1-Ref")
    pub label: [u8; 16],
    /// Transducer type (80 bytes, space-padded)
    pub transducer_type: [u8; 80],
    /// Physical dimension / units (8 bytes, e.g. "uV")
    pub physical_dimension: [u8; 8],
    /// Physical minimum value
    pub physical_min: f64,
    /// Physical maximum value
    pub physical_max: f64,
    /// Digital minimum value
    pub digital_min: i32,
    /// Digital maximum value
    pub digital_max: i32,
    /// Prefiltering description (80 bytes)
    pub prefiltering: [u8; 80],
    /// Number of samples in each data record
    pub n_samples_per_record: usize,
}

impl EdfSignalHeader {
    /// Signal label as trimmed string.
    pub fn label_str(&self) -> &str {
        let trimmed = trim_ascii(&self.label);
        core::str::from_utf8(trimmed).unwrap_or("")
    }

    /// Physical dimension as trimmed string.
    pub fn physical_dimension_str(&self) -> &str {
        let trimmed = trim_ascii(&self.physical_dimension);
        core::str::from_utf8(trimmed).unwrap_or("")
    }

    /// Whether this is an EDF+ annotations signal.
    pub fn is_annotation(&self) -> bool {
        self.label_str() == "EDF Annotations"
    }

    /// Sample rate derived from samples per record and record duration.
    pub fn sample_rate(&self, record_duration: f64) -> f64 {
        if record_duration > 0.0 {
            self.n_samples_per_record as f64 / record_duration
        } else {
            0.0
        }
    }
}

/// Parse the main EDF header from the first 256 bytes.
pub fn parse_header(bytes: &[u8]) -> Result<EdfHeader, EdfError> {
    if bytes.len() < 256 {
        return Err(EdfError::TruncatedFile);
    }

    // Version: bytes 0-7, must start with '0'
    if bytes[0] != b'0' {
        return Err(EdfError::InvalidHeader);
    }

    let mut patient_id = [b' '; 80];
    patient_id.copy_from_slice(&bytes[8..88]);

    let mut recording_id = [b' '; 80];
    recording_id.copy_from_slice(&bytes[88..168]);

    let mut start_date = [b' '; 8];
    start_date.copy_from_slice(&bytes[168..176]);

    let mut start_time = [b' '; 8];
    start_time.copy_from_slice(&bytes[176..184]);

    let header_bytes = parse_ascii_usize(&bytes[184..192])?;

    // Reserved: bytes 192-235 (44 bytes) -- EDF+ detection
    let reserved = &bytes[192..236];
    let is_edf_plus = reserved.starts_with(b"EDF+");
    let is_continuous = !reserved.starts_with(b"EDF+D");

    let n_records = parse_ascii_i32(&bytes[236..244])?;
    let record_duration = parse_ascii_f64(&bytes[244..252])?;
    let n_signals = parse_ascii_usize(&bytes[252..256])?;

    // Validate header_bytes matches expected value
    let expected = 256 + n_signals * 256;
    if header_bytes != expected {
        return Err(EdfError::InvalidHeader);
    }

    Ok(EdfHeader {
        patient_id,
        recording_id,
        start_date,
        start_time,
        header_bytes,
        n_records,
        record_duration,
        n_signals,
        is_edf_plus,
        is_continuous,
    })
}

/// Parse one signal header at the given index.
///
/// The `bytes` slice must contain the entire file (or at least the full header).
/// `n_signals` is the total number of signals from the main header.
pub fn parse_signal_header(
    bytes: &[u8],
    n_signals: usize,
    index: usize,
) -> Result<EdfSignalHeader, EdfError> {
    if index >= n_signals {
        return Err(EdfError::InvalidSignal);
    }
    let min_size = 256 + n_signals * 256;
    if bytes.len() < min_size {
        return Err(EdfError::TruncatedFile);
    }

    let base = 256usize;
    let ns = n_signals;

    let mut label = [b' '; 16];
    let off = base + index * 16;
    label.copy_from_slice(&bytes[off..off + 16]);

    let mut transducer_type = [b' '; 80];
    let off = base + ns * 16 + index * 80;
    transducer_type.copy_from_slice(&bytes[off..off + 80]);

    let mut physical_dimension = [b' '; 8];
    let off = base + ns * 96 + index * 8;
    physical_dimension.copy_from_slice(&bytes[off..off + 8]);

    let off = base + ns * 104 + index * 8;
    let physical_min = parse_ascii_f64(&bytes[off..off + 8])?;

    let off = base + ns * 112 + index * 8;
    let physical_max = parse_ascii_f64(&bytes[off..off + 8])?;

    let off = base + ns * 120 + index * 8;
    let digital_min = parse_ascii_i32(&bytes[off..off + 8])?;

    let off = base + ns * 128 + index * 8;
    let digital_max = parse_ascii_i32(&bytes[off..off + 8])?;

    let mut prefiltering = [b' '; 80];
    let off = base + ns * 136 + index * 80;
    prefiltering.copy_from_slice(&bytes[off..off + 80]);

    let off = base + ns * 216 + index * 8;
    let n_samples_per_record = parse_ascii_usize(&bytes[off..off + 8])?;

    Ok(EdfSignalHeader {
        label,
        transducer_type,
        physical_dimension,
        physical_min,
        physical_max,
        digital_min,
        digital_max,
        prefiltering,
        n_samples_per_record,
    })
}

/// Compute the byte size of one data record.
pub fn record_byte_size(signals: &[EdfSignalHeader]) -> usize {
    signals.iter().map(|s| s.n_samples_per_record * 2).sum()
}

/// Read one channel from one data record, converting to physical units.
///
/// `output` must have at least `signals[channel].n_samples_per_record` elements.
pub fn read_record(
    bytes: &[u8],
    header: &EdfHeader,
    signals: &[EdfSignalHeader],
    record_idx: usize,
    channel: usize,
    output: &mut [f64],
) -> Result<(), EdfError> {
    if channel >= header.n_signals || channel >= signals.len() {
        return Err(EdfError::InvalidSignal);
    }
    if header.n_records < 0 || record_idx >= header.n_records as usize {
        return Err(EdfError::InvalidRecord);
    }

    let n_samples = signals[channel].n_samples_per_record;
    if output.len() < n_samples {
        return Err(EdfError::BufferTooSmall);
    }

    // Byte offset of this channel within a data record
    let mut channel_byte_offset = 0usize;
    for sig in signals.iter().take(channel) {
        channel_byte_offset += sig.n_samples_per_record * 2;
    }

    let rec_size = record_byte_size(signals);
    let file_offset = header.header_bytes + record_idx * rec_size + channel_byte_offset;
    let end_offset = file_offset + n_samples * 2;
    if end_offset > bytes.len() {
        return Err(EdfError::TruncatedFile);
    }

    // Precompute scaling
    let sig = &signals[channel];
    let digital_range = (sig.digital_max - sig.digital_min) as f64;
    if digital_range == 0.0 {
        // Degenerate: all samples map to physical_min
        for out in output.iter_mut().take(n_samples) {
            *out = sig.physical_min;
        }
        return Ok(());
    }
    let gain = (sig.physical_max - sig.physical_min) / digital_range;

    for (i, out) in output.iter_mut().enumerate().take(n_samples) {
        let byte_idx = file_offset + i * 2;
        let digital = i16::from_le_bytes([bytes[byte_idx], bytes[byte_idx + 1]]) as i32;
        *out = (digital - sig.digital_min) as f64 * gain + sig.physical_min;
    }

    Ok(())
}

/// Read an entire channel across all data records, converting to physical units.
///
/// `output` must have at least `n_records * signals[channel].n_samples_per_record` elements.
/// Returns the total number of samples written.
pub fn read_channel(
    bytes: &[u8],
    header: &EdfHeader,
    signals: &[EdfSignalHeader],
    channel: usize,
    output: &mut [f64],
) -> Result<usize, EdfError> {
    if channel >= header.n_signals || channel >= signals.len() {
        return Err(EdfError::InvalidSignal);
    }
    if header.n_records < 0 {
        return Err(EdfError::InvalidHeader);
    }

    let n_records = header.n_records as usize;
    let n_samples = signals[channel].n_samples_per_record;
    let total_samples = n_records * n_samples;
    if output.len() < total_samples {
        return Err(EdfError::BufferTooSmall);
    }

    for r in 0..n_records {
        let out_start = r * n_samples;
        read_record(
            bytes,
            header,
            signals,
            r,
            channel,
            &mut output[out_start..out_start + n_samples],
        )?;
    }

    Ok(total_samples)
}

// --- Private ASCII parsing helpers ---

/// Trim leading and trailing ASCII spaces.
fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|&b| b != b' ').unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|&b| b != b' ')
        .map(|p| p + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

/// Parse a space-padded ASCII field as i32.
fn parse_ascii_i32(bytes: &[u8]) -> Result<i32, EdfError> {
    let s = trim_ascii(bytes);
    if s.is_empty() {
        return Err(EdfError::InvalidHeader);
    }
    let mut neg = false;
    let mut i = 0;
    if s[0] == b'-' {
        neg = true;
        i = 1;
    } else if s[0] == b'+' {
        i = 1;
    }
    let mut val: i32 = 0;
    while i < s.len() {
        if s[i] < b'0' || s[i] > b'9' {
            return Err(EdfError::InvalidHeader);
        }
        val = val * 10 + (s[i] - b'0') as i32;
        i += 1;
    }
    if neg {
        Ok(-val)
    } else {
        Ok(val)
    }
}

/// Parse a space-padded ASCII field as usize (non-negative integer).
fn parse_ascii_usize(bytes: &[u8]) -> Result<usize, EdfError> {
    let val = parse_ascii_i32(bytes)?;
    if val < 0 {
        return Err(EdfError::InvalidHeader);
    }
    Ok(val as usize)
}

/// Parse a space-padded ASCII field as f64.
///
/// Handles simple decimal format: optional sign, integer part, optional decimal point + fraction.
fn parse_ascii_f64(bytes: &[u8]) -> Result<f64, EdfError> {
    let s = trim_ascii(bytes);
    if s.is_empty() {
        return Err(EdfError::InvalidHeader);
    }
    let mut neg = false;
    let mut i = 0;
    if s[0] == b'-' {
        neg = true;
        i = 1;
    } else if s[0] == b'+' {
        i = 1;
    }

    let mut int_part: f64 = 0.0;
    while i < s.len() && s[i] != b'.' {
        if s[i] < b'0' || s[i] > b'9' {
            return Err(EdfError::InvalidHeader);
        }
        int_part = int_part * 10.0 + (s[i] - b'0') as f64;
        i += 1;
    }

    let mut frac_part: f64 = 0.0;
    if i < s.len() && s[i] == b'.' {
        i += 1;
        let mut factor = 0.1;
        while i < s.len() {
            if s[i] < b'0' || s[i] > b'9' {
                return Err(EdfError::InvalidHeader);
            }
            frac_part += (s[i] - b'0') as f64 * factor;
            factor *= 0.1;
            i += 1;
        }
    }

    let val = int_part + frac_part;
    if neg {
        Ok(-val)
    } else {
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    // Write an integer as left-justified ASCII into a space-padded field
    fn write_int_field(buf: &mut [u8], val: i64) {
        for b in buf.iter_mut() {
            *b = b' ';
        }
        if val == 0 {
            buf[0] = b'0';
            return;
        }
        let neg = val < 0;
        let mut v = if neg { (-val) as u64 } else { val as u64 };
        let mut tmp = [0u8; 20];
        let mut len = 0;
        while v > 0 {
            tmp[len] = b'0' + (v % 10) as u8;
            v /= 10;
            len += 1;
        }
        let mut pos = 0;
        if neg {
            buf[pos] = b'-';
            pos += 1;
        }
        for i in (0..len).rev() {
            buf[pos] = tmp[i];
            pos += 1;
        }
    }

    // Write a string (as bytes) into a space-padded field
    fn write_str_field(buf: &mut [u8], s: &[u8]) {
        for b in buf.iter_mut() {
            *b = b' ';
        }
        let len = s.len().min(buf.len());
        buf[..len].copy_from_slice(&s[..len]);
    }

    /// Build a synthetic EDF file with known data.
    fn make_edf(
        labels: &[&str],
        samples_per_record: &[usize],
        n_records: usize,
        record_duration: f64,
        physical_range: (f64, f64),
        digital_range: (i32, i32),
        data: Option<&[&[i16]]>, // data[channel][all_samples_across_records]
    ) -> Vec<u8> {
        let ns = labels.len();
        let header_bytes = 256 + ns * 256;
        let rec_size: usize = samples_per_record.iter().map(|&s| s * 2).sum();
        let total = header_bytes + n_records * rec_size;
        let mut buf = vec![b' '; total];

        // Main header
        buf[0] = b'0'; // version
        write_str_field(&mut buf[8..88], b"Test Patient");
        write_str_field(&mut buf[88..168], b"Test Recording");
        write_str_field(&mut buf[168..176], b"01.01.26");
        write_str_field(&mut buf[176..184], b"12.00.00");
        write_int_field(&mut buf[184..192], header_bytes as i64);
        // reserved[192..236] = spaces (standard EDF)
        write_int_field(&mut buf[236..244], n_records as i64);
        // record_duration as integer (sufficient for tests)
        write_int_field(&mut buf[244..252], record_duration as i64);
        write_int_field(&mut buf[252..256], ns as i64);

        let base = 256;

        // Signal headers (interleaved across signals)
        for (i, &lbl) in labels.iter().enumerate() {
            // Label (16 bytes each)
            write_str_field(&mut buf[base + i * 16..base + (i + 1) * 16], lbl.as_bytes());
            // Transducer (80 bytes each)
            let off = base + ns * 16 + i * 80;
            write_str_field(&mut buf[off..off + 80], b"AgAgCl");
            // Physical dimension (8 bytes each)
            let off = base + ns * 96 + i * 8;
            write_str_field(&mut buf[off..off + 8], b"uV");
            // Physical min (8 bytes each)
            let off = base + ns * 104 + i * 8;
            write_int_field(&mut buf[off..off + 8], physical_range.0 as i64);
            // Physical max (8 bytes each)
            let off = base + ns * 112 + i * 8;
            write_int_field(&mut buf[off..off + 8], physical_range.1 as i64);
            // Digital min (8 bytes each)
            let off = base + ns * 120 + i * 8;
            write_int_field(&mut buf[off..off + 8], digital_range.0 as i64);
            // Digital max (8 bytes each)
            let off = base + ns * 128 + i * 8;
            write_int_field(&mut buf[off..off + 8], digital_range.1 as i64);
            // Prefiltering (80 bytes each)
            let off = base + ns * 136 + i * 80;
            write_str_field(&mut buf[off..off + 80], b"HP:0.1Hz");
            // N samples per record (8 bytes each)
            let off = base + ns * 216 + i * 8;
            write_int_field(&mut buf[off..off + 8], samples_per_record[i] as i64);
            // Reserved (32 bytes each) -- already spaces
        }

        // Data records
        for r in 0..n_records {
            let mut rec_offset = header_bytes + r * rec_size;
            for (ch, &spr) in samples_per_record.iter().enumerate() {
                for s in 0..spr {
                    let val = if let Some(d) = data {
                        d[ch][r * spr + s]
                    } else {
                        0i16
                    };
                    let bytes = val.to_le_bytes();
                    buf[rec_offset] = bytes[0];
                    buf[rec_offset + 1] = bytes[1];
                    rec_offset += 2;
                }
            }
        }

        buf
    }

    #[test]
    fn test_parse_header() {
        let edf = make_edf(
            &["Fp1", "Fp2"],
            &[256, 256],
            10,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        let header = parse_header(&edf).unwrap();
        assert_eq!(header.n_signals, 2);
        assert_eq!(header.n_records, 10);
        assert!((header.record_duration - 1.0).abs() < 1e-10);
        assert_eq!(header.header_bytes, 256 + 2 * 256);
        assert!(!header.is_edf_plus);
        assert!(header.is_continuous);
        assert_eq!(header.patient_id_str(), "Test Patient");
        assert_eq!(header.recording_id_str(), "Test Recording");
    }

    #[test]
    fn test_parse_signal_headers() {
        let edf = make_edf(
            &["EEG Fp1", "EEG Fp2"],
            &[256, 256],
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        let header = parse_header(&edf).unwrap();

        let sig0 = parse_signal_header(&edf, header.n_signals, 0).unwrap();
        assert_eq!(sig0.label_str(), "EEG Fp1");
        assert_eq!(sig0.physical_dimension_str(), "uV");
        assert!((sig0.physical_min - (-3200.0)).abs() < 1e-10);
        assert!((sig0.physical_max - 3200.0).abs() < 1e-10);
        assert_eq!(sig0.digital_min, -32768);
        assert_eq!(sig0.digital_max, 32767);
        assert_eq!(sig0.n_samples_per_record, 256);
        assert!((sig0.sample_rate(1.0) - 256.0).abs() < 1e-10);

        let sig1 = parse_signal_header(&edf, header.n_signals, 1).unwrap();
        assert_eq!(sig1.label_str(), "EEG Fp2");
    }

    #[test]
    fn test_invalid_header_too_short() {
        let buf = vec![0u8; 100];
        assert!(matches!(parse_header(&buf), Err(EdfError::TruncatedFile)));
    }

    #[test]
    fn test_invalid_header_bad_version() {
        let mut edf = make_edf(
            &["Ch1"],
            &[256],
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        edf[0] = b'X'; // corrupt version
        assert!(matches!(parse_header(&edf), Err(EdfError::InvalidHeader)));
    }

    #[test]
    fn test_physical_scaling() {
        // digital_min=-32768, digital_max=32767, physical_min=-3200, physical_max=3200
        // digital 0 -> physical ~0.0 (slight offset due to asymmetric digital range)
        // digital 32767 -> physical 3200.0
        // digital -32768 -> physical -3200.0
        let data_ch0: Vec<i16> = vec![0, 32767, -32768, 16384];
        let edf = make_edf(
            &["Ch1"],
            &[4],
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            Some(&[&data_ch0]),
        );
        let header = parse_header(&edf).unwrap();
        let sig = parse_signal_header(&edf, header.n_signals, 0).unwrap();

        let mut output = [0.0f64; 4];
        read_record(&edf, &header, &[sig], 0, 0, &mut output).unwrap();

        // digital -32768 -> physical -3200
        assert!((output[2] - (-3200.0)).abs() < 0.1);
        // digital 32767 -> physical 3200
        assert!((output[1] - 3200.0).abs() < 0.1);
        // digital 0 -> physical ~0 (slight asymmetry)
        assert!(output[0].abs() < 1.0);
        // digital 16384 -> ~1600
        assert!((output[3] - 1600.0).abs() < 1.0);
    }

    #[test]
    fn test_read_channel_all_records() {
        let n_records = 3;
        let spr = 4;
        // Channel 0 data across 3 records: [0,1,2,3, 4,5,6,7, 8,9,10,11]
        let data_ch0: Vec<i16> = (0..12).collect();
        let data_ch1: Vec<i16> = (100..112).collect();
        let edf = make_edf(
            &["Ch0", "Ch1"],
            &[spr, spr],
            n_records,
            1.0,
            (0.0, 100.0),
            (0, 100),
            Some(&[&data_ch0, &data_ch1]),
        );
        let header = parse_header(&edf).unwrap();
        let sig0 = parse_signal_header(&edf, header.n_signals, 0).unwrap();
        let sig1 = parse_signal_header(&edf, header.n_signals, 1).unwrap();
        let signals = [sig0, sig1];

        let mut output = [0.0f64; 12];
        let n = read_channel(&edf, &header, &signals, 0, &mut output).unwrap();
        assert_eq!(n, 12);

        // With physical=digital (range 0-100, digital 0-100): physical = digital
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - i as f64).abs() < 0.01,
                "sample {} = {}, expected {}",
                i,
                val,
                i
            );
        }
    }

    #[test]
    fn test_different_sample_rates() {
        // Channel 0: 256 samples/record, Channel 1: 128 samples/record
        let spr = [256, 128];
        let data_ch0: Vec<i16> = vec![0; 256];
        let data_ch1: Vec<i16> = vec![100; 128];
        let edf = make_edf(
            &["EEG", "EOG"],
            &spr,
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            Some(&[&data_ch0, &data_ch1]),
        );
        let header = parse_header(&edf).unwrap();
        let sig0 = parse_signal_header(&edf, header.n_signals, 0).unwrap();
        let sig1 = parse_signal_header(&edf, header.n_signals, 1).unwrap();

        assert_eq!(sig0.n_samples_per_record, 256);
        assert_eq!(sig1.n_samples_per_record, 128);
        assert!((sig0.sample_rate(1.0) - 256.0).abs() < 1e-10);
        assert!((sig1.sample_rate(1.0) - 128.0).abs() < 1e-10);

        // Read channel 1
        let mut output = [0.0f64; 128];
        read_record(&edf, &header, &[sig0, sig1], 0, 1, &mut output).unwrap();
        // All values should be consistent (digital 100 maps to some physical value)
        let expected = output[0];
        for &val in &output[1..] {
            assert!((val - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_edf_plus_detection() {
        let mut edf = make_edf(
            &["Ch1"],
            &[256],
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        // Write "EDF+C" into reserved field
        edf[192..197].copy_from_slice(b"EDF+C");
        let header = parse_header(&edf).unwrap();
        assert!(header.is_edf_plus);
        assert!(header.is_continuous);

        // Write "EDF+D" for discontinuous
        edf[192..197].copy_from_slice(b"EDF+D");
        let header = parse_header(&edf).unwrap();
        assert!(header.is_edf_plus);
        assert!(!header.is_continuous);
    }

    #[test]
    fn test_single_record_single_signal() {
        let data: Vec<i16> = vec![1000, -1000, 0, 500];
        let edf = make_edf(
            &["X"],
            &[4],
            1,
            2.0,
            (-500.0, 500.0),
            (-32768, 32767),
            Some(&[&data]),
        );
        let header = parse_header(&edf).unwrap();
        assert_eq!(header.n_signals, 1);
        assert!((header.record_duration - 2.0).abs() < 1e-10);
        assert!((header.duration() - 2.0).abs() < 1e-10);

        let sig = parse_signal_header(&edf, header.n_signals, 0).unwrap();
        assert!((sig.sample_rate(2.0) - 2.0).abs() < 1e-10);

        let mut output = [0.0f64; 4];
        read_record(&edf, &header, &[sig], 0, 0, &mut output).unwrap();

        // digital 1000 with range [-32768, 32767] -> physical
        // gain = 1000 / 65535 = ~0.01526
        // physical = (1000 - (-32768)) * gain + (-500)
        //          = 33768 * 0.01526... - 500 = 515.27... - 500 = 15.27
        assert!(output[0] > 0.0, "digital 1000 should map positive");
        assert!(output[1] < 0.0, "digital -1000 should map negative");
    }

    #[test]
    fn test_truncated_file_error() {
        let edf = make_edf(
            &["Ch1"],
            &[256],
            10,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        let header = parse_header(&edf).unwrap();
        let sig = parse_signal_header(&edf, header.n_signals, 0).unwrap();

        // Truncate the file to cut off data records
        let truncated = &edf[..header.header_bytes + 100]; // not enough for a full record
        let mut output = [0.0f64; 256];
        assert_eq!(
            read_record(truncated, &header, &[sig], 0, 0, &mut output),
            Err(EdfError::TruncatedFile)
        );
    }

    #[test]
    fn test_invalid_signal_index() {
        let edf = make_edf(
            &["Ch1"],
            &[256],
            1,
            1.0,
            (-3200.0, 3200.0),
            (-32768, 32767),
            None,
        );
        assert!(matches!(
            parse_signal_header(&edf, 1, 5),
            Err(EdfError::InvalidSignal)
        ));
    }

    #[test]
    fn test_ascii_parsing() {
        assert_eq!(parse_ascii_i32(b"  42  "), Ok(42));
        assert_eq!(parse_ascii_i32(b"-1      "), Ok(-1));
        assert_eq!(parse_ascii_i32(b"0       "), Ok(0));
        assert!((parse_ascii_f64(b"2.75    ").unwrap() - 2.75).abs() < 0.001);
        assert!((parse_ascii_f64(b"-3200   ").unwrap() - (-3200.0)).abs() < 0.001);
        assert!((parse_ascii_f64(b"1       ").unwrap() - 1.0).abs() < 1e-10);
    }
}
