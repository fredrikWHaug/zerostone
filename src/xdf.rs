//! XDF (Extensible Data Format) file parser.
//!
//! Parses XDF files produced by Lab Streaming Layer (LSL). The parser is
//! `no_std` compatible: it operates on byte slices with zero allocation
//! in the core chunk iteration path.
//!
//! XDF stores multi-stream data (EEG + markers + auxiliary sensors) as
//! sequential chunks with variable-length integer size encoding.

/// Errors during XDF parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdfError {
    /// File does not start with "XDF:" magic
    InvalidMagic,
    /// File is shorter than expected
    TruncatedFile,
    /// Chunk structure is invalid
    InvalidChunk,
    /// XML metadata cannot be parsed
    InvalidXml,
    /// Variable-length integer has invalid prefix
    InvalidVarLen,
    /// Channel format is not supported
    UnsupportedFormat,
    /// Requested stream was not found
    StreamNotFound,
    /// Output buffer is too small
    BufferTooSmall,
}

/// XDF channel data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdfChannelFormat {
    Float32,
    Double64,
    Int8,
    Int16,
    Int32,
    Int64,
    StringFormat,
}

impl XdfChannelFormat {
    /// Parse from the string representation used in XDF XML.
    pub fn from_bytes(s: &[u8]) -> Result<Self, XdfError> {
        match s {
            b"float32" => Ok(Self::Float32),
            b"double64" => Ok(Self::Double64),
            b"int8" => Ok(Self::Int8),
            b"int16" => Ok(Self::Int16),
            b"int32" => Ok(Self::Int32),
            b"int64" => Ok(Self::Int64),
            b"string" => Ok(Self::StringFormat),
            _ => Err(XdfError::UnsupportedFormat),
        }
    }

    /// Byte size per sample value (0 for string format).
    pub fn sample_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Double64 => 8,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::StringFormat => 0,
        }
    }
}

/// Parsed stream metadata from a StreamHeader chunk.
#[derive(Debug, Clone)]
pub struct XdfStreamInfo<'a> {
    pub stream_id: u32,
    pub name: &'a [u8],
    pub stream_type: &'a [u8],
    pub channel_count: u32,
    pub nominal_srate: f64,
    pub channel_format: XdfChannelFormat,
}

/// A single parsed chunk from the XDF file.
#[derive(Debug, Clone)]
pub struct XdfChunk<'a> {
    pub tag: u16,
    pub stream_id: Option<u32>,
    pub content: &'a [u8],
}

/// A clock offset measurement pair.
#[derive(Debug, Clone, Copy)]
pub struct ClockOffsetPair {
    pub collection_time: f64,
    pub offset_value: f64,
}

// Chunk tag constants
pub const TAG_FILE_HEADER: u16 = 1;
pub const TAG_STREAM_HEADER: u16 = 2;
pub const TAG_SAMPLES: u16 = 3;
pub const TAG_CLOCK_OFFSET: u16 = 4;
pub const TAG_BOUNDARY: u16 = 5;
pub const TAG_STREAM_FOOTER: u16 = 6;

/// Validate the XDF magic bytes ("XDF:") at the start of the file.
pub fn validate_magic(bytes: &[u8]) -> Result<(), XdfError> {
    if bytes.len() < 4 {
        return Err(XdfError::TruncatedFile);
    }
    if &bytes[0..4] != b"XDF:" {
        return Err(XdfError::InvalidMagic);
    }
    Ok(())
}

/// Read a variable-length integer from `bytes` at `*pos`, advancing `*pos`.
///
/// XDF varlen encoding: first byte is 1, 4, or 8, then that many bytes LE unsigned.
pub fn read_varlen(bytes: &[u8], pos: &mut usize) -> Result<u64, XdfError> {
    if *pos >= bytes.len() {
        return Err(XdfError::TruncatedFile);
    }
    let prefix = bytes[*pos];
    *pos += 1;

    match prefix {
        1 => {
            if *pos >= bytes.len() {
                return Err(XdfError::TruncatedFile);
            }
            let val = bytes[*pos] as u64;
            *pos += 1;
            Ok(val)
        }
        4 => {
            if *pos + 4 > bytes.len() {
                return Err(XdfError::TruncatedFile);
            }
            let val = u32::from_le_bytes([
                bytes[*pos],
                bytes[*pos + 1],
                bytes[*pos + 2],
                bytes[*pos + 3],
            ]) as u64;
            *pos += 4;
            Ok(val)
        }
        8 => {
            if *pos + 8 > bytes.len() {
                return Err(XdfError::TruncatedFile);
            }
            let val = u64::from_le_bytes([
                bytes[*pos],
                bytes[*pos + 1],
                bytes[*pos + 2],
                bytes[*pos + 3],
                bytes[*pos + 4],
                bytes[*pos + 5],
                bytes[*pos + 6],
                bytes[*pos + 7],
            ]);
            *pos += 8;
            Ok(val)
        }
        _ => Err(XdfError::InvalidVarLen),
    }
}

/// Parse the next chunk from the XDF file, advancing `*pos`.
///
/// Returns `Ok(None)` at end of file.
pub fn next_chunk<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<Option<XdfChunk<'a>>, XdfError> {
    if *pos >= bytes.len() {
        return Ok(None);
    }

    let chunk_len = read_varlen(bytes, pos)? as usize;
    if *pos + chunk_len > bytes.len() {
        return Err(XdfError::TruncatedFile);
    }

    if chunk_len < 2 {
        return Err(XdfError::InvalidChunk);
    }

    let tag = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
    let content_start = *pos + 2;
    let content_end = *pos + chunk_len;
    let content = &bytes[content_start..content_end];

    // Tags 2-6 have a stream_id as the first 4 bytes of content
    let stream_id = if (TAG_STREAM_HEADER..=TAG_STREAM_FOOTER).contains(&tag) && content.len() >= 4
    {
        Some(u32::from_le_bytes([
            content[0], content[1], content[2], content[3],
        ]))
    } else {
        None
    };

    *pos = content_end;

    Ok(Some(XdfChunk {
        tag,
        stream_id,
        content,
    }))
}

/// Extract the text content of a simple XML tag from a byte slice.
///
/// Finds `<tag>value</tag>` and returns `value` as a byte slice.
/// Only handles top-level, non-nested tags. Sufficient for XDF stream headers.
pub fn find_tag_value<'a>(xml: &'a [u8], tag: &[u8]) -> Option<&'a [u8]> {
    // Build "<tag>" pattern
    let mut open = [0u8; 64];
    if tag.len() + 2 > open.len() {
        return None;
    }
    open[0] = b'<';
    open[1..1 + tag.len()].copy_from_slice(tag);
    open[1 + tag.len()] = b'>';
    let open_len = tag.len() + 2;
    let open_pat = &open[..open_len];

    // Build "</tag>" pattern
    let mut close = [0u8; 65];
    if tag.len() + 3 > close.len() {
        return None;
    }
    close[0] = b'<';
    close[1] = b'/';
    close[2..2 + tag.len()].copy_from_slice(tag);
    close[2 + tag.len()] = b'>';
    let close_len = tag.len() + 3;
    let close_pat = &close[..close_len];

    // Search for open tag
    let start = find_subsequence(xml, open_pat)?;
    let value_start = start + open_len;

    // Search for close tag after the value
    let remaining = &xml[value_start..];
    let end = find_subsequence(remaining, close_pat)?;

    Some(&xml[value_start..value_start + end])
}

/// Find the first occurrence of `needle` in `haystack`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=(haystack.len() - needle.len())).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Parse a decimal number from ASCII bytes (no_std f64 parser).
fn parse_f64_bytes(bytes: &[u8]) -> Option<f64> {
    if bytes.is_empty() {
        return None;
    }
    let mut i = 0;
    let neg = if bytes[0] == b'-' {
        i = 1;
        true
    } else if bytes[0] == b'+' {
        i = 1;
        false
    } else {
        false
    };

    let mut int_part: f64 = 0.0;
    while i < bytes.len() && bytes[i] != b'.' {
        if !bytes[i].is_ascii_digit() {
            return None;
        }
        int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
        i += 1;
    }

    let mut frac_part: f64 = 0.0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let mut factor = 0.1;
        while i < bytes.len() {
            if !bytes[i].is_ascii_digit() {
                return None;
            }
            frac_part += (bytes[i] - b'0') as f64 * factor;
            factor *= 0.1;
            i += 1;
        }
    }

    let val = int_part + frac_part;
    Some(if neg { -val } else { val })
}

/// Parse an unsigned integer from ASCII bytes.
fn parse_u32_bytes(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() {
        return None;
    }
    let mut val: u32 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        val = val * 10 + (b - b'0') as u32;
    }
    Some(val)
}

/// Parse stream metadata from a StreamHeader chunk's XML content.
///
/// The `content` slice should be the chunk content after the stream_id (i.e., the XML).
pub fn parse_stream_info<'a>(xml: &'a [u8], stream_id: u32) -> Result<XdfStreamInfo<'a>, XdfError> {
    let name = find_tag_value(xml, b"name").ok_or(XdfError::InvalidXml)?;
    let stream_type = find_tag_value(xml, b"type").unwrap_or(b"");
    let channel_count_str = find_tag_value(xml, b"channel_count").ok_or(XdfError::InvalidXml)?;
    let channel_count = parse_u32_bytes(channel_count_str).ok_or(XdfError::InvalidXml)?;

    let nominal_srate = if let Some(srate_str) = find_tag_value(xml, b"nominal_srate") {
        parse_f64_bytes(srate_str).unwrap_or(0.0)
    } else {
        0.0
    };

    let format_str = find_tag_value(xml, b"channel_format").ok_or(XdfError::InvalidXml)?;
    let channel_format = XdfChannelFormat::from_bytes(format_str)?;

    Ok(XdfStreamInfo {
        stream_id,
        name,
        stream_type,
        channel_count,
        nominal_srate,
        channel_format,
    })
}

/// Parse a ClockOffset chunk's content (after the 4-byte stream_id).
///
/// Content layout: collection_time (f64 LE) + offset_value (f64 LE) = 16 bytes.
pub fn parse_clock_offset(content: &[u8]) -> Result<ClockOffsetPair, XdfError> {
    if content.len() < 16 {
        return Err(XdfError::TruncatedFile);
    }
    let collection_time = f64::from_le_bytes([
        content[0], content[1], content[2], content[3], content[4], content[5], content[6],
        content[7],
    ]);
    let offset_value = f64::from_le_bytes([
        content[8],
        content[9],
        content[10],
        content[11],
        content[12],
        content[13],
        content[14],
        content[15],
    ]);
    Ok(ClockOffsetPair {
        collection_time,
        offset_value,
    })
}

/// Decode samples from a Samples chunk into f64 buffers.
///
/// The `content` slice is the chunk content after the 4-byte stream_id.
/// Each sample: 1-byte timestamp flag, optional 8-byte timestamp, then channel values.
/// If timestamp flag byte is 8, the next 8 bytes are the timestamp (f64 LE).
/// If timestamp flag byte is 0, no timestamp is present (NaN is written).
///
/// Returns the number of samples decoded.
pub fn decode_samples_f64(
    content: &[u8],
    channel_count: u32,
    format: &XdfChannelFormat,
    values: &mut [f64],
    timestamps: &mut [f64],
) -> Result<usize, XdfError> {
    if matches!(format, XdfChannelFormat::StringFormat) {
        return Err(XdfError::UnsupportedFormat);
    }

    let ch = channel_count as usize;
    let sample_bytes = format.sample_bytes();
    let mut pos = 0usize;
    let mut sample_idx = 0usize;

    while pos < content.len() {
        // Timestamp flag byte
        if pos >= content.len() {
            break;
        }
        let ts_flag = content[pos];
        pos += 1;

        // Read timestamp
        let ts = if ts_flag == 8 {
            if pos + 8 > content.len() {
                return Err(XdfError::TruncatedFile);
            }
            let t = f64::from_le_bytes([
                content[pos],
                content[pos + 1],
                content[pos + 2],
                content[pos + 3],
                content[pos + 4],
                content[pos + 5],
                content[pos + 6],
                content[pos + 7],
            ]);
            pos += 8;
            t
        } else {
            f64::NAN
        };

        // Check output capacity
        let val_start = sample_idx * ch;
        if val_start + ch > values.len() || sample_idx >= timestamps.len() {
            return Err(XdfError::BufferTooSmall);
        }

        timestamps[sample_idx] = ts;

        // Read channel values
        let needed = ch * sample_bytes;
        if pos + needed > content.len() {
            return Err(XdfError::TruncatedFile);
        }

        for c in 0..ch {
            let off = pos + c * sample_bytes;
            let v = match format {
                XdfChannelFormat::Float32 => f32::from_le_bytes([
                    content[off],
                    content[off + 1],
                    content[off + 2],
                    content[off + 3],
                ]) as f64,
                XdfChannelFormat::Double64 => f64::from_le_bytes([
                    content[off],
                    content[off + 1],
                    content[off + 2],
                    content[off + 3],
                    content[off + 4],
                    content[off + 5],
                    content[off + 6],
                    content[off + 7],
                ]),
                XdfChannelFormat::Int8 => content[off] as i8 as f64,
                XdfChannelFormat::Int16 => {
                    i16::from_le_bytes([content[off], content[off + 1]]) as f64
                }
                XdfChannelFormat::Int32 => i32::from_le_bytes([
                    content[off],
                    content[off + 1],
                    content[off + 2],
                    content[off + 3],
                ]) as f64,
                XdfChannelFormat::Int64 => i64::from_le_bytes([
                    content[off],
                    content[off + 1],
                    content[off + 2],
                    content[off + 3],
                    content[off + 4],
                    content[off + 5],
                    content[off + 6],
                    content[off + 7],
                ]) as f64,
                XdfChannelFormat::StringFormat => unreachable!(),
            };
            values[val_start + c] = v;
        }
        pos += needed;
        sample_idx += 1;
    }

    Ok(sample_idx)
}

/// Count the number of samples in a Samples chunk without decoding values.
///
/// The `content` slice is the chunk content after the 4-byte stream_id.
pub fn count_chunk_samples(
    content: &[u8],
    channel_count: u32,
    format: &XdfChannelFormat,
) -> Result<usize, XdfError> {
    if matches!(format, XdfChannelFormat::StringFormat) {
        return count_chunk_samples_string(content, channel_count);
    }

    let ch = channel_count as usize;
    let sample_bytes = format.sample_bytes();
    let row_bytes = ch * sample_bytes;
    let mut pos = 0usize;
    let mut count = 0usize;

    while pos < content.len() {
        let ts_flag = content[pos];
        pos += 1;
        if ts_flag == 8 {
            pos += 8;
        }
        pos += row_bytes;
        if pos > content.len() {
            return Err(XdfError::TruncatedFile);
        }
        count += 1;
    }

    Ok(count)
}

/// Count samples in a string-format Samples chunk.
fn count_chunk_samples_string(content: &[u8], channel_count: u32) -> Result<usize, XdfError> {
    let ch = channel_count as usize;
    let mut pos = 0usize;
    let mut count = 0usize;

    while pos < content.len() {
        // Timestamp flag
        if pos >= content.len() {
            break;
        }
        let ts_flag = content[pos];
        pos += 1;
        if ts_flag == 8 {
            pos += 8;
        }
        if pos > content.len() {
            return Err(XdfError::TruncatedFile);
        }
        // For each channel, read varlen-prefixed string
        for _ in 0..ch {
            let str_len = read_varlen(content, &mut pos)? as usize;
            pos += str_len;
            if pos > content.len() {
                return Err(XdfError::TruncatedFile);
            }
        }
        count += 1;
    }

    Ok(count)
}

/// Decode string samples from a Samples chunk.
///
/// Returns a flat Vec of (timestamp, Vec<string_bytes>) tuples would require alloc.
/// Instead, calls a callback for each sample with (timestamp, channel_index, string_bytes).
/// This is used by the Python bindings which handle collection.
///
/// The `content` slice is the chunk content after the 4-byte stream_id.
/// Returns the number of samples decoded.
pub fn decode_samples_string<F>(
    content: &[u8],
    channel_count: u32,
    timestamps: &mut [f64],
    mut callback: F,
) -> Result<usize, XdfError>
where
    F: FnMut(usize, usize, &[u8]),
{
    let ch = channel_count as usize;
    let mut pos = 0usize;
    let mut sample_idx = 0usize;

    while pos < content.len() {
        let ts_flag = content[pos];
        pos += 1;

        let ts = if ts_flag == 8 {
            if pos + 8 > content.len() {
                return Err(XdfError::TruncatedFile);
            }
            let t = f64::from_le_bytes([
                content[pos],
                content[pos + 1],
                content[pos + 2],
                content[pos + 3],
                content[pos + 4],
                content[pos + 5],
                content[pos + 6],
                content[pos + 7],
            ]);
            pos += 8;
            t
        } else {
            f64::NAN
        };

        if sample_idx >= timestamps.len() {
            return Err(XdfError::BufferTooSmall);
        }
        timestamps[sample_idx] = ts;

        for c in 0..ch {
            let str_len = read_varlen(content, &mut pos)? as usize;
            if pos + str_len > content.len() {
                return Err(XdfError::TruncatedFile);
            }
            callback(sample_idx, c, &content[pos..pos + str_len]);
            pos += str_len;
        }
        sample_idx += 1;
    }

    Ok(sample_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a varlen-encoded integer.
    fn varlen_bytes(val: u64) -> Vec<u8> {
        if val <= 255 {
            let mut v = vec![1u8];
            v.push(val as u8);
            v
        } else if val <= u32::MAX as u64 {
            let mut v = vec![4u8];
            v.extend_from_slice(&(val as u32).to_le_bytes());
            v
        } else {
            let mut v = vec![8u8];
            v.extend_from_slice(&val.to_le_bytes());
            v
        }
    }

    /// Build a minimal chunk (varlen length + tag + content).
    fn make_chunk(tag: u16, content: &[u8]) -> Vec<u8> {
        let chunk_len = 2 + content.len();
        let mut v = varlen_bytes(chunk_len as u64);
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(content);
        v
    }

    /// Build a StreamHeader chunk with XML content.
    fn make_stream_header(
        stream_id: u32,
        name: &str,
        stype: &str,
        ch_count: u32,
        srate: f64,
        format: &str,
    ) -> Vec<u8> {
        let xml = alloc::format!(
            "<?xml version=\"1.0\"?><info><name>{}</name><type>{}</type>\
             <channel_count>{}</channel_count><nominal_srate>{}</nominal_srate>\
             <channel_format>{}</channel_format></info>",
            name,
            stype,
            ch_count,
            srate,
            format
        );
        let mut content = stream_id.to_le_bytes().to_vec();
        content.extend_from_slice(xml.as_bytes());
        make_chunk(TAG_STREAM_HEADER, &content)
    }

    /// Build a Samples chunk with f32 values.
    fn make_samples_f32(stream_id: u32, ch_count: u32, samples: &[(f64, &[f32])]) -> Vec<u8> {
        let mut content = stream_id.to_le_bytes().to_vec();
        for (ts, values) in samples {
            assert_eq!(values.len(), ch_count as usize);
            if ts.is_nan() {
                content.push(0);
            } else {
                content.push(8);
                content.extend_from_slice(&ts.to_le_bytes());
            }
            for &v in *values {
                content.extend_from_slice(&v.to_le_bytes());
            }
        }
        make_chunk(TAG_SAMPLES, &content)
    }

    /// Build a Samples chunk with f64 values.
    fn make_samples_f64(stream_id: u32, ch_count: u32, samples: &[(f64, &[f64])]) -> Vec<u8> {
        let mut content = stream_id.to_le_bytes().to_vec();
        for (ts, values) in samples {
            assert_eq!(values.len(), ch_count as usize);
            if ts.is_nan() {
                content.push(0);
            } else {
                content.push(8);
                content.extend_from_slice(&ts.to_le_bytes());
            }
            for &v in *values {
                content.extend_from_slice(&v.to_le_bytes());
            }
        }
        make_chunk(TAG_SAMPLES, &content)
    }

    /// Build a Samples chunk with i16 values.
    fn make_samples_i16(stream_id: u32, ch_count: u32, samples: &[(f64, &[i16])]) -> Vec<u8> {
        let mut content = stream_id.to_le_bytes().to_vec();
        for (ts, values) in samples {
            assert_eq!(values.len(), ch_count as usize);
            if ts.is_nan() {
                content.push(0);
            } else {
                content.push(8);
                content.extend_from_slice(&ts.to_le_bytes());
            }
            for &v in *values {
                content.extend_from_slice(&v.to_le_bytes());
            }
        }
        make_chunk(TAG_SAMPLES, &content)
    }

    /// Build a Samples chunk with i32 values.
    fn make_samples_i32(stream_id: u32, ch_count: u32, samples: &[(f64, &[i32])]) -> Vec<u8> {
        let mut content = stream_id.to_le_bytes().to_vec();
        for (ts, values) in samples {
            assert_eq!(values.len(), ch_count as usize);
            if ts.is_nan() {
                content.push(0);
            } else {
                content.push(8);
                content.extend_from_slice(&ts.to_le_bytes());
            }
            for &v in *values {
                content.extend_from_slice(&v.to_le_bytes());
            }
        }
        make_chunk(TAG_SAMPLES, &content)
    }

    /// Build a ClockOffset chunk.
    fn make_clock_offset(stream_id: u32, collection_time: f64, offset_value: f64) -> Vec<u8> {
        let mut content = stream_id.to_le_bytes().to_vec();
        content.extend_from_slice(&collection_time.to_le_bytes());
        content.extend_from_slice(&offset_value.to_le_bytes());
        make_chunk(TAG_CLOCK_OFFSET, &content)
    }

    /// Build a complete XDF file from parts.
    fn make_xdf(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut file = b"XDF:".to_vec();
        // File header chunk
        let header_xml = b"<?xml version=\"1.0\"?><info><version>1.0</version></info>";
        file.extend_from_slice(&make_chunk(TAG_FILE_HEADER, header_xml));
        for chunk in chunks {
            file.extend_from_slice(chunk);
        }
        file
    }

    // --- Varlen tests ---

    #[test]
    fn test_varlen_1byte() {
        let data = [1u8, 42];
        let mut pos = 0;
        assert_eq!(read_varlen(&data, &mut pos).unwrap(), 42);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_varlen_4byte() {
        let mut data = vec![4u8];
        data.extend_from_slice(&300u32.to_le_bytes());
        let mut pos = 0;
        assert_eq!(read_varlen(&data, &mut pos).unwrap(), 300);
        assert_eq!(pos, 5);
    }

    #[test]
    fn test_varlen_8byte() {
        let big: u64 = 5_000_000_000;
        let mut data = vec![8u8];
        data.extend_from_slice(&big.to_le_bytes());
        let mut pos = 0;
        assert_eq!(read_varlen(&data, &mut pos).unwrap(), big);
        assert_eq!(pos, 9);
    }

    #[test]
    fn test_varlen_invalid_prefix() {
        let data = [3u8, 0, 0];
        let mut pos = 0;
        assert_eq!(read_varlen(&data, &mut pos), Err(XdfError::InvalidVarLen));
    }

    #[test]
    fn test_varlen_truncated() {
        let data = [4u8, 0, 0]; // needs 4 bytes but only 2 available
        let mut pos = 0;
        assert_eq!(read_varlen(&data, &mut pos), Err(XdfError::TruncatedFile));
    }

    // --- Magic tests ---

    #[test]
    fn test_valid_magic() {
        assert_eq!(validate_magic(b"XDF:rest of file"), Ok(()));
    }

    #[test]
    fn test_invalid_magic() {
        assert_eq!(validate_magic(b"EDF:nope"), Err(XdfError::InvalidMagic));
    }

    #[test]
    fn test_truncated_magic() {
        assert_eq!(validate_magic(b"XD"), Err(XdfError::TruncatedFile));
    }

    // --- XML tag extraction ---

    #[test]
    fn test_find_tag_simple() {
        let xml = b"<info><name>EEG</name><type>EEG</type></info>";
        assert_eq!(find_tag_value(xml, b"name"), Some(b"EEG" as &[u8]));
        assert_eq!(find_tag_value(xml, b"type"), Some(b"EEG" as &[u8]));
    }

    #[test]
    fn test_find_tag_missing() {
        let xml = b"<info><name>EEG</name></info>";
        assert_eq!(find_tag_value(xml, b"missing"), None);
    }

    #[test]
    fn test_find_tag_numeric() {
        let xml = b"<info><channel_count>32</channel_count></info>";
        assert_eq!(find_tag_value(xml, b"channel_count"), Some(b"32" as &[u8]));
    }

    #[test]
    fn test_find_tag_multiple_same() {
        // Returns first occurrence
        let xml = b"<a><name>first</name><sub><name>second</name></sub></a>";
        assert_eq!(find_tag_value(xml, b"name"), Some(b"first" as &[u8]));
    }

    // --- Chunk iteration ---

    #[test]
    fn test_chunk_iteration() {
        let sh = make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32");
        let samples = make_samples_f32(1, 2, &[(1.0, &[0.5, -0.5])]);
        let xdf = make_xdf(&[sh, samples]);

        let mut pos = 4; // skip magic
        let mut tags = Vec::new();
        while let Some(chunk) = next_chunk(&xdf, &mut pos).unwrap() {
            tags.push(chunk.tag);
        }
        assert_eq!(tags, vec![TAG_FILE_HEADER, TAG_STREAM_HEADER, TAG_SAMPLES]);
    }

    #[test]
    fn test_chunk_stream_id() {
        let sh = make_stream_header(42, "Test", "EEG", 1, 256.0, "float32");
        let xdf = make_xdf(&[sh]);

        let mut pos = 4;
        // Skip file header
        next_chunk(&xdf, &mut pos).unwrap();
        // Stream header
        let chunk = next_chunk(&xdf, &mut pos).unwrap().unwrap();
        assert_eq!(chunk.tag, TAG_STREAM_HEADER);
        assert_eq!(chunk.stream_id, Some(42));
    }

    // --- Stream info parsing ---

    #[test]
    fn test_parse_stream_info() {
        let xml = b"<?xml version=\"1.0\"?><info><name>MyEEG</name><type>EEG</type>\
                     <channel_count>32</channel_count><nominal_srate>256.0</nominal_srate>\
                     <channel_format>float32</channel_format></info>";
        let info = parse_stream_info(xml, 1).unwrap();
        assert_eq!(info.stream_id, 1);
        assert_eq!(info.name, b"MyEEG");
        assert_eq!(info.stream_type, b"EEG");
        assert_eq!(info.channel_count, 32);
        assert!((info.nominal_srate - 256.0).abs() < 1e-10);
        assert_eq!(info.channel_format, XdfChannelFormat::Float32);
    }

    // --- Sample decoding ---

    #[test]
    fn test_decode_f32_samples() {
        let samples_data = make_samples_f32(1, 2, &[(1.0, &[1.5f32, -2.5]), (2.0, &[3.0, 4.0])]);
        // Extract content after stream_id from the chunk
        // The chunk structure: varlen(chunk_len) + tag(2) + stream_id(4) + sample_data
        // We need to get the sample data part (after stream_id)
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4; // skip tag + stream_id
        let content = &samples_data[content_start..pos + chunk_len];

        let mut values = [0.0f64; 4];
        let mut timestamps = [0.0f64; 2];
        let n = decode_samples_f64(
            content,
            2,
            &XdfChannelFormat::Float32,
            &mut values,
            &mut timestamps,
        )
        .unwrap();

        assert_eq!(n, 2);
        assert!((timestamps[0] - 1.0).abs() < 1e-10);
        assert!((timestamps[1] - 2.0).abs() < 1e-10);
        assert!((values[0] - 1.5).abs() < 1e-5);
        assert!((values[1] - (-2.5)).abs() < 1e-5);
        assert!((values[2] - 3.0).abs() < 1e-5);
        assert!((values[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_decode_f64_samples() {
        let samples_data = make_samples_f64(1, 1, &[(10.0, &[99.99])]);
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4;
        let content = &samples_data[content_start..pos + chunk_len];

        let mut values = [0.0f64; 1];
        let mut timestamps = [0.0f64; 1];
        let n = decode_samples_f64(
            content,
            1,
            &XdfChannelFormat::Double64,
            &mut values,
            &mut timestamps,
        )
        .unwrap();

        assert_eq!(n, 1);
        assert!((timestamps[0] - 10.0).abs() < 1e-10);
        assert!((values[0] - 99.99).abs() < 1e-10);
    }

    #[test]
    fn test_decode_i16_samples() {
        let samples_data = make_samples_i16(1, 2, &[(5.0, &[1000i16, -2000])]);
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4;
        let content = &samples_data[content_start..pos + chunk_len];

        let mut values = [0.0f64; 2];
        let mut timestamps = [0.0f64; 1];
        let n = decode_samples_f64(
            content,
            2,
            &XdfChannelFormat::Int16,
            &mut values,
            &mut timestamps,
        )
        .unwrap();

        assert_eq!(n, 1);
        assert!((values[0] - 1000.0).abs() < 1e-10);
        assert!((values[1] - (-2000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_decode_i32_samples() {
        let samples_data = make_samples_i32(1, 1, &[(0.0, &[100_000i32])]);
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4;
        let content = &samples_data[content_start..pos + chunk_len];

        let mut values = [0.0f64; 1];
        let mut timestamps = [0.0f64; 1];
        let n = decode_samples_f64(
            content,
            1,
            &XdfChannelFormat::Int32,
            &mut values,
            &mut timestamps,
        )
        .unwrap();

        assert_eq!(n, 1);
        assert!((values[0] - 100_000.0).abs() < 1e-10);
    }

    // --- Timestamp handling ---

    #[test]
    fn test_deduced_timestamps() {
        // Samples without timestamps (flag byte = 0)
        let samples_data = make_samples_f32(1, 1, &[(f64::NAN, &[1.0f32]), (f64::NAN, &[2.0])]);
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4;
        let content = &samples_data[content_start..pos + chunk_len];

        let mut values = [0.0f64; 2];
        let mut timestamps = [0.0f64; 2];
        let n = decode_samples_f64(
            content,
            1,
            &XdfChannelFormat::Float32,
            &mut values,
            &mut timestamps,
        )
        .unwrap();

        assert_eq!(n, 2);
        assert!(timestamps[0].is_nan());
        assert!(timestamps[1].is_nan());
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 2.0).abs() < 1e-5);
    }

    // --- Clock offset ---

    #[test]
    fn test_clock_offset_parsing() {
        let co = make_clock_offset(1, 100.5, -0.003);
        let mut pos = 0;
        let chunk_len = read_varlen(&co, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4; // skip tag + stream_id
        let content = &co[content_start..pos + chunk_len];

        let pair = parse_clock_offset(content).unwrap();
        assert!((pair.collection_time - 100.5).abs() < 1e-10);
        assert!((pair.offset_value - (-0.003)).abs() < 1e-10);
    }

    // --- Count samples ---

    #[test]
    fn test_count_samples() {
        let samples_data = make_samples_f32(
            1,
            3,
            &[
                (1.0, &[1.0f32, 2.0, 3.0]),
                (2.0, &[4.0, 5.0, 6.0]),
                (3.0, &[7.0, 8.0, 9.0]),
            ],
        );
        let mut pos = 0;
        let chunk_len = read_varlen(&samples_data, &mut pos).unwrap() as usize;
        let content_start = pos + 2 + 4;
        let content = &samples_data[content_start..pos + chunk_len];

        let n = count_chunk_samples(content, 3, &XdfChannelFormat::Float32).unwrap();
        assert_eq!(n, 3);
    }

    // --- Multi-stream file ---

    #[test]
    fn test_multi_stream_file() {
        let sh1 = make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32");
        let sh2 = make_stream_header(2, "Markers", "Markers", 1, 0.0, "float32");
        let s1 = make_samples_f32(1, 2, &[(1.0, &[0.1f32, 0.2])]);
        let s2 = make_samples_f32(2, 1, &[(1.5, &[1.0f32])]);
        let co = make_clock_offset(1, 10.0, -0.001);
        let xdf = make_xdf(&[sh1, sh2, s1, s2, co]);

        let mut pos = 4;
        let mut stream_ids = Vec::new();
        let mut tags = Vec::new();
        while let Some(chunk) = next_chunk(&xdf, &mut pos).unwrap() {
            tags.push(chunk.tag);
            if let Some(id) = chunk.stream_id {
                stream_ids.push(id);
            }
        }
        // FileHeader + 2 StreamHeaders + 2 Samples + 1 ClockOffset = 6 chunks
        assert_eq!(tags.len(), 6);
        assert_eq!(stream_ids, vec![1, 2, 1, 2, 1]);
    }

    // --- Empty stream edge case ---

    #[test]
    fn test_empty_stream() {
        // Stream header but no samples
        let sh = make_stream_header(1, "Empty", "EEG", 4, 256.0, "float32");
        let xdf = make_xdf(&[sh]);

        let mut pos = 4;
        let mut sample_count = 0;
        while let Some(chunk) = next_chunk(&xdf, &mut pos).unwrap() {
            if chunk.tag == TAG_SAMPLES {
                sample_count += 1;
            }
        }
        assert_eq!(sample_count, 0);
    }

    // --- Channel format parsing ---

    #[test]
    fn test_channel_format_from_bytes() {
        assert_eq!(
            XdfChannelFormat::from_bytes(b"float32"),
            Ok(XdfChannelFormat::Float32)
        );
        assert_eq!(
            XdfChannelFormat::from_bytes(b"double64"),
            Ok(XdfChannelFormat::Double64)
        );
        assert_eq!(
            XdfChannelFormat::from_bytes(b"int16"),
            Ok(XdfChannelFormat::Int16)
        );
        assert_eq!(
            XdfChannelFormat::from_bytes(b"int32"),
            Ok(XdfChannelFormat::Int32)
        );
        assert_eq!(
            XdfChannelFormat::from_bytes(b"string"),
            Ok(XdfChannelFormat::StringFormat)
        );
        assert_eq!(
            XdfChannelFormat::from_bytes(b"unknown"),
            Err(XdfError::UnsupportedFormat)
        );
    }
}
