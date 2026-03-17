//! MDA (MountainSort Data Array) format reader.
//!
//! Parses the binary MDA format used by SpikeForest and MountainSort
//! for storing multi-dimensional arrays of neural recording data.
//!
//! MDA is a simple binary format: a header encoding data type and
//! dimensions, followed immediately by raw data in column-major order.
//! All integers in the header are little-endian `i32` values.
//!
//! # Type codes
//!
//! | Code | Type    |
//! |------|---------|
//! | -2   | uint8   |
//! | -3   | float32 |
//! | -4   | int16   |
//! | -5   | int32   |
//! | -6   | uint16  |
//! | -7   | float64 |
//! | -8   | uint32  |
//!
//! # Example
//!
//! ```
//! use zerostone::mda::{parse_mda_header, read_mda_f64, MdaDataType};
//!
//! // Build a tiny MDA file: float32, 1-D, 3 elements
//! let mut data = Vec::new();
//! data.extend_from_slice(&(-3i32).to_le_bytes()); // dtype = float32
//! data.extend_from_slice(&1i32.to_le_bytes());    // ndims = 1
//! data.extend_from_slice(&3i32.to_le_bytes());    // dim[0] = 3
//! data.extend_from_slice(&1.0f32.to_le_bytes());
//! data.extend_from_slice(&2.0f32.to_le_bytes());
//! data.extend_from_slice(&3.0f32.to_le_bytes());
//!
//! let header = parse_mda_header(&data).unwrap();
//! assert_eq!(header.data_type, MdaDataType::Float32);
//! assert_eq!(header.ndims, 1);
//! assert_eq!(header.dims[0], 3);
//!
//! let mut output = [0.0f64; 3];
//! let n = read_mda_f64(&data, &mut output).unwrap();
//! assert_eq!(n, 3);
//! assert!((output[0] - 1.0).abs() < 1e-6);
//! assert!((output[1] - 2.0).abs() < 1e-6);
//! assert!((output[2] - 3.0).abs() < 1e-6);
//! ```

/// Maximum number of dimensions supported.
pub const MAX_DIMS: usize = 6;

/// MDA data type codes.
///
/// Each variant corresponds to a numeric code in the MDA header.
///
/// ```
/// use zerostone::mda::{MdaDataType, mda_element_size};
/// assert_eq!(mda_element_size(MdaDataType::Float32), 4);
/// assert_eq!(mda_element_size(MdaDataType::Int16), 2);
/// assert_eq!(mda_element_size(MdaDataType::Float64), 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdaDataType {
    /// Unsigned 8-bit integer (code -2).
    Uint8,
    /// 32-bit float (code -3).
    Float32,
    /// Signed 16-bit integer (code -4).
    Int16,
    /// Signed 32-bit integer (code -5).
    Int32,
    /// Unsigned 16-bit integer (code -6).
    Uint16,
    /// 64-bit float (code -7).
    Float64,
    /// Unsigned 32-bit integer (code -8).
    Uint32,
}

impl MdaDataType {
    /// Convert from the integer type code in the MDA header.
    ///
    /// ```
    /// use zerostone::mda::MdaDataType;
    /// assert_eq!(MdaDataType::from_code(-3), Some(MdaDataType::Float32));
    /// assert_eq!(MdaDataType::from_code(-7), Some(MdaDataType::Float64));
    /// assert_eq!(MdaDataType::from_code(0), None);
    /// ```
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            -2 => Some(Self::Uint8),
            -3 => Some(Self::Float32),
            -4 => Some(Self::Int16),
            -5 => Some(Self::Int32),
            -6 => Some(Self::Uint16),
            -7 => Some(Self::Float64),
            -8 => Some(Self::Uint32),
            _ => None,
        }
    }
}

/// Parsed MDA file header.
///
/// Contains the data type, number of dimensions, dimension sizes,
/// and the byte offset where data begins.
///
/// ```
/// use zerostone::mda::{parse_mda_header, mda_num_elements, MdaDataType};
///
/// let mut data = Vec::new();
/// data.extend_from_slice(&(-4i32).to_le_bytes()); // int16
/// data.extend_from_slice(&2i32.to_le_bytes());    // 2-D
/// data.extend_from_slice(&4i32.to_le_bytes());    // 4 rows
/// data.extend_from_slice(&100i32.to_le_bytes());  // 100 columns
///
/// let header = parse_mda_header(&data).unwrap();
/// assert_eq!(header.data_type, MdaDataType::Int16);
/// assert_eq!(header.ndims, 2);
/// assert_eq!(header.dims[0], 4);
/// assert_eq!(header.dims[1], 100);
/// assert_eq!(mda_num_elements(&header), 400);
/// assert_eq!(header.data_offset, 16); // 4 * (1 + 1 + 2) = 16
/// ```
#[derive(Debug, Clone)]
pub struct MdaHeader {
    /// The element data type.
    pub data_type: MdaDataType,
    /// Number of dimensions (1 to [`MAX_DIMS`]).
    pub ndims: usize,
    /// Dimension sizes. Unused dimensions (index >= ndims) are 0.
    pub dims: [usize; MAX_DIMS],
    /// Byte offset in the file where data begins.
    pub data_offset: usize,
}

/// Parse an MDA header from a byte slice.
///
/// Returns `None` if the header is truncated, the data type code is
/// unrecognised, ndims is zero or exceeds [`MAX_DIMS`], or any
/// dimension size is negative.
///
/// ```
/// use zerostone::mda::{parse_mda_header, MdaDataType};
///
/// // float64, 1-D, 5 elements
/// let mut buf = Vec::new();
/// buf.extend_from_slice(&(-7i32).to_le_bytes());
/// buf.extend_from_slice(&1i32.to_le_bytes());
/// buf.extend_from_slice(&5i32.to_le_bytes());
///
/// let hdr = parse_mda_header(&buf).unwrap();
/// assert_eq!(hdr.data_type, MdaDataType::Float64);
/// assert_eq!(hdr.ndims, 1);
/// assert_eq!(hdr.dims[0], 5);
/// assert_eq!(hdr.data_offset, 12);
/// ```
pub fn parse_mda_header(data: &[u8]) -> Option<MdaHeader> {
    // Need at least 8 bytes for dtype + ndims
    if data.len() < 8 {
        return None;
    }

    let dtype_code = read_i32_le(data, 0)?;
    let data_type = MdaDataType::from_code(dtype_code)?;

    let ndims_i32 = read_i32_le(data, 4)?;
    if ndims_i32 <= 0 || ndims_i32 as usize > MAX_DIMS {
        return None;
    }
    let ndims = ndims_i32 as usize;

    // Need 4 bytes per dimension after the 8-byte prefix
    let header_size = 8 + ndims * 4;
    if data.len() < header_size {
        return None;
    }

    let mut dims = [0usize; MAX_DIMS];
    for (i, dim) in dims.iter_mut().enumerate().take(ndims) {
        let dim_i32 = read_i32_le(data, 8 + i * 4)?;
        if dim_i32 < 0 {
            return None;
        }
        *dim = dim_i32 as usize;
    }

    Some(MdaHeader {
        data_type,
        ndims,
        dims,
        data_offset: header_size,
    })
}

/// Total number of elements described by the header.
///
/// This is the product of all dimension sizes.
///
/// ```
/// use zerostone::mda::{MdaHeader, MdaDataType, mda_num_elements, MAX_DIMS};
///
/// let header = MdaHeader {
///     data_type: MdaDataType::Float32,
///     ndims: 2,
///     dims: {
///         let mut d = [0usize; MAX_DIMS];
///         d[0] = 32;
///         d[1] = 1000;
///         d
///     },
///     data_offset: 16,
/// };
/// assert_eq!(mda_num_elements(&header), 32_000);
/// ```
pub fn mda_num_elements(header: &MdaHeader) -> usize {
    if header.ndims == 0 {
        return 0;
    }
    let mut n = 1usize;
    for i in 0..header.ndims {
        n = n.saturating_mul(header.dims[i]);
    }
    n
}

/// Bytes per element for a given data type.
///
/// ```
/// use zerostone::mda::{MdaDataType, mda_element_size};
///
/// assert_eq!(mda_element_size(MdaDataType::Uint8), 1);
/// assert_eq!(mda_element_size(MdaDataType::Int16), 2);
/// assert_eq!(mda_element_size(MdaDataType::Uint16), 2);
/// assert_eq!(mda_element_size(MdaDataType::Float32), 4);
/// assert_eq!(mda_element_size(MdaDataType::Int32), 4);
/// assert_eq!(mda_element_size(MdaDataType::Uint32), 4);
/// assert_eq!(mda_element_size(MdaDataType::Float64), 8);
/// ```
pub fn mda_element_size(dtype: MdaDataType) -> usize {
    match dtype {
        MdaDataType::Uint8 => 1,
        MdaDataType::Int16 | MdaDataType::Uint16 => 2,
        MdaDataType::Float32 | MdaDataType::Int32 | MdaDataType::Uint32 => 4,
        MdaDataType::Float64 => 8,
    }
}

/// Read MDA data as `f64` samples into a caller-provided buffer.
///
/// Parses the header, then converts each element from the source data
/// type to `f64`. Returns the number of elements written, or `None` if
/// the data is truncated or the header is invalid.
///
/// ```
/// use zerostone::mda::read_mda_f64;
///
/// // int16, 1-D, 2 elements: [100, -200]
/// let mut buf = Vec::new();
/// buf.extend_from_slice(&(-4i32).to_le_bytes());
/// buf.extend_from_slice(&1i32.to_le_bytes());
/// buf.extend_from_slice(&2i32.to_le_bytes());
/// buf.extend_from_slice(&100i16.to_le_bytes());
/// buf.extend_from_slice(&(-200i16).to_le_bytes());
///
/// let mut out = [0.0f64; 2];
/// let n = read_mda_f64(&buf, &mut out).unwrap();
/// assert_eq!(n, 2);
/// assert!((out[0] - 100.0).abs() < 1e-10);
/// assert!((out[1] - (-200.0)).abs() < 1e-10);
/// ```
pub fn read_mda_f64(data: &[u8], output: &mut [f64]) -> Option<usize> {
    let header = parse_mda_header(data)?;
    let n = mda_num_elements(&header);
    if n == 0 {
        return Some(0);
    }
    let elem_size = mda_element_size(header.data_type);
    let data_end = header.data_offset.checked_add(n.checked_mul(elem_size)?)?;
    if data.len() < data_end {
        return None;
    }
    let count = n.min(output.len());
    let raw = &data[header.data_offset..];

    for (i, out) in output.iter_mut().enumerate().take(count) {
        let off = i * elem_size;
        *out = read_element_f64(raw, off, header.data_type);
    }

    Some(count)
}

/// Read MDA data as `f32` samples into a caller-provided buffer.
///
/// Same as [`read_mda_f64`] but outputs `f32` values.
///
/// ```
/// use zerostone::mda::read_mda_f32;
///
/// // float32, 1-D, 2 elements: [1.5, -2.5]
/// let mut buf = Vec::new();
/// buf.extend_from_slice(&(-3i32).to_le_bytes());
/// buf.extend_from_slice(&1i32.to_le_bytes());
/// buf.extend_from_slice(&2i32.to_le_bytes());
/// buf.extend_from_slice(&1.5f32.to_le_bytes());
/// buf.extend_from_slice(&(-2.5f32).to_le_bytes());
///
/// let mut out = [0.0f32; 2];
/// let n = read_mda_f32(&buf, &mut out).unwrap();
/// assert_eq!(n, 2);
/// assert!((out[0] - 1.5).abs() < 1e-6);
/// assert!((out[1] - (-2.5)).abs() < 1e-6);
/// ```
pub fn read_mda_f32(data: &[u8], output: &mut [f32]) -> Option<usize> {
    let header = parse_mda_header(data)?;
    let n = mda_num_elements(&header);
    if n == 0 {
        return Some(0);
    }
    let elem_size = mda_element_size(header.data_type);
    let data_end = header.data_offset.checked_add(n.checked_mul(elem_size)?)?;
    if data.len() < data_end {
        return None;
    }
    let count = n.min(output.len());
    let raw = &data[header.data_offset..];

    for (i, out) in output.iter_mut().enumerate().take(count) {
        let off = i * elem_size;
        *out = read_element_f64(raw, off, header.data_type) as f32;
    }

    Some(count)
}

// --- Private helpers ---

/// Read a little-endian i32 from `data` at byte offset `off`.
fn read_i32_le(data: &[u8], off: usize) -> Option<i32> {
    if off + 4 > data.len() {
        return None;
    }
    Some(i32::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
    ]))
}

/// Convert one element from the raw data region to f64.
///
/// Caller must ensure `off + element_size <= raw.len()`.
fn read_element_f64(raw: &[u8], off: usize, dtype: MdaDataType) -> f64 {
    match dtype {
        MdaDataType::Uint8 => raw[off] as f64,
        MdaDataType::Int16 => i16::from_le_bytes([raw[off], raw[off + 1]]) as f64,
        MdaDataType::Uint16 => u16::from_le_bytes([raw[off], raw[off + 1]]) as f64,
        MdaDataType::Float32 => {
            f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as f64
        }
        MdaDataType::Int32 => {
            i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as f64
        }
        MdaDataType::Uint32 => {
            u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as f64
        }
        MdaDataType::Float64 => f64::from_le_bytes([
            raw[off],
            raw[off + 1],
            raw[off + 2],
            raw[off + 3],
            raw[off + 4],
            raw[off + 5],
            raw[off + 6],
            raw[off + 7],
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    /// Build MDA bytes: header + raw data.
    fn make_mda(dtype_code: i32, dims: &[i32], raw_data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&dtype_code.to_le_bytes());
        buf.extend_from_slice(&(dims.len() as i32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(raw_data);
        buf
    }

    // --- Header parsing tests ---

    #[test]
    fn test_parse_header_float32_1d() {
        let data = make_mda(-3, &[10], &[]);
        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(hdr.data_type, MdaDataType::Float32);
        assert_eq!(hdr.ndims, 1);
        assert_eq!(hdr.dims[0], 10);
        assert_eq!(hdr.data_offset, 12);
    }

    #[test]
    fn test_parse_header_int16_2d() {
        let data = make_mda(-4, &[32, 30000], &[]);
        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(hdr.data_type, MdaDataType::Int16);
        assert_eq!(hdr.ndims, 2);
        assert_eq!(hdr.dims[0], 32);
        assert_eq!(hdr.dims[1], 30000);
        assert_eq!(hdr.data_offset, 16);
    }

    #[test]
    fn test_parse_header_float64_3d() {
        let data = make_mda(-7, &[2, 3, 4], &[]);
        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(hdr.data_type, MdaDataType::Float64);
        assert_eq!(hdr.ndims, 3);
        assert_eq!(hdr.dims[0], 2);
        assert_eq!(hdr.dims[1], 3);
        assert_eq!(hdr.dims[2], 4);
        assert_eq!(hdr.data_offset, 20);
    }

    #[test]
    fn test_parse_header_all_dtypes() {
        let cases: &[(i32, MdaDataType)] = &[
            (-2, MdaDataType::Uint8),
            (-3, MdaDataType::Float32),
            (-4, MdaDataType::Int16),
            (-5, MdaDataType::Int32),
            (-6, MdaDataType::Uint16),
            (-7, MdaDataType::Float64),
            (-8, MdaDataType::Uint32),
        ];
        for &(code, expected) in cases {
            let data = make_mda(code, &[1], &[0; 8]);
            let hdr = parse_mda_header(&data).unwrap();
            assert_eq!(hdr.data_type, expected, "dtype code {}", code);
        }
    }

    #[test]
    fn test_parse_header_truncated_too_short() {
        // Less than 8 bytes
        assert!(parse_mda_header(&[0u8; 4]).is_none());
    }

    #[test]
    fn test_parse_header_truncated_missing_dims() {
        // Header says 3 dims but only provides space for 1
        let mut buf = Vec::new();
        buf.extend_from_slice(&(-3i32).to_le_bytes());
        buf.extend_from_slice(&3i32.to_le_bytes());
        buf.extend_from_slice(&10i32.to_le_bytes()); // only 1 dim
        assert!(parse_mda_header(&buf).is_none());
    }

    #[test]
    fn test_parse_header_invalid_dtype() {
        let data = make_mda(0, &[1], &[]);
        assert!(parse_mda_header(&data).is_none());

        let data = make_mda(-1, &[1], &[]);
        assert!(parse_mda_header(&data).is_none());

        let data = make_mda(-9, &[1], &[]);
        assert!(parse_mda_header(&data).is_none());

        let data = make_mda(42, &[1], &[]);
        assert!(parse_mda_header(&data).is_none());
    }

    #[test]
    fn test_parse_header_zero_ndims() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(-3i32).to_le_bytes());
        buf.extend_from_slice(&0i32.to_le_bytes());
        assert!(parse_mda_header(&buf).is_none());
    }

    #[test]
    fn test_parse_header_negative_dim() {
        let data = make_mda(-3, &[-1], &[]);
        assert!(parse_mda_header(&data).is_none());
    }

    #[test]
    fn test_parse_header_ndims_exceeds_max() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(-3i32).to_le_bytes());
        buf.extend_from_slice(&7i32.to_le_bytes()); // MAX_DIMS = 6
        for _ in 0..7 {
            buf.extend_from_slice(&1i32.to_le_bytes());
        }
        assert!(parse_mda_header(&buf).is_none());
    }

    // --- Element count ---

    #[test]
    fn test_num_elements() {
        let data = make_mda(-3, &[4, 100], &[]);
        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(mda_num_elements(&hdr), 400);
    }

    #[test]
    fn test_num_elements_with_zero_dim() {
        let data = make_mda(-3, &[4, 0, 10], &[]);
        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(mda_num_elements(&hdr), 0);
    }

    // --- Element size ---

    #[test]
    fn test_element_sizes() {
        assert_eq!(mda_element_size(MdaDataType::Uint8), 1);
        assert_eq!(mda_element_size(MdaDataType::Int16), 2);
        assert_eq!(mda_element_size(MdaDataType::Uint16), 2);
        assert_eq!(mda_element_size(MdaDataType::Float32), 4);
        assert_eq!(mda_element_size(MdaDataType::Int32), 4);
        assert_eq!(mda_element_size(MdaDataType::Uint32), 4);
        assert_eq!(mda_element_size(MdaDataType::Float64), 8);
    }

    // --- Data reading: f64 output ---

    #[test]
    fn test_read_f64_from_float32() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.5f32.to_le_bytes());
        raw.extend_from_slice(&(-2.5f32).to_le_bytes());
        raw.extend_from_slice(&0.0f32.to_le_bytes());
        let data = make_mda(-3, &[3], &raw);

        let mut out = [0.0f64; 3];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 3);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - (-2.5)).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_f64_from_float64() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&core::f64::consts::PI.to_le_bytes());
        raw.extend_from_slice(&(-1e10f64).to_le_bytes());
        let data = make_mda(-7, &[2], &raw);

        let mut out = [0.0f64; 2];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - core::f64::consts::PI).abs() < 1e-15);
        assert!((out[1] - (-1e10)).abs() < 1e-5);
    }

    #[test]
    fn test_read_f64_from_int16() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&100i16.to_le_bytes());
        raw.extend_from_slice(&(-200i16).to_le_bytes());
        raw.extend_from_slice(&32767i16.to_le_bytes());
        raw.extend_from_slice(&(-32768i16).to_le_bytes());
        let data = make_mda(-4, &[4], &raw);

        let mut out = [0.0f64; 4];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 4);
        assert!((out[0] - 100.0).abs() < 1e-10);
        assert!((out[1] - (-200.0)).abs() < 1e-10);
        assert!((out[2] - 32767.0).abs() < 1e-10);
        assert!((out[3] - (-32768.0)).abs() < 1e-10);
    }

    #[test]
    fn test_read_f64_from_int32() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1_000_000i32.to_le_bytes());
        raw.extend_from_slice(&(-999i32).to_le_bytes());
        let data = make_mda(-5, &[2], &raw);

        let mut out = [0.0f64; 2];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - 1_000_000.0).abs() < 1e-10);
        assert!((out[1] - (-999.0)).abs() < 1e-10);
    }

    #[test]
    fn test_read_f64_from_uint8() {
        let raw = [0u8, 127, 255];
        let data = make_mda(-2, &[3], &raw);

        let mut out = [0.0f64; 3];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 3);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 127.0).abs() < 1e-10);
        assert!((out[2] - 255.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_f64_from_uint16() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u16.to_le_bytes());
        raw.extend_from_slice(&65535u16.to_le_bytes());
        let data = make_mda(-6, &[2], &raw);

        let mut out = [0.0f64; 2];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 65535.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_f64_from_uint32() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&4_000_000_000u32.to_le_bytes());
        let data = make_mda(-8, &[2], &raw);

        let mut out = [0.0f64; 2];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 4_000_000_000.0).abs() < 1e-5);
    }

    // --- Data reading: f32 output ---

    #[test]
    fn test_read_f32_from_float32() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&42.0f32.to_le_bytes());
        raw.extend_from_slice(&(-7.5f32).to_le_bytes());
        let data = make_mda(-3, &[2], &raw);

        let mut out = [0.0f32; 2];
        let n = read_mda_f32(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - 42.0).abs() < 1e-6);
        assert!((out[1] - (-7.5)).abs() < 1e-6);
    }

    #[test]
    fn test_read_f32_from_int16() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&500i16.to_le_bytes());
        let data = make_mda(-4, &[1], &raw);

        let mut out = [0.0f32; 1];
        let n = read_mda_f32(&data, &mut out).unwrap();
        assert_eq!(n, 1);
        assert!((out[0] - 500.0).abs() < 1e-4);
    }

    // --- Multi-dimensional ---

    #[test]
    fn test_read_f64_2d_column_major() {
        // 2 rows x 3 cols, float32, column-major: col0=[1,2], col1=[3,4], col2=[5,6]
        let mut raw = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = make_mda(-3, &[2, 3], &raw);

        let hdr = parse_mda_header(&data).unwrap();
        assert_eq!(mda_num_elements(&hdr), 6);

        let mut out = [0.0f64; 6];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 6);
        // Column-major order: elements read linearly
        for (i, val) in out.iter().enumerate() {
            assert!((val - (i as f64 + 1.0)).abs() < 1e-6);
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_read_truncated_data() {
        // Header says 10 float32 but only provide 2
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        raw.extend_from_slice(&2.0f32.to_le_bytes());
        let data = make_mda(-3, &[10], &raw);

        let mut out = [0.0f64; 10];
        assert!(read_mda_f64(&data, &mut out).is_none());
    }

    #[test]
    fn test_read_buffer_smaller_than_data() {
        // 4 elements but buffer has only 2 slots
        let mut raw = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = make_mda(-3, &[4], &raw);

        let mut out = [0.0f64; 2];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_zero_dim_element() {
        let data = make_mda(-3, &[0], &[]);

        let mut out = [0.0f64; 4];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_read_invalid_header_returns_none() {
        let mut out = [0.0f64; 4];
        assert!(read_mda_f64(&[0u8; 4], &mut out).is_none());
    }

    // --- Round-trip tests ---

    #[test]
    fn test_roundtrip_float32() {
        let values = [1.0f32, -2.5, 0.0, 1e6, -1e-6];
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = make_mda(-3, &[values.len() as i32], &raw);

        let mut out = [0.0f32; 5];
        let n = read_mda_f32(&data, &mut out).unwrap();
        assert_eq!(n, 5);
        for i in 0..5 {
            assert_eq!(out[i], values[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_roundtrip_float64() {
        let values = [core::f64::consts::PI, -core::f64::consts::E, 0.0, 1e300];
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = make_mda(-7, &[values.len() as i32], &raw);

        let mut out = [0.0f64; 4];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 4);
        for i in 0..4 {
            assert!(
                (out[i] - values[i]).abs() < 1e-15,
                "mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_roundtrip_int16() {
        let values = [0i16, 1, -1, 32767, -32768];
        let mut raw = Vec::new();
        for &v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let data = make_mda(-4, &[values.len() as i32], &raw);

        let mut out = [0.0f64; 5];
        let n = read_mda_f64(&data, &mut out).unwrap();
        assert_eq!(n, 5);
        for i in 0..5 {
            assert!(
                (out[i] - values[i] as f64).abs() < 1e-10,
                "mismatch at index {}",
                i
            );
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `parse_mda_header` never panics on any 24-byte input.
    #[kani::proof]
    #[kani::unwind(8)]
    fn parse_mda_header_no_panic() {
        let b0: u8 = kani::any();
        let b1: u8 = kani::any();
        let b2: u8 = kani::any();
        let b3: u8 = kani::any();
        let b4: u8 = kani::any();
        let b5: u8 = kani::any();
        let b6: u8 = kani::any();
        let b7: u8 = kani::any();
        let b8: u8 = kani::any();
        let b9: u8 = kani::any();
        let b10: u8 = kani::any();
        let b11: u8 = kani::any();
        let b12: u8 = kani::any();
        let b13: u8 = kani::any();
        let b14: u8 = kani::any();
        let b15: u8 = kani::any();
        let b16: u8 = kani::any();
        let b17: u8 = kani::any();
        let b18: u8 = kani::any();
        let b19: u8 = kani::any();
        let b20: u8 = kani::any();
        let b21: u8 = kani::any();
        let b22: u8 = kani::any();
        let b23: u8 = kani::any();

        let data = [
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18,
            b19, b20, b21, b22, b23,
        ];

        // Must not panic regardless of input
        let _ = parse_mda_header(&data);
    }
}
