//! Python bindings for probe geometry.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::probe::{self as zs_probe, ProbeLayout as ZsProbeLayout};

// --- Macro for const-generic dispatch ---

macro_rules! make_probe_inner {
    ($name:ident, $c:expr) => {
        #[allow(non_camel_case_types)]
        struct $name(ZsProbeLayout<$c>);

        impl $name {
            fn from_positions(positions: &[[f64; 2]]) -> Self {
                let mut pos = [[0.0f64; 2]; $c];
                for (i, p) in positions.iter().enumerate().take($c) {
                    pos[i] = *p;
                }
                Self(ZsProbeLayout::new(pos))
            }

            fn linear(pitch: f64) -> Self {
                Self(ZsProbeLayout::<$c>::linear(pitch))
            }

            fn channel_distance(&self, ch_a: usize, ch_b: usize) -> f64 {
                self.0.channel_distance(ch_a, ch_b)
            }

            fn neighbor_channels(&self, channel: usize, radius: f64) -> Vec<usize> {
                let mut output = vec![0usize; $c];
                let n = self.0.neighbor_channels(channel, radius, &mut output);
                output.truncate(n);
                output
            }

            fn nearest_channels(&self, channel: usize, k: usize) -> Vec<usize> {
                let mut output = vec![0usize; $c];
                let n = self.0.nearest_channels(channel, k, &mut output);
                output.truncate(n);
                output
            }

            fn spatial_extent(&self) -> (f64, f64) {
                self.0.spatial_extent()
            }

            fn n_channels(&self) -> usize {
                self.0.n_channels()
            }
        }
    };
}

make_probe_inner!(Probe4, 4);
make_probe_inner!(Probe8, 8);
make_probe_inner!(Probe16, 16);
make_probe_inner!(Probe32, 32);
make_probe_inner!(Probe64, 64);
make_probe_inner!(Probe128, 128);

enum ProbeInner {
    C4(Probe4),
    C8(Probe8),
    C16(Probe16),
    C32(Box<Probe32>),
    C64(Box<Probe64>),
    C128(Box<Probe128>),
}

/// Probe geometry for multi-channel electrode arrays.
///
/// Stores 2D positions for each channel on a probe (Neuropixels, tetrodes,
/// high-density arrays) and provides spatial queries used by spike sorting.
///
/// Supports 4, 8, 16, 32, 64, or 128 channels.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// probe = zbci.ProbeLayout.linear(8, 25.0)
/// d = probe.channel_distance(0, 3)
/// neighbors = probe.neighbor_channels(3, 30.0)
/// ```
#[pyclass]
pub struct ProbeLayout {
    inner: ProbeInner,
}

#[pymethods]
impl ProbeLayout {
    /// Create a probe from explicit channel positions.
    ///
    /// Args:
    ///     positions (list[list[float]]): List of [x, y] positions in micrometers.
    ///         Length must be 4, 8, 16, 32, 64, or 128.
    ///
    /// Returns:
    ///     ProbeLayout: A probe geometry object.
    #[new]
    fn new(positions: Vec<[f64; 2]>) -> PyResult<Self> {
        let n = positions.len();
        let inner = match n {
            4 => ProbeInner::C4(Probe4::from_positions(&positions)),
            8 => ProbeInner::C8(Probe8::from_positions(&positions)),
            16 => ProbeInner::C16(Probe16::from_positions(&positions)),
            32 => ProbeInner::C32(Box::new(Probe32::from_positions(&positions))),
            64 => ProbeInner::C64(Box::new(Probe64::from_positions(&positions))),
            128 => ProbeInner::C128(Box::new(Probe128::from_positions(&positions))),
            _ => {
                return Err(PyValueError::new_err(
                    "n_channels must be 4, 8, 16, 32, 64, or 128",
                ));
            }
        };
        Ok(Self { inner })
    }

    /// Create a linear probe with channels spaced along the y-axis.
    ///
    /// Args:
    ///     n_channels (int): Number of channels (4, 8, 16, 32, 64, or 128).
    ///     pitch (float): Spacing between channels in micrometers.
    ///
    /// Returns:
    ///     ProbeLayout: A linear probe geometry.
    #[staticmethod]
    fn linear(n_channels: usize, pitch: f64) -> PyResult<Self> {
        let inner = match n_channels {
            4 => ProbeInner::C4(Probe4::linear(pitch)),
            8 => ProbeInner::C8(Probe8::linear(pitch)),
            16 => ProbeInner::C16(Probe16::linear(pitch)),
            32 => ProbeInner::C32(Box::new(Probe32::linear(pitch))),
            64 => ProbeInner::C64(Box::new(Probe64::linear(pitch))),
            128 => ProbeInner::C128(Box::new(Probe128::linear(pitch))),
            _ => {
                return Err(PyValueError::new_err(
                    "n_channels must be 4, 8, 16, 32, 64, or 128",
                ));
            }
        };
        Ok(Self { inner })
    }

    /// Create a 4-channel tetrode geometry.
    ///
    /// Channels are placed at the corners of a square with side length `pitch`
    /// centered on the origin.
    ///
    /// Args:
    ///     pitch (float): Side length of the tetrode square in micrometers.
    ///
    /// Returns:
    ///     ProbeLayout: A tetrode probe geometry (4 channels).
    #[staticmethod]
    fn tetrode(pitch: f64) -> Self {
        let probe = zs_probe::tetrode(pitch);
        Self {
            inner: ProbeInner::C4(Probe4(probe)),
        }
    }

    /// Euclidean distance between two channels.
    ///
    /// Args:
    ///     ch_a (int): First channel index.
    ///     ch_b (int): Second channel index.
    ///
    /// Returns:
    ///     float: Distance in micrometers, or NaN if index out of range.
    fn channel_distance(&self, ch_a: usize, ch_b: usize) -> f64 {
        match &self.inner {
            ProbeInner::C4(p) => p.channel_distance(ch_a, ch_b),
            ProbeInner::C8(p) => p.channel_distance(ch_a, ch_b),
            ProbeInner::C16(p) => p.channel_distance(ch_a, ch_b),
            ProbeInner::C32(p) => p.channel_distance(ch_a, ch_b),
            ProbeInner::C64(p) => p.channel_distance(ch_a, ch_b),
            ProbeInner::C128(p) => p.channel_distance(ch_a, ch_b),
        }
    }

    /// Find all channels within a radius of a given channel.
    ///
    /// Args:
    ///     channel (int): Query channel index.
    ///     radius (float): Search radius in micrometers.
    ///
    /// Returns:
    ///     list[int]: Indices of neighbor channels (excluding the query channel).
    fn neighbor_channels(&self, channel: usize, radius: f64) -> Vec<usize> {
        match &self.inner {
            ProbeInner::C4(p) => p.neighbor_channels(channel, radius),
            ProbeInner::C8(p) => p.neighbor_channels(channel, radius),
            ProbeInner::C16(p) => p.neighbor_channels(channel, radius),
            ProbeInner::C32(p) => p.neighbor_channels(channel, radius),
            ProbeInner::C64(p) => p.neighbor_channels(channel, radius),
            ProbeInner::C128(p) => p.neighbor_channels(channel, radius),
        }
    }

    /// Find the k nearest channels to a given channel.
    ///
    /// Args:
    ///     channel (int): Query channel index.
    ///     k (int): Number of nearest neighbors to find.
    ///
    /// Returns:
    ///     list[int]: Indices of nearest channels, sorted by ascending distance.
    fn nearest_channels(&self, channel: usize, k: usize) -> Vec<usize> {
        match &self.inner {
            ProbeInner::C4(p) => p.nearest_channels(channel, k),
            ProbeInner::C8(p) => p.nearest_channels(channel, k),
            ProbeInner::C16(p) => p.nearest_channels(channel, k),
            ProbeInner::C32(p) => p.nearest_channels(channel, k),
            ProbeInner::C64(p) => p.nearest_channels(channel, k),
            ProbeInner::C128(p) => p.nearest_channels(channel, k),
        }
    }

    /// Spatial extent of the probe.
    ///
    /// Returns:
    ///     tuple[float, float]: (x_range, y_range) in micrometers.
    fn spatial_extent(&self) -> (f64, f64) {
        match &self.inner {
            ProbeInner::C4(p) => p.spatial_extent(),
            ProbeInner::C8(p) => p.spatial_extent(),
            ProbeInner::C16(p) => p.spatial_extent(),
            ProbeInner::C32(p) => p.spatial_extent(),
            ProbeInner::C64(p) => p.spatial_extent(),
            ProbeInner::C128(p) => p.spatial_extent(),
        }
    }

    /// Number of channels.
    #[getter]
    fn n_channels(&self) -> usize {
        match &self.inner {
            ProbeInner::C4(p) => p.n_channels(),
            ProbeInner::C8(p) => p.n_channels(),
            ProbeInner::C16(p) => p.n_channels(),
            ProbeInner::C32(p) => p.n_channels(),
            ProbeInner::C64(p) => p.n_channels(),
            ProbeInner::C128(p) => p.n_channels(),
        }
    }

    fn __repr__(&self) -> String {
        format!("ProbeLayout(n_channels={})", self.n_channels())
    }
}

impl ProbeLayout {
    /// Get the number of channels (accessible from other modules in the crate).
    pub(crate) fn n_channels_inner(&self) -> usize {
        match &self.inner {
            ProbeInner::C4(p) => p.n_channels(),
            ProbeInner::C8(p) => p.n_channels(),
            ProbeInner::C16(p) => p.n_channels(),
            ProbeInner::C32(p) => p.n_channels(),
            ProbeInner::C64(p) => p.n_channels(),
            ProbeInner::C128(p) => p.n_channels(),
        }
    }
}

/// Access the inner `ZsProbeLayout<C>` reference for a given const-generic C.
///
/// Returns `Err` if the probe's channel count does not match `C`.
pub fn with_probe_ref<const C: usize, F, R>(probe: &ProbeLayout, f: F) -> PyResult<R>
where
    F: FnOnce(&ZsProbeLayout<C>) -> R,
{
    // We need to match the const-generic C against the stored variant.
    // Since Rust doesn't allow runtime matching on const generics, we check
    // the stored channel count and transmute only when it matches.
    let n = probe.n_channels();
    if n != C {
        return Err(PyValueError::new_err(format!(
            "Probe has {} channels but operation expects {}",
            n, C
        )));
    }

    // Extract the reference based on the enum variant.
    // We only reach the matching branch because we checked n == C above.
    macro_rules! try_extract {
        ($variant:ident, $inner_field:expr, $expected_c:expr) => {
            if C == $expected_c {
                if let ProbeInner::$variant(ref p) = probe.inner {
                    // SAFETY: C == $expected_c is guaranteed by the if-guard,
                    // so ZsProbeLayout<$expected_c> has the same type as ZsProbeLayout<C>.
                    let ptr = &p.0 as *const ZsProbeLayout<$expected_c> as *const ZsProbeLayout<C>;
                    return Ok(f(unsafe { &*ptr }));
                }
            }
        };
    }

    try_extract!(C4, inner, 4);
    try_extract!(C8, inner, 8);
    try_extract!(C16, inner, 16);
    try_extract!(C32, inner, 32);
    try_extract!(C64, inner, 64);
    try_extract!(C128, inner, 128);

    Err(PyValueError::new_err(format!(
        "Unsupported probe channel count {}",
        n
    )))
}

/// Register probe geometry classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ProbeLayout>()?;
    Ok(())
}
