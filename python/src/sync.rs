//! Python bindings for clock synchronization primitives.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    ClockOffset as ZsClockOffset, LinearDrift as ZsLinearDrift, OffsetBuffer as ZsOffsetBuffer,
    SampleClock as ZsSampleClock,
};

/// Clock offset measurement from a synchronization exchange.
///
/// Represents the estimated offset between local and remote clocks,
/// along with round-trip time for quality assessment.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// # From NTP-style timestamps
/// offset = zbci.ClockOffset.from_ntp(t1=1.0, t2=1.001, t3=1.002, t4=1.003)
/// print(offset.offset, offset.rtt, offset.quality)
/// ```
#[pyclass]
pub struct ClockOffset {
    pub(crate) inner: ZsClockOffset,
}

#[pymethods]
impl ClockOffset {
    /// Create a new clock offset measurement.
    ///
    /// Args:
    ///     offset (float): Estimated clock offset in seconds.
    ///     local_time (float): Local time of measurement in seconds.
    ///     rtt (float): Round-trip time in seconds.
    #[new]
    fn new(offset: f64, local_time: f64, rtt: f64) -> Self {
        Self {
            inner: ZsClockOffset::new(offset, local_time, rtt),
        }
    }

    /// Create a clock offset from NTP-style timestamps.
    ///
    /// Args:
    ///     t1 (float): Client send time.
    ///     t2 (float): Server receive time.
    ///     t3 (float): Server send time.
    ///     t4 (float): Client receive time.
    #[staticmethod]
    fn from_ntp(t1: f64, t2: f64, t3: f64, t4: f64) -> Self {
        Self {
            inner: ZsClockOffset::from_ntp(t1, t2, t3, t4),
        }
    }

    /// Estimated clock offset in seconds.
    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset()
    }

    /// Local time of measurement in seconds.
    #[getter]
    fn local_time(&self) -> f64 {
        self.inner.local_time()
    }

    /// Round-trip time in seconds.
    #[getter]
    fn rtt(&self) -> f64 {
        self.inner.rtt()
    }

    /// Quality metric (inverse of RTT). Higher is better.
    #[getter]
    fn quality(&self) -> f64 {
        self.inner.quality()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockOffset(offset={:.6}, rtt={:.6}, quality={:.2})",
            self.inner.offset(),
            self.inner.rtt(),
            self.inner.quality()
        )
    }
}

/// Sample clock for converting between sample indices and timestamps.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// clock = zbci.SampleClock(start_time=0.0, sample_rate=250.0)
/// t = clock.sample_to_time(1000)  # 4.0 seconds
/// idx = clock.time_to_sample(4.0)  # 1000
/// ```
#[pyclass]
pub struct SampleClock {
    inner: ZsSampleClock,
}

#[pymethods]
impl SampleClock {
    /// Create a new sample clock.
    ///
    /// Args:
    ///     start_time (float): Start time in seconds.
    ///     sample_rate (float): Sample rate in Hz.
    #[new]
    fn new(start_time: f64, sample_rate: f64) -> Self {
        Self {
            inner: ZsSampleClock::new(start_time, sample_rate),
        }
    }

    /// Convert a sample index to a timestamp.
    fn sample_to_time(&self, sample_index: u64) -> f64 {
        self.inner.sample_to_time(sample_index)
    }

    /// Convert a timestamp to the nearest sample index.
    fn time_to_sample(&self, time: f64) -> u64 {
        self.inner.time_to_sample(time)
    }

    /// Convert a timestamp to a fractional sample index.
    fn time_to_sample_frac(&self, time: f64) -> f64 {
        self.inner.time_to_sample_frac(time)
    }

    /// Start time in seconds.
    #[getter]
    fn start_time(&self) -> f64 {
        self.inner.start_time()
    }

    /// Sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    fn __repr__(&self) -> String {
        format!(
            "SampleClock(start_time={}, sample_rate={} Hz)",
            self.inner.start_time(),
            self.inner.sample_rate()
        )
    }
}

/// Linear drift estimator for clock correction.
///
/// Uses linear regression on clock offset measurements to estimate
/// and correct for linear clock drift.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// drift = zbci.LinearDrift()
/// drift.add_measurement(0.0, 0.001)
/// drift.add_measurement(10.0, 0.002)
/// corrected = drift.correct(5.0)
/// ```
#[pyclass]
pub struct LinearDrift {
    inner: ZsLinearDrift,
}

#[pymethods]
impl LinearDrift {
    /// Create a new linear drift estimator.
    #[new]
    fn new() -> Self {
        Self {
            inner: ZsLinearDrift::new(),
        }
    }

    /// Add a time/offset measurement pair.
    ///
    /// Args:
    ///     time (float): Local time of measurement in seconds.
    ///     offset (float): Measured clock offset in seconds.
    fn add_measurement(&mut self, time: f64, offset: f64) {
        self.inner.add_measurement(time, offset);
    }

    /// Add a ClockOffset measurement.
    ///
    /// Args:
    ///     offset (ClockOffset): A clock offset measurement.
    fn add_offset(&mut self, offset: &ClockOffset) {
        self.inner.add_offset(&offset.inner);
    }

    /// Correct a local timestamp for estimated drift.
    ///
    /// Args:
    ///     local_time (float): Local time to correct.
    ///
    /// Returns:
    ///     float: Corrected time estimate.
    fn correct(&self, local_time: f64) -> f64 {
        self.inner.correct(local_time)
    }

    /// Estimated drift slope (seconds of offset per second).
    #[getter]
    fn slope(&self) -> f64 {
        self.inner.slope()
    }

    /// Estimated intercept (initial offset in seconds).
    #[getter]
    fn intercept(&self) -> f64 {
        self.inner.intercept()
    }

    /// Number of measurements added.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Reset the estimator state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearDrift(count={}, slope={:.6e}, intercept={:.6e})",
            self.inner.count(),
            self.inner.slope(),
            self.inner.intercept()
        )
    }
}

// --- OffsetBuffer enum dispatch ---

enum OffsetBufferInner {
    N8(ZsOffsetBuffer<8>),
    N16(ZsOffsetBuffer<16>),
    N32(ZsOffsetBuffer<32>),
    N64(ZsOffsetBuffer<64>),
    N128(ZsOffsetBuffer<128>),
}

/// Circular buffer of clock offset measurements for filtering.
///
/// Stores recent clock offset measurements and provides access to
/// the best (lowest RTT), latest, or median offset.
///
/// # Example
/// ```python
/// import zpybci as zbci
///
/// buf = zbci.OffsetBuffer(capacity=32)
/// for i in range(10):
///     offset = zbci.ClockOffset(offset=0.001 * i, local_time=float(i), rtt=0.01)
///     buf.add(offset)
/// best = buf.best_offset()
/// ```
#[pyclass]
pub struct OffsetBuffer {
    inner: OffsetBufferInner,
}

#[pymethods]
impl OffsetBuffer {
    /// Create a new offset buffer.
    ///
    /// Args:
    ///     capacity (int): Buffer capacity. Must be 8, 16, 32, 64, or 128.
    #[new]
    #[pyo3(signature = (capacity=32))]
    fn new(capacity: usize) -> PyResult<Self> {
        let inner = match capacity {
            8 => OffsetBufferInner::N8(ZsOffsetBuffer::new()),
            16 => OffsetBufferInner::N16(ZsOffsetBuffer::new()),
            32 => OffsetBufferInner::N32(ZsOffsetBuffer::new()),
            64 => OffsetBufferInner::N64(ZsOffsetBuffer::new()),
            128 => OffsetBufferInner::N128(ZsOffsetBuffer::new()),
            _ => {
                return Err(PyValueError::new_err(
                    "capacity must be 8, 16, 32, 64, or 128",
                ))
            }
        };
        Ok(Self { inner })
    }

    /// Add a clock offset measurement to the buffer.
    ///
    /// Args:
    ///     offset (ClockOffset): The clock offset to add.
    fn add(&mut self, offset: &ClockOffset) {
        macro_rules! add {
            ($buf:expr) => {
                $buf.add(offset.inner)
            };
        }
        match &mut self.inner {
            OffsetBufferInner::N8(b) => add!(b),
            OffsetBufferInner::N16(b) => add!(b),
            OffsetBufferInner::N32(b) => add!(b),
            OffsetBufferInner::N64(b) => add!(b),
            OffsetBufferInner::N128(b) => add!(b),
        }
    }

    /// Get the best offset (lowest RTT) in the buffer.
    ///
    /// Returns:
    ///     ClockOffset or None: The offset with the lowest round-trip time.
    fn best_offset(&self) -> Option<ClockOffset> {
        macro_rules! best {
            ($buf:expr) => {
                $buf.best_offset().map(|o| ClockOffset { inner: o })
            };
        }
        match &self.inner {
            OffsetBufferInner::N8(b) => best!(b),
            OffsetBufferInner::N16(b) => best!(b),
            OffsetBufferInner::N32(b) => best!(b),
            OffsetBufferInner::N64(b) => best!(b),
            OffsetBufferInner::N128(b) => best!(b),
        }
    }

    /// Get the most recently added offset.
    ///
    /// Returns:
    ///     ClockOffset or None: The most recent offset.
    fn latest_offset(&self) -> Option<ClockOffset> {
        macro_rules! latest {
            ($buf:expr) => {
                $buf.latest_offset().map(|o| ClockOffset { inner: o })
            };
        }
        match &self.inner {
            OffsetBufferInner::N8(b) => latest!(b),
            OffsetBufferInner::N16(b) => latest!(b),
            OffsetBufferInner::N32(b) => latest!(b),
            OffsetBufferInner::N64(b) => latest!(b),
            OffsetBufferInner::N128(b) => latest!(b),
        }
    }

    /// Get the median offset by value.
    ///
    /// Returns:
    ///     ClockOffset or None: The median offset.
    fn median_offset(&self) -> Option<ClockOffset> {
        macro_rules! median {
            ($buf:expr) => {
                $buf.median_offset().map(|o| ClockOffset { inner: o })
            };
        }
        match &self.inner {
            OffsetBufferInner::N8(b) => median!(b),
            OffsetBufferInner::N16(b) => median!(b),
            OffsetBufferInner::N32(b) => median!(b),
            OffsetBufferInner::N64(b) => median!(b),
            OffsetBufferInner::N128(b) => median!(b),
        }
    }

    /// Number of offsets currently stored.
    #[getter]
    fn count(&self) -> usize {
        match &self.inner {
            OffsetBufferInner::N8(b) => b.count(),
            OffsetBufferInner::N16(b) => b.count(),
            OffsetBufferInner::N32(b) => b.count(),
            OffsetBufferInner::N64(b) => b.count(),
            OffsetBufferInner::N128(b) => b.count(),
        }
    }

    /// Reset the buffer, removing all stored offsets.
    fn reset(&mut self) {
        match &mut self.inner {
            OffsetBufferInner::N8(b) => b.reset(),
            OffsetBufferInner::N16(b) => b.reset(),
            OffsetBufferInner::N32(b) => b.reset(),
            OffsetBufferInner::N64(b) => b.reset(),
            OffsetBufferInner::N128(b) => b.reset(),
        }
    }

    fn __repr__(&self) -> String {
        format!("OffsetBuffer(count={})", self.count())
    }
}
