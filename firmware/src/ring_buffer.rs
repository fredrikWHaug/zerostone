//! Fixed-size, single-producer single-consumer ring buffer for sample frames.
//!
//! Stores raw i16 ADC frames from the Intan SPI driver. Designed for
//! cooperative Embassy tasks where producer and consumer never run
//! concurrently, so no atomics are needed.

/// A fixed-size ring buffer holding multi-channel sample frames.
///
/// - `C`: number of channels per frame (e.g., 32).
/// - `N`: buffer capacity in frames (must be >= 1).
///
/// All storage is inline (stack or static) with zero heap allocation.
pub struct FrameRingBuffer<const C: usize, const N: usize> {
    /// Backing storage for `N` frames of `C` channels each.
    buf: [[i16; C]; N],
    /// Index of the next slot to write into.
    head: usize,
    /// Index of the next slot to read from.
    tail: usize,
    /// Number of frames currently stored.
    count: usize,
}

impl<const C: usize, const N: usize> FrameRingBuffer<C, N> {
    /// Creates a new empty ring buffer.
    pub const fn new() -> Self {
        Self {
            buf: [[0i16; C]; N],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Pushes a sample frame into the buffer.
    ///
    /// Returns `true` if the frame was stored, `false` if the buffer is full.
    pub fn push(&mut self, frame: &[i16; C]) -> bool {
        if self.count == N {
            return false;
        }
        self.buf[self.head] = *frame;
        self.head = (self.head + 1) % N;
        self.count += 1;
        true
    }

    /// Pops the oldest sample frame from the buffer.
    ///
    /// Returns `true` if a frame was copied into `out`, `false` if empty.
    pub fn pop(&mut self, out: &mut [i16; C]) -> bool {
        if self.count == 0 {
            return false;
        }
        *out = self.buf[self.tail];
        self.tail = (self.tail + 1) % N;
        self.count -= 1;
        true
    }

    /// Returns the number of frames currently in the buffer.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the buffer has no frames.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.count == N
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer_is_empty() {
        let buf = FrameRingBuffer::<4, 8>::new();
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn push_pop_fifo_ordering() {
        let mut buf = FrameRingBuffer::<2, 4>::new();
        assert!(buf.push(&[10, 20]));
        assert!(buf.push(&[30, 40]));
        assert_eq!(buf.len(), 2);

        let mut out = [0i16; 2];
        assert!(buf.pop(&mut out));
        assert_eq!(out, [10, 20]);
        assert!(buf.pop(&mut out));
        assert_eq!(out, [30, 40]);
        assert!(buf.is_empty());
    }

    #[test]
    fn full_buffer_rejects_push() {
        let mut buf = FrameRingBuffer::<1, 2>::new();
        assert!(buf.push(&[1]));
        assert!(buf.push(&[2]));
        assert!(buf.is_full());
        assert!(!buf.push(&[3]));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn empty_buffer_rejects_pop() {
        let mut buf = FrameRingBuffer::<1, 4>::new();
        let mut out = [0i16; 1];
        assert!(!buf.pop(&mut out));
    }

    #[test]
    fn wrap_around_behavior() {
        // Capacity 3: fill, drain, refill to force wrap-around.
        let mut buf = FrameRingBuffer::<1, 3>::new();
        let mut out = [0i16; 1];

        // Fill completely.
        assert!(buf.push(&[1]));
        assert!(buf.push(&[2]));
        assert!(buf.push(&[3]));
        assert!(buf.is_full());

        // Drain two.
        assert!(buf.pop(&mut out));
        assert_eq!(out, [1]);
        assert!(buf.pop(&mut out));
        assert_eq!(out, [2]);

        // Push two more (head wraps around).
        assert!(buf.push(&[4]));
        assert!(buf.push(&[5]));
        assert!(buf.is_full());

        // Drain all — should be 3, 4, 5.
        assert!(buf.pop(&mut out));
        assert_eq!(out, [3]);
        assert!(buf.pop(&mut out));
        assert_eq!(out, [4]);
        assert!(buf.pop(&mut out));
        assert_eq!(out, [5]);
        assert!(buf.is_empty());
    }

    #[test]
    fn multichannel_frame_integrity() {
        let mut buf = FrameRingBuffer::<4, 2>::new();
        let frame_a = [100, -200, 300, -400];
        let frame_b = [i16::MIN, i16::MAX, 0, 1];

        assert!(buf.push(&frame_a));
        assert!(buf.push(&frame_b));

        let mut out = [0i16; 4];
        assert!(buf.pop(&mut out));
        assert_eq!(out, frame_a);
        assert!(buf.pop(&mut out));
        assert_eq!(out, frame_b);
    }
}
