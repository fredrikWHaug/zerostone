#![no_std]

use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

mod stats;
pub use stats::OnlineStats;

/// Lock-free circular buffer for single-producer/single-consumer scenarios.
/// Uses atomic operations for wait-free push/pop with deterministic latency.
///
/// # Memory Layout
/// - `buffer`: Static allocation of N elements using `MaybeUninit<T>`
/// - `head`: Atomic write position (producer increments)
/// - `tail`: Atomic read position (consumer increments)
///
/// # Thread Safety
/// Safe for concurrent access with one producer thread (push) and one consumer thread (pop).
pub struct CircularBuffer<T, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T, const N: usize> CircularBuffer<T, N> {
    /// Creates a new empty circular buffer.
    ///
    /// # Panics
    /// Panics if N is 0 or not a power of 2 (for efficient modulo via bit masking).
    pub const fn new() -> Self {
        assert!(N > 0 && N.is_power_of_two(), "N must be a power of 2");
        Self {
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Pushes a value into the buffer (producer operation).
    ///
    /// Returns `Err(value)` if the buffer is full.
    ///
    /// # Memory Ordering
    /// Uses `Relaxed` for tail read and `Release` for head write to ensure
    /// the written data is visible to the consumer thread.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);

        // Check if buffer is full
        let next_head = (head + 1) & (N - 1);
        if next_head == tail {
            return Err(value);
        }

        // Write the value
        self.buffer[head].write(value);

        // Publish the write with Release ordering
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Attempts to push without requiring mutable reference.
    /// Useful for scenarios where the buffer is behind shared reference.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let next_head = (head + 1) & (N - 1);
        if next_head == tail {
            return Err(value);
        }

        unsafe {
            let buffer_ptr = self.buffer.as_ptr() as *mut MaybeUninit<T>;
            (*buffer_ptr.add(head)).write(value);
        }

        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Pops a value from the buffer (consumer operation).
    ///
    /// Returns `None` if the buffer is empty.
    ///
    /// # Memory Ordering
    /// Uses `Acquire` for head read to ensure we see all writes from the producer,
    /// and `Release` for tail write to publish consumption.
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if buffer is empty
        if tail == head {
            return None;
        }

        // Read the value
        let value = unsafe { self.buffer[tail].assume_init_read() };

        // Publish the read with Release ordering
        let next_tail = (tail + 1) & (N - 1);
        self.tail.store(next_tail, Ordering::Release);

        Some(value)
    }

    /// Returns the number of elements currently in the buffer.
    ///
    /// Note: In concurrent scenarios, this value may be stale immediately after reading.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);

        if head >= tail {
            head - tail
        } else {
            N - tail + head
        }
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head == tail
    }

    /// Returns `true` if the buffer is full.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let next_head = (head + 1) & (N - 1);
        next_head == tail
    }

    /// Returns the capacity of the buffer.
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<T, const N: usize> Drop for CircularBuffer<T, N> {
    fn drop(&mut self) {
        // Drain all remaining elements to ensure proper cleanup
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_buffer_new() {
        let buffer: CircularBuffer<i32, 8> = CircularBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.capacity(), 8);
    }

    #[test]
    fn test_push_pop_single() {
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        assert_eq!(buffer.push(42), Ok(()));
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());

        assert_eq!(buffer.pop(), Some(42));
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_push_pop_multiple() {
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        for i in 0..5 {
            assert_eq!(buffer.push(i), Ok(()));
        }
        assert_eq!(buffer.len(), 5);

        for i in 0..5 {
            assert_eq!(buffer.pop(), Some(i));
        }
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_full() {
        let mut buffer: CircularBuffer<i32, 4> = CircularBuffer::new();

        // Fill buffer (capacity - 1 = 3 elements for power-of-2 sized buffer)
        assert_eq!(buffer.push(1), Ok(()));
        assert_eq!(buffer.push(2), Ok(()));
        assert_eq!(buffer.push(3), Ok(()));

        // Buffer should now be full
        assert!(buffer.is_full());
        assert_eq!(buffer.push(4), Err(4));
    }

    #[test]
    fn test_wrap_around() {
        let mut buffer: CircularBuffer<i32, 4> = CircularBuffer::new();

        // Fill and drain multiple times to test wrap-around
        for cycle in 0..3 {
            for i in 0..3 {
                assert_eq!(buffer.push(cycle * 10 + i), Ok(()));
            }
            for i in 0..3 {
                assert_eq!(buffer.pop(), Some(cycle * 10 + i));
            }
        }
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_try_push() {
        let buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        assert_eq!(buffer.try_push(1), Ok(()));
        assert_eq!(buffer.try_push(2), Ok(()));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_drop_cleanup() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROP_COUNT.store(0, Ordering::Relaxed);
        {
            let mut buffer: CircularBuffer<DropCounter, 8> = CircularBuffer::new();
            buffer.push(DropCounter).unwrap();
            buffer.push(DropCounter).unwrap();
            buffer.push(DropCounter).unwrap();
            // Buffer goes out of scope here
        }

        // All 3 elements should have been dropped
        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_streaming_scenario() {
        let mut buffer: CircularBuffer<f32, 16> = CircularBuffer::new();

        // Simulate streaming: push some, pop some, repeat
        for chunk in 0..5 {
            // Push 10 samples
            for i in 0..10 {
                let sample = (chunk * 10 + i) as f32;
                buffer.push(sample).unwrap();
            }

            // Process 10 samples
            for i in 0..10 {
                let expected = (chunk * 10 + i) as f32;
                assert_eq!(buffer.pop(), Some(expected));
            }
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_capacity_minus_one_elements() {
        // For a power-of-2 buffer, we can store N-1 elements
        let mut buffer: CircularBuffer<i32, 8> = CircularBuffer::new();

        // Fill with 7 elements (8-1)
        for i in 0..7 {
            assert_eq!(buffer.push(i), Ok(()));
        }

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 7);
    }
}
