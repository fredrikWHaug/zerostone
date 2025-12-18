use core::mem::MaybeUninit;

pub struct CircularBuffer<T, const N:usize> {
    buffer: [MaybeUninit<T>; N],
    head: usize,
    tail: usize,
    len: usize,
}

fn main() {
    let mut buf: CircularBuffer<i64, 1064> = CircularBuffer::new();
    buf.push(7);
}
