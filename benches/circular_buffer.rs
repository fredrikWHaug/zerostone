use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zerostone::CircularBuffer;

// Target from proposal: 1024-channel ring buffer insert <1 μs (30M samples/sec)
fn bench_push_pop_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("circular_buffer_throughput");

    for size in [64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_channels", size)),
            size,
            |b, &size| {
                let mut buffer: CircularBuffer<f32, 2048> = CircularBuffer::new();

                b.iter(|| {
                    for i in 0..size {
                        let _ = buffer.push(black_box(i as f32));
                    }
                    for _ in 0..size {
                        let _ = black_box(buffer.pop());
                    }
                });
            },
        );
    }

    group.finish();
}

// Measure single push latency - target <1 μs for 1024 channels
fn bench_single_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_push_latency");

    let mut buffer: CircularBuffer<f32, 2048> = CircularBuffer::new();

    group.bench_function("push_single_sample", |b| {
        b.iter(|| {
            let _ = buffer.push(black_box(42.0f32));
        });
    });

    group.finish();
}

// Measure single pop latency
fn bench_single_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pop_latency");

    let mut buffer: CircularBuffer<f32, 2048> = CircularBuffer::new();
    // Pre-fill the buffer
    for i in 0..1000 {
        buffer.push(i as f32).unwrap();
    }

    group.bench_function("pop_single_sample", |b| {
        b.iter(|| {
            let _ = black_box(buffer.pop());
        });
    });

    group.finish();
}

// Simulate streaming multi-channel BCI data
fn bench_streaming_bci_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_bci");

    // Struct representing a multi-channel sample
    #[derive(Clone, Copy)]
    struct Sample32 {
        channels: [f32; 32],
    }

    for channels in [32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_ch_streaming", channels)),
            channels,
            |b, _| {
                let mut buffer: CircularBuffer<f32, 2048> = CircularBuffer::new();

                b.iter(|| {
                    // Simulate acquiring 100 samples from each channel
                    for sample in 0..100 {
                        for ch in 0..*channels {
                            let value = (sample * channels + ch) as f32;
                            let _ = buffer.push(black_box(value));
                        }
                    }

                    // Process the samples
                    for _ in 0..(100 * channels) {
                        let _ = black_box(buffer.pop());
                    }
                });
            },
        );
    }

    group.finish();
}

// Test try_push (shared reference) performance
fn bench_try_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("try_push_performance");

    let buffer: CircularBuffer<f32, 2048> = CircularBuffer::new();

    group.bench_function("try_push_single", |b| {
        b.iter(|| {
            let _ = buffer.try_push(black_box(42.0f32));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_push_pop_throughput,
    bench_single_push,
    bench_single_pop,
    bench_streaming_bci_data,
    bench_try_push
);
criterion_main!(benches);
