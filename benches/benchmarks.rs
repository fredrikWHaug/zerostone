use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zerostone::{BiquadCoeffs, CircularBuffer, FirFilter, IirFilter, ThresholdDetector};

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

// IirFilter benchmarks - target: <100 ns/sample for 32 channels @ 4th order
fn bench_iir_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iir_filter");

    // 4th-order Butterworth lowpass (2 biquad sections)
    let mut filter: IirFilter<2> = IirFilter::new([
        BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
        BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
    ]);

    group.bench_function("single_sample_4th_order", |b| {
        b.iter(|| {
            let _ = black_box(filter.process_sample(black_box(1.0)));
        });
    });

    // Multi-channel simulation (32 channels)
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_4th_order", |b| {
        let mut filters: Vec<IirFilter<2>> = (0..32)
            .map(|_| {
                IirFilter::new([
                    BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
                    BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
                ])
            })
            .collect();

        b.iter(|| {
            for filter in filters.iter_mut() {
                let _ = black_box(filter.process_sample(black_box(1.0)));
            }
        });
    });

    // Block processing
    group.bench_function("block_256_samples", |b| {
        let mut samples = [0.5f32; 256];
        b.iter(|| {
            filter.process_block(black_box(&mut samples));
        });
    });

    // Notch filter (60 Hz powerline removal)
    let mut notch_filter: IirFilter<1> = IirFilter::new([BiquadCoeffs::notch(1000.0, 60.0, 30.0)]);

    group.bench_function("notch_single_sample", |b| {
        b.iter(|| {
            let _ = black_box(notch_filter.process_sample(black_box(1.0)));
        });
    });

    // Multi-channel notch (32 channels)
    group.bench_function("notch_32_channels", |b| {
        let mut notch_filters: Vec<IirFilter<1>> = (0..32)
            .map(|_| IirFilter::new([BiquadCoeffs::notch(1000.0, 60.0, 30.0)]))
            .collect();

        b.iter(|| {
            for filter in notch_filters.iter_mut() {
                let _ = black_box(filter.process_sample(black_box(1.0)));
            }
        });
    });

    group.finish();
}

// FirFilter benchmarks
fn bench_fir_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("fir_filter");

    // 32-tap FIR (common for BCI applications)
    let mut filter: FirFilter<32> = FirFilter::moving_average();

    group.bench_function("single_sample_32_taps", |b| {
        b.iter(|| {
            let _ = black_box(filter.process_sample(black_box(1.0)));
        });
    });

    // 64-tap FIR
    let mut filter64: FirFilter<64> = FirFilter::moving_average();
    group.bench_function("single_sample_64_taps", |b| {
        b.iter(|| {
            let _ = black_box(filter64.process_sample(black_box(1.0)));
        });
    });

    // Block processing
    group.bench_function("block_256_samples_32_taps", |b| {
        let mut samples = [0.5f32; 256];
        b.iter(|| {
            filter.process_block(black_box(&mut samples));
        });
    });

    group.finish();
}

// ThresholdDetector benchmarks - target: <10 μs for 1024 channels
fn bench_threshold_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_detector");

    for num_channels in [32, 128, 512, 1024].iter() {
        group.throughput(Throughput::Elements(*num_channels as u64));

        match *num_channels {
            32 => {
                let mut detector: ThresholdDetector<32> = ThresholdDetector::new(3.0, 100);
                let samples = [1.0f32; 32];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{}_channels", num_channels)),
                    num_channels,
                    |b, _| {
                        b.iter(|| {
                            let _ = black_box(detector.process_sample(black_box(&samples)));
                        });
                    },
                );
            }
            128 => {
                let mut detector: ThresholdDetector<128> = ThresholdDetector::new(3.0, 100);
                let samples = [1.0f32; 128];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{}_channels", num_channels)),
                    num_channels,
                    |b, _| {
                        b.iter(|| {
                            let _ = black_box(detector.process_sample(black_box(&samples)));
                        });
                    },
                );
            }
            512 => {
                let mut detector: ThresholdDetector<512> = ThresholdDetector::new(3.0, 100);
                let samples = [1.0f32; 512];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{}_channels", num_channels)),
                    num_channels,
                    |b, _| {
                        b.iter(|| {
                            let _ = black_box(detector.process_sample(black_box(&samples)));
                        });
                    },
                );
            }
            1024 => {
                let mut detector: ThresholdDetector<1024> = ThresholdDetector::new(3.0, 100);
                let samples = [1.0f32; 1024];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("{}_channels", num_channels)),
                    num_channels,
                    |b, _| {
                        b.iter(|| {
                            let _ = black_box(detector.process_sample(black_box(&samples)));
                        });
                    },
                );
            }
            _ => {}
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_push_pop_throughput,
    bench_single_push,
    bench_single_pop,
    bench_streaming_bci_data,
    bench_try_push,
    bench_iir_filter,
    bench_fir_filter,
    bench_threshold_detector
);
criterion_main!(benches);
