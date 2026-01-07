use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zerostone::{
    AcCoupler, AdaptiveCsp, AdaptiveThresholdDetector, BandPower, BiquadCoeffs, CircularBuffer,
    Complex, Decimator, EnvelopeFollower, Fft, FirFilter, IirFilter, OnlineCov, Rectification,
    ThresholdDetector, UpdateConfig,
};

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

// AcCoupler benchmarks
fn bench_ac_coupler(c: &mut Criterion) {
    let mut group = c.benchmark_group("ac_coupler");

    // Single-channel AC coupler
    let mut coupler1: AcCoupler<1> = AcCoupler::new(250.0, 0.1);
    group.bench_function("single_channel", |b| {
        b.iter(|| {
            let _ = black_box(coupler1.process(black_box(&[1.0])));
        });
    });

    // Multi-channel AC coupler (32 channels)
    let mut coupler32: AcCoupler<32> = AcCoupler::new(250.0, 0.1);
    let samples32 = [1.0f32; 32];
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels", |b| {
        b.iter(|| {
            let _ = black_box(coupler32.process(black_box(&samples32)));
        });
    });

    // Block processing (256 samples, 4 channels)
    let mut coupler4: AcCoupler<4> = AcCoupler::new(250.0, 0.1);
    group.bench_function("block_256_samples_4ch", |b| {
        let mut block: Vec<[f32; 4]> = (0..256).map(|_| [1.0; 4]).collect();
        b.iter(|| {
            coupler4.process_block(black_box(&mut block));
        });
    });

    group.finish();
}

// Decimator benchmarks
fn bench_decimator(c: &mut Criterion) {
    let mut group = c.benchmark_group("decimator");

    // Single sample processing (32 channels, factor 4)
    let mut dec32: Decimator<32> = Decimator::new(4);
    let samples32 = [1.0f32; 32];
    group.bench_function("32_channels_factor_4", |b| {
        b.iter(|| {
            let _ = black_box(dec32.process(black_box(&samples32)));
        });
    });

    // Block processing (256 samples, 4 channels, factor 4)
    let mut dec4: Decimator<4> = Decimator::new(4);
    group.bench_function("block_256_samples_4ch_factor_4", |b| {
        let input: Vec<[f32; 4]> = (0..256).map(|i| [i as f32; 4]).collect();
        let mut output = vec![[0.0f32; 4]; 64];
        b.iter(|| {
            dec4.reset();
            let _ = black_box(dec4.process_block(black_box(&input), black_box(&mut output)));
        });
    });

    group.finish();
}

// EnvelopeFollower benchmarks
fn bench_envelope_follower(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope_follower");

    // Single sample processing (32 channels)
    let mut env32: EnvelopeFollower<32> =
        EnvelopeFollower::new(250.0, 0.010, 0.100, Rectification::Absolute);
    let samples32 = [0.5f32; 32];
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_absolute", |b| {
        b.iter(|| {
            let _ = black_box(env32.process(black_box(&samples32)));
        });
    });

    // Squared rectification
    let mut env32_sq: EnvelopeFollower<32> =
        EnvelopeFollower::new(250.0, 0.010, 0.100, Rectification::Squared);
    group.bench_function("32_channels_squared", |b| {
        b.iter(|| {
            let _ = black_box(env32_sq.process(black_box(&samples32)));
        });
    });

    // Block processing
    let mut env4: EnvelopeFollower<4> =
        EnvelopeFollower::new(250.0, 0.010, 0.100, Rectification::Absolute);
    group.bench_function("block_256_samples_4ch", |b| {
        let mut block: Vec<[f32; 4]> = (0..256).map(|i| [(i as f32 * 0.01).sin(); 4]).collect();
        b.iter(|| {
            env4.process_block(black_box(&mut block));
        });
    });

    group.finish();
}

// FFT benchmarks - target: <10 μs for 256-point
fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");

    // 128-point FFT
    let fft128 = Fft::<128>::new();
    group.bench_function("128_point_forward", |b| {
        let mut data = [Complex::from_real(1.0); 128];
        b.iter(|| {
            fft128.forward(black_box(&mut data));
        });
    });

    // 256-point FFT (target: <10 μs)
    let fft256 = Fft::<256>::new();
    group.bench_function("256_point_forward", |b| {
        let mut data = [Complex::from_real(1.0); 256];
        b.iter(|| {
            fft256.forward(black_box(&mut data));
        });
    });

    // 512-point FFT
    let fft512 = Fft::<512>::new();
    group.bench_function("512_point_forward", |b| {
        let mut data = [Complex::from_real(1.0); 512];
        b.iter(|| {
            fft512.forward(black_box(&mut data));
        });
    });

    // 1024-point FFT
    let fft1024 = Fft::<1024>::new();
    group.bench_function("1024_point_forward", |b| {
        let mut data = [Complex::from_real(1.0); 1024];
        b.iter(|| {
            fft1024.forward(black_box(&mut data));
        });
    });

    // Power spectrum computation
    let fft256_ps = Fft::<256>::new();
    group.bench_function("256_point_power_spectrum", |b| {
        let signal = [1.0f32; 256];
        let mut output = [0.0f32; 129];
        b.iter(|| {
            fft256_ps.power_spectrum(black_box(&signal), black_box(&mut output));
        });
    });

    // Band power (alpha band)
    let fft256_bp = Fft::<256>::new();
    let mut bp = BandPower::new(250.0);
    group.bench_function("256_point_alpha_band_power", |b| {
        let signal = [1.0f32; 256];
        b.iter(|| {
            let _ = black_box(bp.compute(&fft256_bp, black_box(&signal), 8.0, 12.0));
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

// AdaptiveThresholdDetector benchmarks
fn bench_adaptive_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_detector");

    // Pre-calibrated 32-channel detector
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_calibrated", |b| {
        let mut detector: AdaptiveThresholdDetector<32> =
            AdaptiveThresholdDetector::new(4.0, 100, 500);

        // Pre-calibrate
        for i in 0..500 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val; 32]);
        }
        detector.freeze();

        let samples = [0.05f32; 32];
        b.iter(|| {
            let _ = black_box(detector.process_sample(black_box(&samples)));
        });
    });

    // Adapting mode (updating stats each sample)
    group.bench_function("32_channels_adapting", |b| {
        let mut detector: AdaptiveThresholdDetector<32> =
            AdaptiveThresholdDetector::new(4.0, 100, 500);

        // Pre-calibrate
        for i in 0..500 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val; 32]);
        }
        // Don't freeze - keep adapting

        let samples = [0.05f32; 32];
        b.iter(|| {
            let _ = black_box(detector.process_sample(black_box(&samples)));
        });
    });

    // 128-channel frozen
    group.throughput(Throughput::Elements(128));
    group.bench_function("128_channels_calibrated", |b| {
        let mut detector: AdaptiveThresholdDetector<128> =
            AdaptiveThresholdDetector::new(4.0, 100, 500);

        for i in 0..500 {
            let val = if i % 2 == 0 { 0.1 } else { -0.1 };
            detector.process_sample(&[val; 128]);
        }
        detector.freeze();

        let samples = [0.05f32; 128];
        b.iter(|| {
            let _ = black_box(detector.process_sample(black_box(&samples)));
        });
    });

    group.finish();
}

// OnlineCov benchmarks - essential for CSP and Riemannian geometry BCI methods
fn bench_online_cov(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_cov");

    // 4-channel update (typical for motor imagery)
    let mut cov4: OnlineCov<4, 16> = OnlineCov::new();
    let samples4 = [1.0, 2.0, 3.0, 4.0];
    group.bench_function("update_4_channels", |b| {
        b.iter(|| {
            cov4.update(black_box(&samples4));
        });
    });

    // 8-channel update
    let mut cov8: OnlineCov<8, 64> = OnlineCov::new();
    let samples8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    group.bench_function("update_8_channels", |b| {
        b.iter(|| {
            cov8.update(black_box(&samples8));
        });
    });

    // 16-channel update
    let mut cov16: OnlineCov<16, 256> = OnlineCov::new();
    let samples16 = [1.0; 16];
    group.bench_function("update_16_channels", |b| {
        b.iter(|| {
            cov16.update(black_box(&samples16));
        });
    });

    // 32-channel update
    let mut cov32: OnlineCov<32, 1024> = OnlineCov::new();
    let samples32 = [1.0; 32];
    group.bench_function("update_32_channels", |b| {
        b.iter(|| {
            cov32.update(black_box(&samples32));
        });
    });

    // Covariance matrix retrieval (4 channels)
    let mut cov4_retrieval: OnlineCov<4, 16> = OnlineCov::new();
    for _ in 0..100 {
        cov4_retrieval.update(&[1.0, 2.0, 3.0, 4.0]);
    }
    group.bench_function("covariance_4_channels", |b| {
        b.iter(|| {
            let _ = black_box(cov4_retrieval.covariance());
        });
    });

    // Correlation matrix retrieval (4 channels)
    group.bench_function("correlation_4_channels", |b| {
        b.iter(|| {
            let _ = black_box(cov4_retrieval.correlation());
        });
    });

    // Element access
    group.bench_function("get_element_4_channels", |b| {
        b.iter(|| {
            let _ = black_box(cov4_retrieval.get(0, 1));
        });
    });

    // Covariance matrix retrieval (32 channels)
    let mut cov32_retrieval: OnlineCov<32, 1024> = OnlineCov::new();
    for i in 0..100 {
        let sample: [f64; 32] = core::array::from_fn(|j| (i + j) as f64);
        cov32_retrieval.update(&sample);
    }
    group.bench_function("covariance_32_channels", |b| {
        b.iter(|| {
            let _ = black_box(cov32_retrieval.covariance());
        });
    });

    group.finish();
}

// CSP benchmarks - Common Spatial Patterns for motor imagery BCI
fn bench_csp(c: &mut Criterion) {
    let mut group = c.benchmark_group("csp");

    // Filter application benchmark (hot path) - 8 channels, 4 filters
    {
        let mut csp: AdaptiveCsp<8, 64, 4, 32> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            ..Default::default()
        });

        // Generate training data
        for _ in 0..100 {
            let mut trial = [[0.0; 8]; 20];
            for sample in &mut trial {
                sample[0] = 1.0;
                sample[1] = 0.5;
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 8]; 20];
            for sample in &mut trial2 {
                sample[2] = 1.0;
                sample[3] = 0.5;
            }
            csp.update_class2(&trial2);
        }

        csp.recompute_filters().unwrap();

        let sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        group.bench_function("apply_8ch_4filters", |b| {
            b.iter(|| {
                let _ = black_box(csp.apply(black_box(&sample)).unwrap());
            });
        });
    }

    // Filter application - 16 channels, 6 filters
    {
        let mut csp: AdaptiveCsp<16, 256, 6, 96> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            ..Default::default()
        });

        for _ in 0..100 {
            let mut trial = [[0.0; 16]; 20];
            for sample in &mut trial {
                for i in 0..4 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 16]; 20];
            for sample in &mut trial2 {
                for i in 4..8 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class2(&trial2);
        }

        csp.recompute_filters().unwrap();

        let sample = [1.0; 16];
        group.bench_function("apply_16ch_6filters", |b| {
            b.iter(|| {
                let _ = black_box(csp.apply(black_box(&sample)).unwrap());
            });
        });
    }

    // Filter application - 32 channels, 6 filters (target: <1 μs)
    {
        let mut csp: AdaptiveCsp<32, 1024, 6, 192> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        for _ in 0..100 {
            let mut trial = [[0.0; 32]; 20];
            for sample in &mut trial {
                for i in 0..8 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 32]; 20];
            for sample in &mut trial2 {
                for i in 8..16 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class2(&trial2);
        }

        csp.recompute_filters().unwrap();

        let sample = [1.0; 32];
        group.bench_function("apply_32ch_6filters", |b| {
            b.iter(|| {
                let _ = black_box(csp.apply(black_box(&sample)).unwrap());
            });
        });
    }

    // Filter recomputation - 8 channels (eigendecomposition)
    {
        let mut csp: AdaptiveCsp<8, 64, 4, 32> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            ..Default::default()
        });

        for _ in 0..100 {
            let mut trial = [[0.0; 8]; 20];
            for sample in &mut trial {
                sample[0] = 1.0;
                sample[1] = 0.5;
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 8]; 20];
            for sample in &mut trial2 {
                sample[2] = 1.0;
                sample[3] = 0.5;
            }
            csp.update_class2(&trial2);
        }

        group.bench_function("recompute_8ch", |b| {
            b.iter(|| {
                let _ = black_box(csp.recompute_filters());
            });
        });
    }

    // Filter recomputation - 16 channels
    {
        let mut csp: AdaptiveCsp<16, 256, 6, 96> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            ..Default::default()
        });

        for _ in 0..100 {
            let mut trial = [[0.0; 16]; 20];
            for sample in &mut trial {
                for i in 0..4 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 16]; 20];
            for sample in &mut trial2 {
                for i in 4..8 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class2(&trial2);
        }

        group.bench_function("recompute_16ch", |b| {
            b.iter(|| {
                let _ = black_box(csp.recompute_filters());
            });
        });
    }

    // Filter recomputation - 32 channels (target: <100 ms)
    {
        let mut csp: AdaptiveCsp<32, 1024, 6, 192> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            regularization: 1e-4,
            ..Default::default()
        });

        for _ in 0..100 {
            let mut trial = [[0.0; 32]; 20];
            for sample in &mut trial {
                for i in 0..8 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 32]; 20];
            for sample in &mut trial2 {
                for i in 8..16 {
                    sample[i] = 1.0;
                }
            }
            csp.update_class2(&trial2);
        }

        group.bench_function("recompute_32ch", |b| {
            b.iter(|| {
                let _ = black_box(csp.recompute_filters());
            });
        });
    }

    // Online adaptation - updating with new trials
    {
        let mut csp: AdaptiveCsp<8, 64, 4, 32> = AdaptiveCsp::new(UpdateConfig {
            min_samples: 50,
            update_interval: 0,
            ..Default::default()
        });

        // Pre-fill with data
        for _ in 0..100 {
            let trial1 = [[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 20];
            csp.update_class1(&trial1);

            let trial2 = [[0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]; 20];
            csp.update_class2(&trial2);
        }

        let new_trial = [[1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]; 20];
        group.bench_function("update_class1_8ch", |b| {
            b.iter(|| {
                csp.update_class1(black_box(&new_trial));
            });
        });
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
    bench_ac_coupler,
    bench_decimator,
    bench_envelope_follower,
    bench_fft,
    bench_threshold_detector,
    bench_adaptive_detector,
    bench_online_cov,
    bench_csp
);
criterion_main!(benches);
