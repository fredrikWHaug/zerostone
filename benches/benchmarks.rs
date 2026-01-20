use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zerostone::{
    apply_window,
    hilbert::{hilbert_batch, HilbertTransform},
    xcorr::{autocorr, autocorr_batch, xcorr, xcorr_batch, Normalization},
    AcCoupler, AdaptiveCsp, AdaptiveThresholdDetector, ArtifactDetector, BandPower, BiquadCoeffs,
    CircularBuffer, Complex, Cwt, Decimator, EnvelopeFollower, Fft, FirFilter, IirFilter,
    Interpolator, LmsFilter, MultiChannelCwt, NlmsFilter, OasisDeconvolution, OnlineCov,
    Rectification, Stft, StreamingPercentile, ThresholdDetector, UpdateConfig, WindowType,
    WindowedRms, ZscoreArtifact,
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
                for elem in sample.iter_mut().take(4) {
                    *elem = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 16]; 20];
            for sample in &mut trial2 {
                for elem in sample.iter_mut().take(8).skip(4) {
                    *elem = 1.0;
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
                for elem in sample.iter_mut().take(8) {
                    *elem = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 32]; 20];
            for sample in &mut trial2 {
                for elem in sample.iter_mut().take(16).skip(8) {
                    *elem = 1.0;
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
                for elem in sample.iter_mut().take(4) {
                    *elem = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 16]; 20];
            for sample in &mut trial2 {
                for elem in sample.iter_mut().take(8).skip(4) {
                    *elem = 1.0;
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
                for elem in sample.iter_mut().take(8) {
                    *elem = 1.0;
                }
            }
            csp.update_class1(&trial);

            let mut trial2 = [[0.0; 32]; 20];
            for sample in &mut trial2 {
                for elem in sample.iter_mut().take(16).skip(8) {
                    *elem = 1.0;
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

// Window function benchmarks
fn bench_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("window");

    // Benchmark different window sizes
    for size in [64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        match *size {
            64 => {
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("hann_{}", size)),
                    size,
                    |b, _| {
                        let mut signal = [1.0f32; 64];
                        b.iter(|| {
                            apply_window(black_box(&mut signal), WindowType::Hann);
                        });
                    },
                );
            }
            256 => {
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("hann_{}", size)),
                    size,
                    |b, _| {
                        let mut signal = [1.0f32; 256];
                        b.iter(|| {
                            apply_window(black_box(&mut signal), WindowType::Hann);
                        });
                    },
                );
            }
            1024 => {
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("hann_{}", size)),
                    size,
                    |b, _| {
                        let mut signal = [1.0f32; 1024];
                        b.iter(|| {
                            apply_window(black_box(&mut signal), WindowType::Hann);
                        });
                    },
                );
            }
            _ => {}
        }
    }

    // Benchmark different window types at 256 samples
    let mut signal256 = [1.0f32; 256];

    group.bench_function("rectangular_256", |b| {
        b.iter(|| {
            apply_window(black_box(&mut signal256), WindowType::Rectangular);
        });
    });

    group.bench_function("hamming_256", |b| {
        b.iter(|| {
            apply_window(black_box(&mut signal256), WindowType::Hamming);
        });
    });

    group.bench_function("blackman_256", |b| {
        b.iter(|| {
            apply_window(black_box(&mut signal256), WindowType::Blackman);
        });
    });

    group.bench_function("blackman_harris_256", |b| {
        b.iter(|| {
            apply_window(black_box(&mut signal256), WindowType::BlackmanHarris);
        });
    });

    group.finish();
}

// StreamingPercentile benchmarks - P² algorithm for streaming quantile estimation
fn bench_streaming_percentile(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_percentile");

    // Single-channel update (hot path)
    let mut est1: StreamingPercentile<1> = StreamingPercentile::new(0.5);
    // Pre-initialize
    for i in 0..5 {
        est1.update(&[i as f64]);
    }
    group.bench_function("update_1_channel", |b| {
        b.iter(|| {
            est1.update(black_box(&[42.0]));
        });
    });

    // 8-channel update (typical for motor imagery BCI)
    let mut est8: StreamingPercentile<8> = StreamingPercentile::new(0.08);
    for i in 0..5 {
        est8.update(&[i as f64; 8]);
    }
    let samples8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    group.throughput(Throughput::Elements(8));
    group.bench_function("update_8_channels", |b| {
        b.iter(|| {
            est8.update(black_box(&samples8));
        });
    });

    // 32-channel update
    let mut est32: StreamingPercentile<32> = StreamingPercentile::new(0.5);
    for i in 0..5 {
        est32.update(&[i as f64; 32]);
    }
    let samples32 = [1.0; 32];
    group.throughput(Throughput::Elements(32));
    group.bench_function("update_32_channels", |b| {
        b.iter(|| {
            est32.update(black_box(&samples32));
        });
    });

    // Percentile retrieval
    let mut est_retrieve: StreamingPercentile<8> = StreamingPercentile::new(0.5);
    for i in 0..1000 {
        est_retrieve.update(&[i as f64; 8]);
    }
    group.bench_function("percentile_8_channels", |b| {
        b.iter(|| {
            let _ = black_box(est_retrieve.percentile());
        });
    });

    // Different percentiles (8th percentile for baseline estimation)
    let mut est_p8: StreamingPercentile<8> = StreamingPercentile::new(0.08);
    for i in 0..5 {
        est_p8.update(&[i as f64; 8]);
    }
    group.bench_function("update_8ch_p08", |b| {
        b.iter(|| {
            est_p8.update(black_box(&samples8));
        });
    });

    // 99th percentile
    let mut est_p99: StreamingPercentile<8> = StreamingPercentile::new(0.99);
    for i in 0..5 {
        est_p99.update(&[i as f64; 8]);
    }
    group.bench_function("update_8ch_p99", |b| {
        b.iter(|| {
            est_p99.update(black_box(&samples8));
        });
    });

    group.finish();
}

// OASIS deconvolution benchmarks - calcium imaging spike inference
fn bench_oasis_deconvolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("oasis_deconvolution");

    // Single channel update (hot path)
    group.bench_function("single_channel_update", |b| {
        let mut deconv: OasisDeconvolution<1, 256> = OasisDeconvolution::new(0.95, 0.1);
        let fluorescence = [5.0];
        let baseline = [1.0];

        b.iter(|| {
            black_box(deconv.update(black_box(&fluorescence), black_box(&baseline)));
        });
    });

    // Multi-channel (8 channels) - typical calcium imaging
    group.throughput(Throughput::Elements(8));
    group.bench_function("8_channels_update", |b| {
        let mut deconv: OasisDeconvolution<8, 256> = OasisDeconvolution::new(0.95, 0.1);
        let fluorescence = [5.0; 8];
        let baseline = [1.0; 8];

        b.iter(|| {
            black_box(deconv.update(black_box(&fluorescence), black_box(&baseline)));
        });
    });

    // Multi-channel (32 channels) - large FOV imaging
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_update", |b| {
        let mut deconv: OasisDeconvolution<32, 256> = OasisDeconvolution::new(0.95, 0.1);
        let fluorescence = [5.0; 32];
        let baseline = [1.0; 32];

        b.iter(|| {
            black_box(deconv.update(black_box(&fluorescence), black_box(&baseline)));
        });
    });

    // From tau constructor (GCaMP6f typical parameters)
    group.bench_function("from_tau_constructor", |b| {
        b.iter(|| {
            let _deconv: OasisDeconvolution<8, 256> =
                black_box(OasisDeconvolution::from_tau(30.0, 0.1, 0.1));
        });
    });

    group.finish();
}

// Full calcium imaging pipeline: baseline estimation + deconvolution
fn bench_oasis_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("oasis_full_pipeline");

    // Complete pipeline: baseline estimation + deconvolution (8 channels)
    group.throughput(Throughput::Elements(8));
    group.bench_function("baseline_plus_deconv_8ch", |b| {
        let mut baseline: StreamingPercentile<8> = StreamingPercentile::new(0.08);
        let mut deconv: OasisDeconvolution<8, 256> = OasisDeconvolution::new(0.95, 0.1);

        b.iter(|| {
            let fluorescence = black_box([5.0; 8]);
            let fluor_f64 = fluorescence.map(|x| x as f64);
            baseline.update(&fluor_f64);

            if let Some(b64) = baseline.percentile() {
                let b = b64.map(|x| x as f32);
                black_box(deconv.update(&fluorescence, &b));
            }
        });
    });

    // Realistic imaging scenario with varying signal
    group.bench_function("realistic_imaging_8ch", |b| {
        let mut baseline: StreamingPercentile<8> = StreamingPercentile::new(0.08);
        let mut deconv: OasisDeconvolution<8, 256> = OasisDeconvolution::new(0.95, 0.1);

        let mut frame_counter = 0;

        b.iter(|| {
            frame_counter += 1;
            // Simulate varying fluorescence with spikes
            let mut fluorescence = [2.0; 8];
            if frame_counter % 20 == 0 {
                // Simulate spike
                fluorescence[frame_counter % 8] = 10.0;
            }

            let fluor_f64 = fluorescence.map(|x| x as f64);
            baseline.update(&fluor_f64);

            if let Some(b64) = baseline.percentile() {
                let b = b64.map(|x| x as f32);
                let _result = black_box(deconv.update(&fluorescence, &b));
            }
        });
    });

    group.finish();
}

// Continuous Wavelet Transform benchmarks - time-frequency analysis
fn bench_cwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cwt");

    // 256-point CWT with 8 scales (typical BCI use case)
    {
        let cwt = Cwt::<256, 8>::new(250.0, 1.0, 50.0);
        let signal = [0.5f32; 256];

        group.bench_function("256_point_8_scales_transform", |b| {
            let mut output = [[Complex::new(0.0, 0.0); 256]; 8];
            b.iter(|| {
                cwt.transform(black_box(&signal), black_box(&mut output));
            });
        });

        group.bench_function("256_point_8_scales_power", |b| {
            let mut output = [[0.0f32; 256]; 8];
            b.iter(|| {
                cwt.power(black_box(&signal), black_box(&mut output));
            });
        });
    }

    // 512-point CWT with 16 scales (higher resolution)
    {
        let cwt = Cwt::<512, 16>::new(250.0, 1.0, 100.0);
        let signal = [0.5f32; 512];

        group.bench_function("512_point_16_scales_power", |b| {
            let mut output = [[0.0f32; 512]; 16];
            b.iter(|| {
                cwt.power(black_box(&signal), black_box(&mut output));
            });
        });
    }

    // Single scale transform (streaming use case)
    {
        let cwt = Cwt::<256, 8>::new(250.0, 1.0, 50.0);
        let signal = [0.5f32; 256];

        group.bench_function("256_point_single_scale", |b| {
            let mut output = [Complex::new(0.0, 0.0); 256];
            b.iter(|| {
                cwt.transform_scale(black_box(&signal), 4, black_box(&mut output));
            });
        });
    }

    // Multi-channel CWT (8 channels)
    {
        let cwt = MultiChannelCwt::<256, 8, 8>::new(250.0, 1.0, 50.0);
        let signals = [[0.5f32; 256]; 8];

        group.throughput(Throughput::Elements(8));
        group.bench_function("256_point_8_scales_8_channels", |b| {
            let mut output = [[[0.0f32; 256]; 8]; 8];
            b.iter(|| {
                cwt.power(black_box(&signals), black_box(&mut output));
            });
        });
    }

    group.finish();
}

// Short-Time Fourier Transform benchmarks - spectrogram analysis
fn bench_stft(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft");

    // 256-point window, 50% overlap, typical EEG analysis
    {
        let stft = Stft::<256>::new(128, WindowType::Hann);
        let signal = [0.5f32; 1024];

        group.bench_function("256_window_128_hop_power", |b| {
            let mut output = [[0.0f32; 256]; 7];
            b.iter(|| {
                stft.power(black_box(&signal), black_box(&mut output));
            });
        });

        group.bench_function("256_window_128_hop_transform", |b| {
            let mut output = [[Complex::new(0.0, 0.0); 256]; 7];
            b.iter(|| {
                stft.transform(black_box(&signal), black_box(&mut output));
            });
        });
    }

    // 512-point window for higher frequency resolution
    {
        let stft = Stft::<512>::new(256, WindowType::Hann);
        let signal = [0.5f32; 2048];

        group.bench_function("512_window_256_hop_power", |b| {
            let mut output = [[0.0f32; 512]; 7];
            b.iter(|| {
                stft.power(black_box(&signal), black_box(&mut output));
            });
        });
    }

    // Single frame processing (streaming use case)
    {
        let stft = Stft::<256>::new(128, WindowType::Hann);
        let signal = [0.5f32; 256];

        group.bench_function("256_single_frame", |b| {
            let mut output = [Complex::new(0.0, 0.0); 256];
            b.iter(|| {
                stft.transform_frame(black_box(&signal), 0, black_box(&mut output));
            });
        });
    }

    // 75% overlap for smoother spectrogram
    {
        let stft = Stft::<256>::new(64, WindowType::Hann);
        let signal = [0.5f32; 1024];

        group.bench_function("256_window_64_hop_power", |b| {
            let mut output = [[0.0f32; 256]; 13];
            b.iter(|| {
                stft.power(black_box(&signal), black_box(&mut output));
            });
        });
    }

    group.finish();
}

// Artifact detection benchmarks
fn bench_artifact_detector(c: &mut Criterion) {
    let mut group = c.benchmark_group("artifact_detector");

    // 8-channel artifact detection
    group.throughput(Throughput::Elements(8));
    group.bench_function("8_channels_detect", |b| {
        let mut detector: ArtifactDetector<8> = ArtifactDetector::new(100.0, 50.0);
        let samples = [50.0f32; 8];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    // 32-channel artifact detection
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_detect", |b| {
        let mut detector: ArtifactDetector<32> = ArtifactDetector::new(100.0, 50.0);
        let samples = [50.0f32; 32];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    // 128-channel artifact detection
    group.throughput(Throughput::Elements(128));
    group.bench_function("128_channels_detect", |b| {
        let mut detector: ArtifactDetector<128> = ArtifactDetector::new(100.0, 50.0);
        let samples = [50.0f32; 128];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    // Detailed detection (returns ArtifactType per channel)
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_detect_detailed", |b| {
        let mut detector: ArtifactDetector<32> = ArtifactDetector::new(100.0, 50.0);
        let samples = [50.0f32; 32];
        b.iter(|| {
            let _ = black_box(detector.detect_detailed(black_box(&samples)));
        });
    });

    group.finish();
}

// Z-score artifact detection benchmarks
fn bench_zscore_artifact(c: &mut Criterion) {
    let mut group = c.benchmark_group("zscore_artifact");

    // 8-channel z-score detection (frozen)
    group.throughput(Throughput::Elements(8));
    group.bench_function("8_channels_frozen", |b| {
        let mut detector: ZscoreArtifact<8> = ZscoreArtifact::new(3.0, 100);
        // Calibrate
        for i in 0..100 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val; 8]);
        }
        detector.freeze();

        let samples = [0.5f32; 8];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    // 32-channel z-score detection (frozen)
    group.throughput(Throughput::Elements(32));
    group.bench_function("32_channels_frozen", |b| {
        let mut detector: ZscoreArtifact<32> = ZscoreArtifact::new(3.0, 100);
        for i in 0..100 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val; 32]);
        }
        detector.freeze();

        let samples = [0.5f32; 32];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    // 32-channel z-score detection (adapting - includes stats update)
    group.bench_function("32_channels_adapting", |b| {
        let mut detector: ZscoreArtifact<32> = ZscoreArtifact::new(3.0, 100);
        for i in 0..100 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val; 32]);
        }
        // Don't freeze - keep adapting

        let samples = [0.5f32; 32];
        b.iter(|| {
            let _ = black_box(detector.update_and_detect(black_box(&samples)));
        });
    });

    // 128-channel z-score detection (frozen)
    group.throughput(Throughput::Elements(128));
    group.bench_function("128_channels_frozen", |b| {
        let mut detector: ZscoreArtifact<128> = ZscoreArtifact::new(3.0, 100);
        for i in 0..100 {
            let val = if i % 2 == 0 { 1.0 } else { -1.0 };
            detector.update(&[val; 128]);
        }
        detector.freeze();

        let samples = [0.5f32; 128];
        b.iter(|| {
            let _ = black_box(detector.detect(black_box(&samples)));
        });
    });

    group.finish();
}

// Interpolator benchmarks
fn bench_interpolator(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolator");

    // 8-channel zero-order interpolation (4x)
    group.throughput(Throughput::Elements(8 * 4));
    group.bench_function("8ch_zero_order_4x", |b| {
        let mut interp: Interpolator<8> = Interpolator::zero_order(4);
        let input = [1.0f32; 8];
        let mut output = [[0.0f32; 8]; 4];
        b.iter(|| {
            let _ = black_box(interp.process(black_box(&input), black_box(&mut output)));
        });
    });

    // 8-channel linear interpolation (4x)
    group.bench_function("8ch_linear_4x", |b| {
        let mut interp: Interpolator<8> = Interpolator::linear(4);
        let input = [1.0f32; 8];
        let mut output = [[0.0f32; 8]; 4];
        // Initialize
        interp.process(&input, &mut output);
        b.iter(|| {
            let _ = black_box(interp.process(black_box(&input), black_box(&mut output)));
        });
    });

    // 32-channel zero-order interpolation (4x)
    group.throughput(Throughput::Elements(32 * 4));
    group.bench_function("32ch_zero_order_4x", |b| {
        let mut interp: Interpolator<32> = Interpolator::zero_order(4);
        let input = [1.0f32; 32];
        let mut output = [[0.0f32; 32]; 4];
        b.iter(|| {
            let _ = black_box(interp.process(black_box(&input), black_box(&mut output)));
        });
    });

    // 32-channel linear interpolation (4x)
    group.bench_function("32ch_linear_4x", |b| {
        let mut interp: Interpolator<32> = Interpolator::linear(4);
        let input = [1.0f32; 32];
        let mut output = [[0.0f32; 32]; 4];
        interp.process(&input, &mut output);
        b.iter(|| {
            let _ = black_box(interp.process(black_box(&input), black_box(&mut output)));
        });
    });

    // Block processing: 32-channel, 4x, 100 samples -> 400 output
    group.throughput(Throughput::Elements(32 * 400));
    group.bench_function("32ch_block_100_linear_4x", |b| {
        let mut interp: Interpolator<32> = Interpolator::linear(4);
        let input = [[1.0f32; 32]; 100];
        let mut output = [[0.0f32; 32]; 400];
        b.iter(|| {
            interp.reset();
            let _ = black_box(interp.process_block(black_box(&input), black_box(&mut output)));
        });
    });

    group.finish();
}

// Cross-correlation benchmarks
fn bench_xcorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("xcorr");

    // 64×64 cross-correlation (small, typical for feature extraction)
    {
        let x = [1.0f32; 64];
        let y = [0.5f32; 64];
        let mut output = [0.0f32; 127];

        group.bench_function("xcorr_64x64", |b| {
            b.iter(|| {
                xcorr(
                    black_box(&x),
                    black_box(&y),
                    black_box(&mut output),
                    Normalization::None,
                );
            });
        });
    }

    // 256×256 cross-correlation (typical BCI segment)
    {
        let x = [1.0f32; 256];
        let y = [0.5f32; 256];
        let mut output = [0.0f32; 511];

        group.bench_function("xcorr_256x256", |b| {
            b.iter(|| {
                xcorr(
                    black_box(&x),
                    black_box(&y),
                    black_box(&mut output),
                    Normalization::None,
                );
            });
        });

        // With coefficient normalization
        group.bench_function("xcorr_256x256_coeff", |b| {
            b.iter(|| {
                xcorr(
                    black_box(&x),
                    black_box(&y),
                    black_box(&mut output),
                    Normalization::Coeff,
                );
            });
        });
    }

    // Auto-correlation (optimized for symmetry)
    {
        let x = [1.0f32; 256];
        let mut output = [0.0f32; 511];

        group.bench_function("autocorr_256", |b| {
            b.iter(|| {
                autocorr(black_box(&x), black_box(&mut output), Normalization::None);
            });
        });

        group.bench_function("autocorr_256_coeff", |b| {
            b.iter(|| {
                autocorr(black_box(&x), black_box(&mut output), Normalization::Coeff);
            });
        });
    }

    // Batch cross-correlation (8 channels)
    {
        let x: [[f32; 64]; 8] = [[1.0f32; 64]; 8];
        let y: [[f32; 64]; 8] = [[0.5f32; 64]; 8];
        let mut output = [[0.0f32; 127]; 8];

        group.throughput(Throughput::Elements(8));
        group.bench_function("xcorr_batch_8ch_64x64", |b| {
            b.iter(|| {
                xcorr_batch(
                    black_box(&x),
                    black_box(&y),
                    black_box(&mut output),
                    Normalization::None,
                );
            });
        });
    }

    // Batch auto-correlation (8 channels)
    {
        let x: [[f32; 64]; 8] = [[1.0f32; 64]; 8];
        let mut output = [[0.0f32; 127]; 8];

        group.throughput(Throughput::Elements(8));
        group.bench_function("autocorr_batch_8ch_64", |b| {
            b.iter(|| {
                autocorr_batch(black_box(&x), black_box(&mut output), Normalization::None);
            });
        });
    }

    // Different signal lengths (asymmetric correlation)
    {
        let x = [1.0f32; 256];
        let y = [0.5f32; 64];
        let mut output = [0.0f32; 319]; // 256 + 64 - 1

        group.bench_function("xcorr_256x64", |b| {
            b.iter(|| {
                xcorr(
                    black_box(&x),
                    black_box(&y),
                    black_box(&mut output),
                    Normalization::None,
                );
            });
        });
    }

    group.finish();
}

fn bench_hilbert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hilbert_transform");

    // Generate test signal (4 Hz sine wave at 256 Hz sample rate)
    let mut signal_128 = [0.0f32; 128];
    let mut signal_256 = [0.0f32; 256];
    let mut signal_512 = [0.0f32; 512];

    for (i, sample) in signal_128.iter_mut().enumerate() {
        let t = i as f32 / 256.0;
        *sample = (2.0 * core::f32::consts::PI * 4.0 * t).sin();
    }

    for (i, sample) in signal_256.iter_mut().enumerate() {
        let t = i as f32 / 256.0;
        *sample = (2.0 * core::f32::consts::PI * 4.0 * t).sin();
    }

    for (i, sample) in signal_512.iter_mut().enumerate() {
        let t = i as f32 / 256.0;
        *sample = (2.0 * core::f32::consts::PI * 4.0 * t).sin();
    }

    // Benchmark analytic_signal for different sizes
    {
        let hilbert = HilbertTransform::<128>::new();
        let mut output = [Complex::new(0.0, 0.0); 128];

        group.bench_function("analytic_signal_128", |b| {
            b.iter(|| {
                hilbert.analytic_signal(black_box(&signal_128), black_box(&mut output));
            });
        });
    }

    {
        let hilbert = HilbertTransform::<256>::new();
        let mut output = [Complex::new(0.0, 0.0); 256];

        group.bench_function("analytic_signal_256", |b| {
            b.iter(|| {
                hilbert.analytic_signal(black_box(&signal_256), black_box(&mut output));
            });
        });
    }

    {
        let hilbert = HilbertTransform::<512>::new();
        let mut output = [Complex::new(0.0, 0.0); 512];

        group.bench_function("analytic_signal_512", |b| {
            b.iter(|| {
                hilbert.analytic_signal(black_box(&signal_512), black_box(&mut output));
            });
        });
    }

    // Benchmark instantaneous amplitude
    {
        let hilbert = HilbertTransform::<256>::new();
        let mut output = [0.0f32; 256];

        group.bench_function("instantaneous_amplitude_256", |b| {
            b.iter(|| {
                hilbert.instantaneous_amplitude(black_box(&signal_256), black_box(&mut output));
            });
        });
    }

    // Benchmark instantaneous phase
    {
        let hilbert = HilbertTransform::<256>::new();
        let mut output = [0.0f32; 256];

        group.bench_function("instantaneous_phase_256", |b| {
            b.iter(|| {
                hilbert.instantaneous_phase(black_box(&signal_256), black_box(&mut output));
            });
        });
    }

    // Benchmark instantaneous frequency
    {
        let hilbert = HilbertTransform::<256>::new();
        let mut output = [0.0f32; 255];

        group.bench_function("instantaneous_frequency_256", |b| {
            b.iter(|| {
                let _ = hilbert.instantaneous_frequency(
                    black_box(&signal_256),
                    black_box(&mut output),
                    256.0,
                );
            });
        });
    }

    // Benchmark batch processing (8 channels × 256 samples)
    {
        let signals = [[0.0f32; 256]; 8];
        let mut output = [[Complex::new(0.0, 0.0); 256]; 8];

        group.bench_function("batch_8ch_x_256", |b| {
            b.iter(|| {
                hilbert_batch(black_box(&signals), black_box(&mut output));
            });
        });
    }

    group.finish();
}

// LMS adaptive filter benchmarks - target: <500 ns/sample for N=32, <1 μs for N=64
fn bench_lms_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("lms_filter");

    // 32-tap LMS (typical for powerline/EOG cancellation)
    {
        let mut lms = LmsFilter::<32>::new(0.01);
        group.bench_function("lms_32_taps", |b| {
            b.iter(|| black_box(lms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // 64-tap LMS (typical for EMG artifact removal)
    {
        let mut lms = LmsFilter::<64>::new(0.01);
        group.bench_function("lms_64_taps", |b| {
            b.iter(|| black_box(lms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // 128-tap LMS (high-order adaptive filtering)
    {
        let mut lms = LmsFilter::<128>::new(0.01);
        group.bench_function("lms_128_taps", |b| {
            b.iter(|| black_box(lms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // Prediction mode (no adaptation)
    {
        let mut lms = LmsFilter::<32>::new(0.01);
        group.bench_function("lms_32_predict_only", |b| {
            b.iter(|| black_box(lms.predict(black_box(1.0))));
        });
    }

    // Block processing (256 samples, 32 taps)
    {
        use zerostone::AdaptiveOutput;
        let mut lms = LmsFilter::<32>::new(0.01);
        group.bench_function("lms_32_block_256", |b| {
            let input = [1.0f32; 256];
            let desired = [0.5f32; 256];
            let mut output = [AdaptiveOutput {
                output: 0.0,
                error: 0.0,
            }; 256];
            b.iter(|| {
                lms.process_block(
                    black_box(&input),
                    black_box(&desired),
                    black_box(&mut output),
                );
            });
        });
    }

    group.finish();
}

// NLMS adaptive filter benchmarks - normalized step size for stability
fn bench_nlms_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("nlms_filter");

    // 32-tap NLMS
    {
        let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
        group.bench_function("nlms_32_taps", |b| {
            b.iter(|| black_box(nlms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // 64-tap NLMS
    {
        let mut nlms = NlmsFilter::<64>::new(0.5, 0.01);
        group.bench_function("nlms_64_taps", |b| {
            b.iter(|| black_box(nlms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // 128-tap NLMS
    {
        let mut nlms = NlmsFilter::<128>::new(0.5, 0.01);
        group.bench_function("nlms_128_taps", |b| {
            b.iter(|| black_box(nlms.process_sample(black_box(1.0), black_box(0.5))));
        });
    }

    // Prediction mode (no adaptation)
    {
        let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
        group.bench_function("nlms_32_predict_only", |b| {
            b.iter(|| black_box(nlms.predict(black_box(1.0))));
        });
    }

    // Block processing (256 samples, 32 taps)
    {
        use zerostone::AdaptiveOutput;
        let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);
        group.bench_function("nlms_32_block_256", |b| {
            let input = [1.0f32; 256];
            let desired = [0.5f32; 256];
            let mut output = [AdaptiveOutput {
                output: 0.0,
                error: 0.0,
            }; 256];
            b.iter(|| {
                nlms.process_block(
                    black_box(&input),
                    black_box(&desired),
                    black_box(&mut output),
                );
            });
        });
    }

    // Comparison with varying amplitude (NLMS advantage)
    {
        let mut lms = LmsFilter::<32>::new(0.01);
        let mut nlms = NlmsFilter::<32>::new(0.5, 0.01);

        group.bench_function("lms_32_varying_amplitude", |b| {
            let mut counter = 0u32;
            b.iter(|| {
                counter = counter.wrapping_add(1);
                let amplitude = 1.0 + 0.5 * (counter as f32 * 0.1).sin();
                let input = amplitude;
                let desired = amplitude * 0.5;
                black_box(lms.process_sample(black_box(input), black_box(desired)))
            });
        });

        group.bench_function("nlms_32_varying_amplitude", |b| {
            let mut counter = 0u32;
            b.iter(|| {
                counter = counter.wrapping_add(1);
                let amplitude = 1.0 + 0.5 * (counter as f32 * 0.1).sin();
                let input = amplitude;
                let desired = amplitude * 0.5;
                black_box(nlms.process_sample(black_box(input), black_box(desired)))
            });
        });
    }

    group.finish();
}

// WindowedRms benchmarks - windowed RMS and power computation
fn bench_windowed_rms(c: &mut Criterion) {
    let mut group = c.benchmark_group("windowed_rms");

    // Single-sample processing (typical EEG window: 64 samples, 32 channels)
    {
        let mut rms: WindowedRms<32, 64> = WindowedRms::new();
        let sample = [1.0f32; 32];
        group.throughput(Throughput::Elements(32));
        group.bench_function("process_64win_32ch", |b| {
            b.iter(|| {
                rms.process(black_box(&sample));
            });
        });
    }

    // RMS retrieval cost
    {
        let mut rms: WindowedRms<32, 64> = WindowedRms::new();
        for _ in 0..64 {
            rms.process(&[1.0; 32]);
        }
        group.bench_function("rms_retrieval_64win_32ch", |b| {
            b.iter(|| black_box(rms.rms()));
        });
    }

    // Power retrieval (no sqrt)
    {
        let mut rms: WindowedRms<32, 64> = WindowedRms::new();
        for _ in 0..64 {
            rms.process(&[1.0; 32]);
        }
        group.bench_function("power_retrieval_64win_32ch", |b| {
            b.iter(|| black_box(rms.power()));
        });
    }

    // Different window sizes (32 channels)
    for window in [16, 64, 256].iter() {
        let window_size = *window;
        match window_size {
            16 => {
                let mut rms: WindowedRms<32, 16> = WindowedRms::new();
                let sample = [1.0f32; 32];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_{}win_32ch", window)),
                    window,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            64 => {
                let mut rms: WindowedRms<32, 64> = WindowedRms::new();
                let sample = [1.0f32; 32];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_{}win_32ch", window)),
                    window,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            256 => {
                let mut rms: WindowedRms<32, 256> = WindowedRms::new();
                let sample = [1.0f32; 32];
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_{}win_32ch", window)),
                    window,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            _ => {}
        }
    }

    // Different channel counts (64-sample window)
    for channels in [8, 32, 128].iter() {
        let channel_count = *channels;
        match channel_count {
            8 => {
                let mut rms: WindowedRms<8, 64> = WindowedRms::new();
                let sample = [1.0f32; 8];
                group.throughput(Throughput::Elements(8));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_64win_{}ch", channels)),
                    channels,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            32 => {
                let mut rms: WindowedRms<32, 64> = WindowedRms::new();
                let sample = [1.0f32; 32];
                group.throughput(Throughput::Elements(32));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_64win_{}ch", channels)),
                    channels,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            128 => {
                let mut rms: WindowedRms<128, 64> = WindowedRms::new();
                let sample = [1.0f32; 128];
                group.throughput(Throughput::Elements(128));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!("process_64win_{}ch", channels)),
                    channels,
                    |b, _| {
                        b.iter(|| {
                            rms.process(black_box(&sample));
                        });
                    },
                );
            }
            _ => {}
        }
    }

    // Block processing (256 samples, 4 channels, 64-sample window)
    {
        let mut rms: WindowedRms<4, 64> = WindowedRms::new();
        group.bench_function("block_256samples_4ch_64win", |b| {
            let mut block: Vec<[f32; 4]> = (0..256).map(|i| [(i as f32 * 0.01).sin(); 4]).collect();
            b.iter(|| {
                rms.process_block(black_box(&mut block));
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
    bench_csp,
    bench_window,
    bench_streaming_percentile,
    bench_oasis_deconvolution,
    bench_oasis_full_pipeline,
    bench_cwt,
    bench_stft,
    bench_artifact_detector,
    bench_zscore_artifact,
    bench_interpolator,
    bench_xcorr,
    bench_hilbert,
    bench_lms_filter,
    bench_nlms_filter,
    bench_windowed_rms
);
criterion_main!(benches);
