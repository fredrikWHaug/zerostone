//! Performance benchmarking with visualization.
//!
//! This example measures the performance of key signal processing primitives
//! and generates bar charts showing latency and throughput characteristics.
//!
//! Run with: `cargo run --example benchmark_viz --release`
//!
//! Output: PNG plots and CSV data file in output/ directory

mod common;

use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use zerostone::{
    BiquadCoeffs, CommonAverageReference, Decimator, EnvelopeFollower, FirFilter, IirFilter,
    Rectification, SurfaceLaplacian,
};

/// Number of warmup iterations before measurement
const WARMUP_ITERS: usize = 10000;

/// Number of measurement iterations
const BENCH_ITERS: usize = 1000000;

/// Benchmark result with timing statistics
#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    latency_ns: f64,      // Average latency in nanoseconds
    throughput_msps: f64, // Throughput in millions of samples per second
    channel_count: usize, // Number of channels (if applicable)
}

impl BenchResult {
    fn latency_us(&self) -> f64 {
        self.latency_ns / 1000.0
    }
}

/// Benchmark a single-sample operation
fn bench_single_sample<F>(name: &str, channels: usize, mut op: F) -> BenchResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..WARMUP_ITERS {
        op();
    }

    // Measurement
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        op();
    }
    let elapsed = start.elapsed();

    // Calculate latency with a minimum threshold to avoid 0.0 ns
    let raw_latency_ns = elapsed.as_nanos() as f64 / BENCH_ITERS as f64;
    let latency_ns = raw_latency_ns.max(0.1);

    // Calculate throughput, capping at a reasonable maximum
    let raw_throughput = if channels > 0 {
        (BENCH_ITERS * channels) as f64 / elapsed.as_secs_f64() / 1_000_000.0
    } else {
        BENCH_ITERS as f64 / elapsed.as_secs_f64() / 1_000_000.0
    };
    let throughput_msps = raw_throughput.min(1_000_000.0); // Cap at 1 TSPS

    BenchResult {
        name: name.to_string(),
        latency_ns,
        throughput_msps,
        channel_count: channels,
    }
}

/// Run benchmarks for IIR filters with different channel counts
fn bench_iir_filters() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // 4th-order Butterworth lowpass (2 biquad sections)
    let coeffs = [
        BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
        BiquadCoeffs::butterworth_lowpass(1000.0, 40.0),
    ];

    // Single channel
    {
        let mut filter: IirFilter<2> = IirFilter::new(coeffs);
        results.push(bench_single_sample("IIR 1ch", 1, || {
            let _ = filter.process_sample(1.0);
        }));
    }

    // 8 channels
    {
        let mut filters: Vec<IirFilter<2>> = (0..8).map(|_| IirFilter::new(coeffs)).collect();
        results.push(bench_single_sample("IIR 8ch", 8, || {
            for filter in filters.iter_mut() {
                let _ = filter.process_sample(1.0);
            }
        }));
    }

    // 32 channels
    {
        let mut filters: Vec<IirFilter<2>> = (0..32).map(|_| IirFilter::new(coeffs)).collect();
        results.push(bench_single_sample("IIR 32ch", 32, || {
            for filter in filters.iter_mut() {
                let _ = filter.process_sample(1.0);
            }
        }));
    }

    // 128 channels
    {
        let mut filters: Vec<IirFilter<2>> = (0..128).map(|_| IirFilter::new(coeffs)).collect();
        results.push(bench_single_sample("IIR 128ch", 128, || {
            for filter in filters.iter_mut() {
                let _ = filter.process_sample(1.0);
            }
        }));
    }

    results
}

/// Run benchmarks for FIR filters with different tap counts
fn bench_fir_filters() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // 32-tap FIR
    {
        let mut filter: FirFilter<32> = FirFilter::moving_average();
        results.push(bench_single_sample("FIR 32tap", 1, || {
            let _ = filter.process_sample(1.0);
        }));
    }

    // 64-tap FIR
    {
        let mut filter: FirFilter<64> = FirFilter::moving_average();
        results.push(bench_single_sample("FIR 64tap", 1, || {
            let _ = filter.process_sample(1.0);
        }));
    }

    // 128-tap FIR
    {
        let mut filter: FirFilter<128> = FirFilter::moving_average();
        results.push(bench_single_sample("FIR 128tap", 1, || {
            let _ = filter.process_sample(1.0);
        }));
    }

    results
}

/// Run benchmarks for spatial filters (CAR, Laplacian)
fn bench_spatial_filters() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Common Average Reference - 8 channels
    {
        let car: CommonAverageReference<8> = CommonAverageReference::new();
        let samples = [1.0f32; 8];
        results.push(bench_single_sample("CAR 8ch", 8, || {
            let _ = car.process(&samples);
        }));
    }

    // Common Average Reference - 32 channels
    {
        let car: CommonAverageReference<32> = CommonAverageReference::new();
        let samples = [1.0f32; 32];
        results.push(bench_single_sample("CAR 32ch", 32, || {
            let _ = car.process(&samples);
        }));
    }

    // Common Average Reference - 128 channels
    {
        let car: CommonAverageReference<128> = CommonAverageReference::new();
        let samples = [1.0f32; 128];
        results.push(bench_single_sample("CAR 128ch", 128, || {
            let _ = car.process(&samples);
        }));
    }

    // Surface Laplacian - 32 channels (ring topology)
    {
        let mut neighbors = [[u16::MAX; 4]; 32];
        for (i, neighbor) in neighbors.iter_mut().enumerate() {
            neighbor[0] = ((i + 31) % 32) as u16;
            neighbor[1] = ((i + 1) % 32) as u16;
        }
        let laplacian: SurfaceLaplacian<32, 4> = SurfaceLaplacian::unweighted(neighbors);
        let samples = [1.0f32; 32];
        results.push(bench_single_sample("Laplacian 32ch", 32, || {
            let _ = laplacian.process(&samples);
        }));
    }

    results
}

/// Run benchmarks for streaming primitives
fn bench_streaming_primitives() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // EnvelopeFollower - 32 channels
    {
        let mut env: EnvelopeFollower<32> =
            EnvelopeFollower::new(250.0, 0.010, 0.100, Rectification::Absolute);
        let samples = [0.5f32; 32];
        results.push(bench_single_sample("Envelope 32ch", 32, || {
            let _ = env.process(&samples);
        }));
    }

    // Decimator - 32 channels, factor 4
    {
        let mut dec: Decimator<32> = Decimator::new(4);
        let samples = [1.0f32; 32];
        results.push(bench_single_sample("Decimator 32ch", 32, || {
            let _ = dec.process(&samples);
        }));
    }

    results
}

/// Generate latency comparison bar chart
fn plot_latency_comparison(
    results: &[BenchResult],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_latency = results.iter().map(|r| r.latency_ns).fold(0.0f64, f64::max);
    let y_max = (max_latency * 1.1).ceil();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Primitive Latency Comparison",
            ("sans-serif", 40).into_font(),
        )
        .margin(20)
        .x_label_area_size(100)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..(results.len() as f64), 0.0..y_max)?;

    chart
        .configure_mesh()
        .y_desc("Latency (nanoseconds)")
        .x_desc("Primitive")
        .x_labels(results.len())
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < results.len() {
                results[idx].name.clone()
            } else {
                String::new()
            }
        })
        .label_style(("sans-serif", 15))
        .draw()?;

    // Draw bars
    for (i, result) in results.iter().enumerate() {
        let color = if result.name.starts_with("IIR") {
            &BLUE
        } else if result.name.starts_with("FIR") {
            &GREEN
        } else if result.name.starts_with("CAR") {
            &RED
        } else if result.name.starts_with("Laplacian") {
            &MAGENTA
        } else {
            &CYAN
        };

        chart.draw_series(std::iter::once(Rectangle::new(
            [(i as f64 + 0.1, 0.0), (i as f64 + 0.9, result.latency_ns)],
            color.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Generate channel scaling chart
fn plot_channel_scaling(results: &[BenchResult]) -> Result<(), Box<dyn Error>> {
    // Filter results by primitive type and sort by channel count
    let iir_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("IIR"))
        .collect();
    let car_results: Vec<_> = results
        .iter()
        .filter(|r| r.name.starts_with("CAR"))
        .collect();

    let root = BitMapBackend::new("output/benchmark_scaling.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_channels = results.iter().map(|r| r.channel_count).max().unwrap_or(128);
    let max_latency = results.iter().map(|r| r.latency_ns).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Performance Scaling with Channel Count",
            ("sans-serif", 40).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..(max_channels as f64 * 1.1), 0.0..(max_latency * 1.2))?;

    chart
        .configure_mesh()
        .x_desc("Channel Count")
        .y_desc("Latency (nanoseconds)")
        .label_style(("sans-serif", 15))
        .draw()?;

    // Plot IIR results
    if !iir_results.is_empty() {
        chart
            .draw_series(LineSeries::new(
                iir_results
                    .iter()
                    .map(|r| (r.channel_count as f64, r.latency_ns)),
                &BLUE,
            ))?
            .label("IIR Filter (4th order)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart.draw_series(
            iir_results
                .iter()
                .map(|r| Circle::new((r.channel_count as f64, r.latency_ns), 4, BLUE.filled())),
        )?;
    }

    // Plot CAR results
    if !car_results.is_empty() {
        chart
            .draw_series(LineSeries::new(
                car_results
                    .iter()
                    .map(|r| (r.channel_count as f64, r.latency_ns)),
                &RED,
            ))?
            .label("Common Average Reference")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart.draw_series(
            car_results
                .iter()
                .map(|r| Circle::new((r.channel_count as f64, r.latency_ns), 4, RED.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 20))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Write results to CSV
fn write_csv(results: &[BenchResult], path: &str) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "Primitive,Channels,Latency_ns,Latency_us,Throughput_MSPS"
    )?;

    for result in results {
        writeln!(
            file,
            "{},{},{:.2},{:.4},{:.2}",
            result.name,
            result.channel_count,
            result.latency_ns,
            result.latency_us(),
            result.throughput_msps
        )?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Zerostone Performance Benchmark ===\n");
    println!("IMPORTANT: Run with --release flag for accurate results!");
    println!("  cargo run --example benchmark_viz --release\n");

    #[cfg(debug_assertions)]
    {
        println!("WARNING: Running in debug mode. Results will be significantly slower.");
        println!("Use --release for accurate benchmarks.\n");
    }

    // Create output directory
    std::fs::create_dir_all("output")?;

    let mut all_results = Vec::new();

    // IIR filters
    println!("Benchmarking IIR filters...");
    let iir_results = bench_iir_filters();
    for r in &iir_results {
        println!(
            "  {} - {:.1} ns/sample ({:.2} MSPS)",
            r.name, r.latency_ns, r.throughput_msps
        );
    }
    all_results.extend(iir_results);

    // FIR filters
    println!("\nBenchmarking FIR filters...");
    let fir_results = bench_fir_filters();
    for r in &fir_results {
        println!("  {} - {:.1} ns/sample", r.name, r.latency_ns);
    }
    all_results.extend(fir_results);

    // Spatial filters
    println!("\nBenchmarking spatial filters...");
    let spatial_results = bench_spatial_filters();
    for r in &spatial_results {
        println!(
            "  {} - {:.1} ns/sample ({:.2} MSPS)",
            r.name, r.latency_ns, r.throughput_msps
        );
    }
    all_results.extend(spatial_results);

    // Streaming primitives
    println!("\nBenchmarking streaming primitives...");
    let streaming_results = bench_streaming_primitives();
    for r in &streaming_results {
        println!(
            "  {} - {:.1} ns/sample ({:.2} MSPS)",
            r.name, r.latency_ns, r.throughput_msps
        );
    }
    all_results.extend(streaming_results);

    // Generate plots
    println!("\nGenerating visualizations...");
    plot_latency_comparison(&all_results, "output/benchmark_latency.png")?;
    plot_channel_scaling(&all_results)?;

    // Write CSV
    write_csv(&all_results, "output/benchmark_results.csv")?;

    println!("\nResults saved:");
    println!("  - output/benchmark_latency.png  (latency comparison)");
    println!("  - output/benchmark_scaling.png  (channel scaling)");
    println!("  - output/benchmark_results.csv  (raw data)");

    println!("\n=== Performance Summary ===");
    println!(
        "Fastest primitive: {} ({:.1} ns)",
        all_results
            .iter()
            .min_by(|a, b| a.latency_ns.partial_cmp(&b.latency_ns).unwrap())
            .unwrap()
            .name,
        all_results
            .iter()
            .map(|r| r.latency_ns)
            .fold(f64::INFINITY, f64::min)
    );

    println!(
        "Highest throughput: {} ({:.2} MSPS)",
        all_results
            .iter()
            .max_by(|a, b| a.throughput_msps.partial_cmp(&b.throughput_msps).unwrap())
            .unwrap()
            .name,
        all_results
            .iter()
            .map(|r| r.throughput_msps)
            .fold(0.0, f64::max)
    );

    Ok(())
}
