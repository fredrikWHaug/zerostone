//! Multi-stage pipeline demonstration with visualization.
//!
//! This example demonstrates composing multiple signal processing primitives into a
//! processing pipeline, showing intermediate results at each stage.
//!
//! **Pipeline:**
//! 1. Raw "EEG-like" signal (8 channels, multiple frequency bands + common noise)
//! 2. Bandpass filter (8-30 Hz) - isolate alpha and beta bands
//! 3. Common Average Reference (CAR) - remove common-mode noise
//! 4. Decimation (4x) - downsample from 1000 Hz to 250 Hz
//!
//! Run with: `cargo run --example pipeline_demo`
//!
//! Output: PNG plot showing all pipeline stages and CSV data files

mod common;

use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use zerostone::{BiquadCoeffs, CommonAverageReference, Decimator, IirFilter};

const SAMPLE_RATE: f32 = 1000.0; // 1 kHz
const DURATION_SECS: f32 = 2.0; // 2 seconds for better visualization
const SAMPLES: usize = (SAMPLE_RATE * DURATION_SECS) as usize;
const CHANNELS: usize = 8;

/// Data bundle for all pipeline stages
struct PipelineData {
    /// Raw input signal (before any processing)
    raw: Vec<[f32; CHANNELS]>,
    /// After bandpass filter (8-30 Hz)
    filtered: Vec<[f32; CHANNELS]>,
    /// After Common Average Reference
    car: Vec<[f32; CHANNELS]>,
    /// After decimation (4x downsampling)
    decimated: Vec<[f32; CHANNELS]>,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Zerostone Pipeline Demo ===\n");
    println!("Demonstrating multi-stage signal processing pipeline:");
    println!("  Raw → Bandpass (8-30 Hz) → CAR → Decimation (4x)\n");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // ========== Stage 0: Generate Synthetic "EEG-like" Signal ==========
    println!("Generating synthetic 8-channel EEG-like signal...");
    println!("  Signal components per channel:");
    println!("    - Delta (2 Hz, amp 0.3) - slow waves");
    println!("    - Theta (6 Hz, amp 0.4) - drowsiness");
    println!("    - Alpha (10 Hz, amp 1.0) - relaxed state [TARGET]");
    println!("    - Beta (20 Hz, amp 0.6) - active thinking [TARGET]");
    println!("    - Gamma (40 Hz, amp 0.3) - high frequency");
    println!("    - White noise (amp 0.15)");
    println!("  Common-mode noise (affects all channels):");
    println!("    - 50 Hz powerline interference (amp 0.5)");
    println!("    - Low-frequency drift (0.5 Hz, amp 0.4)\n");

    let raw = generate_eeg_like_signal();

    // ========== Stage 1: Bandpass Filter (8-30 Hz) ==========
    println!("Stage 1: Applying bandpass filter (8-30 Hz)...");
    println!("  Purpose: Isolate alpha (8-13 Hz) and beta (13-30 Hz) bands");
    println!("  Filter: 4th-order Butterworth bandpass");

    let coeffs = BiquadCoeffs::butterworth_bandpass(SAMPLE_RATE, 8.0, 30.0);
    // Each channel needs its own filter instance to maintain independent state
    let mut filters: [IirFilter<2>; CHANNELS] = std::array::from_fn(|_| {
        IirFilter::new([coeffs, coeffs]) // 2 sections = 4th order
    });

    let mut filtered = Vec::with_capacity(SAMPLES);
    for sample in &raw {
        let mut output = [0.0f32; CHANNELS];
        for (ch, &val) in sample.iter().enumerate() {
            output[ch] = filters[ch].process_sample(val);
        }
        filtered.push(output);
    }

    println!("  ✓ Filtered {} samples", filtered.len());

    // ========== Stage 2: Common Average Reference (CAR) ==========
    println!("\nStage 2: Applying Common Average Reference...");
    println!("  Purpose: Remove common-mode artifacts (powerline, movement)");
    println!("  Method: Subtract mean of all channels from each channel");

    let car_filter: CommonAverageReference<CHANNELS> = CommonAverageReference::new();
    let car: Vec<[f32; CHANNELS]> = filtered
        .iter()
        .map(|sample| car_filter.process(sample))
        .collect();

    // Verify CAR property: sum across channels should be near zero
    let car_sums: Vec<f32> = car.iter().map(|sample| sample.iter().sum()).collect();
    let max_sum = car_sums.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    println!("  ✓ Applied CAR (max channel sum: {:.2e})", max_sum);

    // ========== Stage 3: Decimation (4x) ==========
    println!("\nStage 3: Decimating by factor 4...");
    println!("  Purpose: Reduce data rate (1000 Hz → 250 Hz)");
    println!("  Note: Bandpass filter serves as anti-aliasing filter");

    let mut decimator: Decimator<CHANNELS> = Decimator::new(4);
    let mut decimated = Vec::with_capacity(SAMPLES / 4);

    for sample in &car {
        if let Some(out) = decimator.process(sample) {
            decimated.push(out);
        }
    }

    println!(
        "  ✓ Decimated {} samples → {} samples",
        car.len(),
        decimated.len()
    );

    // ========== Output Results ==========
    let data = PipelineData {
        raw,
        filtered,
        car,
        decimated,
    };

    // Write CSV files for each stage
    println!("\nWriting CSV files...");
    write_stage_csv("output/pipeline_raw.csv", &data.raw, SAMPLE_RATE)?;
    write_stage_csv("output/pipeline_filtered.csv", &data.filtered, SAMPLE_RATE)?;
    write_stage_csv("output/pipeline_car.csv", &data.car, SAMPLE_RATE)?;
    write_stage_csv(
        "output/pipeline_decimated.csv",
        &data.decimated,
        SAMPLE_RATE / 4.0,
    )?;
    println!("  ✓ Wrote 4 CSV files: pipeline_{{raw,filtered,car,decimated}}.csv");

    // Generate visualization
    println!("\nGenerating plot...");
    generate_plot(&data, "output/pipeline_demo.png")?;
    println!("  ✓ Wrote output/pipeline_demo.png");

    println!("\n=== Pipeline Complete ===");
    println!(
        "Data reduction: {} samples → {} samples ({:.1}x smaller)",
        data.raw.len(),
        data.decimated.len(),
        data.raw.len() as f32 / data.decimated.len() as f32
    );
    println!("\nOpen output/pipeline_demo.png to see the results.");

    Ok(())
}

/// Generate synthetic multi-channel EEG-like signal.
///
/// Creates 8 channels with:
/// - Individual frequency components (delta, theta, alpha, beta, gamma)
/// - Per-channel noise
/// - Common-mode interference (powerline, drift) affecting all channels equally
fn generate_eeg_like_signal() -> Vec<[f32; CHANNELS]> {
    let mut signal = vec![[0.0f32; CHANNELS]; SAMPLES];

    // Generate per-channel components (each channel has slightly different phase)
    #[allow(clippy::needless_range_loop)]
    for ch in 0..CHANNELS {
        let phase_offset = (ch as f32) * 0.3; // Slight phase shift per channel

        let delta = common::sine_wave(SAMPLES, SAMPLE_RATE, 2.0, 0.3, phase_offset);
        let theta = common::sine_wave(SAMPLES, SAMPLE_RATE, 6.0, 0.4, phase_offset);
        let alpha = common::sine_wave(SAMPLES, SAMPLE_RATE, 10.0, 1.0, phase_offset);
        let beta = common::sine_wave(SAMPLES, SAMPLE_RATE, 20.0, 0.6, phase_offset);
        let gamma = common::sine_wave(SAMPLES, SAMPLE_RATE, 40.0, 0.3, phase_offset);
        let noise = common::white_noise(SAMPLES, 0.15, 42 + ch as u64);

        for i in 0..SAMPLES {
            signal[i][ch] += delta[i] + theta[i] + alpha[i] + beta[i] + gamma[i] + noise[i];
        }
    }

    // Add common-mode noise (affects all channels equally)
    let powerline = common::sine_wave(SAMPLES, SAMPLE_RATE, 50.0, 0.5, 0.0);
    let drift = common::sine_wave(SAMPLES, SAMPLE_RATE, 0.5, 0.4, 0.0);

    for i in 0..SAMPLES {
        let common_noise = powerline[i] + drift[i];
        #[allow(clippy::needless_range_loop)]
        for ch in 0..CHANNELS {
            signal[i][ch] += common_noise;
        }
    }

    signal
}

/// Write a pipeline stage to CSV file.
fn write_stage_csv(
    path: &str,
    data: &[[f32; CHANNELS]],
    sample_rate: f32,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    // Header
    write!(file, "sample,time_ms")?;
    for ch in 0..CHANNELS {
        write!(file, ",ch{}", ch)?;
    }
    writeln!(file)?;

    // Data rows
    for (i, sample) in data.iter().enumerate() {
        let time_ms = (i as f32 / sample_rate) * 1000.0;
        write!(file, "{},{:.3}", i, time_ms)?;
        for &val in sample.iter() {
            write!(file, ",{:.6}", val)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Generate 4-panel visualization showing all pipeline stages.
fn generate_plot(data: &PipelineData, output_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((4, 1));

    // Display settings: show first 500ms of data
    let display_samples = (SAMPLE_RATE * 0.5) as usize;

    // Color palette for channels (use distinct colors)
    let colors = [
        RGBColor(220, 50, 50),   // Red
        RGBColor(50, 150, 50),   // Green
        RGBColor(50, 100, 220),  // Blue
        RGBColor(220, 150, 50),  // Orange
        RGBColor(150, 50, 220),  // Purple
        RGBColor(50, 200, 200),  // Cyan
        RGBColor(220, 50, 150),  // Magenta
        RGBColor(100, 100, 100), // Gray
    ];

    // Panel 1: Raw signal
    plot_multichannel(
        &areas[0],
        &data.raw,
        display_samples,
        SAMPLE_RATE,
        "Stage 0: Raw Signal (with common-mode noise)",
        &colors,
    )?;

    // Panel 2: Filtered signal
    plot_multichannel(
        &areas[1],
        &data.filtered,
        display_samples,
        SAMPLE_RATE,
        "Stage 1: Bandpass Filtered (8-30 Hz)",
        &colors,
    )?;

    // Panel 3: CAR signal
    plot_multichannel(
        &areas[2],
        &data.car,
        display_samples,
        SAMPLE_RATE,
        "Stage 2: Common Average Reference (common-mode removed)",
        &colors,
    )?;

    // Panel 4: Decimated signal (show equivalent time window)
    let decimated_display = display_samples / 4;
    plot_multichannel(
        &areas[3],
        &data.decimated,
        decimated_display,
        SAMPLE_RATE / 4.0, // New sample rate
        "Stage 3: Decimated 4x (250 Hz)",
        &colors,
    )?;

    root.present()?;
    Ok(())
}

/// Plot multi-channel signal with stacked traces.
fn plot_multichannel(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[[f32; CHANNELS]],
    display_samples: usize,
    sample_rate: f32,
    title: &str,
    colors: &[RGBColor],
) -> Result<(), Box<dyn Error>> {
    let display_samples = display_samples.min(data.len());
    let display_time_ms = (display_samples as f32 / sample_rate) * 1000.0;

    // Find y-axis range
    let y_min = data[..display_samples]
        .iter()
        .flat_map(|sample| sample.iter())
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let y_max = data[..display_samples]
        .iter()
        .flat_map(|sample| sample.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let y_margin = (y_max - y_min) * 0.1;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..display_time_ms,
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Amplitude")
        .draw()?;

    // Draw each channel
    for ch in 0..CHANNELS {
        let color = colors[ch % colors.len()];

        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / sample_rate) * 1000.0;
                    (t, data[i][ch])
                }),
                ShapeStyle::from(&color).stroke_width(1),
            ))?
            .label(format!("Ch{}", ch))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}
