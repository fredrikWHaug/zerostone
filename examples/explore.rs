//! TOML-based exploration tool for signal processing demonstrations.
//!
//! This example allows runtime configuration of signal processing chains through TOML files.
//! It includes a recipe system for common use cases.
//!
//! Run with:
//!   cargo run --example explore                          # Uses explore.toml
//!   cargo run --example explore -- --recipe lowpass      # Uses embedded recipe
//!   cargo run --example explore -- --config my.toml      # Uses custom file

mod common;

use clap::Parser;
use plotters::prelude::*;
use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::io::Write;
use zerostone::{BiquadCoeffs, CommonAverageReference, IirFilter, SurfaceLaplacian};

// ============================================================================
// Recipe Constants
// ============================================================================

const RECIPE_LOWPASS: &str = r#"
[signal]
duration = 1.0
sample_rate = 1000.0
channels = 1

[signal.components]
sine_waves = [[10.0, 1.0], [60.0, 0.3]]
noise_amplitude = 0.2
noise_seed = 42

[[filters]]
type = "lowpass"
cutoff = 30.0
order = 4

[output]
plot_path = "output/explore_lowpass.png"
csv_path = "output/explore_lowpass.csv"
layout = "stacked"
"#;

const RECIPE_HIGHPASS: &str = r#"
[signal]
duration = 1.0
sample_rate = 1000.0
channels = 1

[signal.components]
sine_waves = [[5.0, 0.5], [60.0, 1.0]]
noise_amplitude = 0.2
noise_seed = 42

[[filters]]
type = "highpass"
cutoff = 20.0
order = 4

[output]
plot_path = "output/explore_highpass.png"
csv_path = "output/explore_highpass.csv"
layout = "stacked"
"#;

const RECIPE_BANDPASS: &str = r#"
[signal]
duration = 1.0
sample_rate = 1000.0
channels = 1

[signal.components]
sine_waves = [[5.0, 0.3], [20.0, 1.0], [60.0, 0.3]]
noise_amplitude = 0.2
noise_seed = 42

[[filters]]
type = "bandpass"
low = 15.0
high = 25.0
order = 4

[output]
plot_path = "output/explore_bandpass.png"
csv_path = "output/explore_bandpass.csv"
layout = "stacked"
"#;

const RECIPE_SPATIAL: &str = r#"
[signal]
duration = 1.0
sample_rate = 1000.0
channels = 8

[signal.components]
sine_waves = [[10.0, 0.5], [20.0, 0.3]]
noise_amplitude = 0.1
noise_seed = 42

[signal.common_noise]
frequency = 50.0
amplitude = 0.5

[[filters]]
type = "car"

[output]
plot_path = "output/explore_spatial.png"
csv_path = "output/explore_spatial.csv"
layout = "stacked"
"#;

const RECIPE_PIPELINE: &str = r#"
[signal]
duration = 1.0
sample_rate = 1000.0
channels = 8

[signal.components]
sine_waves = [[3.0, 0.2], [7.0, 0.3], [10.0, 0.5], [20.0, 0.4], [40.0, 0.2]]
noise_amplitude = 0.1
noise_seed = 42

[signal.common_noise]
frequency = 50.0
amplitude = 0.5

[[filters]]
type = "bandpass"
low = 8.0
high = 30.0
order = 4

[[filters]]
type = "car"

[output]
plot_path = "output/explore_pipeline.png"
csv_path = "output/explore_pipeline.csv"
layout = "stacked"
"#;

// ============================================================================
// Configuration Structures
// ============================================================================

#[derive(Deserialize)]
struct ExploreConfig {
    signal: SignalConfig,
    filters: Vec<FilterConfig>,
    output: OutputConfig,
}

#[derive(Deserialize)]
struct SignalConfig {
    duration: f32,
    sample_rate: f32,
    channels: usize,
    components: SignalComponents,
    #[serde(default)]
    common_noise: Option<CommonNoiseConfig>,
}

#[derive(Deserialize)]
struct SignalComponents {
    sine_waves: Vec<(f32, f32)>,
    noise_amplitude: f32,
    noise_seed: u64,
}

#[derive(Deserialize)]
struct CommonNoiseConfig {
    frequency: f32,
    amplitude: f32,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum FilterConfig {
    Lowpass { cutoff: f32, order: u8 },
    Highpass { cutoff: f32, order: u8 },
    Bandpass { low: f32, high: f32, order: u8 },
    Car,
    Laplacian,
}

#[derive(Deserialize)]
struct OutputConfig {
    plot_path: String,
    csv_path: String,
    layout: PlotLayout,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum PlotLayout {
    Stacked,
    Grid,
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Recipe name (lowpass, highpass, bandpass, spatial, pipeline)
    #[arg(short, long)]
    recipe: Option<String>,

    /// Custom config file path
    #[arg(short, long)]
    config: Option<String>,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== Zerostone Exploration Tool ===\n");

    // Load config (priority: --config > --recipe > explore.toml)
    let config_str = if let Some(recipe_name) = args.recipe {
        println!("Using recipe: {}", recipe_name);
        load_recipe(&recipe_name)?
    } else if let Some(config_path) = args.config {
        println!("Loading config: {}", config_path);
        fs::read_to_string(config_path)?
    } else {
        match fs::read_to_string("explore.toml") {
            Ok(content) => {
                println!("Using explore.toml");
                content
            }
            Err(_) => {
                println!("No explore.toml found, using lowpass recipe");
                RECIPE_LOWPASS.to_string()
            }
        }
    };

    let config: ExploreConfig =
        toml::from_str(&config_str).map_err(|e| format!("Failed to parse TOML config: {}", e))?;

    println!();
    println!("Configuration:");
    println!("  Duration: {:.1}s", config.signal.duration);
    println!("  Sample rate: {:.1} Hz", config.signal.sample_rate);
    println!("  Channels: {}", config.signal.channels);
    println!("  Filters: {}", config.filters.len());
    println!();

    // Generate signal
    let samples = (config.signal.duration * config.signal.sample_rate) as usize;

    let signal = if config.signal.channels == 1 {
        generate_signal_single(&config.signal, samples)
    } else {
        generate_signal_multi(&config.signal, samples)
    };

    // Apply filters sequentially
    let mut stages: Vec<Vec<Vec<f32>>> = vec![signal.clone()];
    let mut stage_names = vec!["Raw".to_string()];

    for filter_cfg in &config.filters {
        validate_filter(filter_cfg, config.signal.channels)?;

        let filtered = apply_filter(stages.last().unwrap(), filter_cfg, &config.signal)?;
        stage_names.push(filter_name(filter_cfg));
        stages.push(filtered);
    }

    // Create output directory
    if let Some(parent) = std::path::Path::new(&config.output.plot_path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Export results
    write_csv(&stages, &stage_names, &config)?;
    generate_plot(&stages, &stage_names, &config)?;

    println!("Done!");
    println!("  Plot: {}", config.output.plot_path);
    println!("  CSV: {}", config.output.csv_path);

    Ok(())
}

// ============================================================================
// Config Loading
// ============================================================================

fn load_recipe(name: &str) -> Result<String, Box<dyn Error>> {
    match name {
        "lowpass" => Ok(RECIPE_LOWPASS.to_string()),
        "highpass" => Ok(RECIPE_HIGHPASS.to_string()),
        "bandpass" => Ok(RECIPE_BANDPASS.to_string()),
        "spatial" => Ok(RECIPE_SPATIAL.to_string()),
        "pipeline" => Ok(RECIPE_PIPELINE.to_string()),
        _ => Err(format!(
            "Unknown recipe '{}'. Available recipes: lowpass, highpass, bandpass, spatial, pipeline",
            name
        )
        .into()),
    }
}

// ============================================================================
// Signal Generation
// ============================================================================

/// Generate single-channel signal
fn generate_signal_single(config: &SignalConfig, samples: usize) -> Vec<Vec<f32>> {
    let signal = common::composite_signal(
        samples,
        config.sample_rate,
        &config.components.sine_waves,
        config.components.noise_amplitude,
        config.components.noise_seed,
    );

    vec![signal]
}

/// Generate multi-channel signal with per-channel variation
fn generate_signal_multi(config: &SignalConfig, samples: usize) -> Vec<Vec<f32>> {
    let mut channels = Vec::with_capacity(config.channels);

    // Generate base signal for each channel with phase offset
    for ch in 0..config.channels {
        let mut signal = vec![0.0; samples];

        // Add sine components with per-channel phase offset
        for &(freq, amp) in &config.components.sine_waves {
            let phase = ch as f32 * 0.3;
            let sine = common::sine_wave(samples, config.sample_rate, freq, amp, phase);
            for (i, &s) in sine.iter().enumerate() {
                signal[i] += s;
            }
        }

        // Add per-channel noise
        let noise_seed = config.components.noise_seed + ch as u64;
        let noise = common::white_noise(samples, config.components.noise_amplitude, noise_seed);
        for (i, &n) in noise.iter().enumerate() {
            signal[i] += n;
        }

        channels.push(signal);
    }

    // Add common noise if specified (affects all channels equally)
    if let Some(common_cfg) = &config.common_noise {
        let common_noise = common::sine_wave(
            samples,
            config.sample_rate,
            common_cfg.frequency,
            common_cfg.amplitude,
            0.0,
        );

        for channel in &mut channels {
            for (i, &cn) in common_noise.iter().enumerate() {
                channel[i] += cn;
            }
        }
    }

    channels
}

// ============================================================================
// Filter Application
// ============================================================================

fn validate_filter(filter: &FilterConfig, channels: usize) -> Result<(), Box<dyn Error>> {
    match filter {
        FilterConfig::Car | FilterConfig::Laplacian => {
            if channels < 2 {
                return Err(format!(
                    "{} filter requires at least 2 channels, got {}",
                    filter_name(filter),
                    channels
                )
                .into());
            }
        }
        _ => {}
    }
    Ok(())
}

fn apply_filter(
    input: &[Vec<f32>],
    filter_cfg: &FilterConfig,
    signal_cfg: &SignalConfig,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    match filter_cfg {
        FilterConfig::Lowpass { cutoff, order } => {
            apply_lowpass_filter(input, *cutoff, *order, signal_cfg.sample_rate)
        }
        FilterConfig::Highpass { cutoff, order } => {
            apply_highpass_filter(input, *cutoff, *order, signal_cfg.sample_rate)
        }
        FilterConfig::Bandpass { low, high, order } => {
            apply_bandpass_filter(input, *low, *high, *order, signal_cfg.sample_rate)
        }
        FilterConfig::Car => apply_car_filter(input, signal_cfg.channels),
        FilterConfig::Laplacian => apply_laplacian_filter(input, signal_cfg.channels),
    }
}

fn apply_lowpass_filter(
    input: &[Vec<f32>],
    cutoff: f32,
    order: u8,
    sample_rate: f32,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(input.len());
    let num_sections = (order / 2) as usize;

    for channel in input {
        let coeffs = BiquadCoeffs::butterworth_lowpass(sample_rate, cutoff);
        let filtered = match num_sections {
            1 => apply_filter_1_section(channel, coeffs),
            2 => apply_filter_2_sections(channel, coeffs),
            3 => apply_filter_3_sections(channel, coeffs),
            4 => apply_filter_4_sections(channel, coeffs),
            _ => return Err(format!("Unsupported filter order: {}", order).into()),
        };
        output.push(filtered);
    }

    Ok(output)
}

fn apply_highpass_filter(
    input: &[Vec<f32>],
    cutoff: f32,
    order: u8,
    sample_rate: f32,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(input.len());
    let num_sections = (order / 2) as usize;

    for channel in input {
        let coeffs = BiquadCoeffs::butterworth_highpass(sample_rate, cutoff);
        let filtered = match num_sections {
            1 => apply_filter_1_section(channel, coeffs),
            2 => apply_filter_2_sections(channel, coeffs),
            3 => apply_filter_3_sections(channel, coeffs),
            4 => apply_filter_4_sections(channel, coeffs),
            _ => return Err(format!("Unsupported filter order: {}", order).into()),
        };
        output.push(filtered);
    }

    Ok(output)
}

fn apply_bandpass_filter(
    input: &[Vec<f32>],
    low: f32,
    high: f32,
    order: u8,
    sample_rate: f32,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut output = Vec::with_capacity(input.len());
    let num_sections = (order / 2) as usize;

    for channel in input {
        let coeffs = BiquadCoeffs::butterworth_bandpass(sample_rate, low, high);
        let filtered = match num_sections {
            1 => apply_filter_1_section(channel, coeffs),
            2 => apply_filter_2_sections(channel, coeffs),
            3 => apply_filter_3_sections(channel, coeffs),
            4 => apply_filter_4_sections(channel, coeffs),
            _ => return Err(format!("Unsupported filter order: {}", order).into()),
        };
        output.push(filtered);
    }

    Ok(output)
}

fn apply_filter_1_section(channel: &[f32], coeffs: BiquadCoeffs) -> Vec<f32> {
    let mut filter: IirFilter<1> = IirFilter::new([coeffs]);
    channel.iter().map(|&x| filter.process_sample(x)).collect()
}

fn apply_filter_2_sections(channel: &[f32], coeffs: BiquadCoeffs) -> Vec<f32> {
    let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
    channel.iter().map(|&x| filter.process_sample(x)).collect()
}

fn apply_filter_3_sections(channel: &[f32], coeffs: BiquadCoeffs) -> Vec<f32> {
    let mut filter: IirFilter<3> = IirFilter::new([coeffs, coeffs, coeffs]);
    channel.iter().map(|&x| filter.process_sample(x)).collect()
}

fn apply_filter_4_sections(channel: &[f32], coeffs: BiquadCoeffs) -> Vec<f32> {
    let mut filter: IirFilter<4> = IirFilter::new([coeffs, coeffs, coeffs, coeffs]);
    channel.iter().map(|&x| filter.process_sample(x)).collect()
}

fn apply_car_filter(input: &[Vec<f32>], channels: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // Handle different channel counts
    match channels {
        4 => apply_car_typed::<4>(input),
        8 => apply_car_typed::<8>(input),
        _ => Err(format!("CAR filter only supports 4 or 8 channels, got {}", channels).into()),
    }
}

fn apply_car_typed<const C: usize>(input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let car = CommonAverageReference::<C>::new();
    let samples = input[0].len();
    let mut output = vec![vec![0.0; samples]; C];

    for i in 0..samples {
        let mut sample = [0.0f32; C];
        for (ch, channel) in input.iter().enumerate().take(C) {
            sample[ch] = channel[i];
        }

        let filtered = car.process(&sample);
        for (ch, &val) in filtered.iter().enumerate() {
            output[ch][i] = val;
        }
    }

    Ok(output)
}

fn apply_laplacian_filter(
    input: &[Vec<f32>],
    channels: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // Handle different channel counts
    match channels {
        4 => apply_laplacian_4ch(input),
        8 => apply_laplacian_8ch(input),
        _ => Err(format!(
            "Laplacian filter only supports 4 or 8 channels, got {}",
            channels
        )
        .into()),
    }
}

fn apply_laplacian_4ch(input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // 2x2 grid layout
    let neighbors = [
        [1, 2, u16::MAX], // Ch 0: neighbors 1, 2
        [0, 3, u16::MAX], // Ch 1: neighbors 0, 3
        [0, 3, u16::MAX], // Ch 2: neighbors 0, 3
        [1, 2, u16::MAX], // Ch 3: neighbors 1, 2
    ];

    let laplacian: SurfaceLaplacian<4, 3> = SurfaceLaplacian::unweighted(neighbors);
    let samples = input[0].len();
    let mut output = vec![vec![0.0; samples]; 4];

    for i in 0..samples {
        let mut sample = [0.0f32; 4];
        for (ch, channel) in input.iter().enumerate().take(4) {
            sample[ch] = channel[i];
        }

        let filtered = laplacian.process(&sample);
        for (ch, &val) in filtered.iter().enumerate() {
            output[ch][i] = val;
        }
    }

    Ok(output)
}

fn apply_laplacian_8ch(input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // Linear array for simplicity
    let neighbors = [
        [1, u16::MAX, u16::MAX], // Ch 0: neighbor 1
        [0, 2, u16::MAX],        // Ch 1: neighbors 0, 2
        [1, 3, u16::MAX],        // Ch 2: neighbors 1, 3
        [2, 4, u16::MAX],        // Ch 3: neighbors 2, 4
        [3, 5, u16::MAX],        // Ch 4: neighbors 3, 5
        [4, 6, u16::MAX],        // Ch 5: neighbors 4, 6
        [5, 7, u16::MAX],        // Ch 6: neighbors 5, 7
        [6, u16::MAX, u16::MAX], // Ch 7: neighbor 6
    ];

    let laplacian: SurfaceLaplacian<8, 3> = SurfaceLaplacian::unweighted(neighbors);
    let samples = input[0].len();
    let mut output = vec![vec![0.0; samples]; 8];

    for i in 0..samples {
        let mut sample = [0.0f32; 8];
        for (ch, channel) in input.iter().enumerate().take(8) {
            sample[ch] = channel[i];
        }

        let filtered = laplacian.process(&sample);
        for (ch, &val) in filtered.iter().enumerate() {
            output[ch][i] = val;
        }
    }

    Ok(output)
}

fn filter_name(filter: &FilterConfig) -> String {
    match filter {
        FilterConfig::Lowpass { cutoff, order } => {
            format!("Lowpass {:.1} Hz (order {})", cutoff, order)
        }
        FilterConfig::Highpass { cutoff, order } => {
            format!("Highpass {:.1} Hz (order {})", cutoff, order)
        }
        FilterConfig::Bandpass { low, high, order } => {
            format!("Bandpass {:.1}-{:.1} Hz (order {})", low, high, order)
        }
        FilterConfig::Car => "Common Average Reference".to_string(),
        FilterConfig::Laplacian => "Surface Laplacian".to_string(),
    }
}

// ============================================================================
// CSV Export
// ============================================================================

fn write_csv(
    stages: &[Vec<Vec<f32>>],
    names: &[String],
    config: &ExploreConfig,
) -> Result<(), Box<dyn Error>> {
    let mut file = fs::File::create(&config.output.csv_path)?;

    for (stage_idx, (stage, name)) in stages.iter().zip(names.iter()).enumerate() {
        writeln!(file, "# Stage {}: {}", stage_idx, name)?;

        // Write header
        write!(file, "sample,time_ms")?;
        for ch in 0..stage.len() {
            write!(file, ",channel_{}", ch)?;
        }
        writeln!(file)?;

        // Write data
        let samples = stage[0].len();
        let dt = 1000.0 / config.signal.sample_rate; // time step in ms

        for i in 0..samples {
            write!(file, "{},{:.3}", i, i as f32 * dt)?;
            for channel in stage {
                write!(file, ",{:.6}", channel[i])?;
            }
            writeln!(file)?;
        }

        writeln!(file)?;
    }

    Ok(())
}

// ============================================================================
// Plotting
// ============================================================================

fn generate_plot(
    stages: &[Vec<Vec<f32>>],
    names: &[String],
    config: &ExploreConfig,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(&config.output.plot_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine layout
    let (rows, cols) = match config.output.layout {
        PlotLayout::Stacked => (stages.len(), 1),
        PlotLayout::Grid => {
            let n = stages.len();
            let cols = ((n as f32).sqrt().ceil() as usize).max(2);
            let rows = n.div_ceil(cols);
            (rows, cols)
        }
    };

    let panels = root.split_evenly((rows, cols));

    // Plot first 500ms of data
    let plot_duration_ms = 500.0;
    let plot_samples = (plot_duration_ms * config.signal.sample_rate / 1000.0) as usize;

    for (idx, (stage, name)) in stages.iter().zip(names.iter()).enumerate() {
        if idx >= panels.len() {
            break;
        }

        plot_stage(&panels[idx], stage, name, plot_samples, config)?;
    }

    root.present()?;
    Ok(())
}

fn plot_stage(
    panel: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    stage: &[Vec<f32>],
    name: &str,
    plot_samples: usize,
    config: &ExploreConfig,
) -> Result<(), Box<dyn Error>> {
    let samples = plot_samples.min(stage[0].len());
    let time_vec: Vec<f32> = (0..samples)
        .map(|i| i as f32 * 1000.0 / config.signal.sample_rate)
        .collect();

    // Find y-axis range
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;

    for channel in stage {
        for &val in &channel[..samples] {
            y_min = y_min.min(val);
            y_max = y_max.max(val);
        }
    }

    let margin = (y_max - y_min) * 0.1;
    y_min -= margin;
    y_max += margin;

    let mut chart = ChartBuilder::on(panel)
        .caption(name, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f32..time_vec[samples - 1], y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Time (ms)")
        .y_desc("Amplitude")
        .draw()?;

    // Plot each channel with different color
    let colors = [
        &BLUE,
        &RED,
        &GREEN,
        &MAGENTA,
        &CYAN,
        &BLACK,
        &YELLOW,
        &full_palette::ORANGE,
    ];

    for (ch_idx, channel) in stage.iter().enumerate() {
        let color = colors[ch_idx % colors.len()];
        chart.draw_series(LineSeries::new(
            time_vec
                .iter()
                .zip(&channel[..samples])
                .map(|(&t, &v)| (t, v)),
            color.stroke_width(1),
        ))?;
    }

    Ok(())
}
