//! Spatial filter visualization with multi-channel plots.
//!
//! This example demonstrates spatial filtering techniques for multi-channel neural recordings:
//!
//! 1. **Common Average Reference (CAR)**: Removes common-mode noise by subtracting the mean
//!    of all channels from each channel
//! 2. **Surface Laplacian**: Estimates the second spatial derivative, sharpening local activity
//!    and reducing volume conduction effects
//! 3. **ChannelRouter**: Demonstrates channel selection, reordering, and subset extraction
//!
//! Run with: `cargo run --example spatial_demo`
//!
//! Output: PNG plots and CSV data files in the output/ directory

mod common;

use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use zerostone::{ChannelRouter, CommonAverageReference, SurfaceLaplacian};

const SAMPLE_RATE: f32 = 1000.0; // 1 kHz
const DURATION_SECS: f32 = 1.0;
const SAMPLES: usize = (SAMPLE_RATE * DURATION_SECS) as usize;
const CHANNELS: usize = 8;

/// Data bundle for spatial filter comparison
struct SpatialDemoData {
    /// Raw input signal with common-mode noise
    raw: Vec<[f32; CHANNELS]>,
    /// After Common Average Reference
    car: Vec<[f32; CHANNELS]>,
    /// After Surface Laplacian
    laplacian: Vec<[f32; CHANNELS]>,
}

/// Data bundle for ChannelRouter demonstration
struct RouterDemoData {
    /// Original 8-channel signal
    original: Vec<[f32; CHANNELS]>,
    /// Selected subset (channels 0, 2, 4, 6)
    selected: Vec<[f32; 4]>,
    /// Permuted order (reversed)
    permuted: Vec<[f32; CHANNELS]>,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Zerostone Spatial Filter Demo ===\n");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // ========== Part 1: Spatial Filter Comparison ==========
    println!("Part 1: Spatial Filter Comparison");
    println!("==================================\n");

    // Generate synthetic multi-channel signal with common-mode noise
    println!("Generating 8-channel synthetic signal...");
    println!("  Per-channel components:");
    println!("    - Unique signal (10 Hz, varying amplitude per channel)");
    println!("    - Channel-specific noise (white noise, amp 0.15)");
    println!("  Common-mode artifacts (affects ALL channels equally):");
    println!("    - 50 Hz powerline interference (amp 0.8)");
    println!("    - Low-frequency drift (0.5 Hz, amp 0.5)\n");

    let raw = generate_multichannel_signal();

    // Apply Common Average Reference
    println!("Applying Common Average Reference (CAR)...");
    println!("  Purpose: Remove global artifacts (powerline, movement)");
    println!("  Method: Subtract mean across channels from each channel");

    let car_filter: CommonAverageReference<CHANNELS> = CommonAverageReference::new();
    let car: Vec<[f32; CHANNELS]> = raw.iter().map(|s| car_filter.process(s)).collect();

    // Verify CAR property: channels should sum to near-zero
    let car_sum: f32 =
        car.iter().map(|s| s.iter().sum::<f32>().abs()).sum::<f32>() / SAMPLES as f32;
    println!(
        "  Average absolute channel sum after CAR: {:.2e} (should be ~0)\n",
        car_sum
    );

    // Apply Surface Laplacian
    println!("Applying Surface Laplacian...");
    println!("  Purpose: Enhance local activity, reduce volume conduction");
    println!("  Method: Subtract weighted average of neighbors from each channel");
    println!("  Topology: Linear array (each channel has 1-2 neighbors)\n");

    // Define neighbor topology for linear array: 0-1-2-3-4-5-6-7
    let neighbors: [[u16; 2]; CHANNELS] = [
        [1, u16::MAX], // Channel 0: only neighbor 1
        [0, 2],        // Channel 1: neighbors 0, 2
        [1, 3],        // Channel 2: neighbors 1, 3
        [2, 4],        // Channel 3: neighbors 2, 4
        [3, 5],        // Channel 4: neighbors 3, 5
        [4, 6],        // Channel 5: neighbors 4, 6
        [5, 7],        // Channel 6: neighbors 5, 7
        [6, u16::MAX], // Channel 7: only neighbor 6
    ];

    let laplacian_filter: SurfaceLaplacian<CHANNELS, 2> = SurfaceLaplacian::unweighted(neighbors);
    let laplacian: Vec<[f32; CHANNELS]> = raw.iter().map(|s| laplacian_filter.process(s)).collect();

    // Bundle data
    let spatial_data = SpatialDemoData {
        raw,
        car,
        laplacian,
    };

    // Calculate noise reduction metrics
    println!("Computing noise reduction metrics...");
    let common_mode_reduction = compute_common_mode_reduction(&spatial_data);
    println!(
        "  Common-mode power reduction (CAR): {:.1} dB",
        common_mode_reduction.0
    );
    println!(
        "  Common-mode power reduction (Laplacian): {:.1} dB\n",
        common_mode_reduction.1
    );

    // Write CSV files
    println!("Writing CSV files...");
    write_spatial_csv("output/spatial_raw.csv", &spatial_data.raw)?;
    write_spatial_csv("output/spatial_car.csv", &spatial_data.car)?;
    write_spatial_csv("output/spatial_laplacian.csv", &spatial_data.laplacian)?;
    println!("  Wrote: spatial_{{raw,car,laplacian}}.csv");

    // Generate spatial filter comparison plot
    println!("\nGenerating spatial filter comparison plot...");
    generate_spatial_plot(&spatial_data, "output/spatial_demo.png")?;
    println!("  Wrote: output/spatial_demo.png");

    // ========== Part 2: ChannelRouter Demonstration ==========
    println!("\n\nPart 2: ChannelRouter Demonstration");
    println!("====================================\n");

    // Use the raw signal from Part 1
    let original = spatial_data.raw.clone();

    // Demonstrate channel selection: pick every other channel (0, 2, 4, 6)
    println!("1. Channel Selection: Extracting channels [0, 2, 4, 6] from 8 channels");
    let selector: ChannelRouter<8, 4> = ChannelRouter::select([0, 2, 4, 6]);
    let selected: Vec<[f32; 4]> = original.iter().map(|s| selector.process(s)).collect();
    println!("   Input: 8 channels -> Output: 4 channels");

    // Demonstrate channel permutation: reverse order
    println!("\n2. Channel Permutation: Reversing channel order [7,6,5,4,3,2,1,0]");
    let permuter: ChannelRouter<8, 8> = ChannelRouter::permute([7, 6, 5, 4, 3, 2, 1, 0]);
    let permuted: Vec<[f32; 8]> = original.iter().map(|s| permuter.process(s)).collect();
    println!("   Input: [Ch0..Ch7] -> Output: [Ch7..Ch0]");

    // Demonstrate identity (pass-through)
    println!("\n3. Identity Router: Pass-through (useful as pipeline placeholder)");
    let identity: ChannelRouter<8, 8> = ChannelRouter::identity();
    let _identity_out: Vec<[f32; 8]> = original.iter().map(|s| identity.process(s)).collect();
    println!("   Input unchanged");

    // Bundle router demo data
    let router_data = RouterDemoData {
        original,
        selected,
        permuted,
    };

    // Generate router demonstration plot
    println!("\nGenerating ChannelRouter demonstration plot...");
    generate_router_plot(&router_data, "output/router_demo.png")?;
    println!("  Wrote: output/router_demo.png");

    // Summary
    println!("\n=== Demo Complete ===");
    println!("\nOutput files:");
    println!("  - output/spatial_demo.png  : CAR vs Laplacian comparison (3 panels)");
    println!("  - output/router_demo.png   : ChannelRouter demonstration (3 panels)");
    println!("  - output/spatial_*.csv     : Raw data for each processing stage");

    Ok(())
}

/// Generate synthetic 8-channel signal with common-mode artifacts.
///
/// Each channel has:
/// - A unique 10 Hz signal with channel-dependent amplitude
/// - Channel-specific white noise
///
/// All channels share:
/// - 50 Hz powerline interference
/// - Low-frequency drift
fn generate_multichannel_signal() -> Vec<[f32; CHANNELS]> {
    let mut signal = vec![[0.0f32; CHANNELS]; SAMPLES];

    // Generate per-channel unique signals
    // Needs the index for amplitude/phase computation, not just array access
    #[allow(clippy::needless_range_loop)]
    for ch in 0..CHANNELS {
        // Each channel has a 10 Hz signal with different amplitude and phase
        let amplitude = 0.5 + (ch as f32) * 0.1; // 0.5 to 1.2
        let phase = (ch as f32) * 0.4; // Phase offset per channel
        let unique_signal = common::sine_wave(SAMPLES, SAMPLE_RATE, 10.0, amplitude, phase);

        // Add channel-specific noise
        let noise = common::white_noise(SAMPLES, 0.15, 100 + ch as u64);

        for (i, sig) in signal.iter_mut().enumerate() {
            sig[ch] = unique_signal[i] + noise[i];
        }
    }

    // Add common-mode artifacts (same on all channels)
    let powerline = common::sine_wave(SAMPLES, SAMPLE_RATE, 50.0, 0.8, 0.0);
    let drift = common::sine_wave(SAMPLES, SAMPLE_RATE, 0.5, 0.5, 0.0);

    for (i, sig) in signal.iter_mut().enumerate() {
        let common_noise = powerline[i] + drift[i];
        for ch_val in sig.iter_mut() {
            *ch_val += common_noise;
        }
    }

    signal
}

/// Compute common-mode power reduction in dB for CAR and Laplacian.
fn compute_common_mode_reduction(data: &SpatialDemoData) -> (f32, f32) {
    // Common-mode component = mean across channels at each time point
    let compute_common_mode_power = |signal: &[[f32; CHANNELS]]| -> f32 {
        let mut power = 0.0f32;
        for sample in signal {
            let mean: f32 = sample.iter().sum::<f32>() / CHANNELS as f32;
            power += mean * mean;
        }
        power / signal.len() as f32
    };

    let raw_power = compute_common_mode_power(&data.raw);
    let car_power = compute_common_mode_power(&data.car);
    let lap_power = compute_common_mode_power(&data.laplacian);

    // Convert to dB reduction
    let car_reduction = 10.0 * (raw_power / car_power.max(1e-10)).log10();
    let lap_reduction = 10.0 * (raw_power / lap_power.max(1e-10)).log10();

    (car_reduction, lap_reduction)
}

/// Write multi-channel signal to CSV file.
fn write_spatial_csv(path: &str, data: &[[f32; CHANNELS]]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    // Header
    write!(file, "sample,time_ms")?;
    for ch in 0..CHANNELS {
        write!(file, ",ch{}", ch)?;
    }
    writeln!(file)?;

    // Data rows
    for (i, sample) in data.iter().enumerate() {
        let time_ms = (i as f32 / SAMPLE_RATE) * 1000.0;
        write!(file, "{},{:.3}", i, time_ms)?;
        for &val in sample.iter() {
            write!(file, ",{:.6}", val)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Generate 3-panel plot comparing raw, CAR, and Laplacian filtered signals.
fn generate_spatial_plot(data: &SpatialDemoData, output_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((3, 1));

    // Display settings: show first 500ms
    let display_samples = (SAMPLE_RATE * 0.5) as usize;

    // Color palette for channels
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

    // Panel 1: Raw signal (with common-mode noise)
    plot_multichannel(
        &areas[0],
        &data.raw,
        display_samples,
        "Raw Signal (with 50 Hz powerline + drift artifacts)",
        &colors,
    )?;

    // Panel 2: After CAR
    plot_multichannel(
        &areas[1],
        &data.car,
        display_samples,
        "After Common Average Reference (common-mode removed)",
        &colors,
    )?;

    // Panel 3: After Surface Laplacian
    plot_multichannel(
        &areas[2],
        &data.laplacian,
        display_samples,
        "After Surface Laplacian (spatially sharpened)",
        &colors,
    )?;

    root.present()?;
    Ok(())
}

/// Generate 3-panel plot demonstrating ChannelRouter operations.
fn generate_router_plot(data: &RouterDemoData, output_path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((3, 1));

    // Display settings: show first 300ms for clarity
    let display_samples = (SAMPLE_RATE * 0.3) as usize;

    // Color palette
    let colors_8 = [
        RGBColor(220, 50, 50),   // Ch0 - Red
        RGBColor(50, 150, 50),   // Ch1 - Green
        RGBColor(50, 100, 220),  // Ch2 - Blue
        RGBColor(220, 150, 50),  // Ch3 - Orange
        RGBColor(150, 50, 220),  // Ch4 - Purple
        RGBColor(50, 200, 200),  // Ch5 - Cyan
        RGBColor(220, 50, 150),  // Ch6 - Magenta
        RGBColor(100, 100, 100), // Ch7 - Gray
    ];

    // Colors for selected channels (0, 2, 4, 6)
    let colors_4 = [
        RGBColor(220, 50, 50),  // Ch0 - Red
        RGBColor(50, 100, 220), // Ch2 - Blue
        RGBColor(150, 50, 220), // Ch4 - Purple
        RGBColor(220, 50, 150), // Ch6 - Magenta
    ];

    // Panel 1: Original 8-channel signal
    plot_multichannel(
        &areas[0],
        &data.original,
        display_samples,
        "Original Signal (8 channels: Ch0-Ch7)",
        &colors_8,
    )?;

    // Panel 2: Selected subset (channels 0, 2, 4, 6)
    plot_multichannel_generic(
        &areas[1],
        &data.selected,
        display_samples,
        "Channel Selection: select([0, 2, 4, 6]) -> 4 channels",
        &colors_4,
        &["Ch0", "Ch2", "Ch4", "Ch6"],
    )?;

    // Panel 3: Permuted (reversed) order
    // Use reversed color order to show the permutation visually
    let colors_reversed = [
        RGBColor(100, 100, 100), // Ch7 -> position 0
        RGBColor(220, 50, 150),  // Ch6 -> position 1
        RGBColor(50, 200, 200),  // Ch5 -> position 2
        RGBColor(150, 50, 220),  // Ch4 -> position 3
        RGBColor(220, 150, 50),  // Ch3 -> position 4
        RGBColor(50, 100, 220),  // Ch2 -> position 5
        RGBColor(50, 150, 50),   // Ch1 -> position 6
        RGBColor(220, 50, 50),   // Ch0 -> position 7
    ];

    plot_multichannel_with_labels(
        &areas[2],
        &data.permuted,
        display_samples,
        "Channel Permutation: permute([7,6,5,4,3,2,1,0]) - reversed order",
        &colors_reversed,
        &["Ch7", "Ch6", "Ch5", "Ch4", "Ch3", "Ch2", "Ch1", "Ch0"],
    )?;

    root.present()?;
    Ok(())
}

/// Plot multi-channel signal with stacked traces.
fn plot_multichannel<const C: usize>(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[[f32; C]],
    display_samples: usize,
    title: &str,
    colors: &[RGBColor],
) -> Result<(), Box<dyn Error>> {
    let labels: Vec<String> = (0..C).map(|i| format!("Ch{}", i)).collect();
    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
    plot_multichannel_generic(area, data, display_samples, title, colors, &label_refs)
}

/// Plot multi-channel signal with custom labels.
fn plot_multichannel_with_labels<const C: usize>(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[[f32; C]],
    display_samples: usize,
    title: &str,
    colors: &[RGBColor],
    labels: &[&str],
) -> Result<(), Box<dyn Error>> {
    plot_multichannel_generic(area, data, display_samples, title, colors, labels)
}

/// Generic multi-channel plotter.
fn plot_multichannel_generic<const C: usize>(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[[f32; C]],
    display_samples: usize,
    title: &str,
    colors: &[RGBColor],
    labels: &[&str],
) -> Result<(), Box<dyn Error>> {
    let display_samples = display_samples.min(data.len());
    let display_time_ms = (display_samples as f32 / SAMPLE_RATE) * 1000.0;

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
    for ch in 0..C {
        let color = colors[ch % colors.len()];
        let label = if ch < labels.len() {
            labels[ch].to_string()
        } else {
            format!("Ch{}", ch)
        };

        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                    (t, data[i][ch])
                }),
                ShapeStyle::from(&color).stroke_width(1),
            ))?
            .label(label)
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
