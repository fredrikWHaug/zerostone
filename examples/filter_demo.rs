//! Filter demonstration with visualization.
//!
//! This example demonstrates IIR filtering on synthetic signals and generates
//! PNG plots showing the effect of filtering.
//!
//! Run with: `cargo run --example filter_demo`
//!
//! Output: `output/filter_demo.png` and `output/filter_demo.csv`

mod common;

use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use zerostone::{BiquadCoeffs, IirFilter};

const SAMPLE_RATE: f32 = 1000.0; // 1 kHz
const DURATION_SECS: f32 = 1.0;
const SAMPLES: usize = (SAMPLE_RATE * DURATION_SECS) as usize;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Zerostone Filter Demo ===\n");

    // Generate test signal: 10 Hz sine + 60 Hz noise + white noise
    println!("Generating test signal...");
    println!("  - 10 Hz sine wave (amplitude 1.0) - the 'signal'");
    println!("  - 60 Hz sine wave (amplitude 0.3) - simulated powerline interference");
    println!("  - White noise (amplitude 0.2)");

    let signal_10hz = common::sine_wave(SAMPLES, SAMPLE_RATE, 10.0, 1.0, 0.0);
    let noise_60hz = common::sine_wave(SAMPLES, SAMPLE_RATE, 60.0, 0.3, 0.0);
    let white_noise = common::white_noise(SAMPLES, 0.2, 42);

    let input: Vec<f32> = signal_10hz
        .iter()
        .zip(noise_60hz.iter())
        .zip(white_noise.iter())
        .map(|((&s, &n60), &wn)| s + n60 + wn)
        .collect();

    // Create lowpass filter at 30 Hz (will remove 60 Hz interference)
    println!("\nApplying 4th-order Butterworth lowpass filter (30 Hz cutoff)...");
    let coeffs = BiquadCoeffs::butterworth_lowpass(SAMPLE_RATE, 30.0);
    let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]); // 2 sections = 4th order

    let filtered: Vec<f32> = input.iter().map(|&x| filter.process_sample(x)).collect();

    // Calculate statistics (skip first 100ms for filter settling)
    // Using standard SNR formula: SNR(dB) = 10 * log10(signal_power / noise_power)
    let skip_samples = (SAMPLE_RATE * 0.1) as usize;
    let n = (input.len() - skip_samples) as f32;

    // Signal power (from pure 10 Hz reference)
    let signal_power: f32 = signal_10hz[skip_samples..]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        / n;

    // Noise power before filtering: noise = input - pure_signal
    let noise_before_power: f32 = input[skip_samples..]
        .iter()
        .zip(&signal_10hz[skip_samples..])
        .map(|(inp, sig)| {
            let noise = inp - sig;
            noise * noise
        })
        .sum::<f32>()
        / n;

    // Noise power after filtering: noise = filtered - pure_signal
    let noise_after_power: f32 = filtered[skip_samples..]
        .iter()
        .zip(&signal_10hz[skip_samples..])
        .map(|(filt, sig)| {
            let noise = filt - sig;
            noise * noise
        })
        .sum::<f32>()
        / n;

    // SNR in dB
    let snr_before = 10.0 * (signal_power / noise_before_power).log10();
    let snr_after = 10.0 * (signal_power / noise_after_power).log10();
    let snr_improvement = snr_after - snr_before;

    // Correlation with original signal (secondary metric)
    let correlation = {
        let mean_orig: f32 = signal_10hz[skip_samples..].iter().sum::<f32>() / n;
        let mean_filt: f32 = filtered[skip_samples..].iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_orig = 0.0f32;
        let mut var_filt = 0.0f32;

        for i in skip_samples..signal_10hz.len() {
            let o = signal_10hz[i] - mean_orig;
            let f = filtered[i] - mean_filt;
            cov += o * f;
            var_orig += o * o;
            var_filt += f * f;
        }

        cov / (var_orig.sqrt() * var_filt.sqrt())
    };

    println!("\nSignal Statistics (after 100ms settling):");
    println!("  SNR before filtering:     {:.1} dB", snr_before);
    println!("  SNR after filtering:      {:.1} dB", snr_after);
    println!("  SNR improvement:          {:.1} dB", snr_improvement);
    println!(
        "  Correlation with pure:    {:.4} (1.0 = perfect)",
        correlation
    );
    println!();
    println!("Note: Negative SNR improvement is expected - IIR filters introduce phase");
    println!("shift that appears as 'error' vs the pure reference. The 60Hz interference");
    println!("IS removed (see plot). Use filtfilt or linear-phase FIR for zero distortion.");

    // Write CSV
    println!("\nWriting CSV to output/filter_demo.csv...");
    write_csv(&input, &filtered, &signal_10hz)?;

    // Generate plot
    println!("Generating plot to output/filter_demo.png...");
    generate_plot(&input, &filtered, &signal_10hz)?;

    println!("\nDone! Open output/filter_demo.png to see the results.");
    println!("Open output/filter_demo.csv to inspect the raw data.");

    Ok(())
}

fn write_csv(input: &[f32], filtered: &[f32], original: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create("output/filter_demo.csv")?;
    writeln!(file, "sample,time_ms,input,filtered,original_10hz")?;

    for i in 0..input.len() {
        let time_ms = (i as f32 / SAMPLE_RATE) * 1000.0;
        writeln!(
            file,
            "{},{:.3},{:.6},{:.6},{:.6}",
            i, time_ms, input[i], filtered[i], original[i]
        )?;
    }

    Ok(())
}

fn generate_plot(input: &[f32], filtered: &[f32], original: &[f32]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("output/filter_demo.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 1));

    // Plot 1: Full signal comparison (first 500ms)
    let display_samples = (SAMPLE_RATE * 0.5) as usize; // 500ms

    {
        let mut chart = ChartBuilder::on(&areas[0])
            .caption(
                "IIR Lowpass Filter Demo (30 Hz cutoff)",
                ("sans-serif", 24).into_font(),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..500f32, -2.0f32..2.0f32)?;

        chart
            .configure_mesh()
            .x_desc("Time (ms)")
            .y_desc("Amplitude")
            .draw()?;

        // Input signal (noisy) - light gray
        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                    (t, input[i])
                }),
                ShapeStyle::from(&RGBColor(200, 200, 200)).stroke_width(1),
            ))?
            .label("Input (noisy)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(200, 200, 200)));

        // Filtered signal - blue
        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                    (t, filtered[i])
                }),
                ShapeStyle::from(&BLUE).stroke_width(2),
            ))?
            .label("Filtered (30 Hz LP)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Original pure signal - green dashed
        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                    (t, original[i])
                }),
                ShapeStyle::from(&GREEN).stroke_width(1),
            ))?
            .label("Original 10 Hz (reference)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    // Plot 2: Zoomed view (first 200ms)
    let zoom_samples = (SAMPLE_RATE * 0.2) as usize;

    {
        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Zoomed View (first 200ms)", ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..200f32, -2.0f32..2.0f32)?;

        chart
            .configure_mesh()
            .x_desc("Time (ms)")
            .y_desc("Amplitude")
            .draw()?;

        // Input signal
        chart.draw_series(LineSeries::new(
            (0..zoom_samples).map(|i| {
                let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                (t, input[i])
            }),
            ShapeStyle::from(&RGBColor(200, 200, 200)).stroke_width(1),
        ))?;

        // Filtered signal
        chart.draw_series(LineSeries::new(
            (0..zoom_samples).map(|i| {
                let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                (t, filtered[i])
            }),
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?;

        // Original
        chart.draw_series(LineSeries::new(
            (0..zoom_samples).map(|i| {
                let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                (t, original[i])
            }),
            ShapeStyle::from(&GREEN).stroke_width(2),
        ))?;
    }

    root.present()?;
    Ok(())
}
