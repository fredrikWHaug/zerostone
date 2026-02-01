//! Filter demonstration with visualization.
//!
//! This example demonstrates IIR filtering on synthetic signals and generates
//! PNG plots showing the effect of filtering in both time and frequency domain.
//!
//! Run with: `cargo run --example filter_demo [OPTIONS]`
//!
//! Options:
//!   -c, --cutoff <HZ>          Filter cutoff frequency (default: 30.0)
//!   -t, --filter-type <TYPE>   Filter type: lowpass, highpass, bandpass (default: lowpass)
//!   -o, --output <FILE>        Output PNG filename (default: output/filter_demo.png)
//!
//! Output: PNG plot and CSV data file

mod common;

use clap::Parser;
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use zerostone::{BiquadCoeffs, IirFilter};

const SAMPLE_RATE: f32 = 1000.0; // 1 kHz
const DURATION_SECS: f32 = 1.0;
const SAMPLES: usize = (SAMPLE_RATE * DURATION_SECS) as usize;

/// Data bundle for plotting and CSV export
struct FilterDemoData {
    input: Vec<f32>,
    filtered: Vec<f32>,
    reference: Vec<f32>,
    freq_input: Vec<f32>,
    psd_input: Vec<f32>,
    psd_filtered: Vec<f32>,
    freq_response: Vec<f32>,
    mag_response: Vec<f32>,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Filter cutoff frequency in Hz
    #[arg(short, long, default_value_t = 30.0)]
    cutoff: f32,

    /// Filter type: lowpass, highpass, or bandpass
    #[arg(short = 't', long, default_value = "lowpass")]
    filter_type: String,

    /// Output PNG filename
    #[arg(short, long, default_value = "output/filter_demo.png")]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("=== Zerostone Filter Demo ===\n");
    println!("Filter type: {}", args.filter_type);
    println!("Cutoff frequency: {:.1} Hz", args.cutoff);
    println!();

    // Generate test signal based on filter type
    let (input, reference, _description) = match args.filter_type.as_str() {
        "lowpass" => {
            println!("Generating test signal for lowpass demo...");
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

            (input, signal_10hz, "10 Hz signal + 60 Hz noise")
        }
        "highpass" => {
            println!("Generating test signal for highpass demo...");
            println!("  - 5 Hz low-frequency drift (amplitude 0.5)");
            println!("  - 60 Hz sine wave (amplitude 1.0) - the 'signal'");
            println!("  - White noise (amplitude 0.2)");

            let drift = common::sine_wave(SAMPLES, SAMPLE_RATE, 5.0, 0.5, 0.0);
            let signal_60hz = common::sine_wave(SAMPLES, SAMPLE_RATE, 60.0, 1.0, 0.0);
            let white_noise = common::white_noise(SAMPLES, 0.2, 42);

            let input: Vec<f32> = drift
                .iter()
                .zip(signal_60hz.iter())
                .zip(white_noise.iter())
                .map(|((&d, &s), &wn)| d + s + wn)
                .collect();

            (input, signal_60hz, "60 Hz signal + low-freq drift")
        }
        "bandpass" => {
            println!("Generating test signal for bandpass demo...");
            println!("  - 5 Hz low-frequency component (amplitude 0.3)");
            println!("  - 20 Hz target signal (amplitude 1.0)");
            println!("  - 60 Hz high-frequency noise (amplitude 0.3)");
            println!("  - White noise (amplitude 0.2)");

            let low_freq = common::sine_wave(SAMPLES, SAMPLE_RATE, 5.0, 0.3, 0.0);
            let signal_20hz = common::sine_wave(SAMPLES, SAMPLE_RATE, 20.0, 1.0, 0.0);
            let high_freq = common::sine_wave(SAMPLES, SAMPLE_RATE, 60.0, 0.3, 0.0);
            let white_noise = common::white_noise(SAMPLES, 0.2, 42);

            let input: Vec<f32> = low_freq
                .iter()
                .zip(signal_20hz.iter())
                .zip(high_freq.iter())
                .zip(white_noise.iter())
                .map(|(((&lf, &s), &hf), &wn)| lf + s + hf + wn)
                .collect();

            (input, signal_20hz, "Multi-frequency signal")
        }
        _ => {
            eprintln!("Error: Unknown filter type '{}'", args.filter_type);
            eprintln!("Valid options: lowpass, highpass, bandpass");
            std::process::exit(1);
        }
    };

    // Create filter based on type
    let (coeffs, filter_description) = match args.filter_type.as_str() {
        "lowpass" => {
            let coeffs = BiquadCoeffs::butterworth_lowpass(SAMPLE_RATE, args.cutoff);
            (
                coeffs,
                format!("4th-order Butterworth lowpass ({:.1} Hz)", args.cutoff),
            )
        }
        "highpass" => {
            let coeffs = BiquadCoeffs::butterworth_highpass(SAMPLE_RATE, args.cutoff);
            (
                coeffs,
                format!("4th-order Butterworth highpass ({:.1} Hz)", args.cutoff),
            )
        }
        "bandpass" => {
            // For bandpass, use cutoff as center, with ±5 Hz bandwidth
            let low = args.cutoff - 5.0;
            let high = args.cutoff + 5.0;
            let coeffs = BiquadCoeffs::butterworth_bandpass(SAMPLE_RATE, low, high);
            (
                coeffs,
                format!("4th-order Butterworth bandpass ({:.1}-{:.1} Hz)", low, high),
            )
        }
        _ => unreachable!(),
    };

    println!("\nApplying {}...", filter_description);
    let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]); // 2 sections = 4th order

    let filtered: Vec<f32> = input.iter().map(|&x| filter.process_sample(x)).collect();

    // Calculate statistics (skip first 100ms for filter settling)
    let skip_samples = (SAMPLE_RATE * 0.1) as usize;
    let n = (input.len() - skip_samples) as f32;

    // Signal power (from pure reference)
    let signal_power: f32 = reference[skip_samples..].iter().map(|x| x * x).sum::<f32>() / n;

    // Noise power before filtering: noise = input - pure_signal
    let noise_before_power: f32 = input[skip_samples..]
        .iter()
        .zip(&reference[skip_samples..])
        .map(|(inp, sig)| {
            let noise = inp - sig;
            noise * noise
        })
        .sum::<f32>()
        / n;

    // Noise power after filtering: noise = filtered - pure_signal
    let noise_after_power: f32 = filtered[skip_samples..]
        .iter()
        .zip(&reference[skip_samples..])
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

    // Correlation with original signal
    let correlation = {
        let mean_orig: f32 = reference[skip_samples..].iter().sum::<f32>() / n;
        let mean_filt: f32 = filtered[skip_samples..].iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_orig = 0.0f32;
        let mut var_filt = 0.0f32;

        for i in skip_samples..reference.len() {
            let o = reference[i] - mean_orig;
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

    // Compute frequency domain data
    println!("\nComputing frequency domain analysis...");
    let (freq_input, psd_input) = common::compute_power_spectrum(&input, SAMPLE_RATE);
    let (_freq_filtered, psd_filtered) = common::compute_power_spectrum(&filtered, SAMPLE_RATE);
    let (freq_response, mag_response, _phase_response) =
        common::compute_filter_response(&[coeffs, coeffs], SAMPLE_RATE, 512);

    let data = FilterDemoData {
        input,
        filtered,
        reference,
        freq_input,
        psd_input,
        psd_filtered,
        freq_response,
        mag_response,
    };

    // Write CSV
    let csv_path = args.output.replace(".png", ".csv");
    println!("\nWriting CSV to {}...", csv_path);
    write_csv(&data, &csv_path)?;

    // Generate plot
    println!("Generating plot to {}...", args.output);
    generate_plot(&data, args.cutoff, &filter_description, &args.output)?;

    println!("\nDone! Open {} to see the results.", args.output);
    println!("Open {} to inspect the raw data.", csv_path);

    Ok(())
}

fn write_csv(data: &FilterDemoData, path: &str) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    // Time domain section
    writeln!(file, "# Time Domain")?;
    writeln!(file, "sample,time_ms,input,filtered,reference")?;
    for i in 0..data.input.len() {
        let time_ms = (i as f32 / SAMPLE_RATE) * 1000.0;
        writeln!(
            file,
            "{},{:.3},{:.6},{:.6},{:.6}",
            i, time_ms, data.input[i], data.filtered[i], data.reference[i]
        )?;
    }

    // Frequency domain section
    writeln!(file, "\n# Frequency Domain - Input Signal PSD")?;
    writeln!(file, "frequency_hz,power_db")?;
    for i in 0..data.freq_input.len() {
        writeln!(file, "{:.3},{:.6}", data.freq_input[i], data.psd_input[i])?;
    }

    writeln!(file, "\n# Frequency Domain - Filtered Signal PSD")?;
    writeln!(file, "frequency_hz,power_db")?;
    for i in 0..data.psd_filtered.len() {
        writeln!(
            file,
            "{:.3},{:.6}",
            data.freq_input[i], data.psd_filtered[i]
        )?;
    }

    // Filter response section
    writeln!(file, "\n# Filter Frequency Response")?;
    writeln!(file, "frequency_hz,magnitude_db")?;
    for i in 0..data.freq_response.len() {
        writeln!(
            file,
            "{:.3},{:.6}",
            data.freq_response[i], data.mag_response[i]
        )?;
    }

    Ok(())
}

fn generate_plot(
    data: &FilterDemoData,
    cutoff: f32,
    filter_description: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1400, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((3, 1));

    // Panel 1: Time domain comparison (first 500ms)
    let display_samples = (SAMPLE_RATE * 0.5) as usize;

    {
        let mut chart = ChartBuilder::on(&areas[0])
            .caption(
                format!("Time Domain - {}", filter_description),
                ("sans-serif", 22).into_font(),
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
                    (t, data.input[i])
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
                    (t, data.filtered[i])
                }),
                ShapeStyle::from(&BLUE).stroke_width(2),
            ))?
            .label("Filtered")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Reference signal - green
        chart
            .draw_series(LineSeries::new(
                (0..display_samples).map(|i| {
                    let t = (i as f32 / SAMPLE_RATE) * 1000.0;
                    (t, data.reference[i])
                }),
                ShapeStyle::from(&GREEN).stroke_width(1),
            ))?
            .label("Reference (clean)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    // Panel 2: Frequency domain PSD overlay
    {
        let max_freq = 100.0; // Focus on 0-100 Hz range

        // Find data range for y-axis
        let max_idx = data
            .freq_input
            .iter()
            .position(|&f| f > max_freq)
            .unwrap_or(data.freq_input.len());
        let y_min = data.psd_input[..max_idx]
            .iter()
            .chain(&data.psd_filtered[..max_idx])
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .floor()
            - 10.0;
        let y_max = data.psd_input[..max_idx]
            .iter()
            .chain(&data.psd_filtered[..max_idx])
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            + 10.0;

        let mut chart = ChartBuilder::on(&areas[1])
            .caption(
                "Frequency Domain - Power Spectral Density",
                ("sans-serif", 22).into_font(),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..max_freq, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Frequency (Hz)")
            .y_desc("Power (dB)")
            .draw()?;

        // Input PSD - gray with transparency
        chart
            .draw_series(LineSeries::new(
                data.freq_input
                    .iter()
                    .zip(data.psd_input.iter())
                    .filter(|(f, _)| **f <= max_freq)
                    .map(|(f, p)| (*f, *p)),
                ShapeStyle::from(&RGBColor(150, 150, 150)).stroke_width(2),
            ))?
            .label("Input PSD")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(150, 150, 150)));

        // Filtered PSD - blue
        chart
            .draw_series(LineSeries::new(
                data.freq_input
                    .iter()
                    .zip(data.psd_filtered.iter())
                    .filter(|(f, _)| **f <= max_freq)
                    .map(|(f, p)| (*f, *p)),
                ShapeStyle::from(&BLUE).stroke_width(2),
            ))?
            .label("Filtered PSD")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Cutoff frequency marker (dashed vertical line)
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(cutoff, y_min), (cutoff, y_max)],
            ShapeStyle::from(&RED).stroke_width(2),
        )))?;

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    // Panel 3: Filter frequency response
    {
        let max_freq = 100.0;

        // Find data range
        let max_idx = data
            .freq_response
            .iter()
            .position(|&f| f > max_freq)
            .unwrap_or(data.freq_response.len());
        let y_min = data.mag_response[..max_idx]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .floor()
            - 10.0;
        let y_max = 10.0; // 0 dB at top

        let mut chart = ChartBuilder::on(&areas[2])
            .caption("Filter Frequency Response", ("sans-serif", 22).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..max_freq, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Frequency (Hz)")
            .y_desc("Magnitude (dB)")
            .draw()?;

        // Filter magnitude response - red
        chart
            .draw_series(LineSeries::new(
                data.freq_response
                    .iter()
                    .zip(data.mag_response.iter())
                    .filter(|(f, _)| **f <= max_freq)
                    .map(|(f, m)| (*f, *m)),
                ShapeStyle::from(&RED).stroke_width(3),
            ))?
            .label("Filter Response")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        // -3dB line (passband edge reference)
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0, -3.0), (max_freq, -3.0)],
            ShapeStyle::from(&BLACK).stroke_width(1),
        )))?;

        // Cutoff frequency marker
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(cutoff, y_min), (cutoff, y_max)],
            ShapeStyle::from(&RGBColor(100, 100, 100)).stroke_width(1),
        )))?;

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}
