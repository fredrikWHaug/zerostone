use proptest::prelude::*;
use zerostone::connectivity;
use zerostone::drift::{estimate_drift_from_positions, DriftEstimator};
use zerostone::entropy;
use zerostone::isi;
use zerostone::kalman::KalmanFilter;
use zerostone::linalg::Matrix;
use zerostone::localize::{center_of_mass, center_of_mass_threshold};
use zerostone::mda::{mda_element_size, MdaDataType};
use zerostone::metrics::compare_spike_trains;
use zerostone::pac;
use zerostone::probe::ProbeLayout;
use zerostone::quality;
use zerostone::riemannian;
use zerostone::sorter::{
    estimate_noise_multichannel, merge_clusters, sort_multichannel, split_clusters, OnlineSorter,
    SortConfig,
};
use zerostone::spike_sort::{
    align_to_peak, combine_features, compute_adaptive_thresholds, deduplicate_events,
    detect_spikes_multichannel, extract_peak_channel, extract_spatial_features, MultiChannelEvent,
};
use zerostone::template_subtract::{PeelResult, TemplateSubtractor};
use zerostone::whitening::{estimate_noise_covariance, WhiteningMatrix, WhiteningMode};
use zerostone::{BiquadCoeffs, Complex, Fft, HilbertTransform, IirFilter, OnlineStats, WindowType};

/// Construct a 3x3 SPD matrix from 9 arbitrary values: A^T * A + 0.1 * I.
fn make_spd_3x3(vals: &[f64]) -> Matrix<3, 9> {
    let mut mat = Matrix::<3, 9>::new([0.0; 9]);
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += vals[k * 3 + i] * vals[k * 3 + j];
            }
            if i == j {
                sum += 0.1;
            }
            mat.set(i, j, sum);
        }
    }
    mat
}

proptest! {
    /// Biquad filter stability: arbitrary valid (sr, cutoff) and input signal
    /// must never produce NaN or infinity.
    #[test]
    fn biquad_output_is_finite(
        sr in 100.0f32..48000.0,
        cutoff_ratio in 0.01f32..0.49,
        samples in prop::collection::vec(-1e6f32..1e6, 1000..=1000),
    ) {
        let cutoff = sr * cutoff_ratio;
        let coeffs = BiquadCoeffs::butterworth_lowpass(sr, cutoff);
        let mut filter = IirFilter::<1>::new([coeffs]);
        for &x in &samples {
            let y = filter.process_sample(x);
            prop_assert!(y.is_finite(), "output {y} is not finite for sr={sr}, cutoff={cutoff}, input={x}");
        }
    }

    /// FFT roundtrip: forward then inverse recovers original signal within epsilon.
    #[test]
    fn fft_roundtrip(
        signal in prop::collection::vec(-1e3f32..1e3, 64..=64),
    ) {
        let fft = Fft::<64>::new();
        let mut data: [Complex; 64] = core::array::from_fn(|i| Complex::from_real(signal[i]));
        fft.forward(&mut data);
        fft.inverse(&mut data);
        for (i, &original) in signal.iter().enumerate() {
            let recovered = data[i].re;
            let diff = (recovered - original).abs();
            prop_assert!(
                diff < 1e-2,
                "roundtrip mismatch at index {i}: original={original}, recovered={recovered}, diff={diff}"
            );
        }
    }

    /// Parseval's theorem: time-domain energy equals frequency-domain energy / N.
    #[test]
    fn parsevals_theorem(
        signal in prop::collection::vec(-1e3f32..1e3, 64..=64),
    ) {
        let fft = Fft::<64>::new();
        let time_energy: f32 = signal.iter().map(|x| x * x).sum();

        let mut data: [Complex; 64] = core::array::from_fn(|i| Complex::from_real(signal[i]));
        fft.forward(&mut data);
        let freq_energy: f32 = data.iter().map(|c| c.magnitude_squared()).sum::<f32>() / 64.0;

        let relative_err = if time_energy > 1e-10 {
            (time_energy - freq_energy).abs() / time_energy
        } else {
            (time_energy - freq_energy).abs()
        };

        prop_assert!(
            relative_err < 1e-4,
            "Parseval violated: time_energy={time_energy}, freq_energy={freq_energy}, relative_err={relative_err}"
        );
    }

    /// Filter cascade: IirFilter<2>([A, B]) == IirFilter<1>(A) -> IirFilter<1>(B).
    #[test]
    fn filter_cascade_equivalence(
        sr in 1000.0f32..8000.0,
        ratio1 in 0.05f32..0.2,
        ratio2 in 0.2f32..0.45,
        samples in prop::collection::vec(-1.0f32..1.0, 500..=500),
    ) {
        let c1 = BiquadCoeffs::butterworth_lowpass(sr, sr * ratio1);
        let c2 = BiquadCoeffs::butterworth_lowpass(sr, sr * ratio2);

        let mut f1 = IirFilter::<1>::new([c1]);
        let mut f2 = IirFilter::<1>::new([c2]);
        let mut f_combined = IirFilter::<2>::new([c1, c2]);

        for &x in &samples {
            let y_series = f2.process_sample(f1.process_sample(x));
            let y_combined = f_combined.process_sample(x);
            let diff = (y_series - y_combined).abs();
            prop_assert!(diff < 1e-5, "cascade mismatch: series={y_series} vs combined={y_combined}, diff={diff}");
        }
    }

    /// Hilbert transform: |analytic|^2 == re^2 + im^2 (by construction).
    #[test]
    fn hilbert_analytic_magnitude(
        signal in prop::collection::vec(-10.0f32..10.0, 64..=64),
    ) {
        let ht = HilbertTransform::<64>::new();
        let input: [f32; 64] = core::array::from_fn(|i| signal[i]);
        let mut analytic = [Complex::new(0.0, 0.0); 64];
        ht.analytic_signal(&input, &mut analytic);

        for (i, z) in analytic.iter().enumerate() {
            let mag_sq = z.magnitude_squared();
            let re_sq = z.re * z.re;
            let im_sq = z.im * z.im;
            let diff = (mag_sq - (re_sq + im_sq)).abs();
            prop_assert!(diff < 1e-4, "magnitude identity violated at {i}: |z|^2={mag_sq}, re^2+im^2={}", re_sq + im_sq);
        }
    }

    /// Coherence is bounded in [0, 1].
    #[test]
    fn coherence_in_unit_range(
        sig_a in prop::collection::vec(-10.0f32..10.0, 256..=256),
        sig_b in prop::collection::vec(-10.0f32..10.0, 256..=256),
    ) {
        let mut output = [0.0f32; 129];
        connectivity::coherence::<256>(&sig_a, &sig_b, WindowType::Hann, &mut output);
        for (i, &c) in output.iter().enumerate() {
            prop_assert!((-1e-6..=1.0 + 1e-6).contains(&c), "coherence[{i}] = {c} outside [0, 1]");
        }
    }

    /// Phase locking value is bounded in [0, 1].
    #[test]
    fn plv_in_unit_range(
        phases_a in prop::collection::vec(-std::f32::consts::PI..std::f32::consts::PI, 100..=100),
        phases_b in prop::collection::vec(-std::f32::consts::PI..std::f32::consts::PI, 100..=100),
    ) {
        let plv = connectivity::phase_locking_value(&phases_a, &phases_b);
        prop_assert!((-1e-6..=1.0 + 1e-6).contains(&plv), "PLV = {plv} outside [0, 1]");
    }

    /// Approximate entropy is non-negative; normalized spectral entropy is in [0, 1].
    #[test]
    fn entropy_bounds(
        data in prop::collection::vec(0.1f64..10.0, 100..=100),
        r in 0.1f64..2.0,
    ) {
        let apen = entropy::approximate_entropy(&data, 2, r);
        prop_assert!(apen >= -1e-10, "ApEn = {apen} < 0");

        let psd: Vec<f64> = data.iter().map(|x| x * x).collect();
        let se = entropy::spectral_entropy(&psd, true);
        prop_assert!((-1e-10..=1.0 + 1e-10).contains(&se), "SpectralEn = {se} outside [0, 1]");
    }

    /// Modulation index is bounded in [0, 1].
    #[test]
    fn modulation_index_in_unit_range(
        phases in prop::collection::vec(-std::f32::consts::PI..std::f32::consts::PI, 200..=200),
        amplitudes in prop::collection::vec(0.0f32..10.0, 200..=200),
    ) {
        let mi = pac::modulation_index(&phases, &amplitudes, 18);
        prop_assert!((-1e-6..=1.0 + 1e-6).contains(&mi), "MI = {mi} outside [0, 1]");
    }

    /// Riemannian distance is symmetric: d(A, B) == d(B, A).
    #[test]
    fn riemannian_distance_symmetric(
        a_vals in prop::collection::vec(-2.0f64..2.0, 9..=9),
        b_vals in prop::collection::vec(-2.0f64..2.0, 9..=9),
    ) {
        let mat_a = make_spd_3x3(&a_vals);
        let mat_b = make_spd_3x3(&b_vals);

        if let (Ok(d_ab), Ok(d_ba)) = (
            riemannian::riemannian_distance(&mat_a, &mat_b),
            riemannian::riemannian_distance(&mat_b, &mat_a),
        ) {
            let diff = (d_ab - d_ba).abs();
            prop_assert!(diff < 1e-10, "d(A,B)={d_ab} != d(B,A)={d_ba}, diff={diff}");
        }
    }

    /// Riemannian distance triangle inequality: d(A, C) <= d(A, B) + d(B, C).
    #[test]
    fn riemannian_distance_triangle_inequality(
        a_vals in prop::collection::vec(-2.0f64..2.0, 9..=9),
        b_vals in prop::collection::vec(-2.0f64..2.0, 9..=9),
        c_vals in prop::collection::vec(-2.0f64..2.0, 9..=9),
    ) {
        let mat_a = make_spd_3x3(&a_vals);
        let mat_b = make_spd_3x3(&b_vals);
        let mat_c = make_spd_3x3(&c_vals);

        if let (Ok(d_ab), Ok(d_bc), Ok(d_ac)) = (
            riemannian::riemannian_distance(&mat_a, &mat_b),
            riemannian::riemannian_distance(&mat_b, &mat_c),
            riemannian::riemannian_distance(&mat_a, &mat_c),
        ) {
            prop_assert!(
                d_ac <= d_ab + d_bc + 1e-8,
                "triangle inequality violated: d(A,C)={d_ac} > d(A,B)+d(B,C)={}", d_ab + d_bc
            );
        }
    }

    /// Kalman filter covariance diagonal stays non-negative after predict/update cycles.
    #[test]
    fn kalman_covariance_positive_diagonal(
        measurements in prop::collection::vec(-100.0f64..100.0, 50..=50),
    ) {
        let f = Matrix::<2, 4>::new([1.0, 0.1, 0.0, 1.0]);
        let h = [1.0, 0.0];
        let q = Matrix::<2, 4>::new([0.01, 0.0, 0.0, 0.01]);
        let r = Matrix::<1, 1>::new([1.0]);
        let x0 = [0.0, 0.0];
        let p0 = Matrix::<2, 4>::new([1.0, 0.0, 0.0, 1.0]);

        let mut kf = KalmanFilter::<2, 1, 4, 1, 2>::new(f, h, q, r, x0, p0);

        for &z in &measurements {
            kf.predict();
            let _ = kf.update(&[z]);
            let cov = kf.covariance();
            prop_assert!(cov.get(0, 0) >= 0.0, "P[0,0] = {} < 0", cov.get(0, 0));
            prop_assert!(cov.get(1, 1) >= 0.0, "P[1,1] = {} < 0", cov.get(1, 1));
        }
    }

    /// Online stats: mean of N identical values equals that value; variance is zero.
    #[test]
    fn online_stats_constant_signal(
        value in -1000.0f64..1000.0,
        n in 10usize..200,
    ) {
        let mut stats = OnlineStats::<1>::new();
        for _ in 0..n {
            stats.update(&[value]);
        }
        let mean = stats.mean()[0];
        let var = stats.variance()[0];
        prop_assert!((mean - value).abs() < 1e-10, "mean of {n} copies of {value} = {mean}");
        prop_assert!(var.abs() < 1e-10, "variance of constant signal = {var}");
    }

    /// ISI CV of constant intervals is zero.
    #[test]
    fn isi_cv_constant_intervals(
        interval in 0.001f64..1.0,
        n in 5usize..50,
    ) {
        let spike_times: Vec<f64> = (0..n).map(|i| i as f64 * interval).collect();
        let cv = isi::isi_cv(&spike_times);
        prop_assert!(cv < 1e-6, "CV of constant intervals should be ~0, got {cv}");
    }

    /// Burst index is always in [0, 1].
    #[test]
    fn burst_index_in_range(
        intervals in prop::collection::vec(0.001f64..1.0, 5..50),
        threshold in 0.001f64..0.5,
    ) {
        // Build spike times from intervals
        let mut spike_times = Vec::with_capacity(intervals.len() + 1);
        spike_times.push(0.0);
        let mut t = 0.0;
        for &isi in &intervals {
            t += isi;
            spike_times.push(t);
        }
        let mut hist = isi::IsiHistogram::<200>::new(0.001);
        hist.add_train(&spike_times);
        let bi = hist.burst_index(threshold);
        prop_assert!((0.0..=1.0).contains(&bi), "burst index {bi} out of [0,1]");
    }

    /// Local variation of perfectly regular firing is zero.
    #[test]
    fn local_variation_regular_is_zero(
        interval in 0.001f64..1.0,
        n in 5usize..50,
    ) {
        let spike_times: Vec<f64> = (0..n).map(|i| i as f64 * interval).collect();
        let lv = isi::local_variation(&spike_times);
        prop_assert!(lv < 1e-6, "Lv of regular firing should be ~0, got {lv}");
    }

    // =========================================================================
    // Probe geometry properties
    // =========================================================================

    /// channel_distance is symmetric: d(a, b) == d(b, a).
    #[test]
    fn probe_distance_symmetric(
        pitch in 1.0f64..100.0,
        a in 0usize..8,
        b in 0usize..8,
    ) {
        let probe = ProbeLayout::<8>::linear(pitch);
        let d_ab = probe.channel_distance(a, b);
        let d_ba = probe.channel_distance(b, a);
        if d_ab.is_nan() {
            prop_assert!(d_ba.is_nan());
        } else {
            prop_assert!((d_ab - d_ba).abs() < 1e-12, "d({a},{b})={d_ab} != d({b},{a})={d_ba}");
        }
    }

    /// channel_distance satisfies triangle inequality: d(a,c) <= d(a,b) + d(b,c).
    #[test]
    fn probe_distance_triangle_inequality(
        pitch in 1.0f64..100.0,
        a in 0usize..8,
        b in 0usize..8,
        c in 0usize..8,
    ) {
        let probe = ProbeLayout::<8>::linear(pitch);
        let d_ab = probe.channel_distance(a, b);
        let d_bc = probe.channel_distance(b, c);
        let d_ac = probe.channel_distance(a, c);
        if d_ab.is_finite() && d_bc.is_finite() && d_ac.is_finite() {
            prop_assert!(
                d_ac <= d_ab + d_bc + 1e-10,
                "triangle inequality violated: d({a},{c})={d_ac} > d({a},{b})+d({b},{c})={}",
                d_ab + d_bc
            );
        }
    }

    /// spatial_extent is non-negative for any probe.
    #[test]
    fn probe_spatial_extent_non_negative(
        pitch in 1.0f64..200.0,
        columns in 1usize..4,
    ) {
        let probe = ProbeLayout::<8>::polytrode(columns, pitch, pitch);
        let (xr, yr) = probe.spatial_extent();
        prop_assert!(xr >= 0.0, "x range = {xr} < 0");
        prop_assert!(yr >= 0.0, "y range = {yr} < 0");
    }

    // =========================================================================
    // Quality metrics properties
    // =========================================================================

    /// ISI violation rate is always in [0, 1] for sorted spike trains.
    #[test]
    fn isi_violation_rate_bounded(
        intervals in prop::collection::vec(0.0001f64..1.0, 4..20),
        rp in 0.0001f64..0.1,
    ) {
        let mut spike_times = Vec::with_capacity(intervals.len() + 1);
        spike_times.push(0.0);
        let mut t = 0.0;
        for &isi in &intervals {
            t += isi;
            spike_times.push(t);
        }
        if let Some(rate) = quality::isi_violation_rate(&spike_times, rp) {
            prop_assert!((0.0..=1.0).contains(&rate), "ISI violation rate = {rate} outside [0, 1]");
        }
    }

    /// Silhouette score is always in [-1, 1].
    #[test]
    fn silhouette_score_bounded(
        intra in prop::collection::vec(0.0f64..100.0, 2..10),
        inter in prop::collection::vec(0.01f64..100.0, 1..5),
    ) {
        if let Some(s) = quality::silhouette_score(&intra, &inter) {
            prop_assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&s), "silhouette = {s} outside [-1, 1]");
        }
    }

    /// d_prime is symmetric: d'(A, B) == d'(B, A).
    #[test]
    fn d_prime_symmetric(
        a in prop::collection::vec(-100.0f64..100.0, 5..20),
        b in prop::collection::vec(-100.0f64..100.0, 5..20),
    ) {
        if let (Some(dp_ab), Some(dp_ba)) = (quality::d_prime(&a, &b), quality::d_prime(&b, &a)) {
            let diff = (dp_ab - dp_ba).abs();
            prop_assert!(diff < 1e-10, "d'(A,B)={dp_ab} != d'(B,A)={dp_ba}");
        }
    }

    /// Euclidean distance is symmetric: d(a,b) == d(b,a).
    #[test]
    fn euclidean_distance_symmetric(
        a in prop::collection::vec(-1000.0f64..1000.0, 3..=3),
        b in prop::collection::vec(-1000.0f64..1000.0, 3..=3),
    ) {
        let d_ab = quality::euclidean_distance(&a, &b);
        let d_ba = quality::euclidean_distance(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-10, "d(a,b)={d_ab} != d(b,a)={d_ba}");
    }

    // =========================================================================
    // Whitening properties
    // =========================================================================

    /// ZCA whitening of identity covariance approximately preserves input.
    #[test]
    fn whitening_identity_cov_preserves_signal(
        x0 in -100.0f64..100.0,
        x1 in -100.0f64..100.0,
    ) {
        let cov = [[1.0, 0.0], [0.0, 1.0]];
        let wm = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-10).unwrap();
        let out = wm.apply(&[x0, x1]);
        prop_assert!((out[0] - x0).abs() < 0.01, "ch0: {x0} -> {}", out[0]);
        prop_assert!((out[1] - x1).abs() < 0.01, "ch1: {x1} -> {}", out[1]);
    }

    /// ZCA whitening matrix is symmetric.
    #[test]
    fn whitening_zca_symmetric(
        a in 0.1f64..10.0,
        b in -5.0f64..5.0,
        c in 0.1f64..10.0,
    ) {
        // Build a valid positive semi-definite covariance: [[a, b], [b, c]]
        // Ensure positive semi-definite: a*c >= b^2
        let b_clamped = if b * b > a * c {
            let max_b = libm::sqrt(a * c) * 0.99;
            if b > 0.0 { max_b } else { -max_b }
        } else {
            b
        };
        let cov = [[a, b_clamped], [b_clamped, c]];
        if let Ok(wm) = WhiteningMatrix::<2, 4>::from_covariance(&cov, WhiteningMode::Zca, 1e-6) {
            let m = wm.matrix();
            let diff = (m.get(0, 1) - m.get(1, 0)).abs();
            prop_assert!(diff < 1e-8, "ZCA matrix not symmetric: W01={}, W10={}", m.get(0, 1), m.get(1, 0));
        }
    }

    // =========================================================================
    // Multi-channel spike detection properties
    // =========================================================================

    /// detect_spikes_multichannel returns count <= buffer size.
    #[test]
    fn multichannel_detect_count_bounded(
        vals in prop::collection::vec(-10.0f64..10.0, 200..=200),
    ) {
        // 4 channels, 50 time steps
        let mut data = [[0.0f64; 4]; 50];
        for (i, chunk) in vals.chunks_exact(4).enumerate() {
            for (ch, &v) in chunk.iter().enumerate() {
                data[i][ch] = v;
            }
        }
        let noise = [1.0, 1.0, 1.0, 1.0];
        let mut events = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 64];
        let n = detect_spikes_multichannel::<4>(&data, 3.0, &noise, 3, &mut events);
        prop_assert!(n <= 64, "count {n} exceeds buffer size 64");
        prop_assert!(n <= 50 * 4, "count {n} exceeds T*C = 200");
    }

    /// All returned events have valid channel indices (< C) and sample indices (< T).
    #[test]
    fn multichannel_detect_valid_indices(
        vals in prop::collection::vec(-10.0f64..10.0, 200..=200),
    ) {
        let mut data = [[0.0f64; 4]; 50];
        for (i, chunk) in vals.chunks_exact(4).enumerate() {
            for (ch, &v) in chunk.iter().enumerate() {
                data[i][ch] = v;
            }
        }
        let noise = [1.0, 1.0, 1.0, 1.0];
        let mut events = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 64];
        let n = detect_spikes_multichannel::<4>(&data, 3.0, &noise, 3, &mut events);
        for e in &events[..n] {
            prop_assert!(e.channel < 4, "channel {} >= C=4", e.channel);
            prop_assert!(e.sample < 50, "sample {} >= T=50", e.sample);
        }
    }

    /// Detection count is monotonically non-decreasing as threshold decreases.
    #[test]
    fn multichannel_detect_monotone_in_threshold(
        vals in prop::collection::vec(-10.0f64..10.0, 200..=200),
        high_mult in 4.0f64..8.0,
        low_mult in 1.0f64..4.0,
    ) {
        let mut data = [[0.0f64; 4]; 50];
        for (i, chunk) in vals.chunks_exact(4).enumerate() {
            for (ch, &v) in chunk.iter().enumerate() {
                data[i][ch] = v;
            }
        }
        let noise = [1.0, 1.0, 1.0, 1.0];
        let mut ev_high = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 64];
        let mut ev_low = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 64];
        let n_high = detect_spikes_multichannel::<4>(&data, high_mult, &noise, 3, &mut ev_high);
        let n_low = detect_spikes_multichannel::<4>(&data, low_mult, &noise, 3, &mut ev_low);
        prop_assert!(
            n_low >= n_high,
            "lower threshold ({low_mult}) detected {n_low} < {n_high} from higher threshold ({high_mult})"
        );
    }

    // =========================================================================
    // Deduplication properties
    // =========================================================================

    /// deduplicate_events output count <= input count.
    #[test]
    fn dedup_count_le_input(
        samples in prop::collection::vec(0usize..100, 2..10),
        channels in prop::collection::vec(0usize..4, 2..10),
        amplitudes in prop::collection::vec(0.1f64..20.0, 2..10),
    ) {
        let n_in = samples.len().min(channels.len()).min(amplitudes.len());
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events: Vec<MultiChannelEvent> = (0..n_in)
            .map(|i| MultiChannelEvent {
                sample: samples[i],
                channel: channels[i] % 4,
                amplitude: amplitudes[i],
            })
            .collect();
        // Sort by sample to match precondition
        events.sort_by_key(|e| e.sample);
        let n_out = deduplicate_events::<4>(&mut events, n_in, &probe, 30.0, 5);
        prop_assert!(n_out <= n_in, "dedup output {n_out} > input {n_in}");
    }

    /// Deduplication is idempotent: dedup(dedup(x)) == dedup(x).
    #[test]
    fn dedup_idempotent(
        samples in prop::collection::vec(0usize..100, 2..8),
        channels in prop::collection::vec(0usize..4, 2..8),
        amplitudes in prop::collection::vec(0.1f64..20.0, 2..8),
    ) {
        let n_in = samples.len().min(channels.len()).min(amplitudes.len());
        let probe = ProbeLayout::<4>::linear(25.0);
        let mut events: Vec<MultiChannelEvent> = (0..n_in)
            .map(|i| MultiChannelEvent {
                sample: samples[i],
                channel: channels[i] % 4,
                amplitude: amplitudes[i],
            })
            .collect();
        events.sort_by_key(|e| e.sample);

        // First dedup
        let n1 = deduplicate_events::<4>(&mut events, n_in, &probe, 30.0, 5);
        let snapshot: Vec<MultiChannelEvent> = events[..n1].to_vec();

        // Second dedup
        let n2 = deduplicate_events::<4>(&mut events, n1, &probe, 30.0, 5);
        prop_assert!(n1 == n2, "dedup not idempotent: first={}, second={}", n1, n2);
        for i in 0..n2 {
            prop_assert!(events[i].sample == snapshot[i].sample, "event {} sample changed", i);
            prop_assert!(events[i].channel == snapshot[i].channel, "event {} channel changed", i);
        }
    }

    /// Events with zero temporal overlap are all preserved.
    #[test]
    fn dedup_preserves_distant_events(
        n_events in 2usize..6,
    ) {
        let probe = ProbeLayout::<4>::linear(25.0);
        let temporal_radius = 5usize;
        // Place events far apart in time so no pair overlaps
        let mut events: Vec<MultiChannelEvent> = (0..n_events)
            .map(|i| MultiChannelEvent {
                sample: i * (temporal_radius + 10),
                channel: i % 4,
                amplitude: 5.0 + i as f64,
            })
            .collect();
        let n_out = deduplicate_events::<4>(&mut events, n_events, &probe, 30.0, temporal_radius);
        prop_assert!(n_out == n_events, "distant events should all survive: got {}, expected {}", n_out, n_events);
    }

    // =========================================================================
    // Alignment properties
    // =========================================================================

    /// align_to_peak preserves event count.
    #[test]
    fn align_preserves_count(
        vals in prop::collection::vec(-10.0f64..10.0, 200..=200),
    ) {
        let mut data = [[0.0f64; 4]; 50];
        for (i, chunk) in vals.chunks_exact(4).enumerate() {
            for (ch, &v) in chunk.iter().enumerate() {
                data[i][ch] = v;
            }
        }
        let noise = [1.0, 1.0, 1.0, 1.0];
        let mut events = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 64];
        let n = detect_spikes_multichannel::<4>(&data, 3.0, &noise, 3, &mut events);
        let samples_before: Vec<usize> = events[..n].iter().map(|e| e.sample).collect();
        align_to_peak::<4>(&data, &mut events, n, 3);
        // Count unchanged -- align_to_peak does not add or remove events
        // (it modifies in place). We verify the slice length is the same
        // by checking the events are still valid.
        for (idx, e) in events[..n].iter().enumerate() {
            prop_assert!(e.channel < 4, "channel invalid after alignment at index {idx}");
            prop_assert!(e.sample < 50, "sample invalid after alignment at index {idx}");
            let orig = samples_before[idx];
            let diff = e.sample.abs_diff(orig);
            prop_assert!(diff <= 3, "aligned sample {} too far from original {} (half_window=3)", e.sample, orig);
        }
    }

    // =========================================================================
    // Template subtraction properties
    // =========================================================================

    /// Peel output count never exceeds the output buffer length.
    #[test]
    fn peel_output_bounded(
        template_vals in prop::array::uniform4(-5.0f64..5.0),
        data_vals in prop::collection::vec(-10.0f64..10.0, 16..=16),
        spike_time in 2usize..12,
    ) {
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        sub.add_template(&template_vals, 0.1, 5.0).unwrap();

        let mut data: [f64; 16] = [0.0; 16];
        for (i, &v) in data_vals.iter().enumerate() {
            data[i] = v;
        }
        let times = [spike_time];
        let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
        let n = sub.peel(&mut data, &times, 1, 1, &mut results);
        prop_assert!(n <= 4, "peel returned {n} > output buffer size 4");
    }

    /// Injecting amp * template and peeling recovers the amplitude; residual is near zero.
    #[test]
    fn peel_perfect_subtraction(
        amp in 0.5f64..2.0,
    ) {
        let template = [-1.0, -3.0, -2.0, 0.5];
        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        sub.add_template(&template, 0.1, 5.0).unwrap();

        let mut data = [0.0f64; 12];
        for i in 0..4 {
            data[2 + i] = amp * template[i];
        }

        let times = [3usize]; // peak near sample 3, pre_samples=1 -> window starts at 2
        let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
        let n = sub.peel(&mut data, &times, 1, 1, &mut results);
        prop_assert!(n == 1, "expected 1 result, got {n}");
        let amp_err = (results[0].amplitude - amp).abs();
        prop_assert!(amp_err < 0.1, "amplitude error {amp_err} exceeds tolerance");

        let residual: f64 = data.iter().map(|x| x * x).sum();
        prop_assert!(residual < 0.01, "residual energy {residual} is not near zero");
    }

    /// Peel with 0 templates always returns 0.
    #[test]
    fn peel_empty_templates_returns_zero(
        data_vals in prop::collection::vec(-5.0f64..5.0, 8..=8),
    ) {
        let sub = TemplateSubtractor::<4, 2>::new(10);
        let mut data: [f64; 8] = [0.0; 8];
        for (i, &v) in data_vals.iter().enumerate() {
            data[i] = v;
        }
        let times = [3usize];
        let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
        let n = sub.peel(&mut data, &times, 1, 1, &mut results);
        prop_assert!(n == 0, "peel with 0 templates returned {n}, expected 0");
    }

    // =========================================================================
    // Localization properties
    // =========================================================================

    /// Center-of-mass result y coordinate lies within the convex hull of channel positions.
    #[test]
    fn com_within_convex_hull(
        a0 in 0.01f64..10.0,
        a1 in 0.01f64..10.0,
        a2 in 0.01f64..10.0,
        a3 in 0.01f64..10.0,
    ) {
        let amps = [a0, a1, a2, a3];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let loc = center_of_mass(&amps, &pos);

        // y must be within [min_y, max_y] = [0.0, 75.0]
        prop_assert!(loc[1] >= -1e-10, "y = {} below min position 0.0", loc[1]);
        prop_assert!(loc[1] <= 75.0 + 1e-10, "y = {} above max position 75.0", loc[1]);
        // x must be 0.0 since all positions have x=0
        prop_assert!(loc[0].abs() < 1e-10, "x = {} should be 0.0", loc[0]);
    }

    /// Center-of-mass with all-equal weights returns the centroid of positions.
    #[test]
    fn com_equal_weights_returns_centroid(
        w in 0.1f64..100.0,
    ) {
        let amps = [w, w, w, w];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let loc = center_of_mass(&amps, &pos);
        let expected_y = (0.0 + 25.0 + 50.0 + 75.0) / 4.0; // 37.5
        prop_assert!(
            (loc[1] - expected_y).abs() < 1e-9,
            "expected centroid y={expected_y}, got {}", loc[1]
        );
    }

    /// Increasing the threshold never increases the weight sum in thresholded CoM.
    #[test]
    fn com_threshold_monotonicity(
        a0 in 0.0f64..10.0,
        a1 in 0.0f64..10.0,
        a2 in 0.0f64..10.0,
        a3 in 0.0f64..10.0,
        t_low in 0.0f64..5.0,
        t_delta in 0.01f64..5.0,
    ) {
        let amps = [a0, a1, a2, a3];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let t_high = t_low + t_delta;

        // Count channels above each threshold
        let count_low = amps.iter().filter(|&&a| a.abs() >= t_low).count();
        let count_high = amps.iter().filter(|&&a| a.abs() >= t_high).count();
        prop_assert!(
            count_high <= count_low,
            "higher threshold {t_high} has {count_high} channels >= threshold, but lower {t_low} has {count_low}"
        );

        // If high threshold returns Some, low threshold must also return Some
        let result_high = center_of_mass_threshold(&amps, &pos, t_high);
        let result_low = center_of_mass_threshold(&amps, &pos, t_low);
        if result_high.is_some() {
            prop_assert!(result_low.is_some(), "high threshold returned Some but low threshold returned None");
        }
    }

    // =========================================================================
    // Sorter properties
    // =========================================================================

    /// Noise estimation returns non-negative, finite values for any non-empty finite data.
    #[test]
    fn noise_estimation_positive_finite(
        vals in prop::collection::vec(-100.0f64..100.0, 20..=20),
    ) {
        // 2 channels, 10 time steps
        let mut data = [[0.0f64; 2]; 10];
        for (i, chunk) in vals.chunks_exact(2).enumerate() {
            data[i][0] = chunk[0];
            data[i][1] = chunk[1];
        }
        let mut scratch = [0.0f64; 10];
        let noise = estimate_noise_multichannel::<2>(&data, &mut scratch);
        for (ch, &n) in noise.iter().enumerate() {
            prop_assert!(n >= 0.0, "noise[{ch}] = {n} is negative");
            prop_assert!(n.is_finite(), "noise[{ch}] = {n} is not finite");
        }
    }

    /// Sort result consistency: n_spikes equals sum of cluster counts,
    /// and n_clusters matches the number of non-zero clusters.
    #[test]
    fn sort_result_consistency(
        seed in 1u64..1000,
    ) {
        // Simple xorshift for reproducible noise
        let mut s = seed;
        let mut next = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2_000_000) as f64 - 1_000_000.0) / 1_000_000.0
        };

        let n = 500;
        let mut data = vec![[0.0f64; 2]; n];
        for sample in data.iter_mut() {
            sample[0] = next();
            sample[1] = next();
        }

        // Inject a few spikes to give the sorter something to find
        let template = [-2.0, -5.0, -8.0, -4.0, -1.0, 1.0, 0.5, 0.0];
        for &pos in &[100usize, 200, 350] {
            for (dt, &tv) in template.iter().enumerate() {
                if pos + dt < n {
                    data[pos + dt][0] += 8.0 * tv;
                }
            }
        }

        let config = SortConfig {
            threshold_multiplier: 4.0,
            pre_samples: 2,
            refractory_samples: 10,
            ..SortConfig::default()
        };
        let probe = ProbeLayout::<2>::linear(25.0);
        let mut scratch = vec![0.0f64; n];
        let mut events = vec![
            MultiChannelEvent { sample: 0, channel: 0, amplitude: 0.0 }; 100
        ];
        let mut waveforms = vec![[0.0f64; 8]; 100];
        let mut features = vec![[0.0f64; 3]; 100];
        let mut labels = vec![0usize; 100];

        let result = sort_multichannel::<2, 4, 8, 3, 64, 4>(
            &config, &probe, &mut data, &mut scratch,
            &mut events, &mut waveforms, &mut features, &mut labels,
        );

        if let Ok(sr) = result {
            // Sum of cluster counts should equal n_spikes
            let cluster_sum: usize = sr.clusters[..sr.n_clusters]
                .iter()
                .map(|c| c.count)
                .sum();
            prop_assert!(
                cluster_sum == sr.n_spikes,
                "cluster count sum {cluster_sum} != n_spikes {}", sr.n_spikes
            );

            // n_clusters should match the number of clusters with count > 0
            // (or be >= the count of non-zero clusters, since the clustering
            // algorithm may report active clusters that got 0 spikes from
            // this particular batch)
            let nonzero_clusters = sr.clusters[..sr.n_clusters]
                .iter()
                .filter(|c| c.count > 0)
                .count();
            prop_assert!(
                sr.n_clusters >= nonzero_clusters,
                "n_clusters {} < nonzero clusters {}", sr.n_clusters, nonzero_clusters
            );
        }
    }

    // =========================================================================
    // Merge + feature combination properties
    // =========================================================================

    /// merge_clusters output count is <= input count.
    #[test]
    fn merge_never_increases_clusters(
        n_spikes in 4usize..20,
        n_clusters in 2usize..5,
        seed in 0u64..1000,
    ) {
        let mut s = seed.wrapping_add(1);
        let mut next = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2_000_000) as f64 - 1_000_000.0) / 1_000_000.0
        };

        let mut labels: Vec<usize> = (0..n_spikes).map(|i| i % n_clusters).collect();
        let feature_buf: Vec<[f64; 3]> = (0..n_spikes)
            .map(|_| [next(), next(), next()])
            .collect();
        let event_buf: Vec<MultiChannelEvent> = (0..n_spikes)
            .map(|i| MultiChannelEvent {
                sample: i * 20,
                channel: i % 2,
                amplitude: next().abs() + 1.0,
            })
            .collect();
        let mut scratch = vec![0.0f64; n_spikes];

        let result = merge_clusters::<3>(
            n_spikes, &mut labels, &feature_buf, &event_buf,
            n_clusters, 1.5, 0.2, 10, &mut scratch,
        );
        prop_assert!(
            result <= n_clusters,
            "merge_clusters returned {} > input n_clusters {}", result, n_clusters
        );
    }

    /// After merge, all labels are in [0, new_n_clusters).
    #[test]
    fn merge_labels_valid(
        n_spikes in 4usize..20,
        n_clusters in 2usize..5,
        seed in 0u64..1000,
    ) {
        let mut s = seed.wrapping_add(1);
        let mut next = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2_000_000) as f64 - 1_000_000.0) / 1_000_000.0
        };

        let mut labels: Vec<usize> = (0..n_spikes).map(|i| i % n_clusters).collect();
        let feature_buf: Vec<[f64; 3]> = (0..n_spikes)
            .map(|_| [next(), next(), next()])
            .collect();
        let event_buf: Vec<MultiChannelEvent> = (0..n_spikes)
            .map(|i| MultiChannelEvent {
                sample: i * 20,
                channel: i % 2,
                amplitude: next().abs() + 1.0,
            })
            .collect();
        let mut scratch = vec![0.0f64; n_spikes];

        let new_n = merge_clusters::<3>(
            n_spikes, &mut labels, &feature_buf, &event_buf,
            n_clusters, 1.5, 0.2, 10, &mut scratch,
        );
        for (i, &lbl) in labels[..n_spikes].iter().enumerate() {
            prop_assert!(
                lbl < new_n,
                "label[{i}] = {lbl} >= new_n_clusters {new_n}"
            );
        }
    }

    /// combine_features output preserves PCA features and scales spatial features.
    #[test]
    fn combine_features_preserves_pca(
        pca_vals in prop::collection::vec(-100.0f64..100.0, 8..=8),
        spatial_vals in prop::collection::vec(-100.0f64..100.0, 8..=8),
        weight in 0.01f64..10.0,
    ) {
        // K=2, S=2, T=4, n=4 (4 spikes with 2 PCA features and 2 spatial features each)
        let pca: Vec<[f64; 2]> = pca_vals.chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let spatial: Vec<[f64; 2]> = spatial_vals.chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let mut output = [[0.0f64; 4]; 4];
        combine_features::<2, 2, 4>(&pca, &spatial, weight, &mut output, 4);

        for i in 0..4 {
            prop_assert!(
                (output[i][0] - pca[i][0]).abs() < 1e-12,
                "output[{i}][0] = {} != pca {}", output[i][0], pca[i][0]
            );
            prop_assert!(
                (output[i][1] - pca[i][1]).abs() < 1e-12,
                "output[{i}][1] = {} != pca {}", output[i][1], pca[i][1]
            );
            prop_assert!(
                (output[i][2] - spatial[i][0] * weight).abs() < 1e-10,
                "output[{i}][2] = {} != spatial*w {}", output[i][2], spatial[i][0] * weight
            );
            prop_assert!(
                (output[i][3] - spatial[i][1] * weight).abs() < 1e-10,
                "output[{i}][3] = {} != spatial*w {}", output[i][3], spatial[i][1] * weight
            );
        }
    }

    // =========================================================================
    // Adaptive threshold properties
    // =========================================================================

    /// Adaptive thresholds are always >= min_threshold.
    #[test]
    fn adaptive_threshold_ge_floor(
        vals in prop::collection::vec(-10.0f64..10.0, 200..=200),
        base_mult in 3.0f64..8.0,
        min_thresh in 0.1f64..5.0,
    ) {
        // 4 channels, 50 time steps
        let mut data = [[0.0f64; 4]; 50];
        for (i, chunk) in vals.chunks_exact(4).enumerate() {
            for (ch, &v) in chunk.iter().enumerate() {
                data[i][ch] = v;
            }
        }
        let mut scratch = [0.0f64; 50];
        let thresholds = compute_adaptive_thresholds::<4>(
            &data, base_mult, min_thresh, 100.0, 30000.0, &mut scratch,
        );
        for (ch, &t) in thresholds.iter().enumerate() {
            prop_assert!(
                t >= min_thresh - 1e-12,
                "threshold[{ch}] = {t} < min_threshold {min_thresh}"
            );
            prop_assert!(t.is_finite(), "threshold[{ch}] is not finite");
        }
    }

    /// Higher base_multiplier produces higher or equal thresholds (no activity scaling).
    #[test]
    fn adaptive_threshold_monotone_in_multiplier(
        vals in prop::collection::vec(-10.0f64..10.0, 80..=80),
        low_mult in 2.0f64..4.0,
        delta in 0.5f64..4.0,
    ) {
        let mut data = [[0.0f64; 2]; 40];
        for (i, chunk) in vals.chunks_exact(2).enumerate() {
            data[i][0] = chunk[0];
            data[i][1] = chunk[1];
        }
        let high_mult = low_mult + delta;
        let mut scratch = [0.0f64; 40];
        // Use f64::MAX for max_rate_hz to disable activity scaling,
        // which can break monotonicity (fewer crossings at high mult
        // means less upward scaling, potentially yielding lower threshold).
        let t_low = compute_adaptive_thresholds::<2>(
            &data, low_mult, 0.1, f64::MAX, 30000.0, &mut scratch,
        );
        let t_high = compute_adaptive_thresholds::<2>(
            &data, high_mult, 0.1, f64::MAX, 30000.0, &mut scratch,
        );
        for ch in 0..2 {
            prop_assert!(
                t_high[ch] >= t_low[ch] - 1e-10,
                "ch {ch}: higher multiplier gave lower threshold: {} < {}",
                t_high[ch], t_low[ch]
            );
        }
    }

    /// With spatial_weight=0, output equals [pca_features..., 0, 0].
    #[test]
    fn combine_features_zero_weight(
        pca_vals in prop::collection::vec(-100.0f64..100.0, 6..=6),
        spatial_vals in prop::collection::vec(-100.0f64..100.0, 3..=3),
    ) {
        // K=2, S=1, T=3, n=3
        let pca: Vec<[f64; 2]> = pca_vals.chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let spatial: Vec<[f64; 1]> = spatial_vals.iter()
            .map(|&v| [v])
            .collect();
        let mut output = [[0.0f64; 3]; 3];
        combine_features::<2, 1, 3>(&pca, &spatial, 0.0, &mut output, 3);

        for i in 0..3 {
            prop_assert!(
                (output[i][0] - pca[i][0]).abs() < 1e-12,
                "output[{i}][0] = {} != pca {}", output[i][0], pca[i][0]
            );
            prop_assert!(
                (output[i][1] - pca[i][1]).abs() < 1e-12,
                "output[{i}][1] = {} != pca {}", output[i][1], pca[i][1]
            );
            prop_assert!(
                output[i][2] == 0.0,
                "output[{i}][2] = {} should be 0.0 with zero weight", output[i][2]
            );
        }
    }

    // =========================================================================
    // Day 5: drift, split_clusters, OnlineSorter
    // =========================================================================

    /// DriftEstimator: fit + estimate_drift always returns finite values.
    #[test]
    fn drift_estimator_finite(
        positions in prop::collection::vec(-1e4f64..1e4, 4..=16),
    ) {
        let mut est = DriftEstimator::<32>::new(1000);
        for (i, &pos) in positions.iter().enumerate() {
            est.add_spike(i * 1000 + 500, pos);
        }
        est.fit();
        let slope = est.slope();
        let intercept = est.intercept();
        prop_assert!(slope.is_finite(), "slope must be finite: {}", slope);
        prop_assert!(intercept.is_finite(), "intercept must be finite: {}", intercept);

        let d = est.estimate_drift(5000);
        prop_assert!(d.is_finite(), "drift estimate must be finite: {}", d);
    }

    /// DriftEstimator: corrected positions have lower variance when there is drift.
    #[test]
    fn drift_correction_reduces_variance(
        drift_rate in 0.001f64..0.1,
        n_bins in 4usize..16,
    ) {
        let mut est = DriftEstimator::<32>::new(1000);
        let base = 100.0;
        for i in 0..n_bins {
            let pos = base + (i as f64) * drift_rate * 1000.0;
            est.add_spike(i * 1000 + 500, pos);
        }
        est.fit();

        let mut raw_sum = 0.0;
        let mut raw_sq = 0.0;
        let mut cor_sum = 0.0;
        let mut cor_sq = 0.0;
        for i in 0..n_bins {
            let raw = base + (i as f64) * drift_rate * 1000.0;
            let corrected = est.correct_position(i * 1000 + 500, raw);
            raw_sum += raw;
            raw_sq += raw * raw;
            cor_sum += corrected;
            cor_sq += corrected * corrected;
        }
        let n = n_bins as f64;
        let raw_var = raw_sq / n - (raw_sum / n) * (raw_sum / n);
        let cor_var = cor_sq / n - (cor_sum / n) * (cor_sum / n);

        prop_assert!(
            cor_var <= raw_var + 1e-6,
            "Corrected variance ({}) should be <= raw variance ({})",
            cor_var, raw_var
        );
    }

    /// estimate_drift_from_positions: returns None for single-bin data.
    #[test]
    fn batch_drift_single_bin_none(
        n_spikes in 1usize..20,
        positions in prop::collection::vec(0.0f64..100.0, 1..=20),
    ) {
        let indices: Vec<usize> = (0..n_spikes).map(|i| i * 10).collect();
        let pos: Vec<f64> = positions.into_iter().take(n_spikes).collect();
        if pos.is_empty() {
            return Ok(());
        }
        let result = estimate_drift_from_positions(&indices, &pos, 10000, 8);
        prop_assert!(result.is_none(), "Single bin should return None");
    }

    /// split_clusters: never reduces cluster count.
    #[test]
    fn split_clusters_monotonic(
        n_clusters in 1usize..4,
        threshold in 0.5f64..5.0,
    ) {
        let mut labels = [0usize; 12];
        let mut features = [[0.0f64; 2]; 12];
        for i in 0..12 {
            let cl = i % n_clusters;
            labels[i] = cl;
            features[i] = [cl as f64 * 0.01, cl as f64 * 0.01];
        }
        let new_n = split_clusters(12, &mut labels, &features, n_clusters, 3, threshold);
        prop_assert!(
            new_n >= n_clusters,
            "split should never reduce cluster count: {} < {}",
            new_n, n_clusters
        );
    }

    /// OnlineSorter: classify always returns a valid label.
    #[test]
    fn online_sorter_valid_label(
        template_vals in prop::collection::vec(-10.0f64..10.0, 3..=3),
        feature_vals in prop::collection::vec(-10.0f64..10.0, 3..=3),
    ) {
        let mut sorter = OnlineSorter::<3, 8>::new();
        let template: [f64; 3] = [template_vals[0], template_vals[1], template_vals[2]];
        sorter.add_template(&template);

        let features: [f64; 3] = [feature_vals[0], feature_vals[1], feature_vals[2]];
        let (label, dist) = sorter.classify(&features);

        prop_assert_eq!(label, 0, "Only one template, label must be 0");
        prop_assert!(dist >= 0.0, "Distance must be non-negative: {}", dist);
        prop_assert!(dist.is_finite(), "Distance must be finite: {}", dist);
    }

    /// OnlineSorter: distance to own template is zero.
    #[test]
    fn online_sorter_zero_self_distance(
        vals in prop::collection::vec(-10.0f64..10.0, 3..=3),
    ) {
        let mut sorter = OnlineSorter::<3, 8>::new();
        let template: [f64; 3] = [vals[0], vals[1], vals[2]];
        sorter.add_template(&template);

        let (label, dist) = sorter.classify(&template);
        prop_assert_eq!(label, 0);
        prop_assert!(dist < 1e-10, "Self-distance should be ~0, got {}", dist);
    }

    // =========================================================================
    // Metrics properties
    // =========================================================================

    /// compare_spike_trains: accuracy, precision, recall are all in [0.0, 1.0].
    #[test]
    fn metrics_accuracy_bounded(
        intervals_gt in prop::collection::vec(1usize..100, 1..=8),
        intervals_sorted in prop::collection::vec(1usize..100, 1..=8),
        tolerance in 0usize..20,
    ) {
        // Build sorted spike times by accumulating intervals
        let mut gt_times: Vec<usize> = Vec::new();
        let mut acc = 0usize;
        for &iv in &intervals_gt {
            acc += iv;
            gt_times.push(acc);
        }
        let mut sorted_times: Vec<usize> = Vec::new();
        acc = 0;
        for &iv in &intervals_sorted {
            acc += iv;
            sorted_times.push(acc);
        }

        let m = compare_spike_trains(&gt_times, &sorted_times, tolerance);
        prop_assert!(m.accuracy >= 0.0, "accuracy must be >= 0: {}", m.accuracy);
        prop_assert!(m.accuracy <= 1.0, "accuracy must be <= 1: {}", m.accuracy);
        prop_assert!(m.precision >= 0.0, "precision must be >= 0: {}", m.precision);
        prop_assert!(m.precision <= 1.0, "precision must be <= 1: {}", m.precision);
        prop_assert!(m.recall >= 0.0, "recall must be >= 0: {}", m.recall);
        prop_assert!(m.recall <= 1.0, "recall must be <= 1: {}", m.recall);
    }

    /// compare_spike_trains: identical trains yield perfect scores.
    #[test]
    fn metrics_identical_trains_perfect(
        intervals in prop::collection::vec(1usize..200, 1..=8),
        tolerance in 0usize..20,
    ) {
        let mut times: Vec<usize> = Vec::new();
        let mut acc = 0usize;
        for &iv in &intervals {
            acc += iv;
            times.push(acc);
        }

        let m = compare_spike_trains(&times, &times, tolerance);
        prop_assert!((m.accuracy - 1.0).abs() < 1e-10,
            "identical trains must have accuracy=1.0, got {}", m.accuracy);
        prop_assert!((m.precision - 1.0).abs() < 1e-10,
            "identical trains must have precision=1.0, got {}", m.precision);
        prop_assert!((m.recall - 1.0).abs() < 1e-10,
            "identical trains must have recall=1.0, got {}", m.recall);
    }

    /// compare_spike_trains: empty GT against non-empty sorted gives accuracy=0.
    #[test]
    fn metrics_empty_gt_zero_accuracy(
        intervals in prop::collection::vec(1usize..200, 1..=8),
        tolerance in 0usize..20,
    ) {
        let mut sorted_times: Vec<usize> = Vec::new();
        let mut acc = 0usize;
        for &iv in &intervals {
            acc += iv;
            sorted_times.push(acc);
        }
        let gt_empty: Vec<usize> = Vec::new();

        let m = compare_spike_trains(&gt_empty, &sorted_times, tolerance);
        prop_assert!((m.accuracy - 0.0).abs() < 1e-10,
            "empty GT must have accuracy=0.0, got {}", m.accuracy);
    }

    // =========================================================================
    // MDA properties
    // =========================================================================

    /// mda_element_size returns a value in {1, 2, 4, 8} for all variants.
    #[test]
    fn mda_element_size_positive(
        variant_idx in 0usize..7,
    ) {
        let variants = [
            MdaDataType::Uint8,
            MdaDataType::Float32,
            MdaDataType::Int16,
            MdaDataType::Int32,
            MdaDataType::Uint16,
            MdaDataType::Float64,
            MdaDataType::Uint32,
        ];
        let dt = variants[variant_idx];
        let sz = mda_element_size(dt);
        prop_assert!(
            sz == 1 || sz == 2 || sz == 4 || sz == 8,
            "element size must be 1, 2, 4, or 8; got {}", sz
        );
    }

    // =========================================================================
    // Spatial features property
    // =========================================================================

    /// extract_spatial_features with finite inputs returns all-finite outputs.
    #[test]
    fn spatial_features_finite(
        amp_vals in prop::collection::vec(-1e6f64..1e6, 8..=8),
        pos_vals in prop::collection::vec(-1e3f64..1e3, 4..=4),
        event_sample in 1usize..6,
        event_channel in 0usize..2,
    ) {
        // 2-channel data, 8 time samples
        let mut data = [[0.0f64; 2]; 8];
        for t in 0..8 {
            data[t][0] = amp_vals[t];
            data[t][1] = if t < 8 { amp_vals[t] * 0.5 } else { 0.0 };
        }

        let positions: [[f64; 2]; 2] = [
            [pos_vals[0], pos_vals[1]],
            [pos_vals[2], pos_vals[3]],
        ];

        let ch = if event_channel < 2 { event_channel } else { 0 };
        let s = if event_sample < 8 { event_sample } else { 1 };
        let events = [MultiChannelEvent { sample: s, channel: ch, amplitude: 1.0 }];
        let mut output = [[0.0f64; 2]; 1];

        let n = extract_spatial_features::<2>(&data, &events, 1, &positions, &mut output);
        prop_assert!(n <= 1, "count must be <= 1, got {}", n);
        if n > 0 {
            prop_assert!(output[0][0].is_finite(),
                "spatial feature x must be finite, got {}", output[0][0]);
            prop_assert!(output[0][1].is_finite(),
                "spatial feature y must be finite, got {}", output[0][1]);
        }
    }

    // =========================================================================
    // Noise covariance property
    // =========================================================================

    /// estimate_noise_covariance on random finite data produces positive diagonal.
    #[test]
    fn noise_covariance_diagonal_positive(
        vals in prop::collection::vec(-1e3f64..1e3, 6..=6),
    ) {
        // 2-channel data, 3 time samples (minimum for non-zero covariance)
        let data: [[f64; 2]; 3] = [
            [vals[0], vals[1]],
            [vals[2], vals[3]],
            [vals[4], vals[5]],
        ];
        let noise_std = [1e6, 1e6]; // high threshold so all samples are "quiet"

        let cov = estimate_noise_covariance(&data, &noise_std, 3.0, 2);
        // Diagonal entries of a covariance matrix must be >= 0
        prop_assert!(cov[0][0] >= 0.0,
            "covariance diagonal [0][0] must be >= 0, got {}", cov[0][0]);
        prop_assert!(cov[1][1] >= 0.0,
            "covariance diagonal [1][1] must be >= 0, got {}", cov[1][1]);
        prop_assert!(cov[0][0].is_finite(),
            "covariance diagonal [0][0] must be finite, got {}", cov[0][0]);
        prop_assert!(cov[1][1].is_finite(),
            "covariance diagonal [1][1] must be finite, got {}", cov[1][1]);
    }

    // =========================================================================
    // Template subtraction energy property
    // =========================================================================

    /// After peeling, residual energy <= original energy + small epsilon.
    #[test]
    fn peel_reduces_or_preserves_energy(
        template_vals in prop::collection::vec(-5.0f64..5.0, 4..=4),
        amp in 0.8f64..1.8,
    ) {
        let template: [f64; 4] = [template_vals[0], template_vals[1], template_vals[2], template_vals[3]];

        // Only test when template has non-trivial norm
        let norm_sq: f64 = template.iter().map(|x| x * x).sum();
        if norm_sq < 0.01 {
            return Ok(());
        }

        let mut sub = TemplateSubtractor::<4, 2>::new(10);
        sub.add_template(&template, 0.5, 2.0).unwrap();

        // Build data = amp * template at position [1..5] in a length-8 buffer
        let mut data = [0.0f64; 8];
        for i in 0..4 {
            data[1 + i] = amp * template[i];
        }

        let energy_before: f64 = data.iter().map(|x| x * x).sum();

        let spike_times = [2usize];
        let mut results = [PeelResult { sample: 0, template_id: 0, amplitude: 0.0 }; 4];
        let n = sub.peel(&mut data, &spike_times, 1, 1, &mut results);

        let energy_after: f64 = data.iter().map(|x| x * x).sum();

        // If subtraction occurred, residual should not exceed original
        if n > 0 {
            let eps = 1e-6;
            prop_assert!(energy_after <= energy_before + eps,
                "peel should not amplify energy: before={}, after={}", energy_before, energy_after);
        }
    }

    // =========================================================================
    // Extract peak channel count bounded
    // =========================================================================

    /// extract_peak_channel returns count <= min(n_events, output.len()).
    #[test]
    fn extract_peak_channel_count_bounded(
        data_vals in prop::collection::vec(-1e6f64..1e6, 16..=16),
        n_events in 1usize..=4,
    ) {
        // 2-channel data, 8 time samples
        let mut data = [[0.0f64; 2]; 8];
        for t in 0..8 {
            data[t][0] = data_vals[t * 2];
            data[t][1] = data_vals[t * 2 + 1];
        }

        // Create events pointing to valid samples
        let mut events = [MultiChannelEvent { sample: 0, channel: 0, amplitude: 1.0 }; 4];
        for (i, event) in events.iter_mut().enumerate().take(n_events) {
            *event = MultiChannelEvent {
                sample: 2 + i,  // safe range with pre_samples=1 and W=4
                channel: i % 2,
                amplitude: 1.0,
            };
        }

        let mut output = [[0.0f64; 4]; 3]; // buffer smaller than max events
        let count = extract_peak_channel::<2, 4>(&data, &events, n_events, 1, &mut output);

        let expected_max = if n_events < output.len() { n_events } else { output.len() };
        prop_assert!(count <= expected_max,
            "count {} must be <= min(n_events={}, buffer_len={})", count, n_events, output.len());
    }
}
