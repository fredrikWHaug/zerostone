use proptest::prelude::*;
use zerostone::connectivity;
use zerostone::entropy;
use zerostone::isi;
use zerostone::kalman::KalmanFilter;
use zerostone::linalg::Matrix;
use zerostone::pac;
use zerostone::probe::ProbeLayout;
use zerostone::quality;
use zerostone::riemannian;
use zerostone::whitening::{WhiteningMatrix, WhiteningMode};
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
}
