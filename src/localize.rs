//! Spike localization from multi-channel amplitude distributions.
//!
//! Estimates the spatial origin of a detected spike using peak amplitudes
//! across channels and the known probe geometry. Three methods are provided
//! in increasing accuracy (and cost):
//!
//! 1. **Center-of-mass** ([`center_of_mass`]) -- weighted average of channel
//!    positions, using absolute amplitude as weight. Fastest, but biased near
//!    probe borders.
//! 2. **Thresholded center-of-mass** ([`center_of_mass_threshold`]) -- same
//!    idea but ignores channels below a visibility threshold.
//! 3. **Monopole fit** ([`monopole_localize`]) -- fits a point-source model
//!    where amplitude decays as 1/distance. More accurate on dense probes
//!    (Boussard et al. 2021, Kilosort4).
//!
//! A convenience wrapper ([`localize_spike`]) extracts per-channel peak
//! amplitudes from a waveform snippet and calls center-of-mass using the
//! probe geometry stored in [`ProbeLayout`](crate::probe::ProbeLayout).
//!
//! # References
//!
//! - Boussard et al. (2021). Three-dimensional spike localization and improved
//!   motion correction for Neuropixels recordings. NeurIPS.
//! - Hurwitz et al. (2021). Scalable spike source localization in extracellular
//!   recordings using amortized variational inference.

use crate::probe::ProbeLayout;

/// Compute the center-of-mass position of a spike from per-channel amplitudes.
///
/// Weights each channel position by the absolute value of its amplitude.
/// Returns the weighted average `[x, y]` in the same coordinate system as
/// the channel positions (typically micrometers).
///
/// If all amplitudes are zero, returns `[0.0, 0.0]`.
///
/// # Arguments
///
/// * `amplitudes` -- peak amplitude (or peak-to-peak) on each channel
/// * `positions` -- `(x, y)` position of each channel in micrometers
///
/// # Example
///
/// ```
/// use zerostone::localize::center_of_mass;
///
/// // Spike visible only on channel 2 at position (0, 50)
/// let amps = [0.0, 0.0, -5.0, 0.0];
/// let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
/// let loc = center_of_mass(&amps, &pos);
/// assert!((loc[0] - 0.0).abs() < 1e-9);
/// assert!((loc[1] - 50.0).abs() < 1e-9);
/// ```
pub fn center_of_mass<const C: usize>(
    amplitudes: &[f64; C],
    positions: &[[f64; 2]; C],
) -> [f64; 2] {
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut w_total = 0.0;

    let mut i = 0;
    while i < C {
        let w = libm::fabs(amplitudes[i]);
        wx += w * positions[i][0];
        wy += w * positions[i][1];
        w_total += w;
        i += 1;
    }

    if w_total == 0.0 {
        return [0.0, 0.0];
    }
    [wx / w_total, wy / w_total]
}

/// Center-of-mass localization using only channels above an amplitude threshold.
///
/// Channels with `|amplitude| < threshold` are excluded from the weighted
/// average. Returns `None` if no channel meets the threshold.
///
/// This is useful for dense probes where distant channels contribute only
/// noise to the center-of-mass estimate.
///
/// # Arguments
///
/// * `amplitudes` -- peak amplitude on each channel
/// * `positions` -- `(x, y)` position of each channel
/// * `threshold` -- minimum absolute amplitude to include a channel
///
/// # Example
///
/// ```
/// use zerostone::localize::center_of_mass_threshold;
///
/// let amps = [0.1, 0.2, -5.0, -3.0];
/// let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
/// // Only channels 2 and 3 are above threshold 1.0
/// let loc = center_of_mass_threshold(&amps, &pos, 1.0).unwrap();
/// // Weighted average: (5*50 + 3*75)/(5+3) = 475/8 = 59.375
/// assert!((loc[1] - 59.375).abs() < 1e-9);
/// ```
pub fn center_of_mass_threshold<const C: usize>(
    amplitudes: &[f64; C],
    positions: &[[f64; 2]; C],
    threshold: f64,
) -> Option<[f64; 2]> {
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut w_total = 0.0;

    let mut i = 0;
    while i < C {
        let w = libm::fabs(amplitudes[i]);
        if w >= threshold {
            wx += w * positions[i][0];
            wy += w * positions[i][1];
            w_total += w;
        }
        i += 1;
    }

    if w_total == 0.0 {
        return None;
    }
    Some([wx / w_total, wy / w_total])
}

/// Fit a monopole source model to estimate spike position.
///
/// Models the extracellular potential as a point current source where the
/// recorded amplitude on channel *i* follows:
///
/// ```text
///     a_i = alpha / sqrt((x - x_i)^2 + (y - y_i)^2 + z^2)
/// ```
///
/// The algorithm starts from a weighted center-of-mass estimate and
/// refines (x, y) using iterative reweighted least squares (IRLS) for
/// `n_iter` steps. The perpendicular distance *z* is not estimated --
/// it acts as a regularization parameter (set via `z_offset`).
///
/// Returns `(x, y)` in the same coordinate system as channel positions.
/// Returns `None` if all amplitudes are zero.
///
/// # Arguments
///
/// * `amplitudes` -- absolute peak amplitudes on each channel (must be >= 0)
/// * `positions` -- `(x, y)` positions of each channel
/// * `z_offset` -- assumed perpendicular distance from probe plane (um).
///   Typical value: 1.0-20.0. Prevents division by zero and regularizes
///   the fit.
/// * `n_iter` -- number of refinement iterations (3-10 is typical)
///
/// # Example
///
/// ```
/// use zerostone::localize::monopole_localize;
///
/// // Source at (16.0, 37.5), probe is a 2-column polytrode
/// let positions = [
///     [-16.0, 0.0], [16.0, 0.0],
///     [-16.0, 25.0], [16.0, 25.0],
///     [-16.0, 50.0], [16.0, 50.0],
///     [-16.0, 75.0], [16.0, 75.0],
/// ];
/// // Simulate 1/r amplitudes from source at (16.0, 37.5), z=10
/// let src: [f64; 2] = [16.0, 37.5];
/// let z: f64 = 10.0;
/// let mut amps = [0.0f64; 8];
/// for i in 0..8 {
///     let dx = src[0] - positions[i][0];
///     let dy = src[1] - positions[i][1];
///     let r = (dx * dx + dy * dy + z * z).sqrt();
///     amps[i] = 100.0 / r;
/// }
/// let est = monopole_localize(&amps, &positions, z, 10).unwrap();
/// assert!((est[0] - 16.0).abs() < 1.0);
/// assert!((est[1] - 37.5).abs() < 1.0);
/// ```
#[allow(clippy::needless_range_loop)]
pub fn monopole_localize<const C: usize>(
    amplitudes: &[f64; C],
    positions: &[[f64; 2]; C],
    z_offset: f64,
    n_iter: usize,
) -> Option<[f64; 2]> {
    // Start from amplitude-weighted center of mass
    let mut wx = 0.0;
    let mut wy = 0.0;
    let mut w_total = 0.0;
    for i in 0..C {
        let w = amplitudes[i];
        if w > 0.0 {
            wx += w * positions[i][0];
            wy += w * positions[i][1];
            w_total += w;
        }
    }
    if w_total == 0.0 {
        return None;
    }

    let mut x = wx / w_total;
    let mut y = wy / w_total;
    let z2 = z_offset * z_offset;

    // Gradient descent on the monopole objective:
    //   L = sum_i (a_i - alpha / r_i)^2
    // where r_i = sqrt((x - x_i)^2 + (y - y_i)^2 + z^2).
    //
    // At each step we first estimate alpha from the current (x, y),
    // then take a gradient step on L with respect to (x, y).
    for _iter in 0..n_iter {
        // Estimate alpha = sum(a_i / r_i) / sum(1 / r_i^2)
        // via least-squares: minimize sum(a_i - alpha/r_i)^2 over alpha.
        let mut s1 = 0.0; // sum(a_i / r_i)
        let mut s2 = 0.0; // sum(1 / r_i^2)
        for i in 0..C {
            if amplitudes[i] <= 0.0 {
                continue;
            }
            let dx = x - positions[i][0];
            let dy = y - positions[i][1];
            let r = libm::sqrt(dx * dx + dy * dy + z2);
            s1 += amplitudes[i] / r;
            s2 += 1.0 / (r * r);
        }
        if s2 <= 0.0 {
            break;
        }
        let alpha = s1 / s2;

        // Compute gradient of L w.r.t. (x, y)
        // dL/dx = 2 * sum_i (a_i - alpha/r_i) * (alpha / r_i^3) * (x - x_i)
        let mut grad_x = 0.0;
        let mut grad_y = 0.0;
        let mut loss = 0.0;
        for i in 0..C {
            if amplitudes[i] <= 0.0 {
                continue;
            }
            let dx = x - positions[i][0];
            let dy = y - positions[i][1];
            let r2 = dx * dx + dy * dy + z2;
            let r = libm::sqrt(r2);
            let predicted = alpha / r;
            let residual = amplitudes[i] - predicted;
            // derivative of -alpha/r w.r.t. x: alpha * (x-xi) / r^3
            let coeff = 2.0 * residual * alpha / (r2 * r);
            grad_x += coeff * dx;
            grad_y += coeff * dy;
            loss += residual * residual;
        }

        if loss < 1e-30 {
            break;
        }

        // Adaptive step size via line search (Barzilai-Borwein inspired):
        // Use 1 / (max curvature estimate) as step size.
        let grad_norm = libm::sqrt(grad_x * grad_x + grad_y * grad_y);
        if grad_norm < 1e-15 {
            break;
        }

        // Backtracking line search: start with a reasonable step and halve
        // until loss decreases.
        let mut step = 1.0;
        let dir_x = grad_x / grad_norm;
        let dir_y = grad_y / grad_norm;

        let mut best_loss = loss;
        let mut best_x = x;
        let mut best_y = y;
        let mut k = 0;
        while k < 10 {
            let nx = x - step * dir_x;
            let ny = y - step * dir_y;

            // Evaluate loss at candidate
            let mut sa = 0.0;
            let mut sb = 0.0;
            for i in 0..C {
                if amplitudes[i] <= 0.0 {
                    continue;
                }
                let dx2 = nx - positions[i][0];
                let dy2 = ny - positions[i][1];
                let r = libm::sqrt(dx2 * dx2 + dy2 * dy2 + z2);
                sa += amplitudes[i] / r;
                sb += 1.0 / (r * r);
            }
            let a2 = if sb > 0.0 { sa / sb } else { alpha };
            let mut new_loss = 0.0;
            for i in 0..C {
                if amplitudes[i] <= 0.0 {
                    continue;
                }
                let dx2 = nx - positions[i][0];
                let dy2 = ny - positions[i][1];
                let r = libm::sqrt(dx2 * dx2 + dy2 * dy2 + z2);
                let res = amplitudes[i] - a2 / r;
                new_loss += res * res;
            }

            if new_loss < best_loss {
                best_loss = new_loss;
                best_x = nx;
                best_y = ny;
                // Try larger step
                step *= 2.0;
            } else {
                step *= 0.5;
            }
            k += 1;
        }

        x = best_x;
        y = best_y;
    }

    Some([x, y])
}

/// Localize a spike from a multi-channel waveform snippet using center-of-mass.
///
/// Extracts the peak absolute amplitude from each channel in the waveform
/// and computes the center-of-mass position using the probe geometry.
///
/// # Arguments
///
/// * `waveform` -- `C` channels x `W` samples waveform snippet (row-major:
///   `waveform[ch]` is the waveform on channel `ch`)
/// * `probe` -- probe geometry providing channel positions
///
/// # Example
///
/// ```
/// use zerostone::localize::localize_spike;
/// use zerostone::probe::ProbeLayout;
///
/// let probe = ProbeLayout::<4>::linear(25.0);
/// // Spike only on channel 2 (peak amplitude -8.0)
/// let waveform = [
///     [0.0, 0.1, 0.0, -0.1],   // ch 0
///     [0.0, 0.2, -0.1, 0.0],   // ch 1
///     [0.5, -3.0, -8.0, 2.0],  // ch 2
///     [0.0, -0.5, -0.3, 0.1],  // ch 3
/// ];
/// let loc = localize_spike(&waveform, &probe);
/// // Should be near channel 2 at y=50
/// assert!((loc[1] - 50.0).abs() < 5.0);
/// ```
pub fn localize_spike<const C: usize, const W: usize>(
    waveform: &[[f64; W]; C],
    probe: &ProbeLayout<C>,
) -> [f64; 2] {
    let mut peak_amps = [0.0f64; C];

    let mut ch = 0;
    while ch < C {
        let mut max_abs = 0.0;
        let mut s = 0;
        while s < W {
            let a = libm::fabs(waveform[ch][s]);
            if a > max_abs {
                max_abs = a;
            }
            s += 1;
        }
        peak_amps[ch] = max_abs;
        ch += 1;
    }

    center_of_mass(&peak_amps, probe.positions())
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `center_of_mass` never panics for any finite input.
    #[kani::proof]
    #[kani::unwind(6)]
    fn center_of_mass_no_panic() {
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        let a2: f64 = kani::any();
        let a3: f64 = kani::any();
        kani::assume(a0.is_finite() && a0 >= -1e6 && a0 <= 1e6);
        kani::assume(a1.is_finite() && a1 >= -1e6 && a1 <= 1e6);
        kani::assume(a2.is_finite() && a2 >= -1e6 && a2 <= 1e6);
        kani::assume(a3.is_finite() && a3 >= -1e6 && a3 <= 1e6);

        let amps = [a0, a1, a2, a3];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let result = center_of_mass(&amps, &pos);
        // Should not panic -- result is either [0,0] (all zero) or a weighted avg
        let _ = result;
    }

    /// Prove that `center_of_mass` output is finite when at least one
    /// amplitude is nonzero and all inputs are finite.
    #[kani::proof]
    #[kani::unwind(4)]
    fn center_of_mass_finite_output() {
        let a0: f64 = kani::any();
        let a1: f64 = kani::any();
        kani::assume(a0.is_finite() && a0 >= -1e6 && a0 <= 1e6);
        kani::assume(a1.is_finite() && a1 >= -1e6 && a1 <= 1e6);
        // At least one nonzero
        kani::assume(a0 != 0.0 || a1 != 0.0);

        let amps = [a0, a1];
        let pos = [[0.0, 0.0], [0.0, 25.0]];
        let result = center_of_mass(&amps, &pos);
        assert!(result[0].is_finite(), "x must be finite");
        assert!(result[1].is_finite(), "y must be finite");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::ProbeLayout;

    // ---- center_of_mass ----

    #[test]
    fn com_single_channel_returns_that_position() {
        let amps = [-5.0];
        let pos = [[10.0, 20.0]];
        let loc = center_of_mass(&amps, &pos);
        assert!((loc[0] - 10.0).abs() < 1e-12);
        assert!((loc[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn com_equal_weights_returns_centroid() {
        let amps = [1.0, 1.0, 1.0, 1.0];
        let pos = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        let loc = center_of_mass(&amps, &pos);
        assert!((loc[0] - 5.0).abs() < 1e-12);
        assert!((loc[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn com_all_zero_returns_origin() {
        let amps = [0.0, 0.0, 0.0, 0.0];
        let pos = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        let loc = center_of_mass(&amps, &pos);
        assert!((loc[0]).abs() < 1e-12);
        assert!((loc[1]).abs() < 1e-12);
    }

    #[test]
    fn com_negative_amps_uses_absolute_value() {
        let amps = [-3.0, -1.0];
        let pos = [[0.0, 0.0], [0.0, 100.0]];
        let loc = center_of_mass(&amps, &pos);
        // Weighted: (3*0 + 1*100)/(3+1) = 25
        assert!((loc[1] - 25.0).abs() < 1e-12);
    }

    #[test]
    fn com_dominant_channel_pulls_location() {
        let amps = [0.0, 0.0, -10.0, 0.0];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let loc = center_of_mass(&amps, &pos);
        assert!((loc[0] - 0.0).abs() < 1e-12);
        assert!((loc[1] - 50.0).abs() < 1e-12);
    }

    // ---- center_of_mass_threshold ----

    #[test]
    fn com_threshold_filters_low_channels() {
        let amps = [0.1, 0.2, -5.0, -3.0];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
        let loc = center_of_mass_threshold(&amps, &pos, 1.0).unwrap();
        let expected_y = (5.0 * 50.0 + 3.0 * 75.0) / (5.0 + 3.0);
        assert!((loc[1] - expected_y).abs() < 1e-9);
    }

    #[test]
    fn com_threshold_none_when_all_below() {
        let amps = [0.1, 0.2, 0.05];
        let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0]];
        assert!(center_of_mass_threshold(&amps, &pos, 1.0).is_none());
    }

    #[test]
    fn com_threshold_zero_includes_all_nonzero() {
        let amps = [1.0, 2.0, 3.0];
        let pos = [[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]];
        let with_thresh = center_of_mass_threshold(&amps, &pos, 0.0).unwrap();
        let without_thresh = center_of_mass(&amps, &pos);
        assert!((with_thresh[0] - without_thresh[0]).abs() < 1e-12);
        assert!((with_thresh[1] - without_thresh[1]).abs() < 1e-12);
    }

    // ---- monopole_localize ----

    #[test]
    fn monopole_recovers_known_source() {
        // Source at (16.0, 37.5), z=10
        let positions = [
            [-16.0, 0.0],
            [16.0, 0.0],
            [-16.0, 25.0],
            [16.0, 25.0],
            [-16.0, 50.0],
            [16.0, 50.0],
            [-16.0, 75.0],
            [16.0, 75.0],
        ];
        let src = [16.0, 37.5];
        let z = 10.0;
        let alpha = 100.0;
        let mut amps = [0.0f64; 8];
        for i in 0..8 {
            let dx = src[0] - positions[i][0];
            let dy = src[1] - positions[i][1];
            let r = libm::sqrt(dx * dx + dy * dy + z * z);
            amps[i] = alpha / r;
        }
        let est = monopole_localize(&amps, &positions, z, 10).unwrap();
        assert!(
            (est[0] - 16.0).abs() < 1.0,
            "x error: {}",
            (est[0] - 16.0).abs()
        );
        assert!(
            (est[1] - 37.5).abs() < 1.0,
            "y error: {}",
            (est[1] - 37.5).abs()
        );
    }

    #[test]
    fn monopole_more_accurate_than_com() {
        // Source off-center at (10.0, 30.0), z=5
        let positions = [
            [-16.0, 0.0],
            [16.0, 0.0],
            [-16.0, 25.0],
            [16.0, 25.0],
            [-16.0, 50.0],
            [16.0, 50.0],
            [-16.0, 75.0],
            [16.0, 75.0],
        ];
        let src = [10.0, 30.0];
        let z = 5.0;
        let alpha = 100.0;
        let mut amps = [0.0f64; 8];
        for i in 0..8 {
            let dx = src[0] - positions[i][0];
            let dy = src[1] - positions[i][1];
            let r = libm::sqrt(dx * dx + dy * dy + z * z);
            amps[i] = alpha / r;
        }

        let com = center_of_mass(&amps, &positions);
        let mono = monopole_localize(&amps, &positions, z, 10).unwrap();

        let com_err = libm::sqrt(
            (com[0] - src[0]) * (com[0] - src[0]) + (com[1] - src[1]) * (com[1] - src[1]),
        );
        let mono_err = libm::sqrt(
            (mono[0] - src[0]) * (mono[0] - src[0]) + (mono[1] - src[1]) * (mono[1] - src[1]),
        );

        assert!(
            mono_err < com_err,
            "monopole error ({:.3}) should be less than COM error ({:.3})",
            mono_err,
            com_err
        );
    }

    #[test]
    fn monopole_all_zero_returns_none() {
        let amps = [0.0; 4];
        let pos = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        assert!(monopole_localize(&amps, &pos, 10.0, 5).is_none());
    }

    #[test]
    fn monopole_single_channel() {
        let amps = [5.0];
        let pos = [[42.0, 17.0]];
        let est = monopole_localize(&amps, &pos, 10.0, 5).unwrap();
        assert!((est[0] - 42.0).abs() < 1e-9);
        assert!((est[1] - 17.0).abs() < 1e-9);
    }

    #[test]
    fn monopole_uniform_amplitudes_returns_centroid() {
        let amps = [1.0, 1.0, 1.0, 1.0];
        let pos = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        let est = monopole_localize(&amps, &pos, 10.0, 10).unwrap();
        // With uniform amplitudes, should converge near centroid
        assert!((est[0] - 5.0).abs() < 1.0);
        assert!((est[1] - 5.0).abs() < 1.0);
    }

    // ---- localize_spike ----

    #[test]
    fn localize_spike_dominant_channel() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let waveform = [
            [0.0, 0.1, 0.0, -0.1],  // ch 0, peak 0.1
            [0.0, 0.2, -0.1, 0.0],  // ch 1, peak 0.2
            [0.5, -3.0, -8.0, 2.0], // ch 2, peak 8.0
            [0.0, -0.5, -0.3, 0.1], // ch 3, peak 0.5
        ];
        let loc = localize_spike(&waveform, &probe);
        // Dominant weight on ch 2 (y=50), should be close
        assert!((loc[1] - 50.0).abs() < 5.0);
    }

    #[test]
    fn localize_spike_uniform_waveforms() {
        let probe = ProbeLayout::<4>::linear(25.0);
        // All channels have same peak
        let waveform = [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ];
        let loc = localize_spike(&waveform, &probe);
        // Centroid: y = (0+25+50+75)/4 = 37.5
        assert!((loc[1] - 37.5).abs() < 1e-9);
    }

    #[test]
    fn localize_spike_all_zero() {
        let probe = ProbeLayout::<3>::linear(10.0);
        let waveform = [[0.0; 4]; 3];
        let loc = localize_spike(&waveform, &probe);
        assert!((loc[0]).abs() < 1e-12);
        assert!((loc[1]).abs() < 1e-12);
    }
}
