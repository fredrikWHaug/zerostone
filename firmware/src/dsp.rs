//! DSP primitives optimized for Cortex-M33 FPU hot paths.
//!
//! Provides loop-unrolled dot product, norm, fast inverse square root, and
//! normalized cross-correlation (NCC) for real-time spike classification.
//! All functions are `no_std` compatible and use `#[inline(always)]` for
//! the hot path.

/// Computes the inner product of two slices.
///
/// Loop is manually unrolled by 4 to allow the compiler to emit
/// back-to-back FMA instructions on Cortex-M33.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());

    let mut acc0: f32 = 0.0;
    let mut acc1: f32 = 0.0;
    let mut acc2: f32 = 0.0;
    let mut acc3: f32 = 0.0;

    let chunks = len / 4;
    let mut i = 0;
    let mut c = 0;
    while c < chunks {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
        c += 1;
    }

    // Handle remainder.
    while i < len {
        acc0 += a[i] * b[i];
        i += 1;
    }

    (acc0 + acc1) + (acc2 + acc3)
}

/// Computes the sum of squares (squared L2 norm) of a slice.
#[inline(always)]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot_product(a, a)
}

/// Fast inverse square root using the Quake III algorithm.
///
/// Returns an approximation of `1.0 / sqrt(x)` with one Newton-Raphson
/// refinement step. Accuracy is within ~0.2% for typical neural signal
/// magnitudes (x in 1e-6 .. 1e3).
///
/// Returns `0.0` for non-positive inputs to avoid NaN propagation.
#[inline(always)]
pub fn fast_inv_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    let half_x = 0.5 * x;
    let i = x.to_bits();
    // Magic constant for f32 fast inverse sqrt.
    let i = 0x5f37_59df - (i >> 1);
    let y = f32::from_bits(i);

    // One Newton-Raphson iteration: y = y * (1.5 - half_x * y * y)
    let y = y * (1.5 - half_x * y * y);

    y
}

/// Normalized cross-correlation between a waveform and a template.
///
/// ```text
/// NCC = dot(waveform, template) / sqrt(||waveform||^2 * template_norm_sq)
/// ```
///
/// `template_norm_sq` is the precomputed sum of squares of the template.
/// Returns `0.0` if either signal has zero energy.
#[inline(always)]
pub fn ncc(waveform: &[f32], template: &[f32], template_norm_sq: f32) -> f32 {
    let waveform_norm_sq = norm_sq(waveform);

    if waveform_norm_sq <= 0.0 || template_norm_sq <= 0.0 {
        return 0.0;
    }

    let dot = dot_product(waveform, template);
    let denom_sq = waveform_norm_sq * template_norm_sq;

    // Use fast inverse sqrt to avoid hardware sqrt (Cortex-M33 has no VSQRT).
    // NCC = dot * fast_inv_sqrt(denom_sq)
    dot * fast_inv_sqrt(denom_sq)
}

/// Classifies a waveform against a fixed-size array of templates in one pass.
///
/// - `waveform`: the spike waveform to classify (length `W`).
/// - `templates`: array of `(waveform, norm_sq, cluster_id)` tuples.
/// - `count`: number of active templates in the array (must be <= `N`).
/// - `min_corr`: minimum NCC threshold for a valid match.
///
/// Returns the `cluster_id` of the best-matching template if its NCC exceeds
/// `min_corr`, or `0` (unclassified) otherwise.
#[inline(always)]
pub fn batch_ncc<const W: usize, const N: usize>(
    waveform: &[f32; W],
    templates: &[([f32; W], f32, u8); N],
    count: usize,
    min_corr: f32,
) -> u8 {
    let waveform_norm_sq = norm_sq(waveform);

    if waveform_norm_sq <= 0.0 {
        return 0;
    }

    let mut best_ncc: f32 = -2.0;
    let mut best_id: u8 = 0;

    let limit = if count < N { count } else { N };
    let mut t = 0;
    while t < limit {
        let (ref tmpl_wf, tmpl_norm_sq, cluster_id) = templates[t];

        if tmpl_norm_sq > 0.0 {
            let dot = dot_product(waveform, tmpl_wf);
            let denom_sq = waveform_norm_sq * tmpl_norm_sq;
            let corr = dot * fast_inv_sqrt(denom_sq);

            if corr > best_ncc {
                best_ncc = corr;
                best_id = cluster_id;
            }
        }

        t += 1;
    }

    if best_ncc >= min_corr {
        best_id
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // -- dot_product tests ---------------------------------------------------

    #[test]
    fn dot_product_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = dot_product(&a, &b);
        assert!((result - 70.0).abs() < EPS, "got {result}");
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!((dot_product(&a, &b)).abs() < EPS);
    }

    #[test]
    fn dot_product_zero_vector() {
        let a = [0.0; 8];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!((dot_product(&a, &b)).abs() < EPS);
    }

    #[test]
    fn dot_product_empty_slices() {
        let a: &[f32] = &[];
        let b: &[f32] = &[];
        assert!((dot_product(a, b)).abs() < EPS);
    }

    #[test]
    fn dot_product_single_element() {
        let a = [3.0];
        let b = [7.0];
        assert!((dot_product(&a, &b) - 21.0).abs() < EPS);
    }

    #[test]
    fn dot_product_remainder_path() {
        // Length 5: one chunk of 4 + 1 remainder
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0];
        // 2 + 4 + 6 + 8 + 10 = 30
        assert!((dot_product(&a, &b) - 30.0).abs() < EPS);
    }

    // -- norm_sq tests -------------------------------------------------------

    #[test]
    fn norm_sq_correctness() {
        let a = [3.0, 4.0];
        // 9 + 16 = 25
        assert!((norm_sq(&a) - 25.0).abs() < EPS);
    }

    #[test]
    fn norm_sq_zero() {
        let a = [0.0; 4];
        assert!((norm_sq(&a)).abs() < EPS);
    }

    // -- fast_inv_sqrt tests -------------------------------------------------

    #[test]
    fn fast_inv_sqrt_accuracy() {
        // Test across typical neural signal magnitudes.
        let test_values: &[f32] = &[0.001, 0.01, 0.1, 1.0, 4.0, 25.0, 100.0, 1000.0];
        for &x in test_values {
            let approx = fast_inv_sqrt(x);
            let exact = 1.0 / x.sqrt();
            let rel_err = ((approx - exact) / exact).abs();
            assert!(
                rel_err < 0.01,
                "fast_inv_sqrt({x}): approx={approx}, exact={exact}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn fast_inv_sqrt_zero_returns_zero() {
        assert_eq!(fast_inv_sqrt(0.0), 0.0);
    }

    #[test]
    fn fast_inv_sqrt_negative_returns_zero() {
        assert_eq!(fast_inv_sqrt(-1.0), 0.0);
    }

    // -- ncc tests -----------------------------------------------------------

    #[test]
    fn ncc_identical_signals() {
        let a = [1.0, -0.5, 0.3, -0.8];
        let norm = norm_sq(&a);
        let result = ncc(&a, &a, norm);
        assert!(
            (result - 1.0).abs() < 0.01,
            "NCC of identical signals should be ~1.0, got {result}"
        );
    }

    #[test]
    fn ncc_orthogonal_signals() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let norm_b = norm_sq(&b);
        let result = ncc(&a, &b, norm_b);
        assert!(
            result.abs() < 0.01,
            "NCC of orthogonal signals should be ~0.0, got {result}"
        );
    }

    #[test]
    fn ncc_inverted_signals() {
        let a = [1.0, -0.5, 0.3, -0.8];
        let b = [-1.0, 0.5, -0.3, 0.8];
        let norm_b = norm_sq(&b);
        let result = ncc(&a, &b, norm_b);
        assert!(
            (result - (-1.0)).abs() < 0.01,
            "NCC of inverted signals should be ~-1.0, got {result}"
        );
    }

    #[test]
    fn ncc_zero_waveform() {
        let a = [0.0; 4];
        let b = [1.0, 2.0, 3.0, 4.0];
        let norm_b = norm_sq(&b);
        assert_eq!(ncc(&a, &b, norm_b), 0.0);
    }

    #[test]
    fn ncc_zero_template_norm() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(ncc(&a, &a, 0.0), 0.0);
    }

    // -- batch_ncc tests -----------------------------------------------------

    #[test]
    fn batch_ncc_best_template() {
        let waveform: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

        let t1: [f32; 4] = [1.0, 0.0, 0.0, 0.0]; // perfect match
        let t2: [f32; 4] = [0.0, 1.0, 0.0, 0.0]; // orthogonal

        let templates: [([f32; 4], f32, u8); 4] = [
            (t1, norm_sq(&t1), 1),
            (t2, norm_sq(&t2), 2),
            ([0.0; 4], 0.0, 0),
            ([0.0; 4], 0.0, 0),
        ];

        let result = batch_ncc(&waveform, &templates, 2, 0.5);
        assert_eq!(result, 1);
    }

    #[test]
    fn batch_ncc_below_threshold() {
        let waveform: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
        let t1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

        let templates: [([f32; 4], f32, u8); 2] = [
            (t1, norm_sq(&t1), 1),
            ([0.0; 4], 0.0, 0),
        ];

        // NCC ~ 0.707, threshold is 0.9
        let result = batch_ncc(&waveform, &templates, 1, 0.9);
        assert_eq!(result, 0);
    }

    #[test]
    fn batch_ncc_zero_waveform() {
        let waveform: [f32; 4] = [0.0; 4];
        let t1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

        let templates: [([f32; 4], f32, u8); 1] = [
            (t1, norm_sq(&t1), 1),
        ];

        assert_eq!(batch_ncc(&waveform, &templates, 1, 0.5), 0);
    }

    #[test]
    fn batch_ncc_empty_count() {
        let waveform: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

        let templates: [([f32; 4], f32, u8); 2] = [
            ([1.0, 0.0, 0.0, 0.0], 1.0, 1),
            ([0.0; 4], 0.0, 0),
        ];

        assert_eq!(batch_ncc(&waveform, &templates, 0, 0.5), 0);
    }

    #[test]
    fn batch_ncc_selects_among_three() {
        // waveform aligns best with t2
        let waveform: [f32; 4] = [0.1, 0.9, 0.1, 0.0];

        let t1: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
        let t2: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
        let t3: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

        let templates: [([f32; 4], f32, u8); 3] = [
            (t1, norm_sq(&t1), 1),
            (t2, norm_sq(&t2), 2),
            (t3, norm_sq(&t3), 3),
        ];

        let result = batch_ncc(&waveform, &templates, 3, 0.5);
        assert_eq!(result, 2);
    }
}
