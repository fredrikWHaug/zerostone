//! Event-Related Spectral Perturbation (ERSP) baseline normalization.
//!
//! Provides baseline normalization modes for pre-computed time-frequency power
//! matrices. Used to visualize how brain oscillations change around events.
//!
//! # Overview
//!
//! - [`baseline_normalize`] -- in-place baseline correction (dB, z-score, percentage, log-ratio)
//! - [`epoch_average`] -- average multiple epoch power matrices into one
//! - [`BaselineMode`] -- normalization mode selection
//!
//! # Example
//!
//! ```
//! use zerostone::ersp::{baseline_normalize, epoch_average, BaselineMode};
//!
//! // 3 frames x 2 freq bins, baseline is frame 0
//! let mut power = [1.0, 1.0, 2.0, 2.0, 4.0, 4.0];
//! baseline_normalize(&mut power, 3, 2, 0, 1, BaselineMode::Db).unwrap();
//! // Frame 0 (baseline) should be 0 dB
//! assert!(power[0].abs() < 1e-10);
//! assert!(power[1].abs() < 1e-10);
//! ```

use crate::float::{self, Float};

/// Baseline normalization mode for ERSP computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineMode {
    /// 10 * log10(power / baseline_mean)
    Db,
    /// (power - baseline_mean) / baseline_std
    Zscore,
    /// (power - baseline_mean) / baseline_mean * 100
    Percentage,
    /// log10(power / baseline_mean)
    LogRatio,
}

/// Errors that can occur during ERSP operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErspError {
    /// Baseline start must be less than baseline end
    InvalidBaselineRange,
    /// Baseline end exceeds number of frames
    BaselineOutOfBounds,
    /// Power slice length doesn't match n_frames * n_freqs
    DimensionMismatch,
    /// Power slice is empty
    EmptyPower,
    /// Zero or near-zero baseline mean (can't divide)
    ZeroBaselineMean,
    /// Zero or near-zero baseline std (can't compute z-score)
    ZeroBaselineStd,
    /// Output slice length doesn't match n_frames * n_freqs
    OutputDimensionMismatch,
    /// No epochs provided
    NoEpochs,
    /// Epoch slice length doesn't match expected size
    EpochDimensionMismatch,
}

/// Apply baseline normalization to a time-frequency power matrix in-place.
///
/// The power matrix is stored row-major as `&mut [Float]` with shape
/// `(n_frames, n_freqs)`. For each frequency bin, the baseline mean
/// (and std for z-score) is computed over frames `bl_start..bl_end`,
/// then every frame is normalized.
///
/// # Arguments
///
/// * `power` - Row-major power matrix `[n_frames * n_freqs]`, modified in-place
/// * `n_frames` - Number of time frames
/// * `n_freqs` - Number of frequency bins
/// * `bl_start` - Baseline start frame (inclusive)
/// * `bl_end` - Baseline end frame (exclusive)
/// * `mode` - Normalization mode
///
/// # Returns
///
/// `Ok(())` on success, or `ErspError` on validation failure.
///
/// # Example
///
/// ```
/// use zerostone::ersp::{baseline_normalize, BaselineMode};
/// use zerostone::float;
///
/// // 4 frames x 3 freq bins, baseline = frames 0..2
/// let mut power = [
///     1.0, 2.0, 3.0,  // frame 0
///     1.0, 2.0, 3.0,  // frame 1
///     2.0, 4.0, 6.0,  // frame 2
///     4.0, 8.0, 12.0, // frame 3
/// ];
/// baseline_normalize(&mut power, 4, 3, 0, 2, BaselineMode::Db).unwrap();
///
/// // Baseline frames should be 0 dB (power == baseline_mean)
/// assert!(power[0].abs() < 1e-10);
/// assert!(power[1].abs() < 1e-10);
/// assert!(power[2].abs() < 1e-10);
///
/// // Frame 2 has 2x baseline -> ~3.01 dB
/// assert!((power[6] - 10.0 * float::log10(2.0)).abs() < 1e-10);
/// ```
pub fn baseline_normalize(
    power: &mut [Float],
    n_frames: usize,
    n_freqs: usize,
    bl_start: usize,
    bl_end: usize,
    mode: BaselineMode,
) -> Result<(), ErspError> {
    if power.is_empty() {
        return Err(ErspError::EmptyPower);
    }
    if bl_start >= bl_end {
        return Err(ErspError::InvalidBaselineRange);
    }
    if bl_end > n_frames {
        return Err(ErspError::BaselineOutOfBounds);
    }
    if power.len() != n_frames * n_freqs {
        return Err(ErspError::DimensionMismatch);
    }

    let bl_len = (bl_end - bl_start) as Float;

    for f in 0..n_freqs {
        // Compute baseline mean for this frequency bin
        let mut bl_mean: Float = 0.0;
        for t in bl_start..bl_end {
            bl_mean += power[t * n_freqs + f];
        }
        bl_mean /= bl_len;

        // For z-score, also compute baseline std
        let bl_std = if mode == BaselineMode::Zscore {
            let mut var: Float = 0.0;
            for t in bl_start..bl_end {
                let diff = power[t * n_freqs + f] - bl_mean;
                var += diff * diff;
            }
            let std = float::sqrt(var / bl_len);
            if std < 1e-15 {
                return Err(ErspError::ZeroBaselineStd);
            }
            std
        } else {
            0.0
        };

        // Check for zero baseline mean in modes that divide by it
        if matches!(
            mode,
            BaselineMode::Db | BaselineMode::Percentage | BaselineMode::LogRatio
        ) && float::abs(bl_mean) < 1e-15
        {
            return Err(ErspError::ZeroBaselineMean);
        }

        // Apply normalization to all frames
        for t in 0..n_frames {
            let idx = t * n_freqs + f;
            let val = power[idx];
            power[idx] = match mode {
                BaselineMode::Db => 10.0 * float::log10(val / bl_mean),
                BaselineMode::Zscore => (val - bl_mean) / bl_std,
                BaselineMode::Percentage => (val - bl_mean) / bl_mean * 100.0,
                BaselineMode::LogRatio => float::log10(val / bl_mean),
            };
        }
    }

    Ok(())
}

/// Average multiple epoch power matrices into one.
///
/// Input is a flat `&[Float]` containing `n_epochs` concatenated power matrices,
/// each of size `n_frames * n_freqs`. Output is a single `n_frames * n_freqs`
/// matrix containing the element-wise mean.
///
/// # Arguments
///
/// * `epochs` - Flat array of `n_epochs * n_frames * n_freqs` values
/// * `n_epochs` - Number of epochs
/// * `n_frames` - Number of time frames per epoch
/// * `n_freqs` - Number of frequency bins per frame
/// * `output` - Output buffer of size `n_frames * n_freqs`
///
/// # Returns
///
/// `Ok(())` on success, or `ErspError` on validation failure.
///
/// # Example
///
/// ```
/// use zerostone::ersp::epoch_average;
///
/// // 2 epochs, each 2 frames x 2 freq bins
/// let epochs = [
///     1.0, 2.0, 3.0, 4.0,  // epoch 0
///     3.0, 4.0, 5.0, 6.0,  // epoch 1
/// ];
/// let mut output = [0.0; 4];
/// epoch_average(&epochs, 2, 2, 2, &mut output).unwrap();
///
/// assert!((output[0] - 2.0).abs() < 1e-10);  // mean(1, 3) = 2
/// assert!((output[1] - 3.0).abs() < 1e-10);  // mean(2, 4) = 3
/// assert!((output[2] - 4.0).abs() < 1e-10);  // mean(3, 5) = 4
/// assert!((output[3] - 5.0).abs() < 1e-10);  // mean(4, 6) = 5
/// ```
pub fn epoch_average(
    epochs: &[Float],
    n_epochs: usize,
    n_frames: usize,
    n_freqs: usize,
    output: &mut [Float],
) -> Result<(), ErspError> {
    if n_epochs == 0 {
        return Err(ErspError::NoEpochs);
    }
    let epoch_size = n_frames * n_freqs;
    if epochs.len() != n_epochs * epoch_size {
        return Err(ErspError::EpochDimensionMismatch);
    }
    if output.len() != epoch_size {
        return Err(ErspError::OutputDimensionMismatch);
    }

    let inv_n = 1.0 / n_epochs as Float;

    // Zero output
    for v in output.iter_mut() {
        *v = 0.0;
    }

    // Accumulate
    for e in 0..n_epochs {
        let offset = e * epoch_size;
        for i in 0..epoch_size {
            output[i] += epochs[offset + i];
        }
    }

    // Divide
    for v in output.iter_mut() {
        *v *= inv_n;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate alloc;

    // =========================================================================
    // BaselineMode::Db tests
    // =========================================================================

    #[test]
    fn test_db_known_values() {
        // 3 frames x 2 freq bins, baseline = frame 0
        let mut power = [1.0, 2.0, 2.0, 4.0, 10.0, 20.0 as Float];
        baseline_normalize(&mut power, 3, 2, 0, 1, BaselineMode::Db).unwrap();

        // Frame 0: 10*log10(1/1)=0, 10*log10(2/2)=0
        assert!(power[0].abs() < 1e-10);
        assert!(power[1].abs() < 1e-10);
        // Frame 1: 10*log10(2/1)=~3.01, 10*log10(4/2)=~3.01
        let expected_3db = 10.0 * float::log10(2.0);
        assert!((power[2] - expected_3db).abs() < 1e-10);
        assert!((power[3] - expected_3db).abs() < 1e-10);
        // Frame 2: 10*log10(10/1)=10, 10*log10(20/2)=10
        assert!((power[4] - 10.0).abs() < 1e-10);
        assert!((power[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_baseline_is_zero() {
        // Constant power in baseline -> 0 dB during baseline
        let mut power = [5.0, 5.0, 5.0, 5.0, 10.0, 10.0 as Float];
        baseline_normalize(&mut power, 3, 2, 0, 2, BaselineMode::Db).unwrap();
        assert!(power[0].abs() < 1e-10);
        assert!(power[1].abs() < 1e-10);
        assert!(power[2].abs() < 1e-10);
        assert!(power[3].abs() < 1e-10);
    }

    #[test]
    fn test_db_multi_frame_baseline() {
        // baseline = frames 0..2, mean = (1+3)/2 = 2
        let mut power = [1.0, 3.0, 2.0 as Float];
        baseline_normalize(&mut power, 3, 1, 0, 2, BaselineMode::Db).unwrap();
        // frame 0: 10*log10(1/2) = 10*log10(0.5) ~ -3.01
        // frame 1: 10*log10(3/2) = 10*log10(1.5) ~ 1.76
        // frame 2: 10*log10(2/2) = 0
        assert!((power[0] - 10.0 * float::log10(0.5)).abs() < 1e-10);
        assert!((power[1] - 10.0 * float::log10(1.5)).abs() < 1e-10);
        assert!(power[2].abs() < 1e-10);
    }

    // =========================================================================
    // BaselineMode::Zscore tests
    // =========================================================================

    #[test]
    fn test_zscore_known_values() {
        // baseline = frames 0..2, values [2.0, 4.0] -> mean=3, std=1
        // frame 2: value 5.0 -> (5-3)/1 = 2.0
        let mut power = [2.0, 4.0, 5.0 as Float];
        baseline_normalize(&mut power, 3, 1, 0, 2, BaselineMode::Zscore).unwrap();
        assert!((power[0] - -1.0).abs() < 1e-10);
        assert!((power[1] - 1.0).abs() < 1e-10);
        assert!((power[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_constant_baseline_error() {
        // Constant baseline -> std=0 -> error
        let mut power = [3.0, 3.0, 5.0 as Float];
        let result = baseline_normalize(&mut power, 3, 1, 0, 2, BaselineMode::Zscore);
        assert_eq!(result, Err(ErspError::ZeroBaselineStd));
    }

    // =========================================================================
    // BaselineMode::Percentage tests
    // =========================================================================

    #[test]
    fn test_percentage_known_values() {
        // baseline = frame 0, mean = 2.0
        // frame 1: (4-2)/2*100 = 100%
        let mut power = [2.0, 4.0 as Float];
        baseline_normalize(&mut power, 2, 1, 0, 1, BaselineMode::Percentage).unwrap();
        assert!(power[0].abs() < 1e-10); // (2-2)/2*100 = 0
        assert!((power[1] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentage_decrease() {
        // frame 1 has half the baseline power -> -50%
        let mut power = [4.0, 2.0 as Float];
        baseline_normalize(&mut power, 2, 1, 0, 1, BaselineMode::Percentage).unwrap();
        assert!((power[1] - -50.0).abs() < 1e-10);
    }

    // =========================================================================
    // BaselineMode::LogRatio tests
    // =========================================================================

    #[test]
    fn test_logratio_known_values() {
        // baseline = frame 0, mean = 1.0
        // frame 1: log10(10/1) = 1.0
        let mut power = [1.0, 10.0 as Float];
        baseline_normalize(&mut power, 2, 1, 0, 1, BaselineMode::LogRatio).unwrap();
        assert!(power[0].abs() < 1e-10);
        assert!((power[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_logratio_is_db_divided_by_10() {
        let mut power_db = [1.0, 2.0, 4.0 as Float];
        let mut power_lr = [1.0, 2.0, 4.0 as Float];
        baseline_normalize(&mut power_db, 3, 1, 0, 1, BaselineMode::Db).unwrap();
        baseline_normalize(&mut power_lr, 3, 1, 0, 1, BaselineMode::LogRatio).unwrap();
        for i in 0..3 {
            assert!((power_db[i] - power_lr[i] * 10.0).abs() < 1e-10);
        }
    }

    // =========================================================================
    // Validation tests
    // =========================================================================

    #[test]
    fn test_empty_power_error() {
        let mut power: [Float; 0] = [];
        assert_eq!(
            baseline_normalize(&mut power, 0, 0, 0, 0, BaselineMode::Db),
            Err(ErspError::EmptyPower)
        );
    }

    #[test]
    fn test_invalid_baseline_range() {
        let mut power = [1.0 as Float; 4];
        assert_eq!(
            baseline_normalize(&mut power, 2, 2, 1, 1, BaselineMode::Db),
            Err(ErspError::InvalidBaselineRange)
        );
        assert_eq!(
            baseline_normalize(&mut power, 2, 2, 2, 1, BaselineMode::Db),
            Err(ErspError::InvalidBaselineRange)
        );
    }

    #[test]
    fn test_baseline_out_of_bounds() {
        let mut power = [1.0 as Float; 4];
        assert_eq!(
            baseline_normalize(&mut power, 2, 2, 0, 3, BaselineMode::Db),
            Err(ErspError::BaselineOutOfBounds)
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut power = [1.0 as Float; 5]; // 5 != 2*2
        assert_eq!(
            baseline_normalize(&mut power, 2, 2, 0, 1, BaselineMode::Db),
            Err(ErspError::DimensionMismatch)
        );
    }

    #[test]
    fn test_zero_baseline_mean_error() {
        let mut power = [0.0, 1.0 as Float];
        assert_eq!(
            baseline_normalize(&mut power, 2, 1, 0, 1, BaselineMode::Db),
            Err(ErspError::ZeroBaselineMean)
        );
    }

    #[test]
    fn test_single_frame_baseline() {
        let mut power = [2.0, 4.0 as Float];
        baseline_normalize(&mut power, 2, 1, 0, 1, BaselineMode::Db).unwrap();
        assert!(power[0].abs() < 1e-10);
        assert!((power[1] - 10.0 * float::log10(2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_full_signal_baseline() {
        // All frames are baseline
        let mut power = [2.0, 2.0, 2.0 as Float];
        baseline_normalize(&mut power, 3, 1, 0, 3, BaselineMode::Db).unwrap();
        for &v in &power {
            assert!(v.abs() < 1e-10);
        }
    }

    // =========================================================================
    // epoch_average tests
    // =========================================================================

    #[test]
    fn test_epoch_average_single_epoch() {
        let epochs = [1.0, 2.0, 3.0, 4.0 as Float];
        let mut output = [0.0 as Float; 4];
        epoch_average(&epochs, 1, 2, 2, &mut output).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-10);
        assert!((output[1] - 2.0).abs() < 1e-10);
        assert!((output[2] - 3.0).abs() < 1e-10);
        assert!((output[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_epoch_average_multi_epoch() {
        let epochs = [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0 as Float];
        let mut output = [0.0 as Float; 4];
        epoch_average(&epochs, 2, 2, 2, &mut output).unwrap();
        assert!((output[0] - 2.0).abs() < 1e-10);
        assert!((output[1] - 3.0).abs() < 1e-10);
        assert!((output[2] - 4.0).abs() < 1e-10);
        assert!((output[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_epoch_average_no_epochs() {
        let epochs: [Float; 0] = [];
        let mut output = [0.0 as Float; 4];
        assert_eq!(
            epoch_average(&epochs, 0, 2, 2, &mut output),
            Err(ErspError::NoEpochs)
        );
    }

    #[test]
    fn test_epoch_average_dimension_mismatch() {
        let epochs = [1.0, 2.0, 3.0 as Float]; // 3 != 2*2
        let mut output = [0.0 as Float; 4];
        assert_eq!(
            epoch_average(&epochs, 1, 2, 2, &mut output),
            Err(ErspError::EpochDimensionMismatch)
        );
    }

    #[test]
    fn test_epoch_average_output_mismatch() {
        let epochs = [1.0, 2.0, 3.0, 4.0 as Float];
        let mut output = [0.0 as Float; 3]; // 3 != 2*2
        assert_eq!(
            epoch_average(&epochs, 1, 2, 2, &mut output),
            Err(ErspError::OutputDimensionMismatch)
        );
    }

    // =========================================================================
    // Combined workflow test
    // =========================================================================

    #[test]
    fn test_average_then_normalize() {
        // 2 epochs, 4 frames x 1 freq, baseline = frames 0..2
        // Epoch 0: [1, 1, 2, 4], Epoch 1: [3, 3, 6, 12]
        let epochs = [1.0, 1.0, 2.0, 4.0, 3.0, 3.0, 6.0, 12.0 as Float];
        let mut output = [0.0 as Float; 4];
        epoch_average(&epochs, 2, 4, 1, &mut output).unwrap();
        // Mean: [2, 2, 4, 8]
        assert!((output[0] - 2.0).abs() < 1e-10);
        assert!((output[1] - 2.0).abs() < 1e-10);
        assert!((output[2] - 4.0).abs() < 1e-10);
        assert!((output[3] - 8.0).abs() < 1e-10);

        baseline_normalize(&mut output, 4, 1, 0, 2, BaselineMode::Db).unwrap();
        // Baseline mean = (2+2)/2 = 2
        // Frame 0: 10*log10(2/2) = 0
        // Frame 2: 10*log10(4/2) = ~3.01
        // Frame 3: 10*log10(8/2) = 10*log10(4) = ~6.02
        assert!(output[0].abs() < 1e-10);
        assert!(output[1].abs() < 1e-10);
        assert!((output[2] - 10.0 * float::log10(2.0)).abs() < 1e-10);
        assert!((output[3] - 10.0 * float::log10(4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_freq_bins() {
        // 3 frames x 2 freq bins, baseline = frames 0..1
        // freq0: [1, 2, 4], baseline_mean=1
        // freq1: [10, 20, 40], baseline_mean=10
        let mut power = [1.0, 10.0, 2.0, 20.0, 4.0, 40.0 as Float];
        baseline_normalize(&mut power, 3, 2, 0, 1, BaselineMode::Db).unwrap();

        // Both freqs should have same dB pattern since ratio is same
        assert!(power[0].abs() < 1e-10);
        assert!(power[1].abs() < 1e-10);
        assert!((power[2] - 10.0 * float::log10(2.0)).abs() < 1e-10);
        assert!((power[3] - 10.0 * float::log10(2.0)).abs() < 1e-10);
        assert!((power[4] - 10.0 * float::log10(4.0)).abs() < 1e-10);
        assert!((power[5] - 10.0 * float::log10(4.0)).abs() < 1e-10);
    }
}
