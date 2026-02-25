//! Event-Related Potential (ERP) analysis primitives.
//!
//! Provides algorithms for ERP analysis including epoch averaging and xDAWN
//! spatial filtering. Essential for P300 spellers, error-related potentials,
//! and other ERP-based BCI paradigms.
//!
//! # xDAWN Algorithm
//!
//! xDAWN (Rivet et al., 2009) is a supervised spatial filtering method that
//! maximizes the signal-to-signal-plus-noise ratio (SSNR) of event-related
//! potentials. It solves the generalized eigenvalue problem:
//!
//! ```text
//! Σ_evoked × w = λ × Σ_signal × w
//! ```
//!
//! where Σ_evoked is the covariance of averaged evoked responses and Σ_signal
//! is the covariance of all raw signal data. Eigenvectors with largest
//! eigenvalues are spatial filters that maximize ERP signal vs background noise.
//!
//! # Example
//!
//! ```
//! use zerostone::erp::{epoch_average, xdawn_filters, apply_spatial_filter};
//!
//! // Average ERP across target trials
//! let target_epochs = vec![
//!     &[[1.0, 2.0]; 10][..],  // Trial 1: 10 samples × 2 channels
//!     &[[1.5, 2.5]; 10][..],  // Trial 2
//! ];
//! let mut avg = [[0.0; 2]; 10];
//! epoch_average(&target_epochs, &mut avg);
//!
//! // Learn xDAWN spatial filters (2 channels -> 1 filter)
//! let nontarget_epochs = vec![
//!     &[[0.1, 0.2]; 10][..],
//!     &[[0.15, 0.25]; 10][..],
//! ];
//! let mut evoked_workspace = [[0.0; 2]; 10];
//! let mut filters = [[0.0; 2]; 1];  // 1 filter × 2 channels
//! xdawn_filters::<2, 4, 1>(&target_epochs, &nontarget_epochs, &mut evoked_workspace, &mut filters, 1e-6, 30, 1e-10).unwrap();
//!
//! // Apply filter to new epoch
//! let new_epoch = [[1.2, 2.2]; 10];
//! let mut filtered = [[0.0; 1]; 10];  // 10 samples × 1 component
//! apply_spatial_filter(&new_epoch, &filters, &mut filtered);
//! ```
//!
//! # References
//!
//! - Rivet et al. (2009): "xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain-Computer Interface"

use crate::linalg::{generalized_eigen, Matrix};
use crate::stats::OnlineCov;

/// Errors that can occur during ERP operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErpError {
    /// Not enough epochs to compute average
    InsufficientData,

    /// Filters not yet computed
    FiltersNotReady,

    /// Eigenvalue decomposition failed
    EigenDecompositionFailed,

    /// Matrix is not positive definite
    NotPositiveDefinite,

    /// Numerical instability detected
    NumericalInstability,
}

/// Compute average ERP across multiple epochs.
///
/// This is the foundation of all ERP analysis - averaging time-locked trials
/// to extract the event-related response while canceling out unrelated noise.
///
/// # Type Parameters
///
/// * `C` - Number of channels
///
/// # Arguments
///
/// * `epochs` - Slice of references to epochs, each epoch is `&[[f64; C]]` (samples × channels)
/// * `output` - Output array to store averaged epoch (samples × channels)
///
/// # Example
///
/// ```
/// use zerostone::erp::epoch_average;
///
/// let epochs = vec![
///     &[[1.0, 2.0], [3.0, 4.0]][..],  // Trial 1: 2 samples × 2 channels
///     &[[1.5, 2.5], [3.5, 4.5]][..],  // Trial 2
/// ];
/// let mut avg = [[0.0; 2]; 2];
/// epoch_average(&epochs, &mut avg);
///
/// // Average of [1.0, 1.5] = 1.25, [2.0, 2.5] = 2.25, etc.
/// assert!((avg[0][0] - 1.25).abs() < 1e-10);
/// assert!((avg[0][1] - 2.25).abs() < 1e-10);
/// ```
pub fn epoch_average<const C: usize>(epochs: &[&[[f64; C]]], output: &mut [[f64; C]]) {
    if epochs.is_empty() {
        return;
    }

    let n_samples = output.len();
    let n_epochs = epochs.len() as f64;

    // Zero output
    for sample in output.iter_mut() {
        for ch in sample.iter_mut() {
            *ch = 0.0;
        }
    }

    // Accumulate
    for epoch in epochs.iter() {
        for (t, sample) in epoch.iter().enumerate().take(n_samples) {
            for (ch, &value) in sample.iter().enumerate() {
                output[t][ch] += value;
            }
        }
    }

    // Divide by number of epochs
    for sample in output.iter_mut() {
        for ch in sample.iter_mut() {
            *ch /= n_epochs;
        }
    }
}

/// Learn xDAWN spatial filters from labeled epochs.
///
/// Computes spatial filters that maximize the ratio of evoked response power
/// to overall signal power. The top `F` eigenvectors become the spatial filters.
///
/// # Type Parameters
///
/// * `C` - Number of channels
/// * `M` - Matrix size (must equal C × C)
/// * `F` - Number of spatial filters to extract (F ≤ C)
///
/// # Arguments
///
/// * `target_epochs` - Target/attended stimulus epochs (e.g., P300 targets)
/// * `nontarget_epochs` - Non-target/standard stimulus epochs
/// * `evoked_workspace` - Workspace for averaged epoch (same length as one epoch)
/// * `filters` - Output filters array (F filters × C channels), stored row-major
/// * `regularization` - Regularization parameter for covariance matrices (e.g., 1e-6)
/// * `max_iters` - Max iterations for eigenvalue decomposition (e.g., 30)
/// * `tol` - Convergence tolerance for eigenvalue decomposition (e.g., 1e-10)
///
/// # Returns
///
/// Ok(()) on success, or ErpError on failure.
///
/// # Example
///
/// ```
/// use zerostone::erp::xdawn_filters;
///
/// // Synthetic P300: target has larger amplitude than nontarget
/// let target_epochs = [
///     &[[2.0, 3.0]; 50][..],   // 50 samples × 2 channels
///     &[[2.1, 3.1]; 50][..],
/// ];
/// let nontarget_epochs = [
///     &[[0.5, 0.6]; 50][..],
///     &[[0.4, 0.5]; 50][..],
/// ];
///
/// let mut evoked_workspace = [[0.0; 2]; 50];
/// let mut filters = [[0.0; 2]; 1];  // Extract 1 spatial filter
/// xdawn_filters::<2, 4, 1>(&target_epochs, &nontarget_epochs, &mut evoked_workspace, &mut filters, 1e-6, 30, 1e-10).unwrap();
///
/// // Filter should emphasize the channels with target activity
/// assert!(filters[0][0].abs() > 0.1 || filters[0][1].abs() > 0.1);
/// ```
pub fn xdawn_filters<const C: usize, const M: usize, const F: usize>(
    target_epochs: &[&[[f64; C]]],
    nontarget_epochs: &[&[[f64; C]]],
    evoked_workspace: &mut [[f64; C]],
    filters: &mut [[f64; C]; F],
    regularization: f64,
    max_iters: usize,
    tol: f64,
) -> Result<(), ErpError> {
    if target_epochs.is_empty() || nontarget_epochs.is_empty() {
        return Err(ErpError::InsufficientData);
    }

    // Step 1: Compute averaged evoked response for target epochs
    epoch_average(target_epochs, evoked_workspace);

    // Step 2: Compute evoked response covariance matrix
    // Concatenate evoked response time samples as observations
    let mut cov_evoked: OnlineCov<C, M> = OnlineCov::new();
    for sample in evoked_workspace.iter() {
        cov_evoked.update(sample);
    }

    if cov_evoked.count() < 2 {
        return Err(ErpError::InsufficientData);
    }

    let evoked_cov = cov_evoked.covariance();
    let evoked_cov_mat: Matrix<C, M> = Matrix::from_array(evoked_cov);

    // Step 3: Compute signal covariance from all epochs (target + nontarget)
    let mut cov_signal: OnlineCov<C, M> = OnlineCov::new();

    for epoch in target_epochs.iter() {
        for sample in epoch.iter() {
            cov_signal.update(sample);
        }
    }

    for epoch in nontarget_epochs.iter() {
        for sample in epoch.iter() {
            cov_signal.update(sample);
        }
    }

    if cov_signal.count() < 2 {
        return Err(ErpError::InsufficientData);
    }

    let signal_cov = cov_signal.covariance();
    let signal_cov_mat: Matrix<C, M> = Matrix::from_array(signal_cov);

    // Step 4: Solve generalized eigenvalue problem: Σ_evoked × w = λ × Σ_signal × w
    let eigen = generalized_eigen(
        &evoked_cov_mat,
        &signal_cov_mat,
        regularization,
        max_iters,
        tol,
    )
    .map_err(|_| ErpError::EigenDecompositionFailed)?;

    // Step 5: Extract top F eigenvectors as spatial filters
    // Eigenvectors are stored as columns in the matrix, sorted by descending eigenvalues
    #[allow(clippy::needless_range_loop)]
    for f in 0..F {
        for c in 0..C {
            filters[f][c] = eigen.eigenvectors.get(c, f);
        }
    }

    Ok(())
}

/// Apply spatial filters to an epoch, reducing channels to components.
///
/// Performs projection: output\[t]\[f] = sum_c(epoch\[t]\[c] × filters\[f]\[c])
///
/// # Type Parameters
///
/// * `C` - Number of input channels
/// * `F` - Number of spatial filters (output components)
///
/// # Arguments
///
/// * `epoch` - Input epoch (samples × channels)
/// * `filters` - Spatial filters (F filters × C channels), row-major
/// * `output` - Output filtered epoch (samples × F components)
///
/// # Example
///
/// ```
/// use zerostone::erp::apply_spatial_filter;
///
/// let epoch = [[1.0, 2.0], [3.0, 4.0]];  // 2 samples × 2 channels
/// let filters = [[0.5, 0.5]];             // 1 filter averaging both channels
/// let mut output = [[0.0]; 2];            // 2 samples × 1 component
///
/// apply_spatial_filter(&epoch, &filters, &mut output);
///
/// // output[0][0] = 1.0*0.5 + 2.0*0.5 = 1.5
/// // output[1][0] = 3.0*0.5 + 4.0*0.5 = 3.5
/// assert!((output[0][0] - 1.5).abs() < 1e-10);
/// assert!((output[1][0] - 3.5).abs() < 1e-10);
/// ```
pub fn apply_spatial_filter<const C: usize, const F: usize>(
    epoch: &[[f64; C]],
    filters: &[[f64; C]; F],
    output: &mut [[f64; F]],
) {
    let n_samples = epoch.len();

    for t in 0..n_samples {
        for f in 0..F {
            let mut sum = 0.0;
            for c in 0..C {
                sum += epoch[t][c] * filters[f][c];
            }
            output[t][f] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_epoch_average_basic() {
        let epochs = vec![&[[1.0, 2.0], [3.0, 4.0]][..], &[[1.5, 2.5], [3.5, 4.5]][..]];
        let mut avg = [[0.0; 2]; 2];
        epoch_average(&epochs, &mut avg);

        assert!((avg[0][0] - 1.25).abs() < 1e-10);
        assert!((avg[0][1] - 2.25).abs() < 1e-10);
        assert!((avg[1][0] - 3.25).abs() < 1e-10);
        assert!((avg[1][1] - 4.25).abs() < 1e-10);
    }

    #[test]
    fn test_epoch_average_single_epoch() {
        let epochs = vec![&[[1.0, 2.0], [3.0, 4.0]][..]];
        let mut avg = [[0.0; 2]; 2];
        epoch_average(&epochs, &mut avg);

        assert!((avg[0][0] - 1.0).abs() < 1e-10);
        assert!((avg[0][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_epoch_average_empty() {
        let epochs: &[&[[f64; 2]]] = &[];
        let mut avg = [[0.0; 2]; 2];
        epoch_average(epochs, &mut avg);

        // Should remain zero
        assert_eq!(avg[0][0], 0.0);
    }

    #[test]
    fn test_apply_spatial_filter_basic() {
        let epoch = [[1.0, 2.0], [3.0, 4.0]];
        let filters = [[0.5, 0.5]]; // Average both channels
        let mut output = [[0.0]; 2];

        apply_spatial_filter(&epoch, &filters, &mut output);

        assert!((output[0][0] - 1.5).abs() < 1e-10);
        assert!((output[1][0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_spatial_filter_multi_filter() {
        let epoch = [[1.0, 2.0]];
        let filters = [
            [1.0, 0.0], // Select channel 1
            [0.0, 1.0], // Select channel 2
        ];
        let mut output = [[0.0; 2]; 1];

        apply_spatial_filter(&epoch, &filters, &mut output);

        assert!((output[0][0] - 1.0).abs() < 1e-10);
        assert!((output[0][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_xdawn_filters_synthetic_p300() {
        // Create synthetic P300 data: target has stronger signal than nontarget
        // 2 channels, 50 time samples
        const N_SAMPLES: usize = 50;

        // Target: strong evoked response (amplitude 2.0)
        let mut target1 = [[0.0; 2]; N_SAMPLES];
        let mut target2 = [[0.0; 2]; N_SAMPLES];
        for t in 0..N_SAMPLES {
            let phase = (t as f64 / N_SAMPLES as f64) * 2.0 * core::f64::consts::PI;
            let signal = 2.0 * libm::sin(phase);
            target1[t] = [signal, signal * 1.2];
            target2[t] = [signal * 0.9, signal * 1.1];
        }

        // Nontarget: weak/no evoked response (amplitude ~0.5)
        let mut nontarget1 = [[0.0; 2]; N_SAMPLES];
        let mut nontarget2 = [[0.0; 2]; N_SAMPLES];
        for t in 0..N_SAMPLES {
            nontarget1[t] = [0.1 * (t as f64 / 10.0), 0.12 * (t as f64 / 10.0)];
            nontarget2[t] = [0.09 * (t as f64 / 10.0), 0.11 * (t as f64 / 10.0)];
        }

        let target_epochs = vec![&target1[..], &target2[..]];
        let nontarget_epochs = vec![&nontarget1[..], &nontarget2[..]];

        // Learn 1 spatial filter (2 channels -> 1 component)
        let mut evoked_workspace = [[0.0; 2]; N_SAMPLES];
        let mut filters = [[0.0; 2]; 1];
        #[allow(clippy::needless_borrow)]
        let result = xdawn_filters::<2, 4, 1>(
            &target_epochs,
            &nontarget_epochs,
            &mut evoked_workspace,
            &mut filters,
            1e-6,
            30,
            1e-10,
        );

        assert!(result.is_ok());

        // Filter should have non-zero weights
        let filter_norm = (filters[0][0].powi(2) + filters[0][1].powi(2)).sqrt();
        assert!(
            filter_norm > 0.1,
            "Filter should have significant magnitude"
        );
    }

    #[test]
    fn test_xdawn_filters_insufficient_data() {
        let target_epochs: &[&[[f64; 2]]] = &[];
        let nontarget_epochs = vec![&[[0.0; 2]; 10][..]];
        let mut evoked_workspace = [[0.0; 2]; 10];
        let mut filters = [[0.0; 2]; 1];

        let result = xdawn_filters::<2, 4, 1>(
            target_epochs,
            &nontarget_epochs,
            &mut evoked_workspace,
            &mut filters,
            1e-6,
            30,
            1e-10,
        );

        assert_eq!(result, Err(ErpError::InsufficientData));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_xdawn_end_to_end() {
        // End-to-end test: average -> learn filters -> apply
        const N_SAMPLES: usize = 30;

        // Target epochs with distinct pattern
        let mut target1 = [[0.0; 2]; N_SAMPLES];
        let mut target2 = [[0.0; 2]; N_SAMPLES];
        for t in 0..N_SAMPLES {
            let val = (t as f64) / 10.0;
            target1[t] = [val, val * 1.5];
            target2[t] = [val * 1.1, val * 1.4];
        }

        // Nontarget epochs with different pattern
        let nontarget1 = [[0.1, 0.15]; N_SAMPLES];
        let nontarget2 = [[0.12, 0.14]; N_SAMPLES];

        let target_epochs = vec![&target1[..], &target2[..]];
        let nontarget_epochs = vec![&nontarget1[..], &nontarget2[..]];

        // Learn filter
        let mut evoked_workspace = [[0.0; 2]; N_SAMPLES];
        let mut filters = [[0.0; 2]; 1];
        #[allow(clippy::needless_borrow)]
        {
            xdawn_filters::<2, 4, 1>(
                &target_epochs,
                &nontarget_epochs,
                &mut evoked_workspace,
                &mut filters,
                1e-6,
                30,
                1e-10,
            )
            .unwrap();
        }

        // Apply to new epoch
        let mut new_epoch = [[0.0; 2]; N_SAMPLES];
        for t in 0..N_SAMPLES {
            new_epoch[t] = [(t as f64) / 10.0, (t as f64) * 1.5 / 10.0];
        }

        let mut output = [[0.0; 1]; N_SAMPLES];
        apply_spatial_filter(&new_epoch, &filters, &mut output);

        // Output should contain projected values
        assert!(output[N_SAMPLES - 1][0].abs() > 0.1);
    }
}
