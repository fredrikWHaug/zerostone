//! Edge case tests for week 15 spike sorting modules.
//!
//! Tests boundary conditions that could cause panics, overflows, or incorrect
//! behavior in the sorting pipeline, multi-channel detection, template subtraction,
//! comparison metrics, drift estimation, quality metrics, and localization.

use zerostone::drift::{estimate_drift_from_positions, DriftEstimator};
use zerostone::localize::{center_of_mass, center_of_mass_threshold};
use zerostone::metrics::compare_spike_trains;
use zerostone::probe::ProbeLayout;
use zerostone::quality::{isi_violation_rate, silhouette_score};
use zerostone::sorter::{sort_multichannel, SortConfig};
use zerostone::spike_sort::{
    deduplicate_events, detect_spikes_multichannel, MultiChannelEvent, SortError,
};
use zerostone::template_subtract::{PeelResult, TemplateSubtractor};

// ===========================================================================
// Sorting pipeline edge cases (sorter.rs)
// ===========================================================================

#[test]
fn sort_empty_data() {
    let config = SortConfig::default();
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut data: Vec<[f64; 2]> = vec![];
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let mut waveforms = vec![[0.0f64; 32]; 256];
    let mut features = vec![[0.0f64; 3]; 256];
    let mut labels = vec![0usize; 256];

    let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    // Empty data has fewer samples than W=32, should return InsufficientData
    assert!(result.is_err());
}

#[test]
fn sort_single_sample() {
    let config = SortConfig::default();
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut data: Vec<[f64; 2]> = vec![[0.5, -0.3]];
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let mut waveforms = vec![[0.0f64; 32]; 256];
    let mut features = vec![[0.0f64; 3]; 256];
    let mut labels = vec![0usize; 256];

    let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    // 1 sample < W=32, should return InsufficientData
    assert!(result.is_err());
}

#[test]
fn sort_all_zeros() {
    let config = SortConfig::default();
    let probe = ProbeLayout::<2>::linear(25.0);
    // 64 samples of all zeros -- enough for W=32
    let mut data: Vec<[f64; 2]> = vec![[0.0, 0.0]; 64];
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let mut waveforms = vec![[0.0f64; 32]; 256];
    let mut features = vec![[0.0f64; 3]; 256];
    let mut labels = vec![0usize; 256];

    let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    // All-zero data: noise is zero, whitening may fail (EigenFailed)
    // or produce zero spikes -- either outcome is acceptable
    match result {
        Ok(r) => {
            assert_eq!(r.n_spikes, 0);
        }
        Err(_) => {
            // EigenFailed is expected for zero-variance data
        }
    }
}

#[test]
fn sort_very_high_threshold() {
    let config = SortConfig {
        threshold_multiplier: 1000.0,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);
    // Normal-ish data: small values that won't exceed threshold * noise
    let mut data: Vec<[f64; 2]> = (0..128)
        .map(|i| {
            let t = i as f64 * 0.1;
            [libm::sin(t) * 0.5, libm::cos(t) * 0.3]
        })
        .collect();
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let mut waveforms = vec![[0.0f64; 32]; 256];
    let mut features = vec![[0.0f64; 3]; 256];
    let mut labels = vec![0usize; 256];

    let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    match result {
        Ok(r) => {
            assert_eq!(r.n_spikes, 0);
            assert_eq!(r.n_clusters, 0);
        }
        Err(_) => {
            // Acceptable if whitening fails on low-variance sinusoidal data
        }
    }
}

#[test]
fn sort_all_identical_values() {
    let config = SortConfig::default();
    let probe = ProbeLayout::<2>::linear(25.0);
    // Every sample is [1.0, 1.0] -- zero variance
    let mut data: Vec<[f64; 2]> = vec![[1.0, 1.0]; 64];
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let mut waveforms = vec![[0.0f64; 32]; 256];
    let mut features = vec![[0.0f64; 3]; 256];
    let mut labels = vec![0usize; 256];

    let result = sort_multichannel::<2, 4, 32, 3, 1024, 8>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    // Zero variance -> whitening fails or no spikes found
    match result {
        Ok(r) => {
            assert_eq!(r.n_spikes, 0);
        }
        Err(_) => {
            // EigenFailed expected for constant data
        }
    }
}

// ===========================================================================
// Multi-channel detection edge cases (spike_sort.rs)
// ===========================================================================

#[test]
fn detect_empty_data() {
    let data: Vec<[f64; 2]> = vec![];
    let noise = [1.0, 1.0];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        64
    ];
    let n = detect_spikes_multichannel::<2>(&data, 5.0, &noise, 15, &mut events);
    assert_eq!(n, 0);
}

#[test]
fn detect_single_channel_spike() {
    // 2 channels, spike only on channel 0
    let mut data = vec![[0.0, 0.0]; 100];
    // Inject a large negative spike on channel 0 at sample 50
    data[50] = [-20.0, 0.0];
    data[49] = [-10.0, 0.0];
    data[51] = [-10.0, 0.0];

    let noise = [1.0, 1.0];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        64
    ];
    let n = detect_spikes_multichannel::<2>(&data, 5.0, &noise, 15, &mut events);
    assert!(n >= 1, "Should detect at least 1 spike on channel 0");
    // All detected events should be on channel 0
    for event in events.iter().take(n) {
        assert_eq!(event.channel, 0);
    }
}

#[test]
fn detect_all_above_threshold() {
    // Every sample is a large negative value, so every sample exceeds threshold.
    // Refractory period should limit the number of detections.
    let data: Vec<[f64; 1]> = vec![[-100.0]; 200];
    let noise = [1.0];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        256
    ];
    let refractory = 15;
    let n = detect_spikes_multichannel::<1>(&data, 5.0, &noise, refractory, &mut events);
    // Should not panic, and refractory period limits count
    // Maximum detections: roughly ceil(200 / refractory)
    let max_expected = 200_usize.div_ceil(refractory);
    assert!(
        n <= max_expected + 1,
        "Refractory should limit detections: got {}, max_expected {}",
        n,
        max_expected
    );
}

#[test]
fn dedup_zero_radius() {
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut events = [
        MultiChannelEvent {
            sample: 50,
            channel: 0,
            amplitude: 10.0,
        },
        MultiChannelEvent {
            sample: 50,
            channel: 1,
            amplitude: 8.0,
        },
    ];
    // spatial_radius=0.0 means no channel is within range of another, so no dedup
    let n = deduplicate_events::<2>(&mut events, 2, &probe, 0.0, 5);
    assert_eq!(n, 2, "Zero radius should keep all events");
}

#[test]
fn dedup_single_event() {
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut events = [MultiChannelEvent {
        sample: 50,
        channel: 0,
        amplitude: 10.0,
    }];
    let n = deduplicate_events::<2>(&mut events, 1, &probe, 75.0, 5);
    assert_eq!(n, 1, "Single event should remain");
}

// ===========================================================================
// Template subtraction edge cases (template_subtract.rs)
// ===========================================================================

#[test]
fn peel_no_templates() {
    let sub = TemplateSubtractor::<8, 4>::new(50);
    let mut data = [1.0f64; 20];
    let spike_times = [4usize, 8, 12];
    let mut results = [PeelResult {
        sample: 0,
        template_id: 0,
        amplitude: 0.0,
    }; 8];
    let n = sub.peel(&mut data, &spike_times, 3, 2, &mut results);
    assert_eq!(n, 0, "No templates means no peeling");
}

#[test]
fn peel_empty_signal() {
    let mut sub = TemplateSubtractor::<4, 2>::new(10);
    sub.add_template(&[-1.0, -3.0, -2.0, 0.0], 0.5, 2.0)
        .unwrap();

    // Signal shorter than template window W=4
    let mut data = [0.0f64; 2];
    let spike_times = [0usize];
    let mut results = [PeelResult {
        sample: 0,
        template_id: 0,
        amplitude: 0.0,
    }; 4];
    let n = sub.peel(&mut data, &spike_times, 1, 0, &mut results);
    assert_eq!(n, 0, "Signal shorter than template should return 0");
}

#[test]
fn template_full_capacity() {
    let mut sub = TemplateSubtractor::<4, 3>::new(10);
    // Fill all 3 slots
    assert!(sub.add_template(&[1.0; 4], 0.5, 2.0).is_ok());
    assert!(sub.add_template(&[2.0; 4], 0.5, 2.0).is_ok());
    assert!(sub.add_template(&[3.0; 4], 0.5, 2.0).is_ok());
    assert_eq!(sub.n_templates(), 3);

    // Next one should fail with TemplateFull
    let result = sub.add_template(&[4.0; 4], 0.5, 2.0);
    assert_eq!(result, Err(SortError::TemplateFull));
}

// ===========================================================================
// Metrics edge cases (metrics.rs)
// ===========================================================================

#[test]
fn compare_empty_trains() {
    let gt: [usize; 0] = [];
    let sorted: [usize; 0] = [];
    let m = compare_spike_trains(&gt, &sorted, 5);
    assert_eq!(m.true_positives, 0);
    assert_eq!(m.false_positives, 0);
    assert_eq!(m.false_negatives, 0);
    assert!(m.accuracy.abs() < 1e-12);
    assert!(m.precision.abs() < 1e-12);
    assert!(m.recall.abs() < 1e-12);
}

#[test]
fn compare_single_spike_match() {
    let gt = [100usize];
    let sorted = [100usize];
    let m = compare_spike_trains(&gt, &sorted, 0);
    assert_eq!(m.true_positives, 1);
    assert_eq!(m.false_positives, 0);
    assert_eq!(m.false_negatives, 0);
    assert!((m.accuracy - 1.0).abs() < 1e-12);
    assert!((m.precision - 1.0).abs() < 1e-12);
    assert!((m.recall - 1.0).abs() < 1e-12);
}

#[test]
fn compare_no_matches() {
    // GT and sorted trains with completely disjoint times
    let gt = [100usize, 200, 300];
    let sorted = [500usize, 600, 700];
    let m = compare_spike_trains(&gt, &sorted, 5);
    assert_eq!(m.true_positives, 0);
    assert_eq!(m.false_positives, 3);
    assert_eq!(m.false_negatives, 3);
    assert!(m.accuracy.abs() < 1e-12);
}

// ===========================================================================
// Drift edge cases (drift.rs)
// ===========================================================================

#[test]
fn drift_estimator_no_spikes() {
    let mut est = DriftEstimator::<8>::new(1000);
    est.fit();
    assert!(!est.is_fitted());
    assert!(est.slope().abs() < 1e-15);
    assert!(est.intercept().abs() < 1e-15);
    assert_eq!(est.n_bins_used(), 0);
    // estimate_drift should return 0 when not fitted
    assert!(est.estimate_drift(5000).abs() < 1e-15);
}

#[test]
fn drift_single_spike() {
    let mut est = DriftEstimator::<8>::new(1000);
    est.add_spike(500, 100.0);
    est.fit();
    // Single bin -> cannot fit linear regression (need >= 2 bins)
    assert!(!est.is_fitted());
    assert!(est.slope().abs() < 1e-15);
}

#[test]
fn drift_batch_empty() {
    let result = estimate_drift_from_positions(&[], &[], 1000, 8);
    assert!(result.is_none());
}

#[test]
fn drift_batch_single_bin() {
    // All spikes in the same bin -> only 1 data point -> None
    let samples = [100, 200, 300];
    let positions = [50.0, 55.0, 60.0];
    let result = estimate_drift_from_positions(&samples, &positions, 1000, 8);
    assert!(result.is_none());
}

// ===========================================================================
// Quality edge cases (quality.rs)
// ===========================================================================

#[test]
fn isi_violation_rate_empty() {
    assert!(isi_violation_rate(&[], 0.001).is_none());
}

#[test]
fn isi_violation_rate_single_spike() {
    assert!(isi_violation_rate(&[0.1], 0.001).is_none());
}

#[test]
fn silhouette_single_cluster_no_inter() {
    // Single cluster with no other cluster to compare to -> None
    let intra = [0.1, 0.2, 0.15];
    let inter: [f64; 0] = [];
    let result = silhouette_score(&intra, &inter);
    assert!(result.is_none());
}

#[test]
fn silhouette_empty_intra() {
    let intra: [f64; 0] = [];
    let inter = [5.0];
    let result = silhouette_score(&intra, &inter);
    assert!(result.is_none());
}

// ===========================================================================
// Localization edge cases (localize.rs)
// ===========================================================================

#[test]
fn com_all_zero_amplitudes() {
    let amps = [0.0, 0.0, 0.0, 0.0];
    let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]];
    let loc = center_of_mass(&amps, &pos);
    assert!(loc[0].abs() < 1e-12);
    assert!(loc[1].abs() < 1e-12);
}

#[test]
fn com_single_channel_nonzero() {
    // 2-channel probe, amplitude only on channel 0
    let amps = [-5.0, 0.0];
    let pos = [[10.0, 20.0], [30.0, 40.0]];
    let loc = center_of_mass(&amps, &pos);
    assert!((loc[0] - 10.0).abs() < 1e-12);
    assert!((loc[1] - 20.0).abs() < 1e-12);
}

#[test]
fn com_threshold_all_below() {
    // All amplitudes below threshold -> None
    let amps = [0.01, 0.02, 0.03];
    let pos = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0]];
    let result = center_of_mass_threshold(&amps, &pos, 1.0);
    assert!(result.is_none());
}

// ===========================================================================
// Template subtraction integration edge cases (sorter.rs)
// ===========================================================================

/// Template subtraction on data with zero detected spikes should be a no-op.
#[test]
fn sort_template_subtract_no_spikes() {
    let config = SortConfig {
        threshold_multiplier: 1000.0, // impossibly high to get 0 spikes
        template_subtract: true,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut data: Vec<[f64; 2]> = (0..256)
        .map(|i| {
            let t = i as f64 * 0.1;
            [libm::sin(t) * 0.5, libm::cos(t) * 0.3]
        })
        .collect();
    let max_ev = 256;
    let mut scratch = vec![0.0f64; 256];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0
        };
        max_ev
    ];
    let mut waveforms = vec![[0.0f64; 48]; max_ev];
    let mut features = vec![[0.0f64; 4]; max_ev];
    let mut labels = vec![0usize; max_ev];

    let result = sort_multichannel::<2, 4, 48, 4, 2304, 32>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    match result {
        Ok(r) => {
            assert_eq!(r.n_spikes, 0);
            assert_eq!(r.n_clusters, 0);
        }
        Err(_) => {
            // Whitening failure on low-variance sinusoidal data is acceptable
        }
    }
}

/// Template subtraction with a single cluster should not panic.
#[test]
fn sort_template_subtract_single_cluster() {
    let config = SortConfig {
        threshold_multiplier: 3.0,
        template_subtract: true,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);
    // Single large spike with noise
    let mut data: Vec<[f64; 2]> = vec![[0.1, -0.1]; 512];
    // Inject one spike-like deflection
    for (i, row) in data.iter_mut().enumerate().take(260).skip(240) {
        let t = (i as f64 - 250.0) / 3.0;
        row[0] = -10.0 * (-0.5 * t * t).exp();
    }
    let max_ev = 256;
    let mut scratch = vec![0.0f64; 512];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0
        };
        max_ev
    ];
    let mut waveforms = vec![[0.0f64; 48]; max_ev];
    let mut features = vec![[0.0f64; 4]; max_ev];
    let mut labels = vec![0usize; max_ev];

    let result = sort_multichannel::<2, 4, 48, 4, 2304, 32>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    if let Ok(r) = result {
        assert!(r.n_clusters <= 32);
        for &l in labels[..r.n_spikes].iter() {
            assert!(l < r.n_clusters.max(1));
        }
    }
}

/// Template subtraction with identical spikes at the same location should not panic.
#[test]
fn sort_template_subtract_identical_spikes() {
    let config = SortConfig {
        threshold_multiplier: 3.0,
        template_subtract: true,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut data: Vec<[f64; 2]> = vec![[0.1, -0.1]; 2048];
    // Inject many identical large spikes (same waveform, well-spaced)
    for spike_idx in 0..10 {
        let center = 200 + spike_idx * 100;
        for offset in 0..20 {
            let t = (offset as f64 - 10.0) / 3.0;
            let val = -20.0 * (-0.5 * t * t).exp();
            if center + offset < 2048 {
                data[center + offset][0] += val;
            }
        }
    }
    let max_ev = 256;
    let mut scratch = vec![0.0f64; 2048];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0
        };
        max_ev
    ];
    let mut waveforms = vec![[0.0f64; 48]; max_ev];
    let mut features = vec![[0.0f64; 4]; max_ev];
    let mut labels = vec![0usize; max_ev];

    let result = sort_multichannel::<2, 4, 48, 4, 2304, 32>(
        &config,
        &probe,
        &mut data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    );
    if let Ok(r) = result {
        // Should detect spikes and assign valid labels
        if r.n_spikes > 0 {
            assert!(r.n_clusters >= 1);
            for &l in labels[..r.n_spikes].iter() {
                assert!(l < r.n_clusters);
            }
        }
    }
}

/// Compare sort results with and without template subtraction.
/// Template subtraction should not reduce the base spike count significantly.
#[test]
fn sort_template_vs_no_template() {
    let probe = ProbeLayout::<2>::linear(25.0);
    let mut data: Vec<[f64; 2]> = vec![[0.1, -0.1]; 1024];
    // Inject a few spikes
    for spike_idx in 0..5 {
        let center = 100 + spike_idx * 100;
        for offset in 0..20 {
            let t = (offset as f64 - 10.0) / 3.0;
            let val = -10.0 * (-0.5 * t * t).exp();
            if center + offset < 1024 {
                data[center + offset][0] += val;
            }
        }
    }

    let max_ev = 256;

    // Without template subtraction
    let config_no = SortConfig {
        threshold_multiplier: 4.0,
        template_subtract: false,
        ..SortConfig::default()
    };
    let mut data_no = data.clone();
    let mut scratch = vec![0.0f64; 1024];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0
        };
        max_ev
    ];
    let mut waveforms = vec![[0.0f64; 48]; max_ev];
    let mut features = vec![[0.0f64; 4]; max_ev];
    let mut labels_no = vec![0usize; max_ev];

    let result_no = sort_multichannel::<2, 4, 48, 4, 2304, 32>(
        &config_no,
        &probe,
        &mut data_no,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels_no,
    );

    // With template subtraction
    let config_yes = SortConfig {
        threshold_multiplier: 4.0,
        template_subtract: true,
        ..SortConfig::default()
    };
    let mut data_yes = data.clone();
    let mut scratch2 = vec![0.0f64; 1024];
    let mut events2 = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0
        };
        max_ev
    ];
    let mut waveforms2 = vec![[0.0f64; 48]; max_ev];
    let mut features2 = vec![[0.0f64; 4]; max_ev];
    let mut labels_yes = vec![0usize; max_ev];

    let result_yes = sort_multichannel::<2, 4, 48, 4, 2304, 32>(
        &config_yes,
        &probe,
        &mut data_yes,
        &mut scratch2,
        &mut events2,
        &mut waveforms2,
        &mut features2,
        &mut labels_yes,
    );

    // Both should succeed or both fail (same data)
    match (result_no, result_yes) {
        (Ok(r_no), Ok(r_yes)) => {
            // Template subtraction should find >= as many spikes (it can find
            // additional spikes in residual)
            assert!(
                r_yes.n_spikes >= r_no.n_spikes,
                "template subtraction should find >= base spikes: {} vs {}",
                r_yes.n_spikes,
                r_no.n_spikes
            );
        }
        _ => {
            // Both fail is acceptable
        }
    }
}
