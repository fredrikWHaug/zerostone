//! Cross-platform determinism tests for the Zerostone spike sorter.
//!
//! These tests verify that the sorting pipeline produces bit-identical output
//! regardless of platform (macOS ARM64, Linux x86_64, etc.) by using a custom
//! LCG PRNG (no rand crate) and asserting on a byte-level fingerprint checksum.

use zerostone::probe::ProbeLayout;
use zerostone::sorter::{sort_multichannel, DetectionMode, SortConfig};
use zerostone::spike_sort::MultiChannelEvent;

/// LCG PRNG: Knuth's constants. Returns values in [-1.0, 1.0).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    }
}

/// Generate deterministic 4-channel synthetic recording with injected spikes.
fn generate_test_data() -> Vec<[f64; 4]> {
    let n = 10000;
    let mut rng = Lcg::new(12345);
    let mut data = Vec::with_capacity(n);

    // Fill with LCG noise
    for _ in 0..n {
        let sample = [
            rng.next_f64(),
            rng.next_f64(),
            rng.next_f64(),
            rng.next_f64(),
        ];
        data.push(sample);
    }

    // Inject 5 clear negative spikes on channel 0 at known positions
    let spike_positions: [usize; 5] = [1000, 3000, 5000, 7000, 9000];
    for &pos in &spike_positions {
        data[pos][0] = -15.0;
        // Add realistic spike shape: repolarization
        if pos + 1 < n {
            data[pos + 1][0] = -10.0;
        }
        if pos + 2 < n {
            data[pos + 2][0] = 5.0;
        }
    }

    data
}

/// Run sort_multichannel on the given data and return (events, labels, n_spikes, n_clusters).
fn run_sort(data: &mut [[f64; 4]]) -> (Vec<MultiChannelEvent>, Vec<usize>, usize, usize) {
    let config = SortConfig {
        detection_mode: DetectionMode::Amplitude,
        ..SortConfig::default()
    };
    let probe = ProbeLayout::<4>::linear(25.0);

    let n = data.len();
    let mut scratch = vec![0.0f64; n];
    let mut events = vec![
        MultiChannelEvent {
            sample: 0,
            channel: 0,
            amplitude: 0.0,
        };
        500
    ];
    // W=48, K=4
    let mut waveforms = vec![[0.0f64; 48]; 500];
    let mut features = vec![[0.0f64; 4]; 500];
    let mut labels = vec![0usize; 500];

    // C=4, CM=16, W=48, K=4, WM=2304, N=32
    let result = sort_multichannel::<4, 16, 48, 4, 2304, 32>(
        &config,
        &probe,
        data,
        &mut scratch,
        &mut events,
        &mut waveforms,
        &mut features,
        &mut labels,
    )
    .expect("sort_multichannel failed");

    (events, labels, result.n_spikes, result.n_clusters)
}

/// Build a fingerprint from sort output: n_spikes(u64) + n_clusters(u64) + per-spike(sample u64 + label u64).
fn build_fingerprint(
    events: &[MultiChannelEvent],
    labels: &[usize],
    n_spikes: usize,
    n_clusters: usize,
) -> Vec<u8> {
    let mut fp = Vec::new();
    fp.extend_from_slice(&(n_spikes as u64).to_le_bytes());
    fp.extend_from_slice(&(n_clusters as u64).to_le_bytes());
    for i in 0..n_spikes {
        fp.extend_from_slice(&(events[i].sample as u64).to_le_bytes());
        fp.extend_from_slice(&(labels[i] as u64).to_le_bytes());
    }
    fp
}

/// Simple checksum: sum of all bytes as u64.
fn byte_checksum(data: &[u8]) -> u64 {
    data.iter().map(|&b| b as u64).sum()
}

/// Test 1: Determinism fingerprint -- sorting produces a known checksum.
#[test]
fn determinism_fingerprint() {
    let mut data = generate_test_data();
    let (events, labels, n_spikes, n_clusters) = run_sort(&mut data);

    let fp = build_fingerprint(&events, &labels, n_spikes, n_clusters);
    let checksum = byte_checksum(&fp);

    println!("determinism_fingerprint: n_spikes={n_spikes}, n_clusters={n_clusters}, checksum={checksum}");

    assert_eq!(
        checksum, EXPECTED_CHECKSUM,
        "determinism checksum changed: got {checksum}, expected {EXPECTED_CHECKSUM}"
    );
}

/// Test 2: Byte-exact reproducibility -- sorting the same data twice yields identical output.
#[test]
fn determinism_identical_runs() {
    let mut data1 = generate_test_data();
    let (events1, labels1, n_spikes1, n_clusters1) = run_sort(&mut data1);
    let fp1 = build_fingerprint(&events1, &labels1, n_spikes1, n_clusters1);

    let mut data2 = generate_test_data();
    let (events2, labels2, n_spikes2, n_clusters2) = run_sort(&mut data2);
    let fp2 = build_fingerprint(&events2, &labels2, n_spikes2, n_clusters2);

    assert_eq!(
        fp1, fp2,
        "two identical runs produced different fingerprints (n_spikes: {} vs {}, n_clusters: {} vs {})",
        n_spikes1, n_spikes2, n_clusters1, n_clusters2
    );
}

// Placeholder -- will be updated after first run.
const EXPECTED_CHECKSUM: u64 = 783;
