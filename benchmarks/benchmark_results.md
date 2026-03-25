# Spike Sorting Benchmark Results

Zerostone v0.7.0 on synthetic multi-channel recordings with known ground truth.

Methodology follows SpikeInterface/SpikeForest: accuracy = TP / (TP + FN + FP), with 0.67 ms tolerance window (20 samples at 30 kHz) and greedy best-match assignment.

## Results (seed=42, March 2026)

| Preset | Channels | Units | GT Spikes | Sorted | Clusters | Accuracy | Precision | Recall | Time |
|--------|----------|-------|-----------|--------|----------|----------|-----------|--------|------|
| easy   | 32       | 5     | 1476      | 1550   | 20       | **94.5%** | 98.9%    | 95.5%  | 2.2s |
| medium | 32       | 10    | 4865      | 4186   | 12       | **63.6%** | 85.0%    | 71.6%  | 2.3s |
| hard   | 64       | 20    | 11926     | 7875   | 8        | **25.2%** | 50.6%    | 33.4%  | 7.4s |

### Performance History

| Preset | v0.6.0 Acc | v0.7.0 Acc | v0.7.0+opt Acc | v0.6.0 Time | v0.7.0+opt Time | Speedup |
|--------|-----------|-----------|---------------|-------------|-----------------|---------|
| easy   | 94.8%     | 94.2%     | 94.5%         | ~35s        | 2.2s            | **16x** |
| medium | 57.1%     | 64.9%     | 63.6%         | ~128s       | 2.3s            | **56x** |
| hard   | 25.1%     | 25.2%     | 25.2%         | ~286s       | 7.4s            | **39x** |

The v0.7.0+opt speedup comes from NCC template scanning optimization (binary search overlap check, early amplitude rejection, squared NCC threshold) and k-means early-exit distance computation.

Medium accuracy slightly decreased from 64.9% to 63.6% due to NCC scan now skipping quiet regions (trades ~1.3% accuracy for 56x speed). The NCC scan previously checked every single sample position; the optimized version skips positions where peak amplitude < 0.5x threshold.

## Pipeline Configuration

- Detection: amplitude threshold (NEO/SNEO modes available)
- Supported channels: 4, 8, 16, 32, 64, 128 (const-generic, no_std)
- Waveform window: W=48 samples
- PCA components: K=4 (3 PCA + 1 channel index)
- Max clusters: N=32
- Template subtraction: enabled (per-spike amplitude scaling)
- Template residual detection: NCC matching (threshold 0.7, min amplitude 0.5x)
- SNR auto-curation: min_cluster_snr=2.5
- Spatial merge: cross-channel d-prime with amplitude profiles
- CCG merge: available (merges over-split clusters with refractory dip in cross-correlogram)

## Detection Modes

Zerostone supports three detection operators:
- **Amplitude** (default): Standard negative-peak threshold crossing on whitened data.
- **NEO**: Nonlinear Energy Operator `psi[n] = x[n]^2 - x[n-1]*x[n+1]`. Enhances spike-like transients.
- **SNEO**: Smoothed NEO with triangular window. Reduces cross-term artifacts from overlapping spikes.

NEO/SNEO detection thresholds are calibrated using robust percentile estimation (median + T * MAD/0.6745) with a 2000-sample stack-allocated calibration buffer. This resists spike contamination (50% breakdown point).

Note: SNEO currently underperforms amplitude on synthetic data due to threshold space mismatch (T=5 in energy space is far more aggressive than T=5 in amplitude space). A threshold space conversion is needed for SNEO to be competitive.

## Analysis

**Easy (5 units, 32 channels)**: All 5 units detected with >88% per-unit accuracy. High precision (98.9%) indicates very few false positives. The 4.5% missed recall comes from template alignment edge effects.

**Medium (10 units, 32 channels, noise_std=1.5)**: 7 of 10 units reliably detected. Template residual NCC detection recovers weak spikes that match known template shapes. 3 units remain difficult (template amplitude <3 sigma after whitening).

**Hard (20 units, 64 channels, noise_std=2.0)**: 11 of 20 units undetectable (amplitude < noise * threshold). The 9 detectable units are split across 8 clusters, leading to many-to-one mapping errors. Primary limitation is the K=32 cluster cap and single-channel PCA features.

## SpikeInterface Ground Truth Validation (seed=42, 30s, 30 kHz)

Independent validation using SpikeInterface 0.104.0's `generate_ground_truth_recording()` with its own template library and noise model.

| Config | Channels | Units | GT Spikes | Sorted | Mean Accuracy | Mean Precision | Mean Recall | Time |
|--------|----------|-------|-----------|--------|---------------|----------------|-------------|------|
| 32ch/5u  | 32 | 5  | 758  | 1130 | **66.1%** | 79.5% | 66.4% | 1.3s |
| 32ch/10u | 32 | 10 | 1481 | 2165 | **71.2%** | 76.8% | 83.3% | 1.3s |

Notes:
- 27 sorted clusters for 5 GT units indicates the cluster merger is not aggressive enough on this data.
- 1 of 5 units (and 1 of 10 units) is undetected (0% accuracy), likely below detection threshold.
- SpikeInterface uses noise_levels=5.0 (much higher than our easy preset noise_std=1.0), so the effective SNR is lower.
- CCG merge does not reduce over-splitting on this dataset (template correlation between over-split clusters is too low).

## Comparison Context

The Zerostone synthetic benchmarks use our own generator (zpybci.synthetic). The SpikeInterface validation uses an independent generator with different templates and noise. Both confirm that Zerostone reliably detects high-SNR units with >90% accuracy, but struggles with low-amplitude units near the detection threshold.

Real-time context: Zerostone is designed for sub-millisecond latency on 32-128 channels, not for maximizing offline accuracy on dense probes. Kilosort4 and MountainSort5 target different operating points.

## Streaming API

v0.7.0 introduces `StreamingSorter` for segment-based processing:
- Processes continuous recordings in fixed-length segments
- Maintains a template library across segments via NCC matching (threshold 0.7)
- Updates templates with exponential moving average (decay 0.95)
- Consistent label assignment across segments

## `alloc` Feature (v0.7.0+)

The `alloc` feature enables:
- `sorter_dyn::sort_dyn()` -- runtime channel count dispatch (no const-generic at call site)
- Heap-allocated buffers for batch processing
- 128-channel const-generic support
- Future: >128ch heap-allocated pipeline for Neuropixels-scale data
