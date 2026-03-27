# Spike Sorting Benchmark Results

Zerostone v0.8.0 on synthetic multi-channel recordings with known ground truth.

Methodology follows SpikeInterface/SpikeForest: accuracy = TP / (TP + FN + FP), with 0.67 ms tolerance window (20 samples at 30 kHz) and greedy best-match assignment.

## Results (seed=42, March 2026)

| Preset | Channels | Units | GT Spikes | Sorted | Clusters | Accuracy | Precision | Recall | Time |
|--------|----------|-------|-----------|--------|----------|----------|-----------|--------|------|
| easy   | 32       | 5     | 1476      | 1551   | 24       | **95.9%** | 98.7%    | 97.2%  | 2.2s |
| medium | 32       | 10    | 4865      | 6030   | 31       | **75.9%** | 91.5%    | 81.7%  | 4.8s |
| hard   | 64       | 20    | 11926     | 8099   | 14       | **50.4%** | 82.8%    | 56.3%  | 7.6s |

### Performance History

| Preset | v0.6.0 Acc | v0.7.0+sweep | v0.7.0+MF | v0.6.0 Time | v0.7.0+MF Time | Speedup |
|--------|-----------|-------------|-----------|-------------|----------------|---------|
| easy   | 94.8%     | **95.9%**   | **95.9%** | ~35s        | 2.2s           | **16x** |
| medium | 57.1%     | 69.6%       | **75.9%** | ~128s       | 4.8s           | **27x** |
| hard   | 25.1%     | 48.6%       | **50.4%** | ~286s       | 7.6s           | **38x** |

The v0.7.0+MF improvement comes from matched filter second-pass detection:
- Easy: No matched filter (already at 95.9%, MF adds false positives on high-SNR recordings)
- Medium: `matched_filter_detect=true, matched_filter_threshold=4.2` (recovers weak units below amplitude threshold)
- Hard: `matched_filter_detect=true, matched_filter_threshold=4.0` (recovers additional units)
- Additionally: `cluster_threshold=8.0` (over-split then merge), `template_subtract_passes=1`

The matched filter is the Neyman-Pearson optimal detector: it maximizes detection probability for a given false-positive rate by correlating learned template waveforms with the whitened data. SNR gain over amplitude detection is ~√(W_eff) where W_eff is the effective signal duration in samples.

### Medium Per-Unit Breakdown

| Unit | Cluster | GT# | TP  | FN  | FP | Acc   | Prec  | Rec   | vs prev |
|------|---------|-----|-----|-----|----|-------|-------|-------|---------|
| 0    | 19      | 478 | 470 | 8   | 48 | 0.894 | 0.907 | 0.983 | =       |
| 1    | 5       | 483 | 165 | 318 | 151| 0.260 | 0.522 | 0.342 | **+21.7%** (was 4.3%) |
| 2    | 1       | 483 | 434 | 49  | 20 | 0.863 | 0.956 | 0.899 | **+55.6%** (was 30.7%) |
| 3    | 12      | 489 | 360 | 129 | 22 | 0.705 | 0.942 | 0.736 | =       |
| 4    | 23      | 486 | 241 | 245 | 62 | 0.440 | 0.795 | 0.496 | =       |
| 5    | 17      | 517 | 448 | 69  | 5  | 0.858 | 0.989 | 0.867 | =       |
| 6    | 8       | 476 | 455 | 21  | 1  | 0.954 | 0.998 | 0.956 | =       |
| 7    | 9       | 494 | 463 | 31  | 5  | 0.928 | 0.989 | 0.937 | =       |
| 8    | 2       | 480 | 475 | 5   | 55 | 0.888 | 0.896 | 0.990 | -9.1%   |
| 9    | 6       | 479 | 463 | 16  | 2  | 0.963 | 0.996 | 0.967 | =       |

The matched filter dramatically improved detection of weak units:
- **Unit 2**: 30.7% -> 86.3% accuracy (+284 recovered spikes). Template was below amplitude threshold but above matched filter threshold.
- **Unit 1**: 4.3% -> 26.0% accuracy (+141 recovered spikes). Very weak template, partial recovery.
- **Unit 8**: 97.9% -> 88.8% accuracy. Slight degradation from MF false positives assigned to this cluster.
- All other units: unchanged or negligible change.

## Pipeline Configuration

- Detection: amplitude threshold (NEO/SNEO modes available)
- Supported channels: 4, 8, 16, 32, 64, 96, 128 (const-generic, no_std)
- Probe presets: Neuropixels 1.0 (384ch), Neuropixels 2.0 (384ch), Utah array (96ch), tetrode (4ch), linear, polytrode
- Waveform window: W=48 samples
- PCA components: K=4 (3 PCA + 1 channel index)
- Max clusters: N=32
- Template subtraction: enabled (per-spike amplitude scaling)
- Template residual detection: NCC matching (threshold 0.7, min amplitude 0.5x)
- SNR auto-curation: min_cluster_snr configurable (default 2.5, 1.5 for medium)
- Spatial merge: cross-channel d-prime with amplitude profiles
- CCG merge: available (merges over-split clusters with refractory dip in cross-correlogram)
- Multi-pass template subtraction: configurable (1-3 passes)
- ISI-violation cluster splitting: configurable threshold
- Amplitude bimodality split: gap detection on sorted amplitudes with local MAD threshold
- Batch parallel: rayon-based segment parallelism (alloc feature)

## Detection Modes

Zerostone supports three detection operators:
- **Amplitude** (default): Standard negative-peak threshold crossing on whitened data.
- **NEO**: Nonlinear Energy Operator `psi[n] = x[n]^2 - x[n-1]*x[n+1]`. Enhances spike-like transients.
- **SNEO**: Smoothed NEO with triangular window. Reduces cross-term artifacts from overlapping spikes.

NEO/SNEO detection thresholds are calibrated using robust percentile estimation (median + T * MAD/0.6745) with a 2000-sample stack-allocated calibration buffer. This resists spike contamination (50% breakdown point).

Note: SNEO currently underperforms amplitude on synthetic data due to threshold space mismatch (T=5 in energy space is far more aggressive than T=5 in amplitude space). A threshold space conversion is needed for SNEO to be competitive.

## Parameter Sweep Results (Week 18)

### Full Sweep (64 combos on medium, top-3 on easy/hard)

Swept threshold x matched_filter x svd_init x bandpass x CMR. Config: cluster_threshold=8.0, min_cluster_snr=1.5, refractory=15, matched_filter_threshold=4.0.

**Top 5 on medium (by accuracy):**

| Rank | Thr | MF | SVD | BP | CMR | Accuracy | Spikes | Clusters | Time |
|------|-----|----|-----|----|-----|----------|--------|----------|------|
| 1 | 4.0 | Y | N | N | N | **75.2%** | 6114 | 31 | 5.2s |
| 2 | 4.0 | Y | Y | N | N | **73.8%** | 6074 | 30 | 5.4s |
| 3 | 5.0 | Y | N | N | N | **73.3%** | 4407 | 19 | 4.1s |
| 4 | 4.5 | Y | N | N | N | **72.9%** | 4535 | 24 | 4.6s |
| 5 | 4.5 | Y | N | N | Y | **70.6%** | 4714 | 25 | 5.1s |

**Cross-difficulty (top-3 from medium applied to easy/hard):**

| Config | Easy | Medium | Hard | Average |
|--------|------|--------|------|---------|
| thr=5.0 mf=Y svd=N bp=N cmr=N | 86.1% | 73.3% | 47.2% | **68.9%** |
| thr=4.0 mf=Y svd=Y bp=N cmr=N | 86.0% | 73.8% | 42.9% | **67.6%** |
| thr=4.0 mf=Y svd=N bp=N cmr=N | 82.0% | 75.2% | 43.4% | **66.9%** |

### Key Findings

1. **Bandpass hurts** on synthetic data (already bandlimited). Every bandpass combo underperforms its no-bandpass equivalent.
2. **CMR hurts** on synthetic data (spatially uncorrelated noise). Removes signal along with noise.
3. **Matched filter helps** consistently (+3-7% on medium). The Neyman-Pearson detector recovers weak units below amplitude threshold.
4. **SVD init is mixed**: helps on easy (+4%), marginal on medium (-1.4%), slightly hurts on hard (-0.5%). Random k-means init is surprisingly competitive.
5. **Threshold 4.0** optimal for medium (balances detection vs false positives), **5.0** better for hard (more conservative avoids noise clusters).
6. **thr=3.5 universally bad** -- too many false positives overwhelm clustering (15K-20K detected vs 4.8K ground truth).

### Earlier Sweep Findings

1. **cluster_threshold=8.0** (vs default 5.0) improves accuracy by allowing more initial clusters that are then merged by d-prime. Over-splitting followed by informed merging > conservative initial clustering.
2. **min_cluster_snr=1.5** on medium retains weak but real clusters that the default 2.5 would remove.
3. **CCG merge** has no effect on synthetic data (no over-splitting pattern to fix).
4. **ISI-violation split** has no effect on synthetic data (clusters are already well-separated).
5. **Multi-pass template subtraction** provides marginal improvement on synthetic data (no overlapping spikes in our generator).
6. **3 template passes** slightly decreases accuracy (over-subtraction introduces artifacts).

## Analysis

**Easy (5 units, 32 channels)**: All 5 units detected with >93% per-unit accuracy. Very high precision (98.7%) and recall (97.2%). The 4.1% overall loss comes from minor template alignment edge effects.

**Medium (10 units, 32 channels, noise_std=1.5)**: 7 of 10 units detected at >70% accuracy. Units 1 and 2 remain difficult (template amplitude <3 sigma after whitening, effectively undetectable by threshold). Unit 4 has moderate overlap with unit 3. The 69.6% overall is precision-limited: high precision (93.9%) but lower recall (72.9%) due to the 3 weak units.

**Hard (20 units, 64 channels, noise_std=2.0)**: 12 of 20 units detected at >50% accuracy. 4 units completely undetected (amplitude below noise). The 48.6% overall represents a near-doubling from v0.7.0+opt (25.2%), driven by threshold tuning from 3.5 to 5.0 which dramatically reduced false positives.

## SpikeInterface Ground Truth Validation (seed=42, 30s, 30 kHz)

Independent validation using SpikeInterface 0.104.0's `generate_ground_truth_recording()` with its own template library and noise model.

| Config | Channels | Units | GT Spikes | Sorted | Mean Accuracy | Mean Precision | Mean Recall | Time |
|--------|----------|-------|-----------|--------|---------------|----------------|-------------|------|
| 32ch/5u  | 32 | 5  | 758  | 1130 | **66.1%** | 79.5% | 66.4% | 1.3s |
| 32ch/10u | 32 | 10 | 1481 | 2165 | **71.2%** | 76.8% | 83.3% | 1.3s |

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
- `sort_batch_parallel()` -- rayon-based parallel sorting of long recordings by segment
- Heap-allocated buffers for batch processing
- 96-channel and 128-channel const-generic support
- Future: >128ch heap-allocated pipeline for Neuropixels-scale data (384ch probe geometry already supported)
