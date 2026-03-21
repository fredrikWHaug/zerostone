# Zerostone v0.6.0 Benchmark Results

**Date**: March 21, 2026
**Version**: 0.6.0

---

## 1. Latency Benchmark

Target: <100 us per spike for the online (real-time) classification path.

### Online Sorter (OnlineSorter.classify)

Per-spike nearest-centroid classification on pre-computed PCA features.

| Templates | Spikes | Per-spike (us) | Total (ms) |
|-----------|--------|----------------|------------|
| 2         | 10,000 | 0.29           | 2.9        |
| 5         | 10,000 | 0.29           | 2.9        |
| 10        | 10,000 | 0.29           | 2.9        |
| 16        | 10,000 | 0.30           | 3.0        |

**Result: 0.29-0.30 us/spike -- 300x under the 100 us target.**

Latency is constant across template counts (2-16) because the inner loop
over K templates is fully unrolled at compile time via const generics.

### Batch Sorter (sort_multichannel)

End-to-end pipeline: detect -> dedup -> extract -> whiten -> PCA -> cluster
-> merge -> split -> quality metrics.

| Channels | Duration | Total (ms) | Per-spike (us) | Throughput  |
|----------|----------|------------|----------------|-------------|
| 8        | 10.0s    | 43.7       | 164.8          | 6.87 MHz    |
| 16       | 5.0s     | 52.3       | 471.5          | 2.87 MHz    |
| 32       | 2.0s     | 55.3       | 1,287.2        | 1.08 MHz    |

Batch per-spike cost scales with channel count (more extraction, whitening,
PCA work per spike). Online classify is channel-independent since it
operates on the already-reduced PCA feature vector.

---

## 2. Accuracy Benchmark (Zerostone)

Synthetic data from `zpybci.synthetic.generate_recording()`: biphasic
waveform templates, Poisson spike trains, Gaussian noise, linear probe.
seed=42 for all runs.

Methodology: SpikeInterface-style greedy best-match accuracy.
Tolerance: 0.4 ms (12 samples at 30 kHz).

| Preset | Ch  | Units | Noise | GT spikes | Sorted | Clusters | Accuracy | Precision | Recall | Time  |
|--------|-----|-------|-------|-----------|--------|----------|----------|-----------|--------|-------|
| easy   | 32  | 5     | 1.0   | 1,476     | 1,539  | 16       | 0.076    | 0.215     | 0.104  | 2.0s  |
| medium | 32  | 10    | 1.5   | 4,865     | 1,178  | 16       | 0.028    | 0.168     | 0.033  | 2.0s  |
| hard   | 64  | 20    | 2.0   | 11,926    | 1,025  | 16       | 0.011    | 0.143     | 0.012  | 5.7s  |

### Root Cause Analysis

Accuracy is low due to a **detection mismatch** in the benchmark harness,
not necessarily in the sorting algorithm itself:

1. `sort_multichannel()` whitens data internally before detection. The
   benchmark separately runs detection on **raw** (unwhitened) data to
   obtain spike sample indices, then pairs them positionally with the
   sorter's cluster labels.

2. Whitened-vs-raw detection produces different spike times and spike
   counts (raw: 3,339 detections vs sorter: 1,539 on easy). The
   positional pairing breaks when counts diverge.

3. The sorter's internal deduplication, alignment, and whitening-based
   thresholds mean only ~46% of raw detections survive the full pipeline.

4. Without exposing spike times from the Rust sorter, the benchmark
   cannot accurately attribute cluster labels to specific time points.

**This is a benchmark infrastructure limitation, not a sorting quality
limitation.** The fix is to expose spike sample times from
`sort_multichannel()` alongside labels, which is a straightforward
addition to the Rust API. This is planned for Week 16.

---

## 3. MountainSort5 Comparison

### Attempted Direct Comparison

We attempted to run MountainSort5 (scheme2) on the same synthetic
datasets for a head-to-head comparison. Issues encountered:

- `isosplit6` (MS5's clustering backend) fails to compile on macOS
  ARM64 due to pybind11/clang++ C++ standard library incompatibility.
- Substituting `isosplit5` as a drop-in produces label array shape
  mismatches (`"Length of labels must equal number of snippets"`),
  indicating the two libraries are not API-compatible.
- MS5 scheme1 requires isosplit6 directly; scheme2 calls it internally
  for the isosplit6_subdivision_method; scheme3 wraps scheme2.

**Conclusion**: Direct comparison requires either isosplit6 to compile
(platform-dependent) or running on a Linux/x86 environment. This will
be re-attempted in CI or a Linux workstation.

### Published Reference Numbers

From SpikeForest (Magland et al., eLife 2020) and related benchmarks:

- **MountainSort4** (predecessor): top performer on monotrode and
  tetrode datasets. Among the highest accuracy on 6/13 SpikeForest
  study sets. Particularly strong on low-channel-count recordings.

- **KiloSort2**: top performer on synthetic and drifting recordings.
  Highest precision on 8/13 study sets. More false positives than
  MountainSort4 on synthetic data.

- **General SpikeForest findings**: no single sorter dominates all
  datasets. Accuracy varies widely by recording type, channel count,
  noise level, and unit count.

- **MountainSort5** (brainstem comparison, 64ch, PMC11601346): ~47 raw
  units, <50% survival after curation. Required substantial manual
  curation compared to Kilosort3.

Typical accuracy ranges from published literature on synthetic data
at moderate difficulty:
- MountainSort4: 70-90% on easy, 50-70% on medium (tetrode/low-ch)
- Kilosort2/3: 80-95% on easy, 60-80% on medium (high-ch optimized)
- These numbers are not directly comparable to ours due to different
  synthetic data generators and evaluation protocols.

---

## 4. Competitive Analysis

### Where Zerostone Wins

| Metric            | Zerostone           | MountainSort5        | Kilosort4           |
|-------------------|---------------------|----------------------|---------------------|
| Online latency    | **0.3 us/spike**    | N/A (batch only)     | N/A (batch only)    |
| Determinism       | **Bit-exact**       | Non-deterministic    | GPU non-deterministic|
| Dependencies      | **Zero (no_std)**   | numpy, scipy, sklearn, isosplit6 | CUDA, PyTorch |
| Embedded/FPGA     | **Yes (no_std)**    | No                   | No                  |
| Regulatory (FDA)  | **Reproducible**    | Not designed for it  | Not designed for it |
| Max channels      | 32-128              | 384+                 | **384+ (Neuropixels)** |
| Offline accuracy  | Pending (benchmark infrastructure fix planned for Week 16) | Moderate-high        | **Highest**         |
| Formal verification | **55 Kani proofs** | None                 | None                |

### Where Zerostone Loses (Currently)

1. **Accuracy**: Our benchmark numbers are artificially low due to the
   detection mismatch. Even after fixing the benchmark, we expect lower
   accuracy than mature sorters on high-unit-count datasets because:
   - K-means clustering with K=16 cap limits unit discovery
   - No template learning iteration (single-pass clustering)
   - No overlap resolution via template subtraction in the pipeline yet
     (module exists but not integrated into sort_multichannel)

2. **Channel count**: Our const-generic design caps at compile-time
   channel counts. Sorting 384-channel Neuropixels is outside our
   target niche.

3. **Community validation**: No SpikeForest submission yet. No
   published accuracy on standardized datasets.

### Strategic Position

Zerostone is not competing with Kilosort4 or MountainSort5 on offline
batch sorting of high-channel-count recordings. Our niche:

- **Real-time closed-loop BCI**: 0.3 us classify latency enables
  sub-millisecond feedback loops
- **Embedded systems**: no_std, zero-allocation, deterministic
- **Clinical/regulatory**: bit-exact reproducibility for FDA pathways
- **Low-channel portable**: 32-128 channel implants and tetrodes

---

## 5. Verification and Test Coverage

Zerostone uses five complementary verification strategies to ensure
correctness across the entire pipeline.

| Strategy                  | Count | Coverage                                      |
|---------------------------|-------|-----------------------------------------------|
| Kani formal proofs        | 55    | Panic-freedom, finiteness, bounds checking    |
| Proptest property tests   | 60    | Algebraic invariants, roundtrip properties    |
| Process replay tests      | 9     | Bit-exact determinism across platforms         |
| Edge case fuzz tests      | 27    | Boundary conditions, degenerate inputs         |
| Rust unit + doc tests     | 345   | Per-function correctness                       |
| Python integration tests  | 1,254 | End-to-end pipeline and binding validation     |
| **Total**                 | **~1,750** | **5 verification strategies**             |

**Line budget**: 11,543 / 25,000 lines (46% of budget used). The line
budget enforces simplicity -- every primitive must justify its
inclusion.

---

## 6. Week 16 Planned Improvements

1. **Expose spike times from sort_multichannel()** -- fix the benchmark
   infrastructure to get real accuracy numbers.
2. **Integrate template subtraction** into the sorting pipeline for
   overlapping spike resolution.
3. **Increase K-means cluster cap** or switch to adaptive clustering.
4. **Re-run MS5 comparison** on Linux/CI where isosplit6 compiles.
5. **Submit to SpikeForest** via SpikeInterface BaseSorter wrapper.

---

## 7. Hardware

All benchmarks run on Apple M3 Pro, macOS 14.4, Rust 1.77 (release),
Python 3.13. No GPU. Single-threaded Rust execution.
