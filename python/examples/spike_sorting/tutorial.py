"""Spike sorting tutorial for Zerostone.

Demonstrates the full spike sorting pipeline on synthetic data:
1. Generate a multi-channel recording with known ground truth
2. Sort spikes using sort_multichannel()
3. Evaluate accuracy against ground truth
4. Use StreamingSorter for segment-based processing

Usage:
    python examples/spike_sorting/tutorial.py
"""

import numpy as np
import zpybci as zbci
from zpybci.synthetic import generate_recording

# -- 1. Generate synthetic recording ------------------------------------------

print("1. Generating synthetic recording...")
rec = generate_recording(
    n_channels=32,
    n_units=5,
    noise_std=1.0,
    duration_s=30.0,
    firing_rate=5.0,
    seed=42,
)

data = rec["data"]  # shape: (n_samples, n_channels)
gt_times = rec["all_spike_times"]
gt_labels = rec["spike_labels"]

print(f"   Data shape: {data.shape}")
print(f"   Ground truth: {rec['n_units']} units, {len(gt_times)} spikes")

# -- 2. Create probe geometry -------------------------------------------------

print("\n2. Setting up probe geometry...")
probe = zbci.ProbeLayout.linear(32, 25.0)  # 32ch linear, 25um pitch
print(f"   Probe: {probe}")
print(f"   Spatial extent: {probe.spatial_extent()}")

# Other probe options:
# probe = zbci.ProbeLayout.tetrode(25.0)           # 4ch tetrode
# probe = zbci.ProbeLayout.neuropixels_1()          # 384ch Neuropixels 1.0
# probe = zbci.ProbeLayout.neuropixels_2()          # 384ch Neuropixels 2.0
# probe = zbci.ProbeLayout.utah_array()             # 96ch Utah array

# -- 3. Run batch sorting ------------------------------------------------------

print("\n3. Running sort_multichannel()...")
result = zbci.sort_multichannel(
    data,
    probe,
    threshold=5.0,           # detection threshold in MAD units
    cluster_threshold=8.0,   # distance threshold for new clusters
    template_subtract_passes=1,
)

print(f"   Detected: {result['n_spikes']} spikes in {result['n_clusters']} clusters")

# Access per-spike data
spike_times = np.array(result["spike_times"])   # sample indices
labels = np.array(result["labels"])             # cluster assignment
spike_channels = np.array(result["spike_channels"])  # peak channel

# Access per-cluster quality metrics
for i, cluster in enumerate(result["clusters"]):
    if i >= 5:
        print(f"   ... ({result['n_clusters'] - 5} more clusters)")
        break
    print(f"   Cluster {i}: count={cluster['count']}, "
          f"snr={cluster['snr']:.1f}, "
          f"isi_violation_rate={cluster['isi_violation_rate']:.3f}")

# -- 4. Evaluate accuracy against ground truth ---------------------------------

print("\n4. Evaluating accuracy...")
# compare_sorting expects lists of spike time arrays, one per unit/cluster
gt_trains = [
    gt_times[gt_labels == u].astype(np.int64) for u in range(rec["n_units"])
]
sorted_trains = [
    spike_times[labels == cl].astype(np.int64)
    for cl in range(result["n_clusters"])
]

per_unit = zbci.compare_sorting(gt_trains, sorted_trains, tolerance=20)
for i, match in enumerate(per_unit):
    tp = match["true_positives"]
    fn = match["false_negatives"]
    fp = match["false_positives"]
    print(f"   GT unit {i}: "
          f"acc={match['accuracy']:.1%}, "
          f"prec={match['precision']:.1%}, "
          f"rec={match['recall']:.1%} "
          f"(TP={tp}, FN={fn}, FP={fp})")

# -- 5. Streaming sorting ------------------------------------------------------

print("\n5. Streaming sorting with StreamingSorter...")
sorter = zbci.StreamingSorter(
    n_channels=32,
    decay=0.95,              # template EMA decay
    detection_mode="amplitude",
)

segment_size = 30000  # 1 second at 30 kHz
n_segments = data.shape[0] // segment_size

for i in range(min(n_segments, 5)):
    segment = data[i * segment_size : (i + 1) * segment_size]
    seg_result = sorter.feed(segment, probe)
    print(f"   Segment {i+1}: {seg_result['n_spikes']} spikes, "
          f"{seg_result['n_clusters']} clusters, "
          f"templates={sorter.n_templates}")

print(f"\n   Total segments processed: {sorter.segment_count}")
print(f"   Active templates: {sorter.n_templates}")

# -- 6. Probe geometry queries -------------------------------------------------

print("\n6. Probe geometry queries...")
d = probe.channel_distance(0, 5)
print(f"   Distance ch0 -> ch5: {d:.1f} um")

neighbors = probe.neighbor_channels(16, 50.0)
print(f"   Channels within 50um of ch16: {sorted(neighbors)}")

nearest = probe.nearest_channels(16, 3)
print(f"   3 nearest to ch16: {nearest}")

print("\nDone.")
