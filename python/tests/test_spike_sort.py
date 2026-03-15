"""Tests for spike sorting primitives."""

import numpy as np
import pytest

import zpybci as zbci


# ---- Helpers ----


def make_spike_waveform(amplitude, peak_sample, width, window):
    """Generate a synthetic negative-peak spike waveform."""
    t = np.arange(window)
    return -amplitude * np.exp(-0.5 * ((t - peak_sample) / width) ** 2)


def make_synthetic_recording(
    n_samples, sample_rate, n_neuron_types, noise_std, rng, window=64
):
    """Generate a synthetic recording with embedded spikes.

    Returns (data, spike_info) where spike_info is a list of (times, label) tuples.
    """
    data = rng.standard_normal(n_samples) * noise_std
    pre = window // 4

    # Generate distinct waveform templates
    templates = []
    for i in range(n_neuron_types):
        amp = 6.0 + 4.0 * i  # Amplitudes: 6, 10, 14, ...
        width = 2.0 + i * 0.5
        templates.append(make_spike_waveform(amp, pre, width, window))

    # Insert spikes at regular-ish intervals
    spike_info = []
    min_spacing = int(sample_rate * 0.005)  # 5ms minimum between any two spikes
    for label, template in enumerate(templates):
        n_spikes = 20
        offset = label * min_spacing  # Stagger the neuron types
        spacing = n_samples // (n_spikes + 2)
        times = []
        for k in range(n_spikes):
            t = offset + (k + 1) * spacing
            if t - pre < 0 or t - pre + window >= n_samples:
                continue
            data[t - pre : t - pre + window] += template
            times.append(t)
        spike_info.append((times, label))

    return data, spike_info, templates


# ---- Extraction tests ----


class TestExtractWaveforms:
    def test_shape_and_values(self):
        data = np.arange(100, dtype=np.float64)
        spike_times = np.array([10, 50], dtype=np.int64)
        wf = zbci.extract_waveforms(data, spike_times, window=8)
        assert wf.shape == (2, 8)
        # pre_samples = 8 // 4 = 2, so spike at 10 extracts [8..16]
        np.testing.assert_array_equal(wf[0], np.arange(8, 16, dtype=np.float64))

    def test_edge_spikes_skipped(self):
        data = np.zeros(20, dtype=np.float64)
        # Spike at 0 (too close to start) and 19 (too close to end)
        spike_times = np.array([0, 1, 18, 19], dtype=np.int64)
        wf = zbci.extract_waveforms(data, spike_times, window=8)
        # pre=2: spike at 0,1 -> skip. spike at 18: start=16, end=24 > 20 -> skip
        # spike at 19: start=17, end=25 > 20 -> skip
        assert wf.shape[0] == 0

    def test_custom_pre_samples(self):
        data = np.arange(100, dtype=np.float64)
        spike_times = np.array([20], dtype=np.int64)
        wf = zbci.extract_waveforms(data, spike_times, window=8, pre_samples=4)
        assert wf.shape == (1, 8)
        # pre=4, so extracts [16..24]
        np.testing.assert_array_equal(wf[0], np.arange(16, 24, dtype=np.float64))

    def test_single_spike(self):
        data = np.zeros(50, dtype=np.float64)
        data[25] = -5.0
        spike_times = np.array([25], dtype=np.int64)
        wf = zbci.extract_waveforms(data, spike_times, window=8)
        assert wf.shape == (1, 8)
        # pre=2, extracts [23..31], data[25]=-5.0 should be at index 2
        assert wf[0, 2] == -5.0

    def test_empty_spike_times(self):
        data = np.zeros(100, dtype=np.float64)
        spike_times = np.array([], dtype=np.int64)
        wf = zbci.extract_waveforms(data, spike_times, window=8)
        assert wf.shape == (0, 8)


# ---- PCA tests ----


class TestWaveformPca:
    def test_create_all_sizes(self):
        for w in [32, 48, 64]:
            for k in [3, 5]:
                pca = zbci.WaveformPca(window=w, n_components=k)
                assert pca.window == w
                assert pca.n_components == k
                assert not pca.is_fitted

    def test_fit_transform_roundtrip(self):
        rng = np.random.default_rng(42)
        pca = zbci.WaveformPca(window=32, n_components=3)

        # Structured waveforms
        waveforms = np.zeros((50, 32), dtype=np.float64)
        for i in range(50):
            a = rng.standard_normal() * 5
            b = rng.standard_normal() * 3
            waveforms[i, :8] = a
            waveforms[i, 8:16] = b
            waveforms[i, 16:] = rng.standard_normal(16) * 0.01

        pca.fit(waveforms)
        assert pca.is_fitted
        features = pca.transform(waveforms)
        assert features.shape == (50, 3)
        # Features should be non-trivial
        assert np.std(features) > 0.1

    def test_explained_variance_high(self):
        rng = np.random.default_rng(123)
        pca = zbci.WaveformPca(window=32, n_components=3)

        # Create waveforms with only 3 sources of variance
        waveforms = np.zeros((100, 32), dtype=np.float64)
        for i in range(100):
            a = rng.standard_normal() * 10
            b = rng.standard_normal() * 5
            c = rng.standard_normal() * 2
            waveforms[i, 0] = a
            waveforms[i, 1] = b
            waveforms[i, 2] = c
            waveforms[i, 3:] = rng.standard_normal(29) * 0.01

        pca.fit(waveforms)
        evr = pca.explained_variance_ratio
        total = np.sum(evr)
        assert total > 0.90, f"3 components should explain >90% variance, got {total}"

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            zbci.WaveformPca(window=16, n_components=3)

    def test_invalid_n_components(self):
        with pytest.raises(ValueError):
            zbci.WaveformPca(window=32, n_components=7)

    def test_not_fitted_error(self):
        pca = zbci.WaveformPca(window=32, n_components=3)
        waveforms = np.zeros((5, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="not been fitted"):
            pca.transform(waveforms)

    def test_repr(self):
        pca = zbci.WaveformPca(window=64, n_components=5)
        r = repr(pca)
        assert "WaveformPca" in r
        assert "64" in r
        assert "5" in r


# ---- TemplateMatcher tests ----


class TestTemplateMatcher:
    def test_create_add_match(self):
        matcher = zbci.TemplateMatcher(channels=1, window=32, max_templates=8)
        assert matcher.n_templates == 0

        t1 = np.zeros(32, dtype=np.float64)
        t1[8] = -5.0
        t2 = np.zeros(32, dtype=np.float64)
        t2[16] = -5.0

        idx1 = matcher.add_template(t1)
        idx2 = matcher.add_template(t2)
        assert idx1 == 0
        assert idx2 == 1
        assert matcher.n_templates == 2

        # Match a waveform closer to t1
        wf = np.zeros((1, 32), dtype=np.float64)
        wf[0, 8] = -4.9
        labels, dists = matcher.match_waveforms(wf, method="euclidean")
        assert labels[0] == 0
        assert dists[0] < 1.0

    def test_ncc_matching(self):
        matcher = zbci.TemplateMatcher(channels=1, window=32, max_templates=8)
        t1 = np.zeros(32, dtype=np.float64)
        t1[8] = -5.0
        t1[9] = -3.0
        matcher.add_template(t1)

        # Scaled version should match via NCC
        wf = np.zeros((1, 32), dtype=np.float64)
        wf[0, 8] = -10.0
        wf[0, 9] = -6.0
        labels, scores = matcher.match_waveforms(wf, method="ncc")
        assert labels[0] == 0
        assert scores[0] > 0.99

    def test_template_update(self):
        matcher = zbci.TemplateMatcher(channels=1, window=32, max_templates=4)
        t = np.ones(32, dtype=np.float64)
        matcher.add_template(t)

        old_template = matcher.templates[0].copy()

        # Update with different values
        new_wf = np.ones(32, dtype=np.float64) * 5.0
        matcher.update_template(0, new_wf)

        updated_template = matcher.templates[0]
        # Template should shift toward new values
        assert np.all(updated_template > old_template)

    def test_wrong_shape_error(self):
        matcher = zbci.TemplateMatcher(channels=1, window=32, max_templates=4)
        # Wrong length template
        with pytest.raises(ValueError):
            matcher.add_template(np.zeros(16, dtype=np.float64))

    def test_template_full_error(self):
        matcher = zbci.TemplateMatcher(channels=1, window=32, max_templates=4)
        for i in range(4):
            matcher.add_template(np.ones(32, dtype=np.float64) * i)
        with pytest.raises(ValueError, match="Maximum"):
            matcher.add_template(np.zeros(32, dtype=np.float64))

    def test_repr(self):
        matcher = zbci.TemplateMatcher(channels=1, window=64, max_templates=16)
        r = repr(matcher)
        assert "TemplateMatcher" in r
        assert "64" in r


# ---- End-to-end tests ----


class TestEndToEnd:
    def test_spike_sort_convenience(self):
        rng = np.random.default_rng(42)
        data, spike_info, _ = make_synthetic_recording(
            n_samples=20000,
            sample_rate=30000.0,
            n_neuron_types=3,
            noise_std=1.0,
            rng=rng,
        )
        result = zbci.spike_sort(
            data, sample_rate=30000.0, threshold_std=4.0, window=64, n_clusters=3
        )
        assert "spike_times" in result
        assert "waveforms" in result
        assert "labels" in result
        assert "templates" in result
        assert "pca_features" in result

        n_detected = len(result["spike_times"])
        assert n_detected > 0
        assert result["waveforms"].shape == (n_detected, 64)
        assert result["labels"].shape == (n_detected,)
        assert result["templates"].shape == (3, 64)
        assert result["pca_features"].shape == (n_detected, 3)

    def test_template_matching_accuracy(self):
        """At SNR ~5, template matching should classify >90% correctly."""
        rng = np.random.default_rng(99)
        window = 64
        pre = window // 4

        # Create distinct templates
        t1 = make_spike_waveform(10.0, pre, 2.0, window)
        t2 = make_spike_waveform(8.0, pre, 4.0, window)

        # Generate noisy copies
        n_per_class = 50
        noise_std = 1.5
        waveforms = np.zeros((2 * n_per_class, window), dtype=np.float64)
        true_labels = np.zeros(2 * n_per_class, dtype=np.int64)
        for i in range(n_per_class):
            waveforms[i] = t1 + rng.standard_normal(window) * noise_std
            true_labels[i] = 0
            waveforms[n_per_class + i] = t2 + rng.standard_normal(window) * noise_std
            true_labels[n_per_class + i] = 1

        matcher = zbci.TemplateMatcher(channels=1, window=64, max_templates=8)
        matcher.add_template(t1)
        matcher.add_template(t2)

        labels, _ = matcher.match_waveforms(waveforms, method="euclidean")
        accuracy = np.mean(labels == true_labels)
        assert accuracy > 0.90, f"Template matching accuracy {accuracy:.2%} < 90%"

    def test_different_window_sizes(self):
        rng = np.random.default_rng(7)
        for window in [32, 48, 64]:
            data, _, _ = make_synthetic_recording(
                n_samples=20000,
                sample_rate=30000.0,
                n_neuron_types=2,
                noise_std=1.0,
                rng=rng,
                window=window,
            )
            result = zbci.spike_sort(
                data,
                sample_rate=30000.0,
                threshold_std=4.0,
                window=window,
                n_clusters=2,
            )
            assert result["waveforms"].shape[1] == window

    def test_mad_noise_estimation(self):
        rng = np.random.default_rng(42)
        true_std = 3.0
        data = rng.standard_normal(10000).astype(np.float64) * true_std
        estimated = zbci.estimate_noise_mad(data)
        ratio = estimated / true_std
        assert 0.9 < ratio < 1.1, f"MAD ratio {ratio:.3f} out of [0.9, 1.1]"

    def test_integration_with_bandpass(self):
        """Spike sorting should work with bandpass-filtered data."""
        rng = np.random.default_rng(55)
        sample_rate = 30000.0
        n_samples = 30000

        data, _, _ = make_synthetic_recording(
            n_samples=n_samples,
            sample_rate=sample_rate,
            n_neuron_types=2,
            noise_std=1.0,
            rng=rng,
        )

        # Apply bandpass filter (300-3000 Hz for spike band)
        bpf = zbci.IirFilter.butterworth_bandpass(
            float(sample_rate), 300.0, 3000.0, order=4
        )
        filtered = bpf.process(data.astype(np.float32))

        # Detect spikes on filtered data
        filtered_f64 = filtered.astype(np.float64)
        noise_est = zbci.estimate_noise_mad(filtered_f64)
        assert noise_est > 0
        spike_times = zbci.detect_spikes(filtered_f64, threshold=4.0 * noise_est)
        assert len(spike_times) > 0

        # Extract waveforms
        wf = zbci.extract_waveforms(filtered_f64, spike_times, window=64)
        assert wf.shape[0] > 0
        assert wf.shape[1] == 64


# ---- Multi-channel detection tests ----


class TestDetectSpikesMultichannel:
    def test_known_locations(self):
        """Detect spikes at known positions on different channels."""
        data = np.zeros((100, 4), dtype=np.float64)
        data[20, 0] = -6.0
        data[50, 1] = -8.0
        data[80, 2] = -5.0

        noise = np.ones(4, dtype=np.float64)
        events = zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise, refractory=10)
        assert len(events) == 3
        samples = [e["sample"] for e in events]
        channels = [e["channel"] for e in events]
        assert 20 in samples
        assert 50 in samples
        assert 80 in samples
        assert 0 in channels
        assert 1 in channels
        assert 2 in channels

    def test_no_spikes(self):
        """All values below threshold: no events."""
        data = np.full((50, 2), 0.5, dtype=np.float64)
        noise = np.ones(2, dtype=np.float64)
        events = zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise)
        assert len(events) == 0

    def test_amplitude_correct(self):
        data = np.zeros((20, 2), dtype=np.float64)
        data[10, 0] = -7.5
        noise = np.ones(2, dtype=np.float64)
        events = zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise, refractory=5)
        assert len(events) == 1
        assert abs(events[0]["amplitude"] - 7.5) < 1e-12

    def test_simultaneous_all_channels(self):
        data = np.zeros((50, 4), dtype=np.float64)
        for ch in range(4):
            data[25, ch] = -10.0
        noise = np.ones(4, dtype=np.float64)
        events = zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise, refractory=5)
        assert len(events) == 4
        for e in events:
            assert e["sample"] == 25

    def test_invalid_noise_length(self):
        data = np.zeros((50, 4), dtype=np.float64)
        noise = np.ones(3, dtype=np.float64)  # wrong length
        with pytest.raises(ValueError, match="expected 4"):
            zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise)

    def test_invalid_channel_count(self):
        data = np.zeros((50, 3), dtype=np.float64)
        noise = np.ones(3, dtype=np.float64)
        with pytest.raises(ValueError, match="n_channels"):
            zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise)


class TestDeduplicateEvents:
    def test_removes_neighbor_duplicate(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        events = [
            {"sample": 100, "channel": 1, "amplitude": 5.0},
            {"sample": 102, "channel": 2, "amplitude": 7.0},
        ]
        deduped = zbci.deduplicate_events(events, probe, temporal_radius=5, spatial_radius=30.0)
        assert len(deduped) == 1
        assert deduped[0]["channel"] == 2
        assert abs(deduped[0]["amplitude"] - 7.0) < 1e-12

    def test_keeps_distant_events(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        events = [
            {"sample": 100, "channel": 0, "amplitude": 5.0},
            {"sample": 500, "channel": 1, "amplitude": 6.0},
        ]
        deduped = zbci.deduplicate_events(events, probe)
        assert len(deduped) == 2

    def test_keeps_spatially_distant(self):
        probe = zbci.ProbeLayout.linear(8, 50.0)
        events = [
            {"sample": 100, "channel": 0, "amplitude": 5.0},
            {"sample": 101, "channel": 7, "amplitude": 6.0},
        ]
        deduped = zbci.deduplicate_events(events, probe, temporal_radius=5, spatial_radius=60.0)
        assert len(deduped) == 2

    def test_no_events(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        deduped = zbci.deduplicate_events([], probe)
        assert len(deduped) == 0

    def test_three_events_cluster(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        events = [
            {"sample": 100, "channel": 0, "amplitude": 3.0},
            {"sample": 101, "channel": 1, "amplitude": 8.0},
            {"sample": 102, "channel": 2, "amplitude": 5.0},
        ]
        deduped = zbci.deduplicate_events(events, probe, temporal_radius=5, spatial_radius=30.0)
        assert len(deduped) == 1
        assert deduped[0]["channel"] == 1


class TestAlignToPeak:
    def test_improves_timing(self):
        data = np.zeros((30, 2), dtype=np.float64)
        data[10, 0] = -5.0
        data[11, 0] = -7.0
        data[12, 0] = -9.0  # true peak
        data[13, 0] = -6.0

        events = [{"sample": 10, "channel": 0, "amplitude": 5.0}]
        aligned = zbci.align_to_peak(events, data, half_window=5)
        assert aligned[0]["sample"] == 12
        assert abs(aligned[0]["amplitude"] - 9.0) < 1e-12

    def test_already_aligned(self):
        data = np.zeros((20, 1), dtype=np.float64)
        data[10, 0] = -8.0
        events = [{"sample": 10, "channel": 0, "amplitude": 8.0}]
        aligned = zbci.align_to_peak(events, data, half_window=5)
        assert aligned[0]["sample"] == 10

    def test_no_events(self):
        data = np.zeros((20, 2), dtype=np.float64)
        aligned = zbci.align_to_peak([], data)
        assert len(aligned) == 0

    def test_multichannel_correct_channel(self):
        data = np.zeros((30, 2), dtype=np.float64)
        data[10, 0] = -3.0
        data[12, 0] = -8.0  # ch0 peak
        data[10, 1] = -2.0
        data[11, 1] = -6.0  # ch1 peak

        events = [
            {"sample": 10, "channel": 0, "amplitude": 3.0},
            {"sample": 10, "channel": 1, "amplitude": 2.0},
        ]
        aligned = zbci.align_to_peak(events, data, half_window=5)
        assert aligned[0]["sample"] == 12
        assert aligned[1]["sample"] == 11


class TestMultichannelEndToEnd:
    def test_detect_dedup_align_pipeline(self):
        """Full pipeline: detect -> align -> dedup."""
        rng = np.random.default_rng(77)
        n_samples = 1000
        n_channels = 4

        data = rng.standard_normal((n_samples, n_channels))

        # Spike 1 at t=200, largest on ch1
        data[200, 0] += -6.0
        data[200, 1] += -10.0
        data[200, 2] += -5.0

        # Spike 2 at t=600, largest on ch3
        data[600, 2] += -4.5
        data[600, 3] += -9.0

        noise = np.ones(n_channels, dtype=np.float64)

        # Detect
        events = zbci.detect_spikes_multichannel(data, threshold=4.0, noise=noise, refractory=10)
        assert len(events) >= 4

        # Align
        aligned = zbci.align_to_peak(events, data, half_window=3)
        assert len(aligned) == len(events)

        # Dedup
        probe = zbci.ProbeLayout.linear(4, 25.0)
        deduped = zbci.deduplicate_events(aligned, probe, temporal_radius=5, spatial_radius=30.0)

        assert len(deduped) >= 2
        assert len(deduped) < len(events)

        # Verify the two main spikes are present
        samples = [e["sample"] for e in deduped]
        spike1_found = any(197 <= s <= 203 for s in samples)
        spike2_found = any(597 <= s <= 603 for s in samples)
        assert spike1_found, f"Spike 1 near t=200 not found in {samples}"
        assert spike2_found, f"Spike 2 near t=600 not found in {samples}"
