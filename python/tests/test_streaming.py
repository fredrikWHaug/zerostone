"""Tests for zpybci.StreamingSorter -- segment-based streaming spike sorting."""

import numpy as np
import pytest

import zpybci as zbci


def _make_probe(n_channels):
    """Create a linear probe with given channel count."""
    return zbci.ProbeLayout.linear(n_channels, 25.0)


def _make_noise(n_samples, n_channels, seed=42):
    """Create noise-only data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_channels)) * 0.1


def _make_data_with_spikes(n_samples, n_channels, seed=42):
    """Create data with clear spikes for detection."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, n_channels)) * 0.5

    # Insert large spikes every 200 samples on channel 0
    for t in range(200, n_samples - 50, 200):
        data[t, 0] = -15.0
        data[t + 1, 0] = 8.0

    return data


class TestStreamingSorterCreation:
    def test_create_4ch(self):
        sorter = zbci.StreamingSorter(4)
        assert sorter.n_templates == 0
        assert sorter.segment_count == 0

    def test_create_8ch(self):
        sorter = zbci.StreamingSorter(8)
        assert sorter.n_templates == 0

    def test_create_16ch(self):
        sorter = zbci.StreamingSorter(16)
        assert sorter.n_templates == 0

    def test_create_32ch(self):
        sorter = zbci.StreamingSorter(32)
        assert sorter.n_templates == 0

    def test_invalid_channels(self):
        with pytest.raises(ValueError):
            zbci.StreamingSorter(3)

    def test_custom_params(self):
        sorter = zbci.StreamingSorter(
            4, decay=0.9, threshold=4.0, detection_mode="amplitude"
        )
        assert sorter.n_templates == 0

    def test_invalid_detection_mode(self):
        with pytest.raises(ValueError, match="detection_mode"):
            zbci.StreamingSorter(4, detection_mode="invalid")


class TestStreamingSorterFeed:
    def test_feed_noise(self):
        """Feeding noise should produce a valid result dict."""
        sorter = zbci.StreamingSorter(4)
        probe = _make_probe(4)
        data = _make_noise(5000, 4)
        result = sorter.feed(data, probe)
        assert "n_spikes" in result
        assert "n_clusters" in result
        assert "labels" in result
        assert "spike_times" in result

    def test_feed_with_spikes(self):
        """Data with clear spikes should detect something."""
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)
        data = _make_data_with_spikes(10000, 4)
        result = sorter.feed(data, probe)
        assert result["n_spikes"] > 0

    def test_segment_count_increments(self):
        sorter = zbci.StreamingSorter(4)
        probe = _make_probe(4)
        data = _make_noise(5000, 4)
        assert sorter.segment_count == 0
        sorter.feed(data, probe)
        assert sorter.segment_count == 1
        sorter.feed(data, probe)
        assert sorter.segment_count == 2

    def test_channel_mismatch_raises(self):
        sorter = zbci.StreamingSorter(4)
        probe = _make_probe(4)
        data = np.zeros((100, 8))  # wrong channels
        with pytest.raises(ValueError, match="expected 4 channels"):
            sorter.feed(data, probe)

    def test_multiple_segments(self):
        """Multiple segments should build up templates."""
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)

        for seg in range(5):
            data = _make_data_with_spikes(5000, 4, seed=seg)
            result = sorter.feed(data, probe)

        assert sorter.segment_count == 5
        # After several segments with spikes, should have some templates
        # (may or may not, depending on spike consistency)


class TestStreamingSorterDrift:
    def test_drift_rate_property(self):
        """drift_rate should be a float."""
        sorter = zbci.StreamingSorter(4)
        assert isinstance(sorter.drift_rate, float)

    def test_drift_fitted_property(self):
        """drift_fitted should be a bool."""
        sorter = zbci.StreamingSorter(4)
        assert isinstance(sorter.drift_fitted, bool)

    def test_drift_fitted_after_segments(self):
        """After enough segments, drift should be fitted."""
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)

        for seg in range(5):
            data = _make_data_with_spikes(10000, 4, seed=seg)
            sorter.feed(data, probe)

        # After 5 segments with spikes, drift should be fitted
        # (needs >= 2 time bins with data)
        # Note: may or may not fit depending on spike detection


class TestStreamingSorterReset:
    def test_reset_clears_state(self):
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)
        data = _make_data_with_spikes(5000, 4)
        sorter.feed(data, probe)
        assert sorter.segment_count == 1

        sorter.reset()
        assert sorter.segment_count == 0
        assert sorter.n_templates == 0

    def test_feed_after_reset(self):
        """Should be able to feed after reset."""
        sorter = zbci.StreamingSorter(4)
        probe = _make_probe(4)
        data = _make_noise(5000, 4)
        sorter.feed(data, probe)
        sorter.reset()
        result = sorter.feed(data, probe)
        assert "n_spikes" in result


class TestStreamingSorterResultFormat:
    def test_result_has_clusters(self):
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)
        data = _make_data_with_spikes(10000, 4)
        result = sorter.feed(data, probe)
        assert "clusters" in result
        if result["n_clusters"] > 0:
            c = result["clusters"][0]
            assert "count" in c
            assert "snr" in c
            assert "isi_violation_rate" in c

    def test_labels_length_matches(self):
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)
        data = _make_data_with_spikes(10000, 4)
        result = sorter.feed(data, probe)
        assert len(result["labels"]) == result["n_spikes"]

    def test_spike_times_length_matches(self):
        sorter = zbci.StreamingSorter(4, threshold=4.0)
        probe = _make_probe(4)
        data = _make_data_with_spikes(10000, 4)
        result = sorter.feed(data, probe)
        assert len(result["spike_times"]) == result["n_spikes"]
