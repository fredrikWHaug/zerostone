"""Tests for zpybci.synthetic -- synthetic recording generation."""

import numpy as np
import pytest

from zpybci.synthetic import generate_recording, generate_templates


# ---------------------------------------------------------------------------
# generate_templates tests
# ---------------------------------------------------------------------------


class TestGenerateTemplates:
    def test_output_shapes(self):
        templates, primary = generate_templates(5, 32, n_samples=61, seed=0)
        assert templates.shape == (5, 32, 61)
        assert primary.shape == (5,)

    def test_dtype(self):
        templates, primary = generate_templates(3, 8, seed=1)
        assert templates.dtype == np.float64
        assert primary.dtype in (np.int64, np.intp)

    def test_primary_channels_in_range(self):
        templates, primary = generate_templates(10, 16, seed=2)
        assert np.all(primary >= 0)
        assert np.all(primary < 16)

    def test_biphasic_shape(self):
        """Templates should have a negative trough followed by a positive peak."""
        templates, primary = generate_templates(3, 4, n_samples=61, seed=3)
        for u in range(3):
            ch = primary[u]
            waveform = templates[u, ch, :]
            # Trough should be negative
            assert np.min(waveform) < 0.0
            # There should be a positive phase
            assert np.max(waveform) > 0.0
            # Trough comes before peak (negative min index < positive max index)
            assert np.argmin(waveform) < np.argmax(waveform)

    def test_spatial_falloff(self):
        """Primary channel should have the strongest signal."""
        templates, primary = generate_templates(5, 32, seed=4)
        for u in range(5):
            ch = primary[u]
            primary_power = np.sum(templates[u, ch, :] ** 2)
            for other_ch in range(32):
                if other_ch != ch:
                    other_power = np.sum(templates[u, other_ch, :] ** 2)
                    assert primary_power >= other_power

    def test_reproducibility(self):
        t1, p1 = generate_templates(3, 8, seed=99)
        t2, p2 = generate_templates(3, 8, seed=99)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        t1, _ = generate_templates(3, 8, seed=0)
        t2, _ = generate_templates(3, 8, seed=1)
        assert not np.allclose(t1, t2)

    def test_single_unit_single_channel(self):
        templates, primary = generate_templates(1, 1, n_samples=31, seed=5)
        assert templates.shape == (1, 1, 31)
        assert primary[0] == 0

    def test_nonzero_templates(self):
        templates, _ = generate_templates(4, 8, seed=6)
        for u in range(4):
            assert np.any(templates[u] != 0.0)


# ---------------------------------------------------------------------------
# generate_recording tests
# ---------------------------------------------------------------------------


class TestGenerateRecording:
    def test_output_keys(self):
        rec = generate_recording(n_channels=4, duration_s=1.0, n_units=2, seed=0)
        expected_keys = {
            "data", "spike_times", "spike_labels", "all_spike_times",
            "templates", "primary_channels", "sampling_rate", "n_units",
        }
        assert set(rec.keys()) == expected_keys

    def test_data_shape(self):
        rec = generate_recording(
            n_channels=8, duration_s=2.0, sampling_rate=30000.0, n_units=3, seed=0
        )
        assert rec["data"].shape == (60000, 8)
        assert rec["data"].dtype == np.float64

    def test_template_shape(self):
        rec = generate_recording(n_channels=16, duration_s=1.0, n_units=4, seed=0)
        assert rec["templates"].shape[0] == 4
        assert rec["templates"].shape[1] == 16

    def test_spike_times_per_unit(self):
        rec = generate_recording(
            n_channels=4, duration_s=5.0, n_units=3, firing_rate=10.0, seed=0
        )
        assert len(rec["spike_times"]) == 3
        for u in range(3):
            times = rec["spike_times"][u]
            assert times.ndim == 1
            # Should have some spikes (5s * 10Hz = ~50 expected per unit)
            assert len(times) > 10

    def test_spike_times_within_bounds(self):
        rec = generate_recording(
            n_channels=4, duration_s=2.0, sampling_rate=30000.0, n_units=2, seed=0
        )
        n_samples = int(2.0 * 30000.0)
        for times in rec["spike_times"]:
            if len(times) > 0:
                assert np.all(times >= 0)
                assert np.all(times < n_samples)

    def test_spike_labels_valid(self):
        rec = generate_recording(n_channels=4, duration_s=3.0, n_units=5, seed=0)
        labels = rec["spike_labels"]
        assert labels.ndim == 1
        if len(labels) > 0:
            assert np.all(labels >= 0)
            assert np.all(labels < 5)

    def test_all_spike_times_sorted(self):
        rec = generate_recording(n_channels=4, duration_s=5.0, n_units=3, seed=0)
        times = rec["all_spike_times"]
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0)

    def test_labels_and_times_same_length(self):
        rec = generate_recording(n_channels=8, duration_s=2.0, n_units=4, seed=0)
        assert len(rec["spike_labels"]) == len(rec["all_spike_times"])

    def test_total_spikes_consistent(self):
        rec = generate_recording(n_channels=4, duration_s=3.0, n_units=3, seed=0)
        total_from_per_unit = sum(len(t) for t in rec["spike_times"])
        assert len(rec["all_spike_times"]) == total_from_per_unit

    def test_refractory_period(self):
        """No two spikes from the same unit within refractory period."""
        refractory_ms = 1.0
        rec = generate_recording(
            n_channels=4, duration_s=10.0, n_units=3,
            firing_rate=20.0, refractory_ms=refractory_ms, seed=0,
        )
        refractory_samples = int(refractory_ms * 30000.0 / 1000.0)
        for u in range(3):
            times = rec["spike_times"][u]
            if len(times) > 1:
                isis = np.diff(times)
                assert np.all(isis >= refractory_samples), (
                    f"Unit {u}: min ISI = {isis.min()}, "
                    f"refractory = {refractory_samples}"
                )

    def test_reproducibility(self):
        r1 = generate_recording(n_channels=4, duration_s=1.0, n_units=2, seed=42)
        r2 = generate_recording(n_channels=4, duration_s=1.0, n_units=2, seed=42)
        np.testing.assert_array_equal(r1["data"], r2["data"])
        np.testing.assert_array_equal(r1["spike_labels"], r2["spike_labels"])
        np.testing.assert_array_equal(r1["all_spike_times"], r2["all_spike_times"])

    def test_different_seeds_differ(self):
        r1 = generate_recording(n_channels=4, duration_s=1.0, n_units=2, seed=0)
        r2 = generate_recording(n_channels=4, duration_s=1.0, n_units=2, seed=1)
        assert not np.array_equal(r1["data"], r2["data"])

    def test_metadata(self):
        rec = generate_recording(
            n_channels=16, duration_s=5.0, sampling_rate=20000.0, n_units=7, seed=0
        )
        assert rec["sampling_rate"] == 20000.0
        assert rec["n_units"] == 7

    def test_single_unit_single_channel(self):
        rec = generate_recording(
            n_channels=1, duration_s=2.0, n_units=1, firing_rate=5.0, seed=0
        )
        assert rec["data"].shape[1] == 1
        assert len(rec["spike_times"]) == 1
        assert rec["templates"].shape[:2] == (1, 1)

    def test_short_duration(self):
        """Very short recording should still work."""
        rec = generate_recording(
            n_channels=4, duration_s=0.01, n_units=2, firing_rate=5.0, seed=0
        )
        assert rec["data"].shape[0] == int(0.01 * 30000.0)

    def test_high_firing_rate(self):
        """High firing rate should produce many spikes."""
        rec = generate_recording(
            n_channels=4, duration_s=5.0, n_units=2, firing_rate=50.0, seed=0
        )
        total = len(rec["all_spike_times"])
        # 2 units * 50 Hz * 5 s = 500 expected
        assert total > 200

    def test_zero_noise(self):
        """With zero noise, signal should be purely from templates."""
        rec = generate_recording(
            n_channels=4, duration_s=1.0, n_units=1,
            firing_rate=5.0, noise_std=0.0, seed=0
        )
        # Between spikes, data should be exactly zero
        # (this is hard to check precisely, but overall data should be mostly zero)
        n_nonzero = np.count_nonzero(rec["data"])
        total_elements = rec["data"].size
        # With ~5 spikes in 1s, each affecting 61*4=244 elements,
        # that's ~1220 out of 30000*4=120000
        assert n_nonzero < total_elements * 0.1

    def test_primary_channels_returned(self):
        rec = generate_recording(n_channels=16, duration_s=1.0, n_units=4, seed=0)
        assert rec["primary_channels"].shape == (4,)
        assert np.all(rec["primary_channels"] >= 0)
        assert np.all(rec["primary_channels"] < 16)

    def test_many_channels(self):
        """Test with 64 channels (matches sort_multichannel max)."""
        rec = generate_recording(
            n_channels=64, duration_s=0.5, n_units=3, seed=0
        )
        assert rec["data"].shape == (15000, 64)
