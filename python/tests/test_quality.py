"""Tests for cluster quality metrics bindings."""

import math

import numpy as np
import pytest

import zpybci as zbci


class TestIsiViolationRate:
    def test_no_violations(self):
        spikes = np.array([0.0, 0.050, 0.100, 0.150, 0.200])
        rate = zbci.isi_violation_rate(spikes, 0.001)
        assert rate is not None
        assert rate < 1e-10

    def test_one_violation(self):
        spikes = np.array([0.0, 0.050, 0.100, 0.1005, 0.200])
        rate = zbci.isi_violation_rate(spikes, 0.001)
        assert rate is not None
        assert abs(rate - 0.25) < 1e-10

    def test_all_violations(self):
        spikes = np.array([0.0, 0.0001, 0.0002, 0.0003])
        rate = zbci.isi_violation_rate(spikes, 0.001)
        assert rate is not None
        assert abs(rate - 1.0) < 1e-10

    def test_insufficient_spikes(self):
        assert zbci.isi_violation_rate(np.array([]), 0.001) is None
        assert zbci.isi_violation_rate(np.array([0.0]), 0.001) is None

    def test_two_spikes_clean(self):
        rate = zbci.isi_violation_rate(np.array([0.0, 0.050]), 0.001)
        assert rate is not None
        assert rate < 1e-10


class TestContaminationRate:
    def test_no_violations(self):
        spikes = np.array([0.0, 0.050, 0.100, 0.150, 0.200])
        c = zbci.contamination_rate(spikes, 0.0015, 1.0)
        assert c is not None
        assert c < 1e-10

    def test_with_violations(self):
        spikes = np.array([0.0, 0.050, 0.100, 0.1005, 0.200, 0.300])
        c = zbci.contamination_rate(spikes, 0.0015, 1.0)
        assert c is not None
        assert c > 0.0
        assert c <= 1.0

    def test_clamped_at_one(self):
        spikes = np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004])
        c = zbci.contamination_rate(spikes, 0.001, 0.001)
        assert c is not None
        assert c <= 1.0

    def test_insufficient_input(self):
        assert zbci.contamination_rate(np.array([]), 0.001, 1.0) is None
        assert zbci.contamination_rate(np.array([0.0]), 0.001, 1.0) is None
        assert zbci.contamination_rate(np.array([0.0, 0.1]), 0.001, 0.0) is None
        assert zbci.contamination_rate(np.array([0.0, 0.1]), 0.0, 1.0) is None


class TestSilhouetteScore:
    def test_well_separated(self):
        intra = np.array([0.1, 0.2, 0.15])
        inter = np.array([5.0, 8.0])
        s = zbci.silhouette_score(intra, inter)
        assert s is not None
        assert s > 0.9

    def test_overlapping(self):
        intra = np.array([5.0, 6.0, 7.0])
        inter = np.array([2.0, 3.0])
        s = zbci.silhouette_score(intra, inter)
        assert s is not None
        assert s < 0.0

    def test_perfect_separation(self):
        intra = np.array([0.0, 0.0, 0.0])
        inter = np.array([5.0])
        s = zbci.silhouette_score(intra, inter)
        assert s is not None
        assert abs(s - 1.0) < 1e-10

    def test_empty_input(self):
        assert zbci.silhouette_score(np.array([]), np.array([1.0])) is None
        assert zbci.silhouette_score(np.array([1.0]), np.array([])) is None

    def test_range(self):
        intra = np.array([1.0, 2.0, 3.0])
        inter = np.array([2.5])
        s = zbci.silhouette_score(intra, inter)
        assert s is not None
        assert -1.0 <= s <= 1.0


class TestMeanSilhouette:
    def test_uniform(self):
        intra = np.array([0.1, 0.2, 0.15, 0.1, 0.12, 0.18])
        inter = np.array([5.0, 5.0, 5.0])
        s = zbci.mean_silhouette(3, intra, 2, inter, 1)
        assert s is not None
        assert s > 0.9

    def test_empty(self):
        assert zbci.mean_silhouette(0, np.array([]), 0, np.array([]), 0) is None


class TestWaveformSnr:
    def test_known_snr(self):
        wf = np.array([0.0, -1.0, -5.0, -3.0, 0.0, 1.0, 0.5, 0.0])
        snr = zbci.waveform_snr(wf, 1.0)
        assert snr is not None
        # peak-to-peak = 1.0 - (-5.0) = 6.0, SNR = 6.0 / 2.0 = 3.0
        assert abs(snr - 3.0) < 1e-10

    def test_flat_waveform(self):
        wf = np.full(10, 3.0)
        snr = zbci.waveform_snr(wf, 1.0)
        assert snr is not None
        assert snr < 1e-10

    def test_zero_noise(self):
        wf = np.array([0.0, -5.0, 0.0])
        assert zbci.waveform_snr(wf, 0.0) is None

    def test_negative_noise(self):
        wf = np.array([0.0, -5.0, 0.0])
        assert zbci.waveform_snr(wf, -1.0) is None

    def test_empty_waveform(self):
        assert zbci.waveform_snr(np.array([]), 1.0) is None

    def test_single_sample(self):
        snr = zbci.waveform_snr(np.array([5.0]), 1.0)
        assert snr is not None
        assert snr < 1e-10


class TestDPrime:
    def test_well_separated(self):
        a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        b = np.array([5.0, 5.1, 4.9, 5.05, 4.95])
        dp = zbci.d_prime(a, b)
        assert dp is not None
        assert dp > 10.0

    def test_overlapping(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        dp = zbci.d_prime(a, b)
        assert dp is not None
        assert 0.0 < dp < 3.0

    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dp = zbci.d_prime(a, b)
        assert dp is not None
        assert dp < 1e-10

    def test_symmetric(self):
        a = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        b = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        dp_ab = zbci.d_prime(a, b)
        dp_ba = zbci.d_prime(b, a)
        assert dp_ab is not None and dp_ba is not None
        assert abs(dp_ab - dp_ba) < 1e-10

    def test_insufficient(self):
        assert zbci.d_prime(np.array([1.0]), np.array([2.0, 3.0])) is None
        assert zbci.d_prime(np.array([1.0, 2.0]), np.array([3.0])) is None
        assert zbci.d_prime(np.array([]), np.array([])) is None

    def test_zero_variance(self):
        a = np.array([3.0, 3.0, 3.0])
        b = np.array([7.0, 7.0, 7.0])
        dp = zbci.d_prime(a, b)
        assert dp is not None
        assert math.isinf(dp)


class TestIsolationDistance:
    def test_basic(self):
        distances = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        iso = zbci.isolation_distance(3, distances)
        assert iso is not None
        # Sorted: [1.0, 2.0, 3.0, 5.0, 8.0], 3rd = 3.0
        assert abs(iso - 3.0) < 1e-10

    def test_sorted_input(self):
        distances = np.array([1.0, 2.0, 3.0, 4.0])
        iso = zbci.isolation_distance(2, distances)
        assert iso is not None
        assert abs(iso - 2.0) < 1e-10

    def test_single(self):
        distances = np.array([42.0])
        iso = zbci.isolation_distance(1, distances)
        assert iso is not None
        assert abs(iso - 42.0) < 1e-10

    def test_insufficient(self):
        distances = np.array([1.0, 2.0])
        assert zbci.isolation_distance(3, distances) is None
        assert zbci.isolation_distance(0, distances) is None

    def test_original_not_modified(self):
        """The Python binding copies data, so input should be unchanged."""
        distances = np.array([5.0, 1.0, 3.0])
        original = distances.copy()
        zbci.isolation_distance(2, distances)
        np.testing.assert_array_equal(distances, original)


class TestEuclideanDistance:
    def test_basic(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        d = zbci.euclidean_distance(a, b)
        assert abs(d - math.sqrt(2.0)) < 1e-10

    def test_same_point(self):
        a = np.array([1.0, 2.0, 3.0])
        d = zbci.euclidean_distance(a, a)
        assert d < 1e-10

    def test_empty(self):
        d = zbci.euclidean_distance(np.array([]), np.array([]))
        assert d < 1e-10

    def test_1d(self):
        a = np.array([3.0])
        b = np.array([7.0])
        d = zbci.euclidean_distance(a, b)
        assert abs(d - 4.0) < 1e-10

    def test_known_3_4_5(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        d = zbci.euclidean_distance(a, b)
        assert abs(d - 5.0) < 1e-10
