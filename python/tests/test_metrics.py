"""Tests for zpybci comparison metrics bindings."""

import numpy as np
import pytest

import zpybci as zbci


class TestCompareSpikeTrains:
    def test_perfect_match(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([100, 200, 300], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 0)
        assert m["true_positives"] == 3
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0
        assert abs(m["accuracy"] - 1.0) < 1e-10

    def test_within_tolerance(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([101, 199, 302], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 3
        assert abs(m["accuracy"] - 1.0) < 1e-10

    def test_outside_tolerance(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([110, 220, 330], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 3
        assert m["false_positives"] == 3

    def test_partial_overlap(self):
        gt = np.array([100, 200, 300, 400], dtype=np.int64)
        s = np.array([101, 199, 350, 401], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 3
        assert m["false_negatives"] == 1
        assert m["false_positives"] == 1
        assert abs(m["accuracy"] - 0.6) < 1e-10

    def test_empty_gt(self):
        gt = np.array([], dtype=np.int64)
        s = np.array([100, 200], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0
        assert m["false_positives"] == 2

    def test_empty_sorted(self):
        gt = np.array([100, 200], dtype=np.int64)
        s = np.array([], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 2

    def test_both_empty(self):
        gt = np.array([], dtype=np.int64)
        s = np.array([], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0

    def test_default_tolerance(self):
        gt = np.array([100], dtype=np.int64)
        s = np.array([112], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s)  # default tolerance=12
        assert m["true_positives"] == 1

    def test_precision_recall(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([101, 201, 500, 600], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 2
        assert abs(m["precision"] - 0.5) < 1e-10
        assert abs(m["recall"] - 2.0 / 3.0) < 1e-10

    def test_returns_all_keys(self):
        gt = np.array([100], dtype=np.int64)
        s = np.array([100], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        expected = {"true_positives", "false_positives", "false_negatives",
                    "accuracy", "precision", "recall"}
        assert set(m.keys()) == expected


class TestCompareSorting:
    def test_two_units_perfect(self):
        gt = [np.array([100, 200, 300], dtype=np.int64),
              np.array([150, 250, 350], dtype=np.int64)]
        s = [np.array([101, 201, 301], dtype=np.int64),
             np.array([149, 251, 349], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s, 5)
        assert len(results) == 2
        assert results[0]["true_positives"] == 3
        assert results[1]["true_positives"] == 3

    def test_swapped_order(self):
        gt = [np.array([100, 200, 300], dtype=np.int64),
              np.array([1000, 2000, 3000], dtype=np.int64)]
        s = [np.array([1001, 2001, 3001], dtype=np.int64),
             np.array([101, 201, 301], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s, 5)
        assert results[0]["true_positives"] == 3
        assert results[1]["true_positives"] == 3

    def test_unmatched_gt(self):
        gt = [np.array([100, 200, 300], dtype=np.int64)]
        s = [np.array([1000, 2000, 3000], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s, 5)
        assert results[0]["false_negatives"] == 3
        assert results[0]["accuracy"] < 1e-10

    def test_empty_gt(self):
        s = [np.array([100, 200], dtype=np.int64)]
        results = zbci.compare_sorting([], s, 5)
        assert len(results) == 0

    def test_empty_sorted(self):
        gt = [np.array([100, 200], dtype=np.int64)]
        results = zbci.compare_sorting(gt, [], 5)
        assert len(results) == 1
        # Early return fills with empty matches (no sorted trains to compare against)
        assert results[0]["true_positives"] == 0

    def test_default_tolerance(self):
        gt = [np.array([100], dtype=np.int64)]
        s = [np.array([112], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s)  # default tolerance=12
        assert results[0]["true_positives"] == 1


class TestCompareExpanded:
    def test_compare_empty_trains(self):
        gt = np.array([], dtype=np.int64)
        s = np.array([], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_compare_identical(self):
        times = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        m = zbci.compare_spike_trains(times, times.copy(), 0)
        assert m["true_positives"] == 5
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0
        assert abs(m["accuracy"] - 1.0) < 1e-10

    def test_compare_no_overlap(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([1000, 2000, 3000], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert m["true_positives"] == 0
        assert m["false_negatives"] == 3
        assert m["false_positives"] == 3

    def test_compare_tolerance(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([103, 203, 303], dtype=np.int64)
        # With tolerance 2, offset of 3 should miss
        m2 = zbci.compare_spike_trains(gt, s, 2)
        assert m2["true_positives"] == 0
        # With tolerance 5, offset of 3 should match
        m5 = zbci.compare_spike_trains(gt, s, 5)
        assert m5["true_positives"] == 3

    def test_compare_sorting_basic(self):
        gt = [np.array([100, 300, 500], dtype=np.int64),
              np.array([200, 400, 600], dtype=np.int64)]
        s = [np.array([101, 301, 501], dtype=np.int64),
             np.array([199, 399, 599], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s, 5)
        assert len(results) == 2
        for r in results:
            assert r["true_positives"] == 3
            assert abs(r["accuracy"] - 1.0) < 1e-10

    def test_compare_sorting_empty_units(self):
        gt = [np.array([], dtype=np.int64)]
        s = [np.array([], dtype=np.int64)]
        results = zbci.compare_sorting(gt, s, 5)
        assert len(results) == 1
        assert results[0]["true_positives"] == 0

    def test_accuracy_range(self):
        gt = np.array([100, 200, 300, 400], dtype=np.int64)
        s = np.array([101, 199, 350, 600], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_precision_range(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([101, 201, 500, 600, 700], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert 0.0 <= m["precision"] <= 1.0

    def test_recall_range(self):
        gt = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        s = np.array([101], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 5)
        assert 0.0 <= m["recall"] <= 1.0

    def test_compare_large_tolerance(self):
        gt = np.array([100, 200, 300], dtype=np.int64)
        s = np.array([150, 250, 350], dtype=np.int64)
        m = zbci.compare_spike_trains(gt, s, 100)
        assert m["true_positives"] == 3
        assert abs(m["accuracy"] - 1.0) < 1e-10
