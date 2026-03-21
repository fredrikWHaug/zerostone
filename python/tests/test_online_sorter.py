"""Tests for online spike sorter bindings."""
import pytest
import zpybci as zbci


class TestOnlineSorter:
    def test_basic_classify(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        sorter.add_template([0.0, 1.0, 0.0])

        label, dist = sorter.classify([0.9, 0.1, 0.0])
        assert label == 0
        assert dist < 0.2

        label, dist = sorter.classify([0.1, 0.9, 0.0])
        assert label == 1

    def test_from_centroids(self):
        centroids = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        sorter = zbci.OnlineSorter.from_centroids(centroids)
        assert sorter.n_templates == 2

    def test_reject(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.set_max_distance(1.0)

        result = sorter.classify_or_reject([0.1, 0.1, 0.1])
        assert result is not None

        result = sorter.classify_or_reject([10.0, 10.0, 10.0])
        assert result is None
        assert sorter.n_rejected == 1

    def test_counters(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.classify([0.1, 0.1, 0.1])
        sorter.classify([0.2, 0.2, 0.2])
        assert sorter.n_classified == 2
        sorter.reset_counters()
        assert sorter.n_classified == 0

    def test_reset(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        assert sorter.n_templates == 1
        sorter.reset()
        assert sorter.n_templates == 0

    def test_full(self):
        sorter = zbci.OnlineSorter()
        for i in range(16):
            assert sorter.add_template([float(i), 0.0, 0.0]) is not None
        assert sorter.add_template([99.0, 0.0, 0.0]) is None

    def test_wrong_dim_raises(self):
        sorter = zbci.OnlineSorter()
        with pytest.raises((ValueError, Exception)):
            sorter.classify([1.0, 2.0])  # wrong length

    def test_repr(self):
        sorter = zbci.OnlineSorter()
        assert "OnlineSorter" in repr(sorter)


class TestOnlineSorterExpanded:
    def test_classify_multiple_calls(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        for _ in range(100):
            sorter.classify([1.0, 0.1, 0.0])
        assert sorter.n_classified == 100

    def test_reject_all(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([0.0, 0.0, 0.0])
        sorter.set_max_distance(0.0)
        for _ in range(10):
            result = sorter.classify_or_reject([1.0, 1.0, 1.0])
            assert result is None
        assert sorter.n_rejected == 10

    def test_template_capacity(self):
        sorter = zbci.OnlineSorter()
        added = 0
        for i in range(20):
            idx = sorter.add_template([float(i), 0.0, 0.0])
            if idx is not None:
                added += 1
        assert added == 16  # max capacity is 16
        assert sorter.n_templates == 16

    def test_classify_equidistant(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        sorter.add_template([-1.0, 0.0, 0.0])
        # Point at origin is equidistant from both
        label, dist = sorter.classify([0.0, 0.0, 0.0])
        assert label == 0  # should pick lower index

    def test_zero_distance(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([3.0, 4.0, 5.0])
        label, dist = sorter.classify([3.0, 4.0, 5.0])
        assert label == 0
        assert dist < 1e-10

    def test_from_centroids_empty(self):
        sorter = zbci.OnlineSorter.from_centroids([])
        assert sorter.n_templates == 0

    def test_negative_features(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([-5.0, -3.0, -1.0])
        label, dist = sorter.classify([-5.0, -3.0, -1.0])
        assert label == 0
        assert dist < 1e-10

    def test_large_features(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1e10, 1e10, 1e10])
        label, dist = sorter.classify([1e10, 1e10, 1e10])
        assert label == 0
        # Should not crash or produce NaN

    def test_classify_no_templates(self):
        sorter = zbci.OnlineSorter()
        # classify with 0 templates -- returns (0, inf) or similar
        label, dist = sorter.classify([1.0, 2.0, 3.0])
        # Just verify it doesn't crash; label/dist may be arbitrary
        assert isinstance(label, int)
        assert isinstance(dist, float)

    def test_reset_then_classify(self):
        sorter = zbci.OnlineSorter()
        sorter.add_template([1.0, 0.0, 0.0])
        sorter.classify([1.0, 0.0, 0.0])
        assert sorter.n_classified == 1
        sorter.reset()
        assert sorter.n_templates == 0
        assert sorter.n_classified == 0
        # Re-add and classify again
        sorter.add_template([0.0, 1.0, 0.0])
        label, _ = sorter.classify([0.0, 1.0, 0.0])
        assert label == 0
