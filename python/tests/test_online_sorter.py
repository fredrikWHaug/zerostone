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
