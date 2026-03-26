"""Tests for probe geometry bindings."""

import math

import pytest

import zpybci as zbci


class TestProbeLayoutCreation:
    def test_linear_4ch(self):
        probe = zbci.ProbeLayout.linear(4, 25.0)
        assert probe.n_channels == 4

    def test_linear_8ch(self):
        probe = zbci.ProbeLayout.linear(8, 20.0)
        assert probe.n_channels == 8

    def test_linear_16ch(self):
        probe = zbci.ProbeLayout.linear(16, 10.0)
        assert probe.n_channels == 16

    def test_linear_32ch(self):
        probe = zbci.ProbeLayout.linear(32, 25.0)
        assert probe.n_channels == 32

    def test_linear_64ch(self):
        probe = zbci.ProbeLayout.linear(64, 25.0)
        assert probe.n_channels == 64

    def test_linear_128ch(self):
        probe = zbci.ProbeLayout.linear(128, 25.0)
        assert probe.n_channels == 128

    def test_invalid_channel_count(self):
        with pytest.raises(ValueError, match="n_channels"):
            zbci.ProbeLayout.linear(5, 25.0)

    def test_from_positions(self):
        positions = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
        probe = zbci.ProbeLayout(positions)
        assert probe.n_channels == 4

    def test_from_positions_invalid_count(self):
        positions = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        with pytest.raises(ValueError, match="n_channels"):
            zbci.ProbeLayout(positions)

    def test_tetrode(self):
        probe = zbci.ProbeLayout.tetrode(25.0)
        assert probe.n_channels == 4


class TestChannelDistance:
    def test_linear_adjacent(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        d = probe.channel_distance(0, 1)
        assert abs(d - 10.0) < 1e-9

    def test_linear_two_apart(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        d = probe.channel_distance(0, 2)
        assert abs(d - 20.0) < 1e-9

    def test_self_distance_zero(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        d = probe.channel_distance(2, 2)
        assert abs(d) < 1e-12

    def test_symmetry(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        d12 = probe.channel_distance(1, 5)
        d21 = probe.channel_distance(5, 1)
        assert abs(d12 - d21) < 1e-12

    def test_out_of_range_nan(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        d = probe.channel_distance(0, 10)
        assert math.isnan(d)

    def test_tetrode_side(self):
        probe = zbci.ProbeLayout.tetrode(20.0)
        d = probe.channel_distance(0, 1)
        assert abs(d - 20.0) < 1e-9

    def test_tetrode_diagonal(self):
        probe = zbci.ProbeLayout.tetrode(20.0)
        d = probe.channel_distance(0, 3)
        assert abs(d - 20.0 * math.sqrt(2.0)) < 1e-9


class TestNeighborChannels:
    def test_linear_neighbors(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        neighbors = probe.neighbor_channels(3, 15.0)
        assert sorted(neighbors) == [2, 4]

    def test_large_radius_all(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        neighbors = probe.neighbor_channels(0, 1000.0)
        assert len(neighbors) == 3

    def test_zero_radius_empty(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        neighbors = probe.neighbor_channels(0, 0.0)
        assert len(neighbors) == 0

    def test_out_of_range_empty(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        neighbors = probe.neighbor_channels(10, 100.0)
        assert len(neighbors) == 0

    def test_tetrode_neighbors(self):
        probe = zbci.ProbeLayout.tetrode(20.0)
        # radius 25: side=20 within, diagonal=28.28 not within
        neighbors = probe.neighbor_channels(0, 25.0)
        assert len(neighbors) == 2
        assert 1 in neighbors
        assert 2 in neighbors


class TestNearestChannels:
    def test_linear_nearest(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        nearest = probe.nearest_channels(3, 2)
        assert len(nearest) == 2
        assert set(nearest) == {2, 4}

    def test_k_exceeds_channels(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        nearest = probe.nearest_channels(0, 100)
        assert len(nearest) == 3  # C-1

    def test_k_zero(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        nearest = probe.nearest_channels(0, 0)
        assert len(nearest) == 0

    def test_out_of_range_empty(self):
        probe = zbci.ProbeLayout.linear(4, 10.0)
        nearest = probe.nearest_channels(10, 3)
        assert len(nearest) == 0

    def test_sorted_by_distance(self):
        probe = zbci.ProbeLayout.linear(8, 10.0)
        nearest = probe.nearest_channels(0, 4)
        # Channel 0 neighbors sorted by distance: 1(10), 2(20), 3(30), 4(40)
        assert nearest == [1, 2, 3, 4]


class TestSpatialExtent:
    def test_linear(self):
        probe = zbci.ProbeLayout.linear(8, 20.0)
        xr, yr = probe.spatial_extent()
        assert abs(xr) < 1e-9
        assert abs(yr - 140.0) < 1e-9

    def test_tetrode(self):
        probe = zbci.ProbeLayout.tetrode(30.0)
        xr, yr = probe.spatial_extent()
        assert abs(xr - 30.0) < 1e-9
        assert abs(yr - 30.0) < 1e-9

    def test_custom_positions(self):
        positions = [[0.0, 0.0], [100.0, 0.0], [0.0, 200.0], [100.0, 200.0]]
        probe = zbci.ProbeLayout(positions)
        xr, yr = probe.spatial_extent()
        assert abs(xr - 100.0) < 1e-9
        assert abs(yr - 200.0) < 1e-9


class TestRepr:
    def test_repr_contains_channels(self):
        probe = zbci.ProbeLayout.linear(8, 25.0)
        r = repr(probe)
        assert "ProbeLayout" in r
        assert "8" in r

    def test_tetrode_repr(self):
        probe = zbci.ProbeLayout.tetrode(25.0)
        r = repr(probe)
        assert "4" in r


class TestLargeProbe:
    def test_64ch_operations(self):
        probe = zbci.ProbeLayout.linear(64, 25.0)
        d = probe.channel_distance(0, 63)
        assert abs(d - 63 * 25.0) < 1e-9
        neighbors = probe.neighbor_channels(32, 30.0)
        assert len(neighbors) > 0
        nearest = probe.nearest_channels(32, 5)
        assert len(nearest) == 5

    def test_128ch_operations(self):
        probe = zbci.ProbeLayout.linear(128, 20.0)
        d = probe.channel_distance(0, 127)
        assert abs(d - 127 * 20.0) < 1e-9
        xr, yr = probe.spatial_extent()
        assert abs(yr - 127 * 20.0) < 1e-9


class TestProbePresets:
    def test_neuropixels_1_channels(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        assert probe.n_channels == 384

    def test_neuropixels_1_two_column(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        xr, yr = probe.spatial_extent()
        assert abs(xr - 32.0) < 1e-9  # 2-column, 32um x-pitch

    def test_neuropixels_1_distance(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        d = probe.channel_distance(0, 1)
        assert d > 0.0 and not math.isnan(d)

    def test_neuropixels_1_neighbors(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        neighbors = probe.neighbor_channels(100, 50.0)
        assert len(neighbors) > 0

    def test_neuropixels_1_nearest(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        nearest = probe.nearest_channels(100, 5)
        assert len(nearest) == 5

    def test_neuropixels_2_channels(self):
        probe = zbci.ProbeLayout.neuropixels_2()
        assert probe.n_channels == 384

    def test_neuropixels_2_staggered(self):
        probe = zbci.ProbeLayout.neuropixels_2()
        xr, yr = probe.spatial_extent()
        assert abs(xr - 48.0) < 1e-9  # 2 base cols * 32um + 16um stagger = 48

    def test_utah_array_channels(self):
        probe = zbci.ProbeLayout.utah_array()
        assert probe.n_channels == 96

    def test_utah_array_extent(self):
        probe = zbci.ProbeLayout.utah_array()
        xr, yr = probe.spatial_extent()
        assert abs(xr - 3600.0) < 1e-9  # 10 cols * 400um pitch -> 9*400=3600
        assert abs(yr - 3600.0) < 1e-9

    def test_utah_array_neighbors(self):
        probe = zbci.ProbeLayout.utah_array()
        neighbors = probe.neighbor_channels(50, 500.0)
        assert len(neighbors) > 0

    def test_utah_array_nearest(self):
        probe = zbci.ProbeLayout.utah_array()
        nearest = probe.nearest_channels(50, 4)
        assert len(nearest) == 4

    def test_linear_96ch(self):
        probe = zbci.ProbeLayout.linear(96, 25.0)
        assert probe.n_channels == 96
        d = probe.channel_distance(0, 95)
        assert abs(d - 95 * 25.0) < 1e-9

    def test_linear_384ch(self):
        probe = zbci.ProbeLayout.linear(384, 20.0)
        assert probe.n_channels == 384
        d = probe.channel_distance(0, 383)
        assert abs(d - 383 * 20.0) < 1e-9

    def test_neuropixels_1_repr(self):
        probe = zbci.ProbeLayout.neuropixels_1()
        assert "384" in repr(probe)
