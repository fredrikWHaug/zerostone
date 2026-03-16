"""Tests for spike localization bindings."""

import numpy as np
import pytest

import zpybci as zbci


class TestCenterOfMass:
    def test_single_dominant_channel(self):
        amps = np.array([0.0, 0.0, -5.0, 0.0])
        pos = np.array([[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]])
        loc = zbci.center_of_mass(amps, pos)
        np.testing.assert_allclose(loc, [0.0, 50.0], atol=1e-9)

    def test_equal_weights_returns_centroid(self):
        amps = np.array([1.0, 1.0, 1.0, 1.0])
        pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        loc = zbci.center_of_mass(amps, pos)
        np.testing.assert_allclose(loc, [5.0, 5.0], atol=1e-9)

    def test_all_zero_returns_origin(self):
        amps = np.zeros(4)
        pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        loc = zbci.center_of_mass(amps, pos)
        np.testing.assert_allclose(loc, [0.0, 0.0], atol=1e-9)

    def test_negative_amps_uses_absolute_value(self):
        amps = np.array([-3.0, -1.0, 0.0, 0.0])
        pos = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0], [0.0, 300.0]])
        loc = zbci.center_of_mass(amps, pos)
        expected_y = (3.0 * 0.0 + 1.0 * 100.0) / (3.0 + 1.0)
        np.testing.assert_allclose(loc[1], expected_y, atol=1e-9)

    def test_8_channels(self):
        amps = np.zeros(8)
        amps[3] = -10.0
        pos = np.zeros((8, 2))
        for i in range(8):
            pos[i] = [0.0, i * 25.0]
        loc = zbci.center_of_mass(amps, pos)
        np.testing.assert_allclose(loc[1], 75.0, atol=1e-9)

    def test_invalid_channel_count(self):
        with pytest.raises(ValueError):
            zbci.center_of_mass(np.zeros(3), np.zeros((3, 2)))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            zbci.center_of_mass(np.zeros(4), np.zeros((8, 2)))


class TestCenterOfMassThreshold:
    def test_filters_low_channels(self):
        amps = np.array([0.1, 0.2, -5.0, -3.0])
        pos = np.array([[0.0, 0.0], [0.0, 25.0], [0.0, 50.0], [0.0, 75.0]])
        loc = zbci.center_of_mass_threshold(amps, pos, 1.0)
        assert loc is not None
        expected_y = (5.0 * 50.0 + 3.0 * 75.0) / (5.0 + 3.0)
        np.testing.assert_allclose(loc[1], expected_y, atol=1e-6)

    def test_none_when_all_below(self):
        amps = np.array([0.1, 0.2, 0.05, 0.01])
        pos = np.zeros((4, 2))
        assert zbci.center_of_mass_threshold(amps, pos, 1.0) is None

    def test_zero_threshold_same_as_com(self):
        amps = np.array([1.0, 2.0, 3.0, 4.0])
        pos = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0], [0.0, 30.0]])
        com = zbci.center_of_mass(amps, pos)
        thresh = zbci.center_of_mass_threshold(amps, pos, 0.0)
        assert thresh is not None
        np.testing.assert_allclose(thresh, com, atol=1e-9)


class TestMonopoleLocalize:
    def test_recovers_known_source(self):
        positions = np.array([
            [-16.0, 0.0], [16.0, 0.0],
            [-16.0, 25.0], [16.0, 25.0],
            [-16.0, 50.0], [16.0, 50.0],
            [-16.0, 75.0], [16.0, 75.0],
        ])
        src = [16.0, 37.5]
        z = 10.0
        amps = np.zeros(8)
        for i in range(8):
            dx = src[0] - positions[i, 0]
            dy = src[1] - positions[i, 1]
            r = np.sqrt(dx**2 + dy**2 + z**2)
            amps[i] = 100.0 / r
        loc = zbci.monopole_localize(amps, positions, z, n_iter=10)
        assert loc is not None
        assert abs(loc[0] - 16.0) < 1.0
        assert abs(loc[1] - 37.5) < 1.0

    def test_all_zero_returns_none(self):
        assert zbci.monopole_localize(np.zeros(4), np.zeros((4, 2)), 10.0) is None

    def test_default_n_iter(self):
        amps = np.array([5.0, 1.0, 1.0, 1.0])
        pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        loc = zbci.monopole_localize(amps, pos, 10.0)
        assert loc is not None
        assert np.all(np.isfinite(loc))

    def test_single_channel(self):
        amps = np.array([5.0, 0.0, 0.0, 0.0])
        pos = np.array([[42.0, 17.0], [0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        loc = zbci.monopole_localize(amps, pos, 10.0)
        assert loc is not None
        np.testing.assert_allclose(loc, [42.0, 17.0], atol=1e-6)
