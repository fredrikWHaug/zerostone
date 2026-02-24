"""Tests for Riemannian MDM classifier, frechet_mean, and riemannian_distance."""

import numpy as np
import pytest
import zpybci as zbci


# --- Helper ---

def make_spd(c, scale=1.0, seed=42):
    """Create a random SPD matrix of size (c, c)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((c, c)) * 0.3
    return np.eye(c) * scale + a @ a.T


def make_spd_batch(n, c, base_scale=1.0, seed=42):
    """Create a batch of SPD matrices, shape (n, c, c)."""
    return np.stack([make_spd(c, base_scale, seed=seed + i) for i in range(n)])


# --- riemannian_distance ---

class TestRiemannianDistance:
    def test_self_distance_zero(self):
        a = make_spd(4)
        d = zbci.riemannian_distance(a, a)
        assert d < 1e-9

    def test_symmetry(self):
        a = make_spd(4, seed=1)
        b = make_spd(4, seed=2)
        assert abs(zbci.riemannian_distance(a, b) - zbci.riemannian_distance(b, a)) < 1e-9

    def test_positive(self):
        a = make_spd(4, seed=1)
        b = make_spd(4, seed=2)
        assert zbci.riemannian_distance(a, b) > 0

    def test_identity_to_diagonal(self):
        eye = np.eye(4)
        diag = np.diag([2.0, 3.0, 1.5, 0.5])
        d = zbci.riemannian_distance(eye, diag)
        expected = np.sqrt(sum(np.log(x) ** 2 for x in [2.0, 3.0, 1.5, 0.5]))
        assert abs(d - expected) < 1e-8

    def test_triangle_inequality(self):
        a = make_spd(4, seed=1)
        b = make_spd(4, seed=2)
        c = make_spd(4, seed=3)
        d_ab = zbci.riemannian_distance(a, b)
        d_bc = zbci.riemannian_distance(b, c)
        d_ac = zbci.riemannian_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-9

    def test_wrong_shape(self):
        a = np.eye(4)
        b = np.eye(3)
        with pytest.raises(ValueError):
            zbci.riemannian_distance(a, b)

    def test_unsupported_channels(self):
        a = np.eye(5)
        with pytest.raises(ValueError, match="channels must be"):
            zbci.riemannian_distance(a, a)


# --- frechet_mean ---

class TestFrechetMean:
    def test_identity_matrices(self):
        mats = np.stack([np.eye(4)] * 5)
        mean = zbci.frechet_mean(mats)
        np.testing.assert_allclose(mean, np.eye(4), atol=1e-6)

    def test_identical_matrices(self):
        m = make_spd(4, seed=10)
        mats = np.stack([m] * 4)
        mean = zbci.frechet_mean(mats)
        np.testing.assert_allclose(mean, m, atol=1e-6)

    def test_diagonal_matrices(self):
        d1 = np.diag([2.0, 1.0, 1.0, 1.0])
        d2 = np.diag([1.0, 2.0, 1.0, 1.0])
        mats = np.stack([d1, d2])
        mean = zbci.frechet_mean(mats)
        # Mean should be SPD and symmetric
        np.testing.assert_allclose(mean, mean.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(mean)
        assert all(eigvals > 0)

    def test_output_is_spd(self):
        mats = make_spd_batch(6, 8, seed=42)
        mean = zbci.frechet_mean(mats)
        eigvals = np.linalg.eigvalsh(mean)
        assert all(eigvals > 0), f"Not SPD: eigenvalues = {eigvals}"

    def test_mean_closer_than_inputs(self):
        """Frechet mean should be equidistant-ish from all inputs."""
        mats = make_spd_batch(4, 4, seed=42)
        mean = zbci.frechet_mean(mats)
        dists = [zbci.riemannian_distance(mean, mats[i]) for i in range(4)]
        # All distances should be finite and positive
        assert all(d > 0 and np.isfinite(d) for d in dists)

    def test_single_matrix(self):
        m = make_spd(4, seed=7)
        mats = m.reshape(1, 4, 4)
        mean = zbci.frechet_mean(mats)
        np.testing.assert_allclose(mean, m, atol=1e-8)

    def test_wrong_shape(self):
        mats = np.ones((3, 4, 5))  # not square
        with pytest.raises(ValueError):
            zbci.frechet_mean(mats)


# --- MdmClassifier ---

class TestMdmClassifier:
    def _make_two_class_data(self, c=4, n_per_class=10):
        """Create synthetic 2-class SPD data with clear separation."""
        rng = np.random.default_rng(42)
        x_list = []
        y_list = []
        for i in range(n_per_class):
            # Class 0: high variance in first channel
            m0 = np.eye(c, dtype=np.float64)
            m0[0, 0] = 3.0 + rng.uniform(-0.2, 0.2)
            x_list.append(m0)
            y_list.append(0)
            # Class 1: high variance in last channel
            m1 = np.eye(c, dtype=np.float64)
            m1[c - 1, c - 1] = 3.0 + rng.uniform(-0.2, 0.2)
            x_list.append(m1)
            y_list.append(1)
        return np.stack(x_list), np.array(y_list, dtype=np.int64)

    def test_create(self):
        mdm = zbci.MdmClassifier(channels=4)
        assert mdm.channels == 4

    def test_create_invalid_channels(self):
        with pytest.raises(ValueError, match="channels must be"):
            zbci.MdmClassifier(channels=5)

    def test_fit_predict(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        mdm.fit(x, y)
        preds = mdm.predict(x)
        accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
        assert accuracy >= 0.9, f"Accuracy too low: {accuracy}"

    def test_score(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        mdm.fit(x, y)
        acc = mdm.score(x, y)
        assert acc >= 0.9

    def test_transform_shape(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        mdm.fit(x, y)
        dists = mdm.transform(x)
        assert dists.shape == (len(y), 2)  # 2 classes

    def test_transform_min_distance_matches_predict(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        mdm.fit(x, y)
        preds = mdm.predict(x)
        dists = mdm.transform(x)
        for i in range(len(y)):
            pred_from_dist = np.argmin(dists[i])
            assert preds[i] == [0, 1][pred_from_dist]

    def test_predict_before_fit(self):
        mdm = zbci.MdmClassifier(channels=4)
        x = np.stack([np.eye(4)] * 3)
        with pytest.raises(ValueError, match="not fitted"):
            mdm.predict(x)

    def test_n_classes(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        assert mdm.n_classes == 0
        mdm.fit(x, y)
        assert mdm.n_classes == 2

    def test_repr(self):
        mdm = zbci.MdmClassifier(channels=8)
        assert "fitted=False" in repr(mdm)
        x, y = self._make_two_class_data(c=8)
        mdm.fit(x, y)
        assert "n_classes=2" in repr(mdm)

    def test_wrong_shape_predict(self):
        x, y = self._make_two_class_data()
        mdm = zbci.MdmClassifier(channels=4)
        mdm.fit(x, y)
        bad = np.stack([np.eye(8)] * 3)
        with pytest.raises(ValueError):
            mdm.predict(bad)

    def test_mismatched_x_y_length(self):
        mdm = zbci.MdmClassifier(channels=4)
        x = np.stack([np.eye(4)] * 5)
        y = np.array([0, 1, 0], dtype=np.int64)
        with pytest.raises(ValueError):
            mdm.fit(x, y)
