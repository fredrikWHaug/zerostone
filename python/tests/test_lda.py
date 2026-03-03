"""Tests for LDA (Linear Discriminant Analysis) Python bindings."""

import numpy as np
import pytest
import zpybci as zbci


def make_gaussian_data(n_per_class, n_features, mean0, mean1, std=1.0, seed=42):
    """Generate two-class Gaussian data."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(mean0, std, size=(n_per_class, n_features))
    X1 = rng.normal(mean1, std, size=(n_per_class, n_features))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return X, y


class TestLdaConstruction:
    @pytest.mark.parametrize("f", [2, 4, 6, 8, 12, 16, 32, 64])
    def test_supported_features(self, f):
        lda = zbci.Lda(features=f)
        assert lda.features == f
        assert not lda.is_fitted

    def test_unsupported_features(self):
        with pytest.raises(ValueError, match="features must be"):
            zbci.Lda(features=3)

    def test_custom_shrinkage(self):
        lda = zbci.Lda(features=4, shrinkage=0.1)
        assert not lda.is_fitted

    def test_invalid_shrinkage(self):
        with pytest.raises(ValueError, match="shrinkage"):
            zbci.Lda(features=4, shrinkage=-0.1)

    def test_repr(self):
        lda = zbci.Lda(features=4)
        assert "Lda" in repr(lda)
        assert "features=4" in repr(lda)
        assert "fitted=false" in repr(lda)


class TestLdaPerfectSeparation:
    def test_100_percent_accuracy(self):
        mean0 = [0.0, 0.0]
        mean1 = [10.0, 10.0]
        X, y = make_gaussian_data(50, 2, mean0, mean1, std=0.5)

        lda = zbci.Lda(features=2, shrinkage=0.01)
        lda.fit(X, y)
        assert lda.is_fitted

        accuracy = lda.score(X, y)
        assert accuracy > 0.99, f"Expected near-perfect accuracy, got {accuracy}"

    def test_predictions_match_labels(self):
        mean0 = [0.0, 0.0, 0.0, 0.0]
        mean1 = [5.0, 5.0, 5.0, 5.0]
        X, y = make_gaussian_data(50, 4, mean0, mean1, std=0.3)

        lda = zbci.Lda(features=4)
        lda.fit(X, y)

        preds = lda.predict(X)
        assert preds.shape == (100,)
        np.testing.assert_array_equal(preds, y)


class TestLdaOverlapping:
    def test_reasonable_accuracy(self):
        mean0 = [0.0, 0.0]
        mean1 = [2.0, 2.0]
        X, y = make_gaussian_data(100, 2, mean0, mean1, std=1.0, seed=55)

        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        accuracy = lda.score(X, y)
        assert accuracy > 0.70, f"Expected reasonable accuracy, got {accuracy}"


class TestLdaProbabilities:
    def test_extreme_probabilities(self):
        mean0 = [0.0, 0.0]
        mean1 = [5.0, 5.0]
        X, y = make_gaussian_data(50, 2, mean0, mean1, std=0.5)

        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        # Far class 0 point
        far0 = np.array([[-5.0, -5.0]])
        p0 = lda.predict_proba(far0)
        assert p0[0] > 0.95, f"Far class 0 should have P>0.95, got {p0[0]}"

        # Far class 1 point
        far1 = np.array([[10.0, 10.0]])
        p1 = lda.predict_proba(far1)
        assert p1[0] < 0.05, f"Far class 1 should have P<0.05, got {p1[0]}"

    def test_boundary_probability(self):
        mean0 = [0.0, 0.0]
        mean1 = [5.0, 5.0]
        X, y = make_gaussian_data(50, 2, mean0, mean1, std=0.5)

        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        w = lda.weights
        t = lda.threshold
        # Point on boundary: x such that w^T*x = t
        boundary = (w * t).reshape(1, -1)
        p = lda.predict_proba(boundary)
        assert abs(p[0] - 0.5) < 0.1, f"Boundary P should be ~0.5, got {p[0]}"

    def test_proba_shape(self):
        X, y = make_gaussian_data(50, 2, [0, 0], [3, 3])
        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        probs = lda.predict_proba(X)
        assert probs.shape == (100,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


class TestLdaDecisionFunction:
    def test_sign_matches_predict(self):
        X, y = make_gaussian_data(50, 2, [0, 0], [5, 5], std=0.5)
        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        preds = lda.predict(X)
        decisions = lda.decision_function(X)

        for i in range(len(preds)):
            if preds[i] == 0:
                assert decisions[i] >= 0.0
            else:
                assert decisions[i] < 0.0

    def test_magnitude_increases_with_distance(self):
        X, y = make_gaussian_data(50, 2, [0, 0], [5, 5], std=0.5)
        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        near = np.array([[2.5, 2.5]])
        far = np.array([[-5.0, -5.0]])

        d_near = abs(lda.decision_function(near)[0])
        d_far = abs(lda.decision_function(far)[0])
        assert d_far > d_near

    def test_output_shape(self):
        X, y = make_gaussian_data(50, 2, [0, 0], [3, 3])
        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        d = lda.decision_function(X)
        assert d.shape == (100,)


class TestLdaShrinkage:
    def test_regularization_helps(self):
        rng = np.random.default_rng(88)
        # Nearly collinear features
        x = rng.normal(0, 1, size=10)
        X0 = np.column_stack([x, x + rng.normal(0, 0.001, size=10)])
        x = rng.normal(3, 1, size=10)
        X1 = np.column_stack([x, x + rng.normal(0, 0.001, size=10)])
        X = np.vstack([X0, X1])
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

        lda = zbci.Lda(features=2, shrinkage=0.1)
        lda.fit(X, y)
        assert lda.is_fitted


class TestLdaWeights:
    def test_weight_direction(self):
        # Isotropic clusters -> w should be parallel to (m0 - m1)
        X, y = make_gaussian_data(200, 2, [0, 0], [3, 4], std=1.0)

        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        w = lda.weights
        # Expected direction: (m0-m1) normalized
        diff = np.array([-3.0, -4.0])
        expected = diff / np.linalg.norm(diff)

        cos_angle = abs(np.dot(w, expected))
        assert cos_angle > 0.9, f"cos(angle) = {cos_angle}, expected > 0.9"

    def test_unit_norm(self):
        X, y = make_gaussian_data(50, 4, [0, 0, 0, 0], [3, 3, 3, 3])
        lda = zbci.Lda(features=4)
        lda.fit(X, y)

        w = lda.weights
        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-10)

    def test_weights_before_fit(self):
        lda = zbci.Lda(features=2)
        with pytest.raises(ValueError, match="not been fitted"):
            _ = lda.weights


class TestLdaErrors:
    def test_insufficient_data(self):
        X = np.array([[0.0, 0.0]], dtype=np.float64)
        y = np.array([0], dtype=np.int64)
        lda = zbci.Lda(features=2)
        with pytest.raises(ValueError, match="Insufficient"):
            lda.fit(X, y)

    def test_not_fitted(self):
        lda = zbci.Lda(features=2)
        X = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="not been fitted"):
            lda.predict(X)
        with pytest.raises(ValueError, match="not been fitted"):
            lda.predict_proba(X)
        with pytest.raises(ValueError, match="not been fitted"):
            lda.decision_function(X)
        with pytest.raises(ValueError, match="not been fitted"):
            lda.score(X, np.array([0], dtype=np.int64))

    def test_feature_mismatch(self):
        X_train, y_train = make_gaussian_data(50, 2, [0, 0], [3, 3])
        lda = zbci.Lda(features=2)
        lda.fit(X_train, y_train)

        X_bad = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="features"):
            lda.predict(X_bad)


class TestLdaHigherDim:
    @pytest.mark.parametrize("f", [4, 8, 16])
    def test_higher_dimensions(self, f):
        mean0 = np.zeros(f)
        mean1 = np.ones(f) * 3.0
        X, y = make_gaussian_data(100, f, mean0, mean1, std=1.0)

        lda = zbci.Lda(features=f)
        lda.fit(X, y)

        accuracy = lda.score(X, y)
        assert accuracy > 0.85, f"features={f}: accuracy={accuracy}"


class TestLdaScore:
    def test_score_matches_manual(self):
        X, y = make_gaussian_data(50, 2, [0, 0], [5, 5], std=0.5)
        lda = zbci.Lda(features=2)
        lda.fit(X, y)

        score = lda.score(X, y)
        preds = lda.predict(X)
        manual_score = np.mean(preds == y)
        np.testing.assert_allclose(score, manual_score)
