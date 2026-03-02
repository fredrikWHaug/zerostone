"""Tests for ICA (Independent Component Analysis) Python bindings."""

import numpy as np
import pytest

import zpybci as zbci


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ch", [4, 8, 16, 32, 64])
def test_create_all_channel_counts(ch):
    ica = zbci.Ica(channels=ch)
    assert ica.channels == ch
    assert not ica.is_fitted


def test_invalid_channels():
    with pytest.raises(ValueError):
        zbci.Ica(channels=3)
    with pytest.raises(ValueError):
        zbci.Ica(channels=5)
    with pytest.raises(ValueError):
        zbci.Ica(channels=128)


def test_invalid_contrast():
    with pytest.raises(ValueError):
        zbci.Ica(channels=4, contrast="invalid")


def test_all_contrast_functions():
    for cf in ("logcosh", "exp", "cube"):
        ica = zbci.Ica(channels=4, contrast=cf)
        assert not ica.is_fitted


def test_repr():
    ica = zbci.Ica(channels=4, contrast="exp")
    r = repr(ica)
    assert "channels=4" in r
    assert "exp" in r
    assert "fitted=false" in r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cocktail_party(n_sources, n_samples=2000, seed=42):
    """Generate a cocktail party problem with known sources and mixing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_samples, endpoint=False)

    sources = []
    # Sine
    sources.append(np.sin(2 * np.pi * 5 * t))
    # Square wave
    sources.append(np.sign(np.sin(2 * np.pi * 11 * t)))
    # Sawtooth
    if n_sources >= 3:
        sources.append(2 * (7 * t % 1) - 1)
    # Another sine at different freq
    if n_sources >= 4:
        sources.append(np.sin(2 * np.pi * 23 * t))

    S = np.column_stack(sources[:n_sources])
    A = rng.standard_normal((n_sources, n_sources))
    # Make A well-conditioned
    A = A + np.eye(n_sources) * 2
    X = S @ A.T  # (samples, channels)
    return S, A, X


def abs_correlation(a, b):
    """Absolute Pearson correlation."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-15:
        return 0.0
    return abs((a * b).sum() / denom)


def best_match_corr(S_true, S_est):
    """Min of best-match absolute correlations."""
    n_comp = S_true.shape[1]
    min_corr = 1.0
    for i in range(n_comp):
        best = max(abs_correlation(S_true[:, i], S_est[:, j]) for j in range(S_est.shape[1]))
        min_corr = min(min_corr, best)
    return min_corr


# ---------------------------------------------------------------------------
# Fit / transform / inverse_transform
# ---------------------------------------------------------------------------


def test_fit_transform_round_trip():
    _, _, X = make_cocktail_party(4, n_samples=2000)
    ica = zbci.Ica(channels=4)
    ica.fit(X)
    assert ica.is_fitted

    sources = ica.transform(X)
    assert sources.shape == X.shape

    reconstructed = ica.inverse_transform(sources)
    assert reconstructed.shape == X.shape
    np.testing.assert_allclose(reconstructed, X, atol=1e-8)


def test_cocktail_party_separation():
    S, _, X = make_cocktail_party(4, n_samples=5000)
    ica = zbci.Ica(channels=4)
    ica.fit(X)
    sources = ica.transform(X)

    corr = best_match_corr(S, sources)
    assert corr > 0.8, f"Cocktail party separation: min correlation = {corr}"


def test_remove_components_changes_output():
    _, _, X = make_cocktail_party(4, n_samples=2000)
    ica = zbci.Ica(channels=4)
    ica.fit(X)

    cleaned = ica.remove_components(X, [0])
    assert cleaned.shape == X.shape
    # Cleaned should differ from original
    diff = np.abs(cleaned - X).sum()
    assert diff > 1.0, f"Removing component should change output, diff={diff}"


# ---------------------------------------------------------------------------
# Matrix properties
# ---------------------------------------------------------------------------


def test_mixing_unmixing_shapes():
    _, _, X = make_cocktail_party(4, n_samples=2000)
    ica = zbci.Ica(channels=4)
    ica.fit(X)

    A = ica.mixing_matrix
    W = ica.unmixing_matrix
    assert A.shape == (4, 4)
    assert W.shape == (4, 4)


def test_unmixing_times_mixing_approx_identity():
    _, _, X = make_cocktail_party(4, n_samples=2000)
    ica = zbci.Ica(channels=4)
    ica.fit(X)

    W = ica.unmixing_matrix
    A = ica.mixing_matrix
    product = W @ A
    np.testing.assert_allclose(product, np.eye(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_transform_before_fit():
    ica = zbci.Ica(channels=4)
    X = np.random.randn(100, 4)
    with pytest.raises(ValueError, match="not been fitted"):
        ica.transform(X)


def test_wrong_channel_count():
    ica = zbci.Ica(channels=4)
    X = np.random.randn(100, 8)
    with pytest.raises(ValueError, match="channels"):
        ica.fit(X)


def test_insufficient_data():
    ica = zbci.Ica(channels=4)
    X = np.random.randn(5, 4)  # Need at least 8 (2*4)
    with pytest.raises(ValueError, match="Insufficient"):
        ica.fit(X)


# ---------------------------------------------------------------------------
# All contrast functions separate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contrast", ["logcosh", "exp", "cube"])
def test_all_contrasts_separate(contrast):
    S, _, X = make_cocktail_party(4, n_samples=5000, seed=123)
    ica = zbci.Ica(channels=4, contrast=contrast)
    ica.fit(X)
    sources = ica.transform(X)
    corr = best_match_corr(S, sources)
    assert corr > 0.75, f"contrast={contrast}: min correlation = {corr}"


# ---------------------------------------------------------------------------
# Blink artifact removal (synthetic)
# ---------------------------------------------------------------------------


def test_blink_artifact_removal():
    """Synthetic blink artifact test: verify MSE improves after removal."""
    rng = np.random.default_rng(42)
    n = 3000
    t = np.linspace(0, 3, n, endpoint=False)

    # Clean EEG: 4 channels of summed sinusoids
    clean = np.column_stack(
        [
            np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 22 * t),
            0.8 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 15 * t),
            np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 20 * t),
            0.6 * np.sin(2 * np.pi * 12 * t) + np.sin(2 * np.pi * 18 * t),
        ]
    )

    # Blink artifact: Gaussian pulses concentrated in frontal channels
    blink_times = [500, 1200, 2000, 2700]
    blink_signal = np.zeros(n)
    for bt in blink_times:
        blink_signal += 5.0 * np.exp(-0.5 * ((np.arange(n) - bt) / 30) ** 2)

    # Blink affects channels differently (frontal > posterior)
    blink_weights = np.array([1.0, 0.8, 0.2, 0.1])
    artifact = np.outer(blink_signal, blink_weights)

    contaminated = clean + artifact

    ica = zbci.Ica(channels=4)
    ica.fit(contaminated)
    sources = ica.transform(contaminated)

    # Find the component most correlated with blink
    blink_corrs = [abs_correlation(blink_signal, sources[:, i]) for i in range(4)]
    blink_component = int(np.argmax(blink_corrs))

    # Remove that component
    cleaned = ica.remove_components(contaminated, [blink_component])

    # MSE should improve
    mse_before = np.mean((contaminated - clean) ** 2)
    mse_after = np.mean((cleaned - clean) ** 2)
    assert mse_after < mse_before, (
        f"MSE should improve: before={mse_before:.4f}, after={mse_after:.4f}"
    )
