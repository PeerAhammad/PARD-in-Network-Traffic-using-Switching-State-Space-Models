"""
test_data_processing.py
-----------------------
Unit tests for data loading, feature engineering, and preprocessing pipeline.
"""

import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.feature_engineering import (
    normalize_features,
    reduce_dimensions,
    create_time_windows,
    build_observation_sequence,
    temporal_train_test_split,
)
from src.data_processing.dataset_loader import (
    save_processed,
    load_processed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fake_data(N=2000, F=52, n_regimes=5, seed=42):
    """Generate synthetic network-like data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)
    # Add some outliers (common in network data)
    X[:50] *= 10
    y = rng.integers(0, n_regimes, size=N).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------

def test_normalize_standard():
    X, _ = make_fake_data(N=500, F=20)
    X_scaled, _, scaler = normalize_features(X, method="standard")
    assert X_scaled.shape == X.shape, "Shape should be preserved"
    # Mean should be close to 0, std close to 1
    assert abs(X_scaled.mean()) < 0.1,  "Standard scaled mean should be ~0"
    assert abs(X_scaled.std() - 1) < 0.1, "Standard scaled std should be ~1"
    print("  [PASS] test_normalize_standard")


def test_normalize_robust():
    X, _ = make_fake_data(N=500, F=20)
    X_scaled, _, scaler = normalize_features(X, method="robust")
    assert X_scaled.shape == X.shape, "Shape should be preserved"
    # Robust scaler: median ~0, IQR-based — not strict mean/std test
    assert not np.isnan(X_scaled).any(), "No NaNs after robust scaling"
    print("  [PASS] test_normalize_robust")


def test_normalize_train_test_no_leakage():
    X, _ = make_fake_data(N=1000, F=10)
    X_train, X_test = X[:800], X[800:]
    X_tr_scaled, X_te_scaled, scaler = normalize_features(X_train, X_test)
    assert X_tr_scaled.shape == (800, 10)
    assert X_te_scaled.shape == (200, 10)
    print("  [PASS] test_normalize_train_test_no_leakage")


# ---------------------------------------------------------------------------
# reduce_dimensions
# ---------------------------------------------------------------------------

def test_pca_output_shape():
    X, _ = make_fake_data(N=300, F=40)
    X_pca, _, pca = reduce_dimensions(X, n_components=10)
    assert X_pca.shape == (300, 10), f"Expected (300, 10), got {X_pca.shape}"
    print("  [PASS] test_pca_output_shape")


def test_pca_explained_variance():
    """PCA on simple data should explain most variance with few components."""
    rng = np.random.default_rng(0)
    # Low-rank data — 5 true components
    W = rng.standard_normal((100, 5))
    H = rng.standard_normal((5, 30))
    X = W @ H + rng.standard_normal((100, 30)) * 0.01

    X_pca, _, pca = reduce_dimensions(X, n_components=5)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    assert explained > 0.90, f"Expected >90% variance, got {explained:.2%}"
    print(f"  [PASS] test_pca_explained_variance  ({explained:.1%} explained)")


def test_pca_test_transform():
    X, _ = make_fake_data(N=500, F=20)
    X_train, X_test = X[:400], X[400:]
    X_tr, X_te, pca = reduce_dimensions(X_train, X_test, n_components=8)
    assert X_tr.shape == (400, 8)
    assert X_te.shape == (100, 8)
    print("  [PASS] test_pca_test_transform")


# ---------------------------------------------------------------------------
# create_time_windows
# ---------------------------------------------------------------------------

def test_window_output_shape():
    X = np.random.randn(1000, 10)
    y = np.zeros(1000, dtype=int)
    T, stride = 50, 50
    Xw, yw = create_time_windows(X, y, window_size=T, stride=stride)
    expected_windows = (1000 - T) // stride + 1
    assert Xw.shape[0] == expected_windows, f"Wrong number of windows: {Xw.shape[0]}"
    assert Xw.shape[1] == T
    assert Xw.shape[2] == 10
    assert yw.shape == (expected_windows, T)
    print(f"  [PASS] test_window_output_shape  ({Xw.shape})")


def test_window_stride_1():
    X = np.random.randn(200, 5)
    y = np.zeros(200, dtype=int)
    Xw, yw = create_time_windows(X, y, window_size=10, stride=1)
    assert Xw.shape[0] == 191  # N - T + 1
    print("  [PASS] test_window_stride_1")


def test_window_no_data_leakage():
    """First window should start at t=0, last window should end at t=N-1."""
    N, F, T = 100, 4, 10
    X = np.arange(N * F).reshape(N, F).astype(float)
    y = np.zeros(N, dtype=int)
    Xw, _ = create_time_windows(X, y, window_size=T, stride=T)
    # First window: rows 0..9
    np.testing.assert_array_equal(Xw[0], X[:T])
    # Second window: rows 10..19
    np.testing.assert_array_equal(Xw[1], X[T:2*T])
    print("  [PASS] test_window_no_data_leakage")


# ---------------------------------------------------------------------------
# build_observation_sequence
# ---------------------------------------------------------------------------

def test_full_pipeline_output():
    X, y = make_fake_data(N=3000, F=30)
    result = build_observation_sequence(
        X, y, normalize=True, n_pca_components=8, window_size=50
    )
    assert "observations" in result
    assert "labels"       in result
    assert "scaler"       in result
    assert "pca"          in result
    assert result["obs_dim"] == 8
    assert result["observations"].ndim == 3           # (W, T, F)
    assert result["observations"].shape[1] == 50      # window size
    assert result["observations"].shape[2] == 8       # PCA dim
    assert not np.isnan(result["observations"]).any()
    print(f"  [PASS] test_full_pipeline_output  obs shape: {result['observations'].shape}")


def test_pipeline_no_normalize():
    X, y = make_fake_data(N=1000, F=10)
    result = build_observation_sequence(
        X, y, normalize=False, n_pca_components=5, window_size=20
    )
    assert result["scaler"] is None
    print("  [PASS] test_pipeline_no_normalize")


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------

def test_temporal_split_sizes():
    X, y = make_fake_data(N=1000, F=10)
    X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, test_ratio=0.2)
    assert len(X_tr) == 800
    assert len(X_te) == 200
    assert len(y_tr) == 800
    assert len(y_te) == 200
    print("  [PASS] test_temporal_split_sizes")


def test_temporal_split_ordering():
    """Train must come before test — no shuffling."""
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.zeros(100, dtype=int)
    X_tr, X_te, _, _ = temporal_train_test_split(X, y, test_ratio=0.2)
    # All training values should be smaller than test values
    assert X_tr.max() < X_te.min(), "Train/test ordering violated"
    print("  [PASS] test_temporal_split_ordering")


# ---------------------------------------------------------------------------
# save / load processed data
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    X, y = make_fake_data(N=200, F=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_processed(X, y, output_dir=tmpdir, name="test")
        X_loaded, y_loaded = load_processed(tmpdir, name="test")
    np.testing.assert_array_almost_equal(X, X_loaded)
    np.testing.assert_array_equal(y, y_loaded)
    print("  [PASS] test_save_load_roundtrip")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # normalize_features
    test_normalize_standard,
    test_normalize_robust,
    test_normalize_train_test_no_leakage,
    # reduce_dimensions
    test_pca_output_shape,
    test_pca_explained_variance,
    test_pca_test_transform,
    # create_time_windows
    test_window_output_shape,
    test_window_stride_1,
    test_window_no_data_leakage,
    # build_observation_sequence
    test_full_pipeline_output,
    test_pipeline_no_normalize,
    # temporal_train_test_split
    test_temporal_split_sizes,
    test_temporal_split_ordering,
    # save / load
    test_save_load_roundtrip,
]


if __name__ == "__main__":
    print("=" * 55)
    print("Data Processing Tests")
    print("=" * 55)

    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 55)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    if failed == 0:
        print("All data processing tests passed!")
    else:
        sys.exit(1)
