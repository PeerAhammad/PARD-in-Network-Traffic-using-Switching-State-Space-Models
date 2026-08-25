"""
feature_engineering.py
-----------------------
Feature extraction, normalization, and windowing for network telemetry.

Converts raw feature matrices into sequences suitable for state-space models.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = "robust",
) -> Tuple[np.ndarray, Optional[np.ndarray], object]:
    """
    Normalize feature matrix.

    Parameters
    ----------
    X_train : np.ndarray  (N, F)
    X_test  : np.ndarray  (M, F) — optional held-out set
    method  : 'standard' | 'robust'
              RobustScaler is preferred for network data (many outliers)

    Returns
    -------
    X_train_scaled, X_test_scaled (or None), scaler
    """
    if method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Dimensionality Reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_components: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray], PCA]:
    """
    Apply PCA for dimensionality reduction.

    Network telemetry often has 40-80 features; reducing to 10-20 principal
    components speeds up Kalman filtering significantly.

    Parameters
    ----------
    X_train     : (N, F)
    X_test      : (M, F) optional
    n_components: target dimension

    Returns
    -------
    X_train_pca, X_test_pca (or None), pca object
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) if X_test is not None else None

    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA: {n_components} components explain {explained:.1%} of variance")

    return X_train_pca, X_test_pca, pca


# ---------------------------------------------------------------------------
# Temporal Windowing
# ---------------------------------------------------------------------------

def create_time_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 20,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the feature sequence to create temporal sequences.

    For state-space models we process data as time series. Each window
    is a sequence of T observations.

    Parameters
    ----------
    X           : (N, F) — feature matrix (time-ordered)
    y           : (N,)   — regime labels
    window_size : T      — number of time steps per window
    stride      : step size between windows

    Returns
    -------
    X_windows : (W, T, F)
    y_windows : (W, T)     — label at each time step in window
    """
    N, F = X.shape
    windows_X = []
    windows_y = []

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        windows_X.append(X[start:end])
        windows_y.append(y[start:end])

    X_windows = np.stack(windows_X)  # (W, T, F)
    y_windows = np.stack(windows_y)  # (W, T)

    print(f"Created {len(X_windows)} windows of shape (T={window_size}, F={F})")
    return X_windows, y_windows


# ---------------------------------------------------------------------------
# Observation sequence builder (for KF input)
# ---------------------------------------------------------------------------

def build_observation_sequence(
    X: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
    n_pca_components: int = 10,
    window_size: int = 50,
) -> dict:
    """
    Full pipeline: normalize → PCA → window → ready for SSM inference.

    Parameters
    ----------
    X                 : raw feature matrix (N, F)
    y                 : regime labels (N,)
    normalize         : apply RobustScaler
    n_pca_components  : PCA output dimension (observation dim for KF)
    window_size       : length of each time window

    Returns
    -------
    dict with keys:
        'observations'  : (W, T, obs_dim) — input to Kalman filter
        'labels'        : (W, T)           — ground truth regimes
        'scaler'        : fitted scaler
        'pca'           : fitted PCA
        'obs_dim'       : observation dimension
    """
    # 1. Normalize
    if normalize:
        X_scaled, _, scaler = normalize_features(X, method="robust")
    else:
        X_scaled, scaler = X, None

    # 2. PCA
    X_pca, _, pca = reduce_dimensions(X_scaled, n_components=n_pca_components)

    # 3. Window
    X_windows, y_windows = create_time_windows(X_pca, y, window_size=window_size, stride=window_size)

    return {
        "observations": X_windows.astype(np.float64),
        "labels": y_windows,
        "scaler": scaler,
        "pca": pca,
        "obs_dim": n_pca_components,
    }


# ---------------------------------------------------------------------------
# Train/test split for temporal data
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time-ordered dataset into train and test without shuffling.
    Shuffling would leak future information into the past for time series.
    """
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Feature engineering sanity check...")

    # Simulate network telemetry
    np.random.seed(42)
    N, F = 5000, 52
    X_fake = np.random.randn(N, F).astype(np.float32)
    y_fake = np.random.randint(0, 5, size=N).astype(np.int32)

    result = build_observation_sequence(
        X_fake, y_fake,
        normalize=True,
        n_pca_components=10,
        window_size=50,
    )

    print(f"observations shape : {result['observations'].shape}")
    print(f"labels shape       : {result['labels'].shape}")
    print(f"obs_dim            : {result['obs_dim']}")
    print("All good!")
