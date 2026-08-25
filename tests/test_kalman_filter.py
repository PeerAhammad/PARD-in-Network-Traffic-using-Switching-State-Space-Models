# tests/test_kalman_filter.py
"""Unit tests for the Kalman Filter."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.kalman_filter import KalmanFilter


def test_kalman_filter_runs():
    """Filter should run without errors and return correct shapes."""
    T, obs_dim, state_dim = 100, 5, 3
    np.random.seed(0)
    obs = np.random.randn(T, obs_dim)
    kf = KalmanFilter.init_from_data(obs, state_dim=state_dim)
    result = kf.filter(obs)

    assert result.filtered_means.shape == (T, state_dim)
    assert result.filtered_covs.shape == (T, state_dim, state_dim)
    assert result.log_likelihood < 0   # log-likelihood is negative


def test_kalman_smoother_runs():
    """RTS smoother should produce smoothed estimates."""
    T, obs_dim, state_dim = 50, 4, 2
    np.random.seed(1)
    obs = np.random.randn(T, obs_dim)
    kf = KalmanFilter.init_from_data(obs, state_dim=state_dim)
    result = kf.filter(obs)
    result = kf.smooth(result)

    assert result.smoothed_means is not None
    assert result.smoothed_means.shape == (T, state_dim)


def test_kalman_covariance_positive_definite():
    """Filtered covariance should be positive definite at all steps."""
    T, obs_dim, state_dim = 80, 4, 3
    np.random.seed(2)
    obs = np.random.randn(T, obs_dim)
    kf = KalmanFilter.init_from_data(obs, state_dim=state_dim)
    result = kf.filter(obs)

    for t in range(T):
        eigvals = np.linalg.eigvalsh(result.filtered_covs[t])
        assert np.all(eigvals > -1e-8), f"Non-PD covariance at t={t}"


if __name__ == "__main__":
    test_kalman_filter_runs()
    test_kalman_smoother_runs()
    test_kalman_covariance_positive_definite()
    print("All Kalman Filter tests passed!")
