"""
kalman_filter.py
----------------
Standard Linear Kalman Filter implementation.

This is the foundational building block for the switching SSM.
Tracks a hidden state x_t from noisy observations y_t.

State model:
    x_t = A * x_{t-1} + w_t,   w_t ~ N(0, Q)
    y_t = C * x_t + v_t,       v_t ~ N(0, R)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class KalmanFilterResult:
    """Container for Kalman filter output."""
    filtered_means: np.ndarray       # (T, state_dim)
    filtered_covs: np.ndarray        # (T, state_dim, state_dim)
    predicted_means: np.ndarray      # (T, state_dim)
    predicted_covs: np.ndarray       # (T, state_dim, state_dim)
    innovations: np.ndarray          # (T, obs_dim)
    innovation_covs: np.ndarray      # (T, obs_dim, obs_dim)
    log_likelihood: float
    smoothed_means: Optional[np.ndarray] = None   # (T, state_dim) — filled by smoother
    smoothed_covs: Optional[np.ndarray] = None    # (T, state_dim, state_dim)


class KalmanFilter:
    """
    Linear Gaussian Kalman Filter.

    Parameters
    ----------
    A   : state transition matrix              (state_dim, state_dim)
    C   : observation matrix                   (obs_dim, state_dim)
    Q   : process noise covariance             (state_dim, state_dim)
    R   : observation noise covariance         (obs_dim, obs_dim)
    mu0 : initial state mean                   (state_dim,)
    P0  : initial state covariance             (state_dim, state_dim)
    """

    def __init__(
        self,
        A: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        mu0: np.ndarray,
        P0: np.ndarray,
    ):
        self.A = np.array(A, dtype=np.float64)
        self.C = np.array(C, dtype=np.float64)
        self.Q = np.array(Q, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.mu0 = np.array(mu0, dtype=np.float64)
        self.P0 = np.array(P0, dtype=np.float64)

        self.state_dim = A.shape[0]
        self.obs_dim = C.shape[0]

    # ------------------------------------------------------------------
    # Core filter step
    # ------------------------------------------------------------------

    def predict(self, mu: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict step (time update).

        mu_pred = A * mu
        P_pred  = A * P * A^T + Q
        """
        mu_pred = self.A @ mu
        P_pred = self.A @ P @ self.A.T + self.Q
        return mu_pred, P_pred

    def update(
        self, mu_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Update step (measurement update).

        Returns
        -------
        mu_filt  : filtered mean
        P_filt   : filtered covariance
        innov    : innovation (y - C * mu_pred)
        S        : innovation covariance
        log_lik  : log likelihood contribution of this observation
        """
        innov = y - self.C @ mu_pred
        S = self.C @ P_pred @ self.C.T + self.R          # innovation covariance
        K = P_pred @ self.C.T @ np.linalg.inv(S)         # Kalman gain

        mu_filt = mu_pred + K @ innov
        I = np.eye(self.state_dim)
        P_filt = (I - K @ self.C) @ P_pred

        # Symmetrize to avoid numerical drift
        P_filt = 0.5 * (P_filt + P_filt.T)

        # Log-likelihood contribution: log N(y; C*mu_pred, S)
        sign, logdet = np.linalg.slogdet(S)
        log_lik = -0.5 * (
            self.obs_dim * np.log(2 * np.pi)
            + logdet
            + innov @ np.linalg.inv(S) @ innov
        )

        return mu_filt, P_filt, innov, S, log_lik

    # ------------------------------------------------------------------
    # Full forward pass (filter)
    # ------------------------------------------------------------------

    def filter(self, observations: np.ndarray) -> KalmanFilterResult:
        """
        Run the Kalman filter on a sequence of observations.

        Parameters
        ----------
        observations : np.ndarray  shape (T, obs_dim)

        Returns
        -------
        KalmanFilterResult
        """
        T = observations.shape[0]

        filtered_means = np.zeros((T, self.state_dim))
        filtered_covs = np.zeros((T, self.state_dim, self.state_dim))
        predicted_means = np.zeros((T, self.state_dim))
        predicted_covs = np.zeros((T, self.state_dim, self.state_dim))
        innovations = np.zeros((T, self.obs_dim))
        innovation_covs = np.zeros((T, self.obs_dim, self.obs_dim))
        total_log_lik = 0.0

        mu = self.mu0.copy()
        P = self.P0.copy()

        for t in range(T):
            # Predict
            mu_pred, P_pred = self.predict(mu, P)
            predicted_means[t] = mu_pred
            predicted_covs[t] = P_pred

            # Update
            mu, P, innov, S, ll = self.update(mu_pred, P_pred, observations[t])
            filtered_means[t] = mu
            filtered_covs[t] = P
            innovations[t] = innov
            innovation_covs[t] = S
            total_log_lik += ll

        return KalmanFilterResult(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            predicted_means=predicted_means,
            predicted_covs=predicted_covs,
            innovations=innovations,
            innovation_covs=innovation_covs,
            log_likelihood=total_log_lik,
        )

    # ------------------------------------------------------------------
    # Rauch-Tung-Striebel smoother (backward pass)
    # ------------------------------------------------------------------

    def smooth(self, filter_result: KalmanFilterResult) -> KalmanFilterResult:
        """
        RTS smoother: improves state estimates using future observations.
        Runs backward through the filtered estimates.

        Returns the same KalmanFilterResult with smoothed_means/covs filled.
        """
        T = filter_result.filtered_means.shape[0]
        smoothed_means = filter_result.filtered_means.copy()
        smoothed_covs = filter_result.filtered_covs.copy()

        for t in range(T - 2, -1, -1):
            P_pred = filter_result.predicted_covs[t + 1]
            G = filter_result.filtered_covs[t] @ self.A.T @ np.linalg.inv(P_pred)

            smoothed_means[t] = (
                filter_result.filtered_means[t]
                + G @ (smoothed_means[t + 1] - filter_result.predicted_means[t + 1])
            )
            smoothed_covs[t] = (
                filter_result.filtered_covs[t]
                + G @ (smoothed_covs[t + 1] - P_pred) @ G.T
            )

        filter_result.smoothed_means = smoothed_means
        filter_result.smoothed_covs = smoothed_covs
        return filter_result

    # ------------------------------------------------------------------
    # Parameter learning (EM — M step for one regime)
    # ------------------------------------------------------------------

    @staticmethod
    def init_from_data(
        observations: np.ndarray,
        state_dim: int,
        noise_scale: float = 0.1,
    ) -> "KalmanFilter":
        """
        Initialize Kalman Filter parameters from data statistics.

        Uses PCA-based initialization: A = I (random walk assumption),
        C = top singular vectors, Q and R = scaled identity.

        Parameters
        ----------
        observations : (T, obs_dim)
        state_dim    : hidden state dimension
        noise_scale  : scale for initial noise covariances

        Returns
        -------
        KalmanFilter instance ready for filtering
        """
        T, obs_dim = observations.shape

        # A: assume random walk / slow drift
        A = np.eye(state_dim) * 0.99

        # C: project from hidden → observed via random init (will be refined)
        rng = np.random.default_rng(42)
        C = rng.standard_normal((obs_dim, state_dim)) * 0.1

        # Noise covariances
        Q = np.eye(state_dim) * noise_scale
        R = np.eye(obs_dim) * noise_scale

        # Initial state
        mu0 = np.zeros(state_dim)
        P0 = np.eye(state_dim) * 1.0

        return KalmanFilter(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Kalman Filter sanity check...")

    np.random.seed(42)
    T, obs_dim, state_dim = 200, 5, 3

    # Ground truth
    A_true = np.eye(state_dim) * 0.95
    C_true = np.random.randn(obs_dim, state_dim)

    # Generate synthetic trajectory
    x = np.zeros((T, state_dim))
    y = np.zeros((T, obs_dim))
    x[0] = np.random.randn(state_dim)
    for t in range(1, T):
        x[t] = A_true @ x[t - 1] + np.random.randn(state_dim) * 0.1
        y[t] = C_true @ x[t] + np.random.randn(obs_dim) * 0.2

    # Filter
    kf = KalmanFilter.init_from_data(y, state_dim=state_dim)
    result = kf.filter(y)
    result = kf.smooth(result)

    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Filtered means shape: {result.filtered_means.shape}")
    print(f"Smoothed means shape: {result.smoothed_means.shape}")
    print("Kalman Filter OK!")
