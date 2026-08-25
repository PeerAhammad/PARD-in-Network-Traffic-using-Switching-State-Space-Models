"""
ukf.py
------
Unscented Kalman Filter (UKF) for nonlinear state-space models.

Uses the Unscented Transform (sigma points) instead of Jacobians.
More accurate than EKF for strongly nonlinear dynamics, and does not
require computing Jacobians manually.

Reference:
    Wan & Van der Merwe (2000). The Unscented Kalman Filter for
    Nonlinear Estimation. IEEE Adaptive Systems Workshop.
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class UKFResult:
    filtered_means: np.ndarray       # (T, state_dim)
    filtered_covs: np.ndarray        # (T, state_dim, state_dim)
    innovations: np.ndarray          # (T, obs_dim)
    log_likelihood: float


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter.

    Parameters
    ----------
    f      : state transition function
    h      : observation function
    Q      : process noise covariance   (state_dim, state_dim)
    R      : observation noise          (obs_dim, obs_dim)
    mu0    : initial state mean         (state_dim,)
    P0     : initial state covariance   (state_dim, state_dim)
    alpha  : sigma point spread (default 1e-3)
    beta   : prior knowledge of distribution (default 2 for Gaussian)
    kappa  : secondary scaling (default 0)
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        mu0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.mu0 = mu0.copy()
        self.P0 = P0.copy()

        self.state_dim = len(mu0)
        n = self.state_dim

        # Scaling parameters
        lam = alpha ** 2 * (n + kappa) - n
        self.lam = lam

        # Weights for mean and covariance
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)

    def _sigma_points(self, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute 2n+1 sigma points around mu using matrix square root.

        Returns
        -------
        sigmas : (2n+1, state_dim)
        """
        n = self.state_dim
        try:
            L = np.linalg.cholesky((n + self.lam) * P)
        except np.linalg.LinAlgError:
            # Add small jitter for numerical stability
            P_reg = P + np.eye(n) * 1e-8
            L = np.linalg.cholesky((n + self.lam) * P_reg)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mu
        for i in range(n):
            sigmas[i + 1] = mu + L[:, i]
            sigmas[n + i + 1] = mu - L[:, i]
        return sigmas

    def _unscented_transform(
        self,
        sigmas: np.ndarray,
        noise_cov: np.ndarray,
        fn: Callable,
    ):
        """
        Apply unscented transform through a function fn.

        Returns
        -------
        mu_out  : weighted mean of transformed sigmas
        P_out   : weighted covariance + noise_cov
        sigmas_f: transformed sigma points
        """
        sigmas_f = np.array([fn(s) for s in sigmas])
        mu_out = np.einsum("i,ij->j", self.Wm, sigmas_f)
        diff = sigmas_f - mu_out
        P_out = np.einsum("i,ij,ik->jk", self.Wc, diff, diff) + noise_cov
        return mu_out, P_out, sigmas_f

    def filter(self, observations: np.ndarray) -> UKFResult:
        """
        Run UKF on observation sequence.

        Parameters
        ----------
        observations : (T, obs_dim)

        Returns
        -------
        UKFResult
        """
        T, obs_dim = observations.shape
        d = self.state_dim

        filtered_means = np.zeros((T, d))
        filtered_covs = np.zeros((T, d, d))
        innovations = np.zeros((T, obs_dim))
        total_log_lik = 0.0

        mu = self.mu0.copy()
        P = self.P0.copy()

        for t in range(T):
            y_t = observations[t]

            # --- Sigma points ---
            sigmas = self._sigma_points(mu, P)

            # --- Predict ---
            mu_pred, P_pred, sigmas_f = self._unscented_transform(sigmas, self.Q, self.f)

            # --- Measurement prediction ---
            mu_y, P_y, sigmas_h = self._unscented_transform(sigmas_f, self.R, self.h)

            # Cross-covariance P_{xy}
            diff_x = sigmas_f - mu_pred
            diff_y = sigmas_h - mu_y
            P_xy = np.einsum("i,ij,ik->jk", self.Wc, diff_x, diff_y)

            # --- Kalman gain & update ---
            K = P_xy @ np.linalg.inv(P_y)
            innov = y_t - mu_y
            mu = mu_pred + K @ innov
            P = P_pred - K @ P_y @ K.T
            P = 0.5 * (P + P.T)

            filtered_means[t] = mu
            filtered_covs[t] = P
            innovations[t] = innov

            # Log-likelihood
            sign, logdet = np.linalg.slogdet(P_y)
            total_log_lik += -0.5 * (
                obs_dim * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(P_y, innov)
            )

        return UKFResult(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            innovations=innovations,
            log_likelihood=total_log_lik,
        )


# ---------------------------------------------------------------------------
# Factory: UKF for network traffic
# ---------------------------------------------------------------------------

def build_network_ukf(
    state_dim: int,
    obs_dim: int,
    alpha: float = 0.98,
) -> UnscentedKalmanFilter:
    """Build a UKF for network telemetry (nonlinear dynamics)."""
    rng = np.random.default_rng(42)
    C = rng.standard_normal((obs_dim, state_dim)) * 0.1

    def f(x):
        return alpha * x + (1 - alpha) * np.tanh(x)

    def h(x):
        return C @ x

    Q = np.eye(state_dim) * 0.1
    R = np.eye(obs_dim) * 0.2
    mu0 = np.zeros(state_dim)
    P0 = np.eye(state_dim)

    return UnscentedKalmanFilter(f=f, h=h, Q=Q, R=R, mu0=mu0, P0=P0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("UKF sanity check...")
    np.random.seed(42)

    T, state_dim, obs_dim = 200, 4, 6
    ukf = build_network_ukf(state_dim=state_dim, obs_dim=obs_dim)
    obs = np.random.randn(T, obs_dim) * 0.5
    result = ukf.filter(obs)

    print(f"Filtered means shape : {result.filtered_means.shape}")
    print(f"Log-likelihood       : {result.log_likelihood:.2f}")
    print("UKF OK!")
