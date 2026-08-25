"""
ekf.py
------
Extended Kalman Filter (EKF) for nonlinear state-space models.

Handles cases where the state transition or observation function is nonlinear
by linearizing around the current state estimate using the Jacobian.

Nonlinear model:
    x_t = f(x_{t-1}) + w_t,    w_t ~ N(0, Q)
    y_t = h(x_t) + v_t,        v_t ~ N(0, R)

For network traffic: nonlinearity arises from log-transforms, ratio features,
and exponential decay in flow statistics.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EKFResult:
    filtered_means: np.ndarray       # (T, state_dim)
    filtered_covs: np.ndarray        # (T, state_dim, state_dim)
    innovations: np.ndarray          # (T, obs_dim)
    log_likelihood: float


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with user-supplied nonlinear functions
    and their Jacobians.

    Parameters
    ----------
    f      : state transition function  x_{t} = f(x_{t-1})
    h      : observation function       y_t = h(x_t)
    F_jac  : Jacobian of f at x        df/dx  (state_dim, state_dim)
    H_jac  : Jacobian of h at x        dh/dx  (obs_dim, state_dim)
    Q      : process noise covariance
    R      : observation noise covariance
    mu0    : initial state mean
    P0     : initial state covariance
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        F_jac: Callable,
        H_jac: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        mu0: np.ndarray,
        P0: np.ndarray,
    ):
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.Q = Q
        self.R = R
        self.mu0 = mu0.copy()
        self.P0 = P0.copy()
        self.state_dim = len(mu0)

    def filter(self, observations: np.ndarray) -> EKFResult:
        """
        Run EKF on observation sequence.

        Parameters
        ----------
        observations : (T, obs_dim)

        Returns
        -------
        EKFResult
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

            # --- Predict ---
            F = self.F_jac(mu)         # Jacobian of f at mu
            mu_pred = self.f(mu)
            P_pred = F @ P @ F.T + self.Q

            # --- Update ---
            H = self.H_jac(mu_pred)    # Jacobian of h at mu_pred
            innov = y_t - self.h(mu_pred)
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)

            mu = mu_pred + K @ innov
            P = (np.eye(d) - K @ H) @ P_pred
            P = 0.5 * (P + P.T)

            filtered_means[t] = mu
            filtered_covs[t] = P
            innovations[t] = innov

            # Log-likelihood
            sign, logdet = np.linalg.slogdet(S)
            total_log_lik += -0.5 * (
                obs_dim * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov)
            )

        return EKFResult(
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            innovations=innovations,
            log_likelihood=total_log_lik,
        )


# ---------------------------------------------------------------------------
# Factory: build EKF for network traffic with linear+nonlinear mix
# ---------------------------------------------------------------------------

def build_network_ekf(
    state_dim: int,
    obs_dim: int,
    alpha: float = 0.98,
    noise_scale: float = 0.1,
) -> ExtendedKalmanFilter:
    """
    Build an EKF suitable for network telemetry.

    State dynamics: soft exponential decay (models traffic bursts)
        f(x) = alpha * x + (1-alpha) * tanh(x)

    Observation: linear projection (for simplicity, extended from KF)
        h(x) = C * x

    Parameters
    ----------
    state_dim   : hidden dimension
    obs_dim     : observation dimension
    alpha       : smoothing coefficient in state dynamics
    noise_scale : Q, R noise scale
    """
    rng = np.random.default_rng(42)
    C = rng.standard_normal((obs_dim, state_dim)) * 0.1

    def f(x):
        return alpha * x + (1 - alpha) * np.tanh(x)

    def F_jac(x):
        # d/dx [alpha*x + (1-alpha)*tanh(x)] = alpha*I + (1-alpha)*diag(sech^2(x))
        sech2 = 1.0 / np.cosh(x) ** 2
        return alpha * np.eye(state_dim) + (1 - alpha) * np.diag(sech2)

    def h(x):
        return C @ x

    def H_jac(x):
        return C

    Q = np.eye(state_dim) * noise_scale
    R = np.eye(obs_dim) * noise_scale
    mu0 = np.zeros(state_dim)
    P0 = np.eye(state_dim)

    return ExtendedKalmanFilter(f=f, h=h, F_jac=F_jac, H_jac=H_jac,
                                 Q=Q, R=R, mu0=mu0, P0=P0)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("EKF sanity check...")
    np.random.seed(42)

    T, state_dim, obs_dim = 200, 4, 6
    ekf = build_network_ekf(state_dim=state_dim, obs_dim=obs_dim)

    # Fake observations
    obs = np.random.randn(T, obs_dim) * 0.5

    result = ekf.filter(obs)
    print(f"Filtered means shape : {result.filtered_means.shape}")
    print(f"Log-likelihood       : {result.log_likelihood:.2f}")
    print("EKF OK!")
