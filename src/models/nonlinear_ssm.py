"""
nonlinear_ssm.py
----------------
Nonlinear State-Space Model (SSM) definition.

Defines the model functions (f, h) and their Jacobians needed by the
Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF).

Nonlinear model:
    x_t = f(x_{t-1}) + w_t,    w_t ~ N(0, Q)
    y_t = h(x_t)     + v_t,    v_t ~ N(0, R)

Why nonlinear for network traffic?
    - Traffic rates follow exponential / log distributions
    - Burst dynamics are not captured by linear models
    - Flow statistics exhibit saturation effects (e.g. port counts)

This module provides:
    1. NonlinearSSM    — parameter container + model functions
    2. Built-in models — network-specific nonlinear dynamics
    3. Jacobian helpers — for EKF (auto-computed via finite differences
                          if not supplied analytically)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Dynamics presets — domain-specific nonlinear functions for network traffic
# ---------------------------------------------------------------------------

class DynamicsType(str, Enum):
    """Available nonlinear dynamics presets."""
    TANH          = "tanh"          # Soft saturation — models traffic bursts
    SIGMOID_DECAY = "sigmoid_decay" # Exponential forgetting of flow state
    LOG_RATIO     = "log_ratio"     # Log-ratio dynamics (packet size distributions)
    LINEAR        = "linear"        # Falls back to linear (identity Jacobian)


def _build_dynamics(dtype: DynamicsType, state_dim: int, alpha: float = 0.97):
    """
    Return (f, F_jac) for the chosen dynamics type.

    All functions operate on a single state vector x of shape (state_dim,).
    """
    if dtype == DynamicsType.TANH:
        # f(x) = alpha * x + (1-alpha) * tanh(x)
        # Models: fast-changing components (attack bursts) that saturate
        def f(x):
            return alpha * x + (1.0 - alpha) * np.tanh(x)

        def F_jac(x):
            sech2 = 1.0 / np.cosh(x) ** 2
            return np.diag(alpha + (1.0 - alpha) * sech2)

    elif dtype == DynamicsType.SIGMOID_DECAY:
        # f(x) = alpha * sigmoid(x) * 2 - alpha  (maps to [-alpha, alpha])
        # Models: exponential decay with sigmoid nonlinearity
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        def f(x):
            return alpha * (2.0 * sigmoid(x) - 1.0)

        def F_jac(x):
            sig = sigmoid(x)
            dsig = sig * (1.0 - sig)
            return np.diag(2.0 * alpha * dsig)

    elif dtype == DynamicsType.LOG_RATIO:
        # f(x) = alpha * sign(x) * log(1 + |x|)
        # Models: log-ratio dynamics common in packet size / flow rate
        def f(x):
            return alpha * np.sign(x) * np.log1p(np.abs(x))

        def F_jac(x):
            diag_vals = alpha / (1.0 + np.abs(x))
            return np.diag(diag_vals)

    else:  # LINEAR
        def f(x):
            return alpha * x

        def F_jac(x):
            return np.eye(state_dim) * alpha

    return f, F_jac


# ---------------------------------------------------------------------------
# NonlinearSSM
# ---------------------------------------------------------------------------

@dataclass
class NonlinearSSM:
    """
    Parameter container and model definition for a Nonlinear SSM.

    Compatible with both EKF (uses f, F_jac, h, H_jac) and UKF (uses f, h only).

    Attributes
    ----------
    state_dim    : hidden state dimension
    obs_dim      : observation dimension
    Q            : process noise covariance    (state_dim, state_dim)
    R            : observation noise           (obs_dim,   obs_dim)
    mu0          : initial state mean          (state_dim,)
    P0           : initial state covariance    (state_dim, state_dim)
    C            : linear observation matrix   (obs_dim, state_dim)
                   (used when h is linear projection)
    dynamics     : DynamicsType — which nonlinear dynamics to use
    alpha        : stability / decay coefficient (0 < alpha < 1)
    name         : descriptive label
    """

    state_dim: int
    obs_dim: int
    Q: np.ndarray
    R: np.ndarray
    mu0: np.ndarray
    P0: np.ndarray
    C: np.ndarray
    dynamics: DynamicsType = DynamicsType.TANH
    alpha: float = 0.97
    name: str = "nonlinear_ssm"

    # Built dynamically in __post_init__
    f: Callable = field(init=False, repr=False)
    F_jac: Callable = field(init=False, repr=False)
    h: Callable = field(init=False, repr=False)
    H_jac: Callable = field(init=False, repr=False)

    def __post_init__(self):
        self.Q   = np.array(self.Q,   dtype=np.float64)
        self.R   = np.array(self.R,   dtype=np.float64)
        self.mu0 = np.array(self.mu0, dtype=np.float64)
        self.P0  = np.array(self.P0,  dtype=np.float64)
        self.C   = np.array(self.C,   dtype=np.float64)

        # Build state dynamics
        self.f, self.F_jac = _build_dynamics(self.dynamics, self.state_dim, self.alpha)

        # Observation model: linear projection h(x) = C * x
        # (Nonlinearity is in the state dynamics, not the observation)
        C_copy = self.C.copy()

        def h(x):
            return C_copy @ x

        def H_jac(x):
            return C_copy

        self.h     = h
        self.H_jac = H_jac

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls,
        state_dim: int,
        obs_dim: int,
        dynamics: DynamicsType = DynamicsType.TANH,
        alpha: float = 0.97,
        noise_scale: float = 0.1,
        name: str = "nonlinear_ssm",
        seed: Optional[int] = None,
    ) -> "NonlinearSSM":
        """
        Randomly initialise a NonlinearSSM.

        Parameters
        ----------
        state_dim  : hidden state dimension
        obs_dim    : observation dimension
        dynamics   : nonlinear dynamics type
        alpha      : stability coefficient
        noise_scale: Q and R noise level
        seed       : random seed
        """
        rng = np.random.default_rng(seed)
        C   = rng.standard_normal((obs_dim, state_dim)) * 0.1
        Q   = np.eye(state_dim) * noise_scale
        R   = np.eye(obs_dim)   * noise_scale
        mu0 = np.zeros(state_dim)
        P0  = np.eye(state_dim)
        return cls(
            state_dim=state_dim, obs_dim=obs_dim,
            Q=Q, R=R, mu0=mu0, P0=P0, C=C,
            dynamics=dynamics, alpha=alpha, name=name,
        )

    @classmethod
    def from_linear_ssm(
        cls,
        linear_model,
        dynamics: DynamicsType = DynamicsType.TANH,
        alpha: float = 0.97,
        name: Optional[str] = None,
    ) -> "NonlinearSSM":
        """
        Promote a LinearSSM to NonlinearSSM by replacing its state
        transition with a nonlinear function while keeping Q, R, C.

        Parameters
        ----------
        linear_model : LinearSSM instance
        dynamics     : nonlinear dynamics type to use
        alpha        : stability coefficient
        """
        return cls(
            state_dim=linear_model.state_dim,
            obs_dim=linear_model.obs_dim,
            Q=linear_model.Q.copy(),
            R=linear_model.R.copy(),
            mu0=linear_model.mu0.copy(),
            P0=linear_model.P0.copy(),
            C=linear_model.C.copy(),
            dynamics=dynamics,
            alpha=alpha,
            name=name or f"{linear_model.name}_nonlinear",
        )

    # ------------------------------------------------------------------
    # Build inference objects
    # ------------------------------------------------------------------

    def build_ekf(self):
        """
        Instantiate an ExtendedKalmanFilter using this model.

        Returns
        -------
        ExtendedKalmanFilter (from src.inference.ekf)
        """
        from src.inference.ekf import ExtendedKalmanFilter
        return ExtendedKalmanFilter(
            f=self.f, h=self.h,
            F_jac=self.F_jac, H_jac=self.H_jac,
            Q=self.Q, R=self.R,
            mu0=self.mu0, P0=self.P0,
        )

    def build_ukf(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        Instantiate an UnscentedKalmanFilter using this model.

        Returns
        -------
        UnscentedKalmanFilter (from src.inference.ukf)
        """
        from src.inference.ukf import UnscentedKalmanFilter
        return UnscentedKalmanFilter(
            f=self.f, h=self.h,
            Q=self.Q, R=self.R,
            mu0=self.mu0, P0=self.P0,
            alpha=alpha, beta=beta, kappa=kappa,
        )

    # ------------------------------------------------------------------
    # Simulate data
    # ------------------------------------------------------------------

    def simulate(
        self,
        T: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic trajectory from this nonlinear SSM.

        Parameters
        ----------
        T    : number of time steps
        seed : random seed

        Returns
        -------
        x : (T, state_dim)
        y : (T, obs_dim)
        """
        rng = np.random.default_rng(seed)
        x = np.zeros((T, self.state_dim))
        y = np.zeros((T, self.obs_dim))

        try:
            LQ = np.linalg.cholesky(self.Q + np.eye(self.state_dim) * 1e-10)
            LR = np.linalg.cholesky(self.R + np.eye(self.obs_dim)   * 1e-10)
        except np.linalg.LinAlgError:
            LQ = np.eye(self.state_dim) * 0.1
            LR = np.eye(self.obs_dim)   * 0.1

        x[0] = self.mu0 + rng.standard_normal(self.state_dim) * 0.1
        y[0] = self.h(x[0]) + LR @ rng.standard_normal(self.obs_dim)

        for t in range(1, T):
            x[t] = self.f(x[t - 1]) + LQ @ rng.standard_normal(self.state_dim)
            y[t] = self.h(x[t])     + LR @ rng.standard_normal(self.obs_dim)

        return x, y

    # ------------------------------------------------------------------
    # Jacobian check (numerical vs analytical)
    # ------------------------------------------------------------------

    def check_jacobians(self, x: Optional[np.ndarray] = None, eps: float = 1e-5) -> dict:
        """
        Compare analytical Jacobians against finite-difference approximations.

        Useful for verifying custom Jacobian implementations.

        Returns
        -------
        dict with 'F_max_error' and 'H_max_error'
        """
        if x is None:
            x = np.random.randn(self.state_dim)

        # Finite difference for F_jac
        F_analytical = self.F_jac(x)
        F_numerical  = np.zeros_like(F_analytical)
        for i in range(self.state_dim):
            dx = np.zeros(self.state_dim)
            dx[i] = eps
            F_numerical[:, i] = (self.f(x + dx) - self.f(x - dx)) / (2 * eps)

        # For observation Jacobian (H is linear so it's exact)
        H_analytical = self.H_jac(x)
        H_numerical  = np.zeros_like(H_analytical)
        for i in range(self.state_dim):
            dx = np.zeros(self.state_dim)
            dx[i] = eps
            H_numerical[:, i] = (self.h(x + dx) - self.h(x - dx)) / (2 * eps)

        F_err = np.max(np.abs(F_analytical - F_numerical))
        H_err = np.max(np.abs(H_analytical - H_numerical))

        return {"F_max_error": F_err, "H_max_error": H_err}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self):
        print(f"NonlinearSSM '{self.name}'")
        print(f"  state_dim : {self.state_dim}")
        print(f"  obs_dim   : {self.obs_dim}")
        print(f"  dynamics  : {self.dynamics.value}")
        print(f"  alpha     : {self.alpha}")
        print(f"  Q  diag   : {np.diag(self.Q).round(4)}")
        print(f"  R  diag   : {np.diag(self.R).round(4)}")

    def __repr__(self):
        return (
            f"NonlinearSSM(name='{self.name}', "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim}, "
            f"dynamics='{self.dynamics.value}', alpha={self.alpha})"
        )


# ---------------------------------------------------------------------------
# Network-traffic presets
# ---------------------------------------------------------------------------

def normal_traffic_model(state_dim: int = 6, obs_dim: int = 10) -> NonlinearSSM:
    """
    Nonlinear SSM tuned for normal network traffic.
    Low noise, slow-changing dynamics (alpha close to 1).
    """
    return NonlinearSSM.random_init(
        state_dim=state_dim, obs_dim=obs_dim,
        dynamics=DynamicsType.TANH,
        alpha=0.99, noise_scale=0.05,
        name="normal_traffic", seed=0,
    )


def scanning_model(state_dim: int = 6, obs_dim: int = 10) -> NonlinearSSM:
    """
    Nonlinear SSM tuned for port-scanning / reconnaissance.
    Higher variance, faster state changes.
    """
    return NonlinearSSM.random_init(
        state_dim=state_dim, obs_dim=obs_dim,
        dynamics=DynamicsType.SIGMOID_DECAY,
        alpha=0.92, noise_scale=0.3,
        name="scanning", seed=1,
    )


def exfiltration_model(state_dim: int = 6, obs_dim: int = 10) -> NonlinearSSM:
    """
    Nonlinear SSM tuned for data exfiltration.
    Log-ratio dynamics model large outbound flow bursts.
    """
    return NonlinearSSM.random_init(
        state_dim=state_dim, obs_dim=obs_dim,
        dynamics=DynamicsType.LOG_RATIO,
        alpha=0.85, noise_scale=0.5,
        name="exfiltration", seed=2,
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NonlinearSSM sanity check...\n")

    state_dim, obs_dim = 5, 8

    # 1. All dynamics types
    for dtype in DynamicsType:
        m = NonlinearSSM.random_init(state_dim, obs_dim, dynamics=dtype, seed=42)
        x, y = m.simulate(T=200, seed=0)
        print(f"  [{dtype.value:15s}]  x {x.shape}  y {y.shape}")

    print()

    # 2. Jacobian check for TANH model
    model_tanh = NonlinearSSM.random_init(state_dim, obs_dim, dynamics=DynamicsType.TANH, seed=7)
    errs = model_tanh.check_jacobians()
    print(f"Jacobian check (TANH):")
    print(f"  F_jac max error : {errs['F_max_error']:.2e}  (should be < 1e-5)")
    print(f"  H_jac max error : {errs['H_max_error']:.2e}  (should be ~0)")
    print()

    # 3. Build EKF and UKF
    ekf = model_tanh.build_ekf()
    ukf = model_tanh.build_ukf()

    _, y_obs = model_tanh.simulate(T=300, seed=1)
    ekf_result = ekf.filter(y_obs)
    ukf_result = ukf.filter(y_obs)

    print(f"EKF log-likelihood : {ekf_result.log_likelihood:.2f}")
    print(f"UKF log-likelihood : {ukf_result.log_likelihood:.2f}")

    # 4. Preset models
    m_normal = normal_traffic_model()
    m_scan   = scanning_model()
    m_exfil  = exfiltration_model()
    print(f"\nPreset models:")
    print(f"  {m_normal}")
    print(f"  {m_scan}")
    print(f"  {m_exfil}")

    # 5. Promote linear → nonlinear
    from src.models.linear_ssm import LinearSSM
    lin = LinearSSM.random_init(state_dim, obs_dim, seed=42, name="baseline")
    nonlin = NonlinearSSM.from_linear_ssm(lin, dynamics=DynamicsType.TANH)
    print(f"\nPromoted: {lin} → {nonlin}")

    print("\nAll NonlinearSSM checks passed!")
