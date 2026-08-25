"""
linear_ssm.py
-------------
Linear Gaussian State-Space Model (SSM) definition.

This module defines the parameter container and initialization strategies
for a Linear Gaussian SSM. It is the foundation for the Kalman Filter
and the per-regime models inside the Switching SSM.

Model:
    x_t = A * x_{t-1} + w_t,    w_t ~ N(0, Q)   ← state dynamics
    y_t = C * x_t   + v_t,      v_t ~ N(0, R)   ← observation model
    x_0 ~ N(mu0, P0)                              ← initial state prior

Typical use:
    model = LinearSSM.from_data(observations, state_dim=8)
    kf    = model.build_kalman_filter()
    result = kf.filter(observations)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.decomposition import PCA


@dataclass
class LinearSSM:
    """
    Parameter container for a Linear Gaussian SSM.

    Attributes
    ----------
    A   : state transition matrix           (state_dim, state_dim)
    C   : observation matrix                (obs_dim,   state_dim)
    Q   : process noise covariance          (state_dim, state_dim)
    R   : observation noise covariance      (obs_dim,   obs_dim)
    mu0 : initial state mean                (state_dim,)
    P0  : initial state covariance          (state_dim, state_dim)
    state_dim : hidden state dimension
    obs_dim   : observation dimension
    name      : optional label (e.g. "normal_traffic", "port_scan")
    """

    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    mu0: np.ndarray
    P0: np.ndarray
    state_dim: int = field(init=False)
    obs_dim: int = field(init=False)
    name: str = "linear_ssm"

    def __post_init__(self):
        self.A   = np.array(self.A,   dtype=np.float64)
        self.C   = np.array(self.C,   dtype=np.float64)
        self.Q   = np.array(self.Q,   dtype=np.float64)
        self.R   = np.array(self.R,   dtype=np.float64)
        self.mu0 = np.array(self.mu0, dtype=np.float64)
        self.P0  = np.array(self.P0,  dtype=np.float64)
        self.state_dim = self.A.shape[0]
        self.obs_dim   = self.C.shape[0]
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        assert self.A.shape  == (self.state_dim, self.state_dim), "A shape mismatch"
        assert self.C.shape  == (self.obs_dim,   self.state_dim), "C shape mismatch"
        assert self.Q.shape  == (self.state_dim, self.state_dim), "Q shape mismatch"
        assert self.R.shape  == (self.obs_dim,   self.obs_dim),   "R shape mismatch"
        assert self.mu0.shape == (self.state_dim,),               "mu0 shape mismatch"
        assert self.P0.shape  == (self.state_dim, self.state_dim),"P0 shape mismatch"

    # ------------------------------------------------------------------
    # Constructors / Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls,
        state_dim: int,
        obs_dim: int,
        noise_scale: float = 0.1,
        stability: float = 0.95,
        name: str = "linear_ssm",
        seed: Optional[int] = None,
    ) -> "LinearSSM":
        """
        Random initialisation — useful for testing and as a starting point
        for EM-based learning.

        Parameters
        ----------
        state_dim   : dimension of hidden state
        obs_dim     : dimension of observations
        noise_scale : scale of Q and R
        stability   : spectral radius of A (< 1 keeps dynamics stable)
        seed        : random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # A: random stable matrix — eigenvalues inside unit circle
        A_raw = rng.standard_normal((state_dim, state_dim))
        eigvals = np.linalg.eigvals(A_raw)
        spectral_radius = np.max(np.abs(eigvals))
        A = (A_raw / spectral_radius) * stability

        # C: random projection from hidden → observed
        C = rng.standard_normal((obs_dim, state_dim)) * 0.1

        # Noise covariances: diagonal, positive definite
        Q = np.diag(rng.uniform(noise_scale * 0.5, noise_scale * 1.5, state_dim))
        R = np.diag(rng.uniform(noise_scale * 0.5, noise_scale * 1.5, obs_dim))

        mu0 = np.zeros(state_dim)
        P0  = np.eye(state_dim)

        return cls(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0, name=name)

    @classmethod
    def from_data(
        cls,
        observations: np.ndarray,
        state_dim: int,
        noise_scale: float = 0.1,
        method: str = "pca",
        name: str = "linear_ssm",
    ) -> "LinearSSM":
        """
        Data-driven initialisation.

        Uses PCA to set C so that the observation matrix points along the
        principal directions of variance in the data — a much better
        starting point than purely random.

        Parameters
        ----------
        observations : (T, obs_dim) — training sequence
        state_dim    : hidden state dimension (should be ≤ obs_dim)
        noise_scale  : noise level
        method       : 'pca' (recommended) | 'random'
        """
        T, obs_dim = observations.shape

        if method == "pca" and state_dim <= obs_dim:
            # C initialised via PCA: top state_dim principal components
            pca = PCA(n_components=state_dim)
            pca.fit(observations)
            # C maps hidden (state_dim) → observed (obs_dim)
            # pca.components_ is (state_dim, obs_dim) → transpose
            C = pca.components_.T   # (obs_dim, state_dim)

            # Estimate noise from residuals
            X_pca = pca.transform(observations)          # (T, state_dim)
            recon  = pca.inverse_transform(X_pca)        # (T, obs_dim)
            resid  = observations - recon
            R_diag = np.var(resid, axis=0) + 1e-6
            R = np.diag(R_diag)

            # A: estimate from lagged covariance of PCA scores
            if T > 2:
                Xlag0 = X_pca[:-1]     # x_{t-1}
                Xlag1 = X_pca[1:]      # x_t
                # Least-squares: x_t ≈ A * x_{t-1}
                A_raw, _, _, _ = np.linalg.lstsq(Xlag0, Xlag1, rcond=None)
                A = A_raw.T
                # Clip spectral radius for stability
                eigvals = np.linalg.eigvals(A)
                sr = np.max(np.abs(eigvals))
                if sr >= 1.0:
                    A = A / (sr + 0.05)
            else:
                A = np.eye(state_dim) * 0.95

            # Q from residuals of A fit
            pred  = (A @ Xlag0.T).T
            Q_diag = np.var(Xlag1 - pred, axis=0) + 1e-6
            Q = np.diag(Q_diag)

            mu0 = X_pca[0]
            P0  = np.eye(state_dim)

        else:
            return cls.random_init(state_dim, obs_dim, noise_scale, name=name)

        return cls(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0, name=name)

    @classmethod
    def identity_init(
        cls,
        state_dim: int,
        obs_dim: int,
        noise_scale: float = 0.1,
        name: str = "linear_ssm",
    ) -> "LinearSSM":
        """
        Simple identity / diagonal initialisation.
        Fast and interpretable — useful as a sanity-check baseline.
        Requires state_dim == obs_dim.
        """
        assert state_dim == obs_dim, "identity_init requires state_dim == obs_dim"
        A   = np.eye(state_dim) * 0.99
        C   = np.eye(obs_dim)
        Q   = np.eye(state_dim) * noise_scale
        R   = np.eye(obs_dim)   * noise_scale
        mu0 = np.zeros(state_dim)
        P0  = np.eye(state_dim)
        return cls(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0, name=name)

    # ------------------------------------------------------------------
    # Build inference object
    # ------------------------------------------------------------------

    def build_kalman_filter(self):
        """
        Instantiate and return a KalmanFilter using this model's parameters.

        Returns
        -------
        KalmanFilter instance (from src.inference.kalman_filter)
        """
        from src.inference.kalman_filter import KalmanFilter
        return KalmanFilter(
            A=self.A, C=self.C, Q=self.Q, R=self.R,
            mu0=self.mu0, P0=self.P0,
        )

    # ------------------------------------------------------------------
    # Simulate data from this model
    # ------------------------------------------------------------------

    def simulate(
        self,
        T: int,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate a synthetic trajectory from this SSM.

        Useful for unit testing and verifying that the filter can recover
        the hidden state.

        Parameters
        ----------
        T    : number of time steps
        seed : random seed

        Returns
        -------
        x : (T, state_dim) — true hidden states
        y : (T, obs_dim)   — noisy observations
        """
        rng = np.random.default_rng(seed)

        x = np.zeros((T, self.state_dim))
        y = np.zeros((T, self.obs_dim))

        # Sample initial state
        try:
            L0 = np.linalg.cholesky(self.P0)
            x[0] = self.mu0 + L0 @ rng.standard_normal(self.state_dim)
        except np.linalg.LinAlgError:
            x[0] = self.mu0

        y[0] = self.C @ x[0] + np.linalg.cholesky(self.R) @ rng.standard_normal(self.obs_dim)

        LQ = np.linalg.cholesky(self.Q + np.eye(self.state_dim) * 1e-10)
        LR = np.linalg.cholesky(self.R + np.eye(self.obs_dim)   * 1e-10)

        for t in range(1, T):
            x[t] = self.A @ x[t - 1] + LQ @ rng.standard_normal(self.state_dim)
            y[t] = self.C @ x[t]     + LR @ rng.standard_normal(self.obs_dim)

        return x, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_stable(self) -> bool:
        """Check if the state transition matrix A is stable (spectral radius < 1)."""
        sr = np.max(np.abs(np.linalg.eigvals(self.A)))
        return bool(sr < 1.0)

    def spectral_radius(self) -> float:
        """Return spectral radius of A."""
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def summary(self):
        """Print a compact summary of model parameters."""
        print(f"LinearSSM '{self.name}'")
        print(f"  state_dim : {self.state_dim}")
        print(f"  obs_dim   : {self.obs_dim}")
        print(f"  stable    : {self.is_stable()} (spectral radius = {self.spectral_radius():.4f})")
        print(f"  A  shape  : {self.A.shape}")
        print(f"  C  shape  : {self.C.shape}")
        print(f"  Q  diag   : {np.diag(self.Q).round(4)}")
        print(f"  R  diag   : {np.diag(self.R).round(4)}")

    def __repr__(self):
        return (
            f"LinearSSM(name='{self.name}', "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim}, "
            f"stable={self.is_stable()})"
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("LinearSSM sanity check...\n")

    # 1. Random init
    model_rand = LinearSSM.random_init(state_dim=4, obs_dim=8, seed=42, name="random")
    model_rand.summary()
    print()

    # 2. Simulate data, then recover with from_data
    x_true, y_obs = model_rand.simulate(T=500, seed=0)
    print(f"Simulated trajectory: x {x_true.shape}, y {y_obs.shape}")

    # 3. Data-driven init from the observations
    model_pca = LinearSSM.from_data(y_obs, state_dim=4, method="pca", name="pca_init")
    model_pca.summary()
    print()

    # 4. Run Kalman filter
    kf = model_pca.build_kalman_filter()
    result = kf.filter(y_obs)
    print(f"KF log-likelihood : {result.log_likelihood:.2f}")
    print(f"Filtered shape    : {result.filtered_means.shape}")

    # 5. Identity init (state_dim == obs_dim)
    model_id = LinearSSM.identity_init(state_dim=5, obs_dim=5, name="identity")
    print(f"\nIdentity model: {model_id}")

    print("\nAll LinearSSM checks passed!")
