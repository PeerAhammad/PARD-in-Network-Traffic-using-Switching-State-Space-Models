"""
switching_ssm.py
----------------
Switching State-Space Model (SSSM) — the core model for PARD.

Each attack regime has its own linear SSM parameters.
A discrete Markov chain governs transitions between regimes.

The model:
    s_t ~ Markov(Π)                        — discrete regime (attack stage)
    x_t = A_{s_t} * x_{t-1} + w_t         — continuous hidden state
    y_t = C_{s_t} * x_t + v_t             — observed telemetry

Inference: approximate forward filtering via regime-conditioned Kalman filters
plus a viterbi/forward algorithm over regime probabilities.

Reference:
    Ghahramani & Hinton (1996). Switching State-Space Models.
    Murphy (1998). Switching Kalman Filters.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.special import logsumexp

from .kalman_filter import KalmanFilter, KalmanFilterResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeSSM:
    """Parameters for one regime's linear SSM."""
    regime_id: int
    A: np.ndarray   # state transition   (state_dim, state_dim)
    C: np.ndarray   # observation matrix (obs_dim, state_dim)
    Q: np.ndarray   # process noise      (state_dim, state_dim)
    R: np.ndarray   # obs noise          (obs_dim, obs_dim)


@dataclass
class SwitchingSSMResult:
    """Output of the Switching SSM forward filter."""
    regime_probs: np.ndarray          # (T, n_regimes) — P(s_t | y_{1:t})
    filtered_means: np.ndarray        # (T, state_dim) — E[x_t | y_{1:t}]
    filtered_covs: np.ndarray         # (T, state_dim, state_dim)
    viterbi_path: np.ndarray          # (T,) — most likely regime sequence
    log_likelihood: float
    predicted_observations: np.ndarray  # (T, obs_dim) — y_{t|t-1}


# ---------------------------------------------------------------------------
# Switching SSM
# ---------------------------------------------------------------------------

class SwitchingSSM:
    """
    Switching State-Space Model with K regimes.

    Inference is performed using the GPB2 (Generalized Pseudo-Bayesian order 2)
    approximation — a practical approximation to exact inference which is
    intractable due to exponential growth of mixture components.

    Parameters
    ----------
    n_regimes  : K — number of attack regimes
    state_dim  : dimension of hidden state x_t
    obs_dim    : dimension of observations y_t
    """

    def __init__(self, n_regimes: int, state_dim: int, obs_dim: int):
        self.n_regimes = n_regimes
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # Markov transition matrix (row i = P(s_t | s_{t-1}=i))
        self.transition_matrix = self._init_transition_matrix()

        # Initial regime distribution
        self.initial_regime_probs = np.ones(n_regimes) / n_regimes

        # Per-regime SSM parameters (initialized, to be learned via EM)
        self.regimes: List[RegimeSSM] = []
        self._init_regime_params()

        # Initial state per regime
        self.mu0 = [np.zeros(state_dim) for _ in range(n_regimes)]
        self.P0 = [np.eye(state_dim) * 1.0 for _ in range(n_regimes)]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_transition_matrix(self) -> np.ndarray:
        """
        Initialize Markov transition matrix.

        High self-transition (0.95) with small probability of switching,
        reflecting that attack stages persist for multiple time steps.
        """
        K = self.n_regimes
        stay = 0.95
        leave = (1.0 - stay) / (K - 1) if K > 1 else 0.0
        Pi = np.full((K, K), leave)
        np.fill_diagonal(Pi, stay)
        return Pi

    def _init_regime_params(self):
        """Initialize regime SSM parameters."""
        rng = np.random.default_rng(42)
        for k in range(self.n_regimes):
            # Slightly different dynamics per regime
            A = np.eye(self.state_dim) * (0.90 + 0.02 * k)
            C = rng.standard_normal((self.obs_dim, self.state_dim)) * 0.1
            Q = np.eye(self.state_dim) * (0.1 + 0.05 * k)
            R = np.eye(self.obs_dim) * (0.2 + 0.05 * k)
            self.regimes.append(RegimeSSM(regime_id=k, A=A, C=C, Q=Q, R=R))

    def set_regime_params(
        self,
        k: int,
        A: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        """Manually set parameters for regime k."""
        r = self.regimes[k]
        if A is not None: r.A = np.array(A)
        if C is not None: r.C = np.array(C)
        if Q is not None: r.Q = np.array(Q)
        if R is not None: r.R = np.array(R)

    # ------------------------------------------------------------------
    # Forward filtering (GPB2 approximation)
    # ------------------------------------------------------------------

    def filter(self, observations: np.ndarray) -> SwitchingSSMResult:
        """
        Run switching Kalman filter on observation sequence.

        Uses GPB2 (Generalized Pseudo-Bayesian) approximation:
        At each step, we maintain K Kalman filters (one per regime)
        and compute the mixture weights (regime probabilities).

        Parameters
        ----------
        observations : (T, obs_dim)

        Returns
        -------
        SwitchingSSMResult
        """
        T, obs_dim = observations.shape
        K = self.n_regimes
        d = self.state_dim

        # Storage
        regime_probs = np.zeros((T, K))          # P(s_t | y_{1:t})
        filtered_means = np.zeros((T, d))
        filtered_covs = np.zeros((T, d, d))
        predicted_obs = np.zeros((T, obs_dim))
        total_log_lik = 0.0

        # --- Initialize ---
        # Per-regime state estimates: mu[k], P[k]
        mu = [self.mu0[k].copy() for k in range(K)]
        P = [self.P0[k].copy() for k in range(K)]
        regime_prob = self.initial_regime_probs.copy()

        for t in range(T):
            y_t = observations[t]

            # --- Predict step for each regime ---
            mu_pred = []
            P_pred = []
            for k in range(K):
                r = self.regimes[k]
                mp = r.A @ mu[k]
                Pp = r.A @ P[k] @ r.A.T + r.Q
                mu_pred.append(mp)
                P_pred.append(Pp)

            # --- Predicted regime probs (Chapman-Kolmogorov) ---
            # P(s_t = j | y_{1:t-1}) = sum_i P(s_t=j|s_{t-1}=i) * P(s_{t-1}=i)
            pred_regime_prob = self.transition_matrix.T @ regime_prob  # (K,)

            # --- Measurement likelihood per regime ---
            log_liks = np.zeros(K)
            for k in range(K):
                r = self.regimes[k]
                innov = y_t - r.C @ mu_pred[k]
                S = r.C @ P_pred[k] @ r.C.T + r.R
                sign, logdet = np.linalg.slogdet(S)
                log_liks[k] = -0.5 * (
                    obs_dim * np.log(2 * np.pi)
                    + logdet
                    + innov @ np.linalg.solve(S, innov)
                )

            # --- Update regime probabilities ---
            log_joint = log_liks + np.log(pred_regime_prob + 1e-300)
            log_norm = logsumexp(log_joint)
            regime_prob = np.exp(log_joint - log_norm)
            regime_prob = np.clip(regime_prob, 1e-10, 1.0)
            regime_prob /= regime_prob.sum()
            regime_probs[t] = regime_prob
            total_log_lik += log_norm

            # --- Update state estimate per regime (Kalman update) ---
            mu_filt = []
            P_filt = []
            for k in range(K):
                r = self.regimes[k]
                innov = y_t - r.C @ mu_pred[k]
                S = r.C @ P_pred[k] @ r.C.T + r.R
                Kgain = P_pred[k] @ r.C.T @ np.linalg.inv(S)
                mf = mu_pred[k] + Kgain @ innov
                Pf = (np.eye(d) - Kgain @ r.C) @ P_pred[k]
                Pf = 0.5 * (Pf + Pf.T)
                mu_filt.append(mf)
                P_filt.append(Pf)

            # --- Collapse mixture (GPB1 approximation for next step) ---
            mu_collapse = sum(regime_prob[k] * mu_filt[k] for k in range(K))
            P_collapse = sum(
                regime_prob[k] * (
                    P_filt[k]
                    + np.outer(mu_filt[k] - mu_collapse, mu_filt[k] - mu_collapse)
                )
                for k in range(K)
            )
            mu = [mu_collapse.copy() for _ in range(K)]
            P = [P_collapse.copy() for _ in range(K)]

            # Weighted mean state
            filtered_means[t] = mu_collapse
            filtered_covs[t] = P_collapse

            # Predicted observation (mixture)
            y_pred = sum(
                regime_prob[k] * self.regimes[k].C @ mu_pred[k]
                for k in range(K)
            )
            predicted_obs[t] = y_pred

        # --- Viterbi decoding (most likely regime path) ---
        viterbi = self._viterbi(observations)

        return SwitchingSSMResult(
            regime_probs=regime_probs,
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            viterbi_path=viterbi,
            log_likelihood=total_log_lik,
            predicted_observations=predicted_obs,
        )

    # ------------------------------------------------------------------
    # Viterbi algorithm
    # ------------------------------------------------------------------

    def _viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm to find the most likely regime sequence.

        Returns
        -------
        path : (T,) integer array of most likely regimes
        """
        T, obs_dim = observations.shape
        K = self.n_regimes

        log_Pi = np.log(self.transition_matrix + 1e-300)
        log_delta = np.log(self.initial_regime_probs + 1e-300)
        psi = np.zeros((T, K), dtype=int)

        # Initialize
        for k in range(K):
            r = self.regimes[k]
            y = observations[0]
            innov = y - r.C @ self.mu0[k]
            S = r.C @ self.P0[k] @ r.C.T + r.R
            _, logdet = np.linalg.slogdet(S)
            log_delta[k] += -0.5 * (obs_dim * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov))

        # Forward
        for t in range(1, T):
            for k in range(K):
                r = self.regimes[k]
                # Transition scores
                scores = log_delta + log_Pi[:, k]
                psi[t, k] = np.argmax(scores)
                log_delta_new = scores[psi[t, k]]

                # Emission
                y = observations[t]
                mp = r.A @ self.mu0[k]  # simplified
                innov = y - r.C @ mp
                S = r.C @ self.P0[k] @ r.C.T + r.R
                _, logdet = np.linalg.slogdet(S)
                log_delta_new += -0.5 * (obs_dim * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov))
                log_delta[k] = log_delta_new

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta)
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    # ------------------------------------------------------------------
    # EM learning (simplified M-step parameter update)
    # ------------------------------------------------------------------

    def fit(
        self,
        observations: np.ndarray,
        n_iter: int = 20,
        verbose: bool = True,
    ) -> List[float]:
        """
        Fit model parameters via EM (Expectation-Maximization).

        E-step: run switching filter to get regime responsibilities
        M-step: update transition matrix from regime assignments

        Parameters
        ----------
        observations : (T, obs_dim)
        n_iter       : EM iterations
        verbose      : print log-likelihood each iteration

        Returns
        -------
        log_likelihoods : list of log-likelihood per iteration
        """
        log_likelihoods = []

        for i in range(n_iter):
            # E-step
            result = self.filter(observations)
            ll = result.log_likelihood
            log_likelihoods.append(ll)

            if verbose:
                print(f"  EM iter {i+1:3d} | log-likelihood: {ll:.4f}")

            # M-step: update transition matrix from soft counts
            probs = result.regime_probs  # (T, K)
            T = probs.shape[0]
            counts = np.zeros((self.n_regimes, self.n_regimes))
            for t in range(T - 1):
                counts += np.outer(probs[t], probs[t + 1])

            # Normalize rows
            row_sums = counts.sum(axis=1, keepdims=True)
            self.transition_matrix = counts / (row_sums + 1e-10)

            # Update initial distribution
            self.initial_regime_probs = probs[0]

        return log_likelihoods


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Switching SSM sanity check...")
    np.random.seed(42)

    T, obs_dim, state_dim, n_regimes = 300, 5, 3, 3

    # Synthetic observations with 3 regimes (each lasting ~100 steps)
    obs = np.concatenate([
        np.random.randn(100, obs_dim) * 0.5,           # normal
        np.random.randn(100, obs_dim) * 2.0 + 3.0,     # attack
        np.random.randn(100, obs_dim) * 0.3,           # recovered
    ])

    model = SwitchingSSM(n_regimes=n_regimes, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    print(f"regime_probs shape : {result.regime_probs.shape}")
    print(f"viterbi_path[:10]  : {result.viterbi_path[:10]}")
    print(f"log_likelihood     : {result.log_likelihood:.2f}")
    print(f"Dominant regime at t=50 : {result.regime_probs[50].argmax()}")
    print(f"Dominant regime at t=150: {result.regime_probs[150].argmax()}")
    print("Switching SSM OK!")
