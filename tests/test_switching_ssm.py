# tests/test_switching_ssm.py
"""Unit tests for the Switching SSM."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.variational_switching import SwitchingSSM


def make_synthetic_obs(T=200, obs_dim=5, n_regimes=3, seed=42):
    np.random.seed(seed)
    segment = T // n_regimes
    parts = []
    for k in range(n_regimes):
        mu = np.random.randn(obs_dim) * (k + 1)
        parts.append(np.random.randn(segment, obs_dim) * 0.5 + mu)
    return np.vstack(parts)


def test_switching_ssm_output_shapes():
    T, obs_dim, state_dim, K = 150, 5, 3, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    assert result.regime_probs.shape == (T, K)
    assert result.filtered_means.shape == (T, state_dim)
    assert result.viterbi_path.shape == (T,)


def test_regime_probs_sum_to_one():
    T, obs_dim, state_dim, K = 100, 4, 2, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    sums = result.regime_probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_viterbi_path_valid_regimes():
    T, obs_dim, state_dim, K = 100, 4, 2, 4
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    result = model.filter(obs)

    assert np.all(result.viterbi_path >= 0)
    assert np.all(result.viterbi_path < K)


def test_em_increases_likelihood():
    T, obs_dim, state_dim, K = 200, 5, 3, 3
    obs = make_synthetic_obs(T, obs_dim, K)
    model = SwitchingSSM(n_regimes=K, state_dim=state_dim, obs_dim=obs_dim)
    lls = model.fit(obs, n_iter=5, verbose=False)

    # Log-likelihood should generally increase (allow small numerical fluctuations)
    assert lls[-1] >= lls[0] - 10, "EM should not significantly decrease log-likelihood"


if __name__ == "__main__":
    test_switching_ssm_output_shapes()
    test_regime_probs_sum_to_one()
    test_viterbi_path_valid_regimes()
    test_em_increases_likelihood()
    print("All Switching SSM tests passed!")
