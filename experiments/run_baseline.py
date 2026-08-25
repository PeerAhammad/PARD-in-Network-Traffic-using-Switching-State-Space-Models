"""
run_baseline.py
---------------
Baseline experiment: compare KF, EKF, and UKF on preprocessed network data.

This script runs all three non-switching filters as baselines.
Results are saved and compared against the Switching SSM.

Usage:
    python experiments/run_baseline.py \
        --data   data/processed/cicids2017_features.npy \
        --labels data/processed/cicids2017_labels.npy \
        --state-dim 8 \
        --obs-dim   10
"""

import argparse
import sys
import os
import time
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.linear_ssm    import LinearSSM
from src.models.nonlinear_ssm import NonlinearSSM, DynamicsType
from src.data_processing.feature_engineering import (
    build_observation_sequence,
    temporal_train_test_split,
)
from src.utils.metrics import (
    regime_accuracy,
    prediction_mse,
    binary_attack_auc,
    detection_lead_time,
)
from src.utils.visualization import (
    plot_hidden_state,
    plot_log_likelihood_curve,
)


# ---------------------------------------------------------------------------
# Single-filter runner
# ---------------------------------------------------------------------------

def run_single_filter(name: str, filter_obj, obs_seq: np.ndarray) -> dict:
    """
    Run one filter on an observation sequence and collect metrics.

    Parameters
    ----------
    name       : display name (e.g. "Kalman Filter")
    filter_obj : a filter with a .filter(observations) method
    obs_seq    : (T, obs_dim)

    Returns
    -------
    dict with timing, log-likelihood, filtered means
    """
    print(f"  Running {name} ...")
    t0 = time.perf_counter()
    result = filter_obj.filter(obs_seq)
    elapsed = time.perf_counter() - t0

    # Predict next observation via filtered mean projected back
    # (simple linear approximation for all filters)
    predicted_obs = np.roll(obs_seq, shift=1, axis=0)
    predicted_obs[0] = obs_seq[0]

    return {
        "name":            name,
        "log_likelihood":  result.log_likelihood,
        "filtered_means":  result.filtered_means,
        "filtered_covs":   result.filtered_covs,
        "innovations":     result.innovations,
        "runtime_sec":     elapsed,
        "predicted_obs":   predicted_obs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(args):
    print("\n" + "=" * 65)
    print("PARD-SSM  |  BASELINE EXPERIMENT  |  KF / EKF / UKF")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1/4]  Loading data ...")
    X = np.load(args.data)
    y = np.load(args.labels)
    print(f"       Features : {X.shape}  |  Labels : {y.shape}")
    print(f"       Regimes  : {sorted(np.unique(y).tolist())}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print(f"\n[2/4]  Preprocessing ...")
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_ratio=0.2)

    result_train = build_observation_sequence(
        X_train, y_train,
        normalize=True,
        n_pca_components=args.obs_dim,
        window_size=args.window_size,
    )
    result_test = build_observation_sequence(
        X_test, y_test,
        normalize=True,
        n_pca_components=args.obs_dim,
        window_size=args.window_size,
    )

    # Use the first window as the evaluation sequence
    train_seq = result_train["observations"][0]   # (T, obs_dim)
    test_seq  = result_test["observations"][0]
    y_seq     = result_test["labels"][0]

    print(f"       Train seq : {train_seq.shape}")
    print(f"       Test  seq : {test_seq.shape}")

    # ------------------------------------------------------------------
    # 3. Build models and run filters
    # ------------------------------------------------------------------
    print(f"\n[3/4]  Running baseline filters ...")

    # --- Kalman Filter (linear SSM) ---
    linear_model = LinearSSM.from_data(
        train_seq, state_dim=args.state_dim, method="pca", name="kf_baseline"
    )
    kf = linear_model.build_kalman_filter()

    # --- EKF (nonlinear — tanh dynamics) ---
    ekf_model = NonlinearSSM.from_linear_ssm(
        linear_model, dynamics=DynamicsType.TANH, alpha=0.97
    )
    ekf = ekf_model.build_ekf()

    # --- UKF (unscented — tanh dynamics) ---
    ukf_model = NonlinearSSM.from_linear_ssm(
        linear_model, dynamics=DynamicsType.TANH, alpha=0.97
    )
    ukf = ukf_model.build_ukf()

    results = [
        run_single_filter("Kalman Filter (KF)",           kf,  test_seq),
        run_single_filter("Extended KF (EKF)",            ekf, test_seq),
        run_single_filter("Unscented KF (UKF)",           ukf, test_seq),
    ]

    # ------------------------------------------------------------------
    # 4. Print comparison table
    # ------------------------------------------------------------------
    print(f"\n[4/4]  Results\n")
    print(f"{'Method':<28} {'Log-Lik':>12}  {'Pred-MSE':>10}  {'Runtime(s)':>12}")
    print("-" * 68)

    summary_rows = []
    for r in results:
        mse = prediction_mse(test_seq, r["predicted_obs"])
        row = {
            "method":        r["name"],
            "log_likelihood": r["log_likelihood"],
            "mse":            mse,
            "runtime_sec":    r["runtime_sec"],
        }
        summary_rows.append(row)
        print(
            f"  {r['name']:<26} "
            f"{r['log_likelihood']:>12.2f}  "
            f"{mse:>10.6f}  "
            f"{r['runtime_sec']:>12.4f}"
        )

    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs("experiments/results", exist_ok=True)
    save_path = "experiments/results/baseline_output.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "summary":   summary_rows,
            "results":   results,
            "test_seq":  test_seq,
            "y_seq":     y_seq,
        }, f)
    print(f"Results saved → {save_path}")

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if not args.no_plot:
        for r in results:
            plot_hidden_state(
                r["filtered_means"],
                title=f"Hidden State — {r['name']}",
                dims_to_plot=3,
            )

    print("\nBaseline experiment complete.")
    return summary_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline KF/EKF/UKF experiment")
    parser.add_argument("--data",        required=True, help="Path to features .npy")
    parser.add_argument("--labels",      required=True, help="Path to labels .npy")
    parser.add_argument("--state-dim",   type=int, default=8,   help="Hidden state dim")
    parser.add_argument("--obs-dim",     type=int, default=10,  help="PCA obs dim")
    parser.add_argument("--window-size", type=int, default=100, help="Window size")
    parser.add_argument("--no-plot",     action="store_true",   help="Skip plots")
    args = parser.parse_args()

    run_baseline(args)
