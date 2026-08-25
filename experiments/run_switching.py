"""
run_switching.py
----------------
Full Switching SSM experiment for attack regime detection.

Trains the Switching State-Space Model using EM on network telemetry,
then runs inference on the test set to detect attack stages.

Usage:
    python experiments/run_switching.py \
        --data        data/processed/cicids2017_features.npy \
        --labels      data/processed/cicids2017_labels.npy \
        --regimes     4 \
        --state-dim   8 \
        --obs-dim     10 \
        --em-iters    20 \
        --window-size 100
"""

import argparse
import sys
import os
import time
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.variational_switching import SwitchingSSM
from src.data_processing.feature_engineering import (
    build_observation_sequence,
    temporal_train_test_split,
)
from src.utils.metrics import full_evaluation_report
from src.utils.visualization import (
    plot_regime_probabilities,
    plot_hidden_state,
    plot_confusion_matrix,
    plot_log_likelihood_curve,
    plot_detection_timeline,
)


def run_switching_experiment(args):
    print("\n" + "=" * 65)
    print("PARD-SSM  |  SWITCHING SSM EXPERIMENT")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1/6]  Loading data ...")
    X = np.load(args.data)
    y = np.load(args.labels)
    print(f"       Features : {X.shape}  |  Labels : {y.shape}")
    print(f"       Regimes  : {sorted(np.unique(y).tolist())}")

    n_true_regimes = len(np.unique(y))
    if args.regimes != n_true_regimes:
        print(
            f"       [NOTE] --regimes={args.regimes} differs from "
            f"actual label count={n_true_regimes}. "
            f"Using {args.regimes} as the model's regime count."
        )

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print(f"\n[2/6]  Preprocessing ...")
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

    # Use first window (extend to all windows in Phase II)
    train_seq  = result_train["observations"][0]
    test_seq   = result_test["observations"][0]
    y_test_seq = result_test["labels"][0]

    print(f"       Train seq : {train_seq.shape}")
    print(f"       Test  seq : {test_seq.shape}")

    # ------------------------------------------------------------------
    # 3. Build Switching SSM
    # ------------------------------------------------------------------
    print(f"\n[3/6]  Building Switching SSM ...")
    print(f"       n_regimes={args.regimes}, state_dim={args.state_dim}, obs_dim={args.obs_dim}")

    model = SwitchingSSM(
        n_regimes=args.regimes,
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
    )

    # ------------------------------------------------------------------
    # 4. EM training
    # ------------------------------------------------------------------
    print(f"\n[4/6]  EM training ({args.em_iters} iterations) ...")
    t0 = time.perf_counter()
    log_likelihoods = model.fit(train_seq, n_iter=args.em_iters, verbose=True)
    train_time = time.perf_counter() - t0
    print(f"       Training time : {train_time:.2f}s")
    print(f"       Final LL      : {log_likelihoods[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Inference on test set
    # ------------------------------------------------------------------
    print(f"\n[5/6]  Inference on test set ...")
    t0 = time.perf_counter()
    inference_result = model.filter(test_seq)
    infer_time = time.perf_counter() - t0
    print(f"       Inference time : {infer_time:.4f}s")
    print(f"       Test LL        : {inference_result.log_likelihood:.4f}")

    # Align lengths
    T = min(
        len(inference_result.viterbi_path),
        len(y_test_seq),
        len(test_seq),
    )
    predicted_labels = inference_result.viterbi_path[:T]
    true_labels      = y_test_seq[:T]
    regime_probs     = inference_result.regime_probs[:T]
    filtered_means   = inference_result.filtered_means[:T]
    predicted_obs    = inference_result.predicted_observations[:T]
    obs_seq          = test_seq[:T]

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print(f"\n[6/6]  Evaluation ...")
    attack_regimes = list(range(1, args.regimes))

    metrics = full_evaluation_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        regime_probs=regime_probs,
        observations=obs_seq,
        predicted_observations=predicted_obs,
        log_likelihood=inference_result.log_likelihood,
        attack_regimes=attack_regimes,
    )

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    save_path = "experiments/results/switching_output.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "metrics":         metrics,
            "regime_probs":    regime_probs,
            "viterbi_path":    predicted_labels,
            "true_labels":     true_labels,
            "log_likelihoods": log_likelihoods,
            "filtered_means":  filtered_means,
            "obs_seq":         obs_seq,
            "train_time":      train_time,
            "infer_time":      infer_time,
        }, f)
    print(f"\nResults saved → {save_path}")

    # Visualise
    if not args.no_plot:
        print("\nGenerating plots ...")
        plot_log_likelihood_curve(log_likelihoods, title="EM Training — Log-Likelihood")
        plot_regime_probabilities(
            regime_probs,
            true_labels=true_labels,
            viterbi_path=predicted_labels,
            title="Attack Regime Probabilities (Switching SSM)",
        )
        plot_hidden_state(filtered_means, dims_to_plot=3)
        plot_confusion_matrix(true_labels, predicted_labels)
        plot_detection_timeline(
            regime_probs, true_labels,
            attack_regime_ids=attack_regimes,
        )

    print("\nSwitching SSM experiment complete.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switching SSM experiment")
    parser.add_argument("--data",        required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--regimes",     type=int, default=4)
    parser.add_argument("--state-dim",   type=int, default=8)
    parser.add_argument("--obs-dim",     type=int, default=10)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--em-iters",    type=int, default=20)
    parser.add_argument("--no-plot",     action="store_true")
    args = parser.parse_args()
    run_switching_experiment(args)
