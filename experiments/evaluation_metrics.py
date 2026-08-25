"""
evaluation_metrics.py
---------------------
Aggregate, compare, and print results from all experiments.

Loads saved .pkl outputs from run_baseline.py and run_switching.py,
then produces a unified comparison table and summary plots.

Usage:
    # After running both experiments:
    python experiments/evaluation_metrics.py

    # Or point to specific result files:
    python experiments/evaluation_metrics.py \
        --baseline  experiments/results/baseline_output.pkl \
        --switching experiments/results/switching_output.pkl
"""

import argparse
import sys
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    regime_accuracy,
    prediction_mse,
    binary_attack_auc,
    detection_lead_time,
)
from src.utils.visualization import (
    plot_regime_probabilities,
    plot_confusion_matrix,
    plot_log_likelihood_curve,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(path: str) -> dict:
    """Load a pickle file and return its contents."""
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_fmt(val, fmt=".4f") -> str:
    """Format a value, returning 'N/A' if None or NaN."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "  N/A  "
        return f"{val:{fmt}}"
    except Exception:
        return "  N/A  "


# ---------------------------------------------------------------------------
# Per-method metric computation
# ---------------------------------------------------------------------------

def compute_metrics_from_baseline(baseline: dict, attack_regimes: list) -> list:
    """
    Extract per-filter metrics from baseline results.
    baseline_output.pkl stores a list of filter results.
    """
    rows = []
    y_seq = baseline.get("y_seq", None)

    for r in baseline.get("results", []):
        name    = r["name"]
        ll      = r["log_likelihood"]
        runtime = r["runtime_sec"]

        # Without regime probabilities, we can only report LL and MSE
        mse = prediction_mse(
            baseline["test_seq"],
            r["predicted_obs"],
        )

        rows.append({
            "method":          name,
            "log_likelihood":  ll,
            "mse":             mse,
            "regime_accuracy": None,   # baselines have no regime output
            "auc_roc":         None,
            "lead_time":       None,
            "runtime_sec":     runtime,
        })

    return rows


def compute_metrics_from_switching(switching: dict, attack_regimes: list) -> dict:
    """
    Extract metrics from switching SSM results.
    """
    true_labels  = switching["true_labels"]
    pred_labels  = switching["viterbi_path"]
    regime_probs = switching["regime_probs"]
    obs_seq      = switching.get("obs_seq", None)

    acc  = regime_accuracy(true_labels, pred_labels)
    lead = detection_lead_time(true_labels, pred_labels, attack_regimes)
    auc  = binary_attack_auc(true_labels, regime_probs, attack_regimes)
    ll   = switching["metrics"].get("log_likelihood", None)
    mse  = switching["metrics"].get("prediction_mse", None)

    return {
        "method":          "Switching SSM (PARD)",
        "log_likelihood":  ll,
        "mse":             mse,
        "regime_accuracy": acc,
        "auc_roc":         auc,
        "lead_time":       lead,
        "runtime_sec":     switching.get("train_time", None),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(rows: list):
    """Print a formatted comparison table of all methods."""
    col_w = {
        "method":          28,
        "log_likelihood":  12,
        "mse":             10,
        "regime_accuracy": 12,
        "auc_roc":         10,
        "lead_time":       12,
        "runtime_sec":     12,
    }

    header = (
        f"  {'Method':<28}  "
        f"{'Log-Lik':>12}  "
        f"{'Pred-MSE':>10}  "
        f"{'Acc':>12}  "
        f"{'AUC-ROC':>10}  "
        f"{'Lead(steps)':>12}  "
        f"{'Time(s)':>10}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  PARD-SSM — FULL EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        line = (
            f"  {r['method']:<28}  "
            f"{safe_fmt(r['log_likelihood']):>12}  "
            f"{safe_fmt(r['mse']):>10}  "
            f"{safe_fmt(r['regime_accuracy']):>12}  "
            f"{safe_fmt(r['auc_roc']):>10}  "
            f"{safe_fmt(r['lead_time'], fmt='.1f'):>12}  "
            f"{safe_fmt(r['runtime_sec'], fmt='.2f'):>10}"
        )
        print(line)

    print("=" * len(header))
    print()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(rows: list):
    """Highlight best method per metric."""
    metrics_to_compare = {
        "log_likelihood":  ("highest", max),
        "mse":             ("lowest",  min),
        "regime_accuracy": ("highest", max),
        "auc_roc":         ("highest", max),
    }

    print("  Best per metric:")
    for metric, (direction, fn) in metrics_to_compare.items():
        valid = [(r["method"], r[metric]) for r in rows if r[metric] is not None]
        if not valid:
            continue
        best_method, best_val = fn(valid, key=lambda x: x[1])
        print(f"    {metric:<22} → {best_method}  ({best_val:.4f})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(args):
    print("\n" + "=" * 65)
    print("PARD-SSM  |  EVALUATION & COMPARISON")
    print("=" * 65)

    all_rows = []
    attack_regimes = [1, 2, 3, 4]

    # --- Baseline results ---
    if args.baseline and Path(args.baseline).exists():
        print(f"\nLoading baseline results from: {args.baseline}")
        baseline = load_pkl(args.baseline)
        baseline_rows = compute_metrics_from_baseline(baseline, attack_regimes)
        all_rows.extend(baseline_rows)
        print(f"  Loaded {len(baseline_rows)} baseline method(s)")
    else:
        print(f"\n[SKIP] Baseline results not found at: {args.baseline}")
        print("       Run experiments/run_baseline.py first.")

    # --- Switching SSM results ---
    if args.switching and Path(args.switching).exists():
        print(f"Loading switching results from: {args.switching}")
        switching = load_pkl(args.switching)
        switching_row = compute_metrics_from_switching(switching, attack_regimes)
        all_rows.append(switching_row)
        print(f"  Loaded Switching SSM results")
    else:
        print(f"\n[SKIP] Switching results not found at: {args.switching}")
        print("       Run experiments/run_switching.py first.")

    if not all_rows:
        print("\nNo results found. Run the experiments first.")
        return

    # --- Print comparison ---
    print_comparison_table(all_rows)
    print_summary(all_rows)

    # --- Detailed regime report (switching only) ---
    if args.switching and Path(args.switching).exists():
        switching = load_pkl(args.switching)
        print("  Regime-Level Metrics (Switching SSM):")
        print(f"    True regime distribution  : "
              f"{ {k: int((switching['true_labels']==k).sum()) for k in np.unique(switching['true_labels'])} }")
        print(f"    Predicted regime dist     : "
              f"{ {k: int((switching['viterbi_path']==k).sum()) for k in np.unique(switching['viterbi_path'])} }")
        print()

    # --- Optional plots ---
    if PLOT_AVAILABLE and not args.no_plot:
        if args.switching and Path(args.switching).exists():
            switching = load_pkl(args.switching)
            print("Generating comparison plots ...")

            plot_log_likelihood_curve(
                switching["log_likelihoods"],
                title="EM Training — Log-Likelihood",
            )
            plot_regime_probabilities(
                switching["regime_probs"],
                true_labels=switching["true_labels"],
                viterbi_path=switching["viterbi_path"],
                title="Switching SSM — Regime Probabilities",
            )
            plot_confusion_matrix(
                switching["true_labels"],
                switching["viterbi_path"],
                title="Switching SSM — Confusion Matrix",
            )

    # --- Save consolidated report ---
    os.makedirs("experiments/results", exist_ok=True)
    report_path = "experiments/results/consolidated_report.pkl"
    with open(report_path, "wb") as f:
        pickle.dump({"all_rows": all_rows}, f)
    print(f"Consolidated report saved → {report_path}")

    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and compare all experiment results")
    parser.add_argument(
        "--baseline",
        default="experiments/results/baseline_output.pkl",
        help="Path to baseline_output.pkl",
    )
    parser.add_argument(
        "--switching",
        default="experiments/results/switching_output.pkl",
        help="Path to switching_output.pkl",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = parser.parse_args()

    run_evaluation(args)
