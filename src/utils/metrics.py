"""
metrics.py
----------
Evaluation metrics for regime detection and state-space model performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from typing import Optional, Dict


def regime_accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Classification accuracy over all time steps."""
    return accuracy_score(true_labels, predicted_labels)


def prediction_mse(
    observations: np.ndarray,
    predicted_observations: np.ndarray,
) -> float:
    """Mean squared error of one-step-ahead predictions."""
    return float(np.mean((observations - predicted_observations) ** 2))


def detection_lead_time(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    attack_regimes: list,
) -> float:
    """
    Average number of time steps the model detects an attack BEFORE
    the attack fully manifests.

    A positive lead time means early detection.
    Negative means late detection.

    Parameters
    ----------
    true_labels      : (T,) ground truth regime labels
    predicted_labels : (T,) model's predicted regime
    attack_regimes   : list of integer regime IDs considered as attacks

    Returns
    -------
    mean lead time in time steps
    """
    lead_times = []
    in_attack = False
    attack_start = None

    for t in range(len(true_labels)):
        is_true_attack = true_labels[t] in attack_regimes
        is_pred_attack = predicted_labels[t] in attack_regimes

        if is_true_attack and not in_attack:
            # True attack just started
            attack_start = t
            in_attack = True

        if in_attack and is_pred_attack:
            # Model detected it
            lead_times.append(attack_start - t)  # positive = early
            in_attack = False
            attack_start = None

        if not is_true_attack:
            in_attack = False
            attack_start = None

    return float(np.mean(lead_times)) if lead_times else 0.0


def binary_attack_auc(
    true_labels: np.ndarray,
    regime_probs: np.ndarray,
    attack_regimes: list,
) -> float:
    """
    Compute AUC-ROC treating all attack regimes as positive class.

    Parameters
    ----------
    true_labels   : (T,)
    regime_probs  : (T, K)
    attack_regimes: list of attack regime IDs

    Returns
    -------
    AUC-ROC score
    """
    y_true = np.isin(true_labels, attack_regimes).astype(int)
    y_score = regime_probs[:, attack_regimes].sum(axis=1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return roc_auc_score(y_true, y_score)


def full_evaluation_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    regime_probs: np.ndarray,
    observations: np.ndarray,
    predicted_observations: np.ndarray,
    log_likelihood: float,
    attack_regimes: list = None,
    regime_names: dict = None,
) -> Dict:
    """
    Compute and print full evaluation report.

    Returns
    -------
    dict of all metric values
    """
    if attack_regimes is None:
        attack_regimes = [1, 2, 3, 4]
    if regime_names is None:
        regime_names = {0: "Normal", 1: "Reconnaissance", 2: "Intrusion", 3: "DoS", 4: "Exfiltration"}

    acc = regime_accuracy(true_labels, predicted_labels)
    mse = prediction_mse(observations, predicted_observations)
    lead = detection_lead_time(true_labels, predicted_labels, attack_regimes)
    auc = binary_attack_auc(true_labels, regime_probs, attack_regimes)

    print("=" * 60)
    print("EVALUATION REPORT — PARD-SSM")
    print("=" * 60)
    print(f"  Regime Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Prediction MSE       : {mse:.6f}")
    print(f"  Log-Likelihood       : {log_likelihood:.2f}")
    print(f"  AUC-ROC (attack)     : {auc:.4f}")
    print(f"  Detection Lead Time  : {lead:.1f} steps")
    print()
    print("Per-Regime Classification Report:")
    print(classification_report(
        true_labels, predicted_labels,
        target_names=[regime_names.get(k, str(k)) for k in sorted(regime_names)],
        zero_division=0,
    ))
    print("=" * 60)

    return {
        "regime_accuracy": acc,
        "prediction_mse": mse,
        "log_likelihood": log_likelihood,
        "auc_roc": auc,
        "detection_lead_time": lead,
    }
