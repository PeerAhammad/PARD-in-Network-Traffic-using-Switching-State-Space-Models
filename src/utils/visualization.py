"""
visualization.py
----------------
Plotting utilities for regime detection results.

Generates:
- Regime probability timelines
- Hidden state trajectories
- Confusion matrices
- Attack detection timeline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional, List


# Regime color palette
REGIME_COLORS = {
    0: "#2ecc71",   # Normal — green
    1: "#f39c12",   # Scanning — orange
    2: "#e74c3c",   # Intrusion — red
    3: "#8e44ad",   # DoS — purple
    4: "#2c3e50",   # Exfiltration — dark
}

REGIME_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Intrusion",
    3: "DoS/DDoS",
    4: "Exfiltration",
}


def plot_regime_probabilities(
    regime_probs: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    viterbi_path: Optional[np.ndarray] = None,
    title: str = "Attack Regime Probabilities Over Time",
    save_path: Optional[str] = None,
):
    """
    Plot regime probabilities as stacked area chart.

    Parameters
    ----------
    regime_probs  : (T, K) — P(s_t = k | y_{1:t})
    true_labels   : (T,)   — ground truth (optional)
    viterbi_path  : (T,)   — Viterbi decoded regime (optional)
    title         : plot title
    save_path     : if given, save figure to this path
    """
    T, K = regime_probs.shape
    t = np.arange(T)

    n_rows = 1 + (true_labels is not None) + (viterbi_path is not None)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # --- Regime probability plot ---
    ax = axes[0]
    colors = [REGIME_COLORS.get(k, f"C{k}") for k in range(K)]
    labels = [REGIME_NAMES.get(k, f"Regime {k}") for k in range(K)]

    ax.stackplot(t, regime_probs.T, labels=labels, colors=colors, alpha=0.8)
    ax.set_ylabel("P(regime | observations)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=K, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    row = 1

    # --- Viterbi path ---
    if viterbi_path is not None:
        ax2 = axes[row]
        cmap = plt.cm.get_cmap("Set1", K)
        ax2.plot(t, viterbi_path, drawstyle="steps-post", color="navy", lw=1.5, label="Viterbi")
        ax2.set_ylabel("Decoded Regime", fontsize=11)
        ax2.set_yticks(range(K))
        ax2.set_yticklabels([REGIME_NAMES.get(k, f"R{k}") for k in range(K)], fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        row += 1

    # --- Ground truth ---
    if true_labels is not None:
        ax3 = axes[row]
        ax3.plot(t, true_labels, drawstyle="steps-post", color="darkgreen", lw=1.5, label="True Regime")
        ax3.set_ylabel("True Regime", fontsize=11)
        ax3.set_yticks(range(K))
        ax3.set_yticklabels([REGIME_NAMES.get(k, f"R{k}") for k in range(K)], fontsize=8)
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=9)

    axes[-1].set_xlabel("Time Step", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_hidden_state(
    filtered_means: np.ndarray,
    smoothed_means: Optional[np.ndarray] = None,
    dims_to_plot: int = 3,
    title: str = "Hidden State Trajectory",
    save_path: Optional[str] = None,
):
    """
    Plot the hidden state dimensions over time.

    Parameters
    ----------
    filtered_means  : (T, state_dim)
    smoothed_means  : (T, state_dim) — optional, from RTS smoother
    dims_to_plot    : how many state dimensions to show
    """
    T, d = filtered_means.shape
    dims = min(dims_to_plot, d)
    t = np.arange(T)

    fig, axes = plt.subplots(dims, 1, figsize=(14, 3 * dims), sharex=True)
    if dims == 1:
        axes = [axes]

    for i in range(dims):
        ax = axes[i]
        ax.plot(t, filtered_means[:, i], label="Filtered", color="steelblue", lw=1.5)
        if smoothed_means is not None:
            ax.plot(t, smoothed_means[:, i], label="Smoothed", color="tomato", lw=1.5, linestyle="--")
        ax.set_ylabel(f"x[{i}]", fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("Time Step", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    regime_names: Optional[dict] = None,
    title: str = "Regime Classification Confusion Matrix",
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix for regime classification.
    """
    from sklearn.metrics import confusion_matrix

    if regime_names is None:
        regime_names = REGIME_NAMES

    labels = sorted(np.unique(np.concatenate([true_labels, predicted_labels])))
    label_names = [regime_names.get(l, str(l)) for l in labels]

    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted Regime", fontsize=11)
    ax.set_ylabel("True Regime", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_log_likelihood_curve(
    log_likelihoods: List[float],
    title: str = "EM Training: Log-Likelihood",
    save_path: Optional[str] = None,
):
    """Plot EM training log-likelihood curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log_likelihoods, marker="o", color="steelblue", lw=2, markersize=5)
    ax.set_xlabel("EM Iteration", fontsize=11)
    ax.set_ylabel("Log-Likelihood", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_detection_timeline(
    regime_probs: np.ndarray,
    true_labels: np.ndarray,
    attack_regime_ids: List[int],
    threshold: float = 0.5,
    title: str = "Attack Detection Timeline",
    save_path: Optional[str] = None,
):
    """
    Visualize when the model detects an attack vs when it actually started.

    Colors background by true regime, overlays attack probability.
    """
    T, K = regime_probs.shape
    t = np.arange(T)

    # Combined attack probability = sum of all attack regime probs
    attack_prob = regime_probs[:, attack_regime_ids].sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # --- Top: attack probability ---
    ax1 = axes[0]
    ax1.fill_between(t, attack_prob, alpha=0.6, color="tomato", label="Attack Probability")
    ax1.axhline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold = {threshold}")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(Attack)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- Bottom: true regime background ---
    ax2 = axes[1]
    for k in range(K):
        mask = true_labels == k
        color = REGIME_COLORS.get(k, f"C{k}")
        ax2.fill_between(t, 0, 1, where=mask, color=color, alpha=0.5, label=REGIME_NAMES.get(k, f"R{k}"))
    ax2.set_ylabel("True Regime", fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc="upper right", ncol=K, fontsize=9)
    ax2.set_xlabel("Time Step", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating demo plots...")
    np.random.seed(42)
    T, K = 300, 3

    # Simulate regime probabilities with transitions
    probs = np.zeros((T, K))
    for t in range(T):
        if t < 100:
            probs[t] = [0.85, 0.1, 0.05]
        elif t < 200:
            probs[t] = [0.1, 0.7, 0.2]
        else:
            probs[t] = [0.05, 0.1, 0.85]
        probs[t] += np.random.dirichlet([5, 5, 5]) * 0.1
        probs[t] = np.clip(probs[t], 0, 1)
        probs[t] /= probs[t].sum()

    true_labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    viterbi = np.argmax(probs, axis=1)

    plot_regime_probabilities(probs, true_labels=true_labels, viterbi_path=viterbi)
    print("Demo plots done.")
