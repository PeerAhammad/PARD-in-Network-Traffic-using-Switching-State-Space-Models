import matplotlib
matplotlib.use("TkAgg")

import shap
import numpy as np
import matplotlib.pyplot as plt


def run_shap_analysis(model, X_train, X_test, rf_dim, sample_size=2000):
    print("\nRunning SHAP Analysis...")

    # ---------- Sample (new-style RNG — no FutureWarning) ----------
    rng = np.random.default_rng(42)

    def sample(arr):
        if len(arr) > sample_size:
            return arr[rng.choice(len(arr), sample_size, replace=False)]
        return arr

    X_train_s = sample(X_train)
    X_test_s  = sample(X_test)

    # ---------- Feature names (dynamic — no assumed formula) ----------
    total    = X_test_s.shape[1]
    n_sssm   = 2
    n_rf     = rf_dim
    n_temp   = total - n_rf - n_sssm   # remainder = temporal diff features

    if n_temp < 0:
        raise ValueError(
            f"Inferred n_temp={n_temp} is negative. "
            f"Check rf_dim={rf_dim} vs total columns={total}."
        )

    feature_names = (
        [f"RF_{i}"   for i in range(n_rf)]   +
        [f"TEMP_{i}" for i in range(n_temp)] +
        ["SSSM_0", "SSSM_1"]
    )

    # Hard guard — any mismatch surfaces immediately with a clear message
    assert len(feature_names) == total, (
        f"Feature name count ({len(feature_names)}) != actual columns ({total}). "
        f"rf_dim={rf_dim}, n_temp={n_temp}, n_sssm={n_sssm}."
    )

    # ---------- SHAP values ----------
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test_s)

    # Handle both new Explanation objects and legacy list format
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        sv = shap_values.values[:, :, 1]   # Attack class
    elif hasattr(shap_values, "values"):
        sv = shap_values.values
    else:
        sv = shap_values[1]

    # ---------- Group importance ----------
    rf_imp   = float(np.mean(np.abs(sv[:, :n_rf])))
    temp_imp = float(np.mean(np.abs(sv[:, n_rf: n_rf + n_temp])))
    sssm_imp = float(np.mean(np.abs(sv[:, n_rf + n_temp:])))

    print("\n--- SHAP Group Importance ---")
    print(f"RF Features:       {rf_imp:.4f}")
    print(f"Temporal Features: {temp_imp:.4f}")
    print(f"SSSM Features:     {sssm_imp:.4f}")

    # ---------- Summary (beeswarm) ----------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_test_s, feature_names=feature_names, show=False)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Summary — Feature Impact on Attack Detection", pad=20)
    plt.tight_layout()
    plt.show()

    # ---------- Bar (global importance) ----------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        sv, X_test_s, feature_names=feature_names, plot_type="bar", show=False
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Global)", pad=20)
    plt.tight_layout()
    plt.show()

    # ---------- Group bar ----------
    plt.close("all")
    labels = ["RF Features", "Temporal Features", "SSSM Features"]
    values = [rf_imp, temp_imp, sssm_imp]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#378ADD", "#1D9E75", "#D85A30"])
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(values) * 0.01,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=10
        )
    ax.set_title("Feature Group Contribution (SHAP-Based)", pad=15)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.show()
