import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.utils.visualization import plot_regime_probabilities


# ==================== Kalman Filter ====================
class KalmanFilter:

    def __init__(self, A, C, Q, R, x0, P0):
        self.A  = A
        self.C  = C
        self.Q  = Q
        self.R  = R
        self.x  = x0.copy()
        self.P  = P0.copy()

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        S  = self.C @ self.P @ self.C.T + self.R
        S += np.eye(S.shape[0]) * 1e-6

        K      = self.P @ self.C.T @ np.linalg.inv(S)
        y_pred = self.C @ self.x
        diff   = y - y_pred

        self.x = self.x + K @ diff

        I_KC   = np.eye(self.P.shape[0]) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T

        sign, log_det_S = np.linalg.slogdet(S)
        log_det_S       = max(log_det_S, -1e6)
        dim             = y.shape[0]
        log_like        = (
            -0.5 * float(diff.T @ np.linalg.inv(S) @ diff)
            - 0.5 * log_det_S
            - 0.5 * dim * np.log(2 * np.pi)
        )
        log_like = np.clip(log_like, -500, 0)
        return np.exp(log_like)

    def reset(self, x0, P0):
        self.x = x0.copy()
        self.P = P0.copy()


# ==================== Switching State-Space Model ====================
class SwitchingStateSpaceModel:

    def __init__(self, filters, transition_matrix, prior=None):
        self.filters      = filters
        self.M            = len(filters)
        self.T            = transition_matrix
        self.regime_probs = prior if prior is not None else np.ones(self.M) / self.M

    def step(self, y):
        likelihoods = np.zeros(self.M)
        for i, kf in enumerate(self.filters):
            kf.predict()
            likelihoods[i] = kf.update(y)

        prior     = self.T.T @ self.regime_probs
        posterior = likelihoods * prior + 1e-8
        posterior = posterior / posterior.sum()

        self.regime_probs = posterior
        return posterior.copy()


# ==================== Helpers ====================
def causal_smooth(probs, alpha, seed=None):
    smoothed  = probs.copy().astype(float)
    start_row = seed if seed is not None else smoothed[0]
    smoothed[0] = alpha * smoothed[0] + (1 - alpha) * start_row
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def detect_early_detection_latency(y_true, attack_probs, threshold=0.5):
    """
    NeurIPS metric: how many steps BEFORE ground truth does
    the model detect the attack? Positive = early detection.
    """
    latencies    = []
    in_attack    = False
    attack_start = None

    for t in range(len(y_true)):
        if y_true[t] == 1 and not in_attack:
            in_attack    = True
            attack_start = t
        elif y_true[t] == 0 and in_attack:
            in_attack    = False
            attack_start = None

        if in_attack and attack_start is not None:
            search_start = max(0, attack_start - 50)
            for s in range(search_start, attack_start + 1):
                if attack_probs[s] >= threshold:
                    latencies.append(attack_start - s)
                    attack_start = None
                    break

    if latencies:
        return np.mean(latencies), np.std(latencies)
    return 0.0, 0.0


# ==================== MAIN ====================
def main():

    # ---------- Argument parsing — supports both datasets ----------
    parser = argparse.ArgumentParser(
        description="Switching SSM for Network Intrusion Detection"
    )
    parser.add_argument(
        "--data",
        default="data/processed/cicids2017_features.npy",
        help="Path to features .npy file",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/cicids2017_labels.npy",
        help="Path to labels .npy file",
    )
    parser.add_argument(
        "--dataset-name",
        default="CICIDS2017",
        help="Dataset name for plot titles and reports",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Maximum samples to use",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis (faster run)",
    )
    args = parser.parse_args()

    # ---------- Load ----------
    X = np.load(args.data)
    y = np.load(args.labels)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset_name}")
    print(f"{'='*60}")
    print("Total samples loaded:", X.shape[0])

    if X.shape[0] > args.max_samples:
        X = X[:args.max_samples]
        y = y[:args.max_samples]
        print(f"Using subset of {args.max_samples} samples")

    print("Final dataset size:", X.shape[0])

    true_regimes = (y != 0).astype(int)
    print(f"Attack ratio (full): {true_regimes.mean():.3f}")

    # ---------- Temporal split FIRST (no leakage) ----------
    n     = len(X)
    split = int(0.8 * n)
    print(f"\nData split — train: {split}, test: {n - split}")

    X_tr, X_te = X[:split],            X[split:]
    y_tr, y_te = true_regimes[:split],  true_regimes[split:]
    attack_ratio = y_tr.mean()
    print(f"Attack ratio in training set: {attack_ratio:.3f}")

    # ---------- Feature selection (fit on train only) ----------
    n_features = X_tr.shape[1]
    k_select   = min(20, n_features)
    print(f"SelectKBest k={k_select} (dataset has {n_features} features)")

    selector = SelectKBest(score_func=f_classif, k=k_select)
    selector.fit(X_tr, y_tr)

    rf_tr_raw = selector.transform(X_tr)
    rf_te_raw = selector.transform(X_te)

    rf_mean = rf_tr_raw.mean(axis=0)
    rf_std  = rf_tr_raw.std(axis=0) + 1e-6
    rf_tr   = (rf_tr_raw - rf_mean) / rf_std
    rf_te   = (rf_te_raw - rf_mean) / rf_std
    rf_dim  = rf_tr.shape[1]

    # ---------- SSSM input: top-8 features ----------
    k_sssm  = min(8, n_features)
    top_idx = np.argsort(selector.scores_)[::-1][:k_sssm]

    sssm_tr_raw = X_tr[:, top_idx]
    sssm_te_raw = X_te[:, top_idx]

    sssm_mean = sssm_tr_raw.mean(axis=0)
    sssm_std  = sssm_tr_raw.std(axis=0) + 1e-6
    sssm_tr   = (sssm_tr_raw - sssm_mean) / sssm_std
    sssm_te   = (sssm_te_raw - sssm_mean) / sssm_std

    dim = sssm_tr.shape[1]
    print(f"SSSM input dimension: {dim}")

    # ---------- Kalman filter setup ----------
    A1 = np.eye(dim) * 0.95
    Q1 = np.eye(dim) * 0.001

    A2 = np.eye(dim) * 1.5
    Q2 = np.eye(dim) * 0.5

    C  = np.eye(dim)
    R  = np.eye(dim) * 0.2
    x0 = np.zeros(dim)
    P0 = np.eye(dim)

    kf1 = KalmanFilter(A1, C, Q1, R, x0, P0)
    kf2 = KalmanFilter(A2, C, Q2, R, x0, P0)

    p_stay_normal = 0.90
    p_stay_attack = 0.85
    transition_matrix = np.array([
        [p_stay_normal,      1 - p_stay_normal],
        [1 - p_stay_attack,  p_stay_attack    ],
    ])

    prior = np.array([1 - attack_ratio, attack_ratio])
    model = SwitchingStateSpaceModel([kf1, kf2], transition_matrix, prior=prior)

    # ---------- SSSM — train window ----------
    sssm_probs_tr = []
    for obs in sssm_tr:
        sssm_probs_tr.append(model.step(obs))
    sssm_probs_tr = np.array(sssm_probs_tr)

    # ---------- SSSM — test window ----------
    sssm_probs_te = []
    for obs in sssm_te:
        sssm_probs_te.append(model.step(obs))
    sssm_probs_te = np.array(sssm_probs_te)

    # ---------- Causal smoothing ----------
    alpha          = 0.7
    smooth_tr      = causal_smooth(sssm_probs_tr, alpha)
    smooth_te      = causal_smooth(sssm_probs_te, alpha, seed=smooth_tr[-1])
    smooth_history = np.vstack([smooth_tr, smooth_te])

    # ---------- SSSM standalone accuracy ----------
    sssm_pred_tr = np.argmax(smooth_tr, axis=1)
    sssm_pred_te = np.argmax(smooth_te, axis=1)
    print(f"\nSSSM accuracy (train): {accuracy_score(y_tr, sssm_pred_tr):.4f}")
    print(f"SSSM accuracy (test):  {accuracy_score(y_te, sssm_pred_te):.4f}")

    # ---------- Early detection latency (NeurIPS metric) ----------
    attack_probs_te         = smooth_te[:, 1]
    lead_mean, lead_std     = detect_early_detection_latency(y_te, attack_probs_te)
    print(f"Early Detection Lead : {lead_mean:.2f} ± {lead_std:.2f} steps")

    # ---------- Temporal diff features ----------
    diff_tr = np.diff(rf_tr, axis=0, prepend=rf_tr[0:1])
    diff_te = np.diff(rf_te, axis=0, prepend=rf_te[0:1])

    # ---------- Assemble combined feature matrices ----------
    X_train_comb = np.hstack([rf_tr, diff_tr, smooth_tr])
    X_test_comb  = np.hstack([rf_te, diff_te, smooth_te])

    # ---------- Oversampling (train only) ----------
    rng      = np.random.default_rng(42)
    mask_min = y_tr == 1
    mask_maj = y_tr == 0
    X_min, y_min = X_train_comb[mask_min], y_tr[mask_min]
    X_maj, y_maj = X_train_comb[mask_maj], y_tr[mask_maj]

    idx_up           = rng.choice(len(X_min), size=len(X_maj), replace=True)
    X_train_balanced = np.vstack([X_maj, X_min[idx_up]])
    y_train_balanced = np.hstack([y_maj, y_min[idx_up]])

    # ---------- Random Forest ----------
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_balanced, y_train_balanced)

    # ---------- Optional SHAP ----------
    if not args.no_shap:
        try:
            from src.explainablity.shap_explainer import run_shap_analysis
            run_shap_analysis(clf, X_train_balanced, X_test_comb, rf_dim)
        except Exception as e:
            print(f"SHAP skipped: {e}")

    # ---------- Test evaluation ----------
    y_pred  = clf.predict(X_test_comb)
    y_probs = clf.predict_proba(X_test_comb)[:, 1]

    print(f"\n{'='*60}")
    print(f"HYBRID MODEL RESULTS — {args.dataset_name} (threshold 0.5)")
    print(f"{'='*60}")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=["Normal", "Attack"]))

    threshold = 0.35
    y_pred_th = (y_probs >= threshold).astype(int)
    print(f"HYBRID MODEL RESULTS — {args.dataset_name} (threshold {threshold}):")
    print("Accuracy:", accuracy_score(y_te, y_pred_th))
    print(classification_report(y_te, y_pred_th, target_names=["Normal", "Attack"]))

    fpr, tpr, _          = roc_curve(y_te, y_probs)
    roc_auc              = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_te, y_probs)
    pr_auc               = average_precision_score(y_te, y_probs)

    # ---------- NeurIPS summary ----------
    from sklearn.metrics import recall_score, precision_score, f1_score
    print(f"\n{'='*60}")
    print(f"NEURIPS RESULTS SUMMARY — {args.dataset_name}")
    print(f"{'='*60}")
    print(f"  Accuracy          : {accuracy_score(y_te, y_pred):.4f}")
    print(f"  Attack Recall     : {recall_score(y_te, y_pred):.4f}")
    print(f"  Attack Precision  : {precision_score(y_te, y_pred):.4f}")
    print(f"  F1-Score          : {f1_score(y_te, y_pred):.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  PR-AUC            : {pr_auc:.4f}")
    print(f"  Early Detection   : {lead_mean:.2f} ± {lead_std:.2f} steps")
    print(f"{'='*60}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    axes[0].plot([0,1],[0,1],"--")
    axes[0].set_title(f"ROC Curve — {args.dataset_name}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(); axes[0].grid()

    axes[1].plot(recall, precision, label=f"PR (AP={pr_auc:.4f})")
    axes[1].set_title(f"PR Curve — {args.dataset_name}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(); axes[1].grid()
    plt.tight_layout(); plt.show()

    cm   = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {args.dataset_name} Test Set")
    plt.tight_layout(); plt.show()

    X_full      = np.vstack([X_train_comb, X_test_comb])
    y_pred_full = clf.predict(X_full)
    print(f"\nFULL DATASET — {args.dataset_name}:")
    print("Accuracy:", accuracy_score(true_regimes, y_pred_full))
    print(classification_report(true_regimes, y_pred_full,
                                target_names=["Normal", "Attack"]))

    window = 300
    plt.figure(figsize=(12, 4))
    plt.plot(smooth_history[split:split+window, 1], label="Attack Prob")
    plt.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="Threshold")
    plt.title(f"Attack Regime Probability — {args.dataset_name}")
    plt.xlabel("Time Step"); plt.ylabel("Probability")
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    plot_regime_probabilities(smooth_history)


if __name__ == "__main__":
    main()