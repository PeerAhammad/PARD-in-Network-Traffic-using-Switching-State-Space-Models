"""
baselines.py
------------
NeurIPS baseline comparison — HMM and LSTM vs Switching SSM.

Baselines:
  1. Hidden Markov Model (HMM)       — classical sequential regime detection
  2. LSTM Classifier                 — deep learning sequential baseline
  3. Isolation Forest                — unsupervised anomaly detection baseline
  4. Switching SSM (ours)            — loaded from saved results

All baselines report the SAME metrics:
  - Recall, Precision, F1, ROC-AUC, PR-AUC
  - Early Detection Lead Time (steps before ground truth)

Run:
    python experiments/baselines.py --dataset cicids2017
    python experiments/baselines.py --dataset unswnb15
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ==================== Shared Utilities ====================

def causal_smooth(probs, alpha=0.7, seed=None):
    s    = probs.copy().astype(float)
    s[0] = alpha * s[0] + (1 - alpha) * (seed if seed is not None else s[0])
    for i in range(1, len(s)):
        s[i] = alpha * s[i] + (1 - alpha) * s[i - 1]
    return s


def compute_early_detection(y_true, attack_probs, threshold=0.5, lookback=50):
    """
    Measures how many steps BEFORE attack onset the model fires.
    Positive = early detection. Negative = late detection.
    """
    episodes = []
    in_attack = False; start = None
    for t in range(len(y_true)):
        if y_true[t] == 1 and not in_attack:
            in_attack = True; start = t
        elif y_true[t] == 0 and in_attack:
            in_attack = False; episodes.append((start, t - 1))
    if in_attack:
        episodes.append((start, len(y_true) - 1))

    all_leads = []
    for ep_start, ep_end in episodes:
        search_start = max(0, ep_start - lookback)
        found = False
        for s in range(search_start, ep_start + 1):
            if attack_probs[s] >= threshold:
                all_leads.append(ep_start - s)
                found = True; break
        if not found:
            for s in range(ep_start, min(ep_end + 1, len(y_true))):
                if attack_probs[s] >= threshold:
                    all_leads.append(ep_start - s); break

    if all_leads:
        return float(np.mean(all_leads)), float(np.std(all_leads))
    return 0.0, 0.0


def make_result(name, y_true, y_pred, y_probs, lead_mean, lead_std):
    return {
        "name"      : name,
        "accuracy"  : accuracy_score(y_true, y_pred),
        "recall"    : recall_score(y_true, y_pred,    zero_division=0),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred,        zero_division=0),
        "roc_auc"   : roc_auc_score(y_true, y_probs),
        "pr_auc"    : average_precision_score(y_true, y_probs),
        "lead_mean" : lead_mean,
        "lead_std"  : lead_std,
    }


# ==================== Kalman / SSSM (Our Method) ====================

class KalmanFilter:
    def __init__(self, A, C, Q, R, x0, P0):
        self.A=A; self.C=C; self.Q=Q; self.R=R
        self.x=x0.copy(); self.P=P0.copy()

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        S  = self.C @ self.P @ self.C.T + self.R + np.eye(self.R.shape[0]) * 1e-6
        K  = self.P @ self.C.T @ np.linalg.inv(S)
        diff = y - self.C @ self.x
        self.x = self.x + K @ diff
        I_KC = np.eye(self.P.shape[0]) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T
        _, lds = np.linalg.slogdet(S)
        ll = (-0.5*float(diff.T@np.linalg.inv(S)@diff)
              - 0.5*max(lds,-1e6)
              - 0.5*y.shape[0]*np.log(2*np.pi))
        return np.exp(np.clip(ll, -500, 0))


class SwitchingSSM:
    def __init__(self, filters, T, prior=None):
        self.filters = filters; self.M = len(filters); self.T = T
        self.regime_probs = prior if prior is not None else np.ones(self.M)/self.M

    def step(self, y):
        liks = np.zeros(self.M)
        for i, kf in enumerate(self.filters):
            kf.predict(); liks[i] = kf.update(y)
        post = liks * (self.T.T @ self.regime_probs) + 1e-8
        self.regime_probs = post / post.sum()
        return self.regime_probs.copy()


def run_our_sssm(X_tr, X_te, y_tr, y_te, attack_ratio,
                 k_select=20, k_sssm=8, threshold=0.5):
    """Full Switching SSM + RF pipeline (our method)."""
    n_features = X_tr.shape[1]
    k_select   = min(k_select, n_features)
    k_sssm     = min(k_sssm,   n_features)

    selector = SelectKBest(f_classif, k=k_select).fit(X_tr, y_tr)
    rf_tr = (selector.transform(X_tr)); rf_te = selector.transform(X_te)
    mu = rf_tr.mean(0); sg = rf_tr.std(0) + 1e-6
    rf_tr = (rf_tr - mu) / sg; rf_te = (rf_te - mu) / sg

    top     = np.argsort(selector.scores_)[::-1][:k_sssm]
    s_tr    = X_tr[:, top]; s_te = X_te[:, top]
    sm = s_tr.mean(0); ss = s_tr.std(0) + 1e-6
    s_tr = (s_tr-sm)/ss; s_te = (s_te-sm)/ss

    dim = s_tr.shape[1]
    C=np.eye(dim); R=np.eye(dim)*0.2; x0=np.zeros(dim); P0=np.eye(dim)
    kf1 = KalmanFilter(np.eye(dim)*0.95, C, np.eye(dim)*0.001, R, x0, P0)
    kf2 = KalmanFilter(np.eye(dim)*1.5,  C, np.eye(dim)*0.5,   R, x0, P0)
    T   = np.array([[0.90,0.10],[0.15,0.85]])
    mdl = SwitchingSSM([kf1,kf2], T, np.array([1-attack_ratio, attack_ratio]))

    p_tr = causal_smooth(np.array([mdl.step(o) for o in s_tr]))
    p_te = causal_smooth(np.array([mdl.step(o) for o in s_te]), seed=p_tr[-1])

    d_tr = np.diff(rf_tr, axis=0, prepend=rf_tr[0:1])
    d_te = np.diff(rf_te, axis=0, prepend=rf_te[0:1])
    X_comb_tr = np.hstack([rf_tr, d_tr, p_tr])
    X_comb_te = np.hstack([rf_te, d_te, p_te])

    rng = np.random.default_rng(42)
    mm = y_tr==1; mj = y_tr==0
    idx = rng.choice(mm.sum(), size=mj.sum(), replace=True)
    Xb  = np.vstack([X_comb_tr[mj], X_comb_tr[mm][idx]])
    yb  = np.hstack([y_tr[mj],      y_tr[mm][idx]])

    clf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                  class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    clf.fit(Xb, yb)
    y_probs = clf.predict_proba(X_comb_te)[:, 1]
    y_pred  = (y_probs >= threshold).astype(int)

    lead_m, lead_s = compute_early_detection(y_te, p_te[:, 1], threshold)
    return make_result("Switching SSM + RF (Ours)", y_te, y_pred,
                       y_probs, lead_m, lead_s)


# ==================== Baseline 1: HMM ====================

def run_hmm(X_tr, X_te, y_tr, y_te, n_components=4, threshold=0.5):
    """
    Hidden Markov Model baseline using hmmlearn.
    Fits a Gaussian HMM on training data, uses posterior
    state probabilities as attack score.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  Installing hmmlearn...")
        os.system("pip install hmmlearn -q")
        from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler().fit(X_tr)
    Xtr_s  = scaler.transform(X_tr)
    Xte_s  = scaler.transform(X_te)

    # Fit separate HMMs for normal and attack
    mask_n = y_tr == 0; mask_a = y_tr == 1
    Xn = Xtr_s[mask_n]; Xa = Xtr_s[mask_a]

    hmm_n = GaussianHMM(n_components=2, covariance_type="diag",
                         n_iter=50, random_state=42)
    hmm_a = GaussianHMM(n_components=n_components, covariance_type="diag",
                         n_iter=50, random_state=42)

    hmm_n.fit(Xn)
    hmm_a.fit(Xa)

    # Score each test sample: P(attack) via log-likelihood ratio
    chunk   = 500   # score in chunks to avoid memory issues
    scores  = np.zeros(len(Xte_s))
    for i in range(0, len(Xte_s), chunk):
        chunk_data = Xte_s[i:i+chunk]
        try:
            ll_n = hmm_n.score_samples(chunk_data)[0]
            ll_a = hmm_a.score_samples(chunk_data)[0]
        except Exception:
            ll_n = np.zeros(len(chunk_data))
            ll_a = np.zeros(len(chunk_data))
        # Sigmoid of log-likelihood ratio → attack probability
        ratio = ll_a - ll_n
        scores[i:i+chunk] = 1 / (1 + np.exp(-ratio / (np.abs(ratio).max() + 1e-8)))

    # Smooth scores causally
    scores_smooth = causal_smooth(scores.reshape(-1, 1), alpha=0.7).squeeze()
    scores_smooth = np.clip(scores_smooth, 0, 1)

    y_pred = (scores_smooth >= threshold).astype(int)
    lead_m, lead_s = compute_early_detection(y_te, scores_smooth, threshold)
    return make_result("HMM (Gaussian)", y_te, y_pred,
                       scores_smooth, lead_m, lead_s)


# ==================== Baseline 2: LSTM ====================

def run_lstm(X_tr, X_te, y_tr, y_te,
             seq_len=20, hidden=64, epochs=10,
             threshold=0.5):
    """
    LSTM classifier baseline.
    Uses sliding windows of seq_len timesteps as input sequences.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not found — installing...")
        os.system("pip install torch --quiet")
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

    scaler  = StandardScaler().fit(X_tr)
    Xtr_s   = scaler.transform(X_tr).astype(np.float32)
    Xte_s   = scaler.transform(X_te).astype(np.float32)

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    Xtr_seq, ytr_seq = make_sequences(Xtr_s, y_tr, seq_len)
    Xte_seq, yte_seq = make_sequences(Xte_s, y_te, seq_len)

    # Oversample minority class in training
    rng    = np.random.default_rng(42)
    mm     = ytr_seq == 1; mj = ytr_seq == 0
    idx    = rng.choice(mm.sum(), size=mj.sum(), replace=True)
    Xb_seq = np.vstack([Xtr_seq[mj], Xtr_seq[mm][idx]])
    yb_seq = np.hstack([ytr_seq[mj], ytr_seq[mm][idx]])

    # Shuffle
    perm   = np.random.permutation(len(Xb_seq))
    Xb_seq = Xb_seq[perm]; yb_seq = yb_seq[perm]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  LSTM training on: {device}")

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                batch_first=True, dropout=0.3)
            self.fc   = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model = LSTMClassifier(Xtr_s.shape[1], hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Class weights for imbalanced data
    pos_weight = torch.tensor([(mj.sum() / (mm.sum() + 1e-8))]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Override forward for BCEWithLogitsLoss (remove sigmoid)
    class LSTMClassifierLogits(nn.Module):
        def __init__(self, input_dim, hidden, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                                batch_first=True, dropout=0.3)
            self.fc   = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(32, 1),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze()

    model = LSTMClassifierLogits(Xtr_s.shape[1], hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # DataLoader
    batch_size = 512
    Xt = torch.FloatTensor(Xb_seq).to(device)
    yt = torch.FloatTensor(yb_seq).to(device)
    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    # Inference
    model.eval()
    with torch.no_grad():
        Xte_t  = torch.FloatTensor(Xte_seq).to(device)
        logits = model(Xte_t).cpu().numpy()
        probs  = 1 / (1 + np.exp(-logits))   # sigmoid

    # Prepend zeros for the first seq_len steps (no sequence available)
    full_probs = np.concatenate([np.zeros(seq_len), probs])
    full_probs = causal_smooth(full_probs.reshape(-1,1), alpha=0.7).squeeze()
    full_probs = np.clip(full_probs, 0, 1)

    # Align with y_te
    full_probs = full_probs[:len(y_te)]
    y_pred     = (full_probs >= threshold).astype(int)

    lead_m, lead_s = compute_early_detection(y_te, full_probs, threshold)
    return make_result("LSTM Classifier", y_te, y_pred,
                       full_probs, lead_m, lead_s)


# ==================== Baseline 3: Isolation Forest ====================

def run_isolation_forest(X_tr, X_te, y_tr, y_te, threshold=0.5):
    """
    Isolation Forest — unsupervised anomaly detection baseline.
    Trained on normal traffic only (as in real deployment).
    """
    scaler = StandardScaler().fit(X_tr)
    Xtr_s  = scaler.transform(X_tr)
    Xte_s  = scaler.transform(X_te)

    # Train on normal only (unsupervised setting)
    X_normal = Xtr_s[y_tr == 0]
    iso = IsolationForest(n_estimators=200, contamination=0.1,
                          random_state=42, n_jobs=-1)
    iso.fit(X_normal)

    # score_samples: lower = more anomalous → negate for attack prob
    raw_scores  = iso.score_samples(Xte_s)
    # Normalize to [0,1]: higher = more likely attack
    scores      = 1 - (raw_scores - raw_scores.min()) / \
                      (raw_scores.max() - raw_scores.min() + 1e-8)
    scores      = causal_smooth(scores.reshape(-1,1), alpha=0.7).squeeze()
    scores      = np.clip(scores, 0, 1)

    y_pred      = (scores >= threshold).astype(int)
    lead_m, lead_s = compute_early_detection(y_te, scores, threshold)
    return make_result("Isolation Forest", y_te, y_pred,
                       scores, lead_m, lead_s)


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="NeurIPS Baseline Comparison")
    parser.add_argument("--dataset", choices=["cicids2017","unswnb15"],
                        default="cicids2017")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--lstm-epochs", type=int,   default=10)
    parser.add_argument("--skip-lstm",   action="store_true",
                        help="Skip LSTM (no PyTorch installed)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # ---------- Load ----------
    X = np.load(f"data/processed/{args.dataset}_features.npy")
    y = np.load(f"data/processed/{args.dataset}_labels.npy")
    if X.shape[0] > args.max_samples:
        X = X[:args.max_samples]; y = y[:args.max_samples]
    y = (y != 0).astype(int)

    n = len(X); split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    attack_ratio = y_tr.mean()

    print(f"\nBaseline Comparison — {args.dataset.upper()}")
    print(f"Train: {split} | Test: {n-split} | Attack ratio: {attack_ratio:.3f}\n")

    results = []

    # ---------- Run all baselines ----------
    print("="*50)
    print("[1/4] Running HMM baseline...")
    print("="*50)
    r_hmm = run_hmm(X_tr, X_te, y_tr, y_te, threshold=args.threshold)
    results.append(r_hmm)
    print(f"  ✓ Recall={r_hmm['recall']:.4f}  F1={r_hmm['f1']:.4f}  "
          f"Lead={r_hmm['lead_mean']:+.1f}±{r_hmm['lead_std']:.1f}")

    print("\n" + "="*50)
    print("[2/4] Running Isolation Forest baseline...")
    print("="*50)
    r_iso = run_isolation_forest(X_tr, X_te, y_tr, y_te, threshold=args.threshold)
    results.append(r_iso)
    print(f"  ✓ Recall={r_iso['recall']:.4f}  F1={r_iso['f1']:.4f}  "
          f"Lead={r_iso['lead_mean']:+.1f}±{r_iso['lead_std']:.1f}")

    if not args.skip_lstm:
        print("\n" + "="*50)
        print("[3/4] Running LSTM baseline...")
        print("="*50)
        r_lstm = run_lstm(X_tr, X_te, y_tr, y_te,
                          epochs=args.lstm_epochs, threshold=args.threshold)
        results.append(r_lstm)
        print(f"  ✓ Recall={r_lstm['recall']:.4f}  F1={r_lstm['f1']:.4f}  "
              f"Lead={r_lstm['lead_mean']:+.1f}±{r_lstm['lead_std']:.1f}")
    else:
        print("\n[3/4] LSTM skipped (--skip-lstm flag)")

    print("\n" + "="*50)
    print("[4/4] Running Our Switching SSM + RF...")
    print("="*50)
    r_ours = run_our_sssm(X_tr, X_te, y_tr, y_te, attack_ratio,
                           threshold=args.threshold)
    results.append(r_ours)
    print(f"  ✓ Recall={r_ours['recall']:.4f}  F1={r_ours['f1']:.4f}  "
          f"Lead={r_ours['lead_mean']:+.1f}±{r_ours['lead_std']:.1f}")

    # ---------- NeurIPS Comparison Table ----------
    print(f"\n{'='*90}")
    print(f"BASELINE COMPARISON — {args.dataset.upper()}")
    print(f"{'='*90}")
    print(f"{'Method':<30} {'Recall':>7} {'Prec':>7} {'F1':>7} "
          f"{'ROC-AUC':>9} {'PR-AUC':>8} {'Lead(steps)':>13}")
    print(f"{'-'*90}")
    for r in results:
        marker = " ◄ OURS" if "Ours" in r["name"] else ""
        lead_s = f"{r['lead_mean']:+.1f}±{r['lead_std']:.1f}"
        print(f"{r['name']:<30} {r['recall']:>7.4f} {r['precision']:>7.4f} "
              f"{r['f1']:>7.4f} {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f} "
              f"{lead_s:>13}{marker}")
    print(f"{'='*90}")

    # ---------- Delta vs best baseline ----------
    baselines_only = [r for r in results if "Ours" not in r["name"]]
    ours           = next(r for r in results if "Ours" in r["name"])
    best_bl        = max(baselines_only, key=lambda r: r["recall"])

    print(f"\nOUR METHOD vs BEST BASELINE ({best_bl['name']}):")
    print(f"  ΔRecall   : {ours['recall']  - best_bl['recall']:+.4f}")
    print(f"  ΔF1       : {ours['f1']      - best_bl['f1']:+.4f}")
    print(f"  ΔROC-AUC  : {ours['roc_auc'] - best_bl['roc_auc']:+.4f}")
    print(f"  ΔLead     : {ours['lead_mean']- best_bl['lead_mean']:+.2f} steps")

    # ---------- Plots ----------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    names  = [r["name"].replace(" (Ours)","*") for r in results]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"][:len(results)]
    ours_idx = next(i for i,r in enumerate(results) if "Ours" in r["name"])

    def bar_plot(ax, values, title, ylabel, ylim_pad=0.05):
        bars = ax.bar(range(len(results)), values, color=colors, alpha=0.85)
        bars[ours_idx].set_edgecolor("black"); bars[ours_idx].set_linewidth(2.5)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(min(values) - ylim_pad, 1.01)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    bar_plot(fig.add_subplot(gs[0,0]),
             [r["recall"]  for r in results], "Attack Recall",  "Recall")
    bar_plot(fig.add_subplot(gs[0,1]),
             [r["pr_auc"]  for r in results], "PR-AUC",         "PR-AUC")

    # Lead time plot (most important)
    ax3   = fig.add_subplot(gs[1,:])
    leads = [r["lead_mean"] for r in results]
    stds  = [r["lead_std"]  for r in results]
    bars3 = ax3.bar(range(len(results)), leads, color=colors, alpha=0.85,
                    yerr=stds, capsize=7, error_kw={"linewidth":2})
    bars3[ours_idx].set_edgecolor("black"); bars3[ours_idx].set_linewidth(2.5)

    ax3.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ymax = max(leads) + max(stds) + 5
    ymin = min(min(leads) - 3, -2)
    ax3.set_ylim(ymin, ymax)
    ax3.fill_between([-0.5, len(results)-0.5], 0, ymax,
                     alpha=0.06, color="green", label="Early detection zone")
    ax3.fill_between([-0.5, len(results)-0.5], ymin, 0,
                     alpha=0.06, color="red",   label="Reactive / late zone")
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(names, fontsize=10)
    ax3.set_ylabel("Lead Time (steps)\n+ = early  |  − = late", fontsize=11)
    ax3.set_title("Early Detection Lead Time — Key NeurIPS Contribution\n"
                  "* = Our method", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10); ax3.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars3, results):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 r["lead_mean"] + r["lead_std"] + 0.5,
                 f"{r['lead_mean']:+.1f}", ha="center",
                 va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(f"Baseline Comparison — {args.dataset.upper()}",
                 fontsize=15, fontweight="bold")
    out_path = f"results/baselines_{args.dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved → {out_path}")

    # ---------- Final NeurIPS summary ----------
    print(f"\n{'='*60}")
    print(f"NEURIPS TABLE READY — {args.dataset.upper()}")
    print(f"{'='*60}")
    for r in results:
        tag = " ← OUR METHOD" if "Ours" in r["name"] else ""
        print(f"  {r['name']:<30} "
              f"Recall={r['recall']:.4f}  "
              f"F1={r['f1']:.4f}  "
              f"AUC={r['roc_auc']:.4f}  "
              f"Lead={r['lead_mean']:+.1f}±{r['lead_std']:.1f}{tag}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()