"""
statistical_significance.py
----------------------------
NeurIPS statistical significance testing.

Runs the full Switching SSM + RF pipeline N times with different
random seeds and reports:
  - Mean ± std for all metrics
  - 95% confidence intervals
  - Paired t-test vs best baseline (Isolation Forest)
  - Wilcoxon signed-rank test (non-parametric)

Run:
    python experiments/statistical_significance.py --dataset cicids2017 --runs 5
    python experiments/statistical_significance.py --dataset unswnb15 --runs 5
"""

import argparse
import os
import warnings
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")


# ==================== Shared Utilities ====================

def causal_smooth(probs, alpha=0.7, seed=None):
    s    = probs.copy().astype(float)
    s[0] = alpha * s[0] + (1 - alpha) * (seed if seed is not None else s[0])
    for i in range(1, len(s)):
        s[i] = alpha * s[i] + (1 - alpha) * s[i - 1]
    return s


def compute_early_detection(y_true, attack_probs, threshold=0.5, lookback=50):
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


def precise_early_detection(lead_mean, f1):
    """PED = Lead × F1 — rewards early detection only when precise."""
    return lead_mean * f1


# ==================== Kalman / SSSM ====================

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
        self.filters=filters; self.M=len(filters); self.T=T
        self.regime_probs = prior if prior is not None else np.ones(self.M)/self.M

    def step(self, y):
        liks = np.zeros(self.M)
        for i, kf in enumerate(self.filters):
            kf.predict(); liks[i] = kf.update(y)
        post = liks * (self.T.T @ self.regime_probs) + 1e-8
        self.regime_probs = post / post.sum()
        return self.regime_probs.copy()


# ==================== Single Run ====================

def run_one_seed(X_tr, X_te, y_tr, y_te, attack_ratio,
                 seed=42, threshold=0.5, method="ours"):
    """
    Run one method with one random seed.
    Returns dict of metrics for that run.
    """
    n_features = X_tr.shape[1]
    k_select   = min(20, n_features)
    k_sssm     = min(8,  n_features)

    # Feature selection — fixed (deterministic, no seed needed)
    selector = SelectKBest(f_classif, k=k_select).fit(X_tr, y_tr)
    rf_tr_r  = selector.transform(X_tr)
    rf_te_r  = selector.transform(X_te)
    mu = rf_tr_r.mean(0); sg = rf_tr_r.std(0) + 1e-6
    rf_tr = (rf_tr_r - mu) / sg
    rf_te = (rf_te_r - mu) / sg

    if method == "ours":
        # SSSM features
        top     = np.argsort(selector.scores_)[::-1][:k_sssm]
        s_tr_r  = X_tr[:, top]; s_te_r = X_te[:, top]
        sm = s_tr_r.mean(0); ss = s_tr_r.std(0) + 1e-6
        s_tr = (s_tr_r - sm) / ss; s_te = (s_te_r - sm) / ss

        dim = s_tr.shape[1]
        C=np.eye(dim); R=np.eye(dim)*0.2
        x0=np.zeros(dim); P0=np.eye(dim)
        kf1 = KalmanFilter(np.eye(dim)*0.95, C, np.eye(dim)*0.001, R, x0, P0)
        kf2 = KalmanFilter(np.eye(dim)*1.5,  C, np.eye(dim)*0.5,   R, x0, P0)
        T   = np.array([[0.90,0.10],[0.15,0.85]])
        mdl = SwitchingSSM([kf1,kf2], T,
                           np.array([1-attack_ratio, attack_ratio]))

        p_tr = causal_smooth(np.array([mdl.step(o) for o in s_tr]))
        p_te = causal_smooth(np.array([mdl.step(o) for o in s_te]),
                             seed=p_tr[-1])

        d_tr = np.diff(rf_tr, axis=0, prepend=rf_tr[0:1])
        d_te = np.diff(rf_te, axis=0, prepend=rf_te[0:1])
        X_comb_tr = np.hstack([rf_tr, d_tr, p_tr])
        X_comb_te = np.hstack([rf_te, d_te, p_te])
        attack_prob_te = p_te[:, 1]

    elif method == "iso":
        scaler   = StandardScaler().fit(rf_tr)
        Xn_s     = scaler.transform(rf_tr[y_tr==0])
        Xte_s    = scaler.transform(rf_te)
        iso      = IsolationForest(n_estimators=200, contamination=0.1,
                                   random_state=seed, n_jobs=-1)
        iso.fit(Xn_s)
        raw      = iso.score_samples(Xte_s)
        scores   = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        scores   = causal_smooth(scores.reshape(-1,1), alpha=0.7).squeeze()
        attack_prob_te = np.clip(scores, 0, 1)
        X_comb_tr = rf_tr; X_comb_te = rf_te

    # Oversample with this run's seed
    rng = np.random.default_rng(seed)
    mm  = y_tr == 1; mj = y_tr == 0
    idx = rng.choice(mm.sum(), size=mj.sum(), replace=True)
    Xb  = np.vstack([X_comb_tr[mj], X_comb_tr[mm][idx]])
    yb  = np.hstack([y_tr[mj],      y_tr[mm][idx]])

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    clf.fit(Xb, yb)

    if method == "iso":
        y_pred  = (attack_prob_te >= threshold).astype(int)
        y_probs = attack_prob_te
    else:
        y_probs = clf.predict_proba(X_comb_te)[:, 1]
        y_pred  = (y_probs >= threshold).astype(int)

    lead_m, lead_s = compute_early_detection(
        y_te, attack_prob_te if method == "ours" else y_probs, threshold
    )
    f1  = f1_score(y_te, y_pred, zero_division=0)
    ped = precise_early_detection(lead_m, f1)

    return {
        "accuracy"  : accuracy_score(y_te, y_pred),
        "recall"    : recall_score(y_te, y_pred,    zero_division=0),
        "precision" : precision_score(y_te, y_pred, zero_division=0),
        "f1"        : f1,
        "roc_auc"   : roc_auc_score(y_te, y_probs),
        "pr_auc"    : average_precision_score(y_te, y_probs),
        "lead_mean" : lead_m,
        "lead_std"  : lead_s,
        "ped"       : ped,
    }


def run_n_seeds(X_tr, X_te, y_tr, y_te, attack_ratio,
                n_runs=5, threshold=0.5, method="ours"):
    seeds   = [42, 123, 256, 789, 1337][:n_runs]
    results = []
    for i, seed in enumerate(seeds):
        print(f"    Seed {seed} ({i+1}/{n_runs})...", end=" ", flush=True)
        r = run_one_seed(X_tr, X_te, y_tr, y_te, attack_ratio,
                         seed=seed, threshold=threshold, method=method)
        results.append(r)
        print(f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}  "
              f"Lead={r['lead_mean']:+.1f}")
    return results


def aggregate(results):
    """Mean ± std + 95% CI for each metric."""
    metrics = ["accuracy","recall","precision","f1",
               "roc_auc","pr_auc","lead_mean","ped"]
    agg = {}
    for m in metrics:
        vals       = np.array([r[m] for r in results])
        mean       = vals.mean()
        std        = vals.std(ddof=1)
        n          = len(vals)
        se         = std / np.sqrt(n)
        ci95       = stats.t.ppf(0.975, df=n-1) * se
        agg[m]     = {"mean": mean, "std": std, "ci95": ci95, "vals": vals}
    return agg


def ci_str(agg, metric):
    m = agg[metric]
    return f"{m['mean']:.4f} ± {m['std']:.4f} (95% CI: ±{m['ci95']:.4f})"


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cicids2017","unswnb15"],
                        default="cicids2017")
    parser.add_argument("--runs",        type=int,   default=5)
    parser.add_argument("--max-samples", type=int,   default=50_000)
    parser.add_argument("--threshold",   type=float, default=0.5)
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

    print(f"\nStatistical Significance — {args.dataset.upper()}")
    print(f"Runs: {args.runs} | Train: {split} | Test: {n-split}")
    print(f"Attack ratio: {attack_ratio:.3f}\n")

    # ---------- Run both methods N times ----------
    print(f"[1/2] Running Switching SSM + RF (Ours) × {args.runs} seeds...")
    ours_runs = run_n_seeds(X_tr, X_te, y_tr, y_te, attack_ratio,
                            n_runs=args.runs, threshold=args.threshold,
                            method="ours")

    print(f"\n[2/2] Running Isolation Forest (Best Baseline) × {args.runs} seeds...")
    iso_runs  = run_n_seeds(X_tr, X_te, y_tr, y_te, attack_ratio,
                            n_runs=args.runs, threshold=args.threshold,
                            method="iso")

    # ---------- Aggregate ----------
    ours_agg = aggregate(ours_runs)
    iso_agg  = aggregate(iso_runs)

    # ---------- Statistical Tests ----------
    metrics_to_test = ["recall", "f1", "roc_auc", "pr_auc", "ped"]
    test_results    = {}

    for m in metrics_to_test:
        ours_vals = ours_agg[m]["vals"]
        iso_vals  = iso_agg[m]["vals"]

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(ours_vals, iso_vals)

        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(ours_vals - iso_vals)
        except Exception:
            w_stat, w_pval = float("nan"), float("nan")

        test_results[m] = {
            "t_stat"  : t_stat,
            "p_val"   : p_val,
            "w_stat"  : w_stat,
            "w_pval"  : w_pval,
            "sig"     : "***" if p_val < 0.001 else
                        "**"  if p_val < 0.01  else
                        "*"   if p_val < 0.05  else "ns",
        }

    # ---------- Print Full NeurIPS Table ----------
    print(f"\n{'='*75}")
    print(f"STATISTICAL SIGNIFICANCE RESULTS — {args.dataset.upper()}")
    print(f"({args.runs} independent runs, 95% confidence intervals)")
    print(f"{'='*75}")

    print(f"\n{'Metric':<14} {'Ours (mean±std)':<28} "
          f"{'IsoForest (mean±std)':<28} {'p-value':>10} {'Sig':>5}")
    print(f"{'-'*75}")

    for m in metrics_to_test:
        ours_str = f"{ours_agg[m]['mean']:.4f} ± {ours_agg[m]['std']:.4f}"
        iso_str  = f"{iso_agg[m]['mean']:.4f} ± {iso_agg[m]['std']:.4f}"
        p        = test_results[m]["p_val"]
        sig      = test_results[m]["sig"]
        p_str    = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        print(f"{m:<14} {ours_str:<28} {iso_str:<28} {p_str:>10} {sig:>5}")

    print(f"\n{'='*75}")
    print(f"Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print(f"Tests: paired t-test (parametric) + Wilcoxon signed-rank (non-parametric)")

    # ---------- 95% CI Table ----------
    print(f"\n{'='*75}")
    print(f"95% CONFIDENCE INTERVALS — OUR METHOD")
    print(f"{'='*75}")
    for m in ["recall", "f1", "roc_auc", "pr_auc", "lead_mean", "ped"]:
        print(f"  {m:<14}: {ci_str(ours_agg, m)}")

    # ---------- PED explanation ----------
    print(f"\n{'='*75}")
    print(f"PRECISE EARLY DETECTION (PED = Lead × F1)")
    print(f"{'='*75}")
    print(f"  Ours      : {ours_agg['ped']['mean']:.2f} ± {ours_agg['ped']['std']:.2f}")
    print(f"  IsoForest : {iso_agg['ped']['mean']:.2f} ± {iso_agg['ped']['std']:.2f}")
    print(f"  Ours advantage: +{ours_agg['ped']['mean'] - iso_agg['ped']['mean']:.2f} PED points")
    print(f"\n  Interpretation: PED rewards models that detect attacks early")
    print(f"  AND precisely. High lead time with low F1 scores poorly.")

    # ---------- Plots ----------
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    plot_metrics = [
        ("recall",    "Attack Recall"),
        ("f1",        "F1-Score"),
        ("roc_auc",   "ROC-AUC"),
        ("pr_auc",    "PR-AUC"),
        ("lead_mean", "Early Detection Lead (steps)"),
        ("ped",       "PED Score (Lead × F1)"),
    ]

    seeds_x = list(range(1, args.runs + 1))

    for idx, (metric, title) in enumerate(plot_metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        ours_vals = ours_agg[metric]["vals"]
        iso_vals  = iso_agg[metric]["vals"]

        ax.plot(seeds_x, ours_vals,  "o-", color="#C44E52",
                label="Ours",          linewidth=2, markersize=8)
        ax.plot(seeds_x, iso_vals,   "s--", color="#4C72B0",
                label="IsoForest",     linewidth=2, markersize=8)

        # Mean lines
        ax.axhline(ours_agg[metric]["mean"], color="#C44E52",
                   linestyle=":", alpha=0.6, linewidth=1.5)
        ax.axhline(iso_agg[metric]["mean"],  color="#4C72B0",
                   linestyle=":", alpha=0.6, linewidth=1.5)

        # CI shading
        om = ours_agg[metric]["mean"]; oci = ours_agg[metric]["ci95"]
        im = iso_agg[metric]["mean"];  ici = iso_agg[metric]["ci95"]
        ax.fill_between(seeds_x, om-oci, om+oci, alpha=0.15, color="#C44E52")
        ax.fill_between(seeds_x, im-ici, im+ici, alpha=0.15, color="#4C72B0")

        sig = test_results.get(metric, {}).get("sig", "")
        ax.set_title(f"{title}\n(p{test_results.get(metric,{}).get('p_val',1):.4f} {sig})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Run (seed)", fontsize=9)
        ax.set_xticks(seeds_x)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"Statistical Significance — {args.dataset.upper()}\n"
        f"Switching SSM+RF (Ours) vs Isolation Forest  "
        f"({args.runs} runs, shaded = 95% CI)",
        fontsize=13, fontweight="bold"
    )

    out_path = f"results/significance_{args.dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved → {out_path}")

    # ---------- Save raw results ----------
    import json
    summary = {
        "dataset"  : args.dataset,
        "n_runs"   : args.runs,
        "ours"     : {m: {"mean": float(ours_agg[m]["mean"]),
                          "std":  float(ours_agg[m]["std"]),
                          "ci95": float(ours_agg[m]["ci95"])}
                      for m in ours_agg},
        "iso"      : {m: {"mean": float(iso_agg[m]["mean"]),
                          "std":  float(iso_agg[m]["std"]),
                          "ci95": float(iso_agg[m]["ci95"])}
                      for m in iso_agg},
        "p_values" : {m: float(test_results[m]["p_val"])
                      for m in test_results},
    }
    json_path = f"results/significance_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Raw results saved → {json_path}")


if __name__ == "__main__":
    main()