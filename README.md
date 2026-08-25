<div align="center">

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=E74C3C&center=true&vCenter=true&width=700&lines=Probabilistic+Attack+Regime+Detection;in+Network+Traffic;Using+Switching+State-Space+Models" alt="Typing SVG" />
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Cybersecurity-E74C3C?style=for-the-badge&logo=shield&logoColor=white"/>
  <img src="https://img.shields.io/badge/ML-State--Space_Models-8E44AD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-F39C12?style=for-the-badge"/>
</p>

<p align="center">
  <b>Early detection of multi-stage cyber attacks in network telemetry<br>
  using probabilistic switching dynamical systems.</b>
</p>

<br>

![Regime Detection](docs/regime_detection.gif)

<sub><i>Live inference — the model continuously tracks attack stage probabilities as network telemetry arrives, detecting threats before they fully manifest.</i></sub>

</div>

---

## What is PARD-SSM?

Most intrusion detection systems fire an alert **after** an attack is obvious — when thresholds are breached or signatures match. By then, significant damage may already be done.

**PARD-SSM takes a fundamentally different approach.** It models network traffic as a *hidden dynamical system* — one where the true network condition (normal, scanning, intrusion, exfiltration) is a latent variable that must be inferred from noisy, high-dimensional observations over time.

By continuously estimating the **probability of each attack stage**, the system can detect threats **before they fully manifest** — tracking the subtle statistical fingerprints of reconnaissance, privilege escalation, and lateral movement long before threshold-based systems would trigger.

---

## Key Features

- **Probabilistic regime tracking** — outputs a full probability distribution over attack stages at every time step, not just a binary alert
- **Four inference engines** — Kalman Filter, EKF, UKF, and Variational Switching KF, from simple baselines to the full switching model
- **Early detection** — tracks reconnaissance and intrusion phases before exfiltration begins
- **EM learning** — transition probabilities and model parameters learned from data
- **Benchmark evaluated** — tested on CICIDS2017 and UNSW-NB15, the two most widely used intrusion detection benchmarks
- **Fully tested** — 20+ unit tests across models, inference, and data processing

---

## Attack Regime Modeling

Cyber attacks follow structured multi-stage progressions aligned with the **MITRE ATT&CK** framework:

| # | Regime | Observable Signals |
|---|--------|--------------------|
| 0 | 🟢 **Normal Operation** | Stable flows, known protocols, predictable rates |
| 1 | 🟡 **Reconnaissance** | Port sweeps, DNS enumeration, SYN floods |
| 2 | 🔴 **Intrusion Attempts** | Auth failures, unusual service access |
| 3 | 🟠 **Privilege Escalation** | Admin logins, policy changes, new accounts |
| 4 | 🔵 **Lateral Movement** | Unexpected internal connection spikes |
| 5 | 🟣 **Data Exfiltration** | Large outbound flows, unusual destinations |

Rather than detecting isolated events, PARD-SSM models **attack progression as regime transitions in a continuous-time stochastic process** — catching the full kill chain, not just its endpoint.

---

## Mathematical Formulation

The system is a **Switching State-Space Model (SSSM)** where both the continuous hidden network state and the discrete attack regime must be inferred simultaneously from observations.

### State Dynamics (Regime-Specific)

```
x_t = A_{s_t} · x_{t-1} + w_t        w_t ~ N(0, Q_{s_t})
```

Each attack regime `s_t` has its own transition matrix `A_{s_t}` — capturing how network state evolves differently during normal traffic vs. an active intrusion.

### Observation Model

```
y_t = C_{s_t} · x_t + v_t             v_t ~ N(0, R_{s_t})
```

The observed telemetry `y_t` (packet counts, flow stats, protocol ratios) is a noisy linear projection of the hidden state.

### Regime Transition (Markov Chain)

```
P(s_t = j | s_{t-1} = i) = π_{ij}
```

The transition matrix `Π` is **learned from data via EM** — capturing that attack stages persist over multiple time steps and escalate in structured patterns.

### Joint Inference Goal

```
P(x_t, s_t | y_{1:t})   →   regime probabilities + hidden state estimate
```

---

## Model Architecture

```
Raw Network Telemetry  (PCAP / NetFlow / CICIDS CSV)
              │
              ▼
    ┌─────────────────────┐
    │  Feature Extraction  │   packet stats · flow metadata · protocol ratios
    └──────────┬──────────┘
               │  Normalize → PCA → Sliding Window
               ▼
    ┌─────────────────────────────────────────────┐
    │          Switching State-Space Model         │
    │                                             │
    │   Hidden State   x_t  ∈ ℝᵈ                 │
    │   Attack Regime  s_t  ∈ {0, 1, 2, 3, 4, 5} │
    │   Transition     Π    (learned via EM)      │
    └──────────────────┬──────────────────────────┘
                       │
             Inference Engine
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
    Kalman Filter    EKF / UKF    Variational
    (linear SSM)   (nonlinear)   Switching KF
                                      │
                                      ▼
                       Regime Probabilities P(s_t | y_{1:t})
                                      │
                                      ▼
                         ⚠️  Early Attack Stage Detection
```

---

## Inference Methods

| Method | Model Type | Key Property | Complexity |
|--------|-----------|--------------|------------|
| **Kalman Filter (KF)** | Linear Gaussian | Exact posterior, fast baseline | Low |
| **Extended KF (EKF)** | Nonlinear | Local Jacobian linearization | Medium |
| **Unscented KF (UKF)** | Nonlinear | Sigma-point transform, no Jacobians needed | Medium-High |
| **Variational Switching KF** | Full SSSM | GPB2 approximation + Viterbi + EM learning | High |

All methods output:
- Filtered state estimate `x_{t|t}` and covariance `P_{t|t}`
- Regime probability vector `P(s_t | y_{1:t})`
- Viterbi-decoded most-likely attack path `s*_{1:T}`
- Log-likelihood for model comparison

---

## Repository Structure

```
PARD-SSM/
│
├── README.md
├── LICENSE                             ← Apache 2.0
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                            ← Downloaded datasets (not tracked by git)
│   └── processed/                      ← Preprocessed .npy arrays
│
├── src/
│   ├── models/
│   │   ├── linear_ssm.py               ← Linear Gaussian SSM (3 init strategies + simulate)
│   │   ├── nonlinear_ssm.py            ← Nonlinear SSM (TANH / SIGMOID_DECAY / LOG_RATIO)
│   │   └── switching_ssm.py            ← Full Switching SSM parameter container
│   │
│   ├── inference/
│   │   ├── kalman_filter.py            ← KF + RTS smoother
│   │   ├── ekf.py                      ← Extended Kalman Filter
│   │   ├── ukf.py                      ← Unscented Kalman Filter (sigma points)
│   │   └── variational_switching.py    ← Switching KF: GPB2 + Viterbi + EM
│   │
│   ├── data_processing/
│   │   ├── dataset_loader.py           ← CICIDS2017 + UNSW-NB15 loaders
│   │   └── feature_engineering.py     ← Normalize → PCA → Window pipeline
│   │
│   └── utils/
│       ├── visualization.py            ← 5 plot types: regime probs, timeline, confusion
│       └── metrics.py                  ← Accuracy, AUC-ROC, MSE, detection lead time
│
├── experiments/
│   ├── run_baseline.py                 ← KF / EKF / UKF side-by-side comparison
│   ├── run_switching.py                ← Full SSSM end-to-end experiment
│   └── evaluation_metrics.py          ← Unified comparison table across all methods
│
├── tests/
│   ├── test_kalman_filter.py           ← KF correctness, PD covariance, smoother
│   ├── test_switching_ssm.py           ← Regime prob sums, Viterbi validity, EM LL
│   └── test_data_processing.py        ← 14 tests: normalize, PCA, windows, save/load
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_kalman_filter_demo.ipynb
│   ├── 03_switching_ssm_demo.ipynb
│   └── 04_regime_detection_results.ipynb
│
└── docs/
    ├── regime_detection.gif            ← README animation (generated by make_gif.py)
    ├── make_gif.py                     ← Script to regenerate the animation
    ├── math_derivations.md             ← Full derivations: KF → RTS → GPB2 → Viterbi
    └── dataset_guide.md                ← Download instructions + synthetic data fallback
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/prakulhiremath/PARD-in-Network-Traffic-using-Switching-State-Space-Models-.git
cd PARD-in-Network-Traffic-using-Switching-State-Space-Models-

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Quick start — no dataset download needed

```python
# generate_synthetic.py
import numpy as np

X = np.vstack([
    np.random.randn(800, 10) * 0.5,           # Normal traffic
    np.random.randn(600, 10) * 2.0 + 3.0,    # Scanning / attack
    np.random.randn(600, 10) * 1.0 - 2.0,    # Exfiltration
])
y = np.array([0]*800 + [1]*600 + [2]*600)

np.save("data/processed/synthetic_features.npy", X)
np.save("data/processed/synthetic_labels.npy",   y)
```

```bash
python experiments/run_switching.py \
    --data   data/processed/synthetic_features.npy \
    --labels data/processed/synthetic_labels.npy   \
    --regimes 3
```

### Full pipeline on CICIDS2017

```bash
# Step 1 — Preprocess raw dataset
python src/data_processing/dataset_loader.py \
    --dataset cicids2017 \
    --input   data/raw/cicids2017/ \
    --output  data/processed/

# Step 2 — Run baseline filters (KF / EKF / UKF)
python experiments/run_baseline.py \
    --data   data/processed/cicids2017_features.npy \
    --labels data/processed/cicids2017_labels.npy

# Step 3 — Run full Switching SSM
python experiments/run_switching.py \
    --data     data/processed/cicids2017_features.npy \
    --labels   data/processed/cicids2017_labels.npy   \
    --regimes  4   \
    --em-iters 20

# Step 4 — Compare all methods
python experiments/evaluation_metrics.py
```

### Run all tests

```bash
python tests/test_kalman_filter.py
python tests/test_switching_ssm.py
python tests/test_data_processing.py
```

---

## Datasets

### CICIDS 2017
- **Source:** [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Size:** ~1.2 GB — 8 CSV files, one per day of a simulated work week
- **Attacks:** Brute Force, DoS/DDoS, Infiltration, Port Scan, Botnet, Web Attacks
- **Features:** 78 flow-level features extracted by CICFlowMeter

### UNSW-NB15
- **Source:** [UNSW Canberra Cyber](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Size:** ~100 MB — 4 CSV files, ~2.5 million records
- **Attacks:** Reconnaissance, Backdoors, DoS, Exploits, Shellcode, Worms — 9 categories
- **Features:** 49 features including packet stats, flow metadata, and protocol behavior

> See `docs/dataset_guide.md` for full download steps and preprocessing commands.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Regime Accuracy** | Classification accuracy of Viterbi-decoded path vs ground truth labels |
| **Prediction MSE** | Mean squared error of one-step-ahead telemetry prediction |
| **Log-Likelihood** | Model fit on held-out data — higher is better |
| **AUC-ROC** | Attack vs. normal discrimination (all attack regimes pooled as positive class) |
| **Detection Lead Time** | How many time steps early the model flags an incoming attack |
| **Confusion Matrix** | Per-regime precision/recall breakdown |

---

## Results

> Results will be populated as experiments complete on CICIDS2017 and UNSW-NB15.

| Method | Regime Accuracy | AUC-ROC | Detection Lead Time | Log-Likelihood |
|--------|:--------------:|:-------:|:-------------------:|:--------------:|
| Kalman Filter (baseline) | — | — | — | — |
| Extended KF | — | — | — | — |
| Unscented KF | — | — | — | — |
| **Variational Switching SSM** | — | — | — | — |

---

## Future Work — Phase II

- [ ] Real-time streaming inference for live network monitoring
- [ ] Integration with SIEM platforms (Splunk, Elastic Security)
- [ ] Deep Kalman Filter — neural network state-space hybrid models
- [ ] Unsupervised regime discovery — no attack labels required
- [ ] Interactive visualization dashboard (Plotly Dash / Streamlit)
- [ ] Deployment on enterprise-scale network telemetry

---

## Team

<div align="center">

| Name | USN | Contribution |
|------|-----|--------------|
| **Prakul Sunil Hiremath** | 2VX23CS013 | Lead — Inference engines & core models |
| **Peerahamad Bagawan** | 2VX23CS029 | Data processing & evaluation pipeline |
| **Sahil Bekane** | 2VX23CS042 | Experiments, testing & benchmarking |
| **Hemanth B. K.** | 2VX23CS012 | Visualization & documentation |

**Institution:** Visvesvaraya Technological University, Belagavi , Karnataka, India
**Programme:** B. Tech. Computer Science & Engineering — Major Project 2026–27

</div>

---

## Citation

If you use this work in your research, please cite our preprint:

```bibtex
@article{hiremath2026pard,
  title={PARD-SSM: Probabilistic Cyber-Attack Regime Detection via Variational Switching State-Space Models},
  author={Hiremath, Prakul Sunil and Bhekane, Sahil and Bagawan, PeerAhammad M},
  journal={arXiv preprint arXiv:2604.02299},
  year={2026}
}
```

---

## References

- Ghahramani, Z. & Hinton, G.E. (1996). *Switching State-Space Models.* University of Toronto.
- Murphy, K.P. (1998). *Switching Kalman Filters.* UC Berkeley Technical Report.
- Shumway, R.H. & Stoffer, D.S. *Time Series Analysis and Its Applications.* Springer.
- MITRE Corporation. *ATT&CK® Framework for Enterprise.* https://attack.mitre.org
- Sharafaldin, I. et al. (2018). *Toward Generating a New Intrusion Detection Dataset.* ICISSP.
- Moustafa, N. & Slay, J. (2015). *UNSW-NB15: A Comprehensive Dataset.* MilCIS.

---

## License

Copyright 2026 Prakul Sunil Hiremath, Peerahamad Bagawan, Sahil Bekane, Hemanth B. K.

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE) for full terms.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

<div align="center">
<sub>Built with 🔬 for early cyber-attack detection &nbsp;·&nbsp; VTU Karnataka &nbsp;·&nbsp; 2026</sub>
</div>
