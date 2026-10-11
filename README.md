# 🛡️ CyberGuard Core

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Framework-XGBoost%20%7C%20TensorFlow-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Domain-Cybersecurity%20%26%20NIDS-red.svg" alt="Domain">
  <img src="https://img.shields.io/badge/Zero--Day-Detection%20Ready-brightgreen.svg" alt="Zero Day">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **CyberGuard** is a next-generation hybrid Network Intrusion Detection System (NIDS) designed to bridge the gap between high-confidence known attack classification and real-time detection of unseen zero-day exploits using Explainable AI (XAI).

---

## ⚡ Core Architecture: Dual-Path Hybrid Engine

Traditional intrusion detection systems suffer from a trade-off: supervised models fail against zero-day exploits, while unsupervised anomaly detectors suffer from high false positive rates. **CyberGuard** overcomes this by deploying a dual-path pipeline:

```
                          ┌────────────────────────┐
                          │  Network Flow Vector   │
                          │   x ∈ ℝ¹⁸ (d = 18)     │
                          └───────────┬────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      ┌─────────────────────┐                   ┌─────────────────────┐
      │       PATH A        │                   │       PATH B        │
      │  Supervised XGBoost │                   │  Deep Autoencoder   │
      │   (K = 6 Classes)   │                   │  (Trained on 𝒟_ben) │
      └──────────┬──────────┘                   └──────────┬──────────┘
                 │ P(y=k|x)                                │ ℒ_MSE(x, x̂)
                 ▼                                         ▼
      ┌───────────────────────────────────────────────────────────────┐
      │                  Confidence-Gated Fusion Engine               │
      │     If P(max) ≥ θ_conf  ➜ Known Attack Signature Flagged       │
      │     If P(max) < θ_conf & ℒ_MSE > τ ➜ Zero-Day Alert           │
      └───────────────────────────────┬───────────────────────────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │   XAI Root-Cause Layer   │
                        │      (SHAP & LIME)       │
                        └──────────────────────────┘
```

* For full mathematical equations, objective functions, and threshold derivations, see [📐 Mathematics Documentation](Mathematics/Mathematics.md).

---

## 📊 Performance & Benchmarks

The framework was evaluated on adversarial and zero-day simulated network flow traces:

| Metric | Path A (XGBoost) | Path B (Deep Autoencoder) | Hybrid Fusion |
| :--- | :---: | :---: | :---: |
| **Known Attack Detection** | **99.4%** | 88.2% | **99.6%** |
| **Zero-Day Discovery Rate** | 12.3% *(evaded)* | **96.8%** | **97.1%** |
| **False Positive Rate** | **0.3%** | 1.5% | **0.4%** |
| **Inference Latency** | < 1.2 ms | < 2.8 ms | < 3.5 ms |

---

## 🔬 Explainable AI (XAI)
CyberGuard integrates **SHAP (Shapley Additive exPlanations)** and **LIME** to provide Security Operations Center (SOC) analysts with root-cause transparency, highlighting which specific packet headers and traffic metrics triggered an alert.

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/parnabeginneruser/CyberGuard_Core.git
cd CyberGuard_Core
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Path A Evaluation
```bash
python eval_path_a.py
```

---

## 📂 Repository Structure
```
├── Mathematics/          # Mathematical formulations & LaTeX proofs
├── data/                 # Raw & synthetic benchmark flow traces
├── models/               # Pretrained XGBoost & Autoencoder models
├── reports/              # Figures, summaries, and briefing decks
├── src/                  # Core feature extraction & inference logic
├── requirements.txt      # Python dependencies
└── README.md             # Project overview & documentation
```
