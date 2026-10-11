# 0.1 CyberGuard Architecture and Methodology

---

### A. Path A: Supervised XGBoost Classifier

For an input network flow vector **x** ∈ ℝᵈ (*d* = 18), Path A utilizes gradient tree boosting to classify known attack signatures across *K* = 6 classes. The categorical probability distribution is computed via the Softmax activation function:

```
                  exp(z_k(x))
P(y = k | x) = ───────────────────                  (1)
                K-1
                ∑  exp(z_j(x))
                j=0
```

> **LaTeX Formula:**
> $$P(y = k \mid \mathbf{x}) = \frac{\exp(z_k(\mathbf{x}))}{\sum_{j=0}^{K-1} \exp(z_j(\mathbf{x}))}$$

The model minimizes a regularized multi-class cross-entropy objective function using second-order Taylor expansion for optimal split decisions.

---

### B. Path B: Deep Autoencoder Anomaly Detector

To detect unseen zero-day attacks, Path B employs a deep autoencoder trained exclusively on benign traffic 𝒟_benign. The network compresses the feature space through an encoder *E* : ℝᵈ → ℝ^{d_z} (*d_z* = 8) and reconstructs it via decoder *D*. The flow-level anomaly score is defined by the Mean Squared Error (MSE):

```
                   1
ℒ_MSE(x, x̂) = ── ∥x - x̂∥₂²                         (2)
                   d
```

> **LaTeX Formula:**
> $$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d} \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$$

The dynamic zero-day threshold τ is established using the upper boundary of the benign reconstruction loss distribution:

```
τ = max( P_98.5(ℒ_MSE) ,  μ_benign + 3σ_benign )     (3)
```

> **LaTeX Formula:**
> $$\tau = \max\left(P_{98.5}(\mathcal{L}_{\text{MSE}}),\, \mu_{\text{benign}} + 3\sigma_{\text{benign}}\right)$$

Flows exceeding this threshold (ℒ_MSE > τ) are flagged as zero-day anomalies.

---

### C. Hybrid Triage Fusion & Explainability (XAI)

The hybrid engine fuses both paths using a confidence-gated decision rule (θ_conf = 0.60). If Path A fails to identify a known signature with sufficient confidence, but Path B flags an anomaly, the flow is categorized as a suspected zero-day. To ensure interpretability, local feature attributions ϕᵢ(x) are generated using SHAP (Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), providing security analysts with root-cause transparency.
