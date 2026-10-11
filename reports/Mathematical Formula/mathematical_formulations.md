0.1 CyberGuard Architecture and Methodology
A. Path A: Supervised XGBoost Classifier
For an input network flow vector x ∈ Rd (d = 18), Path A utilizes gradient tree
boosting to classify known attack signatures across K = 6 classes. The categor-
ical probability distribution is computed via the Softmax activation function:
P (y = k | x) = exp (zk(x))∑︁
K−1
j=0 exp (zj (x)) (1)
The model minimizes a regularized multi-class cross-entropy objective function
using second-order Taylor expansion for optimal split decisions.
B. Path B: Deep Autoencoder Anomaly Detector
To detect unseen zero-day attacks, Path B employs a deep autoencoder trained
exclusively on benign traffic Dbenign. The network compresses the feature space
through an encoder E : Rd → Rdz (dz = 8) and reconstructs it via decoder D.
The flow-level anomaly score is defined by the Mean Squared Error (MSE):
LMSE(x, ˆx) = 1
d ∥x − ˆx∥2
2 (2)
The dynamic zero-day threshold τ is established using the upper boundary of
the benign reconstruction loss distribution:
τ = max (P98.5 (LMSE) , μbenign + 3σbenign) (3)
Flows exceeding this threshold (LMSE > τ ) are flagged as zero-day anomalies.
C. Hybrid Triage Fusion & Explainability (XAI)
The hybrid engine fuses both paths using a confidence-gated decision rule (θconf =
0.60). If Path A fails to identify a known signature with sufficient confidence, but
Path B flags an anomaly, the flow is categorized as a suspected zero-day. To en-
sure interpretability, local feature attributions ϕi(x) are generated using SHAP
(Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic
Explanations), providing security analysts with root-cause transparency