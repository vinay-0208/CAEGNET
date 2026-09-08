# CAEG-Net Research Track — Phase 5: Shrinkage-Regularized Adaptive Routing (CAEG-Net SR) Design Specification

**Document Status**: COMPLETED & APPROVED FOR IMPLEMENTATION  
**Research Track**: Phase 5 Optimization & Validation  
**Corpus**: PJM Hourly Load Forecasting Benchmark  
**Reference Baselines**:
- Static Standalone Equal Ensemble: $\text{MAE} = 232.51 \pm 3.48\text{ MW}$ (Primary Benchmark)
- Decoupled CAEG-Net (D-CAEG): $\text{MAE} = 239.53 \pm 10.91\text{ MW}$
- CAEG-Net V2 Full: $\text{MAE} = 239.67 \pm 5.26\text{ MW}$
- Standalone Causal TCN: $\text{MAE} = 241.78 \pm 6.45\text{ MW}$

---

## 1. Problem Statement & Audit Findings

The Phase 7 investigation established a vital scientific result:
1. **Decoupled Training Solves Representation Collapse**: Training experts independently preserves their full predictive capability, matching joint V2 accuracy ($239.53\text{ MW}$ vs. $239.67\text{ MW}$) while reducing trainable parameters by $98.2\%$ (from $142,433$ down to $2,595$).
2. **The Unregularized Router Deviates Too Aggressively**: The standard D-CAEG router learned an average allocation of $7.4\%$ GRU, $36.4\%$ TCN, and $56.2\%$ Patch. While this allocation yielded a $-7.20\text{ MW}$ improvement over equal averaging on the *validation* set, it underperformed the **Static Standalone Equal Ensemble** ($232.51\text{ MW}$) on the *test* set by $+7.02\text{ MW}$.
3. **The Variance-Bias Trade-Off in Adaptive Routing**:
   The variance of an ensemble forecast error $e = \sum_i w_i e_i$ with pairwise covariance matrix $\Sigma$ is:
   $$\text{Var}(e) = w^T \Sigma w = \sum_{i=1}^3 w_i^2 \sigma_i^2 + 2 \sum_{i < j} w_i w_j \sigma_{ij}$$
   When errors are diverse ($\bar{r} \approx 0.74$), the uniform weight vector $w_0 = [1/3, 1/3, 1/3]^T$ minimizes the Euclidean norm $\|w\|_2^2 = \sum_i w_i^2 = 1/3$. Any unconstrained routing shift toward a single expert (such as $w = [0.07, 0.36, 0.57]$) sharply increases weight concentration to $\|w\|_2^2 = 0.460$ (a $+38\%$ increase in quadratic weight penalty).
   Unless the router's conditional selection gain exceeds this variance penalty on every origin, the unregularized router loses to simple equal averaging.

---

## 2. Core Research Hypothesis (Hypothesis SR)

> **Hypothesis SR (Shrinkage-Regularization)**:  
> *"Adaptive routing can improve upon equal averaging only if its deviations from equal weighting are sufficiently regularized or shrunk to prevent estimation variance from overwhelming conditional selection gains."*

By anchoring the router to the zero-complexity equal ensemble ($w_0 = [1/3, 1/3, 1/3]$) and penalizing deviations, we formulate a nested model family:
$$\text{Equal Ensemble } (\alpha=0, \lambda=0) \longrightarrow \text{Small Deviations} \longrightarrow \text{Moderate Deviations} \longrightarrow \text{Unconstrained Router } (\alpha=1, \lambda=0)$$

This creates a strictly fair, controlled test:
- If $\alpha = 0$, the model reproduces the $232.51\text{ MW}$ equal ensemble exactly.
- If a small, strictly positive $\alpha^* \in (0, 1)$ or regularizer $\lambda^* > 0$ yields lower validation MAE, the adaptive router successfully extracts conditional regime gains without sacrificing overall variance reduction.
- If $\alpha = 0$ remains optimal on validation data, the hypothesis that equal averaging is unbeatable in this regime is definitively confirmed.

---

## 3. Mathematical Architecture: CAEG-Net SR

### 3.1 Expert Backbones (Stage 1: Frozen)
Three independently trained and frozen neural experts:
1. **Expert 1**: Gated Recurrent Unit (`GatedRecurrentExpert`, $31,344$ params)
2. **Expert 2**: Multi-Scale Causal Dilated TCN (`MultiScaleCausalTCNExpert`, $44,870$ params)
3. **Expert 3**: PatchTemporal (`PatchTemporalExpert`, $63,624$ params)
All backbone parameters have `requires_grad = False` and execute in `eval()` mode.

### 3.2 Router Input $u_t \in \mathbb{R}^9$
- **6 Causal Context Features ($C_t$)**:
  1. Trend slope $\beta_1$ over last $168\text{h}$
  2. Volatility $\sigma_{\text{diff}}$ over last $168\text{h}$
  3. Recent $48\text{h}$ range ratio
  4. Diurnal lag-24 autocorrelation $r_{24}$
  5. Weekly lag-168 profile similarity
  6. Recent Ridge forecaster absolute error $e_{\text{ridge}}$
- **3 Detached Expert Disagreements ($D_t$)**:
  7. Mean pairwise disagreement: $\frac{1}{3}(\|y_1 - y_2\|_1 + \|y_1 - y_3\|_1 + \|y_2 - y_3\|_1)$
  8. Inter-expert standard deviation across $24\text{h}$ horizon
  9. Peak-to-trough range across candidate predictions

All inputs $u_t$ are strictly detached from autograd graph: zero gradient flows to experts.

### 3.3 Shrinkage Formulation
Let $w_0 = [1/3, 1/3, 1/3]^T$ be the uniform base prior.
The router MLP computes unconstrained logits $\delta_t = \text{RouterHead}(\text{ContextEncoder}(u_t)) \in \mathbb{R}^3$.

The shrinkage routing weights are defined as:
$$w_t = \text{softmax}\left( \log(w_0) + \frac{\alpha}{\tau} \cdot \delta_t \right)$$

**Key Properties**:
1. When $\alpha = 0$:  
   $$w_t = \text{softmax}(\log(w_0)) = \text{softmax}\left([-\log 3, -\log 3, -\log 3]^T\right) = [1/3, 1/3, 1/3]^T = w_0$$
   Guarantees exact identity with the $232.51\text{ MW}$ Static Equal Ensemble.
2. When $\alpha > 0$: $\alpha$ acts as a shrinkage hyperparameter that restricts the log-odds deviation of the router from the equal prior.
3. Temperature $\tau = 0.5$ preserves nominal calibration.

### 3.4 Routing Deviation Regularizer (KL Penalty)
In addition to the forecast loss, we define the deviation penalty:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{dev}} \cdot \mathcal{D}_{\text{KL}}(w_t \parallel w_0)$$

Where:
$$\mathcal{D}_{\text{KL}}(w_t \parallel w_0) = \sum_{i=1}^3 w_{t,i} \log\left( \frac{w_{t,i}}{1/3} \right) = \sum_{i=1}^3 w_{t,i} \log w_{t,i} + \log 3 = \log 3 - \mathcal{H}(w_t)$$

**Properties**:
- $\mathcal{D}_{\text{KL}}(w_t \parallel w_0) \ge 0$ for all valid probability vectors $w_t$.
- $\mathcal{D}_{\text{KL}}(w_t \parallel w_0) = 0$ if and only if $w_t = [1/3, 1/3, 1/3]$.
- $\mathcal{D}_{\text{KL}}(w_t \parallel w_0) = \log 3 \approx 1.0986$ under complete one-hot collapse.
- Convex, smooth, and analytically tractable.

---

## 4. Out-of-Fold (OOF) Prediction Protocol & Leakage Controls

### 4.1 The In-Sample Meta-Training Problem
When frozen experts generate predictions on their own training data, their errors and disagreements are systematically deflated compared to unseen test periods. This can cause the meta-router to over-index on in-sample expert accuracy.

### 4.2 Leakage-Free OOF Protocol
To ensure rigorous meta-training:
1. **Chronological 2-Fold Split on Train Data**:
   - Fold 1: First $50\%$ of training sequence ($t = 1 \dots 2966$)
   - Fold 2: Second $50\%$ of training sequence ($t = 2967 \dots 5933$)
2. **Causal Cross-Validation**:
   - Expert set $\mathcal{E}_1$ trained on Fold 1 generates predictions for Fold 2.
   - Expert set $\mathcal{E}_2$ trained on Fold 2 generates predictions for Fold 1.
3. **Meta-Training**:
   - The meta-router is trained on these stacked out-of-fold predictions.
4. **Validation Separation**:
   - Hyperparameter selection for $\alpha$ and $\lambda_{\text{dev}}$ is evaluated strictly on the **Validation Partition** ($1,294$ windows), which was never used for training either the experts or the router.
5. **Untouched Test Benchmark**:
   - Test data ($1,294$ origins, $53$ daily blocks) remains strictly sequestered until validation decisions are locked.

---

## 5. Candidate Hyperparameter Grids & Selection Protocol

All decisions are locked exclusively on the **Validation Partition** before touching the test set.

### 5.1 Shrinkage Grid ($\alpha$)
$$\alpha \in \{0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00\}$$

### 5.2 KL Regularizer Grid ($\lambda_{\text{dev}}$)
$$\lambda_{\text{dev}} \in \{0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10\}$$

### 5.3 Controlled Ablation Matrix
- **Experiment A**: Static Equal Ensemble ($\alpha = 0.0$, $\lambda_{\text{dev}} = 0.0$)
- **Experiment B**: D-CAEG Unconstrained Baseline ($\alpha = 1.0$, $\lambda_{\text{dev}} = 0.0$)
- **Experiment C**: Shrinkage Only ($\alpha = \alpha^*$, $\lambda_{\text{dev}} = 0.0$)
- **Experiment D**: KL Regularization Only ($\alpha = 1.0$, $\lambda_{\text{dev}} = \lambda^*$)
- **Experiment E**: Joint Shrinkage + KL Regularization ($\alpha = \alpha^*$, $\lambda_{\text{dev}} = \lambda^*$)

---

## 6. Success Criteria (C1 through C9)

| Criterion | Specification | Threshold / Standard |
| :--- | :--- | :--- |
| **C1** | Improve over Standalone Experts | $\text{MAE} < 241.78\text{ MW}$ (TCN) |
| **C2** | Improve over D-CAEG Baseline | $\text{MAE} < 239.53\text{ MW}$ |
| **C3** | Improve over Static Equal Ensemble | $\text{MAE} < 232.51\text{ MW}$ |
| **C4** | Statistical Significance | $p < 0.05$ on $53$ daily blocks (or clearly documented non-significance) |
| **C5** | 5-Seed Reproducibility | Verified across Seeds $42, 43, 44, 45, 46$ |
| **C6** | Zero Temporal Leakage | Strictly causal $192\text{h}$ buffer and sequential Ridge error |
| **C7** | Interpretable Routing | No expert collapse; weights stay well-calibrated |
| **C8** | Measurable Routing Deviation | Mean Euclidean distance $\|w_t - w_0\|_2$ quantified |
| **C9** | Parameter & Computational Efficiency | Trainable parameters $\le 2,595$; Stage 3 runtime $\le 30\text{s}$ per seed |

---

## 7. Implementation Roadmap

1. **Step 1**: Implement `ShrinkageRegularizedCAEG` in [`research/models.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/models.py).
2. **Step 2**: Implement `train_shrinkage_router` and evaluation functions in [`research/training.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/training.py).
3. **Step 3**: Implement unit tests in [`research/tests/test_phase5_shrinkage.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/tests/test_phase5_shrinkage.py).
4. **Step 4**: Implement experimental runner in [`research/experiments/run_phase5_shrinkage.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/experiments/run_phase5_shrinkage.py).
5. **Step 5**: Run smoke test and validation grid search.
6. **Step 6**: Lock optimal $(\alpha^*, \lambda_{\text{dev}}^*)$ based on validation MAE.
7. **Step 7**: Run full 5-seed test benchmark.
8. **Step 8**: Perform 53 daily block statistical hypothesis testing and regime analysis.
9. **Step 9**: Generate 6 publication plots and compile findings into [`PHASE_5_SHRINKAGE_RESULTS.md`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/PHASE_5_SHRINKAGE_RESULTS.md).
10. **Step 10**: Commit and push research milestone to `origin/research-track`.
