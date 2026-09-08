# CAEG-Net Research Track — Phase 2: Formal Research Model Specification (Revised)

**Document Version:** 2.1.0-REVISED  
**Date:** September 8, 2026  
**Status:** SPECIFICATION REVISED — HUMAN REVIEW PASS INCORPORATED  
**Repository:** `vinay-0208/CAEGNET`  
**Target Execution Environment:** Python 3.11.15, PyTorch 2.11.0+cu128 (`.venv`), NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Primary Reference Baseline:** CAEG-Net V1 ($251.44 \pm 9.74\text{ MW}$)  

---

## 1. Research Objective

### Primary Research Question
> *"Can a causally designed, context-aware mixture of complementary temporal experts improve short-term electricity load forecasting across operating regimes and regional power grids compared with individual deep models, fixed ensembles, and conventional learned mixtures?"*

### Working Hypothesis
> *"Different temporal forecasting architectures (recurrent gating, dilated causal convolution, and patch-based sub-series tokenization) possess distinct and complementary inductive biases across different operational load dynamics. Explicitly representing causally observable temporal context (volatility, periodicity, trend) alongside inference-time forecast difficulty (inter-expert disagreement) and auxiliary baseline error, and using this representation to adaptively combine complementary experts, should improve accuracy and robustness, particularly during volatile or difficult forecasting regimes."*

### Falsification Standard
This hypothesis is **falsifiable**. If the proposed model:
1. Does not outperform a static learned-weight ensemble of the same three experts under non-overlapping block statistical tests ($p \ge 0.05$); or
2. Exhibits near-static routing weights across operational regimes despite temperature scaling and auxiliary loss; or
3. Degrades relative to the single best standalone expert across regional grid datasets;
then the hypothesis is **rejected**, and the paper will report that context-dependent dynamic routing did not provide a defensible advantage over simpler ensembling.

---

## 2. Immutable V1 Reference Baseline

CAEG-Net V1 serves as the frozen historical benchmark. It is never modified, retrained, or overwritten.

### Canonical Specifications
- **Input Dimension:** 1 (univariate electricity load)
- **Lookback Length ($L$):** 168 hours (7 full days of hourly history)
- **Forecast Horizon ($H$):** 24 hours (day-ahead hourly vector)
- **Experts:**
  - 2-layer LSTM ($h=64$, 56,152 params)
  - Dilated Causal TCN (6 blocks, dilations 1–32, 36,952 params)
  - 1D multi-stage CNN (3 stages + pooling, 27,400 params)
- **Context Vector ($C_t$):** 4 dimensions [Trend, Volatility, Periodicity, Recent Error]
- **Encoder:** Linear(4, 16) $\to$ LayerNorm $\to$ ReLU $\to$ Linear(16, 16) $\to$ ReLU (384 params)
- **Router:** Linear(16, 32) $\to$ ReLU $\to$ Dropout(0.1) $\to$ Linear(32, 3) $\to$ Softmax (643 params)
- **Total Parameters:** **121,531**
- **Canonical 5-Seed Evaluation:**
  $$\text{MAE} = 251.44 \pm 9.74\text{ MW},\quad \text{RMSE} = 334.32 \pm 11.09\text{ MW},\quad R^2 = 0.8723 \pm 0.0086,\quad \text{MAPE} = 4.71 \pm 0.21\%$$

---

## 3. Methodological Foundation (Phase 1 Audit & Human Review Integration)

The Phase 1 audit and subsequent human review establish eight governing methodological principles:
1. **Preserve Causal Data Pipeline:** The chronological 70/15/15 timeline split, train-only `StandardScaler` normalization, and partition-boundary historical prepending are mathematically sound, strictly causal, and 100% leakage-free. They are retained unaltered.
2. **Mitigate Gating Weight Collapse:** In V1, routing weights collapsed into a near-static blend ($\bar{w} \approx [0.405, 0.308, 0.288]$ with $\sigma < 0.0045$). Phase 2 introduces temperature scaling ($\tau \in (0, 1]$), entropy regularization, and auxiliary expert supervision to encourage routing differentiation.
3. **Replace the Deficient CNN Expert with a Research Candidate:** Standalone CNN in V1 achieved an MAE of $469.53\text{ MW}$ ($R^2 = 0.5452$), acting as dead weight. It is eliminated. A patch-based temporal linear model (`PatchTemporalExpert`) is introduced as the primary research candidate.
4. **Enforce Strict Capacity Parity Across Experts:** V1 allocated 46% of parameters to LSTM, 30% to TCN, and 23% to CNN. Phase 2 balances all three experts at $\sim 38,000$ parameters each ($\sim 32.6\%$ share per expert), eliminating capacity confounding.
5. **Classify Information Types Explicitly:** Distinguish strictly between:
   - *Type A: Observable Temporal Context* ($y \le t$)
   - *Type B: Post-Forecast Error Feedback* (past completed cycle residuals)
   - *Type C: Forecast Difficulty* (inference-time candidate expert disagreement)
6. **Correct Causal History Length for Weekly Seasonality:** Comparing the current 24-hour profile $[t-23:t]$ to the corresponding 24-hour profile one week earlier requires access to $[t-191:t-168]$. The context extractor receives a 192-hour causal historical buffer, while forecasting experts retain their 168-hour input window.
7. **Correct Entropy Regularization Formulation:** The regularizer is defined as $H(w) = -\sum_i w_i \ln(w_i + \epsilon)$ and added positively ($+\beta H(w)$) to the minimized loss to penalize high-entropy (uniform) distributions.
8. **Empirical Verification over Unsubstantiated Guarantees:** Temperature scaling and entropy regularization are recognized as mechanisms that *encourage* routing differentiation; whether they produce useful, non-collapsed dynamics remains an empirical research hypothesis.

---

## 4. Core Design Principles

1. **Strict Operational Causality:** Every context, difficulty, and routing feature available at forecast origin $t$ must depend strictly on historical observations $y \le t$. Zero access to future actuals $y > t$ is permitted.
2. **Convex Fusion Boundary:** Gating weights satisfy $w_{i,t} \ge 0$ and $\sum_{i=1}^3 w_{i,t} = 1.0$. The fused output $\hat{y}_t$ lies strictly within the convex hull of candidate predictions:
   $$\hat{y}_t = \sum_{i=1}^3 w_{i,t} \hat{y}_{i,t}$$
   Preventing unphysical, unbounded forecast divergence.
3. **Independent Expert Competence:** Each expert must be supervised such that it is encouraged to generate an accurate standalone 24h forecast.
4. **Inductive Complementarity:** The three experts must embody fundamentally distinct mathematical modeling paradigms: recurrent state tracking, dilated causal convolution, and patch-based sub-series tokenization.
5. **Inspectable & Interpretable Routing:** Gating weights must be logged per horizon step to enable descriptive correlation analysis against physical grid regimes.
6. **Controlled Complexity & Parameter Parity:** Target budget is approximately 120,000 parameters. The expected architecture comprises **116,569 parameters** ($-4.08\%$ vs V1's 121,531), ensuring lightweight execution on an RTX 4050 6 GB GPU.

---

## 5. Proposed Research Architecture: CAEG-Net V2

The proposed model comprises four decoupled, mathematically defined sub-systems:
```
                      Historical Load Sequence X_t in R^[168, 1]
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
      │   Expert 1    │       │   Expert 2    │       │   Expert 3    │
      │  Gated Rec.   │       │ Multi-Scale   │       │ Patch-Based   │
      │   (GRU-54)    │       │   (TCN-34)    │       │ (PatchLinear) │
      └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
              │                       │                       │
      ŷ_1 in R^24             ŷ_2 in R^24             ŷ_3 in R^24
              │                       │                       │
              ├───────────────────────┴───────────────────────┤
              │                                               │
              │ ──> [Inter-Expert Disagreement Module] ──> D_t in R^3 (detached)
              │                                               │
              │         Observable Context Vector C_t in R^6  │
              │           (from 192h causal history buffer)   │
              │                               │               │
              │                               ▼               │
              │                      [Context Concatenation]  │
              │                        u_t = [C_t || D_t]     │
              │                           in R^9              │
              │                               │               │
              │                               ▼               │
              │                    ┌─────────────────────┐    │
              │                    │   Context Encoder   │    │
              │                    │     (MLP + LN)      │    │
              │                    └──────────┬──────────┘    │
              │                               │ e_t in R^32   │
              │                               ▼               │
              │                    ┌─────────────────────┐    │
              │                    │   Adaptive Router   │    │
              │                    │  Softmax(z / tau)   │    │
              │                    └──────────┬──────────┘    │
              │                               │               │
              │                     Routing Weights w_t in R^3│
              │                               │               │
              ▼                               ▼               ▼
      ─────────────────────────────────────────────────────────
             DYNAMIC CONVEX FUSION: ŷ_t = w_1*ŷ_1 + w_2*ŷ_2 + w_3*ŷ_3
      ─────────────────────────────────────────────────────────
```
---

## 6. Expert Architecture Definitions

To ensure strict inductive complementarity and replace the deficient CNN architecture, three distinct architectural paradigms are selected:

### Expert 1: Gated Recurrent Unit Expert (`GatedRecurrentExpert`)
- **Inductive Bias:** Continuous chronological state tracking, autoregressive transition bias, sensitivity to gradual diurnal baseline drift and sequential autocorrelative memory.
- **Specification:**
  - Input: $X_t \in \mathbb{R}^{B \times 168 \times 1}$
  - Recurrent Backbone: 2-layer stacked GRU, `hidden_dim = 54`, `dropout = 0.1`, `batch_first = True`
  - Sequence Pooling: Final hidden state $h_L \in \mathbb{R}^{B \times 54}$
  - Projection Head: `Linear(54, 54)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(54, 24)`
  - Output: $\hat{y}_1 \in \mathbb{R}^{B \times 24}$
  - **Estimated Trainable Parameters:** **38,148**

### Expert 2: Multi-Scale Dilated Causal TCN Expert (`MultiScaleCausalTCNExpert`)
- **Inductive Bias:** Hierarchical multi-scale temporal motif extraction, strictly causal convolutional receptive field, fast parallelized feedforward execution, immunity to exploding/vanishing recurrent gradients.
- **Specification:**
  - Input: $X_t \in \mathbb{R}^{B \times 1 \times 168}$ (transposed)
  - Causal Conv Blocks: 5 residual blocks with kernel size $k=4$ and exponentially increasing dilations $d \in \{1, 2, 4, 8, 16\}$
  - Channels: $c = 34$ in all blocks
  - Block Topology: Conv1d(in, 34, k=4, dilation=d) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.1) $\to$ Conv1d(34, 34, k=4, dilation=d) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.1) $+$ Residual Shortcut
  - Causal Truncation: Right-padding of size $(k-1) \cdot d$ truncated on each conv layer.
  - Receptive Field:
    $$\text{RF} = 1 + 2 \sum_{i=0}^4 (4 - 1) \cdot 2^i = 1 + 6 \cdot (1 + 2 + 4 + 8 + 16) = 1 + 6 \cdot 31 = 187\text{ hours} > 168\text{ hours}$$
    The causal receptive field fully covers the entire 168-hour input sequence.
  - Projection Head: Last causal step $z_L \in \mathbb{R}^{B \times 34}$ $\to$ `Linear(34, 34)` $\to$ `ReLU` $\to$ `Dropout(0.1)` $\to$ `Linear(34, 24)`
  - Output: $\hat{y}_2 \in \mathbb{R}^{B \times 24}$
  - **Estimated Trainable Parameters:** **37,978**

### Expert 3: Patch-Based Temporal Linear Expert (`PatchTemporalExpert` — Research Candidate)
- **Inductive Bias:** Sub-series patch tokenization (Nie et al., ICLR 2023; Zeng et al., AAAI 2023). Divides the 168-hour history into localized 24-hour daily patches, extracting diurnal shape tokens, and mapping cross-patch temporal dependencies directly to the 24-hour horizon without step-by-step sequential decay.
- **Scientific Role & Hypothesis:**
  The Patch-based linear architecture is selected as a **research candidate** because:
  1. It provides an orthogonal inductive bias to recurrent transitions and dilated convolutions.
  2. It fits the capacity budget ($\sim 37.8\text{k}$ params) and preserves local diurnal shape.
  3. It offers a principled replacement for the weak V1 CNN.
  *Empirical Caveat:* Its standalone forecasting competence and degree of complementarity are treated as an **experimental hypothesis** to be verified in Phase 3. If Phase 3 benchmarking reveals that the Patch expert fails to provide independent competence or useful complementarity, the research protocol permits a scientifically justified replacement before final model freeze.
- **Specification:**
  - Input: $X_t \in \mathbb{R}^{B \times 168}$
  - Patching: Patch length $P_L = 24$ hours (1 diurnal cycle), Stride $S = 12$ hours (50% overlap).
  - Number of Patches: $P = \lfloor (168 - 24)/12 \rfloor + 1 = 13$ patches.
  - Patch Embedding: Linear projection of each 24-hour patch to $d_p = 48$ dimensions: `Linear(24, 48)` $\to$ `LayerNorm(48)` $\to$ `GELU`
  - Inter-Patch Dense Mixer: Flattened representation $\mathbb{R}^{B \times (13 \times 48)} = \mathbb{R}^{B \times 624}$ projected through bottleneck: `Linear(624, 96)` $\to$ `GELU` $\to$ `Dropout(0.1)` $\to$ `Linear(96, 24)`
  - Output: $\hat{y}_3 \in \mathbb{R}^{B \times 24}$
  - **Estimated Trainable Parameters:** **37,848**

---

## 7. Parameter Budget & Capacity Allocation

Strict parameter parity is enforced relative to V1 ($121,531$ params):

| Component | Architecture | Trainable Parameters | Budget Share (%) | Target Comparison vs V1 |
|:----------|:-------------|:--------------------:|:----------------:|:------------------------|
| **Expert 1 (Recurrent)** | GRU ($L=2, h=54$) + Linear Head | 38,148 | 32.73% | Balanced down from 56,152 (LSTM) |
| **Expert 2 (Causal Conv)** | Causal TCN (5 blocks, $c=34, k=4$) | 37,978 | 32.58% | Balanced from 36,952 (TCN) |
| **Expert 3 (Patch Linear)**| Patch Model ($P=13, P_L=24, d=48$) | 37,848 | 32.47% | Research candidate replacing weak CNN |
| **Context Encoder** | MLP: Linear(9, 32) + LN + Linear(32, 32) | 1,408 | 1.21% | Scaled up from 384 (V1) |
| **Adaptive Router** | Routing MLP: Linear(32, 32) + Linear(32, 3) | 1,187 | 1.02% | Scaled from 643 (V1) |
| **Disagreement Module** | Parameter-free tensor math (detached) | 0 | 0.00% | Zero parameter overhead |
| **TOTAL RESEARCH MODEL** | **Tri-Expert Balanced MoE** | **116,569** | **100.0%** | **Target Budget: ~120,000 ($-4.08\%$ vs V1)** |

**Capacity Control:** All three experts now possess virtually identical parameter counts ($\sim 38,000 \pm 150$ params, $32.6\% \pm 0.15\%$ share). Capacity confounding is completely eliminated.

---

## 8. Context Representation & Causal History Buffers

### Dual-Window History Protocol
To resolve the weekly seasonality lookback requirement without expanding expert computational cost:
1. **Forecasting Expert Input ($X_t \in \mathbb{R}^{168 \times 1}$):** Observations $[y_{t-167}, \dots, y_t]$ (168 hours / 7 days).
2. **Context Extractor Buffer ($X_{\text{ctx},t} \in \mathbb{R}^{192}$):** Observations $[y_{t-191}, \dots, y_t]$ (192 hours / 8 days).
   *Causal Availability:* Because all timestamps in $[t-191:t]$ occurred strictly at or before forecast origin $t$, this buffer introduces **zero future target leakage**. In real transmission grid operations, historical load records extending 8+ days into the past are fully observable.

### Context Features (6 Dimensions in $C_t$)
#### Group A: Observable Local Temporal Dynamics (3 features)
1. **Trend Slope ($\beta_{1,t} \in \mathbb{R}$):**
   Normalized Ordinary Least Squares (OLS) linear regression slope across the 168-hour lookback:
   $$\bar{i} = \frac{167}{2} = 83.5,\quad \beta_{1,t} = \frac{\sum_{i=0}^{167} (i - \bar{i})(z_i - \bar{z})}{\sum_{i=0}^{167} (i - \bar{i})^2},\quad \text{Trend}_t = \frac{\beta_{1,t}}{\sigma(z) + 10^{-6}}$$
2. **Short-Term Volatility ($v_t \in \mathbb{R}^+$):**
   Sample standard deviation of first differences of the standardized lookback series:
   $$\Delta z_i = z_i - z_{i-1},\quad v_t = \sqrt{\frac{1}{166} \sum_{i=1}^{167} (\Delta z_i - \bar{\Delta z})^2}$$
3. **Recent Demand Range Ratio ($R_{48,t} \in \mathbb{R}^+$):**
   The peak-to-trough range over the most recent 48 hours relative to standard deviation:
   $$R_{48,t} = \frac{\max(z_{t-47:t}) - \min(z_{t-47:t})}{\sigma(z_{t-47:t}) + 10^{-6}}$$

#### Group B: Observable Seasonality (2 features)
4. **Diurnal Rhythmicity ($r_{24,t} \in [-1, 1]$):**
   Lag-24 sample autocorrelation across the 168-hour lookback:
   $$r_{24,t} = \frac{\sum_{i=24}^{167} (z_i - \bar{z})(z_{i-24} - \bar{z})}{\sum_{i=0}^{167} (z_i - \bar{z})^2 + 10^{-6}}$$
5. **Weekly Profile Similarity ($r_{168,t} \in [-1, 1]$):**
   Correlation between the most recent 24-hour cycle $[z_{t-23:t}]$ and the corresponding 24-hour cycle exactly one week prior $[z_{t-191:t-168}]$ from the 192-hour context buffer:
   $$r_{168,t} = \text{Corr}(z_{t-23:t}, z_{t-191:t-168})$$
   *Note on Exact Indexing:* $t - 168$ is the exact corresponding hour one week prior. The 24-hour cycle ending at $t - 168$ spans from $t - 168 - 23 = t - 191$ to $t - 168$. Access through $t-191$ is mathematically required and causally sound.

#### Group C: Post-Forecast Error Feedback (1 feature)
6. **Recent Linear Baseline Error ($E_{\text{base},t} \in \mathbb{R}^+$):**
   Out-of-sample MAE of an expanding-window multi-step Ridge regression model evaluating the 24-hour cycle completed at origin $t$:
   $$E_{\text{base},t} = \frac{1}{24} \sum_{h=1}^{24} |\hat{y}_{\text{Ridge}}[t - 24 + h] - y[t - 24 + h]|$$
   *Causality:* Evaluates predictions issued at $t-24$ against observations $y \le t$. Fully observable at time $t$.

---

## 9. Tripartite Information Taxonomy

To maintain scientific rigor and operational clarity, information available at forecast origin $t$ is partitioned into three distinct classes:
1. **Observable Temporal Context ($C_{\text{obs},t} \in \mathbb{R}^5$):** Features computed directly from past load observations $y \le t$ (Trend, Volatility, Range, Diurnal periodicity, Weekly profile similarity).
2. **Post-Forecast Error Feedback ($E_{\text{base},t} \in \mathbb{R}^1$):** Prediction residuals from previously completed forecast cycles evaluated against historical ground truth.
3. **Forecast Difficulty ($D_t \in \mathbb{R}^3$):** Inference-time consensus metrics reflecting disagreement among candidate expert forecasts generated at origin $t$. Does not require future ground-truth targets.

*Total Observable Context:* $C_t = [C_{\text{obs},t} \parallel E_{\text{base},t}] \in \mathbb{R}^6$.

---

## 10. Forecast-Aware Routing & Inter-Expert Disagreement

### Mathematical Definition of Forecast-Aware Routing
The architecture is designated **forecast-aware** because the router observes candidate expert forecasts before determining final fusion weights:
$$\hat{y}_{i,t} = f_i(X_t) \quad \text{for } i \in \{1, 2, 3\}$$
$$D_t = \text{detach}\left( g(\hat{y}_{1,t}, \hat{y}_{2,t}, \hat{y}_{3,t}) \right) \in \mathbb{R}^3$$
$$u_t = [C_t \parallel D_t] \in \mathbb{R}^9$$
$$w_t = \text{Router}(u_t) \in \mathbb{R}^3$$
$$\hat{y}_t = \sum_{i=1}^3 w_{i,t} \hat{y}_{i,t}$$

### Disagreement Metric Formulations ($D_t \in \mathbb{R}^3$)
1. **Mean Pairwise Discrepancy ($d_{\text{pair},t} \in \mathbb{R}^+$):**
   $$d_{\text{pair},t} = \frac{1}{24} \sum_{h=1}^{24} \frac{|\hat{y}_{1,t,h} - \hat{y}_{2,t,h}| + |\hat{y}_{1,t,h} - \hat{y}_{3,t,h}| + |\hat{y}_{2,t,h} - \hat{y}_{3,t,h}|}{3}$$
2. **Inter-Expert Standard Deviation ($d_{\text{std},t} \in \mathbb{R}^+$):**
   $$d_{\text{std},t} = \frac{1}{24} \sum_{h=1}^{24} \sqrt{\frac{1}{3} \sum_{i=1}^3 (\hat{y}_{i,t,h} - \bar{\hat{y}}_{t,h})^2},\quad \text{where } \bar{\hat{y}}_{t,h} = \frac{1}{3} \sum_{i=1}^3 \hat{y}_{i,t,h}$$
3. **Inter-Expert Horizon Range ($d_{\text{range},t} \in \mathbb{R}^+$):**
   $$d_{\text{range},t} = \frac{1}{24} \sum_{h=1}^{24} \left( \max_{i} \hat{y}_{i,t,h} - \min_{i} \hat{y}_{i,t,h} \right)$$

### Causality and Detachment Guarantees
- **Causality:** Every candidate forecast $\hat{y}_i$ is computed exclusively from historical window $X_t$. Zero future ground truth is consulted.
- **Gradient Detachment:** $D_t$ is explicitly detached from the PyTorch autograd graph:
  $$\frac{\partial D_t}{\partial \theta_{\text{expert}_i}} = 0$$
  This prevents experts from modifying their predictions to artificially manipulate routing weights through disagreement gradients.
---

## 11. Routing Network Mathematics & Temperature Scaling

### Input Formulation
The complete context representation fed into the router is the concatenated 9-dimensional vector:
$$u_t = [C_t \parallel D_t] \in \mathbb{R}^9$$
where $C_t \in \mathbb{R}^6$ represents physical grid context and baseline error, and $D_t \in \mathbb{R}^3$ represents detached inter-expert disagreement.

### Latent Context Encoding
$$e_t = \text{ReLU}\left( \text{LayerNorm}\left( W_{e1} u_t + b_{e1} \right) \right)$$
$$e_t = \text{ReLU}\left( W_{e2} e_t + b_{e2} \right) \in \mathbb{R}^{32}$$
where $W_{e1} \in \mathbb{R}^{32 \times 9}, W_{e2} \in \mathbb{R}^{32 \times 32}$.

### Routing Logits & Temperature-Scaled Softmax
$$z_t = W_{r2} \left( \text{Dropout}_{0.1}\left( \text{ReLU}\left( W_{r1} e_t + b_{r1} \right) \right) \right) + b_{r2} \in \mathbb{R}^3$$
The convex routing weights $w_t = [w_{1,t}, w_{2,t}, w_{3,t}] \in \mathbb{R}^3$ are computed via temperature-scaled Softmax:
$$w_{i,t} = \frac{\exp(z_{i,t} / \tau)}{\sum_{j=1}^3 \exp(z_{j,t} / \tau)}$$

### Scientific Clarification on Temperature Scaling
- **Controlled Concentration Mechanism:** Temperature scaling is introduced as a controlled mathematical mechanism for adjusting routing concentration. Lower values of $\tau$ increase the sharpness of the Softmax distribution for a given logit spread.
- **Hypothesis Status (No Guaranteed De-Collapse):** Temperature scaling does **not** guarantee that routing weights vary across samples or regimes. It only sharpens the distribution. Whether temperature scaling produces meaningful, non-collapsed dynamic specialization must be evaluated empirically in Phase 3.
- **Specification Baseline vs Tuning Protocol:**
  - *Proposed Specification Value:* $\tau = 0.5$ (initial baseline).
  - *Validation Tuning Protocol:* If $\tau$ is tuned, candidates $\tau \in \{0.2, 0.5, 0.8, 1.0\}$ must be selected strictly by minimizing validation MSE on the validation partition. **The test set must never be accessed to select or tune $\tau$.**

### Global vs. Horizon-Aware Routing Specification
- **Primary Model (Global Routing):** A single weight vector $w_t \in \mathbb{R}^3$ applies to all 24 horizon steps: $\hat{y}_t = \sum_{i=1}^3 w_{i,t} \hat{y}_{i,t}$.
- **Controlled Ablation (Horizon-Aware Routing):** Logits projected to $\mathbb{R}^{24 \times 3}$, producing step-specific convex weights $w_{t,h} \in \mathbb{R}^3$ for $h \in \{1, \dots, 24\}$. Evaluated as ablation A10 to determine if multi-step routing justifies its additional variance.

---

## 12. Routing Collapse Diagnostics & Entropy Regularization

### Scientific Diagnostics for Routing Behavior
Rather than imposing an arbitrary variance threshold, routing behavior will be reported descriptively and analyzed scientifically using seven quantitative diagnostics:
1. **Per-Expert Mean Weight:** $\bar{w}_i = \frac{1}{N} \sum_{t=1}^N w_{i,t}$.
2. **Per-Expert Weight Standard Deviation:** $\sigma(w_i) = \sqrt{\frac{1}{N} \sum_{t=1}^N (w_{i,t} - \bar{w}_i)^2}$.
3. **Mean Routing Entropy:**
   $$H(w) = \frac{1}{N} \sum_{t=1}^N \left( -\sum_{i=1}^3 w_{i,t} \ln(w_{i,t} + 10^{-8}) \right)$$
   Maximum entropy (uniform distribution, $w_i = 1/3$) is $\ln(3) \approx 1.0986$. Lower values indicate more decisive expert allocation.
4. **Effective Number of Experts ($N_{\text{eff}}$):**
   $$N_{\text{eff},t} = \exp\left( -\sum_{i=1}^3 w_{i,t} \ln(w_{i,t} + 10^{-8}) \right) \in [1.0, 3.0]$$
5. **Observed Weight Range:** $(\max_t w_{i,t} - \min_t w_{i,t})$ for each expert.
6. **Regime-Conditioned Weight Variation:** Shift in mean weights across Low, Medium, and High volatility/disagreement tertiles: $|\bar{w}_{i,\text{High}} - \bar{w}_{i,\text{Low}}|$.
7. **Routing-Loss Alignment:** Pearson correlation between $w_{i,t}$ and the relative error advantage of expert $i$ over its peers.

### Corrected Entropy Regularization Formulation
To encourage the router toward more decisive expert allocations, an entropy penalty is added to the minimized objective:
$$H(w_t) = -\sum_{i=1}^3 w_{i,t} \ln(w_{i,t} + 10^{-8})$$
$$\mathcal{L}_{\text{entropy}} = \frac{1}{N} \sum_{t=1}^N H(w_t)$$
Because high entropy corresponds to diffuse/uniform distributions and low entropy corresponds to concentrated distributions, adding $+\beta H(w)$ (with $\beta > 0$) to a minimized loss function penalizes uniform collapse.

*Scientific Caveat:* This regularizer encourages lower routing entropy / more concentrated expert weighting. It does **not** guarantee that the concentration aligns with optimal forecasting or dynamic regime tracking; that alignment must be demonstrated empirically.

---

## 13. Fusion Mathematics

The fused prediction $\hat{y}_t \in \mathbb{R}^{24}$ is computed strictly via convex combination:
$$\hat{y}_{t,h} = w_{1,t} \hat{y}_{1,t,h} + w_{2,t} \hat{y}_{2,t,h} + w_{3,t} \hat{y}_{3,t,h}, \quad \forall h \in \{1, \dots, 24\}$$
subject to:
$$w_{i,t} \ge 0 \quad \text{and} \quad \sum_{i=1}^3 w_{i,t} = 1.0$$

### Mathematical Constraints of Convex Fusion
1. **Bounded Convex Hull:** For every forecast step $h$, the prediction is mathematically bounded: $\min_i \hat{y}_{i,t,h} \le \hat{y}_{t,h} \le \max_i \hat{y}_{i,t,h}$. The fused forecast cannot output unphysical values exceeding the range of expert outputs.
2. **Direct Interpretability:** The weights $w_{i,t}$ represent exact proportional attribution.
3. **Absence of Arbitrary Additive Residuals:** No unconstrained linear additive residual terms are included, preventing baseline bias drift under non-stationary regimes.

---

## 14. Training Objective & Auxiliary Expert Supervision

### The Joint Objective Function
\mathcal{L}_{\text{expert\_aux}} = \frac{1}{3} \sum_{i=1}^3 \mathcal{L}_{\text{expert}}(\hat{y}_i, y)
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}}(\hat{y}, y) + 0.15 \cdot \mathcal{L}_{\text{expert\_aux}} + 0.001 \cdot H(w)
\text{Equivalently: } \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}}(\hat{y}, y) + 0.15 \cdot \left( \frac{1}{3} \sum_{i=1}^3 \mathcal{L}_{\text{expert}}(\hat{y}_i, y) \right) + 0.001 \cdot H(w)

### Component Formulations
1. **Primary Fused Loss ($\mathcal{L}_{\text{fused}}$):**
   Mean Squared Error on standardized targets across the full 24-hour horizon:
   $$\mathcal{L}_{\text{fused}} = \frac{1}{24} \sum_{h=1}^{24} (\hat{y}_{t,h} - y_{t,h})^2$$
2. **Auxiliary Expert Supervision Loss ($\mathcal{L}_{\text{expert}}$):**
   $$\mathcal{L}_{\text{expert}}(\hat{y}_i, y) = \frac{1}{24} \sum_{h=1}^{24} (\hat{y}_{i,t,h} - y_{t,h})^2$$
   \mathcal{L}_{\text{expert\_aux}} = \frac{1}{3} \sum_{i=1}^3 \mathcal{L}_{\text{expert}}(\hat{y}_i, y)
   - **Weighting Coefficient:** $\lambda = 0.15$, normalized across the 3 experts (/3$).
   - **Scientific Purpose:** Supervising individual expert branches aims to encourage independent forecasting competence and reduce the risk of destructive co-adaptation. Each expert is penalized for its own forecast error, providing a training signal even when down-weighted by the router.
3. **Entropy Regularization Term ($H(w)$):**
   $$H(w) = -\sum_{i=1}^3 w_{i,t} \ln(w_{i,t} + 10^{-8})$$
   - **Weighting Coefficient:** $\beta = 0.001$.

### Optimization Protocol
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)`
- **Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-6)`
- **Early Stopping:** Monitored strictly on Validation Fused MSE Loss ($\mathcal{L}_{\text{fused}}$) with `patience = 7` epochs. Best validation checkpoint is restored.
- **Batch Size:** 64.
- **Maximum Epochs:** 50.

---

## 15. Standardized Data Protocol

### Universal Preprocessing Rules
1. **Frequency & Regularity:** Enforce uniform 1-hour cadence. Missing timestamps are reindexed to a continuous hourly timeline.
2. **Gap Filling Strategy:** Missing intervals $\le 6$ hours are filled via time-aware linear interpolation. Gaps $> 6$ hours trigger dataset rejection or partition boundary splitting.
3. **Chronological Partitioning:**
   $$\text{Train } (70\%) \longrightarrow \text{Validation } (15\%) \longrightarrow \text{Test } (15\%)$$
   Partitioning is executed along the raw chronological timeline prior to window generation.
4. **Train-Only Normalization:**
   $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{t=1}^{N_{\text{train}}} y_t, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}-1} \sum_{t=1}^{N_{\text{train}}} (y_t - \mu_{\text{train}})^2}$$
   Validation and test partitions are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$. Zero future distribution statistics are accessible.
5. **Partition Boundary Lookback Prepending:**
   Validation and test partitions prepend the final 191 observations of the preceding partition to provide the 192-hour context buffer and 168-hour expert input. Target windows begin exactly at partition boundary index 0.
---

## 16. Multi-Horizon Forecasting Protocol

### Primary Experimental Horizon
- **Day-Ahead 24-Hour Horizon ($H=24$):** Evaluates day-ahead dispatch scheduling, matching Independent System Operator (ISO) market requirements.

### Lead-Time Evaluation Breakdown
A single multi-output model produces all 24 steps simultaneously: $\hat{y}_t \in \mathbb{R}^{24}$. Multi-horizon performance is extracted directly from the primary 24-hour model by evaluating sub-horizon slices:
- $h = 1$ hour ahead (Immediate dispatch)
- $h = 6$ hours ahead (Intra-day re-commitment)
- $h = 12$ hours ahead (Half-day cycle)
- $h = 24$ hours ahead (Full day-ahead closure)

A separate long-horizon benchmark ($H = 48$ hours) will be evaluated on multi-year datasets (GEFCom2014) as a secondary generalization test.

---

## 17. Baseline Benchmark Matrix

The research model is benchmarked against 10 controlled baseline models:

| # | Model Identifier | Architectural Family | Parameter Budget | Primary Scientific Role |
|:--|:-----------------|:--------------------:|:----------------:|:------------------------|
| **B1** | `Seasonal_Naive_24` | Deterministic Rule | 0 | Ground-floor benchmark: $\hat{y}_{t+h} = y_{t+h-24}$. |
| **B2** | `DLinear_Standalone` | Direct Linear Model | $\sim 4,056$ | SOTA linear baseline (Zeng et al., AAAI 2023) to test if deep MoE provides value over direct linear decomposition. |
| **B3** | `GRU_Standalone` | Recurrent Neural Net | 38,148 | Evaluates individual recurrent expert competence. |
| **B4** | `TCN_Standalone` | Dilated Causal Conv | 37,978 | Evaluates individual convolutional expert competence. |
| **B5** | `PatchTemporal_Standalone` | Patch-Based Linear | 37,848 | Evaluates individual patch-based expert competence. |
| **B6** | `Static_Equal_Ensemble` | Fixed Rule Ensemble | 113,974 | Averages the three proposed experts equally ($1/3, 1/3, 1/3$). Tests whether dynamic routing improves over fixed averaging. |
| **B7** | `Static_Learned_Ensemble` | Linear Gating (No Context)| 113,977 | Learns fixed global weights $[w_1, w_2, w_3]$ via Softmax with no input context. |
| **B8** | `Standard_Input_MoE` | Conventional Deep MoE | 119,482 | Gating network receives raw sequence $X_t \in \mathbb{R}^{168}$ directly instead of domain context. |
| **B9** | `CAEG_Net_V1_Canonical` | Historical Reference | 121,531 | Immutable frozen reference baseline ($251.44\text{ MW}$). |
| **B10**| **Proposed Research CAEG-Net** | Context-Adaptive MoE | **116,569** | Full proposed architecture. |

---

## 18. Comprehensive Ablation Matrix

To isolate the contribution of each architectural component, 12 controlled ablations are defined:

| Code | Model Variant | Modification Relative to Full Proposed Model | Scientific Attribution Target |
|:-----|:--------------|:---------------------------------------------|:------------------------------|
| **A0** | Single Best Expert | Best standalone model among B3, B4, B5 | Value of ensembling over single best architecture. |
| **A1** | Static Equal Ensemble | Fixed weights $w = [1/3, 1/3, 1/3]$ | Value of learned weighting over static average. |
| **A2** | Static Learned Ensemble | Fixed learned weights $w \in \mathbb{R}^3$ (no context input) | Value of dynamic input-conditioned routing over optimal fixed blend. |
| **A3** | Standard Input-MoE | Router takes raw sequence $X_t$ instead of context $u_t$ | Value of curated domain context vs opaque raw sequence. |
| **A4** | No Physical Context | Remove $C_{\text{obs},t}$; router receives only Disagreement $D_t$ and $E_{\text{base},t}$ | Isolate contribution of observable grid dynamics. |
| **A5** | No Seasonality Features | Remove Diurnal & Weekly autocorrelations ($r_{24}, r_{168}$) | Test role of periodic rhythmicity. |
| **A6** | No Difficulty/Baseline Error | Remove $E_{\text{base},t}$ from context | Test utility of auxiliary baseline error feature. |
| **A7** | No Expert Disagreement | Remove Disagreement vector $D_t$; router receives only $C_t$ | Isolate contribution of inference-time disagreement. |
| **A8** | No Auxiliary Expert Loss | Train with $\lambda = 0$ (fused loss only) | Test whether expert supervision reduces co-adaptation risk. |
| **A9** | Unscaled Temperature | Fix temperature $\tau = 1.0$ (no sharpening) | Measure degree to which temperature scaling alters routing concentration. |
| **A10**| Horizon-Aware Routing | Router outputs step-specific weights $w_{t,h} \in \mathbb{R}^{24 \times 3}$ | Test whether multi-step routing outperforms global scalar routing. |
| **A11**| **Full Proposed Model** | Complete CAEG-Net V2 with $\tau=0.5, \lambda=0.15, \beta=0.001$ | Full synergistic system performance. |

---

## 19. Operational Regime Analysis

The research model is hypothesized to deliver its primary utility during high-stress operational conditions. We define three objective, training-quantized regimes:

### 1. Volatility Regime (Grid Turbulence)
- Metric: Lookback first-difference standard deviation $v_t$.
- Tertile Thresholds derived strictly from the training partition:
  - Low Volatility: $v_t < Q_{33}(\text{Train})$
  - Medium Volatility: $Q_{33}(\text{Train}) \le v_t \le Q_{66}(\text{Train})$
  - High Volatility: $v_t > Q_{66}(\text{Train})$

### 2. Disagreement Regime (Model Uncertainty)
- Metric: Inference-time pairwise expert discrepancy $d_{\text{pair},t}$.
- Tertile Thresholds derived from candidate expert forecasts across the training set:
  - Low Disagreement (Consensus): $d_{\text{pair}} < Q_{33}$
  - Medium Disagreement: $Q_{33} \le d_{\text{pair}} \le Q_{66}$
  - High Disagreement (Conflict): $d_{\text{pair}} > Q_{66}$

### 3. Diurnal Peak/Ramp Regime (Peak Load Stress)
- Morning Ramp: Hours 06:00 to 09:00 local time
- Evening Peak: Hours 17:00 to 21:00 local time
- Base Load / Trough: Hours 01:00 to 05:00 local time

*Protocol Requirement:* In each regime, MAE, RMSE, MAPE, and average expert weights $[\bar{w}_1, \bar{w}_2, \bar{w}_3]$ are evaluated separately to examine whether the router shifts mass systematically toward the best-performing expert in that regime.

---

## 20. Statistical Testing & Significance Protocol

### Mitigating Overlapping Horizon Autocorrelation
Because windows slide by 1 hour with a 24-hour horizon, test residuals exhibit an $\text{MA}(23)$ autocorrelation structure. Naive $t$-tests across 1,294 samples violate i.i.d. assumptions and produce invalid, inflated $t$-statistics.

### Statistical Test Suite & Block Count Verification
1. **Non-Overlapping Daily Block Analysis (Primary Sample-Level Test):**
   - Subsample test origins every 24 hours: $t \in \{0, 24, 48, \dots, 1272\}$.
   - **Exact Block Count Specification:**
     - Evaluating `np.arange(0, 1294, 24)` yields **54 evaluation origins**.
     - Exactly **53 complete, non-overlapping 24-hour episodes** fit within the 1,294-horizon test partition ($53 \times 24 = 1,272 \le 1,294$). The final block at index 1272 spans hours 1272 to 1293 (22 steps) and may either be included as a 54th origin or truncated to 53 strictly complete episodes. Both conventions are supported and explicitly documented.
   - Run paired Wilcoxon signed-rank test and paired Student's $t$-test on daily MAEs:
     $$\Delta \text{MAE}_d = \text{MAE}_{\text{Baseline}, d} - \text{MAE}_{\text{Proposed}, d}, \quad d \in \{1, \dots, 53\}$$
   - Report mean difference, 95% bootstrap confidence interval (10,000 resamples), and $p$-value.
2. **Multi-Seed Paired $t$-Test (Cross-Initialization Robustness):**
   - Train across the 5 canonical random seeds: `[42, 123, 999, 2024, 3407]`.
   - Compute seed-level paired differences ($df = 4$).
   - Report mean delta, standard deviation, and paired $t$-test $p$-value.
3. **Harvey-Leybourne-Newbold (HLN) Diebold-Mariano Test:**
   - Evaluated on the full test series with rectangular lag truncation $k = 24 - 1 = 23$, applying the HLN finite-sample correction factor:
     $$\text{DM}_{\text{HLN}} = \text{DM} \cdot \sqrt{\frac{N + 1 - 2h + h(h-1)/N}{N}}$$
---

## 21. Computational Profiling & Efficiency Metrics

The research model is designed to operate within an edge/workstation compute envelope (RTX 4050 6 GB GPU). Telemetry recorded for every run:
1. **Model Size:** Total parameters and trainable parameters (target: 116,569).
2. **Computational Complexity:** Multiply-Accumulate operations (MACs / FLOPs) per 24h forward pass.
3. **Training Throughput:** Epoch training time (seconds/epoch) and convergence time to best validation checkpoint.
4. **Inference Latency:** Batch-1 latency ($\mu\text{s}$ per single day-ahead forecast) and bulk test evaluation runtime.
5. **Memory Footprint:** Peak CUDA memory allocated (`torch.cuda.max_memory_allocated()`) and GPU cache reserved.

---

## 22. Multi-Dataset External Generalization Plan

To demonstrate that the model is an algorithmic advance rather than an artifact of single-zone tuning:

### Dataset 1: Modern PJM (Development & Ablation Benchmark)
- **Source:** `data/Modern_PJM/pjm_load.csv`
- **Characteristics:** 8,784 hourly load records (1 year: Oct 2023 – Oct 2024), single regional load curve, mean load $5,552\text{ MW}$.
- **Split:** 70% train (6,148 rows), 15% validation (1,318 rows), 15% test (1,318 rows / 54 days).

### Dataset 2: GEFCom2014 Electricity Load Benchmark (External Publication Validation)
- **Source:** Global Energy Forecasting Competition 2014 (Hong et al., International Journal of Forecasting 2016).
- **Characteristics:** 15 distinct spatial transmission zones, multi-year hourly series, extreme weather events.
- **Adapter:** `research/data_adapter_gefcom.py` and `research/configs/gefcom2014_config.json`.
- **Protocol:** Identical model architecture and hyperparameters evaluated across multiple zones without zone-specific structural tuning.

---

## 23. Objective Scientific Acceptance Targets

Rather than mechanically forcing artificial dynamic behavior, the following targets serve as empirical development milestones and falsification criteria:

| # | Criterion | Development Target / Hypothesis | Evaluation Role |
|:--|:----------|:--------------------------------|:----------------|
| **C1** | **Outperform Standalone Experts** | Lower test MAE than all three standalone experts (GRU, TCN, Patch) on PJM. | Validates benefit of ensembling over individual architectures. |
| **C2** | **Outperform Fixed Ensembles** | Statistically significant improvement over `Static_Equal_Ensemble` ($p < 0.05$). | Tests whether learned weighting improves over static averaging. |
| **C3** | **Outperform Learned Ensembles** | Statistically significant improvement over `Static_Learned_Ensemble` ($p < 0.05$). | Tests whether input-conditioned routing improves over optimal fixed blend. |
| **C4** | **Outperform Input-MoE** | Lower test MAE than `Standard_Input_MoE` with equal capacity. | Tests whether curated domain context provides value over raw input. |
| **C5** | **Improve Over Canonical V1** | Test whether proposed model achieves lower test MAE than V1 ($251.44\text{ MW}$). | Primary comparative benchmark against frozen reference. |
| **C6** | **Descriptive Routing Diagnostics** | Examine whether routing weights exhibit meaningful variation across samples and regimes. | Descriptive analysis; no arbitrary variance threshold required. |
| **C7** | **High-Stress Resilience** | Test whether model achieves higher relative advantage in High-Disagreement regime. | Evaluates core hypothesis regarding operational difficulty. |
| **C8** | **Multi-Zone Generalization** | Superior average rank across GEFCom2014 zones compared to standalone baselines. | Evaluates external multi-climate robustness. |
| **C9** | **Parameter Efficiency** | Total parameters remain within $\pm 10\%$ of V1 (expected: $116,569$). | Capacity parity preservation. |

*Reporting Commitment:* If the proposed architecture fails to meet any of these targets (e.g., if it does not beat a static learned ensemble), the result will be reported honestly as a scientific finding rather than suppressed.

---

## 24. Expected Novelty & Literature Positioning

The paper's scientific contributions are positioned as follows:
1. **Explicit Operational Context Conditioning:** Moving beyond conventional end-to-end Input-MoE (which routes on opaque raw inputs) by constructing a compact, causally verified domain context vector (volatility, trend, periodicity, range, baseline error) that mirrors actual transmission system operator situational awareness.
2. **Forecast-Aware Inter-Expert Disagreement Routing:** A parameter-free mechanism that computes consensus metrics (pairwise discrepancy, variance, range) dynamically from candidate expert forecasts and feeds them to the router. When inductive models conflict, the network recognizes regime uncertainty and adjusts gating weights accordingly.
3. **Capacity-Balanced Tri-Inductive Ensemble:** Combining recurrent state tracking (GRU), multi-scale causal convolution (TCN), and patch-based sub-series representation (PatchLinear) under strict capacity balance ($\sim 38\text{k}$ params each), eliminating capacity confounding.
4. **Auxiliary Expert Supervision:** Mitigating destructive expert co-adaptation by jointly supervising individual expert branches ($\lambda = 0.15$), encouraging every expert to maintain independent forecasting competence.

---

## 25. Risks, Failure Modes, and Built-in Mitigations

| Risk / Failure Mode | Likelihood | Impact | Built-in Scientific Mitigation |
|:--------------------|:----------:|:------:|:-------------------------------|
| **1. Gating Uniformity:** Router outputs near-uniform weights across regimes. | Moderate | High | Entropy regularization ($+\beta H(w)$) penalizes high-entropy distributions; temperature parameter $\tau$ is tunable via validation-only protocol. |
| **2. Co-Adaptation of Experts:** Experts become unstable when evaluated individually. | Low | Moderate | Auxiliary loss $\lambda \sum \mathcal{L}_i$ directly supervises each expert branch to minimize individual MSE. |
| **3. Disagreement Gradient Contamination:** Experts attempt to artificially alter disagreement to manipulate gate. | Very Low | Critical | Explicit autograd detachment prevents gradient flow through the disagreement-routing pathway, substantially mitigating this risk, without claiming that all possible implementation errors are mathematically impossible. |
| **4. Patch Expert Underperformance:** PatchLinear fails to provide sufficient accuracy or diversity. | Moderate | Moderate | Standalone Patch model is evaluated in Phase 3; protocol allows replacement if experimental evidence demonstrates deficiency. |
| **5. Multi-Step Autocorrelation False Positive:** Overlapping hourly test windows inflate statistical significance. | High | High | Primary statistical significance test is conducted on non-overlapping 24h blocks with Wilcoxon and HLN Diebold-Mariano tests. |

---

## 26. Phase 3 Implementation Requirements

When human review is approved, Phase 3 execution must follow these rules:
1. **Dedicated Source Files:** Implement new classes in `research/models.py` and `research/data.py`. Do **NOT** modify `caeg_net.py` or `data_utils.py`.
2. **Deterministic Random Seeding:** Seed PyTorch, NumPy, and CUDA identically across all 5 canonical seeds (`[42, 123, 999, 2024, 3407]`).
3. **Self-Contained Experiment Runners:** Implement `research/experiments/run_phase3_benchmark.py` with automated logging to CSV and NPZ prediction caches.
4. **Automated Sanity Tests:** Run forward-pass shape tests and future target perturbation tests before launching 5-seed training.

---

## Implementation Readiness Checklist

| # | Architecture / Protocol Item | Status | Detailed Implementation Agreement |
|:--|:-----------------------------|:------:|:-----------------------------------|
| 1 | **Primary Expert 1 (Recurrent)** | **PASS** | 2-layer GRU ($h=54$), Linear head, $\sim 38,148$ parameters. |
| 2 | **Primary Expert 2 (Causal Conv)** | **PASS** | Causal TCN, 5 blocks ($c=34, k=4, d \in \{1,2,4,8,16\}$), RF=187h, $\sim 37,978$ params. |
| 3 | **Primary Expert 3 (Patch Candidate)** | **PASS** | PatchLinear ($P=13, P_L=24, S=12, d_p=48$), $\sim 37,848$ params. Research candidate status. |
| 4 | **Context Feature Set & History** | **PASS** | 6 features (Trend, Volatility, Range, Diurnal/Weekly periodicity, Baseline error) with 192h causal buffer. |
| 5 | **Disagreement Formulation** | **PASS** | 3 features (Pairwise discrepancy, Inter-expert std, Range), gradient-detached. |
| 6 | **Routing Architecture** | **PASS** | MLP 9 $\to$ 32 $\to$ LN $\to$ ReLU $\to$ 32 $\to$ ReLU $\to$ Drop(0.1) $\to$ 3 with $\tau=0.5$ (validation tuning protocol). |
| 7 | **Fusion Equation** | **PASS** | Strictly convex combination: $\hat{y} = \sum w_i \hat{y}_i$, $w_i \ge 0, \sum w_i = 1.0$. |
| 8 | **Training Loss Formulation** | **PASS** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fused}} + 0.15 \cdot \left( \frac{1}{3} \sum_{i=1}^3 \mathcal{L}_{\text{expert}} \right) + 0.001 \cdot H(w)$, with (w) = -\sum w_i \ln(w_i+\epsilon)$. |
| 9 | **Parameter Budget Balance** | **PASS** | Total $116,569$ params ($-4.08\%$ vs V1), all 3 experts balanced at $\sim 32.6\%$. |
| 10| **Data Protocol Preservation** | **PASS** | 70/15/15 chronological split, train-only scaling, boundary lookback prepend. |
| 11| **Statistical Testing Suite** | **PASS** | 53 full non-overlapping daily blocks (54 sampling origins), 5-seed paired tests, HLN Diebold-Mariano. |
| 12| **Canonical V1 Untouched** | **PASS** | Reference checkpoints and code frozen; zero modification of V1 artifacts. |