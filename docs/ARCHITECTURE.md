# CAEG-Net: System Architecture & Technical Specification

## Context-Adaptive Expert Gating Network for Short-Term Load Forecasting

- **Public Model Name:** **CAEG-Net**
- **Internal Experimental Identifier:** `F2 / A2-OOF` (`ConfidenceFallbackCAEGNet`)
- **Certification Status:** Final Model Locked — Development Complete
- **Primary Operational Objective:** Day-Ahead Short-Term Electricity Load Forecasting (168-Hour Lookback -> 24-Hour Forecast Horizon)
- **Total Trainable Parameters:** **121,724** (<0.5 MB storage footprint)

---

## 1. Architectural Philosophy & Overview

Short-Term Electricity Load Forecasting (STLF) on power grids involves non-stationary multi-scale dynamics:
1. **Diurnal and weekly periodicity** governed by consumer rhythms and business cycles.
2. **Persistent baseline drift** caused by multi-day temperature regimes and industrial scheduling.
3. **Abrupt demand shocks and ramping events** triggered by weather extremes or sudden generation/load switching.

Monolithic deep architectures embody fixed inductive biases that create structural performance trade-offs:
- Recurrent networks (LSTM) excel at continuous diurnal state memory but risk gradient saturation and slow response to steep ramps.
- Dilated convolutional networks (TCN) capture expansive causal receptive fields (253 hours) without recursive gradient decay, but can over-smooth localized transients.
- Multi-scale convolutional extractors (CNN) rapidly detect localized edge and peak motifs, but lack global sequential memory.
- Static ensembles (e.g. unweighted averaging) fail to adapt when operating regimes alternate between steady baselines and volatile ramps.

**CAEG-Net** integrates three heterogeneous temporal experts coordinated by an explicit **context-adaptive gating router** and stabilized by an **empirical confidence fallback head**.

```text
INPUT LOAD SEQUENCE: X_t in R^{B x 168 x 1} (7 Days Lookback)
  │
  ├──► LSTM Expert (56,152 params) ─────────────► y_hat_LSTM in R^{B x 24}
  │
  ├──► TCN Expert  (36,952 params, RF=253h) ────► y_hat_TCN  in R^{B x 24}
  │
  ├──► CNN Expert  (27,400 params) ─────────────► y_hat_CNN  in R^{B x 24}
  │
  └──► Context Vector c_t in R^7
         │
         ▼
     Context Feature Encoder (432 params) ──────► e_C in R^{B x 16}
         │
         ▼
     Context Gating Network (643 params) ───────► Simplex Weights g_t in Delta^2
         │
         ├──► Adaptive Combination: y_hat_adaptive = sum_i g_{t,i} * y_hat_{t,i}
         │
     Confidence Fallback Head (145 params) ─────► Fallback Weight lambda_t in (0, 1)
         │                                         (Empirical lambda approx 0.51)
         │
         ▼
     Equal-Expert Centroid: y_hat_equal = 1/3 * (y_hat_LSTM + y_hat_TCN + y_hat_CNN)
         │
         ▼
     Final Composite Forecast: y_hat_t = lambda_t * y_hat_adaptive + (1 - lambda_t) * y_hat_equal
     Total Parameters: 121,724  |  Horizon: 24 Hours Ahead
```

---

## 2. Authoritative Parameter Budget

Parameter counts are audited directly from the authoritative source implementation (`caeg_net.py` and `src/models/`):

| Component Module | Class Name | Architectural Role | Trainable Parameters | Parameter Share |
| :--- | :--- | :--- | :---: | :---: |
| **LSTM Expert** | `LSTMExpert` | 2-Layer Stacked LSTM + Linear Prediction Head | **56,152** | 46.13% |
| **TCN Expert** | `TCNExpert` | 6-Stage Dilated Causal Residual Conv1D ($RF=253\mathrm{h}$) | **36,952** | 30.36% |
| **CNN Expert** | `CNNExpert` | 3-Stage Multi-Kernel Conv1D with Adaptive Pooling | **27,400** | 22.51% |
| *Backbone Subtotal* | *Expert Pool* | *Three Heterogeneous Temporal Feature Extractors* | ***120,504*** | ***98.99%*** |
| **Context Encoder** | `ContextFeatureEncoder` | Non-linear latent embedding ($7 \to 16 \to 16$) | **432** | 0.35% |
| **Gating Network** | `ContextGatingNetwork` | Softmax simplex expert router ($16 \to 32 \to 3$) | **643** | 0.53% |
| *Router Subtotal* | *Adaptive Router* | *Context Embedding + Simplex Expert Weights* | ***1,075*** | ***0.88%*** |
| **Confidence Head** | `ConfidenceFallbackCAEGNet` | Dynamic shrinkage interpolation head ($7 \to 16 \to 1$) | **145** | 0.12% |
| **TOTAL CAEG-Net** | `ConfidenceFallbackCAEGNet` | **Full Locked Champion Model (F2 / A2-OOF)** | **121,724** | **100.00%** |

---

## 3. Submodule Specifications

### 3.1. LSTM Expert (`LSTMExpert` — 56,152 Parameters)
- **Input:** $X_t \in \mathbb{R}^{B \times 168 \times 1}$
- **Layer 1:** LSTM cell, `input_size=1`, `hidden_size=64`, `dropout=0.1`
- **Layer 2:** LSTM cell, `input_size=64`, `hidden_size=64`, `dropout=0.1`
- **Pooling:** Final causal step extraction ($h_{168} \in \mathbb{R}^{B \times 64}$)
- **Head:** `Linear(64, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 24)`
- **Output:** $\hat{y}_{\text{LSTM}} \in \mathbb{R}^{B \times 24}$
- **Inductive Bias:** Recurrent memory captures diurnal continuous cycles and multi-day load baselines.

### 3.2. TCN Expert (`TCNExpert` — 36,952 Parameters)
- **Input:** $X_t^T \in \mathbb{R}^{B \times 1 \times 168}$
- **Residual Blocks:** 6 dilated causal blocks with kernel size $k=3$ and exponential dilations:
  $$\mathcal{D} = \{1, 2, 4, 8, 16, 32\}$$
- **Block Topology:**
  $$\text{Conv1D}(d) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Conv1D}(d) \to \text{BatchNorm} \to \text{ReLU} + \text{Residual}$$
- **Causal Receptive Field:**
  $$RF = 1 + 2 \times (k - 1) \times \sum_{i=0}^5 2^i = 1 + 2 \times 2 \times 63 = 253 \text{ hours}$$
  Because $253 > 168$, every output point causally perceives the entire 7-day history without gradient vanishing.
- **Pooling:** Last causal time step ($Z_{:, :, -1} \in \mathbb{R}^{B \times 32}$)
- **Head:** `Linear(32, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 24)`
- **Output:** $\hat{y}_{\text{TCN}} \in \mathbb{R}^{B \times 24}$

### 3.3. CNN Expert (`CNNExpert` — 27,400 Parameters)
- **Input:** $X_t^T \in \mathbb{R}^{B \times 1 \times 168}$
- **Stage 1:** `Conv1D(1 -> 32, k=3, pad=1) -> BatchNorm -> ReLU -> MaxPool1D(2)` ($168 \to 84$)
- **Stage 2:** `Conv1D(32 -> 64, k=5, pad=2) -> BatchNorm -> ReLU -> MaxPool1D(2)` ($84 \to 42$)
- **Stage 3:** `Conv1D(64 -> 64, k=3, pad=1) -> BatchNorm -> ReLU -> AdaptiveAvgPool1D(1)` ($42 \to 1$)
- **Head:** `Linear(64, 48) -> ReLU -> Dropout(0.1) -> Linear(48, 24)`
- **Output:** $\hat{y}_{\text{CNN}} \in \mathbb{R}^{B \times 24}$
- **Inductive Bias:** Multi-scale spatial convolutions isolate localized morning/evening demand ramps and steep transitions.

---

## 4. Context Vector & Gating Mechanism

### 4.1. The 7D Causal Context Vector ($c_t \in \mathbb{R}^7$)
To avoid expert collapse during end-to-end training, the router does not condition on raw high-dimensional sequence inputs. Instead, it evaluates explicit physical and performance domain features computed strictly at or before forecast origin $t$:

1. **Trend Slope ($\beta_{\text{trend}}$):** First-order linear regression coefficient fitted across the 168-hour lookback window, capturing multi-day seasonal slope.
2. **Normalized Volatility ($\sigma_{\text{norm}}$):** Standard deviation of load across the immediate preceding 24 hours ($[t-23, t]$), capturing demand variability.
3. **Diurnal Autocorrelation ($r_{\text{lag-24}}$):** Pearson correlation coefficient between $[t-47, t-24]$ and $[t-23, t]$, measuring diurnal cycle consistency.
4. **Recent Tracking Error ($\text{MAE}_{\text{recent}}$):** Mean absolute error of the completed historical 24-hour forecast ending at origin $t$.
5. **OOF Historical Error — LSTM ($\text{OOF}_{\text{LSTM}}$):** Causally tracked historical validation error residual for the LSTM expert.
6. **OOF Historical Error — TCN ($\text{OOF}_{\text{TCN}}$):** Causally tracked historical validation error residual for the TCN expert.
7. **OOF Historical Error — CNN ($\text{OOF}_{\text{CNN}}$):** Causally tracked historical validation error residual for the CNN expert.

### 4.2. Context Feature Encoder (432 Parameters)
$$e_C = \text{ReLU}(\text{Linear}_{16 \to 16}(\text{ReLU}(\text{LayerNorm}_{16}(\text{Linear}_{7 \to 16}(c_t)))))$$

### 4.3. Softmax Gating Network (643 Parameters)
$$g_t = \text{Softmax}(\text{Linear}_{32 \to 3}(\text{Dropout}_{0.1}(\text{ReLU}(\text{Linear}_{16 \to 32}(e_C))))) \in \Delta^2$$
Guarantees strictly positive, convex expert weights summing to unity: $\sum_{i=1}^3 g_{t, i} = 1.0, \; g_{t, i} > 0$.

---

## 5. Confidence Fallback & Centroid Shrinkage

### 5.1. Motivation
Unregularized dynamic gating routers can become overconfident in anomalous conditions, assigning extreme weights to a single expert. To prevent catastrophic failure modes, CAEG-Net incorporates an empirical shrinkage mechanism toward the unweighted ensemble centroid:
$$\hat{y}_{\text{equal}} = \frac{1}{3} (\hat{y}_{\text{LSTM}} + \hat{y}_{\text{TCN}} + \hat{y}_{\text{CNN}})$$

### 5.2. Mathematical Formulation (145 Parameters)
$$\lambda_t = \sigma(\text{Linear}_{16 \to 1}(\text{ReLU}(\text{Linear}_{7 \to 16}(c_t)))) \in (0, 1)$$
$$\hat{y}_{t} = \lambda_t \hat{y}_{\text{adaptive}} + (1 - \lambda_t) \hat{y}_{\text{equal}}$$

### 5.3. Empirical Behavior
In verified multi-grid evaluations, the learned coefficient $\lambda_t$ exhibits low temporal variance around $\approx 0.51$:
- **PJM:** Mean $\lambda = 0.5066 \pm 0.0038$ ($CV = 0.75\%$)
- **GEFCom2014:** Mean $\lambda = 0.5170 \pm 0.0055$ ($CV = 1.06\%$)
- **UCI Electricity:** Mean $\lambda = 0.5064 \pm 0.0030$ ($CV = 0.59\%$)

This demonstrates that the confidence head functions primarily as an effective, regularized shrinkage anchor, stabilizing predictions around the equal-expert centroid while allowing calibrated dynamic departures.

---

## 6. Conventional MoE vs. CAEG-Net Comparison

| Dimension | Conventional Input-Gated MoE | CAEG-Net (Context-Adaptive Gating) |
| :--- | :--- | :--- |
| **Gating Input** | Raw lookback sequence $X_t \in \mathbb{R}^{168 \times 1}$ | Distilled 7D physical context vector $c_t \in \mathbb{R}^7$ |
| **Router Training** | End-to-end backpropagation with expert starvation risk | Out-of-fold causal performance calibration |
| **Fail-Safe Mechanism** | None (unconstrained softmax) | Dynamic shrinkage to empirical centroid ($\lambda \approx 0.51$) |
| **PJM Benchmark MAE** | $276.30 \pm 13.64$ MW | **$250.97 \pm 10.69$ MW** ($-9.17\%$ error reduction) |
| **Parameters** | 126,011 parameters | **121,724 parameters** |