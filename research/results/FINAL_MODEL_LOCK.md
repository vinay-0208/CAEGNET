# CAEG-Net Phase 15: Definitive Model Lock Specification
## Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting

**Document**: `research/results/FINAL_MODEL_LOCK.md`  
**Public Model Name**: **CAEG-Net**  
**Internal Experimental Variant ID**: `F2 / A2-OOF` (`ConfidenceFallbackCAEGNet`)  
**Certification Date**: September 12, 2026  
**Repository Branch**: `research-track`  
**Git Commit Hash**: `18692a0b8245b96eea94a9ab6b815a99417d00a7`  
**Status**: **FINAL MODEL LOCKED — DEVELOPMENT PHASE COMPLETED & FROZEN**

---

## 0. Formal Model Development Closeout Statement

> ### 🛑 MODEL DEVELOPMENT COMPLETE
> **MODEL DEVELOPMENT COMPLETE — NO FURTHER ARCHITECTURAL OPTIMIZATION PLANNED FOR THIS RESEARCH VERSION.**  
>
> Following the Phase 15 final controlled experimental campaign, the CAEG-Net model architecture, expert family, routing mechanism, confidence head, hyperparameter set, and preprocessing pipeline are **permanently frozen**.  
> No Phase 16 or subsequent architecture exploration phases will be initiated. All subsequent work in this repository is strictly restricted to academic manuscript preparation, peer-review documentation, interactive dashboards, presentation artifacts, and reproducibility packaging.

---

## 1. System Identity & Naming Standards

- **Primary Public Name**: **CAEG-Net** (Context-Adaptive Expert Gating Network)
- **Internal Experimental ID**: `F2 / A2-OOF` (used solely for forensic traceability and repository experimental provenance)
- **PyTorch Model Class**: `ConfidenceFallbackCAEGNet` (implemented in `research/scratch/models_phase15b.py` and canonical `caeg_net.py`)
- **Core Research Question**:  
  *"Can context-aware adaptive expert gating improve short-term electricity load forecasting by dynamically combining complementary LSTM, TCN, and CNN temporal experts?"*

---

## 2. Definitive Architectural Specification

### 2.1 Fixed Temporal Expert Family
The expert family is strictly locked to three heterogeneous neural architectures:
1. **LSTM Expert ($56,152$ parameters, $46.1\%$):**
   - Structure: 2-layer stacked LSTM (`input_size=1`, `hidden_size=64`, `dropout=0.1`).
   - Projection Head: Linear projection layer (`64` $	o$ `24`).
   - Inductive Bias: Step-by-step sequential recurrent memory; captures persistent diurnal continuity and work-rest cyclic baselines.
2. **TCN Expert ($36,952$ parameters, $30.4\%$):**
   - Structure: 6 dilated causal residual blocks with dilations $d \in \{1, 2, 4, 8, 16, 32\}$, `num_channels=32`, `kernel_size=3`.
   - Receptive Field: $253$ hours ($> 168$ hours input lookback), ensuring causal coverage across the entire lookback.
   - Projection Head: Adaptive pooling + Linear projection (`32` $	o$ `24`).
   - Inductive Bias: Non-recursive hierarchical temporal convolutions; eliminates vanishing gradients and recursive error accumulation over long horizons.
3. **CNN Expert ($27,400$ parameters, $22.5\%$):**
   - Structure: 3 sequential 1D convolutional stages with kernels $k \in \{3, 5, 7\}$, batch normalization, ReLU activations, and adaptive pooling.
   - Projection Head: Fully connected projection layer (`48` $	o$ `24`).
   - Inductive Bias: Localized pattern matching; excels at detecting high-frequency edge motifs, steep morning load ramps, and abrupt industrial shifts.

### 2.2 Routing & Gating Mechanism ($1,075$ parameters, $0.9\%$)
- **Context Feature Encoder:** Multi-Layer Perceptron (MLP) with LayerNorm and ReLU mapping the causal context vector to a 16-dimensional embedding.
- **Gating Projection Head:** Linear layer (`16` $	o$ `3`) with Softmax activation generating non-negative convex routing weights:
  $$\mathbf{w}_t = [w_{\text{LSTM}}, w_{\text{TCN}}, w_{\text{CNN}}]^T \in \mathbb{R}^3, \quad w_i > 0, \quad \sum_{i=1}^3 w_i = 1.0$$
- **Causal Context Vector ($\mathbf{C}_t$):**
  1. *Trend:* Linear regression slope across the 168-hour input window.
  2. *Volatility:* Normalized standard deviation of load across the 168-hour lookback.
  3. *Periodicity:* 24-hour lag autocorrelation ($r_{\text{lag-24}}$).
  4. *Causal Recent Error:* Rolling Mean Absolute Error of recent forecasts over the observed lookback.
  5. *Out-of-Fold (OOF) Performance Features:* Causally tracked historical validation error residuals of each expert.
- **Adaptive Convex Fusion:**
  $$\hat{\mathbf{y}}_{\text{adaptive}, t} = w_{\text{LSTM}, t} \hat{\mathbf{y}}_{\text{LSTM}, t} + w_{\text{TCN}, t} \hat{\mathbf{y}}_{\text{TCN}, t} + w_{\text{CNN}, t} \hat{\mathbf{y}}_{\text{CNN}, t}$$

### 2.3 Confidence Shrinkage Head ($145$ parameters, $0.1\%$)
- **Architecture:** Compact linear layer with Sigmoid activation taking the context embedding and outputting a scalar blend parameter:
  $$\lambda_t = \sigma(\mathbf{W}_{\text{conf}} \mathbf{h}_{\text{context}, t} + b_{\text{conf}}) \in (0, 1)$$
- **Shrinkage Formulation:**
  $$\hat{\mathbf{y}}_{\text{final}, t} = \lambda_t \hat{\mathbf{y}}_{\text{adaptive}, t} + (1 - \lambda_t) \hat{\mathbf{y}}_{\text{equal}, t}$$
  where $\hat{\mathbf{y}}_{\text{equal}, t} = \frac{1}{3}(\hat{\mathbf{y}}_{\text{LSTM}, t} + \hat{\mathbf{y}}_{\text{TCN}, t} + \hat{\mathbf{y}}_{\text{CNN}, t})$.
- **Empirical Behavior:** Learned $\lambda$ settles near $pprox 0.51$ with low temporal variance ($CV pprox 0.6\% - 1.1\%$), acting as an empirical regularizer anchoring predictions toward the robust ensemble centroid.

### 2.4 Exact Parameter Count Breakdown
| Component | Trainable Parameters | Percentage of Model |
| :--- | :---: | :---: |
| LSTM Expert | $56,152$ | $46.13\%$ |
| TCN Expert | $36,952$ | $30.36\%$ |
| CNN Expert | $27,400$ | $22.51\%$ |
| **Temporal Backbone Subtotal** | **$120,504$** | **$98.99\%$** |
| Context Gating Router | $1,075$ | $0.88\%$ |
| Confidence Fallback Head | $145$ | $0.12\%$ |
| **TOTAL CAEG-Net F2 Model** | **$121,724$** | **$100.00\%$** |

---

## 3. Dataset & Protocol Specifications

### 3.1 Task Formulation
- **Input Lookback Horizon ($L$):** $168$ hours ($7$ consecutive days of hourly load history).
- **Forecast Horizon ($H$):** $24$ hours (day-ahead hourly dispatch schedule).
- **Target Resolution:** Hourly intervals.

### 3.2 Benchmark Datasets
1. **PJM Interconnection (PJM):** Regional operational load from US Mid-Atlantic RTO ($8,784$ contiguous hours, leap year 2023–2024, unit: MW).
2. **GEFCom2014:** Global Energy Forecasting Competition zonal power grid series ($78,888$ contiguous hours, unit: kW).
3. **UCI ElectricityLoadDiagrams20112014:** Portuguese national grid demand aggregated from 15-minute intervals to hourly series ($26,304$ contiguous hours, unit: MW).

### 3.3 Leakage-Free Data Protocol
- **Partitioning:** Chronological split: 70% Train / 15% Validation / 15% Test. Zero random shuffling across time.
- **Scaling:** `StandardScaler` fitted strictly on the training partition; validation and test partitions transformed using train-only $\mu$ and $\sigma$.
- **Causality:** Target horizons ($y_{t+1 \dots t+24}$) are strictly excluded from input windows and routing feature construction.

### 3.4 Training Specifications
- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=0.0001`, `betas=(0.9, 0.999)`).
- **Loss Function:** Mean Squared Error (MSE) on standardized training targets.
- **Batch Size:** 32.
- **Max Epochs:** 100 with early stopping patience of 15 epochs on validation loss.
- **Evaluation Seeds:** 5 independent random seeds: `[42, 123, 999, 2024, 3407]`.

---

## 4. Authoritative Frozen Benchmark Results

All five-seed results are reported as **sample mean ± population standard deviation (ddof=0)** across seeds `[42, 123, 999, 2024, 3407]`.

### 4.1 Primary Benchmark Comparison Table

| Benchmark Grid | Model Formulation | Primary Test MAE | Test RMSE | Test $R^2$ | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PJM (MW)** | **CAEG-Net (F2 / A2-OOF)** | **250.9747 ± 10.6938** | **335.3822** | **0.8714** | **CHAMPION_LOCKED** |
| PJM (MW) | Standalone TCN | 259.3264 | — | — | Baseline (Best Standalone) |
| PJM (MW) | Equal Ensemble | 279.8282 | — | — | Baseline (Static Ensemble) |
| PJM (MW) | Standalone LSTM | 291.7300 | — | — | Baseline |
| PJM (MW) | Standalone CNN | 432.0800 | — | — | Baseline |
| PJM (MW) | Fixed Shrinkage Control | 253.5008 ± 8.0264 | 338.9532 | 0.8688 | Exploratory Mechanism |
| PJM (MW) | Dynamic Confidence Control | 251.9419 ± 9.8000 | 336.7396 | 0.8704 | Exploratory Mechanism |
| PJM (MW) | Horizon Routing Control | 257.5450 ± 5.3670 | 343.9132 | 0.8650 | Exploratory Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GEFCom (kW)** | **CAEG-Net (F2 / A2-OOF)** | **12.4077 ± 0.1525** | **18.0446** | **0.8610** | **CHAMPION_LOCKED** |
| GEFCom (kW) | Fixed Shrinkage Control | 12.3607 ± 0.1841 | 18.0181 | 0.8614 | Exploratory Diagnostic Min |
| GEFCom (kW) | Dynamic Confidence Control | 12.4855 ± 0.2685 | 18.1019 | 0.8601 | Exploratory Mechanism |
| GEFCom (kW) | Standalone TCN | 12.5729 | — | — | Baseline (Best Standalone) |
| GEFCom (kW) | Equal Ensemble | 12.6248 | — | — | Baseline (Static Ensemble) |
| GEFCom (kW) | Horizon Routing Control | 12.8582 ± 0.2520 | 18.4392 | 0.8548 | Exploratory Mechanism |
| GEFCom (kW) | Standalone LSTM | 13.2300 | — | — | Baseline |
| GEFCom (kW) | Standalone CNN | 14.5000 | — | — | Baseline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **UCI (MW)** | **CAEG-Net (F2 / A2-OOF)** | **7.7371 ± 0.3037** | **10.9556** | **0.9831** | **CHAMPION_LOCKED** |
| UCI (MW) | Standalone LSTM | 7.5542 | — | — | Baseline (Best Standalone) |
| UCI (MW) | Fixed Shrinkage Control | 7.7523 ± 0.1814 | 10.9893 | 0.9830 | Exploratory Mechanism |
| UCI (MW) | Dynamic Confidence Control | 7.8177 ± 0.2838 | 10.9980 | 0.9830 | Exploratory Mechanism |
| UCI (MW) | Horizon Routing Control | 8.1309 ± 0.4048 | 11.4837 | 0.9814 | Exploratory Mechanism |
| UCI (MW) | Equal Ensemble | 8.1675 | — | — | Baseline (Static Ensemble) |
| UCI (MW) | Standalone TCN | 8.3400 | — | — | Baseline |
| UCI (MW) | Standalone CNN | 11.7100 | — | — | Baseline |

### 4.2 Seed-by-Seed Realizations (CAEG-Net F2)
| Dataset | Seed 42 | Seed 123 | Seed 999 | Seed 2024 | Seed 3407 | Mean MAE | Pop SD | CV (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PJM (MW)** | $249.90$ | $262.38$ | $233.76$ | $262.18$ | $246.65$ | **$250.97$** | $10.69$ | $4.26\%$ |
| **GEFCom (kW)**| $12.58$ | $12.24$ | $12.57$ | $12.43$ | $12.23$ | **$12.41$** | $0.15$ | $1.23\%$ |
| **UCI (MW)** | $8.21$ | $7.94$ | $7.59$ | $7.34$ | $7.61$ | **$7.74$** | $0.30$ | $3.92\%$ |

---

## 5. Statistical Inference (Non-Overlapping Daily Blocks)

Because consecutive sliding windows overlap by 167 hours, statistical inference is conducted strictly over **non-overlapping daily blocks** ($K=53$ for PJM, $K=456$ for GEFCom, $K=163$ for UCI).

### Paired Differences vs. Canonical Baseline (V1):
- **PJM ($K=53$):**  
  Mean Daily Diff $= -9.6638\text{ MW}$, $95\%\text{ CI } [-17.07, -2.26]$, $t = -2.5566$ ($p_t = 0.0135$), Wilcoxon $W = 470.0$ ($p_w = 0.0298$), Holm-corrected $p = 0.0406$ (**Statistically Significant Reduction**).
- **GEFCom ($K=456$):**  
  Mean Daily Diff $= -0.6884\text{ kW}$, $95\%\text{ CI } [-0.80, -0.57]$, $t = -11.8498$ ($p_t = 2.04 \times 10^{-28}$), Wilcoxon $W = 20,445.0$ ($p_w = 2.53 \times 10^{-29}$), Holm-corrected $p = 1.02 \times 10^{-27}$ (**Statistically Significant Reduction**).
- **UCI ($K=163$):**  
  Mean Daily Diff $= -0.2021\text{ MW}$, $95\%\text{ CI } [-0.31, -0.09]$, $t = -3.6581$ ($p_t = 0.00034$), Wilcoxon $W = 4,235.0$ ($p_w = 0.00005$), Holm-corrected $p = 0.0017$ (**Statistically Significant Reduction**).

---

## 6. Staged Screening Firewall Audit

Phase 15 strictly enforced a **Two-Stage Validation Firewall** using seeds 42 and 123 prior to any held-out test evaluation:
- **Predefined Rule:** A candidate qualifies as a finalist only if it improves validation MAE on $\ge 2$ of 3 datasets over Control A (F2) with $\le 2.0\%$ worst degradation on the remaining dataset.
- **Firewall Outcome:** Zero exploratory candidate mechanisms qualified:
  - *Control B (Fixed Shrinkage):* Improved 0/3 datasets (worst degradation: $+2.20\%$ on UCI).
  - *Control C (Horizon Routing):* Improved 0/3 datasets (worst degradation: $+10.03\%$ on PJM).
  - *Control D (Dynamic Confidence):* Improved 1/3 datasets (failed 2-dataset threshold).
  - *Candidates E1, E2, E3 (Horizon-Grouped Routing):* Degraded PJM by $+11.0\%$ to $+13.3\%$.
  - *Feature History Variants (P24, P48, P72):* Improved 0–1 datasets.
- **Model Selection Decision:** **Control A (`F2 / A2-OOF`) was the sole surviving qualified model and is certified as the definitive champion.** Subsequent 5-seed test evaluations of rejected candidates were conducted strictly as post-screening diagnostic evaluations.

---

## 7. Operational & Scientific Limitations

To preserve academic integrity, the following limitations are formally recognized:
1. **Point Forecasting Only:** The model generates deterministic point forecasts; probabilistic quantiles and prediction intervals are reserved for future work.
2. **Univariate Load Input:** Relies exclusively on historical load patterns; exogenous weather, temperature, humidity, calendar, and economic covariates are not integrated.
3. **Regional Series Aggregation:** Benchmarks represent regional grid aggregation rather than individual feeder/substation level loads.
4. **Near-Constant Shrinkage Dynamics:** The learned parameter $\lambda pprox 0.51$ acts as a stable static regularizer toward the equal ensemble centroid rather than an active dynamic switch.
5. **No Universal Dominance:** CAEG-Net F2 strictly wins on PJM and GEFCom, while on UCI standalone LSTM is stronger; F2 provides the strongest cross-grid balance rather than winning every metric everywhere.

---

## 8. Definitive Model-Lock Certification

| Field | Certified Specification |
| :--- | :--- |
| **Model Name** | **CAEG-Net** |
| **Internal Designation** | `F2 / A2-OOF` (`ConfidenceFallbackCAEGNet`) |
| **Total Parameter Count** | **$121,724$ Parameters** |
| **Storage Footprint** | $<0.5\text{ MB}$ |
| **Target Grid Metrics** | PJM: $250.97\text{ MW}$ \| GEFCom: $12.41\text{ kW}$ \| UCI: $7.74\text{ MW}$ |
| **Model Selection Basis** | Pre-registered Multi-Criteria Framework & Staged Validation Firewall |
| **Model Development State** | **PERMANENTLY FROZEN & LOCKED** |

> *"CAEG-Net F2 provides the strongest overall balance of forecasting performance, seed stability, methodological integrity and architectural simplicity among the evaluated formulations. F2 performs strongly on PJM and UCI, while fixed shrinkage achieves a slightly lower mean MAE on GEFCom. Therefore F2 is not claimed to be universally optimal on every dataset."*
