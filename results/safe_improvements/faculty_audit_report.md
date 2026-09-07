# CAEG-Net: Final Pre-Defense Forensic Audit & Model Improvement Report

**Generated Date:** September 7, 2026  
**Auditor:** Antigravity Advanced Agentic Pair Programmer  
**Project:** CAEG-Net (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting)  
**Target Submission / Review:** M.Tech Final Project Faculty Defense  

---

## 1. Executive Summary & Final Champion Decision

In the final preparation phase prior to faculty demonstration:
1. A controlled, low-risk training optimization experiment was conducted evaluating learning rates ($\eta \in \{10^{-3}, 5 \times 10^{-4}\}$) and objective functions (MSELoss vs. HuberLoss with $\delta=1.0$) around the canonical CAEG-Net V1 configuration.
2. An analytical inquiry into expanding context features with weekly periodicity ($k=168$) confirmed that within a frozen $168$-hour lookback, lag-$168$ autocorrelation has mathematically zero available pairs and cannot be computed without lookback distortion.
3. Under strict zero-leakage protocol (model selection based exclusively on validation performance), **Candidate OPT-1-V1-Baseline (AdamW $\eta=10^{-3}$, MSE Loss)** achieved the lowest Validation MAE ($411.16 \text{ MW}$), outperforming lower learning rates ($443.98 \text{ MW}$) and Huber loss variants ($416.91 \text{ MW}$).

### Final Decision:
- **PRIMARY PREDICTIVE CHAMPION:** **CAEG-Net V1** ($121,531$ parameters, Global Context Gating, MSE Loss).  
  $$\text{Test MAE} = 251.44 \pm 9.74 \text{ MW} \quad \mid \quad \text{Test RMSE} = 334.32 \pm 11.09 \text{ MW} \quad \mid \quad R^2 = 0.8723 \pm 0.0086$$
- **EXPERIMENTAL / ABLATION VARIANT:** **CAEG-Net V2** ($125,232$ parameters, Horizon-Dependent Matrix Gating).  
  $$\text{Test MAE} = 255.72 \pm 9.76 \text{ MW} \quad \mid \quad \text{Test RMSE} = 340.38 \pm 10.54 \text{ MW} \quad \mid \quad R^2 = 0.8677 \pm 0.0081$$
  *(Retained strictly as an interpretability and ablation variant illustrating step-specific diurnal weight transitions across $h=1 \dots 24$.)*

---

## 2. Controlled Training Optimization Experiments

Conducted on training windows ($N=5,957$) with early stopping (patience $= 6$ epochs, max epochs $= 25$) and evaluated on Validation ($N=1,294$) and Test ($N=1,294$) partitions:

| Candidate ID | Configuration Description | Actual Epochs | Best Epoch | Best Val Loss (scaled) | Val MAE (MW) | Test MAE (MW) | Test RMSE (MW) | Test $R^2$ | Test MAPE (%) | Outcome / Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **OPT-1-V1-Baseline** | AdamW $\eta=10^{-3}$, MSE Loss | 10 | 4 | $0.37603$ | **$411.16$** | $255.29$ | $343.14$ | $0.8656$ | $4.80\%$ | **CHAMPION: Lowest Val MAE** |
| **OPT-2-Lower-LR** | AdamW $\eta=5 \times 10^{-4}$, MSE Loss | 10 | 4 | $0.44290$ | $443.98$ | $258.48$ | $346.75$ | $0.8628$ | $4.77\%$ | Rejected: $+32.82 \text{ MW}$ worse on Val |
| **OPT-3-Huber-1e3** | AdamW $\eta=10^{-3}$, Huber ($\delta=1.0$) | 10 | 4 | $0.18568$ | $416.91$ | $248.32$ | $331.79$ | $0.8744$ | $4.63\%$ | Rejected: $+5.75 \text{ MW}$ worse on Val |
| **OPT-4-Huber-5e4** | AdamW $\eta=5 \times 10^{-4}$, Huber ($\delta=1.0$) | 10 | 4 | $0.19357$ | $429.04$ | $267.03$ | $354.09$ | $0.8569$ | $4.96\%$ | Rejected: $+17.88 \text{ MW}$ worse on Val |

*Key Insight:* Although Huber loss with $\eta=10^{-3}$ yielded slightly lower test error on Seed 42, its validation loss did not surpass MSE loss during early-stopped selection. In adherence to strict scientific rigor, we do not perform post-hoc model selection using test data. Baseline V1 with MSE loss remains the validated champion.

---

## 3. Investigation of Context Feature Expansion (Weekly Periodicity)

- **Hypothesis:** Augmenting the 4D context vector $\mathbf{C}_t$ with an explicit weekly periodicity indicator ($k = 168$).
- **Mathematical Audit:**
  $$\text{Lookback Window Length } L = 168 \text{ hours}$$
  $$\text{Lag Index } k = 168 \implies \text{Available Pairs } (z_i, z_{i-168}) = 168 - 168 = 0$$
- **Finding:** Within a fixed 168-hour lookback, lag-168 autocorrelation has zero degrees of freedom. Computing it would require expanding the lookback to at least $336$ hours (2 weeks), fundamentally modifying the input tensor $[B, 168, 1]$ and breaking the architecture of all three neural experts.
- **Decision:** **Skipped without faking.** Preserved the verified, compact 4D context vector:
  $$\mathbf{C}_t = [\text{Trend}, \text{Volatility}, \text{Lag-24 Autocorrelation}, \text{Causal Recent Forecast Error}]^T \in \mathbb{R}^4$$

---

## 4. Final Performance Benchmark Table Across All Models (5-Seed Aggregate)

Evaluated across all $1,294$ test windows on the original Megawatt (MW) scale:

| Model Architecture | Parameters | Test MAE (MW) | Test RMSE (MW) | Test $R^2$ Score | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence (Naive-24)** | Baseline | $285.19 \pm 0.00$ | $388.86 \pm 0.00$ | $0.8274 \pm 0.0000$ | $5.31 \pm 0.00\%$ |
| **LSTM Standalone** | 56,152 | $301.45 \pm 19.17$ | $414.81 \pm 21.86$ | $0.8032 \pm 0.0204$ | $5.57 \pm 0.38\%$ |
| **TCN Standalone** | 36,952 | $259.86 \pm 8.54$ | $351.37 \pm 11.19$ | $0.8590 \pm 0.0089$ | $4.80 \pm 0.16\%$ |
| **CNN Standalone** | 27,400 | $469.53 \pm 58.34$ | $628.85 \pm 61.63$ | $0.5452 \pm 0.0896$ | $8.85 \pm 1.24\%$ |
| **Static Equal Ensemble** | 120,504 | $295.48 \pm 15.54$ | $404.24 \pm 14.61$ | $0.8133 \pm 0.0136$ | $5.48 \pm 0.33\%$ |
| **Standard Input MoE** | 126,011 | $276.30 \pm 13.64$ | $370.01 \pm 17.26$ | $0.8435 \pm 0.0145$ | $5.15 \pm 0.24\%$ |
| **CAEG-Net (No Recent Error)** | 121,515 | $254.55 \pm 16.10$ | $339.43 \pm 19.53$ | $0.8682 \pm 0.0152$ | $4.75 \pm 0.34\%$ |
| **Full CAEG-Net V1 (Champion)** | **121,531** | **$251.44 \pm 9.74$** | **$334.32 \pm 11.09$** | **$0.8723 \pm 0.0086$** | **$4.71 \pm 0.21\%$** |
| **CAEG-Net V2 (Horizon Variant)**| 125,232 | $255.72 \pm 9.76$ | $340.38 \pm 10.54$ | $0.8677 \pm 0.0081$ | $4.77 \pm 0.22\%$ |

---

## 5. Summary of Ablations and Statistical Tests

1. **Full CAEG-Net V1 vs. CAEG-Net Without Recent Error:**
   - Improvement: $+3.12 \pm 15.57 \text{ MW}$ MAE reduction; $39.52\%$ reduction in cross-seed variance.
   - Paired test: $p = 0.6776$ (improved in 3 of 5 seeds). Not statistically significant.
2. **Full CAEG-Net V1 vs. Standard Input MoE:**
   - Advantage: $+24.87 \pm 21.34 \text{ MW}$ lower mean MAE; CAEG-Net wins in 4 of 5 seeds.
   - Paired $t$-test: $p = 0.0597$ (marginally misses $\alpha = 0.05$).
3. **Full CAEG-Net V1 vs. Static Equal Ensemble:**
   - Advantage: $+44.04 \pm 16.78 \text{ MW}$ lower mean MAE.
   - Paired $t$-test: $t = -5.87, p = 0.0039$ (statistically significant at $p < 0.01$).
4. **Full CAEG-Net V1 vs. CAEG-Net V2:**
   - Difference: $+4.28 \pm 13.87 \text{ MW}$ ($p = 0.5279$, not statistically significant). V1 remains champion.

---

## 6. Faculty Review Notebook Verification Status

- **Path:** `notebooks/CAEG_Net_Faculty_Review.ipynb`
- **Execution Verification:** Executed top-to-bottom via `jupyter nbconvert --to notebook --execute --inplace` from a clean kernel.
- **Exit Code:** $0$ (zero errors, all 23 cells executed cleanly).
- **All 8 Required Visualizations Rendered:**
  1. Architecture Flowchart (ASCII block diagram)
  2. Training vs. Validation Loss Curves (with best epoch marked)
  3. Actual vs. Predicted 24-Hour Forecast Profile (MW labeled)
  4. Forecast Residual / Error Curve (MW labeled)
  5. Model MAE Comparison Bar Chart (9 models color-coded)
  6. Model RMSE Comparison Bar Chart (9 models color-coded)
  7. CAEG V1 Expert Routing Allocation Chart (40.5% LSTM, 30.8% TCN, 28.8% CNN)
  8. V2 Horizon-Dependent Routing Weights Trajectory ($h=1 \dots 24$)
