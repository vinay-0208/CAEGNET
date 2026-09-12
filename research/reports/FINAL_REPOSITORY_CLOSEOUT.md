# CAEG-Net Final Repository Closeout & Consistency Report

**Project:** CAEG-Net — Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting  
**Author:** Vinay Vishwanathan  
**Department:** Advanced Predictive Analytics  
**Date:** 2026-09-12  
**Certification Status:** **FINAL MODEL LOCKED — DEVELOPMENT COMPLETE**  

---

## 1. Final Public Model Identity & Provenance

- **Public Model Name:** **CAEG-Net** (Context-Adaptive Expert Gating Network)
- **Internal Experimental Identifier:** `F2 / A2-OOF` (`ConfidenceFallbackCAEGNet`)
- **PyTorch Model Class:** `ConfidenceFallbackCAEGNet` (implemented in `caeg_net.py`, `src/models/`, and `research/models_phase15b.py`)
- **Model Development Status:** **CLOSED**. No further architectural optimization or retraining is planned.

---

## 2. Architecture & Parameter Verification

The final locked model architecture has been strictly audited and verified against the canonical PyTorch implementation:

| Component | Architecture Specification | Trainable Parameters | Parameter Share |
| :--- | :--- | :---: | :---: |
| **LSTM Expert** | 2-layer Recurrent Neural Network (hidden dim $= 64$, dropout $= 0.1$) | $56,152$ | $46.12\%$ |
| **TCN Expert** | 6-stage Dilated Causal Residual Conv (32 channels, kernel $= 3$, dilations $[1, 2, 4, 8, 16, 32]$, RF $= 253$h) | $36,952$ | $30.36\%$ |
| **CNN Expert** | 3-stage 1D Multi-Kernel Conv (kernels $[3, 5, 3]$, channels $[32, 64, 64]$, adaptive pooling) | $27,400$ | $22.51\%$ |
| **Backbone Subtotal** | **Unified Three-Expert Temporal Feature Extractors** | **$120,504$** | **$98.99\%$** |
| **Context-Adaptive Router** | 7D context $\to$ 16D latent embedding $\to$ 3-class Softmax convex gating | $1,075$ | $0.88\%$ |
| **Confidence Fallback Head** | 7D context $\to$ 16D latent embedding $\to$ Sigmoid shrinkage scalar $\lambda$ | $145$ | $0.12\%$ |
| **TOTAL CAEG-Net** | **End-to-End Champion Formulation (`F2_A2_OOF`)** | **$121,724$** | **$100.00\%$** |

### Context Vector Specification ($\mathbf{c} \in \mathbb{R}^7$)
1. `trend`: Linear slope over the 168-hour lookback window.
2. `volatility`: Sample standard deviation over the 168-hour lookback window.
3. `lag-24 autocorrelation`: Autocorrelation at diurnal lag 24.
4. `causal recent forecast error`: Trailing 24-hour absolute forecast error of the preceding forecast window.
5. `causal out-of-fold relative error (LSTM)`: $r_{\text{LSTM}}(t) = e_{\text{LSTM}} / (e_{\text{LSTM}} + e_{\text{TCN}} + e_{\text{CNN}} + \epsilon)$.
6. `causal out-of-fold relative error (TCN)`: $r_{\text{TCN}}(t) = e_{\text{TCN}} / (e_{\text{LSTM}} + e_{\text{TCN}} + e_{\text{CNN}} + \epsilon)$.
7. `causal out-of-fold relative error (CNN)`: $r_{\text{CNN}}(t) = e_{\text{CNN}} / (e_{\text{LSTM}} + e_{\text{TCN}} + e_{\text{CNN}} + \epsilon)$.

---

## 3. Final Benchmark Results

Evaluated across 5 random seeds (`[42, 123, 999, 2024, 3407]`) under a strict chronological 70/15/15 split with train-only StandardScaler:

| Benchmark Grid | Grid Classification & Geography | Authoritative Test MAE (Mean ± SD) | Test RMSE | Test $R^2$ Score | Multi-Seed Stability (CV) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PJM Interconnection** | Regional US Transmission Grid (Mid-Atlantic) | **$250.9747 \pm 10.6938$ MW** | $335.3822$ MW | $0.8714$ | $4.26\%$ |
| **GEFCom2014** | Zonal Competition Network | **$12.4077 \pm 0.1525$ kW** | $18.0446$ kW | $0.8610$ | $1.23\%$ |
| **UCI Electricity** | Aggregated Consumer Smart Meter Cohort | **$7.7371 \pm 0.3037$ MW** | $10.9556$ MW | $0.9831$ | $3.92\%$ |

*Dispersion definition: Population standard deviation (`ddof=0`).*

---

## 4. Baseline Comparison & Interpretation

| Model Architecture | PJM MAE (MW) | GEFCom MAE (kW) | UCI MAE (MW) | Parameters | Model Classification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Standalone LSTM** | $291.73$ | $13.23$ | **$7.55$ (Lowest)** | $56,152$ | Monolithic Recurrent Baseline |
| **Standalone TCN** | $259.33$ | $12.57$ | $8.34$ | $36,952$ | Monolithic Dilated Causal Baseline |
| **Standalone CNN** | $432.08$ | $14.50$ | $11.71$ | $27,400$ | Monolithic Ramp Baseline |
| **Static Equal Ensemble** | $279.83$ | $12.62$ | $8.17$ | $120,504$ | Fixed Uniform 1/3 Weighting |
| **Fixed Shrinkage Control** | $253.50$ | **$12.36$ (Lowest)** | $7.75$ | $121,579$ | Non-Adaptive Regularization |
| **CAEG-Net (`F2_A2_OOF`)** | **$250.97$ (Lowest)** | $12.41$ | $7.74$ | **$121,724$** | **Context-Adaptive Champion** |
| *Empirical Ex-Post Oracle* | *$234.12$* | *$11.20$* | *$6.95$* | — | *Non-deployable diagnostic reference* |

> **Comparative Scientific Position:**  
> CAEG-Net achieved the strongest overall performance–robustness balance among the evaluated formulations, although it was not the best-performing model on every individual dataset:
> - PJM: CAEG-Net is the strongest among evaluated models ($250.97$ MW).
> - GEFCom: Fixed shrinkage achieved slightly lower mean MAE ($12.36$ kW vs $12.41$ kW).
> - UCI: Standalone LSTM achieved slightly lower mean MAE ($7.55$ MW vs $7.74$ MW).
> CAEG-Net is selected as the best overall compromise balancing accuracy, multi-expert resilience, and seed stability.

---

## 5. Routing & Confidence Head Diagnostics

1. **Routing Weights:**
   - PJM: $w_{\text{LSTM}} = 0.3511, w_{\text{TCN}} = 0.3283, w_{\text{CNN}} = 0.3207$ ($N_{\text{eff}} = 2.9931$)
   - GEFCom: $w_{\text{LSTM}} = 0.3528, w_{\text{TCN}} = 0.2982, w_{\text{CNN}} = 0.3490$ ($N_{\text{eff}} = 2.9765$)
   - UCI: $w_{\text{LSTM}} = 0.3428, w_{\text{TCN}} = 0.2994, w_{\text{CNN}} = 0.3578$ ($N_{\text{eff}} = 2.9842$)
   - *Takeaway:* The model maintains a broadly distributed convex mixture across the three experts in the evaluated benchmarks ($N_{\text{eff}} \approx 2.98 - 2.99$).
2. **Confidence Parameter $\lambda$:**
   - PJM: $0.5066 \pm 0.0038$ ($CV = 0.75\%$)
   - GEFCom: $0.5170 \pm 0.0055$ ($CV = 1.06\%$)
   - UCI: $0.5064 \pm 0.0030$ ($CV = 0.59\%$)
   - *Takeaway:* The learned fallback coefficient remains close to $0.5$ with low temporal variation, functioning primarily as an empirical stabilization mechanism toward the equal-expert centroid.

---

## 6. Statistical Significance on Daily Blocks

Evaluated on non-overlapping daily blocks ($K=53, 456, 163$):
- **PJM ($K=53$):** Mean paired difference $= -9.66$ MW, paired $t$-test adjusted $p = 0.0406$, Wilcoxon $p = 0.0893$. Statistically significant under paired $t$-test.
- **GEFCom ($K=456$):** Mean paired difference $= -0.688$ kW, paired $t$-test adjusted $p = 1.02 \times 10^{-27}$, Wilcoxon $p = 1.27 \times 10^{-28}$. Statistically significant under both tests.
- **UCI ($K=163$):** Mean paired difference $= -0.202$ MW, paired $t$-test adjusted $p = 0.0017$, Wilcoxon $p = 0.00020$. Statistically significant under both tests.
- *Methodological note:* The two tests can differ on small samples (PJM), and statistical conclusions depend on the chosen inference procedure.

---

## 7. Faculty Review Notebook Audit

- **Authoritative Location:** `research/notebooks/CAEG_Net_Faculty_Review.ipynb`.
- **Duplicate Removal:** `notebooks/CAEG_Net_Faculty_Review.ipynb` was removed via `git rm`.
- **Total Cells:** 35 cells (23 structured sections, 7 code execution blocks, 18 complete Viva defense questions).
- **Execution Status:** Pre-executed top-to-bottom via `jupyter nbconvert --to notebook --execute --inplace` with **0 errors** (451 KB).
- **Zero-Fabrication Guarantee:** Prediction trajectory visualization is omitted because verified trajectory arrays were not archived; lead-time error degradation is evaluated directly from authoritative retrospective horizon statistics (`phase15b_horizon_results.csv`).

---

## 8. Interactive Research Dashboard

- **Main File:** `dashboard/app.py` (Streamlit).
- **Documentation:** `dashboard/README.md`.
- **Navigation Panels (11 total):**
  1. Overview (KPI cards, research question)
  2. Architecture (diagram, equations, parameter table)
  3. Dataset & Protocol (provenance, 70/15/15 split, train-only scaling)
  4. Benchmark Results (authoritative 5-seed metrics)
  5. Baseline Comparison (standalone, ensembles, oracle)
  6. Routing Behaviour (weights bar chart, $N_{\text{eff}}$)
  7. Confidence / Fallback ($\lambda$ mean and CV)
  8. Statistical Evidence (daily blocks, $t$-test and Wilcoxon)
  9. Horizon Analysis ($h=1 \dots 24$ step MAE curves)
  10. Research Findings (5 primary empirical conclusions)
  11. Limitations (transparent technical boundaries)

---

## 9. Verification & Quality Assurance

1. **Root Unit Tests:** 4 of 4 passed (`Ran 4 tests in 10.159s, OK`).
2. **Research Unit Tests:** 172 of 172 passed (`Ran 172 tests in 346.282s, OK`).
3. **Compileall:** 0 syntax errors across all repository Python source files.
4. **Security Scan:** 0 credentials, secrets, tokens, or private keys found.
5. **Duplicate Cleanup:** Removed duplicate faculty review notebook from `notebooks/` and archived historical early-phase scripts into `research/archive/historical_scripts/`.

---

## 10. Known Research Limitations

1. **Point Forecasting Only:** Generates deterministic point forecasts; prediction intervals are left to future work.
2. **Univariate Formulation:** Relies strictly on historical load profiles; weather and calendar covariates are not incorporated.
3. **Regional Aggregation:** Evaluated on regionally aggregated load series rather than nodal substation loads.
4. **Finite Seeds:** Five seeds measure initialization sensitivity rather than multi-year structural drift.
5. **Near-Constant Shrinkage:** The learned parameter $\lambda \approx 0.51$ acts primarily as an empirical static regularizer.
6. **Absence of Universal Dominance:** CAEG-Net is not claimed to win every individual benchmark, but achieves the strongest cross-dataset balance.
7. **Empirical Ex-Post Oracle:** The oracle is an exploratory, non-deployable diagnostic that uses future realized observations.

---

## 11. Final Git State

- **Branch:** `research-track`
- **Working Tree:** Clean
- **Closeout Declaration:** Model development is **CLOSED**. Next activities are paper writing, faculty presentation, and demonstration.
