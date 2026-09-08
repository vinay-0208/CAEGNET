# CAEG-Net Phase 5: Statistical Methodology Audit & Correction
**Document**: `PHASE5_STATISTICAL_AUDIT.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: AUDIT COMPLETE  

---

## 1. Executive Summary & Purpose

Earlier phases in this project reported statistical tests (paired $t$-tests and Diebold-Mariano tests) with contradictory conclusions—in some instances displaying $|\text{DM}| > 15$ ($p \approx 0$) alongside paired $t$-test $p$-values $> 0.30$. 

This document identifies the mathematical and methodological causes of these discrepancies and establishes the **corrected statistical evaluation protocol** for Phase 5.

---

## 2. Core Statistical Pathologies Identified

### 2.1 The Overlapping Forecast Fallacy
- For day-ahead forecasting with horizon $H=24$ and sliding step $s=1$ hour, forecast origins $t$ and $t+1$ predict horizons $[t+1 : t+24]$ and $[t+2 : t+25]$, sharing 23 hours of common realization.
- Errors $e_t$ and $e_{t+1}$ are heavily serially autocorrelated by construction (MA(23) process).
- **Flaw**: Treating each hourly forecast origin as an independent sample ($N=1,272$) artificially deflates the standard error of the mean difference by $\sqrt{24} \approx 4.9\times$, producing inflated $t$-statistics and false-positive significance claims.
- **Correction**: Aggregate hourly evaluations into **$K=53$ non-overlapping daily blocks** ($53 \text{ days} \times 24 \text{ hours} = 1,272 \text{ hours}$). Each daily block represents a distinct, non-overlapping 24-hour day-ahead operational period.

### 2.2 Re-Derivation of Diebold-Mariano & Harvey-Leybourne-Newbold Correction
Let $d_t = L(e_{1, t}) - L(e_{2, t})$ be the loss differential sequence (e.g., absolute error difference).

1. **Spectral Variance Estimator**:
   The long-run variance of $\bar{d}$ must account for autocovariances up to truncation lag $h-1$:
   $$\hat{\gamma}_k = \frac{1}{n} \sum_{t=k+1}^n (d_t - \bar{d})(d_{t-k} - \bar{d})$$
   $$\hat{V}(\bar{d}) = \frac{1}{n} \left[ \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k \right]$$
   For daily block aggregates ($h=1$), $\hat{V}(\bar{d}) = \hat{\gamma}_0 / n$.
2. **Harvey-Leybourne-Newbold (HLN) Small-Sample Correction**:
   To correct for severe size distortions in finite samples ($n=53$), the HLN adjustment is applied:
   $$\text{DM}_{\text{HLN}} = \text{DM} \cdot \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$
   The resulting statistic is evaluated against Student's $t$ with $n-1 = 52$ degrees of freedom.

### 2.3 Seed-Level vs. Time-Origin Independence
- **5 Random Seeds** ($[42, 123, 2024, 3407, 999]$) quantify stochastic training variance on the **same single dataset**.
- They are **NOT** 5 independent realizations of an underlying stochastic process.
- Seed results must be summarized as $\text{Mean} \pm \text{Std}$ over seeds to quantify model stability, while hypothesis testing must be conducted across the temporal test blocks.

### 2.4 Multiple Comparison Control
When comparing an improved candidate against $M$ baselines (Equal Ensemble, Standalone TCN, Standalone LSTM, Standalone CNN, Input MoE, V1), the **Holm-Bonferroni step-down procedure** must be applied:
$$\alpha_k = \frac{0.05}{M - k + 1}, \quad k = 1, \dots, M$$
This controls the family-wise error rate (FWER) below $0.05$.
