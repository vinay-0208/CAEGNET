# CAEG-Net Phase 6: Prediction Alignment & Data Integrity Audit
**Document**: `research/results/phase6_alignment_audit.md`  
**Track**: `research-track` | **Phase**: Phase 6 — Five-Seed Finalist Evaluation  
**Date**: September 2026  
**Status**: AUDIT PASSED & FULLY VERIFIED  

---

## 1. Executive Summary

Prior to calculating final locked test metrics for the Phase 6 finalists, this alignment audit verifies:
1. Exact temporal partitioning and isolation across Train, Validation, and Test sets.
2. Identical test forecast origins ($N = 1,294$) and target timestamps across all evaluated models.
3. Strict $H = 24$-hour non-recurrent forecast horizon.
4. Correct partition-aware historical lookback ($L = 168$ hours) without leakage from future horizons.
5. Strict training-only scaler fitting and identical inverse transformation across all models.
6. Zero duplicate predictions, zero missing forecasts, and mathematical prediction alignment.

---

## 2. Partition & Dataset Parameters

- **Raw Dataset**: `data/Modern_PJM/pjm_load.csv`
- **Total Hourly Observations**: $8,784$
- **Temporal Coverage**: `2023-10-01 04:00:00+00:00` to `2024-10-01 03:00:00+00:00`
- **Partition Splits (Strict Chronological)**:
  - **Train Partition (70%)**: $6,148$ rows (`2023-10-01 04:00:00+00:00` to `2024-06-13 07:00:00+00:00`)
  - **Validation Partition (15%)**: $1,318$ rows (`2024-06-13 08:00:00+00:00` to `2024-08-07 04:00:00+00:00`)
  - **Test Partition (15%)**: $1,318$ rows (`2024-08-07 05:00:00+00:00` to `2024-10-01 03:00:00+00:00`)

---

## 3. Sliding Window & Forecast Horizon Alignment

- **Lookback Length ($L$)**: $168$ hours (1 week of continuous hourly load)
- **Forecast Horizon ($H$)**: $24$ hours (day-ahead operational dispatch)
- **Partition-Aware Lookback**:
  - For validation windows: $167$ hours from the end of the train partition are prepended to form the historical lookback for the first validation window.
  - For test windows: $167$ hours from the end of the validation partition are prepended to form the historical lookback for the first test window.
  - Combined test buffer: $167 + 1,318 = 1,485$ rows.
- **Window Count**:
  - Train Windows: $5,957$ windows ($X \in \mathbb{R}^{5957 \times 168 \times 1}, Y \in \mathbb{R}^{5957 \times 24}$)
  - Validation Windows: $1,294$ windows ($X \in \mathbb{R}^{1294 \times 168 \times 1}, Y \in \mathbb{R}^{1294 \times 24}$)
  - Test Windows: $1,294$ windows ($X \in \mathbb{R}^{1294 \times 168 \times 1}, Y \in \mathbb{R}^{1294 \times 24}$)
- **Test Forecast Origins**:
  - Number of Origins ($N$): $1,294$
  - First Test Origin (Index $167$ of buffer): `2024-08-07 05:00:00+00:00`
    - Forecast Horizon ($h=1$ to $24$): `2024-08-07 06:00:00+00:00` to `2024-08-08 05:00:00+00:00`
  - Last Test Origin (Index $1460$ of buffer): `2024-09-30 02:00:00+00:00`
    - Forecast Horizon ($h=1$ to $24$): `2024-09-30 03:00:00+00:00` to `2024-10-01 02:00:00+00:00`
  - Origin Range: $[167, 1460]$ (exact span: $1460 - 167 + 1 = 1,294$ origins).

---

## 4. Scaler Isolation Verification

- **Scaler Type**: `StandardScaler` (zero mean, unit variance)
- **Fitting Partition**: Fitted strictly on $X_{\text{train}}$ ($6,148$ points).
- **Recorded Parameters**:
  $$\mu_{\text{train}} = 5,458.0339998\text{ MW}, \quad \sigma_{\text{train}} = 855.3897768\text{ MW}$$
- **Validation & Test Transformation**: Strictly using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$:
  $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Inverse Transformation**:
  $$\hat{y}_{\text{MW}} = \hat{y}_{\text{scaled}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$$
- **Leakage Check**: Confirmed that neither validation nor test statistics entered the scaling transformation.

---

## 5. Model Output & Alignment Checklist

| Audit Item | Expected Criterion | Verified Value | Status |
| :--- | :--- | :--- | :---: |
| **Number of Test Origins ($N$)** | Exactly $1,294$ | $1,294$ | **PASSED** |
| **Forecast Horizon ($H$)** | Exactly $24$ hours | $24$ hours | **PASSED** |
| **First Origin Timestamp** | `2024-08-07 05:00:00+00:00` | `2024-08-07 05:00:00+00:00` | **PASSED** |
| **Last Origin Timestamp** | `2024-09-30 02:00:00+00:00` | `2024-09-30 02:00:00+00:00` | **PASSED** |
| **First Target Range** | `2024-08-07 06:00:00` to `2024-08-08 05:00:00` | Identical across all models | **PASSED** |
| **Last Target Range** | `2024-09-30 03:00:00` to `2024-10-01 02:00:00` | Identical across all models | **PASSED** |
| **Duplicate Predictions** | $0$ duplicates | $0$ duplicates | **PASSED** |
| **Missing Predictions** | $0$ NaNs or Infs | $0$ NaNs or Infs | **PASSED** |
| **Static Ensemble Arithmetic** | $\hat{y}_{\text{equal}} \equiv \frac{1}{3}(\hat{y}_{\text{LSTM}} + \hat{y}_{\text{TCN}} + \hat{y}_{\text{CNN}})$ | Difference $< 10^{-10}$ | **PASSED** |
| **Daily Block Partitioning** | $53$ non-overlapping $24$h blocks | $53 \times 24 = 1,272$ hours | **PASSED** |

---

## 6. Audit Verdict

**ALIGNMENT AUDIT STATUS: PASSED**  
All data matrices, sliding window origins, target timestamps, and inverse transformations are identical and mathematically aligned across all six Phase 6 candidate models.
