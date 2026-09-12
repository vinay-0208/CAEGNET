# Dataset Documentation & Preprocessing Protocols

This directory contains metadata and instructions for the three primary power grid benchmark datasets used in **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting).

> **Important License & Repository Notice:**  
> In accordance with data copyright and repository hygiene practices, raw external benchmark dataset archives are **not committed** to the Git repository. Preprocessed cache directories (`data/pjm/`, `data/gefcom/`, `data/uci/`) are populated during initial pipeline setup using publicly available public sources described below.

---

## 0. Dataset Acquisition Sources

- **PJM Interconnection:** Hourly metered load archives available directly from the PJM Data Miner portal ([pjm.com](https://dataminer2.pjm.com/)).
- **GEFCom2014:** Global Energy Forecasting Competition 2014 electric load tracks available on Kaggle / IEEE DataPort.
- **UCI Electricity Load Diagrams 2011–2014:** Available from the UCI Machine Learning Repository ([archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)).


## 1. Benchmark Grid Cohorts Overview

| Dataset | Grid Type / Geography | Raw Resolution | Time Span | Total Hours | Physical Unit | Preprocessing & Aggregation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PJM Interconnection** | Regional US Transmission Grid (Mid-Atlantic) | 1-Hour | 2023-01-01 to 2024-01-01 (Leap Year) | 8,784 | MW | Extracted from hourly metered system-wide load; verified continuous time index. |
| **GEFCom2014** | Global Energy Forecasting Competition (Zonal) | 1-Hour | Multi-year zonal competition history | 78,888 | kW | Standard benchmark competition zonal load series; standard imputation for missing records. |
| **UCI Electricity** | Portuguese Consumer Meter Demand | 15-Minute | 2011-01-01 to 2014-12-31 | 26,304 | MW | 370 client meters aggregated to system-wide hourly demand; sum divided by 1,000 for MW scale. |

---

## 2. Leakage-Free Causal Protocol

All data pipelines in CAEG-Net strictly adhere to a **zero-leakage temporal design**:

1. **Chronological Splitting:**
   - **Training Set:** First 70% of chronological records.
   - **Validation Set:** Next 15% of chronological records (used for model screening, checkpoint selection, early stopping).
   - **Held-Out Test Set:** Final 15% of chronological records (audited strictly once per candidate model).
   - *No random shuffling or temporal folding is applied.*

2. **Train-Only Parameter Estimation:**
   - The feature scaler (`StandardScaler`) is fitted strictly on the **training set**:
     $$\mu = \text{mean}(y_{\text{train}}), \quad \sigma = \text{std}(y_{\text{train}})$$
   - Validation and test splits are transformed using training parameters $(\mu, \sigma)$. No test data informs scaling.

3. **Causal Horizon Framing:**
   - Input feature lookback: $L = 168$ hours (exact 7-day lookback).
   - Forecast target lead: $H = 24$ hours (day-ahead scheduling horizon).
   - Sliding window step: $\Delta t = 1$ hour.
   - Ground-truth target horizons $[t+1, \dots, t+24]$ are strictly occluded from the model input and context-gating router.

4. **Out-of-Fold (OOF) Causal Residual Generation:**
   - For variant `A2-OOF`, historical performance residuals are derived strictly from validation/preceding partitions without access to future test targets.

---

## 3. Directory Structure

```text
data/
├── README.md               # This document
├── pjm/                    # PJM hourly series cache / artifacts
├── gefcom/                 # GEFCom2014 hourly benchmark data
└── uci/                    # UCI Electricity hourly aggregated dataset
```

---

## 4. Reproducing Data Ingestion

To verify data preprocessing and window generation locally:

```python
from data_utils import load_dataset, create_sliding_windows

# Example: Load PJM benchmark series
df_pjm = load_dataset("pjm")
X_train, y_train, X_val, y_val, X_test, y_test = create_sliding_windows(
    df_pjm, lookback=168, horizon=24, train_ratio=0.70, val_ratio=0.15
)
print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")
```
