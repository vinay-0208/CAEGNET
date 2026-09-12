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

## 3. Dataset Acquisition & Local Placement Instructions

In compliance with repository hygiene and data copyright governance, raw external dataset files are excluded from git tracking. To replicate local ingestion from scratch:

### 1. PJM Interconnection (Regional US Transmission Grid)
- **Source:** Download hourly metered load records from [PJM Data Miner 2](https://dataminer2.pjm.com/) (hourly load archive 2023–2024).
- **Target Local Path:** `data/Modern_PJM/pjm_load.csv`
- **Schema:** Requires a datetime column (e.g. `Datetime` or `timestamp`) and a numerical load demand column in MW (e.g. `load`).

### 2. GEFCom2014 (Zonal Competition Grid)
- **Source:** IEEE DataPort / Global Energy Forecasting Competition 2014 electric load track.
- **Target Local Path:** `data/gefcom/load.csv` (or pre-extracted zone series).

### 3. UCI Electricity (Portuguese Consumer Aggregation)
- **Source:** [UCI Machine Learning Repository: ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
- **Target Local Path:** `data/uci/LD2011_2014.txt`

---

## 4. Reproducing Data Ingestion

To verify data preprocessing, causal splitting, scaling, and window generation using the modular package:

```python
from src.data import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
)

# Example: Ingest PJM dataset once placed locally
# If raw CSV is not present, generate_synthetic_load_data() can be used for pipeline testing
file_path = "data/Modern_PJM/pjm_load.csv"
df, diagnostics = load_and_clean_data(file_path)

train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)

print(f"Test input windows X: {windows['test']['X'].shape}")
print(f"Test target windows Y: {windows['test']['Y'].shape}")
```
