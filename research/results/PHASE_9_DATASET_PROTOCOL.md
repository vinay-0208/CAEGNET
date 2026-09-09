# PHASE 9 — UCI ELECTRICITY LOAD DIAGRAMS: DATASET INSPECTION & GENERALIZATION PROTOCOL

**Document Version:** 1.0.0  
**Date:** September 9, 2026  
**Repository Branch:** `research-track`  
**Status:** Protocol Established — Awaiting Approval Before Neural Benchmark  

---

## Executive Summary

This document establishes the formal dataset inspection, semantic reconciliation, and experimental protocol for **Phase 9: Cross-Dataset Generalization on the UCI ElectricityLoadDiagrams20112014 Benchmark**.

In strict accordance with the canonical CAEG-Net research roadmap, this phase performs **dataset inspection and protocol design only**. No neural model training, architecture exploration, hyperparameter tuning, or test-set optimization is performed.

The canonical proposed research model remains the **Original CAEG-Net V1**:
- **Temporal Experts:** LSTM + TCN + CNN
- **History Lookback:** 168 hours (7 days)
- **Forecast Horizon:** 24 hours ahead
- **Gating Mechanism:** Context-Aware Adaptive Soft Gating with 4D dynamic context vector
- **Primary Research Metric:** Mean Absolute Error (MAE)
- **Training Loss / Checkpoint Selection:** Mean Squared Error (MSE)

---

## 1. Dataset Provenance

* **Dataset Name:** ElectricityLoadDiagrams20112014
* **Source Archive:** UCI Machine Learning Repository (Dataset ID: 321)
* **DOI:** [10.24432/C58C86](https://doi.org/10.24432/C58C86)
* **Donor:** Artur Trindade (2015), University of Porto / INESC TEC, Portugal
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Canonical Paper / Baseline Reference:** 
  - Trindade, A. (2015). *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository.
  - Lai, G., Chang, W.-C., Yang, Y., & Liu, H. (2018). *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (LSTNet)*. SIGIR 2018.
  - Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*. International Journal of Forecasting.

---

## 2. Raw File Description

* **Repository Location:** `data/ElectricityLoadDiagrams20112014/LD2011_2014.txt`
* **File Size:** $710,998,915$ bytes ($678.06\text{ MB}$)
* **SHA-256 Checksum:** `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`
* **File Encoding:** UTF-8
* **Delimiter:** Semicolon (`;`)
* **Header Structure:** 371 columns:
  - First column: Timestamp header (empty quoted string `""`)
  - Columns 2 through 371: Client identifiers `MT_001` through `MT_370`

> [!IMPORTANT]
> The raw dataset file remains completely unaltered and byte-preserved. In accordance with Step 16 git discipline, the `data/ElectricityLoadDiagrams20112014/` path is strictly excluded via `.gitignore`. No metadata from `__MACOSX` is utilized.

---

## 3. Dataset Structure & Dimensions

* **Total Raw Rows:** $140,256$ observation rows (plus 1 header row)
* **Total Columns:** $371$ ($1$ timestamp + $370$ client series)
* **Total Value Cells:** $140,256 \times 370 = 51,894,720$ individual measurements
* **Calendar Duration:** $1,461$ full calendar days ($4$ complete Gregorian years: 2011, 2012 [leap], 2013, 2014)
* **Daily Interval Structure:** Exactly $96$ fifteen-minute intervals per calendar day ($96 \times 1,461 = 140,256$)

---

## 4. Sampling Frequency & Temporal Spacing

* **Nominal Resolution:** 15 minutes ($900$ seconds)
* **Empirical Inter-Arrival Time Distribution:**
  - Exactly $140,255$ transitions of $900$ seconds ($100.00\%$)
  - $0$ duplicate timestamps
  - $0$ non-increasing / out-of-order timestamps
  - $0$ missing timestamps across the 4-year span
* **Spacing Property:** Strictly uniform.

---

## 5. Date Range & Boundaries

* **First Observation:** `2011-01-01 00:15:00`
* **Last Observation:** `2015-01-01 00:00:00` (representing the final 15-minute interval of calendar year 2014)
* **Timestamp Format:** Standard ISO string format `YYYY-MM-DD HH:MM:SS`

---

## 6. Number of Series & Client Heterogeneity

* **Number of Monitored Clients:** $370$ individual delivery points / electricity meters (`MT_001` to `MT_370`)
* **Client Heterogeneity:**
  - Micro-residential consumers (mean load $< 1\text{ kW}$, e.g. `MT_001` with mean $0.39\text{ kW}$)
  - Commercial / small industrial consumers (mean load $10\text{--}100\text{ kW}$)
  - Large industrial consumers (mean load $> 1,000\text{ kW}$, e.g. `MT_321` with mean $2,258.4\text{ kW}$, max $5,775.0\text{ kW}$)
* **Client Lifetime & Zero-Filling Findings:**
  - **Active from Row 0 (`2011-01-01 00:15:00`):** $158$ clients
  - **Active by End of 2011 (`2011-12-31`):** $160$ clients
  - **Active on `2012-01-01`:** $318$ clients
  - **Active by First Week of Jan 2012:** $321$ clients (the canonical cohort in deep learning forecasting literature)
  - **Added during 2012–2014:** $49$ clients (bringing the total to $370$)
  - **Always Zero:** $0$ clients (all 370 clients have non-zero load by the end of 2014)

---

## 7. Measurement Units & Physical Meaning

* **Official Documented Unit:** **Kilowatts (kW)** representing active power demand for each 15-minute interval.
* **Energy Equivalence:** Because each interval spans $\Delta t = 0.25\text{ hour}$, the energy consumed during interval $i$ is:
  $$E_i = P_i \times 0.25\text{ kWh} = \frac{P_i}{4}\text{ kWh}$$
* **Physical Dimension:** Power ($[\text{Energy}] / [\text{Time}]$).
* **Negative Values:** Exactly $0$ ($0.0\%$). No negative values or regenerative back-feeding exist.
* **Missing (NaN / Null) Values:** Exactly $0$ ($0.0\%$). The table is completely populated.

---

## 8. Data-Quality & Zero-Value Findings

* **Total Zero Values:** $10,457,342$ values out of $51,894,720$ ($20.15\%$)
* **Root Cause of Zeros:**
  1. *Client Pre-Creation Zeros:* As stated in official UCI metadata: *"Some clients were created after 2011. In these cases consumption were considered zero."*
  2. *March Daylight-Saving Adjustment:* Zero values between 01:00 and 02:00 on the spring-forward day.
  3. *Inactivity / Vacancy:* Legitimate zero power demand during meter shutdown or vacancy.
* **Non-Zero Values:** $41,437,378$ values ($79.85\%$)

---

## 9. Timestamp & Time-Zone Findings (Daylight Saving Time)

* **Reference Time Zone:** Portuguese Local Time (Western European Time / WEST, UTC+0 in winter, UTC+1 in summer).
* **DST Artifact Resolution:** The data provider normalized all days to exactly 96 rows:
  - *March Time Change (23-hour day):* Values between 01:00 and 02:00 were set to zero for all points to preserve the 96-interval grid.
  - *October Time Change (25-hour day):* Values between 01:00 and 02:00 aggregated two hours of consumption to preserve the 96-interval grid.
* **Impact on Forecasting:** Every calendar day has an identical 96-point length, making calendar alignment and 24-hour periodicity invariant.

---

## 10. 15-Minute $\to$ Hourly Transformation Decision

The canonical CAEG-Net formulation operates on **hourly observations** ($168\text{ hours}$ input $\to$ $24\text{ hours}$ forecast).

* **Scientific Justification:** Because the raw values represent average power in kW over $\Delta t = 0.25\text{ h}$, transforming to hourly resolution requires computing the average hourly power demand.
* **Mathematical Derivation:**
  For any hour $h$ comprising four consecutive 15-minute intervals $i \in \{1, 2, 3, 4\}$:
  $$E_h = \sum_{i=1}^4 \left( P_i \times 0.25\text{ h} \right) = 0.25 \sum_{i=1}^4 P_i\quad (\text{kWh})$$
  The average power during hour $h$ is:
  $$P_{\text{hour}} = \frac{E_h}{1.0\text{ h}} = \frac{1}{4} \sum_{i=1}^4 P_i\quad (\text{kW})$$
* **Conclusion:** Hourly aggregation via the **arithmetic mean of four consecutive 15-minute intervals** is mathematically exact and preserves physical energy conservation.
* **Resulting Time Series Length:** $140,256 / 4 = 35,064\text{ hours}$ ($1,461\text{ days} \times 24\text{ h/day}$).

---

## 11. Exact Aggregation Formula

For each client $c \in \{1, \dots, 370\}$ and each hour $t \in \{1, \dots, 35,064\}$:
$$P_{t, c}^{\text{hour}} = \frac{P_{4t-3, c}^{15\text{m}} + P_{4t-2, c}^{15\text{m}} + P_{4t-1, c}^{15\text{m}} + P_{4t, c}^{15\text{m}}}{4}$$

For the **System Aggregate Load** across all $C$ clients:
$$L_t^{\text{kW}} = \sum_{c=1}^{C} P_{t, c}^{\text{hour}}$$
$$L_t^{\text{MW}} = \frac{L_t^{\text{kW}}}{1000.0}$$

---

## 12. Series & Client Handling Decision

We evaluated the four candidate formulations specified in Step 4:

| Option | Formulation | Methodological Validity | Computational Feasibility | Direct CAEG Comparability |
| :--- | :--- | :---: | :---: | :---: |
| **Option A** | 370 Independent Client Models | Low (extreme sparsity/intermittency) | Infeasible ($9,250$ neural training runs) | Low (microscopic load vs macroscopic grid) |
| **Option B** | System Aggregate Load Series | **High (continuous macroscopic load)** | **High ($25$ neural training runs)** | **Highest (matches PJM & GEFCom2014)** |
| **Option C** | Multi-Client Pooled Evaluation | Moderate | Moderate-Low ($>1,000$ runs) | Moderate |
| **Option D** | Representative Client Cohort | Moderate | Moderate | Moderate (susceptible to selection bias) |

### Formal Recommendation: Option B — System Aggregate Load
1. **Research Question Alignment:** The core CAEG-Net question is short-term grid/system load forecasting. Summing all meters yields the total electrical demand of the monitored Portuguese distribution network.
2. **Comparability:** Allows exact 1-to-1 comparison with Modern PJM (RTO-level load) and GEFCom2014 (zonal utility load).
3. **Period Selection:**
   - **Full 4-Year Period (2011–2014, 35,064 hours):** Exhibits a structural level shift at Jan 1, 2012 due to 160 new clients onboarding.
   - **Canonical Literature Period (2012–2014, 26,304 hours):** Standard in literature (LSTNet, DeepAR, Autoformer) because the client pool is stable ($321$ active clients), yielding a stationary, continuous, seasonal macroscopic load series.

---

## 13. Chronological Split Protocol

Preserving the established CAEG-Net evaluation methodology:
* **Train Partition:** $70\%$ chronological ($18,413\text{ hours}$, ~2.1 years: `2012-01-01 01:00` to `2014-02-07 05:00`)
* **Validation Partition:** $15\%$ chronological ($3,945\text{ hours}$, ~5.4 months: `2014-02-07 06:00` to `2014-07-21 14:00`)
* **Test Partition:** $15\%$ chronological ($3,946\text{ hours}$, ~5.4 months: `2014-07-21 15:00` to `2014-12-31 00:00`)

> [!NOTE]
> All splits are established **strictly prior to sliding-window creation**. No random splitting or cross-validation shuffling is permitted.

---

## 14. 168 $\to$ 24 Window Protocol

* **Input Sequence ($X_t$):** Previous $168$ hourly observations ($y[t - 167 : t]$), shape `[N, 168, 1]`.
* **Forecast Target ($Y_t$):** Next $24$ hourly observations ($y[t + 1 : t + 24]$), shape `[N, 24]`.
* **Step Size:** $1$ hour (single-step rolling origin).
* **Causal Window Prepending:** For validation and test partitions, the preceding $167$ historical observations from the prior partition are prepended so that target evaluation begins precisely at the split boundary with zero discarded forecasts.

---

## 15. Scaling Protocol

* **Scaler Type:** `StandardScaler` (zero mean, unit variance).
* **Fitting Rule:** Strict in-sample fitting on the training partition:
  $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{t \in \text{train}} L_t, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{t \in \text{train}} (L_t - \mu_{\text{train}})^2}$$
* **Transformation:** Validation and test partitions are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
* **Target Inversion:** All loss and error metrics (MAE, RMSE, MAPE) are evaluated on original scale (MW or kW).

---

## 16. Context-Feature Protocol

The canonical 4D dynamic context vector $c_t \in \mathbb{R}^4$ is extracted strictly causally from historical window $X_t$:
1. **Trend ($c_{t, 1}$):** Normalized linear regression slope over the 168-hour historical lookback.
2. **Volatility ($c_{t, 2}$):** Standard deviation of the 168-hour input sequence.
3. **Lag-24 Periodicity ($c_{t, 3}$):** Autocorrelation at lag 24 within the historical window:
   $$\rho_{24} = \text{Corr}(X_t[0:144], X_t[24:168])$$
4. **Recent Forecast Error ($c_{t, 4}$):** Causal out-of-sample forecast error from a completed historical 24-hour forecast.

---

## 17. Recent-Error Protocol

* **Causal Principle:** $c_{t, 4}$ must use **strictly completed** forecasts whose entire 24-step horizon concluded at or before origin $t$.
* **Implementation:** Standard multi-step Ridge Regression expanding-window walk-forward forecaster from `data_utils.py`.
* **Information Boundary:** Zero future target information or test-period leakage enters the error feedback vector.

---

## 18. Planned Benchmark Models

In accordance with Step 10, the benchmark evaluates the canonical 5-model suite:
1. **Standalone LSTM:** Canonical 2-layer LSTM expert ($64$ hidden units).
2. **Standalone TCN:** Canonical Dilated Temporal Convolutional Network ($64$ channels, kernel $3$, dilations $[1, 2, 4, 8]$).
3. **Standalone CNN:** Canonical Multi-layer 1D CNN ($64$ filters, kernel $3$).
4. **Static Equal Ensemble:** Uniform average of the three standalone experts ($\frac{1}{3} \hat{y}_{\text{LSTM}} + \frac{1}{3} \hat{y}_{\text{TCN}} + \frac{1}{3} \hat{y}_{\text{CNN}}$).
5. **Original CAEG-Net V1:** Proposed architecture with LSTM + TCN + CNN experts, Context Encoder, Adaptive Soft Gate, and Weighted Fusion.

> [!WARNING]
> No V2, V3, GRU, Patch, Transformer, PatchTST, shrinkage, or decoupled routing variants will be introduced.

---

## 19. Primary & Secondary Metrics

* **Primary Research Metric:** **Mean Absolute Error (MAE)** in MW (or kW).
* **Training Objective:** Mean Squared Error (MSE) loss.
* **Checkpoint Selection:** Validation MSE (consistent with Phase 6A methodology audit).
* **Secondary Evaluation Metrics:**
  - Root Mean Squared Error (RMSE)
  - Mean Squared Error (MSE)
  - Coefficient of Determination ($R^2$)
  - Mean Absolute Percentage Error (MAPE)

---

## 20. Leakage Safeguards & Integrity Audit

1. **Strictly Chronological Splitting:** Zero random cross-validation.
2. **Train-Only Scaling:** Scaler parameters computed exclusively on training partition.
3. **Causal Context Construction:** Context at origin $t$ uses exclusively observations $\le t$.
4. **Out-of-Sample Recent Error:** Derived from causal walk-forward forecaster.
5. **Pre-Allocation Isolation:** Raw 15-minute TXT file is read-only and never modified.

---

## 21. Reproducibility Procedure

The entire protocol is programmatic, deterministic, and self-contained:
1. **Inspection Artifact:** `research/results/phase9_dataset_inspection.json`
2. **Summary Artifact:** `research/results/phase9_dataset_summary.csv`
3. **Dataset Adapter:** `research/data_adapter_uci.py`
4. **Protocol Unit Test Suite:** `research/tests/test_phase9_uci_protocol.py` (10/10 tests passed)

---

## 22. Known Dataset Limitations

1. **Client Onboarding Shift (2011):** 160 clients onboarded on Jan 1, 2012, producing a structural level shift if 2011 is included. Using the stable 2012–2014 period ($26,304$ hours) completely resolves this issue.
2. **Artificial DST Days:** March 01:00–02:00 set to zero; October 01:00–02:00 aggregates two hours. These 2 calendar days per year represent mild artificial metering conventions.
3. **Heterogeneous Meter Resolution:** Individual meters range from $0.1\text{ kW}$ to $5,000\text{ kW}$. System-level aggregation smooths out individual meter noise while preserving the macro-system dynamics.

---

## 23. Decisions Requiring User Approval Before Benchmark

Before launching the full five-seed neural benchmark, the following operational decisions are formally submitted for user review:

1. **Target Series Formulation:**
   - *(Option B, Recommended)* **System Aggregate Load:** Sum of all clients in MW.
   - *(Option C)* Benchmark a subset of individual clients in parallel.
2. **Time Span Selection:**
   - *(Recommended)* **Stable 3-Year Benchmark (2012–2014, 26,304 hours):** Standard in literature (LSTNet, DeepAR, Autoformer); client base is completely onboarded and active.
   - *(Full Span)* **Full 4-Year Benchmark (2011–2014, 35,064 hours):** Includes 2011 with the onboarding step shift.
3. **Measurement Unit:**
   - **MW (Megawatts):** $1\text{ MW} = 1000\text{ kW}$, aligning scale directly with PJM (MW) and GEFCom2014 (kW $\to$ MW scale).
   - **kW (Kilowatts):** Raw native unit ($1000\times$ larger numerical scale).

---

## Verification & Test Results

```text
research/tests/test_phase9_uci_protocol.py:
  test_01_raw_timestamp_ordering ......... OK
  test_02_frequency_detection ............. OK
  test_03_aggregation_correctness ......... OK
  test_04_no_duplicate_hourly_timestamps .. OK
  test_05_no_unintended_missing_intervals . OK
  test_06_chronological_split ............. OK
  test_07_no_leakage_across_partitions .... OK
  test_08_window_construction_shapes ...... OK
  test_09_train_only_scaler_fitting ....... OK
  test_10_causal_context_construction ..... OK

Ran 10 tests in 3.400s
OK
```
