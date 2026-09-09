# PHASE 8 — GEFCom2014 DATASET & EVALUATION PROTOCOL
**CAEG-Net Research Track | Original 14-Phase Research Roadmap**

---

## A. Dataset Structure
- **Dataset**: Global Energy Forecasting Competition 2014 (GEFCom2014) — Load Forecasting Track (`GEFCom2014-L`).
- **Location**: `data/Load/` (containing `Task 1` through `Task 15`, `Solution to Task 15`, and `Instructions.txt`).
- **Physical Series**: Regional electricity load time series recorded at 1-hour intervals.
- **Series Scale**: Load values range from $16.1$ kW to $317.5$ kW, with overall mean $\mu = 146.18$ kW and standard deviation $\sigma = 42.88$ kW.
- **Span**: 7 full calendar years, from **2005-01-01 01:00** to **2012-01-01 00:00** ($61,344$ contiguous hourly observations).
- **Contiguity**: Zero missing intervals; zero NaN values in the load series across the 7-year timeline.

---

## B. Task Structure
GEFCom2014 was structured by the competition organizers as **15 sequential monthly forecasting rounds (tasks)**:
1. **Pre-competition History (Task 1 initial data)**:
   - Covers 2005-01-01 to 2010-09-30 ($50,376$ hours = 69 months).
   - Provided to all participants at competition commencement.
2. **Sequential Rolling Monthly Tasks**:
   - **Task 1** (October 2010, 744 hours): Official warm-up round 1.
   - **Task 2** (November 2010, 720 hours): Official warm-up round 2.
   - **Task 3** (December 2010, 744 hours): Official warm-up round 3.
   - **Tasks 4–15** (January 2011 to December 2011, 8,760 hours = 365 days): Official competition evaluation rounds spanning a full 4-season annual cycle.
3. **Information Release Mechanics**:
   - For Task $t$, participants forecast month $t$.
   - Ground truth for month $t$ was revealed in file $L_{t+1}\text{-train.csv}$ when round $t+1$ commenced.
   - Ground truth for final Task 15 was released in `Solution to Task 15/solution15_L.csv`.

---

## C. Forecast Horizon
- **Horizon**: $H = 24$ hours (day-ahead short-term load forecasting).
- **Inductive Alignment**: Exactly preserves the canonical $168 \to 24$ operational formulation established in Phases 1–7 on Modern PJM.
- **Evaluation Partition**: In each evaluation month, forecasting origins advance in non-overlapping 24-hour steps ($D_t \in \{28, 30, 31\}$ daily blocks per month), yielding complete monthly coverage without overlapping-window inflation.

---

## D. Available History & Lookback
- **Lookback Window**: $L = 168$ hours (7 days $\times$ 24 hours), capturing full diurnal cycles and weekly seasonality.

---

## E. Evaluation Unit
1. **Primary Paired Statistical Unit**: Non-overlapping 24-hour daily blocks ($K = 457$ daily blocks across Tasks 1–15; $K = 365$ daily blocks across official competition Tasks 4–15).
2. **Task-Level Unit**: 15 distinct monthly evaluation tasks, reporting per-task mean MAE and cross-task stability.

---

## F. Training, Validation, and Test Partitioning
To guarantee strict causality and eliminate leakage while evaluating all 15 tasks:
- **Training Partition (Pre-Competition Base)**:
  - Timeline: **2005-01-01 01:00 to 2009-09-30 24:00** ($41,616$ hours, $67.8\%$ of total timeline).
  - Used exclusively for model parameter optimization.
- **Validation Partition (Pre-Competition Holdout)**:
  - Timeline: **2009-10-01 01:00 to 2010-09-30 24:00** ($8,760$ hours, 1 full calendar year, $14.3\%$ of total timeline).
  - Used exclusively for early stopping, learning rate scheduling, and checkpoint selection.
- **Locked Test Partition (Competition Tasks 1–15)**:
  - Timeline: **2010-10-01 01:00 to 2011-12-31 24:00** ($10,968$ hours, 457 days, $17.9\%$ of total timeline).
  - Sub-periods reported:
    - **All Tasks (Tasks 1–15)**: $457$ daily evaluation blocks ($10,968$ hours).
    - **Official Competition Period (Tasks 4–15)**: $365$ daily evaluation blocks ($8,760$ hours).

---

## G. Scaling Protocol
- **Scaler Fitting**: `StandardScaler` fitted strictly on training partition load values ($2005\text{--}2009$):
  $$\mu_{\text{train}} = 144.9754 \text{ kW}, \quad \sigma_{\text{train}} = 42.6688 \text{ kW}$$
- **Transformation**: Validation and test partitions transformed using training statistics only:
  $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Inverse Transformation**: All model predictions mapped back to physical engineering units (kW) before metric computation:
  $$\hat{y}_{\text{kW}} = \hat{y}_{\text{scaled}} \times \sigma_{\text{train}} + \mu_{\text{train}}$$

---

## H. Missing Data Handling
- Analysis of all 15 task files and Solution 15 reveals **zero missing values** in the load series from 2005-01-01 onward ($61,344$ continuous observations).
- Pre-2005 temperature rows in `L1-train.csv` (rows 0 to 35,063) contain NaN load and are discarded prior to load forecasting window construction.

---

## I. Window Construction
- Windows generated via `create_partition_windows_with_context()`:
  - Train: $X \in \mathbb{R}^{41425 \times 168 \times 1}$, $Y \in \mathbb{R}^{41425 \times 24}$
  - Val: $X \in \mathbb{R}^{8737 \times 168 \times 1}$, $Y \in \mathbb{R}^{8737 \times 24}$
  - Test: $X \in \mathbb{R}^{10945 \times 168 \times 1}$, $Y \in \mathbb{R}^{10945 \times 24}$
- For daily block evaluation ($H=24$, step=24):
  - Origins spaced every 24 hours: $t_k = k \times 24$ for $k = 0, \dots, 456$.
  - Each origin maps to its respective month and calendar date.

---

## J. Zone and Series Handling
- `ZONEID` is identically 1 across all files. The problem is a single regional aggregate load series.
- Only the univariate load series is used as input, preserving the exact architecture of CAEG-Net V1 without adding exogenous weather inputs.

---

## K. Hyperparameter Policy
- **Frozen Parameters (Zero Search)**:
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
  - Scheduler: `StepLR(step_size=15, gamma=0.5)`
  - Max Epochs: 45, Early Stopping Patience: 7
  - Loss Criterion: MSE
  - Context Dimension: 4
  - Router: Softmax soft fusion with base prior $w_0 = [1/3, 1/3, 1/3]$
- **No tuning on Test partition**: Test performance is evaluated once on locked checkpoints.

---

## L. Leakage Prevention
1. Zero test data enters `StandardScaler.fit()`.
2. Zero test targets enter lookback windows.
3. Out-of-sample Recent Error features generated causally via walk-forward Ridge baseline.
4. Final solution file `Solution to Task 15/solution15_L.csv` isolated from training, validation, and checkpoint selection.

---

## M. Primary and Secondary Metrics
- **Primary Research Metric**: **Mean Absolute Error (MAE, in kW)**.
- **Secondary Metrics**:
  - Root Mean Squared Error (RMSE, in kW)
  - Mean Squared Error (MSE, in $\text{kW}^2$)
  - Coefficient of Determination ($R^2$)
  - Mean Absolute Percentage Error (MAPE, in $\%$)

---

## N. Evaluated Models
1. **CAEG-Net V1** (Proposed Architecture: LSTM + TCN + CNN + Soft Dynamic Gating, 121,531 params, 5 seeds).
2. **Static Equal Ensemble** (Uniform 1/3 fusion of LSTM + TCN + CNN, 120,504 params, 5 seeds).
3. **Standalone LSTM** (2-layer LSTM, 56,152 params, 5 seeds).
4. **Standalone TCN** (6-stage causal TCN, 36,952 params, 5 seeds).
5. **Standalone CNN** (Multi-scale CNN, 27,400 params, 5 seeds).
6. **Official GEFCom2014 Benchmark** (Same-month-last-year seasonal persistence from `Lt-benchmark.csv`, 0 params).
7. **Ridge Regression** (Multi-output linear AR mapping $168 \to 24$, 4,056 params).
8. **Naive-24** (Day-ahead persistence, 0 params).
9. **Seasonal Naive-168** (Week-ahead persistence, 0 params).

---

## O. Exact Files Used
- `data/Load/Task 1/L1-train.csv` (historical training series)
- `data/Load/Task 2/L2-train.csv` through `Task 15/L15-train.csv` (task load increments)
- `data/Load/Task 1/L1-benchmark.csv` through `Task 15/L15-benchmark.csv` (official benchmarks)
- `data/Load/Solution to Task 15/solution15_L.csv` (ground truth for Task 15 evaluation)

---

## P. Exact Files Excluded from Training/Tuning
- `Solution to Task 15/solution15_L.csv` (strictly excluded from training and validation)
- All benchmark files `Lt-benchmark.csv` (strictly excluded from model training and tuning)
