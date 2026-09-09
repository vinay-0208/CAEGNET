# CAEG-Net Phase 7: Strong Baseline Evaluation — Experimental Design & Protocol Specification
**Document**: `research/results/PHASE_7_STRONG_BASELINES_DESIGN.md`  
**Track**: `research-track` | **Phase**: Phase 7 — Strong Baseline Evaluation  
**Roadmap**: 14-Phase Original Research Roadmap  
**Date**: September 2026  
**Status**: DESIGN LOCKED & READY FOR EXECUTION  

---

## 1. Scientific Objective & Research Scope

Phase 7 of the original 14-phase CAEG-Net research roadmap evaluates the canonical **CAEG-Net V1** model against a compact, methodologically sound set of strong forecasting baselines.

### Primary Research Question
> *"How does the canonical CAEG-Net compare with strong established short-term electricity load forecasting baselines under the same data, horizon, and evaluation protocol?"*

### Governing Architectural Commitments
1. **Primary Proposed Model Preserved**: **CAEG-Net V1** (121,531 parameters, LSTM + TCN + CNN, 4D context, softmax soft router) remains strictly unaltered.
2. **Optimized Variant**: **Bounded CAEG-Net ($\rho=0.50$)** is documented strictly as a secondary conservative routing variant.
3. **No Architecture Exploration**: No new mixture-of-experts combinations, no router modifications, and no hyperparameter tuning of the proposed model.
4. **Primary Evaluation Metric**: Mean Absolute Error (**MAE in Megawatts, MW**) is the primary research metric. Secondary metrics are RMSE (MW), MSE (MW$^2$), $R^2$, and MAPE (%).
5. **Locked Test Partition**: The test set was frozen at the onset of Phase 6 and is treated as a locked final evaluation after development. Zero test information is accessed for baseline selection or hyperparameter tuning.

---

## 2. Baseline Taxonomy & Selection Rationale

In accordance with Phase 7 instructions, the baseline suite is kept **compact, transparent, and representative** of four foundational paradigms in short-term load forecasting:

| Baseline ID | Model Name | Paradigm | Parameter Count | Key Scientific Motivation |
| :---: | :--- | :--- | :---: | :--- |
| **B1** | **Naive-24** | Day-Ahead Persistence | $0$ | Canonical un-modelled benchmark reflecting 24-hour diurnal cycle ($\hat{y}_{t+h} = y_{t+h-24}$). |
| **B2** | **Seasonal Naive-168** | Week-Ahead Seasonal | $0$ | Captures strong weekly periodicity by persisting identical day-of-week load ($\hat{y}_{t+h} = y_{t+h-168}$). |
| **B3** | **Ridge Regression** | Regularized Linear AR | $4,056$ | Multi-output $L_2$-regularized linear autoregression mapping $168$ input lags $\to 24$ forecast steps. Classical statistical learning benchmark. |
| **B4** | **Standalone GRU** | Recurrent Neural Seq | $31,344$ | Standard 2-layer Gated Recurrent Unit sequence-to-vector model. Widely deployed recurrent baseline in energy forecasting. |
| **B5** | **Standalone TCN** | Causal Convolutional | $36,952$ | The single strongest individual temporal expert identified in Phase 6. Phase 6 frozen reference: $254.43 \pm 4.08$ MW. |
| **B6** | **Static Equal Ensemble**| Static Expert Fusion | $120,504$ | Fixed arithmetic mean of canonical LSTM, TCN, and CNN experts. Phase 6 frozen reference: $279.79 \pm 8.98$ MW. |

---

## 3. Data & Training Protocols

### Chronological Partitioning (Identical to Phase 6)
- **Dataset**: Modern PJM Load Dataset (`data/Modern_PJM/pjm_load.csv`, 8,784 hourly timestamps).
- **Partitions**: 70% Train ($6,148$ rows), 15% Validation ($1,318$ rows), 15% Test ($1,318$ rows).
- **Windows**: Lookback $L=168$ hours, Horizon $H=24$ hours.
  - Train: $5,957$ windows.
  - Validation: $1,294$ windows.
  - Test: $1,294$ windows (origins $167$ to $1460$).
- **Scaling**: `StandardScaler` fitted strictly on Train ($\mu = 5458.034$ MW, $\sigma = 855.390$ MW).

### Baseline-Specific Training Protocols
1. **Deterministic Baselines (Naive-24, Seasonal Naive-168)**:
   - Evaluated directly on unscaled test partition. Deterministic across all runs.
2. **Ridge Regression (Multi-output $L_2$ Autoregression)**:
   - Maps $X_{\text{2D}} \in \mathbb{R}^{N \times 168} \to Y \in \mathbb{R}^{N \times 24}$.
   - Hyperparameter $\alpha$ tuned strictly on **Validation MAE** across $\alpha \in \{0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0\}$.
   - Best $\alpha^*$ locked before test inference. Deterministic on test.
3. **Standalone GRU**:
   - Architecture: 2-layer stacked GRU (input=1, hidden=54, dropout=0.1) + Linear projection head ($54 \to 54 \to 24$).
   - Trained across the **five canonical seeds**: `[42, 123, 999, 2024, 3407]`.
   - Optimizer: AdamW (`lr=1e-3, weight_decay=1e-4`).
   - Scheduler: StepLR (`step_size=15, gamma=0.5`).
   - Early Stopping: `max_epochs=45, patience=7`, restoring best weights by Validation MSE.

---

## 4. Statistical Significance Framework ($K=53$ Daily Blocks)

Consistent with the verified Phase 6 methodology audit:
1. Hourly predictions are aggregated into **$K=53$ contiguous, non-overlapping 24-hour daily blocks** ($53 \times 24 = 1,272$ hours).
2. For each day $k \in \{1, \dots, 53\}$, daily MAE is computed for CAEG-Net V1 and each baseline:
   $$d_k = \bar{e}_k^{\text{CAEG-Net V1}} - \bar{e}_k^{\text{Baseline}}$$
3. Hypotheses:
   - Paired Student's $t$-test ($df = 52$).
   - Wilcoxon signed-rank test.
   - Verified Diebold-Mariano test with Harvey-Leybourne-Newbold correction ($\text{DM}_{\text{HLN}} \equiv t_{\text{paired}}$ at $h=1$).
   - 95% Confidence Intervals and Cohen's $d$ effect sizes.
   - Holm-Bonferroni step-down correction across all 6 planned baseline comparisons.

---

## 5. Pre-Declared Comparison Matrix

The primary evaluation table will report:
- `Model`
- `MAE (MW) [Mean ± SD]`
- `RMSE (MW) [Mean ± SD]`
- `MSE (MW²) [Mean ± SD]`
- `R² [Mean ± SD]`
- `MAPE (%) [Mean ± SD]`
- `Total Parameters`
- `Training Time (s)`
