# PHASE 7A — RIDGE BASELINE REPRODUCIBILITY & METHODOLOGY AUDIT
**CAEG-Net Research Track | Original 14-Phase Research Roadmap**

---

## A. Audit Objective
The objective of this audit is to conduct an independent, rigorous methodology and reproducibility verification of the unusually strong **Ridge Regression** baseline reported in Phase 7 ($222.23$ MW MAE) prior to freezing the results. Specifically, this audit investigates whether:
1. The Ridge baseline operates on the exact same data partitioning, temporal window construction, and target alignment as CAEG-Net V1.
2. Normalization/scaling is strictly fitted on the training partition without test or validation leakage.
3. The regularization hyperparameter $\alpha^* = 0.01$ was selected strictly on Validation MAE without consulting the test set.
4. The final Ridge model was trained according to the pre-specified protocol without test contamination.
5. The reported test performance is fully reproducible within negligible floating-point tolerance.
6. The statistical significance testing conforms to the non-overlapping $K=53$ daily block unit.
7. The reported parameter count ($4,056$) reflects the exact mathematical parameterization of the model.

---

## B. Original Phase 7 Ridge Result
From `research/results/phase7_strong_baseline_results.csv` and `PHASE_7_STRONG_BASELINES_RESULTS.md`:

- **Model**: Multi-Output $L_2$-Regularized Autoregression (`sklearn.linear_model.Ridge`)
- **Selected Hyperparameter**: $\alpha^* = 0.01$
- **Test MAE [Primary Metric]**: **$222.23$ MW**
- **Test RMSE**: $312.73$ MW
- **Test $R^2$**: $0.8884$
- **Test MAPE**: $4.18\%$
- **Reported Parameter Count**: $4,056$
- **CAEG-Net V1 Benchmark Reference**: $251.74 \pm 7.79$ MW MAE
- **Numerical Advantage**: $251.74 - 222.23 = 29.51$ MW MAE in favor of Ridge.

---

## C. Implementation Findings
An inspection of the Phase 7 benchmark script (`research/experiments/run_phase7_strong_baselines.py`, lines 280–321) reveals:
- **Model Formulation**: Direct multi-output linear mapping from $168$ input lag features to $24$ forecast steps.
- **Fitting Protocol Determination**:
  - **Option A (Train-Only Fitting)**: Fits `Ridge(alpha=best_alpha).fit(X_tr, Y_tr)`.
  - **Option B (Train + Validation Refitting)**: Refits on concatenated `[X_tr, X_val]` and `[Y_tr, Y_val]`.
  - **Actual Protocol in Phase 7**: **Option A (Train-Only Fitting)** was strictly implemented. After identifying $\alpha^* = 0.01$ on validation MAE, the model was fitted solely on `(X_tr, Y_tr)`.
  - An empirical comparison during this audit confirmed:
    - Train Only (Option A): Test MAE = $222.2312$ MW (matches Phase 7 reported $222.23$ MW).
    - Train + Val Refit (Option B): Test MAE = $223.1840$ MW.
  - Therefore, the model evaluated on the test partition was trained exclusively on the $70\%$ training partition.

---

## D. Data and Alignment Verification
- **Raw Dataset**: `data/Modern_PJM/pjm_load.csv` (8,784 total clean hourly rows spanning `2023-10-01 04:00:00+00:00` to `2024-10-01 03:00:00+00:00`).
- **Chronological Split**: Strict chronological split performed *before* window generation:
  - **Train**: 6,148 rows (69.99%) [`2023-10-01 04:00:00+00:00` to `2024-06-13 07:00:00+00:00`]
  - **Validation**: 1,318 rows (15.00%) [`2024-06-13 08:00:00+00:00` to `2024-08-07 05:00:00+00:00`]
  - **Test**: 1,318 rows (15.00%) [`2024-08-07 06:00:00+00:00` to `2024-10-01 03:00:00+00:00`]
- **Window Dimensions**: Lookback = $168$ hours, Horizon = $24$ hours.
  - Train: $X \in \mathbb{R}^{5957 \times 168}$, $Y \in \mathbb{R}^{5957 \times 24}$
  - Val: $X \in \mathbb{R}^{1294 \times 168}$, $Y \in \mathbb{R}^{1294 \times 24}$
  - Test: $X \in \mathbb{R}^{1294 \times 168}$, $Y \in \mathbb{R}^{1294 \times 24}$
- **Temporal Target Alignment**:
  - The first test origin in `windows["test"]["X"][0, -1]` is $5046.821$ MW (corresponds to `test_df.iloc[0]`).
  - The first test target in `windows["test"]["Y"][0, 0]` is $4939.595$ MW (corresponds to `test_df.iloc[1]`).
  - Both CAEG-Net V1 and Ridge Regression evaluate against the exact same ground-truth array `windows["test"]["Y"]`.

---

## E. Scaling Verification
- **Scaler Fitting**: `StandardScaler` was fitted strictly on `train_df[['load']].values` via `fit_and_transform_scaler()`.
  - Training mean: $\mu_{\text{train}} = 5458.034000$ MW
  - Training standard deviation: $\sigma_{\text{train}} = 855.389777$ MW
- **Transformations**:
  - `val_df` and `test_df` were transformed strictly using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$. No test or validation statistics were used to compute or adjust scaling factors.
- **Inverse Transformation**:
  - Ridge predictions in scaled space were transformed back to engineering units (MW) via:
    $$\hat{Y}_{\text{MW}} = \hat{Y}_{\text{scaled}} \times \sigma_{\text{train}} + \mu_{\text{train}}$$
  - Identical inverse transformation used for CAEG-Net V1 and all baselines.

---

## F. Validation-Only Alpha Selection
The Phase 7 pipeline evaluated the pre-specified candidate grid:
$$\alpha \in \{0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0\}$$

Validation metrics obtained during grid evaluation:
| $\alpha$ Candidate | Validation MAE (MW) | Validation RMSE (MW) |
| :---: | :---: | :---: |
| **$0.01$** | **$307.8174$** | **$412.5320$** |
| $0.10$ | $307.8337$ | $412.5505$ |
| $1.00$ | $308.0620$ | $412.7878$ |
| $10.00$ | $310.6229$ | $415.1021$ |
| $50.00$ | $316.5467$ | $420.3533$ |
| $100.00$ | $320.6103$ | $424.0150$ |
| $500.00$ | $334.2106$ | $437.0479$ |
| $1000.00$ | $342.6986$ | $445.9916$ |

- **Verification**: $\alpha^* = 0.01$ strictly yielded the minimum Validation MAE ($307.8174$ MW).
- **Test Integrity**: Test performance was not consulted during selection.
- **Confirmation**: **Validation-only hyperparameter selection verified.**

---

## G. Test Leakage Verification
- **Test Input $X_{\text{te}}$**: Not used in fitting the regression coefficients $\beta$ or intercept.
- **Test Target $Y_{\text{te}}$**: Not used in fitting or hyperparameter tuning.
- **Evaluation Timing**: Test metrics were computed exclusively on the locked model after hyperparameter selection was finalized.
- **Confirmation**: **No test-target leakage detected.**

---

## H. Reproduction Result
A single deterministic reproduction run was executed using the exact code path and parameters:

| Metric | Original Reported (Phase 7) | Reproduced (Phase 7A) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **MAE (MW)** | **$222.23$** | **$222.2312$** | $0.0012$ MW |
| **RMSE (MW)** | **$312.73$** | **$312.7291$** | $0.0009$ MW |
| **$R^2$** | **$0.8884$** | **$0.888422$** | $0.000022$ |
| **MAPE (%)** | **$4.18\%$** | **$4.1758\%$** | $0.0042\%$ |

The differences between the original reported numbers and the reproduced run are strictly $< 0.002$ MW, originating purely from two-decimal formatting rounding.

---

## I. Daily-Block Statistical Alignment Verification
The statistical comparison between CAEG-Net V1 and Ridge Regression was verified on the exact non-overlapping $K=53$ daily 24-hour evaluation blocks ($k \times 24$ for $k=0, \dots, 52$):
- **Mean Paired Block Difference** ($\bar{e}_k^{\text{CAEG V1}} - \bar{e}_k^{\text{Ridge}}$): **$+28.72$ MW** ($95\%$ CI: $[+3.25, +54.20]$ MW)
- **Paired $t$-Test**: $t = +2.2625$, $p = 0.027870$ (unadjusted $p < 0.05$)
- **Wilcoxon Signed-Rank Test**: $W = 415.0$, $p = 0.007808$
- **Holm-Bonferroni Adjusted $p$-Value**: **$p = 0.139348$**
- **Statistical Significance under FWER ($\alpha=0.05$)**: **False (Inconclusive)**

**Scientific Context**: Although Ridge holds a notable numerical advantage of $29.51$ MW over CAEG-Net V1 across the entire test partition (and an unadjusted $p=0.028$ on daily blocks), the comparison is strictly inconclusive after controlling for family-wise error rate across the planned baseline comparisons.

---

## J. Parameter Count Verification
- **Model Parameterization**:
  $$\hat{Y}_{t, h} = \sum_{l=0}^{167} W_{h, l} X_{t, l} + b_h \quad \text{for } h \in \{1, \dots, 24\}$$
- **Mathematical Count**:
  $$\text{Weights}: 168 \text{ input features} \times 24 \text{ output horizons} = 4,032$$
  $$\text{Biases}: 24 \text{ output intercepts} = 24$$
  $$\text{Total Parameters} = 4,032 + 24 = 4,056$$
- **`scikit-learn` Attribute Confirmation**:
  - `Ridge.coef_.shape` = `(24, 168)` $\implies 4,032$
  - `Ridge.intercept_.shape` = `(24,)` $\implies 24$
  - Total = $4,056$.

### Inductive Bias Analysis (Why Ridge Excels on Aggregate Load)
Inspection of the fitted coefficients matrix $W \in \mathbb{R}^{24 \times 168}$ shows that Ridge heavily weights specific temporal lags:
1. $t - 0\text{h}$ (most recent observation): Mean $|\beta| = 1.3563$
2. $t - 1\text{h}$ (prior hour): Mean $|\beta| = 0.8408$
3. $t - 24\text{h}$ (exact daily diurnal lag): Mean $|\beta| = 0.3832$
4. $t - 144\text{h}$ (multi-day lag): Mean $|\beta| = 0.2497$
5. $t - 167\text{h}$ (weekly seasonal lag): Mean $|\beta| = 0.2316$

Because regional electricity demand in PJM is highly aggregated and dominated by strong diurnal and weekly autoregressive inertia, a regularized direct linear mapping directly exploits these structural lags without suffering from the non-convex optimization challenges, initialization variance, or over-parameterization of deep networks ($4,056$ parameters vs $121,531$).

---

## K. Direct Methodological Comparability Checklist
| Criterion | Status | Detail |
| :--- | :---: | :--- |
| 1. Same raw dataset? | **YES** | `data/Modern_PJM/pjm_load.csv` |
| 2. Same chronological split? | **YES** | 70% Train, 15% Val, 15% Test split before windowing |
| 3. Same 168→24 formulation? | **YES** | Lookback = 168, Horizon = 24 |
| 4. Same target timestamps? | **YES** | Identical ground-truth target windows |
| 5. Same train-only scaling? | **YES** | StandardScaler fitted exclusively on training rows |
| 6. Same test partition? | **YES** | Exactly 1,294 test origins |
| 7. No test leakage? | **YES** | Test set untouched during fitting and tuning |
| 8. Validation-only alpha selection? | **YES** | $\alpha^*=0.01$ picked on Validation MAE |
| 9. Same daily-block statistical unit? | **YES** | $K=53$ non-overlapping 24h daily blocks |

---

## L. Final Verdict

# **VERIFIED**

The Phase 7 Ridge Regression result ($222.23$ MW MAE, $312.73$ MW RMSE, $R^2 = 0.8884$, MAPE = $4.18\%$) is completely methodologically sound, reproducible within floating-point precision, free from test leakage, and directly comparable to CAEG-Net V1.
