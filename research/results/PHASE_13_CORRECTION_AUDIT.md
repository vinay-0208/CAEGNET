# Phase 13: Final Scientific Correction, Audit & Reconciliation Report

**Execution Protocol:** Phase 13 Genuine Out-of-Fold (OOF) Performance-Aware Routing Validation  
**Branch:** `research-track`  
**Conda Environment:** `C:\Users\vinay\anaconda3\envs\caeg-gpu\python.exe` (PyTorch `2.13.0+cu130`, NVIDIA GeForce RTX 4050 Laptop GPU)  
**Historical Phase 13 Git Commit:** `bc3a029` (Preserved & Frozen)  
**Audit Purpose:** Comprehensive scientific verification, numerical provenance trace, statistical reconciliation, and documentation correction across all Phase 13 findings.  
**Auditor Classification:** Senior ML Research Engineer, Statistical Auditor, and Reproducibility Reviewer  

---

## 1. Audit Objective & Executive Status

Phase 13 was designed to eliminate the in-sample expert training error leakage identified at the conclusion of Phase 12. The experimental execution completed successfully on GPU across Modern PJM, GEFCom2014, and UCI Cohort 320 Aggregate under 70/15/15 chronological partitioning.

However, a post-execution review of the generated reports and documentation identified several issues in:
1. Description of canonical V1 context features (prose incorrectly listed time-of-day/day-of-week instead of canonical features).
2. Calculation and reporting of "Beats Best Standalone Expert" counts.
3. Overclaimed causal and mechanistic language.
4. Interpretation of learned confidence parameter $\lambda_t$.
5. Reconciliation between Seed 42 and 5-seed pooled metrics on Modern PJM.

**Audit Status:**
- **Zero Category C (Implementation Errors affecting numerical validity):** The training, inference, and feature extraction code executed correctly.
- **Zero Category D (Methodological Validity Failures):** Expanding-window OOF generation was strictly causal; zero in-sample leakage occurred; validation screening firewall was intact.
- **Identified Issues are Strictly Category A (Documentation-only) and Category B (Derived Report / Table Inconsistency):** Model retraining is **unnecessary**.
- All historical raw records (`phase13_*.csv`, `PHASE_13_REPORT.md`, `PHASE_13_POST_EXECUTION_AUDIT.md`, `PHASE_13_WALKTHROUGH.md`, and commit `bc3a029`) remain frozen to preserve complete provenance.

---

## 2. Comprehensive Issue Inventory & Classification

| Issue # | Description | Severity | Category | Affects Model Weights / Results? | Root Cause | Action Taken | Status |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1** | Canonical V1 Context Feature Misdescription | High (Clarity) | **Category A** | **NO** | Walkthrough markdown text misdescribed canonical context as `[mean, std, tod, dow]` instead of `[Trend, Volatility, Periodicity, Recent_Error]`. Code was 100% correct. | Corrected documentation in report and audit. Verified `extract_context_features` in `data_utils.py`. | **RESOLVED** |
| **2** | Standalone Expert Comparison Inconsistency | High (Numerical) | **Category B** | **NO** | `phase13_dataset_summary.csv` logged validation MAE (7.55 MW) for UCI LSTM. Comparing test results (7.59 MW) against 7.55 MW yields 2/3, but report claimed 3/3 (matching Phase 11 test benchmark of 7.79 MW). | Documented both 7.55 MW (validation reference) and 7.79 MW (Phase 11 test benchmark). Fixed counts: 2/3 under 7.55 MW, 3/3 under 7.79 MW. Created corrected comparison CSV. | **RESOLVED** |
| **3** | Overclaimed Scientific Language | Medium (Tone) | **Category A** | **NO** | Use of words such as "proves", "indispensable", "optimal 50/50 shrinkage", "universally superior". | Replaced with publication-grade probabilistic and empirical phrasing ("suggests", "is consistent with", "statistically outperformed"). | **RESOLVED** |
| **4** | $\lambda$ Shrinkage Interpretation | Medium (Concept) | **Category A/B** | **NO** | Described $\lambda_t \approx 0.50$ as globally optimal shrinkage rather than observed near-constant shrinkage. | Clarified that $\lambda_t$ exhibited very low temporal variance ($CV < 2\%$) and behaved predominantly as a near-constant convex shrinkage toward equal ensemble. | **RESOLVED** |
| **5** | A2 vs. A3 Mechanistic Over-interpretation | Medium (Concept) | **Category A/B** | **NO** | Over-generalized ablation results as universal proofs. | Explicitly stated that on GEFCom, A3 matched A2 indicating trailing error features were not necessary, whereas on UCI, A2 statistically outperformed A3 ($p < 10^{-4}$). | **RESOLVED** |
| **6** | Modern PJM Aggregation Discrepancy | High (Clarity) | **Category A/B** | **NO** | Different metrics reported for Seed 42 ($\Delta = -16.90\text{ MW}$, $p=0.0688$) vs. 5-seed pooled ensemble ($\Delta = -5.51\text{ MW}$, $p=0.1136$) vs. 5-seed mean MAE ($255.49$ vs $253.41\text{ MW}$). | Mathematically defined all three estimands. Traced discrepancy to high V1 seed-42 error and Jensen's inequality reduction on ensembling. | **RESOLVED** |
| **7** | Daily-Block Statistical Unit & Direction | Low (Verification) | **Category A** | **NO** | Ensured daily blocks are strictly non-overlapping ($K=53, 456, 163$) and confirmed negative difference means candidate is better. | Verified code implementation in runner lines 734-748 and added defensive unit test `test_10`. | **RESOLVED** |
| **8** | Validation Screening Firewall | Low (Verification) | **Category A** | **NO** | Verified that test data was never consulted during Stage B candidate qualification. | Verified code lines 471-568. Test sets were strictly locked until Stage C. | **RESOLVED** |
| **9** | OOF Temporal Causality | Critical (Method) | **Category A** | **NO** | Verified that 4-block expanding window completely isolates training folds and trailing error features lag by 24h. | Verified code lines 248-375. Perturbation unit tests confirmed target perturbation invariance. | **RESOLVED** |
| **10** | Exact Parameter Counts | Low (Verification) | **Category A** | **NO** | Confirmed parameter counts from instantiated PyTorch models. | Verified: Core 120,504; V1 121,531; A1 121,579; A2 121,724; A3 121,628. | **RESOLVED** |
| **11** | Selection of Standalone Expert Baseline | Low (Verification) | **Category A** | **NO** | Confirmed best standalone expert was chosen strictly from validation partition. | Verified: TCN on PJM, TCN on GEFCom, LSTM on UCI chosen from validation MAE. | **RESOLVED** |
| **12** | PJM Validation vs. Test MAE Shift | Medium (Clarity) | **Category A** | **NO** | PJM Val MAE (~438 MW) was substantially higher than Test MAE (~253 MW). | Verified empirical data distribution: PJM validation partition coincides with extreme summer peak (mean 6,198 MW, std 1,183 MW) vs moderate test partition (mean 5,346 MW, std 935 MW). | **RESOLVED** |

---

## 3. Detailed Technical Verifications

### 3.1 Canonical V1 Context Features (Critical Issue #1)
- **Investigation:** Code inspection of `run_phase13_oof_performance_aware_gating.py` (line 457) shows that `c_base` is loaded directly from `datasets[d_key]["contexts"]["A0"]`. In `run_phase10_optimization.py` (line 134), `ctx_a0` is computed via `extract_context_features(X, rec)`. In `data_utils.py` (lines 476-550), `extract_context_features` extracts:
  1. **Trend ($C_1$):** Normalized OLS slope over the 168-hour lookback divided by lookback standard deviation.
  2. **Volatility ($C_2$):** Sample standard deviation of first differences $\sigma(\Delta z)$.
  3. **Periodicity ($C_3$):** Lag-24 sample autocorrelation $r_{24} \in [-1, 1]$.
  4. **Recent Error ($C_4$):** Out-of-sample MAE of the most recently completed 24-hour forecast cycle $[t-23 : t]$.
- **Finding:** Canonical V1 in Phase 13 strictly utilized the true 4D canonical context features established in Phase 1-10. Phase 13 did NOT modify V1. The phrasing in the earlier walkthrough describing them as mean load, standard deviation, time-of-day, and day-of-week was an erroneous prose description.
- **Classification:** Category A (Documentation-only).

### 3.2 Reconciliation of "Beats Best Expert" Counts (Critical Issue #2)
- **Baseline Sources:**
  - On Modern PJM: Best Standalone Expert is **TCN** ($259.33\text{ MW}$ test MAE).
  - On GEFCom2014: Best Standalone Expert is **TCN** ($12.57\text{ kW}$ test MAE).
  - On UCI Cohort 320:
    - *Validation Selection Reference:* Standalone LSTM validation MAE is **$7.55\text{ MW}$** (logged in `phase13_dataset_summary.csv`).
    - *Test Benchmark Reference:* Standalone LSTM held-out test MAE in Phase 11 is **$7.79\text{ MW}$** (logged in `phase11_dataset_summary.csv`).
- **Empirical Five-Seed Test Results:**
  - A1: PJM $250.63\text{ MW}$ (< 259.33, Win), GEFCom $12.64\text{ kW}$ (> 12.57, Loss), UCI $7.98\text{ MW}$ (> 7.55 / > 7.79, Loss).
    - **Beats Best Expert Count:** **1 / 3** (PJM only).
  - A2: PJM $255.49\text{ MW}$ (< 259.33, Win), GEFCom $12.46\text{ kW}$ (< 12.57, Win), UCI $7.59\text{ MW}$.
    - vs. $7.55\text{ MW}$ baseline: $7.59 > 7.55$ (Loss) $\to$ **2 / 3** (PJM, GEFCom).
    - vs. $7.79\text{ MW}$ test benchmark: $7.59 < 7.79$ (Win) $\to$ **3 / 3**.
  - A3: PJM $255.66\text{ MW}$ (< 259.33, Win), GEFCom $12.44\text{ kW}$ (< 12.57, Win), UCI $7.71\text{ MW}$.
    - vs. $7.55\text{ MW}$ baseline: $7.71 > 7.55$ (Loss) $\to$ **2 / 3** (PJM, GEFCom).
    - vs. $7.79\text{ MW}$ test benchmark: $7.71 < 7.79$ (Win) $\to$ **3 / 3**.
- **Correction:** Under the primary baseline ($7.55\text{ MW}$), the correct count is **1/3 for A1, 2/3 for A2, and 2/3 for A3**. The prior report mistakenly reported 3/3 by blending the 7.55 baseline label with the 7.79 test benchmark comparison. Both references are now explicitly reported and reconciled.

### 3.3 Confidence Fallback ($\lambda_t$) Behavior (Critical Issue #4)
- Audit of `phase13_lambda_distribution.csv` across all seeds and datasets:
  - Mean $\lambda$: PJM $= 0.5061$, GEFCom $= 0.5190$, UCI $= 0.5095$ (Candidate A2).
  - Median $\lambda$: PJM $= 0.5060$, GEFCom $= 0.5193$, UCI $= 0.5096$.
  - Standard Deviation of $\lambda$: PJM $= 0.0038$, GEFCom $= 0.0045$, UCI $= 0.0032$.
  - Range: Minimum $\lambda \approx 0.470$, Maximum $\lambda \approx 0.559$.
  - Fraction $< 0.1$: Exactly $0.0\%$.
  - Fraction $> 0.9$: Exactly $0.0\%$.
- **Scientific Synthesis:** The network converges to $\lambda_t \approx 0.50 - 0.52$ with negligible temporal variation ($CV = \sigma/\mu < 2\%$) across all forecast horizons and conditions. The learned fallback does not act as a dynamic, responsive confidence indicator; rather, it functions as a **near-constant convex shrinkage** that blends adaptive routing and the static equal ensemble in roughly equal measure. This stabilizes routing predictions by cutting estimator variance in half on collinear datasets.

### 3.4 Modern PJM Aggregation Reconciliation (Critical Issue #6)
The three reported quantities on Modern PJM measure fundamentally distinct statistical estimands:
1. **Estimand 1: Mean of 5 Individual Seed MAEs** ($255.49 \pm 6.36\text{ MW}$ for A2 vs. $253.41 \pm 9.12\text{ MW}$ for V1):
   - Computes MAE for each seed model independently, then averages the 5 scalars.
   - Difference: $+2.08\text{ MW}$ ($+0.82\%$), well within the seed standard deviation of $6-9\text{ MW}$.
2. **Estimand 2: Paired Daily Difference on Primary Seed 42** ($-16.90\text{ MW}$, $p = 0.0688$):
   - Evaluated on Seed 42 across the $K=53$ non-overlapping 24-hour daily blocks.
   - On Seed 42, V1 suffered an unfavorable initialization ($MAE \approx 266\text{ MW}$), leading to a marginal advantage for A2 ($p = 0.0688$) and significant advantage for A1 ($-15.37\text{ MW}$, $p = 0.0215$).
3. **Estimand 3: Paired Daily Difference of Seed-Ensemble Predictions** ($-5.51\text{ MW}$, $p = 0.1136$):
   - Predictions are first averaged across all 5 seeds at each hour: $\bar{\hat{y}} = \frac{1}{5}\sum_{s} \hat{y}^{(s)}$.
   - Ensembling substantially reduces the variance of Canonical V1's predictions. The paired daily block difference between the ensembled predictions narrows to $-5.51\text{ MW}$, which is not statistically significant ($p = 0.1136$, Cohen's $d_z = -0.221$).
- **Conclusion:** There is no numerical contradiction. The difference stems from whether individual models or ensembled predictions are evaluated, and the natural seed sensitivity of Canonical V1 on PJM.

### 3.5 OOF Temporal Causality & Firewall (Critical Issue #8 & #9)
- **Expanding Window Protocol:**
  - Training partition divided into 4 chronological blocks $[B_0, B_1, B_2, B_3, B_4]$.
  - Fold 1 trained on $[B_0 : B_1]$ $\to$ predicts $[B_1 : B_2]$.
  - Fold 2 trained on $[B_0 : B_2]$ $\to$ predicts $[B_2 : B_3]$.
  - Fold 3 trained on $[B_0 : B_3]$ $\to$ predicts $[B_3 : B_4]$.
  - Block 1 receives baseline prior from Fold 1.
- **24-Hour Causal Lag:**
  - At origin $t$, trailing error feature $e_i(t)$ is drawn from origin $t - 24$, corresponding to forecast horizon $[t-23, \dots, t]$.
  - Future actuals $y_{t+1:t+24}$ are strictly unobserved.
- **Validation Screening Firewall:**
  - Stage B screening on seeds 42 and 123 evaluated exclusively on validation sets.
  - Zero test set records were accessed or evaluated prior to finalist selection.

---

## 4. Final Model Evaluation & Decision Matrix

| Evaluation Criterion | A0_Canonical_V1 | A1_P2_OOF | A2_C1_OOF (Winner) | A3_Confidence_Only | Criterion Winner |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OOF Methodological Rigor** | N/A (No OOF) | Genuine 4-Fold OOF | Genuine 4-Fold OOF | N/A (No error feats) | A1, A2 |
| **Stage B Validation Qualification** | Control (0/3) | Qualified (2/3) | Qualified (3/3, -8.8% PJM) | Qualified (3/3, -6.9% PJM) | **A2** |
| **PJM 5-Seed Test MAE (MW)** | $253.41 \pm 9.12$ | $\mathbf{250.63 \pm 5.55}$ | $255.49 \pm 6.36$ | $255.66 \pm 6.41$ | **A1** |
| **GEFCom 5-Seed Test MAE (kW)** | $12.88 \pm 0.25$ | $12.64 \pm 0.22$ | $12.46 \pm 0.12$ | $\mathbf{12.44 \pm 0.21}$ | **A3** (A2 parity) |
| **UCI 5-Seed Test MAE (MW)** | $7.94 \pm 0.17$ | $7.98 \pm 0.20$ | $\mathbf{7.59 \pm 0.15}$ | $7.71 \pm 0.18$ | **A2** |
| **Beats Canonical V1 (# / 3)** | Baseline (0/3) | 2 / 3 (PJM, GEFCom) | **2 / 3** (GEFCom, UCI) | **2 / 3** (GEFCom, UCI) | A1, A2, A3 (Tie) |
| **Beats Equal Ensemble (# / 3)** | 2 / 3 | 2 / 3 | **3 / 3** | **3 / 3** | **A2, A3** |
| **Beats Standalone Expert (7.55 BL)** | 1 / 3 | 1 / 3 | **2 / 3** | **2 / 3** | **A2, A3** |
| **Beats Standalone Expert (7.79 BL)** | 1 / 3 | 1 / 3 | **3 / 3** | **3 / 3** | **A2, A3** |
| **Statistical Advantage on Collinear Grid** | Baseline | Decisive ($p < 10^{-14}$) | Decisive ($p < 10^{-22}$) | Decisive ($p < 10^{-19}$) | **A2, A3** |
| **Statistical Advantage on Consumer Aggregate** | Baseline | Parity ($p = 0.54$) | Decisive ($p < 10^{-3}$) | Decisive ($p < 10^{-3}$) | **A2** |
| **Ablation Superiority (A2 vs A3)** | N/A | N/A | **Significant on UCI** ($p < 10^{-3}$) | Matches A2 on GEFCom | **A2** |
| **Parameter Overhead vs. V1** | **0 (Baseline)** | +48 (+0.04%) | +193 (+0.16%) | +97 (+0.08%) | A0 / A1 |

### Recommendation: **A2_C1_OOF is the LEADING FINAL-MODEL CANDIDATE**
- **Justification:**
  1. A2 achieves the lowest MAE on heterogeneous consumer profiles (UCI: $7.59\text{ MW}$), statistically outperforming both Canonical V1 ($p < 10^{-4}$) and the fallback-only ablation A3 ($p < 10^{-3}$).
  2. A2 achieves strong performance on regional grid load (GEFCom: $12.46\text{ kW}$), statistically outperforming V1 ($p < 10^{-22}$) with the lowest seed variance ($\pm 0.12\text{ kW}$).
  3. A2 beats the Static Equal Ensemble on all 3 datasets, and beats standalone experts on 2/3 datasets under the strict 7.55 MW baseline (3/3 under the 7.79 MW test benchmark).
  4. On Modern PJM, A2 remains within $0.82\%$ of Canonical V1 across 5 seeds, well inside the seed standard deviation ($6-9\text{ MW}$).
  5. The parameter overhead is negligible (+193 parameters, +0.16%).
- **Important Qualification:** A2 should be described as the **leading final-model candidate**, not an unconditionally universal winner, because A1 achieved the best MAE on PJM ($250.63\text{ MW}$) and A3 achieved comparable performance on GEFCom ($12.44\text{ kW}$).
