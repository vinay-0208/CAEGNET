# CAEG-Net Phase 10: Scientific Audit, Statistical Reconciliation & Report Correction

**Audit Date:** September 10, 2026  
**Audited Commit:** `13809fe` (`research: complete Phase 10 cross-dataset CAEG optimization`)  
**Branch:** `research-track`  
**Auditor:** CAEG-Net Research Assistant  
**Status:** Completed & Verified  

---

## 1. Scope & Purpose of the Audit

This audit was conducted to ensure that all claims, statistical interpretations, test-set descriptions, and numerical figures reported in Phase 10 are publication-safe, statistically rigorous, and fully defensible against the experimental data.

### Strict Experimental Constraints:
- **No Experiment Rerun:** Phase 10 experiments were NOT rerun.
- **No Model / Result Changes:** No underlying model parameters, training code, test splits, or CSV results were modified.
- **Source of Truth:** Numerical CSV artifacts generated during Phase 10 execution served as the sole source of truth.

---

## 2. Artifacts Inspected

The following primary Phase 10 artifacts were audited:
1. `research/results/phase10_validation_results.csv` (14 candidates $\times$ 3 datasets = 42 validation screening runs)
2. `research/results/phase10_cross_dataset_validation.csv` (Relative validation changes and qualification flags)
3. `research/results/phase10_candidate_comparison.csv` (5-seed test evaluations for A0, B1, and C3)
4. `research/results/phase10_routing_analysis.csv` (Routing weights, entropy, and effective expert counts across 5 seeds)
5. `research/results/phase10_baselines.csv` (Standalone single experts and Static Equal Ensemble)
6. `research/results/phase10_expert_complementarity.csv` (Pairwise residual error correlation matrices)
7. `research/results/phase10_statistical_tests.csv` (Daily-block paired $t$-tests and Wilcoxon tests)
8. `research/experiments/run_phase10_optimization.py` (Driver script)
9. `research/tests/test_phase10_optimization.py` (Unit tests)
10. `research/results/PHASE_10_OPTIMIZATION_DESIGN.md` (Design specifications)
11. `research/results/PHASE_10_OPTIMIZATION_RESULTS.md` (Prose report)
12. `walkthrough.md` (Summary documentation)

---

## 3. Inconsistencies Identified & Corrections Made

### 3.1 Test-Set Terminology
- **Issue:** Prior documentation used phrases such as *"untouched test sets"*, *"untouched test partitions"*, and *"pristine unseen test sets"*. These phrases overstate the novelty of the data, as earlier research phases evaluated these partitions.
- **Correction:** Replaced with scientifically accurate terminology: *"held-out test partitions under the locked Phase 10 protocol"* or *"previously held-out test partitions evaluated after Phase 10 candidate selection was frozen"*.
- **Explicit Clarification:** The report explicitly clarifies that test partitions were held out during Phase 10 model training and selection procedures, but are not pristine first-ever test data because prior phases evaluated them.

### 3.2 Modern PJM Statistical Significance
- **Issue:** Prior prose claimed that *"CAEG V1 significantly beats Equal Ensemble on PJM"*.
- **Audit Findings:** In `phase10_statistical_tests.csv`, the daily-block paired statistics for PJM ($K=53$ blocks) are:
  - Mean daily difference ($\text{MAE}_{\text{CAEG}} - \text{MAE}_{\text{Ens}}$): $-11.15$ MW
  - $t$-statistic: $-1.494$
  - Paired $t$-test $p$-value: $0.1413$
  - Wilcoxon signed-rank $p$-value: $0.5861$
- **Correction:** Because $p > 0.05$, the paired difference is **not statistically significant**. The claim was corrected to:
  > *"CAEG-Net V1 achieved a 9.4% lower aggregate test MAE (251.17 vs 277.13 MW; 245.98 vs 277.13 MW on seed 42) than the Static Equal Ensemble on PJM; however, the daily-block paired comparison did not reach statistical significance (paired t-test p=0.1413; Wilcoxon p=0.5861, K=53 blocks)."*

### 3.3 UCI Statistical Interpretation
- **Issue:** Prior prose claimed "parity" on UCI or implied CAEG matched/exceeded the Equal Ensemble.
- **Audit Findings:** In `phase10_statistical_tests.csv`, the daily-block paired statistics for UCI ($K=163$ blocks) are:
  - Mean daily difference: $+0.38$ MW
  - $t$-statistic: $+3.300$
  - Paired $t$-test $p$-value: $0.0012$
  - Wilcoxon signed-rank $p$-value: $0.0036$
- **Correction:** In daily-block paired testing, the positive difference ($\text{MAE}_{\text{CAEG}} - \text{MAE}_{\text{Ens}} = +0.38$ MW) statistically favors the Equal Ensemble ($p < 0.01$). The report now explicitly distinguishes:
  1. *Sliding-window aggregate test MAE:* CAEG-Net V1 achieved $8.08 \pm 0.39$ MW vs Static Equal Ensemble $8.19$ MW.
  2. *Daily-block paired analysis:* Non-overlapping 24-hour block analysis favored the Equal Ensemble ($p=0.0012$; Wilcoxon $p=0.0036$).
  These represent distinct aggregation perspectives and are not conflated into "parity".

### 3.4 GEFCom2014 Statistical Interpretation
- **Audit Findings:** In `phase10_statistical_tests.csv`, the daily-block paired statistics for GEFCom ($K=456$ blocks) are:
  - Mean daily difference: $+0.48$ kW
  - $t$-statistic: $+5.730$
  - Paired $t$-test $p$-value: $1.82 \times 10^{-8}$
  - Wilcoxon signed-rank $p$-value: $9.66 \times 10^{-10}$
- **Correction:** The daily-block paired test statistically favors the Static Equal Ensemble. Given that expert residual errors are strongly correlated ($r \in [0.848, 0.908]$), uniform averaging is highly competitive ($12.66$ vs $12.81$ kW).

### 3.5 Numerical Reconciliation of 5-Seed Held-Out Test Partitions
- **Discrepancy:** The previous prose report contained draft/placeholder figures (`248.63`, `12.75`, `8.11`) that differed slightly from the actual generated CSV artifact `phase10_candidate_comparison.csv`.
- **Source of Truth Reconciliation:** In accordance with the source-of-truth hierarchy, all prose tables were corrected to match `phase10_candidate_comparison.csv` exactly:
  - **A0_Canonical_V1:**
    - PJM: MAE $251.17 \pm 12.61$ MW, RMSE $334.51 \pm 14.32$ MW, MSE $112,103 \pm 9,626$ MW², $R^2 = 0.8393 \pm 0.0182$, MAPE $4.63 \pm 0.21\%$
    - GEFCom: MAE $12.81 \pm 0.34$ kW, RMSE $18.43 \pm 0.41$ kW, MSE $339.91 \pm 15.30$ kW², $R^2 = 0.8217 \pm 0.0170$, MAPE $8.73 \pm 0.27\%$
    - UCI: MAE $8.08 \pm 0.39$ MW, RMSE $11.38 \pm 0.49$ MW, MSE $129.69 \pm 11.24$ MW², $R^2 = 0.9813 \pm 0.0016$, MAPE $4.16 \pm 0.30\%$
  - **B1_Zero_Recent_Error_3D:**
    - PJM: MAE $255.73 \pm 5.96$ MW, RMSE $342.42 \pm 9.70$ MW, MSE $117,343 \pm 6,717$ MW², $R^2 = 0.8220 \pm 0.0172$, MAPE $4.72 \pm 0.14\%$
    - GEFCom: MAE $12.77 \pm 0.48$ kW, RMSE $18.47 \pm 0.46$ kW, MSE $341.27 \pm 17.32$ kW², $R^2 = 0.8275 \pm 0.0107$, MAPE $8.77 \pm 0.30\%$
    - UCI: MAE $8.18 \pm 0.20$ MW, RMSE $11.66 \pm 0.37$ MW, MSE $136.01 \pm 8.68$ MW², $R^2 = 0.9802 \pm 0.0011$, MAPE $4.13 \pm 0.11\%$
  - **C3_Softer_Temperature:**
    - PJM: MAE $254.28 \pm 4.18$ MW, RMSE $339.09 \pm 7.90$ MW, MSE $115,047 \pm 5,355$ MW², $R^2 = 0.8328 \pm 0.0169$, MAPE $4.69 \pm 0.08\%$
    - GEFCom: MAE $12.86 \pm 0.28$ kW, RMSE $18.51 \pm 0.34$ kW, MSE $342.73 \pm 12.81$ kW², $R^2 = 0.8184 \pm 0.0116$, MAPE $8.76 \pm 0.22\%$
    - UCI: MAE $8.34 \pm 0.62$ MW, RMSE $11.86 \pm 0.88$ MW, MSE $141.38 \pm 22.03$ MW², $R^2 = 0.9795 \pm 0.0029$, MAPE $4.21 \pm 0.28\%$

### 3.6 Causal & Scientific Language Corrections
1. **Expert Correlation:** Replaced "Adaptive gating thrives when pairwise correlation is approximately 0.60" with:
   > *"The observed results are consistent with the hypothesis that greater expert error diversity provides more opportunity for adaptive routing to improve over uniform averaging."*
2. **Context Scale Features:** Replaced "raw physical units are essential" with:
   > *"Within the tested formulations, retaining raw-scale context features provided better validation performance, particularly on PJM."*
3. **Recent Error Feedback:** Replaced "Recent error feedback anchors the gating network during inference" with:
   > *"The results suggest that recent forecast-error feedback provides useful information to the routing mechanism."*
4. **UCI Load Dynamics:** Replaced unsupported causal assertions with:
   > *"The UCI results are consistent with a setting in which the LSTM expert is particularly effective and adaptive routing assigns greater average weight to it."*
5. **GEFCom Load Dynamics:** Replaced overgeneralized assertions with:
   > *"The high residual correlations are consistent with reduced opportunity for adaptive routing to exploit complementary expert errors."*

### 3.7 Verification of Daily-Block Counts ($K$)
- The daily-block counts were audited against `phase10_statistical_tests.csv` and the test set size:
  - **PJM:** $N_{\text{test}} = 1,294$ windows $\implies 1,294 // 24 = 53$ non-overlapping blocks.
  - **GEFCom:** $N_{\text{test}} = 10,944$ windows $\implies 10,944 // 24 = 456$ non-overlapping blocks.
  - **UCI:** $N_{\text{test}} = 3,922$ windows $\implies 3,922 // 24 = 163$ non-overlapping blocks.
- All $K$ values reported in the prose report match the actual computation.

### 3.8 Temperature Parameter Verification
- Candidate `C3_Softer_Temperature` was verified in `run_phase10_optimization.py` (line 525) and `PHASE_10_OPTIMIZATION_DESIGN.md` (line 110) as having temperature **$\tau = 1.25$** (and `C3_Sharper_Temperature` having **$\tau = 0.8$**).
- Earlier draft text loosely noted "$\tau=1.5$" and "$\tau=0.7$"; this was reconciled across all documentation to match the true execution code ($\tau = 1.25$ and $\tau = 0.8$).

---

## 4. Verification of the Central Scientific Decision

The core research question investigated whether controlled modifications could improve CAEG-Net across datasets:
1. Two candidates passed validation screening (**B1** and **C3**).
2. On the subsequent 5-seed evaluation on held-out test partitions:
   - **B1** showed higher error than Canonical V1 on PJM ($255.73$ vs $251.17$ MW) and UCI ($8.18$ vs $8.08$ MW), with near-identical error on GEFCom ($12.77$ vs $12.81$ kW).
   - **C3** showed higher error than Canonical V1 on all three datasets ($254.28$ vs $251.17$ MW on PJM; $12.86$ vs $12.81$ kW on GEFCom; $8.34$ vs $8.08$ MW on UCI).
3. **Canonical CAEG-Net V1** achieved the lowest test MAE on PJM and UCI, and competitive test MAE on GEFCom, demonstrating the strongest overall cross-dataset empirical generalization among tested formulations.

**Final Decision:** **Canonical CAEG-Net V1 is retained as the frozen research architecture.**

---

## 5. Certification Statement

- No experiment was rerun during this audit.
- No CSV artifact or numerical result was altered.
- All prose claims, tables, and statistical summaries now strictly align with the CSV source of truth and defensive scientific standards.

**Audited by:** CAEG-Net Research Assistant  
**Date:** September 10, 2026
