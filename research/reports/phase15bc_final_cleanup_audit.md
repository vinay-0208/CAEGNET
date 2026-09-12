# Phase 15B-C: Final Cleanup Audit & Model-Lock Verification
## CAEG-Net Research Program — Definitive Model-Development Closeout

**Authoritative Report — Final Model-Lock Verification**  
**Date:** September 2026  
**Status:** COMPLETE, VERIFIED & LOCKED  

---

## 1. Historical Commit Preservation

The historical integrity of the repository is preserved across all development stages:
- **Historical Phase 15B Experimental Commit:** `47d05bc` (*research: complete Phase 15B controlled mechanism experiments and lock final model*)
- **Historical Phase 15B-C Correction Commit:** `fa7dbea` (*research: correct Phase 15B interpretation and final model audit*)
- **Current Final Cleanup Commit:** Created upon completion of this pass (`research: finalize Phase 15B model-lock wording`).

Neither historical commit has been amended, rebased, or altered.

---

## 2. Verification of the Two Required Wording Corrections

### Wording Correction #1 (Shrinkage Stabilization):
- **Audited Target:** Replaced all variants of *"confirming that pulling adaptive weights toward the equal-expert centroid provides an essential practical stabilization mechanism"* or universal necessity claims.
- **Adopted Wording:**
  > *"indicating that constraining adaptive weights toward the equal-expert centroid can provide an important practical stabilization mechanism in the evaluated settings."*
- **Verification:** Verified across `phase15b_experiment_report.md`, `phase15b_final_model_selection.md`, and `walkthrough.md`. No claims of "mandatory regularizer" or "Bayesian prior" remain.

### Wording Correction #2 (Architecture Parsimony vs. Peak Optimality):
- **Audited Target:** Replaced all variants of *"CAEG-Net achieves its peak forecasting performance through a parsimonious architecture"* or claims implying global optimality.
- **Adopted Wording:**
  > *"The final evaluated CAEG-Net formulation provides a strong forecasting-performance/complexity trade-off through a parsimonious architecture."*
- **Verification:** Verified in all reports. No claims of "peak", "globally optimal", or "best possible architecture" are present.

---

## 3. Final Model Verification

The final locked model remains strictly:
- **Model Identifier:** `F2_A2_OOF` (`ConfidenceFallbackCAEGNet`)
- **Module Implementation:** `research.models.ConfidenceFallbackCAEGNet` / `research.models_phase15b.ControlA_CurrentF2`
- **Parameter Breakdown:**
  * LSTM Expert: 56,152 parameters
  * TCN Expert: 36,952 parameters
  * CNN Expert: 27,400 parameters
  * **Frozen Expert Core Total:** **120,504 parameters**
  * Adaptive Router: 1,075 parameters
  * Dynamic Confidence Head: 145 parameters
  * **Grand Total:** **121,724 parameters**
  * **Trainable Parameter Delta vs. Core:** **1,220 parameters** (+1.01% overhead)

---

## 4. Unaltered Numerical Benchmark Results

All five-seed benchmark results are strictly preserved from the Phase 15B experimental execution:

| Benchmark Dataset | Metric | Control A (Current F2) | Control B (Fixed $\lambda=0.51$) | Control C (Horizon Only) | Control D (Dynamic Conf) | Candidate E1 (HGR-FS) | Candidate E2 (HGR-DGS) | Empirical Numerical Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PJM** | **MAE (MW)** | **250.9747 ± 10.6938** | 253.5008 ± 8.0264 | 257.5450 ± 5.3670 | 251.9419 ± 9.7996 | 257.9078 ± 7.3293 | 256.0708 ± 7.1071 | **Control A (F2)** |
| | RMSE (MW) | 335.38 ± 12.35 | 338.95 ± 8.16 | 343.91 ± 4.66 | 336.74 ± 11.11 | 343.54 ± 8.76 | 342.79 ± 7.67 | |
| | $R^2$ | 0.8714 ± 0.0094 | 0.8688 ± 0.0064 | 0.8650 ± 0.0036 | 0.8704 ± 0.0086 | 0.8652 ± 0.0067 | 0.8658 ± 0.0060 | |
| **GEFCom** | **MAE (kW)** | 12.4077 ± 0.1525 | **12.3607 ± 0.1841** | 12.8582 ± 0.2520 | 12.4855 ± 0.2685 | 12.5240 ± 0.1316 | 12.7220 ± 0.2457 | **Control B (Fixed $\lambda$)** |
| | RMSE (kW) | 18.04 ± 0.15 | 18.02 ± 0.15 | 18.44 ± 0.24 | 18.10 ± 0.20 | 18.26 ± 0.09 | 18.35 ± 0.20 | |
| | $R^2$ | 0.8610 ± 0.0024 | 0.8614 ± 0.0023 | 0.8548 ± 0.0038 | 0.8601 ± 0.0030 | 0.8577 ± 0.0014 | 0.8562 ± 0.0031 | |
| **UCI** | **MAE (MW)** | **7.7371 ± 0.3037** | 7.7523 ± 0.1814 | 8.1309 ± 0.4048 | 7.8177 ± 0.2838 | 8.1932 ± 0.4413 | 8.2969 ± 0.4746 | **Control A (F2)** |
| | RMSE (MW) | 10.96 ± 0.32 | 10.99 ± 0.14 | 11.48 ± 0.63 | 11.00 ± 0.27 | 11.66 ± 0.71 | 11.81 ± 0.73 | |
| | $R^2$ | 0.9831 ± 0.0010 | 0.9830 ± 0.0004 | 0.9814 ± 0.0021 | 0.9830 ± 0.0008 | 0.9808 ± 0.0024 | 0.9803 ± 0.0025 | |

*Note: All values reflect mean ± population standard deviation (ddof=0) across 5 seeds {42, 123, 999, 2024, 3407}. F2 achieves the lowest mean MAE on PJM and UCI; Control B achieves a slightly lower mean MAE on GEFCom (-0.047 kW, p=0.2536). F2 is explicitly NOT claimed to be universally optimal on every dataset.*

---

## 5. Verification of Protocol & Language Standards

1. **Test-Firewall Language:** Verified that zero proposed mechanism candidates qualified under the predefined validation screening rule, and that all post-screening test evaluations for non-anchor candidates are classified strictly as exploratory diagnostics.
2. **Statistical Rigor:** All hypothesis tests utilize non-overlapping daily blocks ($K=53$ PJM, $K=456$ GEFCom, $K=163$ UCI) with Holm-Bonferroni FWER control. Discrepancies between paired $t$-test and Wilcoxon are transparently documented.
3. **F2 Confidence Dynamics:** Reconciled as exhibiting low temporal variation ($	ext{CV} < 1.5\%$, range $[0.48, 0.54]$), acting as near-constant shrinkage toward the equal ensemble.
4. **Horizon-Routing Framing:** Accurately characterized as: *"Retrospective Horizon Specialization Does Not Translate into Improved Explicit Horizon Routing."* No claims of causal failure or gradient fragmentation.
5. **Shrinkage Framing:** Characterized as an empirical practical stabilization mechanism in the evaluated settings, without theoretical necessity or Bayesian prior claims.
6. **Disagreement Framing:** Accurately characterized as: *"Tested disagreement-conditioned mechanisms did not provide robust forecasting gains."*
7. **Latency Reporting:** The unvalidated 0.120 ms headline claim is completely removed.
8. **Scientific Boundaries Section:** Fully established in both `phase15b_experiment_report.md` and `phase15b_final_model_selection.md`.

---

## 6. Verification of Automated Test Suite

- Total automated tests across repository: **172 tests** (17 test modules).
- Status: **172 / 172 PASSED (100% pass rate)**.
- Python compilation check (`py_compile`): **PASS**.

---

## 7. Model-Lock Certification Statement

> **"F2/A2-OOF remains the final model because it provides the strongest overall balance of cross-dataset forecasting performance, seed stability, methodological integrity, and architectural simplicity among the evaluated formulations.**  
> **F2 achieved the lowest five-seed mean MAE on PJM and UCI, while fixed shrinkage achieved a slightly lower mean MAE on GEFCom. Therefore, F2 is not claimed to be universally optimal on every individual dataset."**
