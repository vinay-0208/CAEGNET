# Phase 15A-C — Forensic Diagnostic Correction, Reconciliation & Audit Report
**CAEG-Net: Dynamic Routing, Confidence, Expert Complementarity & Oracle-Ceiling Analysis**

**Date:** 2026-09-11  
**Authoritative Reference Model:** Candidate F2 / A2-OOF (`ConfidenceFallbackCAEGNet`, 121,724 parameters)  
**Evaluation Seeds:** {42, 123, 999, 2024, 3407}  
**Datasets:** Modern PJM (MW), GEFCom2014 (kW), UCI Electricity Cohort 320 Aggregate (MW)  
**Authoritative Commits:** `c67067d8` (Phase 14 Historical), `1d4d8c35` (Scientific Correction), `cfc75e2b` (Documentation Lock), `45443fa` (Phase 15A Original Diagnostics)  
**Status:** COMPLETE, AUDITED, RECONCILED & LOCKED

---

## 1. Executive Summary & Purpose

Phase 15A executed a comprehensive 14-point diagnostic audit of the frozen CAEG-Net canonical model (Candidate F2 / A2-OOF). Subsequent forensic review identified critical provenance, numerical, and interpretation discrepancies:
1. **Critical Seed-42 vs. 5-Seed Provenance (C0):** The exact provenance of 249.901 MW on PJM was traced to the Seed 42 realization within the Phase 14 benchmark, whereas 250.97 MW is the 5-seed mean.
2. **UCI Baseline Inconsistency (C1):** 7.55 MW is the Phase 12 validation baseline, while 7.79 MW is the Phase 11 held-out test benchmark. F2 achieves 7.74 MW, beating the test benchmark.
3. **Residual Correlation Contradiction (C2):** The positive correlation (+0.60 to +0.91) in Phase 10/11 describes standalone networks, whereas the negative correlation (-0.30 to -0.72) in Phase 15A describes co-adapted internal branches of F2.
4. **Routing Dynamicity / Stationarity Wording (C3):** Retired "stationary" terminology in favor of "limited continuous dynamic variation with absence of frequent top-1 rank switching."
5. **Confidence Head Calibration & Shrinkage (C4 & C5):** Verified that $\lambda$ operates as stationary learned shrinkage ($\lambda \approx 0.51$, CV $< 1.5\%$) rather than dynamic calibrated confidence ($p > 0.20$).
6. **Decision Quality & Regret Units (C6 & C7):** Replaced colloquial phrasing with formal selection regret and fusion regret in physical units (MW/kW).
7. **Oracle Demarcation & Headroom (C8 & C9):** Explicitly certified oracle convex fusion as a retrospective non-deployable upper bound and replaced misleading headroom percentages with absolute MAE gaps.
8. **Routing vs. Shrinkage Causal Claims (C10):** Formatted as a descriptive linear mixture, noting that percentages $>100\%$ represent counteracting descriptive vectors rather than independent causal contributions.
9. **Disagreement, Complementarity, Horizon, Features & Regimes (C11 – C15):** Verified and updated all supporting analyses without causal claims.

---

## 2. Comprehensive Correction Register (Issues C0 – C15)

| ID | Issue | Previous Phase 15A Claim | Source of Truth | Investigation | Correction | Recomputed? | Final Status | Affected Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **C0** | **PJM 249.901 vs. 250.97 MW Provenance** | Claimed 249.901 MW matched Phase 14 benchmark without noting seed | `phase14_test_results.csv` (5-seed) vs `phase14_routing_statistics.csv` (Seed 42) | Verified that Seed 42 test MAE is exactly 249.9014 MW, while 250.9747 MW is the 5-seed mean ({42: 249.90, 123: 262.38, 999: 233.76, 2024: 262.18, 3407: 246.65}) | Formally distinguished Seed 42 deterministic evaluation from 5-seed aggregate benchmark | Yes | **RESOLVED** | `phase15a_phase14_provenance_reconciliation.csv`, report |
| **C1** | **UCI Standalone Baseline Inconsistency** | 7.79 MW in Phase 15A vs 7.55 MW in Phase 13/14 | `phase12_dataset_summary.csv` (Val) vs `phase11_dataset_summary.csv` (Test) | Traced 7.554 MW to Phase 12 validation split and 7.794 MW to Phase 11 held-out test split | Classified 7.554 MW as Val BL and 7.794 MW as Test BM; F2 (7.737 MW) beats Test BM by -0.057 MW (-0.74%) | Yes | **RESOLVED** | `phase15a_baseline_provenance.csv`, report |
| **C2** | **Residual Correlation Contradiction** | Negative (-0.30 to -0.72) vs Phase 10/11 Positive (+0.60 to +0.91) | `phase10_expert_complementarity.csv` vs `phase15a_complementarity.csv` | Proved that standalone models have positive covariance due to macro-shocks, while co-adapted F2 branches have negative covariance due to mixture cancellation | Separated into Standalone Forecast Error Residuals (+0.60~+0.91) vs Co-Adapted Internal Branch Residuals (-0.30~-0.72) | Yes | **RESOLVED** | `phase15a_correlation_reconciliation.csv`, report |
| **C3** | **Routing Dynamicity / Stationarity** | Router called "stationary" because top expert rarely changed | Continuous routing weights ($s_w = 0.0036 - 0.0383$, $r_{\mathrm{lag1}} > 0.95$) | Calculated population SD ($ddof=0$), percentiles, and run lengths | Replaced "stationary" with "limited continuous dynamic variation with absence of frequent top-1 rank switching" | Yes | **RESOLVED** | `phase15a_routing_dynamicity_corrected.csv`, report |
| **C4** | **Confidence $\lambda$ Characterization** | Described as instance-varying confidence | `phase14_confidence_statistics.csv` | Evaluated dispersion: mean $\approx 0.509 - 0.513$, pop SD $< 0.008$, CV $< 1.5\%$ | Characterized as Category C (Approximately Constant) / effectively learned fixed shrinkage | Yes | **RESOLVED** | Report, analysis tables |
| **C5** | **Confidence Calibration** | Called "calibrated" without decision-value evidence | Daily-block paired differences ($K=53, 456, 163$) | Quantile binning and Spearman correlation yielded $p > 0.20$ across all datasets | Clarified that $\lambda$ is uncalibrated to instance difficulty; value comes from fixed shrinkage | Yes | **RESOLVED** | `phase15a_confidence_calibration_corrected.csv` |
| **C6** | **Router Decision Quality** | Lacked window-level error comparison to retrospective best expert | Window-level test errors in `phase11_window_metrics.csv` | Evaluated top-1 agreement ($22.9\% - 41.3\%$) and selection regret ($0.76 - 28.51$ units) | Documented that router operates close to equal weighting, capturing limited expert selection alpha | Yes | **RESOLVED** | `phase15a_router_decision_quality_corrected.csv` |
| **C7** | **Routing Regret Units** | Vaguely described as "12-25% of load" | Window-level test MAEs in physical units | Computed exact formulas for selection and fusion regret | Reported in physical units (MW/kW) and explicit relative % of standalone benchmark | Yes | **RESOLVED** | `phase15a_routing_regret_corrected.csv` |
| **C8** | **Oracle Convex Fusion Demarcation** | Unqualified "opportunity" | Retrospective 2-simplex daily grid search | Verified weights satisfy $w_i \ge 0$ and $\sum w_i = 1$ using hindsight targets | Formally labeled RETROSPECTIVE ORACLE / NON-DEPLOYABLE UPPER BOUND | Yes | **RESOLVED** | `phase15a_oracle_audit.csv` |
| **C9** | **Oracle Headroom Interpretation** | Claimed F2 "captures X% of headroom" | Hindsight oracle grid search | Removed deployability implications; showed that oracle gaps cannot be closed causally | Replaced with absolute MAE gaps to retrospective bounds | Yes | **RESOLVED** | Report, analysis tables |
| **C10** | **Routing vs. Shrinkage Causal Claims** | Counteracting percentages ($-123\%$ routing, $+223\%$ shrinkage) framed causally | End-to-end trained linear mixture $\hat{y}_{\mathrm{final}} = \lambda \hat{y}_{\mathrm{adaptive}} + (1-\lambda) \hat{y}_{\mathrm{equal}}$ | Demonstrated that pure adaptive routing degrades GEFCom and UCI, but shrinkage offsets it | Framed strictly as descriptive linear mixture; noted mutual co-adaptation | Yes | **RESOLVED** | `phase15a_shrinkage_reconciliation.csv` |
| **C11** | **Disagreement Analysis** | Vague association with difficulty | Binned disagreement spread vs adaptive advantage | Verified that high disagreement regimes correspond to modest adaptive advantage ($53\% - 58\%$) | Clarified descriptive, non-causal association | Yes | **RESOLVED** | `phase15a_disagreement_corrected.csv` |
| **C12** | **Expert Complementarity** | Residual correlation confused with complementarity | Standalone vs co-adapted errors | Evaluated regime- and horizon-dependent complementarity | Documented that standalone models are complementary across horizons despite positive correlation | Yes | **RESOLVED** | `phase15a_complementarity_corrected.csv` |
| **C13** | **Horizon Specialization** | Potential overstatement of horizon routing | Step 1-24 multi-horizon test errors | Verified TCN dominance on short horizons ($h \le 8$) and LSTM on long horizons ($h \ge 17$) | Confirmed empirical crossover, motivating controlled testing without guaranteeing gains | Yes | **RESOLVED** | `phase15a_horizon_specialization_corrected.csv` |
| **C14** | **Performance-Feature Resolution** | Lookback window coarseness | 24h, 48h, 72h, and 168h OOF lookback error features | Confirmed that 168h chronological OOF error achieves lowest test MAE across all benchmarks | Documented feature resolution findings | Yes | **RESOLVED** | `phase15a_performance_resolution_corrected.csv` |
| **C15** | **Regime Analysis** | Potential causal interpretation of regime gains | Volatility and diurnal sub-regime breakdowns | Confirmed that F2 gains concentrate in peak and high-volatility regimes | Clarified retrospective descriptive nature of regime analysis | Yes | **RESOLVED** | `phase15a_regime_diagnostics_corrected.csv` |

---

## 3. Detailed Forensic Tracing of C0 (PJM 249.901 vs. 250.97 MW)

- **The Issue:** Phase 15A reported that F2 achieved a test MAE of 249.901 MW on PJM and stated that this matched Phase 14 records, while the locked Phase 14 benchmark is $250.97 \pm 10.69$ MW.
- **Forensic Investigation:**
  - In `research/results/phase14_routing_statistics.csv` and `research/results/phase14_confidence_statistics.csv`, Seed 42 is explicitly logged:
    - Routing weights: LSTM $= 0.367928$, TCN $= 0.326085$, CNN $= 0.305988$.
    - Confidence $\lambda$: $0.509220 \pm 0.005576$.
  - Re-evaluating the 5 seeds under the Phase 14 protocol yields:
    - **Seed 42:** **249.9014 MW**
    - **Seed 123:** **262.3826 MW**
    - **Seed 999:** **233.7615 MW**
    - **Seed 2024:** **262.1776 MW**
    - **Seed 3407:** **246.6506 MW**
    - **5-Seed Mean:** $\mathbf{250.9747}$ **MW**
    - **5-Seed Population SD ($ddof=0$):** $\mathbf{10.6938}$ **MW**
    - **5-Seed Sample SD ($ddof=1$):** $\mathbf{11.9561}$ **MW**
- **Conclusion:** 249.901 MW is the exact single-seed realization of Seed 42 in Phase 14. Phase 15A used Seed 42 for detailed internal activations and window-level diagnostics. 250.97 MW is the authoritative 5-seed mean. Both numbers are authentic, reproducible, and fully reconciled in `research/analysis/phase15a_phase14_provenance_reconciliation.csv`.

---

## 4. Verification & Audit Certification

- **Parameter Counts:** Frozen at 121,724 parameters.
- **Data Splits & Leakage:** Zero future leakage; test data evaluated strictly post-hoc.
- **Test Suite:** 145 / 145 tests passing project-wide (0 failures, 0 errors).
- **Certification:** The Phase 15A diagnostic evidence base is complete, reconciled, and scientifically sound. The repository is certified **READY FOR PHASE 15B**.
