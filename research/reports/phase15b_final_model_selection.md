# Phase 15B-C: Final Model Selection & Architecture Lock
## CAEG-Net Research Program — Definitive Publication Model Certification

**Authoritative Report — Multi-Criteria Selection & Final Lock**  
**Date:** September 2026  
**Status:** COMPLETE & RECONCILED  
**Certified Final Model:** `F2_A2_OOF` (`ConfidenceFallbackCAEGNet`)  
**Certification Decision:** **OUTCOME B (F2 REMAINS FINAL MODEL WITH QUALIFIED MULTI-CRITERIA JUSTIFICATION)**

---

## 1. Multi-Criteria Decision Framework

The CAEG-Net Phase 15B model selection follows a pre-registered multi-criteria evaluation framework designed to balance predictive accuracy, stability across stochastic seeds, statistical defensibility, architectural simplicity, and methodological integrity.

### Decision Criteria Hierarchy:
1. **Criterion 1 (Validation Screening Firewall):** Only candidates meeting the predefined validation qualification rule (improving $\ge 2/3$ datasets with $\le 2.0\%$ worst degradation) are eligible for held-out finalist adoption.
2. **Criterion 2 (Cross-Dataset Performance Balance):** 5-seed mean Test MAE evaluated across three distinct operational load cohorts (PJM utility-level, GEFCom regional-level, UCI customer-level).
3. **Criterion 3 (Statistical Defensibility):** Paired daily-block hypothesis testing ($K=53, 456, 163$) with Holm-Bonferroni correction.
4. **Criterion 4 (Architectural Simplicity & Parsimony):** Preference for models with minimal parameter overhead and robust operating principles over unregularized complex variants.

---

## 2. Multi-Criteria Decision Matrix

| Evaluation Dimension | Control A (Current F2) | Control B (Fixed $\lambda=0.51$) | Control C (Horizon Only) | Control D (Dynamic Conf) | Candidate E1 (HGR-FS) | Candidate E2 (HGR-DGS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PJM 5-Seed Test MAE (MW)** | **250.97 ± 10.69** (Rank 1) | 253.50 ± 8.03 (Rank 3) | 257.54 ± 5.37 (Rank 4) | 251.94 ± 9.80 (Rank 2) | 257.91 ± 7.33 (Rank 6) | 256.07 ± 7.11 (Rank 5) |
| **GEFCom 5-Seed Test MAE (kW)** | 12.41 ± 0.15 (Rank 2) | **12.36 ± 0.18** (Rank 1) | 12.86 ± 0.25 (Rank 6) | 12.49 ± 0.27 (Rank 3) | 12.52 ± 0.13 (Rank 4) | 12.72 ± 0.25 (Rank 5) |
| **UCI 5-Seed Test MAE (MW)** | **7.74 ± 0.30** (Rank 1) | 7.75 ± 0.18 (Rank 2) | 8.13 ± 0.40 (Rank 4) | 7.82 ± 0.28 (Rank 3) | 8.19 ± 0.44 (Rank 5) | 8.30 ± 0.47 (Rank 6) |
| **Average Benchmark Rank** | **1.33** | 2.00 | 4.67 | 2.67 | 5.00 | 5.33 |
| **Validation Screening Firewall** | **QUALIFIED (Anchor)** | REJECTED (0/3) | REJECTED (0/3) | REJECTED (1/3) | REJECTED (0/3) | REJECTED (0/3) |
| **Statistically Significant Win?** | N/A (Baseline Anchor) | No (Parity) | No (Degraded GEFCom) | No (Degraded GEFCom) | No | No (Degraded GEFCom) |
| **5-Seed Stability (Mean SD)** | ± 3.72 | **± 2.79** | ± 2.01 | ± 3.45 | ± 2.63 | ± 2.61 |
| **Total Parameter Count** | 121,724 | **121,579** (-145) | 122,852 (+1,128) | 121,740 (+16) | 122,852 (+1,128) | 123,079 (+1,355) |
| **Final Selection Status** | **LOCKED FINAL MODEL** | REJECTED (Parity) | REJECTED (Degraded) | REJECTED (Degraded) | REJECTED (Degraded) | REJECTED (Degraded) |

*Note: 5-seed statistics represent mean ± population standard deviation (ddof=0) across seeds {42, 123, 999, 2024, 3407}.*

---

## 3. Exhaustive Rationale for Final Selection

### Explicit Preservation of Empirical Winners:
- **PJM numerical winner:** **Control A (F2)** (250.9747 MW vs Control B 253.5008 MW)
- **GEFCom numerical winner:** **Control B (Fixed Shrinkage)** (12.3607 kW vs Control A 12.4077 kW)
- **UCI numerical winner:** **Control A (F2)** (7.7371 MW vs Control B 7.7523 MW)

### Why F2 is Retained Over Control B (Fixed Shrinkage):
1. **Validation Firewall Decision:** Control B failed the validation qualification firewall (0/3 datasets improved; worst degradation +2.20% on UCI). Under pre-registered protocol rules, no candidate qualified to replace the baseline anchor.
2. **Cross-Dataset Performance Balance:** Control A (F2) achieved lower 5-seed mean MAE on 2 of the 3 benchmark datasets (PJM: -2.53 MW; UCI: -0.015 MW). Control B achieved a slightly lower mean MAE on GEFCom (-0.047 kW), but this difference is not statistically significant under Holm-adjusted paired $t$-testing ($p = 0.2536$). F2 therefore provides the best overall multi-dataset balance.
3. **No Claim of Universal Optimality:** F2 is explicitly **not claimed to be universally optimal on every dataset**, as fixed shrinkage achieved a slightly lower mean MAE on GEFCom.

### Why Horizon Routing (Control C, E1, E2) Was Decisively Rejected:
1. **Screening Failure:** All horizon-partitioned variants degraded validation MAE by $+10.03\%$ to $+13.27\%$ on PJM.
2. **Statistically Significant Degradation:** On GEFCom, Control C and Candidate E2 produced statistically significant degradations relative to F2 ($p = 7.73 \times 10^{-12}$ and $p = 7.16 \times 10^{-9}$).
3. **Lack of Empirical Support:** While retrospective analysis showed expert performance differences across horizons, explicitly parameterizing horizon-specific routing heads failed to improve generalization.

---

## 4. Formal Certification Certificate

```
================================================================================
                    CAEG-Net FINAL MODEL SELECTION CERTIFICATE
================================================================================
CERTIFIED DECISION:
    OUTCOME B: F2 REMAINS FINAL MODEL WITH QUALIFIED MULTI-CRITERIA JUSTIFICATION

MODEL IDENTITY:
    Architecture:        ConfidenceFallbackCAEGNet (F2 / A2-OOF)
    Module Class:        research.models.ConfidenceFallbackCAEGNet
    Code Implementation: research/models.py, research/models_phase15b.py (ControlA_CurrentF2)

PARAMETER SPECIFICATION:
    Total Parameters:    121,724
    Backbone Parameters: 120,504 (LSTM: 56,152 | TCN: 36,952 | CNN: 27,400) [Frozen]
    Router Parameters:   1,075
    Confidence Head:     145
    Parameter Overhead:  0.00% (Authoritative Baseline Reference)

OFFICIAL 5-SEED TEST BENCHMARK RESULTS (Seeds: 42, 123, 999, 2024, 3407):
    PJM (MW):           MAE = 250.9747 ± 10.6938 | RMSE = 335.38 | R2 = 0.8714
    GEFCom (kW):        MAE =  12.4077 ±  0.1525 | RMSE =  18.04 | R2 = 0.8610
    UCI (MW):           MAE =   7.7371 ±  0.3037 | RMSE =  10.96 | R2 = 0.9831

MULTI-CRITERIA JUSTIFICATION:
    1. Highest overall rank across tri-benchmark suite (Average Rank 1.33).
    2. Strict adherence to Two-Stage Validation Firewall (no candidate qualified).
    3. F2 achieved lowest mean MAE on PJM and UCI; Control B was slightly lower on GEFCom.
    4. Statistically superior or equal to all evaluated complex mechanism candidates.
    5. The final evaluated CAEG-Net formulation provides a strong forecasting-performance/complexity trade-off through a parsimonious architecture.

STATUS:
    MODEL FROZEN. NO FURTHER ARCHITECTURAL MODIFICATIONS PERMITTED.
================================================================================
```

---

## 5. Scientific Limitations and Interpretation Boundaries

To ensure complete transparency and rigorous publication defense, the following boundaries govern the interpretation of the CAEG-Net results:

1. **Non-Universal Optimality:** F2 is not claimed to be universally superior across all load series or metrics. Control B (fixed shrinkage $\lambda = 0.51$) achieved a slightly lower mean MAE on GEFCom ($12.36$ kW vs $12.41$ kW).
2. **Nature of 5-Seed Statistics:** Reported `mean ± SD` values reflect population standard deviations (`ddof=0`) across 5 random seeds ({42, 123, 999, 2024, 3407}). These represent stochastic sensitivity to network initialization and data ordering, not independent dataset replications.
3. **Retrospective Oracle Non-Deployability:** The retrospective convex oracle accesses unobserved future targets $\mathbf{y}_t$ to compute an idealized upper-bound ceiling. It is strictly non-deployable and does not represent a causal forecasting method or an achievable operational target.
4. **Horizon Specialization Interpretation:** The observation that individual experts perform better at specific horizons in retrospective analysis did not translate into a successful forward predictive mechanism. The underlying cause of this empirical degradation remains unmeasured.
5. **Confidence Head Behavior:** The learned confidence mechanism in F2 exhibits low temporal variation ($	ext{CV} < 1.5\%$, range $[0.48, 0.54]$) and functions primarily as near-constant shrinkage toward the equal-expert ensemble, rather than instance-level difficulty calibration.
6. **No Bayesian Prior Claim:** The uniform-centroid shrinkage mechanism is an empirical linear interpolation. The experiments do not establish that shrinkage is theoretically mandatory or Bayesian.
7. **No Unvalidated Latency Claims:** Wall-clock inference times measured during batch test evaluation do not constitute a formal micro-benchmark (due to lack of CUDA synchronization and warm-up isolation). Headline real-time latency claims are excluded.
8. **Scope of Data:** PJM represents a single utility-level aggregate balancing authority. External weather covariates and holiday schedules were intentionally excluded to evaluate pure autoregressive gating.
