# Phase 15B: Final Model Selection & Architecture Lock
## CAEG-Net Research Program — Definitive Publication Model Certification

**Authoritative Report — Multi-Criteria Selection & Final Lock**  
**Date:** September 2026  
**Status:** COMPLETE & LOCKED  
**Certified Final Model:** `F2_A2_OOF` (`ConfidenceFallbackCAEGNet`)

---

## 1. Multi-Criteria Decision Framework

The CAEG-Net Phase 15B model selection follows a pre-registered multi-criteria evaluation framework designed to balance predictive accuracy, stability, statistical significance, architectural parsimony, and computational efficiency.

### Decision Criteria Hierarchy
1. **Criterion 1 (Predictive Accuracy):** 5-seed mean Test MAE across all three benchmarks (PJM utility-level, GEFCom regional-level, UCI customer-level).
2. **Criterion 2 (Generalization Breadth):** Candidate must demonstrate superior or equivalent performance across $\ge 2$ of the 3 distinct load cohorts.
3. **Criterion 3 (Statistical Defensibility):** Paired daily-block hypothesis tests ($p < 0.05$ after Holm-Bonferroni correction) confirming improvements are not random artifacts.
4. **Criterion 4 (Parameter Efficiency & Complexity):** Minimal parameter overhead; no unneeded architectural complexity without demonstrated empirical gain.
5. **Criterion 5 (Computational Latency):** Sub-millisecond inference time per 24-hour forecast window suitable for real-time grid deployment.

---

## 2. Multi-Criteria Decision Matrix

| Evaluation Criterion | Control A (Current F2) | Control B (Fixed $\lambda=0.51$) | Control C (Horizon Only) | Control D (Dynamic Conf) | Candidate E1 (HGR-FS) | Candidate E2 (HGR-DGS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PJM Test MAE (MW)** | **250.97 ± 10.69** (Rank 1) | 253.50 ± 8.03 (Rank 3) | 257.54 ± 5.37 (Rank 4) | 251.94 ± 9.80 (Rank 2) | 257.91 ± 7.33 (Rank 6) | 256.07 ± 7.11 (Rank 5) |
| **GEFCom Test MAE (kW)** | 12.41 ± 0.15 (Rank 2) | **12.36 ± 0.18** (Rank 1) | 12.86 ± 0.25 (Rank 6) | 12.49 ± 0.27 (Rank 3) | 12.52 ± 0.13 (Rank 4) | 12.72 ± 0.25 (Rank 5) |
| **UCI Test MAE (MW)** | **7.74 ± 0.30** (Rank 1) | 7.75 ± 0.18 (Rank 2) | 8.13 ± 0.40 (Rank 4) | 7.82 ± 0.28 (Rank 3) | 8.19 ± 0.44 (Rank 5) | 8.30 ± 0.47 (Rank 6) |
| **Average Benchmark Rank** | **1.33** | 2.00 | 4.67 | 2.67 | 5.00 | 5.33 |
| **Validation Screening Firewall** | **QUALIFIED (Anchor)** | FAILED (0/3) | FAILED (0/3) | FAILED (1/3) | FAILED (0/3) | FAILED (0/3) |
| **Statistically Significant Win?** | N/A (Baseline Anchor) | No (Parity) | No (Degraded GEFCom) | No (Degraded GEFCom) | No | No (Degraded GEFCom) |
| **5-Seed Stability (Mean Std)** | ± 3.72 | **± 2.79** | ± 2.01 | ± 3.45 | ± 2.63 | ± 2.61 |
| **Total Parameter Count** | 121,724 | **121,579** (-145) | 122,852 (+1,128) | 121,740 (+16) | 122,852 (+1,128) | 123,079 (+1,355) |
| **Inference Time / Window** | **0.120 ms** | 0.126 ms | 0.074 ms | 0.137 ms | 0.075 ms | 0.082 ms |
| **Final Selection Status** | **LOCKED FINAL MODEL** | REJECTED (Parity) | REJECTED (Degraded) | REJECTED (Degraded) | REJECTED (Degraded) | REJECTED (Degraded) |

---

## 3. Exhaustive Rationale for Final Selection

### Why Control B (Fixed Shrinkage) Was Not Selected Over F2
1. **Marginal Numerical Differences:** Control B reduces parameter count by 145 parameters by removing the confidence head and fixing $\lambda = 0.510$. While it achieves a minor improvement on GEFCom (12.36 kW vs 12.41 kW, $-0.4\%$, statistically non-significant), it degrades PJM by $+2.53$ MW (253.50 MW vs 250.97 MW, $+1.01\%$) and UCI by $+0.015$ MW.
2. **Validation Firewall Rejection:** Control B failed the validation firewall by failing to improve $\ge 2$ datasets during screening.
3. **Loss of Adaptive Expressiveness:** Removing the dynamic confidence head entirely eliminates the theoretical capability to adjust shrinkage in non-stationary conditions, offering no compelling advantage over the proven F2 formulation.

### Why Horizon-Group Routing (Control C, E1, E2) Was Decisively Rejected
1. **Universal Generalization Failure:** Horizon routing degraded performance across every dataset on the validation set ($+10.03\%$ to $+13.27\%$ on PJM) and the test set ($+6.57$ MW on PJM, $+0.45$ kW on GEFCom, $+0.39$ MW on UCI).
2. **Statistically Significant Degeneracy:** On GEFCom, Control C and Candidate E2 produced statistically significant degradations ($p = 7.73 \times 10^{-12}$ and $p = 7.16 \times 10^{-9}$).
3. **Severe Overfitting:** Splitting routing across horizons triples router degrees of freedom and decouples the multi-horizon temporal covariance, proving that horizon-partitioned gating is scientifically unviable for 24-step load forecasting.

### Why Disagreement-Modulated Confidence (Control D) Was Rejected
1. **Screening Rejection:** Control D improved only 1 of 3 validation benchmarks (UCI), while degrading PJM and GEFCom.
2. **Statistically Significant Degradation on GEFCom:** On test daily blocks, Control D was significantly worse than F2 ($p = 0.0008, d_z = +0.158$).
3. **High Variance:** Conditioning on instantaneous expert spread increases seed-to-seed sensitivity without offering consistent accuracy gains.

---

## 4. Formal Certification of Final Locked Model

In accordance with Section 27 and Section 34 of the CAEG-Net Phase 15B specification, the model selection process is officially completed and certified under **Outcome D**:

```
================================================================================
                    CAEG-Net FINAL MODEL SELECTION CERTIFICATE
================================================================================
CERTIFIED OUTCOME:
    OUTCOME D: F2 REMAINS THE FINAL LOCKED MODEL

MODEL IDENTITY:
    Architecture:        ConfidenceFallbackCAEGNet (F2 / A2-OOF)
    Module Class:        research.models.ConfidenceFallbackCAEGNet
    Checkpoint / Code:   research/models.py, research/models_phase15b.py (ControlA_CurrentF2)

PARAMETER SPECIFICATION:
    Total Parameters:    121,724
    Backbone Parameters: 120,504 (LSTM: 56,152 | TCN: 36,952 | CNN: 27,400)
    Router Parameters:   1,075
    Confidence Head:     145
    Parameter Overhead:  0.00% (Authoritative Reference)

OFFICIAL AUTHORITATIVE TEST BENCHMARK RESULTS (5 SEEDS):
    PJM (MW):           MAE = 250.9747 ± 10.6938 | RMSE = 335.38 | R2 = 0.8714
    GEFCom (kW):        MAE =  12.4077 ±  0.1525 | RMSE =  18.04 | R2 = 0.8610
    UCI (MW):           MAE =   7.7371 ±  0.3037 | RMSE =  10.96 | R2 = 0.9831

SCIENTIFIC JUSTIFICATION:
    1. Highest overall rank across tri-benchmark suite (Average Rank 1.33).
    2. Strict adherence to Two-Stage Validation Firewall (no candidate qualified).
    3. Statistically superior or equal to all proposed complex mechanisms.
    4. Optimal balance of dynamic adaptation and centroid regularization.
    5. Real-time deployment latency: 0.120 milliseconds per 24-hour window.

STATUS:
    MODEL FROZEN. NO FURTHER ARCHITECTURAL MODIFICATIONS PERMITTED.
================================================================================
```

---

## 5. Executive Summary Conforming to Section 34

### 1. Final Model Lock Decision
- **Certified Selection:** **`F2 REMAINS FINAL MODEL`** (Outcome D).
- The production model **`F2_A2_OOF`** (`ConfidenceFallbackCAEGNet`, 121,724 parameters) is locked as the definitive research model of CAEG-Net. No more complex candidate justified replacing it.

### 2. Candidate Rejection Matrix
- **Control B (Fixed Shrinkage $\lambda \equiv 0.51$):** REJECTED. Failed validation firewall (0/3 improved). Test performance is at statistical parity with F2 on GEFCom and UCI, but worse on PJM (253.50 vs 250.97 MW). Removing dynamic confidence provides no compelling empirical advantage.
- **Control C (Horizon Routing Only $\lambda \equiv 1.0$):** DECISIVELY REJECTED. Failed validation firewall (0/3 improved, $+10.03\%$ degradation on PJM). Statistically significantly worse than F2 on GEFCom ($p = 7.73 \times 10^{-12}, d_z = +0.329$). Horizon-partitioned routing without shrinkage induces severe overfitting.
- **Control D (Dynamic Confidence with Disagreement):** REJECTED. Failed validation firewall (improved 1/3). Statistically significantly worse on GEFCom daily blocks ($p = 0.0008$).
- **Candidate E1 (HGR + Fixed Shrinkage):** REJECTED. Failed validation firewall (0/3 improved, $+13.27\%$ degradation on PJM). Test MAE worse on all 3 benchmarks.
- **Candidate E2 (HGR + Group Disagreement Confidence):** DECISIVELY REJECTED. Failed validation firewall (0/3 improved, $+11.07\%$ degradation on PJM). Statistically significantly worse on GEFCom ($p = 7.16 \times 10^{-9}, d_z = +0.276$).
- **Candidate E3 (HGR + Global Dynamic Confidence):** REJECTED at Stage 15B-S screening (0/3 improved, $+11.64\%$ degradation on PJM).
- **Feature History Lookbacks (P24, P48, P72):** REJECTED at Stage 15B-S screening. None improved $\ge 2$ benchmarks over canonical 168h OOF performance history.

### 3. Primary Evidence Summary
- **PJM (MW):** F2 achieves **250.97 ± 10.69 MW** (RMSE: 335.38, $R^2$: 0.8714), matching historical benchmark to 4 decimal places.
- **GEFCom (kW):** F2 achieves **12.41 ± 0.15 kW** (RMSE: 18.04, $R^2$: 0.8610).
- **UCI (MW):** F2 achieves **7.74 ± 0.30 MW** (RMSE: 10.96, $R^2$: 0.9831).

### 4. Key Mechanism Findings
1. **Centroid Regularization is Mandatory:** Without centroid shrinkage ($\lambda = 1.0$, Control C), gating performance collapses across all datasets. Shrinking adaptive weights toward uniform ($1/3, 1/3, 1/3$) with $\lambda \approx 0.51$ acts as a Bayesian prior that stabilizes generalization.
2. **Retrospective Horizon Crossover Fails Causally:** While oracle analysis demonstrates retrospective expert specialization across lead times, learning separate causal routing heads per horizon group triples router degrees of freedom, decouples multi-horizon trajectory covariance, and degrades generalization.
3. **Disagreement Does Not Rescue Routing:** Adding inter-expert prediction spread as an input to the router or confidence head increases variance without improving accuracy.

### 5. Transition to Paper Writing
- Phase 15B is formally closed.
- The experimental and model-development cycle of CAEG-Net is complete.
- The repository is ready for Phase 16 (final paper revision and submission preparation) with locked evidence, reproducible figures, and certified benchmarks.

---
**Model Selection Certification is hereby COMPLETE and LOCKED.**
