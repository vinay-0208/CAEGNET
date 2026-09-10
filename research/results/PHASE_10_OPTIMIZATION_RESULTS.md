# CAEG-Net Phase 10: Controlled Cross-Dataset Optimization of Original CAEG-Net V1

**Repository:** CAEG-Net (Context-Adaptive Expert Gating Network)  
**Branch:** `research-track`  
**Date:** September 2026 (Audited & Corrected)  
**Status:** Completed, Audited & Frozen  

---

## Executive Summary

Phase 10 addresses the cross-dataset generalization question for CAEG-Net:
> *"Can context-aware adaptive expert gating improve short-term electricity load forecasting by dynamically combining complementary LSTM, TCN, and CNN temporal experts?"*

Prior phases established that while Canonical CAEG-Net V1 demonstrated strong adaptive gains on aggregate consumer data (UCI Cohort 320, Phase 9) and transmission grid data (PJM, Phase 6), its gains over a static equal ensemble were marginal on zonal sub-station data (GEFCom2014, Phase 8). Phase 10 conducted a strictly controlled optimization study across all three tri-benchmark datasets to determine whether targeted modifications—within the frozen canonical expert family (LSTM, TCN, CNN)—could enhance routing efficacy and cross-dataset robustness without test-set tuning.

### Key Audited Conclusions
1. **Validation Screening Qualification:** Out of 14 candidate formulations across 5 hypothesis groups (A–E), only two candidates satisfied the predefined cross-dataset validation criterion ($\ge 2$ datasets improved, no dataset degraded by $>2.0\%$ relative to Canonical V1):
   - **B1 (Zero-Recent-Error 3D Context):** PJM $+0.04\%$, GEFCom $+6.42\%$, UCI $+4.46\%$.
   - **C3 (Softer Temperature $\tau=1.25$):** PJM $-0.56\%$, GEFCom $+2.16\%$, UCI $+2.19\%$.
2. **Evaluation on Held-Out Test Partitions:** When evaluated across 5 canonical seeds (`[42, 123, 999, 2024, 3407]`) on held-out test partitions under the locked Phase 10 protocol:
   - **PJM Test MAE:** Canonical V1 achieved **$251.17 \pm 12.61$ MW** vs B1 = $255.73 \pm 5.96$ MW vs C3 = $254.28 \pm 4.18$ MW.
   - **GEFCom Test MAE:** Canonical V1 achieved **$12.81 \pm 0.34$ kW** vs B1 = **$12.77 \pm 0.48$ kW** vs C3 = $12.86 \pm 0.28$ kW.
   - **UCI Test MAE:** Canonical V1 achieved **$8.08 \pm 0.39$ MW** vs B1 = $8.18 \pm 0.20$ MW vs C3 = $8.34 \pm 0.62$ MW.
   *(Note on test partitions: These partitions were held out during the Phase 10 candidate training and selection procedures; they are not pristine first-ever test data because earlier research phases evaluated them.)*
3. **Dataset-Dependent Routing Advantage:** The empirical advantage of adaptive gating is domain-dependent rather than universal. On Modern PJM, CAEG-Net V1 achieved a 9.4% lower aggregate test MAE than the Static Equal Ensemble ($251.17$ vs $277.13$ MW), though daily-block paired comparisons did not reach statistical significance ($p=0.1413$). On GEFCom and UCI, daily-block paired analyses favored the Static Equal Ensemble ($p < 0.01$).
4. **Definitive Architectural Decision:** Canonical CAEG-Net V1 demonstrated the strongest overall empirical generalization across the three benchmark datasets among all tested candidates. Neither B1 nor C3 produced a consistent cross-dataset improvement on the held-out test partitions. Consequently, **Canonical CAEG-Net V1 is retained as the frozen research architecture.**

---

## 1. Tri-Benchmark Dataset Profiles

Phase 10 evaluated models across three distinct operational regimes of the electric power grid:

| Dimension | Modern PJM (Phase 6) | GEFCom2014 (Phase 8) | UCI Cohort 320 (Phase 9) |
| :--- | :--- | :--- | :--- |
| **Grid Level** | Bulk Transmission Grid | Zonal Sub-station Grid | Aggregate Consumer Cohort |
| **Primary Unit** | Megawatts (MW) | Kilowatts (kW) | Megawatts (MW) |
| **Load Mean $\pm$ Std** | $3,195.4 \pm 628.9$ MW | $130.5 \pm 27.2$ kW | $44.8 \pm 11.8$ MW |
| **Temporal Span** | 2023–2024 (1 year) | 2005–2011 (7 years) | 2012–2014 (3 years) |
| **Total Windows (Tr/Va/Te)** | 5,957 / 1,294 / 1,294 | 41,425 / 8,736 / 10,944 | 18,221 / 3,922 / 3,922 |
| **Context Dimension** | 4-D (Trend, Vol, Err, DoW) | 4-D (Trend, Vol, Err, DoW) | 4-D (Trend, Vol, Err, DoW) |

---

## 2. Cross-Dataset Validation Screening Results

All 14 candidate formulations were screened on the validation partition. Results are reported relative to the Canonical CAEG-Net V1 control baseline ($A0$):

$$\text{Relative Improvement (\%)} = \frac{\text{MAE}_{A0} - \text{MAE}_{\text{Cand}}}{\text{MAE}_{A0}} \times 100\%$$

Positive values indicate an improvement (lower error); negative values indicate degradation.

| Candidate ID | Description | PJM Rel $\Delta$ | GEFCom Rel $\Delta$ | UCI Rel $\Delta$ | Improved Datasets | Max Degradation | Promising? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0_Canonical_V1** | Control Baseline (Canonical V1) | $0.00\%$ | $0.00\%$ | $0.00\%$ | 0 | $0.00\%$ | Control |
| **A1_Load_Normalized_Trend** | Trend normalized by mean load | $-4.82\%$ | $-1.17\%$ | $+6.63\%$ | 1 | $-4.82\%$ | NO |
| **A2_CoV_Volatility** | Volatility as CoV ($\sigma / \mu$) | $-3.10\%$ | $-0.41\%$ | $+2.00\%$ | 1 | $-3.10\%$ | NO |
| **A3_Relative_Recent_Error** | Relative MAE error feature | $-3.17\%$ | $-0.82\%$ | $+1.03\%$ | 1 | $-3.17\%$ | NO |
| **A4_Dimensionless_Context** | All context features dimensionless | $-1.47\%$ | $-0.71\%$ | $-1.11\%$ | 0 | $-1.47\%$ | NO |
| **B1_Zero_Recent_Error_3D** | Zeroed recent error (3-D context) | $\mathbf{+0.04\%}$ | $\mathbf{+6.42\%}$ | $\mathbf{+4.46\%}$ | **3** | $\mathbf{0.00\%}$ | **YES** |
| **B2_Normalized_Recent_Error**| Normalized recent forecast error | $-3.26\%$ | $-0.52\%$ | $+6.43\%$ | 1 | $-3.26\%$ | NO |
| **C1_Entropy_Regularized** | Entropy bonus ($\beta=0.01$) | $-2.63\%$ | $-1.02\%$ | $+3.08\%$ | 1 | $-2.63\%$ | NO |
| **C2_Stability_Regularized** | Gating KL stability regularizer | $-2.03\%$ | $-0.67\%$ | $+2.28\%$ | 1 | $-2.03\%$ | NO |
| **C3_Sharper_Temperature** | Softmax temperature $\tau=0.8$ | $-8.10\%$ | $+0.18\%$ | $+1.71\%$ | 2 | $-8.10\%$ | NO |
| **C3_Softer_Temperature** | Softmax temperature $\tau=1.25$ | $\mathbf{-0.56\%}$ | $\mathbf{+2.16\%}$ | $\mathbf{+2.19\%}$ | **2** | $\mathbf{-0.56\%}$ | **YES** |
| **D1_Decoupled_Pretraining** | Experts frozen, gate trained | $-0.67\%$ | $+4.66\%$ | $-11.49\%$ | 1 | $-11.49\%$ | NO |
| **D2_Pretrain_FineTune** | Pre-trained experts fine-tuned | $-3.76\%$ | $+4.28\%$ | $+3.17\%$ | 2 | $-3.76\%$ | NO |
| **E1_CNN_H2_TemporalPool8** | CNN 2-head, pool=8 | $-9.23\%$ | $+1.11\%$ | $+3.74\%$ | 2 | $-9.23\%$ | NO |

### Screening Observations
1. **Context Transformations (Group A):** Within the tested formulations, dimensionless normalization (A1, A2, A3) consistently impaired PJM validation performance ($-3.1\%$ to $-4.8\%$). Retaining raw-scale context features provided better validation performance, particularly on PJM.
2. **Error Feedback (Group B):** Eliminating recent forecast error (B1) improved validation metrics across all three datasets during single-seed validation screening.
3. **Temperature Dynamics (Group C):** Sharper gating ($\tau=0.8$) degraded PJM validation performance ($-8.10\%$). Softer temperature ($\tau=1.25$) provided modest validation gains on GEFCom and UCI with minimal degradation on PJM ($-0.56\%$).
4. **Decoupled Training & Architecture (Groups D & E):** Decoupled expert training (D1) degraded UCI validation performance by $-11.49\%$. Altering the CNN expert (E1) degraded PJM by $-9.23\%$.

---

## 3. Five-Seed Finalist Evaluation (Held-Out Test Partitions)

The two qualified candidates (B1, C3) and the control baseline (A0 Canonical V1) were evaluated across 5 canonical seeds (`[42, 123, 999, 2024, 3407]`) on the held-out test partitions.

### 3.1 Primary Metric: Test MAE (Mean $\pm$ Std)

| Candidate ID | Modern PJM (MW) | GEFCom2014 (kW) | UCI Cohort 320 (MW) |
| :--- | :---: | :---: | :---: |
| **A0_Canonical_V1** | $\mathbf{251.17 \pm 12.61}$ | $12.81 \pm 0.34$ | $\mathbf{8.08 \pm 0.39}$ |
| **B1_Zero_Recent_Error_3D** | $255.73 \pm 5.96$ | $\mathbf{12.77 \pm 0.48}$ | $8.18 \pm 0.20$ |
| **C3_Softer_Temperature** | $254.28 \pm 4.18$ | $12.86 \pm 0.28$ | $8.34 \pm 0.62$ |

*(Note: The $\pm$ values are standard deviations across the five seed results.)*

### 3.2 Secondary Metrics Summary

#### Modern PJM Benchmark (Transmission Grid)
| Finalist | MAE (MW) | RMSE (MW) | MSE ($\text{MW}^2$) | $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0_Canonical_V1** | $\mathbf{251.17 \pm 12.61}$ | $\mathbf{334.51 \pm 14.32}$ | $\mathbf{112,103 \pm 9,626}$ | $\mathbf{0.8393 \pm 0.0182}$ | $\mathbf{4.63 \pm 0.21}$ |
| **B1_Zero_Recent_Error_3D** | $255.73 \pm 5.96$ | $342.42 \pm 9.70$ | $117,343 \pm 6,717$ | $0.8220 \pm 0.0172$ | $4.72 \pm 0.14$ |
| **C3_Softer_Temperature** | $254.28 \pm 4.18$ | $339.09 \pm 7.90$ | $115,047 \pm 5,355$ | $0.8328 \pm 0.0169$ | $4.69 \pm 0.08$ |

#### GEFCom2014 Benchmark (Zonal Sub-station Grid)
| Finalist | MAE (kW) | RMSE (kW) | MSE ($\text{kW}^2$) | $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0_Canonical_V1** | $12.81 \pm 0.34$ | $\mathbf{18.43 \pm 0.41}$ | $\mathbf{339.9 \pm 15.3}$ | $0.8217 \pm 0.0170$ | $\mathbf{8.73 \pm 0.27}$ |
| **B1_Zero_Recent_Error_3D** | $\mathbf{12.77 \pm 0.48}$ | $18.47 \pm 0.46$ | $341.3 \pm 17.3$ | $\mathbf{0.8275 \pm 0.0107}$ | $8.77 \pm 0.30$ |
| **C3_Softer_Temperature** | $12.86 \pm 0.28$ | $18.51 \pm 0.34$ | $342.7 \pm 12.8$ | $0.8184 \pm 0.0116$ | $8.76 \pm 0.22$ |

#### UCI Cohort 320 Aggregate Benchmark (Consumer Cohort)
| Finalist | MAE (MW) | RMSE (MW) | MSE ($\text{MW}^2$) | $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0_Canonical_V1** | $\mathbf{8.08 \pm 0.39}$ | $\mathbf{11.38 \pm 0.49}$ | $\mathbf{129.7 \pm 11.2}$ | $\mathbf{0.9813 \pm 0.0016}$ | $4.16 \pm 0.30$ |
| **B1_Zero_Recent_Error_3D** | $8.18 \pm 0.20$ | $11.66 \pm 0.37$ | $136.0 \pm 8.7$ | $0.9802 \pm 0.0011$ | $\mathbf{4.13 \pm 0.11}$ |
| **C3_Softer_Temperature** | $8.34 \pm 0.62$ | $11.86 \pm 0.88$ | $141.4 \pm 22.0$ | $0.9795 \pm 0.0029$ | $4.21 \pm 0.28$ |

---

## 4. Expert Complementarity & Standalone Baselines

Standalone temporal experts (LSTM, TCN, CNN) were trained without gating and evaluated alongside the Static Equal Ensemble and CAEG-Net V1.

### 4.1 Standalone Models vs Ensembles (Test MAE)

| Dataset | LSTM | TCN | CNN | Static Equal Ensemble | CAEG-Net V1 (Seed 42) | CAEG-Net V1 (5-Seed Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PJM (MW)** | $302.63$ | $247.98$ | $422.07$ | $277.13$ | $\mathbf{245.98}$ | $\mathbf{251.17}$ |
| **GEFCom (kW)** | $13.13$ | $13.02$ | $14.01$ | $\mathbf{12.66}$ | $12.87$ | $12.81$ |
| **UCI (MW)** | $\mathbf{7.96}$ | $8.45$ | $11.90$ | $8.19$ | $8.57$ | $\mathbf{8.08}$ |

### 4.2 Expert Residual Error Correlations (Pearson $r$)

| Dataset | $\text{Corr}(e_{\text{LSTM}}, e_{\text{TCN}})$ | $\text{Corr}(e_{\text{LSTM}}, e_{\text{CNN}})$ | $\text{Corr}(e_{\text{TCN}}, e_{\text{CNN}})$ | Complementarity Context |
| :--- | :---: | :---: | :---: | :--- |
| **PJM** | $0.791$ | $0.596$ | $0.636$ | Higher diversity (CNN error less correlated) |
| **GEFCom** | $0.908$ | $0.848$ | $0.866$ | Strongly correlated expert residuals |
| **UCI** | $0.843$ | $0.627$ | $0.701$ | Moderate correlation (LSTM expert strongest) |

### Scientific Discussion:
1. **Modern PJM:** CNN errors exhibit lower correlation with LSTM ($r=0.596$) and TCN ($r=0.636$). Although standalone CNN error is higher ($422.07$ MW), the gating network allocates weight adaptively, yielding an aggregate test MAE ($251.17$ MW 5-seed mean; $245.98$ MW on seed 42) that is lower than the Static Equal Ensemble ($277.13$ MW).
2. **GEFCom2014:** All three temporal experts produce strongly correlated error vectors ($r \in [0.848, 0.908]$). These high residual correlations are consistent with reduced opportunity for adaptive routing to exploit complementary expert errors, in which setting uniform averaging achieves strong performance ($12.66$ kW vs $12.81$ kW for CAEG V1).
3. **UCI Cohort 320:** Standalone LSTM achieved the lowest test MAE ($7.96$ MW) compared to TCN ($8.45$ MW) and CNN ($11.90$ MW). The UCI results are consistent with a setting in which the LSTM expert is particularly effective and adaptive routing assigns greater average weight to it ($46.9\%$).

---

## 5. Gating Diagnostics & Routing Entropy

Across all 5 seeds on the held-out test partitions, gating weights and routing entropy were tracked:

| Finalist | Dataset | Mean $w_{\text{LSTM}}$ | Mean $w_{\text{TCN}}$ | Mean $w_{\text{CNN}}$ | Mean Entropy | Effective Experts ($N_{\text{eff}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0_Canonical_V1** | **PJM** | $0.389$ | $0.289$ | $0.322$ | $1.089$ | $\mathbf{2.97}$ |
| **A0_Canonical_V1** | **GEFCom** | $0.409$ | $0.260$ | $0.330$ | $1.077$ | $\mathbf{2.94}$ |
| **A0_Canonical_V1** | **UCI** | $0.469$ | $0.189$ | $0.342$ | $1.032$ | $\mathbf{2.81}$ |
| **B1_Zero_Recent_Error** | **PJM** | $0.387$ | $0.288$ | $0.325$ | $1.088$ | $2.97$ |
| **B1_Zero_Recent_Error** | **GEFCom** | $0.404$ | $0.262$ | $0.334$ | $1.080$ | $2.95$ |
| **B1_Zero_Recent_Error** | **UCI** | $0.443$ | $0.212$ | $0.345$ | $1.054$ | $2.87$ |
| **C3_Softer_Temperature** | **PJM** | $0.386$ | $0.290$ | $0.324$ | $1.090$ | $2.97$ |
| **C3_Softer_Temperature** | **GEFCom** | $0.404$ | $0.270$ | $0.325$ | $1.082$ | $2.95$ |
| **C3_Softer_Temperature** | **UCI** | $0.464$ | $0.194$ | $0.342$ | $1.038$ | $2.82$ |

### Gating Observations:
- **Expert Utilization:** Routing did not collapse to a single expert. Across all seeds and datasets, the effective number of experts ($N_{\text{eff}} = e^H$) averaged between $2.81$ and $2.97$ (theoretical maximum = $3.0$).
- **Weight Allocation:** On UCI, the router placed higher average weight on the LSTM expert ($46.9\%$), reflecting its standalone empirical accuracy, while maintaining balanced usage on PJM and GEFCom.

---

## 6. Daily-Block Paired Statistical Analysis (CAEG V1 vs Static Equal Ensemble)

### Distinction Between Evaluation Protocols
- **Sliding-Window Evaluation:** Full test set evaluated hourly across all overlapping 24-hour windows.
- **Daily-Block Paired Analysis:** Non-overlapping 24-hour forecast blocks evaluated sequentially to provide independent samples for paired hypothesis testing.

### Paired Hypothesis Testing Results

| Dataset | $K$ Blocks | Mean Daily Diff ($\text{MAE}_{\text{CAEG}} - \text{MAE}_{\text{Ens}}$) | Std Diff | $t$-statistic | $p$-value ($t$-test) | $p$-value (Wilcoxon) | Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PJM** | 53 | $-11.15$ MW | $53.85$ MW | $-1.494$ | $0.1413$ | $0.5861$ | No statistically significant difference ($p > 0.05$) |
| **GEFCom** | 456 | $+0.48$ kW | $1.79$ kW | $+5.730$ | $1.82 \times 10^{-8}$ | $9.66 \times 10^{-10}$ | Statistically favors Equal Ensemble ($p < 0.001$) |
| **UCI** | 163 | $+0.38$ MW | $1.47$ MW | $+3.300$ | $0.0012$ | $0.0036$ | Statistically favors Equal Ensemble ($p < 0.01$) |

### Statistical Synthesis:
1. **Modern PJM:** CAEG-Net V1 achieved a numerically lower aggregate test MAE than the Static Equal Ensemble ($251.17$ vs $277.13$ MW; $245.98$ vs $277.13$ MW on seed 42); however, the daily-block paired comparison did not reach statistical significance (paired $t$-test $p=0.1413$; Wilcoxon $p=0.5861$).
2. **GEFCom2014:** The daily-block paired analysis statistically favored the Static Equal Ensemble ($p < 10^{-7}$). The mean difference of $+0.48$ kW on a $130$ kW average load is modest ($0.37\%$), reflecting high expert error collinearity.
3. **UCI Cohort 320:** Aggregate 5-seed test MAE for CAEG-Net V1 was slightly lower than the Static Equal Ensemble ($8.08$ vs $8.19$ MW), while the daily-block paired analysis favored the equal ensemble ($p=0.0012$; Wilcoxon $p=0.0036$). These two analyses represent different aggregation views (hourly sliding-window average vs non-overlapping daily block differences).

---

## 7. Answers to the 11 Core Scientific Questions

### 1. Core Finding: Can CAEG-Net be improved across datasets without redesigning the architecture?
**No.** Controlled modifications within the canonical expert family—including context feature scaling, error-feature ablations, gating regularizers, temperature scaling, and decoupled pre-training—did not produce a consistent cross-dataset improvement over Canonical CAEG-Net V1 on held-out test data.

### 2. Cross-Dataset Performance: Did any candidate improve performance on $\ge 2$ datasets without degrading the 3rd?
In the validation screening phase, **B1** and **C3** satisfied the qualification criterion. However, in the subsequent 5-seed evaluation on held-out test partitions, **Canonical V1 remained the strongest overall cross-dataset formulation among the tested candidates** (lowest MAE on PJM at $251.17$ MW and UCI at $8.08$ MW, with competitive performance on GEFCom at $12.81$ kW vs $12.77$ kW for B1).

### 3. Dataset Sensitivity: Why did UCI benefit while PJM and GEFCom showed different patterns?
The UCI results are consistent with a setting in which the LSTM expert is particularly effective ($7.96$ MW standalone MAE) and adaptive routing assigns greater average weight to it ($46.9\%$). Modern PJM operates at bulk transmission scale with higher inertia, where CAEG-Net V1 showed numerical aggregate improvement over the static equal ensemble ($251.17$ vs $277.13$ MW). GEFCom consists of zonal load where expert residuals are strongly correlated, reducing opportunities for adaptive routing.

### 4. Expert Complementarity: What are the pairwise residual correlations between LSTM, TCN, and CNN?
- **PJM:** $r(\text{LSTM}, \text{TCN})=0.791$, $r(\text{LSTM}, \text{CNN})=0.596$, $r(\text{TCN}, \text{CNN})=0.636$.
- **GEFCom:** $r(\text{LSTM}, \text{TCN})=0.908$, $r(\text{LSTM}, \text{CNN})=0.848$, $r(\text{TCN}, \text{CNN})=0.866$.
- **UCI:** $r(\text{LSTM}, \text{TCN})=0.843$, $r(\text{LSTM}, \text{CNN})=0.627$, $r(\text{TCN}, \text{CNN})=0.701$.  
The observed results are consistent with the hypothesis that greater expert error diversity provides more opportunity for adaptive routing to improve over uniform averaging.

### 5. Gating Behavior: How did gating weights distribute across experts on each dataset?
Routing remained diversified across all datasets:
- PJM: LSTM $38.9\%$, TCN $28.9\%$, CNN $32.2\%$ ($N_{\text{eff}} = 2.97$).
- GEFCom: LSTM $40.9\%$, TCN $26.0\%$, CNN $33.0\%$ ($N_{\text{eff}} = 2.94$).
- UCI: LSTM $46.9\%$, TCN $18.9\%$, CNN $34.2\%$ ($N_{\text{eff}} = 2.81$).  
No collapse to a single expert was observed.

### 6. Equal Ensemble Comparison: How does adaptive gating compare against static equal weighting?
- On Modern PJM, CAEG-Net V1 achieved a $9.4\%$ lower aggregate test MAE than the Static Equal Ensemble ($251.17$ vs $277.13$ MW); however, the daily-block paired comparison did not reach statistical significance ($p=0.1413$).
- On GEFCom, the Static Equal Ensemble achieved slightly lower error ($12.66$ vs $12.81$ kW), with the daily-block paired test favoring the equal ensemble ($p < 0.001$).
- On UCI, aggregate test MAE was close between CAEG-Net V1 and the equal ensemble ($8.08$ vs $8.19$ MW), while the daily-block paired test favored the equal ensemble ($p=0.0012$).

### 7. Parameter Efficiency: How does CAEG-Net scale relative to single experts and Ridge?
- **Ridge:** 4,056 parameters. Parameter-efficient in linear transmission regimes, but limited in non-linear consumer dynamics.
- **Single Experts:** LSTM (56,152), TCN (36,952), CNN (27,400).
- **CAEG-Net V1:** 121,531 parameters. The routing mechanism adds 1,027 parameters ($0.8\%$ of total capacity), making adaptive gating lightweight relative to the expert ensemble.

### 8. Context Representation: Did normalized trend, volatility, or dimensionless context improve routing?
**No.** Within the tested formulations, retaining raw-scale context features provided better validation performance, particularly on PJM. Normalizing trend (A1), volatility as CoV (A2), and dimensionless context (A4) degraded PJM validation MAE by $1.5\%$ to $4.8\%$.

### 9. Error Feedback: Did zero-recent-error (3D) or normalized recent error help or hurt?
The results suggest that recent forecast-error feedback provides useful information to the routing mechanism. Removing recent error (B1) improved validation screening but resulted in higher test MAE on PJM ($255.73$ vs $251.17$ MW) and UCI ($8.18$ vs $8.08$ MW).

### 10. Gating Regularizers & Temperature: Did regularizers or temperature scaling improve generalization?
- **Regularizers:** Entropy bonuses (C1) and gating KL stability (C2) degraded PJM validation performance.
- **Temperature:** Sharper temperature ($\tau=0.8$) degraded PJM validation performance by $-8.10\%$. Softer temperature ($\tau=1.25$, C3) passed screening but did not improve test MAE on PJM ($254.28$ vs $251.17$ MW) or UCI ($8.34$ vs $8.08$ MW).

### 11. Definitive Recommendation: What is the frozen architecture going forward?
**Retain Canonical CAEG-Net V1 without modification.**  
Canonical CAEG-Net V1 demonstrated the strongest overall empirical generalization across the three benchmark datasets among the tested formulations.

---

## 8. Final Scientific Conclusion

Among the Phase 10 formulations evaluated, Canonical CAEG-Net V1 provided the strongest overall cross-dataset performance, achieving the lowest mean test MAE on two of the three datasets. Neither validation-qualified modification produced a consistent improvement over V1 across datasets.

Phase 10 did not identify a modification that robustly improves Canonical CAEG-Net V1 across all three benchmark regimes. Although B1 and C3 satisfied the predefined validation screening criterion, neither produced consistent gains in the subsequent 5-seed evaluation on held-out test partitions. Canonical CAEG-Net V1 therefore remains the preferred frozen formulation among the tested models. The cross-dataset results suggest that the benefit of adaptive fusion depends on the degree of exploitable diversity among expert forecasts, while simpler fusion can remain highly competitive in regimes with strongly correlated expert errors.

---

## 9. Artifact & Verification Index

| Artifact Type | File Path | Description |
| :--- | :--- | :--- |
| **Validation Screening** | `research/results/phase10_validation_results.csv` | Full screening metrics for 14 candidates across 3 datasets |
| **Cross-Dataset Summary** | `research/results/phase10_cross_dataset_validation.csv` | Relative improvement table and qualification flags |
| **5-Seed Test Finalists** | `research/results/phase10_candidate_comparison.csv` | Multi-seed test MAE, RMSE, MSE, R2, MAPE |
| **Routing Diagnostics** | `research/results/phase10_routing_analysis.csv` | Expert weights, entropy, and effective expert count |
| **Baselines & Ensembles**| `research/results/phase10_baselines.csv` | Standalone LSTM, TCN, CNN, and Static Equal Ensemble |
| **Complementarity Matrix**| `research/results/phase10_expert_complementarity.csv`| Pairwise residual error correlations across datasets |
| **Statistical Tests** | `research/results/phase10_statistical_tests.csv` | Daily-block paired t-tests and Wilcoxon tests |
| **Scientific Audit Log** | `research/results/PHASE_10_SCIENTIFIC_AUDIT.md` | Audit log of statistical reconciliation & corrections |
| **Fig 1: Cross Improvement**| `research/results/phase10_plots/phase10_01_cross_dataset_improvement.png` | Group A–E relative validation improvements |
| **Fig 2: Normalized MAE** | `research/results/phase10_plots/phase10_02_val_mae_comparison.png` | Candidate validation MAE normalized to Canonical V1 |
| **Fig 3: Standalone MAE** | `research/results/phase10_plots/phase10_03_standalone_expert_mae.png` | Relative standalone expert performance |
| **Fig 4: CAEG vs Ensemble**| `research/results/phase10_plots/phase10_04_equal_vs_caeg.png` | Error ratio of CAEG-Net V1 to Static Equal Ensemble |
| **Fig 5: Routing Weights** | `research/results/phase10_plots/phase10_05_routing_weights.png` | Average gating weights per expert across datasets |
| **Fig 6: Routing Entropy** | `research/results/phase10_plots/phase10_06_entropy_effective_experts.png` | Effective number of active experts ($N_{\text{eff}}$) |
| **Fig 7: Residual Heatmap** | `research/results/phase10_plots/phase10_03_residual_correlation.png` | Heatmap of pairwise expert residual correlations |
| **Fig 8: Forecast Profile**| `research/results/phase10_plots/phase10_08_sample_forecasts.png` | Representative 24-hour forecast profile |

---
**End of Audited Phase 10 Report**
