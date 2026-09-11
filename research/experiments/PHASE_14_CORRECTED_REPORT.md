# PHASE 14 CORRECTED FINAL RESEARCH REPORT
## Corrected Final-Model Optimization, Scientific Reconciliation & Cross-Dataset Evaluation
**Project:** CAEG-Net: Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting  
**Historical Experiment Commit:** `c67067d8a70d247832ee1cea5be466c527ec3838`  
**Scientific Correction Commit:** `1d4d8c354e720384f01f72ce8a3e071d7fa253a1`  
**Evaluation Environment:** Python 3.10.13, PyTorch `2.13.0+cu130`, CUDA 13.0, NVIDIA GeForce RTX 4050 Laptop GPU  
**Runtime:** 14,619.45s (~4.06 hours)  
**Status:** Scientifically Reconciled, Publication-Ready & Defense-Locked  

---

## 1. Executive Summary

Phase 14 investigated whether the canonical CAEG-Net V1 forecasting architecture could be improved in a statistically defensible manner through causally valid, out-of-fold (OOF) relative performance feedback and confidence-based fallback toward equal ensembling.

This report presents the final, scientifically reconciled synthesis of the completed Phase 14 experimental record:
1. **Strongest Cross-Dataset Balance:** Candidate **F2_A2_OOF** (7D Context with genuine OOF relative errors + learned confidence fallback head, 121,724 parameters) demonstrated the strongest overall cross-dataset balance among the six evaluated formulations.
2. **Consistent Improvement over Canonical V1:** F2 improved upon Canonical V1 on all three benchmark datasets in the locked five-seed evaluation:
   - **Modern PJM:** $250.97 \pm 10.69\text{ MW}$ vs. $253.41 \pm 9.12\text{ MW}$ ($-2.44\text{ MW}$, $-0.96\%$)
   - **GEFCom2014:** $12.41 \pm 0.15\text{ kW}$ vs. $12.88 \pm 0.25\text{ kW}$ ($-0.47\text{ kW}$, $-3.65\%$)
   - **UCI Cohort 320 Aggregate:** $7.74 \pm 0.30\text{ MW}$ vs. $7.94 \pm 0.17\text{ MW}$ ($-0.20\text{ MW}$, $-2.52\%$)
3. **3/3 Datasets — Lower MAE than the Static Equal Ensemble:** F2 achieved lower MAE than the static equal ensemble on all three benchmark datasets:
   - Modern PJM ($250.97\text{ MW}$ vs. $279.83\text{ MW}$)
   - GEFCom2014 ($12.41\text{ kW}$ vs. $12.62\text{ kW}$)
   - UCI Cohort 320 ($7.74\text{ MW}$ vs. $8.17\text{ MW}$)
4. **Comparison with Verified Best Standalone Experts (2/3 Datasets):** Subject to the verified locked standalone references, F2 outperformed the best standalone expert on two of the three datasets:
   - Modern PJM: **YES** (F2 $= 250.97\text{ MW}$ vs. standalone TCN $= 259.33\text{ MW}$)
   - GEFCom2014: **YES** (F2 $= 12.41\text{ kW}$ vs. standalone TCN $= 12.57\text{ kW}$)
   - UCI Cohort 320: **NO** (F2 $= 7.74\text{ MW}$ does not beat the locked standalone LSTM reference $= 7.55\text{ MW}$; it is comparable to the Phase 11 standalone LSTM test benchmark $= 7.79\text{ MW}$)
5. **Daily-Block Statistical Significance vs. V1:** Under predefined non-overlapping 24-hour daily block paired analysis with Step-Down Holm-Bonferroni correction on the five-seed ensemble forecasts:
   - Modern PJM ($K=53$ blocks): Mean daily diff $= -9.66\text{ MW}$ ($95\%\text{ CI}: [-17.07, -2.26]$, $p_{\text{adj}} = 0.0406$, Cohen's $d_z = -0.351$)
   - GEFCom2014 ($K=456$ blocks): Mean daily diff $= -0.688\text{ kW}$ ($95\%\text{ CI}: [-0.802, -0.575]$, $p_{\text{adj}} = 1.02 \times 10^{-27}$, Cohen's $d_z = -0.555$)
   - UCI Cohort 320 ($K=163$ blocks): Mean daily diff $= -0.202\text{ MW}$ ($95\%\text{ CI}: [-0.310, -0.094]$, $p_{\text{adj}} = 0.0017$, Cohen's $d_z = -0.287$)
   Among the evaluated formulations, F2 was the only candidate showing statistically significant improvement over Canonical V1 across all three datasets under the predefined daily-block paired analysis with Holm correction.
6. **Shrinkage Dynamics & Resolution:** Rather than exhibiting strong temporal regime switching, the learned confidence mechanism behaved approximately as a near-constant shrinkage coefficient toward equal fusion ($\lambda_t \approx 0.51$, with sample standard deviations $< 0.006$ and $CV < 1.1\%$). However, a globally fixed scalar shrinkage coefficient (Candidate F5, $\lambda^* = 0.6233$) failed to maintain cross-dataset stability, severely degrading on UCI Cohort 320 ($8.14\text{ MW}$). A context-conditioned confidence head provided superior cross-dataset balance compared to a global static scalar.
7. **Final Model Recommendation:** F2_A2_OOF is selected as the final CAEG-Net formulation for the paper based on its strongest overall cross-dataset balance, consistent improvement over canonical V1 across all three benchmark datasets, statistically significant daily-block improvement over V1 under the predefined Holm-corrected analysis, lower MAE than the static equal ensemble on all three datasets, superiority over the verified best standalone expert on two of three datasets, causal OOF performance features, and minimal additional parameter complexity (+193 parameters, +0.1588% overhead).  
   *Qualification:* F2 is not the lowest-MAE formulation on every individual dataset; F5 achieved the lowest PJM MAE and F3 achieved the lowest UCI MAE among the Phase 14 candidates.

---

## 2. Research Questions & Hypotheses

### Primary Research Question
Can context-adaptive expert gating be improved in a statistically defensible and cross-dataset robust manner by incorporating causally valid, out-of-fold historical performance feedback alongside temporal-statistical context, coupled with a confidence fallback mechanism toward equal ensembling?

### Secondary Research Question
How does the confidence head $\lambda_t = \sigma(\text{MLP}_c(\mathbf{c}(t)))$ behave across operational load regimes? Does it perform dynamic, state-dependent arbitration, or does it operate primarily as variance shrinkage? Can a single globally fixed scalar parameter $\lambda^*$ replace the neural confidence head?

---

## 3. Scope Boundaries & Frozen Protocols

To maintain publication-level scientific integrity:
1. **Frozen Temporal Expert Core (120,504 Parameters):**
   - LSTM Expert: 2-layer LSTM ($d_{\text{hidden}} = 64$) + linear head ($64 \to 64 \to 24$). Total: 56,152 parameters.
   - TCN Expert: 6 causal residual dilated stages ($k=3$, dilations $1, 2, 4, 8, 16, 32$, channels $32$) + linear head ($32 \to 32 \to 24$). Total: 36,952 parameters.
   - CNN Expert: 3 Conv1D stages with BatchNorm, ReLU, MaxPool, AdaptiveAvgPool1D + linear head ($64 \to 48 \to 24$). Total: 27,400 parameters.
   - Zero Transformers, zero attention layers, zero modifications to base expert backbones.
2. **Frozen Chronological Partitions (70% Train / 15% Validation / 15% Test):**
   - Modern PJM: Hourly system load ($N_{\text{train}}=5,957$, $N_{\text{val}}=1,294$, $N_{\text{test}}=1,294$).
   - GEFCom2014: Zonal electricity load ($N_{\text{train}}=41,425$, $N_{\text{val}}=8,736$, $N_{\text{test}}=10,944$).
   - UCI Cohort 320 Aggregate: Summed 320-client smart meter load ($N_{\text{train}}=18,221$, $N_{\text{val}}=3,922$, $N_{\text{test}}=3,922$).
3. **Forecasting Horizon:** 168 hours history $\to$ 24 hours direct forecast.
4. **Historical Preservation:** All results from Phase 10, 11, 12, and 13, and commits `c67067d` and `1d4d8c3` remain completely preserved.

---

## 4. Evaluated Formulations & Parameter Complexity

*(Source: `research/results/phase14_parameter_counts.csv`)*

| Candidate ID | Context Dimension | Context Features | Routing Head | Confidence Fallback Mechanism | Total Trainable Params | Delta Params vs. V1 | Pct Increase |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **F0_Canonical_V1** | 4 | Trend, Volatility, Lag-24 Autocorr, Recent Error | MLP ($4 \to 16 \to 3$) + Softmax | None (Pure Adaptive: $\hat{y} = \hat{y}_{\text{ad}}$) | 121,531 | +0 | +0.0000% |
| **F1_A1_OOF** | 7 | 4D Context + OOF Relative Errors $[r_L, r_T, r_C]$ | MLP ($7 \to 16 \to 3$) + Softmax | None (Pure Adaptive: $\hat{y} = \hat{y}_{\text{ad}}$) | 121,579 | +48 | +0.0395% |
| **F2_A2_OOF** | 7 | 4D Context + OOF Relative Errors $[r_L, r_T, r_C]$ | MLP ($7 \to 16 \to 3$) + Softmax | Learned MLP Head ($7 \to 8 \to 1$) + Sigmoid $\lambda_t$ | **121,724** | **+193** | **+0.1588%** |
| **F3_Confidence_Only** | 4 | 4D Context (No Error Features) | MLP ($4 \to 16 \to 3$) + Softmax | Learned MLP Head ($4 \to 8 \to 1$) + Sigmoid $\lambda_t$ | 121,628 | +97 | +0.0798% |
| **F4_Smoothed_OOF** | 7 | 4D Context + Smoothed OOF Errors ($s_t$, $\alpha^*=0.25$) | MLP ($7 \to 16 \to 3$) + Softmax | Learned MLP Head ($7 \to 8 \to 1$) + Sigmoid $\lambda_t$ | 121,724 | +193 | +0.1588% |
| **F5_Scalar_Shrinkage** | 7 | 4D Context + OOF Relative Errors $[r_L, r_T, r_C]$ | MLP ($7 \to 16 \to 3$) + Softmax | Static Scalar Shrinkage ($\lambda^* = 0.6233$) | 121,579 | +48 | +0.0395% |

---

## 5. Causal Out-of-Fold (OOF) Feature Construction

To eliminate training-side target leakage, Phase 14 employed an expanding-window chronological cross-validation protocol on the training set:
- Training folds $K=5$ expanding windows were trained strictly on historical folds $1, \dots, k-1$ to generate out-of-sample predictions on fold $k$.
- Relative error calculation: At forecast origin $t$, expert relative error $r_i(t) = \frac{e_i(t)}{\sum_j e_j(t) + \epsilon}$ was evaluated on actual observed load strictly from $t-24$ to $t$.
- Zero future targets in $[t, t+24]$ were accessible.
- Causal exponential smoothing (F4): $s_i(t) = \alpha s_i(t-1) + (1-\alpha)r_i(t)$, where $\alpha^* = 0.25$ was optimized on validation data.

---

## 6. Stage 14B Validation Screening Results

Screening was conducted across Seeds 42 and 123 on validation partitions. Metric represents the two-seed mean.

*(Source: `research/results/phase14_validation_results.csv`)*

| Candidate ID | Modern PJM (MW) | PJM vs. V1 | GEFCom2014 (kW) | GEFCom vs. V1 | UCI Cohort 320 (MW) | UCI vs. V1 | Datasets Improved | Worst Degradation | Qualification Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0_Canonical_V1** | 438.10 | Baseline | 13.26 | Baseline | 6.75 | Baseline | Baseline | Baseline | Control |
| **F1_A1_OOF** | 443.25 | +1.17% | 13.24 | -0.14% | 6.70 | -0.73% | 2 / 3 | +1.17% | **QUALIFIED** |
| **F2_A2_OOF** | **399.50** | **-8.81%** | **13.04** | **-1.68%** | **6.48** | **-4.03%** | **3 / 3** | **-1.68%** | **QUALIFIED** |
| **F3_Confidence_Only** | 407.98 | -6.87% | 13.00 | -1.94% | 6.46 | -4.29% | 3 / 3 | -1.94% | **QUALIFIED** |
| **F4_Smoothed_OOF** | 401.74 | -8.30% | 13.05 | -1.55% | 6.47 | -4.20% | 3 / 3 | -1.55% | **QUALIFIED** |
| **F5_Scalar_Shrinkage** | 413.65 | -5.58% | 13.23 | -0.20% | 6.68 | -1.03% | 3 / 3 | -0.20% | **QUALIFIED** |

*All five active candidates satisfied the qualification rule (improved validation MAE on $\ge 2/3$ datasets with degradation $\le 2.0\%$) and advanced to Stage 14C.*

---

## 7. Stage 14C Five-Seed Test Benchmark Results

Evaluated across seeds `[42, 123, 999, 2024, 3407]` on the locked test sets.  
Values are reported as **$\text{Mean} \pm \text{Population SD}$** ($S=5$, `ddof=0`).

*(Source: `research/results/phase14_test_results.csv`)*

### Publication-Ready Final Results Table

| Model | Params | PJM | GEFCom | UCI |
|---|---:|---:|---:|---:|
| F0 V1 | 121,531 | 253.41 ± 9.12 | 12.88 ± 0.25 | 7.94 ± 0.17 |
| F1 A1-OOF | 121,579 | 250.63 ± 5.55 | 12.64 ± 0.22 | 7.98 ± 0.20 |
| F2 A2-OOF | 121,724 | 250.97 ± 10.69 | 12.41 ± 0.15 | 7.74 ± 0.30 |
| F3 Confidence-Only | 121,628 | 255.99 ± 5.95 | 12.44 ± 0.21 | 7.71 ± 0.18 |
| F4 Smoothed OOF | 121,724 | 247.73 ± 5.97 | 12.55 ± 0.17 | 8.02 ± 0.50 |
| F5 Scalar Shrinkage | 121,579 | 246.71 ± 6.36 | 12.66 ± 0.23 | 8.14 ± 0.10 |

### Secondary Metrics (Five-Seed Mean $\pm$ Population SD)
- **Modern PJM:**
  - F0: RMSE $= 338.34 \pm 12.53\text{ MW}$, MAPE $= 4.65 \pm 0.17\%$, $R^2 = 0.8328 \pm 0.0089$
  - F2: RMSE $= 335.38 \pm 12.35\text{ MW}$, MAPE $= 4.62 \pm 0.19\%$, $R^2 = 0.8467 \pm 0.0141$
- **GEFCom2014:**
  - F0: RMSE $= 18.48 \pm 0.29\text{ kW}$, MAPE $= 8.76 \pm 0.20\%$, $R^2 = 0.8245 \pm 0.0083$
  - F2: RMSE $= 18.04 \pm 0.15\text{ kW}$, MAPE $= 8.48 \pm 0.11\%$, $R^2 = 0.8367 \pm 0.0048$
- **UCI Cohort 320 Aggregate:**
  - F0: RMSE $= 11.18 \pm 0.13\text{ MW}$, MAPE $= 4.05 \pm 0.18\%$, $R^2 = 0.9816 \pm 0.0004$
  - F2: RMSE $= 10.96 \pm 0.32\text{ MW}$, MAPE $= 3.97 \pm 0.25\%$, $R^2 = 0.9825 \pm 0.0010$

---

## 8. Baseline Comparisons & Reconciliation

*(Sources: `research/results/phase14_test_results.csv`, `research/results/phase12_dataset_summary.csv`, `research/results/phase11_dataset_summary.csv`)*

| Model / Baseline | Parameters | Modern PJM (MW) | GEFCom2014 (kW) | UCI Cohort 320 (MW) | vs. Canonical V1 | vs. Equal Ensemble | vs. Best Standalone (Locked Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Equal Ensemble** | 120,504 | 279.83 | 12.62 | 8.17 | 1 / 3 (GEFCom) | — | 0 / 3 |
| **Best Standalone Expert (Locked Ref)** | 36k–56k | 259.33 (TCN) | 12.57 (TCN) | 7.55 (LSTM Val BL) / 7.79 (Test BM) | 0 / 3 | 3 / 3 | — |
| **F0_Canonical_V1** | 121,531 | $253.41 \pm 9.12$ | $12.88 \pm 0.25$ | $7.94 \pm 0.17$ | Control | 2 / 3 | 1 / 3 (PJM) |
| **F1_A1_OOF** | 121,579 | $250.63 \pm 5.55$ | $12.64 \pm 0.22$ | $7.98 \pm 0.20$ | 2 / 3 | 2 / 3 | 1 / 3 (PJM) |
| **F2_A2_OOF (Selected Model)** | **121,724** | $\mathbf{250.97 \pm 10.69}$ | $\mathbf{12.41 \pm 0.15}$ | $\mathbf{7.74 \pm 0.30}$ | **3 / 3** | **3 / 3** | **2 / 3 (PJM, GEFCom)** |
| **F3_Confidence_Only** | 121,628 | $255.99 \pm 5.95$ | $12.44 \pm 0.21$ | $\mathbf{7.71 \pm 0.18}$ | 2 / 3 | 3 / 3 | 2 / 3 (PJM, GEFCom) |
| **F4_Smoothed_OOF** | 121,724 | $247.73 \pm 5.97$ | $12.55 \pm 0.17$ | $8.02 \pm 0.50$ | 2 / 3 | 2 / 3 | 2 / 3 (PJM, GEFCom) |
| **F5_Scalar_Shrinkage** | 121,579 | $\mathbf{246.71 \pm 6.36}$ | $12.66 \pm 0.23$ | $8.14 \pm 0.10$ | 2 / 3 | 2 / 3 | 1 / 3 (PJM) |

### Reconciled Baseline Findings
1. **F2 vs. Static Equal Ensemble:** F2 achieved lower MAE than the static equal ensemble on all three benchmark datasets: Modern PJM ($-10.31\%$), GEFCom2014 ($-1.66\%$), and UCI Cohort 320 ($-5.26\%$).
2. **F2 vs. Best Standalone Expert:** F2 outperformed the best standalone expert on **two of the three datasets** (PJM: YES, GEFCom: YES, UCI: NO). On UCI, F2 ($7.74\text{ MW}$) does not beat the locked standalone LSTM reference ($7.55\text{ MW}$), though it is comparable to the Phase 11 test benchmark ($7.79\text{ MW}$).
3. **Single-Dataset Winners & Balance:** F2 is not the lowest-MAE model on every single dataset. Candidate F5 achieved lower MAE on Modern PJM ($246.71\text{ MW}$) and Candidate F3 achieved lower MAE on UCI Cohort 320 ($7.71\text{ MW}$). F2 was selected as the final formulation because it provided the strongest overall cross-dataset balance rather than the lowest individual-dataset error.

---

## 9. Non-Overlapping Daily-Block Statistical Inference

To prevent degrees-of-freedom inflation from overlapping sliding windows, statistical tests were conducted strictly on non-overlapping 24-hour daily blocks ($K=53$ PJM, $K=456$ GEFCom, $K=163$ UCI). Step-Down Holm-Bonferroni correction was applied within each dataset family across all candidate comparisons against F0_Canonical_V1.

*(Source: `research/results/phase14_statistical_comparisons.csv`, Mode: `5seed_mean`)*

| Comparison vs. F0 | Dataset | $K$ Blocks | Mean Daily Diff | 95% Conf Interval | Paired $t$-stat | Holm-Adjusted $p$-val ($t$) | Wilcoxon $W$ | Holm-Adjusted $p$-val ($W$) | Cohen's $d_z$ | Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F2 vs. F0** | **PJM** | 53 | **-9.66 MW** | **[-17.07, -2.26]** | **-2.557** | **0.0406** | 470.0 | 0.0893 | -0.351 | **Statistically Favors F2** |
| **F2 vs. F0** | **GEFCom** | 456 | **-0.688 kW** | **[-0.802, -0.575]** | **-11.850** | **1.02e-27** | 20445.0 | **1.27e-28** | **-0.555** | **Decisively Favors F2** |
| **F2 vs. F0** | **UCI** | 163 | **-0.202 MW** | **[-0.310, -0.094]** | **-3.658** | **0.0017** | 4235.0 | **0.0002** | -0.287 | **Statistically Favors F2** |
| F1 vs. F0 | PJM | 53 | -1.98 MW | [-5.66, +1.71] | -1.051 | 0.3758 | 679.0 | 0.7859 | -0.144 | Not Significant |
| F1 vs. F0 | GEFCom | 456 | -0.290 kW | [-0.356, -0.224] | -8.566 | 3.34e-16 | 28468.0 | 9.52e-17 | -0.401 | Decisively Favors F1 |
| F1 vs. F0 | UCI | 163 | -0.045 MW | [-0.106, +0.017] | -1.424 | 0.3129 | 5968.0 | 0.4722 | -0.112 | Not Significant |
| F3 vs. F0 | PJM | 53 | -4.35 MW | [-10.74, +2.04] | -1.334 | 0.3758 | 619.0 | 0.7859 | -0.183 | Not Significant |
| F3 vs. F0 | GEFCom | 456 | -0.564 kW | [-0.675, -0.453] | -9.974 | 7.65e-21 | 24851.0 | 1.13e-21 | -0.467 | Decisively Favors F3 |
| F3 vs. F0 | UCI | 163 | -0.153 MW | [-0.252, -0.055] | -3.060 | 0.0104 | 5250.0 | 0.0703 | -0.240 | Favors F3 ($t$ only) |
| F4 vs. F0 | PJM | 53 | -11.04 MW | [-18.83, -3.25] | -2.778 | 0.0378 | 412.0 | 0.0361 | -0.382 | Favors F4 |
| F4 vs. F0 | GEFCom | 456 | -0.491 kW | [-0.587, -0.396] | -10.085 | 4.03e-21 | 24466.0 | 3.92e-22 | -0.472 | Decisively Favors F4 |
| F4 vs. F0 | UCI | 163 | +0.128 MW | [+0.026, +0.231] | +2.452 | **0.0458** | 5362.0 | 0.0858 | +0.192 | **Statistically Degrades** |
| F5 vs. F0 | PJM | 53 | -8.08 MW | [-13.78, -2.38] | -2.779 | 0.0378 | 428.0 | 0.0437 | -0.382 | Favors F5 |
| F5 vs. F0 | GEFCom | 456 | -0.311 kW | [-0.398, -0.224] | -6.999 | 9.26e-12 | 35105.0 | 1.59e-09 | -0.328 | Decisively Favors F5 |
| F5 vs. F0 | UCI | 163 | +0.024 MW | [-0.111, +0.159] | +0.345 | 0.7304 | 6467.0 | 0.7204 | +0.027 | Not Significant |

### Statistical Synthesis
Among the evaluated formulations, F2 was the only candidate showing statistically significant improvement over Canonical V1 across all three datasets under the predefined daily-block paired analysis with Holm correction.

---

## 10. Confidence Dynamics & Shrinkage Resolution

*(Source: `research/results/phase14_confidence_statistics.csv`)*

| Candidate | Dataset | Mean $\lambda$ | Std $\lambda$ (`ddof=0`) | Min $\lambda$ | Max $\lambda$ | Frac $<0.1$ | Frac $>0.9$ | Coeff. of Variation (CV) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F2_A2_OOF** | PJM | 0.5066 | 0.0038 | 0.4948 | 0.5190 | 0.0000 | 0.0000 | 0.75% |
| **F2_A2_OOF** | GEFCom | 0.5170 | 0.0055 | 0.4771 | 0.5326 | 0.0000 | 0.0000 | 1.06% |
| **F2_A2_OOF** | UCI | 0.5064 | 0.0030 | 0.4874 | 0.5130 | 0.0000 | 0.0000 | 0.59% |
| **F3_Conf_Only**| PJM | 0.5046 | 0.0037 | 0.4955 | 0.5159 | 0.0000 | 0.0000 | 0.73% |
| **F3_Conf_Only**| GEFCom | 0.5183 | 0.0067 | 0.4898 | 0.5492 | 0.0000 | 0.0000 | 1.29% |
| **F3_Conf_Only**| UCI | 0.5015 | 0.0032 | 0.4820 | 0.5078 | 0.0000 | 0.0000 | 0.64% |
| **F5_Scalar** | All | 0.6233 | 0.0000 | 0.6233 | 0.6233 | 0.0000 | 0.0000 | 0.00% |

### Resolution of Secondary Research Question
1. **Near-Constant Shrinkage Behavior:** Rather than exhibiting strong temporal regime switching, the learned confidence mechanism behaved approximately as a near-constant shrinkage coefficient toward equal fusion:
   \[
   \hat{y}(t) \approx 0.51 \hat{y}_{\text{adaptive}}(t) + 0.49 \hat{y}_{\text{equal}}(t)
   \]
   Temporal standard deviations remained below $0.006$ ($CV < 1.1\%$).
2. **Context-Conditioned vs. Global Scalar Shrinkage:** Candidate F5 applied a globally fixed scalar shrinkage ($\lambda^* = 0.6233$). While effective on Modern PJM ($246.71\text{ MW}$) and GEFCom ($12.66\text{ kW}$), it degraded on UCI Cohort 320 ($8.14\text{ MW}$, failing to beat V1 or the equal ensemble). A globally fixed shrinkage coefficient did not provide the same cross-dataset robustness as the context-conditioned confidence formulation.

---

## 11. Dissociation of OOF Features vs. Confidence Fallback (F2 vs. F3)

Comparing F2 (7D Context + OOF Performance Features + Fallback) against F3 (4D Context + Fallback Only) reveals:
- On UCI Cohort 320, F3 slightly outperformed F2 ($7.71\text{ MW}$ vs. $7.74\text{ MW}$).
- On Modern PJM, F2 outperformed F3 ($250.97\text{ MW}$ vs. $255.99\text{ MW}$).
- On GEFCom2014, F2 slightly outperformed F3 ($12.41\text{ kW}$ vs. $12.44\text{ kW}$).
- **Scientific Conclusion:** The results suggest that the confidence fallback contributes to robustness, while the additional OOF expert-performance features provide dataset-dependent incremental benefit.

---

## 12. PJM Aggregation Discrepancy Reconciliation

The historical reporting variance on Modern PJM is reconciled by explicitly distinguishing three mathematical estimands:
1. **Estimand 1 — Five-Seed Independent Evaluation Mean:**
   \[
   \text{MAE}_{\text{mean}} = \frac{1}{S}\sum_{s=1}^S \text{MAE}(y, \hat{y}^{(s)}) = \mathbf{250.97 \pm 10.69\text{ MW}} \quad (S=5)
   \]
   *Standard:* This is the primary headline metric for benchmark comparison tables.
2. **Estimand 2 — Seed 42 Paired Daily Blocks:**
   \[
   \text{MAE}_{\text{Seed42}} = \mathbf{237.28\text{ MW}} \quad (K=53\text{ blocks})
   \]
   *Standard:* Represents a single seed realization, reporting higher performance due to favorable initialization.
3. **Estimand 3 — Five-Seed Ensemble Forecast Daily Blocks:**
   \[
   \text{MAE}_{\text{Ens}} = \text{MAE}\left(y, \frac{1}{S}\sum_{s=1}^S \hat{y}^{(s)}\right) = \mathbf{242.81\text{ MW}} \quad (K=53\text{ blocks})
   \]
   *Standard:* Represents the inferential unit for non-overlapping daily-block statistical hypothesis testing (mean daily difference vs. V1 $= -9.66\text{ MW}$, $p_{\text{adj}} = 0.0406$).

---

## 13. Final Model Selection Statement

**Candidate F2_A2_OOF is selected as the final CAEG-Net formulation for the paper based on its strongest overall cross-dataset balance, consistent improvement over canonical V1 across all three benchmark datasets, statistically significant daily-block improvement over V1 under the predefined Holm-corrected analysis, lower MAE than the static equal ensemble on all three datasets, superiority over the verified best standalone expert on two of three datasets, causal OOF performance features, and minimal additional parameter complexity.**

*Explicit Qualification:* F2 is not the lowest-MAE formulation on every individual dataset; F5 achieved the lowest PJM MAE and F3 achieved the lowest UCI MAE among the Phase 14 candidates. F2 was selected as the final formulation because it provided the strongest overall cross-dataset balance rather than the lowest individual-dataset error.

---

## 14. Final Paper-Facing Scientific Conclusion

Across Modern PJM, GEFCom2014, and the UCI Cohort 320 benchmark, F2_A2_OOF provided the strongest overall cross-dataset balance among the evaluated Phase 14 formulations. It improved upon canonical V1 in all three benchmarks and achieved lower MAE than the static equal ensemble on all three. Under the predefined non-overlapping daily-block paired analysis with Holm correction, F2 showed statistically significant improvement over V1 on all three datasets. F2 also outperformed the verified best standalone expert on two of the three datasets. Importantly, F2 was not the lowest-MAE formulation on every individual dataset, so its selection is based on cross-dataset robustness and consistency rather than universal per-dataset optimality.

Phase 14 is scientifically reconciled and ready to serve as the final experimental basis for the research paper.
