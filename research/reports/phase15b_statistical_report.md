# Phase 15B-C: Formal Statistical Significance & Hypothesis Testing Report
## Non-Overlapping Daily-Block Paired Hypothesis Testing & Effect Size Analysis

**Authoritative Report — Experimental Hypothesis Testing**  
**Date:** September 2026  
**Status:** COMPLETE & RECONCILED  
**Reference Baseline Model:** `Control_A_F2` (`F2_A2_OOF`, 121,724 parameters)

---

## 1. Statistical Methodology & Protocol

To establish rigorous scientific validity and eliminate autocorrelation artifacts inherent in rolling time series evaluation, all hypothesis testing in Phase 15B adheres to the following protocol:

1. **Primary Statistical Unit — Non-Overlapping Daily Blocks:**
   - Rather than treating individual overlapping 24-step forecast windows as independent samples (which severely inflates Type I error due to high serial correlation), test sequences are partitioned into non-overlapping 24-hour daily blocks.
   - **PJM:** $K = 53$ independent daily blocks ($N = 1,272$ total hours).
   - **GEFCom:** $K = 456$ independent daily blocks ($N = 10,944$ total hours).
   - **UCI:** $K = 163$ independent daily blocks ($N = 3,912$ total hours).

2. **Paired Hypothesis Tests:**
   - **Paired Student's $t$-test:** Tests the null hypothesis that the mean daily paired error differential $\bar{d} = \frac{1}{K} \sum_{k=1}^K (e_{k, \text{cand}} - e_{k, \text{F2}})$ equals zero:
     $$t = \frac{\bar{d}}{s_d / \sqrt{K}}, \quad \text{df} = K - 1.$$
   - **Wilcoxon Signed-Rank Test:** Non-parametric test evaluating the median of paired differences, robust against non-normal error distributions and extreme load outliers.

3. **Multiple Testing Correction:**
   - **Holm-Bonferroni Procedure:** Family-wise error rate (FWER) is controlled at $\alpha = 0.05$ across the 5 candidate comparisons within each dataset:
     $$p_{(i)} \le \frac{\alpha}{m - i + 1}, \quad m = 5.$$

4. **Standardized Effect Size:**
   - **Cohen's $d_z$ for Paired Samples:**
     $$d_z = \frac{\bar{d}}{s_d} = \frac{t}{\sqrt{K}}.$$
     *Interpretation:* $|d_z| < 0.2$ negligible, $0.2 \le |d_z| < 0.5$ small, $0.5 \le |d_z| < 0.8$ medium, $|d_z| \ge 0.8$ large. Positive $d_z$ indicates candidate error is larger than F2 (candidate is worse).

---

## 2. Complete Statistical Comparison Table

The following table reports the formal statistical comparisons of all candidates against the authoritative reference model **Control A (Current F2)** across non-overlapping daily blocks.

| Comparison | Candidate ID | Baseline ID | Dataset | Blocks ($K$) | Mean Paired Diff | 95% Conf Interval | Cohen's $d_z$ | Paired $t$-stat | $p$-value ($t$) | Wilcoxon Stat | $p$-value (Wilcoxon) | Holm-Bonf $p$ | Significant at $\alpha=0.05$? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control B vs F2** | Control_B_FixedShrinkage | Control_A_F2 | PJM | 53 | +1.0165 MW | [-4.010, +6.043] | +0.0544 | 0.3963 | 0.6935 | 613.0 | 0.3642 | 1.0000 | False |
| **Control C vs F2** | Control_C_HorizonRouting | Control_A_F2 | PJM | 53 | +1.0066 MW | [-6.855, +8.868] | +0.0345 | 0.2510 | 0.8028 | 683.0 | 0.7736 | 0.8028 | False |
| **Control D vs F2** | Control_D_DynamicConfidence | Control_A_F2 | PJM | 53 | -7.0925 MW | [-11.056, -3.129] | -0.4818 | -3.5072 | 0.0009 | 343.0 | 0.0010 | 0.0047 | True (Seed 42 only) |
| **Candidate E1 vs F2** | Candidate_E1_HGR_FS | Control_A_F2 | PJM | 53 | +6.6623 MW | [-4.089, +17.414] | +0.1668 | 1.2145 | 0.2300 | 640.0 | 0.5039 | 0.9202 | False |
| **Candidate E2 vs F2** | Candidate_E2_HGR_DGS | Control_A_F2 | PJM | 53 | -1.2983 MW | [-10.087, +7.491] | -0.0398 | -0.2895 | 0.7733 | 640.0 | 0.5039 | 1.0000 | False |
| **Control B vs F2** | Control_B_FixedShrinkage | Control_A_F2 | GEFCom | 456 | +0.0352 kW | [-0.010, +0.080] | +0.0716 | 1.5296 | 0.1268 | 42211.0 | 0.0004 | 0.2536 | False (Holm $t$-test) |
| **Control C vs F2** | Control_C_HorizonRouting | Control_A_F2 | GEFCom | 456 | **+0.5761 kW** | **[+0.415, +0.737]** | **+0.3291** | **7.0274** | **7.73e-12** | **32743.0** | **6.23e-12** | **3.87e-11** | **True (Degraded)** |
| **Control D vs F2** | Control_D_DynamicConfidence | Control_A_F2 | GEFCom | 456 | +0.0813 kW | [+0.034, +0.129] | +0.1578 | 3.3705 | 0.0008 | 36372.0 | 2.33e-08 | 0.0024 | True (Degraded) |
| **Candidate E1 vs F2** | Candidate_E1_HGR_FS | Control_A_F2 | GEFCom | 456 | -0.1083 kW | [-0.334, +0.117] | -0.0440 | -0.9402 | 0.3476 | 47606.0 | 0.1106 | 0.3476 | False |
| **Candidate E2 vs F2** | Candidate_E2_HGR_DGS | Control_A_F2 | GEFCom | 456 | **+0.4442 kW** | **[+0.297, +0.592]** | **+0.2762** | **5.8986** | **7.16e-09** | **35173.0** | **1.84e-09** | **2.86e-08** | **True (Degraded)** |
| **Control B vs F2** | Control_B_FixedShrinkage | Control_A_F2 | UCI | 163 | -0.4359 MW | [-0.704, -0.167] | -0.2492 | -3.1815 | 0.0018 | 4576.0 | 0.0005 | 0.0088 | True (Seed 42 only) |
| **Control C vs F2** | Control_C_HorizonRouting | Control_A_F2 | UCI | 163 | +0.1318 MW | [-0.029, +0.293] | +0.1258 | 1.6067 | 0.1101 | 5147.0 | 0.0109 | 0.2201 | False |
| **Control D vs F2** | Control_D_DynamicConfidence | Control_A_F2 | UCI | 163 | +0.0873 MW | [-0.009, +0.183] | +0.1396 | 1.7822 | 0.0766 | 5637.0 | 0.0831 | 0.2298 | False |
| **Candidate E1 vs F2** | Candidate_E1_HGR_FS | Control_A_F2 | UCI | 163 | +0.1769 MW | [+0.007, +0.347] | +0.1596 | 2.0382 | 0.0432 | 5252.0 | 0.0177 | 0.1726 | False |
| **Candidate E2 vs F2** | Candidate_E2_HGR_DGS | Control_A_F2 | UCI | 163 | -0.0575 MW | [-0.255, +0.140] | -0.0446 | -0.5693 | 0.5699 | 6211.0 | 0.4342 | 0.5699 | False |

---

## 3. Methodological & Test Discrepancy Reconciliation

### 1. Control B on GEFCom: Parametric vs Non-Parametric Divergence
- **Parametric Test (Paired $t$-test):** Mean paired difference $+0.0352$ kW (95% CI [-0.010, +0.080]), $t = 1.530, p = 0.1268$. After Holm-Bonferroni adjustment across 5 comparisons, $p = 0.2536$ $\implies$ **Not statistically significant.**
- **Non-Parametric Test (Wilcoxon):** $W = 42211.0, p = 0.0004$.
- **Reconciliation:** The Wilcoxon test indicates that the median paired difference favors Control B on a majority of individual days, while the paired $t$-test indicates that the mean difference is within sampling noise due to small negative tails. In the 5-seed aggregate evaluation, Control B achieves a slightly lower mean MAE ($12.3607$ kW vs $12.4077$ kW, $-0.047$ kW). Control B is therefore competitive with F2 on GEFCom, but does not achieve a statistically significant parametric advantage under Holm correction.

### 2. Single-Seed vs Five-Seed Discrepancies (Seed 42 Realizations)
- **Control D on PJM:** On Seed 42, Control D exhibited a statistically significant lower daily MAE ($-7.09$ MW, Holm $p = 0.0047$). However, across the full 5-seed evaluation, Control A achieved a lower mean MAE ($250.9747$ MW vs $251.9419$ MW). Single-seed statistical significance does not extrapolate to general superiority across stochastic initializations.
- **Control B on UCI:** On Seed 42, Control B showed a significant daily-block difference ($-0.436$ MW, Holm $p = 0.0088$). Across 5 seeds, however, Control A achieved lower mean MAE ($7.7371$ MW vs $7.7523$ MW).

### 3. Statistically Robust Negative Results
- **Control C (Horizon Routing Only):** Statistically significantly degraded relative to F2 on GEFCom under both paired $t$-test ($p = 7.73 	imes 10^{-12}$, Holm $p = 3.87 	imes 10^{-11}$) and Wilcoxon ($p = 6.23 	imes 10^{-12}$) with a moderate effect size ($d_z = +0.3291$).
- **Candidate E2 (HGR-DGS):** Statistically significantly degraded relative to F2 on GEFCom under both tests ($t$-test $p = 7.16 	imes 10^{-9}$, Wilcoxon $p = 1.84 	imes 10^{-9}$, $d_z = +0.2762$).
- **Control D (Dynamic Confidence):** Statistically significantly degraded on GEFCom ($t$-test $p = 0.0008$, Wilcoxon $p = 2.33 	imes 10^{-8}$, $d_z = +0.1578$).

---

## 4. Summary of Statistical Findings

1. **F2 is not claimed to be universally statistically superior:** Across all comparisons, F2 is statistically distinguishable from competitors primarily where complex variants degrade (e.g., Control C and Candidate E2 on GEFCom). Between F2 and Control B, differences are small and statistically non-significant under parametric testing with Holm correction.
2. **Horizon routing causes statistically robust degradation on GEFCom:** Removing shrinkage and enforcing horizon partitions results in highly significant performance penalties ($p < 10^{-8}$).
3. **Daily-block testing eliminates serial overlap bias:** Treating non-overlapping 24-hour daily blocks as the sampling unit prevents artificial inflation of test statistics, providing conservative, publication-grade inference.
