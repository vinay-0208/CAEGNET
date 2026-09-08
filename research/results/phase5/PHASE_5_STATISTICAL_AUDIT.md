# CAEG-Net Phase 5: Statistical Methodology Audit & Corrected Testing Framework
**Document**: `PHASE_5_STATISTICAL_AUDIT.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: AUDIT COMPLETE & RATIFIED  

---

## 1. Executive Summary & Problem Diagnosis

Earlier iterations of the CAEG-Net project suffered from statistical contradictions where Diebold-Mariano tests reported extreme significance ($|\text{DM}| > 15, p < 10^{-20}$) while paired $t$-tests reported non-significance ($p > 0.30$).

This audit pinpoints the exact mathematical root causes:
1. **The Overlapping Multi-Step Forecast Fallacy**: Rolling day-ahead forecasts with hourly step sizes share 23 out of 24 target hours. Treating hourly evaluations as independent data points deflates the variance estimator by nearly $5\times$, generating spurious significance.
2. **Missing Newey-West / HAC Truncation**: Standard variance estimators omit autocovariance terms, violating asymptotic assumptions of the Diebold-Mariano test.
3. **Absence of Small-Sample Corrections**: In sample sizes under 100, standard normal critical values over-reject the null hypothesis.

---

## 2. Corrected Statistical Evaluation Framework

### 2.1 The Non-Overlapping Daily Block Partition ($K=53$)
To eliminate serial correlation without sacrificing data integrity:
- The 1,318-hour test series is partitioned into $K=53$ contiguous, non-overlapping 24-hour daily evaluation blocks ($53 \times 24 = 1,272$ hours).
- Each block corresponds to a discrete, operational day-ahead forecast cycle:
  $$\bar{e}_k^{(m)} = \frac{1}{24} \sum_{h=1}^{24} |y_{24(k-1)+h} - \hat{y}_{24(k-1)+h}^{(m)}|$$
- The daily loss differential $d_k = \bar{e}_k^{(A)} - \bar{e}_k^{(B)}$ forms a time series of $K=53$ observations free of the sliding-window overlap pathology.

### 2.2 Harvey-Leybourne-Newbold (HLN) Diebold-Mariano Test
For horizon $h=1$ on daily blocks (or $h=24$ on full series), the Diebold-Mariano statistic is adjusted using the Harvey, Leybourne, and Newbold (1997) small-sample modification:
$$\text{DM}_{\text{HLN}} = \text{DM} \cdot \left[ \frac{T + 1 - 2h + h(h-1)/T}{T} \right]^{1/2}$$
Under the null hypothesis of equal forecast accuracy, $\text{DM}_{\text{HLN}}$ is evaluated against Student's $t$-distribution with $T - 1$ degrees of freedom rather than the standard normal distribution $\mathcal{N}(0, 1)$.

### 2.3 Paired Student's $t$-Test & Wilcoxon Signed-Rank Test
- **Paired $t$-Test**: Measures the parametric difference in daily mean absolute error across the 53 blocks:
  $$t = \frac{\bar{d}}{s_d / \sqrt{53}}$$
- **Wilcoxon Signed-Rank Test**: A non-parametric rank test on $d_k$ that guards against non-normality and outlier days.

### 2.4 Multi-Comparison Control: Holm-Bonferroni Procedure
To prevent false discoveries across multiple competing models (V1, Static Equal Ensemble, Standalone TCN, Standalone LSTM, Standalone CNN), the Holm-Bonferroni step-down procedure is enforced:
1. Sort the $M$ $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$.
2. Compare each $p_{(i)}$ against the adjusted threshold:
   $$\alpha_{(i)} = \frac{\alpha}{M - i + 1}, \quad \text{for } \alpha = 0.05$$
3. The hypothesis $H_{0, (i)}$ is rejected if $p_{(i)} < \alpha_{(i)}$ and all preceding hypotheses $j < i$ were rejected.
