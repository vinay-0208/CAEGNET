# CAEG-Net Phase 6: Rigorous Statistical Significance & Hypothesis Testing Evaluation
**Document**: `research/results/PHASE_6_STATISTICAL_EVALUATION.md`  
**Track**: `research-track` | **Phase**: Phase 6 — Five-Seed Finalist Evaluation  
**Date**: September 2026  
**Status**: COMPLETE AUDIT & RIGOROUS INFERENCE  

---

## 1. Executive Summary

Phase 6 provides the locked statistical evaluation of the Phase-5-selected finalists under the verified statistical methodology. Earlier developmental phases identified two critical inferential fallacies:
1. **The Flattening Fallacy ($T=31,056$)**: Flattening the $(1294, 24)$ forecast array inflated sample size by $24\times$, artificially deflating standard errors and generating astronomically high Diebold-Mariano statistics ($|\text{DM}| > 30$).
2. **Rolling Overlap Dependency**: Adjacent 24-step forecast origins overlap for 23 hours, violating the independence assumption required for standard hypothesis tests.

Under the corrected framework established in Phase 5:
- All models are evaluated across **$K=53$ contiguous, non-overlapping 24-hour daily blocks** ($53 \times 24 = 1,272$ hours).
- For each day $k \in \{1, \dots, 53\}$, the daily block Mean Absolute Error (MAE) is computed.
- The loss differential $d_k = \bar{e}_k^{(1)} - \bar{e}_k^{(2)}$ represents the daily performance delta.
- Statistical significance is assessed using:
  1. **Paired Student's $t$-test** on daily blocks ($df = 52$).
  2. **Wilcoxon signed-rank test** (non-parametric).
  3. **Harvey-Leybourne-Newbold (HLN) adjusted Diebold-Mariano test** on the daily series ($h=1$ day), where $\text{DM}_{\text{HLN}} \equiv t_{\text{paired}}$ algebraically.
  4. **Holm-Bonferroni step-down correction** across all 6 primary pairwise comparisons to control Family-Wise Error Rate (FWER) at $\alpha = 0.05$.

---

## 2. Statistical Testing Results ($K=53$ Daily Blocks)

| Comp ID | Model 1 ($M_1$) | Model 2 ($M_2$) | Mean Paired Diff $\bar{d}$ (MW) | 95% Confidence Interval (MW) | Cohen's $d$ | Paired $t$ ($\text{DM}_{\text{HLN}}$) | Raw $p$-value ($t$) | Wilcoxon Stat | Raw $p$-value ($W$) | Holm-Bonferroni Adjusted $p$ | Significant at $\alpha=0.05$? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | CAEG V1 | Standalone LSTM | $-29.57$ | $[-56.83, -2.30]$ | $-0.30$ | $-2.176$ | $0.0341$ | $487.0$ | $0.0431$ | $0.1705$ | **No** (FWER corrected) |
| **C2** | CAEG V1 | Standalone TCN | $+1.30$ | $[-19.63, +22.24]$ | $+0.02$ | $+0.125$ | $0.9011$ | $668.0$ | $0.6741$ | $1.0000$ | **No** |
| **C3** | CAEG V1 | Standalone CNN | $-96.11$ | $[-136.26, -55.96]$ | $-0.66$ | $-4.803$ | $1.37 \times 10^{-5}$ | $212.0$ | $8.30 \times 10^{-6}$ | **$8.19 \times 10^{-5}$** | **Yes** |
| **C4** | CAEG V1 | Equal Ensemble | $-14.76$ | $[-39.35, +9.83]$ | $-0.17$ | $-1.204$ | $0.2339$ | $573.0$ | $0.2071$ | $0.9355$ | **No** |
| **C5** | Bounded CAEG | CAEG V1 | $+6.16$ | $[-15.40, +27.72]$ | $+0.08$ | $+0.573$ | $0.5691$ | $596.0$ | $0.2901$ | $1.0000$ | **No** |
| **C6** | Bounded CAEG | Equal Ensemble | $-8.60$ | $[-26.93, +9.73]$ | $-0.13$ | $-0.942$ | $0.3507$ | $651.0$ | $0.5680$ | $1.0000$ | **No** |

*Note: Negative $\bar{d}$ indicates Model 1 achieves lower error (superior accuracy).*

---

## 3. Detailed Scientific Findings by Primary Comparison

### C1: CAEG-Net V1 vs Standalone LSTM
- **Numerical Result**: CAEG V1 achieves an average daily MAE reduction of **$29.57$ MW** relative to Standalone LSTM.
- **Inference**: The unadjusted paired $t$-test indicates $p = 0.0341$, and Wilcoxon gives $p = 0.0431$. However, when corrected for family-wise multiple testing via Holm-Bonferroni, the adjusted $p$-value is $0.1705$, which does not meet the $\alpha = 0.05$ significance threshold.
- **Conclusion**: CAEG V1 demonstrates consistent numerical outperformance over LSTM, but after rigorous multiple-testing control, the advantage is statistically suggestive rather than conclusive.

### C2: CAEG-Net V1 vs Standalone TCN
- **Numerical Result**: The mean paired daily difference between CAEG V1 and Standalone TCN is **$+1.30$ MW** ($t = +0.125, p = 0.9011$).
- **Inference**: The 95% confidence interval spans $[-19.63, +22.24]$ MW, perfectly centered near zero. Cohen's $d = 0.02$.
- **Conclusion**: CAEG-Net V1 and Standalone TCN exhibit statistically indistinguishable predictive accuracy on the Modern PJM test partition. The gating network achieves parity with the strongest individual temporal expert without succumbing to the degraded accuracy of the weaker experts.

### C3: CAEG-Net V1 vs Standalone CNN
- **Numerical Result**: CAEG V1 substantially outperforms Standalone CNN by **$-96.11$ MW** ($t = -4.803, p = 1.37 \times 10^{-5}$, Holm-Bonferroni $p = 8.19 \times 10^{-5}$).
- **Inference**: Cohen's $d = -0.66$ (medium-to-large effect size). The 95% confidence interval $[-136.26, -55.96]$ MW excludes zero by a wide margin.
- **Conclusion**: Statistically decisive outperformance. The gating mechanism successfully suppresses the temporal pooling limitations of the un-optimized CNN.

### C4: CAEG-Net V1 vs Static Equal Ensemble (Core Scientific Question)
- **Central Hypothesis**: *Does context-adaptive gating improve upon simple equal-weight fusion of the same LSTM + TCN + CNN experts?*
- **Numerical Result**: CAEG V1 achieves a **$14.76$ MW** daily MAE advantage over the Static Equal Ensemble ($251.74$ MW vs $279.79$ MW across 5 seeds; daily block $\bar{d} = -14.76$ MW).
- **Inference**: The paired $t$-test yields $t = -1.204, p = 0.2339$, and Wilcoxon yields $W = 573.0, p = 0.2071$. The 95% confidence interval is $[-39.35, +9.83]$ MW, which spans zero. Holm-Bonferroni adjusted $p = 0.9355$.
- **Scientific Conclusion**: CAEG-Net V1 demonstrates a **favorable numerical improvement (+28.05 MW 5-seed MAE gain)** over the static equal ensemble by learning to place lower effective weight on the undertrained CNN expert. However, under the non-overlapping daily block protocol, the daily error variance yields a 95% confidence interval that overlaps zero. Thus, while practically advantageous, the difference is **statistically inconclusive**.

### C5: Bounded CAEG-Net ($\rho=0.50$) vs CAEG-Net V1
- **Numerical Result**: Bounded CAEG ($\rho=0.50$) achieves a 5-seed mean MAE of $250.56 \pm 7.51$ MW compared to $251.74 \pm 7.79$ MW for unconstrained V1. The daily block difference is $+6.16$ MW ($t = +0.573, p = 0.5691$).
- **Conclusion**: Bounded routing provides tighter cross-seed variance (SD $7.51$ MW vs $7.79$ MW) and comparable test accuracy, but is statistically indistinguishable from canonical V1.

### C6: Bounded CAEG-Net ($\rho=0.50$) vs Static Equal Ensemble
- **Numerical Result**: Bounded CAEG achieves a daily block error reduction of **$-8.60$ MW** ($[-26.93, +9.73]$ MW, $t = -0.942, p = 0.3507$, adjusted $p = 1.0$).
- **Conclusion**: Bounded CAEG exhibits superior numerical accuracy over the equal ensemble ($\Delta = 29.23$ MW 5-seed mean MAE), but does not establish statistically decisive separation under daily block testing.

---

## 4. Methodological Compliance Verification

1. **Absence of 31,056-Hour Flattening**: Verified that no metric or statistic was computed by treating 31,056 hourly predictions as independent observations.
2. **Strict Daily Block Independence**: All tests evaluated on $K=53$ non-overlapping blocks of 24 contiguous hours.
3. **Equivalence Identity**: Verified that for $h=1$ on daily blocks, $\text{DM}_{\text{HLN}} \equiv t_{\text{paired}} = -1.2044$ for C4.
4. **FWER Multiple Testing Control**: Holm-Bonferroni correction strictly applied across the full family of 6 primary hypotheses.
