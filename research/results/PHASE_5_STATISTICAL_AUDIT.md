# CAEG-Net Phase 5: Rigorous Statistical Testing Audit & DM/HLN Verification
**Document**: `research/results/PHASE_5_STATISTICAL_AUDIT.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: COMPLETE AUDIT & VERIFIED IMPLEMENTATION  

---

## 1. Executive Summary & Core Diagnosis

Earlier phase reports in the CAEG-Net research repository exhibited an apparent statistical contradiction:
- Paired Student's $t$-tests and Wilcoxon signed-rank tests reported moderate or non-significant differences (e.g., $t = -1.49, p = 0.1414$),
- Whereas Diebold-Mariano (DM) tests with Harvey-Leybourne-Newbold (HLN) adjustments reported astronomically extreme statistics (e.g., $\text{DM}_{\text{HLN}} = -111.89, p \approx 0.0$ or $\text{DM}_{\text{HLN}} = +34.04$).

This audit established the mathematical root cause, formulated the corrected framework, implemented regression unit tests, and demonstrated the exact before-and-after results.

---

## 2. Identified Mathematical & Implementation Flaws

### Flaw 1: The Multi-Step Flattening Fallacy ($N \times H$)
In multi-step time series forecasting, the model predicts a 2D matrix of shape $(N, H)$ representing $N$ forecast origins and $H=24$ lead times.
In previous code (`run_phase5_original_caeg.py` line 810):
```python
# Flawed implementation:
e1 = mean_preds[m1].flatten() - test_y_true.flatten()
e2 = mean_preds[m2].flatten() - test_y_true.flatten()
dm_hln, p_dm = compute_hln_diebold_mariano(e1, e2, h=1)
```
- **Error**: Flattening the $(1294, 24)$ or $(1318, 24)$ array yielded a 1D sequence of length $T = 31,056$ or $31,632$.
- **Consequence**: The test was evaluated under the assumption that there were 31,632 **independent 1-step forecast origins**!
- The estimated standard error $\text{SE} = \sqrt{\hat{\gamma}_0 / T}$ was divided by $\sqrt{31,632} \approx 177.8$.
- The test statistic was artificially inflated by $\sqrt{24} \approx 4.9\times$ relative to daily aggregates, and by nearly $25\times$ relative to non-overlapping blocks.

### Flaw 2: Overlapping Horizons Violate Independence
Two contiguous 24-step forecast origins starting at hour $t$ and $t+1$ overlap for 23 hours. Their forecast errors $e_{t}$ and $e_{t+1}$ share 23 common target points. This induces a moving-average error structure of order $\text{MA}(23)$.
Treating rolling origins as independent without proper long-run covariance truncation generates severe size distortions and false-positive rejections.

---

## 3. Corrected Formulation: Non-Overlapping Daily Blocks ($K=53$)

To eliminate the rolling overlap dependency while retaining full day-ahead operational realism:
1. The 1,272 test hours ($53 \times 24$) are partitioned into **$K=53$ contiguous, non-overlapping 24-hour daily blocks**.
2. For each day $k \in \{1, \dots, K\}$, the daily Mean Absolute Error for competing models $A$ and $B$ is:
   $$\bar{e}_k^{(A)} = \frac{1}{24} \sum_{h=1}^{24} |y_{k, h} - \hat{y}_{k, h}^{(A)}|, \quad \bar{e}_k^{(B)} = \frac{1}{24} \sum_{h=1}^{24} |y_{k, h} - \hat{y}_{k, h}^{(B)}|$$
3. The loss differential sequence is:
   $$d_k = \bar{e}_k^{(A)} - \bar{e}_k^{(B)}, \quad k \in \{1, \dots, 53\}$$
4. Because the operational blocks are non-overlapping, the forecast horizon between adjacent blocks is $h=1$ day.

### Theorem: Mathematical Equivalence of HLN and Paired $t$-test at $h=1$
Let $n=K=53$ and $h=1$.
- Sample autocovariance at lag 0: $\hat{\gamma}_0 = \frac{1}{n} \sum_{k=1}^n (d_k - \bar{d})^2 = \frac{n-1}{n} s_d^2$.
- Long-run variance estimator: $\hat{V}(\bar{d}) = \frac{\hat{\gamma}_0}{n} = \frac{n-1}{n^2} s_d^2$.
- Raw DM statistic:
  $$\text{DM}_{\text{raw}} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} = \frac{\bar{d}}{\frac{s_d}{n} \sqrt{n-1}} = \left( \frac{\bar{d}}{s_d / \sqrt{n}} \right) \cdot \sqrt{\frac{n}{n-1}}$$
- Harvey-Leybourne-Newbold (1997) correction factor for $h=1$:
  $$\text{HLN} = \sqrt{\frac{n + 1 - 2(1) + 0}{n}} = \sqrt{\frac{n-1}{n}}$$
- Corrected HLN DM statistic:
  $$\text{DM}_{\text{HLN}} = \text{DM}_{\text{raw}} \times \text{HLN} = \left( \frac{\bar{d}}{s_d / \sqrt{n}} \sqrt{\frac{n}{n-1}} \right) \times \sqrt{\frac{n-1}{n}} \equiv \frac{\bar{d}}{s_d / \sqrt{n}} \equiv t_{\text{paired}}$$

**Proof Verification**: Evaluated with df $= n-1 = 52$ against Student's $t$-distribution, **$\text{DM}_{\text{HLN}}$ is identically equal to the paired $t$-statistic**.

---

## 4. Unit Test Verification

Automated regression tests in [`research/tests/test_statistical_audit.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/tests/test_statistical_audit.py) confirm:
1. `test_mathematical_identity_at_h1`: $\text{DM}_{\text{HLN}} \equiv t_{\text{paired}}$ up to $10^{-6}$ numerical precision.
2. `test_sign_preservation`: $\text{sign}(\text{DM}) \equiv \text{sign}(\bar{d})$.
3. `test_hln_factor_properties`: Monotonically converges to 1.0 from below as $n \to \infty$.

---

## 5. Before vs. After Results Comparison

Applying the corrected implementation on the identical 5-seed benchmark model outputs ($K=53$ daily blocks):

| Competing Pair ($M_1 \text{ vs } M_2$) | Mean Diff (MW) | Paired $t$-stat | Old Flawed DM (Flattened 31k) | Corrected Daily DM HLN | Two-Sided $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 5 vs Standalone CNN** | -199.14 MW | -13.75 | -111.89 (Flawed) | **-13.75** | $5.90 \times 10^{-19}$ |
| **Phase 5 vs Static Equal Ens** | -30.90 MW | -4.74 | -39.47 (Flawed) | **-4.74** | $1.72 \times 10^{-5}$ |
| **Phase 5 vs Original CAEG V1** | +21.17 MW | +4.10 | +34.04 (Flawed) | **+4.10** | $1.43 \times 10^{-4}$ |
| **Phase 5 vs Standalone TCN** | +18.44 MW | +3.16 | +30.75 (Flawed) | **+3.16** | $0.0026$ |
| **Phase 5 vs Standalone LSTM** | -15.16 MW | -1.49 | -16.53 (Flawed) | **-1.49** | $0.1414$ |

### Conclusion:
- The apparent contradiction between $t$-tests and DM tests is **fully resolved**.
- On non-overlapping daily blocks, both tests agree exactly.
- Extreme DM statistics ($|\text{DM}| > 30$) were entirely artefacts of flattening the 2D array and treating dependent hours as 31,000 independent samples.
