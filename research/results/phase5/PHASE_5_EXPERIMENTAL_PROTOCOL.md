# CAEG-Net Phase 5: Experimental Protocol & Statistical Validation Design
**Document**: `PHASE_5_EXPERIMENTAL_PROTOCOL.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: APPROVED PROTOCOL  

---

## 1. Experimental Design & Partitioning

### 1.1 Dataset & Time Horizon
- **Corpus**: Modern PJM regional load dataset (`data/Modern_PJM/pjm_load.csv`).
- **Span**: 8,784 hourly observations (1 complete annual cycle: 2023-10-01 04:00 to 2024-10-01 03:00 UTC).
- **Split Breakdown**:
  - **Training**: First 70% (6,148 hours).
  - **Validation**: Next 15% (1,318 hours).
  - **Test**: Final 15% (1,318 hours).
- **Data Scaling**: `StandardScaler` fitted strictly on training partition ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$). Zero validation or test information is used during fit.
- **Windowing Parameters**: Lookback $L=168$ hours (1 week history), Forecast Horizon $H=24$ hours (next day 24-step forecast). Prepending $L-1=167$ hours at partition boundaries ensures zero missing forecast origins without leakage.

---

## 2. Validation-Only Model Selection Protocol

To prevent test-set overfitting and guarantee statistical validity:
1. **Screening & Ablation (Phase A - D)**:
   - Evaluated strictly on the **Validation Partition** using random seed `42`.
   - Every architectural component (disagreement signal, auxiliary loss weight $\lambda_{\text{aux}}$, entropy penalty $\beta_{\text{ent}}$, bounded routing coefficient $\rho$) is evaluated by its **Validation MAE (MW)**.
2. **Candidate Selection (Phase E)**:
   - The final Phase 5 candidate architecture is synthesized exclusively from mechanisms demonstrating validated improvements on the validation set.
   - The test set is not queried during this synthesis.
3. **Multi-Seed Test Evaluation (Phase F)**:
   - The frozen candidate and all baselines are evaluated across the 5 canonical seeds:
     $$\mathcal{S} = \{42, 123, 2024, 3407, 999\}$$
   - Metrics reported: Mean $\pm$ Standard Deviation for MAE, RMSE, $R^2$, MAPE, and MSE.

---

## 3. Rigorous Statistical Testing Protocol

### 3.1 Non-Overlapping Daily Forecast Blocks
Due to the multi-step nature of the 24-hour forecast, rolling hourly evaluations produce an autocorrelation structure of order $q=23$. To eliminate this dependency:
- The 1,318-hour test partition is evaluated on $K=53$ non-overlapping 24-hour daily blocks ($53 \times 24 = 1,272$ hours).
- For each daily block $k \in \{1, \dots, 53\}$, the mean absolute error is computed for each competing model $A$ and $B$:
  $$\bar{e}_k^{(A)} = \frac{1}{24} \sum_{h=1}^{24} |y_{24(k-1)+h} - \hat{y}_{24(k-1)+h}^{(A)}|$$
- The paired series $d_k = \bar{e}_k^{(A)} - \bar{e}_k^{(B)}$ represents independent daily loss differentials.

### 3.2 Statistical Tests & Correction
1. **Paired Student's $t$-Test**:
   $$t = \frac{\bar{d}}{s_d / \sqrt{K}}, \quad \text{df} = K - 1 = 52$$
2. **Wilcoxon Signed-Rank Test**:
   Non-parametric test on paired block errors without normality assumptions.
3. **Harvey-Leybourne-Newbold (HLN) Diebold-Mariano Test**:
   Diebold-Mariano statistic with small-sample correction:
   $$\text{HLN} = \sqrt{\frac{T + 1 - 2H + H(H-1)/T}{T}} \cdot \text{DM}$$
4. **Holm-Bonferroni FWER Control**:
   $p$-values from the multiple paired comparisons against competing models are sorted and compared to step-down significance thresholds $\alpha / (M - i + 1)$ with $\alpha = 0.05$.
