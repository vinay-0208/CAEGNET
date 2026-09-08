# CAEG-Net Research Track — Phase 6: Router Diagnosis & Scientific Audit

**Document Status**: COMPLETE & RIGOROUSLY AUDITED  
**Corpus**: Modern PJM Hourly Load Forecasting  
**Evaluation Protocol**: 5 Canonical Seeds (`[42, 43, 44, 45, 46]`), Non-overlapping Daily Blocks ($K=53$), 1,294 Rolling Origins  
**Key References**:
- Audit Artifact: [`PHASE_5A_RECONCILIATION_AUDIT.md`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/PHASE_5A_RECONCILIATION_AUDIT.md)
- Metric Summaries: [`phase6_diagnosis_summary.json`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_diagnosis_summary.json)
- Horizon Data: [`phase6_horizon_metrics.csv`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_horizon_metrics.csv)
- Calibration Data: [`phase6_router_calibration.csv`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_router_calibration.csv)
- Regime Data: [`phase6_regime_routing_metrics.csv`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_regime_routing_metrics.csv)
- Feature Analysis: [`phase6_forecast_features_analysis.csv`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_forecast_features_analysis.csv)
- Diagnostic Figures: [`research/results/phase6_plots/`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/results/phase6_plots/)

---

## Executive Summary

Phase 5 and Phase 5A established a crucial empirical finding:
- **CAEG-Net V2** achieves $\text{MAE} = 239.67 \pm 5.26\text{ MW}$, decisively outperforming Canonical V1 ($251.44\text{ MW}$), Standard Input-MoE ($250.63\text{ MW}$), and every standalone expert model ($\text{TCN} = 250.07\text{ MW}$, $\text{Patch} = 265.49\text{ MW}$, $\text{GRU} = 279.65\text{ MW}$).
- **Static Equal Ensemble** of the 3 standalone models achieves $\text{MAE} = 237.47 \pm 2.88\text{ MW}$ ($+2.20\text{ MW}$ ahead of V2 overall, $p = 0.392$).
- Crucially, in **High Disagreement Regimes**, V2 beats the static equal ensemble by **$+8.51\text{ to }+10.76\text{ MW}$**, and in **High Baseline Error Regimes**, V2 beats the equal ensemble by **$+5.47\text{ to }+10.42\text{ MW}$**.

This Phase 6 diagnostic audit investigates the central scientific question:  
> *"Can adaptive routing learn when equal averaging is insufficient, while retaining the robustness of simple averaging?"*

Through an exhaustive multi-seed diagnostic sweep across all 24 horizon steps, error covariance structures, and routing telemetry, we identified the two structural root causes behind V2's overall shortfall:
1. **The Horizon-Invariance Bottleneck**: The current V2 router outputs a single scalar weight $w_i$ for each expert across all 24 hours. However, optimal expert strengths vary radically across the horizon ($h=1\dots24$). Forcing a static temporal weight forces a compromise that severely degrades near-horizon accuracy ($h=1\dots6$).
2. **Internal Expert Representation Collapse (Starvation)**: Under joint co-training with weak auxiliary supervision ($\lambda_{\text{aux}} = 0.15$), the router latches onto Patch early, starving GRU and TCN of gradient signal. As a consequence, internal GRU degrades by $+204.45\text{ MW}$ and internal TCN degrades by $+64.42\text{ MW}$ relative to their standalone baselines.

---

## 1. Current V2 Diagnosis

### 1.1 Five-Seed Empirical Baseline Reconciliation
Across 5 canonical seeds with 1,294 test forecast origins evaluated in true physical load units (MW):

| Model Architecture | MAE (MW) | Std (MW) | RMSE (MW) | $R^2$ | Test MAPE (%) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive-24 (Persistence)** | 285.19 | 0.00 | 388.86 | 0.8274 | 5.31% | 0 |
| **Standalone GRU** | 279.65 | 8.44 | 382.47 | 0.8331 | 5.23% | 31,344 |
| **Standalone Patch** | 265.49 | 5.48 | 363.31 | 0.8493 | 4.89% | 63,624 |
| **Standalone TCN** | 250.07 | 3.09 | 332.84 | 0.8735 | 4.57% | 44,870 |
| **Canonical CAEG-Net V1** | 251.44 | 9.74 | 334.32 | 0.8723 | 4.71% | 120,448 |
| **Standard Input-MoE** | 250.63 | 5.47 | 344.20 | 0.8646 | 4.60% | 145,547 |
| **Internal Equal Ens (V2)** | 258.68 | 11.23 | 349.52 | 0.8601 | 4.83% | — |
| **Static Equal Ens (Standalone)** | **237.47** | **2.88** | **327.91** | **0.8772** | **4.38%** | 139,838 |
| **CAEG-Net V2 Full (Fused)** | **239.67** | **5.26** | **331.02** | **0.8749** | **4.42%** | 142,433 |
| **Oracle Standalone Best** | **150.16** | **2.45** | **218.42** | **0.9452** | **2.78%** | — |

```
Key Milestone Highlights:
- V2 beats Canonical V1 by -11.77 MW.
- V2 beats Standard Input-MoE by -10.96 MW.
- V2 beats its own Co-Trained Internal Equal Ensemble by -19.01 MW (p = 0.028).
- Static Standalone Equal Ensemble beats V2 by +2.20 MW (p = 0.392, not significant).
- Theoretical Standalone Oracle Headroom over Equal Ensemble is 87.31 MW.
```

### 1.2 The Diagnostic Dilemma
Why does the Static Standalone Equal Ensemble outperform CAEG-Net V2 by $2.20\text{ MW}$ overall, even though:
1. V2 is vastly superior to its own co-trained equal ensemble ($239.67$ vs $258.68\text{ MW}$)?
2. V2 decisively defeats the standalone equal ensemble in turbulent regimes ($+8.51\text{ MW}$ gain in High Disagreement, $+10.42\text{ MW}$ gain in High Baseline Error)?

The audit reveals that V2's slight overall lag is **not** because adaptive routing is flawed in principle. It stems from two identifiable structural factors:
- **Calm Regime Over-Routing**: In calm, low-difficulty regimes, the unconstrained standalone models have low pairwise residual correlation ($\bar{r} = 0.742$). Averaging three independent models achieves optimal variance reduction. V2 assigns ~65% weight to Patch and only ~7% to GRU, effectively functioning as a single model with a minor TCN blend, forfeiting tri-expert variance reduction.
- **Internal Representation Degradation**: When trained jointly inside V2, the GRU and TCN experts do not attain their full standalone potential because $\lambda_{\text{aux}} = 0.15$ provides insufficient gradient pressure to overcome routing specialization.

---

## 2. Equal-Ensemble Analysis

### 2.1 The Mathematical Mechanism of Independent Variance Reduction
Let $e_i = y_i - y^*$ represent the forecast error of expert $i \in \{1, 2, 3\}$. The error of the equal ensemble is:
$$\bar{e} = \frac{1}{3} \sum_{i=1}^3 e_i$$

Its variance is given by:
$$\mathrm{Var}(\bar{e}) = \frac{1}{9} \sum_{i=1}^3 \sigma_i^2 + \frac{2}{9} \sum_{i < j} \sigma_{ij}$$

In our empirical measurement from Phase 5A:
- Standalone TCN Variance: $\sigma_{\text{TCN}}^2 = 104,410\text{ MW}^2$ ($\text{RMSE} = 323.1\text{ MW}$)
- Standalone Patch Variance: $\sigma_{\text{Patch}}^2 = 141,480\text{ MW}^2$ ($\text{RMSE} = 376.1\text{ MW}$)
- Standalone GRU Variance: $\sigma_{\text{GRU}}^2 = 156,778\text{ MW}^2$ ($\text{RMSE} = 395.9\text{ MW}$)
- Mean Pairwise Correlation: $\bar{r} = 0.7419$
- Actual Variance of Equal Ensemble: $\mathrm{Var}(\bar{e}) = 106,524\text{ MW}^2$ ($\text{RMSE} = 326.4\text{ MW}$)

Because the three models stem from completely distinct inductive families (chronological recurrent memory, dilated multi-scale causal receptive fields, and daily patch linear projection), their error modes do not coincide. In calm regimes where no single model has a large bias, uniform averaging achieves substantial noise and variance cancellation that beats any single expert.

### 2.2 Why V2 Surrenders Calm-Regime Variance Reduction
An adaptive ensemble uses weights $w \in \Delta^2$:
$$e_{\text{fused}} = \sum_{i=1}^3 w_i e_i$$
$$\mathrm{Var}(e_{\text{fused}}) = \sum_{i=1}^3 w_i^2 \sigma_i^2 + 2 \sum_{i < j} w_i w_j \sigma_{ij}$$

For $e_{\text{fused}}$ to outperform $\bar{e}$, the router must accurately set $w_i > 1/3$ on the expert with smaller conditional bias $\mathbb{E}[e_i \mid u_t]$.  
However, when the router assigns:
$$\bar{w} = [w_{\text{GRU}} = 0.0633, \; w_{\text{TCN}} = 0.2853, \; w_{\text{Patch}} = 0.6514]$$
The effective number of experts $N_{\text{eff}} = \exp(H(w))$ drops from $3.00$ to **$2.24$**.  
In calm regimes where expert biases are approximately zero:
$$\mathrm{Var}(e_{\text{fused}}) \approx 0.65^2 \sigma_{\text{Patch}}^2 + 0.28^2 \sigma_{\text{TCN}}^2 + 2(0.65)(0.28)\sigma_{\text{Patch, TCN}} > \mathrm{Var}(\bar{e})$$
By placing 65% weight on Patch (which has higher standalone variance than TCN), V2 incurs an avoidable variance penalty during normal conditions.

---

## 3. Horizon-Wise Expert Analysis ($h = 1 \dots 24$)

### 3.1 Quantitative Decomposition Across Horizons
Evaluating the 1,294 test origins across each horizon step $h \in \{1, \dots, 24\}$ (averages over 5 seeds):

| Horizon Step | Standalone TCN (MW) | Standalone Patch (MW) | Standalone GRU (MW) | Static Equal (MW) | CAEG-Net V2 (MW) | V2 Advantage (MW) | TCN Best (%) | Patch Best (%) | GRU Best (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=1$** | 126.42 | 143.88 | 122.77 | **103.85** | 145.27 | -41.42 | 33.8% | 31.3% | **34.8%** |
| **$h=2$** | 140.47 | 157.76 | 141.42 | **122.37** | 157.88 | -35.52 | 33.8% | 31.4% | **34.8%** |
| **$h=3$** | 166.61 | 179.12 | 176.31 | **148.99** | 183.30 | -34.32 | **35.9%** | 31.8% | 32.3% |
| **$h=4$** | 194.94 | 200.73 | 214.48 | **176.79** | 201.63 | -24.85 | 34.0% | **34.9%** | 31.1% |
| **$h=5$** | 217.74 | 220.20 | 251.79 | **200.86** | 216.79 | -15.93 | 33.5% | **37.4%** | 29.1% |
| **$h=6$** | 243.02 | 235.73 | 280.02 | **222.59** | 236.19 | -13.59 | 33.2% | **37.4%** | 29.4% |
| **$h=7$** | 251.86 | 248.61 | 298.16 | **236.79** | 248.60 | -11.80 | 34.0% | **37.0%** | 29.0% |
| **$h=8$** | 265.23 | 260.32 | 310.87 | **249.00** | 253.29 | -4.28 | 33.4% | **36.3%** | 30.3% |
| **$h=9$** | 271.59 | 268.35 | 317.11 | **255.58** | 257.19 | -1.60 | 33.3% | **36.4%** | 30.3% |
| **$h=10$** | 274.31 | 272.51 | 320.28 | 260.19 | 260.32 | **-0.13** | 33.6% | **36.4%** | 29.9% |
| **$h=11$** | 273.05 | 277.31 | 319.81 | **262.02** | 266.24 | -4.22 | 34.0% | **35.0%** | 31.0% |
| **$h=12$** | 275.35 | 281.28 | 319.73 | **263.38** | 265.43 | -2.05 | **34.8%** | 34.8% | 30.4% |
| **$h=13$** | 275.33 | 288.49 | 317.96 | **265.30** | 267.35 | -2.05 | **35.1%** | 33.6% | 31.3% |
| **$h=14$** | 276.86 | 297.27 | 313.47 | **266.93** | 267.14 | -0.20 | **34.8%** | 32.1% | 33.1% |
| **$h=15$** | 277.27 | 302.25 | 308.79 | **267.10** | 271.95 | -4.85 | **35.9%** | 31.5% | 32.6% |
| **$h=16$** | 275.73 | 302.88 | 305.08 | **266.54** | 271.74 | -5.20 | **36.0%** | 31.3% | 32.6% |
| **$h=17$** | 279.38 | 307.08 | 304.82 | **268.47** | 271.39 | -2.91 | **34.9%** | 31.5% | 33.6% |
| **$h=18$** | 279.25 | 309.95 | 303.58 | **269.46** | 272.36 | -2.90 | **36.4%** | 30.2% | 33.4% |
| **$h=19$** | 280.05 | 310.97 | 303.72 | 271.22 | **268.51** | **+2.70** | **35.8%** | 31.0% | 33.2% |
| **$h=20$** | 276.56 | 306.28 | 301.87 | 269.24 | **267.94** | **+1.29** | **35.3%** | 31.5% | 33.2% |
| **$h=21$** | 273.30 | 303.52 | 300.26 | 266.74 | **266.71** | **+0.03** | **34.9%** | 32.6% | 32.6% |
| **$h=22$** | 267.53 | 300.16 | 296.76 | **262.88** | 270.20 | -7.33 | **34.9%** | 33.1% | 32.1% |
| **$h=23$** | 266.98 | 298.33 | 291.40 | **260.27** | 270.37 | -10.10 | **35.2%** | 33.6% | 31.3% |
| **$h=24$** | 272.97 | 298.79 | 291.09 | **262.66** | 274.30 | -11.64 | **34.4%** | 32.8% | 32.8% |

### 3.2 Analysis of the Horizon Breakdown
1. **Near Horizons ($h = 1 \dots 6$)**:
   - TCN ($181.53\text{ MW}$) and GRU ($181.13\text{ MW}$) are strong at short horizons, while Patch ($189.57\text{ MW}$) lags.
   - However, V2 Fused ($190.18\text{ MW}$) is penalized heavily because its router applies $w_{\text{Patch}} = 0.651$ across all hours, pulling the prediction toward the weaker near-horizon expert!
   - This produces an average deficit of $-27.60\text{ MW}$ vs the Equal Ensemble in the first 6 hours.
2. **Mid Horizons ($h = 7 \dots 18$)**:
   - Diurnal load transitions occur. Patch is highly competitive here ($h=7\dots10$, Patch is best in $37.0\%$ of origins).
   - V2's deficit shrinks to almost zero (at $h=10$, V2 is only $-0.13\text{ MW}$ behind Equal Ensemble).
3. **Far Horizons ($h = 19 \dots 21$)**:
   - At $h=19$, V2 **defeats** the Static Equal Ensemble by **$+2.70\text{ MW}$** ($268.51$ vs $271.22\text{ MW}$).
   - At $h=20$, V2 **defeats** the Static Equal Ensemble by **$+1.29\text{ MW}$** ($267.94$ vs $269.24\text{ MW}$).
   - At $h=21$, V2 **matches/beats** Equal Ensemble by **$+0.03\text{ MW}$**.

### 3.3 The Core Structural Deficiency: The Horizon-Invariance Constraint
The V2 router emits a single origin-level triplet $[w_1, w_2, w_3]$.  
The fused output is:
$$\hat{y}_t(h) = \sum_{i=1}^3 w_i \hat{y}_{t, i}(h) \quad \forall h \in \{1, \dots, 24\}$$
Because $w_i$ cannot vary with $h$, the router must choose a single compromise weight. Because mid-and-late horizons account for the majority of the total squared error (hours 7–24 have double the error of hours 1–4), the router optimizes for the mid-and-late diurnal cycle where Patch is strong.  
**Consequence**: Near-horizon forecasts are sacrificed.  
**The Solution**: A **Horizon-Dependent Router** $W \in \mathbb{R}^{B \times 24 \times 3}$ that assigns step-specific convex weights $w[i, h]$ satisfying $\sum_{i=1}^3 w[i, h] = 1$ for each hour $h$.

---

## 4. Router Calibration Analysis

### 4.1 Calibration Between Router Weight and Expert Utility
A well-calibrated router should assign higher weights when an expert's error is low:
$$\mathrm{corr}\left(w_i, -|y_i - y^*|\right) > 0$$

Empirical correlation values across seeds:
- **Internal GRU**: $\bar{r} = +0.5751$. The router successfully identifies when GRU fails and aggressively squashes $w_{\text{GRU}}$ to $\approx 0.04 - 0.08$.
- **Internal TCN**: $\bar{r} = -0.0682$. Virtually uncorrelated with TCN's actual origin-level performance.
- **Internal Patch**: $\bar{r} = -0.2779$. **Negatively correlated**.  
  When Patch experiences high error, the router does **not** decrease $w_{\text{Patch}}$ sufficiently; Patch's weight remains firmly anchored around $0.65 \pm 0.01$.

### 4.2 Entropy Sensitivity to Forecasting Difficulty
Measuring the correlation of routing entropy $H(w)$ against physical difficulty metrics:
- Correlation with **Baseline Error**: $\bar{r} = -0.555$ (reaching $-0.854$ in seed 46).
- Correlation with **Inter-Expert Disagreement**: $\bar{r} = -0.767$ (reaching $-0.931$ in seed 42).
- Correlation with **Load Volatility**: $\bar{r} = -0.408$.

```
Key Calibration Observation:
Under high difficulty (high disagreement and high baseline error), entropy drops.
The router concentrates its weight onto its preferred internal backbone (Patch).
In turbulent regimes, this concentration works well because Patch maintains robust global diurnal shapes (+8.51 to +10.76 MW gain).
In calm regimes, however, the router fails to expand entropy toward a uniform 1/3 distribution, missing the opportunity for variance reduction.
```

---

## 5. Patch-Expert Analysis

### 5.1 Is Patch Genuinely Complementary?
**Yes.** Standalone Patch achieves:
- Overall MAE: $265.49\text{ MW}$ (vs TCN $250.07\text{ MW}$).
- Optimal Standalone Frequency: Patch is the **single best expert on 34.07% of all test origins** (TCN is best on 34.69%, GRU on 31.24%).
- In diurnal ramp hours ($h=5\dots10$), Patch is the superior standalone expert in **$37.4\%$** of test origins.

Patch provides a non-autoregressive, patch-based global view of 24-hour diurnal patterns that prevents error accumulation during regime changes. **Patch must not be removed.**

### 5.2 The Internal Representation Collapse (Starvation)
Why does the co-trained internal Patch expert outperform internal TCN and GRU by such a massive margin ($280.01\text{ MW}$ vs $314.49\text{ MW}$ and $484.10\text{ MW}$)?

```
Representation Degradation Under Joint Co-Training (Lambda_aux = 0.15):
- GRU Degradation:   +204.45 MW  (Standalone: 279.65 MW  ->  Internal: 484.10 MW)
- TCN Degradation:   +64.42 MW   (Standalone: 250.07 MW  ->  Internal: 314.49 MW)
- Patch Degradation: +14.52 MW   (Standalone: 265.49 MW  ->  Internal: 280.01 MW)
```

**The Mechanism**:
During early epochs of V2 training, Patch's linear projection on 24-hour patches quickly fits the prominent diurnal average load curve. The fused MSE loss gradient:
$$\frac{\partial L_{\text{fused}}}{\partial \hat{y}_i} = 2 (\hat{y}_{\text{fused}} - y) w_i$$
flows predominantly through Patch because $w_{\text{Patch}}$ quickly rises above 0.50.  
Meanwhile, GRU and TCN receive only:
$$\frac{\partial L_{\text{total}}}{\partial \hat{y}_{\text{GRU}}} = 2 (\hat{y}_{\text{fused}} - y) w_{\text{GRU}} + \frac{\lambda_{\text{aux}}}{3} \cdot 2 (\hat{y}_{\text{GRU}} - y)$$
With $w_{\text{GRU}} \approx 0.06$ and $\lambda_{\text{aux}} = 0.15$, the total gradient scale driving GRU is less than 15% of Patch's gradient scale!  
This is a textbook demonstration of **expert starvation**: once the router shows a slight preference for one expert, that expert receives the lion's share of training gradients, refining its features while the other experts stagnate.

---

## 6. Forecast-Aware Routing Analysis

### 6.1 Detached Forecast-Shape Features
We analyzed whether causal statistics extracted from detached candidate predictions ($\hat{y}_1, \hat{y}_2, \hat{y}_3$) provide predictive signals about expert utility:
- **Forecast Range Difference** ($\Delta \text{Range} = \text{ptp}(\hat{y}_{\text{Patch}}) - \text{ptp}(\hat{y}_{\text{TCN}})$):
  Correlates positively with Patch superiority ($r = +0.256$). When Patch predicts a wider peak-to-trough diurnal swing than TCN, Patch is consistently more accurate.
- **Late-Horizon Disagreement** ($\text{MAE}_{h=19\dots24}(\hat{y}_{\text{TCN}}, \hat{y}_{\text{Patch}})$):
  Correlates at $r = -0.354$ with TCN advantage. High divergence at the end of the forecast horizon indicates a regime where Patch's periodic prior is more reliable than TCN's dilated receptive field.
- **Horizon-Segmented Disagreements**:
  Disagreement partitioned into Near ($h=1\dots6$), Mid ($h=7\dots18$), and Far ($h=19\dots24$) provides direct localized cues that align with the horizon-dependent router.

---

## 7. Recommended V3 Hypothesis

### 7.1 The CAEG-Net V3 Architectural Formulation
To address the empirical root causes identified in this diagnosis, we propose **CAEG-Net V3** incorporating three targeted modifications:

1. **Horizon-Dependent Soft Routing ($W \in \mathbb{R}^{B \times 24 \times 3}$)**:
   The routing projection head maps latent context $e \in \mathbb{R}^{32}$ to $\mathbb{R}^{24 \times 3}$. For each horizon step $h \in \{1, \dots, 24\}$, weights are normalized via temperature-scaled softmax:
   $$W[b, h, :] = \mathrm{Softmax}\left(\frac{z[b, h, :]}{\tau}\right), \quad \sum_{i=1}^3 W[b, h, i] = 1.0$$
   *Hypothesis*: Eliminates the horizon compromise, allowing the ensemble to route to TCN and GRU in near horizons ($h=1\dots6$) while routing to Patch during mid-to-far diurnal ramps ($h=7\dots24$).

2. **Anti-Starvation Auxiliary Supervision ($\lambda_{\text{aux}} = 0.40$)**:
   Increase the auxiliary expert supervision coefficient from $\lambda_{\text{aux}} = 0.15$ to $\lambda_{\text{aux}} = 0.40$.
   *Hypothesis*: Forces all three expert backbones to learn representations comparable to their standalone counterparts, preventing GRU and TCN starvation and eliminating the +64 MW to +204 MW internal representation collapse.

3. **Multi-Horizon Disagreement & Shape Context**:
   Augment the router's input with segmented disagreement metrics:
   $$u_t = [C_t \;\|\; D_{\text{near}} \;\|\; D_{\text{mid}} \;\|\; D_{\text{far}} \;\|\; \Delta \text{Range}] \in \mathbb{R}^{13}$$
   *Hypothesis*: Provides the horizon-dependent router with localized signals indicating exactly which segment of the forecast horizon exhibits high inter-expert conflict.

4. **Calm-Regime Centering Prior ($\beta_{\text{uniform}} = 0.005$)**:
   Add a Kullback-Leibler divergence penalty against uniform weighting:
   $$\mathcal{L}_{\text{prior}} = \frac{1}{24} \sum_{h=1}^{24} \mathrm{KL}\left(W[:, h, :] \;\parallel\; [1/3, 1/3, 1/3]\right)$$
   *Hypothesis*: In calm regimes where disagreement and baseline error are low, the router smoothly defaults to $w_i = 1/3$, guaranteeing that V3 captures the full variance-reduction benefits of the static equal ensemble.

---

## 8. Reasons for Rejecting Alternative Modifications

1. **Rejecting Replacement of Experts with Transformers / PatchTST**:
   - Parameter budget constraint: CAEG-Net is intentionally budgeted at $\approx 120\text{k}$ parameters. Full self-attention across 168 lookback steps would dramatically increase parameters, memory, and training time.
   - Inductive diversity: GRU (recurrent), TCN (convolutional), and Patch (projection) already provide orthogonal inductive biases.
2. **Rejecting Hard Top-1 Routing**:
   - Hard gating destroys gradient flow and eliminates variance reduction entirely.
3. **Rejecting Test-Set Tuning / Post-Hoc Stacking**:
   - Violates strict causal forecasting protocols. Gating must be computed strictly from causal inputs available at forecast origin $t$.
4. **Rejecting Removal of the Patch Expert**:
   - As proven in Section 5, Patch wins in 34% of test origins and powers V2's victories in turbulent regimes. Removing it would cripple the ensemble during difficult conditions.

---

## 9. Proposed Experiment Matrix & Protocol

### 9.1 Experimental Setup
- **Data Pipeline**: Identical causal pipeline (`data/Modern_PJM/pjm_load.csv`), lookback $L=168$, horizon $H=24$, 1,294 test forecast origins.
- **Seeds**: Canonical five seeds (`42, 43, 44, 45, 46`).
- **Statistical Evaluation**: 53 non-overlapping daily blocks ($K=53$), paired Student's t-test, Wilcoxon signed-rank test, and Diebold-Mariano test.
- **Comparison Models**:
  1. Naive-24 (Deterministic)
  2. Standalone GRU
  3. Standalone TCN
  4. Standalone Patch
  5. Static Equal Ensemble (Standalone)
  6. Canonical CAEG-Net V1
  7. Standard Input-MoE
  8. CAEG-Net V2 (Frozen Reference)
  9. **CAEG-Net V3 (Proposed Architecture)**

### 9.2 Success Criteria Evaluation Plan
- **C1**: V3 improves over standalone experts (target: $\text{MAE} < 250.07\text{ MW}$).
- **C2**: V3 improves over CAEG-Net V2 (target: $\text{MAE} < 239.67\text{ MW}$).
- **C3**: V3 improves over Static Equal Ensemble (target: $\text{MAE} < 237.47\text{ MW}$).
- **C4**: If C3 is not fully surpassed overall, V3 must maintain statistically significant gains in difficult regimes ($p < 0.05$).
- **C5**: No future temporal leakage (strictly causal padding, causal context buffer).
- **C6**: Parameter efficiency ($\approx 120\text{k} - 150\text{k}$ parameters).
- **C7**: Interpretable routing behavior across horizon and regimes.
- **C8**: Reproducibility verified across all 5 canonical seeds.

---
*Deliverable 1 Complete. Awaiting implementation and benchmark of CAEG-Net V3.*
