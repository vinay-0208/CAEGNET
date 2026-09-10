# Phase 11: Regime, Difficulty & Expert-Complementarity Analysis

**Research Track:** Canonical CAEG-Net V1  
**Author:** CAEG-Net Research Assistant  
**Date:** September 10, 2026  
**Status:** Completed, Verified & Frozen  

---

## 1. Executive Summary & Core Research Question

### Primary Research Question:
> *Under what forecasting conditions does adaptive expert gating provide a measurable advantage, and when is static fusion or a single expert better?*

### Key Empirical Findings:
1. **Adaptive Gating vs. Static Equal Ensemble:**
   - On **Modern PJM** (bulk transmission), CAEG-Net V1 achieved a lower test MAE than the Static Equal Ensemble ($266.58$ vs. $279.83\text{ MW}$, $\Delta = -13.25\text{ MW}$, $-4.74\%$), winning $60.4\%$ of all $24\text{h}$ forecast windows. In non-overlapping daily-block paired testing ($K=53$), the overall difference did not reach statistical significance ($t = -1.548, p = 0.128$, Wilcoxon $p = 0.041$).
   - On **UCI Cohort 320** (consumer aggregation), CAEG-Net V1 achieved a slightly lower test MAE than the Equal Ensemble ($8.05$ vs. $8.17\text{ MW}$, $\Delta = -0.11\text{ MW}$, $-1.39\%$), winning $57.3\%$ of windows ($K=163, t = -1.246, p = 0.215$).
   - On **GEFCom2014** (zonal distribution), the **Static Equal Ensemble outperformed CAEG-Net V1** ($12.62$ vs. $12.99\text{ kW}$, $\Delta = +0.36\text{ kW}$, $+2.87\%$), statistically favoring uniform averaging ($K=456, t = +3.790, p_{\text{adj}} = 0.00119$).

2. **Adaptive Gating vs. Best Individual Expert (Major Negative Finding):**
   - In **all three benchmark datasets**, the single best specialized temporal expert outperformed CAEG-Net V1 overall:
     - On **PJM**, standalone **TCN** achieved $259.33\text{ MW}$ (vs. CAEG $266.58\text{ MW}$).
     - On **GEFCom**, standalone **TCN** achieved $12.57\text{ kW}$ (vs. CAEG $12.99\text{ kW}$).
     - On **UCI**, standalone **LSTM** achieved $7.79\text{ MW}$ (vs. CAEG $8.05\text{ MW}$).
   - CAEG-Net V1 won against the best individual expert in only $41.3\%$ (PJM), $22.9\%$ (GEFCom), and $38.9\%$ (UCI) of forecast windows.

3. **Expert Disagreement Hypothesis (Empirically Refuted):**
   - The hypothesis that *"greater expert disagreement increases the utility of adaptive gating"* is **not supported by the empirical data**.
   - Correlations between expert prediction disagreement and CAEG's relative advantage ($\Delta_{\text{Equal}}$) were negligible ($r = -0.072$ on PJM, $r = +0.038$ on GEFCom, $r = +0.012$ on UCI).
   - On PJM, CAEG's significant advantage occurred in the **low-disagreement regime** ($\Delta = -30.20\text{ MW}, p_{\text{adj}} = 0.00628$), whereas under high disagreement, gating blended divergent expert errors poorly ($\Delta = +32.75\text{ MW}$).

4. **Expert Error Residual Correlation (Empirically Supported):**
   - Across datasets, the relative performance of adaptive fusion vs. uniform averaging strongly corresponds to the degree of residual error collinearity:
     - When expert residuals exhibited lower correlation (PJM mean $r = 0.703$; UCI mean $r = 0.734$), adaptive gating outperformed static equal averaging ($-4.74\%$ and $-1.39\%$).
     - When expert residuals were highly collinear (GEFCom mean $r = 0.871$), uniform equal averaging outperformed adaptive gating ($+2.87\%$).

---

## 2. Methodology & Leakage Firewall

### 2.1 Frozen Architecture & Tri-Benchmark Datasets
- **Canonical Model:** CAEG-Net V1 (121,531 parameters: LSTM 56,152, TCN 36,952, CNN 27,400, Context Encoder 384, Router 643).
- **Benchmark Partitions:** Strict chronological splits ($70\%$ train, $15\%$ validation, $15\%$ test) on Modern PJM, GEFCom2014, and UCI Cohort 320 Aggregate.
- **Reproducibility Utility:** Deterministic seeding via `research/deterministic.py` enforcing PyTorch CPU/CUDA RNGs, NumPy RNG, Python hash seeds, and `torch.backends.cudnn.deterministic = True`.

### 2.2 Feature Categorization & Zero-Leakage Protocol
To strictly prevent hindsight leakage, features were divided into two explicit categories:
1. **Forecast-Time Features (Available Causal Context):**
   - Historical lookback load level (mean, median, max, min over 168h).
   - Historical volatility: standard deviation of first differences $\Delta z$ over 168h lookback.
   - Historical trend slope: linear slope over trailing 24h, 48h, and 168h.
   - Lag-24 autocorrelation and causal recent forecast error.
2. **Post-Hoc Realized Features (Descriptive Explanations ONLY):**
   - Realized target standard deviation, range, CoV, ramp magnitude, and max hourly change.
   - Realized best expert identity.
   - *Firewall Guarantee:* Post-hoc features were evaluated strictly after forecast generation and never used as inputs to the model or as test-time selection criteria.

### 2.3 Train/Validation Regime Threshold Calibration
All continuous difficulty, volatility, load, ramp, and disagreement thresholds were derived strictly as terciles ($33.3\%$ and $66.7\%$ quantiles) on **Train + Validation partitions**. Zero test data was used to select or optimize thresholds.

| Threshold Category | Modern PJM (MW) | GEFCom2014 (kW) | UCI Cohort 320 (MW) |
| :--- | :---: | :---: | :---: |
| **Historical Volatility (Terciles)** | $[164.30, 185.10]$ | $[9.10, 11.29]$ | $[17.59, 23.59]$ |
| **Historical Trend Slope (Terciles)**| $[-2.39, +2.68]$ | $[-0.07, +0.09]$ | $[-0.06, +0.07]$ |
| **Mean Load Level (Terciles)** | $[5222.40, 5904.27]$ | $[126.13, 161.96]$ | $[181.57, 207.46]$ |
| **Realized Ramp Magnitude (Terciles)**| $[192.67, 464.83]$ | $[8.20, 21.00]$ | $[6.99, 18.28]$ |
| **Expert Disagreement (Terciles)** | $[293.89, 400.89]$ | $[6.40, 8.92]$ | $[4.76, 6.71]$ |

---

## 3. Dataset-Level Summary & Model Comparison

Across all test windows ($N=1,294$ PJM, $N=10,944$ GEFCom, $N=3,922$ UCI):

| Dataset | Metric Unit | CAEG V1 MAE | Equal Ens MAE | LSTM MAE | TCN MAE | CNN MAE | $\Delta_{\text{Equal}}$ | Relative $\Delta$ (%) | CAEG Win vs Equal | CAEG Win vs Best |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | MW | $266.58$ | $279.83$ | $287.84$ | $\mathbf{259.33}$ | $411.97$ | $\mathbf{-13.25}$ | $\mathbf{-4.74\%}$ | $\mathbf{60.4\%}$ | $41.3\%$ |
| **GEFCom2014** | kW | $12.99$ | $\mathbf{12.62}$ | $13.33$ | $\mathbf{12.57}$ | $14.33$ | $+0.36$ | $+2.87\%$ | $45.0\%$ | $22.9\%$ |
| **UCI Cohort 320**| MW | $8.05$ | $8.17$ | $\mathbf{7.79}$ | $8.57$ | $11.66$ | $\mathbf{-0.11}$ | $\mathbf{-1.39\%}$ | $\mathbf{57.3\%}$ | $38.9\%$ |

*(Note: $\Delta_{\text{Equal}} = \text{MAE}_{\text{CAEG}} - \text{MAE}_{\text{Equal}}$. Negative values favor CAEG-Net V1. Source: `research/results/phase11_dataset_summary.csv`)*

---

## 4. Non-Overlapping Daily-Block Statistical Hypothesis Testing

To avoid artificial degrees-of-freedom inflation from overlapping sliding hourly windows, inferential testing was conducted strictly on non-overlapping 24-hour daily blocks ($K=53$ PJM, $K=456$ GEFCom, $K=163$ UCI). Step-down Holm-Bonferroni correction was applied within each dataset family.

| Dataset | Evaluation Subset | $K$ Blocks | Mean Daily Diff | 95% CI | $t$-statistic | Raw $p$ ($t$-test) | Holm $p_{\text{adj}}$ ($t$) | Wilcoxon $p$ | Holm $p_{\text{adj}}$ (W) | Cohen's $d_z$ | Statistical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PJM** | **Overall Test Set** | 53 | $-13.91\text{ MW}$ | $[-31.52, +3.71]$ | $-1.548$ | $0.1278$ | $0.3834$ | $0.0413$ | $0.1239$ | $-0.213$ | Inconclusive ($p > 0.05$ after Holm) |
| PJM | Volatility: High | 49 | $-12.82\text{ MW}$ | $[-31.73, +6.08]$ | $-1.330$ | $0.1899$ | $0.3834$ | $0.0817$ | $0.1634$ | $-0.190$ | Not statistically significant |
| PJM | **Disagreement: Low** | 38 | $\mathbf{-30.20\text{ MW}}$ | $[-47.54, -12.86]$ | $\mathbf{-3.413}$ | $\mathbf{0.0016}$ | $\mathbf{0.0063}$ | $\mathbf{0.0004}$ | $\mathbf{0.0015}$ | $\mathbf{-0.554}$ | **Statistically favors CAEG ($p < 0.01$)** |
| PJM | Disagreement: High | 11 | $+32.75\text{ MW}$ | $[-10.32, +75.81]$ | $+1.490$ | $0.1670$ | $0.3834$ | $0.2402$ | $0.2402$ | $+0.449$ | Not statistically significant |
| **GEFCom** | **Overall Test Set** | 456 | $\mathbf{+0.36\text{ kW}}$ | $[+0.17, +0.55]$ | $\mathbf{+3.790}$ | $\mathbf{0.0002}$ | $\mathbf{0.0012}$ | $\mathbf{0.0002}$ | $\mathbf{0.0011}$ | $\mathbf{+0.178}$ | **Statistically favors Equal ($p < 0.01$)** |
| GEFCom | Volatility: Low | 195 | $+0.33\text{ kW}$ | $[+0.14, +0.51]$ | $+3.510$ | $0.0006$ | $0.0032$ | $0.0003$ | $0.0016$ | $+0.251$ | Statistically favors Equal ($p < 0.01$) |
| GEFCom | Disagreement: Low | 205 | $+0.31\text{ kW}$ | $[+0.14, +0.49]$ | $+3.519$ | $0.0005$ | $0.0032$ | $0.0005$ | $0.0026$ | $+0.246$ | Statistically favors Equal ($p < 0.01$) |
| GEFCom | Disagreement: High | 138 | $+0.65\text{ kW}$ | $[+0.16, +1.14]$ | $+2.585$ | $0.0108$ | $0.0431$ | $0.0242$ | $0.0968$ | $+0.220$ | Statistically favors Equal ($p < 0.05$) |
| **UCI** | **Overall Test Set** | 163 | $-0.11\text{ MW}$ | $[-0.28, +0.06]$ | $-1.246$ | $0.2145$ | $0.8580$ | $0.1186$ | $0.5928$ | $-0.098$ | Inconclusive ($p > 0.05$) |
| UCI | Volatility: Low | 37 | $-0.29\text{ MW}$ | $[-0.59, +0.00]$ | $-1.933$ | $0.0611$ | $0.3665$ | $0.0653$ | $0.3915$ | $-0.318$ | Marginal trend favors CAEG ($p \approx 0.06$) |
| UCI | Disagreement: Medium | 32 | $-0.34\text{ MW}$ | $[-0.67, -0.02]$ | $-2.063$ | $0.0475$ | $0.3328$ | $0.0057$ | $0.0400$ | $-0.365$ | Statistically favors CAEG in Wilcoxon |

---

## 5. Detailed Findings for Core Analyses A–F

### Core Analysis A: Forecasting Difficulty Regimes
- **Volatility Regimes:** On PJM, CAEG's largest relative reduction in error occurred in medium volatility ($\Delta = -21.09\text{ MW}$, $60.2\%$ win rate) and high volatility ($\Delta = -12.77\text{ MW}$, $60.6\%$ win rate). On UCI, CAEG provided its greatest advantage under low volatility ($\Delta = -0.34\text{ MW}$, $63.5\%$ win rate). On GEFCom, the Equal Ensemble maintained its advantage across all volatility terciles ($+0.32\text{ kW}$ low, $+0.38\text{ kW}$ med, $+0.42\text{ kW}$ high).
- **Ramp Regimes (Realized Dynamics):** On PJM, CAEG achieved its largest absolute improvement during high ramp periods ($\Delta = -28.23\text{ MW}$, $63.8\%$ win rate). On UCI, high ramp periods similarly produced the largest margin over the equal ensemble ($\Delta = -0.25\text{ MW}$, $58.1\%$ win rate).

### Core Analysis B: Expert Disagreement Dynamics
- Binned analysis demonstrated that expert prediction disagreement does not linearly scale CAEG's advantage.
- On Modern PJM, CAEG achieved its strongest performance when disagreement was low to moderate ($\le 400\text{ MW}$). Under extreme disagreement, the soft router failed to isolate the correct expert, yielding an average degradation of $+32.75\text{ MW}$ in daily blocks.
- On GEFCom, high expert disagreement widened the gap in favor of the Static Equal Ensemble ($+0.65\text{ kW}$ difference).

### Core Analysis C: Residual Correlation vs. Fusion Advantage
Evaluating pairwise residual correlations across non-overlapping blocks confirmed that expert error collinearity dictates fusion efficacy:
- **Modern PJM:** $\text{Corr}(\text{LSTM}, \text{TCN}) = 0.803$, $\text{Corr}(\text{LSTM}, \text{CNN}) = 0.634$, $\text{Corr}(\text{TCN}, \text{CNN}) = 0.671$. Diversity in CNN errors allowed adaptive gating to outperform uniform averaging ($-4.74\%$).
- **UCI Cohort 320:** $\text{Corr}(\text{LSTM}, \text{TCN}) = 0.848$, $\text{Corr}(\text{LSTM}, \text{CNN}) = 0.676$, $\text{Corr}(\text{TCN}, \text{CNN}) = 0.679$. Moderate error diversity supported adaptive weighting towards LSTM ($-1.39\%$).
- **GEFCom2014:** $\text{Corr}(\text{LSTM}, \text{TCN}) = 0.897$, $\text{Corr}(\text{LSTM}, \text{CNN}) = 0.852$, $\text{Corr}(\text{TCN}, \text{CNN}) = 0.864$. The three experts made highly collinear errors; with no unique orthogonal signal to exploit, adaptive weighting added parameter noise, allowing uniform averaging to prevail ($+2.87\%$).

### Core Analysis D: Realized Expert Win Rates vs. Router Allocation
- **PJM:** TCN was the best individual expert on $52.6\%$ of test windows, LSTM on $36.1\%$, and CNN on $11.4\%$. However, the CAEG router allocated an average weight of only $28.8\%$ to TCN and $43.3\%$ to LSTM when TCN won. The router was structurally sluggish in recognizing TCN's dominance.
- **UCI:** LSTM was the best expert on $62.9\%$ of test windows. Here, the router effectively tracked this preference, allocating an average weight of $45.4\%$ to LSTM.
- **GEFCom:** TCN won $44.2\%$ of windows and LSTM won $32.1\%$; however, the router assigned $43.7\%$ weight to LSTM and only $26.2\%$ to TCN.

### Core Analysis E & F: Diurnal, Weekly, and Peak/Off-Peak Regimes
- **Diurnal Regimes:**
  - On PJM, CAEG delivered its largest gains during the **morning** ($\Delta = -51.77\text{ MW}$, $73.8\%$ win rate) and **afternoon** ($\Delta = -17.17\text{ MW}$, $67.6\%$ win rate). However, CAEG suffered significant degradation during the **evening ramp/peak** ($\Delta = +32.14\text{ MW}$, $45.7\%$ win rate), where the equal ensemble was superior.
  - On UCI, morning ($-0.33\text{ MW}$) and afternoon ($-0.33\text{ MW}$) favored CAEG, while evening ($+0.18\text{ MW}$) favored the equal ensemble.
- **Load Regimes (Peak vs. Off-Peak):**
  - On PJM, CAEG was superior in **off-peak** ($\Delta = -25.51\text{ MW}$, $63.0\%$ win rate) and **normal load** ($\Delta = -26.91\text{ MW}$, $64.4\%$ win rate), but degraded during **peak load** ($\Delta = +37.94\text{ MW}$).
  - On UCI, CAEG's advantage was concentrated in **off-peak** ($\Delta = -0.44\text{ MW}$, $68.4\%$ win rate).
- **Weekly Regimes:**
  - On PJM, CAEG's advantage was concentrated on **weekdays** ($\Delta = -16.82\text{ MW}$, $63.8\%$ win rate), dropping to near-parity on **weekends** ($\Delta = -3.09\text{ MW}$, $50.9\%$ win rate).

---

## 6. Cross-Dataset Synthesis

| Investigation Dimension | Modern PJM (Transmission) | GEFCom2014 (Sub-Station) | UCI Cohort 320 (Consumer Cohort) | Cross-Dataset Synthesis |
| :--- | :--- | :--- | :--- | :--- |
| **CAEG vs. Equal Ensemble** | Favors CAEG ($-4.74\%$, win $60.4\%$) | Favors Equal ($+2.87\%$, win $45.0\%$) | Favors CAEG ($-1.39\%$, win $57.3\%$) | Adaptive gating beats Equal only when expert error correlation is $< 0.80$. |
| **CAEG vs. Best Single Expert** | Favors TCN ($259.3$ vs $266.6\text{ MW}$) | Favors TCN ($12.57$ vs $12.99\text{ kW}$) | Favors LSTM ($7.79$ vs $8.05\text{ MW}$) | **In no dataset does CAEG beat the best standalone expert overall.** |
| **Expert Disagreement Effect** | Gains in low disagreement; loses in high | Equal beats CAEG across all terciles | Modest gains across all terciles | Disagreement does not predict adaptive advantage; high disagreement degrades gating. |
| **Expert Error Correlation** | Moderate ($r \approx 0.703$) | Collinear ($r \approx 0.871$) | Moderate ($r \approx 0.734$) | Collinear residuals eliminate adaptive gating utility. |
| **Diurnal Regime Pattern** | Strong morning/afternoon; weak evening | Uniform equal ensemble wins all hours | Strong morning/afternoon; weak evening | Morning load transitions favor adaptive weighting; evening peaks favor static averaging. |
| **Load Regime Pattern** | Off-peak & normal favor CAEG; peak hurts | Equal ensemble wins all load regimes | Off-peak strongly favors CAEG | Off-peak baseload favors adaptive blend; peak volatility causes gating misallocation. |

---

## 7. Artifact & Figure Index

| Artifact Filename | File Path | Description |
| :--- | :--- | :--- |
| `phase11_window_metrics.csv` | `research/results/phase11_window_metrics.csv` | Full $16,160$-row dataset of window-level errors, features, and weights |
| `phase11_regime_summary.csv` | `research/results/phase11_regime_summary.csv` | Performance metrics aggregated by 7 regime categories across datasets |
| `phase11_expert_complementarity.csv` | `research/results/phase11_expert_complementarity.csv` | Disagreement correlations, expert win rates, and non-overlapping block residuals |
| `phase11_routing_by_regime.csv` | `research/results/phase11_routing_by_regime.csv` | Mean expert weights, entropy, and effective expert counts by regime |
| `phase11_statistical_tests.csv` | `research/results/phase11_statistical_tests.csv` | Inferential daily-block tests ($K$), Holm-adjusted $p$-values, and effect sizes |
| `phase11_dataset_summary.csv` | `research/results/phase11_dataset_summary.csv` | High-level dataset metrics, win rates, and relative percentage changes |
| Figure 1 | `research/results/phase11_plots/phase11_01_error_by_volatility.png` | Forecast MAE across historical volatility terciles |
| Figure 2 | `research/results/phase11_plots/phase11_02_advantage_vs_disagreement.png` | Relative fusion advantage ($\Delta_{\text{Equal}}$) vs. expert prediction disagreement |
| Figure 3 | `research/results/phase11_plots/phase11_03_routing_vs_volatility.png` | Gating weights allocation across volatility regimes |
| Figure 4 | `research/results/phase11_plots/phase11_04_routing_vs_disagreement.png` | Gating weights allocation across expert disagreement levels |
| Figure 5 | `research/results/phase11_plots/phase11_05_difficulty_regimes.png` | Forecast MAE across realized ramp magnitude regimes |
| Figure 6 | `research/results/phase11_plots/phase11_06_expert_win_distribution.png` | Realized expert win frequencies across datasets |
| Figure 7 | `research/results/phase11_plots/phase11_07_dataset_summary_advantage.png` | Win rate percentages vs. Equal Ensemble and Best Expert |
| Figure 8 | `research/results/phase11_plots/phase11_08_residual_corr_vs_fusion.png` | Pairwise residual correlation vs. relative fusion advantage |

---
**End of Phase 11 Regime & Expert-Complementarity Report**
