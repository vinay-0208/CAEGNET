# CAEG-Net Phase 11: Scientific Audit & Methodological Integrity Review

**Audit Date:** September 10, 2026  
**Audited Commit:** Working tree (Phase 11 completion)  
**Branch:** `research-track`  
**Auditor:** CAEG-Net Research Assistant  
**Status:** Completed & Defensively Certified  

---

## 1. Research Question & Purpose
The primary research question investigated in Phase 11 is:
> *"Under what forecasting conditions does adaptive expert gating provide a measurable advantage, and when is static fusion or a single expert better?"*

This phase was strictly diagnostic and empirical. No model parameters were optimized, no architectures modified, and no loss objectives altered.

---

## 2. Frozen Methodology Compliance
- **Canonical Model Architecture:** CAEG-Net V1 remained frozen at 121,531 parameters. The expert family remained strictly LSTM (56,152), TCN (36,952), and CNN (27,400).
- **Context Features:** Causal 4D context vector ($c = [\text{trend}, \text{volatility}, \text{lag24\_autocorr}, \text{recent\_error}]$) was retained without modifications.
- **No Test-Set Architecture Search:** Phase 10 CSV results and model code remained completely untouched.

---

## 3. Dataset Integrity & Partitioning Verification
All three benchmark datasets were verified to match the canonical research track definitions:
1. **Modern PJM Benchmark:** Bulk transmission load (MW), 168h lookback, 24h horizon, 70/15/15 chronological split ($N_{\text{train}}=5,957, N_{\text{val}}=1,294, N_{\text{test}}=1,294$).
2. **GEFCom2014:** Zonal distribution load (kW), 70/15/15 chronological split ($N_{\text{train}}=41,425, N_{\text{val}}=8,736, N_{\text{test}}=10,944$).
3. **UCI ElectricityLoadDiagrams20112014 Cohort 320:** Aggregate cohort load (MW), 2012–2014, 70/15/15 chronological split ($N_{\text{train}}=18,221, N_{\text{val}}=3,922, N_{\text{test}}=3,922$).

---

## 4. Leakage Verification & Methodology Firewall

### 4.1 Causal vs. Post-Hoc Feature Distinction
The audit verified that features were strictly segregated into two categories:
- **Forecast-Time Causal Features:** Historical load levels, lookback volatility ($\sigma(\Delta z)$), trend slopes, and lag-24 autocorrelation depend strictly on the historical $168\text{h}$ window $[t-167, \dots, t]$. Unit tests confirmed zero variation in historical features when future targets $Y$ were artificially perturbed.
- **Post-Hoc Realized Descriptors:** Target standard deviation, range, ramp magnitude, and realized expert winner were calculated strictly post-forecast and used solely for retrospective diagnostic slicing. They were never passed into the model or used as test-time selection criteria.

### 4.2 Firewall Against Test-Set Threshold Tuning
All continuous regime thresholds (volatility terciles, trend terciles, load level terciles, ramp terciles, and expert disagreement terciles) were calibrated **strictly on Train + Validation partitions**. Zero test data was inspected, optimized, or used to choose quantile cutoffs.

---

## 5. Forward-Looking Reproducibility Verification
In response to the Phase 10 reproducibility audit findings, a reusable deterministic module was implemented: [`research/deterministic.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/deterministic.py).
- Configures Python `random.seed`, `os.environ["PYTHONHASHSEED"]`, NumPy `np.random.seed`, PyTorch CPU `torch.manual_seed`, and CUDA `torch.cuda.manual_seed_all`.
- Enforces `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- Implements `make_deterministic_loader` binding a dedicated seeded `torch.Generator` to every DataLoader with shuffling.
- Unit tests in [`research/tests/test_phase11_regime_analysis.py`](file:///c:/Fall%20Semister/2026/Advanced%20Predictive%20Analytics/research/tests/test_phase11_regime_analysis.py) verified bitwise reproducibility across independent sequential executions.

---

## 6. Statistical Unit Definition & Independence Guarantee
- **Overlapping-Window Fallacy Avoided:** Sliding hourly windows ($N=1,294$ PJM, $N=10,944$ GEFCom, $N=3,922$ UCI) share 23 hours of common targets between consecutive origins, creating extreme serial correlation. Naive $t$-tests across sliding windows produce severe pseudo-replication.
- **Non-Overlapping Blocks ($K$):** Inferential hypothesis testing was conducted strictly on non-overlapping 24-hour daily blocks:
  - PJM: $K = 1,294 // 24 = 53$ independent daily blocks.
  - GEFCom: $K = 10,944 // 24 = 456$ independent daily blocks.
  - UCI: $K = 3,922 // 24 = 163$ independent daily blocks.
- **Multiplicity Adjustments:** Within each dataset family, all hypothesis tests were corrected using the step-down **Holm-Bonferroni** procedure. Both parametric paired $t$-tests and non-parametric Wilcoxon signed-rank tests were reported.

---

## 7. Explicit Negative Findings

To ensure scientific honesty and publication defensibility, the following negative findings are explicitly documented:

1. **CAEG-Net V1 Does NOT Outperform the Best Standalone Expert Overall:**
   - In all three benchmark datasets, the single best domain expert achieved a lower test MAE than CAEG-Net V1:
     - On PJM: TCN ($259.33\text{ MW}$) beat CAEG ($266.58\text{ MW}$).
     - On GEFCom: TCN ($12.57\text{ kW}$) beat CAEG ($12.99\text{ kW}$) and Equal ($12.62\text{ kW}$).
     - On UCI: LSTM ($7.79\text{ MW}$) beat CAEG ($8.05\text{ MW}$).
   - While adaptive gating improved upon static equal averaging on PJM ($-13.25\text{ MW}$) and UCI ($-0.11\text{ MW}$), it did not surpass the best standalone expert.

2. **The Expert Disagreement Hypothesis is Empirically Refuted:**
   - Prior research conjectured that adaptive gating provides the most value when expert predictions diverge significantly.
   - Empirical correlation between expert disagreement and CAEG advantage was negligible ($r \in [-0.072, +0.038]$).
   - On PJM, CAEG's statistically significant advantage was confined to the **low-disagreement regime** ($\Delta = -30.20\text{ MW}, p_{\text{adj}} = 0.00628$); under high disagreement, soft gating suffered severe misallocation ($\Delta = +32.75\text{ MW}$).

3. **Equal Ensemble Statistically Beats CAEG-Net on GEFCom:**
   - On GEFCom, the Static Equal Ensemble demonstrated a statistically significant advantage over CAEG-Net V1 ($t = +3.790, p_{\text{adj}} = 0.00119$). In environments where expert errors are strongly collinear ($r > 0.85$), uniform averaging is provably superior to soft adaptive gating.

4. **Router Inertia / Under-Allocation to Dominant Experts:**
   - On PJM, TCN won $52.6\%$ of all forecast windows, but the router allocated an average weight of only $28.8\%$ to TCN.
   - The router exhibited high entropy ($N_{\text{eff}} \approx 2.87 - 2.94$), behaving as a sluggish convex blend rather than an agile expert selector.

---

## 8. Supported Claims vs. Unsubstantiated Assertions

| Potential Claim | Audit Verdict | Permissible Scientific Phrasing |
| :--- | :---: | :--- |
| *"CAEG-Net is universally superior to individual temporal experts."* | **REJECTED (False)** | The single best individual expert outperformed CAEG-Net V1 overall across all three datasets. |
| *"High volatility causes CAEG to select TCN."* | **REJECTED (Unproven Causality)** | Higher volatility regimes were empirically associated with higher aggregate forecast error for all models; routing weights remained relatively stable across volatility terciles. |
| *"Greater expert disagreement increases the benefit of adaptive gating."* | **REJECTED (Refuted by Data)** | Expert disagreement was not positively correlated with CAEG advantage; on PJM, CAEG's benefit was statistically significant only when disagreement was low. |
| *"Lower residual error correlation between experts creates opportunity for adaptive gating to improve over uniform averaging."* | **ACCEPTED (Supported)** | Across datasets, lower pairwise residual correlation ($r \approx 0.70$ on PJM/UCI vs. $r \approx 0.87$ on GEFCom) corresponded with adaptive gating outperforming uniform averaging. |
| *"CAEG-Net provides statistically significant daily-block improvement on PJM during low-disagreement conditions."* | **ACCEPTED (Supported)** | In non-overlapping daily-block paired testing ($K=38$), CAEG-Net V1 achieved a mean daily difference of $-30.20\text{ MW}$ ($p_{\text{adj}} = 0.00628$) relative to the static equal ensemble. |

---

## 9. Limitations
1. **Single Seed Evaluation for Window Analysis:** Window-level regime metrics were generated using canonical seed 42 to enable direct per-window alignment across models and residual correlation calculations. Multi-seed finalist averages remain locked from Phase 10.
2. **Fixed Architecture Experts:** Experts were evaluated with canonical hyperparameters without dataset-specific tuning.
3. **Observational Regime Slicing:** Regime categories reflect observational partitions of real load data rather than controlled experimental interventions; observed associations must not be interpreted as causal mechanisms.

---

## 10. Certification Statement
- No Phase 10 CSV results or frozen benchmarks were modified.
- All report figures and tables are traceable directly to generated CSV artifacts.
- The methodology firewall strictly preserved train/val isolation and prevented hindsight leakage.
- All statistical tests strictly honored temporal independence via non-overlapping blocks.

**Audited by:** CAEG-Net Research Assistant  
**Date:** September 10, 2026
