# Phase 15: Final Comprehensive Adaptive Routing, Robustness, and Model-Lock Report
## CAEG-Net Research Program — Definitive Publication Synthesis

**Document**: `research/reports/phase15_comprehensive_final_report.md`  
**Date**: September 2026  
**Status**: EXPERIMENTS COMPLETE — MODEL DEVELOPMENT LOCKED  
**Certified Final Model**: **CAEG-Net** (Internal ID: `F2 / A2-OOF`, `ConfidenceFallbackCAEGNet`, $121,724$ parameters)  

---

## 1. Executive Summary & Research Question Resolution

Phase 15 executed the final controlled model-development campaign for **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting). The primary research question investigated was:

> *"Can richer causally available information about context, expert performance, expert disagreement, forecast horizon, and forecasting difficulty make the CAEG-Net routing meaningfully more adaptive and/or more accurate without sacrificing robustness?"*

### Key Empirical Findings Across Phase 15 Hypotheses:
1. **15-A (Dynamic Performance-Aware Routing):**  
   Conditioning routing weights on causally tracked out-of-fold historical performance residuals (`A2-OOF`) provides a consistent, regularizing inductive bias that outperforms unregularized context gating (`V1`). However, forcing weights to undergo rapid temporal switching without confidence stabilization inflates variance and degrades hold-out forecasting accuracy.
2. **15-B (Horizon-Aware Routing $W_t \in \mathbb{R}^{24 	imes 3}$):**  
   Assigning independent or grouped routing weights across lead steps $h=1 \dots 24$ (`Control_C`, `Candidate_E1`, `Candidate_E2`) consistently degraded validation and test performance across all three benchmarks (PJM MAE degraded from $250.97$ to $257.55$ MW; GEFCom from $12.41$ to $12.86$ kW; UCI from $7.74$ to $8.13$ MW). Retrospective horizon specialization differences between LSTM and TCN do not generalize to unconstrained step-wise routing. Global context routing remains superior.
3. **15-C (Enhanced Causal Context):**  
   Expanding context lookback beyond 168 hours or introducing multi-scale ramp features failed to surpass the compact 4D physical context vector combined with OOF performance residuals at validation screening.
4. **15-D (Expert Complementarity & CNN Control):**  
   Ablations in Phase 5B and Phase 15 confirmed that while standalone CNN MAE can be varied through pooling and kernel depth, the canonical CNN ($27,400$ params) provides the essential localized high-frequency edge-detection bias that complements LSTM and TCN during steep morning ramps.
5. **15-E (Training Objective Robustness):**  
   Comparing MSE, MAE, and Huber losses confirmed that MSE on standardized targets provides the smoothest gradient surfaces and yields the lowest physical-scale test MAE after inverse scaling.
6. **15-F (Adaptive Stability / Confidence Head):**  
   The learned confidence blend parameter $\lambda$ settles near $pprox 0.51$ across all three grids ($CV pprox 0.6\% - 1.1\%$). It behaves primarily as an empirical regularizer anchoring predictions toward the robust equal-expert centroid rather than a wild binary switch.
7. **15-G (Uncertainty Extensions):**  
   Formally documented as future work to maintain absolute integrity of the deterministic point forecasting benchmark.

---

## 2. Staged Screening Firewall Audit

To conserve compute and eliminate multiple-testing bias, Phase 15 strictly applied **Staged Screening**:
- **Stage 15-S:** Screened compact individual variants on the validation set across Seeds 42 and 123.
- **Validation Gate:** Only candidates improving $\ge 2$ of 3 datasets with $\le 2.0\%$ worst degradation were eligible to qualify.
- **Firewall Verdict:** All exploratory candidates failed validation qualification. Control A (`F2 / A2-OOF`) was the sole qualified anchor.
- **Exploratory Test Diagnostics:** Candidates B, C, D, E1, and E2 were evaluated on held-out test data strictly as exploratory post-screening diagnostics. In accordance with pre-registered protocol, no test result was used to retroactively alter candidate qualification.

---

## 3. Final Multi-Grid Benchmark Verification

| Benchmark Grid | Model Name | Internal ID | Parameters | Primary Test MAE | Test RMSE | Test $R^2$ | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PJM (MW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **250.9747 ± 10.6938** | **335.3822** | **0.8714** | **CHAMPION_LOCKED** |
| PJM (MW) | TCN Expert | TCN_Standalone | 36,952 | 259.3264 | — | — | Baseline |
| PJM (MW) | Equal Ensemble | Static_Equal | 120,504 | 279.8282 | — | — | Baseline |
| PJM (MW) | LSTM Expert | LSTM_Standalone | 56,152 | 291.7300 | — | — | Baseline |
| PJM (MW) | CNN Expert | CNN_Standalone | 27,400 | 432.0800 | — | — | Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **GEFCom (kW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **12.4077 ± 0.1525** | **18.0446** | **0.8610** | **CHAMPION_LOCKED** |
| GEFCom (kW) | Fixed Shrinkage Control | Control_B | 121,579 | 12.3607 ± 0.1841 | 18.0181 | 0.8614 | Exploratory Diagnostic |
| GEFCom (kW) | TCN Expert | TCN_Standalone | 36,952 | 12.5729 | — | — | Baseline |
| GEFCom (kW) | Equal Ensemble | Static_Equal | 120,504 | 12.6248 | — | — | Baseline |
| GEFCom (kW) | LSTM Expert | LSTM_Standalone | 56,152 | 13.2300 | — | — | Baseline |
| GEFCom (kW) | CNN Expert | CNN_Standalone | 27,400 | 14.5000 | — | — | Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **UCI (MW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **7.7371 ± 0.3037** | **10.9556** | **0.9831** | **CHAMPION_LOCKED** |
| UCI (MW) | LSTM Expert | LSTM_Standalone | 56,152 | 7.5542 | — | — | Baseline (Best Standalone) |
| UCI (MW) | Equal Ensemble | Static_Equal | 120,504 | 8.1675 | — | — | Baseline |
| UCI (MW) | TCN Expert | TCN_Standalone | 36,952 | 8.3400 | — | — | Baseline |
| UCI (MW) | CNN Expert | CNN_Standalone | 27,400 | 11.7100 | — | — | Baseline |

---

## 4. Final Scientific Interpretation

1. **Definitive Decision:** **Outcome B certified.** CAEG-Net F2 (`ConfidenceFallbackCAEGNet`) is locked as the final publication model.
2. **Balanced Cross-Grid Advantage:**  
   CAEG-Net F2 provides the strongest overall balance of forecasting performance, seed stability, methodological integrity, and architectural simplicity among all evaluated formulations.
3. **Absence of Universal Dominance:**  
   F2 strictly outperforms all individual standalone models and the equal ensemble on PJM and GEFCom. On UCI, standalone LSTM is slightly lower in MAE (7.55 MW vs 7.74 MW); however, F2 maintains multi-expert robustness without single-architecture collapse. F2 is therefore not claimed to be universally optimal on every individual dataset.
4. **Closeout of Architecture Optimization:**  
   All architectural optimization is formally closed. The repository is ready for final publication packaging.
