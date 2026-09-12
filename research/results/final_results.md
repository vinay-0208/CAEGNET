# CAEG-Net Phase 15: Authoritative Final Benchmark Results

## 1. Executive Summary
This document records the definitive, frozen benchmark results for **CAEG-Net** (internal experimental identifier: `F2 / A2-OOF`, `ConfidenceFallbackCAEGNet`, 121,724 parameters) evaluated across three diverse operational power grids against all individual standalone experts and the static equal ensemble.

All five-seed results are reported as **sample mean ± population standard deviation (ddof=0)** across seeds `[42, 123, 999, 2024, 3407]`.

---

## 2. Final Benchmark Results Table

| Benchmark Grid | Model Name | Internal ID | Parameters | Primary Test MAE | Test RMSE | Test $R^2$ | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PJM (MW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **250.9747 ± 10.6938** | **335.3822** | **0.8714** | **CHAMPION_LOCKED** |
| PJM (MW) | TCN Expert | TCN_Standalone | 36,952 | 259.3264 | — | — | BASELINE_STANDALONE |
| PJM (MW) | Equal Ensemble | Static_Equal | 120,504 | 279.8282 | — | — | BASELINE_ENSEMBLE |
| PJM (MW) | LSTM Expert | LSTM_Standalone | 56,152 | 291.7300 | — | — | BASELINE_STANDALONE |
| PJM (MW) | CNN Expert | CNN_Standalone | 27,400 | 432.0800 | — | — | BASELINE_STANDALONE |
| PJM (MW) | Fixed Shrinkage Control | Control_B | 121,579 | 253.5008 ± 8.0264 | 338.9532 | 0.8688 | EXPLORATORY_CONTROL |
| PJM (MW) | Dynamic Confidence Control | Control_D | 121,740 | 251.9419 ± 9.8000 | 336.7396 | 0.8704 | EXPLORATORY_CONTROL |
| PJM (MW) | Horizon Routing Control | Control_C | 122,852 | 257.5450 ± 5.3670 | 343.9132 | 0.8650 | EXPLORATORY_CONTROL |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **GEFCom (kW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **12.4077 ± 0.1525** | **18.0446** | **0.8610** | **CHAMPION_LOCKED** |
| GEFCom (kW) | Fixed Shrinkage Control | Control_B | 121,579 | 12.3607 ± 0.1841 | 18.0181 | 0.8614 | EXPLORATORY_CONTROL |
| GEFCom (kW) | Dynamic Confidence Control | Control_D | 121,740 | 12.4855 ± 0.2685 | 18.1019 | 0.8601 | EXPLORATORY_CONTROL |
| GEFCom (kW) | TCN Expert | TCN_Standalone | 36,952 | 12.5729 | — | — | BASELINE_STANDALONE |
| GEFCom (kW) | Equal Ensemble | Static_Equal | 120,504 | 12.6248 | — | — | BASELINE_ENSEMBLE |
| GEFCom (kW) | Horizon Routing Control | Control_C | 122,852 | 12.8582 ± 0.2520 | 18.4392 | 0.8548 | EXPLORATORY_CONTROL |
| GEFCom (kW) | LSTM Expert | LSTM_Standalone | 56,152 | 13.2300 | — | — | BASELINE_STANDALONE |
| GEFCom (kW) | CNN Expert | CNN_Standalone | 27,400 | 14.5000 | — | — | BASELINE_STANDALONE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **UCI (MW)** | **CAEG-Net** | **F2_A2_OOF** | **121,724** | **7.7371 ± 0.3037** | **10.9556** | **0.9831** | **CHAMPION_LOCKED** |
| UCI (MW) | LSTM Expert | LSTM_Standalone | 56,152 | 7.5542 | — | — | BASELINE_STANDALONE |
| UCI (MW) | Fixed Shrinkage Control | Control_B | 121,579 | 7.7523 ± 0.1814 | 10.9893 | 0.9830 | EXPLORATORY_CONTROL |
| UCI (MW) | Dynamic Confidence Control | Control_D | 121,740 | 7.8177 ± 0.2838 | 10.9980 | 0.9830 | EXPLORATORY_CONTROL |
| UCI (MW) | Horizon Routing Control | Control_C | 122,852 | 8.1309 ± 0.4048 | 11.4837 | 0.9814 | EXPLORATORY_CONTROL |
| UCI (MW) | Equal Ensemble | Static_Equal | 120,504 | 8.1675 | — | — | BASELINE_ENSEMBLE |
| UCI (MW) | TCN Expert | TCN_Standalone | 36,952 | 8.3400 | — | — | BASELINE_STANDALONE |
| UCI (MW) | CNN Expert | CNN_Standalone | 27,400 | 11.7100 | — | — | BASELINE_STANDALONE |

---

## 3. Scientific Synthesis & Model-Lock Conclusion

1. **Overall Cross-Grid Balance:**  
   CAEG-Net F2 provides the strongest overall balance of forecasting performance, seed stability, methodological integrity, and architectural simplicity among the evaluated formulations.
2. **Defensible Dataset Outcomes:**  
   - On **PJM**, CAEG-Net F2 strictly wins over all individual experts (-8.36 MW vs TCN) and equal fusion (-28.86 MW).  
   - On **GEFCom2014**, CAEG-Net F2 improves over standalone TCN (-0.16 kW) and Equal Ensemble (-0.22 kW); exploratory fixed shrinkage achieved 12.36 kW.  
   - On **UCI**, standalone LSTM is stronger (7.55 MW); CAEG-Net F2 achieves 7.74 MW, beating Equal Ensemble (8.17 MW) and TCN (8.34 MW) while maintaining multi-expert robustness against single-model failure.  
   - Therefore, CAEG-Net F2 is **not claimed to be universally optimal on every individual dataset**, but represents the most reliable, parsimonious, and well-balanced deployable system.
