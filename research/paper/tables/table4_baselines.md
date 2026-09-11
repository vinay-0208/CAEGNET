# Table 4: Standalone Experts and Conventional Baselines Performance

| Model Architecture | Parameters | Modern PJM (MW) | GEFCom2014 (kW) | UCI Cohort 320 (MW) |
| :--- | :---: | :---: | :---: | :---: |
| **Static Equal Ensemble** | 120,504 | 279.83 | 12.62 | 8.17 |
| **Standalone LSTM** | 50,264 | 285.42 | 13.12 | 7.55 (Val BL) / 7.79 (Test BM) |
| **Standalone TCN** | 35,800 | **259.33** | **12.57** | 8.42 |
| **Standalone CNN** | 34,440 | 294.15 | 13.45 | 8.85 |
| **Canonical V1 (F0)** | 121,531 | .41 \pm 9.12$ | .88 \pm 0.25$ | .94 \pm 0.17$ |
| **Final CAEG-Net (F2)** | **121,724** | **250.97** $\pm$ **10.69** | **12.41** $\pm$ **0.15** | **7.74** $\pm$ **0.30** |

*Note: For UCI, the locked Phase 14 validation baseline for standalone LSTM is 7.55 MW, whereas the test benchmark realization is 7.79 MW. CAEG-Net F2 beats the standalone experts on 2 out of 3 datasets (Modern PJM and GEFCom2014).*
