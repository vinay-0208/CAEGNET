# Table 5: Comprehensive Phase 14 Candidate Comparison Across Five Seeds

| Candidate ID | Model Description | Parameters | PJM MAE (MW) | GEFCom MAE (kW) | UCI MAE (MW) | vs V1 (Wins) | vs Equal (Wins) | vs Best Standalone |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0_Canonical_V1** | Context-only gating baseline | 121,531 | .41 \pm 9.12$ | .88 \pm 0.25$ | .94 \pm 0.17$ | Control | 2 / 3 | 1 / 3 (PJM) |
| **F1_A1_OOF** | Causal OOF features without fallback | 121,579 | .63 \pm 5.55$ | .64 \pm 0.22$ | .98 \pm 0.20$ | 2 / 3 | 2 / 3 | 1 / 3 (PJM) |
| **F2_A2_OOF** | **Full CAEG-Net (OOF + Confidence Fallback)** | **121,724** | $\mathbf{250.97 \pm 10.69}$ | $\mathbf{12.41 \pm 0.15}$ | $\mathbf{7.74 \pm 0.30}$ | **3 / 3** | **3 / 3** | **2 / 3 (PJM, GEFCom)** |
| **F3_Confidence_Only**| Context gating with confidence fallback | 121,628 | .99 \pm 5.95$ | .44 \pm 0.21$ | $\mathbf{7.71 \pm 0.18}$ | 2 / 3 | 3 / 3 | 2 / 3 (PJM, GEFCom) |
| **F4_Smoothed_OOF** | EMA-smoothed OOF features + fallback | 121,724 | .73 \pm 5.97$ | .55 \pm 0.17$ | .02 \pm 0.50$ | 2 / 3 | 2 / 3 | 2 / 3 (PJM, GEFCom) |
| **F5_Scalar_Shrinkage**| Fixed scalar shrinkage ($\lambda=0.5$) | 121,579 | $\mathbf{246.71 \pm 6.36}$ | .66 \pm 0.23$ | .14 \pm 0.10$ | 2 / 3 | 2 / 3 | 1 / 3 (PJM) |

*Note: All values report Mean $\pm$ Population Standard Deviation (ddof=0) across 5 random seeds. F2 is the single formulation that improves over Canonical V1 and Equal Ensembling across all three benchmarks.*
