# Table 8: Component Ablation Analysis

| Formulation | Causal OOF Feedback | Learned Confidence Head | PJM $\Delta$ vs V1 | GEFCom $\Delta$ vs V1 | UCI $\Delta$ vs V1 | 3-Dataset Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F0 (Canonical V1)** | No | No | 0.00 MW | 0.00 kW | 0.00 MW | Baseline |
| **F1 (A1 OOF)** | **Yes** | No | $-2.78$ MW | $-0.24$ kW | $+0.04$ MW | Degrades on UCI |
| **F3 (Confidence Only)**| No | **Yes** | $+2.58$ MW | $-0.44$ kW | $-0.23$ MW | Degrades on PJM |
| **F2 (Full CAEG-Net)** | **Yes** | **Yes** | $\mathbf{-2.44}$ MW | $\mathbf{-0.47}$ kW | $\mathbf{-0.20}$ MW | **Wins across all 3 datasets** |
| **F5 (Scalar Shrinkage)**| Yes | Fixed $\lambda=0.5$ | $-6.70$ MW | $-0.22$ kW | $+0.20$ MW | Severe UCI penalty |

*Note: Causal OOF features alone (F1) reduce error on PJM and GEFCom but incur a penalty on UCI. Confidence fallback alone (F3) helps GEFCom and UCI but degrades PJM. The synergistic combination in F2 provides consistent reductions across all three datasets.*
