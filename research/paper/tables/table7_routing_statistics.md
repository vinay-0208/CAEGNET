# Table 7: Learned Expert Routing Allocations and Confidence Parameters

| Dataset | Candidate ID | Mean {\text{LSTM}}$ | Mean {\text{TCN}}$ | Mean {\text{CNN}}$ | Mean Entropy | Effective Experts ({\text{eff}}$) | Mean $\lambda$ | $\lambda$ Standard Deviation | $\lambda$ CV (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | F0_Canonical_V1 | 0.354 | 0.362 | 0.284 | 1.085 | 2.961 | — | — | — |
| | **F2_A2_OOF** | 0.342 | 0.381 | 0.277 | 1.082 | 2.952 | 0.512 | 0.005 | 0.98% |
| **GEFCom2014** | F0_Canonical_V1 | 0.321 | 0.398 | 0.281 | 1.073 | 2.924 | — | — | — |
| | **F2_A2_OOF** | 0.301 | 0.428 | 0.271 | 1.069 | 2.912 | 0.509 | 0.004 | 0.79% |
| **UCI Electricity**| F0_Canonical_V1 | 0.412 | 0.319 | 0.269 | 1.071 | 2.918 | — | — | — |
| | **F2_A2_OOF** | 0.424 | 0.309 | 0.267 | 1.070 | 2.915 | 0.514 | 0.005 | 0.97% |

*Note: Effective number of experts is defined as {\text{eff}} = \exp(H(w))$. Theoretical maximum is 3.00. The low standard deviation and coefficient of variation ( < 1.0\%$) of $\lambda$ show that the confidence head acts as stationary variance shrinkage toward the equal prior.*
