# Table 6: Non-Overlapping Daily-Block Statistical Inference (F2 vs. Canonical V1)

| Dataset | Aggregation Mode | Blocks ($) | Mean Daily Diff. ($\Delta$) | 95% Confidence Interval | Paired $-stat | Raw $-value ($) | Holm-Adj. $ ($) | Wilcoxon $ | Holm-Adj. $ ($) | Cohen's $ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | 5-seed Mean Forecast | 53 | $-9.66$ MW | $[-17.07, -2.26]$ | $-2.557$ | .0135$ | **0.0406** | 470.0 | 0.0893 | $-0.351$ |
| **GEFCom2014** | 5-seed Mean Forecast | 456 | $-0.688$ kW | $[-0.802, -0.575]$ | $-11.850$ | .04 \times 10^{-28}$ | **.02 \times 10^{-27}$** | 20445.0 | **.27 \times 10^{-28}$** | $-0.555$ |
| **UCI Electricity**| 5-seed Mean Forecast | 163 | $-0.202$ MW | $[-0.310, -0.094]$ | $-3.658$ | .43 \times 10^{-4}$ | **0.0017** | 4235.0 | **.49 \times 10^{-4}$** | $-0.287$ |
| *Modern PJM* | *Single Realization (Seed 42)* | 53 | $-16.90$ MW | $[-34.72, +0.92]$ | $-1.858$ | .0688$ | 0.1376 | 659.0 | 1.0000 | $-0.255$ |
| *GEFCom2014* | *Single Realization (Seed 42)* | 456 | $-0.410$ kW | $[-0.541, -0.280]$ | $-6.157$ | .63 \times 10^{-9}$ | **.52 \times 10^{-9}$** | 34914.0 | **.16 \times 10^{-9}$** | $-0.288$ |
| *UCI Electricity*| *Single Realization (Seed 42)* | 163 | $+0.149$ MW | $[-0.045, +0.344]$ | $+1.502$ | .1350$ | 0.2896 | 5718.0 | 0.3295 | $+0.118$ |

*Note: The primary inferential estimand is the 5-seed ensemble forecast evaluated over non-overlapping 24-hour daily blocks. Under Holm-Bonferroni correction across all candidate comparisons, F2 achieves statistically significant reductions in MAE over Canonical V1 on all three benchmarks.*
