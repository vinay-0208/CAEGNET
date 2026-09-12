# CAEG-Net Faculty Review & Viva Guide

## Overview
This directory contains the authoritative faculty demonstration and defense notebook for the **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting) research project.

The notebook is explicitly designed for a **10–15 minute faculty review and oral viva**. It requires **zero model retraining** and executes in **under 10 seconds on standard CPU**.

---

## 1. File Locations
- **Authoritative Faculty Review Notebook:**  
  `research/notebooks/CAEG_Net_Faculty_Review.ipynb`
- **Mirrored Copy:**  
  `notebooks/CAEG_Net_Faculty_Review.ipynb`
- **Review Guide & Documentation:**  
  `research/notebooks/README_Faculty_Review.md`

---

## 2. Quickstart: How to Open and Run the Notebook

### Option A: VS Code (Recommended for Presentation)
1. Open the project root folder in **VS Code**.
2. Open `research/notebooks/CAEG_Net_Faculty_Review.ipynb`.
3. Select your Python kernel (`caeg-gpu` or any Python 3.10+ kernel with `numpy`, `pandas`, `matplotlib`, and `jupyter`).
4. Click **Run All** (or execute cells sequentially).

### Option B: JupyterLab / Classic Jupyter Notebook
From the repository root in PowerShell or terminal:
```powershell
jupyter lab research/notebooks/CAEG_Net_Faculty_Review.ipynb
```
or
```powershell
jupyter notebook research/notebooks/CAEG_Net_Faculty_Review.ipynb
```

### Option C: Headless In-Place Execution
To re-run all 47 cells and re-render all embedded plots and tables headlessly:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace research/notebooks/CAEG_Net_Faculty_Review.ipynb
```

---

## 3. Execution & Validation Audit
- **Total Cells:** 47 (35 rich explanatory Markdown cells, 12 executable Python code cells).
- **Execution Errors:** **0 (Zero)** — 100% of cells passed cleanly.
- **Rendered Visualizations:** 10 multi-panel figure objects covering 12+ individual charts.
- **Hardware Requirement:** Standard CPU only (no GPU needed).
- **Execution Time:** ~6 to 8 seconds.
- **Prediction Traces:** Final sample-level prediction curves were not stored on disk in repository evaluation caches; per strict scientific transparency, fake prediction traces are omitted with a clear explanatory notice in Section 19.
- **Horizon Data:** Real step-by-step ($h=1 \dots 24$) lead-time error curves are loaded and plotted directly from `phase15b_horizon_results.csv`.
- **Metric Hygiene:** MAPE was removed; only audited MAE, RMSE, and $R^2$ are presented.

---

## 4. Key Authoritative Results Summary

### Final Model: CAEG-Net F2 (121,724 Parameters)
- **LSTM Expert:** 56,152 parameters (46.1%)
- **TCN Expert:** 36,952 parameters (30.4%)
- **CNN Expert:** 27,400 parameters (22.5%)
- **Backbone Subtotal:** 120,504 parameters (98.9%)
- **Context Router:** 1,075 parameters (0.9%)
- **Confidence Head:** 145 parameters (0.1%)

### Benchmark Performance (5-Seed Mean ± Population SD, ddof=0):
- **PJM:** MAE = $250.9747 \pm 10.6938$ MW | RMSE = $335.3822$ MW | $R^2 = 0.8714$
- **GEFCom2014:** MAE = $12.4077 \pm 0.1525$ kW | RMSE = $18.0446$ kW | $R^2 = 0.8610$
- **UCI:** MAE = $7.7371 \pm 0.3037$ MW | RMSE = $10.9556$ MW | $R^2 = 0.9831$

### Baseline Standalone & Equal Ensemble Comparison:
- **PJM (MW):** F2 = 250.97 | TCN = 259.33 | Equal = 279.83 | LSTM = 291.73 | CNN = 432.08
  *(F2 strictly outperforms all individual experts and equal ensemble).*
- **GEFCom (kW):** F2 = 12.41 | TCN = 12.57 | Equal = 12.62 | LSTM = 13.23 | CNN = 14.50
  *(F2 is competitive, beating TCN and Equal; fixed shrinkage control achieved 12.36 kW).*
- **UCI (MW):** LSTM = 7.55 | F2 = 7.74 | Equal = 8.17 | TCN = 8.34 | CNN = 11.71
  *(LSTM is stronger standalone; F2 beats Equal Ensemble and TCN without single-expert collapse).*

### Scientific Interpretation for Faculty:
> *"CAEG-Net F2 provides the strongest overall balance of forecasting performance, seed stability, methodological integrity and architectural simplicity among the evaluated formulations. F2 performs strongly on PJM and UCI, while fixed shrinkage achieves a slightly lower mean MAE on GEFCom. Therefore F2 is not claimed to be universally optimal on every dataset."*

---

## 5. Visualizations Included (with Faculty Explanations)
Every major chart in the notebook includes a dedicated `### How to explain this to faculty` callout:
1. **Chart A:** Architecture Flow Diagram (`matplotlib.patches`).
2. **Chart B:** Parameter Distribution Horizontal Bar Chart ($121,724$ params).
3. **Chart C:** Executive Result Dashboard (PJM, GEFCom, UCI MAE).
4. **Charts D, E, F:** Separate Baseline Comparisons for PJM (MW), GEFCom (kW), UCI (MW).
5. **Charts G & H:** Secondary Metrics (RMSE and $R^2$ Variance Explained).
6. **Chart I:** Five-Seed Sensitivity Analysis (Relative SD / Coefficient of Variation).
7. **Chart J:** Grouped Bar Chart of Routing Weights ($w_{\text{LSTM}}, w_{\text{TCN}}, w_{\text{CNN}}$).
8. **Chart K:** Confidence Parameter $\lambda$ Error-Bar Plot (Settling near $\approx 0.51$).
9. **Chart L:** Phase 15B Controlled Mechanism Comparison (6 formulations across 3 grids).
10. **Chart M:** Step-by-Step MAE across the 24-Hour Horizon ($h=1 \dots 24$).
