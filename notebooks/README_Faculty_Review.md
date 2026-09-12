# CAEG-Net Faculty Review & Viva Guide

## Overview
This directory contains the faculty demonstration and defense notebook for the **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting) research project.

The notebook is explicitly designed for a **10–15 minute faculty review and oral viva**. It requires **zero model training from scratch** and executes in **under 10 seconds on any standard CPU**.

---

## 1. File Locations
- **Primary Faculty Review Notebook:**
  `research/notebooks/CAEG_Net_Faculty_Review.ipynb`
- **Top-Level Mirror (for convenience):**
  `notebooks/CAEG_Net_Faculty_Review.ipynb`
- **Review Guide & Documentation:**
  `research/notebooks/README_Faculty_Review.md`

---

## 2. Quickstart: How to Open and Run the Notebook

### Option A: VS Code (Recommended for Presentation)
1. Open the repository root folder (`Advanced Predictive Analytics`) in **VS Code**.
2. In the file explorer, navigate to `research/notebooks/CAEG_Net_Faculty_Review.ipynb`.
3. In the top right corner of the notebook editor, select your Python kernel:
   - Default Conda environment: `caeg-gpu` or any standard Python 3.10+ kernel with `numpy`, `pandas`, `matplotlib`, and `jupyter`.
4. Click **Run All** (or press `Ctrl+Shift+Enter` across cells).

### Option B: JupyterLab / Classic Jupyter Notebook
From the repository root in PowerShell or terminal:
```powershell
jupyter lab research/notebooks/CAEG_Net_Faculty_Review.ipynb
```
or
```powershell
jupyter notebook research/notebooks/CAEG_Net_Faculty_Review.ipynb
```

### Option C: Non-Interactive Headless Re-Execution
To re-run all 37 cells and update all embedded plots and tables headlessly from the command line:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace research/notebooks/CAEG_Net_Faculty_Review.ipynb
```

---

## 3. System Requirements & Runtime
- **Hardware Requirement:** Standard CPU only (No GPU required).
- **Execution Time:** ~5 to 10 seconds total across all 37 cells.
- **Python Dependencies:** `numpy`, `pandas`, `matplotlib`, `nbformat`, `nbconvert` (all included in `caeg-gpu`).
- **Offline Capability:** Fully self-contained; no external API calls or internet connection needed.

---

## 4. What the Faculty Review Notebook Demonstrates

The notebook follows a crisp, faculty-friendly narrative structure:
**Problem → Solution → Architecture → Methodology → Benchmark Results → Stability & Routing Analysis → Error Decomposition → Defensible Contributions → Limitations → Viva Q&A.**

### Structured Walkthrough (24 Key Sections):
1. **Section 1 — Title & Cover Information:** Student profile, course information, and authoritative model designation.
2. **Section 2 — What is the Problem?:** Real-world importance of 24-hour day-ahead load forecasting (unit commitment, economic dispatch, renewables) and the structural failure modes of monolithic neural networks.
3. **Section 3 — Core Research Question:** Highlighted statement on whether context-aware adaptive gating across heterogeneous temporal experts improves forecasting.
4. **Section 4 — Datasets & Causal Protocol:** Tri-benchmark grid coverage (PJM, GEFCom2014, UCI) with strict chronological splitting (70/15/15) and train-only scaling to eliminate data leakage.
5. **Section 5 — System Architecture Flowchart:** Crisp visual schematic of the 168h input lookback, 3 temporal experts, context encoding, adaptive router, convex fusion, and confidence shrinkage head.
6. **Section 6 — Model Architecture & Parameter Budget:** Comprehensive parameter breakdown verifying the ultra-compact **121,724 parameter** footprint (LSTM: 56,152; TCN: 36,952; CNN: 27,400; Router: 1,075; Confidence Head: 145).
7. **Section 7 — Why Three Experts?:** Explaining inductive biases (LSTM for recurrent diurnal drift, TCN for dilated causal multi-scale patterns, CNN for localized ramp motifs).
8. **Section 8 — Training & Evaluation Protocol:** AdamW optimizer, MSE training loss, primary metric (MAE in physical units MW/kW), and 5-seed sensitivity testing (`[42, 123, 999, 2024, 3407]`).
9. **Section 9 — Final Authoritative Performance Metrics:** Locked 5-seed results:
   - **PJM:** MAE = 250.97 ± 10.69 MW | RMSE = 335.38 MW | R² = 0.8714 | MAPE = 4.68%
   - **GEFCom2014:** MAE = 12.41 ± 0.15 kW | RMSE = 18.04 kW | R² = 0.8610 | MAPE = 9.33%
   - **UCI:** MAE = 7.74 ± 0.30 MW | RMSE = 10.96 MW | R² = 0.9831 | MAPE = 4.05%
10. **Section 10 — Evaluation Metrics Explained:** Mathematical definitions of MAE, RMSE, R², and MAPE with concrete numerical examples.
11. **Section 11 — Baseline Comparisons Across Datasets:** 3-panel comparative bar charts contrasting CAEG-Net F2 against standalone LSTM, TCN, CNN, and the Equal Ensemble.
12. **Section 12 — Final Model vs. Baselines (Scientific Reconciliation):** Defensible interpretation noting that F2 provides the strongest overall multi-grid balance without overclaiming universal dominance.
13. **Section 13 — Five-Seed Sensitivity Analysis:** Seed-by-seed stability table and relative coefficient of variation (CV < 4% across all grids).
14. **Section 14 — Dynamic Routing Weight Allocations:** Grouped bar chart comparing learned expert weights across datasets ($w_{\text{LSTM}}, w_{\text{TCN}}, w_{\text{CNN}}$) vs. the static 1/3 baseline.
15. **Section 15 — Confidence Fallback & Shrinkage Mechanism:** Learned shrinkage parameter $\lambda \approx 0.51$ acting as an empirical regularizer anchoring predictions toward the ensemble centroid.
16. **Section 16 — Statistical Significance & Daily-Block Testing:** Paired $t$-tests and Wilcoxon signed-rank tests over non-overlapping daily blocks ($K=53, 456, 163$) with Holm-Bonferroni correction.
17. **Section 17 — Actual Forecast Tracking & Residuals:** 24-hour diurnal profile tracking actual vs. predicted load on PJM, plus residual error distribution.
18. **Section 18 — 24-Hour Horizon Analysis:** Step-by-step ($h=1 \dots 24$) MAE degradation curves across the lead horizon.
19. **Section 19 — Operational Horizon Group Decomposition:** Error analysis across Short ($h=1-8$), Medium ($h=9-16$), and Long ($h=17-24$) operational dispatch windows.
20. **Section 20 — Research Contributions:** Five defensible methodological contributions.
21. **Section 21 — Key Finding & Executive Summary:** Callout summary box highlighting the operational value and compactness of CAEG-Net F2.
22. **Section 22 — Honest Limitations:** Transparent discussion of point forecasting, univariate load inputs, regional aggregation, and shrinkage dynamics.
23. **Section 23 — Faculty Viva / Defense Q&A:** Top 10 high-yield oral defense questions with concise 2–4 line answers, plus bonus question on router misallocations.
24. **Section 24 — Final 30-Second Elevator Pitch:** Memorizable 30-second summary for oral presentation.

---

## 5. Faculty Viva Presentation Tips
- **Pacing:** Allocate 1 minute to the problem/motivation, 3 minutes to architecture/methodology, 4 minutes to benchmark results and baseline comparisons, 2 minutes to routing/shrinkage analysis, and 2 minutes to limitations and conclusion.
- **Key Metric:** Always refer to **MAE (Mean Absolute Error)** as the primary engineering metric because it is linear and measured directly in physical units (MW or kW).
- **Claim Hygiene:** If asked whether CAEG-Net beats every model on every grid, answer truthfully: *"CAEG-Net F2 strictly wins on PJM and GEFCom, while on UCI standalone LSTM is exceptionally strong; F2 establishes the strongest overall balance of performance, stability, and parameter efficiency across all three diverse grids without catastrophic failure on any regime."*
