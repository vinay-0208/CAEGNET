# CAEG-Net Interactive Research Dashboard

The **CAEG-Net Research Dashboard** is a standalone, web-based demonstration and exploration platform for the **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting) research project.

It provides a polished, interactive interface for faculty reviews, peer examinations, and technical presentations, operating strictly from authoritative locked experimental artifacts.

---

## 1. Quickstart & Execution

### Prerequisites
Ensure dependencies are installed in your Python environment:
```bash
pip install -r requirements.txt
```

### Launch the Dashboard
From the repository root directory, run:
```bash
streamlit run dashboard/app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 2. Dashboard Architecture & Navigation

The application is structured into 11 specialized academic panels accessible via the sidebar:

1. **Overview:** Executive KPI cards (121,724 parameters, 168h lookback, 24h horizon), research question, and benchmark summary.
2. **Architecture:** Detailed modular diagram, mathematical formulation (equations 1–3), and complete parameter breakdown table.
3. **Dataset & Protocol:** Documentation of the PJM, GEFCom2014, and UCI datasets, along with chronological 70/15/15 splitting and train-only scaling.
4. **Benchmark Results:** Authoritative 5-seed benchmark results table (MAE, RMSE, $R^2$) with physical engineering units.
5. **Baseline Comparison:** Dataset-wise comparisons across standalone LSTM, TCN, CNN, static equal ensemble, fixed shrinkage, CAEG-Net, and the diagnostic empirical ex-post oracle.
6. **Routing Behaviour:** Grouped bar charts showing mean expert routing weights and effective expert counts ($N_{\text{eff}} \approx 2.98 - 2.99$).
7. **Confidence / Fallback:** Visualization and interpretation of the learned shrinkage parameter $\lambda \approx 0.51$ and its low temporal variance ($CV < 1.1\%$).
8. **Statistical Evidence:** Hypothesis testing tables for non-overlapping daily blocks ($K=53, 456, 163$) showing paired $t$-test, Wilcoxon signed-rank, and Holm-Bonferroni corrections.
9. **Horizon Analysis:** Interactive step-by-step ($h=1 \dots 24$) lead-time error curves plotted directly from `research/analysis/phase15b_horizon_results.csv`.
10. **Research Findings:** Synthesis of the five primary empirical discoveries from controlled Phase 15 experimentation.
11. **Limitations:** Honest disclosure of technical boundaries, univariate scope, deterministic point forecasting, and the zero-fabrication academic guarantee.

---

## 3. Data & Provenance Integrity

- **Artifact-Driven:** All numerical metrics and plots are generated exclusively from audited repository artifacts (`research/results/final_results.csv`, `research/analysis/phase15b_horizon_results.csv`, `research/results/final_model_config.json`).
- **Zero-Fabrication Guarantee:** The dashboard does **not** generate synthetic prediction arrays, fake actual-vs-predicted curves, or simulated real-time traces.
- **No Live Inference Pretence:** The dashboard presents certified retrospective research findings and does not claim to execute real-time grid stream inference.

---

## 4. Hardware Requirements

- **CPU:** Standard multi-core laptop CPU (no GPU required).
- **RAM:** $\le 1$ GB memory footprint.
- **Startup Time:** Under 2 seconds.
