# CAEG-Net Interactive Research Dashboard

The **CAEG-Net Research Dashboard** is a standalone, web-based demonstration and exploration platform for the **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting) research project.

It provides a polished, interactive interface for faculty reviews, peer examinations, and technical presentations, operating strictly from authoritative locked experimental artifacts.

---

## 1. Quickstart & Execution (Windows PowerShell)

### Environment Setup
The dashboard and live PyTorch inference simulator are verified in the existing `caeg-gpu` Anaconda environment (`Python 3.14.7`, `torch 2.13.0+cu130`, `streamlit 1.65.0`, `plotly 7.1.0`).

If setting up a fresh environment, install the required packages via:
```powershell
pip install -r requirements.txt
```

### Launch the Dashboard (Windows PowerShell)
From the repository root directory (`C:\Fall Semister\2026\Advanced Predictive Analytics`), run either:

**Option A — Direct One-Line PowerShell Command (`caeg-gpu` environment):**
```powershell
& "C:\Users\vinay\anaconda3\envs\caeg-gpu\Scripts\streamlit.exe" run dashboard/app.py
```
*(Or via Python module execution: `& "C:\Users\vinay\anaconda3\envs\caeg-gpu\python.exe" -m streamlit run dashboard/app.py`)*

**Option B — After Activating `caeg-gpu` in Conda PowerShell:**
```powershell
conda activate caeg-gpu
streamlit run dashboard/app.py
```

The application will launch locally in your browser at `http://localhost:8501`.

---

## 2. Dashboard Architecture & Navigation

The application is structured into 12 specialized academic panels accessible via the sidebar:

1. **Overview:** Executive KPI cards (121,724 parameters, 168h lookback, 24h horizon), research question, and benchmark summary.
2. **◈ Prediction / Forecast:** Dual-mode forecasting and evaluation hub:
   - **Tab 1 — Interactive CAEG-Net V1 Forecast Simulator (Primary Live Inference Interface):**
     - Runs genuine live PyTorch inference (`model.eval()`, `torch.no_grad()`) using the frozen **Canonical CAEG-Net V1** architecture (`caeg_net.CAEGNet`, `121,531` parameters) and verified checkpoints (`checkpoints/seed_{seed}/caeg_full.pt` for seeds `42, 123, 999, 2024, 3407` or `5-Seed Ensemble`).
     - **Method A (Enter / Paste or Edit Hourly Table):** Supports pasting newline/comma-separated hourly load values (`MW`) or editing an interactive hourly table (`168h` minimum lookback; `192h` recommended for exact causal walk-forward `Recent_Error`).
     - **Method B (Upload CSV):** Strict CSV schema and chronological 1-hour interval validation (`load`/`load_mw` column + optional timestamp column).
     - **Optional Historical-Window Live Validation Mode:** Runs live inference on any of the `1,294` PJM test windows and verifies numerical parity against `results/phase5_multiseed_cache.npz` (`atol = 1e-3 MW` for predictions, `1e-6` for routing weights).
     - Displays the 168-hour lookback + 24-hour forecast trajectory, individual LSTM/TCN/CNN expert forecasts, equal-weight average, 4D causal context vector (`Trend, Volatility, Periodicity, Recent_Error`), routing weights (`w_LSTM + w_TCN + w_CNN = 1.0`), and downloadable 24-hour CSV export.
   - **Tab 2 — Cached Benchmark Evaluation Viewer:** Real historical test evaluation viewer loaded directly from `dashboard/assets/data/research_multiseed_cache.npz` and `dashboard/assets/data/pjm_test_context.npz`. Features:
     - Full 192-hour trajectory plot (168-hour historical lookback context + 24-hour lead forecast in physical MW).
     - Actual ground truth demand vs CAEG-Net forecast with shaded absolute error area.
     - Optional expert baseline overlays (LSTM, TCN, CNN, Equal Ensemble).
     - Zoomed 24-hour forecast profile and step-by-step residual bar chart ($y_{\text{true}} - \hat{y}$).
     - Window-level KPIs (Window MAE, RMSE, Peak Error, and exact dynamic routing weights $[w_L, w_T, w_C]$).
     - Curated benchmark presets plus manual index slider (0 to 1,293).
3. **Architecture & Formulation:** Detailed modular diagram, mathematical formulation (equations 1–3), and complete parameter breakdown table.
4. **Dataset & Causal Protocol:** Documentation of the PJM, GEFCom2014, and UCI datasets, along with chronological 70/15/15 splitting and train-only scaling.
5. **Benchmark Results:** Authoritative 5-seed benchmark results table (MAE, RMSE, $R^2$) with physical engineering units and seed-by-seed breakdown.
6. **Baseline Comparison:** Dataset-wise comparisons across standalone LSTM, TCN, CNN, static equal ensemble, fixed shrinkage, CAEG-Net, and the diagnostic empirical ex-post oracle.
7. **Routing Behaviour:** Grouped bar charts showing mean expert routing weights and effective expert counts ($N_{\text{eff}} \approx 2.98 - 2.99$).
8. **Confidence / Fallback:** Visualization and interpretation of the learned shrinkage parameter $\lambda \approx 0.51$ and its low temporal variance ($CV < 1.1\%$).
9. **Statistical Evidence:** Hypothesis testing tables for non-overlapping daily blocks ($K=53, 456, 163$) showing paired $t$-test, Wilcoxon signed-rank, and Holm-Bonferroni corrections.
10. **Horizon Analysis:** Interactive step-by-step ($h=1 \dots 24$) lead-time error curves plotted directly from `dashboard/assets/data/horizon_results.csv`.
11. **Research Findings:** Synthesis of the five primary empirical discoveries from controlled experimental ablation studies.
12. **Limitations & Ethics:** Honest disclosure of technical boundaries, univariate scope, deterministic point forecasting, and the zero-fabrication academic guarantee.

---

## 3. Data & Provenance Integrity

- **Artifact-Driven Benchmark Evaluation:** All retrospective benchmark metrics and tables are generated exclusively from audited repository artifacts (`dashboard/assets/data/research_multiseed_cache.npz`, `dashboard/assets/data/pjm_test_context.npz`, `dashboard/assets/data/horizon_results.csv`, `dashboard/assets/data/final_results.csv`).
- **Genuine Live Simulator Inference:** The Interactive CAEG-Net V1 Forecast Simulator executes real PyTorch forward passes (`caeg_net.CAEGNet`) using the training-fitted PJM `StandardScaler` (`mean = 5458.0340 MW`, `scale = 855.3898 MW`) and causal context feature pipeline (`data_utils.py`).
- **Zero-Fabrication Guarantee:** Hypothetical custom scenarios are explicitly labelled as such and never display fabricated ground-truth accuracy metrics.

---

## 4. Hardware Requirements

- **CPU:** Standard multi-core laptop CPU (no GPU required).
- **RAM:** $\le 1$ GB memory footprint.
- **Startup Time:** Under 2 seconds.
