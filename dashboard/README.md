# CAEG-Net Interactive Results Dashboard

Interactive visualization dashboard for **CAEG-Net** (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting).

---

## Quickstart

Run the dashboard using Streamlit:

```bash
streamlit run dashboard/app.py
```

---

## Features
- **Executive Summary:** Live metrics across PJM, GEFCom, and UCI.
- **Model Architecture Explorer:** Visual parameter breakdown (121,724 parameters) and expert routing dynamics.
- **Multi-Grid Benchmark Explorer:** Interactive comparison against standalone baselines, static ensembles, and oracle bounds.
- **Forecast Horizon Analysis:** Step-by-step ($h=1 \dots 24$) lead-time error curves.
- **Statistical Significance Viewer:** Paired $t$-test, Wilcoxon, and Holm-Bonferroni results on non-overlapping daily blocks.
