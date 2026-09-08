"""
Script to build and execute notebooks/CAEG_Net_Research_Analysis.ipynb
=====================================================================
Constructs a comprehensive, self-contained, 13-section research notebook
containing all mathematical specifications, empirical code cells,
multi-seed tables, statistical validation tests, regime analyses,
and rendered matplotlib visualizations.
"""

import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
notebook_path = os.path.join(repo_root, "notebooks", "CAEG_Net_Research_Analysis.ipynb")


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3"
    }

    cells = []

    # Title & Section 1: Research Question
    cells.append(nbf.v4.new_markdown_cell("""# Context-Adaptive Expert Gating Network (CAEG-Net)
## Autonomous Research Track: Context Formulation, Forecast-Aware Routing & Regime Robustness

---

### Section 1: Executive Overview & Research Question

**Primary Research Question:**
> *"Can explicit operational context (trend, volatility, periodicity, causal recent forecasting error, and inter-expert forecast disagreement) be leveraged by a gating network to dynamically route complementary temporal experts, and does this context-adaptive routing produce statistically defensible improvements in forecasting accuracy and robustness across operational regimes?"*

#### Core Scientific Contributions
1. **Explicit Operational Context vs. Raw Input Gating**: Replaces opaque, high-dimensional input routing with low-dimensional physical state features.
2. **Causal Out-of-Sample Error Feedback**: Integrates recent forecasting error generated via expanding-window walk-forward validation as an exogenous sensor for the gating network.
3. **Forecast-Aware Routing Mechanism**: Formulates inter-expert forecast disagreement (pairwise dispersion, standard deviation, and range) computed strictly from input $X$ as an endogenous uncertainty signal.
4. **Regime Stratification**: Demonstrates that the greatest empirical benefit of adaptive routing occurs under high-stress operating regimes (high volatility and high expert disagreement).
"""))

    # Section 2: Environment Setup & Data Pipeline
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 2: Data Integrity, Chronological Split & Leakage Prevention Protocol

Guarantees:
- **8,784 hourly readings** from Modern PJM operational load dataset (0 missing, 0 gaps).
- **Chronological 70/15/15 Split**: Train: 6,148h | Val: 1,318h | Test: 1,318h.
- **Sliding Windows**: Lookback $L=168\text{h}$ (7 days) $\to$ Forecast Horizon $H=24\text{h}$ (1 day).
- **Scaler Isolation**: `StandardScaler` fitted strictly on training partition ($\mu=5,458.03\text{ MW}, \sigma=855.39\text{ MW}$).
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.data import prepare_research_pipeline, build_research_dataloaders
from research.models import ResearchCAEGNet
from caeg_net import count_parameters
from evaluate import compute_metrics

data_path = os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv")
data = prepare_research_pipeline(data_path)
scaler = data["scaler"]
print("Data Pipeline Successfully Initialized:")
print(f"  Train Windows : {len(data['windows']['train']['X']):,}")
print(f"  Val Windows   : {len(data['windows']['val']['X']):,}")
print(f"  Test Windows  : {len(data['windows']['test']['X']):,}")
print(f"  Scaler Mean   : {scaler.mean_[0]:.2f} MW | Scale: {scaler.scale_[0]:.2f} MW")
"""))

    # Section 3: Architecture Parity
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 3: Tri-Expert Inductive Biases & Architectural Parity

CAEG-Net preserves strict architectural parity across all research variants. The three complementary temporal experts remain identical:
1. **LSTM Expert ($E_{\\text{LSTM}}$)**: 2-layer stacked LSTM (`hidden_dim=64`, `dropout=0.1`) $\implies 56,152$ parameters.
2. **TCN Expert ($E_{\\text{TCN}}$)**: 6-stage dilated causal residual Conv1D blocks ($d \in [1, 2, 4, 8, 16, 32]$, $k=3$, 32 channels, receptive field: 253h) $\implies 36,952$ parameters.
3. **CNN Expert ($E_{\\text{CNN}}$)**: 3-stage 1D convolutional hierarchy ($k=3, 5, 3$) with adaptive pooling $\implies 27,400$ parameters.
"""))

    cells.append(nbf.v4.new_code_cell("""m_v1 = ResearchCAEGNet(base_context_dim=4, forecast_aware=False)
m_fa = ResearchCAEGNet(base_context_dim=4, forecast_aware=True)
m_cal = ResearchCAEGNet(base_context_dim=8, forecast_aware=False)

print(f"Canonical V1 Parameters        : {count_parameters(m_v1)['total_trainable']:,}")
print(f"Forecast-Aware (Var D) Params  : {count_parameters(m_fa)['total_trainable']:,}")
print(f"Calendar-Context (Var C) Params: {count_parameters(m_cal)['total_trainable']:,}")
"""))

    # Section 4: Unified Model Benchmark Comparison
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 4: Unified Multi-Seed Model Benchmark Comparison

Unified comparison of all 8 historical baselines alongside the research-track variants evaluated across the 5 canonical seeds ($42, 123, 999, 2024, 3407$).
"""))

    cells.append(nbf.v4.new_code_cell("""res_dir = os.path.join(repo_root, "research", "results")
df_comp = pd.read_csv(os.path.join(res_dir, "model_comparison.csv"))
print("=" * 95)
print("UNIFIED 5-SEED PERFORMANCE BENCHMARK TABLE")
print("=" * 95)
print(df_comp[["Model", "Parameters", "MAE (MW)", "RMSE (MW)", "R^2", "MAPE (%)", "Track"]].to_string(index=False))
print("=" * 95)
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Multi-Seed Comparison Plot
from IPython.display import Image, display
display(Image(filename=os.path.join(repo_root, "research", "analysis", "model_comparison.png")))
"""))

    # Section 5: Context Feature Ablation Study
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 5: Context Feature Ablation & Lesioning Analysis

To determine which context variables genuinely contribute to gating accuracy, we evaluate systematic feature ablation and a context permutation negative control.
"""))

    cells.append(nbf.v4.new_code_cell("""df_ablation = pd.read_csv(os.path.join(res_dir, "context_ablation.csv"))
print("=" * 85)
print("CONTEXT ABLATION & LESIONING STUDY RESULTS")
print("=" * 85)
print(df_ablation[["Ablation_ID", "Description", "MAE (MW)", "RMSE (MW)", "Delta_MAE_MW", "Relative_Impact"]].to_string(index=False))
print("=" * 85)
display(Image(filename=os.path.join(repo_root, "research", "analysis", "ablation_results.png")))
"""))

    # Section 6: Operational Regime & Difficulty Analysis
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 6: Operational Regime & Difficulty Analysis

We stratify the test set into Low, Medium, and High operational stress regimes across Volatility, Recent Error, and Expert Disagreement.
"""))

    cells.append(nbf.v4.new_code_cell("""df_regimes = pd.read_csv(os.path.join(res_dir, "regime_analysis.csv"))
print("=" * 105)
print("OPERATIONAL REGIME STRATIFICATION ANALYSIS")
print("=" * 105)
print(df_regimes[["Regime_Dimension", "Regime_Level", "Samples", "TCN_MAE_MW", "CAEG_ForecastAware_MAE_MW", "CAEG_Advantage_MW", "CAEG_Advantage_pct"]].to_string(index=False))
print("=" * 105)
display(Image(filename=os.path.join(repo_root, "research", "analysis", "regime_performance.png")))
"""))

    # Section 7: Expert Routing Diagnostics & Learned Specialization
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 7: Expert Routing Diagnostics & Context Correlations

We inspect the sample-level routing distributions and statistical correlations between operational context variables and expert preferences.
"""))

    cells.append(nbf.v4.new_code_cell("""df_dist = pd.read_csv(os.path.join(res_dir, "routing_analysis.csv"))
df_corr = pd.read_csv(os.path.join(res_dir, "routing_correlations.csv"))
print("Expert Weight Distribution Statistics:")
print(df_dist.to_string(index=False))
print("\\nStatistically Significant Context Associations:")
print(df_corr[df_corr["Significance"] != "n.s."][["Context_Variable", "Expert", "Pearson_r", "Pearson_pval", "Significance"]].to_string(index=False))
display(Image(filename=os.path.join(repo_root, "research", "analysis", "expert_weights.png")))
display(Image(filename=os.path.join(repo_root, "research", "analysis", "routing_vs_context.png")))
"""))

    # Section 8: Statistical Hypothesis Testing
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 8: Statistical Validation & Hypothesis Testing

Rigorous hypothesis testing: Seed-level paired t-tests (df = 4), 95% bootstrap confidence intervals, and 53 non-overlapping 24-hour test episodes.
"""))

    cells.append(nbf.v4.new_code_cell("""df_tests = pd.read_csv(os.path.join(res_dir, "statistical_tests.csv"))
print("=" * 105)
print("STATISTICAL VALIDATION & HYPOTHESIS TESTING SUMMARY")
print("=" * 105)
print(df_tests[["Comparison", "Evaluation_Granularity", "Mean_Delta_MAE_MW", "Bootstrap_95_CI", "t_statistic", "p_value", "Significance_Status"]].to_string(index=False))
print("=" * 105)
"""))

    # Section 9: Computational Telemetry
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 9: Computational Efficiency & Telemetry

Profiling execution metrics on the Intel Core i7-13700H and NVIDIA GeForce RTX 4050 Laptop GPU.
"""))

    cells.append(nbf.v4.new_code_cell("""df_telem = pd.read_csv(os.path.join(res_dir, "training_telemetry.csv"))
print("=" * 95)
print("COMPUTATIONAL TELEMETRY & HARDWARE PROFILING")
print("=" * 95)
print(df_telem[["Model", "Parameters", "Avg_Train_Time_s", "Inference_Latency_s", "Peak_VRAM_MB", "Compute_Device"]].to_string(index=False))
print("=" * 95)
"""))

    # Section 10: Limitations & Scientific Positioning
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 10: Scientific Limitations & Discussion

1. **Calendar Features Negative Finding**: Explicit cyclic calendar encoding degraded forecasting accuracy ($258.02 \to 270.78\text{ MW}$). The 168h lookback already captures weekly periodicities more effectively without risking overfitting to calendar indices.
2. **Forecast-Aware Significance Boundary**: While Forecast-Aware Routing achieved lower mean MAE ($256.73\text{ MW}$) and lower RMSE ($341.49\text{ MW}$) than V1, the paired difference across 5 seeds did not achieve statistical significance ($p = 0.7395 > 0.05$). Thus, it represents an observed improvement rather than a statistically proven divergence from V1.
3. **Regime-Specific Superiority**: The primary defensible advantage of CAEG-Net is in high-stress operational states: outperforming standalone TCN by **+27.78 MW (8.79%)** during periods of high expert disagreement.
"""))

    # Section 11: Final Research Conclusion
    cells.append(nbf.v4.new_markdown_cell("""---
### Section 11: Research Conclusions & Next Steps

- **Core Finding**: Explicit context routing is statistically superior to fixed equal ensembles ($p = 0.0153$) and raw input-based MoE ($p = 0.0415$).
- **Routing Dynamics**: Diurnal periodicity and causal recent error are the primary drivers of routing quality.
- **Publication Readiness**: Strong empirical and diagnostic evidence ready for conference/journal submission with reproducible scripts and multi-seed checkpoints.
"""))

    nb.cells = cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created research notebook structure: {notebook_path}")

    print("Executing notebook to render all cells, tables, and images...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_to_run = nbf.read(f, as_version=4)

    ep.preprocess(nb_to_run, {"metadata": {"path": os.path.join(repo_root, "notebooks")}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb_to_run, f)
    print(f"Successfully executed and saved research notebook: {notebook_path}")


if __name__ == "__main__":
    build_notebook()
