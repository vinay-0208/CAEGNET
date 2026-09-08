"""
CAEG-Net Research Track: Phase 7 Regime & Operational Difficulty Analysis
========================================================================
Investigates whether CAEG-Net dynamic routing provides outsized benefits
under severe operational regimes (high volatility, high recent error, high expert disagreement).

Methodology:
- Stratifies the 1,294 test windows into 3 empirically derived regimes (Low, Medium, High)
  using 33.3% and 66.7% percentiles strictly from origin-available information.
- Compares CAEG-Net against the strongest standalone baseline (TCN).
- Computes expert routing distributions across each operating regime.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.data import prepare_research_pipeline, build_research_dataloaders
from caeg_net import TCNExpert
from evaluate import compute_metrics, evaluate_model_on_loader


def run_regime_analysis():
    print("=" * 85)
    print("CAEG-NET RESEARCH TRACK: PHASE 7 REGIME & DIFFICULTY ANALYSIS")
    print("=" * 85)

    res_dir = os.path.join(repo_root, "research", "results")
    cache_path = os.path.join(res_dir, "research_multiseed_cache.npz")
    cache = np.load(cache_path)

    y_true = cache["y_test_true_raw"]
    y_pred_fa = cache["pred_CAEG_Net_Forecast_Aware_seed_42"]
    weights_fa = cache["weights_CAEG_Net_Forecast_Aware_seed_42"]
    y_pred_v1 = cache["pred_CAEG_Net_V1_Canonical_seed_42"]

    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    scaler = data["scaler"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load canonical TCN Standalone baseline
    tcn_model = TCNExpert(input_dim=1, channels=32, dilations=(1, 2, 4, 8, 16, 32), horizon=24).to(device)
    tcn_ckpt = os.path.join(repo_root, "checkpoints", "seed_42", "tcn.pt")
    if os.path.isfile(tcn_ckpt):
        tcn_model.load_state_dict(torch.load(tcn_ckpt, map_location=device))
    else:
        # Fallback to general checkpoint
        tcn_model.load_state_dict(torch.load(os.path.join(repo_root, "checkpoints", "tcn_standalone.pt"), map_location=device))

    loaders = build_research_dataloaders(data, context_type="4d", batch_size=64, seed=42)
    tcn_eval = evaluate_model_on_loader(tcn_model, loaders["test"], scaler)
    y_pred_tcn = tcn_eval["y_pred_raw"]

    # Extract origin context dimensions
    C_test = data["context_4d"]["test"]
    volatility = C_test[:, 1]
    recent_error = C_test[:, 3]

    # Compute expert disagreement from model predictions if available
    # Or compute pairwise standard deviation across experts
    # Disagreement can be derived from the expert forecasts in CAEG
    disagreement = np.std(np.stack([y_pred_fa, y_pred_v1, y_pred_tcn], axis=-1), axis=-1).mean(axis=-1)

    regime_definitions = [
        ("Volatility", volatility, "MW (scaled)"),
        ("Recent_Error", recent_error, "MW (scaled)"),
        ("Expert_Disagreement", disagreement, "MW"),
    ]

    regime_records = []

    for reg_name, reg_vals, unit in regime_definitions:
        q33 = np.percentile(reg_vals, 33.33)
        q67 = np.percentile(reg_vals, 66.67)

        bins = [
            ("Low", reg_vals <= q33),
            ("Medium", (reg_vals > q33) & (reg_vals <= q67)),
            ("High", reg_vals > q67),
        ]

        for level, mask in bins:
            n_samples = int(mask.sum())
            y_t = y_true[mask]
            y_fa = y_pred_fa[mask]
            y_v1 = y_pred_v1[mask]
            y_tc = y_pred_tcn[mask]
            w_sub = weights_fa[mask]

            m_fa = compute_metrics(y_fa, y_t)
            m_v1 = compute_metrics(y_v1, y_t)
            m_tc = compute_metrics(y_tc, y_t)

            improvement_over_tcn = m_tc["MAE"] - m_fa["MAE"]
            pct_improvement = (improvement_over_tcn / m_tc["MAE"]) * 100.0

            mean_w_lstm = float(w_sub[:, 0].mean())
            mean_w_tcn = float(w_sub[:, 1].mean())
            mean_w_cnn = float(w_sub[:, 2].mean())

            regime_records.append({
                "Regime_Dimension": reg_name,
                "Regime_Level": level,
                "Samples": n_samples,
                "Threshold_Range": f"<= {q33:.3f}" if level == "Low" else (f"> {q67:.3f}" if level == "High" else f"({q33:.3f}, {q67:.3f}]"),
                "TCN_MAE_MW": round(m_tc["MAE"], 2),
                "TCN_RMSE_MW": round(m_tc["RMSE"], 2),
                "CAEG_V1_MAE_MW": round(m_v1["MAE"], 2),
                "CAEG_ForecastAware_MAE_MW": round(m_fa["MAE"], 2),
                "CAEG_ForecastAware_RMSE_MW": round(m_fa["RMSE"], 2),
                "CAEG_Advantage_MW": round(improvement_over_tcn, 2),
                "CAEG_Advantage_pct": round(pct_improvement, 2),
                "Mean_W_LSTM": round(mean_w_lstm, 3),
                "Mean_W_TCN": round(mean_w_tcn, 3),
                "Mean_W_CNN": round(mean_w_cnn, 3),
            })

    df_regimes = pd.DataFrame(regime_records)
    csv_path = os.path.join(res_dir, "regime_analysis.csv")
    df_regimes.to_csv(csv_path, index=False)
    print(f"Saved regime analysis to: {csv_path}\n")

    print("=" * 105)
    print("REGIME ANALYSIS SUMMARY TABLE (PERFORMANCE ACROSS OPERATIONAL CONDITIONS):")
    print("=" * 105)
    print(df_regimes[[
        "Regime_Dimension", "Regime_Level", "Samples", "TCN_MAE_MW", "CAEG_ForecastAware_MAE_MW",
        "CAEG_Advantage_MW", "CAEG_Advantage_pct", "Mean_W_LSTM", "Mean_W_TCN", "Mean_W_CNN"
    ]].to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    run_regime_analysis()
