"""
CAEG-Net Research Track: Phase 8 Expert Routing Diagnostic Analysis
===================================================================
Investigates the behavior, dynamic distribution, and context associations
of the learned expert gating weights (w_LSTM, w_TCN, w_CNN).

Guarantees:
- Strictly empirical: Associations are described as correlations, not causal mechanisms.
- Quantifies specialization patterns across temporal, volatility, and error regimes.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.data import prepare_research_pipeline


def run_routing_analysis():
    print("=" * 85)
    print("CAEG-NET RESEARCH TRACK: PHASE 8 EXPERT ROUTING DIAGNOSTIC ANALYSIS")
    print("=" * 85)

    res_dir = os.path.join(repo_root, "research", "results")
    cache_path = os.path.join(res_dir, "research_multiseed_cache.npz")
    cache = np.load(cache_path)

    weights_fa = cache["weights_CAEG_Net_Forecast_Aware_seed_42"]  # [1294, 3]
    w_lstm = weights_fa[:, 0]
    w_tcn = weights_fa[:, 1]
    w_cnn = weights_fa[:, 2]

    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    C_test_4d = data["context_4d"]["test"]
    ts_te = data["origin_timestamps"]["test"]

    trend = C_test_4d[:, 0]
    volatility = C_test_4d[:, 1]
    periodicity = C_test_4d[:, 2]
    recent_error = C_test_4d[:, 3]

    hour = pd.to_datetime(ts_te).dt.hour.values
    dow = pd.to_datetime(ts_te).dt.dayofweek.values

    # Overall weight distribution statistics
    expert_names = ["LSTM_Expert", "TCN_Expert", "CNN_Expert"]
    weights_matrix = [w_lstm, w_tcn, w_cnn]

    dist_records = []
    for name, w in zip(expert_names, weights_matrix):
        dist_records.append({
            "Expert": name,
            "Mean": round(float(np.mean(w)), 4),
            "Std": round(float(np.std(w)), 4),
            "Min": round(float(np.min(w)), 4),
            "Q25": round(float(np.percentile(w, 25)), 4),
            "Median": round(float(np.median(w)), 4),
            "Q75": round(float(np.percentile(w, 75)), 4),
            "Max": round(float(np.max(w)), 4),
        })

    df_dist = pd.DataFrame(dist_records)
    dist_csv = os.path.join(res_dir, "routing_analysis.csv")
    df_dist.to_csv(dist_csv, index=False)
    print(f"Saved routing distribution statistics to: {dist_csv}\n")

    # Correlations between expert weights and context variables
    context_vars = [
        ("Trend", trend),
        ("Volatility", volatility),
        ("Periodicity (Lag-24)", periodicity),
        ("Recent_Forecast_Error", recent_error),
        ("Hour_of_Day", hour),
        ("Day_of_Week", dow),
    ]

    corr_records = []
    for c_name, c_val in context_vars:
        for e_name, w in zip(expert_names, weights_matrix):
            p_r, p_val = pearsonr(w, c_val)
            s_rho, s_val = spearmanr(w, c_val)
            corr_records.append({
                "Context_Variable": c_name,
                "Expert": e_name,
                "Pearson_r": round(float(p_r), 4),
                "Pearson_pval": f"{p_val:.2e}",
                "Spearman_rho": round(float(s_rho), 4),
                "Spearman_pval": f"{s_val:.2e}",
                "Significance": "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")),
            })

    df_corr = pd.DataFrame(corr_records)
    corr_csv = os.path.join(res_dir, "routing_correlations.csv")
    df_corr.to_csv(corr_csv, index=False)
    print(f"Saved routing correlation matrix to: {corr_csv}\n")

    print("=" * 85)
    print("LEARNED EXPERT ROUTING WEIGHT DISTRIBUTIONS:")
    print("=" * 85)
    print(df_dist.to_string(index=False))
    print("=" * 85)

    print("\n" + "=" * 95)
    print("STATISTICAL ASSOCIATIONS BETWEEN CONTEXT SIGNALS AND EXPERT ROUTING PREFERENCES:")
    print("=" * 95)
    print(df_corr[["Context_Variable", "Expert", "Pearson_r", "Pearson_pval", "Spearman_rho", "Significance"]].to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    run_routing_analysis()
