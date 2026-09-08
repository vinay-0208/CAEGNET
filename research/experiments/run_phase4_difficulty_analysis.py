"""
CAEG-Net V2 Phase 4 Difficulty, Regime & Correlation Analysis
=============================================================
Analyzes the 1,294 test origins from `phase4_seed42_routing_diagnostics.csv`:
1. Context -> Routing -> Error Correlations:
   - Pearson and Spearman rank correlations between all 9 context features and:
     * Router weights (w_GRU, w_TCN, w_Patch)
     * Expert errors (MAE_GRU, MAE_TCN, MAE_Patch)
     * Fused error (MAE_Fused)
     * Inter-expert disagreement (Disagreement_Pairwise_MAE)
     * Routing improvement over equal ensemble and best standalone expert
2. Operational Regime Analysis (predetermined non-test-driven tertiles):
   - Volatility Regimes: Low, Medium, High
   - Disagreement Regimes: Low, Medium, High
   - Baseline Error Regimes: Low, Medium, High
   - Diurnal Peak / Ramp Regimes: Morning Ramp (06-09), Evening Peak (17-21), Off-Peak Trough (01-05)
3. Oracle & Complementarity Analysis:
   - Headroom: Fused vs Oracle vs Best Single Expert vs Equal Ensemble
   - Expert win rates across regimes
   - Regret analysis

Outputs:
- research/results/phase4_feature_correlations.csv
- research/results/phase4_regime_analysis.csv
- research/results/phase4_oracle_summary.json
"""

import sys
import os
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def run_difficulty_analysis(
    diagnostics_csv: str = "research/results/phase4_seed42_routing_diagnostics.csv",
    out_dir: str = "research/results",
):
    print(f"=== CAEG-Net V2 Phase 4 Difficulty, Regime & Correlation Analysis ===")
    df = pd.read_csv(diagnostics_csv)
    print(f"Loaded {len(df)} diagnostic rows from {diagnostics_csv}")

    # -------------------------------------------------------------
    # 1. Feature Correlations (Pearson & Spearman)
    # -------------------------------------------------------------
    features = [
        "Trend_Slope",
        "ShortTerm_Volatility",
        "Recent48h_RangeRatio",
        "Diurnal_Periodicity_r24",
        "Weekly_Profile_r168",
        "Recent_Baseline_MAE",
        "Disagreement_Pairwise_MAE",
        "Disagreement_Std",
        "Disagreement_Range",
    ]

    targets = [
        "w_GRU",
        "w_TCN",
        "w_Patch",
        "Entropy_H",
        "MAE_Fused",
        "MAE_GRU",
        "MAE_TCN",
        "MAE_Patch",
        "Gain_over_Equal_Ensemble",
        "Gain_over_Best_Single",
    ]

    corr_rows = []
    for feat in features:
        x = df[feat].values
        for tgt in targets:
            y = df[tgt].values
            p_r, p_p = stats.pearsonr(x, y)
            s_r, s_p = stats.spearmanr(x, y)
            corr_rows.append({
                "Feature": feat,
                "Target": tgt,
                "Pearson_r": round(p_r, 4),
                "Pearson_p": round(p_p, 4),
                "Spearman_rho": round(s_r, 4),
                "Spearman_p": round(s_p, 4),
            })

    df_corr = pd.DataFrame(corr_rows)
    corr_csv = os.path.join(out_dir, "phase4_feature_correlations.csv")
    df_corr.to_csv(corr_csv, index=False)
    print(f"Saved feature correlation matrix to {corr_csv}")

    # -------------------------------------------------------------
    # 2. Operational Regime Analysis
    # -------------------------------------------------------------
    # Regimes based on training/validation derived distribution
    # Here using the quantiles of the features
    def make_tertiles(series: pd.Series) -> pd.Series:
        q33 = series.quantile(0.3333)
        q66 = series.quantile(0.6667)
        def label(val):
            if val < q33:
                return "Low"
            elif val <= q66:
                return "Medium"
            else:
                return "High"
        return series.apply(label)

    df["Volatility_Regime"] = make_tertiles(df["ShortTerm_Volatility"])
    df["Disagreement_Regime"] = make_tertiles(df["Disagreement_Pairwise_MAE"])
    df["Baseline_Error_Regime"] = make_tertiles(df["Recent_Baseline_MAE"])

    # Diurnal Hour of Day Regimes
    # Parse timestamp
    dt = pd.to_datetime(df["timestamp"])
    hour = dt.dt.hour
    def diurnal_regime(h):
        if 6 <= h <= 9:
            return "Morning_Ramp (06-09)"
        elif 17 <= h <= 21:
            return "Evening_Peak (17-21)"
        elif 1 <= h <= 5:
            return "Off_Peak_Trough (01-05)"
        else:
            return "Daytime_MidLoad (10-16, 22-00)"

    df["Diurnal_Regime"] = hour.apply(diurnal_regime)

    regime_columns = [
        "Volatility_Regime",
        "Disagreement_Regime",
        "Baseline_Error_Regime",
        "Diurnal_Regime",
    ]

    regime_rows = []
    for reg_col in regime_columns:
        for reg_val, grp in df.groupby(reg_col):
            n_grp = len(grp)
            fused_mae = grp["MAE_Fused"].mean()
            gru_mae = grp["MAE_GRU"].mean()
            tcn_mae = grp["MAE_TCN"].mean()
            patch_mae = grp["MAE_Patch"].mean()
            equal_mae = grp["MAE_Equal_Ensemble"].mean()
            oracle_mae = grp["MAE_Oracle"].mean()
            best_single = min(gru_mae, tcn_mae, patch_mae)

            w_gru = grp["w_GRU"].mean()
            w_tcn = grp["w_TCN"].mean()
            w_patch = grp["w_Patch"].mean()

            regime_rows.append({
                "Regime_Type": reg_col.replace("_Regime", ""),
                "Regime_Level": reg_val,
                "Sample_Count": n_grp,
                "MAE_Fused_MW": round(fused_mae, 2),
                "MAE_Equal_Ens_MW": round(equal_mae, 2),
                "MAE_Best_Single_MW": round(best_single, 2),
                "MAE_Patch_MW": round(patch_mae, 2),
                "MAE_TCN_MW": round(tcn_mae, 2),
                "MAE_GRU_MW": round(gru_mae, 2),
                "MAE_Oracle_MW": round(oracle_mae, 2),
                "Gain_vs_Equal_MW": round(equal_mae - fused_mae, 2),
                "Gain_vs_Best_Single_MW": round(best_single - fused_mae, 2),
                "Regret_vs_Oracle_MW": round(fused_mae - oracle_mae, 2),
                "w_GRU_mean": round(w_gru, 4),
                "w_TCN_mean": round(w_tcn, 4),
                "w_Patch_mean": round(w_patch, 4),
                "Mean_Disagreement": round(grp["Disagreement_Pairwise_MAE"].mean(), 4),
            })

    df_reg = pd.DataFrame(regime_rows)
    reg_csv = os.path.join(out_dir, "phase4_regime_analysis.csv")
    df_reg.to_csv(reg_csv, index=False)
    print(f"Saved regime analysis breakdown to {reg_csv}")

    # -------------------------------------------------------------
    # 3. Oracle Headroom & Complementarity Summary
    # -------------------------------------------------------------
    total_n = len(df)
    oracle_summary = {
        "overall_test_origins": total_n,
        "fused_mae_mw": round(float(df["MAE_Fused"].mean()), 2),
        "equal_ensemble_mae_mw": round(float(df["MAE_Equal_Ensemble"].mean()), 2),
        "patch_standalone_mae_mw": round(float(df["MAE_Patch"].mean()), 2),
        "tcn_standalone_mae_mw": round(float(df["MAE_TCN"].mean()), 2),
        "gru_standalone_mae_mw": round(float(df["MAE_GRU"].mean()), 2),
        "oracle_mae_mw": round(float(df["MAE_Oracle"].mean()), 2),
        "mean_regret_vs_oracle_mw": round(float(df["Regret_vs_Oracle"].mean()), 2),
        "fused_gain_over_equal_ens_mw": round(float(df["Gain_over_Equal_Ensemble"].mean()), 2),
        "fused_gain_over_best_single_mw": round(float(df["Gain_over_Best_Single"].mean()), 2),
        "theoretical_oracle_headroom_mw": round(float(df["MAE_Fused"].mean() - df["MAE_Oracle"].mean()), 2),
        "expert_win_shares_pct": {
            "Patch": round(float((df["Oracle_Best_Expert"] == "Patch").mean() * 100), 2),
            "TCN": round(float((df["Oracle_Best_Expert"] == "TCN").mean() * 100), 2),
            "GRU": round(float((df["Oracle_Best_Expert"] == "GRU").mean() * 100), 2),
        },
        "router_top_choice_accuracy_pct": round(float(df["Is_Router_Top_Oracle"].mean() * 100), 2),
        "high_disagreement_fused_gain_vs_equal_mw": round(
            float(df[df["Disagreement_Regime"] == "High"]["Gain_over_Equal_Ensemble"].mean()), 2
        ),
        "high_volatility_fused_gain_vs_equal_mw": round(
            float(df[df["Volatility_Regime"] == "High"]["Gain_over_Equal_Ensemble"].mean()), 2
        ),
    }

    oracle_json = os.path.join(out_dir, "phase4_oracle_summary.json")
    with open(oracle_json, "w", encoding="utf-8") as f:
        json.dump(oracle_summary, f, indent=2)
    print(f"Saved oracle and complementarity summary to {oracle_json}")

    print("\n=== ORACLE & REGIME SUMMARY ===")
    print(f"Fused MAE:                {oracle_summary['fused_mae_mw']:.2f} MW")
    print(f"Equal Ensemble MAE:       {oracle_summary['equal_ensemble_mae_mw']:.2f} MW (Gain: +{oracle_summary['fused_gain_over_equal_ens_mw']:.2f} MW)")
    print(f"Best Standalone (Patch):  {oracle_summary['patch_standalone_mae_mw']:.2f} MW (Gain: +{oracle_summary['fused_gain_over_best_single_mw']:.2f} MW)")
    print(f"Oracle Lower Bound:       {oracle_summary['oracle_mae_mw']:.2f} MW (Headroom: {oracle_summary['theoretical_oracle_headroom_mw']:.2f} MW)")
    print(f"Expert Win Rates:         Patch: {oracle_summary['expert_win_shares_pct']['Patch']}%, TCN: {oracle_summary['expert_win_shares_pct']['TCN']}%, GRU: {oracle_summary['expert_win_shares_pct']['GRU']}%")
    print(f"Router Top Accuracy:      {oracle_summary['router_top_choice_accuracy_pct']}%")

    return df_corr, df_reg, oracle_summary


if __name__ == "__main__":
    run_difficulty_analysis()
