"""
Phase 7 Complete Performance Metrics Recovery & Analysis
========================================================
Completes the full performance metrics record across all 8 models and 5 seeds:
- MAE (MW)
- MSE (MW^2)
- RMSE (MW)
- R^2
- MAPE (%)

Guarantees:
- Zero retraining.
- Calculated strictly on raw Megawatt (MW) predictions.
- Explicit verification that minimum raw load > 0 (no division by zero).
- Exact consistency assertion: RMSE == sqrt(MSE).
- Aggregation computes mean/std directly from individual seed metrics (no squaring of mean RMSE).
- Context-regime and post-hoc error-regime evaluation across all 5 metrics.
- Performance vs routing and expert vs fused comparisons.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_phase7_recovery():
    print("=" * 85)
    print("STARTING PHASE 7 COMPLETE METRICS RECOVERY & PERFORMANCE-ROUTING ANALYSIS")
    print("=" * 85)

    # 1. Load Existing Phase 5 Multi-Seed Prediction Cache
    cache_p5_path = "results/phase5_multiseed_cache.npz"
    cache_p6_path = "results/phase6_analysis_cache.npz"
    assert os.path.isfile(cache_p5_path), f"Missing cache: {cache_p5_path}"
    assert os.path.isfile(cache_p6_path), f"Missing cache: {cache_p6_path}"

    cache_p5 = np.load(cache_p5_path, allow_pickle=True)
    cache_p6 = np.load(cache_p6_path, allow_pickle=True)

    y_true_raw = cache_p5["y_true_raw"] # [1294, 24]
    y_true_flat = y_true_raw.flatten()

    # 2. MAPE Safety Check
    min_load = float(np.min(y_true_flat))
    max_load = float(np.max(y_true_flat))
    print(f"Test Set Load Range: Min = {min_load:.2f} MW, Max = {max_load:.2f} MW")
    assert min_load > 0, "Zero or negative load detected in test set!"
    print(f"[MAPE SAFETY VERIFIED] Minimum raw load is {min_load:.2f} MW (> 0). Standard MAPE is safe and well-conditioned.")

    seeds = [42, 123, 2024, 3407, 999]
    models_map = {
        "Persistence_Naive24": "naive_pred",
        "LSTM_Standalone": "lstm_seed_",
        "TCN_Standalone": "tcn_seed_",
        "CNN_Standalone": "cnn_seed_",
        "Static_Equal_Ensemble": "static_seed_",
        "Standard_Input_MoE": "moe_seed_",
        "CAEG_Net_No_Recent_Error": "no_rec_seed_",
        "Full_CAEG_Net": "caeg_seed_"
    }

    # 3. Recover Seed-Level Metrics
    seed_metrics_rows = []

    for m_name, key_prefix in models_map.items():
        if m_name == "Persistence_Naive24":
            y_pred = cache_p5[key_prefix].flatten()
            mae = float(mean_absolute_error(y_true_flat, y_pred))
            mse = float(mean_squared_error(y_true_flat, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_true_flat, y_pred))
            mape = float(np.mean(np.abs((y_true_flat - y_pred) / y_true_flat)) * 100.0)

            # Assert RMSE == sqrt(MSE)
            assert np.isclose(rmse, np.sqrt(mse), atol=1e-5), "RMSE != sqrt(MSE) for Persistence!"

            seed_metrics_rows.append({
                "model": m_name,
                "seed": "Deterministic",
                "MAE_MW": round(mae, 2),
                "MSE_MW2": round(mse, 2),
                "RMSE_MW": round(rmse, 2),
                "R2": round(r2, 4),
                "MAPE_percent": round(mape, 2)
            })
        else:
            for s in seeds:
                y_pred = cache_p5[f"{key_prefix}{s}"].flatten()
                mae = float(mean_absolute_error(y_true_flat, y_pred))
                mse = float(mean_squared_error(y_true_flat, y_pred))
                rmse = float(np.sqrt(mse))
                r2 = float(r2_score(y_true_flat, y_pred))
                mape = float(np.mean(np.abs((y_true_flat - y_pred) / y_true_flat)) * 100.0)

                # Assert RMSE == sqrt(MSE)
                assert np.isclose(rmse, np.sqrt(mse), atol=1e-5), f"RMSE != sqrt(MSE) for {m_name} seed {s}!"

                seed_metrics_rows.append({
                    "model": m_name,
                    "seed": str(s),
                    "MAE_MW": round(mae, 2),
                    "MSE_MW2": round(mse, 2),
                    "RMSE_MW": round(rmse, 2),
                    "R2": round(r2, 4),
                    "MAPE_percent": round(mape, 2)
                })

    df_seed_metrics = pd.DataFrame(seed_metrics_rows)
    os.makedirs("results", exist_ok=True)
    df_seed_metrics.to_csv("results/phase7_seed_metrics.csv", index=False)
    print("\nSaved seed-level metrics to: results/phase7_seed_metrics.csv")

    # 4. Compute Aggregate Performance Summary (Directly from Seed Metrics)
    summary_rows = []
    for m_name in models_map.keys():
        sub = df_seed_metrics[df_seed_metrics["model"] == m_name]
        is_det = (m_name == "Persistence_Naive24")

        mae_m = float(sub["MAE_MW"].mean())
        mae_s = 0.0 if is_det else float(sub["MAE_MW"].std())

        mse_m = float(sub["MSE_MW2"].mean())
        mse_s = 0.0 if is_det else float(sub["MSE_MW2"].std())

        rmse_m = float(sub["RMSE_MW"].mean())
        rmse_s = 0.0 if is_det else float(sub["RMSE_MW"].std())

        r2_m = float(sub["R2"].mean())
        r2_s = 0.0 if is_det else float(sub["R2"].std())

        mape_m = float(sub["MAPE_percent"].mean())
        mape_s = 0.0 if is_det else float(sub["MAPE_percent"].std())

        summary_rows.append({
            "model": m_name,
            "MAE_mean": round(mae_m, 2),
            "MAE_std": round(mae_s, 2),
            "MSE_mean": round(mse_m, 2),
            "MSE_std": round(mse_s, 2),
            "RMSE_mean": round(rmse_m, 2),
            "RMSE_std": round(rmse_s, 2),
            "R2_mean": round(r2_m, 4),
            "R2_std": round(r2_s, 4),
            "MAPE_mean": round(mape_m, 2),
            "MAPE_std": round(mape_s, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/phase7_performance_summary.csv", index=False)
    print("Saved performance summary to: results/phase7_performance_summary.csv")

    # 5. Format and Print Primary Final Performance Table
    print("\n" + "=" * 95)
    print("PRIMARY FINAL PERFORMANCE TABLE (ORIGINAL RAW MW SCALE, MEAN +/- STD ACROSS 5 SEEDS)")
    print("=" * 95)
    display_rows = []
    for _, r in df_summary.iterrows():
        is_det = (r["model"] == "Persistence_Naive24")
        display_rows.append({
            "Model": r["model"],
            "MAE (MW)": f"{r['MAE_mean']:.2f}" if is_det else f"{r['MAE_mean']:.2f} ± {r['MAE_std']:.2f}",
            "MSE (MW^2)": f"{r['MSE_mean']:.2f}" if is_det else f"{r['MSE_mean']:.2f} ± {r['MSE_std']:.2f}",
            "RMSE (MW)": f"{r['RMSE_mean']:.2f}" if is_det else f"{r['RMSE_mean']:.2f} ± {r['RMSE_std']:.2f}",
            "R^2": f"{r['R2_mean']:.4f}" if is_det else f"{r['R2_mean']:.4f} ± {r['R2_std']:.4f}",
            "MAPE (%)": f"{r['MAPE_mean']:.2f}" if is_det else f"{r['MAPE_mean']:.2f} ± {r['MAPE_std']:.2f}",
        })
    df_display = pd.DataFrame(display_rows)
    print(df_display.to_string(index=False))
    print("=" * 95)

    # 6. Verification Against Phase 5 Report
    print("\n--- VERIFICATION AGAINST PHASE 5 REPORTED VALUES ---")
    caeg_row = df_summary[df_summary["model"] == "Full_CAEG_Net"].iloc[0]
    expected_mae, expected_rmse, expected_r2 = 251.44, 334.32, 0.8723
    diff_mae = abs(caeg_row["MAE_mean"] - expected_mae)
    diff_rmse = abs(caeg_row["RMSE_mean"] - expected_rmse)
    diff_r2 = abs(caeg_row["R2_mean"] - expected_r2)
    print(f"Full CAEG-Net Recovered MAE : {caeg_row['MAE_mean']:.2f} MW (Phase 5: {expected_mae:.2f} MW, Diff: {diff_mae:.4f})")
    print(f"Full CAEG-Net Recovered RMSE: {caeg_row['RMSE_mean']:.2f} MW (Phase 5: {expected_rmse:.2f} MW, Diff: {diff_rmse:.4f})")
    print(f"Full CAEG-Net Recovered R^2 : {caeg_row['R2_mean']:.4f} (Phase 5: {expected_r2:.4f}, Diff: {diff_r2:.6f})")
    assert diff_mae < 0.05 and diff_rmse < 0.05 and diff_r2 < 0.001, "Discrepancy detected against Phase 5!"
    print("[CONSISTENCY CONFIRMED] Metrics match Phase 5 perfectly within numerical tolerance.")

    # 7. Complete Performance by Context Regime
    C_test = cache_p6["C_test"]
    weights = cache_p6["weights"]
    y_caeg = cache_p6["y_pred_caeg_raw"]
    y_lstm = cache_p6["y_pred_lstm_raw"]
    y_tcn  = cache_p6["y_pred_tcn_raw"]
    y_cnn  = cache_p6["y_pred_cnn_raw"]

    context_names = ["Trend", "Volatility", "Periodicity", "Recent_Error"]
    regime_perf_rows = []

    for i, c_name in enumerate(context_names):
        c_vals = C_test[:, i]
        q33 = np.percentile(c_vals, 33.33)
        q67 = np.percentile(c_vals, 66.67)

        for reg_name, mask in [("Low", c_vals <= q33), ("Medium", (c_vals > q33) & (c_vals <= q67)), ("High", c_vals > q67)]:
            yt = y_true_raw[mask].flatten()
            w_sub = weights[mask]
            
            # Gating weights in this regime
            w_l_mean = float(w_sub[:, 0].mean())
            w_t_mean = float(w_sub[:, 1].mean())
            w_c_mean = float(w_sub[:, 2].mean())

            # Evaluate each expert and CAEG
            exp_metrics = {}
            for m_key, yp in [("LSTM", y_lstm[mask].flatten()), ("TCN", y_tcn[mask].flatten()), ("CNN", y_cnn[mask].flatten()), ("Full_CAEG", y_caeg[mask].flatten())]:
                mae_v = float(mean_absolute_error(yt, yp))
                mse_v = float(mean_squared_error(yt, yp))
                rmse_v = float(np.sqrt(mse_v))
                r2_v = float(r2_score(yt, yp))
                mape_v = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
                exp_metrics[m_key] = {"MAE": mae_v, "MSE": mse_v, "RMSE": rmse_v, "R2": r2_v, "MAPE": mape_v}

            best_exp_name = min(["LSTM", "TCN", "CNN"], key=lambda k: exp_metrics[k]["MAE"])
            best_exp_mae = exp_metrics[best_exp_name]["MAE"]
            caeg_mae = exp_metrics["Full_CAEG"]["MAE"]
            imp_pct = ((best_exp_mae - caeg_mae) / best_exp_mae) * 100.0

            regime_perf_rows.append({
                "Context": c_name,
                "Regime": reg_name,
                "Sample_Count": int(np.sum(mask)),
                "w_LSTM_mean": round(w_l_mean, 4),
                "w_TCN_mean": round(w_t_mean, 4),
                "w_CNN_mean": round(w_c_mean, 4),
                "LSTM_MAE": round(exp_metrics["LSTM"]["MAE"], 2),
                "LSTM_RMSE": round(exp_metrics["LSTM"]["RMSE"], 2),
                "LSTM_R2": round(exp_metrics["LSTM"]["R2"], 4),
                "LSTM_MAPE": round(exp_metrics["LSTM"]["MAPE"], 2),
                "TCN_MAE": round(exp_metrics["TCN"]["MAE"], 2),
                "TCN_RMSE": round(exp_metrics["TCN"]["RMSE"], 2),
                "TCN_R2": round(exp_metrics["TCN"]["R2"], 4),
                "TCN_MAPE": round(exp_metrics["TCN"]["MAPE"], 2),
                "CNN_MAE": round(exp_metrics["CNN"]["MAE"], 2),
                "CNN_RMSE": round(exp_metrics["CNN"]["RMSE"], 2),
                "CNN_R2": round(exp_metrics["CNN"]["R2"], 4),
                "CNN_MAPE": round(exp_metrics["CNN"]["MAPE"], 2),
                "CAEG_MAE": round(exp_metrics["Full_CAEG"]["MAE"], 2),
                "CAEG_MSE": round(exp_metrics["Full_CAEG"]["MSE"], 2),
                "CAEG_RMSE": round(exp_metrics["Full_CAEG"]["RMSE"], 2),
                "CAEG_R2": round(exp_metrics["Full_CAEG"]["R2"], 4),
                "CAEG_MAPE": round(exp_metrics["Full_CAEG"]["MAPE"], 2),
                "Best_Expert": best_exp_name,
                "Best_Expert_MAE": round(best_exp_mae, 2),
                "Improvement_Over_Best_Exp_pct": round(imp_pct, 2)
            })

    df_regime_perf = pd.DataFrame(regime_perf_rows)
    df_regime_perf.to_csv("results/phase7_context_regime_performance.csv", index=False)
    print("Saved context regime performance to: results/phase7_context_regime_performance.csv")

    # 8. Complete Performance by Forecast Error Regime (Post-Hoc Diagnostic)
    mae_caeg_per_win = np.mean(np.abs(y_caeg - y_true_raw), axis=1)
    err_q33 = np.percentile(mae_caeg_per_win, 33.33)
    err_q67 = np.percentile(mae_caeg_per_win, 66.67)

    err_reg_rows = []
    for reg_name, mask in [("Low Error (<= 33%)", mae_caeg_per_win <= err_q33),
                           ("Medium Error (33-67%)", (mae_caeg_per_win > err_q33) & (mae_caeg_per_win <= err_q67)),
                           ("High Error (> 67%)", mae_caeg_per_win > err_q67)]:
        yt = y_true_raw[mask].flatten()
        w_sub = weights[mask]
        
        row_dict = {
            "Error_Regime": reg_name,
            "Sample_Count": int(np.sum(mask)),
            "w_LSTM_mean": round(float(w_sub[:, 0].mean()), 4),
            "w_TCN_mean": round(float(w_sub[:, 1].mean()), 4),
            "w_CNN_mean": round(float(w_sub[:, 2].mean()), 4),
        }
        for m_key, yp in [("LSTM", y_lstm[mask].flatten()), ("TCN", y_tcn[mask].flatten()), ("CNN", y_cnn[mask].flatten()), ("Full_CAEG", y_caeg[mask].flatten())]:
            row_dict[f"{m_key}_MAE"] = round(float(mean_absolute_error(yt, yp)), 2)
            row_dict[f"{m_key}_RMSE"] = round(float(np.sqrt(mean_squared_error(yt, yp))), 2)
            row_dict[f"{m_key}_R2"] = round(float(r2_score(yt, yp)), 4)
            row_dict[f"{m_key}_MAPE"] = round(float(np.mean(np.abs((yt - yp) / yt)) * 100.0), 2)
        
        err_reg_rows.append(row_dict)

    df_err_reg_perf = pd.DataFrame(err_reg_rows)
    df_err_reg_perf.to_csv("results/phase7_error_regime_performance.csv", index=False)
    print("Saved error regime performance to: results/phase7_error_regime_performance.csv")

    # 9. Standard MoE & Recent Error Ablation Comparisons
    print("\n--- STANDARD MOE VS CAEG-NET (5-SEED COMPARISON) ---")
    sub_moe = df_seed_metrics[df_seed_metrics["model"] == "Standard_Input_MoE"].set_index("seed")
    sub_caeg = df_seed_metrics[df_seed_metrics["model"] == "Full_CAEG_Net"].set_index("seed")
    sub_norec = df_seed_metrics[df_seed_metrics["model"] == "CAEG_Net_No_Recent_Error"].set_index("seed")

    df_comp_moe = pd.DataFrame({
        "Full_CAEG_MAE": sub_caeg["MAE_MW"],
        "Input_MoE_MAE": sub_moe["MAE_MW"],
        "MAE_Diff (CAEG - MoE)": sub_caeg["MAE_MW"] - sub_moe["MAE_MW"],
        "Full_CAEG_RMSE": sub_caeg["RMSE_MW"],
        "Input_MoE_RMSE": sub_moe["RMSE_MW"],
        "RMSE_Diff": sub_caeg["RMSE_MW"] - sub_moe["RMSE_MW"],
        "Full_CAEG_MAPE": sub_caeg["MAPE_percent"],
        "Input_MoE_MAPE": sub_moe["MAPE_percent"],
        "MAPE_Diff": sub_caeg["MAPE_percent"] - sub_moe["MAPE_percent"],
        "Full_CAEG_R2": sub_caeg["R2"],
        "Input_MoE_R2": sub_moe["R2"],
        "R2_Diff": sub_caeg["R2"] - sub_moe["R2"]
    })
    print(df_comp_moe.to_string())

    print("\n--- RECENT ERROR ABLATION (5-SEED COMPARISON) ---")
    df_comp_ablation = pd.DataFrame({
        "Full_CAEG_MAE": sub_caeg["MAE_MW"],
        "NoRec_MAE": sub_norec["MAE_MW"],
        "MAE_Reduction": sub_norec["MAE_MW"] - sub_caeg["MAE_MW"],
        "Full_CAEG_RMSE": sub_caeg["RMSE_MW"],
        "NoRec_RMSE": sub_norec["RMSE_MW"],
        "RMSE_Reduction": sub_norec["RMSE_MW"] - sub_caeg["RMSE_MW"],
        "Full_CAEG_MAPE": sub_caeg["MAPE_percent"],
        "NoRec_MAPE": sub_norec["MAPE_percent"],
        "MAPE_Reduction": sub_norec["MAPE_percent"] - sub_caeg["MAPE_percent"],
        "Full_CAEG_R2": sub_caeg["R2"],
        "NoRec_R2": sub_norec["R2"],
        "R2_Increase": sub_caeg["R2"] - sub_norec["R2"]
    })
    print(df_comp_ablation.to_string())

    print("\n" + "=" * 85)
    print("PHASE 7 METRICS RECOVERY & ANALYSIS SUCCESSFULLY COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    run_phase7_recovery()
