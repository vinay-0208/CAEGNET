"""
Phase 6 Context, Routing & Error-Regime Analysis Script
======================================================
Investigates the internal behavior of CAEG-Net's context-adaptive gating mechanism.

Core Research Question:
"Does CAEG-Net actually adapt its expert combination according to forecasting context?"

Outputs:
- results/phase6_routing_analysis.csv
- results/phase6_correlations.csv
- results/phase6_regimes.csv
- results/phase6_expert_specialization.csv
- results/phase6_analysis_cache.npz
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
    TimeSeriesContextDataset,
    create_dataloaders
)
from caeg_net import CAEGNet, count_parameters
from evaluate import compute_metrics


def run_phase6_analysis():
    print("=" * 80)
    print("STARTING PHASE 6 CONTEXT, ROUTING & ERROR-REGIME ANALYSIS")
    print("=" * 80)

    # 1. Load Data and Pipeline (Preserved exactly from Phase 1-5)
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows)

    C_test = extract_context_features(windows["test"]["X"], rec_test)
    X_test = windows["test"]["X"]
    Y_test = windows["test"]["Y"]
    test_origins = windows["test"]["origins"]

    scale = scaler.scale_[0]
    mean = scaler.mean_[0]

    # Align timestamps for forecast origins
    # test_df index begins at split boundary; test_origins are relative to the test continuous array
    test_start_dt = test_df["timestamp"].iloc[0]
    origin_timestamps = [test_start_dt + pd.Timedelta(hours=int(idx)) for idx in test_origins]

    # 2. Load Trained Full CAEG-Net Model (Seed 42 / Representative Primary Checkpoint)
    ckpt_path = "checkpoints/seed_42/caeg_full.pt"
    assert os.path.isfile(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    model = CAEGNet(input_dim=1, horizon=24, context_dim=4, latent_context_dim=16)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # 3. Extract Forward Pass Predictions, Diagnostical Expert Outputs, and Routing Weights
    bx = torch.tensor(X_test, dtype=torch.float32)
    bc = torch.tensor(C_test, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled, weights_tensor, expert_dict = model(bx, bc, return_diagnostics=True)

    weights = weights_tensor.cpu().numpy() # [1294, 3]
    y_pred_caeg_raw = y_pred_scaled.cpu().numpy() * scale + mean
    y_pred_lstm_raw = expert_dict["lstm"].cpu().numpy() * scale + mean
    y_pred_tcn_raw = expert_dict["tcn"].cpu().numpy() * scale + mean
    y_pred_cnn_raw = expert_dict["cnn"].cpu().numpy() * scale + mean
    y_true_raw = Y_test * scale + mean

    # Per-window MAE (MW)
    mae_caeg = np.mean(np.abs(y_pred_caeg_raw - y_true_raw), axis=1)
    mae_lstm = np.mean(np.abs(y_pred_lstm_raw - y_true_raw), axis=1)
    mae_tcn = np.mean(np.abs(y_pred_tcn_raw - y_true_raw), axis=1)
    mae_cnn = np.mean(np.abs(y_pred_cnn_raw - y_true_raw), axis=1)

    # Verify routing weights properties
    assert np.allclose(weights.sum(axis=1), np.ones(len(weights)), atol=1e-4)
    assert (weights >= 0.0).all()

    # 4. Save results/phase6_routing_analysis.csv
    df_routing_full = pd.DataFrame({
        "forecast_origin_idx": test_origins,
        "timestamp": origin_timestamps,
        "Trend": C_test[:, 0],
        "Volatility": C_test[:, 1],
        "Periodicity": C_test[:, 2],
        "Recent_Error": C_test[:, 3],
        "w_LSTM": weights[:, 0],
        "w_TCN": weights[:, 1],
        "w_CNN": weights[:, 2],
        "MAE_CAEG (MW)": mae_caeg,
        "MAE_LSTM (MW)": mae_lstm,
        "MAE_TCN (MW)": mae_tcn,
        "MAE_CNN (MW)": mae_cnn,
    })
    os.makedirs("results", exist_ok=True)
    df_routing_full.to_csv("results/phase6_routing_analysis.csv", index=False)
    print("Saved routing analysis to: results/phase6_routing_analysis.csv")

    # 5. Context Feature Distribution Statistics
    context_stats = []
    context_names = ["Trend", "Volatility", "Periodicity", "Recent_Error"]
    for i, c_name in enumerate(context_names):
        vals = C_test[:, i]
        context_stats.append({
            "Feature": c_name,
            "Mean": np.mean(vals),
            "Std": np.std(vals),
            "Min": np.min(vals),
            "Q25": np.percentile(vals, 25),
            "Median": np.median(vals),
            "Q75": np.percentile(vals, 75),
            "Max": np.max(vals),
        })
    df_context_stats = pd.DataFrame(context_stats)
    print("\n--- CONTEXT FEATURE DISTRIBUTION STATISTICS (TEST SET) ---")
    print(df_context_stats.round(4).to_string(index=False))

    # 6. Context vs Routing Relationships (Pearson & Spearman Correlations)
    corr_rows = []
    weight_names = ["w_LSTM", "w_TCN", "w_CNN"]
    for i, c_name in enumerate(context_names):
        c_vals = C_test[:, i]
        row = {"Context_Feature": c_name}
        for j, w_name in enumerate(weight_names):
            w_vals = weights[:, j]
            p_corr, p_pval = stats.pearsonr(c_vals, w_vals)
            s_corr, s_pval = stats.spearmanr(c_vals, w_vals)
            row[f"{w_name}_Pearson"] = round(p_corr, 4)
            row[f"{w_name}_Spearman"] = round(s_corr, 4)
        corr_rows.append(row)
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv("results/phase6_correlations.csv", index=False)
    print("\n--- CONTEXT VS ROUTING WEIGHT CORRELATIONS ---")
    print(df_corr.to_string(index=False))

    # 7. Context Regime Analysis (Low, Medium, High Terciles)
    regime_rows = []
    for i, c_name in enumerate(context_names):
        c_vals = C_test[:, i]
        q33 = np.percentile(c_vals, 33.33)
        q67 = np.percentile(c_vals, 66.67)

        low_mask = c_vals <= q33
        med_mask = (c_vals > q33) & (c_vals <= q67)
        high_mask = c_vals > q67

        for reg_name, mask in [("Low (<=33%)", low_mask), ("Medium (33-67%)", med_mask), ("High (>67%)", high_mask)]:
            regime_rows.append({
                "Context_Feature": c_name,
                "Regime": reg_name,
                "Sample_Count": int(np.sum(mask)),
                "w_LSTM_Mean": round(float(weights[mask, 0].mean()), 4),
                "w_TCN_Mean": round(float(weights[mask, 1].mean()), 4),
                "w_CNN_Mean": round(float(weights[mask, 2].mean()), 4),
            })
    df_regimes = pd.DataFrame(regime_rows)
    df_regimes.to_csv("results/phase6_regimes.csv", index=False)
    print("\n--- CONTEXT REGIME ANALYSIS (MEAN ROUTING WEIGHTS) ---")
    print(df_regimes.to_string(index=False))

    # 8. Forecast Error Regime Analysis (Post-Hoc Actual Difficulty Terciles)
    err_q33 = np.percentile(mae_caeg, 33.33)
    err_q67 = np.percentile(mae_caeg, 66.67)
    err_low_mask = mae_caeg <= err_q33
    err_med_mask = (mae_caeg > err_q33) & (mae_caeg <= err_q67)
    err_high_mask = mae_caeg > err_q67

    err_regime_rows = []
    for reg_name, mask in [("Low Error (<=33%)", err_low_mask), ("Medium Error (33-67%)", err_med_mask), ("High Error (>67%)", err_high_mask)]:
        err_regime_rows.append({
            "Error_Regime": reg_name,
            "Threshold (MW)": f"<= {err_q33:.1f}" if "Low" in reg_name else (f"{err_q33:.1f} - {err_q67:.1f}" if "Med" in reg_name else f"> {err_q67:.1f}"),
            "Sample_Count": int(np.sum(mask)),
            "w_LSTM_Mean": round(float(weights[mask, 0].mean()), 4),
            "w_TCN_Mean": round(float(weights[mask, 1].mean()), 4),
            "w_CNN_Mean": round(float(weights[mask, 2].mean()), 4),
            "CAEG_MAE (MW)": round(float(mae_caeg[mask].mean()), 2),
        })
    df_err_regimes = pd.DataFrame(err_regime_rows)
    print("\n--- FORECAST ERROR REGIME ANALYSIS (POST-HOC COMPLETED ERROR) ---")
    print(df_err_regimes.to_string(index=False))

    # 9. Expert Specialization Analysis (Standalone Expert Errors across Context Regimes)
    spec_rows = []
    for i, c_name in enumerate(context_names):
        c_vals = C_test[:, i]
        q33 = np.percentile(c_vals, 33.33)
        q67 = np.percentile(c_vals, 66.67)
        for reg_name, mask in [("Low", c_vals <= q33), ("Medium", (c_vals > q33) & (c_vals <= q67)), ("High", c_vals > q67)]:
            best_exp = "TCN" if mae_tcn[mask].mean() < min(mae_lstm[mask].mean(), mae_cnn[mask].mean()) else ("LSTM" if mae_lstm[mask].mean() < mae_cnn[mask].mean() else "CNN")
            highest_w = "LSTM" if weights[mask, 0].mean() > max(weights[mask, 1].mean(), weights[mask, 2].mean()) else ("TCN" if weights[mask, 1].mean() > weights[mask, 2].mean() else "CNN")
            spec_rows.append({
                "Context": c_name,
                "Regime": reg_name,
                "LSTM_MAE": round(float(mae_lstm[mask].mean()), 2),
                "TCN_MAE": round(float(mae_tcn[mask].mean()), 2),
                "CNN_MAE": round(float(mae_cnn[mask].mean()), 2),
                "CAEG_MAE": round(float(mae_caeg[mask].mean()), 2),
                "Best_Expert": best_exp,
                "Highest_Weight": highest_w,
                "Aligned": best_exp == highest_w
            })
    df_spec = pd.DataFrame(spec_rows)
    df_spec.to_csv("results/phase6_expert_specialization.csv", index=False)
    print("\n--- EXPERT SPECIALIZATION ANALYSIS ---")
    print(df_spec.to_string(index=False))

    # 10. Post-Hoc Linear Regression on the Gate: Can Context Explain Gating Weights?
    ols_lstm = LinearRegression().fit(C_test, weights[:, 0])
    ols_tcn = LinearRegression().fit(C_test, weights[:, 1])
    ols_cnn = LinearRegression().fit(C_test, weights[:, 2])
    print("\n--- POST-HOC LINEAR EXPLANATORY CAPACITY OF THE GATE (OLS R^2) ---")
    print(f"w_LSTM R^2: {ols_lstm.score(C_test, weights[:, 0]):.4f}")
    print(f"w_TCN  R^2: {ols_tcn.score(C_test, weights[:, 1]):.4f}")
    print(f"w_CNN  R^2: {ols_cnn.score(C_test, weights[:, 2]):.4f}")

    # 11. Objective Identification of Case Studies:
    # A. Peak Demand: origin with highest maximum load in horizon
    peak_idx = int(np.argmax(np.max(y_true_raw, axis=1)))
    # B. Rapid Upward Ramp: origin with steepest positive 1h target diff
    up_ramps = np.max(np.diff(y_true_raw, axis=1), axis=1)
    up_idx = int(np.argmax(up_ramps))
    # C. Rapid Downward Ramp: origin with steepest negative 1h target diff
    down_ramps = np.min(np.diff(y_true_raw, axis=1), axis=1)
    down_idx = int(np.argmin(down_ramps))
    # D. High Volatility: origin with highest context volatility
    high_vol_idx = int(np.argmax(C_test[:, 1]))
    # E. Low Volatility: origin with lowest context volatility
    low_vol_idx = int(np.argmin(C_test[:, 1]))

    case_studies = {
        "Peak_Demand": peak_idx,
        "Rapid_Upward_Ramp": up_idx,
        "Rapid_Downward_Ramp": down_idx,
        "High_Volatility": high_vol_idx,
        "Low_Volatility": low_vol_idx,
    }
    print("\n--- PRE-DEFINED OBJECTIVE CASE STUDIES ---")
    for k, idx in case_studies.items():
        print(f"  {k:20s}: Origin index {idx:4d} (Timestamp: {origin_timestamps[idx]})")

    # 12. Save cache for notebook plotting
    np.savez(
        "results/phase6_analysis_cache.npz",
        C_test=C_test,
        weights=weights,
        y_true_raw=y_true_raw,
        y_pred_caeg_raw=y_pred_caeg_raw,
        y_pred_lstm_raw=y_pred_lstm_raw,
        y_pred_tcn_raw=y_pred_tcn_raw,
        y_pred_cnn_raw=y_pred_cnn_raw,
        case_studies_json=json.dumps(case_studies)
    )
    print("\nPhase 6 analysis complete and cache saved successfully.")


if __name__ == "__main__":
    run_phase6_analysis()
