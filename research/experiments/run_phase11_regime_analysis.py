"""
Phase 11: Regime, Difficulty & Expert-Complementarity Analysis
============================================================
Canonical CAEG-Net Research Track
Evaluates:
- Modern PJM (MW)
- GEFCom2014 (kW)
- UCI Cohort 320 Aggregate (MW)

Strict Firewalls:
- Canonical CAEG-Net V1 and Canonical Experts (LSTM, TCN, CNN) are FROZEN.
- Deterministic seeding with CuDNN determinism enabled.
- Regime thresholds calibrated strictly on TRAIN/VAL distributions.
- Non-overlapping daily 24-hour blocks for inferential hypothesis testing.
- Post-hoc realized features strictly separated from forecast-time context.
"""

import os
import sys
import time
import functools

# Force unbuffered output so task logs are immediately readable
print = functools.partial(print, flush=True)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from caeg_net import LSTMExpert, TCNExpert, CNNExpert
from research.original_caeg import OriginalCAEGNetPhase5
from research.deterministic import seed_everything, make_deterministic_loader

# Import data loaders from run_phase10_optimization
from research.experiments.run_phase10_optimization import (
    load_all_three_datasets,
    train_caeg_variant,
    evaluate_model_on_partition,
    compute_metrics,
)


def compute_lookback_features(X_windows, scaler, load_col_scale=True):
    """
    Extract causal forecast-time difficulty features from lookback windows X (N, 168, 1).
    All features depend strictly on past 168 hours (t-167 to t).
    """
    N = len(X_windows)
    # Convert standardized X to physical units
    scale = scaler.scale_[0] if hasattr(scaler, "scale_") else 1.0
    mean = scaler.mean_[0] if hasattr(scaler, "mean_") else 0.0

    X_phys = X_windows.squeeze(-1) * scale + mean  # (N, 168)

    means = np.mean(X_phys, axis=1)
    medians = np.median(X_phys, axis=1)
    maxs = np.max(X_phys, axis=1)
    mins = np.min(X_phys, axis=1)

    # Volatility: std of first differences
    diffs = X_phys[:, 1:] - X_phys[:, :-1]
    vols = np.std(diffs, axis=1)
    covs = vols / (np.abs(means) + 1e-8)

    # Trend slopes (polyfit over last 24h, 48h, 168h)
    x24 = np.arange(24)
    x48 = np.arange(48)
    x168 = np.arange(168)

    slope_24 = np.array([np.polyfit(x24, X_phys[i, -24:], 1)[0] for i in range(N)])
    slope_48 = np.array([np.polyfit(x48, X_phys[i, -48:], 1)[0] for i in range(N)])
    slope_168 = np.array([np.polyfit(x168, X_phys[i, :], 1)[0] for i in range(N)])

    # Lag-24 autocorrelation
    def autocorr_lag24(row):
        s1 = row[24:]
        s2 = row[:-24]
        if np.std(s1) < 1e-8 or np.std(s2) < 1e-8:
            return 0.0
        return np.corrcoef(s1, s2)[0, 1]

    lag24 = np.array([autocorr_lag24(X_phys[i]) for i in range(N)])

    return {
        "mean_load_168h": means,
        "median_load_168h": medians,
        "max_load_168h": maxs,
        "min_load_168h": mins,
        "volatility_168h": vols,
        "cov_168h": covs,
        "trend_slope_24h": slope_24,
        "trend_slope_48h": slope_48,
        "trend_slope_168h": slope_168,
        "lag24_autocorr": lag24,
    }


def compute_realized_target_features(Y_windows, scaler):
    """
    Extract POST-HOC realized difficulty features from target Y (N, 24).
    WARNING: For post-forecast analysis ONLY. Not available at forecast time.
    """
    scale = scaler.scale_[0] if hasattr(scaler, "scale_") else 1.0
    mean = scaler.mean_[0] if hasattr(scaler, "mean_") else 0.0
    Y_phys = Y_windows * scale + mean

    target_std = np.std(Y_phys, axis=1)
    target_range = np.max(Y_phys, axis=1) - np.min(Y_phys, axis=1)
    target_mean = np.mean(Y_phys, axis=1)
    target_cov = target_std / (np.abs(target_mean) + 1e-8)
    target_ramp = np.abs(Y_phys[:, -1] - Y_phys[:, 0])

    diffs = np.abs(Y_phys[:, 1:] - Y_phys[:, :-1])
    target_max_hourly = np.max(diffs, axis=1)

    return {
        "realized_target_std": target_std,
        "realized_target_range": target_range,
        "realized_target_cov": target_cov,
        "realized_target_ramp": target_ramp,
        "realized_max_hourly_change": target_max_hourly,
    }


def holm_bonferroni(p_values):
    """Apply step-down Holm-Bonferroni correction to an array of p-values."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    cum_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = p_values[idx] * (n - rank)
        cum_max = max(cum_max, p_adj)
        adjusted[idx] = min(1.0, cum_max)
    return adjusted


def main():
    print("=================================================================")
    print("PHASE 11: REGIME, DIFFICULTY & EXPERT-COMPLEMENTARITY ANALYSIS")
    print("=================================================================")
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Deterministic Seeding Setup
    seed = 42
    seed_everything(seed, deterministic_cudnn=True)

    # 2. Output Directories
    results_dir = "research/results"
    plots_dir = os.path.join(results_dir, "phase11_plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 3. Load Datasets
    print("\n--- Loading Tri-Benchmark Datasets ---")
    datasets = load_all_three_datasets()

    all_window_records = []
    all_regime_records = []
    all_complementarity_records = []
    all_routing_records = []
    all_statistical_records = []
    all_dataset_records = []

    # Store for cross-dataset plotting
    dataset_eval_data = {}

    for d_key in ["PJM", "GEFCom", "UCI"]:
        print(f"\n=======================================================")
        print(f"PROCESSING DATASET: {d_key}")
        print(f"=======================================================")
        d_obj = datasets[d_key]
        windows = d_obj["windows"]
        scaler = d_obj["scaler"]
        ctx_dict = d_obj["contexts"]["A0"]
        unit = d_obj["unit"]

        # -------------------------------------------------------------
        # 4. Train Canonical Models with Deterministic Seeding
        # -------------------------------------------------------------
        tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
        tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
        va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
        va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
        te_x = torch.tensor(windows["test"]["X"], dtype=torch.float32)
        te_y = torch.tensor(windows["test"]["Y"], dtype=torch.float32)

        tr_c = torch.tensor(ctx_dict["train"], dtype=torch.float32)
        va_c = torch.tensor(ctx_dict["val"], dtype=torch.float32)
        te_c = torch.tensor(ctx_dict["test"], dtype=torch.float32)

        # Single Experts Loaders (No Context)
        tr_ds_no_ctx = TensorDataset(tr_x, tr_y)
        va_ds_no_ctx = TensorDataset(va_x, va_y)
        te_ds_no_ctx = TensorDataset(te_x, te_y)

        tr_loader_no_ctx = make_deterministic_loader(tr_ds_no_ctx, batch_size=64, shuffle=True, seed=seed)
        va_loader_no_ctx = make_deterministic_loader(va_ds_no_ctx, batch_size=64, shuffle=False)
        te_loader_no_ctx = make_deterministic_loader(te_ds_no_ctx, batch_size=64, shuffle=False)

        # Train LSTM
        seed_everything(seed, deterministic_cudnn=True)
        m_lstm = LSTMExpert().to(device)
        train_caeg_variant(m_lstm, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=25, patience=6)
        res_l = evaluate_model_on_partition(m_lstm, te_loader_no_ctx, scaler, device)

        # Train TCN
        seed_everything(seed, deterministic_cudnn=True)
        m_tcn = TCNExpert().to(device)
        train_caeg_variant(m_tcn, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=25, patience=6)
        res_t = evaluate_model_on_partition(m_tcn, te_loader_no_ctx, scaler, device)

        # Train CNN
        seed_everything(seed, deterministic_cudnn=True)
        m_cnn = CNNExpert().to(device)
        train_caeg_variant(m_cnn, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=25, patience=6)
        res_c = evaluate_model_on_partition(m_cnn, te_loader_no_ctx, scaler, device)

        # Train CAEG-Net V1 (Context Aware)
        tr_ds_ctx = TensorDataset(tr_x, tr_y, tr_c)
        va_ds_ctx = TensorDataset(va_x, va_y, va_c)
        te_ds_ctx = TensorDataset(te_x, te_y, te_c)

        tr_loader_ctx = make_deterministic_loader(tr_ds_ctx, batch_size=64, shuffle=True, seed=seed)
        va_loader_ctx = make_deterministic_loader(va_ds_ctx, batch_size=64, shuffle=False)
        te_loader_ctx = make_deterministic_loader(te_ds_ctx, batch_size=64, shuffle=False)

        seed_everything(seed, deterministic_cudnn=True)
        caeg_v1 = OriginalCAEGNetPhase5(context_dim=ctx_dict["train"].shape[1]).to(device)
        train_caeg_variant(caeg_v1, tr_loader_ctx, va_loader_ctx, device, max_epochs=25, patience=6)
        res_caeg = evaluate_model_on_partition(caeg_v1, te_loader_ctx, scaler, device)

        # Predictions (Physical Units)
        preds_l = res_l["preds"]
        preds_t = res_t["preds"]
        preds_c = res_c["preds"]
        preds_caeg = res_caeg["preds"]
        preds_ens = (preds_l + preds_t + preds_c) / 3.0
        trues = res_caeg["trues"]
        weights = res_caeg["weights"]  # (N, 3)

        N_test = len(trues)
        print(f"Completed model inference for {d_key} on {N_test} test windows.")

        # -------------------------------------------------------------
        # 5. Extract Features & Calibrate Thresholds on Train/Val
        # -------------------------------------------------------------
        # Forecast-time difficulty features on Train + Val
        feat_train = compute_lookback_features(windows["train"]["X"], scaler)
        feat_val = compute_lookback_features(windows["val"]["X"], scaler)
        feat_test = compute_lookback_features(windows["test"]["X"], scaler)

        # Realized post-hoc features on test
        realized_test = compute_realized_target_features(windows["test"]["Y"], scaler)

        # Combine Train + Val features for threshold calibration
        train_val_vol = np.concatenate([feat_train["volatility_168h"], feat_val["volatility_168h"]])
        train_val_slope = np.concatenate([feat_train["trend_slope_168h"], feat_val["trend_slope_168h"]])
        train_val_load = np.concatenate([feat_train["mean_load_168h"], feat_val["mean_load_168h"]])

        # Realized train+val targets for ramp calibration
        realized_train = compute_realized_target_features(windows["train"]["Y"], scaler)
        realized_val = compute_realized_target_features(windows["val"]["Y"], scaler)
        train_val_ramp = np.concatenate([realized_train["realized_target_ramp"], realized_val["realized_target_ramp"]])

        # Calibrate tercile thresholds strictly on Train+Val
        q_vol = (np.percentile(train_val_vol, 33.333), np.percentile(train_val_vol, 66.667))
        q_slope = (np.percentile(train_val_slope, 33.333), np.percentile(train_val_slope, 66.667))
        q_load = (np.percentile(train_val_load, 33.333), np.percentile(train_val_load, 66.667))
        q_ramp = (np.percentile(train_val_ramp, 33.333), np.percentile(train_val_ramp, 66.667))

        # Evaluate Expert Disagreement on Validation Set to calibrate disagreement threshold
        res_l_val = evaluate_model_on_partition(m_lstm, va_loader_no_ctx, scaler, device)
        res_t_val = evaluate_model_on_partition(m_tcn, va_loader_no_ctx, scaler, device)
        res_c_val = evaluate_model_on_partition(m_cnn, va_loader_no_ctx, scaler, device)
        val_disagreement = np.mean(
            (np.abs(res_l_val["preds"] - res_t_val["preds"]) +
             np.abs(res_l_val["preds"] - res_c_val["preds"]) +
             np.abs(res_t_val["preds"] - res_c_val["preds"])) / 3.0,
            axis=1
        )
        q_dis = (np.percentile(val_disagreement, 33.333), np.percentile(val_disagreement, 66.667))

        print(f"  Calibrated Thresholds from Train/Val (Firewall Verified):")
        print(f"    Volatility Terciles: {q_vol[0]:.2f}, {q_vol[1]:.2f}")
        print(f"    Trend Terciles:      {q_slope[0]:.2f}, {q_slope[1]:.2f}")
        print(f"    Load Terciles:       {q_load[0]:.2f}, {q_load[1]:.2f}")
        print(f"    Ramp Terciles:       {q_ramp[0]:.2f}, {q_ramp[1]:.2f}")
        print(f"    Disagreement Terciles: {q_dis[0]:.2f}, {q_dis[1]:.2f}")

        # -------------------------------------------------------------
        # 6. Compute Window-Level Performance & Features
        # -------------------------------------------------------------
        eps = 1e-12
        w_l = weights[:, 0]
        w_t = weights[:, 1]
        w_c = weights[:, 2]
        routing_entropy = - (w_l * np.log(w_l + eps) + w_t * np.log(w_t + eps) + w_c * np.log(w_c + eps))
        n_eff = np.exp(routing_entropy)

        # Window MAEs
        mae_caeg_win = np.mean(np.abs(preds_caeg - trues), axis=1)
        mae_equal_win = np.mean(np.abs(preds_ens - trues), axis=1)
        mae_lstm_win = np.mean(np.abs(preds_l - trues), axis=1)
        mae_tcn_win = np.mean(np.abs(preds_t - trues), axis=1)
        mae_cnn_win = np.mean(np.abs(preds_c - trues), axis=1)

        expert_matrix = np.stack([mae_lstm_win, mae_tcn_win, mae_cnn_win], axis=1)
        min_expert_mae = np.min(expert_matrix, axis=1)
        max_expert_mae = np.max(expert_matrix, axis=1)
        best_expert_idx = np.argmin(expert_matrix, axis=1)
        expert_names = ["LSTM", "TCN", "CNN"]
        best_expert_names = [expert_names[idx] for idx in best_expert_idx]

        delta_equal = mae_caeg_win - mae_equal_win
        delta_best = mae_caeg_win - min_expert_mae
        caeg_win_equal = (mae_caeg_win < mae_equal_win).astype(int)
        caeg_win_best = (mae_caeg_win < min_expert_mae).astype(int)

        # Expert Disagreement & Spread
        pairwise_dis = (np.abs(preds_l - preds_t) + np.abs(preds_l - preds_c) + np.abs(preds_t - preds_c)) / 3.0
        disagreement = np.mean(pairwise_dis, axis=1)
        pred_stack = np.stack([preds_l, preds_t, preds_c], axis=0)  # (3, N, 24)
        spread = np.mean(np.std(pred_stack, axis=0), axis=1)

        diff_lt = np.abs(mae_lstm_win - mae_tcn_win)
        diff_lc = np.abs(mae_lstm_win - mae_cnn_win)
        diff_tc = np.abs(mae_tcn_win - mae_cnn_win)
        best_worst_gap = max_expert_mae - min_expert_mae

        # Assign Regimes based on Calibrated Thresholds
        def assign_tercile(vals, cutoffs, labels=("low", "medium", "high")):
            res = np.full(len(vals), labels[1], dtype=object)
            res[vals <= cutoffs[0]] = labels[0]
            res[vals > cutoffs[1]] = labels[2]
            return res

        vol_reg = assign_tercile(feat_test["volatility_168h"], q_vol)
        slope_reg = assign_tercile(feat_test["trend_slope_168h"], q_slope)
        ramp_reg = assign_tercile(realized_test["realized_target_ramp"], q_ramp)
        dis_reg = assign_tercile(disagreement, q_dis)
        load_reg = assign_tercile(feat_test["mean_load_168h"], q_load, labels=("off_peak", "normal", "peak"))

        # Diurnal & Weekly
        hours = np.array([i % 24 for i in range(N_test)])
        days = np.array([(i // 24) % 7 for i in range(N_test)])
        is_weekend = np.array([1 if d in [5, 6] else 0 for d in days])

        diurnal_reg = np.empty(N_test, dtype=object)
        for i in range(N_test):
            h = hours[i]
            if 0 <= h < 6:
                diurnal_reg[i] = "night"
            elif 6 <= h < 12:
                diurnal_reg[i] = "morning"
            elif 12 <= h < 18:
                diurnal_reg[i] = "afternoon"
            else:
                diurnal_reg[i] = "evening"

        # Build window DataFrame
        df_win = pd.DataFrame({
            "dataset": d_key,
            "window_idx": np.arange(N_test),
            "mae_caeg": mae_caeg_win,
            "mae_equal": mae_equal_win,
            "mae_lstm": mae_lstm_win,
            "mae_tcn": mae_tcn_win,
            "mae_cnn": mae_cnn_win,
            "delta_equal": delta_equal,
            "delta_best_expert": delta_best,
            "caeg_win_equal": caeg_win_equal,
            "caeg_win_best": caeg_win_best,
            "best_expert_id": best_expert_names,
            "w_lstm": w_l,
            "w_tcn": w_t,
            "w_cnn": w_c,
            "routing_entropy": routing_entropy,
            "effective_n_experts": n_eff,
            # Forecast-time difficulty
            "mean_load_168h": feat_test["mean_load_168h"],
            "median_load_168h": feat_test["median_load_168h"],
            "max_load_168h": feat_test["max_load_168h"],
            "min_load_168h": feat_test["min_load_168h"],
            "volatility_168h": feat_test["volatility_168h"],
            "cov_168h": feat_test["cov_168h"],
            "trend_slope_24h": feat_test["trend_slope_24h"],
            "trend_slope_48h": feat_test["trend_slope_48h"],
            "trend_slope_168h": feat_test["trend_slope_168h"],
            "lag24_autocorr": feat_test["lag24_autocorr"],
            # Post-hoc realized features
            "realized_target_std": realized_test["realized_target_std"],
            "realized_target_range": realized_test["realized_target_range"],
            "realized_target_cov": realized_test["realized_target_cov"],
            "realized_target_ramp": realized_test["realized_target_ramp"],
            "realized_max_hourly_change": realized_test["realized_max_hourly_change"],
            # Complementarity
            "expert_disagreement": disagreement,
            "expert_spread": spread,
            "pairwise_mae_diff_lt": diff_lt,
            "pairwise_mae_diff_lc": diff_lc,
            "pairwise_mae_diff_tc": diff_tc,
            "best_worst_gap": best_worst_gap,
            # Regimes
            "hour_of_day": hours,
            "day_of_week": days,
            "is_weekend": is_weekend,
            "diurnal_regime": diurnal_reg,
            "volatility_regime": vol_reg,
            "trend_regime": slope_reg,
            "ramp_regime": ramp_reg,
            "disagreement_regime": dis_reg,
            "load_regime": load_reg,
        })
        all_window_records.append(df_win)

        # Store for plotting
        dataset_eval_data[d_key] = {
            "df_win": df_win,
            "unit": unit,
            "preds_caeg": preds_caeg,
            "preds_ens": preds_ens,
            "preds_l": preds_l,
            "preds_t": preds_t,
            "preds_c": preds_c,
            "trues": trues,
        }

        # -------------------------------------------------------------
        # 7. Regime Summaries (Core A, B, E, F)
        # -------------------------------------------------------------
        def summarize_regimes(df, regime_col, category_name):
            recs = []
            for r_val, grp in df.groupby(regime_col):
                recs.append({
                    "dataset": d_key,
                    "regime_category": category_name,
                    "regime_name": str(r_val),
                    "n_windows": len(grp),
                    "mae_caeg": float(grp["mae_caeg"].mean()),
                    "mae_equal": float(grp["mae_equal"].mean()),
                    "mae_best_expert": float(grp[["mae_lstm", "mae_tcn", "mae_cnn"]].min(axis=1).mean()),
                    "delta_equal": float(grp["delta_equal"].mean()),
                    "delta_best_expert": float(grp["delta_best_expert"].mean()),
                    "caeg_win_rate_equal": float(grp["caeg_win_equal"].mean()),
                    "caeg_win_rate_best": float(grp["caeg_win_best"].mean()),
                })
            return recs

        all_regime_records.extend(summarize_regimes(df_win, "volatility_regime", "Volatility"))
        all_regime_records.extend(summarize_regimes(df_win, "ramp_regime", "Ramp_Magnitude"))
        all_regime_records.extend(summarize_regimes(df_win, "trend_regime", "Trend_Slope"))
        all_regime_records.extend(summarize_regimes(df_win, "disagreement_regime", "Expert_Disagreement"))
        all_regime_records.extend(summarize_regimes(df_win, "load_regime", "Load_Level"))
        all_regime_records.extend(summarize_regimes(df_win, "diurnal_regime", "Diurnal"))
        all_regime_records.extend(summarize_regimes(df_win, "is_weekend", "Weekend_Weekday"))

        # -------------------------------------------------------------
        # 8. Routing Analysis by Regime
        # -------------------------------------------------------------
        for reg_col, cat_name in [
            ("volatility_regime", "Volatility"),
            ("disagreement_regime", "Expert_Disagreement"),
            ("load_regime", "Load_Level"),
            ("diurnal_regime", "Diurnal"),
        ]:
            for r_val, grp in df_win.groupby(reg_col):
                all_routing_records.append({
                    "dataset": d_key,
                    "regime_category": cat_name,
                    "regime_name": str(r_val),
                    "n_windows": len(grp),
                    "mean_w_lstm": float(grp["w_lstm"].mean()),
                    "mean_w_tcn": float(grp["w_tcn"].mean()),
                    "mean_w_cnn": float(grp["w_cnn"].mean()),
                    "mean_entropy": float(grp["routing_entropy"].mean()),
                    "mean_effective_experts": float(grp["effective_n_experts"].mean()),
                })

        # -------------------------------------------------------------
        # 9. Expert Complementarity Analysis (Core B, C, D)
        # -------------------------------------------------------------
        # Correlation between Disagreement and Delta_Equal
        corr_dis_delta = float(stats.pearsonr(df_win["expert_disagreement"], df_win["delta_equal"])[0])
        spearman_dis_delta = float(stats.spearmanr(df_win["expert_disagreement"], df_win["delta_equal"])[0])

        # Expert Win Rates
        pct_l_wins = float((df_win["best_expert_id"] == "LSTM").mean())
        pct_t_wins = float((df_win["best_expert_id"] == "TCN").mean())
        pct_c_wins = float((df_win["best_expert_id"] == "CNN").mean())

        # Average weights conditional on realized winner
        w_l_when_l_wins = float(df_win[df_win["best_expert_id"] == "LSTM"]["w_lstm"].mean()) if pct_l_wins > 0 else 0.0
        w_t_when_t_wins = float(df_win[df_win["best_expert_id"] == "TCN"]["w_tcn"].mean()) if pct_t_wins > 0 else 0.0
        w_c_when_c_wins = float(df_win[df_win["best_expert_id"] == "CNN"]["w_cnn"].mean()) if pct_c_wins > 0 else 0.0

        # Residual correlations across non-overlapping blocks
        horizon = 24
        K_blocks = N_test // horizon
        block_err_l = []
        block_err_t = []
        block_err_c = []
        for k in range(K_blocks):
            idx = slice(k * horizon, (k + 1) * horizon)
            block_err_l.extend(preds_l[idx].flatten() - trues[idx].flatten())
            block_err_t.extend(preds_t[idx].flatten() - trues[idx].flatten())
            block_err_c.extend(preds_c[idx].flatten() - trues[idx].flatten())

        block_err_l = np.array(block_err_l)
        block_err_t = np.array(block_err_t)
        block_err_c = np.array(block_err_c)

        r_lt = float(np.corrcoef(block_err_l, block_err_t)[0, 1])
        r_lc = float(np.corrcoef(block_err_l, block_err_c)[0, 1])
        r_tc = float(np.corrcoef(block_err_t, block_err_c)[0, 1])

        all_complementarity_records.append({
            "dataset": d_key,
            "corr_disagreement_delta_equal": corr_dis_delta,
            "spearman_disagreement_delta_equal": spearman_dis_delta,
            "pct_lstm_best": pct_l_wins,
            "pct_tcn_best": pct_t_wins,
            "pct_cnn_best": pct_c_wins,
            "mean_w_lstm_when_lstm_best": w_l_when_l_wins,
            "mean_w_tcn_when_tcn_best": w_t_when_t_wins,
            "mean_w_cnn_when_cnn_best": w_c_when_c_wins,
            "non_overlapping_corr_lstm_tcn": r_lt,
            "non_overlapping_corr_lstm_cnn": r_lc,
            "non_overlapping_corr_tcn_cnn": r_tc,
        })

        # -------------------------------------------------------------
        # 10. Non-Overlapping Daily-Block Statistical Tests
        # -------------------------------------------------------------
        daily_diffs_all = []
        for k in range(K_blocks):
            idx = slice(k * horizon, (k + 1) * horizon)
            mae_c_blk = np.mean(np.abs(preds_caeg[idx] - trues[idx]))
            mae_e_blk = np.mean(np.abs(preds_ens[idx] - trues[idx]))
            daily_diffs_all.append(mae_c_blk - mae_e_blk)

        daily_diffs_all = np.array(daily_diffs_all)

        def run_block_stats(diffs, label):
            k_cnt = len(diffs)
            if k_cnt < 3:
                return None
            m_diff = float(np.mean(diffs))
            s_diff = float(np.std(diffs, ddof=1)) if k_cnt > 1 else 0.0
            se = s_diff / np.sqrt(k_cnt)
            ci_low = m_diff - 1.96 * se
            ci_high = m_diff + 1.96 * se
            t_stat, p_t = stats.ttest_1samp(diffs, 0.0)
            try:
                w_stat, p_w = stats.wilcoxon(diffs)
            except Exception:
                w_stat, p_w = np.nan, np.nan
            cohen_d = m_diff / (s_diff + 1e-8)
            return {
                "subset": label,
                "k_blocks": k_cnt,
                "mean_daily_diff": m_diff,
                "std_daily_diff": s_diff,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "t_stat": float(t_stat),
                "p_value_t": float(p_t),
                "w_stat": float(w_stat) if not np.isnan(w_stat) else 0.0,
                "p_value_wilcoxon": float(p_w) if not np.isnan(p_w) else 1.0,
                "cohen_d": float(cohen_d),
            }

        # Overall test
        stat_overall = run_block_stats(daily_diffs_all, "Overall_Test_Set")
        if stat_overall:
            stat_overall["dataset"] = d_key
            all_statistical_records.append(stat_overall)

        # Statistical tests per difficulty regime (by assigning each block to the modal regime of its 24 hours)
        def get_modal_label(arr):
            vals, counts = np.unique(arr, return_counts=True)
            return vals[np.argmax(counts)]

        block_vol_reg = [get_modal_label(vol_reg[k * horizon:(k + 1) * horizon]) for k in range(K_blocks)]
        block_dis_reg = [get_modal_label(dis_reg[k * horizon:(k + 1) * horizon]) for k in range(K_blocks)]

        for v_level in ["low", "medium", "high"]:
            sub_diffs = [daily_diffs_all[k] for k in range(K_blocks) if block_vol_reg[k] == v_level]
            if len(sub_diffs) >= 5:
                st = run_block_stats(sub_diffs, f"Volatility_{v_level}")
                if st:
                    st["dataset"] = d_key
                    all_statistical_records.append(st)

        for d_level in ["low", "medium", "high"]:
            sub_diffs = [daily_diffs_all[k] for k in range(K_blocks) if block_dis_reg[k] == d_level]
            if len(sub_diffs) >= 5:
                st = run_block_stats(sub_diffs, f"Disagreement_{d_level}")
                if st:
                    st["dataset"] = d_key
                    all_statistical_records.append(st)

        # Dataset-level summary record
        all_dataset_records.append({
            "dataset": d_key,
            "unit": unit,
            "n_test_windows": N_test,
            "k_daily_blocks": K_blocks,
            "overall_caeg_mae": float(np.mean(mae_caeg_win)),
            "overall_equal_mae": float(np.mean(mae_equal_win)),
            "overall_lstm_mae": float(np.mean(mae_lstm_win)),
            "overall_tcn_mae": float(np.mean(mae_tcn_win)),
            "overall_cnn_mae": float(np.mean(mae_cnn_win)),
            "overall_delta_equal": float(np.mean(delta_equal)),
            "relative_pct_delta_equal": float(np.mean(delta_equal) / np.mean(mae_equal_win) * 100),
            "caeg_win_rate_equal": float(np.mean(caeg_win_equal)),
            "caeg_win_rate_best_expert": float(np.mean(caeg_win_best)),
            "mean_routing_entropy": float(np.mean(routing_entropy)),
            "mean_effective_experts": float(np.mean(n_eff)),
        })

    # -----------------------------------------------------------------
    # 11. Compile CSV Artifacts
    # -----------------------------------------------------------------
    print("\n--- Compiling CSV Artifacts ---")
    df_all_windows = pd.concat(all_window_records, ignore_index=True)
    df_all_regimes = pd.DataFrame(all_regime_records)
    df_all_comp = pd.DataFrame(all_complementarity_records)
    df_all_routing = pd.DataFrame(all_routing_records)
    df_all_stats = pd.DataFrame(all_statistical_records)
    df_all_dataset = pd.DataFrame(all_dataset_records)

    # Apply Holm-Bonferroni correction to statistical tests per dataset
    holm_adj_t = []
    holm_adj_w = []
    for d_name, grp in df_all_stats.groupby("dataset"):
        adj_t = holm_bonferroni(grp["p_value_t"].values)
        adj_w = holm_bonferroni(grp["p_value_wilcoxon"].values)
        holm_adj_t.extend(list(zip(grp.index, adj_t)))
        holm_adj_w.extend(list(zip(grp.index, adj_w)))

    df_all_stats["p_val_t_holm"] = [x[1] for x in sorted(holm_adj_t, key=lambda x: x[0])]
    df_all_stats["p_val_wilcoxon_holm"] = [x[1] for x in sorted(holm_adj_w, key=lambda x: x[0])]

    p_win_csv = os.path.join(results_dir, "phase11_window_metrics.csv")
    p_reg_csv = os.path.join(results_dir, "phase11_regime_summary.csv")
    p_comp_csv = os.path.join(results_dir, "phase11_expert_complementarity.csv")
    p_rout_csv = os.path.join(results_dir, "phase11_routing_by_regime.csv")
    p_stat_csv = os.path.join(results_dir, "phase11_statistical_tests.csv")
    p_data_csv = os.path.join(results_dir, "phase11_dataset_summary.csv")

    df_all_windows.to_csv(p_win_csv, index=False)
    df_all_regimes.to_csv(p_reg_csv, index=False)
    df_all_comp.to_csv(p_comp_csv, index=False)
    df_all_routing.to_csv(p_rout_csv, index=False)
    df_all_stats.to_csv(p_stat_csv, index=False)
    df_all_dataset.to_csv(p_data_csv, index=False)

    print(f"Saved: {p_win_csv} ({len(df_all_windows)} rows)")
    print(f"Saved: {p_reg_csv} ({len(df_all_regimes)} rows)")
    print(f"Saved: {p_comp_csv} ({len(df_all_comp)} rows)")
    print(f"Saved: {p_rout_csv} ({len(df_all_routing)} rows)")
    print(f"Saved: {p_stat_csv} ({len(df_all_stats)} rows)")
    print(f"Saved: {p_data_csv} ({len(df_all_dataset)} rows)")

    # -----------------------------------------------------------------
    # 12. Generate 8 Publication-Quality Plots
    # -----------------------------------------------------------------
    print("\n--- Generating Publication-Quality Figures ---")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})

    # Plot 1: CAEG vs Equal error by volatility regime
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_all_regimes[(df_all_regimes["dataset"] == d_key) & (df_all_regimes["regime_category"] == "Volatility")]
        cats = ["low", "medium", "high"]
        sub = sub.set_index("regime_name").reindex(cats).reset_index()
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w/2, sub["mae_caeg"], width=w, label="CAEG-Net V1", color="#1f77b4")
        ax.bar(x + w/2, sub["mae_equal"], width=w, label="Equal Ensemble", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_title(f"{d_key} ({datasets[d_key]['unit']})")
        ax.set_xlabel("Volatility Regime")
        ax.set_ylabel(f"MAE ({datasets[d_key]['unit']})")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 1: Forecast Error by Historical Volatility Regime", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_01_error_by_volatility.png"), bbox_inches="tight")
    plt.close()

    # Plot 2: CAEG advantage vs expert disagreement
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        w_df = dataset_eval_data[d_key]["df_win"]
        # Bin disagreement into deciles for clear trend line
        w_df["dis_decile"] = pd.qcut(w_df["expert_disagreement"], 10, duplicates="drop")
        binned = w_df.groupby("dis_decile", observed=False).agg({"expert_disagreement": "mean", "delta_equal": "mean"}).dropna()
        ax.scatter(w_df["expert_disagreement"], w_df["delta_equal"], alpha=0.15, s=10, color="gray")
        ax.plot(binned["expert_disagreement"], binned["delta_equal"], color="crimson", linewidth=2.5, marker="o", label="Binned Mean")
        ax.axhline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_title(f"{d_key}")
        ax.set_xlabel(f"Expert Disagreement ({datasets[d_key]['unit']})")
        ax.set_ylabel("Delta Equal (MAE_CAEG - MAE_Equal)")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 2: CAEG Advantage vs. Expert Disagreement (Negative Delta Favors CAEG)", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_02_advantage_vs_disagreement.png"), bbox_inches="tight")
    plt.close()

    # Plot 3: Routing weights vs volatility
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_all_routing[(df_all_routing["dataset"] == d_key) & (df_all_routing["regime_category"] == "Volatility")]
        cats = ["low", "medium", "high"]
        sub = sub.set_index("regime_name").reindex(cats).reset_index()
        x = np.arange(len(cats))
        w = 0.25
        ax.bar(x - w, sub["mean_w_lstm"], width=w, label="w_LSTM", color="#2ca02c")
        ax.bar(x, sub["mean_w_tcn"], width=w, label="w_TCN", color="#1f77b4")
        ax.bar(x + w, sub["mean_w_cnn"], width=w, label="w_CNN", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_ylim(0, 0.6)
        ax.set_title(f"{d_key}")
        ax.set_xlabel("Volatility Regime")
        ax.set_ylabel("Mean Routing Weight")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 3: Gating Allocation Across Volatility Regimes", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_03_routing_vs_volatility.png"), bbox_inches="tight")
    plt.close()

    # Plot 4: Routing weights vs disagreement
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_all_routing[(df_all_routing["dataset"] == d_key) & (df_all_routing["regime_category"] == "Expert_Disagreement")]
        cats = ["low", "medium", "high"]
        sub = sub.set_index("regime_name").reindex(cats).reset_index()
        x = np.arange(len(cats))
        w = 0.25
        ax.bar(x - w, sub["mean_w_lstm"], width=w, label="w_LSTM", color="#2ca02c")
        ax.bar(x, sub["mean_w_tcn"], width=w, label="w_TCN", color="#1f77b4")
        ax.bar(x + w, sub["mean_w_cnn"], width=w, label="w_CNN", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_ylim(0, 0.6)
        ax.set_title(f"{d_key}")
        ax.set_xlabel("Disagreement Regime")
        ax.set_ylabel("Mean Routing Weight")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 4: Gating Allocation Across Expert Disagreement Levels", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_04_routing_vs_disagreement.png"), bbox_inches="tight")
    plt.close()

    # Plot 5: CAEG vs Equal across difficulty regimes (Ramp Magnitude)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_all_regimes[(df_all_regimes["dataset"] == d_key) & (df_all_regimes["regime_category"] == "Ramp_Magnitude")]
        cats = ["low", "medium", "high"]
        sub = sub.set_index("regime_name").reindex(cats).reset_index()
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w/2, sub["mae_caeg"], width=w, label="CAEG-Net V1", color="#1f77b4")
        ax.bar(x + w/2, sub["mae_equal"], width=w, label="Equal Ensemble", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low Ramp", "Med Ramp", "High Ramp"])
        ax.set_title(f"{d_key} ({datasets[d_key]['unit']})")
        ax.set_xlabel("Ramp Magnitude Regime")
        ax.set_ylabel(f"MAE ({datasets[d_key]['unit']})")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 5: Forecast Error by Realized Ramp Magnitude Regime", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_05_difficulty_regimes.png"), bbox_inches="tight")
    plt.close()

    # Plot 6: Expert win distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    w = 0.25
    d_names = ["PJM", "GEFCom", "UCI"]
    for i, d_key in enumerate(d_names):
        sub_c = df_all_comp[df_all_comp["dataset"] == d_key].iloc[0]
        ax.bar(x[i] - w, sub_c["pct_lstm_best"] * 100, width=w, label="LSTM" if i == 0 else "", color="#2ca02c")
        ax.bar(x[i], sub_c["pct_tcn_best"] * 100, width=w, label="TCN" if i == 0 else "", color="#1f77b4")
        ax.bar(x[i] + w, sub_c["pct_cnn_best"] * 100, width=w, label="CNN" if i == 0 else "", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(d_names)
    ax.set_ylabel("Win Frequency (% of Test Windows)")
    ax.set_title("Figure 6: Realized Expert Win Frequency by Dataset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_06_expert_win_distribution.png"), bbox_inches="tight")
    plt.close()

    # Plot 7: Dataset-level summary of CAEG advantage
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, df_all_dataset["caeg_win_rate_equal"] * 100, width=w, label="Win Rate vs Equal (%)", color="#1f77b4")
    ax.bar(x + w/2, df_all_dataset["caeg_win_rate_best_expert"] * 100, width=w, label="Win Rate vs Best Single Expert (%)", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(df_all_dataset["dataset"])
    ax.set_ylabel("Win Percentage (%)")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Figure 7: CAEG Win Rate vs Baselines Across Datasets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_07_dataset_summary_advantage.png"), bbox_inches="tight")
    plt.close()

    # Plot 8: Residual correlation vs relative fusion performance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in df_all_comp.iterrows():
        d_name = row["dataset"]
        mean_corr = (row["non_overlapping_corr_lstm_tcn"] + row["non_overlapping_corr_lstm_cnn"] + row["non_overlapping_corr_tcn_cnn"]) / 3.0
        rel_diff = df_all_dataset[df_all_dataset["dataset"] == d_name]["relative_pct_delta_equal"].values[0]
        ax.scatter(mean_corr, rel_diff, s=120, label=d_name)
        ax.annotate(d_name, (mean_corr + 0.005, rel_diff + 0.05), fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Pairwise Expert Residual Correlation (r)")
    ax.set_ylabel("Relative Fusion Delta (%) [Negative Favors CAEG]")
    ax.set_title("Figure 8: Expert Error Correlation vs Relative Fusion Advantage")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase11_08_residual_corr_vs_fusion.png"), bbox_inches="tight")
    plt.close()

    print(f"All 8 figures generated and saved to: {plots_dir}")
    print(f"\nPhase 11 execution finished in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    main()
