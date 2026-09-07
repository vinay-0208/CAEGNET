"""
CAEG-Net V2 Exploratory Controlled Experiments (Validation-Driven)
==================================================================
Systematically evaluates improvements on the VALIDATION SET ONLY:
- Experiment A: Training Loss Optimization (MSE vs Huber vs MAE)
- Experiment B: Context Feature Set (4 context vs 5 context with multi-lag periodicity)
- Experiment C: Expert Contribution Analysis (TCN vs LSTM+TCN vs LSTM+TCN+CNN)
- Experiment D: Gating Network Regularization & Capacity
- Experiment E: Horizon-Dependent Gating (24 x 3 weights vs 1 x 3 weights)
- Experiment F: Recent Error Representation (MAE vs MAE + Signed Bias)

Rule:
Zero test set access during exploration. Model selection is driven strictly by Validation MSE/MAE.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from caeg_net import CAEGNet, LSTMExpert, TCNExpert, CNNExpert
from train import create_criterion, create_optimizer_and_scheduler, train_model


def run_exploratory_suite():
    print("=" * 80)
    print("STARTING CAEG-NET V2 EXPLORATORY CONTROLLED EXPERIMENTS")
    print("Rule: Model evaluation and selection strictly on the VALIDATION SET.")
    print("=" * 80)

    # 1. Setup Data Pipeline
    data_path = "data/Modern_PJM/pjm_load.csv"
    df, _ = load_and_clean_data(data_path)
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    scale_mw = float(scaler.scale_[0])
    mean_mw = float(scaler.mean_[0])

    # Standard 4 Context Features
    C_tr = extract_context_features(windows["train"]["X"], rec_tr)
    C_val = extract_context_features(windows["val"]["X"], rec_val)

    # Extended 5 Context Features (with Lag-48 Multi-Day Harmonic Periodicity)
    # Autocorrelation at lag 48 over lookback
    def extract_5d_context(X_win, rec_err):
        C_4d = extract_context_features(X_win, rec_err)
        Z = X_win[:, :, 0] if X_win.ndim == 3 else X_win
        N, L = Z.shape
        z_mean = np.mean(Z, axis=1, keepdims=True)
        z_var = np.sum((Z - z_mean) ** 2, axis=1) + 1e-6
        # Lag 48 correlation
        r48 = np.sum((Z[:, 48:] - z_mean) * (Z[:, :-48] - z_mean), axis=1) / z_var
        return np.column_stack([C_4d[:, :3], r48, C_4d[:, 3:]])

    C_tr_5d = extract_5d_context(windows["train"]["X"], rec_tr)
    C_val_5d = extract_5d_context(windows["val"]["X"], rec_val)

    # Datasets and Loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_loaders(c_train, c_val, batch_size=32, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], c_train)
        ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], c_val)
        dls = create_dataloaders({"train": ds_tr, "val": ds_va}, batch_size=batch_size, shuffle_train=True, seed=seed)
        return dls["train"], dls["val"]

    y_val_raw = windows["val"]["Y"] * scale_mw + mean_mw

    experiment_records = []

    def evaluate_model_on_val(model, loader, is_5d=False):
        model.eval()
        preds_scaled = []
        with torch.no_grad():
            for bx, by, bc in loader:
                bx = bx.to(device)
                bc = bc.to(device)
                out = model(bx, bc)
                y_p = out[0] if isinstance(out, tuple) else out
                preds_scaled.append(y_p.cpu().numpy())
        p_scaled = np.concatenate(preds_scaled, axis=0)
        p_raw = p_scaled * scale_mw + mean_mw
        val_mse_scaled = float(mean_squared_error(windows["val"]["Y"], p_scaled))
        val_mae_mw = float(mean_absolute_error(y_val_raw, p_raw))
        val_rmse_mw = float(np.sqrt(mean_squared_error(y_val_raw, p_raw)))
        return val_mse_scaled, val_mae_mw, val_rmse_mw

    # -------------------------------------------------------------
    # BASELINE: CAEG-Net V1 Baseline (Seed 42)
    # -------------------------------------------------------------
    print("\n--- Running Baseline V1 (Global Gating, 4 Context, MSE Loss) ---")
    tr_loader, val_loader = get_loaders(C_tr, C_val, seed=42)
    m_base = CAEGNet(horizon_dependent=False, context_dim=4).to(device)
    crit_base = nn.MSELoss()
    opt_base, sched_base = create_optimizer_and_scheduler(m_base, lr=1e-3, weight_decay=1e-4)
    train_model(m_base, tr_loader, val_loader, crit_base, opt_base, sched_base, max_epochs=25, patience=6, verbose=False)
    v_mse_b, v_mae_b, v_rmse_b = evaluate_model_on_val(m_base, val_loader)
    print(f"Baseline V1: Val MSE = {v_mse_b:.5f}, Val MAE = {v_mae_b:.2f} MW, Val RMSE = {v_rmse_b:.2f} MW")
    experiment_records.append({
        "Experiment_ID": "EXP-00-Baseline-V1",
        "Hypothesis": "Current V1 Baseline (Global Gating, 4 Context, MSE Loss)",
        "Val_MSE_Scaled": round(v_mse_b, 5),
        "Val_MAE_MW": round(v_mae_b, 2),
        "Val_RMSE_MW": round(v_rmse_b, 2),
        "Decision": "BASELINE",
        "Notes": "Reference baseline"
    })

    # -------------------------------------------------------------
    # EXPERIMENT A: Training Loss (Huber Loss vs MSE)
    # -------------------------------------------------------------
    print("\n--- Running Experiment A1: Huber Loss (delta=1.0) ---")
    m_huber = CAEGNet(horizon_dependent=False, context_dim=4).to(device)
    crit_huber = nn.HuberLoss(delta=1.0)
    opt_huber, sched_huber = create_optimizer_and_scheduler(m_huber, lr=1e-3, weight_decay=1e-4)
    train_model(m_huber, tr_loader, val_loader, crit_huber, opt_huber, sched_huber, max_epochs=25, patience=6, verbose=False)
    v_mse_h, v_mae_h, v_rmse_h = evaluate_model_on_val(m_huber, val_loader)
    dec_a = "KEEP" if v_mae_h < v_mae_b else "REJECT"
    print(f"Huber Loss: Val MSE = {v_mse_h:.5f}, Val MAE = {v_mae_h:.2f} MW (Diff vs Baseline: {v_mae_h - v_mae_b:+.2f} MW) -> {dec_a}")
    experiment_records.append({
        "Experiment_ID": "EXP-A1-Huber-Loss",
        "Hypothesis": "Huber loss improves robust convergence over MSE against load spikes",
        "Val_MSE_Scaled": round(v_mse_h, 5),
        "Val_MAE_MW": round(v_mae_h, 2),
        "Val_RMSE_MW": round(v_rmse_h, 2),
        "Decision": dec_a,
        "Notes": f"Diff vs Baseline: {v_mae_h - v_mae_b:+.2f} MW MAE"
    })

    # -------------------------------------------------------------
    # EXPERIMENT B: Context Feature Expansion (5 Context Features)
    # -------------------------------------------------------------
    print("\n--- Running Experiment B1: 5 Context Features (Multi-lag Periodicity) ---")
    tr_5d_loader, val_5d_loader = get_loaders(C_tr_5d, C_val_5d, seed=42)
    m_5d = CAEGNet(horizon_dependent=False, context_dim=5).to(device)
    crit_5d = nn.MSELoss()
    opt_5d, sched_5d = create_optimizer_and_scheduler(m_5d, lr=1e-3, weight_decay=1e-4)
    train_model(m_5d, tr_5d_loader, val_5d_loader, crit_5d, opt_5d, sched_5d, max_epochs=25, patience=6, verbose=False)
    v_mse_5d, v_mae_5d, v_rmse_5d = evaluate_model_on_val(m_5d, val_5d_loader)
    dec_b = "KEEP" if v_mae_5d < v_mae_b else "REJECT"
    print(f"5 Context Features: Val MSE = {v_mse_5d:.5f}, Val MAE = {v_mae_5d:.2f} MW (Diff vs Baseline: {v_mae_5d - v_mae_b:+.2f} MW) -> {dec_b}")
    experiment_records.append({
        "Experiment_ID": "EXP-B1-Context-5D",
        "Hypothesis": "Multi-lag harmonic periodicity provides richer context representation",
        "Val_MSE_Scaled": round(v_mse_5d, 5),
        "Val_MAE_MW": round(v_mae_5d, 2),
        "Val_RMSE_MW": round(v_rmse_5d, 2),
        "Decision": dec_b,
        "Notes": f"Diff vs Baseline: {v_mae_5d - v_mae_b:+.2f} MW MAE"
    })

    # -------------------------------------------------------------
    # EXPERIMENT D: Gating Network Capacity (Hidden Dim 48 vs 32)
    # -------------------------------------------------------------
    print("\n--- Running Experiment D1: Gating Hidden Dim 48 ---")
    m_d1 = CAEGNet(horizon_dependent=False, context_dim=4, latent_context_dim=24).to(device)
    crit_d1 = nn.MSELoss()
    opt_d1, sched_d1 = create_optimizer_and_scheduler(m_d1, lr=1e-3, weight_decay=1e-4)
    train_model(m_d1, tr_loader, val_loader, crit_d1, opt_d1, sched_d1, max_epochs=25, patience=6, verbose=False)
    v_mse_d1, v_mae_d1, v_rmse_d1 = evaluate_model_on_val(m_d1, val_loader)
    dec_d = "KEEP" if v_mae_d1 < v_mae_b else "REJECT"
    print(f"Gating Hidden 24/48: Val MSE = {v_mse_d1:.5f}, Val MAE = {v_mae_d1:.2f} MW (Diff vs Baseline: {v_mae_d1 - v_mae_b:+.2f} MW) -> {dec_d}")
    experiment_records.append({
        "Experiment_ID": "EXP-D1-Gating-Capacity",
        "Hypothesis": "Higher latent context capacity (24-dim) improves context routing fidelity",
        "Val_MSE_Scaled": round(v_mse_d1, 5),
        "Val_MAE_MW": round(v_mae_d1, 2),
        "Val_RMSE_MW": round(v_rmse_d1, 2),
        "Decision": dec_d,
        "Notes": f"Diff vs Baseline: {v_mae_d1 - v_mae_b:+.2f} MW MAE"
    })

    # -------------------------------------------------------------
    # EXPERIMENT E: Horizon-Dependent Gating (High Priority)
    # -------------------------------------------------------------
    print("\n--- Running Experiment E1: Horizon-Dependent Gating (24 x 3 Routing Matrix) ---")
    m_horizon = CAEGNet(horizon_dependent=True, context_dim=4).to(device)
    crit_hor = nn.MSELoss()
    opt_hor, sched_hor = create_optimizer_and_scheduler(m_horizon, lr=1e-3, weight_decay=1e-4)
    train_model(m_horizon, tr_loader, val_loader, crit_hor, opt_hor, sched_hor, max_epochs=25, patience=6, verbose=False)
    v_mse_e, v_mae_e, v_rmse_e = evaluate_model_on_val(m_horizon, val_loader)
    dec_e = "KEEP" if v_mae_e < v_mae_b else "REJECT"
    print(f"Horizon-Dependent Gating: Val MSE = {v_mse_e:.5f}, Val MAE = {v_mae_e:.2f} MW (Diff vs Baseline: {v_mae_e - v_mae_b:+.2f} MW) -> {dec_e}")
    experiment_records.append({
        "Experiment_ID": "EXP-E1-Horizon-Gating",
        "Hypothesis": "Horizon-dependent gating allows hour-specific expert specialization across the 24h cycle",
        "Val_MSE_Scaled": round(v_mse_e, 5),
        "Val_MAE_MW": round(v_mae_e, 2),
        "Val_RMSE_MW": round(v_rmse_e, 2),
        "Decision": dec_e,
        "Notes": f"Diff vs Baseline: {v_mae_e - v_mae_b:+.2f} MW MAE"
    })

    # Save Experiment Decision Table
    os.makedirs("results/caeg_v2", exist_ok=True)
    df_exp = pd.DataFrame(experiment_records)
    df_exp.to_csv("results/caeg_v2/exploratory_experiments_validation.csv", index=False)
    print("\n" + "=" * 80)
    print("EXPLORATORY EXPERIMENT SUMMARY (VALIDATION-DRIVEN)")
    print("=" * 80)
    print(df_exp.to_string(index=False))
    return df_exp


if __name__ == "__main__":
    run_exploratory_suite()
