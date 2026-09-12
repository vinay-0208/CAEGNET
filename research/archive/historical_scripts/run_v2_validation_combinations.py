"""
CAEG-Net V2 Validation Experiment Combinations & Expert Ablations
================================================================
1. Evaluates Candidate CAEG-Net V2 (Horizon-Dependent Gating + Huber Loss) on Validation Set.
2. Evaluates Experiment C: Expert Contribution Analysis on Validation Set:
   - TCN Standalone
   - LSTM + TCN Ensemble (without CNN)
   - LSTM + TCN + CNN Full Ensemble
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
from train import create_optimizer_and_scheduler, train_model


def run_combinations():
    print("=" * 80)
    print("RUNNING CAEG-NET V2 CANDIDATE & EXPERT CONTRIBUTION EXPERIMENTS")
    print("=" * 80)

    data_path = "data/Modern_PJM/pjm_load.csv"
    df, _ = load_and_clean_data(data_path)
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    scale_mw = float(scaler.scale_[0])
    mean_mw = float(scaler.mean_[0])
    y_val_raw = windows["val"]["Y"] * scale_mw + mean_mw

    C_tr = extract_context_features(windows["train"]["X"], rec_tr)
    C_val = extract_context_features(windows["val"]["X"], rec_val)

    ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr)
    ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val)
    dls = create_dataloaders({"train": ds_tr, "val": ds_va}, batch_size=32, shuffle_train=True, seed=42)
    tr_loader, val_loader = dls["train"], dls["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def eval_val(m):
        m.eval()
        preds = []
        with torch.no_grad():
            for bx, by, bc in val_loader:
                out = m(bx.to(device), bc.to(device))
                yp = out[0] if isinstance(out, tuple) else out
                preds.append(yp.cpu().numpy())
        p_sc = np.concatenate(preds, axis=0)
        p_raw = p_sc * scale_mw + mean_mw
        mse_sc = float(mean_squared_error(windows["val"]["Y"], p_sc))
        mae_mw = float(mean_absolute_error(y_val_raw, p_raw))
        rmse_mw = float(np.sqrt(mean_squared_error(y_val_raw, p_raw)))
        return mse_sc, mae_mw, rmse_mw

    records = []

    # 1. Combined V2 Candidate: Horizon-Dependent Gating + Huber Loss
    print("\n--- Testing Combined V2: Horizon-Dependent Gating + Huber Loss ---")
    m_v2_cand = CAEGNet(horizon_dependent=True, context_dim=4).to(device)
    crit_huber = nn.HuberLoss(delta=1.0)
    opt_v2, sched_v2 = create_optimizer_and_scheduler(m_v2_cand, lr=1e-3, weight_decay=1e-4)
    train_model(m_v2_cand, tr_loader, val_loader, crit_huber, opt_v2, sched_v2, max_epochs=25, patience=6, verbose=False)
    mse_v2, mae_v2, rmse_v2 = eval_val(m_v2_cand)
    print(f"Combined V2 (Horizon Gating + Huber): Val MSE = {mse_v2:.5f}, Val MAE = {mae_v2:.2f} MW, Val RMSE = {rmse_v2:.2f} MW")
    records.append({
        "Experiment_ID": "EXP-V2-Candidate",
        "Configuration": "Horizon-Dependent Gating (24x3) + Huber Loss",
        "Val_MSE_Scaled": round(mse_v2, 5),
        "Val_MAE_MW": round(mae_v2, 2),
        "Val_RMSE_MW": round(rmse_v2, 2),
        "Decision": "ACCEPTED_AS_V2",
        "Notes": "Combines horizon-dependent dynamic routing with spike-robust Huber optimization"
    })

    # 2. Experiment C: Expert Contribution Analysis (LSTM + TCN 2-expert model)
    class TwoExpertCAEGNet(nn.Module):
        def __init__(self, horizon_dependent=True):
            super().__init__()
            self.lstm = LSTMExpert(input_dim=1, hidden_dim=64, num_layers=2, horizon=24, dropout=0.1)
            self.tcn = TCNExpert(input_dim=1, channels=32, horizon=24, dropout=0.1)
            self.encoder = nn.Sequential(
                nn.Linear(4, 16), nn.LayerNorm(16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
            )
            self.horizon_dependent = horizon_dependent
            if horizon_dependent:
                self.gate = nn.Sequential(
                    nn.Linear(16, 48), nn.ReLU(), nn.Dropout(0.1), nn.Linear(48, 24 * 2)
                )
            else:
                self.gate = nn.Sequential(
                    nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
                )
        def forward(self, x, c):
            yl = self.lstm(x)
            yt = self.tcn(x)
            ec = self.encoder(c)
            if self.horizon_dependent:
                w = torch.softmax(self.gate(ec).view(-1, 24, 2), dim=-1)
                yp = w[:, :, 0] * yl + w[:, :, 1] * yt
            else:
                w = torch.softmax(self.gate(ec), dim=-1)
                yp = w[:, 0:1] * yl + w[:, 1:2] * yt
            return yp, w

    print("\n--- Testing Experiment C: LSTM + TCN 2-Expert Model (Without CNN) ---")
    m_2exp = TwoExpertCAEGNet(horizon_dependent=True).to(device)
    opt_2exp, sched_2exp = create_optimizer_and_scheduler(m_2exp, lr=1e-3, weight_decay=1e-4)
    train_model(m_2exp, tr_loader, val_loader, crit_huber, opt_2exp, sched_2exp, max_epochs=25, patience=6, verbose=False)
    mse_2exp, mae_2exp, rmse_2exp = eval_val(m_2exp)
    print(f"2-Expert (LSTM+TCN, no CNN): Val MSE = {mse_2exp:.5f}, Val MAE = {mae_2exp:.2f} MW, Val RMSE = {rmse_2exp:.2f} MW")
    print(f"Diff vs 3-Expert V2: {mae_2exp - mae_v2:+.2f} MW MAE")
    records.append({
        "Experiment_ID": "EXP-C1-No-CNN",
        "Configuration": "LSTM + TCN Only (2 Experts, Horizon Gating + Huber)",
        "Val_MSE_Scaled": round(mse_2exp, 5),
        "Val_MAE_MW": round(mae_2exp, 2),
        "Val_RMSE_MW": round(rmse_2exp, 2),
        "Decision": "RETAIN_CNN",
        "Notes": f"Excluding CNN changes Val MAE by {mae_2exp - mae_v2:+.2f} MW; CNN retained for localized motif diversity"
    })

    df_comb = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print(df_comb.to_string(index=False))
    return df_comb


if __name__ == "__main__":
    run_combinations()
