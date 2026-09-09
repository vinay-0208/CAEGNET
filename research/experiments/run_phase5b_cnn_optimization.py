"""
Phase 5B: Controlled CNN Expert Optimization Suite
=================================================
Evaluates isolated, controlled architectural and training modifications
to the canonical CNN expert strictly on the VALIDATION partition (Seed 42).

Strict Protocol:
- Primary Metric: Validation MAE (MW)
- Secondary: Validation RMSE, MSE, parameter efficiency, stability
- Zero test data access during optimization.
- Controlled ablations: CNN-A through CNN-H (one change at a time).
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)
from caeg_net import CNNExpert, LSTMExpert, TCNExpert, ContextFeatureEncoder, ContextGatingNetwork, CAEGNet
from evaluate import compute_metrics


# =====================================================================
# Controlled CNN Variants (One Change at a Time)
# =====================================================================

class CNN_Control(nn.Module):
    """Canonical V1 CNNExpert (Control Baseline)."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_A_LessPooling(nn.Module):
    """CNN-A: Reduced pooling aggressiveness (Stride 1 on first stage instead of 2)."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # No stage 1 maxpool: keeps 168 resolution into stage 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 168 -> 84
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_B_LayerNorm(nn.Module):
    """CNN-B: GroupNorm (Group=4) instead of BatchNorm1d to prevent batch-statistic drift."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_C_DilatedReceptiveField(nn.Module):
    """CNN-C: Dilated convolutions (dilation=2 in stage 2 and 3) to expand receptive field."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_D_HeadCapacity(nn.Module):
    """CNN-D: Expanded head capacity (64 -> 128 -> 64 -> 24)."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_E_ZeroDropout(nn.Module):
    """CNN-E: Zero dropout in head (dropout=0.0)."""
    def __init__(self, input_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).squeeze(-1)
        return self.head(feat)


class CNN_H_TemporalPooling(nn.Module):
    """CNN-H: Retains temporal structure via AdaptiveAvgPool1d(4) flattened (64*4=256 -> 48 -> 24)."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, pool_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool_size = pool_size
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_size), # Retains 4 sub-temporal intervals across lookback
        )
        self.head = nn.Sequential(
            nn.Linear(64 * pool_size, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans) # [B, 64, pool_size]
        feat_flat = feat.view(feat.size(0), -1) # [B, 64 * pool_size]
        return self.head(feat_flat)


# =====================================================================
# Training & Validation Helper
# =====================================================================

def train_and_eval_cnn(
    model: nn.Module,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    scaler: object,
    device: torch.device,
    lr: float = 1e-3,
    scheduler_type: str = "plateau",
    max_epochs: int = 40,
    patience: int = 8,
) -> Dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    elif scheduler_type == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        for bx, by in tr_loader:
            optimizer.zero_grad()
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = F.mse_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                va_loss += F.mse_loss(model(bx), by).item()
                n_va += 1
        va_loss /= max(n_va, 1)

        if scheduler_type == "plateau":
            scheduler.step(va_loss)
        else:
            scheduler.step()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = ep
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_weights)
    model.eval()

    # Evaluate unscaled validation metrics
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for bx, by in va_loader:
            all_preds.append(model(bx.to(device)).cpu().numpy())
            all_trues.append(by.numpy())

    preds_mw = np.concatenate(all_preds, axis=0) * scale + mean
    trues_mw = np.concatenate(all_trues, axis=0) * scale + mean

    metrics = compute_metrics(trues_mw, preds_mw)
    params = sum(p.numel() for p in model.parameters())

    return {
        "metrics": metrics,
        "params": params,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "preds_mw": preds_mw,
    }


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=========================================================================")
    print(">>> PHASE 5B: CONTROLLED CNN EXPERT OPTIMIZATION (VALIDATION ONLY) <<<")
    print("=========================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Load 70/15/15 chronological dataset
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)

    tr_x = torch.from_numpy(windows["train"]["X"]).float()
    tr_y = torch.from_numpy(windows["train"]["Y"]).float()
    va_x = torch.from_numpy(windows["val"]["X"]).float()
    va_y = torch.from_numpy(windows["val"]["Y"]).float()

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_x, va_y), batch_size=64, shuffle=False)

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    val_true_mw = windows["val"]["Y"] * scale + mean

    # Set seed 42 strictly for controlled validation screening
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # 1. Baseline Canonical Experts on Validation
    print("\n--- Training Baseline Canonical LSTM & TCN for Ensemble Testing ---")
    m_lstm = LSTMExpert().to(device)
    res_lstm = train_and_eval_cnn(m_lstm, tr_loader, va_loader, scaler, device)
    print(f"Canonical LSTM: Val MAE = {res_lstm['metrics']['MAE']:.2f} MW, Params = {res_lstm['params']}")

    m_tcn = TCNExpert().to(device)
    res_tcn = train_and_eval_cnn(m_tcn, tr_loader, va_loader, scaler, device)
    print(f"Canonical TCN:  Val MAE = {res_tcn['metrics']['MAE']:.2f} MW, Params = {res_tcn['params']}")

    # 2. Screening CNN Controlled Variants
    experiments = [
        ("CNN_Control", CNN_Control(), 1e-3, "plateau", "Canonical V1 CNN architecture"),
        ("CNN_A_LessPooling", CNN_A_LessPooling(), 1e-3, "plateau", "Reduced pooling (stride 1 in stage 1)"),
        ("CNN_B_GroupNorm", CNN_B_LayerNorm(), 1e-3, "plateau", "GroupNorm (group=4) instead of BatchNorm"),
        ("CNN_C_DilatedConv", CNN_C_DilatedReceptiveField(), 1e-3, "plateau", "Dilated conv (dilation=2 in stages 2-3)"),
        ("CNN_D_HeadCapacity", CNN_D_HeadCapacity(), 1e-3, "plateau", "Expanded MLP head (64->128->64->24)"),
        ("CNN_E_ZeroDropout", CNN_E_ZeroDropout(), 1e-3, "plateau", "Dropout = 0.0 in head"),
        ("CNN_F_LowerLR", CNN_Control(), 5e-4, "plateau", "Lower learning rate (lr=5e-4)"),
        ("CNN_G_CosineScheduler", CNN_Control(), 1e-3, "cosine", "CosineAnnealingLR scheduler"),
        ("CNN_H1_TemporalPool4", CNN_H_TemporalPooling(pool_size=4), 1e-3, "plateau", "Adaptive temporal pooling (size=4)"),
        ("CNN_H2_TemporalPool8", CNN_H_TemporalPooling(pool_size=8), 1e-3, "plateau", "Adaptive temporal pooling (size=8)"),
    ]

    screening_rows = []
    trained_preds = {}

    for name, model, lr, sched, desc in experiments:
        torch.manual_seed(42)
        np.random.seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        res = train_and_eval_cnn(model, tr_loader, va_loader, scaler, device, lr=lr, scheduler_type=sched)
        trained_preds[name] = res["preds_mw"]

        screening_rows.append({
            "experiment_id": name,
            "description": desc,
            "val_mae_mw": res["metrics"]["MAE"],
            "val_rmse_mw": res["metrics"]["RMSE"],
            "val_mse_mw2": res["metrics"]["MSE"],
            "val_r2": res["metrics"]["R2"],
            "params": res["params"],
            "best_epoch": res["best_epoch"],
            "train_time_s": res["train_time_s"],
            "seed": 42,
        })
        print(f"[{name:<22}] Val MAE: {res['metrics']['MAE']:6.2f} MW | RMSE: {res['metrics']['RMSE']:6.2f} | Params: {res['params']:6d} | Ep: {res['best_epoch']:2d}")

    df_screen = pd.DataFrame(screening_rows).sort_values(by="val_mae_mw")
    control_mae = df_screen.loc[df_screen["experiment_id"] == "CNN_Control", "val_mae_mw"].values[0]
    df_screen["gain_vs_control_mw"] = control_mae - df_screen["val_mae_mw"]
    df_screen["improved_over_control"] = df_screen["gain_vs_control_mw"] > 0

    out_csv = "research/results/phase5b_expert_screening.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_screen.to_csv(out_csv, index=False)
    print(f"\nSaved CNN expert screening results to: {out_csv}")

    # 3. Ensemble & Fusion Interaction Test (Section 10)
    # Evaluate top CNN variant in combination with canonical LSTM and TCN
    best_variant = df_screen.iloc[0]["experiment_id"]
    print(f"\nTop Performing CNN Variant: {best_variant} (Val MAE = {df_screen.iloc[0]['val_mae_mw']:.2f} MW)")

    # Test Equal Ensemble with Control CNN vs Best CNN
    eq_control = (res_lstm["preds_mw"] + res_tcn["preds_mw"] + trained_preds["CNN_Control"]) / 3.0
    eq_best = (res_lstm["preds_mw"] + res_tcn["preds_mw"] + trained_preds[best_variant]) / 3.0

    met_eq_control = compute_metrics(val_true_mw, eq_control)
    met_eq_best = compute_metrics(val_true_mw, eq_best)

    print(f"\n--- Expert + Fusion Interaction on Validation (Seed 42) ---")
    print(f"Standalone Control CNN:           Val MAE = {control_mae:.2f} MW")
    print(f"Standalone {best_variant}: Val MAE = {df_screen.iloc[0]['val_mae_mw']:.2f} MW (Gain: {df_screen.iloc[0]['gain_vs_control_mw']:+.2f} MW)")
    print(f"Equal Ensemble (with Control CNN): Val MAE = {met_eq_control['MAE']:.2f} MW")
    print(f"Equal Ensemble (with {best_variant}): Val MAE = {met_eq_best['MAE']:.2f} MW (Gain: {met_eq_control['MAE'] - met_eq_best['MAE']:+.2f} MW)")

    fusion_df = pd.DataFrame([
        {"setup": "Standalone_CNN_Control", "val_mae_mw": control_mae},
        {"setup": f"Standalone_{best_variant}", "val_mae_mw": df_screen.iloc[0]["val_mae_mw"]},
        {"setup": "Equal_Ensemble_Control_CNN", "val_mae_mw": met_eq_control["MAE"]},
        {"setup": f"Equal_Ensemble_{best_variant}", "val_mae_mw": met_eq_best["MAE"]},
    ])
    fusion_df.to_csv("research/results/phase5b_fusion_interaction.csv", index=False)


if __name__ == "__main__":
    main()

