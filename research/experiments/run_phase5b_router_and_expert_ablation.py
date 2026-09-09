"""
Phase 5B: Controlled LSTM/TCN Optimization & Router Screening Suite
==================================================================
Evaluates:
1. Small controlled adjustments to LSTM and TCN experts (Validation MAE).
2. Controlled router ablations R0 through R6 (Validation MAE).
3. Recent-error sensitivity analysis (Validation MAE).

Decision Protocol:
- Strictly Validation-Only (Seed 42).
- Zero access to test data during screening.
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
from caeg_net import LSTMExpert, TCNExpert, CNNExpert, CAEGNet
from research.original_caeg import (
    OriginalCAEGNetPhase5,
    compute_phase5_loss,
)
from evaluate import compute_metrics


# =====================================================================
# Controlled LSTM & TCN Variants
# =====================================================================

class LSTM_Wider(nn.Module):
    """LSTM with hidden_dim=96 (vs 64)."""
    def __init__(self, input_dim=1, hidden_dim=96, num_layers=2, horizon=24, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class LSTM_HeadCapacity(nn.Module):
    """LSTM with expanded head capacity (64 -> 128 -> 64 -> 24)."""
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, horizon=24, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class TCN_Wider(nn.Module):
    """TCN with channels=48 (vs 32)."""
    def __init__(self, input_dim=1, channels=48, dilations=(1, 2, 4, 8, 16, 32), kernel_size=3, horizon=24, dropout=0.1):
        super().__init__()
        from caeg_net import CausalConv1dBlock
        blocks = []
        in_c = input_dim
        for d in dilations:
            blocks.append(CausalConv1dBlock(in_c, channels, kernel_size, dilation=d, dropout=dropout))
            in_c = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        features = self.network(x_trans)
        return self.head(features[:, :, -1])


def train_eval_model(model, tr_loader, va_loader, scaler, device, max_epochs=40, patience=8, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    best_val_loss = float("inf")
    best_weights = None
    best_ep = 0
    patience_cnt = 0
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            bx = batch[0].to(device)
            by = batch[1].to(device)
            bc = batch[2].to(device) if len(batch) == 3 else None
            if bc is not None and hasattr(model, "gating_network"):
                out = model(bx, bc, return_diagnostics=False)
            elif bc is not None and hasattr(model, "use_disagreement"):
                out, _, _ = model(bx, bc)
            else:
                out = model(bx)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.mse_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in va_loader:
                bx = batch[0].to(device)
                by = batch[1].to(device)
                bc = batch[2].to(device) if len(batch) == 3 else None
                if bc is not None and hasattr(model, "gating_network"):
                    out = model(bx, bc, return_diagnostics=False)
                elif bc is not None and hasattr(model, "use_disagreement"):
                    out, _, _ = model(bx, bc)
                else:
                    out = model(bx)
                if isinstance(out, tuple):
                    out = out[0]
                va_loss += F.mse_loss(out, by).item()
                n_va += 1
        va_loss /= max(n_va, 1)
        sched.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_ep = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_weights)
    model.eval()

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    preds = []
    trues = []
    with torch.no_grad():
        for batch in va_loader:
            bx = batch[0].to(device)
            by = batch[1].to(device)
            bc = batch[2].to(device) if len(batch) == 3 else None
            if bc is not None and hasattr(model, "gating_network"):
                out = model(bx, bc, return_diagnostics=False)
            elif bc is not None and hasattr(model, "use_disagreement"):
                out, _, _ = model(bx, bc)
            else:
                out = model(bx)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
            trues.append(by.cpu().numpy())

    preds_mw = np.concatenate(preds, axis=0) * scale + mean
    trues_mw = np.concatenate(trues, axis=0) * scale + mean
    metrics = compute_metrics(trues_mw, preds_mw)
    params = sum(p.numel() for p in model.parameters())

    return {
        "metrics": metrics,
        "params": params,
        "best_epoch": best_ep,
        "train_time_s": train_time,
        "preds_mw": preds_mw,
    }


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=========================================================================")
    print(">>> PHASE 5B: LSTM/TCN OPTIMIZATION & ROUTER SCREENING (VALIDATION) <<<")
    print("=========================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    # Context feature sets
    C_tr_4d = extract_context_features(windows["train"]["X"], rec_tr)
    C_va_4d = extract_context_features(windows["val"]["X"], rec_val)
    C_tr_3d = C_tr_4d[:, :3]
    C_va_3d = C_va_4d[:, :3]

    # Data loaders
    def make_loader(c_tr, c_va):
        tr = DataLoader(TensorDataset(
            torch.from_numpy(windows["train"]["X"]).float(),
            torch.from_numpy(windows["train"]["Y"]).float(),
            torch.from_numpy(c_tr).float()
        ), batch_size=64, shuffle=True)
        va = DataLoader(TensorDataset(
            torch.from_numpy(windows["val"]["X"]).float(),
            torch.from_numpy(windows["val"]["Y"]).float(),
            torch.from_numpy(c_va).float()
        ), batch_size=64, shuffle=False)
        return tr, va

    tr_4d, va_4d = make_loader(C_tr_4d, C_va_4d)
    tr_3d, va_3d = make_loader(C_tr_3d, C_va_3d)

    # 1. Screen LSTM & TCN Modifications
    print("\n--- Screening LSTM & TCN Controlled Variants (Seed 42) ---")
    torch.manual_seed(42)
    expert_tests = [
        ("LSTM_Control", LSTMExpert(), "Canonical 2-layer LSTM (dim=64)"),
        ("LSTM_Wider_96", LSTM_Wider(), "Wider hidden dimension (hidden=96)"),
        ("LSTM_HeadCap", LSTM_HeadCapacity(), "Expanded MLP head (64->128->64->24)"),
        ("TCN_Control", TCNExpert(), "Canonical 6-stage TCN (channels=32)"),
        ("TCN_Wider_48", TCN_Wider(), "Wider channel width (channels=48)"),
    ]
    exp_results = []
    for name, m, desc in expert_tests:
        torch.manual_seed(42)
        res = train_eval_model(m, tr_4d, va_4d, scaler, device)
        exp_results.append({
            "model": name,
            "description": desc,
            "val_mae_mw": res["metrics"]["MAE"],
            "val_rmse_mw": res["metrics"]["RMSE"],
            "params": res["params"],
            "best_epoch": res["best_epoch"],
        })
        print(f"[{name:<16}] Val MAE: {res['metrics']['MAE']:6.2f} MW | RMSE: {res['metrics']['RMSE']:6.2f} | Params: {res['params']:6d}")

    df_experts = pd.DataFrame(exp_results)
    df_experts.to_csv("research/results/phase5b_lstm_tcn_screening.csv", index=False)

    # 2. Screen Router Variants (R0 through R6 + Recent Error Sensitivity)
    print("\n--- Screening Router Variants (R0 to R6) on Validation (Seed 42) ---")
    router_tests = [
        ("R0_Canonical_V1", CAEGNet(context_dim=4), tr_4d, va_4d, "Canonical V1 Softmax router (4D context)"),
        ("R0_No_Recent_Error_3D", CAEGNet(context_dim=3), tr_3d, va_3d, "Router with Trend, Volatility, Periodicity only (3D context)"),
        ("R3_Detached_Disagreement", OriginalCAEGNetPhase5(context_dim=4, use_disagreement=True), tr_4d, va_4d, "4D context + 3D detached pairwise expert disagreement (7D total)"),
        ("R4_Bounded_Rho_0.10", OriginalCAEGNetPhase5(context_dim=4, conservative_rho=0.10), tr_4d, va_4d, "Conservative bounded routing (rho=0.10)"),
        ("R4_Bounded_Rho_0.20", OriginalCAEGNetPhase5(context_dim=4, conservative_rho=0.20), tr_4d, va_4d, "Conservative bounded routing (rho=0.20)"),
        ("R4_Bounded_Rho_0.30", OriginalCAEGNetPhase5(context_dim=4, conservative_rho=0.30), tr_4d, va_4d, "Conservative bounded routing (rho=0.30)"),
        ("R4_Bounded_Rho_0.50", OriginalCAEGNetPhase5(context_dim=4, conservative_rho=0.50), tr_4d, va_4d, "Conservative bounded routing (rho=0.50)"),
    ]

    router_results = []
    for name, m, tr_l, va_l, desc in router_tests:
        torch.manual_seed(42)
        res = train_eval_model(m, tr_l, va_l, scaler, device)
        router_results.append({
            "router_id": name,
            "description": desc,
            "val_mae_mw": res["metrics"]["MAE"],
            "val_rmse_mw": res["metrics"]["RMSE"],
            "val_r2": res["metrics"]["R2"],
            "params": res["params"],
            "best_epoch": res["best_epoch"],
        })
        print(f"[{name:<25}] Val MAE: {res['metrics']['MAE']:6.2f} MW | RMSE: {res['metrics']['RMSE']:6.2f} | Params: {res['params']:6d}")

    df_routers = pd.DataFrame(router_results).sort_values(by="val_mae_mw")
    r0_mae = df_routers.loc[df_routers["router_id"] == "R0_Canonical_V1", "val_mae_mw"].values[0]
    df_routers["gain_vs_r0_mw"] = r0_mae - df_routers["val_mae_mw"]
    df_routers["improved_over_r0"] = df_routers["gain_vs_r0_mw"] > 0
    df_routers.to_csv("research/results/phase5b_router_screening.csv", index=False)
    print(f"\nSaved router screening results to: research/results/phase5b_router_screening.csv")


if __name__ == "__main__":
    main()
