import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
)
from caeg_net import LSTMExpert, TCNExpert, CNNExpert, CAEGNet
from evaluate import compute_metrics

def run_equal_ensemble_audit():
    print("=================================================================")
    print(">>> SECTION 4: FULL RE-AUDIT OF CANONICAL STANDALONE EXPERTS & EQUAL ENSEMBLE <<<")
    print("=================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset & Window Setup
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    y_test_true = windows["test"]["Y"] * scale + mean
    num_test_origins = len(windows["test"]["origins"])
    first_origin = int(windows["test"]["origins"][0])
    last_origin = int(windows["test"]["origins"][-1])

    tr_x = torch.from_numpy(windows["train"]["X"]).float()
    tr_y = torch.from_numpy(windows["train"]["Y"]).float()
    va_x = torch.from_numpy(windows["val"]["X"]).float()
    va_y = torch.from_numpy(windows["val"]["Y"]).float()
    te_x = torch.from_numpy(windows["test"]["X"]).float()
    te_y = torch.from_numpy(windows["test"]["Y"]).float()

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_x, va_y), batch_size=64, shuffle=False)
    te_loader = DataLoader(TensorDataset(te_x, te_y), batch_size=64, shuffle=False)

    seeds = [42, 123, 2024, 3407, 999]

    # We will test two training regimes for audit:
    # Regime 1: Canonical train.py protocol (StepLR, step_size=15, gamma=0.5, max_epochs=50, patience=7, no grad clip)
    # Regime 2: Stabilized training protocol (patience=15 to avoid premature plateau early stopping, grad_clip=1.0)
    
    for regime_name, use_plateau, patience_val, clip_val in [
        ("Canonical_train_py_StepLR", False, 7, None),
        ("Audited_Stable_Protocol", True, 12, 1.0)
    ]:
        print(f"\n==================================================")
        print(f"EVALUATING REGIME: {regime_name} (patience={patience_val}, clip={clip_val})")
        print(f"==================================================")

        audit_rows = []
        all_equal_errors = []

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(s)

            # Standalone LSTM
            m_lstm = LSTMExpert().to(device)
            p_lstm = train_expert(m_lstm, tr_loader, va_loader, te_loader, scale, mean, device, use_plateau, patience_val, clip_val)

            # Standalone TCN
            m_tcn = TCNExpert().to(device)
            p_tcn = train_expert(m_tcn, tr_loader, va_loader, te_loader, scale, mean, device, use_plateau, patience_val, clip_val)

            # Standalone CNN
            m_cnn = CNNExpert().to(device)
            p_cnn = train_expert(m_cnn, tr_loader, va_loader, te_loader, scale, mean, device, use_plateau, patience_val, clip_val)

            # EQUAL ENSEMBLE: strictly arithmetic mean of the three prediction arrays
            p_equal = (p_lstm + p_tcn + p_cnn) / 3.0

            # Mathematical verification of arithmetic reproduction
            diff_check = np.max(np.abs(p_equal - (p_lstm + p_tcn + p_cnn) / 3.0))
            assert diff_check < 1e-10, f"Arithmetic mismatch in equal ensemble! diff={diff_check}"

            # Compute individual and ensemble metrics
            met_lstm = compute_metrics(y_test_true, p_lstm)
            met_tcn = compute_metrics(y_test_true, p_tcn)
            met_cnn = compute_metrics(y_test_true, p_cnn)
            met_equal = compute_metrics(y_test_true, p_equal)

            audit_rows.append({
                "seed": s,
                "LSTM_MAE": met_lstm["MAE"],
                "TCN_MAE": met_tcn["MAE"],
                "CNN_MAE": met_cnn["MAE"],
                "Equal_Ensemble_MAE": met_equal["MAE"],
                "Equal_Ensemble_RMSE": met_equal["RMSE"],
                "Equal_Ensemble_R2": met_equal["R2"],
                "num_test_origins": num_test_origins,
                "first_origin": first_origin,
                "last_origin": last_origin,
            })
            print(f"Seed {s:4d} | LSTM: {met_lstm['MAE']:6.2f} | TCN: {met_tcn['MAE']:6.2f} | CNN: {met_cnn['MAE']:6.2f} | Equal: {met_equal['MAE']:6.2f} MW")

        df_audit = pd.DataFrame(audit_rows)
        print(f"\n--- Aggregate Summary for {regime_name} ---")
        print(f"LSTM:  MAE = {df_audit['LSTM_MAE'].mean():.2f} +/- {df_audit['LSTM_MAE'].std():.2f} MW")
        print(f"TCN:   MAE = {df_audit['TCN_MAE'].mean():.2f} +/- {df_audit['TCN_MAE'].std():.2f} MW")
        print(f"CNN:   MAE = {df_audit['CNN_MAE'].mean():.2f} +/- {df_audit['CNN_MAE'].std():.2f} MW")
        print(f"Equal: MAE = {df_audit['Equal_Ensemble_MAE'].mean():.2f} +/- {df_audit['Equal_Ensemble_MAE'].std():.2f} MW")

        out_csv = f"research/results/audit_equal_ensemble_{regime_name}.csv"
        df_audit.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")


def train_expert(model, tr_loader, va_loader, te_loader, scale, mean, device, use_plateau, patience, clip_val):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if use_plateau:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)

    best_val_loss = float("inf")
    best_weights = None
    patience_cnt = 0

    for ep in range(1, 45):
        model.train()
        for bx, by in tr_loader:
            opt.zero_grad()
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = nn.functional.mse_loss(out, by)
            loss.backward()
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            opt.step()

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                va_loss += nn.functional.mse_loss(model(bx), by).item()
                n_va += 1
        va_loss /= n_va

        if use_plateau:
            sched.step(va_loss)
        else:
            sched.step()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in te_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0) * scale + mean
    return preds


if __name__ == "__main__":
    run_equal_ensemble_audit()
