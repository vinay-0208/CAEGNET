import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
)
from caeg_net import CNNExpert
from evaluate import compute_metrics

def inspect_cnn_seed3407():
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(windows["train"]["X"]).float(), torch.from_numpy(windows["train"]["Y"]).float()),
        batch_size=64, shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(windows["val"]["X"]).float(), torch.from_numpy(windows["val"]["Y"]).float()),
        batch_size=64, shuffle=False
    )
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(windows["test"]["X"]).float(), torch.from_numpy(windows["test"]["Y"]).float()),
        batch_size=64, shuffle=False
    )

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    y_test_true = windows["test"]["Y"] * scale + mean

    print("=== INSPECTING STANDALONE CNN ON SEED 3407 ===")
    torch.manual_seed(3407)
    np.random.seed(3407)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(3407)

    cnn = CNNExpert().to(device)
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    best_ep = 0

    for ep in range(1, 35):
        cnn.train()
        tr_loss = 0.0
        n_tr = 0
        for bx, by in tr_loader:
            optimizer.zero_grad()
            bx, by = bx.to(device), by.to(device)
            out = cnn(bx)
            loss = nn.functional.mse_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            n_tr += 1

        cnn.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                va_loss += nn.functional.mse_loss(cnn(bx), by).item()
                n_va += 1
        va_loss /= n_va
        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_ep = ep
            best_weights = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:2d} | Train Loss: {tr_loss/n_tr:.4f} | Val Loss: {va_loss:.4f} (Best: {best_val_loss:.4f} at ep {best_ep}) | LR: {optimizer.param_groups[0]['lr']}")

    cnn.load_state_dict(best_weights)
    cnn.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in te_loader:
            preds.append(cnn(bx.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0) * scale + mean
    met = compute_metrics(y_test_true, preds)
    print(f"\nFinal Test Metrics for Seed 3407 CNN: MAE={met['MAE']:.2f} MW, RMSE={met['RMSE']:.2f} MW, R2={met['R2']:.4f}")

if __name__ == "__main__":
    inspect_cnn_seed3407()
