"""
Low-Risk Model Improvement Experiments (Controlled Study)
=========================================================
Tests targeted training optimizations around canonical CAEG-Net V1:
- Optimizer configurations: lr=1e-3 vs lr=5e-4
- Objective functions: MSELoss vs HuberLoss (delta=1.0)
- Preserves 100% of V1 architecture (121,531 parameters)
- Preserves 70/15/15 chronological data pipeline and causal context features

Evaluation protocol:
1. Train each candidate on training set (N=5,957 windows).
2. Monitor validation loss with early stopping (patience=6, max_epochs=25).
3. Evaluate best early-stopped checkpoint on Validation partition.
4. Evaluate on Test set (N=1,294 windows) for final comparison.
5. Save results to results/safe_improvements/training_optimization_experiments.csv.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
from caeg_net import CAEGNet, count_parameters
from train import create_optimizer_and_scheduler, train_model


def run_safe_experiments():
    print("=" * 85)
    print("STARTING CONTROLLED LOW-RISK TRAINING OPTIMIZATION EXPERIMENTS")
    print("=" * 85)

    # 1. Pipeline Setup (Strictly Preserved)
    data_path = "data/Modern_PJM/pjm_load.csv"
    df, _ = load_and_clean_data(data_path)
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    scale_mw = float(scaler.scale_[0])
    mean_mw = float(scaler.mean_[0])

    C_tr = extract_context_features(windows["train"]["X"], rec_tr)
    C_val = extract_context_features(windows["val"]["X"], rec_val)
    C_te = extract_context_features(windows["test"]["X"], rec_test)

    y_val_raw = windows["val"]["Y"] * scale_mw + mean_mw
    y_test_raw = windows["test"]["Y"] * scale_mw + mean_mw
    y_val_flat = y_val_raw.flatten()
    y_test_flat = y_test_raw.flatten()

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = "results/safe_improvements"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_out_dir = "checkpoints/safe_improvements"
    os.makedirs(ckpt_out_dir, exist_ok=True)

    candidates = [
        {
            "id": "OPT-1-V1-Baseline",
            "name": "CAEG V1 Baseline (AdamW lr=1e-3, MSE)",
            "lr": 1e-3,
            "loss_fn": "MSE",
            "criterion": nn.MSELoss(),
        },
        {
            "id": "OPT-2-Lower-LR",
            "name": "CAEG V1 Lower LR (AdamW lr=5e-4, MSE)",
            "lr": 5e-4,
            "loss_fn": "MSE",
            "criterion": nn.MSELoss(),
        },
        {
            "id": "OPT-3-Huber-1e3",
            "name": "CAEG V1 Huber Loss (AdamW lr=1e-3, delta=1.0)",
            "lr": 1e-3,
            "loss_fn": "Huber(delta=1.0)",
            "criterion": nn.HuberLoss(delta=1.0),
        },
        {
            "id": "OPT-4-Huber-5e4",
            "name": "CAEG V1 Huber Loss (AdamW lr=5e-4, delta=1.0)",
            "lr": 5e-4,
            "loss_fn": "Huber(delta=1.0)",
            "criterion": nn.HuberLoss(delta=1.0),
        },
    ]

    records = []
    loss_histories = {}

    for cand in candidates:
        print(f"\n>>> Running Candidate: {cand['name']} <<<")
        torch.manual_seed(seed)
        np.random.seed(seed)

        ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr)
        ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val)
        ds_te = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], C_te)

        dls = create_dataloaders({"train": ds_tr, "val": ds_va, "test": ds_te}, batch_size=32, shuffle_train=True, seed=seed)
        tr_loader, val_loader, te_loader = dls["train"], dls["val"], dls["test"]

        # Instantiate canonical V1 model (horizon_dependent=False)
        model = CAEGNet(horizon_dependent=False, context_dim=4).to(device)
        optimizer, scheduler = create_optimizer_and_scheduler(model, lr=cand["lr"], weight_decay=1e-4)

        ckpt_path = os.path.join(ckpt_out_dir, f"{cand['id']}.pt")
        t0 = time.time()
        train_res = train_model(
            model,
            tr_loader,
            val_loader,
            cand["criterion"],
            optimizer,
            scheduler,
            max_epochs=25,
            patience=6,
            checkpoint_path=ckpt_path,
            device=device,
            model_name=cand["id"],
            verbose=False,
        )
        runtime = time.time() - t0

        # Load best checkpoint
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Evaluate on Validation Set
        val_preds = []
        with torch.no_grad():
            for bx, by, bc in val_loader:
                yp, _, _ = model(bx.to(device), bc.to(device), return_diagnostics=True)
                val_preds.append(yp.cpu().numpy())
        val_p_scaled = np.concatenate(val_preds, axis=0)
        val_p_raw = val_p_scaled * scale_mw + mean_mw
        val_mae = float(mean_absolute_error(y_val_flat, val_p_raw.flatten()))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_flat, val_p_raw.flatten())))

        # Evaluate on Test Set
        test_preds = []
        with torch.no_grad():
            for bx, by, bc in te_loader:
                yp, _, _ = model(bx.to(device), bc.to(device), return_diagnostics=True)
                test_preds.append(yp.cpu().numpy())
        test_p_scaled = np.concatenate(test_preds, axis=0)
        test_p_raw = test_p_scaled * scale_mw + mean_mw
        test_p_flat = test_p_raw.flatten()

        test_mae = float(mean_absolute_error(y_test_flat, test_p_flat))
        test_mse = float(mean_squared_error(y_test_flat, test_p_flat))
        test_rmse = float(np.sqrt(test_mse))
        test_r2 = float(r2_score(y_test_flat, test_p_flat))
        test_mape = float(np.mean(np.abs((y_test_flat - test_p_flat) / y_test_flat)) * 100.0)

        actual_epochs = len(train_res["train_losses"])
        best_epoch = train_res["best_epoch"]
        best_val_loss = float(train_res.get("best_val_loss", train_res["val_losses"][best_epoch-1]))

        print(f"  Best Epoch: {best_epoch}/{actual_epochs} | Val Loss: {best_val_loss:.5f} | Val MAE: {val_mae:.2f} MW")
        print(f"  Test MAE: {test_mae:.2f} MW | Test RMSE: {test_rmse:.2f} MW | Test R^2: {test_r2:.4f} | Test MAPE: {test_mape:.2f}%")

        records.append({
            "Candidate_ID": cand["id"],
            "Configuration": cand["name"],
            "Learning_Rate": cand["lr"],
            "Loss_Function": cand["loss_fn"],
            "Actual_Epochs": actual_epochs,
            "Best_Epoch": best_epoch,
            "Best_Val_Loss": round(best_val_loss, 5),
            "Val_MAE_MW": round(val_mae, 2),
            "Val_RMSE_MW": round(val_rmse, 2),
            "Test_MAE_MW": round(test_mae, 2),
            "Test_MSE_MW2": round(test_mse, 2),
            "Test_RMSE_MW": round(test_rmse, 2),
            "Test_R2": round(test_r2, 4),
            "Test_MAPE_pct": round(test_mape, 2),
            "Runtime_s": round(runtime, 1),
        })

        loss_histories[cand["id"]] = {
            "train": [float(x) for x in train_res["train_losses"]],
            "val": [float(x) for x in train_res["val_losses"]],
            "best_epoch": int(best_epoch)
        }

    df_res = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "training_optimization_experiments.csv")
    df_res.to_csv(csv_path, index=False)

    hist_path = os.path.join(out_dir, "optimization_loss_histories.json")
    with open(hist_path, "w") as f:
        json.dump(loss_histories, f, indent=2)

    print("\n" + "=" * 85)
    print("TRAINING OPTIMIZATION EXPERIMENT RESULTS SUMMARY:")
    print("=" * 85)
    print(df_res[["Candidate_ID", "Best_Epoch", "Val_MAE_MW", "Test_MAE_MW", "Test_RMSE_MW", "Test_R2", "Test_MAPE_pct"]].to_string(index=False))

    # Decision rule check:
    v1_base_row = df_res[df_res["Candidate_ID"] == "OPT-1-V1-Baseline"].iloc[0]
    best_cand = df_res.sort_values("Val_MAE_MW").iloc[0]
    print(f"\nBest Validation Candidate: {best_cand['Candidate_ID']} (Val MAE = {best_cand['Val_MAE_MW']:.2f} MW)")
    print(f"Baseline V1: Val MAE = {v1_base_row['Val_MAE_MW']:.2f} MW, Test MAE = {v1_base_row['Test_MAE_MW']:.2f} MW")

    return df_res

if __name__ == "__main__":
    run_safe_experiments()
