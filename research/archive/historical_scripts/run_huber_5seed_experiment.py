"""
5-Seed Huber Loss vs MSE Benchmark Experiment (Canonical Setup)
=============================================================
Compares Canonical CAEG-Net V1 trained with MSELoss vs HuberLoss (delta=1.0)
across all 5 canonical seeds: [42, 123, 2024, 3407, 999].

Preserves:
- Batch size = 64 (canonical Phase 4/5 setting)
- AdamW lr=1e-3, weight_decay=1e-4
- StepLR(step_size=15, gamma=0.5)
- max_epochs=25, patience=6 early stopping on validation loss
- Architecture: Canonical CAEGNet V1 (121,531 parameters)
- Causal 4D context features and train-only StandardScaler
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
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
from evaluate import evaluate_model_on_loader

SEEDS = [42, 123, 2024, 3407, 999]

def main():
    print("=" * 80)
    print("STARTING 5-SEED HUBER LOSS VS MSE BENCHMARK EXPERIMENT")
    print(f"Seeds: {SEEDS}")
    print("Batch size: 64 | Optimizer: AdamW(lr=1e-3, wd=1e-4) | Scheduler: StepLR(15, 0.5)")
    print("=" * 80)

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

    ckpt_dir = "checkpoints/safe_improvements/huber"
    os.makedirs(ckpt_dir, exist_ok=True)
    res_dir = "results/safe_improvements"
    os.makedirs(res_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load canonical V1 metrics for baseline comparison
    v1_metrics_path = "results/baseline_v1/phase7_seed_metrics.csv"
    v1_df = pd.read_csv(v1_metrics_path)
    v1_caeg = v1_df[v1_df["model"] == "Full_CAEG_Net"].copy()
    v1_caeg["seed"] = v1_caeg["seed"].astype(int)
    v1_map = {row["seed"]: row for _, row in v1_caeg.iterrows()}

    # Canonical V1 validation telemetry
    v1_val_telemetry = {
        42: {"best_epoch": 13, "actual_epochs": 19, "val_mse": 0.44217},
        123: {"best_epoch": 13, "actual_epochs": 19, "val_mse": 0.44424},
        2024: {"best_epoch": 8, "actual_epochs": 14, "val_mse": 0.38535},
        3407: {"best_epoch": 11, "actual_epochs": 17, "val_mse": 0.40850},
        999: {"best_epoch": 8, "actual_epochs": 14, "val_mse": 0.36747}
    }

    huber_rows = []
    criterion_huber = nn.HuberLoss(delta=1.0)

    for s_idx, seed in enumerate(SEEDS):
        print("\n" + "-" * 70)
        print(f"[{s_idx+1}/{len(SEEDS)}] Running Huber Loss for Seed {seed}...")
        print("-" * 70)

        torch.manual_seed(seed)
        np.random.seed(seed)

        ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr)
        ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val)
        ds_te = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], C_te)

        # Strictly batch_size=64 (canonical!)
        dls = create_dataloaders({"train": ds_tr, "val": ds_va, "test": ds_te}, batch_size=64, shuffle_train=True, seed=seed)

        model = CAEGNet(input_dim=1, horizon=24, context_dim=4, latent_context_dim=16, horizon_dependent=False).to(device)
        opt, sch = create_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-4)

        ckpt_path = os.path.join(ckpt_dir, f"caeg_huber_seed_{seed}.pt")
        t0 = time.time()
        train_res = train_model(
            model,
            dls["train"],
            dls["val"],
            criterion_huber,
            opt,
            sch,
            max_epochs=25,
            patience=6,
            checkpoint_path=ckpt_path,
            device=device,
            model_name=f"Huber_{seed}",
            verbose=False
        )
        runtime = time.time() - t0

        # Load best model
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Evaluate on validation loader (both Huber loss and scaled MSE)
        val_eval = evaluate_model_on_loader(model, dls["val"], scaler)
        test_eval = evaluate_model_on_loader(model, dls["test"], scaler)

        # Also compute validation MSE in normalized scale for direct comparison with V1 Val MSE
        with torch.no_grad():
            val_preds_norm = []
            val_targets_norm = []
            for bx, by, bc in dls["val"]:
                bx, by, bc = bx.to(device), by.to(device), bc.to(device)
                p = model(bx, bc)
                if isinstance(p, tuple):
                    p = p[0]
                val_preds_norm.append(p.cpu().numpy())
                val_targets_norm.append(by.cpu().numpy())
            val_preds_norm = np.concatenate(val_preds_norm, axis=0)
            val_targets_norm = np.concatenate(val_targets_norm, axis=0)
            val_mse_norm = float(np.mean((val_preds_norm - val_targets_norm) ** 2))

        v1_row = v1_map[seed]
        v1_val_info = v1_val_telemetry[seed]

        row = {
            "seed": seed,
            "loss_fn": "Huber(delta=1.0)",
            "actual_epochs": len(train_res["val_losses"]),
            "best_epoch": train_res["best_epoch"],
            "val_huber_loss": round(train_res["best_val_loss"], 5),
            "val_mse_norm": round(val_mse_norm, 5),
            "val_mae_mw": round(val_eval["metrics"]["MAE"], 2),
            "val_rmse_mw": round(val_eval["metrics"]["RMSE"], 2),
            "test_mae_mw": round(test_eval["metrics"]["MAE"], 2),
            "test_mse_mw2": round(test_eval["metrics"]["MSE"], 2),
            "test_rmse_mw": round(test_eval["metrics"]["RMSE"], 2),
            "test_r2": round(test_eval["metrics"]["R2"], 4),
            "test_mape_pct": round(test_eval["metrics"]["MAPE"], 2),
            "train_time_s": round(runtime, 1),
            "v1_test_mae_mw": round(v1_row["MAE_MW"], 2),
            "v1_test_rmse_mw": round(v1_row["RMSE_MW"], 2),
            "v1_test_r2": round(v1_row["R2"], 4),
            "v1_val_mse_norm": round(v1_val_info["val_mse"], 5),
            "diff_test_mae_mw": round(test_eval["metrics"]["MAE"] - v1_row["MAE_MW"], 2),
        }
        huber_rows.append(row)

        print(f"    Seed {seed} Complete | Actual Ep: {row['actual_epochs']} | Best Ep: {row['best_epoch']} | Best Val Huber: {row['val_huber_loss']}")
        print(f"    Val MSE (norm): {row['val_mse_norm']} (V1: {row['v1_val_mse_norm']}) | Val MAE: {row['val_mae_mw']:.2f} MW")
        print(f"    Test MAE: {row['test_mae_mw']:.2f} MW (V1: {row['v1_test_mae_mw']:.2f} MW, diff: {row['diff_test_mae_mw']:+.2f} MW)")
        print(f"    Test RMSE: {row['test_rmse_mw']:.2f} MW | Test R2: {row['test_r2']:.4f} | MAPE: {row['test_mape_pct']:.2f}%")

    res_df = pd.DataFrame(huber_rows)
    out_csv = os.path.join(res_dir, "huber_5seed_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved 5-seed Huber results to {out_csv}")

    # Summary Statistics & Paired t-test
    huber_maes = res_df["test_mae_mw"].values
    v1_maes = res_df["v1_test_mae_mw"].values
    huber_val_mses = res_df["val_mse_norm"].values
    v1_val_mses = res_df["v1_val_mse_norm"].values

    diff_test_mae = huber_maes - v1_maes
    diff_val_mse = huber_val_mses - v1_val_mses

    t_test_mae, p_test_mae = stats.ttest_rel(huber_maes, v1_maes)
    t_val_mse, p_val_mse = stats.ttest_rel(huber_val_mses, v1_val_mses)

    print("\n" + "=" * 80)
    print("5-SEED EXPERIMENT SUMMARY & DECISION AUDIT")
    print("=" * 80)
    print(f"Huber Test MAE: {huber_maes.mean():.2f} +/- {huber_maes.std(ddof=1):.2f} MW")
    print(f"Canonical V1 (MSE) Test MAE: {v1_maes.mean():.2f} +/- {v1_maes.std(ddof=1):.2f} MW")
    print(f"Test MAE Paired Difference (Huber - V1): {diff_test_mae.mean():+.2f} +/- {diff_test_mae.std(ddof=1):.2f} MW")
    print(f"Test MAE Paired t-test: t = {t_test_mae:.4f}, p = {p_test_mae:.4f}")
    print("-" * 50)
    print(f"Huber Val MSE (norm): {huber_val_mses.mean():.5f} +/- {huber_val_mses.std(ddof=1):.5f}")
    print(f"Canonical V1 Val MSE (norm): {v1_val_mses.mean():.5f} +/- {v1_val_mses.std(ddof=1):.5f}")
    print(f"Val MSE Paired Difference (Huber - V1): {diff_val_mse.mean():+.5f} +/- {diff_val_mse.std(ddof=1):.5f}")
    print(f"Val MSE Paired t-test: t = {t_val_mse:.4f}, p = {p_val_mse:.4f}")

    # Decision rule output
    print("\n--- DECISION RULE AUDIT ---")
    if diff_val_mse.mean() < 0 and p_val_mse < 0.05:
        print("DECISION: Huber loss improves validation MSE with statistical significance.")
    else:
        print("DECISION: Huber loss does NOT achieve statistically superior validation MSE over Canonical MSE.")
        print("ACADEMIC RULE PRESERVED: Canonical CAEG-Net V1 (MSE) remains the primary champion baseline.")

if __name__ == "__main__":
    main()
