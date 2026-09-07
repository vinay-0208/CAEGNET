"""
CAEG-Net V2 Five-Seed Benchmark & Evaluation Suite
=================================================
Trains and evaluates the final selected CAEG-Net V2 architecture across the 5 canonical seeds:
[42, 123, 2024, 3407, 999]

CAEG-Net V2 Architecture Specification:
- Specialized Forecasting Experts: LSTM, TCN, CNN (Preserved)
- Context Vector: [Trend, Volatility, Periodicity, Causal Recent Forecast Error] in R^4
- Context Feature Encoder: Non-linear MLP mapping 4D context to 16D latent embedding
- Horizon-Dependent Gating Network: Non-linear MLP mapping 16D latent context to 24 x 3 Softmax routing matrix
- Dynamic Horizon-Specific Convex Fusion:
    y_hat[b, h] = sum_{i=1}^3 w_{h, i} * y_i[b, h]  for h in {1, ..., 24}
    where sum_{i=1}^3 w_{h, i} = 1.0, w_{h, i} > 0 for all h.

Strict Scientific Discipline:
- Identical 70/15/15 chronological data split.
- Scaler fitted strictly on training data.
- Checkpoints saved to checkpoints/caeg_v2/seed_{seed}/caeg_v2.pt.
- Test set evaluated strictly post-hoc on raw MW scale.
- Metrics recorded: MAE, MSE, RMSE, R^2, MAPE.
- Dependence-aware block comparison vs Baseline V1.
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

import argparse
import datetime
import json

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
from train import create_criterion, create_optimizer_and_scheduler, train_model


def run_v2_multiseed(force_retrain: bool = False, evaluate_only: bool = False):
    print("=" * 85)
    print("STARTING CAEG-NET V2 5-SEED BENCHMARK TRAINING & EVALUATION")
    print(f"Mode: {'Force Retrain' if force_retrain else ('Evaluate Only' if evaluate_only else 'Train / Verify')}")
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

    y_test_raw = windows["test"]["Y"] * scale_mw + mean_mw
    y_test_flat = y_test_raw.flatten()

    seeds = [42, 123, 2024, 3407, 999]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints/caeg_v2", exist_ok=True)
    os.makedirs("results/caeg_v2", exist_ok=True)

    v2_seed_metrics = []
    v2_predictions = {}
    v2_weights = {}

    for s in seeds:
        print(f"\n>>> Processing CAEG-Net V2 (Horizon-Dependent Gating) on Seed {s} <<<")
        torch.manual_seed(s)
        np.random.seed(s)

        ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr)
        ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val)
        ds_te = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], C_te)

        dls = create_dataloaders({"train": ds_tr, "val": ds_va, "test": ds_te}, batch_size=32, shuffle_train=True, seed=s)
        tr_loader, val_loader, te_loader = dls["train"], dls["val"], dls["test"]

        model = CAEGNet(horizon_dependent=True, context_dim=4).to(device)
        criterion = nn.MSELoss()
        optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-4)

        ckpt_dir = f"checkpoints/caeg_v2/seed_{s}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "caeg_v2.pt")
        meta_path = os.path.join(ckpt_dir, "metadata.json")

        ckpt_exists = os.path.exists(ckpt_path) and os.path.exists(meta_path)
        should_train = force_retrain or (not ckpt_exists and not evaluate_only)

        if should_train:
            print(f"Training Seed {s} from scratch...")
            t0 = time.time()
            train_res = train_model(
                model,
                tr_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                max_epochs=25,
                patience=6,
                checkpoint_path=ckpt_path,
                device=device,
                model_name=f"CAEG_V2_Seed_{s}",
                verbose=False,
            )
            train_time = time.time() - t0
            best_epoch = train_res["best_epoch"]
            best_val_loss = float(train_res.get("best_val_loss", train_res["val_losses"][best_epoch-1]))

            # Save full traceability metadata
            metadata = {
                "model_name": "CAEG_Net_V2",
                "horizon_dependent": True,
                "context_dim": 4,
                "seed": s,
                "best_epoch": int(best_epoch),
                "best_val_loss": round(best_val_loss, 5),
                "train_time_s": round(train_time, 2),
                "parameters": count_parameters(model),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            with open(meta_path, "w") as mf:
                json.dump(metadata, mf, indent=2)
        else:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run with --force-retrain.")
            with open(meta_path, "r") as mf:
                metadata = json.load(mf)
            assert metadata["seed"] == s, f"Seed mismatch: expected {s}, got {metadata['seed']}"
            assert metadata["horizon_dependent"] is True, "Config mismatch: horizon_dependent must be True"
            train_time = metadata.get("train_time_s", 0.0)
            best_epoch = metadata.get("best_epoch", 0)
            print(f"Loaded verified checkpoint for Seed {s} (Best Epoch: {best_epoch}, Runtime: {train_time}s)")

        # Load best early-stopped checkpoint
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Inference on test set
        test_preds = []
        test_w = []
        with torch.no_grad():
            for bx, by, bc in te_loader:
                yp, w, _ = model(bx.to(device), bc.to(device), return_diagnostics=True)
                test_preds.append(yp.cpu().numpy())
                test_w.append(w.cpu().numpy())

        p_scaled = np.concatenate(test_preds, axis=0) # [1294, 24]
        p_raw = p_scaled * scale_mw + mean_mw
        w_all = np.concatenate(test_w, axis=0)         # [1294, 24, 3]

        p_flat = p_raw.flatten()
        mae_mw = float(mean_absolute_error(y_test_flat, p_flat))
        mse_mw = float(mean_squared_error(y_test_flat, p_flat))
        rmse_mw = float(np.sqrt(mse_mw))
        r2 = float(r2_score(y_test_flat, p_flat))
        mape = float(np.mean(np.abs((y_test_flat - p_flat) / y_test_flat)) * 100.0)

        # Assertions
        assert np.isclose(rmse_mw, np.sqrt(mse_mw), atol=1e-5), "RMSE mismatch!"
        assert np.allclose(w_all.sum(axis=-1), np.ones((len(w_all), 24)), atol=1e-4), "Horizon routing weights convexity violated!"

        print(f"Seed {s} Results: MAE = {mae_mw:.2f} MW | RMSE = {rmse_mw:.2f} MW | R^2 = {r2:.4f} | MAPE = {mape:.2f}% | Best Epoch = {best_epoch}")

        v2_seed_metrics.append({
            "model": "CAEG_Net_V2",
            "seed": s,
            "MAE_MW": round(mae_mw, 2),
            "MSE_MW2": round(mse_mw, 2),
            "RMSE_MW": round(rmse_mw, 2),
            "R2": round(r2, 4),
            "MAPE_percent": round(mape, 2),
            "Train_Time_s": round(train_time, 1),
            "Best_Epoch": best_epoch,
        })
        v2_predictions[f"caeg_v2_seed_{s}"] = p_raw
        v2_weights[f"weights_v2_seed_{s}"] = w_all

    df_v2_seeds = pd.DataFrame(v2_seed_metrics)
    df_v2_seeds.to_csv("results/caeg_v2/v2_seed_metrics.csv", index=False)

    # Save multi-seed prediction cache
    np.savez(
        "results/caeg_v2/phase7_v2_multiseed_cache.npz",
        y_true_raw=y_test_raw,
        **v2_predictions,
        **v2_weights
    )

    # Compute Summary Statistics
    summary_v2 = {
        "model": "CAEG_Net_V2",
        "MAE_mean": round(float(df_v2_seeds["MAE_MW"].mean()), 2),
        "MAE_std": round(float(df_v2_seeds["MAE_MW"].std()), 2),
        "MSE_mean": round(float(df_v2_seeds["MSE_MW2"].mean()), 2),
        "MSE_std": round(float(df_v2_seeds["MSE_MW2"].std()), 2),
        "RMSE_mean": round(float(df_v2_seeds["RMSE_MW"].mean()), 2),
        "RMSE_std": round(float(df_v2_seeds["RMSE_MW"].std()), 2),
        "R2_mean": round(float(df_v2_seeds["R2"].mean()), 4),
        "R2_std": round(float(df_v2_seeds["R2"].std()), 4),
        "MAPE_mean": round(float(df_v2_seeds["MAPE_percent"].mean()), 2),
        "MAPE_std": round(float(df_v2_seeds["MAPE_percent"].std()), 2),
    }
    df_v2_sum = pd.DataFrame([summary_v2])
    df_v2_sum.to_csv("results/caeg_v2/v2_performance_summary.csv", index=False)

    print("\n" + "=" * 85)
    print("CAEG-NET V2 (PROPOSED) 5-SEED AGGREGATE RESULTS:")
    print("=" * 85)
    print(df_v2_sum.to_string(index=False))

    # Compare Directly with V1 Baseline
    df_v1_seeds = pd.read_csv("results/baseline_v1/phase7_seed_metrics.csv")
    v1_caeg = df_v1_seeds[df_v1_seeds["model"] == "Full_CAEG_Net"].copy()
    v1_caeg["seed"] = v1_caeg["seed"].astype(int)
    v1_caeg = v1_caeg.sort_values("seed").reset_index(drop=True)

    v2_sorted = df_v2_seeds.sort_values("seed").reset_index(drop=True)

    df_comp = pd.DataFrame({
        "Seed": v2_sorted["seed"],
        "V1_MAE (MW)": v1_caeg["MAE_MW"],
        "V2_MAE (MW)": v2_sorted["MAE_MW"],
        "MAE_Diff (V2 - V1)": round(v2_sorted["MAE_MW"] - v1_caeg["MAE_MW"], 2),
        "V1_RMSE (MW)": v1_caeg["RMSE_MW"],
        "V2_RMSE (MW)": v2_sorted["RMSE_MW"],
        "RMSE_Diff (V2 - V1)": round(v2_sorted["RMSE_MW"] - v1_caeg["RMSE_MW"], 2),
        "V1_MAPE (%)": v1_caeg["MAPE_percent"],
        "V2_MAPE (%)": v2_sorted["MAPE_percent"],
        "MAPE_Diff (%)": round(v2_sorted["MAPE_percent"] - v1_caeg["MAPE_percent"], 2),
        "V1_R2": v1_caeg["R2"],
        "V2_R2": v2_sorted["R2"],
        "R2_Diff": round(v2_sorted["R2"] - v1_caeg["R2"], 4),
    })
    df_comp.to_csv("results/caeg_v2/v1_vs_v2_comparison.csv", index=False)

    print("\n" + "=" * 85)
    print("SEED-BY-SEED COMPARISON: BASELINE V1 vs CAEG-NET V2")
    print("=" * 85)
    print(df_comp.to_string(index=False))

    # Paired Statistics Across 5 Seeds
    diffs = df_comp["MAE_Diff (V2 - V1)"].values
    t_stat, t_pval = stats.ttest_rel(v2_sorted["MAE_MW"], v1_caeg["MAE_MW"])
    w_stat, w_pval = stats.wilcoxon(v2_sorted["MAE_MW"], v1_caeg["MAE_MW"])
    print(f"\nSeed-Level Paired Analysis (n=5): Mean MAE Diff = {np.mean(diffs):+.2f} +/- {np.std(diffs, ddof=1):.2f} MW")
    print(f"Paired t-test p = {t_pval:.4f} | Wilcoxon p = {w_pval:.4f}")

    # Dependence-Aware Non-Overlapping 24-Hour Block Analysis
    block_indices = np.arange(0, len(y_test_raw), 24)
    y_test_blk = y_test_raw[block_indices]
    cache_v1 = np.load("results/baseline_v1/phase5_multiseed_cache.npz", allow_pickle=True)

    block_stats_list = []
    for s in seeds:
        pred_v1_blk = cache_v1[f"caeg_seed_{s}"][block_indices]
        pred_v2_blk = v2_predictions[f"caeg_v2_seed_{s}"][block_indices]

        mae_v1_blk = np.mean(np.abs(pred_v1_blk - y_test_blk), axis=1) # [54]
        mae_v2_blk = np.mean(np.abs(pred_v2_blk - y_test_blk), axis=1) # [54]
        d_blk = mae_v2_blk - mae_v1_blk

        wb_stat, wb_pval = stats.wilcoxon(mae_v2_blk, mae_v1_blk)
        tb_stat, tb_pval = stats.ttest_rel(mae_v2_blk, mae_v1_blk)
        ci95 = 1.96 * np.std(d_blk, ddof=1) / np.sqrt(len(d_blk))

        block_stats_list.append({
            "Seed": s,
            "Mean_Daily_Diff_MW": round(float(np.mean(d_blk)), 2),
            "CI95_MW": round(float(ci95), 2),
            "Wilcoxon_p": round(float(wb_pval), 4),
            "ttest_p": round(float(tb_pval), 4),
            "Favors": "CAEG_V2" if np.mean(d_blk) < 0 else "Baseline_V1"
        })

    df_block_stats = pd.DataFrame(block_stats_list)
    df_block_stats.to_csv("results/caeg_v2/v1_vs_v2_disjoint_blocks.csv", index=False)

    print("\n" + "=" * 85)
    print("DISJOINT 24-HOUR BLOCK PAIRED ANALYSIS (K=54 Non-Overlapping Blocks):")
    print("=" * 85)
    print(df_block_stats.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAEG-Net V2 Multi-Seed Benchmark Runner")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining models from scratch")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing checkpoints without retraining")
    args = parser.parse_args()

    run_v2_multiseed(force_retrain=args.force_retrain, evaluate_only=args.evaluate_only)
