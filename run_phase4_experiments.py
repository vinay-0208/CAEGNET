"""
Phase 4 Controlled Experiments Runner
=====================================
Executes the controlled training and evaluation across all 8 models:
1. Persistence / Naive-24
2. LSTM Standalone
3. TCN Standalone
4. CNN Standalone
5. Static Ensemble (1/3, 1/3, 1/3)
6. Standard Input-Based MoE
7. CAEG-Net (Without Recent Error - 3D Context)
8. Full CAEG-Net (Proposed - 4D Context with Out-of-Sample Recent Error)

All models are trained strictly on the training partition with early stopping on validation MSE.
All final metrics are computed strictly on the raw MW scale on the held-out test partition.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

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
from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    CAEGNet,
    StandardInputMoE,
    count_parameters
)
from train import (
    create_criterion,
    create_optimizer_and_scheduler,
    train_model
)
from evaluate import (
    compute_metrics,
    evaluate_model_on_loader
)
from experiments import (
    evaluate_persistence_naive24,
    evaluate_static_ensemble,
    format_results_table
)


def run_all_experiments(seed: int = 42, max_epochs: int = 25, patience: int = 6):
    print("=" * 70)
    print(f"STARTING PHASE 4 CONTROLLED TRAINING EXPERIMENTS (Seed: {seed})")
    print("=" * 70)
    
    # 1. Load Data and Pipeline
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    
    rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows)
    
    # Full 4D context
    C_tr_4d = extract_context_features(windows["train"]["X"], rec_tr)
    C_val_4d = extract_context_features(windows["val"]["X"], rec_val)
    C_test_4d = extract_context_features(windows["test"]["X"], rec_test)
    
    # 3D context without Recent Error (Trend, Volatility, Periodicity)
    C_tr_3d = C_tr_4d[:, :3]
    C_val_3d = C_val_4d[:, :3]
    C_test_3d = C_test_4d[:, :3]
    
    # Datasets
    ds_train_4d = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr_4d)
    ds_val_4d = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val_4d)
    ds_test_4d = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], C_test_4d)
    
    ds_train_3d = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr_3d)
    ds_val_3d = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], C_val_3d)
    ds_test_3d = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], C_test_3d)
    
    loaders_4d = create_dataloaders({"train": ds_train_4d, "val": ds_val_4d, "test": ds_test_4d}, batch_size=64, shuffle_train=True, seed=seed)
    loaders_3d = create_dataloaders({"train": ds_train_3d, "val": ds_val_3d, "test": ds_test_3d}, batch_size=64, shuffle_train=True, seed=seed)
    
    criterion = create_criterion("mse")
    results = {}
    history = {}
    
    # -----------------------------------------------------------------
    # 1. Baseline: Day-Ahead Persistence / Naive-24
    # -----------------------------------------------------------------
    print("\n>>> [1/8] Evaluating Persistence / Naive-24 Baseline...")
    naive_res = evaluate_persistence_naive24(windows["test"], scaler)
    results["Persistence_Naive24"] = {
        "metrics": naive_res["metrics"],
        "params": 0,
        "train_time": 0.0,
        "val_loss": None
    }
    print("    Naive-24 Test Metrics:", naive_res["metrics"])
    
    # -----------------------------------------------------------------
    # 2. Standalone Expert: LSTM
    # -----------------------------------------------------------------
    print("\n>>> [2/8] Training Standalone LSTM Expert...")
    torch.manual_seed(seed)
    lstm_model = LSTMExpert(input_dim=1, hidden_dim=64, num_layers=2, horizon=24)
    opt_lstm, sch_lstm = create_optimizer_and_scheduler(lstm_model, lr=1e-3, weight_decay=1e-4)
    train_lstm = train_model(
        lstm_model, loaders_4d["train"], loaders_4d["val"], criterion, opt_lstm, sch_lstm,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/lstm_standalone.pt",
        model_name="LSTM_Standalone"
    )
    eval_lstm = evaluate_model_on_loader(train_lstm["model"], loaders_4d["test"], scaler)
    results["LSTM_Standalone"] = {
        "metrics": eval_lstm["metrics"],
        "params": count_parameters(lstm_model)["total_trainable"],
        "train_time": train_lstm["training_time"],
        "val_loss": train_lstm["best_val_loss"]
    }
    history["LSTM_Standalone"] = {"train": train_lstm["train_losses"], "val": train_lstm["val_losses"]}
    print("    LSTM Test Metrics:", eval_lstm["metrics"])
    
    # -----------------------------------------------------------------
    # 3. Standalone Expert: TCN
    # -----------------------------------------------------------------
    print("\n>>> [3/8] Training Standalone TCN Expert...")
    torch.manual_seed(seed)
    tcn_model = TCNExpert(input_dim=1, channels=32, horizon=24)
    opt_tcn, sch_tcn = create_optimizer_and_scheduler(tcn_model, lr=1e-3, weight_decay=1e-4)
    train_tcn = train_model(
        tcn_model, loaders_4d["train"], loaders_4d["val"], criterion, opt_tcn, sch_tcn,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/tcn_standalone.pt",
        model_name="TCN_Standalone"
    )
    eval_tcn = evaluate_model_on_loader(train_tcn["model"], loaders_4d["test"], scaler)
    results["TCN_Standalone"] = {
        "metrics": eval_tcn["metrics"],
        "params": count_parameters(tcn_model)["total_trainable"],
        "train_time": train_tcn["training_time"],
        "val_loss": train_tcn["best_val_loss"]
    }
    history["TCN_Standalone"] = {"train": train_tcn["train_losses"], "val": train_tcn["val_losses"]}
    print("    TCN Test Metrics:", eval_tcn["metrics"])
    
    # -----------------------------------------------------------------
    # 4. Standalone Expert: CNN
    # -----------------------------------------------------------------
    print("\n>>> [4/8] Training Standalone CNN Expert...")
    torch.manual_seed(seed)
    cnn_model = CNNExpert(input_dim=1, horizon=24)
    opt_cnn, sch_cnn = create_optimizer_and_scheduler(cnn_model, lr=1e-3, weight_decay=1e-4)
    train_cnn = train_model(
        cnn_model, loaders_4d["train"], loaders_4d["val"], criterion, opt_cnn, sch_cnn,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/cnn_standalone.pt",
        model_name="CNN_Standalone"
    )
    eval_cnn = evaluate_model_on_loader(train_cnn["model"], loaders_4d["test"], scaler)
    results["CNN_Standalone"] = {
        "metrics": eval_cnn["metrics"],
        "params": count_parameters(cnn_model)["total_trainable"],
        "train_time": train_cnn["training_time"],
        "val_loss": train_cnn["best_val_loss"]
    }
    history["CNN_Standalone"] = {"train": train_cnn["train_losses"], "val": train_cnn["val_losses"]}
    print("    CNN Test Metrics:", eval_cnn["metrics"])
    
    # -----------------------------------------------------------------
    # 5. Baseline: Static Equal-Weighted Ensemble (1/3, 1/3, 1/3)
    # -----------------------------------------------------------------
    print("\n>>> [5/8] Evaluating Static Equal-Weighted Ensemble (LSTM+TCN+CNN)...")
    ens_res = evaluate_static_ensemble(eval_lstm["y_pred_raw"], eval_tcn["y_pred_raw"], eval_cnn["y_pred_raw"], eval_lstm["y_true_raw"])
    total_ens_params = count_parameters(lstm_model)["total_trainable"] + count_parameters(tcn_model)["total_trainable"] + count_parameters(cnn_model)["total_trainable"]
    results["Static_Equal_Ensemble"] = {
        "metrics": ens_res["metrics"],
        "params": total_ens_params,
        "train_time": train_lstm["training_time"] + train_tcn["training_time"] + train_cnn["training_time"],
        "val_loss": None
    }
    print("    Static Ensemble Test Metrics:", ens_res["metrics"])
    
    # -----------------------------------------------------------------
    # 6. Baseline: Standard Input-Based MoE
    # -----------------------------------------------------------------
    print("\n>>> [6/8] Training Standard Input-Based MoE Baseline...")
    torch.manual_seed(seed)
    moe_model = StandardInputMoE(input_dim=1, lookback=168, horizon=24)
    opt_moe, sch_moe = create_optimizer_and_scheduler(moe_model, lr=1e-3, weight_decay=1e-4)
    train_moe = train_model(
        moe_model, loaders_4d["train"], loaders_4d["val"], criterion, opt_moe, sch_moe,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/input_moe.pt",
        model_name="Standard_Input_MoE"
    )
    eval_moe = evaluate_model_on_loader(train_moe["model"], loaders_4d["test"], scaler)
    results["Standard_Input_MoE"] = {
        "metrics": eval_moe["metrics"],
        "params": count_parameters(moe_model)["total_trainable"],
        "train_time": train_moe["training_time"],
        "val_loss": train_moe["best_val_loss"]
    }
    history["Standard_Input_MoE"] = {"train": train_moe["train_losses"], "val": train_moe["val_losses"]}
    print("    Input MoE Test Metrics:", eval_moe["metrics"])
    
    # -----------------------------------------------------------------
    # 7. Ablation: CAEG-Net Without Recent Error (3D Context)
    # -----------------------------------------------------------------
    print("\n>>> [7/8] Training CAEG-Net Without Recent Error (3D Context Ablation)...")
    torch.manual_seed(seed)
    caeg_no_rec = CAEGNet(input_dim=1, horizon=24, context_dim=3, latent_context_dim=16)
    opt_no_rec, sch_no_rec = create_optimizer_and_scheduler(caeg_no_rec, lr=1e-3, weight_decay=1e-4)
    train_no_rec = train_model(
        caeg_no_rec, loaders_3d["train"], loaders_3d["val"], criterion, opt_no_rec, sch_no_rec,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/caeg_net_no_rec_error.pt",
        model_name="CAEG_Net_No_Recent_Error"
    )
    eval_no_rec = evaluate_model_on_loader(train_no_rec["model"], loaders_3d["test"], scaler)
    results["CAEG_Net_No_Recent_Error"] = {
        "metrics": eval_no_rec["metrics"],
        "params": count_parameters(caeg_no_rec)["total_trainable"],
        "train_time": train_no_rec["training_time"],
        "val_loss": train_no_rec["best_val_loss"]
    }
    history["CAEG_Net_No_Recent_Error"] = {"train": train_no_rec["train_losses"], "val": train_no_rec["val_losses"]}
    print("    CAEG-Net (No Recent Error) Test Metrics:", eval_no_rec["metrics"])
    
    # -----------------------------------------------------------------
    # 8. Proposed: Full CAEG-Net (4D Context with Recent Error)
    # -----------------------------------------------------------------
    print("\n>>> [8/8] Training Full Proposed CAEG-Net (4D Context with Causal Recent Error)...")
    torch.manual_seed(seed)
    caeg_full = CAEGNet(input_dim=1, horizon=24, context_dim=4, latent_context_dim=16)
    opt_full, sch_full = create_optimizer_and_scheduler(caeg_full, lr=1e-3, weight_decay=1e-4)
    train_full = train_model(
        caeg_full, loaders_4d["train"], loaders_4d["val"], criterion, opt_full, sch_full,
        max_epochs=max_epochs, patience=patience, checkpoint_path="checkpoints/caeg_net_full_best.pt",
        model_name="Full_CAEG_Net"
    )
    eval_full = evaluate_model_on_loader(train_full["model"], loaders_4d["test"], scaler)
    results["Full_CAEG_Net"] = {
        "metrics": eval_full["metrics"],
        "params": count_parameters(caeg_full)["total_trainable"],
        "train_time": train_full["training_time"],
        "val_loss": train_full["best_val_loss"],
        "weights": eval_full.get("weights")
    }
    history["Full_CAEG_Net"] = {"train": train_full["train_losses"], "val": train_full["val_losses"]}
    print("    Full CAEG-Net Test Metrics:", eval_full["metrics"])
    
    # Format and save comparison table
    df_table = format_results_table(results)
    print("\n" + "=" * 75)
    print("FINAL BENCHMARK COMPARISON TABLE (ORIGINAL RAW MW SCALE)")
    print("=" * 75)
    print(df_table.to_string(index=False))
    print("=" * 75)
    
    os.makedirs("results", exist_ok=True)
    df_table.to_csv("results/phase4_benchmark_results.csv", index=False)
    
    # Save training curves and predictions for notebook plotting
    np.savez(
        "results/phase4_experiment_cache.npz",
        history_json=json.dumps(history),
        y_test_true=eval_full["y_true_raw"],
        y_test_caeg=eval_full["y_pred_raw"],
        caeg_weights=eval_full["weights"],
        y_test_naive=naive_res["y_pred_raw"],
        y_test_static=ens_res["y_pred_raw"],
        y_test_moe=eval_moe["y_pred_raw"],
        y_test_no_rec=eval_no_rec["y_pred_raw"]
    )
    print("\nExperiment artifacts successfully saved to results/phase4_benchmark_results.csv and results/phase4_experiment_cache.npz")
    return results, history, df_table


if __name__ == "__main__":
    run_all_experiments(seed=42, max_epochs=25, patience=6)
