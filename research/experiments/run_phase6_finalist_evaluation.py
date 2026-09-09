"""
CAEG-Net Phase 6: Five-Seed Finalist Evaluation
===============================================
Locked Evaluation of the Original CAEG-Net Architecture
Governing Roadmap: Phase 6 of the Original 14-Phase Research Roadmap

Finalist Set:
1. F1 — Original_CAEGNet_V1: Canonical Softmax router, 4D context, 121,531 params
2. F2 — Bounded_CAEGNet_rho0.50: Bounded router w = (1-rho)w0 + rho*q (rho=0.50), 121,531 params
3. B1 — Static_Equal_Ensemble: y = (y_LSTM + y_TCN + y_CNN) / 3, 120,504 params
4. B2 — Standalone_LSTM: 56,152 params
5. B3 — Standalone_TCN: 36,952 params
6. B4 — Standalone_CNN: 27,400 params

Five Canonical Seeds: [42, 123, 999, 2024, 3407]
Protocol: AdamW (lr=1e-3, weight_decay=1e-4), StepLR (step_size=15, gamma=0.5), max_epochs=45, patience=7.
Checkpoint: Best Validation MSE.
Statistical Evaluation: K=53 non-overlapping 24h daily blocks, Paired t-test, Wilcoxon, DM_HLN, Holm-Bonferroni.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)
from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    count_parameters,
)
from research.original_caeg import OriginalCAEGNetPhase5
from evaluate import compute_metrics


def train_model_canonical(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    step_size: int = 15,
    gamma: float = 0.5,
    max_epochs: int = 45,
    patience: int = 7,
    device: Optional[torch.device] = None,
    conservative_rho: Optional[float] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    start_time = time.time()
    actual_epochs = 0

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "router"):
                # CAEGNet model
                out, _, _ = model(x, c, conservative_rho=conservative_rho)
            else:
                # Standalone expert
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]

            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

                if hasattr(model, "router"):
                    y_pred, _, _ = model(x, c, conservative_rho=conservative_rho)
                else:
                    y_pred = model(x)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]

                val_mse = F.mse_loss(y_pred, y)
                total_val_loss += val_mse.item()
                n_val_batches += 1

        val_loss = total_val_loss / max(n_val_batches, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "best_val_loss": best_val_loss,
        "train_time_seconds": time.time() - start_time,
    }


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    scaler: object,
    device: torch.device,
    conservative_rho: Optional[float] = None,
) -> Dict:
    model.eval()
    all_preds = []
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "router"):
                y_pred, w, _ = model(x, c, conservative_rho=conservative_rho)
                all_weights.append(w.cpu().numpy())
            else:
                y_pred = model(x)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

            all_preds.append(y_pred.cpu().numpy())

    preds_scaled = np.concatenate(all_preds, axis=0)
    preds_mw = preds_scaled * float(scaler.scale_[0]) + float(scaler.mean_[0])
    weights = np.concatenate(all_weights, axis=0) if len(all_weights) > 0 else None

    return {
        "preds_scaled": preds_scaled,
        "preds_mw": preds_mw,
        "weights": weights,
    }


def compute_holmd_bonferroni(p_values: List[float]) -> List[float]:
    """
    Apply step-down Holm-Bonferroni correction.
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        p_adj = p * (m - rank)
        cum_max = max(cum_max, p_adj)
        cum_max = min(cum_max, 1.0)
        adjusted[orig_idx] = cum_max

    return adjusted


def main():
    print("=" * 80)
    print("CAEG-NET PHASE 6: FIVE-SEED FINALIST EVALUATION")
    print("Roadmap Phase: Phase 6 of the Original 14-Phase Research Roadmap")
    print("Test Set Status: FROZEN FROM FURTHER MODEL DEVELOPMENT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    results_dir = os.path.join(repo_root, "research", "results")
    plots_dir = os.path.join(results_dir, "phase6", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Pipeline Data Preparation
    print("\n[1/5] Preparing canonical 70/15/15 PJM dataset...")
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    ctx_train = extract_context_features(windows["train"]["X"], rec_tr)
    ctx_val = extract_context_features(windows["val"]["X"], rec_val)
    ctx_test = extract_context_features(windows["test"]["X"], rec_test)

    # Ground truth targets in MW
    y_test_mw = windows["test"]["Y"] * float(scaler.scale_[0]) + float(scaler.mean_[0])
    N_test, H_test = y_test_mw.shape
    print(f"Test partition ready: N = {N_test} origins, H = {H_test} hours")
    print(f"Scaler parameters: mu = {float(scaler.mean_[0]):.4f} MW, sigma = {float(scaler.scale_[0]):.4f} MW")

    def get_loader(part: str, batch_size: int = 64, shuffle: bool = False):
        x = torch.from_numpy(windows[part]["X"]).float()
        y = torch.from_numpy(windows[part]["Y"]).float()
        c = torch.from_numpy(ctx_train if part == "train" else ctx_val if part == "val" else ctx_test).float()
        ds = TensorDataset(x, y, c)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())

    val_loader = get_loader("val", shuffle=False)
    test_loader = get_loader("test", shuffle=False)

    canonical_seeds = [42, 123, 999, 2024, 3407]
    print(f"Evaluating across 5 canonical seeds: {canonical_seeds}")

    # Data structures for collecting results
    seed_records = []
    test_predictions = {
        "Original_CAEGNet_V1": {},
        "Bounded_CAEGNet_rho0.50": {},
        "Static_Equal_Ensemble": {},
        "Standalone_LSTM": {},
        "Standalone_TCN": {},
        "Standalone_CNN": {},
    }
    routing_stats_list = []

    # Model parameter reference
    dummy_lstm = LSTMExpert()
    dummy_tcn = TCNExpert()
    dummy_cnn = CNNExpert()
    dummy_v1 = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, conservative_rho=None)
    dummy_br = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, conservative_rho=0.50)

    param_counts = {
        "Standalone_LSTM": count_parameters(dummy_lstm)["total_params"],
        "Standalone_TCN": count_parameters(dummy_tcn)["total_params"],
        "Standalone_CNN": count_parameters(dummy_cnn)["total_params"],
        "Static_Equal_Ensemble": (
            count_parameters(dummy_lstm)["total_params"]
            + count_parameters(dummy_tcn)["total_params"]
            + count_parameters(dummy_cnn)["total_params"]
        ),
        "Original_CAEGNet_V1": count_parameters(dummy_v1)["total_params"],
        "Bounded_CAEGNet_rho0.50": count_parameters(dummy_br)["total_params"],
    }
    print(f"Model parameters: {param_counts}")

    print("\n[2/5] Executing Five-Seed Training & Test Evaluation...")
    for seed in canonical_seeds:
        print(f"\n{'=' * 60}\n>>> RUNNING SEED: {seed} <<<\n{'=' * 60}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader = get_loader("train", shuffle=True)

        # 1. Standalone LSTM
        print(f"[{seed}] Training Standalone LSTM...")
        m_lstm = LSTMExpert().to(device)
        tr_lstm = train_model_canonical(m_lstm, train_loader, val_loader, lr=1e-3, device=device)
        pred_lstm = predict_model(m_lstm, test_loader, scaler, device)
        met_lstm = compute_metrics(y_test_mw, pred_lstm["preds_mw"])
        test_predictions["Standalone_LSTM"][seed] = pred_lstm["preds_mw"]
        seed_records.append({
            "model": "Standalone_LSTM",
            "seed": seed,
            "MAE": met_lstm["MAE"],
            "RMSE": met_lstm["RMSE"],
            "MSE": met_lstm["MSE"],
            "R2": met_lstm["R2"],
            "MAPE": met_lstm["MAPE"],
            "best_epoch": tr_lstm["best_epoch"],
            "actual_epochs": tr_lstm["actual_epochs"],
            "train_time_seconds": tr_lstm["train_time_seconds"],
            "params": param_counts["Standalone_LSTM"],
        })
        print(f"  LSTM Seed {seed}: MAE = {met_lstm['MAE']:.2f} MW, R2 = {met_lstm['R2']:.4f} (Epoch {tr_lstm['best_epoch']}/{tr_lstm['actual_epochs']})")

        # 2. Standalone TCN
        print(f"[{seed}] Training Standalone TCN...")
        m_tcn = TCNExpert().to(device)
        tr_tcn = train_model_canonical(m_tcn, train_loader, val_loader, lr=1e-3, device=device)
        pred_tcn = predict_model(m_tcn, test_loader, scaler, device)
        met_tcn = compute_metrics(y_test_mw, pred_tcn["preds_mw"])
        test_predictions["Standalone_TCN"][seed] = pred_tcn["preds_mw"]
        seed_records.append({
            "model": "Standalone_TCN",
            "seed": seed,
            "MAE": met_tcn["MAE"],
            "RMSE": met_tcn["RMSE"],
            "MSE": met_tcn["MSE"],
            "R2": met_tcn["R2"],
            "MAPE": met_tcn["MAPE"],
            "best_epoch": tr_tcn["best_epoch"],
            "actual_epochs": tr_tcn["actual_epochs"],
            "train_time_seconds": tr_tcn["train_time_seconds"],
            "params": param_counts["Standalone_TCN"],
        })
        print(f"  TCN Seed {seed}: MAE = {met_tcn['MAE']:.2f} MW, R2 = {met_tcn['R2']:.4f} (Epoch {tr_tcn['best_epoch']}/{tr_tcn['actual_epochs']})")

        # 3. Standalone CNN
        print(f"[{seed}] Training Standalone CNN...")
        m_cnn = CNNExpert().to(device)
        tr_cnn = train_model_canonical(m_cnn, train_loader, val_loader, lr=1e-3, device=device)
        pred_cnn = predict_model(m_cnn, test_loader, scaler, device)
        met_cnn = compute_metrics(y_test_mw, pred_cnn["preds_mw"])
        test_predictions["Standalone_CNN"][seed] = pred_cnn["preds_mw"]
        seed_records.append({
            "model": "Standalone_CNN",
            "seed": seed,
            "MAE": met_cnn["MAE"],
            "RMSE": met_cnn["RMSE"],
            "MSE": met_cnn["MSE"],
            "R2": met_cnn["R2"],
            "MAPE": met_cnn["MAPE"],
            "best_epoch": tr_cnn["best_epoch"],
            "actual_epochs": tr_cnn["actual_epochs"],
            "train_time_seconds": tr_cnn["train_time_seconds"],
            "params": param_counts["Standalone_CNN"],
        })
        print(f"  CNN Seed {seed}: MAE = {met_cnn['MAE']:.2f} MW, R2 = {met_cnn['R2']:.4f} (Epoch {tr_cnn['best_epoch']}/{tr_cnn['actual_epochs']})")

        # 4. Static Equal Ensemble (B1)
        pred_equal = (pred_lstm["preds_mw"] + pred_tcn["preds_mw"] + pred_cnn["preds_mw"]) / 3.0
        met_equal = compute_metrics(y_test_mw, pred_equal)
        test_predictions["Static_Equal_Ensemble"][seed] = pred_equal
        seed_records.append({
            "model": "Static_Equal_Ensemble",
            "seed": seed,
            "MAE": met_equal["MAE"],
            "RMSE": met_equal["RMSE"],
            "MSE": met_equal["MSE"],
            "R2": met_equal["R2"],
            "MAPE": met_equal["MAPE"],
            "best_epoch": 0,
            "actual_epochs": 0,
            "train_time_seconds": (
                tr_lstm["train_time_seconds"]
                + tr_tcn["train_time_seconds"]
                + tr_cnn["train_time_seconds"]
            ),
            "params": param_counts["Static_Equal_Ensemble"],
        })
        print(f"  Equal Ensemble Seed {seed}: MAE = {met_equal['MAE']:.2f} MW, R2 = {met_equal['R2']:.4f}")

        # 5. Canonical CAEG-Net V1 (F1)
        print(f"[{seed}] Training Canonical CAEG-Net V1...")
        m_v1 = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, conservative_rho=None).to(device)
        tr_v1 = train_model_canonical(m_v1, train_loader, val_loader, lr=1e-3, device=device, conservative_rho=None)
        pred_v1 = predict_model(m_v1, test_loader, scaler, device, conservative_rho=None)
        met_v1 = compute_metrics(y_test_mw, pred_v1["preds_mw"])
        test_predictions["Original_CAEGNet_V1"][seed] = pred_v1["preds_mw"]
        seed_records.append({
            "model": "Original_CAEGNet_V1",
            "seed": seed,
            "MAE": met_v1["MAE"],
            "RMSE": met_v1["RMSE"],
            "MSE": met_v1["MSE"],
            "R2": met_v1["R2"],
            "MAPE": met_v1["MAPE"],
            "best_epoch": tr_v1["best_epoch"],
            "actual_epochs": tr_v1["actual_epochs"],
            "train_time_seconds": tr_v1["train_time_seconds"],
            "params": param_counts["Original_CAEGNet_V1"],
        })
        print(f"  CAEG-Net V1 Seed {seed}: MAE = {met_v1['MAE']:.2f} MW, R2 = {met_v1['R2']:.4f} (Epoch {tr_v1['best_epoch']}/{tr_v1['actual_epochs']})")

        # Diagnostics for V1
        w_v1 = pred_v1["weights"]  # [1294, 3]
        eps = 1e-8
        ent_v1 = -np.sum(w_v1 * np.log(w_v1 + eps), axis=-1)
        eff_v1 = 1.0 / np.sum(w_v1 ** 2, axis=-1)
        routing_stats_list.append({
            "model": "Original_CAEGNet_V1",
            "seed": seed,
            "mean_lstm_weight": float(np.mean(w_v1[:, 0])),
            "mean_tcn_weight": float(np.mean(w_v1[:, 1])),
            "mean_cnn_weight": float(np.mean(w_v1[:, 2])),
            "std_lstm_weight": float(np.std(w_v1[:, 0])),
            "std_tcn_weight": float(np.std(w_v1[:, 1])),
            "std_cnn_weight": float(np.std(w_v1[:, 2])),
            "mean_entropy": float(np.mean(ent_v1)),
            "effective_experts": float(np.mean(eff_v1)),
            "weight_sum_min": float(np.min(np.sum(w_v1, axis=-1))),
            "weight_sum_max": float(np.max(np.sum(w_v1, axis=-1))),
            "weight_min": float(np.min(w_v1)),
            "weight_max": float(np.max(w_v1)),
        })

        # 6. Bounded CAEG-Net rho=0.50 (F2)
        print(f"[{seed}] Training Bounded CAEG-Net (rho=0.50)...")
        m_br = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, conservative_rho=0.50).to(device)
        tr_br = train_model_canonical(m_br, train_loader, val_loader, lr=1e-3, device=device, conservative_rho=0.50)
        pred_br = predict_model(m_br, test_loader, scaler, device, conservative_rho=0.50)
        met_br = compute_metrics(y_test_mw, pred_br["preds_mw"])
        test_predictions["Bounded_CAEGNet_rho0.50"][seed] = pred_br["preds_mw"]
        seed_records.append({
            "model": "Bounded_CAEGNet_rho0.50",
            "seed": seed,
            "MAE": met_br["MAE"],
            "RMSE": met_br["RMSE"],
            "MSE": met_br["MSE"],
            "R2": met_br["R2"],
            "MAPE": met_br["MAPE"],
            "best_epoch": tr_br["best_epoch"],
            "actual_epochs": tr_br["actual_epochs"],
            "train_time_seconds": tr_br["train_time_seconds"],
            "params": param_counts["Bounded_CAEGNet_rho0.50"],
        })
        print(f"  Bounded CAEG (rho=0.50) Seed {seed}: MAE = {met_br['MAE']:.2f} MW, R2 = {met_br['R2']:.4f} (Epoch {tr_br['best_epoch']}/{tr_br['actual_epochs']})")

        # Diagnostics for Bounded
        w_br = pred_br["weights"]  # [1294, 3]
        ent_br = -np.sum(w_br * np.log(w_br + eps), axis=-1)
        eff_br = 1.0 / np.sum(w_br ** 2, axis=-1)
        routing_stats_list.append({
            "model": "Bounded_CAEGNet_rho0.50",
            "seed": seed,
            "mean_lstm_weight": float(np.mean(w_br[:, 0])),
            "mean_tcn_weight": float(np.mean(w_br[:, 1])),
            "mean_cnn_weight": float(np.mean(w_br[:, 2])),
            "std_lstm_weight": float(np.std(w_br[:, 0])),
            "std_tcn_weight": float(np.std(w_br[:, 1])),
            "std_cnn_weight": float(np.std(w_br[:, 2])),
            "mean_entropy": float(np.mean(ent_br)),
            "effective_experts": float(np.mean(eff_br)),
            "weight_sum_min": float(np.min(np.sum(w_br, axis=-1))),
            "weight_sum_max": float(np.max(np.sum(w_br, axis=-1))),
            "weight_min": float(np.min(w_br)),
            "weight_max": float(np.max(w_br)),
        })

    # Save per-seed results CSV
    df_seed_results = pd.DataFrame(seed_records)
    seed_csv_path = os.path.join(results_dir, "phase6_seed_results.csv")
    df_seed_results.to_csv(seed_csv_path, index=False)
    print(f"\n[3/5] Saved per-seed results to: {seed_csv_path}")

    # Compute Model Comparison Summary
    models = [
        "Original_CAEGNet_V1",
        "Bounded_CAEGNet_rho0.50",
        "Static_Equal_Ensemble",
        "Standalone_TCN",
        "Standalone_LSTM",
        "Standalone_CNN",
    ]
    summary_rows = []
    for m in models:
        sub = df_seed_results[df_seed_results["model"] == m]
        summary_rows.append({
            "model": m,
            "mae_mean": float(sub["MAE"].mean()),
            "mae_std": float(sub["MAE"].std()),
            "rmse_mean": float(sub["RMSE"].mean()),
            "rmse_std": float(sub["RMSE"].std()),
            "mse_mean": float(sub["MSE"].mean()),
            "mse_std": float(sub["MSE"].std()),
            "r2_mean": float(sub["R2"].mean()),
            "r2_std": float(sub["R2"].std()),
            "mape_mean": float(sub["MAPE"].mean()),
            "mape_std": float(sub["MAPE"].std()),
            "train_time_mean": float(sub["train_time_seconds"].mean()),
            "params": param_counts[m],
        })

    df_model_comparison = pd.DataFrame(summary_rows)
    comp_csv_path = os.path.join(results_dir, "phase6_model_comparison.csv")
    df_model_comparison.to_csv(comp_csv_path, index=False)
    print(f"Saved model comparison to: {comp_csv_path}")

    # Save routing diagnostics CSV
    df_routing = pd.DataFrame(routing_stats_list)
    routing_csv_path = os.path.join(results_dir, "phase6_routing_diagnostics.csv")
    df_routing.to_csv(routing_csv_path, index=False)
    print(f"Saved routing diagnostics to: {routing_csv_path}")

    # =========================================================================
    # STATISTICAL HYPOTHESIS TESTING (K=53 DAILY BLOCKS)
    # =========================================================================
    print("\n[4/5] Conducting Statistical Significance Testing on K=53 Daily Blocks...")
    mean_preds = {}
    for m in models:
        all_seed_preds = np.stack([test_predictions[m][s] for s in canonical_seeds], axis=0)
        mean_preds[m] = np.mean(all_seed_preds, axis=0)  # [1294, 24]

    K = 53
    daily_origins = [k * 24 for k in range(K)]  # 0, 24, ..., 1248

    daily_errors = {}
    for m in models:
        daily_errors[m] = np.array([
            np.mean(np.abs(mean_preds[m][orig] - y_test_mw[orig]))
            for orig in daily_origins
        ])

    comparisons = [
        ("C1", "Original_CAEGNet_V1", "Standalone_LSTM", "CAEG V1 vs Standalone LSTM"),
        ("C2", "Original_CAEGNet_V1", "Standalone_TCN", "CAEG V1 vs Standalone TCN"),
        ("C3", "Original_CAEGNet_V1", "Standalone_CNN", "CAEG V1 vs Standalone CNN"),
        ("C4", "Original_CAEGNet_V1", "Static_Equal_Ensemble", "CAEG V1 vs Static Equal Ensemble"),
        ("C5", "Bounded_CAEGNet_rho0.50", "Original_CAEGNet_V1", "Bounded CAEG vs CAEG V1"),
        ("C6", "Bounded_CAEGNet_rho0.50", "Static_Equal_Ensemble", "Bounded CAEG vs Static Equal Ensemble"),
    ]

    stat_results = []
    raw_p_values = []

    for comp_id, m1, m2, label in comparisons:
        err1 = daily_errors[m1]
        err2 = daily_errors[m2]
        d = err1 - err2  # negative means m1 is better (lower error)
        d_mean = float(np.mean(d))
        d_std = float(np.std(d, ddof=1))
        se = d_std / np.sqrt(K)

        # Paired t-test
        t_stat, p_val_t = stats.ttest_rel(err1, err2)
        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_w = stats.wilcoxon(err1, err2)
        except Exception:
            w_stat, p_val_w = np.nan, np.nan

        # DM_HLN at h=1 on K=53 daily blocks is mathematically identical to t_stat
        dm_hln = t_stat
        p_val_dm = p_val_t

        # 95% Confidence Interval for mean difference
        ci_half = stats.t.ppf(0.975, df=K-1) * se
        ci_lower = d_mean - ci_half
        ci_upper = d_mean + ci_half

        # Cohen's d
        cohen_d = d_mean / (d_std + 1e-12)

        stat_results.append({
            "comparison_id": comp_id,
            "model_1": m1,
            "model_2": m2,
            "description": label,
            "mean_paired_diff_mw": d_mean,
            "ci_95_lower_mw": ci_lower,
            "ci_95_upper_mw": ci_upper,
            "cohens_d": cohen_d,
            "t_statistic": float(t_stat),
            "p_value_t": float(p_val_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_wilcoxon": float(p_val_w),
            "dm_hln_stat": float(dm_hln),
            "p_value_dm_hln": float(p_val_dm),
        })
        raw_p_values.append(float(p_val_t))

    # Apply Holm-Bonferroni correction
    adjusted_p_values = compute_holmd_bonferroni(raw_p_values)
    for res, adj_p in zip(stat_results, adjusted_p_values):
        res["p_value_holm_bonferroni"] = adj_p
        res["statistically_significant"] = bool(adj_p < 0.05)

    df_stats = pd.DataFrame(stat_results)
    stats_csv_path = os.path.join(results_dir, "phase6_statistical_tests.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Saved statistical test results to: {stats_csv_path}")

    # =========================================================================
    # PUBLICATION PLOTS
    # =========================================================================
    print("\n[5/5] Generating Publication Plots...")

    # Plot 1: Mean +/- SD MAE Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    model_labels = [
        "CAEG-Net V1\n(Primary Proposed)",
        "Bounded CAEG\n(rho=0.50)",
        "Static Equal\nEnsemble",
        "Standalone\nTCN",
        "Standalone\nLSTM",
        "Standalone\nCNN",
    ]
    mae_means = [df_model_comparison[df_model_comparison["model"] == m]["mae_mean"].values[0] for m in models]
    mae_stds = [df_model_comparison[df_model_comparison["model"] == m]["mae_std"].values[0] for m in models]
    colors = ["#2b5c8f", "#418ab3", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    bars = plt.bar(range(len(models)), mae_means, yerr=mae_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    for bar, mean_val, std_val in zip(bars, mae_means, mae_stds):
        plt.text(bar.get_x() + bar.get_width() / 2.0, mean_val + std_val + 5.0, f"{mean_val:.1f} +/- {std_val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.xticks(range(len(models)), model_labels, fontsize=9.5)
    plt.ylabel("Test MAE (MW)", fontsize=11, fontweight="bold")
    plt.title("CAEG-Net Phase 6: Five-Seed Finalist Performance (Mean +/- SD MAE)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0, max(mae_means) + max(mae_stds) + 60)
    plt.tight_layout()
    p1_path = os.path.join(plots_dir, "phase6_mae_comparison.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"Saved: {p1_path}")

    # Plot 2: Per-Seed MAE Comparison
    plt.figure(figsize=(10, 5.5), dpi=300)
    seed_str = [str(s) for s in canonical_seeds]
    x_indices = np.arange(len(canonical_seeds))
    width = 0.13

    for idx, (m, c, l) in enumerate(zip(models, colors, ["CAEG V1", "Bounded CAEG", "Equal Ens", "TCN", "LSTM", "CNN"])):
        vals = [df_seed_results[(df_seed_results["model"] == m) & (df_seed_results["seed"] == s)]["MAE"].values[0] for s in canonical_seeds]
        plt.bar(x_indices + (idx - 2.5) * width, vals, width, label=l, color=c, alpha=0.85, edgecolor="black", linewidth=0.8)

    plt.xticks(x_indices, [f"Seed {s}" for s in canonical_seeds], fontsize=10, fontweight="bold")
    plt.ylabel("Test MAE (MW)", fontsize=11, fontweight="bold")
    plt.title("CAEG-Net Phase 6: Individual Seed MAE across Finalists", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="gray", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    p2_path = os.path.join(plots_dir, "phase6_per_seed_mae.png")
    plt.savefig(p2_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2_path}")

    # Plot 3: Daily Error Distribution
    plt.figure(figsize=(10, 5.5), dpi=300)
    data_to_plot = [daily_errors[m] for m in models]
    box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=["CAEG V1", "Bounded CAEG", "Equal Ens", "TCN", "LSTM", "CNN"], medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    plt.ylabel("Daily Block MAE (MW)", fontsize=11, fontweight="bold")
    plt.title("CAEG-Net Phase 6: Non-Overlapping Daily Error Distribution (K=53 Days)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3_path = os.path.join(plots_dir, "phase6_daily_error_distribution.png")
    plt.savefig(p3_path)
    plt.close()
    print(f"Saved: {p3_path}")

    print("\n============================================================")
    print("PHASE 6 FIVE-SEED EVALUATION EXECUTION COMPLETE")
    print("============================================================")

    # Clean exit to avoid Windows PyTorch CUDA cleanup hang/exception
    os._exit(0)


if __name__ == "__main__":
    main()
