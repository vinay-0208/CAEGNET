"""
CAEG-Net Research Track — Phase 5: Multi-Seed Scientific Validation Suite
=========================================================================
Executes canonical 5-seed benchmark across independent seeds [42, 43, 44, 45, 46].

Evaluates:
1. Naive-24 (Deterministic baseline)
2. Standalone GRU Expert
3. Standalone TCN Expert
4. Standalone PatchTemporalExpert
5. Static Equal-Weight Ensemble (GRU + TCN + Patch)
6. Standard Input-MoE (Raw 168h lookback gating)
7. CAEG-Net V2 Full (9 features: 5 observable + 1 Ridge-error + 3 disagreement)
8. CAEG-Net V2 No Recent Error (8 features: 5 observable + 0 Ridge-error + 3 disagreement)
9. Canonical Frozen V1 Reference Comparison

Performs:
- Origin-level full metric evaluation (MAE, MSE, RMSE, R2, MAPE)
- Non-overlapping 24h daily block analysis (53 independent 24h episodes)
- Paired t-tests, Wilcoxon signed-rank tests, Cohen's d, Holm-Bonferroni correction
- Routing reproducibility and entropy metrics across seeds
- Expert complementarity & Oracle headroom analysis
- Difficulty-regime analysis (Volatility, Disagreement, Baseline Error)
- Controlled auxiliary loss ablation (lambda=0.15 vs lambda=0)

Saves all tabular artifacts to research/results/
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.models import (
    GatedRecurrentExpert,
    MultiScaleCausalTCNExpert,
    PatchTemporalExpert,
    StandardInputMoE,
    CAEGNetV2,
    count_parameters,
    compute_caeg_v2_loss,
)
from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.training import train_caeg_v2, evaluate_caeg_v2


# =====================================================================
# Standalone Expert Training & Evaluation Utilities
# =====================================================================

def train_single_expert(
    expert: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 35,
    patience: int = 7,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, int, nn.Module]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert = expert.to(device)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        expert.train()
        train_loss = 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            y_pred = expert(x)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        expert.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = expert(x)
                loss = F.mse_loss(y_pred, y)
                val_loss += loss.item()
                n_val += 1

        val_mse = val_loss / max(n_val, 1)
        scheduler.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in expert.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    expert.load_state_dict(best_weights)
    total_time = time.time() - start_time

    return {
        "model": expert,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
    }


def evaluate_single_expert(
    expert: nn.Module,
    data_loader: DataLoader,
    scaler: object,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert = expert.to(device)
    expert.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device, non_blocking=True)
            y_pred = expert(x)
            all_preds.append(y_pred.cpu())
            all_trues.append(y)

    preds_sc = torch.cat(all_preds, dim=0).numpy()
    trues_sc = torch.cat(all_trues, dim=0).numpy()

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    preds_mw = preds_sc * scale + mean
    trues_mw = trues_sc * scale + mean

    err = preds_mw - trues_mw
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues_mw - np.mean(trues_mw)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(err / trues_mw)) * 100.0)

    metrics = {
        "mae_mw": mae,
        "mse_mw2": mse,
        "rmse_mw": rmse,
        "r2": r2,
        "mape_pct": mape,
    }
    return metrics, preds_mw


def compute_metrics(preds_mw: np.ndarray, trues_mw: np.ndarray) -> Dict[str, float]:
    err = preds_mw - trues_mw
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues_mw - np.mean(trues_mw)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(err / trues_mw)) * 100.0)
    return {
        "mae_mw": mae,
        "mse_mw2": mse,
        "rmse_mw": rmse,
        "r2": r2,
        "mape_pct": mape,
    }


# =====================================================================
# Main Phase 5 Benchmark Runner
# =====================================================================

def run_phase5_benchmark(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    seeds: List[int] = [42, 43, 44, 45, 46],
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    results_dir: str = "research/results",
):
    device = torch.device(device_str)
    os.makedirs(results_dir, exist_ok=True)
    print(f"=== Starting CAEG-Net Phase 5 Multi-Seed Scientific Validation on {device} ===")
    print(f"Canonical Seeds: {seeds}")

    pipe = prepare_research_v2_pipeline(data_path=data_path)
    scaler = pipe["scaler"]

    # Storage for all evaluation tables
    seed_records = []
    block_records = []
    routing_records = []
    complementarity_records = []
    regime_records = []

    # Get true test targets once
    test_loader_sample = build_research_v2_dataloaders(pipe, batch_size=64, seed=42)["test"]
    all_y_true = []
    all_x_hist = []
    for x, y, _ in test_loader_sample:
        all_y_true.append(y)
        all_x_hist.append(x)
    y_true_sc = torch.cat(all_y_true, dim=0).numpy()
    x_hist_sc = torch.cat(all_x_hist, dim=0).numpy()

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    y_true_mw = y_true_sc * scale + mean
    x_hist_mw = x_hist_sc * scale + mean

    num_test_origins = y_true_mw.shape[0]  # 1294
    num_blocks = num_test_origins // 24    # 53 non-overlapping daily blocks
    block_indices = [i * 24 for i in range(num_blocks)]

    # Compute Naive-24 (Deterministic, identical for all seeds)
    naive_preds_mw = x_hist_mw[:, -24:, 0]
    naive_metrics = compute_metrics(naive_preds_mw, y_true_mw)
    naive_block_maes = [float(np.mean(np.abs(naive_preds_mw[idx] - y_true_mw[idx]))) for idx in block_indices]

    print(f"Evaluated Naive-24 Baseline: MAE = {naive_metrics['mae_mw']:.2f} MW, R2 = {naive_metrics['r2']:.4f}")

    # Record Naive-24 in seed_records
    for s in seeds:
        seed_records.append({
            "model": "Naive-24",
            "seed": s,
            "best_epoch": 0,
            "actual_epochs": 0,
            "best_val_loss": 0.0,
            "test_mae_mw": naive_metrics["mae_mw"],
            "test_mse_mw2": naive_metrics["mse_mw2"],
            "test_rmse_mw": naive_metrics["rmse_mw"],
            "test_r2": naive_metrics["r2"],
            "test_mape_pct": naive_metrics["mape_pct"],
            "training_time_s": 0.0,
            "param_count": 0,
        })

    # -------------------------------------------------------------
    # Loop across independent seeds
    # -------------------------------------------------------------
    for seed in seeds:
        print("\n=======================================================")
        print(f">>> RUNNING CANONICAL BENCHMARK FOR SEED {seed} <<<")
        print(f"=======================================================")

        # Build dataloaders for this seed
        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)

        # Dictionary to store predictions for this seed
        seed_preds = {
            "Naive-24": naive_preds_mw,
        }

        # 1. Standalone GRU
        print("\n--- [Seed {seed}] Training Standalone GRU ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24, dropout=0.1).to(device)
        gru_res = train_single_expert(gru, dataloaders["train"], dataloaders["val"], device=device)
        gru_met, gru_pred_mw = evaluate_single_expert(gru_res["model"], dataloaders["test"], scaler, device=device)
        seed_preds["Standalone_GRU"] = gru_pred_mw
        seed_records.append({
            "model": "Standalone_GRU",
            "seed": seed,
            "best_epoch": gru_res["best_epoch"],
            "actual_epochs": gru_res["total_epochs"],
            "best_val_loss": gru_res["best_val_loss"],
            "test_mae_mw": gru_met["mae_mw"],
            "test_mse_mw2": gru_met["mse_mw2"],
            "test_rmse_mw": gru_met["rmse_mw"],
            "test_r2": gru_met["r2"],
            "test_mape_pct": gru_met["mape_pct"],
            "training_time_s": gru_res["training_time"],
            "param_count": sum(p.numel() for p in gru.parameters()),
        })
        print(f"GRU MAE: {gru_met['mae_mw']:.2f} MW, R2: {gru_met['r2']:.4f}, Runtime: {gru_res['training_time']:.1f}s")

        # 2. Standalone TCN
        print(f"\n--- [Seed {seed}] Training Standalone TCN ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24, dropout=0.1).to(device)
        tcn_res = train_single_expert(tcn, dataloaders["train"], dataloaders["val"], device=device)
        tcn_met, tcn_pred_mw = evaluate_single_expert(tcn_res["model"], dataloaders["test"], scaler, device=device)
        seed_preds["Standalone_TCN"] = tcn_pred_mw
        seed_records.append({
            "model": "Standalone_TCN",
            "seed": seed,
            "best_epoch": tcn_res["best_epoch"],
            "actual_epochs": tcn_res["total_epochs"],
            "best_val_loss": tcn_res["best_val_loss"],
            "test_mae_mw": tcn_met["mae_mw"],
            "test_mse_mw2": tcn_met["mse_mw2"],
            "test_rmse_mw": tcn_met["rmse_mw"],
            "test_r2": tcn_met["r2"],
            "test_mape_pct": tcn_met["mape_pct"],
            "training_time_s": tcn_res["training_time"],
            "param_count": sum(p.numel() for p in tcn.parameters()),
        })
        print(f"TCN MAE: {tcn_met['mae_mw']:.2f} MW, R2: {tcn_met['r2']:.4f}, Runtime: {tcn_res['training_time']:.1f}s")

        # 3. Standalone PatchTemporalExpert
        print(f"\n--- [Seed {seed}] Training Standalone PatchTemporalExpert ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24, dropout=0.1).to(device)
        patch_res = train_single_expert(patch, dataloaders["train"], dataloaders["val"], device=device)
        patch_met, patch_pred_mw = evaluate_single_expert(patch_res["model"], dataloaders["test"], scaler, device=device)
        seed_preds["Standalone_Patch"] = patch_pred_mw
        seed_records.append({
            "model": "Standalone_Patch",
            "seed": seed,
            "best_epoch": patch_res["best_epoch"],
            "actual_epochs": patch_res["total_epochs"],
            "best_val_loss": patch_res["best_val_loss"],
            "test_mae_mw": patch_met["mae_mw"],
            "test_mse_mw2": patch_met["mse_mw2"],
            "test_rmse_mw": patch_met["rmse_mw"],
            "test_r2": patch_met["r2"],
            "test_mape_pct": patch_met["mape_pct"],
            "training_time_s": patch_res["training_time"],
            "param_count": sum(p.numel() for p in patch.parameters()),
        })
        print(f"Patch MAE: {patch_met['mae_mw']:.2f} MW, R2: {patch_met['r2']:.4f}, Runtime: {patch_res['training_time']:.1f}s")

        # 4. Static Equal-Weight Ensemble (Average of the 3 standalone experts)
        equal_ens_mw = (gru_pred_mw + tcn_pred_mw + patch_pred_mw) / 3.0
        ens_met = compute_metrics(equal_ens_mw, y_true_mw)
        seed_preds["Static_Equal_Ensemble"] = equal_ens_mw
        seed_records.append({
            "model": "Static_Equal_Ensemble",
            "seed": seed,
            "best_epoch": 0,
            "actual_epochs": 0,
            "best_val_loss": 0.0,
            "test_mae_mw": ens_met["mae_mw"],
            "test_mse_mw2": ens_met["mse_mw2"],
            "test_rmse_mw": ens_met["rmse_mw"],
            "test_r2": ens_met["r2"],
            "test_mape_pct": ens_met["mape_pct"],
            "training_time_s": 0.0,
            "param_count": 0,
        })
        print(f"Equal Ensemble MAE: {ens_met['mae_mw']:.2f} MW, R2: {ens_met['r2']:.4f}")

        # 5. Standard Input-MoE (Raw 168h Lookback Gating)
        print(f"\n--- [Seed {seed}] Training Standard Input-MoE ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        input_moe = StandardInputMoE(temperature=0.5).to(device)
        imoe_res = train_caeg_v2(
            model=input_moe,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.15,
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            device=device,
            verbose=False,
        )
        imoe_eval = evaluate_caeg_v2(input_moe, dataloaders["test"], scaler, temperature=0.5, device=device)
        imoe_fused_met = imoe_eval["fused_metrics"]
        imoe_pred_mw = imoe_eval["y_pred_mw"]
        seed_preds["Standard_Input_MoE"] = imoe_pred_mw
        seed_records.append({
            "model": "Standard_Input_MoE",
            "seed": seed,
            "best_epoch": imoe_res["best_epoch"],
            "actual_epochs": imoe_res["total_epochs"],
            "best_val_loss": imoe_res["best_val_fused"],
            "test_mae_mw": imoe_fused_met["mae_mw"],
            "test_mse_mw2": imoe_fused_met["rmse_mw"] ** 2,
            "test_rmse_mw": imoe_fused_met["rmse_mw"],
            "test_r2": imoe_fused_met["r2"],
            "test_mape_pct": imoe_fused_met["mape_pct"],
            "training_time_s": imoe_res["training_time_seconds"],
            "param_count": count_parameters(input_moe)["total"],
        })
        print(f"Standard Input-MoE MAE: {imoe_fused_met['mae_mw']:.2f} MW, R2: {imoe_fused_met['r2']:.4f}")

        # 6. CAEG-Net V2 Full (Nominal Proposed Model)
        print(f"\n--- [Seed {seed}] Training CAEG-Net V2 (Full Context, tau=0.5) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        v2_full = CAEGNetV2(forecast_aware=True, temperature=0.5).to(device)
        v2_res = train_caeg_v2(
            model=v2_full,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.15,
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            device=device,
            verbose=False,
        )
        v2_eval = evaluate_caeg_v2(v2_full, dataloaders["test"], scaler, temperature=0.5, device=device)
        v2_fused_met = v2_eval["fused_metrics"]
        v2_pred_mw = v2_eval["y_pred_mw"]
        seed_preds["CAEG_Net_V2_Full"] = v2_pred_mw
        seed_records.append({
            "model": "CAEG_Net_V2_Full",
            "seed": seed,
            "best_epoch": v2_res["best_epoch"],
            "actual_epochs": v2_res["total_epochs"],
            "best_val_loss": v2_res["best_val_fused"],
            "test_mae_mw": v2_fused_met["mae_mw"],
            "test_mse_mw2": v2_fused_met["rmse_mw"] ** 2,
            "test_rmse_mw": v2_fused_met["rmse_mw"],
            "test_r2": v2_fused_met["r2"],
            "test_mape_pct": v2_fused_met["mape_pct"],
            "training_time_s": v2_res["training_time_seconds"],
            "param_count": count_parameters(v2_full)["total"],
        })
        print(f"CAEG-Net V2 Full MAE: {v2_fused_met['mae_mw']:.2f} MW, R2: {v2_fused_met['r2']:.4f}")

        # 7. CAEG-Net V2 Without Recent Error Context (Ablation)
        print(f"\n--- [Seed {seed}] Training CAEG-Net V2 (No Recent Error Feedback) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        mask_no_rec = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        v2_no_rec = CAEGNetV2(forecast_aware=True, temperature=0.5).to(device)
        no_rec_res = train_caeg_v2(
            model=v2_no_rec,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.15,
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            feature_mask=mask_no_rec,
            device=device,
            verbose=False,
        )
        no_rec_eval = evaluate_caeg_v2(v2_no_rec, dataloaders["test"], scaler, temperature=0.5, feature_mask=mask_no_rec, device=device)
        no_rec_fused_met = no_rec_eval["fused_metrics"]
        no_rec_pred_mw = no_rec_eval["y_pred_mw"]
        seed_preds["CAEG_Net_V2_NoRecentError"] = no_rec_pred_mw
        seed_records.append({
            "model": "CAEG_Net_V2_NoRecentError",
            "seed": seed,
            "best_epoch": no_rec_res["best_epoch"],
            "actual_epochs": no_rec_res["total_epochs"],
            "best_val_loss": no_rec_res["best_val_fused"],
            "test_mae_mw": no_rec_fused_met["mae_mw"],
            "test_mse_mw2": no_rec_fused_met["rmse_mw"] ** 2,
            "test_rmse_mw": no_rec_fused_met["rmse_mw"],
            "test_r2": no_rec_fused_met["r2"],
            "test_mape_pct": no_rec_fused_met["mape_pct"],
            "training_time_s": no_rec_res["training_time_seconds"],
            "param_count": count_parameters(v2_no_rec)["total"],
        })
        print(f"CAEG-Net V2 No-Recent-Error MAE: {no_rec_fused_met['mae_mw']:.2f} MW, R2: {no_rec_fused_met['r2']:.4f}")

        # -------------------------------------------------------------
        # Collect Routing Statistics for this seed
        # -------------------------------------------------------------
        v2_weights = v2_eval["weight_stats"]
        w_np = v2_eval["weights"]  # [N, 3]
        top_expert_idx = np.argmax(w_np, axis=-1)
        top_expert_counts = np.bincount(top_expert_idx, minlength=3)
        top_expert_shares = top_expert_counts / len(top_expert_idx)

        routing_records.append({
            "seed": seed,
            "mean_w_GRU": v2_weights["mean_weights"][0],
            "mean_w_TCN": v2_weights["mean_weights"][1],
            "mean_w_Patch": v2_weights["mean_weights"][2],
            "std_w_GRU": v2_weights["std_weights"][0],
            "std_w_TCN": v2_weights["std_weights"][1],
            "std_w_Patch": v2_weights["std_weights"][2],
            "mean_entropy_H": v2_weights["mean_entropy"],
            "mean_n_eff": v2_weights["mean_n_eff"],
            "top_choice_Patch_share": top_expert_shares[2],
            "top_choice_TCN_share": top_expert_shares[1],
            "top_choice_GRU_share": top_expert_shares[0],
        })

        # -------------------------------------------------------------
        # Collect Expert Complementarity & Oracle Headroom
        # -------------------------------------------------------------
        # Compute per-origin MAEs for standalone experts
        origin_gru_mae = np.mean(np.abs(gru_pred_mw - y_true_mw), axis=-1)
        origin_tcn_mae = np.mean(np.abs(tcn_pred_mw - y_true_mw), axis=-1)
        origin_patch_mae = np.mean(np.abs(patch_pred_mw - y_true_mw), axis=-1)
        origin_fused_mae = np.mean(np.abs(v2_pred_mw - y_true_mw), axis=-1)

        # Oracle chooses best standalone expert per origin
        origin_expert_stack = np.stack([origin_gru_mae, origin_tcn_mae, origin_patch_mae], axis=-1)
        oracle_best_idx = np.argmin(origin_expert_stack, axis=-1)
        oracle_min_mae = np.min(origin_expert_stack, axis=-1)
        oracle_mae_mw = float(np.mean(oracle_min_mae))

        oracle_wins = np.bincount(oracle_best_idx, minlength=3)
        oracle_win_share = oracle_wins / len(oracle_best_idx)

        # Best standalone expert across the partition
        standalone_best_mae = min(gru_met["mae_mw"], tcn_met["mae_mw"], patch_met["mae_mw"])
        oracle_headroom = standalone_best_mae - oracle_mae_mw
        fused_gain = standalone_best_mae - v2_fused_met["mae_mw"]
        captured_headroom = (fused_gain / oracle_headroom) if oracle_headroom > 0 else 0.0

        complementarity_records.append({
            "seed": seed,
            "standalone_best_mae_mw": standalone_best_mae,
            "oracle_mae_mw": oracle_mae_mw,
            "fused_v2_mae_mw": v2_fused_met["mae_mw"],
            "oracle_headroom_mw": oracle_headroom,
            "captured_headroom_pct": captured_headroom * 100.0,
            "oracle_win_share_Patch": oracle_win_share[2],
            "oracle_win_share_TCN": oracle_win_share[1],
            "oracle_win_share_GRU": oracle_win_share[0],
            "router_top_matches_oracle_pct": float(np.mean(top_expert_idx == oracle_best_idx)) * 100.0,
        })

        # -------------------------------------------------------------
        # Collect Difficulty Regime Analysis for this seed
        # -------------------------------------------------------------
        # Extract disagreement and context features
        disagree_vec = v2_eval["disagreement"]  # [N, 3] -> column 0 is pairwise MAE
        disagree_scalar = disagree_vec[:, 0] if disagree_vec is not None else np.zeros(num_test_origins)
        context_mat = v2_eval.get("context_6d", v2_eval.get("context"))  # [N, 6] -> col 1 is Volatility, col 5 is Recent Error
        volatility_scalar = context_mat[:, 1]
        baseline_err_scalar = context_mat[:, 5]

        # Predefined tertiles across all test windows
        for reg_name, scalar_vals in [
            ("Disagreement", disagree_scalar),
            ("Volatility", volatility_scalar),
            ("Baseline_Error", baseline_err_scalar),
        ]:
            t1, t2 = np.percentile(scalar_vals, [33.33, 66.67])
            for lvl_name, mask in [
                ("Low", scalar_vals <= t1),
                ("Medium", (scalar_vals > t1) & (scalar_vals <= t2)),
                ("High", scalar_vals > t2),
            ]:
                sub_fused = np.mean(origin_fused_mae[mask])
                sub_ens = np.mean(np.mean(np.abs(equal_ens_mw[mask] - y_true_mw[mask]), axis=-1))
                sub_patch = np.mean(origin_patch_mae[mask])
                regime_records.append({
                    "seed": seed,
                    "regime_type": reg_name,
                    "regime_level": lvl_name,
                    "sample_count": int(np.sum(mask)),
                    "v2_mae_mw": float(sub_fused),
                    "equal_ens_mae_mw": float(sub_ens),
                    "patch_mae_mw": float(sub_patch),
                    "gain_vs_equal_mw": float(sub_ens - sub_fused),
                    "gain_vs_patch_mw": float(sub_patch - sub_fused),
                })

        # -------------------------------------------------------------
        # Non-Overlapping Daily Block MAEs (53 blocks of 24h)
        # -------------------------------------------------------------
        for model_name, p_mw in seed_preds.items():
            for b_idx, start_i in enumerate(block_indices):
                b_err = np.mean(np.abs(p_mw[start_i] - y_true_mw[start_i]))
                block_records.append({
                    "seed": seed,
                    "model": model_name,
                    "block_id": b_idx,
                    "block_mae_mw": float(b_err),
                })

    # =====================================================================
    # Process Results into Final DataFrames & Artifacts
    # =====================================================================
    df_seeds = pd.DataFrame(seed_records)
    df_blocks = pd.DataFrame(block_records)
    df_routing = pd.DataFrame(routing_records)
    df_compl = pd.DataFrame(complementarity_records)
    df_regime = pd.DataFrame(regime_records)

    # 1. Save Seed Results CSV
    csv_seed_path = os.path.join(results_dir, "phase5_seed_results.csv")
    df_seeds.to_csv(csv_seed_path, index=False)
    print("\nSaved seed results to {csv_seed_path}")

    # 2. Model Comparison Summary Table (Across 5 Seeds)
    models_order = [
        "Naive-24",
        "Standalone_GRU",
        "Standalone_TCN",
        "Standalone_Patch",
        "Static_Equal_Ensemble",
        "Standard_Input_MoE",
        "CAEG_Net_V2_Full",
        "CAEG_Net_V2_NoRecentError",
    ]

    comp_rows = []
    for m in models_order:
        sub = df_seeds[df_seeds["model"] == m]
        maes = sub["test_mae_mw"].values
        rmses = sub["test_rmse_mw"].values
        r2s = sub["test_r2"].values
        mapes = sub["test_mape_pct"].values
        mses = sub["test_mse_mw2"].values
        params = sub["param_count"].iloc[0]

        row = {
            "model": m,
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "mae_median": float(np.median(maes)),
            "mae_min": float(np.min(maes)),
            "mae_max": float(np.max(maes)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
            "mape_mean": float(np.mean(mapes)),
            "mape_std": float(np.std(mapes, ddof=1)) if len(mapes) > 1 else 0.0,
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
            "param_count": params,
        }
        comp_rows.append(row)

    # Append Frozen V1 Reference Row
    comp_rows.append({
        "model": "CAEG_Net_V1_Canonical_Reference",
        "mae_mean": 251.44,
        "mae_std": 9.74,
        "mae_median": 247.30,
        "mae_min": 244.00,
        "mae_max": 268.31,
        "rmse_mean": 334.32,
        "rmse_std": 11.09,
        "r2_mean": 0.8723,
        "r2_std": 0.0086,
        "mape_mean": 4.71,
        "mape_std": 0.21,
        "mse_mean": 111865.65,
        "mse_std": 7430.22,
        "param_count": 121531,
    })

    df_comp = pd.DataFrame(comp_rows)
    csv_comp_path = os.path.join(results_dir, "phase5_model_comparison.csv")
    df_comp.to_csv(csv_comp_path, index=False)
    print(f"Saved model comparison to {csv_comp_path}")

    # 3. Recent Error Ablation Table
    rec_abl = []
    for s in seeds:
        mae_full = df_seeds[(df_seeds["model"] == "CAEG_Net_V2_Full") & (df_seeds["seed"] == s)]["test_mae_mw"].iloc[0]
        mae_norec = df_seeds[(df_seeds["model"] == "CAEG_Net_V2_NoRecentError") & (df_seeds["seed"] == s)]["test_mae_mw"].iloc[0]
        rec_abl.append({
            "seed": s,
            "mae_full_v2_mw": mae_full,
            "mae_no_recent_error_mw": mae_norec,
            "delta_mw": mae_full - mae_norec,  # positive means NoRecentError is better
            "superior_model": "CAEG_Net_V2_NoRecentError" if mae_norec < mae_full else "CAEG_Net_V2_Full",
        })
    df_rec = pd.DataFrame(rec_abl)
    diffs = df_rec["delta_mw"].values
    t_stat_rec, p_val_rec = stats.ttest_1samp(diffs, 0.0)
    w_stat_rec, p_val_w_rec = stats.wilcoxon(diffs)
    csv_rec_path = os.path.join(results_dir, "phase5_recent_error_ablation.csv")
    df_rec.to_csv(csv_rec_path, index=False)
    print(f"Saved recent error ablation to {csv_rec_path}")

    # 4. Routing Reproducibility Table
    csv_routing_path = os.path.join(results_dir, "phase5_routing_reproducibility.csv")
    df_routing.to_csv(csv_routing_path, index=False)
    print(f"Saved routing reproducibility to {csv_routing_path}")

    # 5. Expert Complementarity Table
    csv_compl_path = os.path.join(results_dir, "phase5_expert_complementarity.csv")
    df_compl.to_csv(csv_compl_path, index=False)
    print(f"Saved expert complementarity to {csv_compl_path}")

    # 6. Regime Analysis Table
    reg_summary = df_regime.groupby(["regime_type", "regime_level"]).agg({
        "v2_mae_mw": ["mean", "std"],
        "equal_ens_mae_mw": ["mean", "std"],
        "patch_mae_mw": ["mean", "std"],
        "gain_vs_equal_mw": ["mean", "std"],
        "gain_vs_patch_mw": ["mean", "std"],
    }).reset_index()
    reg_summary.columns = [f"{c[0]}_{c[1]}".strip("_") for c in reg_summary.columns]
    csv_reg_path = os.path.join(results_dir, "phase5_regime_analysis.csv")
    reg_summary.to_csv(csv_reg_path, index=False)
    print(f"Saved regime analysis to {csv_reg_path}")

    # 7. Statistical Tests Table (Seed-Level & Non-Overlapping Daily Blocks)
    stat_records = []

    # Primary pairwise comparisons for V2 Full
    comparisons = [
        ("CAEG_Net_V2_Full", "Static_Equal_Ensemble"),
        ("CAEG_Net_V2_Full", "Standard_Input_MoE"),
        ("CAEG_Net_V2_Full", "Standalone_Patch"),
        ("CAEG_Net_V2_Full", "CAEG_Net_V2_NoRecentError"),
    ]

    # Block-level evaluation (averaged across seeds per block or pooled)
    block_pivot = df_blocks.groupby(["model", "block_id"])["block_mae_mw"].mean().unstack(level=0)

    raw_p_values_seed = []
    raw_p_values_block = []

    for m_target, m_base in comparisons:
        # Seed-level test (5 paired observations)
        s_target = df_seeds[df_seeds["model"] == m_target].sort_values("seed")["test_mae_mw"].values
        s_base = df_seeds[df_seeds["model"] == m_base].sort_values("seed")["test_mae_mw"].values
        s_diff = s_target - s_base  # negative means target has lower MAE (better)
        s_t, s_p = stats.ttest_rel(s_target, s_base)
        s_w, s_wp = stats.wilcoxon(s_diff)
        s_d = np.mean(s_diff) / (np.std(s_diff, ddof=1) + 1e-8)

        # Block-level test (53 non-overlapping daily blocks)
        b_target = block_pivot[m_target].values
        b_base = block_pivot[m_base].values
        b_diff = b_target - b_base
        b_t, b_p = stats.ttest_rel(b_target, b_base)
        b_w, b_wp = stats.wilcoxon(b_diff)
        b_d = np.mean(b_diff) / (np.std(b_diff, ddof=1) + 1e-8)
        ci_low, ci_high = stats.t.interval(0.95, df=len(b_diff)-1, loc=np.mean(b_diff), scale=stats.sem(b_diff))

        stat_records.append({
            "target_model": m_target,
            "baseline_model": m_base,
            "seed_mean_diff_mw": float(np.mean(s_diff)),
            "seed_std_diff_mw": float(np.std(s_diff, ddof=1)),
            "seed_cohen_d": float(s_d),
            "seed_paired_t_stat": float(s_t),
            "seed_paired_t_p": float(s_p),
            "seed_wilcoxon_p": float(s_wp),
            "seed_wins": int(np.sum(s_diff < 0)),
            "block_count": len(b_diff),
            "block_mean_diff_mw": float(np.mean(b_diff)),
            "block_std_diff_mw": float(np.std(b_diff, ddof=1)),
            "block_95ci_low": float(ci_low),
            "block_95ci_high": float(ci_high),
            "block_cohen_d": float(b_d),
            "block_paired_t_stat": float(b_t),
            "block_paired_t_p": float(b_p),
            "block_wilcoxon_p": float(b_wp),
            "block_wins": int(np.sum(b_diff < 0)),
        })
        raw_p_values_block.append(b_p)

    # Holm-Bonferroni correction on block p-values
    sorted_p_indices = np.argsort(raw_p_values_block)
    m_tests = len(raw_p_values_block)
    adj_p_values = [0.0] * m_tests
    for rank, idx in enumerate(sorted_p_indices):
        multiplier = m_tests - rank
        adj_p_values[idx] = min(raw_p_values_block[idx] * multiplier, 1.0)

    for i, r in enumerate(stat_records):
        r["block_holm_bonferroni_p"] = adj_p_values[i]
        r["statistically_significant_0_05"] = adj_p_values[i] < 0.05

    df_stat = pd.DataFrame(stat_records)
    csv_stat_path = os.path.join(results_dir, "phase5_statistical_tests.csv")
    df_stat.to_csv(csv_stat_path, index=False)
    print(f"Saved statistical tests to {csv_stat_path}")

    # 8. Controlled Auxiliary Loss Reproducibility Check (Seeds 42 and 43)
    print("\n--- Running Controlled Auxiliary Loss Reproducibility Check (lambda=0.0) ---")
    aux_records = []
    for aux_seed in [42, 43]:
        # Train with lambda=0.0
        torch.manual_seed(aux_seed)
        np.random.seed(aux_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(aux_seed)
        v2_no_aux = CAEGNetV2(forecast_aware=True, temperature=0.5).to(device)
        dl_aux = build_research_v2_dataloaders(pipe, batch_size=64, seed=aux_seed)
        train_caeg_v2(
            model=v2_no_aux,
            train_loader=dl_aux["train"],
            val_loader=dl_aux["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.0,
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            device=device,
            verbose=False,
        )
        eval_no_aux = evaluate_caeg_v2(v2_no_aux, dl_aux["test"], scaler, temperature=0.5, device=device)
        f_no_aux = eval_no_aux["fused_metrics"]
        p_no_aux = eval_no_aux["patch_metrics"]
        t_no_aux = eval_no_aux["tcn_metrics"]
        g_no_aux = eval_no_aux["gru_metrics"]

        # Retrieve matching lambda=0.15 metrics from df_seeds
        m_full_s = df_seeds[(df_seeds["model"] == "CAEG_Net_V2_Full") & (df_seeds["seed"] == aux_seed)].iloc[0]

        aux_records.append({
            "seed": aux_seed,
            "lambda_aux": 0.0,
            "fused_mae_mw": f_no_aux["mae_mw"],
            "patch_mae_mw": p_no_aux["mae_mw"],
            "tcn_mae_mw": t_no_aux["mae_mw"],
            "gru_mae_mw": g_no_aux["mae_mw"],
            "delta_vs_full_fused_mw": f_no_aux["mae_mw"] - m_full_s["test_mae_mw"],
        })
    df_aux = pd.DataFrame(aux_records)
    csv_aux_path = os.path.join(results_dir, "phase5_auxiliary_loss_check.csv")
    df_aux.to_csv(csv_aux_path, index=False)
    print(f"Saved auxiliary loss check to {csv_aux_path}")

    print("\n=== PHASE 5 MULTI-SEED BENCHMARK COMPLETE ===")
    print(df_comp[["model", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "param_count"]].to_string(index=False))

    return {
        "seeds": df_seeds,
        "comparison": df_comp,
        "recent_error": df_rec,
        "routing": df_routing,
        "complementarity": df_compl,
        "regime": reg_summary,
        "statistical_tests": df_stat,
    }


if __name__ == "__main__":
    run_phase5_benchmark()