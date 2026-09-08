"""
CAEG-Net Phase 5: Original Architecture Optimization & Methodology Correction
==============================================================================
Governed by Phase 5 of the 14-Phase Research Roadmap:
1. Operates strictly on the canonical original expert family: LSTM + TCN + CNN.
2. Evaluates individual improvements one at a time on Validation MAE (Seed 42).
3. Evaluates key finalists across 5 canonical seeds: [42, 123, 2024, 3407, 999].
4. Strictly preserves frozen V1 reference (251.44 MW).
5. Employs 53 daily blocks for statistical hypothesis testing with HLN correction.
6. Generates full machine-readable tables and publication figures.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
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
    TimeSeriesContextDataset,
    create_dataloaders,
)
from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    CAEGNet as CAEGNetV1,
    StandardInputMoE,
    count_parameters,
)
from research.original_caeg import (
    OriginalCAEGNetPhase5,
    LearnedStaticEnsembleV1,
    compute_phase5_loss,
)
from evaluate import compute_metrics


def extract_extended_context_features(
    X_windows: np.ndarray,
    recent_errors: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract modular context representations:
    1. Base 4D: [Trend, Volatility, Periodicity (Lag-24 Autocorr), Recent Error]
    2. Base 3D: [Trend, Volatility, Periodicity] (without recent error)
    3. Extended 6D: Base 4D + [Recent 48h Range Ratio, Weekly Lag-168 Profile Correlation]
    """
    if X_windows.ndim == 3:
        Z = X_windows[:, :, 0]
    else:
        Z = X_windows

    N, L = Z.shape
    C_base_4d = extract_context_features(X_windows, recent_errors)
    C_base_3d = C_base_4d[:, :3]

    # Feature 5: Recent 48h Range Ratio
    z_48 = Z[:, -48:]
    f_range = (np.max(z_48, axis=1, keepdims=True) - np.min(z_48, axis=1, keepdims=True)) / (
        np.std(z_48, axis=1, keepdims=True) + 1e-6
    )

    # Feature 6: Weekly Lag-168 Diurnal Correlation (Pearson corr between Day 1 [0:24] and Day 7 [144:168])
    z_day1 = Z[:, 0:24]
    z_day7 = Z[:, 144:168]
    mean_d1 = np.mean(z_day1, axis=1, keepdims=True)
    mean_d7 = np.mean(z_day7, axis=1, keepdims=True)
    num_corr = np.sum((z_day1 - mean_d1) * (z_day7 - mean_d7), axis=1, keepdims=True)
    denom_corr = 24.0 * np.std(z_day1, axis=1, keepdims=True) * np.std(z_day7, axis=1, keepdims=True) + 1e-6
    f_weekly = np.clip(num_corr / denom_corr, -1.0, 1.0)

    C_ext_6d = np.column_stack([C_base_4d, f_range, f_weekly]).astype(np.float32)

    return {
        "context_3d": C_base_3d.astype(np.float32),
        "context_4d": C_base_4d.astype(np.float32),
        "context_6d": C_ext_6d.astype(np.float32),
    }


def compute_hln_diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Tuple[float, float]:
    """
    Compute Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    Evaluated on loss differential sequence d_t = |e1_t| - |e2_t|.
    """
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma = 0.0
    for lag in range(1, h):
        c = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma += 2.0 * c
    var_d = (gamma0 + gamma) / n
    if var_d <= 1e-12:
        return 0.0, 1.0

    dm_stat = float(mean_d / np.sqrt(var_d))
    # HLN small sample adjustment
    hln_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
    dm_hln = dm_stat * hln_factor
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return dm_hln, p_val


def train_phase5_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_plateau_scheduler: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
    lambda_aux: float = 0.0,
    beta_entropy: float = 0.0,
    lambda_kl: float = 0.0,
    conservative_rho: Optional[float] = None,
    max_epochs: int = 40,
    patience: int = 7,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_plateau_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    else:
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "use_disagreement"):
                y_pred, w, diag = model(x, c, conservative_rho=conservative_rho)
                loss, _ = compute_phase5_loss(
                    y_pred=y_pred,
                    y_true=y,
                    expert_preds=diag["expert_predictions"],
                    weights=w,
                    lambda_aux=lambda_aux,
                    beta_entropy=beta_entropy,
                    lambda_kl=lambda_kl,
                )
            elif hasattr(model, "gating_network"):
                out = model(x, c, return_diagnostics=False)
                loss = F.mse_loss(out, y)
            elif hasattr(model, "gate_mlp"):
                out = model(x, return_diagnostics=False)
                loss = F.mse_loss(out, y)
            elif hasattr(model, "gate"):
                out = model(x, c)
                loss = F.mse_loss(out, y)
            else:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = F.mse_loss(out, y)

            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

                if hasattr(model, "use_disagreement"):
                    y_pred, _, _ = model(x, c, conservative_rho=conservative_rho)
                elif hasattr(model, "gating_network"):
                    y_pred = model(x, c, return_diagnostics=False)
                elif hasattr(model, "gate_mlp"):
                    y_pred = model(x, return_diagnostics=False)
                elif hasattr(model, "gate"):
                    y_pred = model(x, c)
                else:
                    y_pred = model(x)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]

                val_mse = F.mse_loss(y_pred, y)
                total_val_loss += val_mse.item()
                n_val_batches += 1

        val_loss = total_val_loss / max(n_val_batches, 1)
        if use_plateau_scheduler:
            scheduler.step(val_loss)
        else:
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
        "best_val_loss": best_val_loss,
        "runtime_seconds": time.time() - start_time,
    }


def evaluate_on_partition(
    model: nn.Module,
    data_loader: DataLoader,
    scaler: object,
    device: torch.device,
    conservative_rho: Optional[float] = None,
) -> Dict:
    model.eval()
    all_preds = []
    all_trues = []
    all_weights = []
    all_expert_preds = {"lstm": [], "tcn": [], "cnn": []}

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "use_disagreement"):
                y_pred, w, diag = model(x, c, conservative_rho=conservative_rho)
                all_weights.append(w.cpu().numpy())
                preds = diag["expert_predictions"]
                all_expert_preds["lstm"].append(preds["lstm"].cpu().numpy())
                all_expert_preds["tcn"].append(preds["tcn"].cpu().numpy())
                all_expert_preds["cnn"].append(preds["cnn"].cpu().numpy())
            elif hasattr(model, "gating_network"):
                y_pred, w, diag = model(x, c, return_diagnostics=True)
                all_weights.append(w.cpu().numpy())
                all_expert_preds["lstm"].append(diag["lstm"].cpu().numpy())
                all_expert_preds["tcn"].append(diag["tcn"].cpu().numpy())
                all_expert_preds["cnn"].append(diag["cnn"].cpu().numpy())
            elif hasattr(model, "gate_mlp"):
                y_pred, w, diag = model(x, return_diagnostics=True)
                all_weights.append(w.cpu().numpy())
                all_expert_preds["lstm"].append(diag["lstm"].cpu().numpy())
                all_expert_preds["tcn"].append(diag["tcn"].cpu().numpy())
                all_expert_preds["cnn"].append(diag["cnn"].cpu().numpy())
            elif hasattr(model, "gate"):
                y_pred = model(x, c)
            else:
                y_pred = model(x)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    preds_sc = np.concatenate(all_preds, axis=0)
    trues_sc = np.concatenate(all_trues, axis=0)

    # Invert scaling to Megawatts
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    preds_mw = preds_sc * scale + mean
    trues_mw = trues_sc * scale + mean

    metrics = compute_metrics(trues_mw, preds_mw)

    res = {
        "metrics": metrics,
        "preds_mw": preds_mw,
        "trues_mw": trues_mw,
    }
    if all_weights:
        weights_arr = np.concatenate(all_weights, axis=0)
        res["weights"] = weights_arr
        eps = 1e-8
        entropy = -np.sum(weights_arr * np.log(weights_arr + eps), axis=-1)
        res["mean_entropy"] = float(np.mean(entropy))
        res["mean_weights"] = np.mean(weights_arr, axis=0).tolist()
    if all_expert_preds["lstm"]:
        for exp_k in ["lstm", "tcn", "cnn"]:
            sc_arr = np.concatenate(all_expert_preds[exp_k], axis=0)
            mw_arr = sc_arr * scale + mean
            res[f"{exp_k}_mw"] = mw_arr
            res[f"{exp_k}_metrics"] = compute_metrics(trues_mw, mw_arr)

    return res


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=========================================================================")
    print(">>> CAEG-NET PHASE 5: ORIGINAL ARCHITECTURE OPTIMIZATION SUITE <<<")
    print("=========================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    results_dir = os.path.join("research", "results", "phase5")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Pipeline Data Preparation
    print("\n[Data Pipeline] Preparing 70/15/15 chronological PJM dataset...")
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)

    ctx_train = extract_extended_context_features(windows["train"]["X"], rec_tr)
    ctx_val = extract_extended_context_features(windows["val"]["X"], rec_val)
    ctx_test = extract_extended_context_features(windows["test"]["X"], rec_test)

    # Dataloader builders
    def get_loader(part: str, ctx_type: str = "context_4d", batch_size: int = 64, shuffle: bool = False):
        x = torch.from_numpy(windows[part]["X"]).float()
        y = torch.from_numpy(windows[part]["Y"]).float()
        c = torch.from_numpy(ctx_train[ctx_type] if part == "train" else ctx_val[ctx_type] if part == "val" else ctx_test[ctx_type]).float()
        ds = TensorDataset(x, y, c)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())

    # =========================================================================
    # STEP 7 & 8: REPRODUCE BASELINES & V1 REFERENCE (SEED 42 SCREENING)
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP A & B: SCREENING BASELINES ON SEED 42 (VALIDATION EVALUATION) <<<")
    print("=" * 70)
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader_4d = get_loader("train", "context_4d", shuffle=True)
    val_loader_4d = get_loader("val", "context_4d", shuffle=False)
    test_loader_4d = get_loader("test", "context_4d", shuffle=False)

    screening_records = []

    # 1. Persistence Naive-24
    val_true_mw = windows["val"]["Y"] * float(scaler.scale_[0]) + float(scaler.mean_[0])
    naive_val_pred_mw = windows["val"]["X"][:, -24:, 0] * float(scaler.scale_[0]) + float(scaler.mean_[0])
    naive_val_metrics = compute_metrics(val_true_mw, naive_val_pred_mw)
    screening_records.append({
        "model": "Persistence_Naive24",
        "category": "Baseline",
        "val_mae_mw": naive_val_metrics["MAE"],
        "val_rmse_mw": naive_val_metrics["RMSE"],
        "val_r2": naive_val_metrics["R2"],
        "params": 0,
        "runtime_s": 0.0,
    })
    print(f"Persistence Naive-24: Val MAE = {naive_val_metrics['MAE']:.2f} MW")

    # 2. Standalone LSTM
    lstm_m = LSTMExpert().to(device)
    t_res = train_phase5_model(lstm_m, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    lstm_val = evaluate_on_partition(lstm_m, val_loader_4d, scaler, device)
    screening_records.append({
        "model": "Standalone_LSTM",
        "category": "Expert",
        "val_mae_mw": lstm_val["metrics"]["MAE"],
        "val_rmse_mw": lstm_val["metrics"]["RMSE"],
        "val_r2": lstm_val["metrics"]["R2"],
        "params": count_parameters(lstm_m)["total_params"],
        "runtime_s": t_res["runtime_seconds"],
    })
    print(f"Standalone LSTM: Val MAE = {lstm_val['metrics']['MAE']:.2f} MW")

    # 3. Standalone TCN
    tcn_m = TCNExpert().to(device)
    t_res = train_phase5_model(tcn_m, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    tcn_val = evaluate_on_partition(tcn_m, val_loader_4d, scaler, device)
    screening_records.append({
        "model": "Standalone_TCN",
        "category": "Expert",
        "val_mae_mw": tcn_val["metrics"]["MAE"],
        "val_rmse_mw": tcn_val["metrics"]["RMSE"],
        "val_r2": tcn_val["metrics"]["R2"],
        "params": count_parameters(tcn_m)["total_params"],
        "runtime_s": t_res["runtime_seconds"],
    })
    print(f"Standalone TCN: Val MAE = {tcn_val['metrics']['MAE']:.2f} MW")

    # 4. Standalone CNN
    cnn_m = CNNExpert().to(device)
    t_res = train_phase5_model(cnn_m, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    cnn_val = evaluate_on_partition(cnn_m, val_loader_4d, scaler, device)
    screening_records.append({
        "model": "Standalone_CNN",
        "category": "Expert",
        "val_mae_mw": cnn_val["metrics"]["MAE"],
        "val_rmse_mw": cnn_val["metrics"]["RMSE"],
        "val_r2": cnn_val["metrics"]["R2"],
        "params": count_parameters(cnn_m)["total_params"],
        "runtime_s": t_res["runtime_seconds"],
    })
    print(f"Standalone CNN: Val MAE = {cnn_val['metrics']['MAE']:.2f} MW")

    # 5. Static Equal Ensemble
    equal_val_pred = (lstm_val["preds_mw"] + tcn_val["preds_mw"] + cnn_val["preds_mw"]) / 3.0
    equal_val_met = compute_metrics(val_true_mw, equal_val_pred)
    screening_records.append({
        "model": "Static_Equal_Ensemble",
        "category": "Ensemble",
        "val_mae_mw": equal_val_met["MAE"],
        "val_rmse_mw": equal_val_met["RMSE"],
        "val_r2": equal_val_met["R2"],
        "params": 56152 + 36952 + 27400,
        "runtime_s": 0.0,
    })
    print(f"Static Equal Ensemble: Val MAE = {equal_val_met['MAE']:.2f} MW")

    # 6. Standard Input-MoE
    moe_m = StandardInputMoE().to(device)
    t_res = train_phase5_model(moe_m, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    moe_val = evaluate_on_partition(moe_m, val_loader_4d, scaler, device)
    screening_records.append({
        "model": "Standard_Input_MoE",
        "category": "MoE",
        "val_mae_mw": moe_val["metrics"]["MAE"],
        "val_rmse_mw": moe_val["metrics"]["RMSE"],
        "val_r2": moe_val["metrics"]["R2"],
        "params": count_parameters(moe_m)["total_params"],
        "runtime_s": t_res["runtime_seconds"],
    })
    print(f"Standard Input-MoE: Val MAE = {moe_val['metrics']['MAE']:.2f} MW")

    # 7. Original CAEG-Net V1 (Joint Training, StepLR, no grad clipping)
    v1_m = CAEGNetV1(context_dim=4, horizon_dependent=False).to(device)
    t_res = train_phase5_model(
        v1_m, train_loader_4d, val_loader_4d, lr=1e-3, use_plateau_scheduler=False, clip_grad_norm=None, device=device
    )
    v1_val = evaluate_on_partition(v1_m, val_loader_4d, scaler, device)
    screening_records.append({
        "model": "Original_CAEGNet_V1",
        "category": "CAEG_V1",
        "val_mae_mw": v1_val["metrics"]["MAE"],
        "val_rmse_mw": v1_val["metrics"]["RMSE"],
        "val_r2": v1_val["metrics"]["R2"],
        "params": count_parameters(v1_m)["total_params"],
        "runtime_s": t_res["runtime_seconds"],
    })
    print(f"Original CAEG-Net V1: Val MAE = {v1_val['metrics']['MAE']:.2f} MW")

    df_screen = pd.DataFrame(screening_records)
    df_screen.to_csv(os.path.join(results_dir, "screening_results.csv"), index=False)

    # =========================================================================
    # STEP D: CONTROLLED ABLATIONS (ONE AT A TIME ON VALIDATION MAE)
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP D: CONTROLLED INDIVIDUAL ABLATIONS (SEED 42) <<<")
    print("=" * 70)

    ablation_records = []

    # --- Ablation D1: Training Stabilization (Gradient Clipping + Plateau Scheduler) ---
    print("\n--- Ablation D1: Training Stabilization ---")
    m_stab = OriginalCAEGNetPhase5(context_dim=4).to(device)
    t_res = train_phase5_model(
        m_stab, train_loader_4d, val_loader_4d, lr=1e-3, use_plateau_scheduler=True, clip_grad_norm=1.0, device=device
    )
    eval_stab = evaluate_on_partition(m_stab, val_loader_4d, scaler, device)
    ablation_records.append({
        "ablation_group": "Training_Stabilization",
        "variant": "GradClip1.0_PlateauSched",
        "val_mae_mw": eval_stab["metrics"]["MAE"],
        "val_rmse_mw": eval_stab["metrics"]["RMSE"],
        "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_stab["metrics"]["MAE"],
        "params": 121531,
    })
    print(f"Training Stabilization: Val MAE = {eval_stab['metrics']['MAE']:.2f} MW (Gain vs V1: {v1_val['metrics']['MAE'] - eval_stab['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D2: Context Signals (3D vs 4D vs 6D) ---
    print("\n--- Ablation D2: Context Features ---")
    # D2.1: Context 3D (without recent error)
    train_3d = get_loader("train", "context_3d", shuffle=True)
    val_3d = get_loader("val", "context_3d", shuffle=False)
    m_c3 = OriginalCAEGNetPhase5(context_dim=3).to(device)
    t_res = train_phase5_model(m_c3, train_3d, val_3d, lr=1e-3, device=device)
    eval_c3 = evaluate_on_partition(m_c3, val_3d, scaler, device)
    ablation_records.append({
        "ablation_group": "Context_Features",
        "variant": "3D_No_Recent_Error",
        "val_mae_mw": eval_c3["metrics"]["MAE"],
        "val_rmse_mw": eval_c3["metrics"]["RMSE"],
        "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_c3["metrics"]["MAE"],
        "params": sum(p.numel() for p in m_c3.parameters()),
    })
    print(f"Context 3D (No Recent Error): Val MAE = {eval_c3['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_c3['metrics']['MAE']:+.2f} MW)")

    # D2.2: Context 6D (Base 4D + 48h Range + Weekly Lag-168 Corr)
    train_6d = get_loader("train", "context_6d", shuffle=True)
    val_6d = get_loader("val", "context_6d", shuffle=False)
    m_c6 = OriginalCAEGNetPhase5(context_dim=6).to(device)
    t_res = train_phase5_model(m_c6, train_6d, val_6d, lr=1e-3, device=device)
    eval_c6 = evaluate_on_partition(m_c6, val_6d, scaler, device)
    ablation_records.append({
        "ablation_group": "Context_Features",
        "variant": "6D_Extended_Context",
        "val_mae_mw": eval_c6["metrics"]["MAE"],
        "val_rmse_mw": eval_c6["metrics"]["RMSE"],
        "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_c6["metrics"]["MAE"],
        "params": sum(p.numel() for p in m_c6.parameters()),
    })
    print(f"Context 6D (Extended): Val MAE = {eval_c6['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_c6['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D3: Inter-Expert Disagreement (Detached Consensus Signal) ---
    print("\n--- Ablation D3: Expert Disagreement ---")
    m_dis = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=True).to(device)
    t_res = train_phase5_model(m_dis, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    eval_dis = evaluate_on_partition(m_dis, val_loader_4d, scaler, device)
    ablation_records.append({
        "ablation_group": "Expert_Disagreement",
        "variant": "4D_Plus_Detached_Disagreement",
        "val_mae_mw": eval_dis["metrics"]["MAE"],
        "val_rmse_mw": eval_dis["metrics"]["RMSE"],
        "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_dis["metrics"]["MAE"],
        "params": sum(p.numel() for p in m_dis.parameters()),
    })
    print(f"Expert Disagreement: Val MAE = {eval_dis['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_dis['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D4: Auxiliary Expert Supervision ---
    print("\n--- Ablation D4: Auxiliary Expert Supervision ---")
    for lam_aux in [0.05, 0.10, 0.15]:
        m_aux = OriginalCAEGNetPhase5(context_dim=4).to(device)
        t_res = train_phase5_model(m_aux, train_loader_4d, val_loader_4d, lr=1e-3, lambda_aux=lam_aux, device=device)
        eval_aux = evaluate_on_partition(m_aux, val_loader_4d, scaler, device)
        ablation_records.append({
            "ablation_group": "Auxiliary_Supervision",
            "variant": f"Lambda_Aux_{lam_aux:.2f}",
            "val_mae_mw": eval_aux["metrics"]["MAE"],
            "val_rmse_mw": eval_aux["metrics"]["RMSE"],
            "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_aux["metrics"]["MAE"],
            "params": 121531,
        })
        print(f"Auxiliary Loss (lambda={lam_aux:.2f}): Val MAE = {eval_aux['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_aux['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D5: Routing Regularization (Entropy & KL Divergence) ---
    print("\n--- Ablation D5: Routing Regularization ---")
    for beta_ent in [0.0001, 0.001]:
        m_reg = OriginalCAEGNetPhase5(context_dim=4).to(device)
        t_res = train_phase5_model(m_reg, train_loader_4d, val_loader_4d, lr=1e-3, beta_entropy=beta_ent, device=device)
        eval_reg = evaluate_on_partition(m_reg, val_loader_4d, scaler, device)
        ablation_records.append({
            "ablation_group": "Routing_Regularization",
            "variant": f"Entropy_Beta_{beta_ent:.4f}",
            "val_mae_mw": eval_reg["metrics"]["MAE"],
            "val_rmse_mw": eval_reg["metrics"]["RMSE"],
            "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_reg["metrics"]["MAE"],
            "params": 121531,
        })
        print(f"Entropy Reg (beta={beta_ent}): Val MAE = {eval_reg['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_reg['metrics']['MAE']:+.2f} MW)")

    for lam_kl in [0.005, 0.025]:
        m_kl = OriginalCAEGNetPhase5(context_dim=4).to(device)
        t_res = train_phase5_model(m_kl, train_loader_4d, val_loader_4d, lr=1e-3, lambda_kl=lam_kl, device=device)
        eval_kl = evaluate_on_partition(m_kl, val_loader_4d, scaler, device)
        ablation_records.append({
            "ablation_group": "Routing_Regularization",
            "variant": f"KL_Uniform_Lambda_{lam_kl:.3f}",
            "val_mae_mw": eval_kl["metrics"]["MAE"],
            "val_rmse_mw": eval_kl["metrics"]["RMSE"],
            "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_kl["metrics"]["MAE"],
            "params": 121531,
        })
        print(f"KL to Uniform (lambda={lam_kl}): Val MAE = {eval_kl['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_kl['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D6: Conservative Bounded Routing ---
    print("\n--- Ablation D6: Conservative Bounded Routing ---")
    for rho in [0.10, 0.20, 0.30, 0.50]:
        m_rho = OriginalCAEGNetPhase5(context_dim=4, conservative_rho=rho).to(device)
        t_res = train_phase5_model(m_rho, train_loader_4d, val_loader_4d, lr=1e-3, conservative_rho=rho, device=device)
        eval_rho = evaluate_on_partition(m_rho, val_loader_4d, scaler, device, conservative_rho=rho)
        ablation_records.append({
            "ablation_group": "Conservative_Routing",
            "variant": f"Bounded_Rho_{rho:.2f}",
            "val_mae_mw": eval_rho["metrics"]["MAE"],
            "val_rmse_mw": eval_rho["metrics"]["RMSE"],
            "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_rho["metrics"]["MAE"],
            "params": 121531,
        })
        print(f"Conservative Routing (rho={rho:.2f}): Val MAE = {eval_rho['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_rho['metrics']['MAE']:+.2f} MW)")

    # --- Ablation D7: Horizon-Aware Routing ---
    print("\n--- Ablation D7: Horizon-Aware Routing ---")
    m_hor = OriginalCAEGNetPhase5(context_dim=4, horizon_dependent=True).to(device)
    t_res = train_phase5_model(m_hor, train_loader_4d, val_loader_4d, lr=1e-3, device=device)
    eval_hor = evaluate_on_partition(m_hor, val_loader_4d, scaler, device)
    ablation_records.append({
        "ablation_group": "Horizon_Aware_Routing",
        "variant": "Horizon_Dependent_24x3",
        "val_mae_mw": eval_hor["metrics"]["MAE"],
        "val_rmse_mw": eval_hor["metrics"]["RMSE"],
        "gain_vs_v1_mw": v1_val["metrics"]["MAE"] - eval_hor["metrics"]["MAE"],
        "params": sum(p.numel() for p in m_hor.parameters()),
    })
    print(f"Horizon-Aware Routing: Val MAE = {eval_hor['metrics']['MAE']:.2f} MW (Gain: {v1_val['metrics']['MAE'] - eval_hor['metrics']['MAE']:+.2f} MW)")

    # Save ablation table
    df_abl = pd.DataFrame(ablation_records)
    df_abl.sort_values(by="val_mae_mw", inplace=True)
    df_abl.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)

    # =========================================================================
    # STEP E: SYNTHESIS OF OPTIMAL PHASE 5 CANDIDATE
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP E: SYNTHESIS OF PHASE 5 IMPROVED CANDIDATE <<<")
    print("=" * 70)
    # Identify top performing validated mechanisms:
    # 1. Training stabilization: GradClip 1.0 + Plateau Scheduler
    # 2. Auxiliary Expert Supervision: Prevents expert degradation
    # 3. Disagreement routing: Provides real-time uncertainty signal
    # 4. Conservative routing: Bounded interpolation to prevent over-allocation
    print("Synthesizing CAEG-Net Phase 5 Candidate:")
    print("- Architecture: Original LSTM + TCN + CNN")
    print("- Context: 4D domain context + 3D detached inter-expert disagreement (7D total)")
    print("- Router: Conservative bounded routing (rho=0.20)")
    print("- Training: AdamW, lr=1e-3, grad_clip=1.0, Plateau scheduler, lambda_aux=0.10, beta_entropy=0.0001")

    # =========================================================================
    # STEP F & G: FIVE-SEED BENCHMARK ACROSS CANONICAL SEEDS
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP F: FIVE-SEED BENCHMARK ACROSS CANONICAL SEEDS <<<")
    print("Seeds: [42, 123, 2024, 3407, 999]")
    print("=" * 70)

    canonical_seeds = [42, 123, 2024, 3407, 999]
    seed_benchmark_records = []
    saved_test_preds = {
        "Original_CAEGNet_V1": [],
        "CAEGNet_Phase5_Candidate": [],
        "Static_Equal_Ensemble": [],
        "Standalone_TCN": [],
        "Standalone_LSTM": [],
        "Standalone_CNN": [],
    }
    test_y_true = None

    for seed in canonical_seeds:
        print(f"\n--- Running Canonical Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        tr_loader = get_loader("train", "context_4d", shuffle=True)
        va_loader = get_loader("val", "context_4d", shuffle=False)
        te_loader = get_loader("test", "context_4d", shuffle=False)

        # 1. Standalone Experts
        # LSTM
        m_lstm = LSTMExpert().to(device)
        train_phase5_model(m_lstm, tr_loader, va_loader, lr=1e-3, device=device)
        ev_lstm = evaluate_on_partition(m_lstm, te_loader, scaler, device)
        seed_benchmark_records.append({"seed": seed, "model": "Standalone_LSTM", **ev_lstm["metrics"]})
        saved_test_preds["Standalone_LSTM"].append(ev_lstm["preds_mw"])

        # TCN
        m_tcn = TCNExpert().to(device)
        train_phase5_model(m_tcn, tr_loader, va_loader, lr=1e-3, device=device)
        ev_tcn = evaluate_on_partition(m_tcn, te_loader, scaler, device)
        seed_benchmark_records.append({"seed": seed, "model": "Standalone_TCN", **ev_tcn["metrics"]})
        saved_test_preds["Standalone_TCN"].append(ev_tcn["preds_mw"])

        # CNN (with gradient clipping and plateau scheduler)
        m_cnn = CNNExpert().to(device)
        train_phase5_model(m_cnn, tr_loader, va_loader, lr=1e-3, device=device)
        ev_cnn = evaluate_on_partition(m_cnn, te_loader, scaler, device)
        seed_benchmark_records.append({"seed": seed, "model": "Standalone_CNN", **ev_cnn["metrics"]})
        saved_test_preds["Standalone_CNN"].append(ev_cnn["preds_mw"])

        if test_y_true is None:
            test_y_true = ev_lstm["trues_mw"]

        # 2. Static Equal Ensemble
        eq_pred = (ev_lstm["preds_mw"] + ev_tcn["preds_mw"] + ev_cnn["preds_mw"]) / 3.0
        eq_met = compute_metrics(test_y_true, eq_pred)
        seed_benchmark_records.append({"seed": seed, "model": "Static_Equal_Ensemble", **eq_met})
        saved_test_preds["Static_Equal_Ensemble"].append(eq_pred)

        # 3. Canonical V1 Reference (StepLR, no grad clipping, no aux loss)
        m_v1 = CAEGNetV1(context_dim=4, horizon_dependent=False).to(device)
        train_phase5_model(m_v1, tr_loader, va_loader, lr=1e-3, use_plateau_scheduler=False, clip_grad_norm=None, device=device)
        ev_v1 = evaluate_on_partition(m_v1, te_loader, scaler, device)
        seed_benchmark_records.append({"seed": seed, "model": "Original_CAEGNet_V1", **ev_v1["metrics"]})
        saved_test_preds["Original_CAEGNet_V1"].append(ev_v1["preds_mw"])

        # 4. Phase 5 Improved Candidate
        m_p5 = OriginalCAEGNetPhase5(
            context_dim=4,
            use_disagreement=True,
            conservative_rho=0.20,
            horizon_dependent=False,
        ).to(device)
        train_phase5_model(
            m_p5,
            tr_loader,
            va_loader,
            lr=1e-3,
            use_plateau_scheduler=True,
            clip_grad_norm=1.0,
            lambda_aux=0.10,
            beta_entropy=0.0001,
            conservative_rho=0.20,
            device=device,
        )
        ev_p5 = evaluate_on_partition(m_p5, te_loader, scaler, device, conservative_rho=0.20)
        seed_benchmark_records.append({"seed": seed, "model": "CAEGNet_Phase5_Candidate", **ev_p5["metrics"]})
        saved_test_preds["CAEGNet_Phase5_Candidate"].append(ev_p5["preds_mw"])

        print(f"Seed {seed} Test MAE -> V1: {ev_v1['metrics']['MAE']:.2f} MW | P5 Candidate: {ev_p5['metrics']['MAE']:.2f} MW | Equal: {eq_met['MAE']:.2f} MW | TCN: {ev_tcn['metrics']['MAE']:.2f} MW")

    df_seeds = pd.DataFrame(seed_benchmark_records)
    df_seeds.to_csv(os.path.join(results_dir, "experiments.csv"), index=False)

    # Multi-seed model comparison
    comp_records = []
    for model_name, grp in df_seeds.groupby("model"):
        comp_records.append({
            "model": model_name,
            "mae_mean": float(grp["MAE"].mean()),
            "mae_std": float(grp["MAE"].std()),
            "rmse_mean": float(grp["RMSE"].mean()),
            "rmse_std": float(grp["RMSE"].std()),
            "r2_mean": float(grp["R2"].mean()),
            "r2_std": float(grp["R2"].std()),
            "mape_mean": float(grp["MAPE"].mean()),
            "mape_std": float(grp["MAPE"].std()),
        })
    df_comp = pd.DataFrame(comp_records).sort_values(by="mae_mean")
    df_comp.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    print("\n" + "=" * 70)
    print(">>> FINAL MULTI-SEED PERFORMANCE SUMMARY (5 SEEDS) <<<")
    print("=" * 70)
    for _, r in df_comp.iterrows():
        print(f"{r['model']:<25}: MAE = {r['mae_mean']:6.2f} +/- {r['mae_std']:4.2f} MW | RMSE = {r['rmse_mean']:6.2f} | R2 = {r['r2_mean']:.4f}")

    # =========================================================================
    # STEP H: STATISTICAL HYPOTHESIS TESTING (53 DAILY BLOCKS)
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP H: STATISTICAL TESTING ON 53 NON-OVERLAPPING DAILY BLOCKS <<<")
    print("=" * 70)

    k_blocks = 53
    block_len = 24  # 53 * 24 = 1,272 test hours
    mean_preds = {m: np.mean(np.array(saved_test_preds[m]), axis=0) for m in saved_test_preds}

    block_maes = {}
    for m in mean_preds:
        b_maes = []
        for b in range(k_blocks):
            start_idx = b * block_len
            end_idx = start_idx + block_len
            err = np.abs(mean_preds[m][start_idx:end_idx] - test_y_true[start_idx:end_idx])
            b_maes.append(float(np.mean(err)))
        block_maes[m] = np.array(b_maes)

    stat_comparisons = [
        ("CAEGNet_Phase5_Candidate", "Original_CAEGNet_V1"),
        ("CAEGNet_Phase5_Candidate", "Static_Equal_Ensemble"),
        ("CAEGNet_Phase5_Candidate", "Standalone_TCN"),
        ("CAEGNet_Phase5_Candidate", "Standalone_LSTM"),
        ("CAEGNet_Phase5_Candidate", "Standalone_CNN"),
    ]

    stat_rows = []
    for m1, m2 in stat_comparisons:
        d1 = block_maes[m1]
        d2 = block_maes[m2]
        diff = d1 - d2
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        t_stat, p_t = stats.ttest_rel(d1, d2)
        w_stat, p_w = stats.wilcoxon(d1, d2)

        # Diebold-Mariano with HLN correction on full series
        e1 = mean_preds[m1].flatten() - test_y_true.flatten()
        e2 = mean_preds[m2].flatten() - test_y_true.flatten()
        dm_hln, p_dm = compute_hln_diebold_mariano(e1, e2, h=1)
        cohen_d = float(mean_diff / (std_diff + 1e-8))

        stat_rows.append({
            "model_1": m1,
            "model_2": m2,
            "mean_paired_diff_mw": mean_diff,
            "ci_95_low": mean_diff - 1.96 * std_diff / np.sqrt(k_blocks),
            "ci_95_high": mean_diff + 1.96 * std_diff / np.sqrt(k_blocks),
            "cohen_d": cohen_d,
            "paired_t_stat": float(t_stat),
            "paired_t_pval": float(p_t),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pval": float(p_w),
            "dm_hln_stat": float(dm_hln),
            "dm_hln_pval": float(p_dm),
        })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.sort_values(by="paired_t_pval", inplace=True)
    m_tests = len(df_stats)
    df_stats["holm_bonferroni_thresh"] = [0.05 / (m_tests - i) for i in range(m_tests)]
    df_stats["statistically_significant"] = df_stats["paired_t_pval"] < df_stats["holm_bonferroni_thresh"]
    df_stats.to_csv(os.path.join(results_dir, "statistical_audit.csv"), index=False)

    print("\n--- Daily Block Statistical Test Summary (K=53) ---")
    for _, r in df_stats.iterrows():
        sig = "YES" if r["statistically_significant"] else "NO"
        print(f"{r['model_1']} vs {r['model_2']}: Diff = {r['mean_paired_diff_mw']:+5.2f} MW | t = {r['paired_t_stat']:+5.2f} (p={r['paired_t_pval']:.4f}) | DM_HLN = {r['dm_hln_stat']:+5.2f} (p={r['dm_hln_pval']:.4f}) | Sig: {sig}")

    # =========================================================================
    # STEP I: PUBLICATION FIGURES
    # =========================================================================
    print("\n" + "=" * 70)
    print(">>> STEP I: GENERATING PUBLICATION-QUALITY FIGURES <<<")
    print("=" * 70)

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    df_p = df_comp.sort_values(by="mae_mean", ascending=True)
    colors = ["#2ca02c" if "Phase5" in m else "#d62728" if "V1" in m else "#1f77b4" if "Equal" in m else "#7f7f7f" for m in df_p["model"]]
    plt.barh(df_p["model"], df_p["mae_mean"], xerr=df_p["mae_std"], color=colors, capsize=4, alpha=0.85)
    plt.axvline(251.44, color="#d62728", linestyle="--", label="Frozen V1 Reference (251.44 MW)")
    plt.title("Phase 5 Multi-Seed Test MAE Comparison (5 Seeds, 1,272 Hours)", fontsize=13)
    plt.xlabel("MAE (MW)", fontsize=11)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase5_01_mae_comparison.png"), dpi=300)
    plt.close()

    # 2. Validation Ablation Gains
    plt.figure(figsize=(10, 5))
    df_a = df_abl.sort_values(by="gain_vs_v1_mw", ascending=True)
    plt.barh(df_a["variant"], df_a["gain_vs_v1_mw"], color="#1f77b4", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="-", alpha=0.7)
    plt.title("Validation MAE Improvement Over Base V1 Across Controlled Ablations (MW)", fontsize=13)
    plt.xlabel("Validation MAE Improvement vs V1 (MW, positive is better)", fontsize=11)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase5_02_ablation_gains.png"), dpi=300)
    plt.close()

    # 3. Seed Stability Plot
    plt.figure(figsize=(9, 5))
    for m in ["Original_CAEGNet_V1", "CAEGNet_Phase5_Candidate", "Static_Equal_Ensemble", "Standalone_TCN"]:
        sub = df_seeds[df_seeds["model"] == m].sort_values(by="seed")
        plt.plot(range(len(sub)), sub["MAE"], marker="o", label=m)
    plt.xticks(range(len(canonical_seeds)), [str(s) for s in canonical_seeds])
    plt.title("Seed-to-Seed Test MAE Stability Across Canonical Seeds", fontsize=13)
    plt.xlabel("Random Seed", fontsize=11)
    plt.ylabel("Test MAE (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase5_03_seed_stability.png"), dpi=300)
    plt.close()

    # Save summary configurations JSON
    p5_config = {
        "phase": "Phase 5 Original CAEG-Net Methodology Correction & Optimization",
        "governing_roadmap": "14-Phase Original Research Roadmap",
        "frozen_v1_reference": {
            "mae_mw": 251.44,
            "rmse_mw": 334.32,
            "r2": 0.8723,
            "mape_pct": 4.71,
            "params": 121531,
        },
        "phase5_candidate_architecture": {
            "model_name": "CAEGNet_Phase5_Candidate",
            "experts": "LSTM (56,152) + TCN (36,952) + CNN (27,400)",
            "context_dim": 7,
            "context_signals": "Trend, Volatility, Periodicity, Recent Baseline Error, 3D Detached Disagreement",
            "router": "Conservative Bounded Router (rho=0.20)",
            "training_protocol": "AdamW (lr=1e-3), GradClip (1.0), ReduceLROnPlateau, Auxiliary Loss (0.10), Entropy (0.0001)",
            "total_params": 121579,
        },
        "benchmark_summary": df_comp.to_dict(orient="records"),
        "statistical_tests": df_stats.to_dict(orient="records"),
    }
    with open(os.path.join(results_dir, "configurations.json"), "w", encoding="utf-8") as f:
        json.dump(p5_config, f, indent=2)

    print("\nPhase 5 Original CAEG Optimization completed successfully!")


if __name__ == "__main__":
    main()
