"""
CAEG-Net V2 Standardized Training & Evaluation Module
=====================================================
Provides rigorous, reproducible training loops, early stopping, and metric evaluation
for CAEG-Net V2 research models and ablation variants.

Guarantees:
- Checkpoint selection strictly driven by validation fused MSE.
- Zero test data access during training, early stopping, or feature selection.
- Safe CUDA transfers with pin_memory and non_blocking transfers.
- Scaler inversion to report unstandardized physical load metrics (MW).
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from research.models import (
    CAEGNetV2,
    CAEGNetV3,
    DecoupledCAEGNet,
    LearnedStaticEnsemble,
    ShrinkageRegularizedCAEG,
    CAEGNetSR,
    BoundedRoutingCAEG,
    CAEGNetBR,
    compute_caeg_v2_loss,
    compute_caeg_v3_loss,
    compute_shrinkage_loss,
    compute_bounded_loss,
    count_parameters,
)


def train_caeg_v2(
    model: CAEGNetV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_aux: float = 0.15,
    beta_entropy: float = 0.001,
    temperature: float = 0.5,
    max_epochs: int = 35,
    patience: int = 7,
    feature_mask: Optional[torch.Tensor] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, List[float]]]:
    """
    Train CAEGNetV2 with early stopping on validation fused MSE.

    Parameters:
    -----------
    feature_mask: optional binary tensor of shape [6] or [9] to zero out masked features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if feature_mask is not None:
        feature_mask = feature_mask.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None

    history = {
        "train_loss_total": [],
        "train_loss_fused": [],
        "train_loss_aux": [],
        "train_loss_entropy": [],
        "val_loss_fused": [],
        "val_loss_total": [],
        "learning_rates": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # 1. Training Phase
        model.train()
        r_total, r_fused, r_aux, r_ent = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            if feature_mask is not None:
                # Apply mask to base context c [B, 6]
                if feature_mask.shape[0] == c.shape[1]:
                    c = c * feature_mask

            optimizer.zero_grad(set_to_none=True)

            y_fused, weights, diag = model(x, c, temperature=temperature)
            loss, telem = compute_caeg_v2_loss(
                y_fused, y, diag["expert_predictions"], weights,
                lambda_aux=lambda_aux, beta_entropy=beta_entropy
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            r_total += telem["loss_total"]
            r_fused += telem["loss_fused"]
            r_aux += telem["loss_aux"]
            r_ent += telem["loss_entropy"]
            n_batches += 1

        ep_train_total = r_total / max(n_batches, 1)
        ep_train_fused = r_fused / max(n_batches, 1)
        ep_train_aux = r_aux / max(n_batches, 1)
        ep_train_ent = r_ent / max(n_batches, 1)

        history["train_loss_total"].append(ep_train_total)
        history["train_loss_fused"].append(ep_train_fused)
        history["train_loss_aux"].append(ep_train_aux)
        history["train_loss_entropy"].append(ep_train_ent)

        # 2. Validation Phase (Evaluated on validation fused MSE)
        model.eval()
        v_fused, v_total = 0.0, 0.0
        n_val = 0

        with torch.no_grad():
            for x, y, c in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                if feature_mask is not None:
                    if feature_mask.shape[0] == c.shape[1]:
                        c = c * feature_mask

                y_fused, weights, diag = model(x, c, temperature=temperature)
                loss, telem = compute_caeg_v2_loss(
                    y_fused, y, diag["expert_predictions"], weights,
                    lambda_aux=lambda_aux, beta_entropy=beta_entropy
                )
                v_fused += telem["loss_fused"]
                v_total += telem["loss_total"]
                n_val += 1

        ep_val_fused = v_fused / max(n_val, 1)
        ep_val_total = v_total / max(n_val, 1)
        history["val_loss_fused"].append(ep_val_fused)
        history["val_loss_total"].append(ep_val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)
        scheduler.step(ep_val_fused)

        if verbose:
            print(f"Epoch {epoch:02d} | Train Fused: {ep_train_fused:.4f} | Val Fused: {ep_val_fused:.4f} | LR: {current_lr:.1e}")

        # Early stopping check on validation fused loss
        if ep_val_fused < best_val_loss - 1e-5:
            best_val_loss = ep_val_fused
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (Val Fused: {best_val_loss:.4f})")
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if checkpoint_path is not None and best_state_dict is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": best_epoch,
            "state_dict": best_state_dict,
            "best_val_fused": best_val_loss,
            "history": history,
            "lambda_aux": lambda_aux,
            "beta_entropy": beta_entropy,
            "temperature": temperature,
        }, checkpoint_path)

    total_time = time.perf_counter() - start_time

    return {
        "best_epoch": best_epoch,
        "best_val_fused": best_val_loss,
        "total_epochs": epoch,
        "training_time_seconds": total_time,
        "history": history,
    }


def evaluate_caeg_v2(
    model: CAEGNetV2,
    data_loader: DataLoader,
    scaler: object,
    temperature: Optional[float] = None,
    feature_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Evaluate CAEGNetV2 on a partition and calculate unstandardized physical load metrics (MW).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_weights = []
    all_gru_pred = []
    all_tcn_pred = []
    all_patch_pred = []
    all_disagreement = []
    all_context = []

    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            if feature_mask is not None:
                if feature_mask.shape[0] == c.shape[1]:
                    c = c * feature_mask.to(device)

            y_fused, weights, diag = model(x, c, temperature=temperature)

            all_y_pred.append(y_fused)
            all_y_true.append(y)
            all_weights.append(weights)
            all_context.append(c)

            preds = diag["expert_predictions"]
            all_gru_pred.append(preds["gru"])
            all_tcn_pred.append(preds["tcn"])
            all_patch_pred.append(preds["patch"])

            if "disagreement" in diag and diag["disagreement"] is not None:
                all_disagreement.append(diag["disagreement"])

    # Bulk GPU -> CPU transfer
    y_pred_sc = torch.cat(all_y_pred, dim=0).cpu().numpy()
    y_true_sc = torch.cat(all_y_true, dim=0).cpu().numpy()
    weights_np = torch.cat(all_weights, dim=0).cpu().numpy()
    context_np = torch.cat(all_context, dim=0).cpu().numpy()

    gru_sc = torch.cat(all_gru_pred, dim=0).cpu().numpy()
    tcn_sc = torch.cat(all_tcn_pred, dim=0).cpu().numpy()
    patch_sc = torch.cat(all_patch_pred, dim=0).cpu().numpy()

    if all_disagreement:
        disagree_np = torch.cat(all_disagreement, dim=0).cpu().numpy()
    else:
        disagree_np = None

    # Invert StandardScaler: load = z * scale_ + mean_
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    y_pred_mw = y_pred_sc * scale + mean
    y_true_mw = y_true_sc * scale + mean
    gru_mw = gru_sc * scale + mean
    tcn_mw = tcn_sc * scale + mean
    patch_mw = patch_sc * scale + mean
    equal_ens_mw = (gru_mw + tcn_mw + patch_mw) / 3.0

    def calc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        mape = float(np.mean(abs_err / np.maximum(np.abs(true), 1e-6)) * 100.0)
        mse_std = float(np.mean((((pred - mean) / scale) - ((true - mean) / scale)) ** 2))
        return {
            "mae_mw": mae,
            "rmse_mw": rmse,
            "r2": r2,
            "mape_pct": mape,
            "mse_standardized": mse_std,
        }

    fused_metrics = calc_metrics(y_pred_mw, y_true_mw)
    gru_metrics = calc_metrics(gru_mw, y_true_mw)
    tcn_metrics = calc_metrics(tcn_mw, y_true_mw)
    patch_metrics = calc_metrics(patch_mw, y_true_mw)
    equal_metrics = calc_metrics(equal_ens_mw, y_true_mw)

    # Routing Diagnostics
    eps = 1e-8
    entropy_per_origin = -np.sum(weights_np * np.log(weights_np + eps), axis=-1)
    mean_entropy = float(np.mean(entropy_per_origin))
    n_eff_per_origin = np.exp(entropy_per_origin)
    mean_n_eff = float(np.mean(n_eff_per_origin))

    weight_stats = {
        "mean_weights": [float(w) for w in np.mean(weights_np, axis=0)],
        "std_weights": [float(s) for s in np.std(weights_np, axis=0)],
        "min_weights": [float(m) for m in np.min(weights_np, axis=0)],
        "max_weights": [float(m) for m in np.max(weights_np, axis=0)],
        "mean_entropy": mean_entropy,
        "mean_n_eff": mean_n_eff,
    }

    return {
        "fused_metrics": fused_metrics,
        "gru_metrics": gru_metrics,
        "tcn_metrics": tcn_metrics,
        "patch_metrics": patch_metrics,
        "equal_ensemble_metrics": equal_metrics,
        "weight_stats": weight_stats,
        "y_pred_mw": y_pred_mw,
        "y_true_mw": y_true_mw,
        "weights": weights_np,
        "context_6d": context_np,
        "disagreement": disagree_np,
        "gru_mw": gru_mw,
        "tcn_mw": tcn_mw,
        "patch_mw": patch_mw,
        "equal_ens_mw": equal_ens_mw,
        "entropy_per_origin": entropy_per_origin,
    }


# =====================================================================
# CAEG-Net V3 Training & Evaluation Functions
# =====================================================================

def train_caeg_v3(
    model: CAEGNetV3,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_aux: float = 0.40,
    beta_prior: float = 0.002,
    temperature: float = 0.5,
    max_epochs: int = 35,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Standardized Training Loop for CAEG-Net V3 with Anti-Starvation Auxiliary Supervision.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None

    history = {
        "train_loss_total": [],
        "train_loss_fused": [],
        "train_loss_aux": [],
        "train_loss_prior": [],
        "val_loss_fused": [],
        "val_loss_total": [],
        "learning_rates": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # 1. Training Phase
        model.train()
        r_total, r_fused, r_aux, r_prior = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_fused, weights, diag = model(x, c, temperature=temperature)
            loss, telem = compute_caeg_v3_loss(
                y_fused, y, diag["expert_predictions"], weights,
                lambda_aux=lambda_aux, beta_prior=beta_prior
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            r_total += telem["loss_total"]
            r_fused += telem["loss_fused"]
            r_aux += telem["loss_aux"]
            r_prior += telem["loss_prior"]
            n_batches += 1

        ep_train_total = r_total / max(n_batches, 1)
        ep_train_fused = r_fused / max(n_batches, 1)
        ep_train_aux = r_aux / max(n_batches, 1)
        ep_train_prior = r_prior / max(n_batches, 1)

        history["train_loss_total"].append(ep_train_total)
        history["train_loss_fused"].append(ep_train_fused)
        history["train_loss_aux"].append(ep_train_aux)
        history["train_loss_prior"].append(ep_train_prior)

        # 2. Validation Phase
        model.eval()
        v_fused, v_total = 0.0, 0.0
        n_val = 0

        with torch.no_grad():
            for x, y, c in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                y_fused, weights, diag = model(x, c, temperature=temperature)
                loss, telem = compute_caeg_v3_loss(
                    y_fused, y, diag["expert_predictions"], weights,
                    lambda_aux=lambda_aux, beta_prior=beta_prior
                )
                v_fused += telem["loss_fused"]
                v_total += telem["loss_total"]
                n_val += 1

        ep_val_fused = v_fused / max(n_val, 1)
        ep_val_total = v_total / max(n_val, 1)
        history["val_loss_fused"].append(ep_val_fused)
        history["val_loss_total"].append(ep_val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)
        scheduler.step(ep_val_fused)

        if verbose:
            print(f"Epoch {epoch:02d} | Train Fused: {ep_train_fused:.4f} | Val Fused: {ep_val_fused:.4f} | LR: {current_lr:.1e}")

        # Early stopping on validation fused MSE
        if ep_val_fused < best_val_loss - 1e-5:
            best_val_loss = ep_val_fused
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (Val Fused: {best_val_loss:.4f})")
                break

        if current_lr < 1e-5:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if checkpoint_path is not None and best_state_dict is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": best_epoch,
            "state_dict": best_state_dict,
            "best_val_fused": best_val_loss,
            "history": history,
            "lambda_aux": lambda_aux,
            "beta_prior": beta_prior,
            "temperature": temperature,
        }, checkpoint_path)

    total_time = time.perf_counter() - start_time

    return {
        "best_epoch": best_epoch,
        "best_val_fused": best_val_loss,
        "total_epochs": epoch,
        "training_time_seconds": total_time,
        "history": history,
    }


def evaluate_caeg_v3(
    model: CAEGNetV3,
    data_loader: DataLoader,
    scaler: object,
    temperature: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Evaluate CAEGNetV3 on a partition and calculate unstandardized physical load metrics (MW)
    along with horizon-dependent routing analytics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_weights = []
    all_gru_pred = []
    all_tcn_pred = []
    all_patch_pred = []
    all_disagreement = []
    all_context = []

    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            y_fused, weights, diag = model(x, c, temperature=temperature)

            all_y_pred.append(y_fused)
            all_y_true.append(y)
            all_weights.append(weights)
            all_context.append(c)

            preds = diag["expert_predictions"]
            all_gru_pred.append(preds["gru"])
            all_tcn_pred.append(preds["tcn"])
            all_patch_pred.append(preds["patch"])

            if "disagreement" in diag and diag["disagreement"] is not None:
                all_disagreement.append(diag["disagreement"])

    y_pred_sc = torch.cat(all_y_pred, dim=0).cpu().numpy()
    y_true_sc = torch.cat(all_y_true, dim=0).cpu().numpy()
    weights_np = torch.cat(all_weights, dim=0).cpu().numpy()  # [N, 24, 3]
    context_np = torch.cat(all_context, dim=0).cpu().numpy()

    gru_sc = torch.cat(all_gru_pred, dim=0).cpu().numpy()
    tcn_sc = torch.cat(all_tcn_pred, dim=0).cpu().numpy()
    patch_sc = torch.cat(all_patch_pred, dim=0).cpu().numpy()

    disagree_np = torch.cat(all_disagreement, dim=0).cpu().numpy() if all_disagreement else None

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    y_pred_mw = y_pred_sc * scale + mean
    y_true_mw = y_true_sc * scale + mean
    gru_mw = gru_sc * scale + mean
    tcn_mw = tcn_sc * scale + mean
    patch_mw = patch_sc * scale + mean
    equal_ens_mw = (gru_mw + tcn_mw + patch_mw) / 3.0

    def calc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        mape = float(np.mean(abs_err / np.maximum(np.abs(true), 1e-6)) * 100.0)
        return {
            "mae_mw": mae,
            "mse_mw2": float(np.mean(err ** 2)),
            "rmse_mw": rmse,
            "r2": r2,
            "mape_pct": mape,
        }

    fused_metrics = calc_metrics(y_pred_mw, y_true_mw)
    gru_metrics = calc_metrics(gru_mw, y_true_mw)
    tcn_metrics = calc_metrics(tcn_mw, y_true_mw)
    patch_metrics = calc_metrics(patch_mw, y_true_mw)
    equal_metrics = calc_metrics(equal_ens_mw, y_true_mw)

    # Routing Diagnostics across [N, 24, 3]
    eps = 1e-8
    entropy_per_step = -np.sum(weights_np * np.log(weights_np + eps), axis=-1)  # [N, 24]
    mean_entropy = float(np.mean(entropy_per_step))
    n_eff_per_step = np.exp(entropy_per_step)
    mean_n_eff = float(np.mean(n_eff_per_step))

    # Weight stats
    # Mean weight per expert across all N and all 24 horizons
    overall_mean_weights = [float(w) for w in np.mean(weights_np, axis=(0, 1))]
    # Horizon profile: [24, 3]
    weights_by_horizon = np.mean(weights_np, axis=0)

    weight_stats = {
        "mean_weights": overall_mean_weights,
        "mean_entropy": mean_entropy,
        "mean_n_eff": mean_n_eff,
        "weights_by_horizon": weights_by_horizon.tolist(),
    }

    return {
        "fused_metrics": fused_metrics,
        "gru_metrics": gru_metrics,
        "tcn_metrics": tcn_metrics,
        "patch_metrics": patch_metrics,
        "equal_ensemble_metrics": equal_metrics,
        "weight_stats": weight_stats,
        "y_pred_mw": y_pred_mw,
        "y_true_mw": y_true_mw,
        "weights": weights_np,
        "context_6d": context_np,
        "disagreement": disagree_np,
        "gru_mw": gru_mw,
        "tcn_mw": tcn_mw,
        "patch_mw": patch_mw,
        "equal_ens_mw": equal_ens_mw,
        "entropy_per_origin": np.mean(entropy_per_step, axis=1),
    }


# =====================================================================
# Decoupled CAEG-Net (D-CAEG) Training & Evaluation Functions
# =====================================================================

def train_decoupled_router(
    model: Union[DecoupledCAEGNet, LearnedStaticEnsemble],
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    max_epochs: int = 35,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, Dict]]:
    """
    Standardized Training Loop for Decoupled Router on Frozen Experts.
    Zero gradients flow into expert backbones.
    Objective: L_router = MSE(y_fused, y_true).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # Ensure experts are frozen
    model.freeze_experts()

    # Optimize ONLY trainable parameters (router and context encoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
    }

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.freeze_experts()  # Enforce eval mode on frozen backbones
        r_loss = 0.0
        n_batches = 0

        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            y_fused, weights, _ = model(x, c, temperature=temperature)
            loss = F.mse_loss(y_fused, y)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            r_loss += loss.item()
            n_batches += 1

        ep_train_loss = r_loss / max(n_batches, 1)
        history["train_loss"].append(ep_train_loss)

        # Validation Phase
        model.eval()
        v_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for x, y, c in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                y_fused, _, _ = model(x, c, temperature=temperature)
                v_loss += F.mse_loss(y_fused, y).item()
                n_val += 1

        ep_val_loss = v_loss / max(n_val, 1)
        history["val_loss"].append(ep_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)
        scheduler.step(ep_val_loss)

        if verbose:
            print(f"Epoch {epoch:02d} | Train Loss: {ep_train_loss:.4f} | Val Loss: {ep_val_loss:.4f} | LR: {current_lr:.1e}")

        # Early stopping on validation fused MSE
        if ep_val_loss < best_val_loss - 1e-5:
            best_val_loss = ep_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break

        if current_lr < 1e-5:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if checkpoint_path is not None and best_state_dict is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": best_epoch,
            "state_dict": best_state_dict,
            "best_val_loss": best_val_loss,
            "history": history,
            "temperature": temperature,
        }, checkpoint_path)

    total_time = time.perf_counter() - start_time

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "training_time_seconds": total_time,
        "history": history,
    }


def evaluate_decoupled_caeg(
    model: Union[DecoupledCAEGNet, LearnedStaticEnsemble],
    data_loader: DataLoader,
    scaler: object,
    temperature: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Evaluate Decoupled CAEG-Net or Learned Static Ensemble on a partition and
    calculate physical load metrics (MW) along with routing weight telemetry.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_weights = []
    all_gru_pred = []
    all_tcn_pred = []
    all_patch_pred = []
    all_disagreement = []
    all_context = []

    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            y_fused, weights, diag = model(x, c, temperature=temperature)

            all_y_pred.append(y_fused)
            all_y_true.append(y)
            all_weights.append(weights)
            all_context.append(c)

            preds = diag["expert_predictions"]
            all_gru_pred.append(preds["gru"])
            all_tcn_pred.append(preds["tcn"])
            all_patch_pred.append(preds["patch"])

            if "disagreement" in diag and diag["disagreement"] is not None:
                all_disagreement.append(diag["disagreement"])

    y_pred_sc = torch.cat(all_y_pred, dim=0).cpu().numpy()
    y_true_sc = torch.cat(all_y_true, dim=0).cpu().numpy()
    weights_np = torch.cat(all_weights, dim=0).cpu().numpy()  # [N, 3]
    context_np = torch.cat(all_context, dim=0).cpu().numpy()

    gru_sc = torch.cat(all_gru_pred, dim=0).cpu().numpy()
    tcn_sc = torch.cat(all_tcn_pred, dim=0).cpu().numpy()
    patch_sc = torch.cat(all_patch_pred, dim=0).cpu().numpy()

    disagree_np = torch.cat(all_disagreement, dim=0).cpu().numpy() if all_disagreement else None

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    y_pred_mw = y_pred_sc * scale + mean
    y_true_mw = y_true_sc * scale + mean
    gru_mw = gru_sc * scale + mean
    tcn_mw = tcn_sc * scale + mean
    patch_mw = patch_sc * scale + mean
    equal_ens_mw = (gru_mw + tcn_mw + patch_mw) / 3.0

    def calc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        mape = float(np.mean(abs_err / np.maximum(np.abs(true), 1e-6)) * 100.0)
        return {
            "mae_mw": mae,
            "mse_mw2": float(np.mean(err ** 2)),
            "rmse_mw": rmse,
            "r2": r2,
            "mape_pct": mape,
        }

    fused_metrics = calc_metrics(y_pred_mw, y_true_mw)
    gru_metrics = calc_metrics(gru_mw, y_true_mw)
    tcn_metrics = calc_metrics(tcn_mw, y_true_mw)
    patch_metrics = calc_metrics(patch_mw, y_true_mw)
    equal_metrics = calc_metrics(equal_ens_mw, y_true_mw)

    # Routing Diagnostics across [N, 3]
    eps = 1e-8
    entropy_per_origin = -np.sum(weights_np * np.log(weights_np + eps), axis=-1)  # [N]
    mean_entropy = float(np.mean(entropy_per_origin))
    n_eff_per_origin = np.exp(entropy_per_origin)
    mean_n_eff = float(np.mean(n_eff_per_origin))

    overall_mean_weights = [float(w) for w in np.mean(weights_np, axis=0)]

    weight_stats = {
        "mean_weights": overall_mean_weights,
        "std_weights": [float(s) for s in np.std(weights_np, axis=0)],
        "min_weights": [float(m) for m in np.min(weights_np, axis=0)],
        "max_weights": [float(m) for m in np.max(weights_np, axis=0)],
        "mean_entropy": mean_entropy,
        "mean_n_eff": mean_n_eff,
    }

    return {
        "fused_metrics": fused_metrics,
        "gru_metrics": gru_metrics,
        "tcn_metrics": tcn_metrics,
        "patch_metrics": patch_metrics,
        "equal_ensemble_metrics": equal_metrics,
        "weight_stats": weight_stats,
        "y_pred_mw": y_pred_mw,
        "y_true_mw": y_true_mw,
        "weights": weights_np,
        "context_6d": context_np,
        "disagreement": disagree_np,
        "gru_mw": gru_mw,
        "tcn_mw": tcn_mw,
        "patch_mw": patch_mw,
        "equal_ens_mw": equal_ens_mw,
        "entropy_per_origin": entropy_per_origin,
    }


# =====================================================================
# 13. Phase 5 Optimization: Shrinkage-Regularized Router Training
# =====================================================================

def train_shrinkage_router(
    model: ShrinkageRegularizedCAEG,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 1.0,
    lambda_dev: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 35,
    patience: int = 7,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, List[float]]]:
    """
    Train Shrinkage-Regularized CAEG Router with frozen backbones and optional KL deviation loss:
    L = MSE(y_fused, y_true) + lambda_dev * KL(w_t || [1/3, 1/3, 1/3])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.freeze_experts()

    # Collect only trainable parameters (context encoder and router)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_router_weights = None
    start_time = time.time()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.freeze_experts()

        total_train_loss = 0.0
        n_train_batches = 0

        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_fused, weights, diag = model(x, c, alpha=alpha)

            loss, _ = compute_shrinkage_loss(y_fused, y, weights, lambda_dev=lambda_dev)
            loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, y, c in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                y_fused, weights, _ = model(x, c, alpha=alpha)
                val_mse = F.mse_loss(y_fused, y)
                total_val_loss += val_mse.item()
                n_val_batches += 1

        val_loss = total_val_loss / max(n_val_batches, 1)
        train_loss_epoch = total_train_loss / max(n_train_batches, 1)

        train_loss_history.append(train_loss_epoch)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_router_weights = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
                if "expert" not in k
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    if best_router_weights is not None:
        model.load_state_dict(best_router_weights, strict=False)

    return {
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_loss": best_val_loss,
        "training_time_seconds": time.time() - start_time,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "alpha": alpha,
        "lambda_dev": lambda_dev,
    }


def evaluate_shrinkage_caeg(
    model: ShrinkageRegularizedCAEG,
    data_loader: DataLoader,
    scaler: object,
    alpha: Optional[float] = None,
    temperature: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Evaluate Shrinkage-Regularized CAEG-Net on a data partition and return
    unstandardized physical metrics (MW) along with shrinkage routing telemetry.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_weights = []
    all_gru_pred = []
    all_tcn_pred = []
    all_patch_pred = []
    all_disagreement = []
    all_context = []
    all_kl = []

    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            y_fused, weights, diag = model(x, c, alpha=alpha, temperature=temperature)

            all_y_pred.append(y_fused)
            all_y_true.append(y)
            all_weights.append(weights)
            all_context.append(c)

            preds = diag["expert_predictions"]
            all_gru_pred.append(preds["gru"])
            all_tcn_pred.append(preds["tcn"])
            all_patch_pred.append(preds["patch"])
            all_kl.append(diag["kl_div"])

            if "disagreement" in diag and diag["disagreement"] is not None:
                all_disagreement.append(diag["disagreement"])

    y_pred_sc = torch.cat(all_y_pred, dim=0).cpu().numpy()
    y_true_sc = torch.cat(all_y_true, dim=0).cpu().numpy()
    weights_np = torch.cat(all_weights, dim=0).cpu().numpy()
    context_np = torch.cat(all_context, dim=0).cpu().numpy()
    kl_np = torch.cat(all_kl, dim=0).cpu().numpy()

    gru_sc = torch.cat(all_gru_pred, dim=0).cpu().numpy()
    tcn_sc = torch.cat(all_tcn_pred, dim=0).cpu().numpy()
    patch_sc = torch.cat(all_patch_pred, dim=0).cpu().numpy()
    disagree_np = torch.cat(all_disagreement, dim=0).cpu().numpy() if all_disagreement else None

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    y_pred_mw = y_pred_sc * scale + mean
    y_true_mw = y_true_sc * scale + mean
    gru_mw = gru_sc * scale + mean
    tcn_mw = tcn_sc * scale + mean
    patch_mw = patch_sc * scale + mean
    equal_ens_mw = (gru_mw + tcn_mw + patch_mw) / 3.0

    def calc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))
        mape = float(np.mean(abs_err / np.maximum(np.abs(true), 1e-6)) * 100.0)
        return {
            "mae_mw": mae,
            "mse_mw2": float(np.mean(err ** 2)),
            "rmse_mw": rmse,
            "r2": r2,
            "mape_pct": mape,
        }

    fused_metrics = calc_metrics(y_pred_mw, y_true_mw)
    gru_metrics = calc_metrics(gru_mw, y_true_mw)
    tcn_metrics = calc_metrics(tcn_mw, y_true_mw)
    patch_metrics = calc_metrics(patch_mw, y_true_mw)
    equal_metrics = calc_metrics(equal_ens_mw, y_true_mw)

    eps = 1e-8
    entropy_per_origin = -np.sum(weights_np * np.log(weights_np + eps), axis=-1)
    mean_entropy = float(np.mean(entropy_per_origin))
    mean_n_eff = float(np.mean(np.exp(entropy_per_origin)))

    # Deviation from uniform [1/3, 1/3, 1/3]
    w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    l2_dev = np.linalg.norm(weights_np - w0, axis=-1)
    mean_l2_dev = float(np.mean(l2_dev))
    max_l2_dev = float(np.max(l2_dev))
    mean_kl = float(np.mean(kl_np))

    overall_mean_weights = [float(w) for w in np.mean(weights_np, axis=0)]

    weight_stats = {
        "mean_weights": overall_mean_weights,
        "std_weights": [float(s) for s in np.std(weights_np, axis=0)],
        "min_weights": [float(m) for m in np.min(weights_np, axis=0)],
        "max_weights": [float(m) for m in np.max(weights_np, axis=0)],
        "mean_entropy": mean_entropy,
        "mean_n_eff": mean_n_eff,
        "mean_l2_dev_from_equal": mean_l2_dev,
        "max_l2_dev_from_equal": max_l2_dev,
        "mean_kl_divergence": mean_kl,
    }

    return {
        "fused_metrics": fused_metrics,
        "gru_metrics": gru_metrics,
        "tcn_metrics": tcn_metrics,
        "patch_metrics": patch_metrics,
        "equal_ensemble_metrics": equal_metrics,
        "weight_stats": weight_stats,
        "y_pred_mw": y_pred_mw,
        "y_true_mw": y_true_mw,
        "weights": weights_np,
        "context_6d": context_np,
        "disagreement": disagree_np,
        "gru_mw": gru_mw,
        "tcn_mw": tcn_mw,
        "patch_mw": patch_mw,
        "equal_ens_mw": equal_ens_mw,
        "entropy_per_origin": entropy_per_origin,
        "kl_divergence": kl_np,
    }


# =====================================================================
# 10. Phase 5 Bounded Routing Trainer & Evaluator
# =====================================================================

def train_bounded_router(
    model: BoundedRoutingCAEG,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rho: float = 0.2,
    lambda_dev: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 35,
    patience: int = 7,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Union[float, int, List[float]]]:
    """
    Train Bounded Routing CAEG-Net (CAEGNetBR) router on frozen expert backbones.

    Specification:
    - Experts remain strictly frozen in eval mode throughout training.
    - Optimizer updates only router and context_encoder parameters (2,595 params).
    - Objective: MSE(y_fused, y_true) + lambda_dev * KL(w || w0).
    - Early stopping strictly governed by validation MSE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.freeze_experts()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_router_weights = None
    start_time = time.time()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        model.freeze_experts()

        total_train_loss = 0.0
        n_train_batches = 0

        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_fused, weights, diag = model(x, c, rho=rho)

            loss, _ = compute_bounded_loss(y_fused, y, weights, lambda_dev=lambda_dev)
            loss.backward()

            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation pass: evaluate fused MSE
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, y, c in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)

                y_fused, weights, _ = model(x, c, rho=rho)
                val_mse = F.mse_loss(y_fused, y)
                total_val_loss += val_mse.item()
                n_val_batches += 1

        val_loss = total_val_loss / max(n_val_batches, 1)
        train_loss_epoch = total_train_loss / max(n_train_batches, 1)

        train_loss_history.append(train_loss_epoch)
        val_loss_history.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_router_weights = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
                if "expert" not in k
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    if best_router_weights is not None:
        model.load_state_dict(best_router_weights, strict=False)

    return {
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_loss": best_val_loss,
        "training_time_seconds": time.time() - start_time,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "rho": rho,
        "lambda_dev": lambda_dev,
    }


def evaluate_bounded_caeg(
    model: BoundedRoutingCAEG,
    data_loader: DataLoader,
    scaler: object,
    rho: Optional[float] = None,
    temperature: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Evaluate Bounded Routing CAEG-Net on a data partition and return
    unstandardized physical metrics (MW) along with routing telemetry.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_y_pred = []
    all_y_true = []
    all_weights = []
    all_q_weights = []
    all_gru_pred = []
    all_tcn_pred = []
    all_patch_pred = []
    all_disagreement = []
    all_context = []
    all_kl = []

    with torch.no_grad():
        for x, y, c in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            y_fused, weights, diag = model(x, c, rho=rho, temperature=temperature)

            all_y_pred.append(y_fused)
            all_y_true.append(y)
            all_weights.append(weights)
            all_q_weights.append(diag["q_weights"])
            all_context.append(c)

            preds = diag["expert_predictions"]
            all_gru_pred.append(preds["gru"])
            all_tcn_pred.append(preds["tcn"])
            all_patch_pred.append(preds["patch"])
            all_kl.append(diag["kl_div"])

            if "disagreement" in diag and diag["disagreement"] is not None:
                all_disagreement.append(diag["disagreement"])

    y_pred_sc = torch.cat(all_y_pred, dim=0).cpu().numpy()
    y_true_sc = torch.cat(all_y_true, dim=0).cpu().numpy()
    weights_np = torch.cat(all_weights, dim=0).cpu().numpy()
    q_weights_np = torch.cat(all_q_weights, dim=0).cpu().numpy()
    context_np = torch.cat(all_context, dim=0).cpu().numpy()
    kl_np = torch.cat(all_kl, dim=0).cpu().numpy()

    gru_sc = torch.cat(all_gru_pred, dim=0).cpu().numpy()
    tcn_sc = torch.cat(all_tcn_pred, dim=0).cpu().numpy()
    patch_sc = torch.cat(all_patch_pred, dim=0).cpu().numpy()
    disagree_np = torch.cat(all_disagreement, dim=0).cpu().numpy() if all_disagreement else None

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    y_pred_mw = y_pred_sc * scale + mean
    y_true_mw = y_true_sc * scale + mean
    gru_mw = gru_sc * scale + mean
    tcn_mw = tcn_sc * scale + mean
    patch_mw = patch_sc * scale + mean
    equal_ens_mw = (gru_mw + tcn_mw + patch_mw) / 3.0

    def calc_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        err = pred - true
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))
        mape = float(np.mean(abs_err / np.maximum(np.abs(true), 1e-6)) * 100.0)
        return {"mae_mw": mae, "rmse_mw": rmse, "r2": r2, "mape_pct": mape}

    fused_metrics = calc_metrics(y_pred_mw, y_true_mw)
    gru_metrics = calc_metrics(gru_mw, y_true_mw)
    tcn_metrics = calc_metrics(tcn_mw, y_true_mw)
    patch_metrics = calc_metrics(patch_mw, y_true_mw)
    equal_metrics = calc_metrics(equal_ens_mw, y_true_mw)

    # Telemetry statistics
    eps = 1e-8
    entropy_per_origin = -np.sum(weights_np * np.log(weights_np + eps), axis=-1)
    mean_entropy = float(np.mean(entropy_per_origin))
    eff_experts = float(np.exp(mean_entropy))

    # Weight deviations from equal prior [1/3, 1/3, 1/3]
    w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    dev_l1_per_origin = np.sum(np.abs(weights_np - w0), axis=-1)
    dev_l2_per_origin = np.sqrt(np.sum((weights_np - w0) ** 2, axis=-1))
    mean_l1_dev = float(np.mean(dev_l1_per_origin))
    mean_l2_dev = float(np.mean(dev_l2_per_origin))
    max_dev = float(np.max(np.abs(weights_np - w0)))
    mean_kl = float(np.mean(kl_np))

    weight_stats = {
        "mean_gru_weight": float(np.mean(weights_np[:, 0])),
        "mean_tcn_weight": float(np.mean(weights_np[:, 1])),
        "mean_patch_weight": float(np.mean(weights_np[:, 2])),
        "std_gru_weight": float(np.std(weights_np[:, 0])),
        "std_tcn_weight": float(np.std(weights_np[:, 1])),
        "std_patch_weight": float(np.std(weights_np[:, 2])),
        "mean_entropy": mean_entropy,
        "effective_num_experts": eff_experts,
        "mean_l1_dev_from_equal": mean_l1_dev,
        "mean_l2_dev_from_equal": mean_l2_dev,
        "max_dev_from_equal": max_dev,
        "mean_kl_divergence": mean_kl,
    }

    return {
        "fused_metrics": fused_metrics,
        "gru_metrics": gru_metrics,
        "tcn_metrics": tcn_metrics,
        "patch_metrics": patch_metrics,
        "equal_ensemble_metrics": equal_metrics,
        "weight_stats": weight_stats,
        "y_pred_mw": y_pred_mw,
        "y_true_mw": y_true_mw,
        "weights": weights_np,
        "q_weights": q_weights_np,
        "context_6d": context_np,
        "disagreement": disagree_np,
        "gru_mw": gru_mw,
        "tcn_mw": tcn_mw,
        "patch_mw": patch_mw,
        "equal_ens_mw": equal_ens_mw,
        "entropy_per_origin": entropy_per_origin,
        "kl_divergence": kl_np,
    }


