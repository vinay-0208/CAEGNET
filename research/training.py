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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from research.models import CAEGNetV2, CAEGNetV3, compute_caeg_v2_loss, compute_caeg_v3_loss, count_parameters


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
