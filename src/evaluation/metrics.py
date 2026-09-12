"""
Evaluation Metrics & Inference Evaluation
"""
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    scaler=None,
    device: str = "cpu",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    model.to(device)
    preds, targets, weights_list = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_c in loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            try:
                res = model(batch_x, batch_c, return_diagnostics=True)
                if isinstance(res, tuple):
                    y_p, w = res[0], res[1]
                    weights_list.append(w.cpu().numpy())
                else:
                    y_p = res
            except TypeError:
                y_p = model(batch_x)

            preds.append(y_p.cpu().numpy())
            targets.append(batch_y.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    all_weights = np.vstack(weights_list) if weights_list else None

    if scaler is not None:
        preds_orig = scaler.inverse_transform(preds)
        targets_orig = scaler.inverse_transform(targets)
    else:
        preds_orig = preds
        targets_orig = targets

    metrics = compute_metrics(targets_orig, preds_orig)
    return metrics, targets_orig, preds_orig, all_weights
