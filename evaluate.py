"""
CAEG-Net Evaluation Module
==========================
Computes formal short-term electricity load forecasting research metrics
strictly on the ORIGINAL RAW MW SCALE:
- Mean Absolute Error (MAE) [MW]
- Mean Squared Error (MSE) [MW^2]
- Root Mean Squared Error (RMSE) [MW]
- Coefficient of Determination (R^2)
- Mean Absolute Percentage Error (MAPE) [%]

Critical Protocol:
Evaluations on the test partition are performed strictly using the best validation checkpoint.
No test data is used for tuning or model selection.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute short-term load forecasting metrics on raw MW values.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    # MAPE: avoid division by near-zero readings
    valid_mask = np.abs(y_t) > 1e-3
    if valid_mask.any():
        mape = float(np.mean(np.abs((y_t[valid_mask] - y_p[valid_mask]) / y_t[valid_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def evaluate_model_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    scaler: StandardScaler,
    device: Optional[torch.device] = None,
    collect_routing: bool = True,
) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
    """
    Run inference on DataLoader, invert standardization to raw MW scale,
    and compute standard evaluation metrics.

    Returns:
    --------
    Dict containing:
      - "metrics": Dict of MAE, MSE, RMSE, R2, MAPE on raw MW
      - "y_pred_raw": np.ndarray of shape [N, 24] in MW
      - "y_true_raw": np.ndarray of shape [N, 24] in MW
      - "weights": np.ndarray of shape [N, 3] (if model supports gating)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    preds_scaled_list = []
    targets_scaled_list = []
    weights_list = []

    scale = scaler.scale_[0]
    mean = scaler.mean_[0]

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                bx, by, bc = batch
                bx = bx.to(device)
                by = by.to(device)
                bc = bc.to(device)
                try:
                    out = model(bx, bc, return_diagnostics=collect_routing)
                except TypeError:
                    try:
                        out = model(bx, bc)
                    except TypeError:
                        out = model(bx)
            else:
                bx, by = batch
                bx = bx.to(device)
                by = by.to(device)
                out = model(bx)

            if isinstance(out, tuple):
                y_pred = out[0]
                if collect_routing and len(out) >= 2 and out[1] is not None:
                    weights_list.append(out[1].cpu().numpy())
            else:
                y_pred = out

            preds_scaled_list.append(y_pred.cpu().numpy())
            targets_scaled_list.append(by.cpu().numpy())

    preds_scaled = np.concatenate(preds_scaled_list, axis=0)
    targets_scaled = np.concatenate(targets_scaled_list, axis=0)

    # Invert to raw MW scale
    preds_raw = preds_scaled * scale + mean
    targets_raw = targets_scaled * scale + mean

    metrics = compute_metrics(targets_raw, preds_raw)

    result: Dict[str, Union[Dict[str, float], np.ndarray]] = {
        "metrics": metrics,
        "y_pred_raw": preds_raw,
        "y_true_raw": targets_raw,
    }

    if weights_list:
        result["weights"] = np.concatenate(weights_list, axis=0)

    return result
