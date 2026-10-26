"""
CAEG-Net Experiments Module
===========================
Defines baseline architectures, static ensemble evaluation, and comparison table utilities.

Benchmarks:
1. Persistence / Naive-24 (Day-Ahead Persistence)
2. LSTM Standalone
3. TCN Standalone
4. CNN Standalone
5. Static Ensemble (Equal 1/3 weights)
6. Standard Input-Based MoE
7. CAEG-Net (Without Recent Error - 3D Context Ablation)
8. CAEG-Net (Full Proposed - 4D Context with Out-of-Sample Feedback)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics


def evaluate_persistence_naive24(
    test_windows: Dict[str, np.ndarray],
    scaler: StandardScaler,
) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
    """
    Day-Ahead Persistence Baseline (Naive-24):
    Predicts the next 24 hours using the most recent 24 hours of the lookback sequence.
    y_hat[t+h] = y[t - 24 + h] for h = 1..24
    """
    X_test = test_windows["X"]  # [N, 168, 1]
    Y_test = test_windows["Y"]  # [N, 24]

    # Last 24 hours of the lookback sequence
    preds_scaled = X_test[:, -24:, 0]  # [N, 24]

    scale = scaler.scale_[0]
    mean = scaler.mean_[0]

    preds_raw = preds_scaled * scale + mean
    targets_raw = Y_test * scale + mean

    metrics = compute_metrics(targets_raw, preds_raw)
    return {
        "metrics": metrics,
        "y_pred_raw": preds_raw,
        "y_true_raw": targets_raw,
    }


def evaluate_static_ensemble(
    preds_lstm_raw: np.ndarray,
    preds_tcn_raw: np.ndarray,
    preds_cnn_raw: np.ndarray,
    targets_raw: np.ndarray,
) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
    """
    Static Equal-Weighted Ensemble:
    Y_hat = (1/3) * Y_LSTM + (1/3) * Y_TCN + (1/3) * Y_CNN
    """
    fused_raw = (preds_lstm_raw + preds_tcn_raw + preds_cnn_raw) / 3.0
    metrics = compute_metrics(targets_raw, fused_raw)
    return {
        "metrics": metrics,
        "y_pred_raw": fused_raw,
        "y_true_raw": targets_raw,
    }


def format_results_table(
    results_dict: Dict[str, Dict[str, Union[Dict[str, float], int, float]]]
) -> pd.DataFrame:
    """
    Format benchmark results into a clean, presentation-ready DataFrame.
    """
    rows = []
    for model_name, info in results_dict.items():
        m = info["metrics"]
        row = {
            "Model": model_name,
            "MAE (MW)": round(m["MAE"], 2),
            "RMSE (MW)": round(m["RMSE"], 2),
            "MAPE (%)": round(m["MAPE"], 2),
            "R^2": round(m["R2"], 4),
            "Params": f"{info.get('params', 0):,}",
            "Train Time (s)": round(info.get("train_time", 0.0), 1),
            "Val MSE (scaled)": round(info.get("val_loss", 0.0), 5) if info.get("val_loss") is not None else "N/A",
        }
        rows.append(row)

    df_res = pd.DataFrame(rows)
    return df_res
