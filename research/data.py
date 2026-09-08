"""
CAEG-Net Research Data Pipeline & Calendar Context Utilities
============================================================
Extends the canonical data pipeline with strictly causal, inference-available
cyclic calendar representations while preserving:
- Chronological 70/15/15 split
- Train-only StandardScaler fitting
- 168h lookback -> 24h horizon sliding window structure
- Strict causality: zero target leakage, zero future timestamp leakage
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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


def extract_cyclic_calendar_features(timestamps: pd.Series) -> np.ndarray:
    """
    Extract continuous cyclic calendar features from origin timestamps.
    At forecast origin t, the timestamp is known with certainty.
    Zero future target or horizon timestamp leakage occurs.

    Features:
    1. sin(2 * pi * hour / 24)
    2. cos(2 * pi * hour / 24)
    3. sin(2 * pi * dayofweek / 7)
    4. cos(2 * pi * dayofweek / 7)

    Returns:
        cal_features: np.ndarray of shape [N, 4], dtype float32
    """
    dt = pd.to_datetime(timestamps)
    hour = dt.dt.hour.values.astype(np.float32)
    dow = dt.dt.dayofweek.values.astype(np.float32)  # Monday=0, Sunday=6

    sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
    cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
    sin_dow = np.sin(2.0 * np.pi * dow / 7.0)
    cos_dow = np.cos(2.0 * np.pi * dow / 7.0)

    cal = np.column_stack([sin_hour, cos_hour, sin_dow, cos_dow]).astype(np.float32)
    return cal


def prepare_research_pipeline(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    lookback: int = 168,
    horizon: int = 24,
) -> Dict[str, Union[Dict, object, np.ndarray]]:
    """
    Execute full causal data pipeline and prepare context representations:
    - 3D context (Trend, Volatility, Periodicity)
    - 4D context (Trend, Volatility, Periodicity, Recent Error)
    - 8D context (4D Operational + 4D Cyclic Calendar)
    """
    # 1. Load and Clean
    df, diagnostics = load_and_clean_data(data_path)
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)

    # 2. Partition windows
    windows = create_partition_windows_with_context(
        train_sc, val_sc, test_sc, lookback=lookback, horizon=horizon
    )

    # 3. Recent Forecast Errors (Walk-forward historical forecaster)
    rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows)

    # 4. Canonical Operational Contexts
    C_tr_4d = extract_context_features(windows["train"]["X"], rec_tr)
    C_val_4d = extract_context_features(windows["val"]["X"], rec_val)
    C_test_4d = extract_context_features(windows["test"]["X"], rec_test)

    C_tr_3d = C_tr_4d[:, :3]
    C_val_3d = C_val_4d[:, :3]
    C_test_3d = C_test_4d[:, :3]

    # 5. Extract Origin Timestamps for Calendar Features
    hist_val_ts = train_df["timestamp"].iloc[-(lookback - 1):]
    val_ts_ext = pd.concat([hist_val_ts, val_df["timestamp"]]).reset_index(drop=True)

    hist_test_ts = val_df["timestamp"].iloc[-(lookback - 1):]
    test_ts_ext = pd.concat([hist_test_ts, test_df["timestamp"]]).reset_index(drop=True)

    origins_tr = windows["train"]["origins"]
    origins_va = windows["val"]["origins"]
    origins_te = windows["test"]["origins"]

    ts_origin_tr = train_df["timestamp"].iloc[origins_tr].reset_index(drop=True)
    ts_origin_va = val_ts_ext.iloc[origins_va].reset_index(drop=True)
    ts_origin_te = test_ts_ext.iloc[origins_te].reset_index(drop=True)

    cal_tr = extract_cyclic_calendar_features(ts_origin_tr)
    cal_va = extract_cyclic_calendar_features(ts_origin_va)
    cal_te = extract_cyclic_calendar_features(ts_origin_te)

    # Combined 8D Context: [Operational 4D, Calendar 4D]
    C_tr_8d = np.column_stack([C_tr_4d, cal_tr]).astype(np.float32)
    C_val_8d = np.column_stack([C_val_4d, cal_va]).astype(np.float32)
    C_test_8d = np.column_stack([C_test_4d, cal_te]).astype(np.float32)

    return {
        "raw_df": df,
        "split_dfs": {"train": train_df, "val": val_df, "test": test_df},
        "scaler": scaler,
        "windows": windows,
        "origin_timestamps": {
            "train": ts_origin_tr,
            "val": ts_origin_va,
            "test": ts_origin_te,
        },
        "context_3d": {"train": C_tr_3d, "val": C_val_3d, "test": C_test_3d},
        "context_4d": {"train": C_tr_4d, "val": C_val_4d, "test": C_test_4d},
        "context_8d": {"train": C_tr_8d, "val": C_val_8d, "test": C_test_8d},
        "calendar_only": {"train": cal_tr, "val": cal_va, "test": cal_te},
    }


def build_research_dataloaders(
    pipeline_data: Dict,
    context_type: str = "4d",
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Construct reproducible PyTorch DataLoaders for train, validation, and test.

    context_type: '3d', '4d', '8d'
    """
    windows = pipeline_data["windows"]
    context_map = {
        "3d": pipeline_data["context_3d"],
        "4d": pipeline_data["context_4d"],
        "8d": pipeline_data["context_8d"],
    }
    if context_type not in context_map:
        raise ValueError(f"Unknown context_type '{context_type}'. Must be one of {list(context_map.keys())}")

    ctx = context_map[context_type]
    ds_train = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], ctx["train"])
    ds_val = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], ctx["val"])
    ds_test = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], ctx["test"])

    dataloaders = create_dataloaders(
        {"train": ds_train, "val": ds_val, "test": ds_test},
        batch_size=batch_size,
        shuffle_train=True,
        seed=seed,
    )
    return dataloaders
