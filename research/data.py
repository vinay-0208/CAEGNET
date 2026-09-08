"""
CAEG-Net V2 Research Data Pipeline & Causal Context Engineering
===============================================================
Formally specified in:
- research/results/PHASE_2_RESEARCH_MODEL_SPECIFICATION.md (Sections 8, 9, 22)
- research/results/PHASE_2_ARCHITECTURE_SCHEMA.json

Key Structural Guarantees:
1. Chronological Timeline Partitioning:
   - 70% train (6,148 rows), 15% validation (1,318 rows), 15% test (1,318 rows).
   - Strict temporal ordering: train < validation < test.
2. Train-Only StandardScaler Fitting:
   - Fitted strictly on train_df['load'].
   - Transform applied identically to train, val, and test. Zero target leakage.
3. Dual-Window Causal History Buffers:
   - Forecasting Model Input (X_t): 168 hours lookback [y_{t-167}, ..., y_t].
   - Context Extractor Buffer (X_ctx,t): 192 hours lookback [y_{t-191}, ..., y_t].
   - Boundary prepending: Exactly 191 historical observations prepended to validation
     and test partitions to guarantee full 192h lookback without losing valid target windows.
4. Tripartite Information Context (6D Observable Context Vector C_t):
   - Group A (Local Dynamics): Trend slope (168h), Volatility (168h), Range ratio (48h).
   - Group B (Seasonality): Diurnal lag-24 autocorrelation (168h), Weekly lag-168 profile
     similarity Corr(z[t-23:t], z[t-191:t-168]) over the 192h context buffer.
   - Group C (Post-Forecast Error Feedback): Recent 24h MAE from chronological walk-forward
     expanding-window Ridge regression baseline.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    compute_causal_recent_forecast_errors,
    TimeSeriesContextDataset,
    create_dataloaders,
)


def compute_causal_observable_context(X_ctx: np.ndarray) -> np.ndarray:
    """
    Compute 5 causal observable context features (Group A + Group B) from 192h buffer.

    Parameters:
    -----------
    X_ctx: np.ndarray of shape [N, 192], standardized load values.

    Returns:
    --------
    C_obs: np.ndarray of shape [N, 5], dtype float32:
        1. Trend slope beta_1 / (sigma(z) + 1e-6) over last 168h
        2. Short-term volatility std(diff(z), ddof=1) over last 168h
        3. Recent 48h range ratio (max - min) / (sigma(z_48) + 1e-6)
        4. Diurnal periodicity: lag-24 autocorrelation r_24 over last 168h
        5. Weekly profile similarity: Corr(z[t-23:t], z[t-191:t-168]) over 192h buffer
    """
    N, L = X_ctx.shape
    if L != 192:
        raise ValueError(f"Expected 192-hour context buffer, but received {L} hours.")

    # The forecasting input sequence corresponds to the final 168 hours of the 192h buffer
    X_168 = X_ctx[:, 24:]  # [N, 168]

    # 1. Trend Slope beta_1 (OLS linear regression slope over 168h normalized by std)
    time_idx = np.arange(168, dtype=np.float32)
    i_bar = 83.5  # (167 / 2)
    denom_t = np.sum((time_idx - i_bar) ** 2)  # Scalar constant: 395,122.0
    z_bar = np.mean(X_168, axis=1, keepdims=True)
    num_t = np.sum((time_idx - i_bar) * (X_168 - z_bar), axis=1, keepdims=True)
    beta_1 = num_t / denom_t
    std_168 = np.std(X_168, axis=1, keepdims=True)
    f_trend = beta_1 / (std_168 + 1e-6)

    # 2. Short-Term Volatility (Sample std of first differences over 168h)
    diff_z = np.diff(X_168, axis=1)  # [N, 167]
    f_vol = np.std(diff_z, axis=1, ddof=1, keepdims=True)

    # 3. Recent 48h Demand Range Ratio: (max(z_48) - min(z_48)) / (std(z_48) + 1e-6)
    z_48 = X_168[:, -48:]
    f_range = (np.max(z_48, axis=1, keepdims=True) - np.min(z_48, axis=1, keepdims=True)) / (
        np.std(z_48, axis=1, keepdims=True) + 1e-6
    )

    # 4. Diurnal Rhythmicity (Lag-24 sample autocorrelation r_24 over 168h)
    z_cent = X_168 - z_bar
    num_r24 = np.sum(z_cent[:, 24:] * z_cent[:, :-24], axis=1, keepdims=True)
    denom_r24 = np.sum(z_cent ** 2, axis=1, keepdims=True) + 1e-6
    f_diurnal = num_r24 / denom_r24

    # 5. Weekly Profile Similarity (Pearson Corr between recent 24h and prior-week 24h)
    # Over 192h buffer:
    # [t-191 : t-168] is X_ctx[:, 0:24]
    # [t-23 : t] is X_ctx[:, 168:192]
    z_prior_week = X_ctx[:, 0:24]
    z_recent_day = X_ctx[:, 168:192]

    mean_pw = np.mean(z_prior_week, axis=1, keepdims=True)
    mean_rd = np.mean(z_recent_day, axis=1, keepdims=True)
    num_corr = np.sum((z_prior_week - mean_pw) * (z_recent_day - mean_rd), axis=1, keepdims=True)
    std_pw = np.std(z_prior_week, axis=1, keepdims=True)
    std_rd = np.std(z_recent_day, axis=1, keepdims=True)
    denom_corr = 24.0 * std_pw * std_rd + 1e-6
    f_weekly = np.clip(num_corr / denom_corr, -1.0, 1.0)

    C_obs = np.concatenate([f_trend, f_vol, f_range, f_diurnal, f_weekly], axis=1).astype(np.float32)
    return C_obs


def create_causal_research_windows(
    series: np.ndarray,
    lookback_model: int = 168,
    lookback_context: int = 192,
    horizon: int = 24,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sliding windows for research CAEG-Net V2:
    - X: [num_windows, 168, 1] (model input)
    - X_ctx: [num_windows, 192] (context buffer)
    - Y: [num_windows, 24] (forecast targets)
    - origin_indices: [num_windows] (chronological origins)
    """
    series_1d = np.asarray(series, dtype=np.float32).flatten()
    n = len(series_1d)

    start_origin = lookback_context - 1  # Origin 191 provides 192 hours of history
    end_origin = n - horizon - 1

    if end_origin < start_origin:
        raise ValueError(f"Series length {n} is insufficient for context lookback {lookback_context} and horizon {horizon}.")

    origins = np.arange(start_origin, end_origin + 1, step)
    num_windows = len(origins)

    X = np.zeros((num_windows, lookback_model, 1), dtype=np.float32)
    X_ctx = np.zeros((num_windows, lookback_context), dtype=np.float32)
    Y = np.zeros((num_windows, horizon), dtype=np.float32)

    for i, t in enumerate(origins):
        X[i, :, 0] = series_1d[t - lookback_model + 1 : t + 1]
        X_ctx[i, :] = series_1d[t - lookback_context + 1 : t + 1]
        Y[i, :] = series_1d[t + 1 : t + horizon + 1]

    return X, X_ctx, Y, origins


def create_partition_windows_with_192h_context(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    lookback_model: int = 168,
    lookback_context: int = 192,
    horizon: int = 24,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate causal dual-window datasets for train, validation, and test.

    Boundary prepending:
    - Validation: Prepend last (lookback_context - 1) = 191 points from train series.
    - Test: Prepend last (lookback_context - 1) = 191 points from val series.
    """
    # 1. Train partition
    X_tr, X_ctx_tr, Y_tr, orig_tr = create_causal_research_windows(
        train_series, lookback_model=lookback_model, lookback_context=lookback_context, horizon=horizon
    )

    # 2. Validation partition with prepended train history
    hist_for_val = train_series[-(lookback_context - 1):]
    val_extended = np.concatenate([hist_for_val, val_series])
    X_va, X_ctx_va, Y_va, orig_va = create_causal_research_windows(
        val_extended, lookback_model=lookback_model, lookback_context=lookback_context, horizon=horizon
    )

    # 3. Test partition with prepended val history
    hist_for_test = val_series[-(lookback_context - 1):]
    test_extended = np.concatenate([hist_for_test, test_series])
    X_te, X_ctx_te, Y_te, orig_te = create_causal_research_windows(
        test_extended, lookback_model=lookback_model, lookback_context=lookback_context, horizon=horizon
    )

    return {
        "train": {"X": X_tr, "X_ctx": X_ctx_tr, "Y": Y_tr, "origins": orig_tr},
        "val": {"X": X_va, "X_ctx": X_ctx_va, "Y": Y_va, "origins": orig_va},
        "test": {"X": X_te, "X_ctx": X_ctx_te, "Y": Y_te, "origins": orig_te},
    }


def prepare_research_v2_pipeline(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    lookback_model: int = 168,
    lookback_context: int = 192,
    horizon: int = 24,
) -> Dict[str, Union[Dict, object, np.ndarray]]:
    """
    Execute complete CAEG-Net V2 data pipeline:
    1. Load and clean raw load time series.
    2. Chronological 70/15/15 timeline split.
    3. Train-only StandardScaler fit and transform.
    4. Causal dual-window generation (168h model input, 192h context buffer).
    5. Causal 6D context extraction (5 observable physical features + 1 expanding Ridge MAE).
    """
    # 1. Load and Clean
    df, diagnostics = load_and_clean_data(data_path)
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)

    train_series = train_sc["load"].values.astype(np.float32)
    val_series = val_sc["load"].values.astype(np.float32)
    test_series = test_sc["load"].values.astype(np.float32)

    # 2. Partition windows with 192h context buffer
    windows = create_partition_windows_with_192h_context(
        train_series, val_series, test_series,
        lookback_model=lookback_model,
        lookback_context=lookback_context,
        horizon=horizon,
    )

    # 3. Recent baseline forecast error (Feature 6 in C_t)
    rec_tr, rec_va, rec_te, forecaster = compute_causal_recent_forecast_errors(windows)

    # 4. Extract observable 5D context features
    C_obs_tr = compute_causal_observable_context(windows["train"]["X_ctx"])
    C_obs_va = compute_causal_observable_context(windows["val"]["X_ctx"])
    C_obs_te = compute_causal_observable_context(windows["test"]["X_ctx"])

    # 5. Assemble complete 6D context vector: C_t = [C_obs_t || E_base,t]
    # Ensure rec error is 2D: [N, 1]
    rec_tr_2d = rec_tr[:, None] if rec_tr.ndim == 1 else rec_tr
    rec_va_2d = rec_va[:, None] if rec_va.ndim == 1 else rec_va
    rec_te_2d = rec_te[:, None] if rec_te.ndim == 1 else rec_te

    C_tr = np.column_stack([C_obs_tr, rec_tr_2d]).astype(np.float32)
    C_va = np.column_stack([C_obs_va, rec_va_2d]).astype(np.float32)
    C_te = np.column_stack([C_obs_te, rec_te_2d]).astype(np.float32)

    return {
        "raw_df": df,
        "split_dfs": {"train": train_df, "val": val_df, "test": test_df},
        "scaler": scaler,
        "windows": windows,
        "context_6d": {"train": C_tr, "val": C_va, "test": C_te},
        "context_obs_5d": {"train": C_obs_tr, "val": C_obs_va, "test": C_obs_te},
        "recent_error_1d": {"train": rec_tr_2d, "val": rec_va_2d, "test": rec_te_2d},
        "forecaster": forecaster,
        "diagnostics": diagnostics,
    }


def build_research_v2_dataloaders(
    pipeline_data: Dict,
    batch_size: int = 64,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    """
    Construct PyTorch DataLoaders for CAEG-Net V2.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    windows = pipeline_data["windows"]
    ctx = pipeline_data["context_6d"]

    ds_train = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], ctx["train"])
    ds_val = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], ctx["val"])
    ds_test = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], ctx["test"])

    dataloaders = create_dataloaders(
        {"train": ds_train, "val": ds_val, "test": ds_test},
        batch_size=batch_size,
        shuffle_train=True,
        seed=seed,
        pin_memory=pin_memory,
    )
    return dataloaders
