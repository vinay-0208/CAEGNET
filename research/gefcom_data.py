"""
GEFCom2014 Load Dataset Adapter
===============================
Standardized data pipeline and windowing adapter for GEFCom2014 Load forecasting.
Constructs canonical 168h lookback -> 24h horizon forecasting windows with:
1. Strict chronological train/validation/test partitions.
2. Train-only scaling (zero leakage).
3. 4 domain context features (trend, volatility, lag-24 autocorrelation, out-of-sample recent error).
4. Task-level mapping for all 15 official monthly rounds.
5. Non-overlapping K=457 daily evaluation blocks.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data_utils import (
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)


TASK_DURATIONS_HOURS = {
    1: 744,
    2: 720,
    3: 744,
    4: 744,
    5: 672,
    6: 744,
    7: 720,
    8: 744,
    9: 720,
    10: 744,
    11: 744,
    12: 720,
    13: 744,
    14: 720,
    15: 744,
}


def load_gefcom2014_full_series(base_dir: str = "data/Load") -> pd.DataFrame:
    """
    Load all task files and construct the complete, contiguous 7-year hourly load series
    from 2005-01-01 01:00 to 2012-01-01 00:00 (61,344 hours).
    """
    dfs = []
    
    # Task 1: Historical training data (discard pre-2005 NaN rows)
    df1_path = os.path.join(base_dir, "Task 1", "L1-train.csv")
    df1 = pd.read_csv(df1_path)
    df1_valid = df1.dropna(subset=["LOAD"]).copy()
    dfs.append(df1_valid[["ZONEID", "TIMESTAMP", "LOAD"]])
    
    # Tasks 2 to 15: Incremental monthly series
    for t in range(2, 16):
        dft_path = os.path.join(base_dir, f"Task {t}", f"L{t}-train.csv")
        dft = pd.read_csv(dft_path)
        dfs.append(dft[["ZONEID", "TIMESTAMP", "LOAD"]])
        
    # Solution to Task 15: December 2011 actual load
    sol_path = os.path.join(base_dir, "Solution to Task 15", "solution15_L.csv")
    sol = pd.read_csv(sol_path)
    dfs.append(sol[["ZONEID", "TIMESTAMP", "LOAD"]])
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["load"] = full_df["LOAD"].astype(np.float32)
    return full_df


def get_gefcom2014_partitions(
    full_df: pd.DataFrame,
    train_hours: int = 41616,
    val_hours: int = 8760,
    test_hours: int = 10968,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological partition:
    - Train: 2005-01-01 to 2009-09-30 (41,616 hours = ~4.75 years)
    - Val:   2009-10-01 to 2010-09-30 (8,760 hours = 1 full year)
    - Test:  2010-10-01 to 2011-12-31 (10,968 hours = Tasks 1-15, 457 days)
    """
    total = len(full_df)
    assert train_hours + val_hours + test_hours == total, f"Sum mismatch: {train_hours} + {val_hours} + {test_hours} != {total}"
    
    train_df = full_df.iloc[:train_hours].copy().reset_index(drop=True)
    val_df = full_df.iloc[train_hours:train_hours + val_hours].copy().reset_index(drop=True)
    test_df = full_df.iloc[train_hours + val_hours:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def prepare_gefcom2014_pipeline(
    base_dir: str = "data/Load",
    lookback: int = 168,
    horizon: int = 24,
) -> Dict:
    """
    Full end-to-end data pipeline for GEFCom2014:
    - Loads series
    - Partitions train/val/test
    - Fits scaler strictly on train
    - Creates sliding windows (lookback=168, horizon=24)
    - Computes causal recent forecast error and extracts 4 context features
    - Computes task day indices and task boundaries
    """
    full_df = load_gefcom2014_full_series(base_dir)
    train_df, val_df, test_df = get_gefcom2014_partitions(full_df)
    
    # 1. Scaling: Strictly on Train
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df, load_col="load")
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    
    # 2. Window Construction
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=lookback, horizon=horizon, load_col="load")
    
    # 3. Context Features
    rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows, warmup=500, update_step=24)
    ctx_tr = extract_context_features(windows["train"]["X"], rec_tr)
    ctx_val = extract_context_features(windows["val"]["X"], rec_val)
    ctx_te = extract_context_features(windows["test"]["X"], rec_test)
    
    # 4. Daily Evaluation Blocks Mapping
    # Non-overlapping daily origins: t = 0, 24, 48, ... in test partition
    K_days = len(windows["test"]["X"]) // horizon  # 10,944 / 24 = 456
    daily_origins = [k * horizon for k in range(K_days)]
    
    # 5. Task-to-Day Mapping
    task_daily_blocks = {}
    current_day = 0
    for t_id in range(1, 16):
        n_days_task = TASK_DURATIONS_HOURS[t_id] // horizon
        if t_id == 15:
            n_days_task = K_days - current_day  # 30 days
        task_daily_blocks[t_id] = list(range(current_day, current_day + n_days_task))
        current_day += n_days_task
        
    assert current_day == K_days, f"Task day sum mismatch: {current_day} != {K_days}"
    
    # 6. Load Official Benchmark Forecasts for Tasks 1-15
    official_benchmarks = {}
    for t_id in range(1, 16):
        bench_file = os.path.join(base_dir, f"Task {t_id}", f"L{t_id}-benchmark.csv")
        bench_df = pd.read_csv(bench_file)
        # Point forecast is identical across quantiles; take column 2 ('0.01' or '0.50')
        official_benchmarks[t_id] = bench_df.iloc[:, 2].values.astype(np.float32)
        
    return {
        "full_df": full_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "scale": scale,
        "mean": mean,
        "windows": windows,
        "context": {
            "train": ctx_tr,
            "val": ctx_val,
            "test": ctx_te,
        },
        "daily_origins": daily_origins,
        "task_daily_blocks": task_daily_blocks,
        "official_benchmarks": official_benchmarks,
        "K_days": K_days,
        "forecaster": forecaster,
    }
