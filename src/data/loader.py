"""
Data Loading, Splitting, and Standardization
============================================
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_load_data(
    num_hours: int = 3000,
    start_date: str = "2024-01-01 00:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range(start=start_date, periods=num_hours, freq="h")
    t = np.arange(num_hours, dtype=np.float64)

    base_load = 5000.0
    daily_cycle = 800.0 * np.sin(2.0 * np.pi * t / 24.0) + 400.0 * np.cos(4.0 * np.pi * t / 24.0)
    weekly_cycle = 600.0 * np.sin(2.0 * np.pi * t / 168.0)
    trend = 0.15 * t
    noise_std = np.where(t > (num_hours / 2.0), 120.0, 60.0)
    noise = rng.normal(loc=0.0, scale=noise_std, size=num_hours)
    load = base_load + daily_cycle + weekly_cycle + trend + noise

    return pd.DataFrame({
        "timestamp": timestamps,
        "load": np.round(load, 2),
    })


def load_and_clean_data(
    file_path: str,
    timestamp_col: Optional[str] = None,
    load_col: Optional[str] = None,
    freq: str = "h",
    fill_strategy: str = "interpolate",
    max_fill_limit: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, str, float, bool]]]:
    df = pd.read_csv(file_path)
    diagnostics: Dict[str, Union[int, str, float, bool]] = {
        "raw_rows": len(df),
        "raw_columns": list(df.columns),
    }

    if timestamp_col is None:
        candidate_ts = [c for c in df.columns if any(k in c.lower() for k in ["datetime", "timestamp", "date", "time"])]
        if not candidate_ts:
            raise ValueError(f"Could not automatically detect timestamp column in {df.columns.tolist()}")
        timestamp_col = candidate_ts[0]
    diagnostics["timestamp_col"] = timestamp_col

    if load_col is None:
        candidate_load = [c for c in df.columns if any(k in c.lower() for k in ["load", "demand", "mw", "kw", "total"])]
        if not candidate_load:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            candidate_load = [c for c in numeric_cols if c != timestamp_col]
        if not candidate_load:
            raise ValueError(f"Could not automatically detect load column in {df.columns.tolist()}")
        load_col = candidate_load[0]
    diagnostics["load_col"] = load_col

    df["parsed_timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    invalid_ts = int(df["parsed_timestamp"].isna().sum())
    diagnostics["invalid_timestamps"] = invalid_ts
    if invalid_ts > 0:
        df = df.dropna(subset=["parsed_timestamp"]).copy()

    df = df.sort_values("parsed_timestamp").reset_index(drop=True)

    num_dups = int(df.duplicated(subset=["parsed_timestamp"]).sum())
    diagnostics["duplicate_timestamps"] = num_dups
    if num_dups > 0:
        df = df.groupby("parsed_timestamp", as_index=False)[load_col].mean()

    clean_series = pd.DataFrame({
        "timestamp": df["parsed_timestamp"],
        "load": pd.to_numeric(df[load_col], errors="coerce"),
    }).dropna(subset=["timestamp"])

    diagnostics["initial_missing_load"] = int(clean_series["load"].isna().sum())

    t_start = clean_series["timestamp"].min()
    t_end = clean_series["timestamp"].max()
    diagnostics["start_time"] = str(t_start)
    diagnostics["end_time"] = str(t_end)

    full_date_range = pd.date_range(start=t_start, end=t_end, freq=freq)
    diagnostics["expected_regular_steps"] = len(full_date_range)
    diagnostics["actual_steps_before_reindex"] = len(clean_series)

    clean_series = clean_series.set_index("timestamp").reindex(full_date_range)
    clean_series.index.name = "timestamp"

    missing_intervals = int(clean_series["load"].isna().sum())
    diagnostics["total_missing_intervals"] = missing_intervals

    is_na = clean_series["load"].isna()
    gap_blocks = (~is_na).cumsum()[is_na]
    max_gap = int(gap_blocks.value_counts().max()) if not gap_blocks.empty else 0
    diagnostics["max_consecutive_gap_hours"] = max_gap

    if missing_intervals > 0:
        if fill_strategy == "interpolate":
            clean_series["load"] = clean_series["load"].interpolate(method="time", limit=max_fill_limit)
        elif fill_strategy == "ffill":
            clean_series["load"] = clean_series["load"].ffill(limit=max_fill_limit)

    diagnostics["remaining_missing_after_fill"] = int(clean_series["load"].isna().sum())
    cleaned_df = clean_series.reset_index()
    return cleaned_df, diagnostics


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Union[int, str, float]]]:
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    split_info = {
        "total_samples": n,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_pct": len(train_df) / n * 100.0,
        "val_pct": len(val_df) / n * 100.0,
        "test_pct": len(test_df) / n * 100.0,
        "train_start": str(train_df["timestamp"].iloc[0]),
        "train_end": str(train_df["timestamp"].iloc[-1]),
        "val_start": str(val_df["timestamp"].iloc[0]),
        "val_end": str(val_df["timestamp"].iloc[-1]),
        "test_start": str(test_df["timestamp"].iloc[0]),
        "test_end": str(test_df["timestamp"].iloc[-1]),
    }
    return train_df, val_df, test_df, split_info


def fit_and_transform_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_col: str = "load",
) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    train_loads = train_df[[load_col]].values.astype(np.float64)
    scaler.fit(train_loads)

    train_transformed = train_df.copy()
    val_transformed = val_df.copy()
    test_transformed = test_df.copy()

    train_transformed[load_col] = scaler.transform(train_df[[load_col]].values)
    val_transformed[load_col] = scaler.transform(val_df[[load_col]].values)
    test_transformed[load_col] = scaler.transform(test_df[[load_col]].values)

    return scaler, train_transformed, val_transformed, test_transformed
