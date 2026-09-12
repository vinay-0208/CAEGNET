"""
Data Loading, Splitting, and Standardization
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_load_data(
    num_hours: int = 3000,
    start_date: str = "2024-01-01 00:00:00",
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=num_hours, freq="h")
    t = np.arange(num_hours)
    diurnal = 300.0 * np.sin(2 * np.pi * t / 24.0)
    weekly = 150.0 * np.sin(2 * np.pi * t / 168.0)
    trend = 0.05 * t
    noise = np.random.normal(0, 40.0, size=num_hours)
    load = 2500.0 + diurnal + weekly + trend + noise
    df = pd.DataFrame({"timestamp": date_range, "load": load})
    return df


def load_and_clean_data(
    filepath: str,
    timestamp_col: str = "timestamp",
    load_col: str = "load",
) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    df[load_col] = df[load_col].interpolate(method="linear").bfill().ffill()
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def fit_and_transform_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str = "load",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[[col]].values).flatten()
    val_scaled = scaler.transform(val_df[[col]].values).flatten()
    test_scaled = scaler.transform(test_df[[col]].values).flatten()
    return train_scaled, val_scaled, test_scaled, scaler
