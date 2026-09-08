"""
GEFCom2014 Benchmark Dataset Adapter for CAEG-Net
=================================================
Provides a reproducible data loading and preprocessing adapter for the
Global Energy Forecasting Competition 2014 (GEFCom2014) electricity load track.

Protocol Preservation:
- Chronological 70/15/15 partition
- In-sample only StandardScaler fitting (zero future target leakage)
- 168-hour input sequence -> 24-hour ahead multi-step forecast
- Identical context feature extraction (Trend, Volatility, Periodicity, Recent Error)
- Exact compatibility with ResearchCAEGNet and CAEG-Net V1 architectures
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from data_utils import (
    create_forecasting_windows,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
    TimeSeriesContextDataset,
    create_dataloaders
)


class GEFCom2014DatasetAdapter:
    """
    Adapter for processing GEFCom2014 hourly electricity load data.
    """
    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scaler = StandardScaler()
        self.is_prepared = False

    def load_and_preprocess(
        self,
        csv_path: str,
        timestamp_col: str = "timestamp",
        load_col: str = "load",
    ) -> Dict[str, Union[pd.DataFrame, Dict, np.ndarray]]:
        """
        Load, validate, chronologically split, scale, and partition GEFCom2014 series.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"GEFCom2014 raw dataset not found at '{csv_path}'. "
                "Download the standard GEFCom2014 load benchmark to this path to execute."
            )

        df = pd.read_csv(csv_path)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
        test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

        # Fit scaler ONLY on training data
        self.scaler.fit(train_df[[load_col]].values)

        train_sc = train_df.copy()
        val_sc = val_df.copy()
        test_sc = test_df.copy()

        train_sc[load_col] = self.scaler.transform(train_df[[load_col]].values)
        val_sc[load_col] = self.scaler.transform(val_df[[load_col]].values)
        test_sc[load_col] = self.scaler.transform(test_df[[load_col]].values)

        # Sliding window construction
        windows = create_partition_windows_with_context(
            train_sc, val_sc, test_sc, lookback=self.lookback, horizon=self.horizon, load_col=load_col
        )

        # Out-of-sample recent error feedback
        rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows)

        # 4D Context
        C_tr = extract_context_features(windows["train"]["X"], rec_tr)
        C_val = extract_context_features(windows["val"]["X"], rec_val)
        C_test = extract_context_features(windows["test"]["X"], rec_test)

        self.is_prepared = True

        return {
            "scaler": self.scaler,
            "windows": windows,
            "context_4d": {"train": C_tr, "val": C_val, "test": C_test},
            "split_info": {
                "total_hours": n,
                "train_hours": len(train_df),
                "val_hours": len(val_df),
                "test_hours": len(test_df),
            },
        }

    def create_dataloaders(
        self,
        processed_data: Dict,
        batch_size: int = 64,
        seed: int = 42,
    ) -> Dict[str, DataLoader]:
        windows = processed_data["windows"]
        ctx = processed_data["context_4d"]

        ds_tr = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], ctx["train"])
        ds_va = TimeSeriesContextDataset(windows["val"]["X"], windows["val"]["Y"], ctx["val"])
        ds_te = TimeSeriesContextDataset(windows["test"]["X"], windows["test"]["Y"], ctx["test"])

        return create_dataloaders(
            {"train": ds_tr, "val": ds_va, "test": ds_te},
            batch_size=batch_size,
            shuffle_train=True,
            seed=seed,
        )
