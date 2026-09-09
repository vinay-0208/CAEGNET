"""
UCI ElectricityLoadDiagrams20112014 Dataset Adapter for CAEG-Net
================================================================
Provides a reproducible, memory-efficient data loading and preprocessing adapter
for the UCI ElectricityLoadDiagrams20112014 dataset with rigorous fixed-cohort selection.

Phase 9A Methodological Correction:
- Avoids non-stationary cohort contamination (where 49 clients onboarded during 2012-2014).
- Implements the objective Fixed Cohort 320 rule:
    Inclusion: first_active_timestamp <= "2012-01-01 00:15:00"
- Invariant cohort membership (exactly 320 clients continuously active throughout 2012-2014).
- Hourly aggregation via exact arithmetic mean of four 15-minute readings:
    P_hour = (P_15m1 + P_15m2 + P_15m3 + P_15m4) / 4.0
- Timestamp Convention: Hour-Ending (00:15, 00:30, 00:45, 01:00 -> 01:00:00).
- Scale: System aggregate load in Megawatts (MW = kW / 1000.0).
- Lookback: 168 hours -> 24 hours forecast.
- Chronological Split: 70% train / 15% val / 15% test prior to windowing.
- Scaler: In-sample only StandardScaler fitted strictly on training partition.
- Context: 4D context vector (trend, volatility, lag-24 periodicity, causal recent error).
"""

import os
import csv
from typing import Dict, List, Optional, Set, Tuple, Union
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


def get_fixed_cohort_client_ids(
    audit_csv_path: Optional[str] = None,
    cohort: str = "cohort_320"
) -> Set[str]:
    """
    Load pre-audited fixed cohort client IDs.
    
    Parameters:
    -----------
    audit_csv_path: Path to phase9_cohort_audit.csv
    cohort: 'cohort_320' (320 clients active by 2012-01-01), 
            'cohort_321' (321 clients active by 2012-01-08),
            or 'all_370' (all clients)
    """
    if cohort == "all_370":
        return set() # Empty set signals include all

    if audit_csv_path is None:
        audit_csv_path = os.path.join(
            os.path.dirname(__file__), "results", "phase9_cohort_audit.csv"
        )
    
    if not os.path.isfile(audit_csv_path):
        raise FileNotFoundError(f"Cohort audit CSV not found at '{audit_csv_path}'. Run audit first.")

    col_name = "included_in_fixed_cohort_320" if cohort == "cohort_320" else "included_in_cohort_321"
    
    cohort_ids = set()
    with open(audit_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(col_name) == "YES":
                cohort_ids.add(row["client_id"])
                
    if not cohort_ids:
        raise ValueError(f"Cohort '{cohort}' returned 0 clients from '{audit_csv_path}'")
        
    return cohort_ids


def stream_uci_hourly_aggregate_load(
    raw_path: str,
    unit: str = "MW",
    start_year: int = 2012,
    cohort: str = "cohort_320",
    audit_csv_path: Optional[str] = None,
    client_subset: Optional[List[str]] = None,
    max_hours: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stream through raw 678 MB UCI file and compute hourly aggregated load
    for a strictly invariant fixed client cohort without loading the full raw dataset.

    Parameters:
    -----------
    raw_path: Path to LD2011_2014.txt
    unit: 'MW' (divide kW by 1000) or 'kW'
    start_year: 2012 (stable 3-year, 26304 hours) or 2011 (full 4-year, 35064 hours)
    cohort: 'cohort_320' (default, invariant fixed cohort), 'cohort_321', or 'all_370'
    audit_csv_path: Optional path to phase9_cohort_audit.csv
    client_subset: Explicit client IDs if custom
    max_hours: Optional cutoff for lightweight testing

    Returns:
    --------
    pd.DataFrame with columns ['timestamp', 'load']
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"UCI Electricity raw dataset not found at '{raw_path}'")

    # Determine client filter
    if client_subset is not None:
        target_clients = set(client_subset)
    elif cohort in ("cohort_320", "cohort_321"):
        target_clients = get_fixed_cohort_client_ids(audit_csv_path=audit_csv_path, cohort=cohort)
    else:
        target_clients = set() # all clients

    hourly_records = []
    # Note: 00:00:00 on Jan 1 is the 24th hour of Dec 31 of prior year.
    # The first interval of the calendar year is 00:15:00.
    min_ts_str = f"{start_year}-01-01 00:15:00"
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        cols = [c.strip().strip('"').strip("'") for c in header_line.split(';')]
        client_cols = cols[1:]
        
        if target_clients:
            selected_indices = [i + 1 for i, c in enumerate(client_cols) if c in target_clients]
        else:
            selected_indices = list(range(1, len(cols)))

        acc_sum = 0.0
        quarter_count = 0
        hour_counter = 0

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            ts_str = parts[0].strip().strip('"').strip("'")
            
            # Skip rows before start_year 00:15:00
            if ts_str < min_ts_str:
                continue
            
            # Sum selected cohort clients for this 15-minute interval
            row_sum = 0.0
            for idx in selected_indices:
                val_str = parts[idx].strip().replace(',', '.')
                row_sum += float(val_str)
            
            acc_sum += row_sum
            quarter_count += 1
            
            if quarter_count == 4:
                # 4 quarters = 1 hour. Average kW across the 4 quarters
                # Timestamp assigned to the hour-ending observation (ts_str)
                hourly_kw = acc_sum / 4.0
                hourly_val = (hourly_kw / 1000.0) if unit == "MW" else hourly_kw
                hourly_records.append({
                    "timestamp": ts_str,
                    "load": hourly_val
                })
                acc_sum = 0.0
                quarter_count = 0
                hour_counter += 1
                
                if max_hours is not None and hour_counter >= max_hours:
                    break

    df = pd.DataFrame(hourly_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


class UCIElectricityDatasetAdapter:
    """
    Adapter for processing UCI ElectricityLoadDiagrams20112014 data for CAEG-Net.
    """
    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 24,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        unit: str = "MW",
        start_year: int = 2012,
        cohort: str = "cohort_320",
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.unit = unit
        self.start_year = start_year
        self.cohort = cohort
        self.scaler = StandardScaler()
        self.is_prepared = False

    def load_and_preprocess(
        self,
        raw_path: str,
        audit_csv_path: Optional[str] = None,
        max_hours: Optional[int] = None,
    ) -> Dict[str, Union[pd.DataFrame, Dict, np.ndarray]]:
        """
        Stream, aggregate, chronologically partition, scale, and extract windows + context.
        """
        df = stream_uci_hourly_aggregate_load(
            raw_path=raw_path,
            unit=self.unit,
            start_year=self.start_year,
            cohort=self.cohort,
            audit_csv_path=audit_csv_path,
            max_hours=max_hours
        )

        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
        test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

        # Scale using ONLY training partition statistics
        self.scaler.fit(train_df[["load"]].values)

        train_sc = train_df.copy()
        val_sc = val_df.copy()
        test_sc = test_df.copy()

        train_sc["load"] = self.scaler.transform(train_df[["load"]].values)
        val_sc["load"] = self.scaler.transform(val_df[["load"]].values)
        test_sc["load"] = self.scaler.transform(test_df[["load"]].values)

        # Build causal sliding windows
        windows = create_partition_windows_with_context(
            train_sc, val_sc, test_sc, lookback=self.lookback, horizon=self.horizon, load_col="load"
        )

        # Compute causal out-of-sample recent error feedback
        rec_tr, rec_val, rec_test, forecaster = compute_causal_recent_forecast_errors(windows)

        # Extract 4D domain context features (Trend, Volatility, Lag-24 Periodicity, Recent Error)
        C_tr = extract_context_features(windows["train"]["X"], rec_tr)
        C_val = extract_context_features(windows["val"]["X"], rec_val)
        C_te = extract_context_features(windows["test"]["X"], rec_test)

        self.is_prepared = True

        return {
            "raw_df": df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "scaler": self.scaler,
            "windows": windows,
            "context": {
                "train": C_tr,
                "val": C_val,
                "test": C_te,
            },
            "recent_error_forecaster": forecaster,
            "split_info": {
                "n_total": n,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": len(test_df),
                "train_start": str(train_df["timestamp"].iloc[0]),
                "train_end": str(train_df["timestamp"].iloc[-1]),
                "val_start": str(val_df["timestamp"].iloc[0]),
                "val_end": str(val_df["timestamp"].iloc[-1]),
                "test_start": str(test_df["timestamp"].iloc[0]),
                "test_end": str(test_df["timestamp"].iloc[-1]),
                "num_train_windows": len(windows["train"]["X"]),
                "num_val_windows": len(windows["val"]["X"]),
                "num_test_windows": len(windows["test"]["X"]),
                "cohort": self.cohort,
            }
        }
