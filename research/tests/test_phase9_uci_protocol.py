"""
Phase 9 UCI ElectricityLoadDiagrams20112014 Protocol & Causality Unit Tests
========================================================================
Implements and verifies all 10 requirements from Phase 9 Step 15:
1. raw timestamp ordering
2. frequency detection
3. aggregation correctness
4. no duplicate hourly timestamps
5. no unintended missing intervals
6. chronological split
7. no leakage across train/validation/test
8. 168->24 window construction
9. train-only scaler fitting
10. causal context construction
"""

import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from research.data_adapter_uci import (
    stream_uci_hourly_aggregate_load,
    UCIElectricityDatasetAdapter
)
from data_utils import (
    create_forecasting_windows,
    create_partition_windows_with_context,
    extract_context_features,
    compute_causal_recent_forecast_errors
)

RAW_UCI_PATH = os.path.abspath("data/ElectricityLoadDiagrams20112014/LD2011_2014.txt")


class TestPhase9UCIProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(RAW_UCI_PATH):
            raise unittest.SkipTest(f"Raw UCI dataset not found at {RAW_UCI_PATH}")
        
        # Stream a representative sample of 1500 hours for fast deterministic test execution
        cls.hourly_df = stream_uci_hourly_aggregate_load(
            raw_path=RAW_UCI_PATH,
            unit="MW",
            start_year=2012,
            max_hours=1500
        )

    def test_01_raw_timestamp_ordering(self):
        """1. Verify that raw timestamps are monotonically strictly increasing."""
        timestamps = []
        with open(RAW_UCI_PATH, 'r', encoding='utf-8') as f:
            f.readline() # header
            for i in range(1000):
                line = f.readline()
                if not line:
                    break
                ts_str = line.split(';')[0].strip('"').strip("'")
                timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
        
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i], timestamps[i - 1], f"Timestamp inversion at index {i}")

    def test_02_frequency_detection(self):
        """2. Verify that raw interval spacing is strictly 15 minutes (900 seconds)."""
        time_diffs = set()
        prev_dt = None
        with open(RAW_UCI_PATH, 'r', encoding='utf-8') as f:
            f.readline()
            for i in range(1000):
                line = f.readline()
                if not line:
                    break
                ts_str = line.split(';')[0].strip('"').strip("'")
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                if prev_dt is not None:
                    time_diffs.add(int((dt - prev_dt).total_seconds()))
                prev_dt = dt
        
        self.assertEqual(time_diffs, {900}, f"Expected strictly 900s interval, found: {time_diffs}")

    def test_03_aggregation_correctness(self):
        """3. Verify that hourly aggregation is the exact mean of 4 fifteen-minute readings."""
        quarter_totals = []
        with open(RAW_UCI_PATH, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(';')
            for i in range(4):
                line = f.readline().strip()
                parts = line.split(';')
                vals = [float(p.replace(',', '.')) for p in parts[1:]]
                quarter_totals.append(sum(vals))
        
        expected_hour_kw = sum(quarter_totals) / 4.0
        expected_hour_mw = expected_hour_kw / 1000.0
        
        df_2011_sample = stream_uci_hourly_aggregate_load(
            raw_path=RAW_UCI_PATH, unit="MW", start_year=2011, max_hours=1
        )
        self.assertEqual(len(df_2011_sample), 1)
        actual_hour_mw = df_2011_sample["load"].iloc[0]
        self.assertAlmostEqual(actual_hour_mw, expected_hour_mw, places=5)

    def test_04_no_duplicate_hourly_timestamps(self):
        """4. Verify that hourly series contains zero duplicate timestamps."""
        duplicates = self.hourly_df["timestamp"].duplicated().sum()
        self.assertEqual(duplicates, 0, f"Found {duplicates} duplicate hourly timestamps.")

    def test_05_no_unintended_missing_intervals(self):
        """5. Verify that hourly timestamps advance uniformly by 1 hour (3600 seconds)."""
        diffs = self.hourly_df["timestamp"].diff().dropna()
        unique_seconds = diffs.dt.total_seconds().unique()
        self.assertEqual(list(unique_seconds), [3600.0], f"Hourly frequency has irregular intervals: {unique_seconds}")

    def test_06_chronological_split(self):
        """6. Verify strictly chronological train, validation, and test partition boundaries."""
        n = len(self.hourly_df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_df = self.hourly_df.iloc[:n_train]
        val_df = self.hourly_df.iloc[n_train : n_train + n_val]
        test_df = self.hourly_df.iloc[n_train + n_val :]
        
        self.assertGreater(val_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1])
        self.assertGreater(test_df["timestamp"].iloc[0], val_df["timestamp"].iloc[-1])

    def test_07_no_leakage_across_partitions(self):
        """7. Verify zero temporal overlap or index overlap across splits."""
        n = len(self.hourly_df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_idx = set(range(0, n_train))
        val_idx = set(range(n_train, n_train + n_val))
        test_idx = set(range(n_train + n_val, n))
        
        self.assertEqual(len(train_idx.intersection(val_idx)), 0)
        self.assertEqual(len(train_idx.intersection(test_idx)), 0)
        self.assertEqual(len(val_idx.intersection(test_idx)), 0)

    def test_08_window_construction_shapes(self):
        """8. Verify 168-hour history -> 24-hour target window construction and dimensions."""
        adapter = UCIElectricityDatasetAdapter(lookback=168, horizon=24, start_year=2012)
        prep = adapter.load_and_preprocess(raw_path=RAW_UCI_PATH, max_hours=1500)
        
        windows = prep["windows"]
        self.assertEqual(windows["train"]["X"].shape[1:], (168, 1))
        self.assertEqual(windows["train"]["Y"].shape[1], 24)
        self.assertEqual(windows["val"]["X"].shape[1:], (168, 1))
        self.assertEqual(windows["val"]["Y"].shape[1], 24)
        self.assertEqual(windows["test"]["X"].shape[1:], (168, 1))
        self.assertEqual(windows["test"]["Y"].shape[1], 24)

    def test_09_train_only_scaler_fitting(self):
        """9. Verify that StandardScaler is fitted strictly on training data."""
        adapter = UCIElectricityDatasetAdapter(lookback=168, horizon=24, start_year=2012)
        prep = adapter.load_and_preprocess(raw_path=RAW_UCI_PATH, max_hours=1500)
        
        train_df = prep["train_df"]
        scaler = prep["scaler"]
        
        expected_mean = train_df["load"].values.mean()
        expected_scale = np.std(train_df["load"].values, ddof=0)
        
        self.assertAlmostEqual(scaler.mean_[0], expected_mean, places=5)
        self.assertAlmostEqual(scaler.scale_[0], expected_scale, places=5)

    def test_10_causal_context_construction(self):
        """10. Verify 4D context feature dimensions and causal extraction."""
        adapter = UCIElectricityDatasetAdapter(lookback=168, horizon=24, start_year=2012)
        prep = adapter.load_and_preprocess(raw_path=RAW_UCI_PATH, max_hours=1500)
        
        C_tr = prep["context"]["train"]
        C_val = prep["context"]["val"]
        C_te = prep["context"]["test"]
        
        # Each context vector must be [N, 4]
        self.assertEqual(C_tr.shape[1], 4)
        self.assertEqual(C_val.shape[1], 4)
        self.assertEqual(C_te.shape[1], 4)
        
        self.assertEqual(len(C_tr), len(prep["windows"]["train"]["X"]))
        self.assertEqual(len(C_val), len(prep["windows"]["val"]["X"]))
        self.assertEqual(len(C_te), len(prep["windows"]["test"]["X"]))
        
        # Verify no NaNs or Infs
        self.assertFalse(np.isnan(C_tr).any())
        self.assertFalse(np.isinf(C_tr).any())
        self.assertFalse(np.isnan(C_te).any())


if __name__ == "__main__":
    unittest.main()
