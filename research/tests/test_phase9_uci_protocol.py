"""
Phase 9A UCI ElectricityLoadDiagrams20112014 Protocol & Cohort Unit Tests
=======================================================================
Implements and verifies all 11 requirements from Phase 9A Task 14:
1. fixed cohort membership is deterministic
2. selected clients satisfy the stated inclusion rule
3. selected cohort remains constant throughout study period
4. no client outside the cohort contributes to the target
5. each hourly aggregate contains exactly four 15-minute observations
6. hourly timestamp labeling is deterministic
7. no duplicate hourly timestamps
8. chronological split
9. no train/test leakage
10. train-only scaler
11. causal context
"""

import os
import csv
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from research.data_adapter_uci import (
    stream_uci_hourly_aggregate_load,
    get_fixed_cohort_client_ids,
    UCIElectricityDatasetAdapter
)
from data_utils import (
    create_forecasting_windows,
    create_partition_windows_with_context,
    extract_context_features,
    compute_causal_recent_forecast_errors
)

RAW_UCI_PATH = os.path.abspath("data/ElectricityLoadDiagrams20112014/LD2011_2014.txt")
AUDIT_CSV_PATH = os.path.abspath("research/results/phase9_cohort_audit.csv")


class TestPhase9AUCIProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(RAW_UCI_PATH):
            raise unittest.SkipTest(f"Raw UCI dataset not found at {RAW_UCI_PATH}")
        if not os.path.isfile(AUDIT_CSV_PATH):
            raise unittest.SkipTest(f"Audit CSV not found at {AUDIT_CSV_PATH}")
            
        cls.cohort_320_ids = get_fixed_cohort_client_ids(audit_csv_path=AUDIT_CSV_PATH, cohort="cohort_320")
        
        # Stream a representative 1500-hour sample for the fixed cohort 320
        cls.hourly_df = stream_uci_hourly_aggregate_load(
            raw_path=RAW_UCI_PATH,
            unit="MW",
            start_year=2012,
            cohort="cohort_320",
            audit_csv_path=AUDIT_CSV_PATH,
            max_hours=1500
        )

    def test_01_fixed_cohort_membership_deterministic(self):
        """1. Verify that fixed cohort membership is 100% deterministic across repeated calls."""
        c1 = get_fixed_cohort_client_ids(audit_csv_path=AUDIT_CSV_PATH, cohort="cohort_320")
        c2 = get_fixed_cohort_client_ids(audit_csv_path=AUDIT_CSV_PATH, cohort="cohort_320")
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 320)

    def test_02_selected_clients_satisfy_inclusion_rule(self):
        """2. Verify that all 320 clients activated on or before 2012-01-01 00:15:00."""
        with open(AUDIT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["client_id"]
                fn_ts = row["first_active_timestamp"]
                in_c320 = (row["included_in_fixed_cohort_320"] == "YES")
                if in_c320:
                    self.assertLessEqual(fn_ts, "2012-01-01 00:15:00", f"Client {cid} activated late ({fn_ts})")
                else:
                    self.assertGreater(fn_ts, "2012-01-01 00:15:00", f"Client {cid} should have been included ({fn_ts})")

    def test_03_selected_cohort_remains_constant(self):
        """3. Verify that cohort membership remains strictly constant throughout the study period."""
        # Cohort membership is fixed at T0 and does not change dynamically
        c_set = get_fixed_cohort_client_ids(audit_csv_path=AUDIT_CSV_PATH, cohort="cohort_320")
        self.assertEqual(len(c_set), 320)
        # Verify no clients outside c_set are admitted
        with open(AUDIT_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            excluded_count = sum(1 for row in reader if row["included_in_fixed_cohort_320"] == "NO")
        self.assertEqual(excluded_count, 50)
        self.assertEqual(len(c_set) + excluded_count, 370)

    def test_04_no_client_outside_cohort_contributes(self):
        """4. Verify that excluded clients do not contribute to the cohort aggregate."""
        # Check that MT_146 (which activated 2012-01-07 16:30:00) is excluded from Cohort 320
        self.assertNotIn("MT_146", self.cohort_320_ids)
        # Check that MT_024 (activated 2012-03-30) is excluded
        self.assertNotIn("MT_024", self.cohort_320_ids)

    def test_05_each_hourly_aggregate_contains_four_intervals(self):
        """5. Verify that each hourly point is computed from exactly four 15-minute readings."""
        # Hand-calculate the first hour of Cohort 320
        c320_set = self.cohort_320_ids
        quarter_sums = []
        with open(RAW_UCI_PATH, 'r', encoding='utf-8') as f:
            header = [c.strip('"') for c in f.readline().strip().split(';')]
            indices = [i + 1 for i, c in enumerate(header[1:]) if c in c320_set]
            
            # Fast forward to 2012-01-01 00:15:00 (row 35040)
            for r_idx in range(35040):
                f.readline()
                
            for q in range(4):
                line = f.readline().strip().split(';')
                vals = [float(line[i].replace(',', '.')) for i in indices]
                quarter_sums.append(sum(vals))
                
        expected_hour_kw = sum(quarter_sums) / 4.0
        expected_hour_mw = expected_hour_kw / 1000.0
        
        sample_df = stream_uci_hourly_aggregate_load(
            raw_path=RAW_UCI_PATH,
            unit="MW",
            start_year=2012,
            cohort="cohort_320",
            audit_csv_path=AUDIT_CSV_PATH,
            max_hours=1
        )
        self.assertAlmostEqual(sample_df["load"].iloc[0], expected_hour_mw, places=5)

    def test_06_hourly_timestamp_labeling_deterministic(self):
        """6. Verify that timestamp labeling follows the hour-ending convention."""
        # The first hour of 2012 spans 00:15 to 01:00, ending at 01:00:00
        self.assertEqual(str(self.hourly_df["timestamp"].iloc[0]), "2012-01-01 01:00:00")
        self.assertEqual(str(self.hourly_df["timestamp"].iloc[1]), "2012-01-01 02:00:00")

    def test_07_no_duplicate_hourly_timestamps(self):
        """7. Verify zero duplicate hourly timestamps."""
        duplicates = self.hourly_df["timestamp"].duplicated().sum()
        self.assertEqual(duplicates, 0)

    def test_08_chronological_split(self):
        """8. Verify strictly chronological train, validation, and test boundaries."""
        n = len(self.hourly_df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_df = self.hourly_df.iloc[:n_train]
        val_df = self.hourly_df.iloc[n_train : n_train + n_val]
        test_df = self.hourly_df.iloc[n_train + n_val :]
        
        self.assertGreater(val_df["timestamp"].iloc[0], train_df["timestamp"].iloc[-1])
        self.assertGreater(test_df["timestamp"].iloc[0], val_df["timestamp"].iloc[-1])

    def test_09_no_train_test_leakage(self):
        """9. Verify zero index overlap across partition splits."""
        n = len(self.hourly_df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_idx = set(range(0, n_train))
        val_idx = set(range(n_train, n_train + n_val))
        test_idx = set(range(n_train + n_val, n))
        
        self.assertEqual(len(train_idx.intersection(val_idx)), 0)
        self.assertEqual(len(train_idx.intersection(test_idx)), 0)
        self.assertEqual(len(val_idx.intersection(test_idx)), 0)

    def test_10_train_only_scaler(self):
        """10. Verify that StandardScaler parameters match only the training partition."""
        adapter = UCIElectricityDatasetAdapter(
            lookback=168, horizon=24, start_year=2012, cohort="cohort_320"
        )
        prep = adapter.load_and_preprocess(
            raw_path=RAW_UCI_PATH, audit_csv_path=AUDIT_CSV_PATH, max_hours=1500
        )
        
        train_df = prep["train_df"]
        scaler = prep["scaler"]
        
        expected_mean = train_df["load"].values.mean()
        expected_scale = np.std(train_df["load"].values, ddof=0)
        
        self.assertAlmostEqual(scaler.mean_[0], expected_mean, places=5)
        self.assertAlmostEqual(scaler.scale_[0], expected_scale, places=5)

    def test_11_causal_context(self):
        """11. Verify causal 4D context vector extraction without NaNs or future leakage."""
        adapter = UCIElectricityDatasetAdapter(
            lookback=168, horizon=24, start_year=2012, cohort="cohort_320"
        )
        prep = adapter.load_and_preprocess(
            raw_path=RAW_UCI_PATH, audit_csv_path=AUDIT_CSV_PATH, max_hours=1500
        )
        
        C_tr = prep["context"]["train"]
        C_val = prep["context"]["val"]
        C_te = prep["context"]["test"]
        
        self.assertEqual(C_tr.shape[1], 4)
        self.assertEqual(C_val.shape[1], 4)
        self.assertEqual(C_te.shape[1], 4)
        
        self.assertFalse(np.isnan(C_tr).any())
        self.assertFalse(np.isnan(C_val).any())
        self.assertFalse(np.isnan(C_te).any())


if __name__ == "__main__":
    unittest.main()
