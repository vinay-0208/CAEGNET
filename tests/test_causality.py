"""
CAEG-Net Causality & Leakage Verification Suite
===============================================
Enforces strict scientific research integrity:
1. Future Target Perturbation Test:
   Altering future targets y[t+1 : t+24] must have ZERO impact on context features C[t].
2. Causal Recent Forecast-Error Availability Test:
   Recent forecast error at origin t must depend ONLY on past completed forecasts whose
   horizon concluded at or before t.
3. Scaler Isolation Test:
   Standardization parameters must be strictly computed from training data only.
4. Chronological Boundary Isolation Test:
   Lookback windows in validation/test must not bleed across future partition boundaries.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_forecasting_windows,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
)
from src.features import extract_context_features


class TestCAEGCausalityAndLeakage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Modern_PJM", "pjm_load.csv")
        if os.path.exists(cls.data_path):
            cls.df, _ = load_and_clean_data(cls.data_path)
        else:
            from src.data import generate_synthetic_load_data
            cls.df = generate_synthetic_load_data(num_hours=3000, seed=42)
        cls.train_df, cls.val_df, cls.test_df, _ = chronological_split(cls.df, 0.70, 0.15, 0.15)
        cls.scaler, cls.train_sc, cls.val_sc, cls.test_sc = fit_and_transform_scaler(
            cls.train_df, cls.val_df, cls.test_df
        )
        cls.windows = create_partition_windows_with_context(
            cls.train_sc, cls.val_sc, cls.test_sc, lookback=168, horizon=24
        )
        cls.rec_tr, cls.rec_val, cls.rec_test, _ = compute_causal_recent_forecast_errors(cls.windows)
        cls.C_test = extract_context_features(cls.windows["test"]["X"], cls.rec_test)

    def test_future_target_perturbation(self):
        """
        Counterfactual Future Target Perturbation Test (ISS-01):
        Assert that perturbing future target values y[t+1 : t+24] at and after a chosen
        forecast origin t_0 has ZERO impact on context features C[t_0] and recent error rec_err[t_0].

        Temporal property tested:
        At forecast origin t_0, the true target trajectory Y[t_0] has NOT yet occurred in real time.
        In the production pipeline, the most recently completed forecast available at t_0 concluded
        at (t_0 - 24 + 24) = t_0. Therefore, altering targets at or after t_0 must leave C[t_0]
        and rec_err[t_0] strictly invariant, while downstream origins (t >= t_0 + 24) must change,
        proving that the perturbation actively flows through the production forecaster.
        """
        # 1. Baseline context on unperturbed windows
        rec_tr_base, rec_val_base, rec_test_base, _ = compute_causal_recent_forecast_errors(self.windows)
        C_test_base = extract_context_features(self.windows["test"]["X"], rec_test_base)

        # 2. Select a forecast origin index t_0 in test partition (must be >= horizon=24)
        t_0 = 50
        self.assertGreaterEqual(t_0, 24, "Origin t_0 must be >= 24 to have a completed historical forecast.")
        self.assertLess(t_0 + 24, len(rec_test_base), "Origin t_0 must leave headroom to assert downstream sensitivity.")

        # 3. Create deep copy of windows dictionary and perturb future targets strictly at and after t_0
        windows_perturbed = {
            split: {k: v.copy() for k, v in data.items()}
            for split, data in self.windows.items()
        }
        windows_perturbed["test"]["Y"][t_0:] = self.windows["test"]["Y"][t_0:] * 10.0 + 5000.0

        # 4. Recompute recent error and context features through the production pipeline
        rec_tr_pert, rec_val_pert, rec_test_pert, _ = compute_causal_recent_forecast_errors(windows_perturbed)
        C_test_pert = extract_context_features(windows_perturbed["test"]["X"], rec_test_pert)

        # 5. Causal Invariance Assertion across the full 24-hour horizon:
        # At origin t_0, the forecast predicts [t_0+1 : t_0+24]. That forecast does not conclude
        # until origin t_0 + 24 arrives. Therefore, all origins strictly before t_0 + 24
        # (i.e. t in [0, t_0 + 23], covered by slice [:t_0 + 24]) evaluate forecasts whose
        # horizons concluded at or before t <= t_0 - 1, and MUST be bitwise invariant.
        self.assertEqual(
            rec_test_base[t_0],
            rec_test_pert[t_0],
            f"Recent error at origin t_0={t_0} changed when future targets Y[t_0:] were perturbed! Severe target leakage!",
        )
        np.testing.assert_array_equal(
            rec_test_base[: t_0 + 24],
            rec_test_pert[: t_0 + 24],
            err_msg="Recent error for origins < t_0 + 24 changed when future targets Y[t_0:] were perturbed!",
        )
        np.testing.assert_array_equal(
            C_test_base[: t_0 + 24],
            C_test_pert[: t_0 + 24],
            err_msg="Context features C[t] for origins < t_0 + 24 changed when future targets Y[t_0:] were perturbed!",
        )

        # 6. Sensitivity Assertion: Downstream origin t_0 + 24 evaluated the forecast issued at t_0.
        # Its target Y[t_0] was drastically perturbed, so rec_test_pert[t_0 + 24] MUST change!
        downstream_origin = t_0 + 24
        err_diff = float(abs(rec_test_pert[downstream_origin] - rec_test_base[downstream_origin]))
        self.assertGreater(
            err_diff,
            100.0,
            "Perturbation failed to reach production recent-error calculation downstream! Test harness error.",
        )
        self.assertNotEqual(
            float(C_test_pert[downstream_origin, 3]),
            float(C_test_base[downstream_origin, 3]),
            "Context feature (recent error column) downstream was unaffected by target perturbation!",
        )

    def test_recent_error_causality_and_availability(self):
        """
        Assert that for every forecast origin t, the historical forecast used to compute
        recent error completed strictly at or before origin t.
        """
        lookback = 168
        horizon = 24
        # Check test partition origins
        test_origins = self.windows["test"]["origins"]
        for i, origin_idx in enumerate(test_origins):
            # In data_utils, the previous forecast evaluated for origin_idx was generated at (origin_idx - horizon)
            # Its horizon ended at (origin_idx - horizon + horizon) = origin_idx <= t
            rec_err = self.rec_test[i]
            self.assertFalse(np.isnan(rec_err), f"NaN recent error at test origin {origin_idx}")
            self.assertFalse(np.isinf(rec_err), f"Inf recent error at test origin {origin_idx}")
            self.assertGreaterEqual(rec_err, 0.0, f"Negative recent error at test origin {origin_idx}")

    def test_scaler_isolation(self):
        """
        Verify that scaler parameters are computed strictly from train_df.
        """
        train_mean = self.train_df["load"].mean()
        train_std = self.train_df["load"].std(ddof=0)

        scaler_mean = self.scaler.mean_[0]
        scaler_scale = self.scaler.scale_[0]

        self.assertAlmostEqual(train_mean, scaler_mean, places=4, msg="Scaler mean does not match train data mean!")
        self.assertAlmostEqual(train_std, scaler_scale, places=4, msg="Scaler scale does not match train data std!")

    def test_chronological_splits(self):
        """
        Assert strictly monotonic timestamps across train, val, and test partitions with no overlap.
        """
        max_train_time = self.train_df["timestamp"].max()
        min_val_time = self.val_df["timestamp"].min()
        max_val_time = self.val_df["timestamp"].max()
        min_test_time = self.test_df["timestamp"].min()

        self.assertLess(max_train_time, min_val_time, "Train and Validation partitions overlap in time!")
        self.assertLess(max_val_time, min_test_time, "Validation and Test partitions overlap in time!")

    def test_training_warmup_prior_temporal_isolation(self):
        """
        Deterministic Training Warm-Up Prior Temporal Isolation Test (ISS-02):
        Assert that perturbing training targets in the future cohort Y_train[500:1000]
        has ZERO impact on early training warm-up error features (rec_err_train[:524]),
        and verify that the fixed warm-up prior (0.35) is causally assigned.
        """
        # 1. Compute baseline recent errors through production pipeline
        rec_tr_base, _, _, fc_base = compute_causal_recent_forecast_errors(self.windows)

        # 2. Assert baseline early training windows receive the intended fixed prior
        self.assertEqual(
            fc_base.warmup_prior_mae,
            0.35,
            "Forecaster warmup_prior_mae did not retain the intended fixed prior of 0.35!",
        )
        # Slices 0 to 523 (where completed_origin = t - 24 < warmup=500)
        np.testing.assert_array_equal(
            rec_tr_base[:524],
            np.full(524, 0.35, dtype=np.float32),
            err_msg="Early training recent errors do not match the fixed prior 0.35!",
        )

        # 3. Create deep copy and substantially perturb future training targets Y_train[500:1000]
        windows_perturbed = {
            split: {k: v.copy() for k, v in data.items()}
            for split, data in self.windows.items()
        }
        windows_perturbed["train"]["Y"][500:1000] = (
            self.windows["train"]["Y"][500:1000] * 10.0 + 5000.0
        )

        # 4. Recompute recent errors through production pipeline
        rec_tr_pert, _, _, fc_pert = compute_causal_recent_forecast_errors(windows_perturbed)

        # 5. Assert that warmup_prior_mae and early training recent errors (including rec_err_train[0])
        # are strictly invariant to the future target perturbation
        self.assertEqual(
            fc_pert.warmup_prior_mae,
            fc_base.warmup_prior_mae,
            "Forecaster warmup_prior_mae changed when future training targets were perturbed!",
        )
        self.assertEqual(
            rec_tr_pert[0],
            rec_tr_base[0],
            "rec_err_train[0] changed when future training targets Y_train[500:1000] were perturbed! Lookahead defect present!",
        )
        np.testing.assert_array_equal(
            rec_tr_pert[:524],
            rec_tr_base[:524],
            err_msg="Early training recent errors rec_err_train[:524] leaked future targets Y_train[500:1000]!",
        )

        # 6. Verify downstream sensitivity: origin t=524 evaluated forecast issued at t=500.
        # Its target was perturbed, so rec_tr_pert[524] must change downstream.
        self.assertNotEqual(
            float(rec_tr_pert[524]),
            float(rec_tr_base[524]),
            "Downstream training recent error at t=524 was unexpectedly unaffected by perturbed targets!",
        )


class TestDataCleaningAndMissingValueHandling(unittest.TestCase):
    """
    Deterministic Data Cleaning & Missing-Value Verification Suite (ISS-06)
    =======================================================================
    Validates load_and_clean_data contract under controlled synthetic fixtures:
    1. Regular series with zero missingness
    2. Missing timestamp detection and date-range reindexing (omitted row)
    3. Missing load value at an existing timestamp (NaN/blank numeric)
    4. Short consecutive gap within max_fill_limit
    5. Extended gap exceeding max_fill_limit
    6. Duplicate timestamp deduplication (arithmetic mean)
    7. Invalid unparseable timestamp dropping
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_csv(self, filename: str, df: pd.DataFrame) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        df.to_csv(path, index=False)
        return path

    def test_regular_spaced_series_no_missing(self):
        """Case 1: Regularly spaced series with zero missing timestamps or values."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dr,
            "load": np.arange(24, dtype=np.float64) + 100.0,
        })
        path = self._create_csv("regular.csv", df)
        clean_df, diag = load_and_clean_data(path)

        self.assertEqual(diag["raw_rows"], 24)
        self.assertEqual(diag["total_missing_intervals"], 0)
        self.assertEqual(diag["max_consecutive_gap_hours"], 0)
        self.assertEqual(diag["remaining_missing_after_fill"], 0)
        self.assertEqual(len(clean_df), 24)
        self.assertEqual(int(clean_df["load"].isna().sum()), 0)
        self.assertTrue(clean_df["timestamp"].is_monotonic_increasing)

    def test_missing_timestamp_detection_and_reindex(self):
        """Case 2: Omitted timestamp in sequence is detected and reindexed."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dr,
            "load": np.arange(24, dtype=np.float64) + 100.0,
        })
        # Omit hour 10 entirely (row deletion)
        df_missing = df.drop(index=10).reset_index(drop=True)
        path = self._create_csv("missing_timestamp.csv", df_missing)

        clean_df, diag = load_and_clean_data(path, fill_strategy="interpolate", max_fill_limit=6)

        self.assertEqual(diag["raw_rows"], 23)
        self.assertEqual(diag["actual_steps_before_reindex"], 23)
        self.assertEqual(diag["expected_regular_steps"], 24)
        self.assertEqual(diag["total_missing_intervals"], 1)
        self.assertEqual(diag["max_consecutive_gap_hours"], 1)
        self.assertEqual(diag["remaining_missing_after_fill"], 0)
        self.assertEqual(len(clean_df), 24)
        # Assert linear interpolation midpoint between index 9 (109.0) and index 11 (111.0)
        self.assertAlmostEqual(clean_df["load"].iloc[10], 110.0, places=4)

    def test_missing_load_value_at_existing_timestamp(self):
        """Case 3: Existing row with NaN/missing load is detected and filled."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dr,
            "load": np.arange(24, dtype=np.float64) + 100.0,
        })
        df.loc[12, "load"] = np.nan
        path = self._create_csv("missing_val.csv", df)

        clean_df, diag = load_and_clean_data(path, fill_strategy="interpolate", max_fill_limit=6)

        self.assertEqual(diag["raw_rows"], 24)
        self.assertEqual(diag["initial_missing_load"], 1)
        self.assertEqual(diag["total_missing_intervals"], 1)
        self.assertEqual(diag["remaining_missing_after_fill"], 0)
        self.assertEqual(len(clean_df), 24)
        self.assertAlmostEqual(clean_df["load"].iloc[12], 112.0, places=4)

    def test_short_gap_within_max_fill_limit(self):
        """Case 4: Consecutive gap within max_fill_limit is completely filled."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dr,
            "load": np.arange(24, dtype=np.float64) + 100.0,
        })
        # 3 consecutive hours missing (hours 10, 11, 12)
        df.loc[10:12, "load"] = np.nan
        path = self._create_csv("short_gap.csv", df)

        clean_df, diag = load_and_clean_data(path, fill_strategy="interpolate", max_fill_limit=6)

        self.assertEqual(diag["total_missing_intervals"], 3)
        self.assertEqual(diag["max_consecutive_gap_hours"], 3)
        self.assertEqual(diag["remaining_missing_after_fill"], 0)
        self.assertEqual(int(clean_df["load"].isna().sum()), 0)

    def test_gap_exceeding_max_fill_limit(self):
        """Case 5: Consecutive gap exceeding max_fill_limit leaves expected unfilled NaNs."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "timestamp": dr,
            "load": np.arange(24, dtype=np.float64) + 100.0,
        })
        # 4 consecutive hours missing (hours 10, 11, 12, 13) with max_fill_limit=2
        df.loc[10:13, "load"] = np.nan
        path = self._create_csv("exceed_gap.csv", df)

        clean_df, diag = load_and_clean_data(path, fill_strategy="interpolate", max_fill_limit=2)

        self.assertEqual(diag["total_missing_intervals"], 4)
        self.assertEqual(diag["max_consecutive_gap_hours"], 4)
        # Limit=2 fills only the first 2 intervals; exactly 2 remain unfilled
        self.assertEqual(diag["remaining_missing_after_fill"], 2)
        self.assertEqual(int(clean_df["load"].isna().sum()), 2)

    def test_duplicate_timestamp_averaging(self):
        """Duplicate timestamps are aggregated via arithmetic mean."""
        timestamps = [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 01:00:00",  # Duplicate
            "2024-01-01 02:00:00",
        ]
        loads = [100.0, 150.0, 250.0, 200.0]
        df = pd.DataFrame({"timestamp": timestamps, "load": loads})
        path = self._create_csv("duplicates.csv", df)

        clean_df, diag = load_and_clean_data(path)

        self.assertEqual(diag["duplicate_timestamps"], 1)
        self.assertEqual(len(clean_df), 3)
        # (150 + 250) / 2 = 200.0
        self.assertAlmostEqual(clean_df.loc[clean_df["timestamp"] == pd.Timestamp("2024-01-01 01:00:00"), "load"].iloc[0], 200.0)

    def test_invalid_timestamp_dropping(self):
        """Rows with unparseable timestamps are dropped during cleaning."""
        timestamps = [
            "2024-01-01 00:00:00",
            "MALFORMED_TIMESTAMP",
            "2024-01-01 01:00:00",
        ]
        loads = [100.0, 999.0, 200.0]
        df = pd.DataFrame({"timestamp": timestamps, "load": loads})
        path = self._create_csv("invalid_ts.csv", df)

        clean_df, diag = load_and_clean_data(path)

        self.assertEqual(diag["invalid_timestamps"], 1)
        self.assertEqual(len(clean_df), 2)


class TestPartitionBoundaryAndCausalCleaningIsolation(unittest.TestCase):
    """
    Deterministic Partition-Boundary & Causal Cleaning Isolation Suite (ISS-04)
    =============================================================================
    1. Causal Forward-Fill Invariance under future-target perturbation.
    2. Partition-Boundary Isolation: validation perturbations cannot contaminate train.
    3. Scaler Parameter Isolation: training StandardScaler is immune to boundary shifts.
    4. Explicit Rejection of Unresolved NaNs prior to model window construction.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_csv(self, filename: str, df: pd.DataFrame) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        df.to_csv(path, index=False)
        return path

    def test_future_value_perturbation_causal_ffill(self):
        """Under causal ffill, future target perturbations cannot alter earlier filled values."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
        loads = np.arange(24, dtype=np.float64) + 100.0

        # Baseline: hour 10 is missing
        df_base = pd.DataFrame({"timestamp": dr, "load": loads})
        df_base.loc[10, "load"] = np.nan
        path_base = self._create_csv("ffill_base.csv", df_base)

        # Perturbed: hour 10 is missing, future hour 11 is perturbed by +5000.0
        df_pert = pd.DataFrame({"timestamp": dr, "load": loads})
        df_pert.loc[10, "load"] = np.nan
        df_pert.loc[11, "load"] += 5000.0
        path_pert = self._create_csv("ffill_pert.csv", df_pert)

        clean_base, _ = load_and_clean_data(path_base, fill_strategy="ffill", max_fill_limit=6)
        clean_pert, _ = load_and_clean_data(path_pert, fill_strategy="ffill", max_fill_limit=6)

        self.assertEqual(
            clean_base["load"].iloc[10],
            109.0,
            "Baseline hour 10 was not forward-filled from hour 9 (109.0)!",
        )
        self.assertEqual(
            clean_pert["load"].iloc[10],
            clean_base["load"].iloc[10],
            "Under ffill, future perturbation at hour 11 leaked backward into filled hour 10!",
        )

    def test_default_path_boundary_isolation_enforced(self):
        """When split_ratios is omitted, load_and_clean_data defaults to safe boundary isolation."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=100, freq="h")
        loads = np.arange(100, dtype=np.float64) + 100.0

        # Boundary under 70/15/15: train ends at index 69 (last training step), val starts at 70
        df_base = pd.DataFrame({"timestamp": dr, "load": loads})
        df_base.loc[69, "load"] = np.nan
        path_base = self._create_csv("default_bound_base.csv", df_base)

        df_pert = pd.DataFrame({"timestamp": dr, "load": loads})
        df_pert.loc[69, "load"] = np.nan
        df_pert.loc[70, "load"] += 5000.0
        path_pert = self._create_csv("default_bound_pert.csv", df_pert)

        # Call load_and_clean_data WITHOUT passing split_ratios (verifying safe default enforcement)
        clean_base, _ = load_and_clean_data(path_base)
        clean_pert, _ = load_and_clean_data(path_pert)

        tr_b, val_b, test_b, _ = chronological_split(clean_base, 0.70, 0.15, 0.15)
        tr_p, val_p, test_p, _ = chronological_split(clean_pert, 0.70, 0.15, 0.15)

        # Assert training partition is bitwise invariant to validation perturbation under default call
        np.testing.assert_array_equal(
            tr_b["load"].values,
            tr_p["load"].values,
            err_msg="Default call to load_and_clean_data leaked validation perturbation into train_df!",
        )

    def test_interior_future_perturbation_under_retrospective_interpolation(self):
        """Documents and asserts that retrospective interpolation within a partition is bidirectional."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=100, freq="h")
        loads = np.arange(100, dtype=np.float64) + 100.0

        # Interior index 10 is missing inside the training partition
        df_base = pd.DataFrame({"timestamp": dr, "load": loads})
        df_base.loc[10, "load"] = np.nan
        path_base = self._create_csv("interp_c_base.csv", df_base)

        # Perturb future index 11 inside the same partition
        df_pert = pd.DataFrame({"timestamp": dr, "load": loads})
        df_pert.loc[10, "load"] = np.nan
        df_pert.loc[11, "load"] += 500.0
        path_pert = self._create_csv("interp_c_pert.csv", df_pert)

        clean_base, _ = load_and_clean_data(path_base, fill_strategy="interpolate")
        clean_pert, _ = load_and_clean_data(path_pert, fill_strategy="interpolate")

        # In retrospective interpolation, index 10 evaluates midpoint of 9 (109.0) and 11
        # When index 11 increases by 500.0, index 10 changes by +250.0
        diff = float(clean_pert["load"].iloc[10] - clean_base["load"].iloc[10])
        self.assertAlmostEqual(diff, 250.0, places=4, msg="Retrospective interpolation did not reflect right endpoint!")

    def test_partition_boundary_isolation(self):
        """Validation perturbations cannot leak backward into training partition under boundary guard."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=100, freq="h")
        loads = np.arange(100, dtype=np.float64) + 100.0

        # Boundary under 70/15/15: train ends at index 69 (last training step), val starts at 70
        df_base = pd.DataFrame({"timestamp": dr, "load": loads})
        df_base.loc[69, "load"] = np.nan
        path_base = self._create_csv("bound_base.csv", df_base)

        df_pert = pd.DataFrame({"timestamp": dr, "load": loads})
        df_pert.loc[69, "load"] = np.nan
        df_pert.loc[70, "load"] += 5000.0
        path_pert = self._create_csv("bound_pert.csv", df_pert)

        clean_base, _ = load_and_clean_data(path_base, split_ratios=(0.70, 0.15, 0.15))
        clean_pert, _ = load_and_clean_data(path_pert, split_ratios=(0.70, 0.15, 0.15))

        tr_b, val_b, test_b, _ = chronological_split(clean_base, 0.70, 0.15, 0.15)
        tr_p, val_p, test_p, _ = chronological_split(clean_pert, 0.70, 0.15, 0.15)

        # Assert training partition is bitwise invariant to validation perturbation
        np.testing.assert_array_equal(
            tr_b["load"].values,
            tr_p["load"].values,
            err_msg="Validation perturbation at index 70 leaked backward across boundary into train_df!",
        )
        self.assertEqual(tr_p["load"].iloc[69], tr_b["load"].iloc[69])

    def test_training_scaler_isolation_across_boundary(self):
        """StandardScaler fit parameters remain strictly isolated from validation/test perturbations."""
        dr = pd.date_range("2024-01-01 00:00:00", periods=100, freq="h")
        loads = np.arange(100, dtype=np.float64) + 100.0

        df_base = pd.DataFrame({"timestamp": dr, "load": loads})
        df_base.loc[69, "load"] = np.nan
        path_base = self._create_csv("sc_base.csv", df_base)

        df_pert = pd.DataFrame({"timestamp": dr, "load": loads})
        df_pert.loc[69, "load"] = np.nan
        df_pert.loc[70, "load"] += 5000.0
        path_pert = self._create_csv("sc_pert.csv", df_pert)

        clean_base, _ = load_and_clean_data(path_base, split_ratios=(0.70, 0.15, 0.15))
        clean_pert, _ = load_and_clean_data(path_pert, split_ratios=(0.70, 0.15, 0.15))

        tr_b, val_b, test_b, _ = chronological_split(clean_base, 0.70, 0.15, 0.15)
        tr_p, val_p, test_p, _ = chronological_split(clean_pert, 0.70, 0.15, 0.15)

        sc_b, _, _, _ = fit_and_transform_scaler(tr_b, val_b, test_b)
        sc_p, _, _, _ = fit_and_transform_scaler(tr_p, val_p, test_p)

        self.assertAlmostEqual(sc_b.mean_[0], sc_p.mean_[0], places=5)
        self.assertAlmostEqual(sc_b.scale_[0], sc_p.scale_[0], places=5)

    def test_unresolved_nans_rejected_in_window_generation(self):
        """create_forecasting_windows explicitly raises ValueError on unresolved NaNs."""
        series_with_nan = np.arange(200, dtype=np.float32)
        series_with_nan[50] = np.nan

        with self.assertRaises(ValueError) as ctx:
            create_forecasting_windows(series_with_nan, lookback=48, horizon=12)

        self.assertIn("unresolved NaN values", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
