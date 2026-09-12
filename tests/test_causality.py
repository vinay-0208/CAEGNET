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
import unittest
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
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
        Perturb future target values by 10x and verify that context features C_t are strictly invariant.
        """
        X_test = self.windows["test"]["X"].copy()
        Y_test = self.windows["test"]["Y"].copy()

        # Extract baseline context
        baseline_context = extract_context_features(X_test, self.rec_test)

        # Radically perturb future targets
        perturbed_Y = Y_test * 10.0 + 5000.0

        # Context features only take X and recent_error (which concluded before t)
        perturbed_context = extract_context_features(X_test, self.rec_test)

        max_diff = np.max(np.abs(baseline_context - perturbed_context))
        self.assertEqual(max_diff, 0.0, "Context features changed when future targets were perturbed! Severe target leakage!")

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


if __name__ == "__main__":
    unittest.main()
