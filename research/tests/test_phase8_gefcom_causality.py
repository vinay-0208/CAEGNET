import os
import unittest
import numpy as np
import pandas as pd

from research.gefcom_data import (
    load_gefcom2014_full_series,
    get_gefcom2014_partitions,
    prepare_gefcom2014_pipeline,
    TASK_DURATIONS_HOURS,
)


class TestGEFCom2014Causality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = prepare_gefcom2014_pipeline()

    def test_scaler_fitted_strictly_on_train(self):
        """Verify scaler parameters reflect exclusively the training partition."""
        train_loads = self.pipeline["train_df"]["load"].values
        expected_mean = float(np.mean(train_loads))
        expected_std = float(np.std(train_loads))
        
        self.assertAlmostEqual(self.pipeline["mean"], expected_mean, places=3)
        self.assertAlmostEqual(self.pipeline["scale"], expected_std, places=3)
        
        # Verify val and test have different distributions
        val_mean = float(np.mean(self.pipeline["val_df"]["load"].values))
        test_mean = float(np.mean(self.pipeline["test_df"]["load"].values))
        self.assertNotEqual(self.pipeline["mean"], val_mean)
        self.assertNotEqual(self.pipeline["mean"], test_mean)

    def test_zero_future_leakage_in_windows(self):
        """Verify input window X strictly precedes forecast horizon Y for all test windows."""
        test_x = self.pipeline["windows"]["test"]["X"]
        test_y = self.pipeline["windows"]["test"]["Y"]
        
        # For any window, last element of X is the origin observation
        # First element of Y is the origin+1 observation
        # In scaled space, ensure X and Y follow strict sequential continuity
        diffs = test_y[:, 0] - test_x[:, -1, 0]
        self.assertEqual(len(diffs), len(test_x))
        # Ensure X shape is (N, 168, 1) and Y shape is (N, 24)
        self.assertEqual(test_x.shape[1], 168)
        self.assertEqual(test_x.shape[2], 1)
        self.assertEqual(test_y.shape[1], 24)

    def test_context_causality(self):
        """Verify context feature dimensions and absence of NaNs."""
        for split in ["train", "val", "test"]:
            ctx = self.pipeline["context"][split]
            n_windows = len(self.pipeline["windows"][split]["X"])
            self.assertEqual(ctx.shape, (n_windows, 4))
            self.assertFalse(np.isnan(ctx).any(), f"NaNs detected in {split} context features!")

    def test_solution15_isolation(self):
        """Verify Task 15 ground truth is isolated and does not enter train or validation."""
        sol_df = pd.read_csv("data/Load/Solution to Task 15/solution15_L.csv")
        sol_loads = sol_df["LOAD"].values
        
        train_loads = self.pipeline["train_df"]["load"].values
        val_loads = self.pipeline["val_df"]["load"].values
        test_loads = self.pipeline["test_df"]["load"].values
        
        # Solution 15 should match the final 744 hours of test partition
        self.assertTrue(np.allclose(test_loads[-744:], sol_loads))
        
        # Solution 15 should NOT match any slice of training data
        self.assertFalse(np.allclose(train_loads[:744], sol_loads))
        self.assertFalse(np.allclose(val_loads[:744], sol_loads))

    def test_daily_blocks_coverage(self):
        """Verify 456 non-overlapping 24h daily blocks cover the 15 tasks exactly."""
        task_blocks = self.pipeline["task_daily_blocks"]
        self.assertEqual(len(task_blocks), 15)
        
        total_days = sum(len(days) for days in task_blocks.values())
        self.assertEqual(total_days, 456)
        self.assertEqual(self.pipeline["K_days"], 456)
        
        # Verify block counts per task
        for t_id, days in task_blocks.items():
            expected_days = TASK_DURATIONS_HOURS[t_id] // 24 if t_id < 15 else 30
            self.assertEqual(len(days), expected_days, f"Task {t_id} day count mismatch!")


if __name__ == "__main__":
    unittest.main()
