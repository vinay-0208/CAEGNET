"""
Unit tests verifying dashboard cache integrity and Prediction/Forecast key mappings.
"""

import os
import unittest
import numpy as np


class TestDashboardPredictionCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.cache_path = os.path.join(cls.repo_root, "dashboard", "assets", "data", "research_multiseed_cache.npz")
        cls.context_path = os.path.join(cls.repo_root, "dashboard", "assets", "data", "pjm_test_context.npz")
        
        cls.assertTrue(cls, os.path.exists(cls.cache_path), f"Cache file not found at {cls.cache_path}")
        cls.assertTrue(cls, os.path.exists(cls.context_path), f"Context file not found at {cls.context_path}")
        
        cls.cache_data = np.load(cls.cache_path)
        cls.context_data = np.load(cls.context_path)

    def test_cache_contains_ground_truth(self):
        """Verify ground truth key exists, is named y_test_true_raw, and has valid shape."""
        self.assertIn("y_test_true_raw", self.cache_data.files)
        y_true = self.cache_data["y_test_true_raw"]
        self.assertEqual(y_true.shape, (1294, 24))
        self.assertFalse(np.isnan(y_true).any(), "Ground truth array contains NaNs")

    def test_cache_contains_canonical_v1_predictions(self):
        """Verify Canonical V1 predictions exist for all 5 evaluation seeds."""
        seeds = [42, 123, 999, 2024, 3407]
        for s in seeds:
            pred_key = f"pred_CAEG_Net_V1_Canonical_seed_{s}"
            self.assertIn(pred_key, self.cache_data.files, f"Missing prediction key: {pred_key}")
            pred_arr = self.cache_data[pred_key]
            self.assertEqual(pred_arr.shape, (1294, 24))
            self.assertFalse(np.isnan(pred_arr).any())

    def test_cache_contains_canonical_v1_routing_weights(self):
        """Verify routing weights exist for all 5 seeds, shape (1294, 3), and form valid simplex."""
        seeds = [42, 123, 999, 2024, 3407]
        for s in seeds:
            w_key = f"weights_CAEG_Net_V1_Canonical_seed_{s}"
            self.assertIn(w_key, self.cache_data.files, f"Missing weights key: {w_key}")
            weights = self.cache_data[w_key]
            self.assertEqual(weights.shape, (1294, 3))
            self.assertTrue(np.all(weights >= 0.0), f"Negative weights found in {w_key}")
            # Each row should sum to 1.0
            row_sums = np.sum(weights, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg=f"Weights in {w_key} do not sum to 1.0")

    def test_standalone_experts_absent_without_fabrication(self):
        """Verify standalone expert keys do NOT exist in research cache (confirming omission requirement)."""
        absent_keys = [
            "lstm_seed_42", "tcn_seed_42", "cnn_seed_42", "static_seed_42",
            "y_lstm", "y_tcn", "y_cnn", "y_static"
        ]
        for k in absent_keys:
            self.assertNotIn(k, self.cache_data.files, f"Key {k} unexpectedly present; must not be assumed")

    def test_context_file_structure(self):
        """Verify context file has X, scaler_mean, and scaler_scale with matched instances."""
        self.assertIn("X", self.context_data.files)
        self.assertIn("scaler_mean", self.context_data.files)
        self.assertIn("scaler_scale", self.context_data.files)
        
        X = self.context_data["X"]
        self.assertEqual(X.shape, (1294, 168, 1))
        self.assertEqual(len(X), len(self.cache_data["y_test_true_raw"]))

    def test_preset_window_indices_within_bounds(self):
        """Verify all curated instance presets (721, 1185, 493, 476, 0) are strictly valid."""
        max_windows = len(self.cache_data["y_test_true_raw"]) - 1
        presets = [721, 1185, 493, 476, 0]
        for idx in presets:
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, max_windows)

    def test_prediction_error_computations(self):
        """Verify window MAE and RMSE computations produce valid finite floats."""
        y_true = self.cache_data["y_test_true_raw"][721]
        y_pred = self.cache_data["pred_CAEG_Net_V1_Canonical_seed_42"][721]
        
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        max_err = float(np.max(np.abs(y_true - y_pred)))
        
        self.assertTrue(np.isfinite(mae))
        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(max_err))
        self.assertGreater(mae, 0)
        self.assertGreater(rmse, 0)


if __name__ == "__main__":
    unittest.main()
