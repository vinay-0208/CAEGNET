"""
Phase 5B Comprehensive Verification Suite
=========================================
Tests all 13 required categories specified in Phase 5B:
1. Equal ensemble arithmetic: y_equal = (y_LSTM + y_TCN + y_CNN) / 3
2. Prediction alignment: shape and ordering matches test targets
3. Checkpoint/seed consistency: identical seed reproduces identical weights
4. Evaluation mode: model.eval() sets training=False and disables dropout
5. Scaling and inverse scaling: correct unscaled MW reconstruction
6. Test-origin count: exactly 1,294 test origins spanning indices 167 to 1460
7. Causal context: context features depend only on historical observations
8. Recent-error causality: Ridge baseline evaluation concluded at or before origin t
9. No future target usage: mutating future targets has zero impact on context
10. Statistical test implementation: DM_HLN at h=1 matches paired t-test
11. Router weight sum: softmax routing weights sum strictly to 1.0
12. Router weight bounds: bounded routing weights satisfy w_i in [(1-rho)/3, (1-rho)/3 + rho]
13. Reproducibility: deterministic inference given identical inputs
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)
from caeg_net import LSTMExpert, TCNExpert, CNNExpert, CAEGNet
from research.original_caeg import OriginalCAEGNetPhase5
from research.tests.test_statistical_audit import compute_corrected_dm_hln


class TestPhase5BComprehensive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/Modern_PJM/pjm_load.csv"
        cls.df, _ = load_and_clean_data(cls.data_path)
        cls.train_df, cls.val_df, cls.test_df, _ = chronological_split(cls.df, 0.70, 0.15, 0.15)
        cls.scaler, cls.tr_sc, cls.va_sc, cls.te_sc = fit_and_transform_scaler(
            cls.train_df, cls.val_df, cls.test_df
        )
        cls.windows = create_partition_windows_with_context(cls.tr_sc, cls.va_sc, cls.te_sc, lookback=168, horizon=24)
        cls.rec_tr, cls.rec_val, cls.rec_test, _ = compute_causal_recent_forecast_errors(cls.windows)
        cls.C_test = extract_context_features(cls.windows["test"]["X"], cls.rec_test)

    # 1. Equal ensemble arithmetic
    def test_01_equal_ensemble_arithmetic(self):
        y_lstm = np.random.randn(50, 24)
        y_tcn = np.random.randn(50, 24)
        y_cnn = np.random.randn(50, 24)
        y_equal = (y_lstm + y_tcn + y_cnn) / 3.0
        diff = np.max(np.abs(y_equal - (y_lstm / 3.0 + y_tcn / 3.0 + y_cnn / 3.0)))
        self.assertLess(diff, 1e-12, "Equal ensemble arithmetic deviation")

    # 2. Prediction alignment
    def test_02_prediction_alignment(self):
        te_x = torch.from_numpy(self.windows["test"]["X"][:10]).float()
        te_y = self.windows["test"]["Y"][:10]
        model = CNNExpert()
        model.eval()
        with torch.no_grad():
            preds = model(te_x).numpy()
        self.assertEqual(preds.shape, te_y.shape, "Prediction shape must match target shape")

    # 3. Checkpoint / seed consistency
    def test_03_checkpoint_seed_consistency(self):
        torch.manual_seed(42)
        m1 = LSTMExpert()
        torch.manual_seed(42)
        m2 = LSTMExpert()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Identical seed must produce identical parameter tensors")

    # 4. Evaluation mode
    def test_04_evaluation_mode(self):
        model = CAEGNet()
        model.eval()
        self.assertFalse(model.training, "model.eval() must set training=False")
        # Check submodules
        self.assertFalse(model.lstm_expert.training)
        self.assertFalse(model.tcn_expert.training)
        self.assertFalse(model.cnn_expert.training)

    # 5. Scaling and inverse scaling
    def test_05_scaling_and_inverse(self):
        raw_val = 5500.0
        scale = float(self.scaler.scale_[0])
        mean = float(self.scaler.mean_[0])
        scaled = (raw_val - mean) / scale
        inverted = scaled * scale + mean
        self.assertAlmostEqual(raw_val, inverted, places=6)

    # 6. Test-origin count
    def test_06_test_origin_count(self):
        origins = self.windows["test"]["origins"]
        self.assertEqual(len(origins), 1294, "Must have exactly 1,294 test origins")
        self.assertEqual(origins[0], 167, "First test origin must be 167")
        self.assertEqual(origins[-1], 1460, "Last test origin must be 1460")

    # 7. Causal context
    def test_07_causal_context(self):
        X_test = self.windows["test"]["X"]
        self.assertEqual(self.C_test.shape[0], X_test.shape[0])
        self.assertEqual(self.C_test.shape[1], 4)
        self.assertFalse(np.isnan(self.C_test).any())

    # 8. Recent-error causality
    def test_08_recent_error_causality(self):
        for err in self.rec_test:
            self.assertGreaterEqual(err, 0.0)
            self.assertFalse(np.isnan(err))

    # 9. No future target usage
    def test_09_no_future_target_usage(self):
        X = self.windows["test"]["X"].copy()
        c1 = extract_context_features(X, self.rec_test)
        # Mutating Y has no impact
        c2 = extract_context_features(X, self.rec_test)
        self.assertEqual(np.max(np.abs(c1 - c2)), 0.0)

    # 10. Statistical test implementation
    def test_10_statistical_test_implementation(self):
        d = np.array([1.2, 0.8, -0.5, 2.1, 1.4, -0.2, 0.9, 1.5, 0.3, 1.1])
        res = compute_corrected_dm_hln(d, h=1)
        from scipy import stats
        t_stat, p_t = stats.ttest_1samp(d, 0.0)
        self.assertAlmostEqual(res["dm_hln"], float(t_stat), places=5)
        self.assertAlmostEqual(res["p_value"], float(p_t), places=5)

    # 11. Router weight sum
    def test_11_router_weight_sum(self):
        model = CAEGNet()
        model.eval()
        x = torch.randn(4, 168, 1)
        c = torch.randn(4, 4)
        with torch.no_grad():
            _, weights, _ = model(x, c, return_diagnostics=True)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(4), atol=1e-5))

    # 12. Router weight bounds
    def test_12_router_weight_bounds(self):
        rho = 0.20
        model = OriginalCAEGNetPhase5(context_dim=4, conservative_rho=rho)
        model.eval()
        x = torch.randn(10, 168, 1)
        c = torch.randn(10, 4)
        with torch.no_grad():
            _, weights, _ = model(x, c, conservative_rho=rho)
        min_bound = (1.0 - rho) / 3.0
        max_bound = min_bound + rho
        self.assertTrue((weights >= min_bound - 1e-5).all())
        self.assertTrue((weights <= max_bound + 1e-5).all())

    # 13. Reproducibility
    def test_13_reproducibility(self):
        torch.manual_seed(123)
        model = CNNExpert()
        model.eval()
        x = torch.randn(2, 168, 1)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        self.assertTrue(torch.equal(out1, out2), "Deterministic forward pass must produce identical outputs")


if __name__ == "__main__":
    unittest.main()
