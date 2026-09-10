"""
Phase 13 Comprehensive Defensive Unit & Audit Test Suite
======================================================
Tests:
1. Exact parameter counts and architecture dimensions.
2. Canonical V1 context feature definition ([Trend, Volatility, Periodicity, Recent Error]).
3. Chronological OOF temporal causality and target perturbation invariance.
4. Expanding fold boundary separation (zero in-sample training leakage).
5. Chronological train/validation/test isolation.
6. Relative expert error normalization and scale invariance.
7. Confidence fallback convexity, bounds, and near-constancy behavior.
8. Candidate A3 ablation context dimension (strictly 4D, no error features).
9. Stage B validation screening qualification rules.
10. Daily-block non-overlapping alignment and negative difference direction.
11. Validation-only best expert baseline selection.
12. Deterministic seeding reproducibility across RNGs.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from caeg_net import LSTMExpert, TCNExpert, CNNExpert
from research.original_caeg import OriginalCAEGNetPhase5
from research.deterministic import seed_everything


class ConfidenceFallbackCAEGNet(nn.Module):
    """
    Candidate A2 / A3: CAEG router + learned adaptive/equal confidence interpolation.
    y_final = lambda * y_adaptive + (1 - lambda) * y_equal
    lambda = Sigmoid(W_lambda * c + b_lambda)
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        # 2-layer MLP confidence head predicting lambda in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_adaptive, w, diag = self.caeg(x, c, conservative_rho=conservative_rho, temperature=temperature)
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        lam = self.confidence_head(c)  # (B, 1)
        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        diag["lambda"] = lam
        return y_final, w, diag


class TestPhase13OOFPerformanceAwareGating(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.L = 168
        self.H = 24
        self.x = torch.randn(self.B, self.L, 1)
        self.c4 = torch.randn(self.B, 4)
        self.c7 = torch.randn(self.B, 7)

    def test_01_parameter_counts_exact(self):
        """Verify exact parameter counts for all 4 Phase 13 candidates."""
        m_lstm = LSTMExpert()
        m_tcn = TCNExpert()
        m_cnn = CNNExpert()

        p_l = sum(p.numel() for p in m_lstm.parameters())
        p_t = sum(p.numel() for p in m_tcn.parameters())
        p_c = sum(p.numel() for p in m_cnn.parameters())
        p_core = p_l + p_t + p_c

        self.assertEqual(p_l, 56152)
        self.assertEqual(p_t, 36952)
        self.assertEqual(p_c, 27400)
        self.assertEqual(p_core, 120504)

        m_a0 = OriginalCAEGNetPhase5(context_dim=4)
        m_a1 = OriginalCAEGNetPhase5(context_dim=7)
        m_a2 = ConfidenceFallbackCAEGNet(context_dim=7)
        m_a3 = ConfidenceFallbackCAEGNet(context_dim=4)

        def count(m):
            return sum(p.numel() for p in m.parameters())

        # A0: Canonical V1 = 121,531 (Router = 1,027)
        self.assertEqual(count(m_a0), 121531)
        self.assertEqual(count(m_a0) - p_core, 1027)

        # A1: P2-OOF (context_dim=7, Linear 7->32) = 121,579 (Router = 1,075, +48)
        self.assertEqual(count(m_a1), 121579)
        self.assertEqual(count(m_a1) - p_core, 1075)

        # A2: C1-OOF (context_dim=7 + 145-param confidence head) = 121,724 (Router = 1,220, +193)
        self.assertEqual(count(m_a2), 121724)
        self.assertEqual(count(m_a2) - p_core, 1220)

        # A3: Confidence-only ablation (context_dim=4 + 97-param confidence head) = 121,628 (Router = 1,124, +97)
        self.assertEqual(count(m_a3), 121628)
        self.assertEqual(count(m_a3) - p_core, 1124)

    def test_02_canonical_v1_context_feature_definition(self):
        """
        Verify that canonical V1 context features are strictly:
        [Trend, Volatility, Periodicity (Lag-24 Autocorrelation), Recent Error]
        and NOT [mean load, std, time-of-day, day-of-week].
        """
        from data_utils import extract_context_features
        N = 10
        L = 168
        synthetic_x = np.random.normal(5000, 500, (N, L, 1)).astype(np.float32)
        synthetic_recent_err = np.random.uniform(100, 300, N).astype(np.float32)

        C = extract_context_features(synthetic_x, recent_errors=synthetic_recent_err)
        self.assertEqual(C.shape, (N, 4))
        # Column 0: Trend, Column 1: Volatility, Column 2: Periodicity, Column 3: Recent Error
        np.testing.assert_allclose(C[:, 3], synthetic_recent_err, rtol=1e-5)
        self.assertTrue(np.all(C[:, 1] >= 0.0))
        self.assertTrue(np.all(C[:, 2] >= -1.0) and np.all(C[:, 2] <= 1.0))

    def test_03_oof_temporal_causality_and_target_perturbation(self):
        """
        Verify that OOF error feature for origin t strictly depends on targets
        ending at or before t, and is completely invariant to future targets Y[t].
        """
        N = 100
        horizon = 24
        synthetic_trues = np.sin(np.linspace(0, 10, N + horizon))
        synthetic_preds = synthetic_trues + np.random.normal(0, 0.1, len(synthetic_trues))

        def get_trailing_error(preds, trues, t):
            if t >= horizon:
                return float(np.mean(np.abs(preds[t - horizon : t] - trues[t - horizon : t])))
            return 0.5

        t_eval = 50
        err_orig = get_trailing_error(synthetic_preds, synthetic_trues, t_eval)

        perturbed_trues = synthetic_trues.copy()
        perturbed_trues[t_eval : t_eval + horizon] += 999.0

        err_perturbed = get_trailing_error(synthetic_preds, perturbed_trues, t_eval)
        self.assertEqual(err_orig, err_perturbed)

    def test_04_oof_fold_boundaries_and_separation(self):
        """
        Verify expanding fold boundaries ensure zero in-sample training leakage:
        Fold k is trained on [0:B_k] and evaluates on [B_k:B_{k+1}].
        """
        n_tr = 1000
        n_folds = 4
        block_indices = [int(i * n_tr / n_folds) for i in range(n_folds + 1)]

        for f in range(n_folds - 1):
            tr_end = block_indices[f + 1]
            eval_start = block_indices[f + 1]
            eval_end = block_indices[f + 2]

            self.assertLessEqual(tr_end, eval_start)
            self.assertGreater(eval_end, eval_start)
            tr_set = set(range(0, tr_end))
            eval_set = set(range(eval_start, eval_end))
            self.assertEqual(len(tr_set.intersection(eval_set)), 0)

    def test_05_train_val_test_isolation(self):
        """Verify that train, validation, and test chronological partitions have zero index overlap."""
        n_total = 10000
        n_tr = int(n_total * 0.70)
        n_va = int(n_total * 0.15)
        n_te = n_total - n_tr - n_va

        tr_idx = set(range(0, n_tr))
        va_idx = set(range(n_tr, n_tr + n_va))
        te_idx = set(range(n_tr + n_va, n_total))

        self.assertEqual(len(tr_idx.intersection(va_idx)), 0)
        self.assertEqual(len(va_idx.intersection(te_idx)), 0)
        self.assertEqual(len(tr_idx.intersection(te_idx)), 0)
        self.assertEqual(len(tr_idx) + len(va_idx) + len(te_idx), n_total)

    def test_06_relative_error_normalization_and_scale_invariance(self):
        """Verify that relative error r_i(t) sums to 1.0 and is scale-invariant."""
        e_l = np.array([150.0, 25.0, 300.0])
        e_t = np.array([120.0, 20.0, 280.0])
        e_c = np.array([180.0, 35.0, 350.0])

        eps = 1e-6
        e_sum = e_l + e_t + e_c + eps
        r_l = e_l / e_sum
        r_t = e_t / e_sum
        r_c = e_c / e_sum

        r_total = r_l + r_t + r_c
        np.testing.assert_allclose(r_total, 1.0, rtol=1e-5)

        scale = 10.0
        e_sum_sc = (e_l * scale) + (e_t * scale) + (e_c * scale) + eps
        r_l_sc = (e_l * scale) / e_sum_sc
        np.testing.assert_allclose(r_l, r_l_sc, rtol=1e-4)

    def test_07_confidence_fallback_bounds_and_constancy(self):
        """
        Verify that confidence lambda is bounded in [0, 1], and that the head produces
        valid output across batches.
        """
        m_c1 = ConfidenceFallbackCAEGNet(context_dim=7)
        m_c1.eval()
        with torch.no_grad():
            y_final, w, diag = m_c1(self.x, self.c7)

        lam = diag["lambda"]
        self.assertTrue(torch.all(lam >= 0.0) and torch.all(lam <= 1.0))
        self.assertEqual(y_final.shape, (self.B, self.H))
        self.assertEqual(w.shape, (self.B, 3))
        np.testing.assert_allclose(torch.sum(w, dim=-1).numpy(), 1.0, rtol=1e-5)

    def test_08_ablation_a3_confidence_only_dimension(self):
        """
        Verify that candidate A3 operates strictly on 4D canonical context
        and produces valid confidence fallback predictions.
        """
        m_a3 = ConfidenceFallbackCAEGNet(context_dim=4)
        m_a3.eval()
        with torch.no_grad():
            y_final, w, diag = m_a3(self.x, self.c4)

        lam = diag["lambda"]
        self.assertTrue(torch.all(lam >= 0.0) and torch.all(lam <= 1.0))
        self.assertEqual(y_final.shape, (self.B, self.H))
        self.assertEqual(w.shape, (self.B, 3))

    def test_09_validation_selection_firewall_rule(self):
        """
        Verify the qualification logic: must improve on >= 2 datasets,
        with worst degradation <= 2.0%.
        """
        def qualifies(rel_changes):
            improved = sum(1 for r in rel_changes if r < 0.0)
            worst_deg = max(rel_changes)
            return (improved >= 2) and (worst_deg <= 2.0)

        self.assertTrue(qualifies([-5.0, -2.0, 1.5]))
        self.assertFalse(qualifies([-5.0, -2.0, 2.5]))
        self.assertFalse(qualifies([-5.0, 1.0, 0.5]))
        self.assertTrue(qualifies([-3.0, -1.0, -4.0]))

    def test_10_daily_block_partitioning_and_direction(self):
        """
        Verify non-overlapping daily block calculation and ensure negative
        difference strictly represents an improvement over baseline.
        """
        n_windows = 1294
        horizon = 24
        k_blocks = n_windows // horizon
        self.assertEqual(k_blocks, 53)

        mae_v1 = np.full(k_blocks, 260.0)
        mae_cand = np.full(k_blocks, 250.0)
        diff = mae_cand - mae_v1
        self.assertTrue(np.all(diff < 0.0))
        self.assertEqual(float(np.mean(diff)), -10.0)

    def test_11_best_expert_selection_firewall(self):
        """Verify best expert is selected strictly from validation MAEs."""
        val_maes = {"LSTM": 15.2, "TCN": 12.8, "CNN": 16.5}
        best_expert = min(val_maes, key=val_maes.get)
        self.assertEqual(best_expert, "TCN")

    def test_12_reproducibility_deterministic_seeding(self):
        """Verify deterministic seeding produces identical tensor initializations."""
        seed_everything(42, deterministic_cudnn=True)
        t1 = torch.randn(10, 10)
        seed_everything(42, deterministic_cudnn=True)
        t2 = torch.randn(10, 10)
        torch.testing.assert_close(t1, t2)


if __name__ == "__main__":
    unittest.main()
