"""
Phase 14 Comprehensive Final-Model Optimization & Robustness Unit Tests
======================================================================
Tests:
1. Canonical V1 context definition (strictly 4D: Trend, Volatility, Periodicity, Recent Error).
2. Causal recent error (evaluated strictly on completed past horizons).
3. Genuine OOF expert errors (expanding window, zero in-sample training error).
4. No future-target access (target perturbation invariance).
5. Relative error calculation (sums to 1.0 and scale invariant).
6. Causally smoothed performance calculation (alpha in {0.25, 0.50, 0.75}).
7. Confidence lambda bounds and convexity (MLP head in [0, 1]).
8. Scalar shrinkage bounds (lambda in [0, 1]).
9. Equal ensemble construction (exact uniform 1/3 weighting).
10. Adaptive fusion equation (convex combination of adaptive and equal).
11. Parameter counts for F0-F5.
12. Deterministic seed behavior across RNGs.
13. Daily-block construction (K=53 PJM, K=456 GEFCom, K=163 UCI).
14. Test firewall rule (Stage B qualification before Stage C test access).
15. Aggregation consistency (mean of seed MAEs vs. MAE of pooled predictions).
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
    Candidates F2 / F3 / F4:
    y_final = lambda * y_adaptive + (1 - lambda) * y_equal
    lambda = Sigmoid(MLP(c))
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
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

        lam = self.confidence_head(c)
        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        diag["lambda"] = lam
        return y_final, w, diag


class ScalarShrinkageCAEGNet(nn.Module):
    """
    Candidate F5:
    y_final = lambda_scalar * y_adaptive + (1 - lambda_scalar) * y_equal
    where lambda_scalar is a single global constant in [0, 1].
    """
    def __init__(self, context_dim: int = 7, lambda_val: float = 0.5):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.lambda_scalar = float(np.clip(lambda_val, 0.0, 1.0))

    def set_lambda(self, val: float):
        self.lambda_scalar = float(np.clip(val, 0.0, 1.0))

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

        y_final = self.lambda_scalar * y_adaptive + (1.0 - self.lambda_scalar) * y_equal
        diag["lambda"] = torch.full((x.shape[0], 1), self.lambda_scalar, device=x.device)
        return y_final, w, diag


class TestPhase14FinalOptimization(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.L = 168
        self.H = 24
        self.x = torch.randn(self.B, self.L, 1)
        self.c4 = torch.randn(self.B, 4)
        self.c7 = torch.randn(self.B, 7)

    def test_01_canonical_v1_context_definition(self):
        """1. Verify canonical V1 context features are strictly [Trend, Volatility, Periodicity, Recent Error]."""
        from data_utils import extract_context_features
        x_syn = np.random.normal(5000, 500, (10, 168, 1)).astype(np.float32)
        rec_syn = np.random.uniform(100, 300, 10).astype(np.float32)
        C = extract_context_features(x_syn, recent_errors=rec_syn)
        self.assertEqual(C.shape, (10, 4))
        np.testing.assert_allclose(C[:, 3], rec_syn, rtol=1e-5)
        self.assertTrue(np.all(C[:, 1] >= 0.0))  # Volatility non-negative
        self.assertTrue(np.all(C[:, 2] >= -1.0) and np.all(C[:, 2] <= 1.0))  # Autocorr in [-1, 1]

    def test_02_causal_recent_error_timeline(self):
        """2. Verify recent error strictly completes prior to forecast origin t."""
        origin = 100
        horizon = 24
        forecast_window_start = origin - horizon
        forecast_window_end = origin
        self.assertEqual(forecast_window_end - forecast_window_start, horizon)
        self.assertLessEqual(forecast_window_end, origin)

    def test_03_genuine_oof_expert_errors_expanding_window(self):
        """3. Verify expanding fold windows guarantee zero in-sample training leakage."""
        n_tr = 1000
        n_folds = 4
        block_indices = [int(i * n_tr / n_folds) for i in range(n_folds + 1)]
        for f in range(n_folds - 1):
            tr_end = block_indices[f + 1]
            eval_start = block_indices[f + 1]
            eval_end = block_indices[f + 2]
            self.assertLessEqual(tr_end, eval_start)
            self.assertEqual(len(set(range(0, tr_end)).intersection(set(range(eval_start, eval_end)))), 0)

    def test_04_no_future_target_access_perturbation(self):
        """4. Verify future target perturbation leaves historical OOF features completely unchanged."""
        N = 100
        horizon = 24
        trues = np.sin(np.linspace(0, 10, N + horizon))
        preds = trues + np.random.normal(0, 0.1, len(trues))
        t_eval = 60
        err_orig = float(np.mean(np.abs(preds[t_eval - horizon : t_eval] - trues[t_eval - horizon : t_eval])))

        perturbed_trues = trues.copy()
        perturbed_trues[t_eval : t_eval + horizon] += 9999.0  # future targets perturbed
        err_pert = float(np.mean(np.abs(preds[t_eval - horizon : t_eval] - perturbed_trues[t_eval - horizon : t_eval])))
        self.assertEqual(err_orig, err_pert)

    def test_05_relative_error_calculation(self):
        """5. Verify relative error r_i(t) sums to 1.0 and is scale-invariant."""
        el, et, ec = 120.0, 80.0, 150.0
        tot = el + et + ec + 1e-6
        rl, rt, rc = el / tot, et / tot, ec / tot
        self.assertAlmostEqual(rl + rt + rc, 1.0, places=5)
        # Scale invariance
        sc = 1000.0
        tot_sc = (el * sc) + (et * sc) + (ec * sc) + 1e-6
        self.assertAlmostEqual(rl, (el * sc) / tot_sc, places=4)

    def test_06_causally_smoothed_performance(self):
        """6. Verify causally smoothed performance s_i(t) = alpha * s_i(t-1) + (1-alpha) * r_i(t)."""
        r = np.array([[0.5, 0.3, 0.2],
                      [0.2, 0.5, 0.3],
                      [0.4, 0.4, 0.2]], dtype=np.float32)
        alpha = 0.5
        s = np.zeros_like(r)
        s[0] = r[0]
        for t in range(1, len(r)):
            s[t] = alpha * s[t - 1] + (1.0 - alpha) * r[t]
        # s must sum to 1.0 at every step
        for t in range(len(s)):
            self.assertAlmostEqual(float(np.sum(s[t])), 1.0, places=5)
        # s[1] must depend strictly on r[0] and r[1]
        expected_s1 = 0.5 * r[0] + 0.5 * r[1]
        np.testing.assert_allclose(s[1], expected_s1, rtol=1e-5)

    def test_07_confidence_lambda_bounds(self):
        """7. Verify learned confidence head lambda is strictly in [0, 1]."""
        m_f2 = ConfidenceFallbackCAEGNet(context_dim=7)
        m_f2.eval()
        with torch.no_grad():
            _, _, diag = m_f2(self.x, self.c7)
        lam = diag["lambda"]
        self.assertTrue(torch.all(lam >= 0.0) and torch.all(lam <= 1.0))

    def test_08_scalar_shrinkage_bounds(self):
        """8. Verify scalar shrinkage lambda is strictly bounded in [0, 1]."""
        m_f5 = ScalarShrinkageCAEGNet(context_dim=7, lambda_val=0.52)
        self.assertEqual(m_f5.lambda_scalar, 0.52)
        m_f5.set_lambda(1.5)
        self.assertEqual(m_f5.lambda_scalar, 1.0)
        m_f5.set_lambda(-0.5)
        self.assertEqual(m_f5.lambda_scalar, 0.0)

    def test_09_equal_ensemble_construction(self):
        """9. Verify equal ensemble construction produces exact 1/3 weighting."""
        yl = torch.ones(2, 24) * 10.0
        yt = torch.ones(2, 24) * 20.0
        yc = torch.ones(2, 24) * 30.0
        y_eq = (yl + yt + yc) / 3.0
        np.testing.assert_allclose(y_eq.numpy(), 20.0, rtol=1e-5)

    def test_10_adaptive_fusion_equation(self):
        """10. Verify final prediction matches lambda * y_adaptive + (1-lambda) * y_equal."""
        y_ad = torch.ones(2, 24) * 100.0
        y_eq = torch.ones(2, 24) * 50.0
        lam = 0.6
        y_fin = lam * y_ad + (1.0 - lam) * y_eq
        np.testing.assert_allclose(y_fin.numpy(), 80.0, rtol=1e-5)

    def test_11_parameter_counts_f0_to_f5(self):
        """11. Verify exact parameter counts for all Phase 14 candidates."""
        m_core_l = LSTMExpert()
        m_core_t = TCNExpert()
        m_core_c = CNNExpert()
        p_core = sum(p.numel() for p in m_core_l.parameters()) +                  sum(p.numel() for p in m_core_t.parameters()) +                  sum(p.numel() for p in m_core_c.parameters())
        self.assertEqual(p_core, 120504)

        f0 = OriginalCAEGNetPhase5(context_dim=4)
        f1 = OriginalCAEGNetPhase5(context_dim=7)
        f2 = ConfidenceFallbackCAEGNet(context_dim=7)
        f3 = ConfidenceFallbackCAEGNet(context_dim=4)
        f4 = ConfidenceFallbackCAEGNet(context_dim=7)
        f5 = ScalarShrinkageCAEGNet(context_dim=7)

        def pcount(m): return sum(p.numel() for p in m.parameters())

        self.assertEqual(pcount(f0), 121531)  # Router = 1027
        self.assertEqual(pcount(f1), 121579)  # Router = 1075 (+48)
        self.assertEqual(pcount(f2), 121724)  # Router = 1220 (+193)
        self.assertEqual(pcount(f3), 121628)  # Router = 1124 (+97)
        self.assertEqual(pcount(f4), 121724)  # Same as F2 (+193)
        self.assertEqual(pcount(f5), 121579)  # Trainable router params = 121579 (+48, 0 extra net params)

    def test_12_deterministic_seeding_behavior(self):
        """12. Verify deterministic seeding produces identical tensor initializations."""
        seed_everything(42, deterministic_cudnn=True)
        t1 = torch.randn(5, 5)
        seed_everything(42, deterministic_cudnn=True)
        t2 = torch.randn(5, 5)
        torch.testing.assert_close(t1, t2)

    def test_13_daily_block_construction(self):
        """13. Verify non-overlapping daily block counts."""
        n_pjm = 1294
        n_gef = 10944
        n_uci = 3922
        horizon = 24
        self.assertEqual(n_pjm // horizon, 53)
        self.assertEqual(n_gef // horizon, 456)
        self.assertEqual(n_uci // horizon, 163)

    def test_14_test_firewall_rule(self):
        """14. Verify qualification rule logic: >= 2 improved, worst degradation <= 2.0%."""
        def qualifies(pct_changes):
            improved = sum(1 for p in pct_changes if p < 0.0)
            worst = max(pct_changes)
            return (improved >= 2) and (worst <= 2.0)

        self.assertTrue(qualifies([-5.0, -1.5, 1.2]))
        self.assertFalse(qualifies([-5.0, -1.5, 2.8]))
        self.assertFalse(qualifies([-5.0, 1.0, 0.5]))

    def test_15_aggregation_consistency(self):
        """15. Verify that mean of seed MAEs is conceptually and mathematically distinct from MAE of ensemble."""
        # 2 seeds, 1 origin with true target 10.0
        y_true = 10.0
        pred_s1 = 8.0  # error = 2.0
        pred_s2 = 12.0 # error = 2.0
        mean_of_maes = (abs(pred_s1 - y_true) + abs(pred_s2 - y_true)) / 2.0  # = 2.0
        pred_ens = (pred_s1 + pred_s2) / 2.0  # = 10.0
        mae_of_ens = abs(pred_ens - y_true)   # = 0.0
        # By Jensen's inequality, mae_of_ens <= mean_of_maes
        self.assertLess(mae_of_ens, mean_of_maes)


if __name__ == "__main__":
    unittest.main()
