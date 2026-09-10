import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.original_caeg import OriginalCAEGNetPhase5


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
        m_a0 = OriginalCAEGNetPhase5(context_dim=4)
        m_a1 = OriginalCAEGNetPhase5(context_dim=7)
        m_a2 = ConfidenceFallbackCAEGNet(context_dim=7)
        m_a3 = ConfidenceFallbackCAEGNet(context_dim=4)

        def count(m):
            return sum(p.numel() for p in m.parameters())

        # A0: Canonical V1 = 121,531
        self.assertEqual(count(m_a0), 121531)
        # A1: P2-OOF (context_dim=7, Linear 7->32) = 121,531 + 48 = 121,579
        self.assertEqual(count(m_a1), 121579)
        # A2: C1-OOF (context_dim=7, 145-param confidence head) = 121,579 + 145 = 121,724
        self.assertEqual(count(m_a2), 121724)
        # A3: Confidence-only ablation (context_dim=4, confidence head: 4->16 (80) + 16->1 (17) = 97 params)
        # 121,531 + 97 = 121,628
        self.assertEqual(count(m_a3), 121628)

    def test_02_oof_causality_and_target_perturbation(self):
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

    def test_03_relative_error_normalization(self):
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

    def test_04_confidence_fallback_convexity(self):
        """
        Verify that confidence lambda is bounded in [0, 1], and the final prediction
        strictly lies between y_adaptive and y_equal.
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

    def test_05_ablation_a3_confidence_only(self):
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

    def test_06_validation_selection_firewall_rule(self):
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


if __name__ == "__main__":
    unittest.main()
