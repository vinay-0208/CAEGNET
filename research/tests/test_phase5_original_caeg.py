"""
Unit Tests for Phase 5 Original CAEG-Net Architecture & Extensions
==================================================================
Validates:
1. Exact parameter accounting for original LSTM + TCN + CNN architecture (121,531 params).
2. Strict equivalence of conservative routing at rho=0 to the Static Equal Ensemble.
3. Equivalence of conservative routing at rho=1 to unconstrained soft gating.
4. Inter-expert disagreement computation shape and autograd detachment.
5. Auxiliary expert loss calculation.
6. Horizon-dependent routing weight shapes and simplex constraints.
7. Learned static ensemble parameter accounting and gradient isolation.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from research.original_caeg import (
    OriginalCAEGNetPhase5,
    LearnedStaticEnsembleV1,
    compute_phase5_loss,
)
from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    ContextFeatureEncoder,
    ContextGatingNetwork,
    CAEGNet as CAEGNetV1,
    count_parameters,
)


class TestPhase5OriginalCAEG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.batch_size = 4
        cls.x = torch.randn(cls.batch_size, 168, 1)
        cls.c = torch.randn(cls.batch_size, 4)

    def test_parameter_counts_exact_v1(self):
        """Verify exact parameter counts of the canonical V1 architecture."""
        m_v1 = CAEGNetV1(context_dim=4, horizon_dependent=False)
        counts = count_parameters(m_v1)
        self.assertEqual(counts["total_params"], 121531, f"Expected 121,531 total params, got {counts['total_params']}")
        self.assertEqual(counts["submodules"]["lstm_expert"], 56152)
        self.assertEqual(counts["submodules"]["tcn_expert"], 36952)
        self.assertEqual(counts["submodules"]["cnn_expert"], 27400)
        self.assertEqual(counts["submodules"]["context_encoder"], 384)
        self.assertEqual(counts["submodules"]["gating_network"], 643)

        # Also verify Phase 5 model without extensions matches V1 counts
        m_p5 = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, horizon_dependent=False)
        p5_total = sum(p.numel() for p in m_p5.parameters())
        self.assertEqual(p5_total, 121531, f"Phase 5 base model expected 121,531 params, got {p5_total}")

    def test_conservative_rho_zero_equivalence(self):
        """Verify that rho=0 produces predictions identical to the equal ensemble within 1e-6."""
        model = OriginalCAEGNetPhase5(context_dim=4, conservative_rho=0.0)
        model.eval()
        with torch.no_grad():
            y_pred, weights, diag = model(self.x, self.c, conservative_rho=0.0)
            
            y_lstm = model.lstm_expert(self.x)
            y_tcn = model.tcn_expert(self.x)
            y_cnn = model.cnn_expert(self.x)
            y_equal = (y_lstm + y_tcn + y_cnn) / 3.0

            diff = torch.abs(y_pred - y_equal).max().item()
            self.assertLess(diff, 1e-5, f"rho=0 differs from equal ensemble by {diff}")

            expected_w = torch.full_like(weights, 1.0 / 3.0)
            w_diff = torch.abs(weights - expected_w).max().item()
            self.assertLess(w_diff, 1e-5, f"Weights at rho=0 differ from 1/3 by {w_diff}")

    def test_conservative_rho_one_equivalence(self):
        """Verify that rho=1 matches unconstrained routing."""
        model = OriginalCAEGNetPhase5(context_dim=4, conservative_rho=1.0)
        model.eval()
        with torch.no_grad():
            y_pred, weights, diag = model(self.x, self.c, conservative_rho=1.0)
            q_weights = diag["q_weights"]
            diff = torch.abs(weights - q_weights).max().item()
            self.assertLess(diff, 1e-5, f"rho=1 differs from q_weights by {diff}")

    def test_disagreement_computation_and_detachment(self):
        """Verify disagreement shape and autograd detachment."""
        model = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=True)
        model.train()
        y_pred, weights, diag = model(self.x, self.c)
        
        disagreement = diag["disagreement"]
        self.assertIsNotNone(disagreement)
        self.assertEqual(disagreement.shape, (self.batch_size, 3))
        self.assertFalse(disagreement.requires_grad, "Disagreement tensor should be detached from autograd!")

    def test_auxiliary_loss_computation(self):
        """Verify tripartite loss computation with auxiliary expert loss."""
        model = OriginalCAEGNetPhase5(context_dim=4)
        model.train()
        y_pred, weights, diag = model(self.x, self.c)
        y_true = torch.randn_like(y_pred)

        loss, telemetry = compute_phase5_loss(
            y_pred=y_pred,
            y_true=y_true,
            expert_preds=diag["expert_predictions"],
            weights=weights,
            lambda_aux=0.10,
            beta_entropy=0.001,
            lambda_kl=0.01,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(telemetry["loss_aux"], 0.0)
        self.assertGreater(telemetry["loss_entropy"], 0.0)

    def test_horizon_dependent_shapes(self):
        """Verify horizon-dependent routing output shape and simplex sum=1.0 per horizon."""
        model = OriginalCAEGNetPhase5(context_dim=4, horizon_dependent=True)
        model.eval()
        with torch.no_grad():
            y_pred, weights, _ = model(self.x, self.c)
            self.assertEqual(weights.shape, (self.batch_size, 24, 3))
            self.assertEqual(y_pred.shape, (self.batch_size, 24))
            sums = weights.sum(dim=-1)  # [B, 24]
            diff = torch.abs(sums - 1.0).max().item()
            self.assertLess(diff, 1e-5)

    def test_learned_static_ensemble(self):
        """Verify learned static ensemble has 3 trainable params and frozen backbones."""
        lstm = LSTMExpert()
        tcn = TCNExpert()
        cnn = CNNExpert()
        ens = LearnedStaticEnsembleV1(lstm, tcn, cnn)
        trainable = sum(p.numel() for p in ens.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ens.parameters())
        self.assertEqual(trainable, 3)
        self.assertEqual(total, 56152 + 36952 + 27400 + 3)


if __name__ == "__main__":
    unittest.main()
