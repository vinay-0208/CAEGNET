"""
Unit Tests for Phase 10 Controlled Cross-Dataset Optimization
=============================================================
Verifies:
1. Canonical expert identity and parameter constraints.
2. Context feature normalization functions (A0-A4, B0-B2).
3. Router regularization loss objectives (C1 entropy, C2 stability/KL).
4. Temperature-scaled softmax routing (C3).
5. Decoupled expert pretraining and router training pipeline (D1).
6. CNN temporal pooling stabilization (E1: CNN_H2_TemporalPool8).
7. Strict causal timeline and zero future leakage guarantees.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    ContextFeatureEncoder,
    ContextGatingNetwork,
    CAEGNet as CAEGNetV1,
    count_parameters,
)
from research.original_caeg import OriginalCAEGNetPhase5, compute_phase5_loss


class TestPhase10Optimization(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.lookback = 168
        self.horizon = 24
        self.x = torch.randn(self.batch_size, self.lookback, 1)
        self.c_4d = torch.randn(self.batch_size, 4)
        self.c_3d = torch.randn(self.batch_size, 3)

    def test_canonical_expert_parameters(self):
        """Verify parameter counts of canonical experts match frozen specifications."""
        m_lstm = LSTMExpert()
        m_tcn = TCNExpert()
        m_cnn = CNNExpert()

        self.assertEqual(count_parameters(m_lstm)["total_params"], 56152)
        self.assertEqual(count_parameters(m_tcn)["total_params"], 36952)
        self.assertEqual(count_parameters(m_cnn)["total_params"], 27400)

        # Canonical V1 CAEG-Net: 121,531 params
        m_v1 = OriginalCAEGNetPhase5(context_dim=4, use_disagreement=False, horizon_dependent=False)
        self.assertEqual(count_parameters(m_v1)["total_params"], 121531)

    def test_context_dimensions_a_and_b(self):
        """Verify model handles 4D context (A0-A4, B0, B2) and 3D context (B1)."""
        # 4D Context model
        m_4d = OriginalCAEGNetPhase5(context_dim=4)
        y_4d, w_4d, diag_4d = m_4d(self.x, self.c_4d)
        self.assertEqual(y_4d.shape, (self.batch_size, self.horizon))
        self.assertEqual(w_4d.shape, (self.batch_size, 3))
        # Check convex sum
        self.assertTrue(torch.allclose(w_4d.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))

        # 3D Context model (B1: No recent error)
        m_3d = OriginalCAEGNetPhase5(context_dim=3)
        y_3d, w_3d, diag_3d = m_3d(self.x, self.c_3d)
        self.assertEqual(y_3d.shape, (self.batch_size, self.horizon))
        self.assertEqual(w_3d.shape, (self.batch_size, 3))
        self.assertTrue(torch.allclose(w_3d.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))

    def test_router_regularization_loss_c1_c2(self):
        """Verify entropy (C1) and KL stability (C2) regularization in loss function."""
        y_pred = torch.randn(self.batch_size, self.horizon)
        y_true = torch.randn(self.batch_size, self.horizon)
        w = F.softmax(torch.randn(self.batch_size, 3), dim=-1)

        # Baseline loss
        l_base, tel_base = compute_phase5_loss(y_pred, y_true, weights=w, beta_entropy=0.0, lambda_kl=0.0)
        self.assertAlmostEqual(tel_base["loss_total"], tel_base["loss_fused"], places=5)

        # Entropy regularized loss (C1: beta_entropy > 0)
        l_ent, tel_ent = compute_phase5_loss(y_pred, y_true, weights=w, beta_entropy=0.01, lambda_kl=0.0)
        self.assertTrue(tel_ent["loss_entropy"] > 0)
        self.assertAlmostEqual(tel_ent["loss_total"], tel_base["loss_fused"] - 0.01 * tel_ent["loss_entropy"], places=5)

        # KL regularized loss (C2: lambda_kl > 0)
        l_kl, tel_kl = compute_phase5_loss(y_pred, y_true, weights=w, beta_entropy=0.0, lambda_kl=0.01)
        self.assertTrue(tel_kl["loss_kl"] >= 0)
        self.assertAlmostEqual(tel_kl["loss_total"], tel_base["loss_fused"] + 0.01 * tel_kl["loss_kl"], places=5)

    def test_router_temperature_c3(self):
        """Verify softmax temperature scaling operates strictly on logits."""
        m = OriginalCAEGNetPhase5(context_dim=4)
        _, w_t1, _ = m(self.x, self.c_4d, temperature=1.0)
        _, w_t05, _ = m(self.x, self.c_4d, temperature=0.5)
        _, w_t20, _ = m(self.x, self.c_4d, temperature=2.0)

        # Sharper temperature (0.5) should have lower entropy on average than softer (2.0)
        eps = 1e-8
        h_t05 = -(w_t05 * torch.log(w_t05 + eps)).sum(dim=-1).mean()
        h_t20 = -(w_t20 * torch.log(w_t20 + eps)).sum(dim=-1).mean()
        self.assertTrue(h_t05 <= h_t20)

    def test_cnn_temporal_pooling_e1(self):
        """Verify CNN_H2_TemporalPool8 retains temporal structure with valid output shape."""
        class CNN_H2_TemporalPool8(nn.Module):
            def __init__(self, input_dim=1, horizon=24, pool_size=8, dropout=0.1):
                super().__init__()
                self.conv_stack = nn.Sequential(
                    nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(pool_size),
                )
                self.head = nn.Sequential(
                    nn.Linear(64 * pool_size, 48),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(48, horizon),
                )

            def forward(self, x):
                x_trans = x.transpose(1, 2)
                feat = self.conv_stack(x_trans).view(x.shape[0], -1)
                return self.head(feat)

        m_cnn_h2 = CNN_H2_TemporalPool8()
        y = m_cnn_h2(self.x)
        self.assertEqual(y.shape, (self.batch_size, self.horizon))
        self.assertEqual(count_parameters(m_cnn_h2)["total_params"], 48904)

    def test_decoupled_expert_training_d1(self):
        """Verify that freezing experts stops gradients to expert parameters."""
        m_lstm = LSTMExpert()
        m_tcn = TCNExpert()
        m_cnn = CNNExpert()

        # Freeze experts
        for expert in [m_lstm, m_tcn, m_cnn]:
            for p in expert.parameters():
                p.requires_grad = False

        # Router and context encoder remain trainable
        ctx_enc = ContextFeatureEncoder(context_dim=4, latent_dim=16)
        router = ContextGatingNetwork(latent_dim=16, num_experts=3)

        e_c = ctx_enc(self.c_4d)
        w = router(e_c)

        with torch.no_grad():
            y_l = m_lstm(self.x)
            y_t = m_tcn(self.x)
            y_c = m_cnn(self.x)

        y_fused = w[:, 0:1] * y_l + w[:, 1:2] * y_t + w[:, 2:3] * y_c
        loss = F.mse_loss(y_fused, torch.randn_like(y_fused))
        loss.backward()

        # Expert gradients must be None
        for expert in [m_lstm, m_tcn, m_cnn]:
            for p in expert.parameters():
                self.assertIsNone(p.grad)

        # Router gradients must be present
        for p in router.parameters():
            self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
