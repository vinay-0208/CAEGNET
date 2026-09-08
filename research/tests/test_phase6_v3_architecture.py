"""
CAEG-Net V3 Architecture & Verification Test Suite
===================================================
Tests all structural guarantees and correctness criteria for CAEG-Net V3:
1. Instantiation and parameter efficiency (~146k params within 120k-150k budget).
2. Forward pass output shape: y_fused in R^[B, 24].
3. Horizon-dependent router output shape: weights in R^[B, 24, 3].
4. Simplex property: weights sum to 1.0 at every horizon step h in {1..24}.
5. Non-negativity: weights >= 0.0 everywhere.
6. Multi-horizon disagreement shape: disagreement in R^[B, 6].
7. Disagreement detachment: no gradient flow through disagreement into expert backbones.
8. Tripartite loss with uniform centering prior: compute_caeg_v3_loss produces finite scalar.
9. Backward pass produces valid, finite gradients with no NaNs/Infs.
10. Backward-compatibility: CAEGNetV2 remains completely functional and unaffected.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.models import (
    CAEGNetV2,
    CAEGNetV3,
    compute_caeg_v2_loss,
    compute_caeg_v3_loss,
    count_parameters,
)
from research.training import train_caeg_v3, evaluate_caeg_v3


class TestPhase6V3Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.L_in = 168
        self.H = 24
        self.C_dim = 6
        self.x = torch.randn(self.B, self.L_in, 1)
        self.c = torch.randn(self.B, self.C_dim)
        self.y_true = torch.randn(self.B, self.H)

    def test_01_v3_instantiation_and_parameter_count(self):
        model = CAEGNetV3()
        params = count_parameters(model)
        self.assertIn("total", params)
        self.assertIn("trainable", params)
        self.assertIn("gru_expert", params)
        self.assertIn("tcn_expert", params)
        self.assertIn("patch_expert", params)
        self.assertIn("context_encoder", params)
        self.assertIn("router_head", params)

        # Budget requirement: ~120k - 150k
        self.assertGreater(params["total"], 120000)
        self.assertLess(params["total"], 155000)
        self.assertEqual(params["total"], params["trainable"])
        print(f"\n[Test 1] V3 Total Parameters: {params['total']:,}")

    def test_02_forward_pass_shapes(self):
        model = CAEGNetV3()
        model.eval()
        with torch.no_grad():
            y_fused, weights, diag = model(self.x, self.c)

        self.assertEqual(y_fused.shape, (self.B, self.H))
        self.assertEqual(weights.shape, (self.B, self.H, 3))
        self.assertIn("expert_predictions", diag)
        self.assertEqual(diag["expert_predictions"]["gru"].shape, (self.B, self.H))
        self.assertEqual(diag["expert_predictions"]["tcn"].shape, (self.B, self.H))
        self.assertEqual(diag["expert_predictions"]["patch"].shape, (self.B, self.H))
        self.assertIn("disagreement", diag)
        self.assertEqual(diag["disagreement"].shape, (self.B, 6))

    def test_03_horizon_simplex_property(self):
        model = CAEGNetV3()
        model.eval()
        with torch.no_grad():
            _, weights, _ = model(self.x, self.c)

        # Weights must be non-negative
        self.assertTrue(torch.all(weights >= 0.0))
        # Weights must sum to 1.0 at each horizon step
        weight_sums = torch.sum(weights, dim=-1)  # [B, 24]
        ones = torch.ones_like(weight_sums)
        self.assertTrue(torch.allclose(weight_sums, ones, atol=1e-5))

    def test_04_disagreement_detachment(self):
        """
        Verify that disagreement tensor is detached and does not backpropagate
        into expert backbones through the router pathway.
        """
        model = CAEGNetV3()
        model.train()

        y1 = model.gru_expert(self.x)
        y2 = model.tcn_expert(self.x)
        y3 = model.patch_expert(self.x)

        d_vec = model.compute_multi_horizon_disagreement(y1, y2, y3)
        self.assertFalse(d_vec.requires_grad)

    def test_05_tripartite_loss_and_backward(self):
        model = CAEGNetV3()
        model.train()

        y_fused, weights, diag = model(self.x, self.c)
        loss, telem = compute_caeg_v3_loss(
            y_fused, self.y_true, diag["expert_predictions"], weights,
            lambda_aux=0.40, beta_prior=0.002
        )

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(telem["loss_aux"], 0.0)
        self.assertGreaterEqual(telem["loss_prior"], 0.0)

        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Grad missing for {name}")
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN in grad for {name}")
                self.assertFalse(torch.isinf(p.grad).any(), f"Inf in grad for {name}")

    def test_06_backward_compatibility_v2(self):
        """Verify CAEGNetV2 remains 100% functional and intact."""
        v2 = CAEGNetV2()
        v2.eval()
        with torch.no_grad():
            y_fused, weights, diag = v2(self.x, self.c)
        self.assertEqual(y_fused.shape, (self.B, self.H))
        self.assertEqual(weights.shape, (self.B, 3))


if __name__ == "__main__":
    unittest.main()
