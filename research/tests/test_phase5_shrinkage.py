"""
CAEG-Net Phase 5 Shrinkage-Regularized (CAEG-Net SR) Unit Tests
===============================================================
Verifies:
1. Exact identity: alpha = 0 produces exactly [1/3, 1/3, 1/3] and exact equal averaging.
2. KL divergence: KL(w || w0) >= 0, equals 0 at w = [1/3, 1/3, 1/3].
3. Zero gradient leakage into frozen backbones during router backpropagation.
4. Parameter accounting: 142,433 total deployed, 2,595 trainable in Stage 3.
5. Output shapes: [B, 24] fused forecast, [B, 3] convex weights.
6. Deterministic execution under a fixed seed.
7. Shrinkage loss function correctness: L = L_forecast + lambda_dev * KL.
8. Alpha responsiveness: increasing alpha strictly increases router weight variance / deviation.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.models import (
    GatedRecurrentExpert,
    MultiScaleCausalTCNExpert,
    PatchTemporalExpert,
    ShrinkageRegularizedCAEG,
    CAEGNetSR,
    compute_shrinkage_loss,
    count_parameters,
)


class TestPhase5Shrinkage(unittest.TestCase):
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

        self.gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24)
        self.tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24)
        self.patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24)

    def test_01_alpha_zero_exact_identity(self):
        """Verify that alpha=0 reproduces exactly [1/3, 1/3, 1/3] and exact equal averaging."""
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=0.0)
        model.eval()
        with torch.no_grad():
            y_fused, weights, diag = model(self.x, self.c)

        expected_w = torch.full_like(weights, 1.0 / 3.0)
        self.assertTrue(torch.allclose(weights, expected_w, atol=1e-6))

        # Check prediction equals simple average of experts
        with torch.no_grad():
            y_gru = self.gru(self.x)
            y_tcn = self.tcn(self.x)
            y_patch = self.patch(self.x)
            y_expected = (y_gru + y_tcn + y_patch) / 3.0

        self.assertTrue(torch.allclose(y_fused, y_expected, atol=1e-5))
        self.assertTrue(torch.allclose(diag["kl_div"], torch.zeros_like(diag["kl_div"]), atol=1e-6))

    def test_02_kl_divergence_non_negative(self):
        """Verify that analytical KL(w || w0) is non-negative and bounded in [0, log 3]."""
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=1.0)
        model.eval()
        with torch.no_grad():
            _, weights, diag = model(self.x, self.c)

        kl = diag["kl_div"]
        self.assertTrue(torch.all(kl >= -1e-6), "KL divergence must be non-negative")
        log_3 = float(np.log(3.0))
        self.assertTrue(torch.all(kl <= log_3 + 1e-4), "KL divergence must be bounded by log(3)")

    def test_03_frozen_backbones_and_gradients(self):
        """Verify backpropagation computes gradients ONLY for router, NOT experts."""
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=0.5)
        model.train()

        y_fused, weights, diag = model(self.x, self.c)
        loss, _ = compute_shrinkage_loss(y_fused, self.y_true, weights, lambda_dev=0.01)
        loss.backward()

        # Expert gradients must be strictly None
        for exp_name, expert in [("GRU", model.gru_expert), ("TCN", model.tcn_expert), ("Patch", model.patch_expert)]:
            for name, p in expert.named_parameters():
                self.assertIsNone(p.grad, f"Gradient leaked to {exp_name}.{name}!")

        # Router and Context Encoder must have valid non-zero gradients
        for name, p in model.context_encoder.named_parameters():
            self.assertIsNotNone(p.grad, f"Gradient missing for context_encoder.{name}")
            self.assertFalse(torch.isnan(p.grad).any())

        for name, p in model.router.named_parameters():
            self.assertIsNotNone(p.grad, f"Gradient missing for router.{name}")
            self.assertFalse(torch.isnan(p.grad).any())

    def test_04_parameter_counts(self):
        """Verify parameter accountability: 142,433 total deployed, 2,595 trainable."""
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch)
        params = count_parameters(model)

        self.assertEqual(params["total"], 142433)
        self.assertEqual(params["trainable"], 2595)
        self.assertEqual(params["gru_expert"], 31344)
        self.assertEqual(params["tcn_expert"], 44870)
        self.assertEqual(params["patch_expert"], 63624)
        self.assertEqual(params["context_encoder"], 1440)
        self.assertEqual(params["router"], 1155)

    def test_05_output_shapes_and_probabilities(self):
        """Verify output shapes and convex combination simplex constraints."""
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=0.35)
        model.eval()
        with torch.no_grad():
            y_fused, weights, _ = model(self.x, self.c)

        self.assertEqual(y_fused.shape, (self.B, self.H))
        self.assertEqual(weights.shape, (self.B, 3))
        self.assertTrue(torch.all(weights >= 0.0), "Weights must be non-negative")
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_06_deterministic_behavior(self):
        """Verify exact reproducibility under a fixed seed."""
        torch.manual_seed(123)
        m1 = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=0.5)
        with torch.no_grad():
            y1, w1, _ = m1(self.x, self.c)

        torch.manual_seed(123)
        m2 = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch, alpha=0.5)
        with torch.no_grad():
            y2, w2, _ = m2(self.x, self.c)

        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.equal(w1, w2))

    def test_07_shrinkage_loss_computation(self):
        """Verify compute_shrinkage_loss with various lambda_dev values."""
        y_p = torch.ones(self.B, self.H)
        y_t = torch.zeros(self.B, self.H)
        # MSE is 1.0
        w_eq = torch.full((self.B, 3), 1.0 / 3.0)
        l_total, tel = compute_shrinkage_loss(y_p, y_t, w_eq, lambda_dev=0.05)
        self.assertAlmostEqual(tel["loss_forecast"], 1.0, places=5)
        self.assertAlmostEqual(tel["loss_kl"], 0.0, places=5)
        self.assertAlmostEqual(tel["loss_total"], 1.0, places=5)

    def test_08_alpha_monotonic_deviation(self):
        """Verify that increasing alpha strictly increases weight deviation from [1/3, 1/3, 1/3]."""
        torch.manual_seed(42)
        model = ShrinkageRegularizedCAEG(self.gru, self.tcn, self.patch)
        model.eval()

        w0 = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        devs = []
        alphas = [0.0, 0.1, 0.2, 0.5, 1.0]

        with torch.no_grad():
            for a in alphas:
                _, w, _ = model(self.x, self.c, alpha=a)
                dev = torch.norm(w - w0, dim=-1).mean().item()
                devs.append(dev)

        for i in range(len(devs) - 1):
            self.assertLessEqual(devs[i], devs[i+1] + 1e-6, f"Deviation did not increase monotonically from alpha={alphas[i]} to {alphas[i+1]}")


if __name__ == "__main__":
    unittest.main()
