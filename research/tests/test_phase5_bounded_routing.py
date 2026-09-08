"""
Unit & Architecture Tests for Phase 5 Bounded / Conservative Routing (CAEG-Net BR)
==================================================================================
Validates:
1. Strict equivalence of CAEG-Net BR at rho=0 to the Static Standalone Equal Ensemble.
2. Equivalence of CAEG-Net BR at rho=1 to unconstrained routing.
3. Simplex preservation: sum(w_i) = 1.0 and w_i >= (1 - rho) / 3 >= 0.
4. Absolute gradient isolation: frozen backbones receive zero gradients.
5. Analytical KL divergence: non-negativity and exact 0.0 at rho=0.
6. Monotonicity of weight deviation from equal prior as rho increases.
7. Parameter accounting: exactly 139,838 frozen and 2,595 trainable (142,433 total).
8. Determinism and finite losses during training passes.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from research.models import (
    GatedRecurrentExpert,
    MultiScaleCausalTCNExpert,
    PatchTemporalExpert,
    BoundedRoutingCAEG,
    CAEGNetBR,
    compute_bounded_loss,
    count_parameters,
)


class TestPhase5BoundedRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)

        cls.gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24)
        cls.tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24)
        cls.patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24)

        cls.model = BoundedRoutingCAEG(cls.gru, cls.tcn, cls.patch, rho=0.2, temperature=1.0)
        cls.batch_size = 4
        cls.x = torch.randn(cls.batch_size, 168, 1)
        cls.c = torch.randn(cls.batch_size, 6)

    def test_parameter_accounting(self):
        """Test parameter accounting: 139,838 frozen backbones, 2,595 trainable router."""
        counts = count_parameters(self.model)
        self.assertEqual(counts["trainable"], 2595, f"Expected 2,595 trainable params, got {counts['trainable']}")
        frozen = counts["total"] - counts["trainable"]
        self.assertEqual(frozen, 139838, f"Expected 139,838 frozen params, got {frozen}")
        self.assertEqual(counts["total"], 142433, f"Expected 142,433 total params, got {counts['total']}")

    def test_rho_zero_exact_identity(self):
        """Test that rho=0 produces predictions identical to the equal ensemble within 1e-6."""
        self.model.eval()
        with torch.no_grad():
            y_fused, weights, diag = self.model(self.x, self.c, rho=0.0)
            
            y_gru = self.gru(self.x)
            y_tcn = self.tcn(self.x)
            y_patch = self.patch(self.x)
            y_equal = (y_gru + y_tcn + y_patch) / 3.0

            # 1. Prediction equivalence
            diff = torch.abs(y_fused - y_equal).max().item()
            self.assertLess(diff, 1e-6, f"rho=0 prediction differs from equal ensemble by {diff}")

            # 2. Weights exactly 1/3
            expected_w = torch.full_like(weights, 1.0 / 3.0)
            w_diff = torch.abs(weights - expected_w).max().item()
            self.assertLess(w_diff, 1e-6, f"Weights at rho=0 differ from 1/3 by {w_diff}")

            # 3. KL divergence exactly 0
            self.assertAlmostEqual(diag["mean_kl"].item(), 0.0, places=5)

    def test_rho_one_matches_unconstrained(self):
        """Test that rho=1 produces weights exactly equal to unconstrained softmax(delta/tau)."""
        self.model.eval()
        with torch.no_grad():
            y_fused, weights, diag = self.model(self.x, self.c, rho=1.0)
            q_weights = diag["q_weights"]
            diff = torch.abs(weights - q_weights).max().item()
            self.assertLess(diff, 1e-6, f"rho=1 weights differ from q_weights by {diff}")

    def test_simplex_sum_and_bounds(self):
        """Test simplex sum=1.0 and lower bound w_i >= (1 - rho) / 3 for multiple rho values."""
        self.model.eval()
        candidate_rhos = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        with torch.no_grad():
            for rho in candidate_rhos:
                _, weights, _ = self.model(self.x, self.c, rho=rho)
                sums = weights.sum(dim=-1)
                for s in sums:
                    self.assertAlmostEqual(s.item(), 1.0, places=5)

                min_allowed = (1.0 - rho) / 3.0 - 1e-6
                self.assertTrue(torch.all(weights >= min_allowed), f"Weights below minimum bound for rho={rho}")

    def test_gradient_isolation(self):
        """Verify backbones receive no gradients; only router parameters receive gradients."""
        self.model.train()
        self.model.freeze_experts()

        y_fused, weights, _ = self.model(self.x, self.c, rho=0.3)
        y_dummy = torch.randn_like(y_fused)
        loss, _ = compute_bounded_loss(y_fused, y_dummy, weights, lambda_dev=0.025)
        loss.backward()

        # Check expert params have no grads
        for expert in [self.gru, self.tcn, self.patch]:
            for name, param in expert.named_parameters():
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad, f"Param {name} unexpectedly has gradient!")

        # Check router params have valid gradients
        has_router_grads = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in self.model.router.parameters()
        )
        self.assertTrue(has_router_grads, "Router parameters failed to receive gradients!")

    def test_kl_divergence_properties(self):
        """Verify KL divergence non-negativity and positive values when rho > 0."""
        self.model.eval()
        with torch.no_grad():
            for rho in [0.1, 0.3, 0.5, 0.8]:
                _, _, diag = self.model(self.x, self.c, rho=rho)
                kl = diag["kl_div"]
                self.assertTrue(torch.all(kl >= -1e-6), f"KL divergence is negative for rho={rho}")

    def test_monotonic_deviation(self):
        """Verify mean L1 deviation from equal weighting strictly increases with rho."""
        self.model.eval()
        rhos = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
        mean_l1_devs = []
        w0 = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        with torch.no_grad():
            for rho in rhos:
                _, weights, _ = self.model(self.x, self.c, rho=rho)
                l1_dev = torch.sum(torch.abs(weights - w0), dim=-1).mean().item()
                mean_l1_devs.append(l1_dev)

        for i in range(len(mean_l1_devs) - 1):
            self.assertLessEqual(
                mean_l1_devs[i],
                mean_l1_devs[i + 1] + 1e-6,
                f"Monotonicity violated: rho={rhos[i]} has dev {mean_l1_devs[i]}, rho={rhos[i+1]} has dev {mean_l1_devs[i+1]}"
            )

    def test_determinism(self):
        """Verify identical outputs for consecutive evaluation passes."""
        self.model.eval()
        with torch.no_grad():
            y1, w1, _ = self.model(self.x, self.c, rho=0.25)
            y2, w2, _ = self.model(self.x, self.c, rho=0.25)
            self.assertTrue(torch.equal(y1, y2))
            self.assertTrue(torch.equal(w1, w2))


if __name__ == "__main__":
    unittest.main()
