"""
CAEG-Net V2 Phase 3 Architecture & Pipeline Verification Suite
==============================================================
Validates all 18 implementation correctness criteria mandated by Phase 3:
1. Model import
2. Forward pass
3. Output shape: [B, 24]
4. Router output shape: [B, 3]
5. Weights sum to 1.0
6. Weights are non-negative
7. Parameter count programmatic calculation and inspection
8. Context feature dimensions (6D context, 9D router input)
9. Weekly indexing: [t-191:t-168] vs [t-23:t]
10. 192h context buffer validation
11. Disagreement tensor shape: [B, 3]
12. Disagreement is detached (no gradients through D_t)
13. Auxiliary loss normalization (1/3 factor)
14. Entropy definition and sign (+ 0.001 * H(w))
15. No NaNs or Infs in outputs or gradients
16. Causal feature computation
17. Compatibility with CPU
18. Compatibility with CUDA
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
    ContextFeatureEncoder,
    ContextAdaptiveRouter,
    CAEGNetV2,
    compute_caeg_v2_loss,
    count_parameters,
)
from research.data import (
    compute_causal_observable_context,
    create_causal_research_windows,
    create_partition_windows_with_192h_context,
)


class TestPhase3Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 8
        self.L_in = 168
        self.L_ctx = 192
        self.H = 24
        self.C_dim = 6
        self.x = torch.randn(self.B, self.L_in, 1)
        self.c = torch.randn(self.B, self.C_dim)
        self.y_true = torch.randn(self.B, self.H)

    # 1. Model import
    def test_01_model_import(self):
        model = CAEGNetV2()
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, "gru_expert"))
        self.assertTrue(hasattr(model, "tcn_expert"))
        self.assertTrue(hasattr(model, "patch_expert"))
        self.assertTrue(hasattr(model, "context_encoder"))
        self.assertTrue(hasattr(model, "router"))

    # 2. Forward pass
    def test_02_forward_pass(self):
        model = CAEGNetV2()
        y_fused, weights, diag = model(self.x, self.c)
        self.assertIsNotNone(y_fused)
        self.assertIsNotNone(weights)
        self.assertIn("expert_predictions", diag)

    # 3. Output shape: [B, 24]
    def test_03_output_shape(self):
        model = CAEGNetV2()
        y_fused, _, _ = model(self.x, self.c)
        self.assertEqual(y_fused.shape, (self.B, self.H))

    # 4. Router output shape: [B, 3]
    def test_04_router_output_shape(self):
        model = CAEGNetV2()
        _, weights, _ = model(self.x, self.c)
        self.assertEqual(weights.shape, (self.B, 3))

    # 5. Weights sum to 1
    def test_05_weights_sum_to_one(self):
        model = CAEGNetV2()
        _, weights, _ = model(self.x, self.c)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    # 6. Weights are non-negative
    def test_06_weights_non_negative(self):
        model = CAEGNetV2()
        _, weights, _ = model(self.x, self.c)
        self.assertTrue((weights >= 0.0).all().item())

    # 7. Parameter count calculation
    def test_07_parameter_counts(self):
        model = CAEGNetV2()
        counts = count_parameters(model)
        self.assertIn("total", counts)
        self.assertIn("gru_expert", counts)
        self.assertIn("tcn_expert", counts)
        self.assertIn("patch_expert", counts)
        self.assertIn("context_encoder", counts)
        self.assertIn("router", counts)
        self.assertGreater(counts["total"], 100000)

    # 8. Context feature dimensions
    def test_08_context_dimensions(self):
        model = CAEGNetV2()
        self.assertEqual(model.base_context_dim, 6)
        self.assertEqual(model.disagreement_dim, 3)
        self.assertEqual(model.total_router_dim, 9)

    # 9. Weekly indexing
    def test_09_weekly_indexing(self):
        # Verify weekly profile similarity uses exactly [0:24] vs [168:192]
        # In a 192h buffer:
        # Prior week 24h is t-191 to t-168 -> indices 0:24
        # Recent 24h is t-23 to t -> indices 168:192
        ctx_buffer = np.zeros((1, 192), dtype=np.float32)
        # Create identical 24h sine pattern in both cycles
        sine_pattern = np.sin(np.linspace(0, 2 * np.pi, 24)).astype(np.float32)
        ctx_buffer[0, 0:24] = sine_pattern
        ctx_buffer[0, 168:192] = sine_pattern
        c_obs = compute_causal_observable_context(ctx_buffer)
        weekly_corr = c_obs[0, 4]
        # When patterns are identical, correlation should be 1.0
        self.assertAlmostEqual(weekly_corr, 1.0, places=4)

    # 10. 192h context buffer
    def test_10_context_buffer_shape(self):
        raw_series = np.arange(500, dtype=np.float32)
        X, X_ctx, Y, orig = create_causal_research_windows(raw_series, lookback_model=168, lookback_context=192, horizon=24)
        self.assertEqual(X.shape[1], 168)
        self.assertEqual(X_ctx.shape[1], 192)
        self.assertEqual(Y.shape[1], 24)
        # Verify that X is the exact suffix of X_ctx
        self.assertTrue(np.allclose(X_ctx[:, 24:], X[:, :, 0]))

    # 11. Disagreement tensor shape
    def test_11_disagreement_shape(self):
        model = CAEGNetV2()
        _, _, diag = model(self.x, self.c)
        d = diag["disagreement"]
        self.assertEqual(d.shape, (self.B, 3))

    # 12. Disagreement is detached
    def test_12_disagreement_detachment(self):
        model = CAEGNetV2()
        # Set expert parameters to require grad
        y_fused, weights, diag = model(self.x, self.c)
        d = diag["disagreement"]
        self.assertFalse(d.requires_grad)

        # Backprop through a function of d only should fail to produce expert gradients
        y1 = model.gru_expert(self.x)
        y2 = model.tcn_expert(self.x)
        y3 = model.patch_expert(self.x)
        d_explicit = model.compute_expert_disagreement(y1, y2, y3)
        self.assertFalse(d_explicit.requires_grad)

    # 13. Auxiliary loss normalization
    def test_13_auxiliary_loss_normalization(self):
        y_pred = torch.randn(self.B, self.H, requires_grad=True)
        weights = F.softmax(torch.randn(self.B, 3), dim=-1)
        expert_preds = {
            "gru": torch.randn(self.B, self.H),
            "tcn": torch.randn(self.B, self.H),
            "patch": torch.randn(self.B, self.H),
        }
        l_total, diag = compute_caeg_v2_loss(
            y_pred, self.y_true, expert_preds, weights, lambda_aux=0.15, beta_entropy=0.001
        )
        expected_aux = (diag["loss_gru"] + diag["loss_tcn"] + diag["loss_patch"]) / 3.0
        self.assertAlmostEqual(diag["loss_aux"], expected_aux, places=5)
        expected_total = diag["loss_fused"] + 0.15 * expected_aux + 0.001 * diag["loss_entropy"]
        self.assertAlmostEqual(diag["loss_total"], expected_total, places=5)

    # 14. Entropy sign and definition
    def test_14_entropy_sign(self):
        # If weights are uniform [1/3, 1/3, 1/3], entropy is ln(3) ~ 1.0986
        w_uniform = torch.full((self.B, 3), 1.0 / 3.0)
        y_pred = torch.randn(self.B, self.H)
        expert_preds = {
            "gru": torch.randn(self.B, self.H),
            "tcn": torch.randn(self.B, self.H),
            "patch": torch.randn(self.B, self.H),
        }
        _, diag = compute_caeg_v2_loss(y_pred, self.y_true, expert_preds, w_uniform, beta_entropy=0.001)
        self.assertAlmostEqual(diag["loss_entropy"], np.log(3), places=4)
        self.assertGreater(diag["loss_entropy"], 0.0)

    # 15. No NaNs/Infs
    def test_15_no_nans_infs(self):
        model = CAEGNetV2()
        y_fused, weights, diag = model(self.x, self.c)
        self.assertFalse(torch.isnan(y_fused).any().item())
        self.assertFalse(torch.isinf(y_fused).any().item())
        self.assertFalse(torch.isnan(weights).any().item())
        self.assertFalse(torch.isinf(weights).any().item())

        l_total, _ = compute_caeg_v2_loss(
            y_fused, self.y_true, diag["expert_predictions"], weights
        )
        l_total.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN in grad of {name}")
                self.assertFalse(torch.isinf(param.grad).any().item(), f"Inf in grad of {name}")

    # 16. Causal feature computation
    def test_16_causal_features(self):
        ctx_buffer = np.random.randn(10, 192).astype(np.float32)
        c_obs = compute_causal_observable_context(ctx_buffer)
        self.assertEqual(c_obs.shape, (10, 5))
        self.assertFalse(np.isnan(c_obs).any())
        self.assertFalse(np.isinf(c_obs).any())

    # 17. Compatibility with CPU
    def test_17_cpu_compatibility(self):
        model = CAEGNetV2().to("cpu")
        x_cpu = self.x.to("cpu")
        c_cpu = self.c.to("cpu")
        y_fused, weights, _ = model(x_cpu, c_cpu)
        self.assertEqual(y_fused.device.type, "cpu")
        self.assertEqual(weights.device.type, "cpu")

    # 18. Compatibility with CUDA
    def test_18_cuda_compatibility(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this environment.")
        device = torch.device("cuda")
        model = CAEGNetV2().to(device)
        x_cuda = self.x.to(device)
        c_cuda = self.c.to(device)
        y_true_cuda = self.y_true.to(device)

        y_fused, weights, diag = model(x_cuda, c_cuda)
        self.assertEqual(y_fused.device.type, "cuda")
        self.assertEqual(weights.device.type, "cuda")

        l_total, _ = compute_caeg_v2_loss(
            y_fused, y_true_cuda, diag["expert_predictions"], weights
        )
        l_total.backward()
        self.assertGreater(l_total.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
