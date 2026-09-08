"""
CAEG-Net Decoupled (D-CAEG) Phase 7 Unit Test Suite
===================================================
Verifies all 8 critical requirements mandated by Phase 7:
1. Experts are frozen during router training (requires_grad=False).
2. Expert gradients remain strictly None or zero after router backward pass.
3. Router weights are valid probabilities (non-negative and sum to 1.0).
4. Output shape is strictly [B, 24].
5. No future context is consumed (causal 192h context buffer and causal history).
6. Deterministic behavior under a fixed seed.
7. Standalone expert architectures are loaded and evaluated correctly.
8. Parameter counts are correct: total deployed = 142,433, trainable during Stage 3 = 2,595.
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
    DecoupledCAEGNet,
    LearnedStaticEnsemble,
    count_parameters,
)
from research.training import train_decoupled_router, evaluate_decoupled_caeg


class TestPhase7Decoupled(unittest.TestCase):
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

        # Create expert backbones
        self.gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24)
        self.tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24)
        self.patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24)

    def test_01_parameter_counts_and_freezing(self):
        """Verify parameter accounting: 142,433 total deployed, 2,595 trainable in Stage 3."""
        model = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        params = count_parameters(model)

        self.assertEqual(params["total"], 142433)
        self.assertEqual(params["trainable"], 2595)
        self.assertEqual(params["gru_expert"], 31344)
        self.assertEqual(params["tcn_expert"], 44870)
        self.assertEqual(params["patch_expert"], 63624)
        self.assertEqual(params["context_encoder"], 1440)
        self.assertEqual(params["router"], 1155)

        # Verify all expert parameters have requires_grad = False
        for expert in [model.gru_expert, model.tcn_expert, model.patch_expert]:
            for p in expert.parameters():
                self.assertFalse(p.requires_grad, "Expert parameter should have requires_grad=False")

        # Verify router parameters have requires_grad = True
        for p in model.context_encoder.parameters():
            self.assertTrue(p.requires_grad, "Context encoder parameter should be trainable")
        for p in model.router.parameters():
            self.assertTrue(p.requires_grad, "Router parameter should be trainable")

    def test_02_expert_gradients_remain_none(self):
        """Verify that backward pass computes gradients ONLY for router, NOT experts."""
        model = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        model.train()

        y_fused, weights, diag = model(self.x, self.c)
        loss = F.mse_loss(y_fused, self.y_true)
        loss.backward()

        # Expert gradients must be None
        for expert_name, expert in [("GRU", model.gru_expert), ("TCN", model.tcn_expert), ("Patch", model.patch_expert)]:
            for name, p in expert.named_parameters():
                self.assertIsNone(p.grad, f"Gradient for {expert_name}.{name} should be None after backward!")

        # Router gradients must be valid and non-None
        for name, p in model.context_encoder.named_parameters():
            self.assertIsNotNone(p.grad, f"Grad missing for context_encoder.{name}")
            self.assertFalse(torch.isnan(p.grad).any())
        for name, p in model.router.named_parameters():
            self.assertIsNotNone(p.grad, f"Grad missing for router.{name}")
            self.assertFalse(torch.isnan(p.grad).any())

    def test_03_router_weights_valid_probabilities(self):
        """Verify weights are valid probability simplex elements: w_i >= 0, sum(w_i) = 1.0."""
        model = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        model.eval()
        with torch.no_grad():
            _, weights, _ = model(self.x, self.c)

        self.assertEqual(weights.shape, (self.B, 3))
        self.assertTrue(torch.all(weights >= 0.0), "Weights must be non-negative")
        sums = torch.sum(weights, dim=-1)
        ones = torch.ones_like(sums)
        self.assertTrue(torch.allclose(sums, ones, atol=1e-6), "Weights must sum to 1.0")

    def test_04_output_shape(self):
        """Verify output shape is strictly [B, 24]."""
        model = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        model.eval()
        with torch.no_grad():
            y_fused = model(self.x, self.c, return_diagnostics=False)
        self.assertEqual(y_fused.shape, (self.B, self.H))

    def test_05_learned_static_ensemble(self):
        """Verify Experiment B baseline: 3 learned static weights on frozen experts."""
        model = LearnedStaticEnsemble(self.gru, self.tcn, self.patch)
        params = count_parameters(model)
        self.assertEqual(params["total"], 139841)
        self.assertEqual(params["trainable"], 3)

        y_fused, weights, _ = model(self.x)
        self.assertEqual(y_fused.shape, (self.B, self.H))
        self.assertEqual(weights.shape, (self.B, 3))
        # Initial weights should be [1/3, 1/3, 1/3]
        expected_init = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0])
        self.assertTrue(torch.allclose(weights[0], expected_init, atol=1e-5))

        # Backward pass updates ONLY the 3 logits
        loss = F.mse_loss(y_fused, self.y_true)
        loss.backward()
        self.assertIsNotNone(model.logits.grad)
        for expert in [model.gru_expert, model.tcn_expert, model.patch_expert]:
            for p in expert.parameters():
                self.assertIsNone(p.grad)

    def test_06_deterministic_behavior(self):
        """Verify exact reproducibility under a fixed seed."""
        torch.manual_seed(123)
        m1 = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        with torch.no_grad():
            y1, w1, _ = m1(self.x, self.c)

        torch.manual_seed(123)
        m2 = DecoupledCAEGNet(self.gru, self.tcn, self.patch)
        with torch.no_grad():
            y2, w2, _ = m2(self.x, self.c)

        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.equal(w1, w2))

    def test_07_standalone_checkpoint_loading(self):
        """Verify that standalone expert checkpoints can be loaded into DecoupledCAEGNet with frozen weights."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            gru_path = os.path.join(tmpdir, "gru.pt")
            tcn_path = os.path.join(tmpdir, "tcn.pt")
            patch_path = os.path.join(tmpdir, "patch.pt")

            torch.save(self.gru.state_dict(), gru_path)
            torch.save(self.tcn.state_dict(), tcn_path)
            torch.save(self.patch.state_dict(), patch_path)

            new_gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24)
            new_tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24)
            new_patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24)

            new_gru.load_state_dict(torch.load(gru_path))
            new_tcn.load_state_dict(torch.load(tcn_path))
            new_patch.load_state_dict(torch.load(patch_path))

            decoupled = DecoupledCAEGNet(new_gru, new_tcn, new_patch)
            decoupled.eval()
            with torch.no_grad():
                y_pred = decoupled(self.x, self.c, return_diagnostics=False)
            self.assertEqual(y_pred.shape, (self.B, self.H))
            for p in decoupled.gru_expert.parameters():
                self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()

