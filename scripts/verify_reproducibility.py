"""
CAEG-Net Reproducibility & Architecture Verification Script
===========================================================
1. Verifies exact model parameter counts (121,724 total).
2. Verifies all expert parameter allocations (LSTM, TCN, CNN).
3. Verifies context gating router and confidence head parameters.
4. Verifies forward pass shapes (B, 168, 1) -> (B, 24).
5. Verifies data loading and artifact schemas.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def verify_architecture():
    print("=" * 70)
    print("  CAEG-Net Architecture & Parameter Verification")
    print("=" * 70)

    try:
        import torch
        from src.models import (
            LSTMExpert,
            TCNExpert,
            CNNExpert,
            ContextFeatureEncoder,
            ContextGatingNetwork,
            ConfidenceFallbackCAEGNet,
        )
    except ImportError as e:
        print(f"[FAIL] Unable to import src.models: {e}")
        return False

    lstm = LSTMExpert()
    tcn = TCNExpert()
    cnn = CNNExpert()
    encoder = ContextFeatureEncoder()
    gating = ContextGatingNetwork()
    model = ConfidenceFallbackCAEGNet()

    lstm_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
    tcn_params = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    router_params = (
        sum(p.numel() for p in encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in gating.parameters() if p.requires_grad)
    )
    conf_params = sum(p.numel() for p in model.confidence_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    checks = [
        ("LSTM Expert", lstm_params, 56152),
        ("TCN Expert", tcn_params, 36952),
        ("CNN Expert", cnn_params, 27400),
        ("Context Router (Encoder + Gating)", router_params, 1075),
        ("Confidence Fallback Head", conf_params, 145),
        ("CAEG-Net Total Parameters", total_params, 121724),
    ]

    all_passed = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] {name:<35}: {actual:>7} (Expected: {expected:>7})")

    # Forward pass verification
    print("-" * 70)
    print("  Verifying Forward Pass Dimensions...")
    batch_size = 4
    x = torch.randn(batch_size, 168, 1)
    context = torch.randn(batch_size, 7)

    model.eval()
    with torch.no_grad():
        out, w, diag = model(x, context)
        weights = model.get_gating_weights(context)

    pred_shape_ok = out.shape == (batch_size, 24)
    weights_shape_ok = weights.shape == (batch_size, 3)
    weights_sum_ok = torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    print(f"  [{'PASS' if pred_shape_ok else 'FAIL'}] Prediction Output Shape : {tuple(out.shape)} (Expected: ({batch_size}, 24))")
    print(f"  [{'PASS' if weights_shape_ok else 'FAIL'}] Gating Weights Shape   : {tuple(weights.shape)} (Expected: ({batch_size}, 3))")
    print(f"  [{'PASS' if weights_sum_ok else 'FAIL'}] Gating Weights Sum to 1 : {weights_sum_ok}")

    forward_passed = pred_shape_ok and weights_shape_ok and weights_sum_ok
    return all_passed and forward_passed


def run_unit_tests():
    print("\n" + "=" * 70)
    print("  Executing Unit Test Suite...")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(str(REPO_ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    arch_ok = verify_architecture()
    tests_ok = run_unit_tests()

    print("\n" + "=" * 70)
    if arch_ok and tests_ok:
        print("  [SUCCESS] All architecture and unit test verifications passed.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("  [FAILURE] One or more verification checks failed.")
        print("=" * 70)
        sys.exit(1)
