"""
Phase 3 Neural Architecture Verification Suite
==============================================
Validates the CAEG-Net neural architecture, specialized expert outputs,
context encoder, gating network, convex fusion, parameter distribution,
and gradient flow.

Checklist:
1. caeg_net.py exists and imports cleanly
2. LSTMExpert forward pass and shape [B, 24]
3. TCNExpert forward pass and shape [B, 24]
4. CNNExpert forward pass and shape [B, 24]
5. ContextFeatureEncoder forward pass and shape [B, 16]
6. ContextGatingNetwork forward pass, shape [B, 3], and Softmax sum=1.0
7. Complete CAEGNet forward pass and shape [B, 24]
8. Dynamic convex fusion verification: y_pred == sum(w_i * y_i)
9. Non-negativity of routing weights: w_i > 0
10. Total parameter count and submodule distribution audit
11. Absence of NaN values across all expert and fused outputs
12. Absence of Inf values across all expert and fused outputs
13. Backward pass gradient flow through all experts and context gating
14. Integration with live PyTorch DataLoader from data_utils.py
15. Notebook notebooks/CAEG_Net_Development.ipynb validation
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import nbformat

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
    TimeSeriesContextDataset,
    create_dataloaders
)
from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    ContextFeatureEncoder,
    ContextGatingNetwork,
    CAEGNet,
    count_parameters,
    print_architecture_summary
)


def run_phase3_verification():
    print("=" * 70)
    print("STARTING CAEG-NET PHASE 3 NEURAL ARCHITECTURE AUDIT")
    print("=" * 70)
    passed = 0
    total = 15

    # Check 1: Files exist
    assert os.path.isfile("caeg_net.py"), "caeg_net.py missing"
    assert os.path.isfile("train.py"), "train.py missing"
    assert os.path.isfile("evaluate.py"), "evaluate.py missing"
    assert os.path.isfile("experiments.py"), "experiments.py missing"
    print("[CHECK 1/15 PASSED] Architecture and interface modules exist.")
    passed += 1

    B = 16
    L = 168
    H = 24
    C_dim = 4

    x_dummy = torch.randn(B, L, 1)
    c_dummy = torch.randn(B, C_dim)

    # Check 2: LSTMExpert
    lstm = LSTMExpert(input_dim=1, hidden_dim=64, num_layers=2, horizon=H)
    y_lstm = lstm(x_dummy)
    assert y_lstm.shape == (B, H), f"LSTM output shape {y_lstm.shape} != {(B, H)}"
    print(f"[CHECK 2/15 PASSED] LSTMExpert verified: output shape {y_lstm.shape}.")
    passed += 1

    # Check 3: TCNExpert
    tcn = TCNExpert(input_dim=1, channels=32, dilations=(1, 2, 4, 8, 16, 32), horizon=H)
    y_tcn = tcn(x_dummy)
    assert y_tcn.shape == (B, H), f"TCN output shape {y_tcn.shape} != {(B, H)}"
    print(f"[CHECK 3/15 PASSED] TCNExpert verified: output shape {y_tcn.shape}.")
    passed += 1

    # Check 4: CNNExpert
    cnn = CNNExpert(input_dim=1, horizon=H)
    y_cnn = cnn(x_dummy)
    assert y_cnn.shape == (B, H), f"CNN output shape {y_cnn.shape} != {(B, H)}"
    print(f"[CHECK 4/15 PASSED] CNNExpert verified: output shape {y_cnn.shape}.")
    passed += 1

    # Check 5: ContextFeatureEncoder
    encoder = ContextFeatureEncoder(context_dim=C_dim, latent_dim=16)
    e_c = encoder(c_dummy)
    assert e_c.shape == (B, 16), f"Encoder output shape {e_c.shape} != {(B, 16)}"
    print(f"[CHECK 5/15 PASSED] ContextFeatureEncoder verified: output shape {e_c.shape}.")
    passed += 1

    # Check 6: ContextGatingNetwork
    gating = ContextGatingNetwork(latent_dim=16, num_experts=3)
    weights = gating(e_c)
    assert weights.shape == (B, 3), f"Gating output shape {weights.shape} != {(B, 3)}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5), "Weights do not sum to 1.0"
    print(f"[CHECK 6/15 PASSED] ContextGatingNetwork verified: shape {weights.shape}, Softmax sum=1.0.")
    passed += 1

    # Check 7: CAEGNet complete forward pass
    model = CAEGNet(input_dim=1, horizon=H, context_dim=C_dim, latent_context_dim=16)
    y_pred, w, exp_dict = model(x_dummy, c_dummy, return_diagnostics=True)
    assert y_pred.shape == (B, H), f"Fused output shape {y_pred.shape} != {(B, H)}"
    print(f"[CHECK 7/15 PASSED] CAEGNet complete forward pass verified: shape {y_pred.shape}.")
    passed += 1

    # Check 8: Dynamic convex fusion verification
    manual = w[:, 0:1] * exp_dict["lstm"] + w[:, 1:2] * exp_dict["tcn"] + w[:, 2:3] * exp_dict["cnn"]
    assert torch.allclose(y_pred, manual, atol=1e-6), "Fusion does not match manual convex sum"
    print("[CHECK 8/15 PASSED] Dynamic convex fusion verified: y_pred == sum(w_i * y_i).")
    passed += 1

    # Check 9: Routing weights strictly positive
    assert (w > 0.0).all(), "Negative weights found"
    print("[CHECK 9/15 PASSED] Routing weights strictly positive: w_i > 0.")
    passed += 1

    # Check 10: Parameter counts
    p_info = count_parameters(model)
    assert p_info["total_trainable"] == 121531, f"Unexpected parameter count: {p_info['total_trainable']}"
    print(f"[CHECK 10/15 PASSED] Parameter counts verified: {p_info['total_trainable']:,} trainable parameters.")
    passed += 1

    # Check 11: Zero NaNs
    assert not torch.isnan(y_pred).any(), "NaN in y_pred"
    assert not torch.isnan(w).any(), "NaN in weights"
    for k, v in exp_dict.items():
        assert not torch.isnan(v).any(), f"NaN in expert {k}"
    print("[CHECK 11/15 PASSED] Zero NaN values across all expert, gating, and fused outputs.")
    passed += 1

    # Check 12: Zero Infs
    assert not torch.isinf(y_pred).any(), "Inf in y_pred"
    assert not torch.isinf(w).any(), "Inf in weights"
    for k, v in exp_dict.items():
        assert not torch.isinf(v).any(), f"Inf in expert {k}"
    print("[CHECK 12/15 PASSED] Zero Inf values across all expert, gating, and fused outputs.")
    passed += 1

    # Check 13: Backward pass gradient flow
    model.train()
    y_out, _, _ = model(x_dummy, c_dummy, return_diagnostics=True)
    loss = y_out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    print("[CHECK 13/15 PASSED] End-to-end gradient flow verified through all 3 experts and context gate.")
    passed += 1

    # Check 14: Integration with live data loader
    df, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    train_df, val_df, test_df, _ = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    rec_tr, rec_val, rec_test, _ = compute_causal_recent_forecast_errors(windows)
    C_tr = extract_context_features(windows["train"]["X"], rec_tr)
    ds = TimeSeriesContextDataset(windows["train"]["X"], windows["train"]["Y"], C_tr)
    loader = create_dataloaders({"train": ds}, batch_size=32, shuffle_train=False)["train"]
    bx, by, bc = next(iter(loader))
    model.eval()
    with torch.no_grad():
        live_pred, live_w, _ = model(bx, bc)
    assert live_pred.shape == (32, 24)
    assert live_w.shape == (32, 3)
    print(f"[CHECK 14/15 PASSED] Live DataLoader integration verified: batch {live_pred.shape}, weights {live_w.shape}.")
    passed += 1

    # Check 15: Notebook valid
    nb_path = os.path.join("notebooks", "CAEG_Net_Development.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_obj = nbformat.read(f, as_version=4)
    assert len(nb_obj.cells) >= 15, "Notebook missing required sections"
    print(f"[CHECK 15/15 PASSED] Jupyter notebook verified: valid nbformat v4 with {len(nb_obj.cells)} cells.")
    passed += 1

    print("\n" + "=" * 70)
    print(f"ALL {passed}/{total} PHASE 3 ARCHITECTURE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase3_verification()
    sys.exit(0 if success else 1)
