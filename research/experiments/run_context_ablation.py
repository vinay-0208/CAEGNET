"""
CAEG-Net Research Track: Phase 6 Context Ablation Study
======================================================
Investigates the individual contribution of each context dimension for
the champion research model: CAEG-Net Forecast-Aware Routing (Variant D).

Evaluates:
1. Full Context (7D: Trend, Volatility, Periodicity, Recent Error, Disagreement)
2. Ablate Trend (zero-out / mask feature 0)
3. Ablate Volatility (zero-out / mask feature 1)
4. Ablate Periodicity (zero-out / mask feature 2)
5. Ablate Recent Error (zero-out / mask feature 3)
6. Ablate Forecast Disagreement (zero-out / mask features 4, 5, 6)
7. Context Permutation / Shuffling Negative Control (permute context vectors across test set)
8. Retrained feature omission models on Seed 42 (rigorous retraining comparison)
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.models import ResearchCAEGNet
from research.data import prepare_research_pipeline, build_research_dataloaders
from caeg_net import count_parameters
from train import create_criterion, create_optimizer_and_scheduler, train_model
from evaluate import compute_metrics, evaluate_model_on_loader


def run_ablation_study():
    print("=" * 80)
    print("CAEG-NET RESEARCH TRACK: PHASE 6 CONTEXT ABLATION STUDY")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    scaler = data["scaler"]
    scale = scaler.scale_[0]
    mean = scaler.mean_[0]

    # Load best checkpoint of Variant D (Seed 42)
    ckpt_path = os.path.join(repo_root, "research", "checkpoints", "multiseed", "CAEG_Net_Forecast_Aware_seed_42.pt")
    model = ResearchCAEGNet(
        input_dim=1,
        horizon=24,
        base_context_dim=4,
        forecast_aware=True,
        latent_context_dim=16,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loaders = build_research_dataloaders(data, context_type="4d", batch_size=64, seed=42)
    test_loader = loaders["test"]

    # -------------------------------------------------------------
    # Part 1: Feature Lesioning & Shuffling on Trained Model
    # -------------------------------------------------------------
    print("\n--- Part 1: Context Lesioning & Shuffling on Trained Model ---")

    ablation_modes = [
        {"id": "Full_Context_7D", "name": "Full Context (7D Complete)", "mask_idx": None, "shuffle": False},
        {"id": "Ablate_Trend", "name": "Ablate Trend (Zero-out Feature 0)", "mask_idx": [0], "shuffle": False},
        {"id": "Ablate_Volatility", "name": "Ablate Volatility (Zero-out Feature 1)", "mask_idx": [1], "shuffle": False},
        {"id": "Ablate_Periodicity", "name": "Ablate Periodicity (Zero-out Feature 2)", "mask_idx": [2], "shuffle": False},
        {"id": "Ablate_Recent_Error", "name": "Ablate Recent Error (Zero-out Feature 3)", "mask_idx": [3], "shuffle": False},
        {"id": "Ablate_Disagreement", "name": "Ablate Disagreement (Zero-out Features 4-6)", "mask_idx": "disagreement", "shuffle": False},
        {"id": "Permutation_Negative_Control", "name": "Context Shuffling (Negative Control)", "mask_idx": None, "shuffle": True},
    ]

    lesion_results = []
    baseline_mae = None

    for mode in ablation_modes:
        preds_list = []
        targets_list = []
        weights_list = []

        # If shuffle mode, prepare shuffled context array
        if mode["shuffle"]:
            rng = np.random.RandomState(42)
            orig_C = data["context_4d"]["test"].copy()
            shuffled_C = orig_C[rng.permutation(len(orig_C))]
            shuffled_C_tensor = torch.from_numpy(shuffled_C).to(device)

        with torch.no_grad():
            batch_start_idx = 0
            for batch in test_loader:
                bx, by, bc = batch
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                bc = bc.to(device, non_blocking=True)

                if mode["shuffle"]:
                    b_size = bx.shape[0]
                    bc = shuffled_C_tensor[batch_start_idx : batch_start_idx + b_size]
                    batch_start_idx += b_size
                elif mode["mask_idx"] is not None and mode["mask_idx"] != "disagreement":
                    bc = bc.clone()
                    for idx in mode["mask_idx"]:
                        bc[:, idx] = 0.0

                # Compute forward pass with optional disagreement masking
                y_lstm = model.lstm_expert(bx)
                y_tcn = model.tcn_expert(bx)
                y_cnn = model.cnn_expert(bx)

                disagreement = model.compute_expert_disagreement(y_lstm, y_tcn, y_cnn)
                if mode["mask_idx"] == "disagreement":
                    disagreement = torch.zeros_like(disagreement)

                full_c = torch.cat([bc, disagreement], dim=-1)
                e_c = model.context_encoder(full_c)
                weights = model.gating_network(e_c)

                y_pred = weights[:, 0:1] * y_lstm + weights[:, 1:2] * y_tcn + weights[:, 2:3] * y_cnn

                preds_list.append(y_pred.detach())
                targets_list.append(by.detach())
                weights_list.append(weights.detach())

        preds_scaled = torch.cat(preds_list, dim=0).cpu().numpy()
        targets_scaled = torch.cat(targets_list, dim=0).cpu().numpy()

        preds_raw = preds_scaled * scale + mean
        targets_raw = targets_scaled * scale + mean
        m = compute_metrics(preds_raw, targets_raw)

        if baseline_mae is None:
            baseline_mae = m["MAE"]
            delta_mae = 0.0
        else:
            delta_mae = m["MAE"] - baseline_mae

        row = {
            "Ablation_ID": mode["id"],
            "Description": mode["name"],
            "MAE (MW)": round(m["MAE"], 2),
            "RMSE (MW)": round(m["RMSE"], 2),
            "R^2": round(m["R2"], 4),
            "MAPE (%)": round(m["MAPE"], 2),
            "Delta_MAE_MW": round(delta_mae, 2),
            "Relative_Impact": "BASELINE" if delta_mae == 0 else ("DEGRADED" if delta_mae > 0 else "IMPROVED"),
        }
        lesion_results.append(row)
        print(f"  {mode['name']:<45s} | MAE: {m['MAE']:.2f} MW | Delta: {delta_mae:+.2f} MW | R2: {m['R2']:.4f}")

    df_ablation = pd.DataFrame(lesion_results)
    res_dir = os.path.join(repo_root, "research", "results")
    csv_path = os.path.join(res_dir, "context_ablation.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"\nSaved context ablation results to: {csv_path}")

    print("\n" + "=" * 90)
    print("CONTEXT ABLATION RESULTS TABLE:")
    print("=" * 90)
    print(df_ablation[["Ablation_ID", "MAE (MW)", "RMSE (MW)", "R^2", "MAPE (%)", "Delta_MAE_MW", "Relative_Impact"]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    run_ablation_study()
