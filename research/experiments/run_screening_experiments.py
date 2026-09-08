"""
CAEG-Net Research Track: Phase 4 Fast Screening Experiments (Seed 42)
=====================================================================
Evaluates 5 candidate variants under strictly identical training, validation,
and testing configurations on Seed 42:

1. Variant A (Canonical V1 Baseline): 4D Operational Context, Global Gating
2. Variant B (No Recent Error): 3D Operational Context (Ablation Control)
3. Variant C (Calendar-Context): 8D Context (4 Operational + 4 Cyclic Calendar)
4. Variant D (Forecast-Aware Routing): 4D Operational + 3D Forecast Disagreement Context
5. Variant E (Calendar + Forecast-Aware): 8D Context + 3D Forecast Disagreement Context
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.models import ResearchCAEGNet
from research.data import prepare_research_pipeline, build_research_dataloaders
from caeg_net import count_parameters
from train import create_criterion, create_optimizer_and_scheduler, train_model
from evaluate import evaluate_model_on_loader


def run_screening():
    print("=" * 80)
    print("CAEG-NET RESEARCH TRACK: PHASE 4 FAST SCREENING EXPERIMENTS (SEED 42)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Load canonical data pipeline
    print("\n[Data] Preparing causal research data pipeline...")
    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    scaler = data["scaler"]

    # Candidate specifications
    candidates = [
        {
            "id": "Variant_A_V1_Canonical",
            "name": "CAEG-Net V1 Canonical (4D Operational)",
            "context_type": "4d",
            "base_context_dim": 4,
            "forecast_aware": False,
            "description": "Trend, Volatility, Periodicity, Causal Recent Error",
        },
        {
            "id": "Variant_B_No_Recent_Error",
            "name": "CAEG-Net (No Recent Error, 3D Operational)",
            "context_type": "3d",
            "base_context_dim": 3,
            "forecast_aware": False,
            "description": "Trend, Volatility, Periodicity (without error feedback)",
        },
        {
            "id": "Variant_C_Calendar_Context",
            "name": "CAEG-Net Calendar-Context (8D Context)",
            "context_type": "8d",
            "base_context_dim": 8,
            "forecast_aware": False,
            "description": "4D Operational + 4D Cyclic Calendar (Hour, Day-of-Week)",
        },
        {
            "id": "Variant_D_Forecast_Aware",
            "name": "CAEG-Net Forecast-Aware Routing (7D Context)",
            "context_type": "4d",
            "base_context_dim": 4,
            "forecast_aware": True,
            "description": "4D Operational + 3D Expert Forecast Disagreement",
        },
        {
            "id": "Variant_E_Calendar_Forecast_Aware",
            "name": "CAEG-Net Calendar + Forecast-Aware (11D Context)",
            "context_type": "8d",
            "base_context_dim": 8,
            "forecast_aware": True,
            "description": "8D Operational/Calendar + 3D Expert Forecast Disagreement",
        },
    ]

    seed = 42
    batch_size = 64
    max_epochs = 25
    patience = 6

    # Dataloader caches for context types
    loaders_cache = {
        "3d": build_research_dataloaders(data, context_type="3d", batch_size=batch_size, seed=seed),
        "4d": build_research_dataloaders(data, context_type="4d", batch_size=batch_size, seed=seed),
        "8d": build_research_dataloaders(data, context_type="8d", batch_size=batch_size, seed=seed),
    }

    results = []
    cache_predictions = {
        "y_true_raw": data["windows"]["test"]["Y"] * scaler.scale_[0] + scaler.mean_[0]
    }

    ckpt_dir = os.path.join(repo_root, "research", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for idx, cand in enumerate(candidates, 1):
        cand_id = cand["id"]
        cand_name = cand["name"]
        print("\n" + "#" * 80)
        print(f"[{idx}/{len(candidates)}] TRAINING CANDIDATE: {cand_name}")
        print(f"Features: {cand['description']}")
        print("#" * 80)

        # Set seeds strictly before instantiation
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()

        # Build clean loaders with seed generator
        loaders = build_research_dataloaders(
            data, context_type=cand["context_type"], batch_size=batch_size, seed=seed
        )

        model = ResearchCAEGNet(
            input_dim=1,
            horizon=24,
            base_context_dim=cand["base_context_dim"],
            forecast_aware=cand["forecast_aware"],
            latent_context_dim=16,
            dropout=0.1,
        ).to(device)

        param_counts = count_parameters(model)
        total_params = param_counts["total_trainable"]
        print(f"Trainable Parameters: {total_params:,}")

        criterion = create_criterion("mse")
        optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-4)

        ckpt_path = os.path.join(ckpt_dir, f"screening_{cand_id}.pt")

        t0 = time.time()
        train_res = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            patience=patience,
            checkpoint_path=ckpt_path,
            device=device,
            model_name=cand_id,
            verbose=True,
        )
        t_train = time.time() - t0

        # Peak VRAM
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Test evaluation with restored best checkpoint
        t_eval_start = time.time()
        best_model = train_res["model"]
        eval_res = evaluate_model_on_loader(best_model, loaders["test"], scaler)
        t_eval = time.time() - t_eval_start

        metrics = eval_res["metrics"]
        actual_epochs = len(train_res["train_losses"])
        best_epoch = train_res["best_epoch"]
        best_val_mse = train_res["best_val_loss"]

        row = {
            "Candidate_ID": cand_id,
            "Model_Name": cand_name,
            "Context_Type": cand["context_type"],
            "Forecast_Aware": cand["forecast_aware"],
            "Total_Context_Dim": model.total_context_dim,
            "Parameters": total_params,
            "Val_MSE_Scaled": round(best_val_mse, 5),
            "Test_MAE_MW": round(metrics["MAE"], 2),
            "Test_RMSE_MW": round(metrics["RMSE"], 2),
            "Test_MSE_MW2": round(metrics["MSE"], 2),
            "Test_MAPE_pct": round(metrics["MAPE"], 2),
            "Test_R2": round(metrics["R2"], 4),
            "Actual_Epochs": actual_epochs,
            "Best_Epoch": best_epoch,
            "Train_Time_s": round(t_train, 1),
            "Inference_Time_s": round(t_eval, 3),
            "Peak_VRAM_MB": round(peak_vram_mb, 1),
        }
        results.append(row)

        cache_predictions[f"pred_{cand_id}"] = eval_res["y_pred_raw"]
        if "weights" in eval_res:
            cache_predictions[f"weights_{cand_id}"] = eval_res["weights"]

        print(f"\n>>> RESULT: {cand_id} <<<")
        print(f"    Val MSE:   {best_val_mse:.5f}")
        print(f"    Test MAE:  {metrics['MAE']:.2f} MW")
        print(f"    Test RMSE: {metrics['RMSE']:.2f} MW")
        print(f"    Test R^2:  {metrics['R2']:.4f}")
        print(f"    Test MAPE: {metrics['MAPE']:.2f}%")
        print(f"    Time:      {t_train:.1f} s | VRAM: {peak_vram_mb:.1f} MB")

    # Save results table
    df_screening = pd.DataFrame(results)
    res_dir = os.path.join(repo_root, "research", "results")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, "screening_comparison.csv")
    df_screening.to_csv(csv_path, index=False)

    npz_path = os.path.join(res_dir, "screening_predictions_cache.npz")
    np.savez_compressed(npz_path, **cache_predictions)

    print("\n" + "=" * 85)
    print("PHASE 4 FAST SCREENING SUMMARY TABLE (SEED 42):")
    print("=" * 85)
    print(df_screening[[
        "Candidate_ID", "Parameters", "Val_MSE_Scaled", "Test_MAE_MW", "Test_RMSE_MW", "Test_R2", "Train_Time_s"
    ]].to_string(index=False))
    print("=" * 85)
    print(f"Saved screening summary to: {csv_path}")
    print(f"Saved prediction caches to: {npz_path}")


if __name__ == "__main__":
    run_screening()
