"""
CAEG-Net V2 Phase 4 Controlled Context & Loss Ablation Suite (Seed 42)
======================================================================
Executes a controlled suite of 11 architectural and context ablations under Seed 42:
1. Full V2 (All 9 features: 5 observable + 1 baseline error + 3 disagreement, lambda=0.15)
2. No Physical Context (Remove Type A: 5 observable context features masked)
3. No Recent Error Feedback (Remove Type B: recent expanding-Ridge MAE masked)
4. No Forecast Difficulty / Disagreement (Remove Type C: 3 disagreement features masked)
5. No Weekly Profile Similarity (Feature 5 masked)
6. No Diurnal Periodicity (Feature 4 masked)
7. No Trend Slope (Feature 1 masked)
8. No Volatility (Feature 2 masked)
9. No Recent 48h Range Ratio (Feature 3 masked)
10. No Context At All (All 6 context features masked)
11. No Auxiliary Expert Loss (lambda = 0, fused MSE objective only)

Saves results to: research/results/phase4_context_ablations.csv
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.models import CAEGNetV2, count_parameters
from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.training import train_caeg_v2, evaluate_caeg_v2


def run_context_ablations(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_csv: str = "research/results/phase4_context_ablations.csv",
):
    device = torch.device(device_str)
    print(f"=== CAEG-Net V2 Phase 4 Context & Loss Ablation Suite (Seed {seed}) on {device} ===")

    # 1. Load pipeline once
    pipe = prepare_research_v2_pipeline(data_path=data_path)
    scaler = pipe["scaler"]

    # Define the 11 ablations
    # Context vector has 6 features:
    # 0: Trend, 1: Volatility, 2: Range, 3: Diurnal, 4: Weekly, 5: Recent Error
    ablations = [
        {
            "id": "A11_Full_V2",
            "name": "Full Proposed CAEG-Net V2",
            "mask_6d": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Full model with all 9 features and auxiliary supervision (lambda=0.15)",
        },
        {
            "id": "A4_No_Physical_Context",
            "name": "No Physical Context (Type A Removed)",
            "mask_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks features 0-4; router receives only Recent Error and Disagreement",
        },
        {
            "id": "A6_No_Recent_Error",
            "name": "No Recent Error Feedback (Type B Removed)",
            "mask_6d": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks feature 5; router receives only 5 Observable features and Disagreement",
        },
        {
            "id": "A7_No_Disagreement",
            "name": "No Forecast Difficulty (Type C Removed)",
            "mask_6d": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forecast_aware": False,
            "lambda_aux": 0.15,
            "description": "Removes 3 disagreement features; router receives only 6D base context",
        },
        {
            "id": "A5a_No_Weekly_Profile",
            "name": "No Weekly Profile Similarity",
            "mask_6d": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks weekly profile correlation r_168 (Feature 4)",
        },
        {
            "id": "A5b_No_Diurnal_Periodicity",
            "name": "No Diurnal Rhythmicity",
            "mask_6d": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks diurnal lag-24 autocorrelation r_24 (Feature 3)",
        },
        {
            "id": "A5c_No_Trend_Slope",
            "name": "No Trend Slope",
            "mask_6d": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks trend slope beta_1 (Feature 0)",
        },
        {
            "id": "A5d_No_Volatility",
            "name": "No Short-Term Volatility",
            "mask_6d": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks short-term volatility sigma(diff) (Feature 1)",
        },
        {
            "id": "A5e_No_Range_Ratio",
            "name": "No Recent 48h Range Ratio",
            "mask_6d": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.15,
            "description": "Masks 48h range ratio R_48 (Feature 2)",
        },
        {
            "id": "A4b_No_Context_At_All",
            "name": "No Context At All (Zero Context)",
            "mask_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "forecast_aware": False,
            "lambda_aux": 0.15,
            "description": "All external context and disagreement masked to zero",
        },
        {
            "id": "A8_No_Auxiliary_Loss",
            "name": "No Auxiliary Expert Loss (lambda=0)",
            "mask_6d": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forecast_aware": True,
            "lambda_aux": 0.0,
            "description": "Trained purely on fused MSE loss; expert supervision disabled (lambda=0)",
        },
    ]

    results = []

    for idx, ab in enumerate(ablations):
        ab_id = ab["id"]
        ab_name = ab["name"]
        print(f"\n--- [{idx+1}/{len(ablations)}] Running Ablation: {ab_id} ({ab_name}) ---")

        # Set seed strictly before data loading and model init
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)

        mask_t = torch.tensor(ab["mask_6d"], dtype=torch.float32)

        model = CAEGNetV2(forecast_aware=ab["forecast_aware"]).to(device)

        train_res = train_caeg_v2(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=ab["lambda_aux"],
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            feature_mask=mask_t,
            device=device,
            verbose=False,
        )

        eval_res = evaluate_caeg_v2(
            model=model,
            data_loader=dataloaders["test"],
            scaler=scaler,
            temperature=0.5,
            feature_mask=mask_t,
            device=device,
        )

        f_met = eval_res["fused_metrics"]
        w_stat = eval_res["weight_stats"]

        res_row = {
            "Ablation_ID": ab_id,
            "Ablation_Name": ab_name,
            "Test_MAE_MW": f_met["mae_mw"],
            "Test_RMSE_MW": f_met["rmse_mw"],
            "Test_R2": f_met["r2"],
            "Test_MAPE_pct": f_met["mape_pct"],
            "Best_Epoch": train_res["best_epoch"],
            "Total_Epochs": train_res["total_epochs"],
            "Best_Val_Fused_MSE": train_res["best_val_fused"],
            "Training_Time_s": round(train_res["training_time_seconds"], 2),
            "w_GRU_mean": w_stat["mean_weights"][0],
            "w_TCN_mean": w_stat["mean_weights"][1],
            "w_Patch_mean": w_stat["mean_weights"][2],
            "w_GRU_std": w_stat["std_weights"][0],
            "w_TCN_std": w_stat["std_weights"][1],
            "w_Patch_std": w_stat["std_weights"][2],
            "Mean_Entropy": w_stat["mean_entropy"],
            "Mean_N_eff": w_stat["mean_n_eff"],
            "GRU_MAE_MW": eval_res["gru_metrics"]["mae_mw"],
            "TCN_MAE_MW": eval_res["tcn_metrics"]["mae_mw"],
            "Patch_MAE_MW": eval_res["patch_metrics"]["mae_mw"],
            "Equal_Ensemble_MAE_MW": eval_res["equal_ensemble_metrics"]["mae_mw"],
            "Delta_vs_Full_MAE": 0.0,  # will compute after loop
        }
        results.append(res_row)
        print(f"Done: Test MAE = {f_met['mae_mw']:.2f} MW | Val MSE = {train_res['best_val_fused']:.4f} | Runtime = {train_res['training_time_seconds']:.1f}s")

    # Compute delta vs Full V2
    full_mae = results[0]["Test_MAE_MW"]
    for r in results:
        r["Delta_vs_Full_MAE"] = round(r["Test_MAE_MW"] - full_mae, 2)

    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_res.to_csv(output_csv, index=False)
    print(f"\nSaved context ablation results to {output_csv}")

    print("\n=== CONTEXT ABLATION SUMMARY (SEED 42) ===")
    print(df_res[["Ablation_ID", "Test_MAE_MW", "Delta_vs_Full_MAE", "Test_R2", "Mean_Entropy", "Patch_MAE_MW"]].to_string(index=False))

    return df_res


if __name__ == "__main__":
    run_context_ablations()
