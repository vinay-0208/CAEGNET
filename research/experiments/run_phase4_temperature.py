"""
CAEG-Net V2 Phase 4 Temperature Study (Validation Partition Selection)
======================================================================
Formally specified in Section 11 of PHASE_2_RESEARCH_MODEL_SPECIFICATION.md:
- Candidate temperatures: tau in {0.2, 0.5, 0.8, 1.0}
- Selection protocol: Strictly evaluated on the VALIDATION partition.
  The test set is NEVER accessed to tune or select tau.
- Reports:
  - Validation Fused MSE
  - Validation MAE (MW)
  - Validation Routing Entropy H(w)
  - Validation Routing Weight Standard Deviations
  - Effective number of experts N_eff
- Once the best temperature is selected on validation, evaluates that temperature on test
  as a single final confirmation.

Saves results to: research/results/phase4_temperature_study.csv
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

from research.models import CAEGNetV2
from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.training import train_caeg_v2, evaluate_caeg_v2


def run_temperature_study(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_csv: str = "research/results/phase4_temperature_study.csv",
):
    device = torch.device(device_str)
    print(f"=== CAEG-Net V2 Phase 4 Temperature Study (Seed {seed}) on {device} ===")

    pipe = prepare_research_v2_pipeline(data_path=data_path)
    scaler = pipe["scaler"]

    candidate_temperatures = [0.2, 0.5, 0.8, 1.0]
    results = []

    for tau in candidate_temperatures:
        print(f"\n--- Training CAEG-Net V2 with tau = {tau} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
        model = CAEGNetV2(temperature=tau).to(device)

        train_res = train_caeg_v2(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.15,
            beta_entropy=0.001,
            temperature=tau,
            max_epochs=35,
            patience=7,
            device=device,
            verbose=False,
        )

        # Evaluate strictly on VALIDATION partition for selection
        val_eval = evaluate_caeg_v2(
            model=model,
            data_loader=dataloaders["val"],
            scaler=scaler,
            temperature=tau,
            device=device,
        )

        # Evaluate on TEST partition strictly for reporting
        test_eval = evaluate_caeg_v2(
            model=model,
            data_loader=dataloaders["test"],
            scaler=scaler,
            temperature=tau,
            device=device,
        )

        v_met = val_eval["fused_metrics"]
        v_wstat = val_eval["weight_stats"]

        t_met = test_eval["fused_metrics"]
        t_wstat = test_eval["weight_stats"]

        row = {
            "Temperature_tau": tau,
            "Best_Epoch": train_res["best_epoch"],
            "Val_Fused_MSE": v_met["mse_standardized"],
            "Val_MAE_MW": round(v_met["mae_mw"], 2),
            "Val_RMSE_MW": round(v_met["rmse_mw"], 2),
            "Val_Entropy": round(v_wstat["mean_entropy"], 4),
            "Val_N_eff": round(v_wstat["mean_n_eff"], 2),
            "Val_w_GRU_std": round(v_wstat["std_weights"][0], 4),
            "Val_w_TCN_std": round(v_wstat["std_weights"][1], 4),
            "Val_w_Patch_std": round(v_wstat["std_weights"][2], 4),
            "Val_w_Patch_mean": round(v_wstat["mean_weights"][2], 4),
            "Test_MAE_MW": round(t_met["mae_mw"], 2),
            "Test_RMSE_MW": round(t_met["rmse_mw"], 2),
            "Test_R2": round(t_met["r2"], 4),
            "Test_Entropy": round(t_wstat["mean_entropy"], 4),
            "Training_Time_s": round(train_res["training_time_seconds"], 2),
        }
        results.append(row)
        print(f"tau={tau} -> Val MSE={v_met['mse_standardized']:.4f} | Val MAE={v_met['mae_mw']:.2f} MW | Val Entropy={v_wstat['mean_entropy']:.4f} | Test MAE={t_met['mae_mw']:.2f} MW")

    df_temp = pd.DataFrame(results)

    # Validation-based selection: select row with lowest Val_Fused_MSE
    best_val_idx = df_temp["Val_Fused_MSE"].idxmin()
    best_tau = df_temp.loc[best_val_idx, "Temperature_tau"]
    df_temp["Selected_by_Validation"] = (df_temp["Temperature_tau"] == best_tau)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_temp.to_csv(output_csv, index=False)
    print(f"\nSaved temperature study results to {output_csv}")

    print(f"\n=== VALIDATION TEMPERATURE SELECTION ===")
    print(f"Selected tau = {best_tau} strictly based on minimum Validation MSE ({df_temp.loc[best_val_idx, 'Val_Fused_MSE']:.4f})")
    print(df_temp[["Temperature_tau", "Val_Fused_MSE", "Val_MAE_MW", "Val_Entropy", "Test_MAE_MW", "Selected_by_Validation"]].to_string(index=False))

    return df_temp


if __name__ == "__main__":
    run_temperature_study()
