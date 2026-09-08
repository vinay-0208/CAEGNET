"""
CAEG-Net V2 Phase 4 Baseline Routing Diagnostics (Seed 42)
===========================================================
Executes a single controlled Seed-42 training and evaluation run of CAEG-Net V2,
generating a comprehensive per-origin routing and error diagnostic dataset.

Collects:
- Fused forecast, individual expert forecasts, and equal ensemble forecasts
- Actual targets (MW)
- 9 context features (6 base context + 3 detached disagreement)
- Router weights [w_GRU, w_TCN, w_Patch], entropy H(w), and N_eff
- Per-origin MAE for Fused, GRU, TCN, Patch, Equal Ensemble, and Oracle
- Oracle best expert identification and router top-choice accuracy
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.models import CAEGNetV2, count_parameters
from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.training import train_caeg_v2, evaluate_caeg_v2


def run_seed42_diagnostics(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_csv: str = "research/results/phase4_seed42_routing_diagnostics.csv",
    checkpoint_path: str = "research/checkpoints/phase4_v2_full_seed42.pt",
) -> Dict:
    device = torch.device(device_str)
    print(f"=== CAEG-Net V2 Phase 4 Routing Diagnostics (Seed {seed}) on {device} ===")

    # Set seeds strictly
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 1. Prepare pipeline
    print("Loading data pipeline...")
    pipe = prepare_research_v2_pipeline(data_path=data_path)
    dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
    scaler = pipe["scaler"]
    test_origins = pipe["windows"]["test"]["origins"]

    # Extract test timestamps
    val_df = pipe["split_dfs"]["val"]
    test_df = pipe["split_dfs"]["test"]
    hist_test_ts = val_df["timestamp"].iloc[-(192 - 1):]
    test_ts_ext = pd.concat([hist_test_ts, test_df["timestamp"]]).reset_index(drop=True)
    test_timestamps = test_ts_ext.iloc[test_origins].reset_index(drop=True)

    # 2. Instantiate and train model
    model = CAEGNetV2().to(device)
    print(f"Instantiated CAEGNetV2 with {count_parameters(model)['total']:,} parameters.")

    train_res = train_caeg_v2(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        lr=1e-3,
        weight_decay=1e-4,
        lambda_aux=0.15,
        beta_entropy=0.001,
        temperature=0.5,
        max_epochs=35,
        patience=7,
        checkpoint_path=checkpoint_path,
        device=device,
        verbose=True,
    )

    print(f"Training completed in {train_res['training_time_seconds']:.2f}s (Best Epoch: {train_res['best_epoch']}, Best Val Fused MSE: {train_res['best_val_fused']:.4f})")

    # 3. Evaluate on test set
    eval_res = evaluate_caeg_v2(
        model=model,
        data_loader=dataloaders["test"],
        scaler=scaler,
        temperature=0.5,
        device=device,
    )

    # 4. Extract arrays
    y_pred_mw = eval_res["y_pred_mw"]      # [N, 24]
    y_true_mw = eval_res["y_true_mw"]      # [N, 24]
    gru_mw = eval_res["gru_mw"]            # [N, 24]
    tcn_mw = eval_res["tcn_mw"]            # [N, 24]
    patch_mw = eval_res["patch_mw"]        # [N, 24]
    equal_ens_mw = eval_res["equal_ens_mw"]# [N, 24]
    weights = eval_res["weights"]          # [N, 3]
    context_6d = eval_res["context_6d"]    # [N, 6]
    disagreement = eval_res["disagreement"]# [N, 3]
    entropy = eval_res["entropy_per_origin"]# [N]
    n_eff = np.exp(entropy)                # [N]

    N = len(y_pred_mw)

    # Per-origin 24h MAEs
    mae_fused = np.mean(np.abs(y_pred_mw - y_true_mw), axis=-1)
    mae_gru = np.mean(np.abs(gru_mw - y_true_mw), axis=-1)
    mae_tcn = np.mean(np.abs(tcn_mw - y_true_mw), axis=-1)
    mae_patch = np.mean(np.abs(patch_mw - y_true_mw), axis=-1)
    mae_equal = np.mean(np.abs(equal_ens_mw - y_true_mw), axis=-1)

    # Expert stack for per-origin oracle: shape [N, 3]
    expert_maes = np.column_stack([mae_gru, mae_tcn, mae_patch])
    oracle_idx = np.argmin(expert_maes, axis=1)  # 0: GRU, 1: TCN, 2: Patch
    mae_oracle = np.min(expert_maes, axis=1)
    expert_names = ["GRU", "TCN", "Patch"]
    oracle_names = [expert_names[i] for i in oracle_idx]

    router_top_idx = np.argmax(weights, axis=1)
    router_top_names = [expert_names[i] for i in router_top_idx]
    is_top_oracle = (router_top_idx == oracle_idx).astype(int)

    regret_vs_oracle = mae_fused - mae_oracle
    gain_over_equal = mae_equal - mae_fused
    best_single_expert_mae = min(float(np.mean(mae_gru)), float(np.mean(mae_tcn)), float(np.mean(mae_patch)))
    gain_over_best_single = best_single_expert_mae - mae_fused

    # Build DataFrame
    df_diag = pd.DataFrame({
        "origin_idx": test_origins,
        "timestamp": test_timestamps,
        # Type A: Observable Physical Context
        "Trend_Slope": context_6d[:, 0],
        "ShortTerm_Volatility": context_6d[:, 1],
        "Recent48h_RangeRatio": context_6d[:, 2],
        "Diurnal_Periodicity_r24": context_6d[:, 3],
        "Weekly_Profile_r168": context_6d[:, 4],
        # Type B: Post-Forecast Error Feedback
        "Recent_Baseline_MAE": context_6d[:, 5],
        # Type C: Forecast Difficulty / Disagreement
        "Disagreement_Pairwise_MAE": disagreement[:, 0] if disagreement is not None else 0.0,
        "Disagreement_Std": disagreement[:, 1] if disagreement is not None else 0.0,
        "Disagreement_Range": disagreement[:, 2] if disagreement is not None else 0.0,
        # Router Weights
        "w_GRU": weights[:, 0],
        "w_TCN": weights[:, 1],
        "w_Patch": weights[:, 2],
        "Entropy_H": entropy,
        "N_eff": n_eff,
        "Max_Weight": np.max(weights, axis=1),
        "Min_Weight": np.min(weights, axis=1),
        "Router_Top_Expert": router_top_names,
        # Absolute Errors (MW)
        "MAE_Fused": mae_fused,
        "MAE_GRU": mae_gru,
        "MAE_TCN": mae_tcn,
        "MAE_Patch": mae_patch,
        "MAE_Equal_Ensemble": mae_equal,
        "MAE_Oracle": mae_oracle,
        "Oracle_Best_Expert": oracle_names,
        "Is_Router_Top_Oracle": is_top_oracle,
        "Regret_vs_Oracle": regret_vs_oracle,
        "Gain_over_Equal_Ensemble": gain_over_equal,
        "Gain_over_Best_Single": gain_over_best_single,
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_diag.to_csv(output_csv, index=False)
    print(f"Saved {len(df_diag)} diagnostic origins to {output_csv}")

    # Summary Statistics
    summary = {
        "seed": seed,
        "num_test_origins": N,
        "test_fused_metrics": eval_res["fused_metrics"],
        "test_gru_metrics": eval_res["gru_metrics"],
        "test_tcn_metrics": eval_res["tcn_metrics"],
        "test_patch_metrics": eval_res["patch_metrics"],
        "test_equal_ensemble_metrics": eval_res["equal_ensemble_metrics"],
        "test_oracle_metrics": {
            "mae_mw": float(np.mean(mae_oracle)),
            "regret_mean_mw": float(np.mean(regret_vs_oracle)),
            "regret_median_mw": float(np.median(regret_vs_oracle)),
        },
        "router_top_accuracy": float(np.mean(is_top_oracle)),
        "expert_win_shares": {
            "GRU": float(np.mean(oracle_idx == 0)),
            "TCN": float(np.mean(oracle_idx == 1)),
            "Patch": float(np.mean(oracle_idx == 2)),
        },
        "router_selection_shares": {
            "GRU": float(np.mean(router_top_idx == 0)),
            "TCN": float(np.mean(router_top_idx == 1)),
            "Patch": float(np.mean(router_top_idx == 2)),
        },
        "weight_stats": eval_res["weight_stats"],
        "training_telemetry": {
            "best_epoch": train_res["best_epoch"],
            "total_epochs": train_res["total_epochs"],
            "best_val_fused": train_res["best_val_fused"],
            "runtime_seconds": train_res["training_time_seconds"],
        }
    }

    print("\n=== SEED 42 DIAGNOSTIC SUMMARY ===")
    print(f"Fused MAE:          {eval_res['fused_metrics']['mae_mw']:.2f} MW | RMSE: {eval_res['fused_metrics']['rmse_mw']:.2f} MW | R2: {eval_res['fused_metrics']['r2']:.4f}")
    print(f"GRU Standalone MAE: {eval_res['gru_metrics']['mae_mw']:.2f} MW")
    print(f"TCN Standalone MAE: {eval_res['tcn_metrics']['mae_mw']:.2f} MW")
    print(f"Patch Standalone MAE:{eval_res['patch_metrics']['mae_mw']:.2f} MW")
    print(f"Equal Ensemble MAE: {eval_res['equal_ensemble_metrics']['mae_mw']:.2f} MW")
    print(f"Oracle MAE:         {summary['test_oracle_metrics']['mae_mw']:.2f} MW (Regret: {summary['test_oracle_metrics']['regret_mean_mw']:.2f} MW)")
    print(f"Router Top Accuracy: {summary['router_top_accuracy']*100:.1f}%")
    print(f"Mean Weights:        GRU={summary['weight_stats']['mean_weights'][0]:.4f}, TCN={summary['weight_stats']['mean_weights'][1]:.4f}, Patch={summary['weight_stats']['mean_weights'][2]:.4f}")
    print(f"Weight Std Dev:      GRU={summary['weight_stats']['std_weights'][0]:.4f}, TCN={summary['weight_stats']['std_weights'][1]:.4f}, Patch={summary['weight_stats']['std_weights'][2]:.4f}")
    print(f"Mean Entropy H(w):   {summary['weight_stats']['mean_entropy']:.4f} (Max: {np.log(3):.4f}) | N_eff: {summary['weight_stats']['mean_n_eff']:.2f}")

    return summary


if __name__ == "__main__":
    run_seed42_diagnostics()
