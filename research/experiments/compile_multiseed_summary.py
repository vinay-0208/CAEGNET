"""
Compile 5-seed research results, evaluate all saved checkpoints,
and generate unified model comparison table and prediction caches.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.models import ResearchCAEGNet
from research.data import prepare_research_pipeline, build_research_dataloaders
from evaluate import evaluate_model_on_loader

SEEDS = [42, 123, 999, 2024, 3407]


def compile_summary():
    print("=" * 80)
    print("COMPILING 5-SEED RESEARCH BENCHMARK SUMMARY & CACHES")
    print("=" * 80)

    res_dir = os.path.join(repo_root, "research", "results")
    seeds_csv = os.path.join(res_dir, "five_seed_results.csv")
    df_seeds = pd.read_csv(seeds_csv)

    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    scaler = data["scaler"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_base_dir = os.path.join(repo_root, "research", "checkpoints", "multiseed")
    cache_dict = {
        "y_test_true_raw": data["windows"]["test"]["Y"] * scaler.scale_[0] + scaler.mean_[0]
    }

    variants = [
        {
            "id": "CAEG_Net_V1_Canonical",
            "name": "CAEG-Net V1 (4D Operational)",
            "context_type": "4d",
            "base_context_dim": 4,
            "forecast_aware": False,
        },
        {
            "id": "CAEG_Net_Forecast_Aware",
            "name": "CAEG-Net Forecast-Aware (Variant D)",
            "context_type": "4d",
            "base_context_dim": 4,
            "forecast_aware": True,
        },
        {
            "id": "CAEG_Net_Calendar_Context",
            "name": "CAEG-Net Calendar-Context (Variant C)",
            "context_type": "8d",
            "base_context_dim": 8,
            "forecast_aware": False,
        },
        {
            "id": "CAEG_Net_No_Recent_Error",
            "name": "CAEG-Net No Recent Error (Variant B)",
            "context_type": "3d",
            "base_context_dim": 3,
            "forecast_aware": False,
        },
    ]

    print("Extracting predictions and routing weights from saved checkpoints...")
    for var in variants:
        var_id = var["id"]
        for seed in SEEDS:
            ckpt_path = os.path.join(ckpt_base_dir, f"{var_id}_seed_{seed}.pt")
            model = ResearchCAEGNet(
                input_dim=1,
                horizon=24,
                base_context_dim=var["base_context_dim"],
                forecast_aware=var["forecast_aware"],
                latent_context_dim=16,
                dropout=0.1,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            loaders = build_research_dataloaders(
                data, context_type=var["context_type"], batch_size=64, seed=seed
            )
            eval_res = evaluate_model_on_loader(model, loaders["test"], scaler)
            cache_dict[f"pred_{var_id}_seed_{seed}"] = eval_res["y_pred_raw"]
            if "weights" in eval_res:
                cache_dict[f"weights_{var_id}_seed_{seed}"] = eval_res["weights"]

    cache_path = os.path.join(res_dir, "research_multiseed_cache.npz")
    np.savez_compressed(cache_path, **cache_dict)
    print(f"Saved complete prediction/weight caches to: {cache_path}")

    # Aggregate 5-seed statistics
    summary_rows = []
    for var in variants:
        var_id = var["id"]
        sub = df_seeds[df_seeds["variant_id"] == var_id]
        name = sub["model_name"].iloc[0]
        params = sub["parameters"].iloc[0]
        mae_m, mae_s = sub["MAE_MW"].mean(), sub["MAE_MW"].std()
        rmse_m, rmse_s = sub["RMSE_MW"].mean(), sub["RMSE_MW"].std()
        r2_m, r2_s = sub["R2"].mean(), sub["R2"].std()
        mape_m, mape_s = sub["MAPE_pct"].mean(), sub["MAPE_pct"].std()
        t_m = sub["train_time_s"].mean()

        summary_rows.append({
            "Model": name,
            "Variant_ID": var_id,
            "Parameters": params,
            "MAE (MW)": f"{mae_m:.2f} ± {mae_s:.2f}",
            "RMSE (MW)": f"{rmse_m:.2f} ± {rmse_s:.2f}",
            "R^2": f"{r2_m:.4f} ± {r2_s:.4f}",
            "MAPE (%)": f"{mape_m:.2f} ± {mape_s:.2f}",
            "MAE_Mean": round(mae_m, 2),
            "MAE_Std": round(mae_s, 2),
            "RMSE_Mean": round(rmse_m, 2),
            "RMSE_Std": round(rmse_s, 2),
            "R2_Mean": round(r2_m, 4),
            "R2_Std": round(r2_s, 4),
            "MAPE_Mean": round(mape_m, 2),
            "MAPE_Std": round(mape_s, 2),
            "Avg_Train_Time_s": round(t_m, 1),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Combine with canonical baseline
    canonical_summary_path = os.path.join(repo_root, "results", "phase5_summary.csv")
    unified_rows = []
    if os.path.isfile(canonical_summary_path):
        df_canon = pd.read_csv(canonical_summary_path)
        for _, r in df_canon.iterrows():
            unified_rows.append({
                "Model": f"[Baseline] {r['Model']}",
                "Parameters": r["Params"],
                "MAE (MW)": r["MAE (MW)"],
                "RMSE (MW)": r["RMSE (MW)"],
                "R^2": r["R^2"],
                "MAPE (%)": r["MAPE (%)"],
                "MAE_Mean": r["MAE Mean"],
                "MAE_Std": r["MAE Std"],
                "RMSE_Mean": r["RMSE Mean"],
                "RMSE_Std": r["RMSE Std"],
                "R2_Mean": r["R2 Mean"],
                "R2_Std": r["R2 Std"],
                "Track": "Canonical Baseline",
            })

    for _, r in df_summary.iterrows():
        unified_rows.append({
            "Model": f"[Research] {r['Model']}",
            "Parameters": r["Parameters"],
            "MAE (MW)": r["MAE (MW)"],
            "RMSE (MW)": r["RMSE (MW)"],
            "R^2": r["R^2"],
            "MAPE (%)": r["MAPE (%)"],
            "MAE_Mean": r["MAE_Mean"],
            "MAE_Std": r["MAE_Std"],
            "RMSE_Mean": r["RMSE_Mean"],
            "RMSE_Std": r["RMSE_Std"],
            "R2_Mean": r["R2_Mean"],
            "R2_Std": r["R2_Std"],
            "Track": "Research Track",
        })

    df_unified = pd.DataFrame(unified_rows)
    unified_csv = os.path.join(res_dir, "model_comparison.csv")
    df_unified.to_csv(unified_csv, index=False)
    print(f"Saved unified comparison to: {unified_csv}")

    print("\n" + "=" * 90)
    print("5-SEED RESEARCH TRACK SUMMARY TABLE:")
    print("=" * 90)
    print(df_summary[["Model", "Parameters", "MAE (MW)", "RMSE (MW)", "R^2", "MAPE (%)", "Avg_Train_Time_s"]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    compile_summary()
