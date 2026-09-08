"""
CAEG-Net Research Track: Phase 5 Multi-Seed Rigorous Benchmark
==============================================================
Runs the shortlisted research variants across the canonical 5 seeds:
SEEDS = [42, 123, 999, 2024, 3407]

Models:
1. CAEG-Net V1 (Canonical 4D Operational Context)
2. CAEG-Net Forecast-Aware Routing (Variant D: 4D Operational + 3D Disagreement)
3. CAEG-Net Calendar-Context (Variant C: 8D Context)
4. CAEG-Net No Recent Error (Variant B: 3D Context)

Integrates with canonical baselines:
- Naive-24
- Standalone LSTM, TCN, CNN
- Static Equal Ensemble
- Standard Input-MoE
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
from evaluate import evaluate_model_on_loader

SEEDS = [42, 123, 999, 2024, 3407]


def run_multiseed_benchmark():
    print("=" * 80)
    print("CAEG-NET RESEARCH TRACK: 5-SEED RIGOROUS BENCHMARK EVALUATION")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = prepare_research_pipeline(os.path.join(repo_root, "data/Modern_PJM/pjm_load.csv"))
    scaler = data["scaler"]

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

    all_seed_records = []
    predictions_cache = {
        "y_test_true_raw": data["windows"]["test"]["Y"] * scaler.scale_[0] + scaler.mean_[0]
    }
    routing_weights_cache = {}

    ckpt_base_dir = os.path.join(repo_root, "research", "checkpoints", "multiseed")
    os.makedirs(ckpt_base_dir, exist_ok=True)

    for var in variants:
        var_id = var["id"]
        var_name = var["name"]
        print("\n" + "#" * 80)
        print(f"EVALUATING VARIANT ACROSS 5 SEEDS: {var_name}")
        print("#" * 80)

        for seed in SEEDS:
            print(f"  >>> Seed {seed} ... ", end="", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()

            loaders = build_research_dataloaders(
                data, context_type=var["context_type"], batch_size=64, seed=seed
            )

            model = ResearchCAEGNet(
                input_dim=1,
                horizon=24,
                base_context_dim=var["base_context_dim"],
                forecast_aware=var["forecast_aware"],
                latent_context_dim=16,
                dropout=0.1,
            ).to(device)

            params = count_parameters(model)["total_trainable"]
            criterion = create_criterion("mse")
            optimizer, scheduler = create_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-4)

            ckpt_path = os.path.join(ckpt_base_dir, f"{var_id}_seed_{seed}.pt")

            t0 = time.time()
            train_res = train_model(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                max_epochs=25,
                patience=6,
                checkpoint_path=ckpt_path,
                device=device,
                model_name=f"{var_id}_{seed}",
                verbose=False,
            )
            t_train = time.time() - t0

            # Evaluate with restored best model
            best_model = train_res["model"]
            eval_res = evaluate_model_on_loader(best_model, loaders["test"], scaler)
            m = eval_res["metrics"]

            peak_vram = 0.0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            record = {
                "variant_id": var_id,
                "model_name": var_name,
                "seed": seed,
                "MAE_MW": round(m["MAE"], 2),
                "RMSE_MW": round(m["RMSE"], 2),
                "MSE_MW2": round(m["MSE"], 2),
                "MAPE_pct": round(m["MAPE"], 2),
                "R2": round(m["R2"], 4),
                "parameters": params,
                "actual_epochs": len(train_res["train_losses"]),
                "best_epoch": train_res["best_epoch"],
                "best_val_mse": round(train_res["best_val_loss"], 5),
                "train_time_s": round(t_train, 1),
                "peak_vram_mb": round(peak_vram, 1),
            }
            all_seed_records.append(record)

            predictions_cache[f"{var_id}_seed_{seed}"] = eval_res["y_pred_raw"]
            if "weights" in eval_res:
                routing_weights_cache[f"{var_id}_seed_{seed}"] = eval_res["weights"]

            print(f"Done in {t_train:.1f}s | Best Ep: {train_res['best_epoch']} | Val MSE: {train_res['best_val_loss']:.5f} | Test MAE: {m['MAE']:.2f} MW | RMSE: {m['RMSE']:.2f} MW | R2: {m['R2']:.4f}")

    # Compile 5-seed results
    df_seeds = pd.DataFrame(all_seed_records)
    res_dir = os.path.join(repo_root, "research", "results")
    os.makedirs(res_dir, exist_ok=True)
    seeds_csv = os.path.join(res_dir, "five_seed_results.csv")
    df_seeds.to_csv(seeds_csv, index=False)
    print(f"\nSaved per-seed results to: {seeds_csv}")

    # Save predictions and routing caches
    cache_path = os.path.join(res_dir, "research_multiseed_cache.npz")
    np.savez_compressed(cache_path, **predictions_cache, **routing_weights_cache)
    print(f"Saved prediction and routing caches to: {cache_path}")

    # Compute multi-seed summary statistics
    summary_rows = []
    for var_id, group in df_seeds.groupby("variant_id"):
        name = group["model_name"].iloc[0]
        params = group["parameters"].iloc[0]
        mae_m, mae_s = group["MAE_MW"].mean(), group["MAE_MW"].std()
        rmse_m, rmse_s = group["RMSE_MW"].mean(), group["RMSE_MW"].std()
        r2_m, r2_s = group["R2"].mean(), group["R2"].std()
        mape_m, mape_s = group["MAPE_pct"].mean(), group["MAPE_pct"].std()
        t_m = group["train_time_s"].mean()

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

    # Merge with canonical baseline summary for unified comparison
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
    print(f"Saved unified model comparison to: {unified_csv}")

    print("\n" + "=" * 85)
    print("5-SEED RESEARCH TRACK SUMMARY TABLE:")
    print("=" * 85)
    print(df_summary[["Model", "Parameters", "MAE (MW)", "RMSE (MW)", "R^2", "MAPE (%)", "Avg_Train_Time_s"]].to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    run_multiseed_benchmark()
