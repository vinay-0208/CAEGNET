"""
CAEG-Net Phase 6: Router Diagnosis & Quantitative Audit
======================================================
This script performs a rigorous, multi-seed quantitative diagnosis of CAEG-Net V2's
routing mechanism, expert complementarity, horizon-dependent accuracy, and calibration.

Evaluates across all 5 canonical seeds (42, 43, 44, 45, 46):
- Horizon-wise decomposition (h = 1..24) for standalone experts, equal ensemble, V2, and oracle.
- Router calibration and difficulty sensitivity.
- Expert starvation / representation collapse analysis (internal vs standalone).
- Forecast-aware features analysis (detached forecast shape statistics).
- Patch expert regime & horizon profile.

Outputs:
- research/results/phase6_horizon_metrics.csv
- research/results/phase6_regime_routing_metrics.csv
- research/results/phase6_router_calibration.csv
- research/results/phase6_forecast_features_analysis.csv
- research/results/phase6_diagnosis_summary.json
- Diagnostic plots in research/results/phase6_plots/
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.models import CAEGNetV2, count_parameters
from research.training import train_caeg_v2, evaluate_caeg_v2


def compute_metrics(preds_mw: np.ndarray, trues_mw: np.ndarray) -> Dict[str, float]:
    err = preds_mw - trues_mw
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues_mw - np.mean(trues_mw)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(err / trues_mw)) * 100.0)
    return {
        "mae_mw": mae,
        "mse_mw2": mse,
        "rmse_mw": rmse,
        "r2": r2,
        "mape_pct": mape,
    }


def run_phase6_diagnosis(
    data_path: str = "data/Modern_PJM/pjm_load.csv",
    seeds: List[int] = [42, 43, 44, 45, 46],
    results_dir: str = "research/results",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "phase6_plots")
    os.makedirs(plots_dir, exist_ok=True)
    preds_dir = os.path.join(results_dir, "phase5a_predictions")

    print(f"=== Running Phase 6 Router Diagnosis on {device} ===")
    print(f"Seeds: {seeds}")

    pipe = prepare_research_v2_pipeline(data_path=data_path)
    scaler = pipe["scaler"]
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    # Load Ground Truth
    y_true_mw = np.load(os.path.join(preds_dir, "y_true_mw.npy"))
    num_origins, horizon = y_true_mw.shape  # 1294, 24
    print(f"Loaded ground truth: shape {y_true_mw.shape}")

    # Load Standalone Predictions from Phase 5A
    standalone_preds = {s: {} for s in seeds}
    equal_ens_preds = {s: {} for s in seeds}
    for s in seeds:
        standalone_preds[s]["gru"] = np.load(os.path.join(preds_dir, f"standalone_gru_seed_{s}.npy"))
        standalone_preds[s]["tcn"] = np.load(os.path.join(preds_dir, f"standalone_tcn_seed_{s}.npy"))
        standalone_preds[s]["patch"] = np.load(os.path.join(preds_dir, f"standalone_patch_seed_{s}.npy"))
        equal_ens_preds[s] = np.load(os.path.join(preds_dir, f"static_equal_ens_seed_{s}.npy"))

    # Containers for V2 internal evaluations across 5 seeds
    v2_fused_preds = {}
    v2_internal_preds = {s: {} for s in seeds}
    v2_weights = {}
    v2_context = {}
    v2_disagreement = {}

    print("\n--- Training / Evaluating CAEG-Net V2 across 5 seeds to capture router dynamics ---")
    for seed in seeds:
        print(f"Training V2 (Seed {seed})...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
        v2 = CAEGNetV2(forecast_aware=True, temperature=0.5).to(device)
        res = train_caeg_v2(
            model=v2,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            lr=1e-3,
            weight_decay=1e-4,
            lambda_aux=0.15,
            beta_entropy=0.001,
            temperature=0.5,
            max_epochs=35,
            patience=7,
            device=device,
            verbose=False,
        )
        eval_res = evaluate_caeg_v2(v2, dataloaders["test"], scaler, temperature=0.5, device=device)

        v2_fused_preds[seed] = eval_res["y_pred_mw"]
        v2_internal_preds[seed]["gru"] = eval_res["gru_mw"]
        v2_internal_preds[seed]["tcn"] = eval_res["tcn_mw"]
        v2_internal_preds[seed]["patch"] = eval_res["patch_mw"]
        v2_weights[seed] = eval_res["weights"]
        v2_context[seed] = eval_res["context_6d"]
        v2_disagreement[seed] = eval_res["disagreement"]

        fused_mae = eval_res["fused_metrics"]["mae_mw"]
        equal_mae = compute_metrics(equal_ens_preds[seed], y_true_mw)["mae_mw"]
        print(f"Seed {seed} complete: V2 MAE = {fused_mae:.2f} MW | Standalone Equal Ens MAE = {equal_mae:.2f} MW")

    # =========================================================================
    # 1. HORIZON-WISE DECOMPOSITION (h = 1..24)
    # =========================================================================
    print("\n--- Computing Horizon-Wise Metrics (h = 1..24) ---")
    horizon_records = []

    for h in range(horizon):
        h_step = h + 1  # 1-indexed

        # Aggregate across seeds
        gru_stand_mae_list = []
        tcn_stand_mae_list = []
        patch_stand_mae_list = []
        gru_stand_rmse_list = []
        tcn_stand_rmse_list = []
        patch_stand_rmse_list = []

        gru_int_mae_list = []
        tcn_int_mae_list = []
        patch_int_mae_list = []

        equal_mae_list = []
        equal_rmse_list = []
        v2_mae_list = []
        v2_rmse_list = []

        w_gru_list = []
        w_tcn_list = []
        w_patch_list = []

        # Oracle calculations per seed
        oracle_stand_mae_list = []
        oracle_stand_best_count = {"gru": 0, "tcn": 0, "patch": 0}

        oracle_int_mae_list = []
        oracle_int_best_count = {"gru": 0, "tcn": 0, "patch": 0}

        for s in seeds:
            yt = y_true_mw[:, h]

            # Standalone
            yg = standalone_preds[s]["gru"][:, h]
            ytcn = standalone_preds[s]["tcn"][:, h]
            yp = standalone_preds[s]["patch"][:, h]
            ye = equal_ens_preds[s][:, h]
            yv2 = v2_fused_preds[s][:, h]

            gru_stand_mae_list.append(np.mean(np.abs(yg - yt)))
            tcn_stand_mae_list.append(np.mean(np.abs(ytcn - yt)))
            patch_stand_mae_list.append(np.mean(np.abs(yp - yt)))

            gru_stand_rmse_list.append(np.sqrt(np.mean((yg - yt) ** 2)))
            tcn_stand_rmse_list.append(np.sqrt(np.mean((ytcn - yt) ** 2)))
            patch_stand_rmse_list.append(np.sqrt(np.mean((yp - yt) ** 2)))

            equal_mae_list.append(np.mean(np.abs(ye - yt)))
            equal_rmse_list.append(np.sqrt(np.mean((ye - yt) ** 2)))
            v2_mae_list.append(np.mean(np.abs(yv2 - yt)))
            v2_rmse_list.append(np.sqrt(np.mean((yv2 - yt) ** 2)))

            # Standalone oracle per origin at horizon h
            errs_stand = np.stack([np.abs(yg - yt), np.abs(ytcn - yt), np.abs(yp - yt)], axis=-1)
            best_idx = np.argmin(errs_stand, axis=-1)
            oracle_stand_mae_list.append(np.mean(np.min(errs_stand, axis=-1)))
            for idx in best_idx:
                name = ["gru", "tcn", "patch"][idx]
                oracle_stand_best_count[name] += 1

            # Internal experts
            ygi = v2_internal_preds[s]["gru"][:, h]
            ytcni = v2_internal_preds[s]["tcn"][:, h]
            ypi = v2_internal_preds[s]["patch"][:, h]

            gru_int_mae_list.append(np.mean(np.abs(ygi - yt)))
            tcn_int_mae_list.append(np.mean(np.abs(ytcni - yt)))
            patch_int_mae_list.append(np.mean(np.abs(ypi - yt)))

            errs_int = np.stack([np.abs(ygi - yt), np.abs(ytcni - yt), np.abs(ypi - yt)], axis=-1)
            oracle_int_mae_list.append(np.mean(np.min(errs_int, axis=-1)))
            best_idx_int = np.argmin(errs_int, axis=-1)
            for idx in best_idx_int:
                name = ["gru", "tcn", "patch"][idx]
                oracle_int_best_count[name] += 1

            # Weights
            w = v2_weights[s]  # [1294, 3]
            w_gru_list.append(np.mean(w[:, 0]))
            w_tcn_list.append(np.mean(w[:, 1]))
            w_patch_list.append(np.mean(w[:, 2]))

        total_obs = num_origins * len(seeds)

        horizon_records.append({
            "horizon_step": h_step,
            "standalone_gru_mae": float(np.mean(gru_stand_mae_list)),
            "standalone_tcn_mae": float(np.mean(tcn_stand_mae_list)),
            "standalone_patch_mae": float(np.mean(patch_stand_mae_list)),
            "standalone_gru_rmse": float(np.mean(gru_stand_rmse_list)),
            "standalone_tcn_rmse": float(np.mean(tcn_stand_rmse_list)),
            "standalone_patch_rmse": float(np.mean(patch_stand_rmse_list)),
            "equal_ensemble_mae": float(np.mean(equal_mae_list)),
            "equal_ensemble_rmse": float(np.mean(equal_rmse_list)),
            "v2_fused_mae": float(np.mean(v2_mae_list)),
            "v2_fused_rmse": float(np.mean(v2_rmse_list)),
            "v2_minus_equal_mae": float(np.mean(v2_mae_list) - np.mean(equal_mae_list)),
            "oracle_standalone_mae": float(np.mean(oracle_stand_mae_list)),
            "oracle_internal_mae": float(np.mean(oracle_int_mae_list)),
            "internal_gru_mae": float(np.mean(gru_int_mae_list)),
            "internal_tcn_mae": float(np.mean(tcn_int_mae_list)),
            "internal_patch_mae": float(np.mean(patch_int_mae_list)),
            "router_w_gru": float(np.mean(w_gru_list)),
            "router_w_tcn": float(np.mean(w_tcn_list)),
            "router_w_patch": float(np.mean(w_patch_list)),
            "pct_standalone_best_tcn": float((oracle_stand_best_count["tcn"] / total_obs) * 100.0),
            "pct_standalone_best_patch": float((oracle_stand_best_count["patch"] / total_obs) * 100.0),
            "pct_standalone_best_gru": float((oracle_stand_best_count["gru"] / total_obs) * 100.0),
            "pct_internal_best_tcn": float((oracle_int_best_count["tcn"] / total_obs) * 100.0),
            "pct_internal_best_patch": float((oracle_int_best_count["patch"] / total_obs) * 100.0),
            "pct_internal_best_gru": float((oracle_int_best_count["gru"] / total_obs) * 100.0),
        })

    df_horizon = pd.DataFrame(horizon_records)
    csv_horizon_path = os.path.join(results_dir, "phase6_horizon_metrics.csv")
    df_horizon.to_csv(csv_horizon_path, index=False)
    print(f"Saved horizon metrics to {csv_horizon_path}")

    # =========================================================================
    # 2. ROUTER CALIBRATION & DIFFICULTY SENSITIVITY
    # =========================================================================
    print("\n--- Auditing Router Calibration and Difficulty Regimes ---")
    calibration_records = []
    regime_records = []

    for s in seeds:
        w = v2_weights[s]  # [1294, 3]
        ctx = v2_context[s]  # [1294, 6]
        dis = v2_disagreement[s]  # [1294, 3]

        # Origin-level MAEs
        yt = y_true_mw
        mae_v2 = np.mean(np.abs(v2_fused_preds[s] - yt), axis=1)
        mae_eq = np.mean(np.abs(equal_ens_preds[s] - yt), axis=1)

        mae_int_gru = np.mean(np.abs(v2_internal_preds[s]["gru"] - yt), axis=1)
        mae_int_tcn = np.mean(np.abs(v2_internal_preds[s]["tcn"] - yt), axis=1)
        mae_int_patch = np.mean(np.abs(v2_internal_preds[s]["patch"] - yt), axis=1)

        mae_std_gru = np.mean(np.abs(standalone_preds[s]["gru"] - yt), axis=1)
        mae_std_tcn = np.mean(np.abs(standalone_preds[s]["tcn"] - yt), axis=1)
        mae_std_patch = np.mean(np.abs(standalone_preds[s]["patch"] - yt), axis=1)

        # Baseline MAE (context col 5)
        base_err = ctx[:, 5] * scale
        # Disagreement pairwise MAE (col 0 in dis)
        dis_pair = dis[:, 0] * scale
        # Volatility (context col 1)
        vol = ctx[:, 1]

        # Entropy of weights
        eps = 1e-8
        entropy_h = -np.sum(w * np.log(w + eps), axis=1)
        n_eff = np.exp(entropy_h)

        corr_int_gru = np.corrcoef(w[:, 0], -mae_int_gru)[0, 1]
        corr_int_tcn = np.corrcoef(w[:, 1], -mae_int_tcn)[0, 1]
        corr_int_patch = np.corrcoef(w[:, 2], -mae_int_patch)[0, 1]

        corr_std_gru = np.corrcoef(w[:, 0], -mae_std_gru)[0, 1]
        corr_std_tcn = np.corrcoef(w[:, 1], -mae_std_tcn)[0, 1]
        corr_std_patch = np.corrcoef(w[:, 2], -mae_std_patch)[0, 1]

        calibration_records.append({
            "seed": s,
            "corr_w_neg_error_int_gru": float(corr_int_gru),
            "corr_w_neg_error_int_tcn": float(corr_int_tcn),
            "corr_w_neg_error_int_patch": float(corr_int_patch),
            "corr_w_neg_error_std_gru": float(corr_std_gru),
            "corr_w_neg_error_std_tcn": float(corr_std_tcn),
            "corr_w_neg_error_std_patch": float(corr_std_patch),
            "mean_entropy": float(np.mean(entropy_h)),
            "mean_n_eff": float(np.mean(n_eff)),
            "corr_entropy_baseline_err": float(np.corrcoef(entropy_h, base_err)[0, 1]),
            "corr_entropy_disagreement": float(np.corrcoef(entropy_h, dis_pair)[0, 1]),
            "corr_entropy_volatility": float(np.corrcoef(entropy_h, vol)[0, 1]),
        })

        # Regime binning (Tertiaries: Low, Medium, High)
        for criterion_name, crit_vals in [("Baseline_Error", base_err), ("Disagreement", dis_pair), ("Volatility", vol)]:
            q33, q66 = np.percentile(crit_vals, [33.33, 66.67])
            regimes = {
                "Low": crit_vals <= q33,
                "Medium": (crit_vals > q33) & (crit_vals <= q66),
                "High": crit_vals > q66,
            }
            for r_name, mask in regimes.items():
                regime_records.append({
                    "seed": s,
                    "criterion": criterion_name,
                    "regime": r_name,
                    "count": int(np.sum(mask)),
                    "v2_mae": float(np.mean(mae_v2[mask])),
                    "equal_ens_mae": float(np.mean(mae_eq[mask])),
                    "v2_advantage_mw": float(np.mean(mae_eq[mask]) - np.mean(mae_v2[mask])),
                    "mean_w_gru": float(np.mean(w[mask, 0])),
                    "mean_w_tcn": float(np.mean(w[mask, 1])),
                    "mean_w_patch": float(np.mean(w[mask, 2])),
                    "mean_entropy": float(np.mean(entropy_h[mask])),
                })

    df_calibration = pd.DataFrame(calibration_records)
    csv_calib_path = os.path.join(results_dir, "phase6_router_calibration.csv")
    df_calibration.to_csv(csv_calib_path, index=False)
    print(f"Saved router calibration metrics to {csv_calib_path}")

    df_regime = pd.DataFrame(regime_records)
    csv_regime_path = os.path.join(results_dir, "phase6_regime_routing_metrics.csv")
    df_regime.to_csv(csv_regime_path, index=False)
    print(f"Saved regime routing metrics to {csv_regime_path}")

    # =========================================================================
    # 3. PATCH EXPERT INVESTIGATION
    # =========================================================================
    print("\n--- Investigating Patch Expert Complementarity ---")
    all_s_best_counts = {"gru": 0, "tcn": 0, "patch": 0}
    total_evals = 0

    for s in seeds:
        yt = y_true_mw
        err_g = np.mean(np.abs(standalone_preds[s]["gru"] - yt), axis=1)
        err_t = np.mean(np.abs(standalone_preds[s]["tcn"] - yt), axis=1)
        err_p = np.mean(np.abs(standalone_preds[s]["patch"] - yt), axis=1)

        best = np.argmin(np.stack([err_g, err_t, err_p], axis=1), axis=1)
        all_s_best_counts["gru"] += np.sum(best == 0)
        all_s_best_counts["tcn"] += np.sum(best == 1)
        all_s_best_counts["patch"] += np.sum(best == 2)
        total_evals += len(best)

    all_i_best_counts = {"gru": 0, "tcn": 0, "patch": 0}
    total_i_evals = 0
    for s in seeds:
        yt = y_true_mw
        err_g = np.mean(np.abs(v2_internal_preds[s]["gru"] - yt), axis=1)
        err_t = np.mean(np.abs(v2_internal_preds[s]["tcn"] - yt), axis=1)
        err_p = np.mean(np.abs(v2_internal_preds[s]["patch"] - yt), axis=1)

        best = np.argmin(np.stack([err_g, err_t, err_p], axis=1), axis=1)
        all_i_best_counts["gru"] += np.sum(best == 0)
        all_i_best_counts["tcn"] += np.sum(best == 1)
        all_i_best_counts["patch"] += np.sum(best == 2)
        total_i_evals += len(best)

    # =========================================================================
    # 4. FORECAST-AWARE ROUTING FEATURES ANALYSIS
    # =========================================================================
    print("\n--- Auditing Candidate Forecast-Aware Features ---")
    feature_analysis_records = []
    for s in seeds:
        yg = standalone_preds[s]["gru"]
        ytcn = standalone_preds[s]["tcn"]
        yp = standalone_preds[s]["patch"]
        yt = y_true_mw

        err_g = np.mean(np.abs(yg - yt), axis=1)
        err_t = np.mean(np.abs(ytcn - yt), axis=1)
        err_p = np.mean(np.abs(yp - yt), axis=1)

        diff_tp = err_t - err_p  # > 0 means Patch wins, < 0 means TCN wins

        slope_t = ytcn[:, -1] - ytcn[:, 0]
        slope_p = yp[:, -1] - yp[:, 0]
        range_t = np.ptp(ytcn, axis=1)
        range_p = np.ptp(yp, axis=1)

        early_dis = np.mean(np.abs(ytcn[:, :6] - yp[:, :6]), axis=1)
        late_dis = np.mean(np.abs(ytcn[:, 18:] - yp[:, 18:]), axis=1)

        corr_slope = np.corrcoef(slope_p - slope_t, diff_tp)[0, 1]
        corr_range = np.corrcoef(range_p - range_t, diff_tp)[0, 1]
        corr_early_dis = np.corrcoef(early_dis, diff_tp)[0, 1]
        corr_late_dis = np.corrcoef(late_dis, diff_tp)[0, 1]

        feature_analysis_records.append({
            "seed": s,
            "corr_slope_diff_vs_tcn_patch_advantage": float(corr_slope),
            "corr_range_diff_vs_tcn_patch_advantage": float(corr_range),
            "corr_early_disagreement_vs_advantage": float(corr_early_dis),
            "corr_late_disagreement_vs_advantage": float(corr_late_dis),
        })

    df_feats = pd.DataFrame(feature_analysis_records)
    csv_feats_path = os.path.join(results_dir, "phase6_forecast_features_analysis.csv")
    df_feats.to_csv(csv_feats_path, index=False)
    print(f"Saved forecast features analysis to {csv_feats_path}")

    # =========================================================================
    # 5. GENERATE DIAGNOSTIC PLOTS
    # =========================================================================
    print("\n--- Generating Diagnostic Plots ---")
    plt.style.use("default")
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

    # Plot 1: Horizon-wise MAE by model
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_horizon["horizon_step"], df_horizon["standalone_tcn_mae"], label="Standalone TCN", color="#1f77b4", lw=2.2)
    ax.plot(df_horizon["horizon_step"], df_horizon["standalone_patch_mae"], label="Standalone Patch", color="#2ca02c", lw=2.2)
    ax.plot(df_horizon["horizon_step"], df_horizon["standalone_gru_mae"], label="Standalone GRU", color="#d62728", lw=1.8, ls="--")
    ax.plot(df_horizon["horizon_step"], df_horizon["equal_ensemble_mae"], label="Static Equal Ensemble", color="#000000", lw=2.5)
    ax.plot(df_horizon["horizon_step"], df_horizon["v2_fused_mae"], label="CAEG-Net V2 Fused", color="#ff7f0e", lw=2.5, ls="-.")
    ax.plot(df_horizon["horizon_step"], df_horizon["oracle_standalone_mae"], label="Oracle Standalone Best", color="#9467bd", lw=2.0, ls=":")

    ax.set_title("Phase 6 Diagnostic 1: Error Decomposition Across Forecast Horizon (h = 1..24)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (Hours Ahead)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MW)", fontsize=12)
    ax.set_xticks(range(1, 25))
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase6_01_horizon_error_decomposition.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Best Standalone Expert Share by Horizon
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df_horizon["horizon_step"] - 0.25, df_horizon["pct_standalone_best_tcn"], width=0.25, label="TCN Best", color="#1f77b4")
    ax.bar(df_horizon["horizon_step"], df_horizon["pct_standalone_best_patch"], width=0.25, label="Patch Best", color="#2ca02c")
    ax.bar(df_horizon["horizon_step"] + 0.25, df_horizon["pct_standalone_best_gru"], width=0.25, label="GRU Best", color="#d62728")
    ax.set_title("Phase 6 Diagnostic 2: Optimal Expert Selection Frequency by Horizon", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (Hours Ahead)", fontsize=12)
    ax.set_ylabel("% Origins Where Expert is Best Standalone", fontsize=12)
    ax.set_xticks(range(1, 25))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase6_02_horizon_best_expert_share.png"), dpi=300)
    plt.close(fig)

    # Plot 3: V2 Fused vs Equal Ensemble Advantage by Horizon
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in df_horizon["v2_minus_equal_mae"]]
    ax.bar(df_horizon["horizon_step"], -df_horizon["v2_minus_equal_mae"], color=colors, width=0.6)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_title("Phase 6 Diagnostic 3: V2 Advantage over Equal Ensemble by Horizon (Positive = V2 Wins)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step h (Hours Ahead)", fontsize=12)
    ax.set_ylabel("Equal MAE - V2 MAE (MW)", fontsize=12)
    ax.set_xticks(range(1, 25))
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase6_03_v2_advantage_by_horizon.png"), dpi=300)
    plt.close(fig)

    # Plot 4: Regime Advantage
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    criteria = ["Baseline_Error", "Disagreement", "Volatility"]
    order = ["Low", "Medium", "High"]
    for idx, crit in enumerate(criteria):
        sub_df = df_regime[df_regime["criterion"] == crit]
        mean_adv = sub_df.groupby("regime")["v2_advantage_mw"].mean().reindex(order)
        std_adv = sub_df.groupby("regime")["v2_advantage_mw"].std().reindex(order)

        axes[idx].bar(order, mean_adv, yerr=std_adv, capsize=5, color=["#1f77b4", "#aec7e8", "#ff7f0e"])
        axes[idx].axhline(0, color="black", lw=1.0, ls="--")
        axes[idx].set_title(f"Advantage by {crit.replace('_', ' ')}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Regime Tertiary", fontsize=11)
        if idx == 0:
            axes[idx].set_ylabel("V2 Gain over Equal Ensemble (MW)", fontsize=11)

    fig.suptitle("Phase 6 Diagnostic 4: CAEG-Net V2 Advantage Across Difficulty Regimes", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase6_04_regime_advantage_comparison.png"), dpi=300)
    plt.close(fig)

    # Plot 5: Representation Collapse Under Joint Co-Training
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ["GRU", "TCN", "Patch"]
    stand_maes = [float(np.mean(df_horizon["standalone_gru_mae"])), float(np.mean(df_horizon["standalone_tcn_mae"])), float(np.mean(df_horizon["standalone_patch_mae"]))]
    int_maes = [float(np.mean(df_horizon["internal_gru_mae"])), float(np.mean(df_horizon["internal_tcn_mae"])), float(np.mean(df_horizon["internal_patch_mae"]))]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, stand_maes, width, label="Standalone Trained", color="#1f77b4")
    ax.bar(x + width/2, int_maes, width, label="Co-Trained inside V2 (lambda=0.15)", color="#ff7f0e")

    ax.set_ylabel("Overall Mean Absolute Error (MW)", fontsize=12)
    ax.set_title("Phase 6 Diagnostic 5: Representation Collapse Under Joint Co-Training", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(loc="upper left")

    for i in range(len(models)):
        deg = int_maes[i] - stand_maes[i]
        sign = "+" if deg > 0 else ""
        ax.text(x[i] + width/2, int_maes[i] + 5, f"{sign}{deg:.1f} MW", ha="center", fontweight="bold", color="#d62728" if deg > 0 else "#2ca02c")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "phase6_05_representation_collapse.png"), dpi=300)
    plt.close(fig)

    print(f"Generated 5 diagnostic plots in {plots_dir}")

    # =========================================================================
    # 6. COMPILE COMPLETE DIAGNOSTIC SUMMARY JSON
    # =========================================================================
    summary = {
        "diagnostic_phase": "Phase 6 Router Diagnosis",
        "seeds": seeds,
        "overall_metrics": {
            "standalone_tcn_mae": float(np.mean(df_horizon["standalone_tcn_mae"])),
            "standalone_patch_mae": float(np.mean(df_horizon["standalone_patch_mae"])),
            "standalone_gru_mae": float(np.mean(df_horizon["standalone_gru_mae"])),
            "static_equal_ensemble_mae": float(np.mean(df_horizon["equal_ensemble_mae"])),
            "v2_fused_mae": float(np.mean(df_horizon["v2_fused_mae"])),
            "oracle_standalone_mae": float(np.mean(df_horizon["oracle_standalone_mae"])),
            "oracle_headroom_over_equal_mw": float(np.mean(df_horizon["equal_ensemble_mae"]) - np.mean(df_horizon["oracle_standalone_mae"])),
            "internal_tcn_mae": float(np.mean(df_horizon["internal_tcn_mae"])),
            "internal_patch_mae": float(np.mean(df_horizon["internal_patch_mae"])),
            "internal_gru_mae": float(np.mean(df_horizon["internal_gru_mae"])),
        },
        "horizon_breakdown": {
            "near_horizon_h1_h6": {
                "tcn_mae": float(df_horizon.loc[0:5, "standalone_tcn_mae"].mean()),
                "patch_mae": float(df_horizon.loc[0:5, "standalone_patch_mae"].mean()),
                "equal_mae": float(df_horizon.loc[0:5, "equal_ensemble_mae"].mean()),
                "v2_mae": float(df_horizon.loc[0:5, "v2_fused_mae"].mean()),
                "tcn_best_pct": float(df_horizon.loc[0:5, "pct_standalone_best_tcn"].mean()),
                "patch_best_pct": float(df_horizon.loc[0:5, "pct_standalone_best_patch"].mean()),
            },
            "mid_horizon_h7_h18": {
                "tcn_mae": float(df_horizon.loc[6:17, "standalone_tcn_mae"].mean()),
                "patch_mae": float(df_horizon.loc[6:17, "standalone_patch_mae"].mean()),
                "equal_mae": float(df_horizon.loc[6:17, "equal_ensemble_mae"].mean()),
                "v2_mae": float(df_horizon.loc[6:17, "v2_fused_mae"].mean()),
                "tcn_best_pct": float(df_horizon.loc[6:17, "pct_standalone_best_tcn"].mean()),
                "patch_best_pct": float(df_horizon.loc[6:17, "pct_standalone_best_patch"].mean()),
            },
            "far_horizon_h19_h24": {
                "tcn_mae": float(df_horizon.loc[18:23, "standalone_tcn_mae"].mean()),
                "patch_mae": float(df_horizon.loc[18:23, "standalone_patch_mae"].mean()),
                "equal_mae": float(df_horizon.loc[18:23, "equal_ensemble_mae"].mean()),
                "v2_mae": float(df_horizon.loc[18:23, "v2_fused_mae"].mean()),
                "tcn_best_pct": float(df_horizon.loc[18:23, "pct_standalone_best_tcn"].mean()),
                "patch_best_pct": float(df_horizon.loc[18:23, "pct_standalone_best_patch"].mean()),
            }
        },
        "regime_breakdown": {
            "high_disagreement_v2_gain": float(df_regime[(df_regime["criterion"] == "Disagreement") & (df_regime["regime"] == "High")]["v2_advantage_mw"].mean()),
            "low_disagreement_v2_gain": float(df_regime[(df_regime["criterion"] == "Disagreement") & (df_regime["regime"] == "Low")]["v2_advantage_mw"].mean()),
            "high_baseline_v2_gain": float(df_regime[(df_regime["criterion"] == "Baseline_Error") & (df_regime["regime"] == "High")]["v2_advantage_mw"].mean()),
            "low_baseline_v2_gain": float(df_regime[(df_regime["criterion"] == "Baseline_Error") & (df_regime["regime"] == "Low")]["v2_advantage_mw"].mean()),
        },
        "representation_collapse_deltas": {
            "gru_degradation_mw": float(np.mean(df_horizon["internal_gru_mae"]) - np.mean(df_horizon["standalone_gru_mae"])),
            "tcn_degradation_mw": float(np.mean(df_horizon["internal_tcn_mae"]) - np.mean(df_horizon["standalone_tcn_mae"])),
            "patch_degradation_mw": float(np.mean(df_horizon["internal_patch_mae"]) - np.mean(df_horizon["standalone_patch_mae"])),
        },
        "router_weights_mean": {
            "w_patch": float(np.mean(df_horizon["router_w_patch"])),
            "w_tcn": float(np.mean(df_horizon["router_w_tcn"])),
            "w_gru": float(np.mean(df_horizon["router_w_gru"])),
        },
        "calibration_correlations": {
            "mean_corr_w_neg_error_int_patch": float(df_calibration["corr_w_neg_error_int_patch"].mean()),
            "mean_corr_w_neg_error_int_tcn": float(df_calibration["corr_w_neg_error_int_tcn"].mean()),
            "mean_corr_w_neg_error_int_gru": float(df_calibration["corr_w_neg_error_int_gru"].mean()),
        }
    }

    summary_json_path = os.path.join(results_dir, "phase6_diagnosis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved diagnosis summary JSON to {summary_json_path}")
    print("\n=== Phase 6 Router Diagnosis Complete ===")


if __name__ == "__main__":
    run_phase6_diagnosis()
