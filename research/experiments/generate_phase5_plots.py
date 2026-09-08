"""
CAEG-Net Research Track — Phase 5 Publication Visualizations
============================================================
Generates the 9 publication-grade figures mandated by Phase 5:
1. Model MAE across 5 seeds (mean +- std bar plot with error bars)
2. Model RMSE across 5 seeds
3. Model R2 across 5 seeds
4. Seed-wise V2 vs strongest baselines (grouped bar plot)
5. Recent-error ablation comparison (Full V2 vs No Recent Error)
6. Routing weight distributions & stability across seeds
7. Oracle headroom & captured headroom across seeds
8. Difficulty-regime performance across seeds (Volatility & Disagreement)
9. Training stability & convergence across models

Saves all figures to: research/results/phase5_plots/
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")  # Headless execution
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = "research/results/phase5_plots"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_DIR = "research/results"
SEED_CSV = os.path.join(RESULTS_DIR, "phase5_seed_results.csv")
COMP_CSV = os.path.join(RESULTS_DIR, "phase5_model_comparison.csv")
REC_CSV = os.path.join(RESULTS_DIR, "phase5_recent_error_ablation.csv")
ROUT_CSV = os.path.join(RESULTS_DIR, "phase5_routing_reproducibility.csv")
COMPL_CSV = os.path.join(RESULTS_DIR, "phase5_expert_complementarity.csv")
REG_CSV = os.path.join(RESULTS_DIR, "phase5_regime_analysis.csv")


def generate_all_plots():
    print(f"=== Generating Phase 5 Publication Visualizations in {OUT_DIR} ===")

    df_seeds = pd.read_csv(SEED_CSV)
    df_comp = pd.read_csv(COMP_CSV)
    df_rec = pd.read_csv(REC_CSV)
    df_rout = pd.read_csv(ROUT_CSV)
    df_compl = pd.read_csv(COMPL_CSV)
    df_reg = pd.read_csv(REG_CSV)

    # Color palette
    colors = {
        "Naive-24": "#8c564b",
        "Standalone_GRU": "#1f77b4",
        "Standalone_TCN": "#ff7f0e",
        "Standalone_Patch": "#9467bd",
        "Static_Equal_Ensemble": "#2ca02c",
        "Standard_Input_MoE": "#e377c2",
        "CAEG_Net_V2_Full": "#17becf",
        "CAEG_Net_V2_NoRecentError": "#bcbd22",
        "CAEG_Net_V1_Canonical_Reference": "#d62728",
    }

    # -------------------------------------------------------------
    # Plot 1: Model MAE across 5 Seeds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    models = df_comp["model"].values
    maes = df_comp["mae_mean"].values
    stds = df_comp["mae_std"].values
    x = np.arange(len(models))

    bar_colors = [colors.get(m, "#7f7f7f") for m in models]
    bars = ax.bar(x, maes, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.8)

    for bar, m_val, s_val in zip(bars, maes, stds):
        h = bar.get_height()
        ax.annotate(f"{m_val:.1f}\n(+-{s_val:.1f})", (bar.get_x() + bar.get_width()/2, h + s_val + 5),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    clean_labels = [m.replace("Standalone_", "").replace("CAEG_Net_", "").replace("_Canonical_Reference", " [V1 Ref]") for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=35, ha="right")
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("CAEG-Net Phase 5: Mean Test MAE Across 5 Independent Random Seeds (MW)")
    ax.set_ylim(0, max(maes) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")
    p1 = os.path.join(OUT_DIR, "01_model_mae_across_seeds.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # Plot 2: Model RMSE across 5 Seeds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    rmses = df_comp["rmse_mean"].values
    rmse_stds = df_comp["rmse_std"].values

    bars = ax.bar(x, rmses, yerr=rmse_stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.8)
    for bar, m_val, s_val in zip(bars, rmses, rmse_stds):
        h = bar.get_height()
        ax.annotate(f"{m_val:.1f}\n(+-{s_val:.1f})", (bar.get_x() + bar.get_width()/2, h + s_val + 7),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=35, ha="right")
    ax.set_ylabel("Test RMSE (MW)")
    ax.set_title("CAEG-Net Phase 5: Mean Test RMSE Across 5 Independent Random Seeds (MW)")
    ax.set_ylim(0, max(rmses) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")
    p2 = os.path.join(OUT_DIR, "02_model_rmse_across_seeds.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # Plot 3: Model R2 across 5 Seeds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    r2s = df_comp["r2_mean"].values
    r2_stds = df_comp["r2_std"].values

    bars = ax.bar(x, r2s, yerr=r2_stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.8)
    for bar, m_val, s_val in zip(bars, r2s, r2_stds):
        h = bar.get_height()
        ax.annotate(f"{m_val:.4f}", (bar.get_x() + bar.get_width()/2, h + s_val + 0.01),
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, rotation=35, ha="right")
    ax.set_ylabel("Coefficient of Determination R^2")
    ax.set_title("CAEG-Net Phase 5: Forecast R^2 Across 5 Independent Random Seeds")
    ax.set_ylim(0.70, 0.92)
    ax.grid(True, alpha=0.3, axis="y")
    p3 = os.path.join(OUT_DIR, "03_model_r2_across_seeds.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # Plot 4: Seed-Wise V2 vs Strongest Baselines
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    sub_models = ["CAEG_Net_V2_Full", "CAEG_Net_V2_NoRecentError", "Standard_Input_MoE", "Static_Equal_Ensemble", "Standalone_Patch"]
    seeds = sorted(df_seeds["seed"].unique())
    x_seeds = np.arange(len(seeds))
    w = 0.16

    for i, m in enumerate(sub_models):
        m_vals = df_seeds[df_seeds["model"] == m].sort_values("seed")["test_mae_mw"].values
        lbl = m.replace("Standalone_", "").replace("CAEG_Net_", "")
        ax.bar(x_seeds + (i - 2)*w, m_vals, w, label=lbl, color=colors[m], alpha=0.85, edgecolor="black", lw=0.5)

    ax.set_xticks(x_seeds)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("CAEG-Net Phase 5: Seed-by-Seed Performance Comparison Against Key Baselines")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.set_ylim(220, 310)
    ax.grid(True, alpha=0.3, axis="y")
    p4 = os.path.join(OUT_DIR, "04_seedwise_v2_vs_strongest_baseline.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------
    # Plot 5: Recent-Error Ablation Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    full_vals = df_rec["mae_full_v2_mw"].values
    norec_vals = df_rec["mae_no_recent_error_mw"].values
    deltas = df_rec["delta_mw"].values  # Full - NoRecent

    x_s = np.arange(len(seeds))
    w_b = 0.35
    ax1.bar(x_s - w_b/2, full_vals, w_b, label="V2 Full (with Ridge feedback)", color=colors["CAEG_Net_V2_Full"], alpha=0.85)
    ax1.bar(x_s + w_b/2, norec_vals, w_b, label="V2 No Recent Error", color=colors["CAEG_Net_V2_NoRecentError"], alpha=0.85)
    ax1.set_xticks(x_s)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylabel("Test MAE (MW)")
    ax1.set_title("Per-Seed MAE: Full vs No Recent Error")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(230, 275)
    ax1.grid(True, alpha=0.3, axis="y")

    # Delta bar plot
    delta_colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]
    ax2.bar(x_s, deltas, color=delta_colors, alpha=0.85, width=0.5, edgecolor="black", lw=0.7)
    ax2.axhline(0, color="black", linestyle="--", lw=1)
    ax2.set_xticks(x_s)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel("Delta MAE (MW) [Full - No Recent Error]")
    ax2.set_title(f"Ablation Impact (Mean Delta: {np.mean(deltas):+.2f} MW)")
    ax2.grid(True, alpha=0.3, axis="y")
    fig.suptitle("CAEG-Net Phase 5: Expanding-Ridge Recent Error Feedback Ablation", y=1.02)
    p5 = os.path.join(OUT_DIR, "05_recent_error_ablation_comparison.png")
    fig.savefig(p5)
    plt.close(fig)
    print(f"Saved: {p5}")

    # -------------------------------------------------------------
    # Plot 6: Routing Weight Distribution Across Seeds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    expert_names = ["Patch", "TCN", "GRU"]
    cols_mean = ["mean_w_Patch", "mean_w_TCN", "mean_w_GRU"]
    cols_std = ["std_w_Patch", "std_w_TCN", "std_w_GRU"]
    e_colors = ["#9467bd", "#ff7f0e", "#1f77b4"]

    x_e = np.arange(len(expert_names))
    means = [df_rout[c].mean() for c in cols_mean]
    stds = [df_rout[c].std() for c in cols_mean]

    bars = ax.bar(x_e, means, yerr=stds, capsize=6, color=e_colors, alpha=0.85, edgecolor="black", lw=0.8, width=0.45)
    for bar, m_val, s_val in zip(bars, means, stds):
        h = bar.get_height()
        ax.annotate(f"{m_val:.3f}\n(+-{s_val:.4f})", (bar.get_x() + bar.get_width()/2, h + 0.03),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(1.0/3.0, color="gray", linestyle="--", alpha=0.7, label="Uniform Weight (0.333)")
    ax.set_xticks(x_e)
    ax.set_xticklabels(expert_names)
    ax.set_ylabel("Mean Expert Routing Weight")
    ax.set_title("CAEG-Net V2: Routing Weight Allocations & Stability Across 5 Seeds")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    p6 = os.path.join(OUT_DIR, "06_routing_weights_distribution_across_seeds.png")
    fig.savefig(p6)
    plt.close(fig)
    print(f"Saved: {p6}")

    # -------------------------------------------------------------
    # Plot 7: Oracle Headroom Across Seeds
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x_c = np.arange(len(seeds))
    w_c = 0.25

    best_single = df_compl["standalone_best_mae_mw"].values
    fused_v2 = df_compl["fused_v2_mae_mw"].values
    oracle = df_compl["oracle_mae_mw"].values
    cap_pct = df_compl["captured_headroom_pct"].values

    ax.bar(x_c - w_c, best_single, w_c, label="Standalone Best (Patch)", color="#9467bd", alpha=0.85)
    ax.bar(x_c, fused_v2, w_c, label="CAEG-Net V2 Fused", color="#17becf", alpha=0.85)
    ax.bar(x_c + w_c, oracle, w_c, label="Theoretical Oracle", color="#e377c2", alpha=0.85)

    for i, c_p in enumerate(cap_pct):
        ax.annotate(f"Cap: {c_p:.1f}%", (i, fused_v2[i] - 18),
                    ha="center", fontsize=8.5, fontweight="bold", color="darkblue")

    ax.set_xticks(x_c)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("CAEG-Net Phase 5: Oracle Headroom & Captured Headroom Across Seeds")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(200, 310)
    ax.grid(True, alpha=0.3, axis="y")
    p7 = os.path.join(OUT_DIR, "07_oracle_headroom_across_seeds.png")
    fig.savefig(p7)
    plt.close(fig)
    print(f"Saved: {p7}")

    # -------------------------------------------------------------
    # Plot 8: Difficulty Regime Performance
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sub_dis = df_reg[df_reg["regime_type"] == "Disagreement"].copy()
    sub_vol = df_reg[df_reg["regime_type"] == "Volatility"].copy()

    order = {"Low": 0, "Medium": 1, "High": 2}
    sub_dis["order"] = sub_dis["regime_level"].map(order)
    sub_dis = sub_dis.sort_values("order")
    sub_vol["order"] = sub_vol["regime_level"].map(order)
    sub_vol = sub_vol.sort_values("order")

    x_r = np.arange(3)
    w_r = 0.25

    # Disagreement plot
    ax1.bar(x_r - w_r, sub_dis["patch_mae_mw_mean"], w_r, label="Patch Expert", color="#9467bd", alpha=0.85)
    ax1.bar(x_r, sub_dis["equal_ens_mae_mw_mean"], w_r, label="Equal Ensemble", color="#2ca02c", alpha=0.85)
    ax1.bar(x_r + w_r, sub_dis["v2_mae_mw_mean"], w_r, label="CAEG-Net V2", color="#17becf", alpha=0.85)
    ax1.set_xticks(x_r)
    ax1.set_xticklabels(["Low", "Medium", "High"])
    ax1.set_xlabel("Inter-Expert Disagreement Regime")
    ax1.set_ylabel("Test MAE (MW)")
    ax1.set_title("Disagreement Regimes")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3, axis="y")

    # Volatility plot
    ax2.bar(x_r - w_r, sub_vol["patch_mae_mw_mean"], w_r, label="Patch Expert", color="#9467bd", alpha=0.85)
    ax2.bar(x_r, sub_vol["equal_ens_mae_mw_mean"], w_r, label="Equal Ensemble", color="#2ca02c", alpha=0.85)
    ax2.bar(x_r + w_r, sub_vol["v2_mae_mw_mean"], w_r, label="CAEG-Net V2", color="#17becf", alpha=0.85)
    ax2.set_xticks(x_r)
    ax2.set_xticklabels(["Low", "Medium", "High"])
    ax2.set_xlabel("Short-Term Volatility Regime")
    ax2.set_ylabel("Test MAE (MW)")
    ax2.set_title("Volatility Regimes")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CAEG-Net Phase 5: Robustness Across Forecast Difficulty & Volatility Regimes", y=1.02)
    p8 = os.path.join(OUT_DIR, "08_difficulty_regime_performance.png")
    fig.savefig(p8)
    plt.close(fig)
    print(f"Saved: {p8}")

    # -------------------------------------------------------------
    # Plot 9: Training Stability & Convergence
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    v2_seeds = df_seeds[df_seeds["model"] == "CAEG_Net_V2_Full"]
    best_eps = v2_seeds["best_epoch"].values
    tot_eps = v2_seeds["actual_epochs"].values
    runtimes = v2_seeds["training_time_s"].values

    x_s5 = np.arange(len(seeds))
    w_e = 0.35
    ax1.bar(x_s5 - w_e/2, best_eps, w_e, label="Best Epoch (Early Stopping)", color="#1f77b4", alpha=0.85)
    ax1.bar(x_s5 + w_e/2, tot_eps, w_e, label="Total Epochs Trained", color="#aec7e8", alpha=0.85)
    ax1.set_xticks(x_s5)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylabel("Epoch Count")
    ax1.set_title(f"Convergence Epochs (Mean Best: {np.mean(best_eps):.1f})")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x_s5, runtimes, color="#2ca02c", alpha=0.85, width=0.45, edgecolor="black", lw=0.7)
    ax2.set_xticks(x_s5)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel("Training Time (seconds)")
    ax2.set_title(f"CUDA Runtime per Seed (Mean: {np.mean(runtimes):.1f}s)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CAEG-Net V2 Phase 5: Training Stability & Convergence Characteristics", y=1.02)
    p9 = os.path.join(OUT_DIR, "09_training_stability_and_convergence.png")
    fig.savefig(p9)
    plt.close(fig)
    print(f"Saved: {p9}")

    print("All 9 publication figures generated successfully!")


if __name__ == "__main__":
    generate_all_plots()