"""
Phase 15B Publication Figure Generator
======================================
Generates all 14 publication-quality figures for Phase 15B:
1. Mean test MAE by dataset
2. Five-seed MAE distributions
3. Candidate vs F2 paired daily-block differences
4. Routing weight distributions
5. Routing dynamicity comparison
6. Confidence lambda distributions
7. Confidence lambda versus difficulty
8. Horizon-level expert MAE
9. Horizon-level candidate MAE
10. Oracle convex fusion gap
11. Selection regret
12. Fusion gain
13. Regime-level performance
14. Parameter count versus performance
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

plots_dir = "research/plots"
os.makedirs(plots_dir, exist_ok=True)

# Load data
df_test = pd.read_csv("research/analysis/phase15b_test_results.csv")
df_stat = pd.read_csv("research/analysis/phase15b_daily_block_statistics.csv")
df_dyn = pd.read_csv("research/analysis/phase15b_routing_dynamicity.csv")
df_regret = pd.read_csv("research/analysis/phase15b_router_decision_quality.csv")
df_oracle = pd.read_csv("research/analysis/phase15b_oracle_gap.csv")
df_conf = pd.read_csv("research/analysis/phase15b_confidence_results.csv")
df_horizon = pd.read_csv("research/analysis/phase15b_horizon_results.csv")
df_regime = pd.read_csv("research/analysis/phase15b_regime_results.csv")
df_params = pd.read_csv("research/analysis/phase15b_parameter_counts.csv")

candidates = [
    "Control_A_F2", "Control_B_FixedShrinkage", "Control_C_HorizonRouting",
    "Control_D_DynamicConfidence", "Candidate_E1_HGR_FS", "Candidate_E2_HGR_DGS"
]
labels = [
    "Control A (F2)", "Control B (Fixed λ)", "Control C (Horizon Only)",
    "Control D (Dynamic Conf)", "Candidate E1 (HGR-FS)", "Candidate E2 (HGR-DGS)"
]
cand_map = dict(zip(candidates, labels))
colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]

# -------------------------------------------------------------
# Fig 1: Mean test MAE by dataset
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_test[df_test["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax = axes[idx]
    bars = ax.bar(range(len(candidates)), sub["test_mae_mean"], yerr=sub["test_mae_std"],
                  capsize=4, color=colors, edgecolor="black", alpha=0.85)
    ax.set_title(f"{d} Test MAE ({sub['unit'].iloc[0]})", fontweight="bold")
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(f"MAE ({sub['unit'].iloc[0]})")
    # Annotate values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig01_test_mae_by_dataset.png"))
plt.close()
print("Saved fig 01")

# -------------------------------------------------------------
# Fig 2: Five-seed MAE distributions
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
seed_cols = ["seed_42_mae", "seed_123_mae", "seed_999_mae", "seed_2024_mae", "seed_3407_mae"]
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_test[df_test["dataset"] == d].set_index("candidate_id").reindex(candidates)
    data = [sub.loc[c, seed_cols].values for c in candidates]
    ax = axes[idx]
    bplot = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(f"{d} Seed Stability (5 Seeds)", fontweight="bold")
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(f"MAE ({sub['unit'].iloc[0]})")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig02_fiveseed_distributions.png"))
plt.close()
print("Saved fig 02")

# -------------------------------------------------------------
# Fig 3: Paired daily-block differences vs Control A (F2)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_stat[df_stat["dataset"] == d].set_index("candidate_id")
    cands_sub = [c for c in candidates if c in sub.index]
    sub = sub.reindex(cands_sub)
    ax = axes[idx]
    y_err = [sub["ci_95_upper"] - sub["mean_paired_diff"], sub["mean_paired_diff"] - sub["ci_95_lower"]]
    bars = ax.bar(range(len(cands_sub)), sub["mean_paired_diff"], yerr=y_err, capsize=4,
                  color=[colors[candidates.index(c)] for c in cands_sub], edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{d} Paired Block Diff vs F2 (95% CI)", fontweight="bold")
    ax.set_xticks(range(len(cands_sub)))
    ax.set_xticklabels([cand_map[c] for c in cands_sub], rotation=40, ha="right")
    ax.set_ylabel("Paired Difference (Candidate - F2)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig03_paired_daily_block_differences.png"))
plt.close()
print("Saved fig 03")

# -------------------------------------------------------------
# Fig 4: Routing weight distributions
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_dyn[df_dyn["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax = axes[idx]
    x = np.arange(len(candidates))
    w = 0.25
    ax.bar(x - w, sub["mean_weight_lstm"], width=w, label="LSTM", color="#1f77b4", edgecolor="black")
    ax.bar(x, sub["mean_weight_tcn"], width=w, label="TCN", color="#ff7f0e", edgecolor="black")
    ax.bar(x + w, sub["mean_weight_cnn"], width=w, label="CNN", color="#2ca02c", edgecolor="black")
    ax.axhline(1.0/3.0, color="gray", linestyle=":", label="Uniform Centroid")
    ax.set_title(f"{d} Mean Routing Weights", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Weight Share")
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig04_routing_weight_distributions.png"))
plt.close()
print("Saved fig 04")

# -------------------------------------------------------------
# Fig 5: Routing dynamicity comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
pjm_dyn = df_dyn[df_dyn["dataset"] == "PJM"].set_index("candidate_id").reindex(candidates)["dynamicity_score"]
gef_dyn = df_dyn[df_dyn["dataset"] == "GEFCom"].set_index("candidate_id").reindex(candidates)["dynamicity_score"]
uci_dyn = df_dyn[df_dyn["dataset"] == "UCI"].set_index("candidate_id").reindex(candidates)["dynamicity_score"]
x = np.arange(len(candidates))
w = 0.25
ax.bar(x - w, pjm_dyn, width=w, label="PJM", color="#4c72b0", edgecolor="black")
ax.bar(x, gef_dyn, width=w, label="GEFCom", color="#55a868", edgecolor="black")
ax.bar(x + w, uci_dyn, width=w, label="UCI", color="#c44e52", edgecolor="black")
ax.set_title("Routing Dynamicity Score Comparison", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("Dynamicity Score")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig05_routing_dynamicity_comparison.png"))
plt.close()
print("Saved fig 05")

# -------------------------------------------------------------
# Fig 6: Confidence lambda distributions
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
sub_conf = df_conf.set_index(["candidate_id", "dataset"])
x = np.arange(len(candidates))
w = 0.25
for i, d in enumerate(["PJM", "GEFCom", "UCI"]):
    vals = [sub_conf.loc[(c, d), "mean_lambda"] for c in candidates]
    errs = [sub_conf.loc[(c, d), "std_lambda"] for c in candidates]
    ax.bar(x + (i - 1)*w, vals, width=w, yerr=errs, capsize=3, label=d, edgecolor="black", alpha=0.85)
ax.axhline(0.51, color="red", linestyle="--", label="Fixed Shrinkage λ=0.51")
ax.set_title("Mean Learned Confidence Fallback (λ)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("Confidence λ")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig06_confidence_lambda_distributions.png"))
plt.close()
print("Saved fig 06")

# -------------------------------------------------------------
# Fig 7: Confidence lambda vs CV (dispersion)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for d, col in zip(["PJM", "GEFCom", "UCI"], ["#1f77b4", "#2ca02c", "#d62728"]):
    sub = df_conf[df_conf["dataset"] == d]
    ax.scatter(sub["mean_lambda"], sub["cv_lambda"] * 100.0, s=80, label=d, color=col)
    for _, r in sub.iterrows():
        ax.annotate(r["candidate_id"].replace("Control_", "C_").replace("Candidate_", "E_"),
                    (r["mean_lambda"], r["cv_lambda"] * 100.0), fontsize=7, alpha=0.7)
ax.axhline(1.5, color="gray", linestyle=":", label="Constant Threshold (CV < 1.5%)")
ax.set_title("Confidence Parameter Dispersion (Mean vs CV)", fontweight="bold")
ax.set_xlabel("Mean Confidence (λ)")
ax.set_ylabel("Coefficient of Variation (%)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig07_confidence_vs_difficulty.png"))
plt.close()
print("Saved fig 07")

# -------------------------------------------------------------
# Fig 8: Horizon-level expert MAE (Crossover)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_horizon[(df_horizon["dataset"] == d) & (df_horizon["candidate_id"] == "Control_A_F2")]
    ax = axes[idx]
    ax.plot(sub["horizon_step"], sub["mae"], marker="o", label="Control A (F2)", color="#1f77b4", linewidth=2)
    # Compare with Control C (Horizon Only)
    sub_c = df_horizon[(df_horizon["dataset"] == d) & (df_horizon["candidate_id"] == "Control_C_HorizonRouting")]
    ax.plot(sub_c["horizon_step"], sub_c["mae"], marker="s", label="Control C (Horizon Only)", color="#d62728", linestyle="--")
    ax.axvspan(1, 8, alpha=0.1, color="blue", label="Short (1-8)" if idx==0 else "")
    ax.axvspan(9, 16, alpha=0.1, color="green", label="Medium (9-16)" if idx==0 else "")
    ax.axvspan(17, 24, alpha=0.1, color="orange", label="Long (17-24)" if idx==0 else "")
    ax.set_title(f"{d} Horizon-Level MAE (h=1..24)", fontweight="bold")
    ax.set_xlabel("Forecast Horizon Step (Hour)")
    ax.set_ylabel("MAE")
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig08_horizon_expert_mae.png"))
plt.close()
print("Saved fig 08")

# -------------------------------------------------------------
# Fig 9: Horizon-level candidate MAE comparison
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    ax = axes[idx]
    for cid, col in zip(["Control_A_F2", "Control_B_FixedShrinkage", "Control_C_HorizonRouting", "Candidate_E1_HGR_FS"],
                        ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]):
        sub = df_horizon[(df_horizon["dataset"] == d) & (df_horizon["candidate_id"] == cid)]
        ax.plot(sub["horizon_step"], sub["mae"], label=cand_map[cid], color=col, linewidth=1.8)
    ax.set_title(f"{d} Candidate Comparison Across Horizons", fontweight="bold")
    ax.set_xlabel("Horizon Step (h=1..24)")
    ax.set_ylabel("MAE")
    if idx == 0:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig09_horizon_candidate_mae.png"))
plt.close()
print("Saved fig 09")

# -------------------------------------------------------------
# Fig 10: Oracle convex fusion gap
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_oracle[df_oracle["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax = axes[idx]
    bars = ax.bar(range(len(candidates)), sub["oracle_gap"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_title(f"{d} Gap to Oracle Convex Fusion", fontweight="bold")
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("MAE Gap to Oracle")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig10_oracle_convex_gap.png"))
plt.close()
print("Saved fig 10")

# -------------------------------------------------------------
# Fig 11: Selection regret
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(candidates))
w = 0.25
for i, (d, col) in enumerate(zip(["PJM", "GEFCom", "UCI"], ["#1f77b4", "#2ca02c", "#d62728"])):
    sub = df_regret[df_regret["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax.bar(x + (i - 1)*w, sub["selection_regret_pct"], width=w, label=d, color=col, edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linestyle="-", linewidth=1)
ax.set_title("Selection Regret (% of Best Standalone Benchmark)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("Selection Regret (%)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig11_selection_regret.png"))
plt.close()
print("Saved fig 11")

# -------------------------------------------------------------
# Fig 12: Fusion gain
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for i, (d, col) in enumerate(zip(["PJM", "GEFCom", "UCI"], ["#1f77b4", "#2ca02c", "#d62728"])):
    sub = df_regret[df_regret["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax.bar(x + (i - 1)*w, sub["fusion_gain_pct"], width=w, label=d, color=col, edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("Fusion Gain (% Improvement over Best Standalone)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("Fusion Gain (%) [Negative = Favorable]")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig12_fusion_gain.png"))
plt.close()
print("Saved fig 12")

# -------------------------------------------------------------
# Fig 13: Regime-level performance (Volatility breakdown)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, d in enumerate(["PJM", "GEFCom", "UCI"]):
    sub = df_regime[df_regime["dataset"] == d].set_index("candidate_id").reindex(candidates)
    ax = axes[idx]
    x_pos = np.arange(len(candidates))
    w = 0.25
    ax.bar(x_pos - w, sub["vol_low_mae"], width=w, label="Low Volatility", color="#4c72b0", edgecolor="black")
    ax.bar(x_pos, sub["vol_med_mae"], width=w, label="Med Volatility", color="#55a868", edgecolor="black")
    ax.bar(x_pos + w, sub["vol_high_mae"], width=w, label="High Volatility", color="#c44e52", edgecolor="black")
    ax.set_title(f"{d} MAE by Volatility Regime", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("MAE")
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig13_regime_performance.png"))
plt.close()
print("Saved fig 13")

# -------------------------------------------------------------
# Fig 14: Parameter count versus performance
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for c in candidates:
    p = df_params[df_params["candidate_id"] == c].iloc[0]["total_parameters"]
    # Average normalized rank across the 3 datasets
    pjm_m = df_test[(df_test["candidate_id"] == c) & (df_test["dataset"] == "PJM")].iloc[0]["test_mae_mean"]
    gef_m = df_test[(df_test["candidate_id"] == c) & (df_test["dataset"] == "GEFCom")].iloc[0]["test_mae_mean"]
    uci_m = df_test[(df_test["candidate_id"] == c) & (df_test["dataset"] == "UCI")].iloc[0]["test_mae_mean"]
    # Mean relative error compared to F2
    f2_p = df_test[(df_test["candidate_id"] == "Control_A_F2") & (df_test["dataset"] == "PJM")].iloc[0]["test_mae_mean"]
    f2_g = df_test[(df_test["candidate_id"] == "Control_A_F2") & (df_test["dataset"] == "GEFCom")].iloc[0]["test_mae_mean"]
    f2_u = df_test[(df_test["candidate_id"] == "Control_A_F2") & (df_test["dataset"] == "UCI")].iloc[0]["test_mae_mean"]
    rel_err = np.mean([(pjm_m - f2_p)/f2_p, (gef_m - f2_g)/f2_g, (uci_m - f2_u)/f2_u]) * 100.0
    
    col = colors[candidates.index(c)]
    ax.scatter(p, rel_err, s=150, color=col, edgecolor="black", label=cand_map[c])
    ax.annotate(cand_map[c], (p, rel_err), xytext=(5, 5), textcoords="offset points", fontsize=9)

ax.axhline(0, color="gray", linestyle="--", label="Control A (F2) Baseline")
ax.set_title("Model Complexity vs Relative Error Increase (%)", fontweight="bold")
ax.set_xlabel("Total Parameters")
ax.set_ylabel("Mean Relative Error vs F2 (%) [Lower = Better]")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "phase15b_fig14_parameter_count_vs_performance.png"))
plt.close()
print("Saved fig 14")

print("All 14 publication figures generated successfully!")
