"""
CAEG-Net V2 Phase 4 Publication Visualization Suite (Pure Matplotlib & Scipy)
=============================================================================
Generates the 10 required research-quality figures mandated by Phase 4:
1. Expert weight trajectories over the test timeline
2. Expert weight distributions (KDE via scipy)
3. Routing entropy H(w) and effective expert count N_eff over time
4. Temporal dynamics of all 6 base context features
5. Context features vs routing weights (scatter & trend)
6. Expert absolute error vs assigned routing weight
7. Inter-expert disagreement vs fused forecast error
8. Inter-expert disagreement vs improvement over TCN and Equal Ensemble
9. Expert suitability and performance across operational regimes
10. CAEG-Net V2 vs Equal Ensemble by difficulty regime

Saves all figures to: research/results/phase4_plots/
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Professional academic styling
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

OUT_DIR = "research/results/phase4_plots"
os.makedirs(OUT_DIR, exist_ok=True)

DIAG_CSV = "research/results/phase4_seed42_routing_diagnostics.csv"
REG_CSV = "research/results/phase4_regime_analysis.csv"


def plot_reg_trend(ax, x, y, color, label=None):
    ax.scatter(x, y, alpha=0.25, color=color, label=label, s=15)
    # Fit linear regression trend line
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 2:
        m, b = np.polyfit(x[mask], y[mask], 1)
        x_sort = np.sort(x[mask])
        ax.plot(x_sort, m * x_sort + b, color="black", lw=1.8, linestyle="-")


def generate_all_plots():
    print(f"=== Generating Phase 4 Publication Plots in {OUT_DIR} ===")
    df = pd.read_csv(DIAG_CSV)
    df_reg = pd.read_csv(REG_CSV)
    dt = pd.to_datetime(df["timestamp"])

    colors = {
        "GRU": "#2b5c8f",
        "TCN": "#d95f02",
        "Patch": "#7570b3",
        "Fused": "#1b9e77",
        "Equal": "#66a61e",
        "Oracle": "#e7298a",
    }

    # -------------------------------------------------------------
    # Plot 1: Expert Weight Trajectories Over Test Timeline
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dt, df["w_Patch"], label=f"Patch Expert (Mean: {df['w_Patch'].mean():.3f})", color=colors["Patch"], lw=1.5)
    ax.plot(dt, df["w_TCN"], label=f"TCN Expert (Mean: {df['w_TCN'].mean():.3f})", color=colors["TCN"], lw=1.5)
    ax.plot(dt, df["w_GRU"], label=f"GRU Expert (Mean: {df['w_GRU'].mean():.3f})", color=colors["GRU"], lw=1.5)
    ax.axhline(1.0/3.0, color="gray", linestyle="--", alpha=0.7, label="Equal Weight (0.333)")
    ax.set_ylabel("Routing Weight w_i")
    ax.set_xlabel("Forecast Origin Timestamp")
    ax.set_title("CAEG-Net V2: Dynamic Expert Routing Weights Across Test Set (Seed 42)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    p1 = os.path.join(OUT_DIR, "01_expert_weight_trajectories.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # Plot 2: Expert Weight Distributions (KDE via scipy)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_grid = np.linspace(0, 1, 500)
    for name, col in [("Patch", "w_Patch"), ("TCN", "w_TCN"), ("GRU", "w_GRU")]:
        vals = df[col].values
        kde = gaussian_kde(vals, bw_method=0.3)
        density = kde(x_grid)
        ax.plot(x_grid, density, color=colors[name], lw=2, label=f"{name} (Mean: {vals.mean():.3f}, std: {vals.std():.4f})")
        ax.fill_between(x_grid, density, alpha=0.3, color=colors[name])

    ax.axvline(1.0/3.0, color="gray", linestyle="--", alpha=0.7, label="Uniform (1/3)")
    ax.set_xlabel("Routing Weight w_i")
    ax.set_ylabel("Empirical Density")
    ax.set_title("CAEG-Net V2: Empirical Distribution of Expert Routing Weights")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    p2 = os.path.join(OUT_DIR, "02_expert_weight_distributions.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # Plot 3: Routing Entropy & Effective Expert Count
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(dt, df["Entropy_H"], color="#4575b4", lw=1.2, label="Routing Entropy H(w)")
    ax1.axhline(np.log(3), color="crimson", linestyle="--", lw=1.2, label=f"Max Uniform Entropy ln(3) = {np.log(3):.3f}")
    ax1.axhline(df["Entropy_H"].mean(), color="black", linestyle=":", lw=1.2, label=f"Mean Entropy = {df['Entropy_H'].mean():.3f}")
    ax1.set_ylabel("Entropy H(w) [nats]")
    ax1.set_title("CAEG-Net V2: Routing Concentration & Effective Number of Experts Over Time")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(dt, df["N_eff"], color="#313695", lw=1.2, label="Effective Number of Experts N_eff")
    ax2.axhline(3.0, color="crimson", linestyle="--", lw=1.2, label="Full Trio (3.0)")
    ax2.axhline(df["N_eff"].mean(), color="black", linestyle=":", lw=1.2, label=f"Mean N_eff = {df['N_eff'].mean():.2f}")
    ax2.set_ylabel("N_eff = exp(H)")
    ax2.set_xlabel("Forecast Origin Timestamp")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)
    p3 = os.path.join(OUT_DIR, "03_routing_entropy_over_time.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # Plot 4: Temporal Dynamics of Context Features
    # -------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axs[0, 0].plot(dt, df["Trend_Slope"], color="#08519c", lw=1.0)
    axs[0, 0].set_title("Trend Slope (Normalized beta_1)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(dt, df["ShortTerm_Volatility"], color="#e6550d", lw=1.0)
    axs[0, 1].set_title("Short-Term Volatility sigma(diff)")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(dt, df["Recent48h_RangeRatio"], color="#756bb1", lw=1.0)
    axs[1, 0].set_title("Recent 48h Range Ratio R_48")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(dt, df["Diurnal_Periodicity_r24"], color="#31a354", lw=1.0)
    axs[1, 1].set_title("Diurnal Rhythmicity (Lag-24 Autocorrelation)")
    axs[1, 1].grid(True, alpha=0.3)

    axs[2, 0].plot(dt, df["Weekly_Profile_r168"], color="#636363", lw=1.0)
    axs[2, 0].set_title("Weekly Profile Similarity r_168 (192h Buffer)")
    axs[2, 0].set_xlabel("Timestamp")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(dt, df["Recent_Baseline_MAE"], color="#bd0026", lw=1.0)
    axs[2, 1].set_title("Recent Expanding-Ridge Baseline MAE (Feedback)")
    axs[2, 1].set_xlabel("Timestamp")
    axs[2, 1].grid(True, alpha=0.3)

    fig.suptitle("CAEG-Net V2: Temporal Evolution of Observable Context Features (Test Partition)", y=0.99)
    p4 = os.path.join(OUT_DIR, "04_context_features_over_time.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------
    # Plot 5: Context Features vs Routing Weights
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_reg_trend(ax1, df["ShortTerm_Volatility"].values, df["w_Patch"].values, colors["Patch"])
    ax1.set_xlabel("Short-Term Volatility")
    ax1.set_ylabel("Patch Routing Weight (w_Patch)")
    ax1.set_title("Volatility vs Patch Routing Weight")
    ax1.grid(True, alpha=0.3)

    plot_reg_trend(ax2, df["Disagreement_Pairwise_MAE"].values, df["w_Patch"].values, colors["TCN"])
    ax2.set_xlabel("Inter-Expert Disagreement (Pairwise MAE)")
    ax2.set_ylabel("Patch Routing Weight (w_Patch)")
    ax2.set_title("Inter-Expert Disagreement vs Patch Weight")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CAEG-Net V2: Empirical Relationship Between Context and Expert Gating", y=1.02)
    p5 = os.path.join(OUT_DIR, "05_context_vs_routing_weights.png")
    fig.savefig(p5)
    plt.close(fig)
    print(f"Saved: {p5}")

    # -------------------------------------------------------------
    # Plot 6: Expert Error vs Assigned Weight
    # -------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    axs[0].scatter(df["w_GRU"], df["MAE_GRU"], alpha=0.25, color=colors["GRU"])
    axs[0].set_xlabel("w_GRU")
    axs[0].set_ylabel("24h MAE (MW)")
    axs[0].set_title("GRU: Weight vs Absolute Error")
    axs[0].grid(True, alpha=0.3)

    axs[1].scatter(df["w_TCN"], df["MAE_TCN"], alpha=0.25, color=colors["TCN"])
    axs[1].set_xlabel("w_TCN")
    axs[1].set_title("TCN: Weight vs Absolute Error")
    axs[1].grid(True, alpha=0.3)

    axs[2].scatter(df["w_Patch"], df["MAE_Patch"], alpha=0.25, color=colors["Patch"])
    axs[2].set_xlabel("w_Patch")
    axs[2].set_title("Patch: Weight vs Absolute Error")
    axs[2].grid(True, alpha=0.3)

    fig.suptitle("CAEG-Net V2: Expert Prediction Error vs Router Weight Allocation", y=1.03)
    p6 = os.path.join(OUT_DIR, "06_expert_error_vs_routing_weight.png")
    fig.savefig(p6)
    plt.close(fig)
    print(f"Saved: {p6}")

    # -------------------------------------------------------------
    # Plot 7: Disagreement vs Fused Forecast Error
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_reg_trend(ax, df["Disagreement_Pairwise_MAE"].values, df["MAE_Fused"].values, colors["Fused"])
    corr_p = df["Disagreement_Pairwise_MAE"].corr(df["MAE_Fused"])
    ax.set_xlabel("Inter-Expert Pairwise Disagreement [Standardized]")
    ax.set_ylabel("CAEG-Net V2 Fused MAE (MW)")
    ax.set_title(f"Forecast Difficulty: Inter-Expert Disagreement vs Fused Error (r = {corr_p:.3f})")
    ax.grid(True, alpha=0.3)
    p7 = os.path.join(OUT_DIR, "07_disagreement_vs_fused_error.png")
    fig.savefig(p7)
    plt.close(fig)
    print(f"Saved: {p7}")

    # -------------------------------------------------------------
    # Plot 8: Disagreement vs Improvement Over Baselines
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    gain_tcn = df["MAE_TCN"].values - df["MAE_Fused"].values
    plot_reg_trend(ax1, df["Disagreement_Pairwise_MAE"].values, gain_tcn, colors["TCN"])
    ax1.axhline(0, color="gray", linestyle="--")
    ax1.set_xlabel("Disagreement (Pairwise MAE)")
    ax1.set_ylabel("Gain Over TCN (MW)")
    ax1.set_title("Disagreement vs Accuracy Gain Over TCN")
    ax1.grid(True, alpha=0.3)

    plot_reg_trend(ax2, df["Disagreement_Pairwise_MAE"].values, df["Gain_over_Equal_Ensemble"].values, colors["Equal"])
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Disagreement (Pairwise MAE)")
    ax2.set_ylabel("Gain Over Equal Ensemble (MW)")
    ax2.set_title("Disagreement vs Gain Over Equal Ensemble")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CAEG-Net V2: Routing Advantage as a Function of Disagreement", y=1.02)
    p8 = os.path.join(OUT_DIR, "08_disagreement_vs_improvement_over_tcn.png")
    fig.savefig(p8)
    plt.close(fig)
    print(f"Saved: {p8}")

    # -------------------------------------------------------------
    # Plot 9: Expert Suitability Across Regimes
    # -------------------------------------------------------------
    sub_reg = df_reg[df_reg["Regime_Type"].isin(["Volatility", "Disagreement", "Baseline_Error"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = [f"{r['Regime_Type']}-{r['Regime_Level']}" for _, r in sub_reg.iterrows()]
    x = np.arange(len(sub_reg))
    width = 0.2

    ax.bar(x - 1.5*width, sub_reg["MAE_Patch_MW"], width, label="Patch Expert", color=colors["Patch"], alpha=0.8)
    ax.bar(x - 0.5*width, sub_reg["MAE_TCN_MW"], width, label="TCN Expert", color=colors["TCN"], alpha=0.8)
    ax.bar(x + 0.5*width, sub_reg["MAE_Equal_Ens_MW"], width, label="Equal Ensemble", color=colors["Equal"], alpha=0.8)
    ax.bar(x + 1.5*width, sub_reg["MAE_Fused_MW"], width, label="CAEG-Net V2 (Fused)", color=colors["Fused"], alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_ylabel("24h MAE (MW)")
    ax.set_title("Expert and Fused Model Performance Across Objective Operational Regimes")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    p9 = os.path.join(OUT_DIR, "09_expert_suitability_by_regime.png")
    fig.savefig(p9)
    plt.close(fig)
    print(f"Saved: {p9}")

    # -------------------------------------------------------------
    # Plot 10: V2 vs Equal Ensemble by Difficulty Regime
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dis_reg = df_reg[df_reg["Regime_Type"] == "Disagreement"].sort_values("Regime_Level").copy()
    order = {"Low": 0, "Medium": 1, "High": 2}
    dis_reg["order"] = dis_reg["Regime_Level"].map(order)
    dis_reg = dis_reg.sort_values("order")

    x_pos = np.arange(len(dis_reg))
    w = 0.3
    ax.bar(x_pos - w/2, dis_reg["MAE_Equal_Ens_MW"], w, label="Equal Ensemble (1/3 each)", color=colors["Equal"], alpha=0.8)
    ax.bar(x_pos + w/2, dis_reg["MAE_Fused_MW"], w, label="CAEG-Net V2 (Context-Adaptive)", color=colors["Fused"], alpha=0.95)

    for i, (_, r) in enumerate(dis_reg.iterrows()):
        gain = r["Gain_vs_Equal_MW"]
        ax.annotate(f"Gain: +{gain:.1f} MW", (i, min(r["MAE_Equal_Ens_MW"], r["MAE_Fused_MW"]) - 15),
                    ha="center", fontsize=9, fontweight="bold", color="darkgreen")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Disagreement: {lvl}" for lvl in dis_reg["Regime_Level"]])
    ax.set_ylabel("24h MAE (MW)")
    ax.set_title("CAEG-Net V2 vs Fixed Equal Ensemble Across Forecast Disagreement Regimes")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3, axis="y")
    p10 = os.path.join(OUT_DIR, "10_v2_vs_equal_ensemble_by_difficulty.png")
    fig.savefig(p10)
    plt.close(fig)
    print(f"Saved: {p10}")
    print("All 10 publication plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()