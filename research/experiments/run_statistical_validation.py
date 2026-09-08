"""
CAEG-Net Research Track: Phase 9 Statistical Validation Suite
============================================================
Performs rigorous hypothesis testing on the champion research model
(CAEG-Net Forecast-Aware Routing) against:
1. Static Equal Ensemble
2. Standalone TCN Expert (strongest standalone)
3. Standard Input-MoE
4. Canonical CAEG-Net V1

Tests Performed:
- Seed-level paired Student's t-test (df = 4)
- 95% Bootstrap Confidence Intervals (10,000 resamples)
- Non-overlapping 24-hour block evaluation (53 independent 24h forecast episodes)
"""

import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def bootstrap_ci(diffs: np.ndarray, num_resamples: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(42)
    means = np.empty(num_resamples)
    n = len(diffs)
    for i in range(num_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        means[i] = np.mean(sample)
    lower = np.percentile(means, 100.0 * (alpha / 2.0))
    upper = np.percentile(means, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def run_statistical_validation():
    print("=" * 85)
    print("CAEG-NET RESEARCH TRACK: PHASE 9 STATISTICAL VALIDATION SUITE")
    print("=" * 85)

    res_dir = os.path.join(repo_root, "research", "results")
    seeds_csv = os.path.join(res_dir, "five_seed_results.csv")
    df_research_seeds = pd.read_csv(seeds_csv)

    canon_csv = os.path.join(repo_root, "results", "phase5_multiseed_results.csv")
    df_canon = pd.read_csv(canon_csv)

    # Extract 5-seed MAE arrays
    SEEDS = [42, 123, 999, 2024, 3407]

    def get_seed_maes(df, model_col, model_val):
        sub = df[df[model_col] == model_val].sort_values("seed")
        return sub["MAE_MW"].values if "MAE_MW" in sub.columns else sub["MAE (MW)"].values

    mae_fa = get_seed_maes(df_research_seeds, "variant_id", "CAEG_Net_Forecast_Aware")
    mae_v1_research = get_seed_maes(df_research_seeds, "variant_id", "CAEG_Net_V1_Canonical")
    mae_norec = get_seed_maes(df_research_seeds, "variant_id", "CAEG_Net_No_Recent_Error")
    mae_tcn = get_seed_maes(df_canon, "model", "TCN_Standalone")
    mae_static = get_seed_maes(df_canon, "model", "Static_Equal_Ensemble")
    mae_moe = get_seed_maes(df_canon, "model", "Standard_Input_MoE")
    mae_v1_canon = get_seed_maes(df_canon, "model", "Full_CAEG_Net")

    # 1. Multi-Seed Paired Tests
    comparisons = [
        ("Forecast-Aware vs. Static Equal Ensemble", mae_static, mae_fa),
        ("Forecast-Aware vs. Standalone TCN Expert", mae_tcn, mae_fa),
        ("Forecast-Aware vs. Standard Input-MoE", mae_moe, mae_fa),
        ("Forecast-Aware vs. CAEG-Net V1 (Research)", mae_v1_research, mae_fa),
        ("Forecast-Aware vs. CAEG-Net V1 (Canonical)", mae_v1_canon, mae_fa),
        ("Forecast-Aware vs. No Recent Error", mae_norec, mae_fa),
        ("CAEG-Net V1 (Canonical) vs. Static Ensemble", mae_static, mae_v1_canon),
    ]

    test_records = []

    print("\n--- 1. Seed-Level Paired Hypothesis Tests (df = 4) ---")
    for name, m_base, m_cand in comparisons:
        diff = m_base - m_cand  # Positive diff = candidate is better (lower MAE)
        t_stat, p_val = ttest_rel(m_base, m_cand)
        ci_low, ci_high = bootstrap_ci(diff)
        mean_diff = float(np.mean(diff))

        sig = "Significant (p < 0.05)" if p_val < 0.05 else "Observed / Not Sig (p >= 0.05)"

        test_records.append({
            "Comparison": name,
            "Evaluation_Granularity": "Seed-Level (N=5)",
            "Mean_Delta_MAE_MW": round(mean_diff, 2),
            "Bootstrap_95_CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "t_statistic": round(float(t_stat), 4),
            "p_value": f"{p_val:.4e}",
            "Significance_Status": sig,
        })
        print(f"  {name:<46s} | Delta: {mean_diff:+6.2f} MW | t = {t_stat:6.3f} | p = {p_val:.4e} | {sig}")

    # 2. Non-Overlapping 24-Hour Block Tests (Test Set Episodes)
    print("\n--- 2. Non-Overlapping 24-Hour Block Evaluation (53 Independent Episodes) ---")
    cache_path = os.path.join(res_dir, "research_multiseed_cache.npz")
    cache = np.load(cache_path)

    y_t = cache["y_test_true_raw"]
    y_fa = cache["pred_CAEG_Net_Forecast_Aware_seed_42"]
    y_v1 = cache["pred_CAEG_Net_V1_Canonical_seed_42"]

    # Block indices: 0, 24, 48, ... up to len - 24
    block_indices = np.arange(0, len(y_t) - 23, 24)
    fa_block_mae = np.array([np.mean(np.abs(y_fa[b : b + 24] - y_t[b : b + 24])) for b in block_indices])
    v1_block_mae = np.array([np.mean(np.abs(y_v1[b : b + 24] - y_t[b : b + 24])) for b in block_indices])

    diff_blocks = v1_block_mae - fa_block_mae
    t_blk, p_blk = ttest_rel(v1_block_mae, fa_block_mae)
    ci_blk_l, ci_blk_h = bootstrap_ci(diff_blocks)
    mean_blk_diff = float(np.mean(diff_blocks))

    blk_sig = "Significant (p < 0.05)" if p_blk < 0.05 else "Observed / Not Sig (p >= 0.05)"
    test_records.append({
        "Comparison": "Forecast-Aware vs. V1 (Non-Overlapping 24h Blocks)",
        "Evaluation_Granularity": "Block-Level (53 Episodes)",
        "Mean_Delta_MAE_MW": round(mean_blk_diff, 2),
        "Bootstrap_95_CI": f"[{ci_blk_l:.2f}, {ci_blk_h:.2f}]",
        "t_statistic": round(float(t_blk), 4),
        "p_value": f"{p_blk:.4e}",
        "Significance_Status": blk_sig,
    })
    print(f"  Non-overlapping 24h Blocks (N=53)             | Delta: {mean_blk_diff:+6.2f} MW | t = {t_blk:6.3f} | p = {p_blk:.4e} | {blk_sig}")

    df_tests = pd.DataFrame(test_records)
    csv_path = os.path.join(res_dir, "statistical_tests.csv")
    df_tests.to_csv(csv_path, index=False)
    print(f"\nSaved statistical test results to: {csv_path}")

    print("\n" + "=" * 105)
    print("STATISTICAL VALIDATION SUMMARY TABLE:")
    print("=" * 105)
    print(df_tests[["Comparison", "Evaluation_Granularity", "Mean_Delta_MAE_MW", "Bootstrap_95_CI", "t_statistic", "p_value", "Significance_Status"]].to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    run_statistical_validation()
