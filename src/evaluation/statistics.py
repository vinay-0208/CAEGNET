"""
Statistical Hypothesis Testing on Non-Overlapping Daily Blocks
"""
from typing import Dict, List
import numpy as np
import scipy.stats as stats


def compute_daily_block_statistics(
    errors_model_a: np.ndarray,
    errors_model_b: np.ndarray,
    k_blocks: int,
) -> Dict[str, float]:
    """
    Computes paired t-test, Wilcoxon signed-rank, and effect size Cohen's d
    over k non-overlapping evaluation blocks.
    """
    diffs = errors_model_a - errors_model_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0

    t_stat, p_val_t = stats.ttest_1samp(diffs, 0.0)
    try:
        w_stat, p_val_w = stats.wilcoxon(diffs)
    except ValueError:
        w_stat, p_val_w = 0.0, 1.0

    return {
        "k_blocks": k_blocks,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohen_d": cohen_d,
        "t_statistic": float(t_stat),
        "p_value_t": float(p_val_t),
        "wilcoxon_statistic": float(w_stat),
        "p_value_wilcoxon": float(p_val_w),
    }
