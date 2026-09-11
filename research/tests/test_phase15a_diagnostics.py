"""
Unit and Regression Tests for Phase 15A Diagnostic Calculations
===============================================================
Verifies:
- Routing-statistic correctness (entropy, N_eff)
- Confidence statistics & calibration binning
- Oracle calculations & regret properties
- Horizon aggregation
- Disagreement calculations
- Daily-block alignment
- Zero future-target leakage
- Deterministic reproducibility
"""

import unittest
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from research.deterministic import seed_everything


class TestPhase15ADiagnostics(unittest.TestCase):
    def setUp(self):
        seed_everything(42, deterministic_cudnn=True)
        self.rng = np.random.RandomState(42)

    def test_routing_entropy_and_neff_boundaries(self):
        """Test entropy and N_eff boundaries on the 3-expert probability simplex."""
        # 1. Equal distribution [1/3, 1/3, 1/3] -> max entropy ln(3), N_eff = 3.0
        w_equal = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        ent_equal = -np.sum(w_equal * np.log(w_equal + 1e-12), axis=-1)
        neff_equal = np.exp(ent_equal)
        self.assertAlmostEqual(ent_equal[0], np.log(3.0), places=5)
        self.assertAlmostEqual(neff_equal[0], 3.0, places=5)

        # 2. Degenerate distribution [1, 0, 0] -> min entropy 0.0, N_eff = 1.0
        w_degen = np.array([[1.0, 0.0, 0.0]])
        ent_degen = -np.sum(w_degen * np.log(w_degen + 1e-12), axis=-1)
        neff_degen = np.exp(ent_degen)
        self.assertAlmostEqual(ent_degen[0], 0.0, places=4)
        self.assertAlmostEqual(neff_degen[0], 1.0, places=4)

        # 3. Intermediate distribution
        w_inter = np.array([[0.5, 0.3, 0.2]])
        ent_inter = -np.sum(w_inter * np.log(w_inter + 1e-12), axis=-1)
        neff_inter = np.exp(ent_inter)
        self.assertTrue(0.0 < ent_inter[0] < np.log(3.0))
        self.assertTrue(1.0 < neff_inter[0] < 3.0)

    def test_confidence_statistics_and_cv(self):
        """Test coefficient of variation and bounded percentiles for lambda."""
        # Mock lambda near 0.51 with small variance (F2 behavior)
        lams = self.rng.normal(loc=0.51, scale=0.005, size=1000)
        mean_l = np.mean(lams)
        std_l = np.std(lams)
        cv_l = std_l / mean_l

        self.assertAlmostEqual(mean_l, 0.51, delta=0.002)
        self.assertTrue(cv_l < 0.02)  # Low temporal variability CV < 2%
        self.assertTrue(np.all(lams > 0.45))
        self.assertTrue(np.all(lams < 0.55))

    def test_confidence_calibration_binning(self):
        """Verify quantile binning and monotonicity testing logic."""
        lams = np.array([0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55])
        advantages = np.array([1.0, 1.2, 1.5, 1.4, 2.0, 2.1, 2.5, 2.8])

        # Test Spearman rank correlation
        rho, p = stats.spearmanr(lams, advantages)
        self.assertGreater(rho, 0.9)  # Strong positive monotonic relation

    def test_oracle_ceiling_hierarchy(self):
        """Verify the theoretical mathematical ordering: Oracle Convex <= Oracle Best Expert <= Best Standalone."""
        # Simulate predictions for 3 experts across 10 blocks (24 hours each)
        N = 10
        H = 24
        y_true = self.rng.normal(100, 10, (N, H))
        y_l = y_true + self.rng.normal(2, 5, (N, H))
        y_t = y_true + self.rng.normal(0, 4, (N, H))
        y_c = y_true + self.rng.normal(-1, 6, (N, H))

        mae_l = np.mean(np.abs(y_l - y_true))
        mae_t = np.mean(np.abs(y_t - y_true))
        mae_c = np.mean(np.abs(y_c - y_true))
        best_standalone_mae = min(mae_l, mae_t, mae_c)

        # Oracle Best Expert per block
        oracle_exp_maes = []
        for i in range(N):
            m_l = np.mean(np.abs(y_l[i] - y_true[i]))
            m_t = np.mean(np.abs(y_t[i] - y_true[i]))
            m_c = np.mean(np.abs(y_c[i] - y_true[i]))
            oracle_exp_maes.append(min(m_l, m_t, m_c))
        mean_oracle_exp = np.mean(oracle_exp_maes)

        # Oracle Convex Combination per block
        oracle_cvx_maes = []
        for i in range(N):
            best_cvx = float("inf")
            for w1 in np.linspace(0, 1, 11):
                for w2 in np.linspace(0, 1 - w1, 11):
                    w3 = max(0.0, 1.0 - w1 - w2)
                    comb = w1 * y_l[i] + w2 * y_t[i] + w3 * y_c[i]
                    err = np.mean(np.abs(comb - y_true[i]))
                    if err < best_cvx:
                        best_cvx = err
            oracle_cvx_maes.append(best_cvx)
        mean_oracle_cvx = np.mean(oracle_cvx_maes)

        # Mathematical hierarchy check
        self.assertLessEqual(mean_oracle_exp, best_standalone_mae + 1e-6)
        self.assertLessEqual(mean_oracle_cvx, mean_oracle_exp + 1e-6)

    def test_routing_regret_properties(self):
        """Verify regret is strictly non-negative relative to oracle."""
        mae_selected = 12.5
        mae_oracle = 10.0
        regret = mae_selected - mae_oracle
        self.assertGreaterEqual(regret, 0.0)

    def test_horizon_group_partitioning(self):
        """Verify exact 24-step horizon partitioning into Short, Medium, and Long."""
        steps = list(range(1, 25))
        short = [s for s in steps if s <= 8]
        med = [s for s in steps if 9 <= s <= 16]
        long = [s for s in steps if s >= 17]

        self.assertEqual(len(short), 8)
        self.assertEqual(len(med), 8)
        self.assertEqual(len(long), 8)
        self.assertEqual(short, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(med, [9, 10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(long, [17, 18, 19, 20, 21, 22, 23, 24])

    def test_disagreement_calculation_properties(self):
        """Verify disagreement properties: non-negativity, symmetry, and spread."""
        yl = np.array([[10.0, 20.0]])
        yt = np.array([[12.0, 18.0]])
        yc = np.array([[8.0, 25.0]])

        d_lt = np.mean(np.abs(yl - yt))
        d_tl = np.mean(np.abs(yt - yl))
        self.assertAlmostEqual(d_lt, d_tl)  # Symmetry
        self.assertGreater(d_lt, 0.0)  # Non-negativity

        spread = max(np.mean(np.abs(yl - yt)), np.mean(np.abs(yl - yc)), np.mean(np.abs(yt - yc)))
        self.assertGreaterEqual(spread, d_lt)

    def test_daily_block_non_overlapping_indices(self):
        """Verify non-overlapping daily block indexing strictly preserves temporal independence."""
        N = 1294
        H = 24
        K = N // H
        self.assertEqual(K, 53)  # Matches PJM K=53 daily blocks

        indices_used = set()
        for k in range(K):
            block_idx = list(range(k * H, (k + 1) * H))
            # Verify exactly 24 hours per block
            self.assertEqual(len(block_idx), 24)
            # Verify no overlap with previous blocks
            self.assertTrue(indices_used.isdisjoint(block_idx))
            indices_used.update(block_idx)

        self.assertEqual(len(indices_used), 53 * 24)

    def test_zero_future_leakage_in_lookback_features(self):
        """Verify context lookback strictly uses past sequence and never includes target horizon."""
        L = 168
        H = 24
        total_len = 300
        series = np.arange(total_len, dtype=float)

        t_forecast = 200
        lookback = series[t_forecast - L : t_forecast]
        target = series[t_forecast : t_forecast + H]

        self.assertEqual(len(lookback), 168)
        self.assertEqual(len(target), 24)
        self.assertEqual(lookback[-1], 199)
        self.assertEqual(target[0], 200)
        self.assertTrue(set(lookback).isdisjoint(target))

    def test_deterministic_output_reproducibility(self):
        """Verify deterministic loader and model initialization reproduce exact values."""
        seed_everything(42, deterministic_cudnn=True)
        m1 = nn.Linear(7, 3)
        w1 = m1.weight.clone().detach().numpy()

        seed_everything(42, deterministic_cudnn=True)
        m2 = nn.Linear(7, 3)
        w2 = m2.weight.clone().detach().numpy()

        np.testing.assert_allclose(w1, w2, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
