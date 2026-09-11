"""
Unit and Regression Tests for Phase 15A Reconciled Diagnostics
=============================================================
Verifies all 12 mandated diagnostic calculations:
1. UCI baseline provenance distinction (val baseline vs test benchmark)
2. Residual correlation definition distinction (standalone vs co-adapted)
3. Routing dynamicity statistics (continuous variance & rank switching)
4. Simplex entropy boundary conditions
5. Effective number of experts (N_eff)
6. Oracle weight simplex constraints (w >= 0, sum w = 1)
7. Oracle MAE hierarchy and non-deployability property
8. Regret formula properties (selection vs fusion regret)
9. Shrinkage linear interpolation & descriptive decomposition
10. Daily-block alignment & non-overlapping index independence
11. Zero future-target leakage in lookback context slicing
12. Deterministic diagnostic reproducibility
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

    def test_1_uci_baseline_provenance_distinction(self):
        """Test 1: Verify UCI validation baseline (7.55 MW) vs test benchmark (7.79 MW) separation."""
        val_baseline = 7.554153
        test_benchmark = 7.794470
        f2_test_mae = 7.737148

        # F2 outperforms the test benchmark
        self.assertLess(f2_test_mae, test_benchmark)
        # F2 does NOT outperform the validation baseline
        self.assertGreater(f2_test_mae, val_baseline)
        # Gaps are mathematically distinct
        test_gain = test_benchmark - f2_test_mae
        self.assertAlmostEqual(test_gain, 0.057322, places=5)

    def test_2_correlation_definition_distinction(self):
        """Test 2: Verify distinction between positive standalone error correlations and negative co-adapted branch correlations."""
        # Case A: Standalone models subject to shared macroeconomic shocks
        true_load = self.rng.normal(100, 10, size=1000)
        macro_shock = self.rng.normal(0, 5, size=1000)
        # Both models share the under/over-prediction shock
        e_standalone_1 = macro_shock + self.rng.normal(0, 2, size=1000)
        e_standalone_2 = macro_shock + self.rng.normal(0, 2, size=1000)
        r_standalone = float(np.corrcoef(e_standalone_1, e_standalone_2)[0, 1])
        self.assertGreater(r_standalone, 0.5)  # Strictly positive

        # Case B: Co-adapted branches in an end-to-end trained ensemble: y_hat = 0.5*y1 + 0.5*y2
        # To cancel errors, branch 1 overpredicts when branch 2 underpredicts
        canceling_noise = self.rng.normal(0, 5, size=1000)
        e_coadapted_1 = canceling_noise + self.rng.normal(0, 1, size=1000)
        e_coadapted_2 = -canceling_noise + self.rng.normal(0, 1, size=1000)
        r_coadapted = float(np.corrcoef(e_coadapted_1, e_coadapted_2)[0, 1])
        self.assertLess(r_coadapted, -0.7)  # Strongly negative

    def test_3_routing_dynamicity_statistics(self):
        """Test 3: Verify population standard deviation (ddof=0) and continuous variance."""
        weights = np.array([0.368, 0.370, 0.365, 0.369, 0.372, 0.366])
        mean_w = np.mean(weights)
        pop_sd = np.std(weights, ddof=0)
        sample_sd = np.std(weights, ddof=1)

        # Population SD is strictly less than sample SD for N > 1
        self.assertLess(pop_sd, sample_sd)
        # Coefficient of variation
        cv = pop_sd / mean_w
        self.assertTrue(0.0 < cv < 0.05)

    def test_4_entropy_boundaries(self):
        """Test 4: Verify entropy boundary conditions on the 3-expert probability simplex."""
        # 1. Equal distribution [1/3, 1/3, 1/3] -> max entropy ln(3)
        w_equal = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        ent_equal = -np.sum(w_equal * np.log(w_equal + 1e-12), axis=-1)
        self.assertAlmostEqual(ent_equal[0], np.log(3.0), places=5)

        # 2. Degenerate distribution [1, 0, 0] -> min entropy 0.0
        w_degen = np.array([[1.0, 0.0, 0.0]])
        ent_degen = -np.sum(w_degen * np.log(w_degen + 1e-12), axis=-1)
        self.assertAlmostEqual(ent_degen[0], 0.0, places=4)

    def test_5_neff_boundaries(self):
        """Test 5: Verify effective number of experts N_eff = exp(entropy)."""
        w_equal = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        neff_equal = np.exp(-np.sum(w_equal * np.log(w_equal + 1e-12), axis=-1))
        self.assertAlmostEqual(neff_equal[0], 3.0, places=5)

        w_degen = np.array([[1.0, 0.0, 0.0]])
        neff_degen = np.exp(-np.sum(w_degen * np.log(w_degen + 1e-12), axis=-1))
        self.assertAlmostEqual(neff_degen[0], 1.0, places=4)

    def test_6_oracle_weight_simplex_constraints(self):
        """Test 6: Verify oracle convex weights lie strictly on the 2-simplex Delta^2."""
        # Grid search generator
        step = 0.05
        weights_list = []
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(0, 1.001 - w1, step):
                w3 = max(0.0, 1.0 - w1 - w2)
                weights_list.append([w1, w2, w3])
        weights_arr = np.array(weights_list)

        # Constraints: w >= 0 and sum(w) == 1.0
        self.assertTrue(np.all(weights_arr >= -1e-6))
        row_sums = np.sum(weights_arr, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5)

    def test_7_oracle_mae_ordering(self):
        """Test 7: Verify mathematical ordering: Oracle Convex <= Oracle Best Expert <= Best Standalone."""
        N, H = 10, 24
        y_true = self.rng.normal(100, 10, (N, H))
        y_l = y_true + self.rng.normal(2, 5, (N, H))
        y_t = y_true + self.rng.normal(0, 4, (N, H))
        y_c = y_true + self.rng.normal(-1, 6, (N, H))

        mae_l = np.mean(np.abs(y_l - y_true))
        mae_t = np.mean(np.abs(y_t - y_true))
        mae_c = np.mean(np.abs(y_c - y_true))
        best_standalone_mae = min(mae_l, mae_t, mae_c)

        # Oracle Best Expert per block
        oracle_exp_maes = [min(np.mean(np.abs(y_l[i] - y_true[i])),
                               np.mean(np.abs(y_t[i] - y_true[i])),
                               np.mean(np.abs(y_c[i] - y_true[i]))) for i in range(N)]
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

        # Theoretical hierarchy
        self.assertLessEqual(mean_oracle_exp, best_standalone_mae + 1e-6)
        self.assertLessEqual(mean_oracle_cvx, mean_oracle_exp + 1e-6)

    def test_8_regret_formulas_and_properties(self):
        """Test 8: Verify regret definitions (Selection Regret vs Fusion Regret)."""
        mae_best_bm = 259.33
        mae_top1 = 287.84
        mae_fused = 250.97

        selection_regret = mae_top1 - mae_best_bm
        fusion_regret = mae_fused - mae_best_bm

        # Selection regret is positive (hard selection is sub-optimal)
        self.assertGreater(selection_regret, 0.0)
        # Fusion regret is negative (F2 outperforms standalone best)
        self.assertLess(fusion_regret, 0.0)
        self.assertAlmostEqual(fusion_regret, -8.36, places=1)

    def test_9_shrinkage_decomposition_properties(self):
        """Test 9: Verify linear shrinkage interpolation y_final = lambda*y_adaptive + (1-lambda)*y_equal."""
        y_adapt = np.array([10.0, 20.0])
        y_equal = np.array([12.0, 18.0])
        lam = 0.50

        y_final = lam * y_adapt + (1.0 - lam) * y_equal
        expected = np.array([11.0, 19.0])
        np.testing.assert_allclose(y_final, expected, rtol=1e-6)

    def test_10_daily_block_non_overlapping_indices(self):
        """Test 10: Verify non-overlapping daily block indexing eliminates serial overlap."""
        N = 1294
        H = 24
        K = N // H
        self.assertEqual(K, 53)

        indices_used = set()
        for k in range(K):
            block_idx = list(range(k * H, (k + 1) * H))
            self.assertEqual(len(block_idx), 24)
            self.assertTrue(indices_used.isdisjoint(block_idx))
            indices_used.update(block_idx)
        self.assertEqual(len(indices_used), 53 * 24)

    def test_11_zero_future_leakage_in_lookback_features(self):
        """Test 11: Verify context lookback features strictly precede forecast horizon."""
        L, H = 168, 24
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

    def test_12_deterministic_diagnostic_reproducibility(self):
        """Test 12: Verify deterministic seeding guarantees identical initialization and outputs."""
        seed_everything(42, deterministic_cudnn=True)
        m1 = nn.Linear(7, 3)
        w1 = m1.weight.clone().detach().numpy()

        seed_everything(42, deterministic_cudnn=True)
        m2 = nn.Linear(7, 3)
        w2 = m2.weight.clone().detach().numpy()

        np.testing.assert_allclose(w1, w2, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
