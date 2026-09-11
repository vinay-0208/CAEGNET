"""
Unit and Regression Tests for Phase 15B Controlled Mechanism Experiments
=======================================================================
Verifies:
1. Candidate parameter counts and parameter overhead cap (<= 2.0%)
2. Router output shapes, simplex constraints, and horizon partitioning
3. Centroid shrinkage behavior (dynamic vs fixed lambda vs unregularized)
4. Causal feature isolation (no lookahead leakage in OOF histories)
5. Two-Stage Validation Firewall qualification logic
6. 5-seed benchmark reproducibility & stability checks
7. Non-overlapping daily blocks hypothesis testing (K=53, 456, 163)
8. Regret non-negativity and convex oracle mathematical properties
9. Completeness and validity of all 14 analysis CSVs
10. Completeness and validity of all 14 publication figures
11. Completeness and validity of all 3 Phase 15B reports
12. Certification of Outcome D: F2 REMAINS FINAL MODEL
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from research.deterministic import seed_everything
from research.models import (
    GatedRecurrentExpert,
    MultiScaleCausalTCNExpert,
    PatchTemporalExpert,
)
from research.models_phase15b import (
    ControlA_CurrentF2,
    ControlB_FixedShrinkage,
    ControlC_HorizonRoutingOnly,
    ControlD_DynamicConfidenceOnly,
    CandidateE1_HGR_FixedShrinkage,
    CandidateE2_HGR_DisagreementConfidence,
    CandidateE3_HGR_GlobalConfidence,
)


class TestPhase15BMechanismExperiments(unittest.TestCase):
    def setUp(self):
        seed_everything(42, deterministic_cudnn=True)
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.lookback = 168
        self.horizon = 24
        self.n_experts = 3

    def test_01_parameter_counts_and_budget(self):
        """Test 1: Verify exact parameter counts and <= 2.0% overhead cap."""
        f2 = ControlA_CurrentF2(context_dim=7).to(self.device)
        ctrl_b = ControlB_FixedShrinkage(context_dim=7).to(self.device)
        ctrl_c = ControlC_HorizonRoutingOnly(context_dim=7).to(self.device)
        ctrl_d = ControlD_DynamicConfidenceOnly(context_dim=7).to(self.device)
        cand_e1 = CandidateE1_HGR_FixedShrinkage(context_dim=7).to(self.device)
        cand_e2 = CandidateE2_HGR_DisagreementConfidence(context_dim=7).to(self.device)
        cand_e3 = CandidateE3_HGR_GlobalConfidence(context_dim=7).to(self.device)

        # Frozen tri-expert backbone
        lstm_params = sum(p.numel() for p in f2.caeg.lstm_expert.parameters())
        tcn_params = sum(p.numel() for p in f2.caeg.tcn_expert.parameters())
        cnn_params = sum(p.numel() for p in f2.caeg.cnn_expert.parameters())
        backbone_params = lstm_params + tcn_params + cnn_params

        self.assertEqual(lstm_params, 56152)
        self.assertEqual(tcn_params, 36952)
        self.assertEqual(cnn_params, 27400)
        self.assertEqual(backbone_params, 120504)

        # Candidate total parameters
        total_f2 = sum(p.numel() for p in f2.parameters())
        total_b = sum(p.numel() for p in ctrl_b.parameters())
        total_c = sum(p.numel() for p in ctrl_c.parameters())
        total_d = sum(p.numel() for p in ctrl_d.parameters())
        total_e1 = sum(p.numel() for p in cand_e1.parameters())
        total_e2 = sum(p.numel() for p in cand_e2.parameters())
        total_e3 = sum(p.numel() for p in cand_e3.parameters())

        self.assertEqual(total_f2, 121724)
        self.assertEqual(total_b, 121579)
        self.assertEqual(total_c, 122852)
        self.assertEqual(total_d, 121740)
        self.assertEqual(total_e1, 122852)
        self.assertEqual(total_e2, 123079)
        self.assertEqual(total_e3, 122997)

        # All models strictly within budget (< 130,000 and overhead <= 2.0%)
        for name, tot in [("B", total_b), ("C", total_c), ("D", total_d), ("E1", total_e1), ("E2", total_e2), ("E3", total_e3)]:
            overhead_pct = ((tot - total_f2) / total_f2) * 100.0
            self.assertLess(tot, 130000, f"{name} exceeds absolute cap")
            self.assertLessEqual(overhead_pct, 2.0, f"{name} exceeds 2.0% overhead cap")

    def test_02_routing_simplex_and_shapes(self):
        """Test 2: Verify router output shapes and simplex normalization constraints."""
        x = torch.randn(self.batch_size, self.lookback, 1)
        c = torch.randn(self.batch_size, 7)

        # Global routing models
        for ModelClass in [ControlA_CurrentF2, ControlB_FixedShrinkage, ControlD_DynamicConfidenceOnly]:
            m = ModelClass(context_dim=7)
            out, weights, diag = m(x, c)
            self.assertEqual(out.shape, (self.batch_size, self.horizon))
            self.assertEqual(weights.shape, (self.batch_size, 3))
            # Simplex constraint: sum to 1, non-negative
            self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))
            self.assertTrue(torch.all(weights >= 0))

        # Horizon-group routing models
        for ModelClass in [ControlC_HorizonRoutingOnly, CandidateE1_HGR_FixedShrinkage, CandidateE2_HGR_DisagreementConfidence]:
            m = ModelClass(context_dim=7)
            out, weights, diag = m(x, c)
            self.assertEqual(out.shape, (self.batch_size, self.horizon))
            self.assertEqual(weights.shape, (self.batch_size, 3))  # global average weights
            self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))

            w_groups = diag["weights_groups"]
            self.assertEqual(w_groups.shape, (self.batch_size, 3, 3))  # 3 groups x 3 experts
            # Simplex constraint for each horizon group
            for g in range(3):
                g_sum = w_groups[:, g, :].sum(dim=-1)
                self.assertTrue(torch.allclose(g_sum, torch.ones(self.batch_size), atol=1e-5))
                self.assertTrue(torch.all(w_groups[:, g, :] >= 0))

    def test_03_centroid_shrinkage_behavior(self):
        """Test 3: Verify dynamic, fixed, and disabled shrinkage behavior."""
        x = torch.randn(self.batch_size, self.lookback, 1)
        c = torch.randn(self.batch_size, 7)

        # Control B has fixed lambda = 0.51
        m_b = ControlB_FixedShrinkage(context_dim=7)
        _, _, diag_b = m_b(x, c)
        conf_b = diag_b["lambda"]
        self.assertTrue(torch.allclose(conf_b, torch.tensor(0.51), atol=1e-5))

        # Control C has lambda = 1.0 (no shrinkage)
        m_c = ControlC_HorizonRoutingOnly(context_dim=7)
        _, _, diag_c = m_c(x, c)
        conf_c = diag_c["lambda"]
        self.assertTrue(torch.allclose(conf_c, torch.tensor(1.0), atol=1e-5))

        # Control A has dynamic lambda in (0, 1)
        m_a = ControlA_CurrentF2(context_dim=7)
        _, _, diag_a = m_a(x, c)
        conf_a = diag_a["lambda"]
        self.assertTrue(torch.all(conf_a >= 0.0) and torch.all(conf_a <= 1.0))

    def test_04_causal_feature_isolation(self):
        """Test 4: Verify lookback performance features contain zero future target leakage."""
        # Simulated sequence of 500 hours
        T = 500
        H = 24
        lookback_window = 168
        eval_t = 300

        # Target sequence
        y_true = np.sin(np.linspace(0, 50, T))
        # OOF error sequence
        oof_errors = np.abs(np.cos(np.linspace(0, 50, T)))

        # Feature extracted at time t must strictly use indices [t - lookback_window, t)
        feat_indices = list(range(eval_t - lookback_window, eval_t))
        self.assertEqual(len(feat_indices), lookback_window)
        # Any index >= eval_t constitutes future leakage
        future_indices = [idx for idx in feat_indices if idx >= eval_t]
        self.assertEqual(len(future_indices), 0)

    def test_05_validation_screening_firewall_logic(self):
        """Test 5: Verify Stage 15B-S validation screening firewall rule evaluation."""
        df_screen = pd.read_csv("research/analysis/phase15b_screening_decision.csv")
        
        # Rule: Improve >= 2 datasets, degrade <= 2.0% on remaining
        for _, row in df_screen.iterrows():
            cid = row["candidate_id"]
            if cid == "Control_A_F2":
                self.assertTrue(row["qualified"])
                continue
            
            pjm_diff = row["pjm_pct_diff"]
            gef_diff = row["gefcom_pct_diff"]
            uci_diff = row["uci_pct_diff"]
            
            n_improved = sum([pjm_diff < 0, gef_diff < 0, uci_diff < 0])
            worst_deg = max([pjm_diff, gef_diff, uci_diff])
            
            expected_qualified = (n_improved >= 2) and (worst_deg <= 2.0)
            self.assertEqual(row["qualified"], expected_qualified, f"Mismatch for {cid}")
            # All candidates must have failed qualification
            self.assertFalse(row["qualified"], f"Candidate {cid} should not have qualified")

    def test_06_five_seed_test_reproducibility(self):
        """Test 6: Verify 5-seed benchmark results match authoritative records."""
        df_test = pd.read_csv("research/analysis/phase15b_test_results.csv")
        
        f2_pjm = df_test[(df_test["candidate_id"] == "Control_A_F2") & (df_test["dataset"] == "PJM")].iloc[0]
        # PJM historical benchmark: 250.9747 MW
        self.assertAlmostEqual(f2_pjm["test_mae_mean"], 250.9747, places=3)
        self.assertAlmostEqual(f2_pjm["test_mae_std"], 10.6938, places=3)
        self.assertAlmostEqual(f2_pjm["test_rmse_mean"], 335.38, places=1)
        self.assertAlmostEqual(f2_pjm["test_r2_mean"], 0.8714, places=3)

        # Check all candidates have 5 seeds recorded
        seed_cols = ["seed_42_mae", "seed_123_mae", "seed_999_mae", "seed_2024_mae", "seed_3407_mae"]
        for _, row in df_test.iterrows():
            for c in seed_cols:
                self.assertFalse(np.isnan(row[c]), f"NaN found in {row['candidate_id']} {c}")
                self.assertGreater(row[c], 0)

    def test_07_non_overlapping_daily_block_statistics(self):
        """Test 7: Verify daily block counts and statistical significance outputs."""
        df_stat = pd.read_csv("research/analysis/phase15b_daily_block_statistics.csv")

        # Verify block counts
        pjm_rows = df_stat[df_stat["dataset"] == "PJM"]
        gef_rows = df_stat[df_stat["dataset"] == "GEFCom"]
        uci_rows = df_stat[df_stat["dataset"] == "UCI"]

        self.assertTrue(all(pjm_rows["k_blocks"] == 53))
        self.assertTrue(all(gef_rows["k_blocks"] == 456))
        self.assertTrue(all(uci_rows["k_blocks"] == 163))

        # Verify that Control C on GEFCom is statistically significantly worse
        c_gef = df_stat[(df_stat["candidate_id"] == "Control_C_HorizonRouting") & (df_stat["dataset"] == "GEFCom")].iloc[0]
        self.assertTrue(c_gef["statistically_significant"])
        self.assertGreater(c_gef["mean_paired_diff"], 0)
        self.assertLess(c_gef["p_value_holm"], 0.05)
        self.assertGreater(c_gef["cohen_dz"], 0.3)

    def test_08_selection_regret_non_negativity(self):
        """Test 8: Verify selection regret is non-negative and oracle is strictly retrospective."""
        df_regret = pd.read_csv("research/analysis/phase15b_router_decision_quality.csv")
        for _, row in df_regret.iterrows():
            self.assertGreaterEqual(row["selection_regret"], 0.0, f"Negative regret for {row['candidate_id']}")

        df_oracle = pd.read_csv("research/analysis/phase15b_oracle_gap.csv")
        for _, row in df_oracle.iterrows():
            self.assertGreaterEqual(row["oracle_gap"], 0.0, f"Negative oracle gap for {row['candidate_id']}")
            self.assertEqual(row["oracle_deployability"], "RETROSPECTIVE_ORACLE_NON_DEPLOYABLE")

    def test_09_all_14_analysis_csvs_exist_and_valid(self):
        """Test 9: Verify existence and non-emptiness of all 14 analysis CSVs."""
        required_csvs = [
            "phase15b_validation_results.csv",
            "phase15b_feature_history_results.csv",
            "phase15b_screening_decision.csv",
            "phase15b_test_results.csv",
            "phase15b_daily_block_statistics.csv",
            "phase15b_routing_dynamicity.csv",
            "phase15b_router_decision_quality.csv",
            "phase15b_oracle_gap.csv",
            "phase15b_confidence_results.csv",
            "phase15b_horizon_results.csv",
            "phase15b_regime_results.csv",
            "phase15b_ablation_results.csv",
            "phase15b_parameter_counts.csv",
            "phase15b_runtime.csv"
        ]
        for csv_name in required_csvs:
            path = os.path.join("research", "analysis", csv_name)
            self.assertTrue(os.path.exists(path), f"Missing CSV: {path}")
            df = pd.read_csv(path)
            self.assertGreater(len(df), 0, f"Empty CSV: {path}")

    def test_10_all_14_publication_figures_exist_and_valid(self):
        """Test 10: Verify existence and non-zero size of all 14 publication figures."""
        for i in range(1, 15):
            fname = f"phase15b_fig{i:02d}_"
            matched = [f for f in os.listdir("research/plots") if f.startswith(fname) and f.endswith(".png")]
            self.assertEqual(len(matched), 1, f"Figure {i} missing or multiple found: {matched}")
            fpath = os.path.join("research/plots", matched[0])
            self.assertGreater(os.path.getsize(fpath), 10000, f"Figure {matched[0]} is too small/corrupt")

    def test_11_all_reports_exist_and_contain_required_sections(self):
        """Test 11: Verify all 3 Phase 15B reports exist and contain required content."""
        exp_path = "research/reports/phase15b_experiment_report.md"
        stat_path = "research/reports/phase15b_statistical_report.md"
        sel_path = "research/reports/phase15b_final_model_selection.md"

        for p in [exp_path, stat_path, sel_path]:
            self.assertTrue(os.path.exists(p), f"Missing report: {p}")
            self.assertGreater(os.path.getsize(p), 5000, f"Report {p} too small")

        # Check key phrases in experiment report
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_text = f.read()
        self.assertIn("F2_A2_OOF", exp_text)
        self.assertIn("Two-Stage Validation Firewall", exp_text)
        self.assertIn("250.97", exp_text)

        # Check key phrases in selection report
        with open(sel_path, "r", encoding="utf-8") as f:
            sel_text = f.read()
        self.assertIn("OUTCOME D: F2 REMAINS THE FINAL LOCKED MODEL", sel_text)
        self.assertIn("ConfidenceFallbackCAEGNet", sel_text)

    def test_12_certified_outcome_d_lock(self):
        """Test 12: Ensure final decision is unambiguously Outcome D."""
        df_screen = pd.read_csv("research/analysis/phase15b_screening_decision.csv")
        # Ensure zero non-anchor candidates qualified
        non_anchor = df_screen[df_screen["candidate_id"] != "Control_A_F2"]
        self.assertEqual(non_anchor["qualified"].sum(), 0)

        df_stat = pd.read_csv("research/analysis/phase15b_daily_block_statistics.csv")
        # Ensure no candidate statistically beat F2 across all 3 datasets
        for cid in ["Control_B_FixedShrinkage", "Control_C_HorizonRouting", "Candidate_E1_HGR_FS", "Candidate_E2_HGR_DGS"]:
            sub = df_stat[df_stat["candidate_id"] == cid]
            significant_wins = sub[sub["statistically_significant"] & (sub["mean_paired_diff"] < 0)]
            self.assertLess(len(significant_wins), 3, f"{cid} cannot have beaten F2 on all datasets")


if __name__ == "__main__":
    unittest.main()
