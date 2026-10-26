"""
Unit tests for the Canonical CAEG-Net V1 Interactive Custom Forecast Simulator.
Verifies strict checkpoint loading, parameter counts, input & CSV validation,
training-only scaling, causal context feature order & recent-error calculation,
routing simplex constraints, evaluation-mode determinism, CSV export, and
historical live-versus-cache parity across all 5 evaluation seeds.
"""

from __future__ import annotations

import io
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import data_utils
from dashboard.simulator_engine import (
    CONTEXT_FEATURE_NAMES,
    EXPECTED_SUBMODULE_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    HORIZON_HOURS,
    LOOKBACK_HOURS,
    SUPPORTED_SEEDS,
    InputValidationError,
    build_forecast_dataframe,
    export_forecast_to_csv_bytes,
    generate_hypothetical_scenario_dataframe,
    load_pjm_training_reference,
    parse_and_validate_Sequence_input,
    parse_and_validate_csv_input,
    parse_and_validate_text_input,
    prepare_standardized_input_and_causal_context,
    run_historical_window_validation,
    run_live_custom_forecast,
    verify_and_load_canonical_v1_model,
)


class TestForecastSimulatorEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_pjm_training_reference(use_cache=True)

    def test_01_canonical_v1_checkpoint_loading_and_parameter_count(self):
        """Verify all 5 Canonical V1 checkpoints load with strict=True and 121,531 parameters."""
        for seed in SUPPORTED_SEEDS:
            model, ckpt_path, counts = verify_and_load_canonical_v1_model(seed, use_cache=False)
            self.assertTrue(os.path.isfile(ckpt_path), f"Missing checkpoint at {ckpt_path}")
            self.assertFalse(model.training, "Model must be in evaluation mode (model.eval())")
            self.assertEqual(counts["total_params"], EXPECTED_TOTAL_PARAMS)
            self.assertEqual(counts["total_trainable"], EXPECTED_TOTAL_PARAMS)
            for sub_name, exp_cnt in EXPECTED_SUBMODULE_PARAMS.items():
                self.assertEqual(
                    counts["submodules"][sub_name],
                    exp_cnt,
                    f"Submodule {sub_name} param mismatch for seed {seed}",
                )

    def test_02_input_and_output_shapes(self):
        """Verify input tensor shapes (1, 168, 1), context (1, 4), and output shapes (24,) and (3,)."""
        scen_df = generate_hypothetical_scenario_dataframe(num_hours=192, seed=42)
        val_in = parse_and_validate_csv_input(scen_df, ref=self.ref)
        X_win, C_win, _, _, _ = prepare_standardized_input_and_causal_context(val_in, self.ref)
        self.assertEqual(X_win.shape, (1, LOOKBACK_HOURS, 1))
        self.assertEqual(C_win.shape, (1, 4))

        res = run_live_custom_forecast(val_in, seed=42, ref=self.ref)
        self.assertEqual(res.lookback_168_mw.shape, (LOOKBACK_HOURS,))
        self.assertEqual(res.y_caeg_mw.shape, (HORIZON_HOURS,))
        self.assertEqual(res.y_lstm_mw.shape, (HORIZON_HOURS,))
        self.assertEqual(res.y_tcn_mw.shape, (HORIZON_HOURS,))
        self.assertEqual(res.y_cnn_mw.shape, (HORIZON_HOURS,))
        self.assertEqual(res.y_equal_mw.shape, (HORIZON_HOURS,))
        self.assertEqual(res.routing_weights.shape, (3,))
        self.assertEqual(res.context_vector.shape, (4,))

    def test_03_invalid_input_rejection(self):
        """Verify rejection of <168 observations, non-numeric, NaN/Inf, negative/zero, and constant series."""
        # 1. Fewer than 168 observations
        short_text = ", ".join(str(5000.0 + i) for i in range(167))
        with self.assertRaises(InputValidationError) as ctx:
            parse_and_validate_text_input(short_text, ref=self.ref)
        self.assertIn("167", str(ctx.exception))

        # 2. Non-numeric token
        bad_tokens = [str(5000.0 + i) for i in range(168)]
        bad_tokens[50] = "invalid_mw"
        with self.assertRaises(InputValidationError) as ctx:
            parse_and_validate_text_input("\n".join(bad_tokens), ref=self.ref)
        self.assertIn("Non-numeric", str(ctx.exception))

        # 3. NaN or Inf values
        nan_seq = [5000.0 + (i % 24) * 20.0 for i in range(168)]
        nan_seq[10] = float("nan")
        with self.assertRaises(InputValidationError):
            parse_and_validate_Sequence_input(nan_seq, ref=self.ref)

        inf_seq = [5000.0 + (i % 24) * 20.0 for i in range(168)]
        inf_seq[20] = float("inf")
        with self.assertRaises(InputValidationError):
            parse_and_validate_Sequence_input(inf_seq, ref=self.ref)

        # 4. Physically invalid negative or zero load
        neg_seq = [5000.0 + (i % 24) * 20.0 for i in range(168)]
        neg_seq[0] = -150.0
        with self.assertRaises(InputValidationError):
            parse_and_validate_Sequence_input(neg_seq, ref=self.ref)

        zero_seq = [5000.0 + (i % 24) * 20.0 for i in range(168)]
        zero_seq[0] = 0.0
        with self.assertRaises(InputValidationError):
            parse_and_validate_Sequence_input(zero_seq, ref=self.ref)

        # 5. Constant zero-variance series
        const_seq = [5000.0] * 168
        with self.assertRaises(InputValidationError):
            parse_and_validate_Sequence_input(const_seq, ref=self.ref)

    def test_04_csv_parsing_and_timestamp_validation(self):
        """Verify CSV schema validation, duplicate/out-of-order/gap timestamp rejection, and >168h slicing."""
        # Valid 200-hour CSV uses the latest 168 hours for lookback
        df_200 = generate_hypothetical_scenario_dataframe(num_hours=200, seed=123)
        val_200 = parse_and_validate_csv_input(df_200, ref=self.ref)
        self.assertEqual(val_200.total_observations, 200)
        self.assertEqual(len(val_200.lookback_168_mw), 168)
        np.testing.assert_allclose(val_200.lookback_168_mw, df_200["load_mw"].values[-168:])

        # Reject ambiguous unknown column names
        df_unknown = pd.DataFrame({"foo_col": df_200["load_mw"].values})
        with self.assertRaises(InputValidationError):
            parse_and_validate_csv_input(df_unknown, ref=self.ref)

        # Reject multiple ambiguous recognized load columns
        df_ambig = pd.DataFrame({
            "load": df_200["load_mw"].values,
            "demand_mw": df_200["load_mw"].values,
        })
        with self.assertRaises(InputValidationError):
            parse_and_validate_csv_input(df_ambig, ref=self.ref)

        # Reject duplicate timestamps
        df_dup = df_200.copy()
        df_dup.loc[25, "timestamp"] = df_dup.loc[24, "timestamp"]
        with self.assertRaises(InputValidationError) as ctx:
            parse_and_validate_csv_input(df_dup, ref=self.ref)
        self.assertIn("Duplicate timestamp", str(ctx.exception))

        # Reject out-of-order timestamps
        df_ooo = df_200.copy()
        ts_a = df_ooo.loc[30, "timestamp"]
        ts_b = df_ooo.loc[31, "timestamp"]
        df_ooo.loc[30, "timestamp"] = ts_b
        df_ooo.loc[31, "timestamp"] = ts_a
        with self.assertRaises(InputValidationError) as ctx:
            parse_and_validate_csv_input(df_ooo, ref=self.ref)
        self.assertIn("Out-of-order", str(ctx.exception))

        # Reject missing hourly interval (2-hour gap)
        df_gap = df_200.drop(index=50).reset_index(drop=True)
        with self.assertRaises(InputValidationError) as ctx:
            parse_and_validate_csv_input(df_gap, ref=self.ref)
        self.assertIn("Missing or non-hourly timestamp interval", str(ctx.exception))

    def test_05_training_only_scaling_and_context_feature_order(self):
        """Verify training-only StandardScaler parameters and exact [Trend, Volatility, Periodicity, Recent_Error] order."""
        # Confirm training scaler matches authoritative PJM training split
        self.assertAlmostEqual(self.ref.scaler_mean, 5458.0340, places=3)
        self.assertAlmostEqual(self.ref.scaler_scale, 855.3898, places=3)

        # Verify custom input with shifted scale does NOT refit scaler
        scen_df = generate_hypothetical_scenario_dataframe(num_hours=168, seed=99)
        val_in = parse_and_validate_csv_input(scen_df, ref=self.ref)
        X_win, C_win, _, _, _ = prepare_standardized_input_and_causal_context(val_in, self.ref)

        expected_std = (val_in.lookback_168_mw - self.ref.scaler_mean) / self.ref.scaler_scale
        np.testing.assert_allclose(X_win[0, :, 0], expected_std.astype(np.float32), rtol=1e-6, atol=1e-6)

        # Verify exact context feature order against data_utils.extract_context_features
        expected_c = data_utils.extract_context_features(
            X_win,
            recent_errors=np.array([self.ref.warmup_prior_mae], dtype=np.float32),
        )
        np.testing.assert_allclose(C_win, expected_c, rtol=1e-6, atol=1e-6)
        self.assertEqual(CONTEXT_FEATURE_NAMES, ("Trend", "Volatility", "Periodicity", "Recent_Error"))

    def test_06_causal_recent_error_and_fallback_disclosure(self):
        """Verify exact causal walk-forward recent error when >=192h vs disclosed fallback prior when 168h."""
        # Case A: Exactly 168 observations -> must use training warmup_prior_mae and disclose fallback
        df_168 = generate_hypothetical_scenario_dataframe(num_hours=168, seed=42)
        val_168 = parse_and_validate_csv_input(df_168, ref=self.ref)
        res_168 = run_live_custom_forecast(val_168, seed=42, ref=self.ref)
        self.assertTrue(res_168.recent_error_fallback_used)
        self.assertEqual(res_168.recent_error_mode, "training_warmup_prior_fallback")
        self.assertAlmostEqual(res_168.context_dict["Recent_Error"], self.ref.warmup_prior_mae, places=6)
        self.assertIn("fallback prior", res_168.recent_error_explanation.lower())

        # Case B: 192 observations -> must compute exact causal walk-forward error without fallback
        df_192 = generate_hypothetical_scenario_dataframe(num_hours=192, seed=42)
        val_192 = parse_and_validate_csv_input(df_192, ref=self.ref)
        res_192 = run_live_custom_forecast(val_192, seed=42, ref=self.ref)
        self.assertFalse(res_192.recent_error_fallback_used)
        self.assertEqual(res_192.recent_error_mode, "exact_causal_walk_forward")
        self.assertNotAlmostEqual(res_192.context_dict["Recent_Error"], self.ref.warmup_prior_mae, places=4)

    def test_07_finite_forecasts_routing_simplex_and_determinism(self):
        """Verify finite forecasts, routing weights summing to 1.0, and deterministic repeated inference."""
        df_192 = generate_hypothetical_scenario_dataframe(num_hours=192, seed=2024)
        val_in = parse_and_validate_csv_input(df_192, ref=self.ref)

        res1 = run_live_custom_forecast(val_in, seed=42, ref=self.ref)
        res2 = run_live_custom_forecast(val_in, seed=42, ref=self.ref)

        # Finite checks
        self.assertTrue(np.all(np.isfinite(res1.y_caeg_mw)))
        self.assertTrue(np.all(np.isfinite(res1.y_lstm_mw)))
        self.assertTrue(np.all(np.isfinite(res1.y_tcn_mw)))
        self.assertTrue(np.all(np.isfinite(res1.y_cnn_mw)))
        self.assertTrue(np.all(np.isfinite(res1.routing_weights)))

        # Routing weights simplex check
        self.assertTrue(np.all(res1.routing_weights >= 0.0))
        self.assertAlmostEqual(float(np.sum(res1.routing_weights)), 1.0, places=6)
        self.assertTrue(res1.routing_weights_valid)

        # Exact determinism across repeated calls in eval mode
        np.testing.assert_array_equal(res1.y_caeg_mw, res2.y_caeg_mw)
        np.testing.assert_array_equal(res1.routing_weights, res2.routing_weights)

        # 5-seed ensemble check
        res_ens = run_live_custom_forecast(val_in, seed="ensemble_5seed", ref=self.ref)
        self.assertTrue(np.all(np.isfinite(res_ens.y_caeg_mw)))
        self.assertAlmostEqual(float(np.sum(res_ens.routing_weights)), 1.0, places=6)
        self.assertEqual(len(res_ens.per_seed_caeg_mw), 5)

    def test_08_csv_forecast_export(self):
        """Verify 24-row forecast table construction and CSV serialization for hypothetical scenarios."""
        df_192 = generate_hypothetical_scenario_dataframe(num_hours=192, seed=42)
        val_in = parse_and_validate_csv_input(df_192, ref=self.ref)
        res = run_live_custom_forecast(val_in, seed=42, ref=self.ref)

        out_df = build_forecast_dataframe(res)
        self.assertEqual(len(out_df), HORIZON_HOURS)
        for col in [
            "Horizon_Step_h",
            "CAEG_Net_V1_Forecast_MW",
            "LSTM_Expert_MW",
            "TCN_Expert_MW",
            "CNN_Expert_MW",
            "Equal_Weight_Avg_MW",
            "Weight_LSTM",
            "Weight_TCN",
            "Weight_CNN",
        ]:
            self.assertIn(col, out_df.columns)

        # Hypothetical scenario export must NOT fabricate ground truth columns
        self.assertNotIn("Observed_Actual_MW", out_df.columns)

        csv_bytes = export_forecast_to_csv_bytes(res)
        reloaded = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(reloaded), HORIZON_HOURS)
        np.testing.assert_allclose(reloaded["CAEG_Net_V1_Forecast_MW"].values, res.y_caeg_mw, atol=1e-3)

    def test_09_historical_live_vs_cache_parity(self):
        """Verify live Canonical V1 inference on PJM test windows matches saved Phase 5 cache across all 5 seeds."""
        test_windows = [0, 476, 493, 721, 1185]
        for seed in SUPPORTED_SEEDS:
            for w_idx in test_windows:
                hist_res = run_historical_window_validation(
                    window_idx=w_idx,
                    seed=seed,
                    ref=self.ref,
                    pred_atol_mw=1e-3,
                    weight_atol=1e-6,
                )
                self.assertTrue(hist_res.historical_parity_passed)
                self.assertLessEqual(hist_res.historical_max_abs_pred_diff_mw, 1e-3)
                self.assertLessEqual(hist_res.historical_max_abs_weight_diff, 1e-6)

    def test_10_context_features_regression_against_authoritative_implementation(self):
        """
        Regression test verifying that the simulator's context feature extraction
        reproduces the exact frozen V1 training pipeline (`data_utils.extract_context_features`
        and `src.features.context.extract_context_features`), including first-difference
        sample standard deviation (ddof=1) for Volatility rather than coefficient of variation.
        """
        from src.features.context import extract_context_features as src_extract_context_features

        # 1. Verify on actual PJM test windows (192 contiguous hours -> exact causal recent error + 4D context)
        for w_idx in [24, 476, 493, 721, 1185]:
            # In test_series_extended_mw, window w_idx has lookback [w_idx : w_idx + 168].
            # Providing 192 hours [w_idx - 24 : w_idx + 168] supplies the exact 24h prior origin.
            raw_192_mw = self.ref.test_series_extended_mw[w_idx - 24 : w_idx + LOOKBACK_HOURS]
            val_in = parse_and_validate_Sequence_input(raw_192_mw, ref=self.ref)
            X_win, C_win, rec_mode, fallback_used, _ = prepare_standardized_input_and_causal_context(
                val_in, self.ref
            )
            self.assertEqual(rec_mode, "exact_causal_walk_forward")
            self.assertFalse(fallback_used)

            # Compare against authoritative precomputed test partition C_test[w_idx]
            np.testing.assert_allclose(C_win[0], self.ref.C_test[w_idx], rtol=1e-6, atol=1e-6)

            # Compare against both data_utils.extract_context_features and src.features.context.extract_context_features
            c_data_utils = data_utils.extract_context_features(
                X_win, recent_errors=np.array([self.ref.rec_err_test[w_idx]], dtype=np.float32)
            )
            c_src_feat = src_extract_context_features(
                X_win, recent_errors=np.array([self.ref.rec_err_test[w_idx]], dtype=np.float32)
            )
            np.testing.assert_allclose(C_win, c_data_utils, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(C_win, c_src_feat, rtol=1e-6, atol=1e-6)

            # 2. Verify each mathematical formula explicitly from first principles
            z = X_win[0, :, 0].astype(np.float64)
            L = len(z)
            self.assertEqual(L, 168)

            # (a) Trend: beta_1 / (std(z, ddof=1) + 1e-6)
            i_idx = np.arange(L, dtype=np.float64)
            w_i = i_idx - (L - 1.0) / 2.0
            z_centered = z - np.mean(z)
            beta_1 = np.sum(z_centered * w_i) / np.sum(w_i ** 2)
            expected_trend = beta_1 / (np.std(z, ddof=1) + 1e-6)
            self.assertAlmostEqual(float(C_win[0, 0]), float(expected_trend), places=6)

            # (b) Volatility: sample std (ddof=1) of first differences Delta z_i = z_i - z_{i-1}
            expected_volatility_first_diff = float(np.std(np.diff(z), ddof=1))
            incorrect_cv_formula = float(np.std(z, ddof=0) / (abs(np.mean(z)) + 1e-6))
            self.assertAlmostEqual(float(C_win[0, 1]), expected_volatility_first_diff, places=6)
            self.assertNotAlmostEqual(float(C_win[0, 1]), incorrect_cv_formula, places=2)

            # (c) Periodicity: unclipped lag-24 sample autocorrelation
            expected_periodicity = float(
                np.sum(z_centered[24:] * z_centered[:-24]) / (np.sum(z_centered ** 2) + 1e-6)
            )
            self.assertAlmostEqual(float(C_win[0, 2]), expected_periodicity, places=6)

            # (d) Recent_Error: causal walk-forward MAE at t-24
            self.assertAlmostEqual(float(C_win[0, 3]), float(self.ref.rec_err_test[w_idx]), places=6)


if __name__ == "__main__":
    unittest.main()

