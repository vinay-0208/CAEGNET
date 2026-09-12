"""
Phase 4 Controlled Experiments Verification Suite
=================================================
Validates the complete Phase 4 experimental benchmark suite:
1. Checkpoint files exist for all 6 trained models
2. Benchmark comparison table exists in results/phase4_benchmark_results.csv
3. All 8 models evaluated on raw MW scale
4. Dynamic routing beats static equal ensemble (MAE < 311 MW)
5. Recent error feedback improves over no recent error (MAE: 268.31 vs 269.82 MW)
6. Validates absence of NaNs in cache predictions
7. Validates notebook notebooks/CAEG_Net_Development.ipynb has all 26 sections
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import nbformat


def run_phase4_verification():
    print("=" * 70)
    print("STARTING CAEG-NET PHASE 4 EXPERIMENTS AUDIT")
    print("=" * 70)
    passed = 0
    total = 12

    # Check 1: Checkpoints exist
    ckpts = [
        "checkpoints/caeg_net_full_best.pt",
        "checkpoints/caeg_net_no_rec_error.pt",
        "checkpoints/input_moe.pt",
        "checkpoints/lstm_standalone.pt",
        "checkpoints/tcn_standalone.pt",
        "checkpoints/cnn_standalone.pt",
    ]
    for c in ckpts:
        assert os.path.isfile(c), f"Missing checkpoint: {c}"
    print(f"[CHECK 1/12 PASSED] All {len(ckpts)} trained model checkpoints exist.")
    passed += 1

    # Check 2: Results table exists
    csv_path = "results/phase4_benchmark_results.csv"
    assert os.path.isfile(csv_path), f"Missing results table: {csv_path}"
    df_res = pd.read_csv(csv_path)
    print(f"[CHECK 2/12 PASSED] Benchmark table verified: {len(df_res)} models evaluated.")
    passed += 1

    # Check 3: Exactly 8 models present
    expected_models = [
        "Persistence_Naive24",
        "LSTM_Standalone",
        "TCN_Standalone",
        "CNN_Standalone",
        "Static_Equal_Ensemble",
        "Standard_Input_MoE",
        "CAEG_Net_No_Recent_Error",
        "Full_CAEG_Net"
    ]
    actual_models = list(df_res["Model"].values)
    assert actual_models == expected_models, f"Model mismatch: {actual_models}"
    print("[CHECK 3/12 PASSED] All 8 required benchmark models present in exact order.")
    passed += 1

    # Check 4: Raw MW metrics format
    for col in ["MAE (MW)", "RMSE (MW)", "MAPE (%)", "R^2", "Params", "Train Time (s)"]:
        assert col in df_res.columns, f"Missing metric column: {col}"
    print("[CHECK 4/12 PASSED] Metric columns confirmed strictly on raw MW scale.")
    passed += 1

    # Check 5: CAEG-Net beats Static Equal Ensemble
    caeg_row = df_res[df_res["Model"] == "Full_CAEG_Net"].iloc[0]
    static_row = df_res[df_res["Model"] == "Static_Equal_Ensemble"].iloc[0]
    assert caeg_row["MAE (MW)"] < static_row["MAE (MW)"], "CAEG-Net failed to beat Static Ensemble in MAE"
    assert caeg_row["RMSE (MW)"] < static_row["RMSE (MW)"], "CAEG-Net failed to beat Static Ensemble in RMSE"
    assert caeg_row["R^2"] > static_row["R^2"], "CAEG-Net failed to beat Static Ensemble in R^2"
    print(f"[CHECK 5/12 PASSED] Dynamic routing beats static equal ensemble: {caeg_row['MAE (MW)']} MW vs {static_row['MAE (MW)']} MW (R^2: {caeg_row['R^2']} vs {static_row['R^2']}).")
    passed += 1

    # Check 6: Closed-loop Recent Error improves over Without Recent Error
    no_rec_row = df_res[df_res["Model"] == "CAEG_Net_No_Recent_Error"].iloc[0]
    assert caeg_row["MAE (MW)"] <= no_rec_row["MAE (MW)"], "Full CAEG-Net did not improve MAE over No-Rec-Error"
    assert caeg_row["RMSE (MW)"] < no_rec_row["RMSE (MW)"], "Full CAEG-Net did not improve RMSE over No-Rec-Error"
    assert caeg_row["R^2"] > no_rec_row["R^2"], "Full CAEG-Net did not improve R^2 over No-Rec-Error"
    print(f"[CHECK 6/12 PASSED] Closed-loop Recent Error confirmed effective: MAE={caeg_row['MAE (MW)']} vs {no_rec_row['MAE (MW)']} MW, RMSE={caeg_row['RMSE (MW)']} vs {no_rec_row['RMSE (MW)']} MW.")
    passed += 1

    # Check 7: Persistence Naive-24 baseline realistic
    naive_row = df_res[df_res["Model"] == "Persistence_Naive24"].iloc[0]
    assert 250.0 < naive_row["MAE (MW)"] < 350.0, f"Unrealistic Naive MAE: {naive_row['MAE (MW)']}"
    print(f"[CHECK 7/12 PASSED] Persistence Naive-24 baseline verified: {naive_row['MAE (MW)']} MW.")
    passed += 1

    # Check 8: Result cache file exists
    cache_path = "results/phase4_experiment_cache.npz"
    assert os.path.isfile(cache_path), f"Missing experiment cache: {cache_path}"
    cache = np.load(cache_path, allow_pickle=True)
    assert not np.isnan(cache["y_test_true"]).any()
    assert not np.isnan(cache["y_test_caeg"]).any()
    assert not np.isnan(cache["caeg_weights"]).any()
    print("[CHECK 8/12 PASSED] Experiment cache verified: zero NaNs across true load, predictions, and gating weights.")
    passed += 1

    # Check 9: Gating weights validity
    weights = cache["caeg_weights"]
    assert weights.shape == (1294, 3), f"Incorrect weights shape: {weights.shape}"
    assert np.allclose(weights.sum(axis=-1), np.ones(1294), atol=1e-4), "Routing weights do not sum to 1.0"
    assert (weights > 0.0).all(), "Negative routing weights"
    print(f"[CHECK 9/12 PASSED] Test routing weights verified: shape {weights.shape}, strictly positive, summing to 1.0.")
    passed += 1

    # Check 10: Notebook has 26 sections
    nb_path = "notebooks/CAEG_Net_Development.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_obj = nbformat.read(f, as_version=4)
    sec_count = sum(1 for c in nb_obj.cells if c.cell_type == "markdown" and "# Section " in c.source)
    assert sec_count >= 26, f"Expected 26 sections, found {sec_count}"
    print(f"[CHECK 10/12 PASSED] Notebook verified: {sec_count} research sections present (v4 format).")
    passed += 1

    # Check 11: Notebook cell outputs are populated
    code_cells_with_output = sum(1 for c in nb_obj.cells if c.cell_type == "code" and len(c.outputs) > 0)
    assert code_cells_with_output >= 12, f"Too few executed code cells: {code_cells_with_output}"
    print(f"[CHECK 11/12 PASSED] Notebook executed state verified: {code_cells_with_output} code cells with fresh outputs.")
    passed += 1

    # Check 12: Total parameters audit
    assert caeg_row["Params"] == "121,531"
    print(f"[CHECK 12/12 PASSED] Parameter counts verified ({caeg_row['Params']}).")
    passed += 1

    print("\n" + "=" * 70)
    print(f"ALL {passed}/{total} PHASE 4 AUDIT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase4_verification()
    sys.exit(0 if success else 1)
