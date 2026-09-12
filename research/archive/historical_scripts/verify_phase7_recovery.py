"""
Phase 7 Verification Suite: Complete Metrics Recovery & Analysis
================================================================
Validates all Phase 7 requirements:
1. phase7_seed_metrics.csv exists with 36 rows
2. phase7_performance_summary.csv exists with 8 model rows
3. phase7_context_regime_performance.csv exists with 12 regime rows
4. phase7_error_regime_performance.csv exists with 3 error regime rows
5. All 5 metrics (MAE, MSE, RMSE, R2, MAPE) are populated with valid non-null values
6. Mathematical consistency: RMSE == sqrt(MSE) for all rows
7. Recovered Phase 7 metrics match Phase 5 reported values within 0.05 tolerance
8. Test set load values strictly positive (no division by zero in MAPE)
9. Notebook contains 57 sections and executed code cells
"""

import os
import sys
import pandas as pd
import numpy as np
import nbformat


def run_phase7_verification():
    print("=" * 80)
    print("STARTING CAEG-NET PHASE 7 METRICS RECOVERY AUDIT")
    print("=" * 80)
    passed = 0
    total = 9

    # 1. Seed metrics CSV
    seed_csv = "results/phase7_seed_metrics.csv"
    assert os.path.isfile(seed_csv), f"Missing file: {seed_csv}"
    df_seed = pd.read_csv(seed_csv)
    assert len(df_seed) == 36, f"Expected 36 rows, found {len(df_seed)}"
    print(f"[CHECK 1/9 PASSED] Seed metrics CSV verified (36 rows: 1 deterministic + 7 models * 5 seeds).")
    passed += 1

    # 2. Performance summary CSV
    sum_csv = "results/phase7_performance_summary.csv"
    assert os.path.isfile(sum_csv), f"Missing file: {sum_csv}"
    df_sum = pd.read_csv(sum_csv)
    assert len(df_sum) == 8, f"Expected 8 models, found {len(df_sum)}"
    print(f"[CHECK 2/9 PASSED] Performance summary CSV verified (all 8 benchmark models).")
    passed += 1

    # 3. Context regime performance CSV
    ctx_csv = "results/phase7_context_regime_performance.csv"
    assert os.path.isfile(ctx_csv), f"Missing file: {ctx_csv}"
    df_ctx = pd.read_csv(ctx_csv)
    assert len(df_ctx) == 12, f"Expected 12 regime rows, found {len(df_ctx)}"
    print(f"[CHECK 3/9 PASSED] Context regime performance CSV verified (12 terciles).")
    passed += 1

    # 4. Error regime performance CSV
    err_csv = "results/phase7_error_regime_performance.csv"
    assert os.path.isfile(err_csv), f"Missing file: {err_csv}"
    df_err = pd.read_csv(err_csv)
    assert len(df_err) == 3, f"Expected 3 error regime rows, found {len(df_err)}"
    print(f"[CHECK 4/9 PASSED] Error regime performance CSV verified (3 diagnostic terciles).")
    passed += 1

    # 5. Non-null metrics
    req_cols = ["MAE_MW", "MSE_MW2", "RMSE_MW", "R2", "MAPE_percent"]
    assert df_seed[req_cols].notnull().all().all(), "Null values found in seed metrics!"
    print(f"[CHECK 5/9 PASSED] All 5 required metrics fully populated with valid non-null values.")
    passed += 1

    # 6. RMSE == sqrt(MSE) consistency
    max_diff = np.max(np.abs(df_seed["RMSE_MW"] - np.sqrt(df_seed["MSE_MW2"])))
    assert max_diff < 0.02, f"RMSE mismatch: max diff = {max_diff}"
    print(f"[CHECK 6/9 PASSED] Mathematical consistency confirmed: RMSE == sqrt(MSE) (max diff = {max_diff:.6f}).")
    passed += 1

    # 7. Verification against Phase 5 reported values
    caeg_row = df_sum[df_sum["model"] == "Full_CAEG_Net"].iloc[0]
    assert abs(caeg_row["MAE_mean"] - 251.44) < 0.05, "MAE mismatch with Phase 5!"
    assert abs(caeg_row["RMSE_mean"] - 334.32) < 0.05, "RMSE mismatch with Phase 5!"
    assert abs(caeg_row["R2_mean"] - 0.8723) < 0.001, "R2 mismatch with Phase 5!"
    print(f"[CHECK 7/9 PASSED] Full CAEG-Net recovered metrics match Phase 5 reported values exactly.")
    passed += 1

    # 8. MAPE safety (minimum load > 0)
    cache = np.load("results/phase5_multiseed_cache.npz", allow_pickle=True)
    min_load = np.min(cache["y_true_raw"])
    assert min_load > 0, "Non-positive load detected!"
    print(f"[CHECK 8/9 PASSED] MAPE safety verified: min load = {min_load:.2f} MW > 0.")
    passed += 1

    # 9. Notebook verified
    nb_path = "notebooks/CAEG_Net_Development.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_obj = nbformat.read(f, as_version=4)
    sec_count = sum(1 for c in nb_obj.cells if c.cell_type == "markdown" and "# Section " in c.source)
    assert sec_count >= 57, f"Expected at least 57 sections, found {sec_count}"
    code_cells = sum(1 for c in nb_obj.cells if c.cell_type == "code" and len(c.outputs) > 0)
    assert code_cells >= 35, f"Too few executed code cells: {code_cells}"
    print(f"[CHECK 9/9 PASSED] Notebook verified: {sec_count} sections present, {code_cells} executed cells.")
    passed += 1

    print("\n" + "=" * 80)
    print(f"ALL {passed}/{total} PHASE 7 AUDIT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_phase7_verification()
    sys.exit(0 if success else 1)
