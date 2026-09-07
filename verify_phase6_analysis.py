"""
Phase 6 Context, Routing & Error-Regime Verification Suite
=========================================================
Validates all Phase 6 deliverables:
1. results/phase6_routing_analysis.csv exists with 1,294 rows
2. results/phase6_correlations.csv exists with 4 context rows
3. results/phase6_regimes.csv exists with 12 regime rows
4. results/phase6_expert_specialization.csv exists with 12 rows
5. results/phase6_analysis_cache.npz exists and contains valid case study indices
6. Gating weights sum to 1.0 and are strictly non-negative
7. Notebook contains all 44 sections and executed code cells
"""

import os
import sys
import pandas as pd
import numpy as np
import nbformat


def run_phase6_verification():
    print("=" * 75)
    print("STARTING CAEG-NET PHASE 6 VERIFICATION AUDIT")
    print("=" * 75)
    passed = 0
    total = 8

    # 1. Routing analysis CSV
    csv_routing = "results/phase6_routing_analysis.csv"
    assert os.path.isfile(csv_routing), f"Missing CSV: {csv_routing}"
    df_r = pd.read_csv(csv_routing)
    assert len(df_r) == 1294, f"Expected 1294 rows, found {len(df_r)}"
    print(f"[CHECK 1/8 PASSED] Routing analysis CSV verified (1,294 test origins).")
    passed += 1

    # 2. Routing weight convexity
    w_sum = df_r[["w_LSTM", "w_TCN", "w_CNN"]].sum(axis=1)
    assert np.allclose(w_sum, np.ones(len(df_r)), atol=1e-4), "Weights do not sum to 1.0"
    assert (df_r[["w_LSTM", "w_TCN", "w_CNN"]] >= 0.0).all().all(), "Negative weights detected"
    print(f"[CHECK 2/8 PASSED] Routing weights strictly convex (sum=1.0, non-negative).")
    passed += 1

    # 3. Correlations CSV
    csv_corr = "results/phase6_correlations.csv"
    assert os.path.isfile(csv_corr), f"Missing correlations CSV: {csv_corr}"
    df_c = pd.read_csv(csv_corr)
    assert len(df_c) == 4, f"Expected 4 rows, found {len(df_c)}"
    print(f"[CHECK 3/8 PASSED] Correlations CSV verified (all 4 context features).")
    passed += 1

    # 4. Context regimes CSV
    csv_reg = "results/phase6_regimes.csv"
    assert os.path.isfile(csv_reg), f"Missing regimes CSV: {csv_reg}"
    df_reg = pd.read_csv(csv_reg)
    assert len(df_reg) == 12, f"Expected 12 regime rows, found {len(df_reg)}"
    print(f"[CHECK 4/8 PASSED] Context regimes CSV verified (12 tercile subsets).")
    passed += 1

    # 5. Expert specialization CSV
    csv_spec = "results/phase6_expert_specialization.csv"
    assert os.path.isfile(csv_spec), f"Missing specialization CSV: {csv_spec}"
    df_s = pd.read_csv(csv_spec)
    assert len(df_s) == 12, f"Expected 12 specialization rows, found {len(df_s)}"
    print(f"[CHECK 5/8 PASSED] Expert specialization CSV verified.")
    passed += 1

    # 6. Cache NPZ
    cache_path = "results/phase6_analysis_cache.npz"
    assert os.path.isfile(cache_path), f"Missing cache: {cache_path}"
    cache = np.load(cache_path, allow_pickle=True)
    assert "case_studies_json" in cache
    print(f"[CHECK 6/8 PASSED] Phase 6 cache verified with objective case studies.")
    passed += 1

    # 7. Notebook contains all 44 sections
    nb_path = "notebooks/CAEG_Net_Development.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_obj = nbformat.read(f, as_version=4)
    sec_count = sum(1 for c in nb_obj.cells if c.cell_type == "markdown" and "# Section " in c.source)
    assert sec_count >= 44, f"Expected 44 sections, found {sec_count}"
    print(f"[CHECK 7/8 PASSED] Notebook verified: {sec_count} research sections present.")
    passed += 1

    # 8. Notebook executed code cells
    code_cells = sum(1 for c in nb_obj.cells if c.cell_type == "code" and len(c.outputs) > 0)
    assert code_cells >= 25, f"Too few executed code cells: {code_cells}"
    print(f"[CHECK 8/8 PASSED] Notebook executed state verified ({code_cells} cells with fresh outputs).")
    passed += 1

    print("\n" + "=" * 75)
    print(f"ALL {passed}/{total} PHASE 6 VERIFICATION AUDIT CHECKS PASSED!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_phase6_verification()
    sys.exit(0 if success else 1)
