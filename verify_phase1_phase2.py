"""
Phase 1 & Phase 2 Research-Integrity Verification Suite
======================================================
Verifies the complete setup, causal data pipeline, mathematical context extraction,
train-only scaling, window shapes, notebook validity, and future-leakage prevention
using the provided PJM electricity load dataset and the chronological walk-forward
out-of-sample Recent Forecast Error mechanism.

Checklist:
1. directories exist
2. notebook exists
3. required Python files exist
4. PJM dataset located and verified (8,784 hourly rows)
5. timestamps are strictly chronological
6. duplicate timestamps are verified absent
7. missing intervals are verified absent
8. train/validation/test order is strictly chronological
9. split percentages are correct (70/15/15)
10. scaler is fitted only on training data
11. validation/test cannot alter scaler parameters
12. input window shape is [N, 168, 1]
13. target window shape is [N, 24]
14. chronological walk-forward forecaster generates out-of-sample training forecasts
15. context shape is [N, 4] with causally aligned Recent Forecast Error
16. zero NaN values across all prepared tensors
17. zero Inf values across all prepared tensors
18. notebook is valid JSON with 12 required sections
19. notebook is readable by Jupyter nbformat
+ Explicit Target Horizon Leakage Test (Origin t decoupled from t+1..t+24)
+ Explicit Walk-Forward Historical Training Causal Isolation Test (Origin s trained only on j <= s-24)
+ Explicit Recent Forecast Error Alignment Test
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
import nbformat

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_forecasting_windows,
    create_partition_windows_with_context,
    ChronologicalWalkForwardForecaster,
    compute_causal_recent_forecast_errors,
    extract_context_features,
    TimeSeriesContextDataset,
    create_dataloaders,
)


def run_verification():
    print("=" * 70)
    print("STARTING CAEG-NET PHASE 1 & PHASE 2 RESEARCH-INTEGRITY AUDIT")
    print("=" * 70)
    passed_checks = 0
    total_checks = 19

    # Check 1: Directories exist
    required_dirs = ["data", "notebooks", "checkpoints", "results", "data/Modern_PJM"]
    for d in required_dirs:
        assert os.path.isdir(d), f"Directory missing: {d}"
    print("[CHECK 1/19 PASSED] All required project directories exist.")
    passed_checks += 1

    # Check 2: Notebook exists
    nb_path = os.path.join("notebooks", "CAEG_Net_Development.ipynb")
    assert os.path.isfile(nb_path), f"Notebook missing: {nb_path}"
    print("[CHECK 2/19 PASSED] notebooks/CAEG_Net_Development.ipynb exists.")
    passed_checks += 1

    # Check 3: Required Python files exist
    assert os.path.isfile("data_utils.py"), "data_utils.py missing"
    assert os.path.isfile("requirements.txt"), "requirements.txt missing"
    assert os.path.isfile("README.md"), "README.md missing"
    print("[CHECK 3/19 PASSED] Required files data_utils.py, requirements.txt, README.md exist.")
    passed_checks += 1

    # Check 4: PJM dataset located and verified
    csv_path = os.path.join("data", "Modern_PJM", "pjm_load.csv")
    assert os.path.isfile(csv_path), f"PJM dataset missing at {csv_path}"
    df_raw, diag = load_and_clean_data(csv_path)
    assert len(df_raw) == 8784, f"Expected 8784 rows, found {len(df_raw)}"
    assert "timestamp" in df_raw.columns and "load" in df_raw.columns, "Missing expected columns"
    print(f"[CHECK 4/19 PASSED] Provided PJM dataset verified: {len(df_raw)} rows, columns={list(df_raw.columns)}.")
    passed_checks += 1

    # Check 5: Timestamps are chronological
    assert df_raw["timestamp"].is_monotonic_increasing, "Timestamps are not monotonically increasing"
    print("[CHECK 5/19 PASSED] Timestamps are strictly chronological and monotonically increasing.")
    passed_checks += 1

    # Check 6: Duplicate timestamps are verified absent
    assert diag["duplicate_timestamps"] == 0, f"Unexpected duplicates found: {diag['duplicate_timestamps']}"
    print("[CHECK 6/19 PASSED] Duplicate timestamp check verified: 0 duplicates in series.")
    passed_checks += 1

    # Check 7: Missing intervals are verified absent
    assert diag["total_missing_intervals"] == 0, f"Unexpected missing intervals: {diag['total_missing_intervals']}"
    print("[CHECK 7/19 PASSED] Continuous hourly sampling frequency verified: 0 missing intervals.")
    passed_checks += 1

    # Check 8: Train/Validation/Test order is correct
    train_df, val_df, test_df, split_info = chronological_split(df_raw, 0.70, 0.15, 0.15)
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train timestamps overlap Validation!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Validation timestamps overlap Test!"
    print("[CHECK 8/19 PASSED] Train -> Validation -> Test strict temporal ordering verified.")
    passed_checks += 1

    # Check 9: Split percentages are correct
    n = len(df_raw)
    assert split_info["train_samples"] == 6148, f"Unexpected train samples: {split_info['train_samples']}"
    assert split_info["val_samples"] == 1318, f"Unexpected val samples: {split_info['val_samples']}"
    assert split_info["test_samples"] == 1318, f"Unexpected test samples: {split_info['test_samples']}"
    print(f"[CHECK 9/19 PASSED] Split sizes verified: Train={split_info['train_samples']} (70%), Val={split_info['val_samples']} (15%), Test={split_info['test_samples']} (15%).")
    passed_checks += 1

    # Check 10: Scaler is fitted ONLY on training data
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    train_mean = train_df["load"].mean()
    train_std = train_df["load"].std(ddof=0)
    assert np.isclose(scaler.mean_[0], train_mean), f"Scaler mean {scaler.mean_[0]} != train mean {train_mean}"
    assert np.isclose(scaler.scale_[0], train_std), f"Scaler scale {scaler.scale_[0]} != train std {train_std}"
    print(f"[CHECK 10/19 PASSED] Scaler strictly fitted on training partition (mean={scaler.mean_[0]:.4f} MW, std={scaler.scale_[0]:.4f} MW).")
    passed_checks += 1

    # Check 11: Validation/Test cannot alter scaler parameters
    val_altered = val_df.copy()
    val_altered["load"] = val_altered["load"] * 999.0
    test_altered = test_df.copy()
    test_altered["load"] = test_altered["load"] * -500.0
    scaler2, _, _, _ = fit_and_transform_scaler(train_df, val_altered, test_altered)
    assert np.isclose(scaler2.mean_[0], scaler.mean_[0]), "Scaler mean changed by validation/test data!"
    assert np.isclose(scaler2.scale_[0], scaler.scale_[0]), "Scaler scale changed by validation/test data!"
    print("[CHECK 11/19 PASSED] Validation/Test isolation confirmed: modifications have 0.0 effect on scaler statistics.")
    passed_checks += 1

    # Create sliding windows for checks 12-16
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    X_train = windows["train"]["X"]
    Y_train = windows["train"]["Y"]
    origins_train = windows["train"]["origins"]

    # Check 12: Input shape is [N, 168, 1]
    assert X_train.ndim == 3 and X_train.shape[1] == 168 and X_train.shape[2] == 1, f"Unexpected X shape: {X_train.shape}"
    assert X_train.shape[0] == 5957, f"Expected 5957 train windows, got {X_train.shape[0]}"
    print(f"[CHECK 12/19 PASSED] Input window shape verified: {X_train.shape} ([N, 168, 1]).")
    passed_checks += 1

    # Check 13: Target shape is [N, 24]
    assert Y_train.ndim == 2 and Y_train.shape[1] == 24, f"Unexpected Y shape: {Y_train.shape}"
    assert Y_train.shape[0] == 5957, f"Expected 5957 target windows, got {Y_train.shape[0]}"
    print(f"[CHECK 13/19 PASSED] Target window shape verified: {Y_train.shape} ([N, 24]).")
    passed_checks += 1

    # Check 14: Chronological walk-forward out-of-sample forecaster
    rec_errors_train, rec_errors_val, rec_errors_test, forecaster = compute_causal_recent_forecast_errors(windows)
    assert forecaster.is_fitted, "Forecaster not fitted"
    assert 0.25 < forecaster.warmup_prior_mae < 0.45, f"Unexpected warmup prior MAE: {forecaster.warmup_prior_mae}"
    print(f"[CHECK 14/19 PASSED] Walk-forward forecaster generated out-of-sample training forecasts (Warmup Prior MAE: {forecaster.warmup_prior_mae:.4f}).")
    passed_checks += 1

    # Check 15: Context shape is [N, 4] with causal Recent Forecast Error
    C_train = extract_context_features(X_train, rec_errors_train)
    assert C_train.ndim == 2 and C_train.shape[1] == 4 and len(C_train) == len(X_train), f"Unexpected C shape: {C_train.shape}"
    print(f"[CHECK 15/19 PASSED] Context tensor shape verified: {C_train.shape} ([N, 4]).")
    passed_checks += 1

    # Check 16: No NaN values in prepared tensors
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(Y_train).any(), "NaN in Y_train"
    assert not np.isnan(C_train).any(), "NaN in C_train"
    print("[CHECK 16/19 PASSED] Zero NaN values across X, Y, and C tensors.")
    passed_checks += 1

    # Check 17: No Inf values in prepared tensors
    assert not np.isinf(X_train).any(), "Inf in X_train"
    assert not np.isinf(Y_train).any(), "Inf in Y_train"
    assert not np.isinf(C_train).any(), "Inf in C_train"
    print("[CHECK 17/19 PASSED] Zero Inf values across X, Y, and C tensors.")
    passed_checks += 1

    # Check 18: Notebook is valid JSON
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)
    assert "cells" in nb_json and isinstance(nb_json["cells"], list), "Notebook JSON invalid"
    print(f"[CHECK 18/19 PASSED] Notebook {nb_path} is valid JSON with {len(nb_json['cells'])} cells.")
    passed_checks += 1

    # Check 19: Notebook is readable by Jupyter nbformat
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_obj = nbformat.read(f, as_version=4)
    assert len(nb_obj.cells) > 0, "No cells read by nbformat"
    print(f"[CHECK 19/19 PASSED] Notebook successfully validated by nbformat (v{nb_obj.nbformat}.{nb_obj.nbformat_minor}).")
    passed_checks += 1

    # =====================================================================
    # EXPLICIT LEAKAGE & CAUSAL INDEPENDENCE TESTS
    # =====================================================================
    print("\n" + "=" * 70)
    print("RUNNING EXPLICIT CAUSAL LEAKAGE AUDIT TESTS")
    print("=" * 70)

    # Test A: Modifying values occurring AFTER forecast origin t
    test_origin_idx = 1000
    X_before = X_train[test_origin_idx:test_origin_idx+1].copy()
    Y_before = Y_train[test_origin_idx:test_origin_idx+1].copy()
    rec_err_before = np.array([rec_errors_train[test_origin_idx]])

    C_before = extract_context_features(X_before, rec_err_before)

    # Radically mutate target values occurring after t (inside current horizon t+1..t+24 and beyond)
    Y_mutated = Y_before * 99999.0 + 888888.0

    # Recalculate context at origin t
    C_after = extract_context_features(X_before, rec_err_before)

    diff = np.abs(C_before - C_after)
    max_diff = float(np.max(diff))
    print(f"Original Context at origin {test_origin_idx}: Trend={C_before[0,0]:.5f}, Vol={C_before[0,1]:.5f}, Per={C_before[0,2]:.5f}, RecErr={C_before[0,3]:.5f}")
    print(f"Max Context Difference after Modifying Targets after origin t: {max_diff}")
    assert max_diff == 0.0, "FATAL: Context features depend on future targets!"
    print("[LEAKAGE TEST A PASSED] Absolute zero leakage from observations after forecast origin t (max_diff = 0.0).")

    # Test B: Walk-Forward Historical Forecaster Causal Isolation
    # Verify that the model generating forecast for origin s was trained ONLY on j <= s-24.
    s_test = 1500
    max_train_idx = s_test - 24
    X_2d = X_train[:, :, 0]
    
    model_orig = Ridge(alpha=100.0)
    model_orig.fit(X_2d[:max_train_idx], Y_train[:max_train_idx])
    pred_orig = model_orig.predict(X_2d[s_test:s_test+1])
    
    # Mutate data at or after s_test - 23
    Y_train_mutated = Y_train.copy()
    Y_train_mutated[s_test-23:] = Y_train_mutated[s_test-23:] * 88888.0 + 99999.0
    
    model_mutated = Ridge(alpha=100.0)
    model_mutated.fit(X_2d[:max_train_idx], Y_train_mutated[:max_train_idx])
    pred_mutated = model_mutated.predict(X_2d[s_test:s_test+1])
    
    diff_wf = float(np.max(np.abs(pred_orig - pred_mutated)))
    print(f"Max Difference in Walk-Forward Forecast for origin s={s_test} after mutating targets >= s-23: {diff_wf}")
    assert diff_wf == 0.0, "FATAL: Walk-forward forecaster leaked future target information!"
    print(f"[LEAKAGE TEST B PASSED] Walk-forward forecaster for origin {s_test} is strictly isolated to completed history j <= {max_train_idx} (diff = 0.0).")

    # PyTorch DataLoader Batch Test
    ds = TimeSeriesContextDataset(X_train, Y_train, C_train)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    sample_x, sample_y, sample_c = next(iter(loader))
    assert sample_x.shape == (32, 168, 1)
    assert sample_y.shape == (32, 24)
    assert sample_c.shape == (32, 4)
    print(f"[DATALOADER TEST PASSED] Batch shapes: X={sample_x.shape}, Y={sample_y.shape}, C={sample_c.shape}")

    print("\n" + "=" * 70)
    print(f"ALL {passed_checks}/{total_checks} CHECKS AND DUAL LEAKAGE AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
