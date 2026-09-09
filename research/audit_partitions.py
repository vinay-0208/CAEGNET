import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch

from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)
from caeg_net import LSTMExpert, TCNExpert, CNNExpert, CAEGNet
from evaluate import compute_metrics

def audit_equal_ensemble():
    data_path = "data/Modern_PJM/pjm_load.csv"
    df, clean_diag = load_and_clean_data(data_path)
    train_df, val_df, test_df, split_info = chronological_split(df, 0.70, 0.15, 0.15)
    scaler, train_sc, val_sc, test_sc = fit_and_transform_scaler(train_df, val_df, test_df)
    windows = create_partition_windows_with_context(train_sc, val_sc, test_sc, lookback=168, horizon=24)
    
    print("=== DATASET & PARTITION AUDIT ===")
    print("Total rows in CSV:", len(df))
    print(f"Train samples: {len(train_df)} ({split_info['train_start']} to {split_info['train_end']})")
    print(f"Val samples: {len(val_df)} ({split_info['val_start']} to {split_info['val_end']})")
    print(f"Test samples: {len(test_df)} ({split_info['test_start']} to {split_info['test_end']})")
    print(f"Train windows: X={windows['train']['X'].shape}, Y={windows['train']['Y'].shape}, origins={len(windows['train']['origins'])}")
    print(f"Val windows: X={windows['val']['X'].shape}, Y={windows['val']['Y'].shape}, origins={len(windows['val']['origins'])}")
    print(f"Test windows: X={windows['test']['X'].shape}, Y={windows['test']['Y'].shape}, origins={len(windows['test']['origins'])}")
    print("First test origin:", windows['test']['origins'][0], "Last test origin:", windows['test']['origins'][-1])

    # Check results/phase5_summary.csv vs research/results/phase5/experiments.csv
    print("\n=== HISTORICAL VS CURRENT PHASE 5 EXPERIMENT COMPARISON ===")
    if os.path.exists("research/results/phase5/experiments.csv"):
        exp_df = pd.read_csv("research/results/phase5/experiments.csv")
        print("Experiments CSV loaded. Seeds:", exp_df["seed"].unique())
        for s in exp_df["seed"].unique():
            sub = exp_df[exp_df["seed"] == s].set_index("model")
            lstm_mae = sub.loc["Standalone_LSTM", "MAE"]
            tcn_mae = sub.loc["Standalone_TCN", "MAE"]
            cnn_mae = sub.loc["Standalone_CNN", "MAE"]
            eq_mae = sub.loc["Static_Equal_Ensemble", "MAE"]
            v1_mae = sub.loc["Original_CAEGNet_V1", "MAE"]
            print(f"Seed {s:4d} | LSTM: {lstm_mae:6.2f} | TCN: {tcn_mae:6.2f} | CNN: {cnn_mae:6.2f} | Equal: {eq_mae:6.2f} | V1: {v1_mae:6.2f}")

if __name__ == "__main__":
    audit_equal_ensemble()
