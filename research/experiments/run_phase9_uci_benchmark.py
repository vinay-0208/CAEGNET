"""
Phase 9B: UCI ElectricityLoadDiagrams20112014 Locked Five-Seed Generalization Benchmark
======================================================================================
Evaluates the canonical Original CAEG-Net V1 against Standalone LSTM, TCN, CNN, and Static Equal Ensemble
on the frozen Fixed Cohort 320 over 2012-2014.

Governing Research Protocol:
- Dataset: UCI ElectricityLoadDiagrams20112014 (LD2011_2014.txt)
- Fixed Cohort: 320 clients (first_active_timestamp <= 2012-01-01 00:15:00)
- Aggregation: Hourly mean power in kW -> System Aggregate Load in MW (sum / 1000.0)
- Horizon: 168-hour history -> 24-hour forecast, rolling step 1
- Chronological Split: 70% train (18,413h), 15% val (3,945h), 15% test (3,946h)
- Scaler: StandardScaler fitted strictly on train partition
- Context: Canonical 4D vector (Trend, Volatility, Lag-24 Autocorrelation, Recent Ridge Error)
- 5 Canonical Seeds: 42, 123, 999, 2024, 3407
- Training: AdamW(lr=1e-3, weight_decay=1e-4), StepLR(step_size=15, gamma=0.5),
            max_epochs=25, patience=6, loss=MSE, checkpoint=val MSE, batch_size=64.
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from caeg_net import LSTMExpert, TCNExpert, CNNExpert, count_parameters
from research.original_caeg import OriginalCAEGNetPhase5
from evaluate import compute_metrics
from research.data_adapter_uci import UCIElectricityDatasetAdapter

RAW_UCI_PATH = os.path.join(repo_root, "data", "ElectricityLoadDiagrams20112014", "LD2011_2014.txt")
AUDIT_CSV_PATH = os.path.join(repo_root, "research", "results", "phase9_cohort_audit.csv")
PLOTS_DIR = os.path.join(repo_root, "research", "results", "phase9_plots")
RESULTS_DIR = os.path.join(repo_root, "research", "results")

SEEDS = [42, 123, 999, 2024, 3407]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    step_size: int = 15,
    gamma: float = 0.5,
    max_epochs: int = 25,
    patience: int = 6,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_weights = None
    start_time = time.time()
    actual_epochs = 0

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "router"):
                out, _, _ = model(x, c)
            else:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]

            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
                c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

                if hasattr(model, "router"):
                    y_pred, _, _ = model(x, c)
                else:
                    y_pred = model(x)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]

                val_mse = F.mse_loss(y_pred, y)
                total_val_loss += val_mse.item()
                n_val_batches += 1

        val_loss = total_val_loss / max(n_val_batches, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "best_val_loss": best_val_loss,
        "train_time_seconds": time.time() - start_time,
    }


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    scaler: object,
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds = []
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            c = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None

            if hasattr(model, "router"):
                y_pred, w, _ = model(x, c)
                all_weights.append(w.cpu().numpy())
            else:
                y_pred = model(x)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

            all_preds.append(y_pred.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0) # [N, 24] scaled
    unscaled_preds = preds * scaler.scale_[0] + scaler.mean_[0]

    result = {"preds": unscaled_preds}
    if all_weights:
        result["weights"] = np.concatenate(all_weights, axis=0) # [N, 3]
    return result


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))
    
    nonzero_mask = np.abs(y_true) > 1e-4
    mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "mape": mape
    }


def run_benchmark():
    print("=" * 80)
    print("PHASE 9B: UCI ELECTRICITY LOAD DIAGRAMS LOCKED FIVE-SEED GENERALIZATION BENCHMARK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Load Data
    print("\nLoading and preprocessing Fixed Cohort 320 over 2012-2014...")
    adapter = UCIElectricityDatasetAdapter(
        lookback=168,
        horizon=24,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        unit="MW",
        start_year=2012,
        cohort="cohort_320"
    )
    prep = adapter.load_and_preprocess(raw_path=RAW_UCI_PATH, audit_csv_path=AUDIT_CSV_PATH)
    
    windows = prep["windows"]
    scaler = prep["scaler"]
    context = prep["context"]
    split_info = prep["split_info"]
    
    print("Dataset summary:")
    print(f"  Total hourly observations: {split_info['n_total']}")
    print(f"  Train: {split_info['n_train']} hours ({split_info['train_start']} to {split_info['train_end']})")
    print(f"  Val:   {split_info['n_val']} hours ({split_info['val_start']} to {split_info['val_end']})")
    print(f"  Test:  {split_info['n_test']} hours ({split_info['test_start']} to {split_info['test_end']})")
    print(f"  Train windows: {len(windows['train']['X'])}, Val: {len(windows['val']['X'])}, Test: {len(windows['test']['X'])}")
    print(f"  Scaler mean: {scaler.mean_[0]:.4f} MW, scale: {scaler.scale_[0]:.4f} MW")

    y_test_scaled = windows["test"]["Y"]
    y_test_unscaled = y_test_scaled * scaler.scale_[0] + scaler.mean_[0]

    X_tr = torch.tensor(windows["train"]["X"], dtype=torch.float32)
    Y_tr = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
    C_tr = torch.tensor(context["train"], dtype=torch.float32)

    X_va = torch.tensor(windows["val"]["X"], dtype=torch.float32)
    Y_va = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
    C_va = torch.tensor(context["val"], dtype=torch.float32)

    X_te = torch.tensor(windows["test"]["X"], dtype=torch.float32)
    Y_te = torch.tensor(windows["test"]["Y"], dtype=torch.float32)
    C_te = torch.tensor(context["test"], dtype=torch.float32)

    train_ds = TensorDataset(X_tr, Y_tr, C_tr)
    val_ds = TensorDataset(X_va, Y_va, C_va)
    test_ds = TensorDataset(X_te, Y_te, C_te)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    seed_results = []
    test_predictions = {
        "LSTM": [],
        "TCN": [],
        "CNN": [],
        "Equal_Ensemble": [],
        "CAEG_Net_V1": []
    }
    caeg_routing_records = []

    m_lstm_temp = LSTMExpert(input_dim=1, hidden_dim=64, num_layers=2, horizon=24, dropout=0.1)
    m_tcn_temp = TCNExpert(input_dim=1, channels=32, dilations=(1, 2, 4, 8, 16, 32), kernel_size=3, horizon=24, dropout=0.1)
    m_cnn_temp = CNNExpert(input_dim=1, horizon=24, dropout=0.1)
    m_caeg_temp = OriginalCAEGNetPhase5(input_dim=1, horizon=24, context_dim=4, latent_context_dim=16, lstm_hidden=64, lstm_layers=2, tcn_channels=32, dropout=0.1)

    param_counts = {
        "Standalone LSTM": count_parameters(m_lstm_temp)["total_params"],
        "Standalone TCN": count_parameters(m_tcn_temp)["total_params"],
        "Standalone CNN": count_parameters(m_cnn_temp)["total_params"],
        "Static Equal Ensemble": count_parameters(m_lstm_temp)["total_params"] + count_parameters(m_tcn_temp)["total_params"] + count_parameters(m_cnn_temp)["total_params"],
        "Original CAEG-Net V1": count_parameters(m_caeg_temp)["total_params"]
    }
    print(f"\nModel Parameter Counts: {param_counts}")

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n==================================================")
        print(f"RUNNING SEED {seed} ({seed_idx}/5)")
        print(f"==================================================")

        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        # 1. Standalone LSTM
        print(f"\n--- [Seed {seed}] Training Standalone LSTM ---")
        set_seed(seed)
        lstm_model = LSTMExpert(input_dim=1, hidden_dim=64, num_layers=2, horizon=24, dropout=0.1)
        lstm_train_info = train_model(lstm_model, train_loader, val_loader, max_epochs=25, patience=6, device=device)
        lstm_preds = predict_model(lstm_model, test_loader, scaler, device)["preds"]
        lstm_metrics = evaluate_metrics(y_test_unscaled, lstm_preds)
        test_predictions["LSTM"].append(lstm_preds)
        print(f"  LSTM Test MAE: {lstm_metrics['mae']:.4f} MW | Best Epoch: {lstm_train_info['best_epoch']} | Time: {lstm_train_info['train_time_seconds']:.1f}s")

        seed_results.append({
            "model": "Standalone LSTM",
            "seed": seed,
            "best_epoch": lstm_train_info["best_epoch"],
            "actual_epochs": lstm_train_info["actual_epochs"],
            "val_mse": lstm_train_info["best_val_loss"],
            "test_mae": lstm_metrics["mae"],
            "test_rmse": lstm_metrics["rmse"],
            "test_mse": lstm_metrics["mse"],
            "test_r2": lstm_metrics["r2"],
            "test_mape": lstm_metrics["mape"],
            "train_time_s": lstm_train_info["train_time_seconds"],
            "params": param_counts["Standalone LSTM"]
        })

        # 2. Standalone TCN
        print(f"\n--- [Seed {seed}] Training Standalone TCN ---")
        set_seed(seed)
        tcn_model = TCNExpert(input_dim=1, channels=32, dilations=(1, 2, 4, 8, 16, 32), kernel_size=3, horizon=24, dropout=0.1)
        tcn_train_info = train_model(tcn_model, train_loader, val_loader, max_epochs=25, patience=6, device=device)
        tcn_preds = predict_model(tcn_model, test_loader, scaler, device)["preds"]
        tcn_metrics = evaluate_metrics(y_test_unscaled, tcn_preds)
        test_predictions["TCN"].append(tcn_preds)
        print(f"  TCN Test MAE: {tcn_metrics['mae']:.4f} MW | Best Epoch: {tcn_train_info['best_epoch']} | Time: {tcn_train_info['train_time_seconds']:.1f}s")

        seed_results.append({
            "model": "Standalone TCN",
            "seed": seed,
            "best_epoch": tcn_train_info["best_epoch"],
            "actual_epochs": tcn_train_info["actual_epochs"],
            "val_mse": tcn_train_info["best_val_loss"],
            "test_mae": tcn_metrics["mae"],
            "test_rmse": tcn_metrics["rmse"],
            "test_mse": tcn_metrics["mse"],
            "test_r2": tcn_metrics["r2"],
            "test_mape": tcn_metrics["mape"],
            "train_time_s": tcn_train_info["train_time_seconds"],
            "params": param_counts["Standalone TCN"]
        })

        # 3. Standalone CNN
        print(f"\n--- [Seed {seed}] Training Standalone CNN ---")
        set_seed(seed)
        cnn_model = CNNExpert(input_dim=1, horizon=24, dropout=0.1)
        cnn_train_info = train_model(cnn_model, train_loader, val_loader, max_epochs=25, patience=6, device=device)
        cnn_preds = predict_model(cnn_model, test_loader, scaler, device)["preds"]
        cnn_metrics = evaluate_metrics(y_test_unscaled, cnn_preds)
        test_predictions["CNN"].append(cnn_preds)
        print(f"  CNN Test MAE: {cnn_metrics['mae']:.4f} MW | Best Epoch: {cnn_train_info['best_epoch']} | Time: {cnn_train_info['train_time_seconds']:.1f}s")

        seed_results.append({
            "model": "Standalone CNN",
            "seed": seed,
            "best_epoch": cnn_train_info["best_epoch"],
            "actual_epochs": cnn_train_info["actual_epochs"],
            "val_mse": cnn_train_info["best_val_loss"],
            "test_mae": cnn_metrics["mae"],
            "test_rmse": cnn_metrics["rmse"],
            "test_mse": cnn_metrics["mse"],
            "test_r2": cnn_metrics["r2"],
            "test_mape": cnn_metrics["mape"],
            "train_time_s": cnn_train_info["train_time_seconds"],
            "params": param_counts["Standalone CNN"]
        })

        # 4. Static Equal Ensemble
        equal_preds = (lstm_preds + tcn_preds + cnn_preds) / 3.0
        equal_metrics = evaluate_metrics(y_test_unscaled, equal_preds)
        test_predictions["Equal_Ensemble"].append(equal_preds)
        print(f"\n--- [Seed {seed}] Static Equal Ensemble ---")
        print(f"  Equal Ensemble Test MAE: {equal_metrics['mae']:.4f} MW")

        seed_results.append({
            "model": "Static Equal Ensemble",
            "seed": seed,
            "best_epoch": "N/A",
            "actual_epochs": "N/A",
            "val_mse": (lstm_train_info["best_val_loss"] + tcn_train_info["best_val_loss"] + cnn_train_info["best_val_loss"]) / 3.0,
            "test_mae": equal_metrics["mae"],
            "test_rmse": equal_metrics["rmse"],
            "test_mse": equal_metrics["mse"],
            "test_r2": equal_metrics["r2"],
            "test_mape": equal_metrics["mape"],
            "train_time_s": lstm_train_info["train_time_seconds"] + tcn_train_info["train_time_seconds"] + cnn_train_info["train_time_seconds"],
            "params": param_counts["Static Equal Ensemble"]
        })

        # 5. Original CAEG-Net V1
        print(f"\n--- [Seed {seed}] Training Original CAEG-Net V1 ---")
        set_seed(seed)
        caeg_model = OriginalCAEGNetPhase5(
            input_dim=1,
            horizon=24,
            context_dim=4,
            latent_context_dim=16,
            lstm_hidden=64,
            lstm_layers=2,
            tcn_channels=32,
            dropout=0.1
        )
        caeg_train_info = train_model(caeg_model, train_loader, val_loader, max_epochs=25, patience=6, device=device)
        caeg_pred_dict = predict_model(caeg_model, test_loader, scaler, device)
        caeg_preds = caeg_pred_dict["preds"]
        caeg_weights = caeg_pred_dict["weights"]
        caeg_metrics = evaluate_metrics(y_test_unscaled, caeg_preds)
        test_predictions["CAEG_Net_V1"].append(caeg_preds)
        
        mean_w_lstm = float(np.mean(caeg_weights[:, 0]))
        mean_w_tcn = float(np.mean(caeg_weights[:, 1]))
        mean_w_cnn = float(np.mean(caeg_weights[:, 2]))
        
        eps = 1e-8
        entropy = -np.sum(caeg_weights * np.log(caeg_weights + eps), axis=-1)
        mean_entropy = float(np.mean(entropy))
        n_eff = float(np.mean(np.exp(entropy)))

        print(f"  CAEG-Net V1 Test MAE: {caeg_metrics['mae']:.4f} MW | Best Epoch: {caeg_train_info['best_epoch']} | Time: {caeg_train_info['train_time_seconds']:.1f}s")
        print(f"  Routing Weights: LSTM={mean_w_lstm:.3f}, TCN={mean_w_tcn:.3f}, CNN={mean_w_cnn:.3f} | N_eff={n_eff:.2f}")

        seed_results.append({
            "model": "Original CAEG-Net V1",
            "seed": seed,
            "best_epoch": caeg_train_info["best_epoch"],
            "actual_epochs": caeg_train_info["actual_epochs"],
            "val_mse": caeg_train_info["best_val_loss"],
            "test_mae": caeg_metrics["mae"],
            "test_rmse": caeg_metrics["rmse"],
            "test_mse": caeg_metrics["mse"],
            "test_r2": caeg_metrics["r2"],
            "test_mape": caeg_metrics["mape"],
            "train_time_s": caeg_train_info["train_time_seconds"],
            "params": param_counts["Original CAEG-Net V1"]
        })

        caeg_routing_records.append({
            "seed": seed,
            "mean_w_lstm": mean_w_lstm,
            "mean_w_tcn": mean_w_tcn,
            "mean_w_cnn": mean_w_cnn,
            "min_w_lstm": float(np.min(caeg_weights[:, 0])),
            "max_w_lstm": float(np.max(caeg_weights[:, 0])),
            "min_w_tcn": float(np.min(caeg_weights[:, 1])),
            "max_w_tcn": float(np.max(caeg_weights[:, 1])),
            "min_w_cnn": float(np.min(caeg_weights[:, 2])),
            "max_w_cnn": float(np.max(caeg_weights[:, 2])),
            "mean_entropy": mean_entropy,
            "effective_n_experts": n_eff
        })

    df_seed_results = pd.DataFrame(seed_results)
    df_seed_results.to_csv(os.path.join(RESULTS_DIR, "phase9_seed_results.csv"), index=False)
    print("\nSaved phase9_seed_results.csv")

    df_routing = pd.DataFrame(caeg_routing_records)
    df_routing.to_csv(os.path.join(RESULTS_DIR, "phase9_routing_diagnostics.csv"), index=False)
    print("Saved phase9_routing_diagnostics.csv")

    models_list = ["Original CAEG-Net V1", "Static Equal Ensemble", "Standalone TCN", "Standalone LSTM", "Standalone CNN"]
    summary_rows = []
    
    for m_name in models_list:
        sub = df_seed_results[df_seed_results["model"] == m_name]
        mae_mean, mae_std = sub["test_mae"].mean(), sub["test_mae"].std()
        rmse_mean, rmse_std = sub["test_rmse"].mean(), sub["test_rmse"].std()
        mse_mean, mse_std = sub["test_mse"].mean(), sub["test_mse"].std()
        r2_mean, r2_std = sub["test_r2"].mean(), sub["test_r2"].std()
        mape_mean, mape_std = sub["test_mape"].mean(), sub["test_mape"].std()
        p_count = sub["params"].iloc[0]

        summary_rows.append({
            "model": m_name,
            "test_mae_mean": mae_mean,
            "test_mae_std": mae_std,
            "test_rmse_mean": rmse_mean,
            "test_rmse_std": rmse_std,
            "test_mse_mean": mse_mean,
            "test_mse_std": mse_std,
            "test_r2_mean": r2_mean,
            "test_r2_std": r2_std,
            "test_mape_mean": mape_mean,
            "test_mape_std": mape_std,
            "parameters": p_count
        })

    df_comparison = pd.DataFrame(summary_rows)
    df_comparison = df_comparison.sort_values("test_mae_mean").reset_index(drop=True)
    df_comparison.to_csv(os.path.join(RESULTS_DIR, "phase9_model_comparison.csv"), index=False)
    print("Saved phase9_model_comparison.csv")

    print("\n" + "=" * 80)
    print("PHASE 9B MODEL COMPARISON TABLE (MW)")
    print("=" * 80)
    print(f"{'Model':<25} | {'MAE (MW)':<16} | {'RMSE (MW)':<16} | {'R²':<12} | {'MAPE (%)':<12} | {'Params'}")
    print("-" * 95)
    for _, r in df_comparison.iterrows():
        print(f"{r['model']:<25} | {r['test_mae_mean']:6.2f} ± {r['test_mae_std']:4.2f} | {r['test_rmse_mean']:6.2f} ± {r['test_rmse_std']:4.2f} | {r['test_r2_mean']:6.4f} | {r['test_mape_mean']:5.2f}% | {int(r['parameters']):,}")

    # Statistical Evaluation: Daily-Block Paired Tests
    print("\n" + "=" * 80)
    print("STATISTICAL EVALUATION: NON-OVERLAPPING DAILY-BLOCK ANALYSIS")
    print("=" * 80)

    ens_preds = {
        "LSTM": np.mean(test_predictions["LSTM"], axis=0),
        "TCN": np.mean(test_predictions["TCN"], axis=0),
        "CNN": np.mean(test_predictions["CNN"], axis=0),
        "Equal_Ensemble": np.mean(test_predictions["Equal_Ensemble"], axis=0),
        "CAEG_Net_V1": np.mean(test_predictions["CAEG_Net_V1"], axis=0)
    }

    step_daily = 24
    daily_indices = np.arange(0, len(y_test_unscaled), step_daily)
    K_blocks = len(daily_indices)
    print(f"Total non-overlapping 24h daily blocks: K = {K_blocks}")

    daily_maes = {}
    for m_key, preds in ens_preds.items():
        daily_maes[m_key] = np.array([
            np.mean(np.abs(y_test_unscaled[idx] - preds[idx])) for idx in daily_indices
        ])

    caeg_daily = daily_maes["CAEG_Net_V1"]
    stat_rows = []

    comparisons = [
        ("CAEG_Net_V1", "Equal_Ensemble", "Static Equal Ensemble"),
        ("CAEG_Net_V1", "TCN", "Standalone TCN"),
        ("CAEG_Net_V1", "LSTM", "Standalone LSTM"),
        ("CAEG_Net_V1", "CNN", "Standalone CNN")
    ]

    raw_p_values = []
    diff_data = []

    for m_caeg, m_base, display_name in comparisons:
        base_daily = daily_maes[m_base]
        diff = caeg_daily - base_daily
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1))
        se = diff_std / np.sqrt(K_blocks)
        
        ci_low = diff_mean - stats.t.ppf(0.975, df=K_blocks - 1) * se
        ci_high = diff_mean + stats.t.ppf(0.975, df=K_blocks - 1) * se
        
        t_stat, p_t = stats.ttest_1samp(diff, 0.0)
        try:
            w_stat, p_w = stats.wilcoxon(diff, alternative='two-sided')
        except Exception:
            w_stat, p_w = float('nan'), float('nan')
            
        raw_p_values.append(p_t)
        diff_data.append({
            "comparison": f"CAEG-Net V1 vs {display_name}",
            "baseline": display_name,
            "mean_diff_mw": diff_mean,
            "std_diff_mw": diff_std,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "t_stat": float(t_stat),
            "p_value_t": float(p_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_wilcoxon": float(p_w),
            "k_blocks": K_blocks
        })

    order = np.argsort(raw_p_values)
    m = len(raw_p_values)
    corrected_p_values = [0.0] * m
    for rank, idx in enumerate(order):
        factor = m - rank
        corrected_p = min(1.0, raw_p_values[idx] * factor)
        corrected_p_values[idx] = corrected_p

    for i, d in enumerate(diff_data):
        d["p_value_holm_bonferroni"] = corrected_p_values[i]
        d["statistically_significant"] = "YES" if corrected_p_values[i] < 0.05 else "NO"
        stat_rows.append(d)

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(os.path.join(RESULTS_DIR, "phase9_statistical_tests.csv"), index=False)
    print("Saved phase9_statistical_tests.csv")

    for r in stat_rows:
        print(f"\n{r['comparison']}:")
        print(f"  Mean Paired Diff: {r['mean_diff_mw']:+.4f} MW (95% CI: [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}] MW)")
        print(f"  t-statistic: {r['t_stat']:+.4f}, p (raw) = {r['p_value_t']:.6f}, p (Holm-Bonferroni) = {r['p_value_holm_bonferroni']:.6f}")
        print(f"  Wilcoxon stat: {r['wilcoxon_stat']}, p = {r['p_value_wilcoxon']:.6f}")
        print(f"  Significant at alpha=0.05: {r['statistically_significant']}")

    # Publication Plots
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 80)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot 1: Model MAE Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
    models_display = df_comparison["model"].tolist()
    maes = df_comparison["test_mae_mean"].tolist()
    stds = df_comparison["test_mae_std"].tolist()

    bars = ax.bar(models_display, maes, yerr=stds, capsize=5, color=bar_colors[:len(models_display)], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Test MAE (MW)", fontsize=12, fontweight="bold")
    ax.set_title("UCI ElectricityLoadDiagrams (Cohort 320): Model MAE Comparison (5 Seeds)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    for bar, m_val, s_val in zip(bars, maes, stds):
        ax.text(bar.get_x() + bar.get_width()/2.0, m_val + s_val + 0.3, f"{m_val:.2f}\n±{s_val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "phase9_01_model_mae_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved {p1}")

    # Plot 2: Per-Seed MAE
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_seeds = df_seed_results.pivot(index="seed", columns="model", values="test_mae")[models_list]
    pivot_seeds.plot(kind="bar", ax=ax, colormap="tab10", width=0.8, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Test MAE (MW)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Random Seed", fontsize=12, fontweight="bold")
    ax.set_title("UCI Benchmark: Per-Seed Model MAE (MW)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "phase9_02_per_seed_mae.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved {p2}")

    # Plot 3: CAEG Routing Weights
    fig, ax = plt.subplots(figsize=(12, 4.5))
    caeg_w_sample = predict_model(caeg_model, test_loader, scaler, device)["weights"]
    time_steps = np.arange(len(caeg_w_sample))
    ax.plot(time_steps[:720], caeg_w_sample[:720, 0], label="LSTM Weight", color="#2b5c8f", alpha=0.8, linewidth=1.5)
    ax.plot(time_steps[:720], caeg_w_sample[:720, 1], label="TCN Weight", color="#d95f02", alpha=0.8, linewidth=1.5)
    ax.plot(time_steps[:720], caeg_w_sample[:720, 2], label="CNN Weight", color="#1b9e77", alpha=0.8, linewidth=1.5)
    ax.axhline(1.0/3.0, color="gray", linestyle="--", alpha=0.7, label="Equal Prior (1/3)")
    ax.set_ylabel("Routing Weight", fontsize=12, fontweight="bold")
    ax.set_xlabel("Test Horizon Hours (First 30 Days)", fontsize=12, fontweight="bold")
    ax.set_title("Original CAEG-Net V1: Dynamic Routing Weights (First 720 Hours of Test Partition)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "phase9_03_caeg_routing_weights.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved {p3}")

    # Plot 4: Forecast vs Actual
    fig, ax = plt.subplots(figsize=(12, 5))
    seg_start = 1000
    seg_len = 336
    t_range = np.arange(seg_len)
    
    actual_seg = [y_test_unscaled[seg_start + i, 0] for i in range(seg_len)]
    caeg_seg = [ens_preds["CAEG_Net_V1"][seg_start + i, 0] for i in range(seg_len)]
    equal_seg = [ens_preds["Equal_Ensemble"][seg_start + i, 0] for i in range(seg_len)]
    tcn_seg = [ens_preds["TCN"][seg_start + i, 0] for i in range(seg_len)]

    ax.plot(t_range, actual_seg, label="Actual System Load (MW)", color="black", linewidth=2.0)
    ax.plot(t_range, caeg_seg, label="CAEG-Net V1 (Ensemble)", color="#2b5c8f", linestyle="-", linewidth=1.5)
    ax.plot(t_range, equal_seg, label="Static Equal Ensemble", color="#d95f02", linestyle="--", linewidth=1.5)
    ax.plot(t_range, tcn_seg, label="Standalone TCN", color="#7570b3", linestyle=":", linewidth=1.2)
    
    ax.set_ylabel("System Load (MW)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hours (14-Day Test Window)", fontsize=12, fontweight="bold")
    ax.set_title("UCI Benchmark: Representative 14-Day Forecast vs Actual Load (Cohort 320)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "phase9_04_forecast_vs_actual.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved {p4}")

    # Plot 5: Residual Error Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    err_caeg = (ens_preds["CAEG_Net_V1"] - y_test_unscaled).flatten()
    err_equal = (ens_preds["Equal_Ensemble"] - y_test_unscaled).flatten()

    ax.hist(err_caeg, bins=80, density=True, alpha=0.55, color="#2b5c8f", label="CAEG-Net V1 Residuals")
    ax.hist(err_equal, bins=80, density=True, alpha=0.55, color="#d95f02", label="Equal Ensemble Residuals")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Forecast Residual (y_pred - y_true) [MW]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.set_title("UCI Benchmark: Forecast Residual Error Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    p5 = os.path.join(PLOTS_DIR, "phase9_05_error_distribution.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved {p5}")

    # Plot 6: Daily-Block MAE Scatter & Cumulative Diff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1.scatter(daily_maes["Equal_Ensemble"], daily_maes["CAEG_Net_V1"], color="#2b5c8f", alpha=0.7, edgecolors='none', s=35)
    min_lim = min(daily_maes["Equal_Ensemble"].min(), daily_maes["CAEG_Net_V1"].min()) - 1.0
    max_lim = max(daily_maes["Equal_Ensemble"].max(), daily_maes["CAEG_Net_V1"].max()) + 1.0
    ax1.plot([min_lim, max_lim], [min_lim, max_lim], color="red", linestyle="--", label="Line of Parity (y = x)")
    ax1.set_xlabel("Static Equal Ensemble Daily MAE (MW)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("CAEG-Net V1 Daily MAE (MW)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Daily Block MAE Scatter (K={K_blocks})", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    cum_diff = np.cumsum(daily_maes["CAEG_Net_V1"] - daily_maes["Equal_Ensemble"])
    ax2.plot(np.arange(1, K_blocks + 1), cum_diff, color="#2b5c8f", linewidth=2.0)
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Consecutive Calendar Days (Test Set)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative MAE Difference (MW)", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Daily MAE Difference (CAEG - Equal)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    p6 = os.path.join(PLOTS_DIR, "phase9_06_daily_block_mae_comparison.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Saved {p6}")

    print("\n" + "=" * 80)
    print("PHASE 9B BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
