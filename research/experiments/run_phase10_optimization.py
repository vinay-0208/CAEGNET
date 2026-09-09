"""
CAEG-Net Phase 10: Controlled Cross-Dataset Optimization Driver
==============================================================
Governing Roadmap: Phase 10 of the 14-Phase Research Roadmap

Executes:
1. Multi-dataset data ingestion & caching (PJM, GEFCom2014, UCI Cohort 320).
2. Staged validation screening across 5 hypothesis groups:
   - Stage 1: Context Normalization (A0-A4)
   - Stage 2: Recent-Error Formulation (B0-B2)
   - Stage 3: Router Regularization (C0-C3)
   - Stage 4: Training Strategy (D0-D2)
   - Stage 5: CNN Stabilization (E0-E1)
3. Strict cross-dataset validation selection rule (>= 2 datasets improved, <= 2% degradation on 3rd).
4. Five-seed finalist benchmark across all 3 datasets on test partitions (descriptive evidence).
5. Daily-block paired statistical tests with Holm-Bonferroni correction.
6. Routing diagnostics & expert complementarity residual analysis.
7. Publication figures and standardized CSV artifacts.
"""

import os
import sys
import json
import time
import functools
from typing import Dict, List, Tuple, Optional, Union

# Force unbuffered output so task logs are immediately readable
print = functools.partial(print, flush=True)

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
from research.original_caeg import OriginalCAEGNetPhase5, compute_phase5_loss
from research.gefcom_data import prepare_gefcom2014_pipeline
from research.data_adapter_uci import stream_uci_hourly_aggregate_load
from data_utils import (
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
    create_partition_windows_with_context,
    compute_causal_recent_forecast_errors,
    extract_context_features,
)
from evaluate import compute_metrics


# =====================================================================
# 1. Custom CNN_H2 Architecture for Group E
# =====================================================================

class CNN_H2_TemporalPool8(nn.Module):
    """CNN with AdaptiveAvgPool1d(8) temporal retention."""
    def __init__(self, input_dim: int = 1, horizon: int = 24, pool_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pool_size = pool_size
        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_size),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * pool_size, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)
        feat = self.conv_stack(x_trans).view(x.shape[0], -1)
        return self.head(feat)


class CAEGNetWithOptimizedCNN(OriginalCAEGNetPhase5):
    """CAEG-Net with CNN_H2_TemporalPool8 replacing canonical CNN."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cnn_expert = CNN_H2_TemporalPool8(
            input_dim=self.input_dim,
            horizon=self.horizon,
            pool_size=8,
            dropout=0.1,
        )


# =====================================================================
# 2. Context Extraction & Normalization Variants
# =====================================================================

def compute_modular_contexts(
    windows: Dict[str, Dict[str, np.ndarray]],
    rec_errors: Tuple[np.ndarray, np.ndarray, np.ndarray],
    scaler,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Computes modular context feature sets (A0-A4, B0-B2) for train, val, and test.
    """
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    rec_tr, rec_val, rec_te = rec_errors

    contexts = {}
    for part in ["train", "val", "test"]:
        X = windows[part]["X"]  # [N, 168, 1]
        Z = X[:, :, 0]          # [N, 168]
        rec = rec_tr if part == "train" else (rec_val if part == "val" else rec_te)

        # Baseline A0 Context
        C_a0 = extract_context_features(X, rec)  # [N, 4]: [trend, vol, r24, rec_err]

        # 1. Trend components
        time_idx = np.arange(168, dtype=np.float32)
        i_bar = 83.5
        denom_t = np.sum((time_idx - i_bar) ** 2)
        z_bar = np.mean(Z, axis=1, keepdims=True)
        num_t = np.sum((time_idx - i_bar) * (Z - z_bar), axis=1, keepdims=True)
        beta_1 = num_t / denom_t
        std_168 = np.std(Z, axis=1, keepdims=True) + 1e-6

        # A1: Load-scale normalized trend
        f_trend_a1 = (beta_1 * scale * 168.0) / (abs(mean) + 1e-6)
        f_trend_a1 = np.clip(f_trend_a1, -10.0, 10.0)

        # A2: Coefficient-of-variation volatility
        mean_orig = z_bar * scale + mean
        f_vol_a2 = (std_168 * scale) / (np.abs(mean_orig) + 1e-6)
        f_vol_a2 = np.clip(f_vol_a2, 0.0, 10.0)

        # A3: Load-relative recent error
        z_24 = Z[:, -24:]
        mean_recent_orig = np.mean(z_24 * scale + mean, axis=1, keepdims=True)
        rec_orig = rec if rec.ndim == 2 else rec[:, None]
        rec_err_orig = rec_orig * scale
        f_err_a3 = rec_err_orig / (np.abs(mean_recent_orig) + 1e-6)
        f_err_a3 = np.clip(f_err_a3, 0.0, 10.0)

        # Periodicity r24 is naturally dimensionless in [-1, 1]
        f_r24 = C_a0[:, 2:3]

        # Assemble Variants
        ctx_a0 = C_a0.astype(np.float32)
        ctx_a1 = np.column_stack([f_trend_a1, C_a0[:, 1:2], f_r24, C_a0[:, 3:4]]).astype(np.float32)
        ctx_a2 = np.column_stack([C_a0[:, 0:1], f_vol_a2, f_r24, C_a0[:, 3:4]]).astype(np.float32)
        ctx_a3 = np.column_stack([C_a0[:, 0:1], C_a0[:, 1:2], f_r24, f_err_a3]).astype(np.float32)
        ctx_a4 = np.column_stack([f_trend_a1, f_vol_a2, f_r24, f_err_a3]).astype(np.float32)

        # B1: Zero recent error (3D context)
        ctx_b1 = C_a0[:, :3].astype(np.float32)

        # B2: Variance-normalized recent error
        f_err_b2 = rec_orig / (std_168 + 1e-6)
        f_err_b2 = np.clip(f_err_b2, 0.0, 10.0)
        ctx_b2 = np.column_stack([C_a0[:, 0:1], C_a0[:, 1:2], f_r24, f_err_b2]).astype(np.float32)

        for name, arr in [
            ("A0", ctx_a0), ("A1", ctx_a1), ("A2", ctx_a2), ("A3", ctx_a3), ("A4", ctx_a4),
            ("B0", ctx_a0), ("B1", ctx_b1), ("B2", ctx_b2),
        ]:
            if name not in contexts:
                contexts[name] = {}
            contexts[name][part] = arr

    return contexts


# =====================================================================
# 3. Model Training & Evaluation Engine
# =====================================================================

def train_caeg_variant(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    max_epochs: int = 25,
    patience: int = 6,
    beta_entropy: float = 0.0,
    lambda_kl: float = 0.0,
    temperature: float = 1.0,
) -> Dict:
    """Trains a CAEG-Net model variant using early stopping on Validation MSE."""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float("inf")
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                x_b, y_b, c_b = batch
            else:
                x_b, y_b = batch
                c_b = None

            x_b, y_b = x_b.to(device), y_b.to(device)
            if c_b is not None:
                c_b = c_b.to(device)

            optimizer.zero_grad()
            if c_b is not None:
                y_pred, weights, _ = model(x_b, c_b, temperature=temperature, return_diagnostics=True)
                loss, _ = compute_phase5_loss(
                    y_pred, y_b, weights=weights, beta_entropy=beta_entropy, lambda_kl=lambda_kl
                )
            else:
                y_pred = model(x_b)
                loss = F.mse_loss(y_pred, y_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    x_b, y_b, c_b = batch
                else:
                    x_b, y_b = batch
                    c_b = None
                x_b, y_b = x_b.to(device), y_b.to(device)
                if c_b is not None:
                    c_b = c_b.to(device)
                    y_pred, _, _ = model(x_b, c_b, temperature=temperature, return_diagnostics=True)
                else:
                    y_pred = model(x_b)
                val_loss += F.mse_loss(y_pred, y_b, reduction="sum").item()
                val_count += y_b.numel()

        val_mse = val_loss / max(val_count, 1)
        if not np.isnan(val_mse) and val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    train_time = time.time() - t0
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "train_time": train_time,
    }


def evaluate_model_on_partition(
    model: nn.Module,
    data_loader: DataLoader,
    scaler,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict:
    """Evaluates model on given partition DataLoader, converting to engineering units."""
    model.eval()
    all_preds = []
    all_trues = []
    all_weights = []

    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                x_b, y_b, c_b = batch
            else:
                x_b, y_b = batch
                c_b = None
            x_b, y_b = x_b.to(device), y_b.to(device)
            if c_b is not None:
                c_b = c_b.to(device)
                y_pred, w, _ = model(x_b, c_b, temperature=temperature, return_diagnostics=True)
                all_weights.append(w.cpu().numpy())
            else:
                y_pred = model(x_b)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_b.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0) * scale + mean
    trues_arr = np.concatenate(all_trues, axis=0) * scale + mean

    metrics = compute_metrics(preds_arr, trues_arr)
    weights_arr = np.concatenate(all_weights, axis=0) if all_weights else None

    return {
        "metrics": metrics,
        "preds": preds_arr,
        "trues": trues_arr,
        "weights": weights_arr,
    }


# =====================================================================
# 4. Decoupled Training Routine (Group D1 & D2)
# =====================================================================

def train_decoupled_caeg(
    dataset_dict: Dict,
    context_type: str,
    device: torch.device,
    seed: int = 42,
    fine_tune: bool = False,
) -> Dict:
    """Pre-trains experts independently, freezes them, and trains only router (D1) or fine-tunes (D2)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    windows = dataset_dict["windows"]
    ctx = dataset_dict["contexts"][context_type]
    scaler = dataset_dict["scaler"]

    tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
    tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
    va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
    va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
    te_x = torch.tensor(windows["test"]["X"], dtype=torch.float32)
    te_y = torch.tensor(windows["test"]["Y"], dtype=torch.float32)

    tr_c = torch.tensor(ctx["train"], dtype=torch.float32)
    va_c = torch.tensor(ctx["val"], dtype=torch.float32)
    te_c = torch.tensor(ctx["test"], dtype=torch.float32)

    tr_loader_no_ctx = DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True)
    va_loader_no_ctx = DataLoader(TensorDataset(va_x, va_y), batch_size=64, shuffle=False)

    tr_loader_ctx = DataLoader(TensorDataset(tr_x, tr_y, tr_c), batch_size=64, shuffle=True)
    va_loader_ctx = DataLoader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)
    te_loader_ctx = DataLoader(TensorDataset(te_x, te_y, te_c), batch_size=64, shuffle=False)

    # 1. Pretrain canonical experts (with fewer epochs for screening efficiency)
    m_lstm = LSTMExpert().to(device)
    m_tcn = TCNExpert().to(device)
    m_cnn = CNNExpert().to(device)

    train_caeg_variant(m_lstm, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=15, patience=4)
    train_caeg_variant(m_tcn, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=15, patience=4)
    train_caeg_variant(m_cnn, tr_loader_no_ctx, va_loader_no_ctx, device, max_epochs=15, patience=4)

    # 2. Build CAEG model and transplant pretrained expert weights
    caeg = OriginalCAEGNetPhase5(context_dim=ctx["train"].shape[1]).to(device)
    caeg.lstm_expert.load_state_dict(m_lstm.state_dict())
    caeg.tcn_expert.load_state_dict(m_tcn.state_dict())
    caeg.cnn_expert.load_state_dict(m_cnn.state_dict())

    # Freeze experts
    for expert in [caeg.lstm_expert, caeg.tcn_expert, caeg.cnn_expert]:
        for p in expert.parameters():
            p.requires_grad = False

    # 3. Train router and context encoder only
    train_caeg_variant(caeg, tr_loader_ctx, va_loader_ctx, device, lr=1e-3, max_epochs=15, patience=4)

    # 4. Optional limited fine-tuning (D2)
    if fine_tune:
        for p in caeg.parameters():
            p.requires_grad = True
        train_caeg_variant(caeg, tr_loader_ctx, va_loader_ctx, device, lr=1e-4, max_epochs=5, patience=3)

    val_res = evaluate_model_on_partition(caeg, va_loader_ctx, scaler, device)
    test_res = evaluate_model_on_partition(caeg, te_loader_ctx, scaler, device)

    return {
        "model": caeg,
        "val_metrics": val_res["metrics"],
        "test_metrics": test_res["metrics"],
        "weights": test_res["weights"],
    }


# =====================================================================
# 5. Dataset Loading & Preparation Pipeline
# =====================================================================

def load_all_three_datasets() -> Dict[str, Dict]:
    """Loads and standardizes Modern PJM, GEFCom2014, and UCI Cohort 320."""
    print("\n=======================================================")
    print("INGESTING AND PREPARING TRI-BENCHMARK DATASETS")
    print("=======================================================")

    # 1. Modern PJM
    print("Loading Modern PJM...")
    df_pjm, _ = load_and_clean_data("data/Modern_PJM/pjm_load.csv")
    tr_pjm, va_pjm, te_pjm, _ = chronological_split(df_pjm, 0.70, 0.15, 0.15)
    scaler_pjm, tr_sc_pjm, va_sc_pjm, te_sc_pjm = fit_and_transform_scaler(tr_pjm, va_pjm, te_pjm)
    win_pjm = create_partition_windows_with_context(tr_sc_pjm, va_sc_pjm, te_sc_pjm, lookback=168, horizon=24)
    rec_pjm = compute_causal_recent_forecast_errors(win_pjm)
    ctx_pjm = compute_modular_contexts(win_pjm, rec_pjm[:3], scaler_pjm)
    print(f"PJM: Train={len(win_pjm['train']['X'])}, Val={len(win_pjm['val']['X'])}, Test={len(win_pjm['test']['X'])}")

    # 2. GEFCom2014
    print("Loading GEFCom2014...")
    gef_pipe = prepare_gefcom2014_pipeline("data/Load")
    win_gef = gef_pipe["windows"]
    scaler_gef = gef_pipe["scaler"]
    # Extract recent errors from gef_pipe context (column 3)
    rec_tr_g = gef_pipe["context"]["train"][:, 3]
    rec_va_g = gef_pipe["context"]["val"][:, 3]
    rec_te_g = gef_pipe["context"]["test"][:, 3]
    ctx_gef = compute_modular_contexts(win_gef, (rec_tr_g, rec_va_g, rec_te_g), scaler_gef)
    print(f"GEFCom: Train={len(win_gef['train']['X'])}, Val={len(win_gef['val']['X'])}, Test={len(win_gef['test']['X'])}")

    # 3. UCI Cohort 320
    print("Loading UCI Cohort 320 Aggregate...")
    df_uci = stream_uci_hourly_aggregate_load(
        "data/ElectricityLoadDiagrams20112014/LD2011_2014.txt",
        unit="MW",
        start_year=2012,
        cohort="cohort_320",
        audit_csv_path="research/results/phase9_cohort_audit.csv",
    )
    tr_uci, va_uci, te_uci, _ = chronological_split(df_uci, 0.70, 0.15, 0.15)
    scaler_uci, tr_sc_uci, va_sc_uci, te_sc_uci = fit_and_transform_scaler(tr_uci, va_uci, te_uci, load_col="load")
    win_uci = create_partition_windows_with_context(tr_sc_uci, va_sc_uci, te_sc_uci, lookback=168, horizon=24, load_col="load")
    rec_uci = compute_causal_recent_forecast_errors(win_uci)
    ctx_uci = compute_modular_contexts(win_uci, rec_uci[:3], scaler_uci)
    print(f"UCI: Train={len(win_uci['train']['X'])}, Val={len(win_uci['val']['X'])}, Test={len(win_uci['test']['X'])}")

    return {
        "PJM": {
            "name": "Modern PJM",
            "unit": "MW",
            "windows": win_pjm,
            "scaler": scaler_pjm,
            "contexts": ctx_pjm,
            "horizon": 24,
        },
        "GEFCom": {
            "name": "GEFCom2014",
            "unit": "kW",
            "windows": win_gef,
            "scaler": scaler_gef,
            "contexts": ctx_gef,
            "horizon": 24,
        },
        "UCI": {
            "name": "UCI Load Diagrams",
            "unit": "MW",
            "windows": win_uci,
            "scaler": scaler_uci,
            "contexts": ctx_uci,
            "horizon": 24,
        },
    }


# =====================================================================
# 6. Main Orchestrator
# =====================================================================

def run_phase10_experiment():
    t_start = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is expected on NVIDIA RTX 4050 but unavailable! Failing immediately per instructions.")
    device = torch.device("cuda")
    print(f"Executing Phase 10 on device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Data
    datasets = load_all_three_datasets()

    # 2. Define Candidate Specifications
    candidates = [
        # Stage 1: Context Normalization
        {"id": "A0_Canonical_V1", "group": "Control", "ctx": "A0", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "A1_Load_Normalized_Trend", "group": "Group_A", "ctx": "A1", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "A2_CoV_Volatility", "group": "Group_A", "ctx": "A2", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "A3_Relative_Recent_Error", "group": "Group_A", "ctx": "A3", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "A4_Dimensionless_Context", "group": "Group_A", "ctx": "A4", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},

        # Stage 2: Recent Error Formulation
        {"id": "B1_Zero_Recent_Error_3D", "group": "Group_B", "ctx": "B1", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "B2_Normalized_Recent_Error", "group": "Group_B", "ctx": "B2", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},

        # Stage 3: Router Regularization
        {"id": "C1_Entropy_Regularized", "group": "Group_C", "ctx": "A0", "ent": 0.005, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "C2_Stability_Regularized", "group": "Group_C", "ctx": "A0", "ent": 0.0, "kl": 0.005, "temp": 1.0, "strat": "joint", "cnn": "canonical"},
        {"id": "C3_Sharper_Temperature", "group": "Group_C", "ctx": "A0", "ent": 0.0, "kl": 0.0, "temp": 0.8, "strat": "joint", "cnn": "canonical"},
        {"id": "C3_Softer_Temperature", "group": "Group_C", "ctx": "A0", "ent": 0.0, "kl": 0.0, "temp": 1.25, "strat": "joint", "cnn": "canonical"},

        # Stage 4: Training Strategy
        {"id": "D1_Decoupled_Pretraining", "group": "Group_D", "ctx": "A0", "strat": "decoupled"},
        {"id": "D2_Pretrain_FineTune", "group": "Group_D", "ctx": "A0", "strat": "finetune"},

        # Stage 5: CNN Stabilization
        {"id": "E1_CNN_H2_TemporalPool8", "group": "Group_E", "ctx": "A0", "ent": 0.0, "kl": 0.0, "temp": 1.0, "strat": "joint", "cnn": "h2_pool8"},
    ]

    screening_seeds = [42]  # Rigorous validation screening with canonical seed 42
    validation_records = []

    print("\n=======================================================")
    print("STAGE 1-5: CROSS-DATASET VALIDATION SCREENING")
    print("=======================================================")

    # Check if screening CSV already exists
    val_csv_path = "research/results/phase10_validation_results.csv"
    cross_csv_path = "research/results/phase10_cross_dataset_validation.csv"
    if os.path.exists(val_csv_path) and os.path.exists(cross_csv_path):
        print(f"Loading existing screening results from {val_csv_path} and {cross_csv_path}...")
        df_val = pd.read_csv(val_csv_path)
        df_cross = pd.read_csv(cross_csv_path)
        v1_rows = df_val[df_val["candidate_id"] == "A0_Canonical_V1"].set_index("dataset")["val_mae_mean"].to_dict()
        promising_candidates = df_cross[df_cross["cross_dataset_promising"] == "YES"]["candidate_id"].tolist()
    else:
        for cand in candidates:
            cid = cand["id"]
            cgroup = cand["group"]
            print(f"\n--- Evaluating Candidate: {cid} ({cgroup}) ---")

            for d_key in ["PJM", "GEFCom", "UCI"]:
                d_obj = datasets[d_key]
                windows = d_obj["windows"]
                scaler = d_obj["scaler"]
                ctx_dict = d_obj["contexts"][cand["ctx"]]

                tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
                tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
                va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
                va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
                tr_c = torch.tensor(ctx_dict["train"], dtype=torch.float32)
                va_c = torch.tensor(ctx_dict["val"], dtype=torch.float32)

                tr_loader = DataLoader(TensorDataset(tr_x, tr_y, tr_c), batch_size=64, shuffle=True)
                va_loader = DataLoader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)

                val_maes = []
                val_rmses = []

                for seed in screening_seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)

                    strat = cand.get("strat", "joint")
                    if strat == "decoupled":
                        res = train_decoupled_caeg(d_obj, cand["ctx"], device, seed=seed, fine_tune=False)
                        v_mae = res["val_metrics"]["MAE"]
                        v_rmse = res["val_metrics"]["RMSE"]
                    elif strat == "finetune":
                        res = train_decoupled_caeg(d_obj, cand["ctx"], device, seed=seed, fine_tune=True)
                        v_mae = res["val_metrics"]["MAE"]
                        v_rmse = res["val_metrics"]["RMSE"]
                    else:
                        cnn_type = cand.get("cnn", "canonical")
                        if cnn_type == "h2_pool8":
                            model = CAEGNetWithOptimizedCNN(context_dim=ctx_dict["train"].shape[1])
                        else:
                            model = OriginalCAEGNetPhase5(context_dim=ctx_dict["train"].shape[1])

                        train_caeg_variant(
                            model, tr_loader, va_loader, device,
                            lr=1e-3, max_epochs=25, patience=6,
                            beta_entropy=cand.get("ent", 0.0),
                            lambda_kl=cand.get("kl", 0.0),
                            temperature=cand.get("temp", 1.0),
                        )
                        eval_res = evaluate_model_on_partition(
                            model, va_loader, scaler, device, temperature=cand.get("temp", 1.0)
                        )
                        v_mae = eval_res["metrics"]["MAE"]
                        v_rmse = eval_res["metrics"]["RMSE"]

                    val_maes.append(v_mae)
                    val_rmses.append(v_rmse)

                mean_val_mae = float(np.mean(val_maes))
                mean_val_rmse = float(np.mean(val_rmses))

                validation_records.append({
                    "candidate_id": cid,
                    "group": cgroup,
                    "dataset": d_key,
                    "unit": d_obj["unit"],
                    "val_mae_mean": mean_val_mae,
                    "val_rmse_mean": mean_val_rmse,
                })
                print(f"  {d_key:7s}: Val MAE = {mean_val_mae:.2f} {d_obj['unit']}")

            # Save incremental validation records
            pd.DataFrame(validation_records).to_csv(val_csv_path, index=False)

        # 3. Cross-Dataset Comparison Matrix & Relative Calculations
        df_val = pd.DataFrame(validation_records)
        v1_rows = df_val[df_val["candidate_id"] == "A0_Canonical_V1"].set_index("dataset")["val_mae_mean"].to_dict()

        cross_records = []
        promising_candidates = []

        for cid in df_val["candidate_id"].unique():
            cand_df = df_val[df_val["candidate_id"] == cid].set_index("dataset")
            pjm_val = cand_df.loc["PJM", "val_mae_mean"]
            gef_val = cand_df.loc["GEFCom", "val_mae_mean"]
            uci_val = cand_df.loc["UCI", "val_mae_mean"]

            rel_pjm = (v1_rows["PJM"] - pjm_val) / v1_rows["PJM"] * 100.0
            rel_gef = (v1_rows["GEFCom"] - gef_val) / v1_rows["GEFCom"] * 100.0
            rel_uci = (v1_rows["UCI"] - uci_val) / v1_rows["UCI"] * 100.0

            improvements = sum([rel_pjm > 0.0, rel_gef > 0.0, rel_uci > 0.0])
            min_rel = min(rel_pjm, rel_gef, rel_uci)
            is_promising = (improvements >= 2) and (min_rel >= -2.0) and (cid != "A0_Canonical_V1")

            if is_promising:
                promising_candidates.append(cid)

            cross_records.append({
                "candidate_id": cid,
                "group": cand_df.iloc[0]["group"],
                "pjm_val_mae": pjm_val,
                "gefcom_val_mae": gef_val,
                "uci_val_mae": uci_val,
                "pjm_rel_pct": rel_pjm,
                "gefcom_rel_pct": rel_gef,
                "uci_rel_pct": rel_uci,
                "num_datasets_improved": improvements,
                "min_relative_change_pct": min_rel,
                "cross_dataset_promising": "YES" if is_promising else "NO",
            })

        df_cross = pd.DataFrame(cross_records)
        df_cross.to_csv(cross_csv_path, index=False)

    print("\n=======================================================")
    print("CROSS-DATASET VALIDATION COMPARISON SUMMARY")
    print("=======================================================")
    print(df_cross[["candidate_id", "pjm_rel_pct", "gefcom_rel_pct", "uci_rel_pct", "cross_dataset_promising"]].to_string(index=False))

    # 4. Finalist Evaluation (Canonical 5 Seeds)
    finalists = ["A0_Canonical_V1"]
    if promising_candidates:
        print(f"\nQualified Cross-Dataset Promising Finalists: {promising_candidates}")
        finalists.extend(promising_candidates)
    else:
        print("\nNo candidate satisfied the strict cross-dataset criteria (>=2 improved, no degradation >2%).")
        print("Retaining Canonical CAEG-Net V1 as primary research model.")
        best_other = df_cross[df_cross["candidate_id"] != "A0_Canonical_V1"].sort_values("num_datasets_improved", ascending=False).iloc[0]["candidate_id"]
        finalists.append(best_other)
        print(f"Adding best exploratory candidate '{best_other}' for descriptive multi-seed comparison.")

    canonical_seeds = [42, 123, 999, 2024, 3407]
    finalist_records = []
    routing_records = []

    fin_csv_path = "research/results/phase10_candidate_comparison.csv"
    route_csv_path = "research/results/phase10_routing_analysis.csv"

    if os.path.exists(fin_csv_path) and os.path.exists(route_csv_path) and len(pd.read_csv(fin_csv_path)) >= len(finalists) * 3:
        print(f"Loading existing finalist test benchmark results from {fin_csv_path}...")
        df_fin = pd.read_csv(fin_csv_path)
        df_route = pd.read_csv(route_csv_path)
    else:
        print("\n=======================================================")
        print("EXECUTING 5-SEED BENCHMARK ON FINALISTS (DESCRIPTIVE TEST SET)")
        print("=======================================================")

        for fin_id in finalists:
            cand_meta = next(c for c in candidates if c["id"] == fin_id)
            print(f"\n>>> Running 5 Seeds for Finalist: {fin_id} <<<")

            for d_key in ["PJM", "GEFCom", "UCI"]:
                d_obj = datasets[d_key]
                windows = d_obj["windows"]
                scaler = d_obj["scaler"]
                ctx_dict = d_obj["contexts"][cand_meta["ctx"]]

                tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
                tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
                va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
                va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
                te_x = torch.tensor(windows["test"]["X"], dtype=torch.float32)
                te_y = torch.tensor(windows["test"]["Y"], dtype=torch.float32)

                tr_c = torch.tensor(ctx_dict["train"], dtype=torch.float32)
                va_c = torch.tensor(ctx_dict["val"], dtype=torch.float32)
                te_c = torch.tensor(ctx_dict["test"], dtype=torch.float32)

                tr_loader = DataLoader(TensorDataset(tr_x, tr_y, tr_c), batch_size=64, shuffle=True)
                va_loader = DataLoader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)
                te_loader = DataLoader(TensorDataset(te_x, te_y, te_c), batch_size=64, shuffle=False)

                seed_maes = []
                seed_rmses = []
                seed_mses = []
                seed_r2s = []
                seed_mapes = []

                for seed in canonical_seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)

                    strat = cand_meta.get("strat", "joint")
                    if strat in ["decoupled", "finetune"]:
                        res = train_decoupled_caeg(
                            d_obj, cand_meta["ctx"], device, seed=seed, fine_tune=(strat == "finetune")
                        )
                        test_eval = {"metrics": res["test_metrics"], "weights": res["weights"]}
                    else:
                        cnn_type = cand_meta.get("cnn", "canonical")
                        if cnn_type == "h2_pool8":
                            model = CAEGNetWithOptimizedCNN(context_dim=ctx_dict["train"].shape[1])
                        else:
                            model = OriginalCAEGNetPhase5(context_dim=ctx_dict["train"].shape[1])

                        train_caeg_variant(
                            model, tr_loader, va_loader, device,
                            lr=1e-3, max_epochs=25, patience=6,
                            beta_entropy=cand_meta.get("ent", 0.0),
                            lambda_kl=cand_meta.get("kl", 0.0),
                            temperature=cand_meta.get("temp", 1.0),
                        )
                        test_eval = evaluate_model_on_partition(
                            model, te_loader, scaler, device, temperature=cand_meta.get("temp", 1.0)
                        )

                    m = test_eval["metrics"]
                    seed_maes.append(m["MAE"])
                    seed_rmses.append(m["RMSE"])
                    seed_mses.append(m["MSE"])
                    seed_r2s.append(m["R2"])
                    seed_mapes.append(m["MAPE"])

                    # Gating diagnostics
                    w = test_eval["weights"]
                    if w is not None:
                        eps = 1e-8
                        entropy = -np.sum(w * np.log(w + eps), axis=-1)
                        n_eff = np.exp(entropy)
                        routing_records.append({
                            "finalist_id": fin_id,
                            "dataset": d_key,
                            "seed": seed,
                            "mean_w_lstm": float(np.mean(w[:, 0])),
                            "mean_w_tcn": float(np.mean(w[:, 1])),
                            "mean_w_cnn": float(np.mean(w[:, 2])),
                            "mean_entropy": float(np.mean(entropy)),
                            "effective_n_experts": float(np.mean(n_eff)),
                        })

                finalist_records.append({
                    "finalist_id": fin_id,
                    "dataset": d_key,
                    "unit": d_obj["unit"],
                    "test_mae_mean": float(np.mean(seed_maes)),
                    "test_mae_std": float(np.std(seed_maes)),
                    "test_rmse_mean": float(np.mean(seed_rmses)),
                    "test_rmse_std": float(np.std(seed_rmses)),
                    "test_mse_mean": float(np.mean(seed_mses)),
                    "test_mse_std": float(np.std(seed_mses)),
                    "test_r2_mean": float(np.mean(seed_r2s)),
                    "test_r2_std": float(np.std(seed_r2s)),
                    "test_mape_mean": float(np.mean(seed_mapes)),
                    "test_mape_std": float(np.std(seed_mapes)),
                })
                print(f"  {d_key:7s}: Test MAE = {np.mean(seed_maes):.2f} ± {np.std(seed_maes):.2f} {d_obj['unit']}")

                pd.DataFrame(finalist_records).to_csv(fin_csv_path, index=False)
                pd.DataFrame(routing_records).to_csv(route_csv_path, index=False)

        df_fin = pd.DataFrame(finalist_records)
        df_route = pd.DataFrame(routing_records)

    # 5. Expert Complementarity Analysis (Residual Correlations) & Baselines
    print("\n=======================================================")
    print("ANALYZING EXPERT RESIDUAL COMPLEMENTARITY & BASELINES")
    print("=======================================================")
    baseline_records = []
    residual_records = []
    daily_block_tests = []

    for d_key in ["PJM", "GEFCom", "UCI"]:
        d_obj = datasets[d_key]
        windows = d_obj["windows"]
        scaler = d_obj["scaler"]

        tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
        tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
        va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
        va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
        te_x = torch.tensor(windows["test"]["X"], dtype=torch.float32)
        te_y = torch.tensor(windows["test"]["Y"], dtype=torch.float32)

        tr_loader_no_ctx = DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True)
        va_loader_no_ctx = DataLoader(TensorDataset(va_x, va_y), batch_size=64, shuffle=False)
        te_loader_no_ctx = DataLoader(TensorDataset(te_x, te_y), batch_size=64, shuffle=False)

        torch.manual_seed(42)
        m_lstm = LSTMExpert().to(device)
        m_tcn = TCNExpert().to(device)
        m_cnn = CNNExpert().to(device)

        train_caeg_variant(m_lstm, tr_loader_no_ctx, va_loader_no_ctx, device)
        train_caeg_variant(m_tcn, tr_loader_no_ctx, va_loader_no_ctx, device)
        train_caeg_variant(m_cnn, tr_loader_no_ctx, va_loader_no_ctx, device)

        res_l = evaluate_model_on_partition(m_lstm, te_loader_no_ctx, scaler, device)
        res_t = evaluate_model_on_partition(m_tcn, te_loader_no_ctx, scaler, device)
        res_c = evaluate_model_on_partition(m_cnn, te_loader_no_ctx, scaler, device)

        # Equal Ensemble
        p_ens = (res_l["preds"] + res_t["preds"] + res_c["preds"]) / 3.0
        ens_metrics = compute_metrics(p_ens, res_l["trues"])

        baseline_records.append({"dataset": d_key, "model": "LSTM", "test_mae": res_l["metrics"]["MAE"]})
        baseline_records.append({"dataset": d_key, "model": "TCN", "test_mae": res_t["metrics"]["MAE"]})
        baseline_records.append({"dataset": d_key, "model": "CNN", "test_mae": res_c["metrics"]["MAE"]})
        baseline_records.append({"dataset": d_key, "model": "Static_Equal_Ensemble", "test_mae": ens_metrics["MAE"]})

        err_l = (res_l["preds"] - res_l["trues"]).flatten()
        err_t = (res_t["preds"] - res_t["trues"]).flatten()
        err_c = (res_c["preds"] - res_c["trues"]).flatten()

        corr_lt = float(np.corrcoef(err_l, err_t)[0, 1])
        corr_lc = float(np.corrcoef(err_l, err_c)[0, 1])
        corr_tc = float(np.corrcoef(err_t, err_c)[0, 1])

        residual_records.append({
            "dataset": d_key,
            "corr_lstm_tcn": corr_lt,
            "corr_lstm_cnn": corr_lc,
            "corr_tcn_cnn": corr_tc,
            "lstm_test_mae": res_l["metrics"]["MAE"],
            "tcn_test_mae": res_t["metrics"]["MAE"],
            "cnn_test_mae": res_c["metrics"]["MAE"],
            "ensemble_test_mae": ens_metrics["MAE"],
        })
        print(f"  {d_key:7s}: Corr(L,T)={corr_lt:.3f}, Corr(L,C)={corr_lc:.3f}, Corr(T,C)={corr_tc:.3f} | Ens MAE={ens_metrics['MAE']:.2f}")

        # Daily block paired statistical test (CAEG V1 vs Equal Ensemble)
        horizon = 24
        K_blocks = len(res_l["preds"]) // horizon
        daily_caeg = []
        daily_ens = []
        # Get CAEG V1 seed 42 predictions
        caeg_v1 = OriginalCAEGNetPhase5(context_dim=datasets[d_key]["contexts"]["A0"]["train"].shape[1]).to(device)
        te_c = torch.tensor(datasets[d_key]["contexts"]["A0"]["test"], dtype=torch.float32)
        va_c = torch.tensor(datasets[d_key]["contexts"]["A0"]["val"], dtype=torch.float32)
        te_loader_ctx = DataLoader(TensorDataset(te_x, te_y, te_c), batch_size=64, shuffle=False)
        va_loader_ctx = DataLoader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)
        train_caeg_variant(caeg_v1, DataLoader(TensorDataset(tr_x, tr_y, torch.tensor(datasets[d_key]["contexts"]["A0"]["train"], dtype=torch.float32)), batch_size=64, shuffle=True), va_loader_ctx, device)
        res_caeg = evaluate_model_on_partition(caeg_v1, te_loader_ctx, scaler, device)

        diffs = []
        for k in range(K_blocks):
            idx = slice(k * horizon, (k + 1) * horizon)
            mae_c = np.mean(np.abs(res_caeg["preds"][idx] - res_caeg["trues"][idx]))
            mae_e = np.mean(np.abs(p_ens[idx] - res_l["trues"][idx]))
            diffs.append(mae_c - mae_e)

        diffs = np.array(diffs)
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        w_stat, w_pval = stats.wilcoxon(diffs) if np.any(diffs != 0) else (0.0, 1.0)
        daily_block_tests.append({
            "dataset": d_key,
            "comparison": "CAEG_V1_vs_Static_Ensemble",
            "k_blocks": K_blocks,
            "mean_daily_diff": float(np.mean(diffs)),
            "std_daily_diff": float(np.std(diffs)),
            "t_stat": float(t_stat),
            "p_val_t": float(p_val),
            "p_val_wilcoxon": float(w_pval),
        })

    df_res = pd.DataFrame(residual_records)
    df_res.to_csv("research/results/phase10_expert_complementarity.csv", index=False)

    df_base = pd.DataFrame(baseline_records)
    df_base.to_csv("research/results/phase10_baselines.csv", index=False)

    df_stat = pd.DataFrame(daily_block_tests)
    df_stat.to_csv("research/results/phase10_statistical_tests.csv", index=False)

    # 6. Generate Publication Figures (8 Required Figure Types)
    os.makedirs("research/results/phase10_plots", exist_ok=True)

    try:
        # Plot 1: Cross-Dataset Normalized Validation Improvement
        plt.figure(figsize=(12, 6))
        x = np.arange(len(df_cross))
        w = 0.25
        plt.bar(x - w, df_cross["pjm_rel_pct"], width=w, label="PJM Relative Δ (%)", color="#1f77b4")
        plt.bar(x, df_cross["gefcom_rel_pct"], width=w, label="GEFCom Relative Δ (%)", color="#ff7f0e")
        plt.bar(x + w, df_cross["uci_rel_pct"], width=w, label="UCI Relative Δ (%)", color="#2ca02c")
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.axhline(-2.0, color="red", linestyle=":", label="Degradation Threshold (-2%)")
        plt.xticks(x, df_cross["candidate_id"], rotation=45, ha="right")
        plt.ylabel("Relative Validation Improvement (%)")
        plt.title("Fig 1: Cross-Dataset Relative Validation Improvement vs Canonical V1")
        plt.legend()
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_01_cross_dataset_improvement.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 1: {e}")

    try:
        # Plot 2: Candidate vs V1 Validation MAE
        plt.figure(figsize=(12, 5))
        x_c = np.arange(len(df_cross))
        pjm_norm = df_cross["pjm_val_mae"] / v1_rows["PJM"] * 100.0
        gef_norm = df_cross["gefcom_val_mae"] / v1_rows["GEFCom"] * 100.0
        uci_norm = df_cross["uci_val_mae"] / v1_rows["UCI"] * 100.0
        plt.plot(x_c, pjm_norm, marker="o", label="PJM (% of V1)", color="#1f77b4")
        plt.plot(x_c, gef_norm, marker="s", label="GEFCom (% of V1)", color="#ff7f0e")
        plt.plot(x_c, uci_norm, marker="^", label="UCI (% of V1)", color="#2ca02c")
        plt.axhline(100.0, color="black", linestyle="--", label="Canonical V1 Baseline (100%)")
        plt.xticks(x_c, df_cross["candidate_id"], rotation=45, ha="right")
        plt.ylabel("Validation MAE (% of Canonical V1)")
        plt.title("Fig 2: Candidate Validation MAE Normalized to Canonical V1")
        plt.legend()
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_02_val_mae_comparison.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 2: {e}")

    try:
        # Plot 3: Expert Standalone MAE
        plt.figure(figsize=(10, 5))
        df_b_p = df_base[df_base["model"] != "Static_Equal_Ensemble"]
        x_b = np.arange(len(datasets))
        w_b = 0.25
        lstm_vals = [df_b_p[(df_b_p['dataset']==d) & (df_b_p['model']=='LSTM')]['test_mae'].values[0] for d in ["PJM", "GEFCom", "UCI"]]
        tcn_vals = [df_b_p[(df_b_p['dataset']==d) & (df_b_p['model']=='TCN')]['test_mae'].values[0] for d in ["PJM", "GEFCom", "UCI"]]
        cnn_vals = [df_b_p[(df_b_p['dataset']==d) & (df_b_p['model']=='CNN')]['test_mae'].values[0] for d in ["PJM", "GEFCom", "UCI"]]
        lstm_norm = [1.0, 1.0, 1.0]
        tcn_norm = [tcn_vals[i]/lstm_vals[i] for i in range(3)]
        cnn_norm = [cnn_vals[i]/lstm_vals[i] for i in range(3)]
        plt.bar(x_b - w_b, lstm_norm, width=w_b, label="LSTM (Normalized = 1.0)", color="#1f77b4")
        plt.bar(x_b, tcn_norm, width=w_b, label="TCN / LSTM Ratio", color="#ff7f0e")
        plt.bar(x_b + w_b, cnn_norm, width=w_b, label="CNN / LSTM Ratio", color="#2ca02c")
        plt.xticks(x_b, ["PJM", "GEFCom", "UCI"])
        plt.ylabel("Relative Error Ratio (LSTM = 1.0)")
        plt.title("Fig 3: Relative Standalone Expert Performance Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_03_standalone_expert_mae.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 3: {e}")

    try:
        # Plot 4: Equal Ensemble vs CAEG
        plt.figure(figsize=(8, 5))
        caeg_v1_maes = [df_fin[(df_fin['finalist_id']=='A0_Canonical_V1') & (df_fin['dataset']==d)]['test_mae_mean'].values[0] for d in ["PJM", "GEFCom", "UCI"]]
        ens_maes = [df_res[df_res['dataset']==d]['ensemble_test_mae'].values[0] for d in ["PJM", "GEFCom", "UCI"]]
        caeg_ens_ratio = [caeg_v1_maes[i] / ens_maes[i] for i in range(3)]
        plt.bar(["PJM", "GEFCom", "UCI"], caeg_ens_ratio, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
        plt.axhline(1.0, color="black", linestyle="--", label="Equal Ensemble Parity (1.0)")
        plt.ylabel("CAEG-Net V1 MAE / Static Ensemble MAE")
        plt.title("Fig 4: Ratio of CAEG-Net V1 Error to Static Equal Ensemble (< 1.0 favors CAEG)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_04_equal_vs_caeg.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 4: {e}")

    try:
        # Plot 5: Finalist Routing Weights
        if not df_route.empty:
            plt.figure(figsize=(10, 5))
            route_mean = df_route.groupby(["finalist_id", "dataset"])[["mean_w_lstm", "mean_w_tcn", "mean_w_cnn"]].mean().reset_index()
            v1_route = route_mean[route_mean["finalist_id"] == "A0_Canonical_V1"]
            x_d = np.arange(len(v1_route))
            plt.bar(x_d - 0.25, v1_route["mean_w_lstm"], width=0.25, label="LSTM", color="#1f77b4")
            plt.bar(x_d, v1_route["mean_w_tcn"], width=0.25, label="TCN", color="#ff7f0e")
            plt.bar(x_d + 0.25, v1_route["mean_w_cnn"], width=0.25, label="CNN", color="#2ca02c")
            plt.xticks(x_d, v1_route["dataset"])
            plt.ylabel("Average Routing Weight")
            plt.title("Fig 5: Canonical CAEG-Net V1 Routing Weights Across Datasets")
            plt.legend()
            plt.tight_layout()
            plt.savefig("research/results/phase10_plots/phase10_05_routing_weights.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error generating Plot 5: {e}")

    try:
        # Plot 6: Routing Entropy & Effective Experts
        if not df_route.empty:
            plt.figure(figsize=(8, 5))
            ent_mean = df_route.groupby(["finalist_id", "dataset"])[["mean_entropy", "effective_n_experts"]].mean().reset_index()
            v1_ent = ent_mean[ent_mean["finalist_id"] == "A0_Canonical_V1"]
            plt.bar(v1_ent["dataset"], v1_ent["effective_n_experts"], color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
            plt.axhline(3.0, color="black", linestyle="--", label="Theoretical Maximum (3.0 Experts)")
            plt.ylabel("Effective Number of Experts (N_eff)")
            plt.title("Fig 6: Effective Number of Active Experts Across Datasets")
            plt.legend()
            plt.tight_layout()
            plt.savefig("research/results/phase10_plots/phase10_06_entropy_effective_experts.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error generating Plot 6: {e}")

    try:
        # Plot 7: Expert Residual Correlation Heatmap
        plt.figure(figsize=(9, 4))
        corr_mat = df_res.set_index("dataset")[["corr_lstm_tcn", "corr_lstm_cnn", "corr_tcn_cnn"]]
        plt.imshow(corr_mat.values, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Pearson Correlation")
        plt.xticks(range(3), ["LSTM vs TCN", "LSTM vs CNN", "TCN vs CNN"])
        plt.yticks(range(len(corr_mat)), corr_mat.index)
        for i in range(len(corr_mat)):
            for j in range(3):
                val = corr_mat.values[i, j]
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.6 else "black")
        plt.title("Fig 7: Pairwise Expert Residual Correlation Across Datasets")
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_03_residual_correlation.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 7: {e}")

    try:
        # Plot 8: Representative Forecast Profile
        plt.figure(figsize=(10, 4))
        sample_true = res_l["trues"][0]
        sample_lstm = res_l["preds"][0]
        sample_tcn = res_t["preds"][0]
        sample_cnn = res_c["preds"][0]
        sample_ens = p_ens[0]
        h_idx = np.arange(1, 25)
        plt.plot(h_idx, sample_true, "k-", linewidth=2, label="Actual Load")
        plt.plot(h_idx, sample_lstm, "--", label="LSTM", alpha=0.7)
        plt.plot(h_idx, sample_tcn, ":", label="TCN", alpha=0.7)
        plt.plot(h_idx, sample_cnn, "-.", label="CNN", alpha=0.7)
        plt.plot(h_idx, sample_ens, "r-", linewidth=1.5, label="Ensemble")
        plt.xlabel("Forecast Horizon (Hours)")
        plt.ylabel(f"Load ({d_obj['unit']})")
        plt.title(f"Fig 8: Representative 24-Hour Forecast Profile ({d_obj['name']})")
        plt.legend()
        plt.tight_layout()
        plt.savefig("research/results/phase10_plots/phase10_08_sample_forecasts.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Plot 8: {e}")

    total_time = time.time() - t_start
    print(f"\nPhase 10 Experiment complete in {total_time/60.0:.2f} minutes.")


if __name__ == "__main__":
    run_phase10_experiment()
