"""
Phase 12: Causally Performance-Aware Adaptive Gating Optimization
===============================================================
Research Question:
Can causally available recent expert-performance information make CAEG-Net's
adaptive routing more effective and robust across PJM, GEFCom2014, and UCI
without sacrificing methodological integrity?

Candidate Models:
- B0: Canonical CAEG-Net V1 (Context 4D)
- B1: Static Equal Ensemble
- B2: Best Standalone Expert on Validation
- P1: Recent Expert MAE (Context 7D: 4D + [e_L, e_T, e_C])
- P2: Relative Recent Expert MAE (Context 7D: 4D + [r_L, r_T, r_C])
- P3: Relative Performance + Performance Trend (Context 10D: 4D + [r_L, r_T, r_C, slope_L, slope_T, slope_C])
- C1: P2 Router + Learned Adaptive/Equal Confidence Interpolation (lambda * y_adaptive + (1 - lambda) * y_equal)

Strict Constraints:
- Expert family remains LSTM, TCN, CNN (121,531 canonical parameters).
- Deterministic seeding with CuDNN determinism enabled.
- Stage A Validation Screening on seeds [42, 123].
- Stage B Five-Seed Finalist Evaluation on seeds [42, 123, 999, 2024, 3407].
- Non-overlapping daily-block statistical inference.
"""

import os
import sys
import time
import functools
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force unbuffered output
print = functools.partial(print, flush=True)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from caeg_net import LSTMExpert, TCNExpert, CNNExpert, count_parameters
from research.original_caeg import OriginalCAEGNetPhase5
from research.deterministic import seed_everything, make_deterministic_loader

# Import data loaders from run_phase10_optimization
from research.experiments.run_phase10_optimization import (
    load_all_three_datasets,
    train_caeg_variant,
    evaluate_model_on_partition,
    compute_metrics,
)


# =====================================================================
# 1. Model Definitions
# =====================================================================

class ConfidenceFallbackCAEGNet(nn.Module):
    """
    Candidate C1: P2 router + learned adaptive/equal confidence interpolation.
    y_final = lambda * y_adaptive + (1 - lambda) * y_equal
    lambda = Sigmoid(W_lambda * c + b_lambda)
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        # Confidence head predicting lambda in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_adaptive, w, diag = self.caeg(x, c, conservative_rho=conservative_rho, temperature=temperature)
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        lam = self.confidence_head(c)  # (B, 1)
        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        diag["lambda"] = lam
        return y_final, w, diag


def train_confidence_variant(
    model: ConfidenceFallbackCAEGNet,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    max_epochs=25,
    patience=6,
):
    """Training loop for ConfidenceFallbackCAEGNet with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x_b, y_b, c_b = [t.to(device) for t in batch]
            optimizer.zero_grad()
            y_pred, _, _ = model(x_b, c_b)
            loss = F.mse_loss(y_pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x_b, y_b, c_b = [t.to(device) for t in batch]
                y_pred, _, _ = model(x_b, c_b)
                val_loss += F.mse_loss(y_pred, y_b).item() * len(x_b)
                n_val += len(x_b)
        val_loss /= max(n_val, 1)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})


# =====================================================================
# 2. Causal Recent Expert Performance Computation
# =====================================================================

def compute_causal_expert_performance_features(
    m_lstm,
    m_tcn,
    m_cnn,
    windows,
    scaler,
    device,
):
    """
    Generate causal expert recent-error features across Train, Val, and Test.
    Strictly uses completed forecasts issued at origin t - 24, evaluating against
    observed targets Y[t - 24].
    """
    horizon = 24
    features = {}

    for part_key in ["train", "val", "test"]:
        part_x = torch.tensor(windows[part_key]["X"], dtype=torch.float32)
        part_y = torch.tensor(windows[part_key]["Y"], dtype=torch.float32)
        ds = TensorDataset(part_x, part_y)
        loader = make_deterministic_loader(ds, batch_size=64, shuffle=False)

        res_l = evaluate_model_on_partition(m_lstm, loader, scaler, device)
        res_t = evaluate_model_on_partition(m_tcn, loader, scaler, device)
        res_c = evaluate_model_on_partition(m_cnn, loader, scaler, device)

        # Window MAEs in physical units
        mae_l = np.mean(np.abs(res_l["preds"] - res_l["trues"]), axis=1)
        mae_t = np.mean(np.abs(res_t["preds"] - res_t["trues"]), axis=1)
        mae_c = np.mean(np.abs(res_c["preds"] - res_c["trues"]), axis=1)

        features[part_key] = {
            "mae_l": mae_l,
            "mae_t": mae_t,
            "mae_c": mae_c,
            "n_windows": len(mae_l),
        }

    # Handoff alignment for Causal Recent Performance
    # At origin t, the most recently completed forecast was issued at t - 24
    n_tr = features["train"]["n_windows"]
    n_val = features["val"]["n_windows"]
    n_te = features["test"]["n_windows"]

    prior_l = float(np.mean(features["train"]["mae_l"][:100]))
    prior_t = float(np.mean(features["train"]["mae_t"][:100]))
    prior_c = float(np.mean(features["train"]["mae_c"][:100]))

    def construct_causal_arrays(target_n, prev_maes, curr_maes):
        err_l = np.zeros(target_n, dtype=np.float32)
        err_t = np.zeros(target_n, dtype=np.float32)
        err_c = np.zeros(target_n, dtype=np.float32)

        for t in range(target_n):
            if t >= horizon:
                err_l[t] = curr_maes["mae_l"][t - horizon]
                err_t[t] = curr_maes["mae_t"][t - horizon]
                err_c[t] = curr_maes["mae_c"][t - horizon]
            else:
                if prev_maes is not None:
                    # Boundary handover from late prior partition
                    prev_len = prev_maes["n_windows"]
                    err_l[t] = prev_maes["mae_l"][prev_len - horizon + t]
                    err_t[t] = prev_maes["mae_t"][prev_len - horizon + t]
                    err_c[t] = prev_maes["mae_c"][prev_len - horizon + t]
                else:
                    err_l[t] = prior_l
                    err_t[t] = prior_t
                    err_c[t] = prior_c
        return err_l, err_t, err_c

    # 1. Train recent errors
    tr_el, tr_et, tr_ec = construct_causal_arrays(n_tr, None, features["train"])
    # 2. Val recent errors (handoff from train)
    val_el, val_et, val_ec = construct_causal_arrays(n_val, features["train"], features["val"])
    # 3. Test recent errors (handoff from val)
    te_el, te_et, te_ec = construct_causal_arrays(n_te, features["val"], features["test"])

    perf_dict = {
        "train": {"el": tr_el, "et": tr_et, "ec": tr_ec},
        "val": {"el": val_el, "et": val_et, "ec": val_ec},
        "test": {"el": te_el, "et": te_et, "ec": te_ec},
    }

    # Derive P1, P2, and P3 representations
    results = {"P1": {}, "P2": {}, "P3": {}}

    for p_key in ["train", "val", "test"]:
        el = perf_dict[p_key]["el"]
        et = perf_dict[p_key]["et"]
        ec = perf_dict[p_key]["ec"]
        n_pts = len(el)

        # P1: Raw recent MAEs
        p1_arr = np.stack([el, et, ec], axis=-1)  # (N, 3)

        # P2: Relative recent performance (sums to 1.0)
        eps = 1e-6
        e_sum = el + et + ec + eps
        rl = el / e_sum
        rt = et / e_sum
        rc = ec / e_sum
        p2_arr = np.stack([rl, rt, rc], axis=-1)  # (N, 3)

        # P3: Relative performance + Short-term trend slope (over past 3 completed forecasts)
        # slope = (err(t-24) - err(t-72)) / 48
        slope_l = np.zeros(n_pts, dtype=np.float32)
        slope_t = np.zeros(n_pts, dtype=np.float32)
        slope_c = np.zeros(n_pts, dtype=np.float32)

        for t in range(n_pts):
            if t >= 72:
                slope_l[t] = (el[t] - el[t - 48]) / 48.0
                slope_t[t] = (et[t] - et[t - 48]) / 48.0
                slope_c[t] = (ec[t] - ec[t - 48]) / 48.0

        p3_arr = np.stack([rl, rt, rc, slope_l, slope_t, slope_c], axis=-1)  # (N, 6)

        results["P1"][p_key] = p1_arr
        results["P2"][p_key] = p2_arr
        results["P3"][p_key] = p3_arr

    return results


# =====================================================================
# 3. Main Orchestrator
# =====================================================================

def run_phase12_experiment():
    print("=================================================================")
    print("PHASE 12: CAUSALLY PERFORMANCE-AWARE ADAPTIVE GATING OPTIMIZATION")
    print("=================================================================")
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    results_dir = "research/results"
    plots_dir = os.path.join(results_dir, "phase12_plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Complexity Audit
    complexity_records = [
        {"model_id": "B0_Canonical_V1", "context_dim": 4, "total_params": 121531, "param_increase": 0, "pct_increase": 0.0},
        {"model_id": "P1_Recent_Expert_MAE", "context_dim": 7, "total_params": 121579, "param_increase": 48, "pct_increase": 0.0395},
        {"model_id": "P2_Relative_Performance", "context_dim": 7, "total_params": 121579, "param_increase": 48, "pct_increase": 0.0395},
        {"model_id": "P3_Performance_Trend", "context_dim": 10, "total_params": 121627, "param_increase": 96, "pct_increase": 0.0790},
        {"model_id": "C1_Confidence_Fallback", "context_dim": 7, "total_params": 121724, "param_increase": 193, "pct_increase": 0.1588},
    ]
    df_complexity = pd.DataFrame(complexity_records)
    df_complexity.to_csv(os.path.join(results_dir, "phase12_complexity.csv"), index=False)
    print("\n--- Parameter Complexity Audit ---")
    print(df_complexity.to_string(index=False))

    # 2. Load Tri-Benchmark Datasets
    print("\n--- Loading Tri-Benchmark Datasets ---")
    datasets = load_all_three_datasets()

    # 3. Pre-train / Cache Standalone Experts & Extract Causal Performance Features
    print("\n--- Training Standalone Experts & Extracting Causal Performance Features ---")
    perf_features_by_dataset = {}
    standalone_val_maes = {}

    for d_key in ["PJM", "GEFCom", "UCI"]:
        print(f"Dataset: {d_key}")
        d_obj = datasets[d_key]
        windows = d_obj["windows"]
        scaler = d_obj["scaler"]

        tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
        tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
        va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
        va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)

        tr_loader = make_deterministic_loader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True, seed=42)
        va_loader = make_deterministic_loader(TensorDataset(va_x, va_y), batch_size=64, shuffle=False)

        seed_everything(42, deterministic_cudnn=True)
        m_lstm = LSTMExpert().to(device)
        train_caeg_variant(m_lstm, tr_loader, va_loader, device, max_epochs=25, patience=6)

        seed_everything(42, deterministic_cudnn=True)
        m_tcn = TCNExpert().to(device)
        train_caeg_variant(m_tcn, tr_loader, va_loader, device, max_epochs=25, patience=6)

        seed_everything(42, deterministic_cudnn=True)
        m_cnn = CNNExpert().to(device)
        train_caeg_variant(m_cnn, tr_loader, va_loader, device, max_epochs=25, patience=6)

        # Standalone validation MAEs
        res_l_val = evaluate_model_on_partition(m_lstm, va_loader, scaler, device)
        res_t_val = evaluate_model_on_partition(m_tcn, va_loader, scaler, device)
        res_c_val = evaluate_model_on_partition(m_cnn, va_loader, scaler, device)

        val_maes = {
            "LSTM": res_l_val["metrics"]["MAE"],
            "TCN": res_t_val["metrics"]["MAE"],
            "CNN": res_c_val["metrics"]["MAE"],
        }
        best_expert = min(val_maes, key=val_maes.get)
        standalone_val_maes[d_key] = {"best_expert": best_expert, "maes": val_maes}
        print(f"  Validation Standalone MAEs: LSTM={val_maes['LSTM']:.2f}, TCN={val_maes['TCN']:.2f}, CNN={val_maes['CNN']:.2f} -> Best: {best_expert}")

        # Extract Causal Performance Features
        perf_feats = compute_causal_expert_performance_features(m_lstm, m_tcn, m_cnn, windows, scaler, device)
        perf_features_by_dataset[d_key] = perf_feats

    # Build Context Dictionaries for Candidates
    # Canonical context is A0 (4D: trend, vol, autocorr, recent_error)
    candidate_contexts = {}
    for d_key in ["PJM", "GEFCom", "UCI"]:
        c_base = datasets[d_key]["contexts"]["A0"]
        p_feats = perf_features_by_dataset[d_key]
        candidate_contexts[d_key] = {
            "B0": c_base,
            "P1": {
                p: np.concatenate([c_base[p], p_feats["P1"][p]], axis=-1) for p in ["train", "val", "test"]
            },
            "P2": {
                p: np.concatenate([c_base[p], p_feats["P2"][p]], axis=-1) for p in ["train", "val", "test"]
            },
            "P3": {
                p: np.concatenate([c_base[p], p_feats["P3"][p]], axis=-1) for p in ["train", "val", "test"]
            },
            "C1": {
                p: np.concatenate([c_base[p], p_feats["P2"][p]], axis=-1) for p in ["train", "val", "test"]
            },
        }

    # =================================================================
    # 4. Stage A: Validation Screening (Seeds 42, 123)
    # =================================================================
    print("\n=======================================================")
    print("STAGE A: VALIDATION SCREENING (SEEDS [42, 123])")
    print("=======================================================")

    screening_seeds = [42, 123]
    candidate_ids = ["B0_Canonical_V1", "P1_Recent_Expert_MAE", "P2_Relative_Performance", "P3_Performance_Trend", "C1_Confidence_Fallback"]
    validation_records = []

    for cid in candidate_ids:
        c_key = cid.split("_")[0]
        print(f"\n--- Screening Candidate: {cid} ---")

        for d_key in ["PJM", "GEFCom", "UCI"]:
            d_obj = datasets[d_key]
            windows = d_obj["windows"]
            scaler = d_obj["scaler"]
            ctx_dict = candidate_contexts[d_key][c_key]

            tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
            tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
            va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
            va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)

            tr_c = torch.tensor(ctx_dict["train"], dtype=torch.float32)
            va_c = torch.tensor(ctx_dict["val"], dtype=torch.float32)

            val_maes = []
            val_rmses = []
            val_entropies = []

            for s in screening_seeds:
                seed_everything(s, deterministic_cudnn=True)
                tr_loader = make_deterministic_loader(TensorDataset(tr_x, tr_y, tr_c), batch_size=64, shuffle=True, seed=s)
                va_loader = make_deterministic_loader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)

                if c_key == "C1":
                    model = ConfidenceFallbackCAEGNet(context_dim=ctx_dict["train"].shape[1]).to(device)
                    train_confidence_variant(model, tr_loader, va_loader, device, max_epochs=25, patience=6)
                else:
                    model = OriginalCAEGNetPhase5(context_dim=ctx_dict["train"].shape[1]).to(device)
                    train_caeg_variant(model, tr_loader, va_loader, device, max_epochs=25, patience=6)

                eval_res = evaluate_model_on_partition(model, va_loader, scaler, device)
                val_maes.append(eval_res["metrics"]["MAE"])
                val_rmses.append(eval_res["metrics"]["RMSE"])

                w = eval_res["weights"]
                if w is not None:
                    eps = 1e-12
                    ent = -np.sum(w * np.log(w + eps), axis=-1)
                    val_entropies.append(np.mean(ent))

            m_mae = float(np.mean(val_maes))
            m_rmse = float(np.mean(val_rmses))
            m_ent = float(np.mean(val_entropies)) if val_entropies else 1.0
            n_eff = float(np.exp(m_ent))

            validation_records.append({
                "candidate_id": cid,
                "dataset": d_key,
                "unit": d_obj["unit"],
                "val_mae_mean": m_mae,
                "val_rmse_mean": m_rmse,
                "mean_entropy": m_ent,
                "effective_n_experts": n_eff,
            })
            print(f"  {d_key:7s}: Val MAE = {m_mae:.2f} {d_obj['unit']} | N_eff = {n_eff:.2f}")

    df_val = pd.DataFrame(validation_records)
    df_val.to_csv(os.path.join(results_dir, "phase12_validation_results.csv"), index=False)

    # 5. Qualification Assessment
    v1_rows = df_val[df_val["candidate_id"] == "B0_Canonical_V1"].set_index("dataset")["val_mae_mean"].to_dict()

    comparison_records = []
    for cid in candidate_ids:
        sub = df_val[df_val["candidate_id"] == cid].set_index("dataset")["val_mae_mean"].to_dict()
        rel_pjm = (sub["PJM"] - v1_rows["PJM"]) / v1_rows["PJM"] * 100
        rel_gef = (sub["GEFCom"] - v1_rows["GEFCom"]) / v1_rows["GEFCom"] * 100
        rel_uci = (sub["UCI"] - v1_rows["UCI"]) / v1_rows["UCI"] * 100

        # Negative relative change means lower error (improvement)
        impr_count = sum(1 for r in [rel_pjm, rel_gef, rel_uci] if r < 0.0)
        worst_degr = max([rel_pjm, rel_gef, rel_uci])
        qualifies = (impr_count >= 2) and (worst_degr <= 2.0)

        comparison_records.append({
            "candidate_id": cid,
            "pjm_val_mae": sub["PJM"],
            "gefcom_val_mae": sub["GEFCom"],
            "uci_val_mae": sub["UCI"],
            "rel_change_pjm_pct": rel_pjm,
            "rel_change_gefcom_pct": rel_gef,
            "rel_change_uci_pct": rel_uci,
            "datasets_improved": impr_count,
            "worst_dataset_degradation_pct": worst_degr,
            "qualified": "YES" if qualifies else "NO",
        })

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(os.path.join(results_dir, "phase12_candidate_comparison.csv"), index=False)
    print("\n--- Validation Screening Comparison & Qualification ---")
    print(df_comp.to_string(index=False))

    qualified_finalists = df_comp[df_comp["qualified"] == "YES"]["candidate_id"].tolist()
    # Always include B0 for comparison
    finalist_ids = ["B0_Canonical_V1"] + [q for q in qualified_finalists if q != "B0_Canonical_V1"]

    print(f"\nFinalists proceeding to Stage B (5-Seed Benchmark): {finalist_ids}")

    # =================================================================
    # 5. Stage B: Five-Seed Finalist Evaluation (Held-Out Test Partitions)
    # =================================================================
    print("\n=======================================================")
    print("STAGE B: FIVE-SEED FINALIST BENCHMARK ON HELD-OUT TEST")
    print("=======================================================")

    canonical_seeds = [42, 123, 999, 2024, 3407]
    five_seed_records = []
    daily_block_records = []
    routing_records = []
    dataset_summary_records = []

    # Cache predictions of best finalist and B0 for block tests
    finalist_predictions = {}

    for cid in finalist_ids:
        c_key = cid.split("_")[0]
        print(f"\n>>> Running 5 Seeds for Finalist: {cid} <<<")

        for d_key in ["PJM", "GEFCom", "UCI"]:
            d_obj = datasets[d_key]
            windows = d_obj["windows"]
            scaler = d_obj["scaler"]
            ctx_dict = candidate_contexts[d_key][c_key]

            tr_x = torch.tensor(windows["train"]["X"], dtype=torch.float32)
            tr_y = torch.tensor(windows["train"]["Y"], dtype=torch.float32)
            va_x = torch.tensor(windows["val"]["X"], dtype=torch.float32)
            va_y = torch.tensor(windows["val"]["Y"], dtype=torch.float32)
            te_x = torch.tensor(windows["test"]["X"], dtype=torch.float32)
            te_y = torch.tensor(windows["test"]["Y"], dtype=torch.float32)

            tr_c = torch.tensor(ctx_dict["train"], dtype=torch.float32)
            va_c = torch.tensor(ctx_dict["val"], dtype=torch.float32)
            te_c = torch.tensor(ctx_dict["test"], dtype=torch.float32)

            seed_maes = []
            seed_rmses = []
            seed_mses = []
            seed_r2s = []
            seed_mapes = []
            seed_preds = []

            for s in canonical_seeds:
                seed_everything(s, deterministic_cudnn=True)
                tr_loader = make_deterministic_loader(TensorDataset(tr_x, tr_y, tr_c), batch_size=64, shuffle=True, seed=s)
                va_loader = make_deterministic_loader(TensorDataset(va_x, va_y, va_c), batch_size=64, shuffle=False)
                te_loader = make_deterministic_loader(TensorDataset(te_x, te_y, te_c), batch_size=64, shuffle=False)

                if c_key == "C1":
                    model = ConfidenceFallbackCAEGNet(context_dim=ctx_dict["train"].shape[1]).to(device)
                    train_confidence_variant(model, tr_loader, va_loader, device, max_epochs=25, patience=6)
                else:
                    model = OriginalCAEGNetPhase5(context_dim=ctx_dict["train"].shape[1]).to(device)
                    train_caeg_variant(model, tr_loader, va_loader, device, max_epochs=25, patience=6)

                test_eval = evaluate_model_on_partition(model, te_loader, scaler, device)
                m = test_eval["metrics"]
                seed_maes.append(m["MAE"])
                seed_rmses.append(m["RMSE"])
                seed_mses.append(m["MSE"])
                seed_r2s.append(m["R2"])
                seed_mapes.append(m["MAPE"])
                seed_preds.append(test_eval["preds"])

                # Routing diagnostics
                w = test_eval["weights"]
                if w is not None:
                    eps = 1e-12
                    ent = -np.sum(w * np.log(w + eps), axis=-1)
                    routing_records.append({
                        "candidate_id": cid,
                        "dataset": d_key,
                        "seed": s,
                        "mean_w_lstm": float(np.mean(w[:, 0])),
                        "mean_w_tcn": float(np.mean(w[:, 1])),
                        "mean_w_cnn": float(np.mean(w[:, 2])),
                        "std_w_lstm": float(np.std(w[:, 0])),
                        "std_w_tcn": float(np.std(w[:, 1])),
                        "std_w_cnn": float(np.std(w[:, 2])),
                        "mean_entropy": float(np.mean(ent)),
                        "effective_n_experts": float(np.mean(np.exp(ent))),
                    })

            # Store predictions of seed 42 for daily block tests
            finalist_predictions[(cid, d_key)] = {
                "preds_seed42": seed_preds[0],
                "trues": test_eval["trues"],
            }

            five_seed_records.append({
                "candidate_id": cid,
                "dataset": d_key,
                "unit": d_obj["unit"],
                "test_mae_mean": float(np.mean(seed_maes)),
                "test_mae_std": float(np.std(seed_maes)),  # Population SD (ddof=0)
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

    df_5seed = pd.DataFrame(five_seed_records)
    df_5seed.to_csv(os.path.join(results_dir, "phase12_five_seed_results.csv"), index=False)

    df_routing = pd.DataFrame(routing_records)
    df_routing.to_csv(os.path.join(results_dir, "phase12_routing_analysis.csv"), index=False)

    # 6. Daily-Block Inferential Statistical Testing
    print("\n--- Non-Overlapping Daily-Block Statistical Inference ---")
    horizon = 24
    for d_key in ["PJM", "GEFCom", "UCI"]:
        trues = finalist_predictions[("B0_Canonical_V1", d_key)]["trues"]
        n_windows = len(trues)
        k_blocks = n_windows // horizon

        p_v1 = finalist_predictions[("B0_Canonical_V1", d_key)]["preds_seed42"]

        for cid in finalist_ids:
            if cid == "B0_Canonical_V1":
                continue
            p_cand = finalist_predictions[(cid, d_key)]["preds_seed42"]

            # Compute non-overlapping daily differences
            diffs = []
            for k in range(k_blocks):
                idx = slice(k * horizon, (k + 1) * horizon)
                mae_cand = np.mean(np.abs(p_cand[idx] - trues[idx]))
                mae_v1 = np.mean(np.abs(p_v1[idx] - trues[idx]))
                diffs.append(mae_cand - mae_v1)

            diffs = np.array(diffs)
            m_diff = float(np.mean(diffs))
            s_diff = float(np.std(diffs, ddof=1))
            se = s_diff / np.sqrt(k_blocks)
            ci_low = m_diff - 1.96 * se
            ci_high = m_diff + 1.96 * se
            t_stat, p_t = stats.ttest_1samp(diffs, 0.0)
            try:
                w_stat, p_w = stats.wilcoxon(diffs)
            except Exception:
                w_stat, p_w = np.nan, np.nan
            cohen_d = m_diff / (s_diff + 1e-8)

            daily_block_records.append({
                "candidate_id": cid,
                "comparison": "vs_Canonical_V1",
                "dataset": d_key,
                "k_blocks": k_blocks,
                "mean_daily_diff": m_diff,
                "std_daily_diff": s_diff,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "t_stat": float(t_stat),
                "p_value_t": float(p_t),
                "w_stat": float(w_stat) if not np.isnan(w_stat) else 0.0,
                "p_value_wilcoxon": float(p_w) if not np.isnan(p_w) else 1.0,
                "cohen_d": float(cohen_d),
            })

    df_stats = pd.DataFrame(daily_block_records)
    if not df_stats.empty:
        # Apply Holm-Bonferroni correction
        def holm_adjust(p_vals):
            p_vals = np.asarray(p_vals)
            n = len(p_vals)
            order = np.argsort(p_vals)
            adj = np.empty(n)
            cum_max = 0.0
            for rank, idx in enumerate(order):
                p = p_vals[idx] * (n - rank)
                cum_max = max(cum_max, p)
                adj[idx] = min(1.0, cum_max)
            return adj

        df_stats["p_val_t_holm"] = holm_adjust(df_stats["p_value_t"].values)
        df_stats["p_val_wilcoxon_holm"] = holm_adjust(df_stats["p_value_wilcoxon"].values)
        df_stats.to_csv(os.path.join(results_dir, "phase12_statistical_tests.csv"), index=False)
        print(df_stats.to_string(index=False))
    else:
        # Save empty structure if no candidate qualified
        df_stats = pd.DataFrame(columns=[
            "candidate_id", "comparison", "dataset", "k_blocks", "mean_daily_diff",
            "std_daily_diff", "ci_95_low", "ci_95_high", "t_stat", "p_value_t",
            "w_stat", "p_value_wilcoxon", "cohen_d", "p_val_t_holm", "p_val_wilcoxon_holm"
        ])
        df_stats.to_csv(os.path.join(results_dir, "phase12_statistical_tests.csv"), index=False)
        print("No candidates qualified for Stage B statistical testing.")

    # 7. Dataset Summary Compilation
    for d_key in ["PJM", "GEFCom", "UCI"]:
        b0_row = df_5seed[(df_5seed["candidate_id"] == "B0_Canonical_V1") & (df_5seed["dataset"] == d_key)].iloc[0]
        dataset_summary_records.append({
            "dataset": d_key,
            "unit": b0_row["unit"],
            "v1_test_mae": b0_row["test_mae_mean"],
            "v1_test_mae_sd": b0_row["test_mae_std"],
            "best_expert_val": standalone_val_maes[d_key]["best_expert"],
            "best_expert_val_mae": standalone_val_maes[d_key]["maes"][standalone_val_maes[d_key]["best_expert"]],
            "n_qualified_finalists": len(qualified_finalists),
        })
    df_dsummary = pd.DataFrame(dataset_summary_records)
    df_dsummary.to_csv(os.path.join(results_dir, "phase12_dataset_summary.csv"), index=False)

    # =================================================================
    # 8. Publication-Quality Plots
    # =================================================================
    print("\n--- Generating Publication Figures ---")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})

    # Fig 1: Validation candidate comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    c_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_val[df_val["dataset"] == d_key]
        x = np.arange(len(sub))
        ax.bar(x, sub["val_mae_mean"], color=c_palette[:len(sub)], width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([c.split("_")[0] for c in sub["candidate_id"]], rotation=0)
        ax.set_title(f"{d_key} ({sub.iloc[0]['unit']})")
        ax.set_ylabel(f"Val MAE ({sub.iloc[0]['unit']})")
        # Highlight B0
        ax.axhline(v1_rows[d_key], color="black", linestyle="--", alpha=0.7, label="B0 V1 Baseline")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 1: Stage A Validation Screening (Mean MAE Across Seeds 42, 123)", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_01_validation_candidate_comparison.png"), bbox_inches="tight")
    plt.close()

    # Fig 2: Five-seed MAE comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_5seed[df_5seed["dataset"] == d_key]
        x = np.arange(len(sub))
        ax.bar(x, sub["test_mae_mean"], yerr=sub["test_mae_std"], capsize=5, color="#1f77b4", width=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([c.split("_")[0] for c in sub["candidate_id"]])
        ax.set_title(f"{d_key} ({sub.iloc[0]['unit']})")
        ax.set_ylabel(f"Test MAE ({sub.iloc[0]['unit']})")
    plt.suptitle("Figure 2: Five-Seed Finalist Benchmark on Held-Out Test Partitions (Mean ± SD)", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_02_five_seed_mae_comparison.png"), bbox_inches="tight")
    plt.close()

    # Fig 3: Paired daily-block differences (if qualified, otherwise bar chart)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df_stats.empty:
        x = np.arange(len(df_stats))
        ax.bar(x, df_stats["mean_daily_diff"], yerr=1.96 * df_stats["std_daily_diff"] / np.sqrt(df_stats["k_blocks"]), capsize=5, color="coral", width=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['candidate_id'].split('_')[0]} ({r['dataset']})" for _, r in df_stats.iterrows()])
        ax.axhline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_ylabel("Mean Daily Difference (Candidate - V1)")
        ax.set_title("Figure 3: Paired Daily-Block MAE Difference vs Canonical V1 (95% CI)")
    else:
        ax.text(0.5, 0.5, "No Candidate Qualified Stage A Validation Screening\n(Canonical V1 Retained as Sole Finalist)", ha="center", va="center", fontsize=12)
        ax.set_title("Figure 3: Daily-Block Inferential Hypothesis Testing (Null Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_03_paired_daily_differences.png"), bbox_inches="tight")
    plt.close()

    # Fig 4: Expert weight distributions across seeds
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, d_key in enumerate(["PJM", "GEFCom", "UCI"]):
        ax = axes[idx]
        sub = df_routing[df_routing["dataset"] == d_key]
        grp = sub.groupby("candidate_id").agg({"mean_w_lstm": "mean", "mean_w_tcn": "mean", "mean_w_cnn": "mean"}).reset_index()
        x = np.arange(len(grp))
        w = 0.25
        ax.bar(x - w, grp["mean_w_lstm"], width=w, label="w_LSTM", color="#2ca02c")
        ax.bar(x, grp["mean_w_tcn"], width=w, label="w_TCN", color="#1f77b4")
        ax.bar(x + w, grp["mean_w_cnn"], width=w, label="w_CNN", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([c.split("_")[0] for c in grp["candidate_id"]])
        ax.set_ylim(0, 0.7)
        ax.set_title(f"{d_key}")
        ax.set_ylabel("Mean Routing Weight")
        if idx == 0:
            ax.legend()
    plt.suptitle("Figure 4: Learned Gating Weights Distribution by Finalist", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_04_expert_weight_distributions.png"), bbox_inches="tight")
    plt.close()

    # Fig 5: Error vs Weight response
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    w = 0.25
    d_names = ["PJM", "GEFCom", "UCI"]
    for i, d_key in enumerate(d_names):
        b0_route = df_routing[(df_routing["dataset"] == d_key) & (df_routing["candidate_id"] == "B0_Canonical_V1")]
        ax.bar(x[i] - w, b0_route["mean_w_lstm"].mean(), width=w, label="LSTM" if i == 0 else "", color="#2ca02c")
        ax.bar(x[i], b0_route["mean_w_tcn"].mean(), width=w, label="TCN" if i == 0 else "", color="#1f77b4")
        ax.bar(x[i] + w, b0_route["mean_w_cnn"].mean(), width=w, label="CNN" if i == 0 else "", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(d_names)
    ax.set_ylabel("Mean Routing Weight")
    ax.set_title("Figure 5: Canonical V1 Router Allocation Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_05_error_vs_weight_response.png"), bbox_inches="tight")
    plt.close()

    # Fig 6: Routing entropy & N_eff comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grp_neff = df_routing.groupby(["candidate_id", "dataset"])["effective_n_experts"].mean().unstack()
    grp_neff.plot(kind="bar", ax=ax, width=0.6)
    ax.set_xticklabels([c.split("_")[0] for c in grp_neff.index], rotation=0)
    ax.axhline(3.0, color="gray", linestyle="--", alpha=0.7, label="Theoretical Max (3.0)")
    ax.set_ylabel("Effective Experts (N_eff = exp(H))")
    ax.set_title("Figure 6: Routing Entropy & Effective Active Experts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_06_entropy_neff_comparison.png"), bbox_inches="tight")
    plt.close()

    # Fig 7: Dataset-level improvement across candidates (Validation)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cands_plot = [c for c in df_comp["candidate_id"] if c != "B0_Canonical_V1"]
    sub_df = df_comp[df_comp["candidate_id"].isin(cands_plot)]
    x = np.arange(len(sub_df))
    w = 0.25
    ax.bar(x - w, sub_df["rel_change_pjm_pct"], width=w, label="PJM", color="#1f77b4")
    ax.bar(x, sub_df["rel_change_gefcom_pct"], width=w, label="GEFCom", color="#ff7f0e")
    ax.bar(x + w, sub_df["rel_change_uci_pct"], width=w, label="UCI", color="#2ca02c")
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.axhline(2.0, color="red", linestyle=":", alpha=0.7, label="Max Permissible Degradation (+2%)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[0] for c in sub_df["candidate_id"]])
    ax.set_ylabel("Relative Change vs V1 (%) [Negative = Improved]")
    ax.set_title("Figure 7: Relative Validation Change vs Canonical V1 Across Datasets")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_07_dataset_level_improvement.png"), bbox_inches="tight")
    plt.close()

    # Fig 8: Complexity vs performance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for _, row in df_complexity.iterrows():
        cid = row["model_id"]
        p_cnt = row["total_params"]
        # Use average relative validation change across datasets
        if cid == "B0_Canonical_V1":
            mean_rel = 0.0
        else:
            match = df_comp[df_comp["candidate_id"] == cid]
            mean_rel = float((match["rel_change_pjm_pct"].values[0] + match["rel_change_gefcom_pct"].values[0] + match["rel_change_uci_pct"].values[0]) / 3.0)
        ax.scatter(p_cnt, mean_rel, s=120, label=cid.split("_")[0])
        ax.annotate(cid.split("_")[0], (p_cnt + 5, mean_rel + 0.05), fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Total Model Parameters")
    ax.set_ylabel("Mean Relative Validation Change (%) [Negative = Improved]")
    ax.set_title("Figure 8: Model Parameter Complexity vs. Mean Validation Change")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase12_08_complexity_vs_performance.png"), bbox_inches="tight")
    plt.close()

    print(f"\nAll 8 publication figures successfully generated in: {plots_dir}")
    print(f"Phase 12 completed in {time.time() - t_start:.2f} seconds.")


if __name__ == "__main__":
    run_phase12_experiment()
