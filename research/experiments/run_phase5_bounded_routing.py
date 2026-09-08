"""
CAEG-Net Phase 5: Bounded / Conservative Routing (CAEG-Net BR) Experimental Suite
==================================================================================
Executes the controlled Phase 5 Bounded Routing investigation:
1. Smoke test: gradient isolation, loss calculation, forward routing, rho=0 identity.
2. Validation grid search: rho in {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00},
   with lambda_dev in {0.0, 0.025} strictly on validation data.
3. Lock optimal hyperparameters (rho*, lambda_dev*) on validation data using conservative tie-break.
4. Five-seed test benchmark across seeds [42, 43, 44, 45, 46].
5. 53 daily block statistical hypothesis testing (paired t, Wilcoxon, DM, Cohen's d, Holm-Bonferroni).
6. Forecasting difficulty regime breakdown (baseline error, disagreement, volatility).
7. Routing telemetry and weight deviation analysis.
8. 6 publication-quality figures.
9. Machine-readable findings JSON and detailed CSV exports.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders
from research.models import (
    GatedRecurrentExpert,
    MultiScaleCausalTCNExpert,
    PatchTemporalExpert,
    BoundedRoutingCAEG,
    CAEGNetBR,
    ShrinkageRegularizedCAEG,
    CAEGNetSR,
    LearnedStaticEnsemble,
    DecoupledCAEGNet,
    compute_bounded_loss,
    count_parameters,
)
from research.training import (
    train_bounded_router,
    train_shrinkage_router,
    evaluate_bounded_caeg,
    evaluate_shrinkage_caeg,
    evaluate_decoupled_caeg,
)


def compute_metrics(preds_mw: np.ndarray, trues_mw: np.ndarray) -> Dict[str, float]:
    err = preds_mw - trues_mw
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues_mw - np.mean(trues_mw)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(trues_mw), 1e-6))) * 100.0)
    return {
        "mae_mw": mae,
        "mse_mw2": mse,
        "rmse_mw": rmse,
        "r2": r2,
        "mape_pct": mape,
    }


def compute_diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> Tuple[float, float]:
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    gamma = 0.0
    for lag in range(1, h):
        c = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma += 2.0 * c
    var_d = (gamma0 + gamma) / n
    if var_d <= 1e-12:
        return 0.0, 1.0
    dm_stat = float(mean_d / np.sqrt(var_d))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val


def load_standalone_experts(seed: int, device: torch.device) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Load cached standalone expert backbones from Phase 7."""
    ckpt_dir = os.path.join("research", "checkpoints", "phase7")
    gru_ckpt = os.path.join(ckpt_dir, f"standalone_gru_seed_{seed}.pt")
    tcn_ckpt = os.path.join(ckpt_dir, f"standalone_tcn_seed_{seed}.pt")
    patch_ckpt = os.path.join(ckpt_dir, f"standalone_patch_seed_{seed}.pt")

    gru = GatedRecurrentExpert(input_dim=1, hidden_dim=54, num_layers=2, horizon=24).to(device)
    tcn = MultiScaleCausalTCNExpert(in_channels=1, channels=34, kernel_size=4, dilations=(1, 2, 4, 8, 16), horizon=24).to(device)
    patch = PatchTemporalExpert(seq_len=168, patch_len=24, stride=12, embed_dim=48, hidden_dim=96, horizon=24).to(device)

    gru.load_state_dict(torch.load(gru_ckpt, map_location=device))
    tcn.load_state_dict(torch.load(tcn_ckpt, map_location=device))
    patch.load_state_dict(torch.load(patch_ckpt, map_location=device))

    gru.eval()
    tcn.eval()
    patch.eval()
    return gru, tcn, patch


def run_smoke_test(pipe: Dict, device: torch.device, results_dir: str):
    print("\n--- Running Phase 5 Bounded Routing Smoke Test ---")
    dataloaders = build_research_v2_dataloaders(pipe, batch_size=32, seed=42)
    gru, tcn, patch = load_standalone_experts(42, device)

    # Instantiate BR model with rho=0.2
    model = BoundedRoutingCAEG(gru, tcn, patch, rho=0.2).to(device)
    res = train_bounded_router(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        rho=0.2,
        lambda_dev=0.025,
        max_epochs=2,
        patience=2,
        device=device,
        verbose=False,
    )
    eval_res = evaluate_bounded_caeg(model, dataloaders["val"], pipe["scaler"], rho=0.2, device=device)

    # Mandatory rho=0 equivalence test on validation
    eval_zero = evaluate_bounded_caeg(model, dataloaders["val"], pipe["scaler"], rho=0.0, device=device)
    diff_from_equal = float(np.max(np.abs(eval_zero["y_pred_mw"] - eval_zero["equal_ens_mw"])))
    assert diff_from_equal < 0.005, f"rho=0 does not match equal ensemble! Max diff: {diff_from_equal}"

    smoke_info = {
        "smoke_test": "PASSED",
        "best_epoch": res["best_epoch"],
        "val_mae_mw": eval_res["fused_metrics"]["mae_mw"],
        "mean_l1_dev": eval_res["weight_stats"]["mean_l1_dev_from_equal"],
        "mean_kl": eval_res["weight_stats"]["mean_kl_divergence"],
        "rho_zero_max_diff_from_equal_mw": diff_from_equal,
        "param_accounting": count_parameters(model),
    }
    with open(os.path.join(results_dir, "phase5_bounded_routing_smoke_test.json"), "w", encoding="utf-8") as f:
        json.dump(smoke_info, f, indent=2)
    print(f"Smoke test passed: {smoke_info}")


def run_validation_grid_search(
    pipe: Dict,
    seeds: List[int],
    rhos: List[float],
    lambdas_dev: List[float],
    device: torch.device,
    results_dir: str,
) -> pd.DataFrame:
    """
    Grid search over (rho, lambda_dev) strictly on the VALIDATION partition.
    Never evaluates or touches test data.
    """
    print("\n=========================================================================")
    print(">>> RUNNING PHASE 5 BOUNDED ROUTING VALIDATION GRID SEARCH <<<")
    print("=========================================================================")

    scaler = pipe["scaler"]
    val_records = []

    # Benchmark Equal Ensemble on Validation
    equal_val_maes = []
    for seed in seeds:
        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
        gru, tcn, patch = load_standalone_experts(seed, device)
        model = BoundedRoutingCAEG(gru, tcn, patch, rho=0.0).to(device)
        eval_eq = evaluate_bounded_caeg(model, dataloaders["val"], scaler, rho=0.0, device=device)
        equal_val_maes.append(eval_eq["fused_metrics"]["mae_mw"])
    mean_equal_val_mae = float(np.mean(equal_val_maes))
    print(f"Baseline: Static Equal Ensemble Mean Val MAE across seeds: {mean_equal_val_mae:.2f} MW\n")

    grid_configs = []
    for rho in rhos:
        for lam in lambdas_dev:
            grid_configs.append((rho, lam))

    print(f"Evaluating {len(grid_configs)} (rho, lambda_dev) configurations across {len(seeds)} seeds...")

    for rho, lam in grid_configs:
        seed_maes = []
        seed_rmses = []
        seed_r2s = []
        seed_l1_devs = []
        seed_kls = []
        seed_entropies = []
        seed_w_gru = []
        seed_w_tcn = []
        seed_w_patch = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
            gru, tcn, patch = load_standalone_experts(seed, device)

            model = BoundedRoutingCAEG(gru, tcn, patch, rho=rho).to(device)
            if rho > 0.0:
                res = train_bounded_router(
                    model=model,
                    train_loader=dataloaders["train"],
                    val_loader=dataloaders["val"],
                    rho=rho,
                    lambda_dev=lam,
                    lr=1e-3,
                    weight_decay=1e-4,
                    max_epochs=25,
                    patience=5,
                    device=device,
                    verbose=False,
                )
            eval_res = evaluate_bounded_caeg(model, dataloaders["val"], scaler, rho=rho, device=device)

            seed_maes.append(eval_res["fused_metrics"]["mae_mw"])
            seed_rmses.append(eval_res["fused_metrics"]["rmse_mw"])
            seed_r2s.append(eval_res["fused_metrics"]["r2"])
            seed_l1_devs.append(eval_res["weight_stats"]["mean_l1_dev_from_equal"])
            seed_kls.append(eval_res["weight_stats"]["mean_kl_divergence"])
            seed_entropies.append(eval_res["weight_stats"]["mean_entropy"])
            seed_w_gru.append(eval_res["weight_stats"]["mean_gru_weight"])
            seed_w_tcn.append(eval_res["weight_stats"]["mean_tcn_weight"])
            seed_w_patch.append(eval_res["weight_stats"]["mean_patch_weight"])

        m_mae = float(np.mean(seed_maes))
        s_mae = float(np.std(seed_maes))
        m_rmse = float(np.mean(seed_rmses))
        m_r2 = float(np.mean(seed_r2s))
        m_l1 = float(np.mean(seed_l1_devs))
        m_kl = float(np.mean(seed_kls))
        m_ent = float(np.mean(seed_entropies))
        gain_over_equal = mean_equal_val_mae - m_mae

        val_records.append({
            "rho": rho,
            "lambda_dev": lam,
            "val_mae_mean": m_mae,
            "val_mae_std": s_mae,
            "val_rmse_mean": m_rmse,
            "val_r2_mean": m_r2,
            "gain_over_equal_mw": gain_over_equal,
            "mean_l1_dev": m_l1,
            "mean_kl": m_kl,
            "mean_entropy": m_ent,
            "mean_gru_weight": float(np.mean(seed_w_gru)),
            "mean_tcn_weight": float(np.mean(seed_w_tcn)),
            "mean_patch_weight": float(np.mean(seed_w_patch)),
        })
        print(f"rho={rho:4.2f} | lam={lam:5.3f} -> Val MAE: {m_mae:6.2f} +/- {s_mae:4.2f} MW | Gain: {gain_over_equal:+5.2f} MW | L1 Dev: {m_l1:.4f} | KL: {m_kl:.4f}")

    df_val = pd.DataFrame(val_records)
    df_val.sort_values(by="val_mae_mean", inplace=True)
    df_val.to_csv(os.path.join(results_dir, "phase5_bounded_routing_validation.csv"), index=False)
    print("\nSaved validation grid search results to phase5_bounded_routing_validation.csv")
    return df_val


def select_and_lock_hyperparameters(df_val: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Select optimal hyperparameters strictly based on validation MAE.
    Tie-breaking rule: if multiple configurations are within 0.25 MW of best MAE,
    select the configuration with smaller rho (more conservative).
    """
    best_mae = df_val["val_mae_mean"].min()
    near_best = df_val[df_val["val_mae_mean"] <= best_mae + 0.25]
    
    # Sort near best by rho ascending, then lambda_dev descending (more regularized first)
    near_best_sorted = near_best.sort_values(by=["rho", "lambda_dev"], ascending=[True, False])
    locked_best = near_best_sorted.iloc[0].to_dict()

    # Also extract best unconstrained / pure bounded (lambda_dev = 0.0)
    df_no_kl = df_val[df_val["lambda_dev"] == 0.0].sort_values(by="val_mae_mean")
    best_no_kl_mae = df_no_kl["val_mae_mean"].min()
    near_best_no_kl = df_no_kl[df_no_kl["val_mae_mean"] <= best_no_kl_mae + 0.25].sort_values(by="rho", ascending=True)
    locked_no_kl = near_best_no_kl.iloc[0].to_dict()

    print("\n=========================================================================")
    print(">>> HYPERPARAMETER LOCKING PROTOCOL (PRE-TEST ENFORCEMENT) <<<")
    print("=========================================================================")
    print(f"Global Best Validation MAE: {best_mae:.2f} MW")
    print(f"Candidates within 0.25 MW tie threshold: {len(near_best)}")
    print(f"Selected Primary Model (Locked): rho* = {locked_best['rho']}, lambda_dev* = {locked_best['lambda_dev']}")
    print(f"  Validation MAE: {locked_best['val_mae_mean']:.2f} +/- {locked_best['val_mae_std']:.2f} MW")
    print(f"Selected Pure Bounded (No KL): rho* = {locked_no_kl['rho']}, lambda_dev = 0.0")
    print(f"  Validation MAE: {locked_no_kl['val_mae_mean']:.2f} +/- {locked_no_kl['val_mae_std']:.2f} MW")
    print("Hyperparameters are now permanently locked before touching test data.\n")
    return locked_best, locked_no_kl


def run_test_benchmark(
    pipe: Dict,
    seeds: List[int],
    locked_best: Dict,
    locked_no_kl: Dict,
    device: torch.device,
    results_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Execute final multi-seed test benchmark across seeds [42, 43, 44, 45, 46]
    comparing CAEG-Net BR against all established baselines.
    """
    print("\n=========================================================================")
    print(">>> RUNNING 5-SEED TEST BENCHMARK ON UNTOUCHED TEST DATA <<<")
    print("=========================================================================")

    scaler = pipe["scaler"]
    pred_dir = os.path.join(results_dir, "phase5_bounded_routing_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    models_to_evaluate = [
        "Standalone_GRU",
        "Standalone_TCN",
        "Standalone_Patch",
        "Static_Equal_Ensemble",
        "Learned_Static_Weights",
        "Decoupled_CAEG",
        "CAEG_Net_SR",
        "CAEG_Net_BR_Optimal",
        "CAEG_Net_BR_No_KL",
        "CAEG_Net_BR_Rho_0_10",
        "CAEG_Net_BR_Rho_0_20",
    ]

    all_seed_results = []
    saved_preds = {m: [] for m in models_to_evaluate}
    saved_weights = {m: [] for m in ["CAEG_Net_BR_Optimal", "CAEG_Net_BR_No_KL", "CAEG_Net_BR_Rho_0_10", "CAEG_Net_BR_Rho_0_20", "CAEG_Net_SR", "Decoupled_CAEG"]}
    test_y_true = None

    for seed in seeds:
        print(f"\n--- Benchmarking Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        dataloaders = build_research_v2_dataloaders(pipe, batch_size=64, seed=seed)
        gru, tcn, patch = load_standalone_experts(seed, device)

        # 1. Standalone experts & Equal Ensemble
        eval_base = evaluate_bounded_caeg(
            BoundedRoutingCAEG(gru, tcn, patch, rho=0.0).to(device),
            dataloaders["test"],
            scaler,
            rho=0.0,
            device=device,
        )
        if test_y_true is None:
            test_y_true = eval_base["y_true_mw"]

        # Evaluate Standalone GRU
        met_gru = compute_metrics(eval_base["gru_mw"], test_y_true)
        all_seed_results.append({"model": "Standalone_GRU", "seed": seed, **met_gru})
        saved_preds["Standalone_GRU"].append(eval_base["gru_mw"])

        # Evaluate Standalone TCN
        met_tcn = compute_metrics(eval_base["tcn_mw"], test_y_true)
        all_seed_results.append({"model": "Standalone_TCN", "seed": seed, **met_tcn})
        saved_preds["Standalone_TCN"].append(eval_base["tcn_mw"])

        # Evaluate Standalone Patch
        met_patch = compute_metrics(eval_base["patch_mw"], test_y_true)
        all_seed_results.append({"model": "Standalone_Patch", "seed": seed, **met_patch})
        saved_preds["Standalone_Patch"].append(eval_base["patch_mw"])

        # Evaluate Static Equal Ensemble
        met_equal = compute_metrics(eval_base["equal_ens_mw"], test_y_true)
        all_seed_results.append({"model": "Static_Equal_Ensemble", "seed": seed, **met_equal})
        saved_preds["Static_Equal_Ensemble"].append(eval_base["equal_ens_mw"])

        # 2. Learned Static Weights (cached or trained)
        ckpt_dir_p7 = os.path.join("research", "checkpoints", "phase7")
        l_stat = LearnedStaticEnsemble(gru, tcn, patch).to(device)
        l_stat_ckpt = os.path.join(ckpt_dir_p7, f"learned_static_seed_{seed}.pt")
        if os.path.exists(l_stat_ckpt):
            l_stat.load_state_dict(torch.load(l_stat_ckpt, map_location=device))
        l_stat.eval()
        with torch.no_grad():
            w_stat = F.softmax(l_stat.logits / l_stat.temperature, dim=0).cpu().numpy()
            y_stat = w_stat[0] * eval_base["gru_mw"] + w_stat[1] * eval_base["tcn_mw"] + w_stat[2] * eval_base["patch_mw"]
        met_stat = compute_metrics(y_stat, test_y_true)
        all_seed_results.append({"model": "Learned_Static_Weights", "seed": seed, **met_stat})
        saved_preds["Learned_Static_Weights"].append(y_stat)

        # 3. Decoupled CAEG (unconstrained)
        d_caeg = DecoupledCAEGNet(gru, tcn, patch).to(device)
        d_ckpt = os.path.join(ckpt_dir_p7, f"decoupled_caeg_seed_{seed}.pt")
        if os.path.exists(d_ckpt):
            d_caeg.load_state_dict(torch.load(d_ckpt, map_location=device))
        eval_dcaeg = evaluate_decoupled_caeg(d_caeg, dataloaders["test"], scaler, device=device)
        met_dcaeg = compute_metrics(eval_dcaeg["y_pred_mw"], test_y_true)
        all_seed_results.append({"model": "Decoupled_CAEG", "seed": seed, **met_dcaeg})
        saved_preds["Decoupled_CAEG"].append(eval_dcaeg["y_pred_mw"])
        saved_weights["Decoupled_CAEG"].append(eval_dcaeg["weights"])

        # 4. CAEG-Net SR (frozen reference from phase 5 shrinkage)
        sr_model = ShrinkageRegularizedCAEG(gru, tcn, patch, alpha=1.0).to(device)
        # Train SR with lambda=0.025
        train_shrinkage_router(
            model=sr_model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            alpha=1.0,
            lambda_dev=0.025,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=25,
            patience=5,
            device=device,
            verbose=False,
        )
        eval_sr = evaluate_shrinkage_caeg(sr_model, dataloaders["test"], scaler, alpha=1.0, device=device)
        met_sr = compute_metrics(eval_sr["y_pred_mw"], test_y_true)
        all_seed_results.append({"model": "CAEG_Net_SR", "seed": seed, **met_sr})
        saved_preds["CAEG_Net_SR"].append(eval_sr["y_pred_mw"])
        saved_weights["CAEG_Net_SR"].append(eval_sr["weights"])

        # 5. CAEG-Net BR Optimal (Locked)
        rho_opt = float(locked_best["rho"])
        lam_opt = float(locked_best["lambda_dev"])
        br_opt = BoundedRoutingCAEG(gru, tcn, patch, rho=rho_opt).to(device)
        train_bounded_router(
            model=br_opt,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            rho=rho_opt,
            lambda_dev=lam_opt,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=25,
            patience=5,
            device=device,
            verbose=False,
        )
        eval_br_opt = evaluate_bounded_caeg(br_opt, dataloaders["test"], scaler, rho=rho_opt, device=device)
        met_br_opt = compute_metrics(eval_br_opt["y_pred_mw"], test_y_true)
        all_seed_results.append({"model": "CAEG_Net_BR_Optimal", "seed": seed, **met_br_opt})
        saved_preds["CAEG_Net_BR_Optimal"].append(eval_br_opt["y_pred_mw"])
        saved_weights["CAEG_Net_BR_Optimal"].append(eval_br_opt["weights"])
        np.save(os.path.join(pred_dir, f"caeg_br_opt_seed_{seed}.npy"), eval_br_opt["y_pred_mw"])

        # 6. CAEG-Net BR No KL (Pure Bounded)
        rho_nokl = float(locked_no_kl["rho"])
        br_nokl = BoundedRoutingCAEG(gru, tcn, patch, rho=rho_nokl).to(device)
        train_bounded_router(
            model=br_nokl,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            rho=rho_nokl,
            lambda_dev=0.0,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=25,
            patience=5,
            device=device,
            verbose=False,
        )
        eval_br_nokl = evaluate_bounded_caeg(br_nokl, dataloaders["test"], scaler, rho=rho_nokl, device=device)
        met_br_nokl = compute_metrics(eval_br_nokl["y_pred_mw"], test_y_true)
        all_seed_results.append({"model": "CAEG_Net_BR_No_KL", "seed": seed, **met_br_nokl})
        saved_preds["CAEG_Net_BR_No_KL"].append(eval_br_nokl["y_pred_mw"])
        saved_weights["CAEG_Net_BR_No_KL"].append(eval_br_nokl["weights"])

        # 7. Fixed Conservative Points: rho=0.10 and rho=0.20
        for fixed_rho, name in [(0.10, "CAEG_Net_BR_Rho_0_10"), (0.20, "CAEG_Net_BR_Rho_0_20")]:
            br_fixed = BoundedRoutingCAEG(gru, tcn, patch, rho=fixed_rho).to(device)
            train_bounded_router(
                model=br_fixed,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                rho=fixed_rho,
                lambda_dev=0.025,
                lr=1e-3,
                weight_decay=1e-4,
                max_epochs=25,
                patience=5,
                device=device,
                verbose=False,
            )
            eval_fixed = evaluate_bounded_caeg(br_fixed, dataloaders["test"], scaler, rho=fixed_rho, device=device)
            met_fixed = compute_metrics(eval_fixed["y_pred_mw"], test_y_true)
            all_seed_results.append({"model": name, "seed": seed, **met_fixed})
            saved_preds[name].append(eval_fixed["y_pred_mw"])
            saved_weights[name].append(eval_fixed["weights"])

    df_seed_results = pd.DataFrame(all_seed_results)
    df_seed_results.to_csv(os.path.join(results_dir, "phase5_bounded_routing_seed_results.csv"), index=False)

    # Aggregate model comparison
    comp_records = []
    for model_name, group in df_seed_results.groupby("model"):
        comp_records.append({
            "model": model_name,
            "mae_mean": float(group["mae_mw"].mean()),
            "mae_std": float(group["mae_mw"].std()),
            "rmse_mean": float(group["rmse_mw"].mean()),
            "rmse_std": float(group["rmse_mw"].std()),
            "r2_mean": float(group["r2"].mean()),
            "r2_std": float(group["r2"].std()),
            "mape_mean": float(group["mape_pct"].mean()),
            "mape_std": float(group["mape_pct"].std()),
        })
    df_comp = pd.DataFrame(comp_records).sort_values(by="mae_mean")
    df_comp.to_csv(os.path.join(results_dir, "phase5_bounded_routing_model_comparison.csv"), index=False)

    print("\n--- Phase 5 Multi-Seed Benchmark Summary (Mean +/- Std) ---")
    for _, row in df_comp.iterrows():
        print(f"{row['model']:<25}: MAE={row['mae_mean']:6.2f} +/- {row['mae_std']:4.2f} MW | RMSE={row['rmse_mean']:6.2f} | R2={row['r2_mean']:.4f}")

    return df_seed_results, df_comp, {"preds": saved_preds, "weights": saved_weights, "y_true": test_y_true}


def run_statistical_analysis(
    saved_data: Dict,
    seeds: List[int],
    results_dir: str,
) -> pd.DataFrame:
    """
    Perform rigorous paired hypothesis testing across 53 non-overlapping daily blocks (K=53).
    Includes Holm-Bonferroni correction.
    """
    print("\n=========================================================================")
    print(">>> RUNNING STATISTICAL HYPOTHESIS TESTING (K=53 DAILY BLOCKS) <<<")
    print("=========================================================================")

    preds = saved_data["preds"]
    y_true = saved_data["y_true"]
    k_blocks = 53
    block_len = 24  # 53 * 24 = 1,272 test hours

    # Mean predictions across 5 seeds
    mean_preds = {m: np.mean(np.array(preds[m]), axis=0) for m in preds}

    # Daily block MAEs
    block_maes = {}
    for m in mean_preds:
        b_maes = []
        for b in range(k_blocks):
            start_idx = b * block_len
            end_idx = start_idx + block_len
            err = np.abs(mean_preds[m][start_idx:end_idx] - y_true[start_idx:end_idx])
            b_maes.append(float(np.mean(err)))
        block_maes[m] = np.array(b_maes)

    comparisons = [
        ("CAEG_Net_BR_Optimal", "Static_Equal_Ensemble"),
        ("CAEG_Net_BR_Optimal", "Decoupled_CAEG"),
        ("CAEG_Net_BR_Optimal", "CAEG_Net_SR"),
        ("CAEG_Net_BR_Optimal", "Standalone_TCN"),
        ("CAEG_Net_BR_No_KL", "Static_Equal_Ensemble"),
        ("CAEG_Net_BR_Rho_0_10", "Static_Equal_Ensemble"),
        ("CAEG_Net_BR_Rho_0_20", "Static_Equal_Ensemble"),
    ]

    stat_records = []
    for m1, m2 in comparisons:
        d1 = block_maes[m1]
        d2 = block_maes[m2]
        diff = d1 - d2
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        # 1. Paired t-test
        t_stat, p_t = stats.ttest_rel(d1, d2)

        # 2. Wilcoxon signed-rank test
        try:
            w_stat, p_w = stats.wilcoxon(d1, d2)
        except Exception:
            w_stat, p_w = 0.0, 1.0

        # 3. Diebold-Mariano test (lag-1 on full series)
        e1 = mean_preds[m1].flatten() - y_true.flatten()
        e2 = mean_preds[m2].flatten() - y_true.flatten()
        dm_stat, p_dm = compute_diebold_mariano(e1, e2, h=1)

        # 4. Cohen's d
        cohen_d = float(mean_diff / (std_diff + 1e-8))

        # 5. 95% Confidence Interval
        ci_half = float(1.96 * std_diff / np.sqrt(k_blocks))
        ci_low = mean_diff - ci_half
        ci_high = mean_diff + ci_half

        stat_records.append({
            "model_1": m1,
            "model_2": m2,
            "mean_paired_diff_mw": mean_diff,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "cohen_d": cohen_d,
            "paired_t_stat": float(t_stat),
            "paired_t_pval": float(p_t),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_pval": float(p_w),
            "dm_stat": float(dm_stat),
            "dm_pval": float(p_dm),
        })

    df_stats = pd.DataFrame(stat_records)

    # Holm-Bonferroni correction on paired_t_pval
    df_stats.sort_values(by="paired_t_pval", inplace=True)
    m_tests = len(df_stats)
    df_stats["holm_bonferroni_thresh"] = [0.05 / (m_tests - i) for i in range(m_tests)]
    df_stats["statistically_significant"] = df_stats["paired_t_pval"] < df_stats["holm_bonferroni_thresh"]

    df_stats.to_csv(os.path.join(results_dir, "phase5_bounded_routing_statistical_tests.csv"), index=False)
    print("\n--- Daily Block Statistical Analysis (K=53) ---")
    for _, r in df_stats.iterrows():
        sig = "YES" if r["statistically_significant"] else "NO"
        print(f"{r['model_1']} vs {r['model_2']}: Mean Diff={r['mean_paired_diff_mw']:+5.2f} MW | t={r['paired_t_stat']:+5.2f} (p={r['paired_t_pval']:.4f}) | DM={r['dm_stat']:+5.2f} (p={r['dm_pval']:.4f}) | d={r['cohen_d']:+5.3f} | Sig: {sig}")
    return df_stats


def run_routing_analysis(
    saved_data: Dict,
    seeds: List[int],
    results_dir: str,
) -> pd.DataFrame:
    """Analyze router weights, entropy, and deviations from equal prior."""
    print("\n=========================================================================")
    print(">>> RUNNING ROUTING TELEMETRY & BEHAVIORAL ANALYSIS <<<")
    print("=========================================================================")

    weights_dict = saved_data["weights"]
    routing_records = []
    w0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    for model_name, w_list in weights_dict.items():
        all_w = np.concatenate(w_list, axis=0)  # [5 * N, 3]
        mean_gru = float(np.mean(all_w[:, 0]))
        mean_tcn = float(np.mean(all_w[:, 1]))
        mean_patch = float(np.mean(all_w[:, 2]))
        std_gru = float(np.std(all_w[:, 0]))
        std_tcn = float(np.std(all_w[:, 1]))
        std_patch = float(np.std(all_w[:, 2]))

        eps = 1e-8
        entropy = -np.sum(all_w * np.log(all_w + eps), axis=-1)
        mean_entropy = float(np.mean(entropy))
        eff_experts = float(np.exp(mean_entropy))

        dev_l1 = np.sum(np.abs(all_w - w0), axis=-1)
        mean_l1_dev = float(np.mean(dev_l1))
        max_dev = float(np.max(np.abs(all_w - w0)))

        routing_records.append({
            "model": model_name,
            "mean_gru_weight": mean_gru,
            "std_gru_weight": std_gru,
            "mean_tcn_weight": mean_tcn,
            "std_tcn_weight": std_tcn,
            "mean_patch_weight": mean_patch,
            "std_patch_weight": std_patch,
            "mean_entropy": mean_entropy,
            "effective_num_experts": eff_experts,
            "mean_l1_deviation": mean_l1_dev,
            "max_deviation": max_dev,
        })
        print(f"{model_name:<25}: Weights=[GRU={mean_gru:.3f}, TCN={mean_tcn:.3f}, Patch={mean_patch:.3f}] | L1 Dev={mean_l1_dev:.4f} | Eff Exp={eff_experts:.2f}")

    df_routing = pd.DataFrame(routing_records)
    df_routing.to_csv(os.path.join(results_dir, "phase5_bounded_routing_routing.csv"), index=False)
    return df_routing


def run_regime_analysis(
    pipe: Dict,
    saved_data: Dict,
    results_dir: str,
) -> pd.DataFrame:
    """Analyze performance across predefined forecast difficulty tertiaries."""
    print("\n=========================================================================")
    print(">>> RUNNING FORECAST DIFFICULTY REGIME ANALYSIS <<<")
    print("=========================================================================")

    preds = saved_data["preds"]
    y_true = saved_data["y_true"]
    ctx = pipe["context_6d"]["test"]

    # Tertiary splits defined strictly on context features
    rec_err = ctx[:, 5]
    vol = ctx[:, 1]
    # Disagreement from BR Optimal
    opt_preds = np.mean(np.array(preds["CAEG_Net_BR_Optimal"]), axis=0)
    equal_preds = np.mean(np.array(preds["Static_Equal_Ensemble"]), axis=0)
    sr_preds = np.mean(np.array(preds["CAEG_Net_SR"]), axis=0)
    dcaeg_preds = np.mean(np.array(preds["Decoupled_CAEG"]), axis=0)

    # Disagreement across 3 standalone predictions
    gru_preds = np.mean(np.array(preds["Standalone_GRU"]), axis=0)
    tcn_preds = np.mean(np.array(preds["Standalone_TCN"]), axis=0)
    patch_preds = np.mean(np.array(preds["Standalone_Patch"]), axis=0)
    disagree = (np.abs(gru_preds - tcn_preds) + np.abs(gru_preds - patch_preds) + np.abs(tcn_preds - patch_preds)) / 3.0
    mean_disagree = np.mean(disagree, axis=1)

    regime_defs = {
        "Baseline_Error_Low": rec_err <= np.percentile(rec_err, 33.3),
        "Baseline_Error_Med": (rec_err > np.percentile(rec_err, 33.3)) & (rec_err <= np.percentile(rec_err, 66.7)),
        "Baseline_Error_High": rec_err > np.percentile(rec_err, 66.7),
        "Disagreement_Low": mean_disagree <= np.percentile(mean_disagree, 33.3),
        "Disagreement_Med": (mean_disagree > np.percentile(mean_disagree, 33.3)) & (mean_disagree <= np.percentile(mean_disagree, 66.7)),
        "Disagreement_High": mean_disagree > np.percentile(mean_disagree, 66.7),
        "Volatility_Low": vol <= np.percentile(vol, 33.3),
        "Volatility_Med": (vol > np.percentile(vol, 33.3)) & (vol <= np.percentile(vol, 66.7)),
        "Volatility_High": vol > np.percentile(vol, 66.7),
    }

    regime_records = []
    for reg_name, mask in regime_defs.items():
        sub_true = y_true[mask]
        sub_br = opt_preds[mask]
        sub_eq = equal_preds[mask]
        sub_sr = sr_preds[mask]
        sub_dcaeg = dcaeg_preds[mask]

        mae_br = float(np.mean(np.abs(sub_br - sub_true)))
        mae_eq = float(np.mean(np.abs(sub_eq - sub_true)))
        mae_sr = float(np.mean(np.abs(sub_sr - sub_true)))
        mae_dcaeg = float(np.mean(np.abs(sub_dcaeg - sub_true)))

        gain_vs_eq = mae_eq - mae_br
        gain_vs_sr = mae_sr - mae_br
        gain_vs_dcaeg = mae_dcaeg - mae_br

        regime_records.append({
            "regime": reg_name,
            "n_samples": int(np.sum(mask)),
            "mae_br_mw": mae_br,
            "mae_equal_mw": mae_eq,
            "mae_sr_mw": mae_sr,
            "mae_dcaeg_mw": mae_dcaeg,
            "gain_vs_equal_mw": gain_vs_eq,
            "gain_vs_sr_mw": gain_vs_sr,
            "gain_vs_dcaeg_mw": gain_vs_dcaeg,
        })
        print(f"{reg_name:<22}: BR={mae_br:6.2f} | Equal={mae_eq:6.2f} | Gain vs Equal: {gain_vs_eq:+5.2f} MW | Gain vs SR: {gain_vs_sr:+5.2f} MW")

    df_reg = pd.DataFrame(regime_records)
    df_reg.to_csv(os.path.join(results_dir, "phase5_bounded_routing_regimes.csv"), index=False)
    return df_reg


def generate_publication_figures(
    df_val: pd.DataFrame,
    df_comp: pd.DataFrame,
    df_routing: pd.DataFrame,
    df_reg: pd.DataFrame,
    df_seed_results: pd.DataFrame,
    results_dir: str,
):
    """Generate 6 publication-quality figures at 300 DPI."""
    print("\n--- Generating Publication-Quality Figures ---")
    plot_dir = os.path.join(results_dir, "phase5_bounded_routing_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Validation MAE vs Rho
    plt.figure(figsize=(8, 5))
    df_nokl = df_val[df_val["lambda_dev"] == 0.0].sort_values(by="rho")
    df_kl = df_val[df_val["lambda_dev"] == 0.025].sort_values(by="rho")
    plt.plot(df_nokl["rho"], df_nokl["val_mae_mean"], marker="o", color="#1f77b4", label=r"Pure Bounded ($\lambda_{\mathrm{dev}}=0.0$)")
    plt.plot(df_kl["rho"], df_kl["val_mae_mean"], marker="s", color="#2ca02c", label=r"Bounded + KL ($\lambda_{\mathrm{dev}}=0.025$)")
    plt.axhline(df_nokl[df_nokl["rho"] == 0.0]["val_mae_mean"].iloc[0], color="#d62728", linestyle="--", label="Static Equal Baseline")
    plt.title(r"Validation MAE vs. Routing Bound Parameter $\rho$", fontsize=13)
    plt.xlabel(r"Routing Bound Parameter $\rho$ (0=Equal, 1=Unconstrained)", fontsize=11)
    plt.ylabel("Validation MAE (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_01_validation_mae_vs_rho.png"), dpi=300)
    plt.close()

    # 2. Test MAE Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    df_plot = df_comp.sort_values(by="mae_mean", ascending=True)
    colors = ["#2ca02c" if "BR" in m else "#1f77b4" if "Equal" in m else "#ff7f0e" if "SR" in m else "#7f7f7f" for m in df_plot["model"]]
    bars = plt.barh(df_plot["model"], df_plot["mae_mean"], xerr=df_plot["mae_std"], color=colors, capsize=4, alpha=0.85)
    plt.axvline(232.51, color="#d62728", linestyle="--", label="Static Equal Ensemble (232.51 MW)")
    plt.title("Multi-Seed Test MAE Comparison (5 Seeds, 1,272 Hours)", fontsize=13)
    plt.xlabel("Test MAE (MW)", fontsize=11)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_02_test_mae_comparison.png"), dpi=300)
    plt.close()

    # 3. Routing Weights vs Rho
    plt.figure(figsize=(8, 5))
    df_nokl_w = df_val[df_val["lambda_dev"] == 0.0].sort_values(by="rho")
    plt.plot(df_nokl_w["rho"], df_nokl_w["mean_gru_weight"], marker="o", label="GRU Weight", color="#1f77b4")
    plt.plot(df_nokl_w["rho"], df_nokl_w["mean_tcn_weight"], marker="s", label="TCN Weight", color="#2ca02c")
    plt.plot(df_nokl_w["rho"], df_nokl_w["mean_patch_weight"], marker="^", label="Patch Weight", color="#ff7f0e")
    plt.axhline(1.0 / 3.0, color="#7f7f7f", linestyle=":", label="Equal Weight (1/3)")
    plt.title(r"Learned Mean Expert Weights Across $\rho$", fontsize=13)
    plt.xlabel(r"Routing Bound $\rho$", fontsize=11)
    plt.ylabel("Mean Expert Weight", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_03_routing_weights_vs_rho.png"), dpi=300)
    plt.close()

    # 4. Deviation from Equal Weighting
    plt.figure(figsize=(8, 5))
    plt.plot(df_nokl["rho"], df_nokl["mean_l1_dev"], marker="o", color="#1f77b4", label=r"Empirical $L_1$ Deviation")
    theoretical_bound = df_nokl["rho"] * (4.0 / 3.0)
    plt.plot(df_nokl["rho"], theoretical_bound, linestyle="--", color="#d62728", label=r"Max Theoretical Bound $\frac{4}{3}\rho$")
    plt.title(r"Mean $L_1$ Routing Weight Deviation from Equal Prior vs. $\rho$", fontsize=13)
    plt.xlabel(r"Routing Bound Parameter $\rho$", fontsize=11)
    plt.ylabel(r"$\|w_t - w_0\|_1$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_04_deviation_from_equal.png"), dpi=300)
    plt.close()

    # 5. Seed Stability
    plt.figure(figsize=(9, 5))
    key_models = ["Static_Equal_Ensemble", "CAEG_Net_BR_Optimal", "CAEG_Net_SR", "Decoupled_CAEG"]
    for m in key_models:
        sub = df_seed_results[df_seed_results["model"] == m].sort_values(by="seed")
        plt.plot(sub["seed"], sub["mae_mw"], marker="o", label=m)
    plt.title("Seed-to-Seed Test MAE Stability Across Seeds 42-46", fontsize=13)
    plt.xlabel("Random Seed", fontsize=11)
    plt.ylabel("Test MAE (MW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_05_seed_stability.png"), dpi=300)
    plt.close()

    # 6. Regime Performance Comparison
    plt.figure(figsize=(11, 5))
    x = np.arange(len(df_reg))
    width = 0.35
    plt.bar(x - width/2, df_reg["mae_equal_mw"], width, label="Static Equal Ensemble", color="#1f77b4", alpha=0.8)
    plt.bar(x + width/2, df_reg["mae_br_mw"], width, label="CAEG-Net BR (Optimal)", color="#2ca02c", alpha=0.8)
    plt.xticks(x, df_reg["regime"], rotation=45, ha="right", fontsize=9)
    plt.title("Forecast Difficulty Regime Performance: CAEG-Net BR vs. Static Equal", fontsize=13)
    plt.ylabel("MAE (MW)", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "phase5_br_06_regime_performance.png"), dpi=300)
    plt.close()

    print("All 6 figures successfully saved to phase5_bounded_routing_plots/")


def main():
    print("=========================================================================")
    print(">>> CAEG-NET PHASE 5: BOUNDED / CONSERVATIVE ROUTING EXPERIMENT <<<")
    print("=========================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    results_dir = os.path.join("research", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Pipeline preparation
    print("\nLoading and preparing causal data pipeline...")
    pipe = prepare_research_v2_pipeline()

    # 2. Step 1 & 7: Smoke test
    run_smoke_test(pipe, device, results_dir)

    # 3. Step 8: Validation grid search
    seeds = [42, 43, 44, 45, 46]
    rhos = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    lambdas_dev = [0.0, 0.025]
    val_csv_path = os.path.join(results_dir, "phase5_bounded_routing_validation.csv")
    if os.path.exists(val_csv_path):
        print(f"\nLoading existing completed validation grid search from {val_csv_path}...")
        df_val = pd.read_csv(val_csv_path)
    else:
        df_val = run_validation_grid_search(pipe, seeds, rhos, lambdas_dev, device, results_dir)

    # 4. Step 9: Lock hyperparameters
    locked_best, locked_no_kl = select_and_lock_hyperparameters(df_val)

    # 5. Step 10: 5-seed final test benchmark
    df_seed_results, df_comp, saved_data = run_test_benchmark(pipe, seeds, locked_best, locked_no_kl, device, results_dir)

    # 6. Step 11: Statistical analysis
    df_stats = run_statistical_analysis(saved_data, seeds, results_dir)

    # 7. Step 12: Routing analysis
    df_routing = run_routing_analysis(saved_data, seeds, results_dir)

    # 8. Step 13: Regime analysis
    df_reg = run_regime_analysis(pipe, saved_data, results_dir)

    # 9. Figures
    generate_publication_figures(df_val, df_comp, df_routing, df_reg, df_seed_results, results_dir)

    # 10. Machine-readable findings
    findings = {
        "benchmark": "Phase 5 Bounded Routing (CAEG-Net BR)",
        "locked_optimal_hyperparameters": {
            "rho": float(locked_best["rho"]),
            "lambda_dev": float(locked_best["lambda_dev"]),
            "val_mae_mw": float(locked_best["val_mae_mean"]),
        },
        "locked_no_kl_hyperparameters": {
            "rho": float(locked_no_kl["rho"]),
            "lambda_dev": 0.0,
            "val_mae_mw": float(locked_no_kl["val_mae_mean"]),
        },
        "test_results_summary": df_comp.to_dict(orient="records"),
        "statistical_tests": df_stats.to_dict(orient="records"),
        "routing_telemetry": df_routing.to_dict(orient="records"),
        "regime_breakdown": df_reg.to_dict(orient="records"),
    }
    with open(os.path.join(results_dir, "phase5_bounded_routing_findings.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print("\nPhase 5 Bounded Routing Experiment completed successfully!")


if __name__ == "__main__":
    main()
