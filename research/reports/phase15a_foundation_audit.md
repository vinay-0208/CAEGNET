# Phase 15A — Foundation & Artifact Audit Report
**CAEG-Net: Dynamic Routing, Confidence, Expert Complementarity & Oracle-Ceiling Analysis**

**Date:** 2026-09-11  
**Authoritative Reference Model:** Candidate F2 / A2-OOF (`ConfidenceFallbackCAEGNet`, 121,724 parameters)  
**Evaluation Seeds:** {42, 123, 999, 2024, 3407}  
**Datasets:** Modern PJM (MW), GEFCom2014 (kW), UCI Electricity Cohort 320 Aggregate (MW)  
**Authoritative Commits:** `c67067d8` (Phase 14), `1d4d8c35` (Scientific Correction), `cfc75e2b` (Documentation Lock), `cc3afcf` (Phase 15A Manuscript), `b5f3140` (Phase 15B Reviewer Audit)

---

## 1. Executive Summary

Phase 15A initiates the comprehensive diagnostic audit of CAEG-Net before Phase 15B implements controlled architectural refinements. The objective is to rigorously diagnose why the current routing and confidence mechanisms behave as they do, whether expert routing is genuinely dynamic or effectively near-equal, why confidence $\lambda$ settles near a constant $0.51$, what oracle headroom exists, and whether the single global routing vector conceals horizon specialization.

This audit establishes the foundation by inventorying existing artifacts, validating experimental provenance, identifying reusable frozen data, and verifying that all Phase 15A diagnostics can be executed without modifying the frozen canonical architecture or retraining the multi-seed benchmark.

---

## 2. Artifact Inventory & Locations

A comprehensive scan of the repository located the following historical and experimental artifacts:

| Category | File Path | Size / Count | Status & Authoritative Scope |
| :--- | :--- | :--- | :--- |
| **Model Implementation** | `research/models.py`, `caeg_net.py`, `research/original_caeg.py` | 50.2 KB, 10.9 KB | Canonical LSTM (56,152), TCN (36,952), CNN (27,400), Core (120,504), V1 (121,531) |
| **Phase 14 Runner** | `research/experiments/run_phase14_final_optimization.py` | 57.3 KB | Canonical F0-F5 implementation, screening protocol, 5-seed evaluation |
| **Cached OOF Features** | `research/results/phase14_cached_oof_features.pkl` | 1.15 MB | **Authoritative:** Train/Val/Test 4-block chronological OOF errors across tri-benchmarks |
| **Phase 14 Test Metrics** | `research/results/phase14_test_results.csv` | 3.8 KB | **Authoritative:** 5-seed test MAE/RMSE/MSE/R2/MAPE for Candidates F0-F5 |
| **Phase 14 Statistical Tests** | `research/results/phase14_statistical_comparisons.csv` | 7.6 KB | **Authoritative:** Non-overlapping 24h daily-block paired $t$-tests, Wilcoxon, Holm $p$-values |
| **Phase 14 Routing Stats** | `research/results/phase14_routing_statistics.csv` | 19.1 KB | **Authoritative:** Mean/Std weights, entropy, $N_{\mathrm{eff}}$ across all seeds & datasets |
| **Phase 14 Confidence Stats**| `research/results/phase14_confidence_statistics.csv` | 9.5 KB | **Authoritative:** $\lambda$ distribution percentiles, min/max, extreme fractions |
| **Phase 14 Regime Results** | `research/results/phase14_regime_results.csv` | 11.0 KB | **Authoritative:** Volatility & load level performance breakdown for F0-F5 |
| **Phase 11 Window Metrics** | `research/results/phase11_window_metrics.csv` | 8.48 MB | **Authoritative:** 16,160 window-level ground truth, expert errors, weights, and regime flags |
| **Multi-Seed Cache** | `research/results/research_multiseed_cache.npz` | 4.88 MB | PJM test ground truth ($1,294 \times 24$) and 5-seed multi-horizon predictions for V1 |
| **Screening Cache** | `research/results/screening_predictions_cache.npz` | 1.23 MB | PJM test ground truth ($1,294 \times 24$) and 24h predictions for screening variants |
| **Phase 13 Artifacts** | `research/results/phase13_*.csv` | Multiple files | Phase 13 candidate comparison, lambda distributions, and statistical tests |

---

## 3. What is Authoritative & Can Be Reused

1. **Frozen Experimental Benchmarks:**
   - Five-seed test results in `phase14_test_results.csv` are authoritative:
     - Modern PJM: $250.97 \pm 10.69$ MW (F2), $253.41 \pm 9.12$ MW (V1), $279.83$ MW (Equal), $259.33$ MW (TCN).
     - GEFCom2014: $12.41 \pm 0.15$ kW (F2), $12.88 \pm 0.25$ kW (V1), $12.62$ kW (Equal), $12.57$ kW (TCN).
     - UCI Cohort 320: $7.74 \pm 0.30$ MW (F2), $7.94 \pm 0.17$ MW (V1), $8.17$ MW (Equal), $7.55$ MW (LSTM Val BL) / $7.79$ MW (Test BM).
2. **Causal OOF Feature Cache:**
   - `phase14_cached_oof_features.pkl` contains the expanding-window chronological OOF errors generated strictly on historical training blocks without target leakage. These features can be reused directly, eliminating the need for repeating the 4-block surrogate expert training.
3. **Statistical Inference Baselines:**
   - `phase14_statistical_comparisons.csv` provides the authoritative daily-block paired hypothesis tests ($K=53, 456, 163$) and Holm-adjusted $p$-values ($p_{\mathrm{adj}} = 0.0406, 1.02 \times 10^{-27}, 0.0017$).
4. **Historical Window-Level Dynamics:**
   - `phase11_window_metrics.csv` provides $16,160$ test windows of ground-truth loads, individual expert predictions, disagreements, and regime classifications across all three datasets.

---

## 4. What Must Be Computed for Phase 15A Diagnostics

While aggregate summaries exist, deep diagnostic answers to specific Phase 15A questions require fine-grained window-by-window and horizon-by-horizon quantities that were not logged to disk in Phase 14:
1. **Window-Level ($\lambda_t, \Delta_t$) Pairs for Diagnostic C (Confidence Calibration):**
   - Phase 14 saved summary percentiles of $\lambda_t$ in `phase14_confidence_statistics.csv`, but not the joint time-series $(\lambda_t, \Delta_t)$ required to evaluate calibration, quantile binning, and monotonicity.
2. **Horizon-Specific Performance for Diagnostic J (Horizon Specialization):**
   - Evaluating whether expert rankings change across forecast horizons $h \in \{1, \dots, 24\}$ requires multi-horizon prediction matrices $\hat{y}_{t, h}$ for each expert (LSTM, TCN, CNN) and F2 across all three datasets.
3. **Oracle Convex Fusion Upper Bound for Diagnostic E (Oracle Ceiling):**
   - Determining the per-window or per-block optimal convex weights $\mathbf{w}^* \in \Delta^2$ minimizing MAE requires window-aligned multi-horizon expert forecasts.
4. **Recent Performance Window Comparisons for Diagnostic K (Resolution Analysis):**
   - Evaluating whether 24h, 48h, 72h, or exponentially smoothed error windows carry differentiated predictive signal requires computing moving error features from lookback windows.

To obtain these fine-grained quantities deterministically without rerunning the full 4-hour Phase 14 suite:
- A targeted diagnostic evaluation under Seed 42 is executed using the cached OOF features in `phase14_cached_oof_features.pkl`.
- Verification test confirmed exact numerical alignment: Seed 42 PJM test MAE $= 249.901$ MW, $\lambda = 0.509220 \pm 0.005576$, matching Phase 14 records to 6 decimal places.

---

## 5. Provenance & Integrity Verification

- **No Historical Commits Altered:** Historical commits `c67067d8`, `1d4d8c35`, `cfc75e2b`, `f5437e1a`, `cc3afcf`, and `b5f3140` remain fully intact in git history.
- **No Test-Driven Model Tuning:** Diagnostic quantities (including oracle selections) are computed strictly post-hoc for diagnostic analysis and are never fed back into model weights, loss functions, or hyperparameter selection.
- **Zero Future Leakage:** Lookback context features use strictly $X_{t-167:t}$. Training-period OOF features strictly use historical blocks.
- **Observational Unit Alignment:** Inferential hypothesis tests and calibration bins respect non-overlapping daily blocks ($K=53, 456, 163$) to eliminate serial pseudoreplication.

---

## 6. Audit Conclusion & Feasibility Certification

**Certification:** The repository possesses complete, uncorrupted experimental provenance. Candidate F2 can be diagnosed thoroughly across all 14 mandated diagnostics without altering the model architecture, without violating data splits, and without invalidating historical records.
