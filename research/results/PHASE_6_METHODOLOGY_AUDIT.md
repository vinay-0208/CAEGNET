# CAEG-Net Phase 6: MAE-Primary Methodology Audit & Checkpoint Selection Reconciliation
**Document**: `research/results/PHASE_6_METHODOLOGY_AUDIT.md`  
**Track**: `research-track` | **Phase**: Phase 6 — Five-Seed Finalist Evaluation  
**Roadmap**: 14-Phase Original Research Roadmap  
**Date**: September 2026  
**Status**: AUDIT COMPLETE — METHODOLOGICALLY ACCEPTABLE & VERIFIED  

---

## 1. Executive Summary

This audit formally investigates and documents the distinction between:
1. **Primary Research and Evaluation Metric**: Mean Absolute Error (**MAE** in Megawatts, MW).
2. **Training Objective Loss Function**: Standardized Mean Squared Error (**MSE**).
3. **Model Checkpoint Selection Criterion**: Standardized **Validation MSE**.

Following an exhaustive inspection of the actual Phase 6 implementation (`research/experiments/run_phase6_finalist_evaluation.py` and `train.py`), this audit verifies that the experimental execution is methodologically sound, strictly leak-free, mathematically aligned, and completely consistent across all evaluated models and seeds.

**Audit Classification**:
> **"METHODOLOGICALLY ACCEPTABLE, BUT MAE IS THE PRIMARY RESEARCH/EVALUATION METRIC WHILE MSE IS THE TRAINING/CHECKPOINT OBJECTIVE."**

Because the execution is leak-free, consistent across all competing models, and adheres strictly to standard machine learning forecasting practice (where smooth $L_2$ objectives are minimized via gradient descent while $L_1$ metrics assess practical day-ahead operational accuracy), **Phase 6 results are fully valid and do NOT require a rerun.**

---

## 2. Granular Code Audit: Seven Core Verification Questions

We directly inspected `research/experiments/run_phase6_finalist_evaluation.py` and the supporting data pipeline modules:

### Question A: What loss is optimized during training?
- **Audited Code** (`run_phase6_finalist_evaluation.py` lines 96-98):
  ```python
  loss = F.mse_loss(out, y)
  loss.backward()
  optimizer.step()
  ```
- **Finding**: Standardized **Mean Squared Error (MSE)** is the training loss function minimized via AdamW.

### Question B: What metric is used for checkpoint selection?
- **Audited Code** (`run_phase6_finalist_evaluation.py` lines 112-124):
  ```python
  val_mse = F.mse_loss(y_pred, y)
  total_val_loss += val_mse.item()
  ...
  val_loss = total_val_loss / max(n_val_batches, 1)
  ...
  if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch
      best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
  ```
- **Finding**: Checkpoints are saved and restored based on the minimum **Validation MSE** on standardized targets.

### Question C: What metric is used for final model comparison?
- **Audited Code** (`run_phase6_finalist_evaluation.py` lines 170-176, 470-520, 530-580):
  ```python
  preds_mw = preds_scaled * float(scaler.scale_[0]) + float(scaler.mean_[0])
  met = compute_metrics(y_test_mw, preds_mw)  # Primary: met["MAE"]
  ```
  And for statistical significance testing on $K=53$ daily blocks:
  ```python
  daily_errors[m] = np.array([
      np.mean(np.abs(mean_preds[m][orig] - y_test_mw[orig]))
      for orig in daily_origins
  ])
  d = err1 - err2  # Daily MAE differential sequence
  ```
- **Finding**: **MAE (in MW)** is the primary metric for all model rankings, percentage gains, multi-seed comparison tables, and inferential hypothesis tests. Secondary metrics (RMSE, MSE, $R^2$, MAPE) provide supporting characterization.

### Question D: Is test-set information used anywhere during checkpoint selection?
- **Audited Code**: `train_model_canonical` receives strictly `train_loader` and `val_loader`. The test data loader (`test_loader`) is only referenced after training is completely terminated and `model.load_state_dict(best_weights)` has restored the best validation checkpoint.
- **Finding**: **Zero test leakage.** The test partition is strictly isolated and frozen from training and checkpoint selection.

### Question E: Is validation information used correctly?
- **Audited Code**: Validation passes strictly execute under `model.eval()` and `torch.no_grad()`. No gradients are backpropagated from the validation set into any model parameters. Validation loss is used exclusively for learning rate scheduling (`StepLR`) and early stopping / checkpoint retention.
- **Finding**: **Validation information is handled with strict methodological integrity.**

### Question F: Are all five seeds treated identically?
- **Audited Code**: All five seeds (`[42, 123, 999, 2024, 3407]`) execute identical loops with deterministic seed setting (`torch.manual_seed`, `np.random.seed`, `torch.cuda.manual_seed_all`), identical hyperparameters (`lr=1e-3, weight_decay=1e-4, max_epochs=45, patience=7`), and identical evaluation functions.
- **Finding**: **All seeds are treated identically.**

### Question G: Is the implementation consistent across all models?
- **Audited Code**:
  - `Original_CAEGNet_V1`: trained with MSE loss, selected on Val MSE, evaluated on MAE.
  - `Bounded_CAEGNet_rho0.50`: trained with MSE loss, selected on Val MSE, evaluated on MAE.
  - `Standalone_LSTM`: trained with MSE loss, selected on Val MSE, evaluated on MAE.
  - `Standalone_TCN`: trained with MSE loss, selected on Val MSE, evaluated on MAE.
  - `Standalone_CNN`: trained with MSE loss, selected on Val MSE, evaluated on MAE.
  - `Static_Equal_Ensemble`: arithmetic combination of the identical standalone predictions: $\hat{y}_{	ext{equal}} = (\hat{y}_{	ext{LSTM}} + \hat{y}_{	ext{TCN}} + \hat{y}_{	ext{CNN}})/3$.
- **Finding**: **100% implementation consistency across all six finalists.**

---

## 3. Methodological Assessment: Is a Rerun Necessary?

### Why a Rerun is NOT Required
1. **Consistency**: Checkpoint selection criteria were identical across every competing model and seed. No model received an unfair advantage from a divergent selection protocol.
2. **Zero Information Leakage**: No test data contaminated the selection process.
3. **Standard Machine Learning Norm**: Minimizing MSE (which yields the conditional mean $\mathbb{E}[Y|X]$) via smooth gradient optimization while evaluating performance via MAE (which reflects $L_1$ operational costs in electric grid dispatch) is a standard, mathematically coherent methodology in time series forecasting literature.
4. **Research Integrity**: Rerunning the experiment with validation MAE checkpoint selection merely to alter the numbers would constitute retrospective metric engineering. Preserving the frozen Phase 6 results upholds scientific transparency.

**Verdict**: The existing Phase 6 test results remain **fully valid, locked, and preserved**.

---

## 4. Formal Metric Hierarchy & Disclosures

To prevent any ambiguity in subsequent academic reporting:

```
=============================================================================
CAEG-NET EVALUATION METRIC HIERARCHY
=============================================================================
PRIMARY RESEARCH & EVALUATION METRIC:
    MAE (MW) — Primary metric for model comparison, ranking, and hypothesis testing.

SECONDARY RESEARCH METRICS:
    RMSE (MW) — Penalizes large peak errors.
    MSE (MW^2) — Supporting variance metric.
    MAPE (%)   — Scale-independent percentage accuracy.
    R^2        — Proportion of total load variance explained.

TRAINING OBJECTIVE LOSS:
    Standardized MSE (smooth L2 gradient optimization).

PHASE 6 CHECKPOINT SELECTION CRITERION:
    Standardized Validation MSE (frozen Phase 6 protocol).
=============================================================================
```

### Required Paper Language
All publications and reports must state:
> *"MAE (in MW) is the primary research and evaluation metric used to assess operational forecasting accuracy. Model training minimizes standardized MSE to ensure smooth gradient convergence, and Phase 6 model checkpoints were selected using validation MSE under the locked evaluation protocol. This distinction is intentional: gradient-based optimization targets the conditional expectation under an $L_2$ objective, while practical utility is evaluated primarily via $L_1$ dispatch error (MAE)."*

---

## 5. Frozen Protocol for Future Phases (Phase 7 Onward)

For subsequent phases (Phase 7 Strong Baselines, Phase 8 GEFCom2014, Phase 9 Dataset 3, Phase 10 Multi-Horizon, etc.):

1. **Pre-Experiment Specification**: The validation checkpoint selection metric (whether Validation MAE or Validation MSE) must be explicitly declared and locked *prior* to running the experiment.
2. **Universal Consistency**: The selected criterion must be applied identically across all models, baselines, and seeds within that experiment.
3. **No Retrospective Switching**: Checkpoint criteria must never be altered retrospectively after observing test performance.
4. **Primary Research Focus**: Test MAE remains the primary research metric for discussing results, plotting comparative figures, and conducting statistical significance tests.
5. **Locked Test Partition**: The test partition remains strictly frozen from model development.

---

## 6. Audit Sign-Off

- **Audit Status**: **PASSED**
- **Existing Phase 6 Results**: **CONFIRMED VALID & PRESERVED**
- **Code Consistency**: **VERIFIED**
- **Test Set Status**: **FROZEN FROM FURTHER DEVELOPMENT**
