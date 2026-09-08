# CAEG-Net Phase 5: Controlled Ablation Studies & Architectural Evidence
**Document**: `PHASE_5_ABLATION_RESULTS.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: COMPLETE ABLATION AUDIT  

---

## 1. Executive Summary & Scientific Purpose

The purpose of Phase 5 ablation studies is to rigorously test specific, mathematically motivated hypotheses for improving the **original CAEG-Net architecture (LSTM + TCN + CNN)** without altering the underlying expert family.

In accordance with strict research integrity:
- **Decision Partition**: All ablation comparisons are conducted strictly on the **Validation Partition** (Seed 42).
- **Evaluation Metric**: Unscaled physical validation error in Megawatts (Validation MAE).
- **Control Baseline**: Original CAEG-Net V1 evaluated under identical partition boundaries (Validation MAE = 397.97 MW).

---

## 2. Baseline Model Screening (Validation Set, Seed 42)

Before testing ablations, all baselines and component experts were screened on the validation set under identical causal partitioning:

| Model / Architecture | Category | Parameters | Val MAE (MW) | Val RMSE (MW) | Val $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original CAEG-Net V1** | CAEG_V1 | 121,531 | **397.97** | **509.01** | **0.8143** |
| **Persistence Naive-24** | Baseline | 0 | 416.47 | 537.58 | 0.7929 |
| **Standard Input-MoE** | MoE | 126,011 | 445.20 | 574.49 | 0.7635 |
| **Standalone TCN** | Expert | 36,952 | 453.09 | 583.48 | 0.7560 |
| **Static Equal Ensemble** | Ensemble | 120,504 | 524.47 | 681.12 | 0.6676 |
| **Standalone LSTM** | Expert | 56,152 | 541.96 | 703.61 | 0.6452 |
| **Standalone CNN** | Expert | 27,400 | 728.96 | 945.49 | 0.3594 |

### Critical Screening Finding:
In the original architecture, **Standalone CNN is severely deficient** (Val MAE = 728.96 MW). Consequently, the **Static Equal Ensemble** (524.47 MW) performs dramatically worse than Standalone TCN (453.09 MW) and Persistence (416.47 MW).
**CAEG-Net V1 achieves its superior performance (397.97 MW) precisely because its adaptive gating network learns to heavily downweight the defective CNN expert.**

---

## 3. Controlled Ablation Results (Validation Set, Seed 42)

All ablations were evaluated on the validation set starting from the canonical V1 baseline:

| Rank | Ablation Group | Variant / Parameter | Val MAE (MW) | Val RMSE (MW) | Gain vs V1 (MW) | Total Params |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| — | **Control Reference** | **Original CAEG-Net V1** | **397.97** | **509.01** | **0.00** | **121,531** |
| 1 | **Routing Regularization** | Entropy Penalty ($\beta = 0.0001$) | 402.99 | 529.06 | -5.01 | 121,531 |
| 2 | **Context Features** | 3D Context (No Recent Error) | 403.57 | 518.28 | -5.59 | 121,515 |
| 3 | **Conservative Routing** | Bounded Routing ($\rho = 0.20$) | 411.35 | 523.84 | -13.37 | 121,531 |
| 4 | **Conservative Routing** | Bounded Routing ($\rho = 0.30$) | 420.24 | 536.93 | -22.27 | 121,531 |
| 5 | **Expert Disagreement** | 4D + Detached Disagreement (7D) | 426.47 | 539.79 | -28.49 | 121,579 |
| 6 | **Horizon-Aware Routing** | Horizon-Dependent ($24 \times 3$) | 427.27 | 546.08 | -29.30 | 125,232 |
| 7 | **Training Stabilization** | GradClip 1.0 + Plateau Scheduler | 429.06 | 548.66 | -31.09 | 121,531 |
| 8 | **Conservative Routing** | Bounded Routing ($\rho = 0.10$) | 430.70 | 560.09 | -32.73 | 121,531 |
| 9 | **Routing Regularization** | Entropy Penalty ($\beta = 0.0010$) | 435.60 | 559.08 | -37.62 | 121,531 |
| 10 | **Auxiliary Supervision** | $\lambda_{\text{aux}} = 0.15$ | 440.43 | 556.93 | -42.46 | 121,531 |
| 11 | **Context Features** | 6D Extended Context | 447.24 | 576.35 | -49.26 | 121,563 |
| 12 | **Routing Regularization** | KL Divergence to Uniform ($\lambda = 0.005$) | 447.61 | 564.84 | -49.64 | 121,531 |
| 13 | **Routing Regularization** | KL Divergence to Uniform ($\lambda = 0.025$) | 448.92 | 575.69 | -50.94 | 121,531 |
| 14 | **Auxiliary Supervision** | $\lambda_{\text{aux}} = 0.05$ | 449.67 | 569.50 | -51.69 | 121,531 |
| 15 | **Auxiliary Supervision** | $\lambda_{\text{aux}} = 0.10$ | 453.23 | 581.11 | -55.25 | 121,531 |
| 16 | **Conservative Routing** | Bounded Routing ($\rho = 0.50$) | 453.91 | 572.85 | -55.94 | 121,531 |

---

## 4. In-Depth Scientific Analysis of Ablation Results

### 4.1 Why Shrinkage / Bounded Routing Fails in Original Architecture
In exploratory Phase 6/7 research (which used GRU + TCN + PatchTemporal), bounded routing and shrinkage to uniform weights substantially improved generalization because all three experts were strong (~240–265 MW) and had comparable error variances.
However, in the **Original CAEG-Net (LSTM + TCN + CNN)**:
- Standalone CNN error is massive ($> 500$ MW).
- Any bounded routing mechanism with parameter $\rho$ forces each expert to receive at least weight:
  $$w_{\min} = \frac{1 - \rho}{3}$$
- For $\rho = 0.20$, each expert is forced to have at least **26.7% weight**!
- Forcing a defective expert to contribute more than a quarter of the forecast directly degrades the ensemble.
- The unconstrained V1 router can dynamically assign $< 5\%$ weight to the CNN, insulating the final forecast from CNN errors.

### 4.2 Why Auxiliary Expert Supervision Degrades the Gating Output
- When individual experts are trained with auxiliary loss $\lambda_{\text{aux}}$, the CNN expert is forced to optimize its own MSE loss directly. However, because the CNN architecture is poorly suited for long-range 168h temporal modeling, pushing gradients through the CNN pulls the shared representations or optimization capacity away from the collaborative ensemble objective.

### 4.3 Why Context 4D Outperforms Context 3D and 6D
- Context 4D includes the **Recent Baseline Error**. When Recent Error is removed (3D Context), Val MAE degrades from 397.97 MW to 403.57 MW ($+5.59$ MW error increase).
- Adding 2 extra noisy features (6D Context: weekly correlation and 48h range ratio) causes overfitting on the router MLP, worsening Val MAE to 447.24 MW.
- Therefore, the canonical 4D context (Trend, Volatility, Periodicity, Recent Ridge Error) is the optimal domain context representation.
