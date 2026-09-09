# CAEG-Net Phase 5B: Controlled Expert Optimization & Fusion Interaction Analysis
**Document**: `research/results/PHASE_5B_EXPERT_OPTIMIZATION.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: VALIDATION SCREENING COMPLETE  

---

## 1. Executive Summary & Objective

The canonical CAEG-Net expert family is strictly fixed to:
$$\mathbf{E} = \{\text{LSTM}, \text{TCN}, \text{CNN}\}$$
In the canonical V1 baseline configuration, the **CNN expert** demonstrated substantially weaker forecasting performance (Validation $\text{MAE} \approx 730\text{ MW}$ vs $\approx 419\text{ MW}$ for TCN and $\approx 542\text{ MW}$ for LSTM).

Rather than replacing the expert family with alternative architectures (e.g. PatchLinear or GRU), Phase 5B conducted **controlled, single-variable experimental ablations** strictly on the **Validation Partition (Seed 42)** to isolate the architectural bottlenecks of the canonical experts and measure their interaction with both the static equal ensemble and adaptive CAEG fusion.

---

## 2. CNN Controlled Architectural & Training Optimization

Ten controlled experiments (CNN-A through CNN-H) were evaluated one modification at a time against the canonical control baseline (`CNN_Control`):

| Experiment ID | Controlled Modification | Parameters | Val MAE (MW) | Val RMSE (MW) | Val $R^2$ | Best Ep | Gain vs Control (MW) | Improved? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN_H2_TemporalPool8** | Adaptive temporal pooling (`AdaptiveAvgPool1d(8)`) | 48,904 | **541.26** | **714.40** | **0.6343** | 7 | **+189.30** | **YES** |
| **CNN_C_DilatedConv** | Dilated conv (dilation=2 in stages 2–3, kernel=3) | **23,304** | **548.14** | **714.50** | **0.6342** | 22 | **+182.42** | **YES** |
| **CNN_G_CosineScheduler** | `CosineAnnealingLR` scheduler | 27,400 | **574.17** | **740.62** | **0.6069** | 34 | **+156.39** | **YES** |
| **CNN_H1_TemporalPool4** | Adaptive temporal pooling (`AdaptiveAvgPool1d(4)`) | 36,616 | **596.49** | **765.60** | **0.5800** | 12 | **+134.07** | **YES** |
| **CNN_E_ZeroDropout** | Linear head `dropout = 0.0` | 27,400 | **671.13** | **859.34** | **0.4708** | 39 | **+59.43** | **YES** |
| **CNN_Control** | **Canonical V1 baseline CNN architecture** | **27,400** | **730.56** | **929.06** | **0.3815** | 39 | **0.00** | — |
| **CNN_F_LowerLR** | Lower learning rate ($\text{lr} = 5 \times 10^{-4}$) | 27,400 | 740.04 | 927.18 | 0.3840 | 36 | -9.48 | NO |
| **CNN_B_GroupNorm** | GroupNorm (group=4) instead of BatchNorm1d | 27,400 | 785.53 | 978.78 | 0.3135 | 25 | -54.97 | NO |
| **CNN_D_HeadCapacity** | Expanded MLP head ($64 \to 128 \to 64 \to 24$) | 41,240 | 875.61 | 1098.16 | 0.1358 | 16 | -145.05 | NO |
| **CNN_A_LessPooling** | Stride 1 on stage 1 conv | 27,400 | 1024.44 | 1275.69 | -0.1662 | 10 | -293.88 | NO |

---

## 3. Scientific Diagnoses of CNN Architecture

1. **The Temporal Bottleneck of `AdaptiveAvgPool1d(1)`**:
   - In canonical V1, `AdaptiveAvgPool1d(1)` collapsed 42 temporal positions down to a single scalar per channel. The model attempted to forecast 24 future hours from 64 static, time-averaged scalar features.
   - Retaining temporal resolution via `AdaptiveAvgPool1d(8)` flattened into the linear head yielded an immediate **$+189.30$ MW improvement** in Validation MAE ($541.26\text{ MW}$ vs $730.56\text{ MW}$).
2. **Receptive Field Expansion via Dilation**:
   - Standard 1D convolutions with small kernels ($k=3, 5, 3$) and unit dilation have a receptive field of only 15 hours.
   - Introducing dilation ($d=2$) in stages 2 and 3 expanded the effective receptive field across multiple diurnal cycles, improving Validation MAE to **$548.14$ MW ($+182.42$ MW gain)** while actually **reducing parameters from 27,400 to 23,304**!
3. **Training Dynamics & Schedulers**:
   - The CNN is sensitive to early step-decay schedulers. `CosineAnnealingLR` maintained gradient flow throughout training, delivering a **$+156.39$ MW gain** without adding a single parameter.

---

## 4. Expert + Fusion Interaction (Validation Partition, Seed 42)

Evaluating whether standalone expert gains translate to ensemble improvements (Section 10 requirement):

| Setup / Configuration | Standalone Expert Val MAE | Equal Ensemble Val MAE | Gain vs Control Ensemble |
| :--- | :---: | :---: | :---: |
| **Canonical Control CNN** | 730.56 MW | 509.31 MW | 0.00 MW |
| **Optimized CNN (`CNN_H2_TemporalPool8`)** | **541.26 MW** | **461.25 MW** | **+48.06 MW** |

### Key Fusion Finding:
Improving the standalone CNN expert directly lifted the entire **Static Equal Ensemble from $509.31$ MW to $461.25$ MW (an unscaled physical improvement of $+48.06$ MW)** on the validation partition. This confirms that expert-level optimization genuinely strengthens collaborative multi-model forecasting.
