# CAEG-Net Phase 5: Repository Restoration & Architecture Boundaries
**Document**: `PHASE_5_REPOSITORY_RESTORATION.md`  
**Track**: `research-track` | **Phase**: Phase 5 (Original 14-Phase Research Roadmap)  
**Date**: September 2026  
**Status**: ACTIVE CANONICAL  

---

## 1. Scope & Objective

This document defines the strict boundary between the **Frozen Canonical V1 Reference Implementation** and the **Phase 5 Research-Guided Optimization Track**. It specifies the exact architectural components, parameter budgets, and how validated insights from exploratory V2/V3 research are legitimately adapted into the canonical model.

---

## 2. Canonical vs. Exploratory Repository Boundaries

### 2.1 Preserved Canonical Modules
The following root modules define the frozen canonical benchmark and standard interfaces:
- **`caeg_net.py`**: Contains the pure canonical V1 implementation (`CAEGNet`, aliased as `CAEGNetV1`), the original expert definitions (`LSTMExpert`, `TCNExpert`, `CNNExpert`), and the canonical gating network. This file is permanently frozen to preserve historical reproducibility.
- **`data_utils.py`**: Canonical causal data loader, standard scaling, and 4D context extraction.
- **`train.py` & `evaluate.py`**: Canonical training loop and evaluation routines.
- **`results/phase5_summary.csv`**: Contains the frozen multi-seed V1 reference metrics ($251.44 \pm 9.74$ MW).

### 2.2 Exploratory Archive Modules
The following modules represent historical exploratory research:
- **`research/models.py`**: Contains V2, V3, Decoupled-CAEG, Shrinkage-CAEG, and Bounded-Routing models built on the alternative GRU + TCN + PatchTemporal backbone. These serve exclusively as historical records and comparative baselines.
- **`research/training.py` & `research/data.py`**: V2/V3 specific trainers and 192h context pipelines.

### 2.3 Phase 5 Research Track Implementation
The Phase 5 research track lives in:
- **`research/original_caeg.py`**: The canonical Phase 5 model (`OriginalCAEGNetPhase5`), maintaining the original **LSTM + TCN + CNN** backbone with research-guided enhancements (detached disagreement input, conservative bounded routing, and auxiliary loss).
- **`research/tests/test_phase5_original_caeg.py`**: Comprehensive unit and causality test suite.
- **`research/experiments/run_phase5_original_caeg.py`**: Fully automated experimental runner.
- **`research/results/phase5/`**: Complete artifact, metrics, and visualization repository.

---

## 3. Parameter Budgets & Accounting

The canonical parameter budget is strictly respected:
- Baseline V1 CAEG-Net: **121,531** parameters.
- Phase 5 CAEG-Net Candidate: **121,579** parameters ($+48$ parameters, $+0.04\%$).
- Learned Static Ensemble: **120,507** parameters.

Neither model introduces heavy parameter bloat or alters the expert capacities.

---

## 4. Transfer of Research-Guided Hypotheses to Original Architecture

From the exploratory V2/V3 research, several valuable mathematical insights were extracted. In Phase 5, these are evaluated strictly within the context of the original LSTM + TCN + CNN architecture:

1. **Detached Inter-Expert Disagreement**:
   - Rather than relying solely on domain context (trend, volatility, periodicity), the router receives a 3-dimensional uncertainty vector representing pairwise expert divergence:
     $$d_t = [|\hat{y}_t^{\text{LSTM}} - \hat{y}_t^{\text{TCN}}|, |\hat{y}_t^{\text{LSTM}} - \hat{y}_t^{\text{CNN}}|, |\hat{y}_t^{\text{TCN}} - \hat{y}_t^{\text{CNN}}|]$$
   - Critical rule: gradients from $d_t$ are detached (`.detach()`) before entering the router to prevent expert corruption.

2. **Conservative Bounded Routing**:
   - In regions of high forecast ambiguity, adaptive routing can suffer from estimation variance. A bounded convex combination with uniform weights:
     $$w_t^{\text{bounded}} = (1 - \rho) \cdot \frac{1}{3} + \rho \cdot w_t^{\text{router}}, \quad \rho \in [0.10, 1.00]$$
     shrinks the router towards equal weighting, guaranteeing robust lower-bound performance.

3. **Auxiliary Expert Supervision**:
   - Joint end-to-end training of MoEs often results in expert co-adaptation and degradation of individual components (as occurred with CNN in V1). Adding auxiliary expert loss:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ensemble}} + \lambda_{\text{aux}} \sum_{i=1}^3 \text{MSE}(\hat{y}^{(i)}, y)$$
     ensures each expert maintains strong standalone predictive competence.
