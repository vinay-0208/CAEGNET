# Phase 14A: Experimental Foundation Lock & Verification Audit

**Project:** CAEG-Net (Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting)  
**Branch:** `research-track`  
**Execution Environment:** `C:\Users\vinay\anaconda3\envs\caeg-gpu\python.exe` (PyTorch `2.13.0+cu130`, NVIDIA GeForce RTX 4050 Laptop GPU)  
**Status:** All 19 Foundational Dimensions Verified & Locked  

---

## 1. Audit Objective

Before launching Phase 14 model training, this audit rigorously verifies every component of the experimental pipeline against the frozen research protocol. Every dimension has been audited by direct code inspection and unit tests to ensure that Phase 14 begins from a verified, leakage-free, and reproducible foundation.

---

## 2. Foundational Dimension Verifications

### A. Expert Architecture
- **LSTM Expert:** 2 stacked recurrent layers (hidden dim 64, dropout 0.1) followed by a 2-layer linear projection head ($64 \to 64 \to 24$). Exactly **56,152** parameters.
- **TCN Expert:** 6 causal residual stages with dilated 1D convolutions (channels 32, kernel size 3, dilations $1, 2, 4, 8, 16, 32$, receptive field 253 hours) followed by projection ($32 \to 32 \to 24$). Exactly **36,952** parameters.
- **CNN Expert:** 3-stage 1D convolutional feature extractor (Conv1D $1 \to 32$ k3, MaxPool, Conv1D $32 \to 64$ k5, MaxPool, Conv1D $64 \to 64$ k3, BatchNorm, AdaptiveAvgPool1d(1)) with linear head ($64 \to 48 \to 24$). Exactly **27,400** parameters.
- **Expert Core Total:** Exactly **120,504** parameters. Zero Transformer, GRU, or attention layers are present.

### B. Parameter Counts of Phase 14 Candidate Formulations
Calculated directly from instantiated PyTorch models:
- **F0 (Canonical V1, 4D context):** Total **121,531** parameters (Temporal Core: 120,504; Context Encoder: 384; Router: 643; Total Router: 1,027).
- **F1 (A1-OOF, 7D context):** Total **121,579** parameters (+48 router parameters vs. V1, +0.040% overhead).
- **F2 (A2-OOF Confidence Fallback, 7D context):** Total **121,724** parameters (+193 parameters vs. V1, +0.159% overhead). Includes 145-parameter 2-layer MLP confidence head.
- **F3 (Confidence-Only Ablation, 4D context):** Total **121,628** parameters (+97 parameters vs. V1, +0.080% overhead). Includes 97-parameter 2-layer MLP confidence head.
- **F4 (Causally Smoothed Performance Router, 7D context):** Total **121,724** parameters (identical architecture to F2; operates on causally smoothed error features).
- **F5 (Simplified Scalar Shrinkage Control):** Total **121,579** model parameters. Uses F1 adaptive router with single scalar shrinkage $\lambda^* \in [0, 1]$ optimized on validation set (+0 network parameters).

### C. Canonical V1 Context
Verified in `data_utils.py` (`extract_context_features`) and `run_phase10_optimization.py`:
- Column 0: **Trend** (OLS regression slope over 168h lookback divided by lookback standard deviation).
- Column 1: **Volatility** (sample standard deviation of first differences $\sigma(\Delta z)$).
- Column 2: **Periodicity** (lag-24 sample autocorrelation $r_{24} \in [-1, 1]$).
- Column 3: **Recent Forecast Error** (causally completed previous-24h forecast error).
- Verified: Phase 14 maintains strictly this 4D canonical context. Prose misdescriptions from earlier phases are permanently corrected.

### D. Causal Recent Error Construction
- At forecast origin $t$, recent error is computed from the forecast issued at origin $t - 24$, predicting $[t-23, \dots, t]$.
- The forecast concludes at or before origin $t$. Future load values $y_{t+1:t+24}$ are strictly unobserved.

### E. Genuine Out-of-Fold (OOF) Performance Features
- For the training partition ($70\%$), an expanding-window 4-block chronological fold scheme is used:
  - Fold 1 (trained on Block 1) predicts Block 2 out-of-sample.
  - Fold 2 (trained on Blocks 1+2) predicts Block 3 out-of-sample.
  - Fold 3 (trained on Blocks 1+2+3) predicts Block 4 out-of-sample.
  - Block 1 receives neutral prior predictions from Fold 1.
- Zero in-sample training error is used to train the router.
- Validation and test error features are generated strictly out-of-sample by models trained on the completed training partition.

### F. Chronological Train / Validation / Test Partitions
- Partition splits: strictly 70% Train, 15% Validation, 15% Test.
- Modern PJM: $N = 5,957$ Train / $1,294$ Val / $1,294$ Test windows.
- GEFCom2014: $N = 41,425$ Train / $8,736$ Val / $10,944$ Test windows.
- UCI Cohort 320: $N = 18,221$ Train / $3,922$ Val / $3,922$ Test windows.
- Zero chronological index overlap across partitions.

### G. Train-Only Scaler
- Standard scaler fits mean $\mu_{\text{train}}$ and scale $\sigma_{\text{train}}$ strictly on the training partition.
- Validation and test sequences are transformed using frozen training statistics.

### H. Input-Output Window Slicing
- Lookback $L = 168$ hours (1 week).
- Horizon $H = 24$ hours (1 day).
- Step size = 1 hour.

### I. Loss Function & Optimization
- Loss: Mean Squared Error (MSE) on scaled targets.
- Optimizer: AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$).
- Gradient clipping: max norm 1.0.
- Learning rate scheduler: StepLR (step size 15, gamma 0.5).

### J. Checkpoint Selection
- Model state restored from the epoch achieving lowest Validation MSE.
- Early stopping patience: 6 epochs (max epochs: 25).

### K. Deterministic Seeding & CUDA/cuDNN Determinism
- Handled by `research/deterministic.py:seed_everything`:
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
- Unit tests verify identical tensor generation across repeated runs.

### L. DataLoader Determinism
- Handled by `make_deterministic_loader`:
  - DataLoader initialized with explicit PyTorch `Generator` keyed to the seed.
  - Worker init function sets individual worker seeds deterministically.

### M. Non-Overlapping Daily-Block Statistical Construction
- Daily blocks: non-overlapping 24-hour windows ($K = \lfloor N / 24 \rfloor$):
  - Modern PJM: $K = 53$ daily blocks ($53 \times 24 = 1,272$ windows).
  - GEFCom2014: $K = 456$ daily blocks ($456 \times 24 = 10,944$ windows).
  - UCI Cohort 320: $K = 163$ daily blocks ($163 \times 24 = 3,912$ windows).
- Paired differences: $D_k = \text{MAE}_k(\text{Candidate}) - \text{MAE}_k(\text{Baseline})$.
  - Negative difference: Candidate outperforms baseline.
  - Positive difference: Candidate degrades relative to baseline.

### N. Five-Seed Finalist Evaluation
- Seeds: `[42, 123, 999, 2024, 3407]`.
- Aggregation: Reported as $\text{Mean} \pm \text{Population SD}$ (`np.std(..., ddof=0)`).

### O. Statistical Inference & Multiple Testing
- Paired two-tailed $t$-test and Wilcoxon signed-rank test.
- Effect size: Cohen's $d_z = \bar{D} / s_D$.
- Multiple comparison correction: Step-down Holm-Bonferroni adjustment across the predefined comparison family.

### P. Validation-Only Selection of Best Standalone Expert
- Best standalone expert identified strictly on the validation partition:
  - Modern PJM: Standalone **TCN** ($461.52\text{ MW}$ Val MAE).
  - GEFCom2014: Standalone **TCN** ($13.37\text{ kW}$ Val MAE).
  - UCI Cohort 320: Standalone **LSTM** ($7.55\text{ MW}$ Val MAE).
- Zero test data inspected during baseline selection.

### Q. Static Equal Ensemble Construction
- Uniform combination: $\hat{y}_{\text{equal}} = \frac{1}{3} \hat{y}_{\text{LSTM}} + \frac{1}{3} \hat{y}_{\text{TCN}} + \frac{1}{3} \hat{y}_{\text{CNN}}$.
- Baseline test results: PJM $= 279.83\text{ MW}$, GEFCom $= 12.62\text{ kW}$, UCI $= 8.17\text{ MW}$.

### R. Historical Artifact Provenance
- Phase 10, Phase 11, Phase 12, and Phase 13 raw files, CSVs, reports, and Git commits are preserved intact.

---

## 3. Foundation Lock Verdict
All foundational components are verified, leakage-free, and locked for Phase 14 execution.
