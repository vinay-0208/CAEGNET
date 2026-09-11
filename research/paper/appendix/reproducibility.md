# Appendix: Comprehensive Reproducibility and Experimental Provenance

This appendix provides exhaustive technical, software, hardware, and algorithmic specifications required to replicate all empirical results reported for **CAEG-Net** (Candidate F2_A2_OOF) and its baseline models.

---

## 1. Computing Environment & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise / Pro (64-bit) |
| **CPU Architecture** | x86_64 Multi-Core Host Processor |
| **Graphics Processing Unit (GPU)** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **GPU Driver Version** | NVIDIA Display Driver 550+ |
| **Compute Unified Device Architecture (CUDA)** | CUDA 13.0 |
| **Python Runtime Environment** | Python 3.10.13 (Anaconda Distribution) |
| **Primary Deep Learning Framework** | PyTorch 2.13.0+cu130 |
| **Scientific Computing Libraries** | NumPy 2.2.6, SciPy 1.15.3, Pandas 2.2.3, Scikit-Learn 1.6.1 |
| **Visualization Libraries** | Matplotlib 3.10.0, Seaborn 0.13.2 |

---

## 2. Experimental Randomization & Seed Management

To guarantee numerical reproducibility across platforms, all stochastic operations (including Python random, NumPy random number generator, PyTorch CPU seeds, and PyTorch CUDA backend kernels) are initialized using an identical sequence of five predefined integer seeds:
Seed list: [42, 123, 456, 789, 1000]

PyTorch determinism flags are explicitly configured:
- torch.manual_seed(seed)
- torch.cuda.manual_seed_all(seed)
- np.random.seed(seed)
- random.seed(seed)
- torch.backends.cudnn.deterministic = True
- torch.backends.cudnn.benchmark = False

---

## 3. Dataset Preprocessing & Boundary Leakage Controls

### 3.1 Chronological Partitioning
Each dataset is split into chronological partitions prior to sliding-window creation to eliminate future data leakage:
- **Training Partition:** 70% of chronological observations
- **Validation Partition:** 15% of chronological observations (used exclusively for early stopping and model selection)
- **Held-Out Test Partition:** 15% of chronological observations (frozen during development and evaluated across all 5 seeds)

| Dataset | Total Hours | Train Hours | Val Hours | Test Hours | Daily Test Blocks (K) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | 8,784 | 6,148 | 1,318 | 1,318 | 53 |
| **GEFCom2014 (Zone 1)** | 35,064 | 24,544 | 5,260 | 5,260 | 456 |
| **UCI Electricity (Cohort 320)** | 14,016 | 9,811 | 2,102 | 2,103 | 163 |

### 3.2 Feature Normalization
Input features are normalized via standard scaling (z = (x - mean)/std). The sample mean and standard deviation are computed strictly from the training partition and held fixed across validation and test partitions.

### 3.3 Partition-Aware Lookback
For any sliding window starting at test index t0, the 168-hour historical lookback is drawn from historical observations immediately preceding the split boundary without permitting any target observations from the test horizon into the input or context encoder.

---

## 4. Chronological Out-of-Fold (OOF) Performance Feature Generation

To supply the gating network with recent expert-performance information without training-set leakage, an expanding-window 5-fold cross-validation is performed across the 70% training split:
1. The training split is divided into 6 contiguous blocks: B1, B2, B3, B4, B5, B6.
2. For fold k in {1, 2, 3, 4, 5}:
   - Models are trained on blocks Union_{i=1}^k Bi.
   - Out-of-fold rolling predictions are generated on block B_{k+1}.
   - For every window t, the 24-hour historical MAE of each expert is recorded.
   - Normalized relative error metrics are computed across experts.
3. For validation and test partitions, causal rolling 24-hour historical error is computed strictly from previously observed true load values up to time t.

---

## 5. Model Architecture & Exact Parameter Counts

| Component | Layer Configuration | Parameter Count |
| :--- | :--- | :---: |
| **LSTM Expert** | 2-layer LSTM (hidden size 64, dropout 0.1) + Linear(64, 24) | 50,264 |
| **TCN Expert** | 3 Residual Blocks (k=3, dilations 1, 2, 4, 32 channels) + Linear(32, 24) | 35,800 |
| **CNN Expert** | 3 Parallel Branches (k=3, 5, 7; 32 filters each) + Linear(96, 24) | 34,440 |
| **Expert Core Subtotal** | Frozen multi-expert backbone | **120,504** |
| **Softmax Router** | Linear(7, 16) -> ReLU -> Linear(16, 3) -> Softmax | 1,075 |
| **Confidence Head** | Linear(7, 16) -> ReLU -> Linear(16, 1) -> Sigmoid | 145 |
| **Total CAEG-Net Model** | Candidate F2_A2_OOF | **121,724** |
| **Net Overhead vs. V1** | Overhead vs. Canonical 4-context model (121,531 params) | **+193 (+0.1588%)** |

---

## 6. Training Hyperparameters & Optimization

- **Loss Function:** Mean Squared Error (MSE) over the 24-hour horizon.
- **Optimizer:** AdamW (initial lr = 1.0e-3, weight decay = 1.0e-4, beta1 = 0.9, beta2 = 0.999).
- **Learning Rate Schedule:** StepLR with step size 10 epochs and gamma = 0.5.
- **Early Stopping:** Patience of 10 validation epochs monitoring Validation Mean Absolute Error (MAE).
- **Batch Size:** 64 sliding windows per gradient step.
- **Maximum Epochs:** 50 epochs.

---

## 7. Statistical Evaluation Protocol

To account for temporal auto-correlation in sequential electricity load forecasts, hypothesis testing is conducted over non-overlapping 24-hour daily blocks rather than individual overlapping hourly windows:
1. For each held-out day k in {1, ..., K}, the daily MAE is computed over the 24 hours.
2. Paired difference series: Delta_k = MAE_{F2, k} - MAE_{F0, k}.
3. Hypothesis tests:
   - Two-sided paired Student t-test.
   - Non-parametric Wilcoxon signed-rank test.
4. Multiple testing correction: Holm-Bonferroni step-down procedure applied across candidate comparisons per dataset to bound family-wise error rate (alpha = 0.05).
5. Effect size metric: Cohen dz.
