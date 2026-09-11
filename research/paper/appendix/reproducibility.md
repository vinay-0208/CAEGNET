# Appendix: Comprehensive Reproducibility and Experimental Provenance

This appendix provides exhaustive technical, software, hardware, and algorithmic specifications required to replicate all empirical results reported for **CAEG-Net** (Candidate `F2_A2_OOF`) and its baseline models.

---

## 1. Computing Environment & Hardware Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise / Pro (64-bit) |
| **CPU Architecture** | x86_64 Multi-Core Host Processor |
| **Graphics Processing Unit (GPU)** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM) |
| **GPU Driver Version** | NVIDIA Display Driver 550+ |
| **Compute Unified Device Architecture (CUDA)** | CUDA 13.0 |
| **Python Runtime Environment** | Python 3.10.13 (Anaconda Distribution) |
| **Primary Deep Learning Framework** | PyTorch 2.13.0+cu130 |
| **Scientific Computing Libraries** | NumPy 2.2.6, SciPy 1.15.3, Pandas 2.2.3, Scikit-Learn 1.6.1 |
| **Visualization Libraries** | Matplotlib 3.10.0, Seaborn 0.13.2 |

---

## 2. Experimental Randomization & Seed Management

To ensure deterministic reproducibility across experiments, all stochastic operations (Python `random`, NumPy RNG, PyTorch CPU seeds, and PyTorch CUDA backend kernels) are initialized using an identical sequence of five predefined integer seeds:
$$\mathcal{S} = \{42, 123, 999, 2024, 3407\}$$

PyTorch determinism flags are explicitly configured:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 3. Dataset Preprocessing & Boundary Leakage Controls

### 3.1 Chronological Partitioning
Each dataset is split into chronological partitions **prior** to sliding-window creation to eliminate future data leakage:
- **Training Partition:** 70% of chronological observations
- **Validation Partition:** 15% of chronological observations (used exclusively for early stopping and model selection)
- **Held-Out Test Partition:** 15% of chronological observations (frozen during development and evaluated across all 5 seeds)

| Dataset | Total Hours | Train Hours | Val Hours | Test Hours | Daily Test Blocks ($K$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modern PJM** | 8,784 | 6,148 | 1,318 | 1,318 | 53 |
| **GEFCom2014 (Zone 1)** | 35,064 | 24,544 | 5,260 | 5,260 | 456 |
| **UCI Electricity (Cohort 320)** | 14,016 | 9,811 | 2,102 | 2,103 | 163 |

### 3.2 Feature Normalization
Input features are normalized via standard scaling ($z = (x - \mu)/\sigma$). The sample mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are computed strictly from the training partition and held fixed across validation and test partitions.

### 3.3 Partition-Aware Lookback
For any sliding window starting at test index $t_0$, the 168-hour historical lookback $X_{t_0-168:t_0}$ is drawn from historical observations immediately preceding the split boundary without permitting any target observations from the test horizon $y_{t_0:t_0+24}$ into the input or context encoder.

---

## 4. Chronological Out-of-Fold (OOF) Performance Feature Generation

To supply the gating network with recent expert-performance information without training-set leakage, an expanding-window four-block cross-validation is performed across the 70% training split:
1. The training split $\mathcal{D}_{\text{train}}$ is divided into 4 contiguous blocks: $B_1, B_2, B_3, B_4$.
2. For fold $k \in \{1, 2, 3\}$:
   - Models are trained on blocks $\bigcup_{i=1}^k B_i$.
   - Out-of-fold rolling predictions are generated on block $B_{k+1}$.
   - For every window $t$, the 24-hour historical MAE of each expert $m \in \{\text{LSTM}, \text{TCN}, \text{CNN}\}$ is recorded:
     $$e_{m, t} = \frac{1}{24} \sum_{h=1}^{24} |y_{t-h} - \hat{y}_{m, t-h}|$$
   - Relative normalized error metrics are computed:
     $$r_{m, t} = \frac{e_{m, t}}{\sum_{j=1}^3 e_{j, t}}$$
3. For the validation and test partitions, the causal rolling 24-hour historical error is computed strictly from previously observed true load values up to time $t$.

This mechanism serves strictly as a forecasting leakage-control procedure, ensuring that expert-performance features are causally available at forecast time without future lookahead. It does not constitute a causal inference or treatment effect model.

---

## 5. Model Architecture & Exact Parameter Counts

| Component | Layer Configuration | Parameter Count |
| :--- | :--- | :---: |
| **LSTM Expert** | 2-layer LSTM ($h=64$, dropout $0.1$)<br>Head: `Linear(64, 64) -> ReLU -> Dropout(0.1) -> Linear(64, 24)` | 56,152 |
| **TCN Expert** | 6 Causal Residual Stages ($k=3$, dilations $\{1, 2, 4, 8, 16, 32\}$, 32 ch, RF=253h)<br>Head: `Linear(32, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 24)` | 36,952 |
| **CNN Expert** | 3-Stage Conv1D ($1 \to 32 \to 64 \to 64$, MaxPool + AdaptiveAvgPool)<br>Head: `Linear(64, 48) -> ReLU -> Dropout(0.1) -> Linear(48, 24)` | 27,400 |
| **Expert Core Subtotal** | *Frozen multi-expert backbone* | **120,504** |
| **Softmax Router** | `nn.Linear(7, 16)`, `nn.ReLU()`, `nn.Linear(16, 3)`, `nn.Softmax(dim=-1)` | 1,075 |
| **Confidence Head** | `nn.Linear(7, 16)`, `nn.ReLU()`, `nn.Linear(16, 1)`, `nn.Sigmoid()` | 145 |
| **Total CAEG-Net Model** | *Candidate `F2_A2_OOF`* | **121,724** |
| **Net Overhead vs. V1** | *Overhead vs. Canonical 4-context model (121,531 params)* | **+193 (+0.1588%)** |

---

## 6. Training Hyperparameters & Optimization

- **Optimization Loss:** Mean Squared Error (MSE) over the 24-hour horizon.
- **Primary Evaluation Metric:** Mean Absolute Error (MAE), providing direct physical interpretability in load units.
- **Optimizer:** AdamW (initial lr = $1.0 \times 10^{-3}$, weight decay = $1.0 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
- **Learning Rate Schedule:** StepLR with step size 15 epochs and multiplicative decay factor $\gamma = 0.5$.
- **Early Stopping:** Patience of 6 validation epochs monitoring validation loss. Optimization halts if validation loss does not improve for 6 consecutive epochs, restoring the checkpoint with lowest error.
- **Batch Size:** 64 sliding windows per gradient step.
- **Maximum Epochs:** 25 epochs.

---

## 7. Statistical Evaluation Protocol

To account for temporal auto-correlation in sequential electricity load forecasts, hypothesis testing is conducted over non-overlapping 24-hour daily blocks:
1. For each held-out day $k \in \{1, \dots, K\}$, the daily MAE is computed over the 24 hours:
   $$\text{MAE}_k = \frac{1}{24} \sum_{h=1}^{24} |y_{k, h} - \hat{y}_{k, h}|$$
2. Paired difference series: $\Delta_k = \text{MAE}_{\text{F2}, k} - \text{MAE}_{\text{F0}, k}$.
3. Hypothesis tests:
   - Two-sided paired Student's $t$-test.
   - Non-parametric Wilcoxon signed-rank test.
4. Multiple testing correction: Holm-Bonferroni step-down procedure applied across candidate comparisons per dataset to strictly bound the family-wise error rate ($\alpha = 0.05$).
5. Effect size metric: Cohen's $d_z = \bar{\Delta} / s_{\Delta}$.
