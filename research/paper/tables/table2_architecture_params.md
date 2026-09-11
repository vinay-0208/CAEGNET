# Table 2: CAEG-Net Component Architectural Specifications and Parameter Overhead

| Component | Layer Specification / Hyperparameters | Output Dim. | Parameters |
| :--- | :--- | :---: | :---: |
| **LSTM Expert** | 2-layer LSTM ($h=64$, dropout $p=0.1$), forecasting head Linear(64, 64) $\to$ ReLU $\to$ Linear(64, 24) | $\mathbb{R}^{24}$ | 56,152 |
| **TCN Expert** | 6 causal residual stages ($k=3$, dilations $\{1, 2, 4, 8, 16, 32\}$, 32 ch, RF=253h), head Linear(32, 32) $\to$ ReLU $\to$ Linear(32, 24) | $\mathbb{R}^{24}$ | 36,952 |
| **CNN Expert** | 3-stage Conv1D ($1 \to 32 \to 64 \to 64$, MaxPool + AdaptiveAvgPool), head Linear(64, 48) $\to$ ReLU $\to$ Linear(48, 24) | $\mathbb{R}^{24}$ | 27,400 |
| **Subtotal: Expert Core** | *Frozen multi-expert temporal feature extraction backbone* | — | **120,504** |
| **Context Encoder (V1)** | 4 state features: trend, volatility, lag-24 autocorrelation, recent error | $\mathbb{R}^4$ | — |
| **Softmax Router (V1)** | Linear(4, 16) $\to$ ReLU $\to$ Linear(16, 3) $\to$ Softmax | $\mathbb{R}^3$ | 1,027 |
| **Total Canonical V1** | *V1 Baseline (`F0_Canonical_V1`)* | $\mathbb{R}^{24}$ | **121,531** |
| **Context Encoder (F2)** | 4 state features + 3 chronological OOF relative expert-performance metrics | $\mathbb{R}^7$ | — |
| **Softmax Router (F2)** | Linear(7, 16) $\to$ ReLU $\to$ Linear(16, 3) $\to$ Softmax | $\mathbb{R}^3$ | 1,075 |
| **Confidence Head (F2)** | Linear(7, 16) $\to$ ReLU $\to$ Linear(16, 1) $\to$ Sigmoid | $\mathbb{R}^1$ | 145 |
| **Total Final F2 Model** | *CAEG-Net `F2_A2_OOF` (Final)* | $\mathbb{R}^{24}$ | **121,724** |
| **Incremental Overhead** | *Additional parameters over Canonical V1* | — | **+193 (+0.1588%)** |
