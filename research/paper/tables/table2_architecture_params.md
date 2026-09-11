# Table 2: CAEG-Net Component Architectural Specifications and Parameter Overhead

| Component | Layer Specification / Hyperparameters | Output Dim. | Parameters |
| :--- | :--- | :---: | :---: |
| **LSTM Expert** | 2-layer LSTM, hidden size =64$, dropout =0.1$, Linear(64, 24) | $\mathbb{R}^{24}$ | 50,264 |
| **TCN Expert** | 3 residual blocks, kernel size 3, dilations $[1, 2, 4]$, 32 channels, Linear(32, 24) | $\mathbb{R}^{24}$ | 35,800 |
| **CNN Expert** | 3 parallel 1D conv branches (kernels 3, 5, 7; 32 filters each), concat (96), Linear(96, 24) | $\mathbb{R}^{24}$ | 34,440 |
| **Subtotal: Expert Core** | *Frozen multi-expert temporal feature extraction backbone* | — | **120,504** |
| **Context Encoder (V1)** | 4 context features (volatility, trend, peak-ratio, cyclical) | $\mathbb{R}^4$ | — |
| **Softmax Router (V1)** | Linear(4, 16) $\to$ ReLU $\to$ Linear(16, 3) $\to$ Softmax | $\mathbb{R}^3$ | 1,027 |
| **Total Canonical V1** | *V1 Baseline (F0_Canonical_V1)* | $\mathbb{R}^{24}$ | **121,531** |
| **Context Encoder (F2)** | 4 context features + 3 causal OOF rolling MAE error metrics | $\mathbb{R}^7$ | — |
| **Softmax Router (F2)** | Linear(7, 16) $\to$ ReLU $\to$ Linear(16, 3) $\to$ Softmax | $\mathbb{R}^3$ | 1,075 |
| **Confidence Head (F2)** | Linear(7, 16) $\to$ ReLU $\to$ Linear(16, 1) $\to$ Sigmoid | $\mathbb{R}^1$ | 145 |
| **Total Final F2 Model** | *CAEG-Net F2_A2_OOF (Final)* | $\mathbb{R}^{24}$ | **121,724** |
| **Incremental Overhead** | *Additional parameters over Canonical V1* | — | **+193 (+0.1588%)** |
