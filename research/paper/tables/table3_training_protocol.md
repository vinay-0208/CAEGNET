# Table 3: Experimental and Training Protocol Parameters

| Parameter | Configuration / Value | Scientific Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Decoupled weight decay regularization |
| **Learning Rate** | $1.0 \times 10^{-3}$ | Stable convergence across recurrent, convolutional, and gating heads |
| **Weight Decay** | $1.0 \times 10^{-4}$ | Regularization on linear projection layers |
| **Batch Size** | 64 | Efficient GPU memory utilization with stable batch-level gradient estimates |
| **Learning Rate Scheduler**| StepLR (step size = 15 epochs, $\gamma = 0.5$) | Gradual decay refining terminal convergence |
| **Maximum Epochs** | 25 | Upper bound on training optimization budget |
| **Early Stopping** | Patience = 6 epochs | Halts optimization upon validation loss plateau; restores best checkpoint |
| **Optimization Loss** | Mean Squared Error (MSE) | Quadratic empirical risk objective during backpropagation |
| **Primary Evaluation Metric**| Mean Absolute Error (MAE) | Direct physical interpretation in load units without quadratic distortion |
| **Random Seeds** | 5 seeds: {42, 123, 999, 2024, 3407} | Rigorous multi-seed variance and population standard deviation assessment |
| **OOF Construction** | 4-block chronological expanding window | Leakage-free historical expert-performance feature generation |
