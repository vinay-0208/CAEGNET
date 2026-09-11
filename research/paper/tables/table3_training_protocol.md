# Table 3: Experimental and Training Protocol Parameters

| Parameter | Configuration / Value | Scientific Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Robust adaptive weight decay decoupled from $ gradient updates |
| **Learning Rate** | .0 \times 10^{-3}$ | Verified stable convergence across recurrent, convolutional, and gating heads |
| **Weight Decay** | .0 \times 10^{-4}$ | Light regularization on dense projection weights |
| **Batch Size** | 64 | Efficient GPU memory utilization with stable batch-level gradient estimates |
| **Learning Rate Scheduler**| StepLR (step size = 10 epochs, $\gamma = 0.5$) | Gradual decay ensuring fine parameter convergence in later epochs |
| **Maximum Epochs** | 50 | Bound on optimization budget |
| **Early Stopping** | Patience = 10 epochs | Halts optimization upon validation MAE degradation; restores best checkpoint |
| **Loss Function** | Mean Squared Error (MSE) | Standard quadratic empirical risk objective for regression |
| **Primary Evaluation Metric**| Mean Absolute Error (MAE) | Direct linear penalty reflecting grid dispatch and tariff economics |
| **Random Seeds** | 5 seeds: {42, 123, 456, 789, 1000} | Rigorous multi-seed variance and population standard deviation assessment |
| **OOF Cross-Validation** | 5-fold expanding chronological window | Leakage-free historical expert-performance feature extraction |
