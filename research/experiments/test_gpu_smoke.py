import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from caeg_net import LSTMExpert, TCNExpert, CNNExpert
from research.original_caeg import OriginalCAEGNetPhase5, compute_phase5_loss

print("=== GPU SMOKE TEST ===", flush=True)
print(f"Python Executable: {sys.executable}", flush=True)
print(f"PyTorch Version: {torch.__version__}", flush=True)
print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is expected but unavailable! Failing immediately per instructions.")

device = torch.device("cuda")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

# 1. Build Model on GPU
model = OriginalCAEGNetPhase5(context_dim=4).to(device)

# Verify submodule devices
print(f"LSTM Expert Device: {next(model.lstm_expert.parameters()).device}", flush=True)
print(f"TCN Expert Device: {next(model.tcn_expert.parameters()).device}", flush=True)
print(f"CNN Expert Device: {next(model.cnn_expert.parameters()).device}", flush=True)
print(f"Context Router Device: {next(model.router.parameters()).device}", flush=True)

# 2. Forward, Loss, Backward pass with synthetic batch
x = torch.randn(64, 168, 1, device=device)
c = torch.randn(64, 4, device=device)
y_true = torch.randn(64, 24, device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

y_pred, weights, diag = model(x, c, return_diagnostics=True)
loss, telemetry = compute_phase5_loss(y_pred, y_true, weights=weights)
loss.backward()
optimizer.step()

# Memory allocated
mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
print(f"Forward + Backward pass successful! Loss = {loss.item():.4f}", flush=True)
print(f"GPU Memory Allocated: {mem_allocated:.2f} MB | Reserved: {mem_reserved:.2f} MB", flush=True)

# 3. Call nvidia-smi
print("\n--- nvidia-smi query ---", flush=True)
try:
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv"],
        capture_output=True, text=True
    )
    print(res.stdout, flush=True)
except Exception as e:
    print(f"Could not run nvidia-smi: {e}", flush=True)

print("=== GPU SMOKE TEST PASSED ===", flush=True)
