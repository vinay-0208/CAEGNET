"""
CAEG-Net V2 Phase 3 Smoke Test Runner
======================================
Executes a minimal, lightweight smoke training run to confirm end-to-end operational
viability on the local development machine (NVIDIA RTX 4050 6 GB GPU / CUDA).

Scope:
- 1 epoch, limited to 5 training mini-batches and 2 validation mini-batches.
- Forward pass, tripartite loss evaluation, autograd backward pass, optimizer step,
  telemetry collection, and evaluation mode validation.
- Records: loss reduction, execution time, peak CUDA memory allocated, and absence of NaNs.
- IMPORTANT: This is an implementation sanity check, NOT an experimental benchmark result.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import json
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from research.models import CAEGNetV2, compute_caeg_v2_loss, count_parameters
from research.data import prepare_research_v2_pipeline, build_research_v2_dataloaders


def run_smoke_test(device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_str)
    print(f"=== Starting CAEG-Net V2 Phase 3 Smoke Test on {device} ===")
    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Pipeline preparation
    t0 = time.perf_counter()
    pipeline_data = prepare_research_v2_pipeline()
    dataloaders = build_research_v2_dataloaders(pipeline_data, batch_size=32, seed=42)
    t_pipe = time.perf_counter() - t0
    print(f"Data pipeline prepared in {t_pipe:.2f}s")

    # 2. Model instantiation
    model = CAEGNetV2().to(device)
    param_counts = count_parameters(model)
    print(f"Model instantiated with {param_counts['total']:,} parameters")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Smoke training loop: 5 batches
    model.train()
    train_telemetry = []
    initial_loss = None
    final_loss = None

    for b_idx, (x, y, c) in enumerate(dataloaders["train"]):
        if b_idx >= 5:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        y_fused, weights, diag = model(x, c)
        loss, telem = compute_caeg_v2_loss(y_fused, y, diag["expert_predictions"], weights)

        assert not torch.isnan(loss), f"NaN encountered in loss at batch {b_idx}"
        assert not torch.isinf(loss), f"Inf encountered in loss at batch {b_idx}"

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if b_idx == 0:
            initial_loss = telem["loss_total"]
        final_loss = telem["loss_total"]

        train_telemetry.append({
            "batch": b_idx + 1,
            "loss_total": telem["loss_total"],
            "loss_fused": telem["loss_fused"],
            "loss_aux": telem["loss_aux"],
            "loss_entropy": telem["loss_entropy"],
            "mean_weights": weights.mean(dim=0).detach().cpu().tolist(),
        })

    # 4. Smoke evaluation loop: 2 batches
    model.eval()
    val_losses = []
    with torch.no_grad():
        for b_idx, (x, y, c) in enumerate(dataloaders["val"]):
            if b_idx >= 2:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            y_fused, weights, diag = model(x, c)
            loss, telem = compute_caeg_v2_loss(y_fused, y, diag["expert_predictions"], weights)
            val_losses.append(telem["loss_total"])

    total_time = time.perf_counter() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    smoke_results = {
        "status": "PASSED",
        "device": str(device),
        "total_parameters": param_counts["total"],
        "trainable_parameters": param_counts["trainable"],
        "parameter_breakdown": param_counts,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "mean_val_loss": sum(val_losses) / len(val_losses),
        "batches_trained": len(train_telemetry),
        "batches_evaluated": len(val_losses),
        "peak_cuda_memory_mb": round(peak_vram_mb, 2),
        "smoke_test_runtime_seconds": round(total_time, 2),
        "train_telemetry": train_telemetry,
    }

    print(f"Smoke Test PASSED in {total_time:.2f}s | Peak VRAM: {peak_vram_mb:.2f} MB")
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} | Val Loss: {smoke_results['mean_val_loss']:.4f}")

    out_path = r"research/results/phase3_smoke_test.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2)
    print(f"Saved smoke test report to {out_path}")
    return smoke_results


if __name__ == "__main__":
    run_smoke_test()
