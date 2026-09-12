"""
Parameter Inspection & Architecture Utilities
"""
from typing import Dict
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    submodule_counts = {}
    for name, module in model.named_children():
        sub_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        submodule_counts[name] = sub_trainable

    return {
        "total_trainable": total_trainable,
        "total_params": total_params,
        "submodules": submodule_counts,
    }


def print_architecture_summary(model: nn.Module):
    counts = count_parameters(model)
    print("=" * 70)
    print("CAEG-NET NEURAL ARCHITECTURE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Total Trainable Parameters: {counts['total_trainable']:,}")
    print(f"Total Model Parameters    : {counts['total_params']:,}")
    print("-" * 70)
    print(f"{'Component':30s} | {'Trainable Parameters':>20s} | {'Share (%)':>10s}")
    print("-" * 70)
    for name, sub_cnt in counts["submodules"].items():
        pct = (sub_cnt / counts["total_trainable"]) * 100.0 if counts["total_trainable"] > 0 else 0.0
        print(f"{name:30s} | {sub_cnt:20,d} | {pct:9.2f}%")
    print("=" * 70)
