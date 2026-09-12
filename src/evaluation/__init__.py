"""
CAEG-Net Evaluation Package
"""
from .metrics import compute_metrics, evaluate_model_on_loader
from .statistics import compute_daily_block_statistics

__all__ = [
    "compute_metrics",
    "evaluate_model_on_loader",
    "compute_daily_block_statistics",
]
