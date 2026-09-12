"""
CAEG-Net Data Pipeline Package
"""
from .loader import (
    generate_synthetic_load_data,
    load_and_clean_data,
    chronological_split,
    fit_and_transform_scaler,
)
from .dataset import (
    LoadDataset,
    create_sliding_windows,
    create_forecasting_windows,
    create_partition_windows_with_context,
    ChronologicalWalkForwardForecaster,
    compute_causal_recent_forecast_errors,
)

__all__ = [
    "generate_synthetic_load_data",
    "load_and_clean_data",
    "chronological_split",
    "fit_and_transform_scaler",
    "LoadDataset",
    "create_sliding_windows",
    "create_forecasting_windows",
    "create_partition_windows_with_context",
    "ChronologicalWalkForwardForecaster",
    "compute_causal_recent_forecast_errors",
]
