"""
CAEG-Net: Context-Adaptive Expert Gating Network
================================================
A parameter-efficient (121,724 parameters) heterogeneous temporal mixture-of-experts
model for 24-hour day-ahead short-term electricity load forecasting.
"""

try:
    from .models import CAEGNet, ConfidenceFallbackCAEGNet, LSTMExpert, TCNExpert, CNNExpert
except ImportError:
    pass
from .utils import count_parameters, set_seed, load_config

__version__ = "1.0.0"
__all__ = [
    "CAEGNet",
    "ConfidenceFallbackCAEGNet",
    "LSTMExpert",
    "TCNExpert",
    "CNNExpert",
    "count_parameters",
    "set_seed",
    "load_config",
]
