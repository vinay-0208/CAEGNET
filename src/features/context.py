"""
Causal Domain-Informed Context Extraction
=========================================
Extracts physical context features strictly at or before forecast origin t:
1. Trend Slope (168h linear regression)
2. Short-Term Volatility (std over 24h)
3. Lag-24 Diurnal Autocorrelation (Pearson r between [t-47:t-24] and [t-23:t])
4. Causal Recent Forecast Error (MAE of completed historical forecast)
5. Causal OOF Expert Residuals (LSTM, TCN, CNN relative performance)
"""

from typing import Tuple
import numpy as np


def compute_trend_slope(window: np.ndarray) -> float:
    t = len(window)
    x = np.arange(t)
    x_mean = (t - 1) / 2.0
    y_mean = np.mean(window)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    numer = np.sum((x - x_mean) * (window - y_mean))
    return float(numer / denom)


def compute_short_term_volatility(window: np.ndarray, last_k: int = 24) -> float:
    sub = window[-last_k:]
    return float(np.std(sub))


def compute_lag24_autocorrelation(window: np.ndarray) -> float:
    if len(window) < 48:
        return 0.0
    x1 = window[-48:-24]
    x2 = window[-24:]
    s1 = np.std(x1)
    s2 = np.std(x2)
    if s1 < 1e-8 or s2 < 1e-8:
        return 0.0
    r = np.corrcoef(x1, x2)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)


def extract_context_features(
    series: np.ndarray,
    lookback: int = 168,
    horizon: int = 24,
    recent_errors: np.ndarray = None,
) -> np.ndarray:
    num_samples = len(series) - lookback - horizon + 1
    c = np.zeros((num_samples, 4), dtype=np.float32)
    for i in range(num_samples):
        window = series[i : i + lookback]
        c[i, 0] = compute_trend_slope(window)
        c[i, 1] = compute_short_term_volatility(window, last_k=24)
        c[i, 2] = compute_lag24_autocorrelation(window)
        if recent_errors is not None and i < len(recent_errors):
            c[i, 3] = recent_errors[i]
        else:
            c[i, 3] = 0.0
    return c
