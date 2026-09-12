"""
Causal Domain-Informed Context Feature Extraction
=================================================
Extracts physical context features strictly at or before forecast origin t:
1. Trend Slope (OLS regression slope over lookback)
2. Short-Term Volatility (sample standard deviation of first differences)
3. Lag-24 Diurnal Autocorrelation (diurnal rhythmicity)
4. Causal Recent Forecast Error (MAE of completed historical forecast)
"""
from typing import Optional
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
    X_windows: np.ndarray,
    recent_errors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract causal 4-dimensional context vector for each lookback window:
    C_t = [Trend, Volatility, Periodicity, Recent Error] in R^4
    """
    if X_windows.ndim == 3:
        Z_lookback = X_windows[:, :, 0]
    elif X_windows.ndim == 2:
        Z_lookback = X_windows
    else:
        raise ValueError("X_windows must be 2D [N, L] or 3D [N, L, 1]")

    N, L = Z_lookback.shape
    assert L >= 48, f"Lookback length {L} must be at least 48 for lag-24 autocorrelation"

    i_indices = np.arange(L, dtype=np.float64)
    i_bar = (L - 1.0) / 2.0
    w_i = i_indices - i_bar
    sum_w_sq = np.sum(w_i ** 2)

    # 1. Trend
    z_bar = np.mean(Z_lookback, axis=1, keepdims=True)
    z_centered = Z_lookback - z_bar
    beta_1 = np.sum(z_centered * w_i, axis=1) / sum_w_sq
    z_std = np.std(Z_lookback, axis=1, ddof=1)
    trend = beta_1 / (z_std + 1e-6)

    # 2. Volatility: Sample standard deviation of first differences (ddof=1)
    diffs = Z_lookback[:, 1:] - Z_lookback[:, :-1]
    volatility = np.std(diffs, axis=1, ddof=1)

    # 3. Periodicity: Lag-24 sample autocorrelation
    var_denom = np.sum(z_centered ** 2, axis=1)
    autocorr_num = np.sum(z_centered[:, 24:] * z_centered[:, :-24], axis=1)
    periodicity = autocorr_num / (var_denom + 1e-6)

    # 4. Recent Error
    if recent_errors is not None:
        rec_err = np.asarray(recent_errors, dtype=np.float32).flatten()
        if len(rec_err) != N:
            raise ValueError(f"recent_errors length ({len(rec_err)}) must match number of windows ({N})")
    else:
        rec_err = np.full(N, 0.35, dtype=np.float32)

    C = np.column_stack([trend, volatility, periodicity, rec_err]).astype(np.float32)
    return C
