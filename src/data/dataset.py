"""
PyTorch Dataset, Sliding Window Generation, and Causal Partition Windowing
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch
from torch.utils.data import Dataset, DataLoader


class LoadDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.c[idx]


def create_sliding_windows(
    series: np.ndarray,
    lookback: int = 168,
    horizon: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = len(series) - lookback - horizon + 1
    if num_samples <= 0:
        raise ValueError("Series length insufficient for lookback + horizon.")
    x = np.zeros((num_samples, lookback, 1), dtype=np.float32)
    y = np.zeros((num_samples, horizon), dtype=np.float32)
    for i in range(num_samples):
        x[i, :, 0] = series[i : i + lookback]
        y[i, :] = series[i + lookback : i + lookback + horizon]
    return x, y


def create_forecasting_windows(
    series: np.ndarray,
    lookback: int = 168,
    horizon: int = 24,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    series_1d = np.asarray(series, dtype=np.float32).flatten()
    n = len(series_1d)

    start_origin = lookback - 1
    end_origin = n - horizon - 1

    if end_origin < start_origin:
        raise ValueError(f"Series length {n} is insufficient for lookback={lookback} and horizon={horizon}")

    origin_indices = np.arange(start_origin, end_origin + 1, step)
    num_windows = len(origin_indices)

    X = np.zeros((num_windows, lookback, 1), dtype=np.float32)
    Y = np.zeros((num_windows, horizon), dtype=np.float32)

    for i, t in enumerate(origin_indices):
        X[i, :, 0] = series_1d[t - lookback + 1 : t + 1]
        Y[i, :] = series_1d[t + 1 : t + horizon + 1]

    return X, Y, origin_indices


def create_partition_windows_with_context(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 24,
    load_col: str = "load",
) -> Dict[str, Dict[str, np.ndarray]]:
    train_series = train_df[load_col].values
    X_train, Y_train, origins_train = create_forecasting_windows(
        train_series, lookback=lookback, horizon=horizon
    )

    history_for_val = train_series[-(lookback - 1):]
    val_series_extended = np.concatenate([history_for_val, val_df[load_col].values])
    X_val, Y_val, origins_val = create_forecasting_windows(
        val_series_extended, lookback=lookback, horizon=horizon
    )

    val_series = val_df[load_col].values
    history_for_test = val_series[-(lookback - 1):]
    test_series_extended = np.concatenate([history_for_test, test_df[load_col].values])
    X_test, Y_test, origins_test = create_forecasting_windows(
        test_series_extended, lookback=lookback, horizon=horizon
    )

    return {
        "train": {"X": X_train, "Y": Y_train, "origins": origins_train, "series": train_series},
        "val": {"X": X_val, "Y": Y_val, "origins": origins_val, "series": val_series_extended},
        "test": {"X": X_test, "Y": Y_test, "origins": origins_test, "series": test_series_extended},
    }


class ChronologicalWalkForwardForecaster:
    def __init__(self, alpha: float = 100.0, warmup: int = 500, update_step: int = 24):
        self.alpha = alpha
        self.warmup = warmup
        self.update_step = update_step
        self.final_deployment_model = Ridge(alpha=alpha)
        self.warmup_prior_mae: float = 0.35
        self.is_fitted = False

    def generate_walk_forward_train_predictions(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> np.ndarray:
        if X_train.ndim == 3:
            X_2d = X_train[:, :, 0]
        else:
            X_2d = X_train

        N = len(X_2d)
        oof_preds = np.zeros_like(Y_train)

        model = Ridge(alpha=self.alpha)

        for s in range(self.warmup, N, self.update_step):
            max_avail_idx = s - 24
            if max_avail_idx >= 100:
                model.fit(X_2d[:max_avail_idx], Y_train[:max_avail_idx])

            block_end = min(s + self.update_step, N)
            oof_preds[s:block_end] = model.predict(X_2d[s:block_end])

        self.final_deployment_model.fit(X_2d, Y_train)
        self.is_fitted = True

        cohort_end = min(self.warmup + 500, N)
        self.warmup_prior_mae = float(np.mean(np.abs(oof_preds[self.warmup:cohort_end] - Y_train[self.warmup:cohort_end])))
        return oof_preds

    def predict_deployment(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted on training partition before deployment.")
        if X.ndim == 3:
            X_2d = X[:, :, 0]
        else:
            X_2d = X
        return self.final_deployment_model.predict(X_2d).astype(np.float32)


def compute_causal_recent_forecast_errors(
    windows_dict: Dict[str, Dict[str, np.ndarray]],
    warmup: int = 500,
    update_step: int = 24,
    alpha: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ChronologicalWalkForwardForecaster]:
    forecaster = ChronologicalWalkForwardForecaster(alpha=alpha, warmup=warmup, update_step=update_step)

    X_tr = windows_dict["train"]["X"]
    Y_tr = windows_dict["train"]["Y"]
    oof_preds_train = forecaster.generate_walk_forward_train_predictions(X_tr, Y_tr)
    mae_train_windows = np.mean(np.abs(oof_preds_train - Y_tr), axis=1)

    preds_val = forecaster.predict_deployment(windows_dict["val"]["X"])
    mae_val_windows = np.mean(np.abs(preds_val - windows_dict["val"]["Y"]), axis=1)

    preds_test = forecaster.predict_deployment(windows_dict["test"]["X"])
    mae_test_windows = np.mean(np.abs(preds_test - windows_dict["test"]["Y"]), axis=1)

    n_tr = len(mae_train_windows)
    n_val = len(mae_val_windows)
    n_test = len(mae_test_windows)
    horizon = 24
    prior = forecaster.warmup_prior_mae

    rec_err_train = np.zeros(n_tr, dtype=np.float32)
    for t in range(n_tr):
        completed_origin = t - horizon
        if completed_origin >= warmup:
            rec_err_train[t] = mae_train_windows[completed_origin]
        else:
            rec_err_train[t] = prior

    rec_err_val = np.zeros(n_val, dtype=np.float32)
    for t in range(n_val):
        if t >= horizon:
            rec_err_val[t] = mae_val_windows[t - horizon]
        else:
            rec_err_val[t] = mae_train_windows[n_tr - horizon + t]

    rec_err_test = np.zeros(n_test, dtype=np.float32)
    for t in range(n_test):
        if t >= horizon:
            rec_err_test[t] = mae_test_windows[t - horizon]
        else:
            rec_err_test[t] = mae_val_windows[n_val - horizon + t]

    return rec_err_train, rec_err_val, rec_err_test, forecaster
