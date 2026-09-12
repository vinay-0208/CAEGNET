"""
PyTorch Dataset and Sliding Window Generation
"""
from typing import Tuple
import numpy as np
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
