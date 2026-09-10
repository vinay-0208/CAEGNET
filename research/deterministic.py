"""
Deterministic Seeding Utility for CAEG-Net Research Track
=========================================================
Ensures bitwise reproducibility across PyTorch, NumPy, Python standard library,
and CuDNN backend execution where supported.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> torch.Generator:
    """
    Seed all random number generators across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, forces CuDNN deterministic behavior and
                             disables CuDNN benchmarking.

    Returns:
        torch.Generator configured with the seed for DataLoader usage.
    """
    # 1. Python standard library RNG & hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy RNG
    np.random.seed(seed)

    # 3. PyTorch CPU & CUDA RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. CuDNN Determinism
    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 5. Create and return seeded PyTorch Generator
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int):
    """
    Worker initialization function for DataLoader to ensure each worker thread
    has a deterministic, distinct seed derived from the PyTorch initial seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_deterministic_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Factory function creating a DataLoader with guaranteed deterministic shuffling.

    Args:
        dataset: PyTorch Dataset or TensorDataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle batches.
        seed: Seed to initialize the loader's dedicated Generator.
        num_workers: Number of subprocesses for data loading.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        Deterministic DataLoader instance.
    """
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        num_workers=num_workers,
        drop_last=drop_last,
    )
