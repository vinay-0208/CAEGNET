"""
CAEG-Net Training Module
========================
Provides standardized training loops, loss functions, optimizer configurations,
early stopping callbacks, and checkpoint persistence.

Enforced Principles:
- Training loss: Standardized MSE.
- Checkpoint selection: Strictly driven by Validation MSE loss.
- Zero test data accessed during training or early stopping.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def create_criterion(loss_type: str = "mse") -> nn.Module:
    """
    Construct loss function on standardized target values.
    """
    if loss_type.lower() == "mse":
        return nn.MSELoss()
    elif loss_type.lower() == "mae" or loss_type.lower() == "l1":
        return nn.L1Loss()
    elif loss_type.lower() == "huber":
        return nn.HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    step_size: int = 15,
    gamma: float = 0.5,
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """
    Configure AdamW optimizer and StepLR scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    max_epochs: int = 50,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, Union[nn.Module, List[float], int, float]]:
    """
    Train a forecasting model with early stopping on validation loss.

    Parameters:
    -----------
    model: nn.Module
    train_loader: DataLoader yielding (bx, by, bc) or (bx, by)
    val_loader: DataLoader yielding (bx, by, bc) or (bx, by)
    criterion: nn.Module (MSELoss)
    optimizer: optim.Optimizer
    scheduler: Optional learning rate scheduler
    max_epochs: Maximum training epochs
    patience: Early stopping patience epochs
    checkpoint_path: Path to save the best model weights
    device: torch.device (default CPU)
    model_name: Display name for logging
    verbose: Whether to print progress per epoch

    Returns:
    --------
    Dict containing best model, train/val loss curves, best epoch, and runtime.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    start_time = time.time()

    if verbose:
        print(f"[{model_name}] Starting training on {device} (Max Epochs: {max_epochs}, Patience: {patience})")

    for epoch in range(1, max_epochs + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if len(batch) == 3:
                bx, by, bc = batch
                bx = bx.to(device)
                by = by.to(device)
                bc = bc.to(device)
                # Forward pass: supports models taking (x, c) or only (x)
                try:
                    out = model(bx, bc, return_diagnostics=False)
                except TypeError:
                    try:
                        out = model(bx, bc)
                    except TypeError:
                        out = model(bx)
            else:
                bx, by = batch
                bx = bx.to(device)
                by = by.to(device)
                out = model(bx)

            # Extract prediction tensor if tuple is returned
            if isinstance(out, tuple):
                y_pred = out[0]
            else:
                y_pred = out

            loss = criterion(y_pred, by)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / max(num_train_batches, 1)
        train_losses.append(avg_train_loss)

        if scheduler is not None:
            scheduler.step()

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    bx, by, bc = batch
                    bx = bx.to(device)
                    by = by.to(device)
                    bc = bc.to(device)
                    try:
                        out = model(bx, bc, return_diagnostics=False)
                    except TypeError:
                        try:
                            out = model(bx, bc)
                        except TypeError:
                            out = model(bx)
                else:
                    bx, by = batch
                    bx = bx.to(device)
                    by = by.to(device)
                    out = model(bx)

                if isinstance(out, tuple):
                    y_pred = out[0]
                else:
                    y_pred = out

                val_loss = criterion(y_pred, by)
                running_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = running_val_loss / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1 or avg_val_loss < best_val_loss):
            print(
                f"  Epoch {epoch:02d}/{max_epochs:02d} | "
                f"Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}"
                f"{'  * Best *' if avg_val_loss < best_val_loss else ''}"
            )

        # Check early stopping improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"[{model_name}] Early stopping triggered at Epoch {epoch} (Best Epoch: {best_epoch}, Val MSE: {best_val_loss:.5f})")
                break

    total_time = time.time() - start_time

    # Restore best checkpoint if saved
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if verbose:
            print(f"[{model_name}] Restored best checkpoint from Epoch {best_epoch} ({checkpoint_path})")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
    }
