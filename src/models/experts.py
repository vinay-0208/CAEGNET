"""
CAEG-Net Specialized Temporal Experts
=====================================
Contains the three complementary temporal neural feature extractors:
1. LSTMExpert: 2-layer recurrent sequence model capturing persistent diurnal baselines (56,152 parameters).
2. TCNExpert: 6-stage dilated causal residual convolutional network with 253h receptive field (36,952 parameters).
3. CNNExpert: 3-stage multi-kernel 1D convolutional feature extractor for localized ramp patterns (27,400 parameters).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMExpert(nn.Module):
    """
    Recurrent forecasting expert with 2 stacked LSTM layers and fully-connected head.
    Captures long-range temporal autoregressive dependencies and diurnal drift.

    Input: [B, 168, 1] (batch, lookback, input_dim)
    Output: [B, 24] (batch, horizon)
    Parameters: 56,152
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc_head(last_step)
        return out


class CausalConv1dBlock(nn.Module):
    """
    Dilated Causal 1D Convolutional Residual Block.
    Enforces strict causality: output at time t depends strictly on inputs <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_conv(x)
        y = self.conv1(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.dropout1(self.relu1(self.bn1(y)))

        y = self.conv2(y)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.dropout2(self.relu2(self.bn2(y)))

        out = F.relu(y + res)
        return out


class TCNExpert(nn.Module):
    """
    Temporal Convolutional Network Expert.
    6 residual blocks with exponential dilation schedule d in {1, 2, 4, 8, 16, 32}.
    Receptive field: 1 + 2 * (3-1) * (1+2+4+8+16+32) = 253 hours (>168h lookback).
    Parameters: 36,952
    """
    def __init__(
        self,
        input_dim: int = 1,
        channels: int = 32,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.dilations = dilations
        self.horizon = horizon

        blocks = []
        for i, d in enumerate(dilations):
            in_c = input_dim if i == 0 else channels
            blocks.append(
                CausalConv1dBlock(
                    in_channels=in_c,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.fc_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)  # [B, 1, 168]
        features = self.network(x_trans)  # [B, 32, 168]
        last_step = features[:, :, -1]   # [B, 32]
        out = self.fc_head(last_step)     # [B, 24]
        return out


class CNNExpert(nn.Module):
    """
    1D Convolutional Neural Network Expert.
    Multi-stage localized temporal feature extractor detecting ramp and peak motifs.
    Conv1D(1->32, k=3) -> Conv1D(32->64, k=5) -> Conv1D(64->64, k=3) -> AdaptiveAvgPool -> Head.
    Parameters: 27,400
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = x.transpose(1, 2)       # [B, 1, 168]
        conv_out = self.conv_layers(x_trans)  # [B, 64, 1]
        pooled = conv_out.squeeze(-1)     # [B, 64]
        out = self.fc_head(pooled)        # [B, 24]
        return out
