"""
CAEG-Net Neural Architecture Module
===================================
Context-Adaptive Expert Gating Network for Short-Term Electricity Load Forecasting.

Components:
1. LSTMExpert: 2-layer recurrent network capturing persistent temporal dependencies.
2. TCNExpert: Dilated causal temporal convolutional network with residual connections.
3. CNNExpert: Multi-stage 1D convolutional network extracting localized temporal motifs.
4. ContextFeatureEncoder: Non-linear MLP mapping 4D context vector C into latent representation e_C.
5. ContextGatingNetwork: MLP + Softmax producing dynamic expert routing weights [w_LSTM, w_TCN, w_CNN].
6. CAEGNet: Complete ensemble with dynamic weighted fusion:
   Y_hat = w_LSTM * Y_hat_LSTM + w_TCN * Y_hat_TCN + w_CNN * Y_hat_CNN

Strict Architectural Constraints:
- Lookback: 168 hours, Input shape: [B, 168, 1]
- Horizon: 24 hours, Target shape: [B, 24]
- Context: 4 features [Trend, Volatility, Periodicity, Recent_Error], Shape: [B, 4]
- Softmax weights sum to 1.0 per sample: sum_i w_i = 1.0, w_i > 0
- NO direct residual output correction (e.g., no Y_hat - constant * Recent_Error).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. LSTM Forecasting Expert
# =====================================================================

class LSTMExpert(nn.Module):
    """
    Recurrent forecasting expert with 2 stacked LSTM layers and fully-connected head.
    Captures long-range temporal autoregressive dependencies and diurnal drift.

    Input: [B, 168, 1] (batch, lookback, input_dim)
    Output: [B, 24] (batch, horizon)
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
        """
        Forward pass.
        x: [B, 168, 1]
        Returns: [B, 24]
        """
        # lstm_out: [B, 168, hidden_dim]
        lstm_out, _ = self.lstm(x)
        # Pool the final hidden state
        last_step = lstm_out[:, -1, :]  # [B, hidden_dim]
        out = self.fc_head(last_step)   # [B, 24]
        return out


# =====================================================================
# 2. TCN (Temporal Convolutional Network) Forecasting Expert
# =====================================================================

class CausalConv1dBlock(nn.Module):
    """
    Dilated Causal 1D Convolutional Residual Block.
    Enforces causality: output at time t depends strictly on inputs <= t.
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
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual shortcut matching channel dimension
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, L]
        """
        # Conv 1 + causal truncation
        y = self.conv1(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]  # Truncate right padding for causal guarantee
        y = self.relu1(self.bn1(y))
        y = self.dropout1(y)

        # Conv 2 + causal truncation
        y = self.conv2(y)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.bn2(y)
        y = self.dropout2(y)

        res = self.residual_conv(x)
        return self.relu2(y + res)


class TCNExpert(nn.Module):
    """
    Temporal Convolutional Network expert with dilated causal residual blocks.
    Receptive field expands exponentially with dilation rates [1, 2, 4, 8, 16, 32].
    Receptive field = 1 + 2 * (3-1) * (1+2+4+8+16+32) = 253 > 168 hours.

    Input: [B, 168, 1]
    Output: [B, 24]
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
        self.channels = channels
        self.horizon = horizon

        blocks = []
        in_c = input_dim
        for d in dilations:
            blocks.append(
                CausalConv1dBlock(
                    in_channels=in_c,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_c = channels
        self.network = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 168, 1] -> transpose to [B, 1, 168]
        Returns: [B, 24]
        """
        # [B, 168, 1] -> [B, 1, 168]
        x_trans = x.transpose(1, 2)
        features = self.network(x_trans)  # [B, channels, 168]
        # Pool the final causal time step (representing lookback ending at t)
        last_step = features[:, :, -1]    # [B, channels]
        out = self.head(last_step)        # [B, 24]
        return out


# =====================================================================
# 3. CNN Forecasting Expert
# =====================================================================

class CNNExpert(nn.Module):
    """
    Multi-stage 1D Convolutional expert with varying kernel sizes and adaptive pooling.
    Specialized for capturing localized temporal motifs, demand spikes, and peak ramps.

    Input: [B, 168, 1]
    Output: [B, 24]
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Stage 1: Local motif detection
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 168 -> 84

            # Stage 2: Intermediate pattern integration
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 84 -> 42

            # Stage 3: High-level abstract feature map
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # 42 -> 1
        )

        self.head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 168, 1] -> transpose to [B, 1, 168]
        Returns: [B, 24]
        """
        x_trans = x.transpose(1, 2)
        features = self.conv_stack(x_trans)  # [B, 64, 1]
        flat = features.squeeze(-1)          # [B, 64]
        out = self.head(flat)                # [B, 24]
        return out


# =====================================================================
# 4. Context Feature Encoder
# =====================================================================

class ContextFeatureEncoder(nn.Module):
    """
    Non-linear Multi-Layer Perceptron projecting the 4-dimensional domain context:
    C = [Trend, Volatility, Periodicity, Recent_Error] in R^4
    into a continuous latent context embedding e_C in R^16.

    Input: [B, 4]
    Output: [B, 16]
    """
    def __init__(
        self,
        context_dim: int = 4,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(context_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: [B, 4]
        Returns: [B, 16]
        """
        return self.encoder(c)


# =====================================================================
# 5. Context-Adaptive Expert Gating Network
# =====================================================================

class ContextGatingNetwork(nn.Module):
    """
    Routing network mapping the latent context embedding e_C to 3 convex expert weights:
    W = Softmax(MLP(e_C)) in R^3, where W = [w_LSTM, w_TCN, w_CNN]

    Guarantees:
    - sum_{i=1}^3 w_i = 1.0 for each sample.
    - w_i > 0 for all i (strictly convex combination).
    - Fully differentiable with respect to context features and routing parameters.

    Input: [B, 16] (latent context embedding)
    Output: [B, 3] (routing weights)
    """
    def __init__(
        self,
        latent_dim: int = 16,
        num_experts: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, e_c: torch.Tensor) -> torch.Tensor:
        """
        e_c: [B, 16]
        Returns: [B, 3] routing weights summing to 1.0 along dim=-1
        """
        logits = self.routing_mlp(e_c)            # [B, 3]
        weights = F.softmax(logits, dim=-1)       # [B, 3]
        return weights


class HorizonContextGatingNetwork(nn.Module):
    """
    Horizon-dependent routing network mapping latent context embedding e_C to 24 x 3 convex expert weights:
    W = Softmax(MLP(e_C)) in R^[24, 3], where for each forecast horizon step h in [1..24]:
    sum_{i=1}^3 w_{h, i} = 1.0, and w_{h, i} > 0.

    Input: [B, 16] (latent context embedding)
    Output: [B, 24, 3] (horizon-dependent routing weights)
    """
    def __init__(
        self,
        latent_dim: int = 16,
        horizon: int = 24,
        num_experts: int = 3,
        hidden_dim: int = 48,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_experts = num_experts
        self.routing_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * num_experts),
        )

    def forward(self, e_c: torch.Tensor) -> torch.Tensor:
        """
        e_c: [B, 16]
        Returns: [B, 24, 3] routing weights summing to 1.0 along dim=-1 for each horizon step
        """
        B = e_c.shape[0]
        logits = self.routing_mlp(e_c).view(B, self.horizon, self.num_experts)  # [B, 24, 3]
        weights = F.softmax(logits, dim=-1)                                    # [B, 24, 3]
        return weights


# =====================================================================
# 6. Complete CAEG-Net Architecture
# =====================================================================

class CAEGNet(nn.Module):
    """
    CAEG-Net: Context-Adaptive Expert Gating Network for Short-Term Load Forecasting.

    Dynamic Fusion Equation (Standard / V1):
    Y_hat_CAEG = w_LSTM * Y_hat_LSTM + w_TCN * Y_hat_TCN + w_CNN * Y_hat_CNN

    Horizon-Dependent Dynamic Fusion Equation (V2):
    Y_hat_CAEG[h] = w_LSTM[h] * Y_hat_LSTM[h] + w_TCN[h] * Y_hat_TCN[h] + w_CNN[h] * Y_hat_CNN[h]
    for each horizon step h in {1, ..., 24}.

    Strict Research Principles:
    - No direct output residual error correction.
    - Context signal exclusively informs the gating mechanism.
    - Zero future leakage: Lookback=168, Horizon=24, Context=4 (or 5).
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        context_dim: int = 4,
        latent_context_dim: int = 16,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        tcn_channels: int = 32,
        dropout: float = 0.1,
        horizon_dependent: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.context_dim = context_dim
        self.horizon_dependent = horizon_dependent

        # 1. Specialized Forecasting Experts
        self.lstm_expert = LSTMExpert(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            horizon=horizon,
            dropout=dropout,
        )
        self.tcn_expert = TCNExpert(
            input_dim=input_dim,
            channels=tcn_channels,
            dilations=(1, 2, 4, 8, 16, 32),
            kernel_size=3,
            horizon=horizon,
            dropout=dropout,
        )
        self.cnn_expert = CNNExpert(
            input_dim=input_dim,
            horizon=horizon,
            dropout=dropout,
        )

        # 2. Context Encoder & Gating Network
        self.context_encoder = ContextFeatureEncoder(
            context_dim=context_dim,
            latent_dim=latent_context_dim,
        )
        if horizon_dependent:
            self.gating_network = HorizonContextGatingNetwork(
                latent_dim=latent_context_dim,
                horizon=horizon,
                num_experts=3,
                hidden_dim=48,
                dropout=dropout,
            )
        else:
            self.gating_network = ContextGatingNetwork(
                latent_dim=latent_context_dim,
                num_experts=3,
                hidden_dim=32,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward Pass of CAEG-Net.

        Parameters:
        -----------
        x: torch.Tensor, shape [B, 168, 1] (Lookback sequence)
        c: torch.Tensor, shape [B, context_dim] (Context vector)
        return_diagnostics: bool (If True, returns predictions, gating weights, and expert outputs)

        Returns:
        --------
        If return_diagnostics is True:
            (y_pred, weights, expert_predictions)
            - y_pred: [B, 24] (CAEG-Net fused prediction)
            - weights: [B, 3] or [B, 24, 3] (Softmax routing weights)
            - expert_predictions: dict with keys "lstm", "tcn", "cnn", each [B, 24]
        Else:
            y_pred: [B, 24]
        """
        # 1. Compute Expert Forecasts
        y_lstm = self.lstm_expert(x)  # [B, 24]
        y_tcn = self.tcn_expert(x)    # [B, 24]
        y_cnn = self.cnn_expert(x)    # [B, 24]

        # 2. Compute Routing Weights from Context Vector
        e_c = self.context_encoder(c)      # [B, latent_dim]
        weights = self.gating_network(e_c) # [B, 3] or [B, 24, 3]

        # 3. Dynamic Weighted Fusion
        if self.horizon_dependent:
            # weights: [B, 24, 3]
            y_pred = (
                weights[:, :, 0] * y_lstm
                + weights[:, :, 1] * y_tcn
                + weights[:, :, 2] * y_cnn
            )  # [B, 24]
        else:
            # weights: [B, 3]
            y_pred = (
                weights[:, 0:1] * y_lstm
                + weights[:, 1:2] * y_tcn
                + weights[:, 2:3] * y_cnn
            )  # [B, 24]

        if return_diagnostics:
            expert_dict = {
                "lstm": y_lstm,
                "tcn": y_tcn,
                "cnn": y_cnn,
            }
            return y_pred, weights, expert_dict
        return y_pred


# =====================================================================
# 7. Standard Input-Based MoE (Baseline Model)
# =====================================================================

class StandardInputMoE(nn.Module):
    """
    Standard Input-Based Mixture-of-Experts Baseline.

    Contrasts with CAEG-Net:
    - Same three specialized forecasting experts (LSTM, TCN, CNN).
    - Gating network receives uncurated, raw input sequence X [B, 168, 1] instead of
      explicit domain context features C [B, 4].
    - Zero access to Trend, Volatility, Periodicity, or Recent Error.
    - Softmax routing weights: [w_LSTM, w_TCN, w_CNN], sum = 1.0.
    """
    def __init__(
        self,
        input_dim: int = 1,
        lookback: int = 168,
        horizon: int = 24,
        lstm_hidden: int = 64,
        tcn_channels: int = 32,
        gate_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon

        # Same 3 experts as CAEG-Net
        self.lstm_expert = LSTMExpert(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            horizon=horizon,
            dropout=dropout,
        )
        self.tcn_expert = TCNExpert(
            input_dim=input_dim,
            channels=tcn_channels,
            horizon=horizon,
            dropout=dropout,
        )
        self.cnn_expert = CNNExpert(
            input_dim=input_dim,
            horizon=horizon,
            dropout=dropout,
        )

        # Input-based gate: projects raw lookback sequence directly
        self.gate_mlp = nn.Sequential(
            nn.Linear(lookback * input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,  # c is ignored by design
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        x: [B, 168, 1]
        c: Ignored (demonstrating conventional input gating)
        """
        y_lstm = self.lstm_expert(x)
        y_tcn = self.tcn_expert(x)
        y_cnn = self.cnn_expert(x)

        # Flatten input lookback for input gate
        x_flat = x.view(x.size(0), -1)  # [B, 168]
        logits = self.gate_mlp(x_flat)  # [B, 3]
        weights = F.softmax(logits, dim=-1)

        y_pred = (
            weights[:, 0:1] * y_lstm
            + weights[:, 1:2] * y_tcn
            + weights[:, 2:3] * y_cnn
        )

        if return_diagnostics:
            return y_pred, weights, {"lstm": y_lstm, "tcn": y_tcn, "cnn": y_cnn}
        return y_pred


# =====================================================================
# 7. Model Diagnostic & Parameter Counting Utilities
# =====================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters for each submodule and entire model.
    """
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    submodule_counts = {}
    for name, module in model.named_children():
        sub_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        submodule_counts[name] = sub_trainable

    return {
        "total_trainable": total_trainable,
        "total_params": total_params,
        "submodules": submodule_counts,
    }


def print_architecture_summary(model: CAEGNet):
    """
    Print formatted architecture diagnostic report.
    """
    counts = count_parameters(model)
    print("=" * 70)
    print("CAEG-NET NEURAL ARCHITECTURE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Total Trainable Parameters: {counts['total_trainable']:,}")
    print(f"Total Model Parameters    : {counts['total_params']:,}")
    print("-" * 70)
    print(f"{'Component':30s} | {'Trainable Parameters':>20s} | {'Share (%)':>10s}")
    print("-" * 70)
    for name, sub_cnt in counts["submodules"].items():
        pct = (sub_cnt / counts["total_trainable"]) * 100.0 if counts["total_trainable"] > 0 else 0.0
        print(f"{name:30s} | {sub_cnt:20,d} | {pct:9.2f}%")
    print("=" * 70)


# Canonical V1 Alias
CAEGNetV1 = CAEGNet
