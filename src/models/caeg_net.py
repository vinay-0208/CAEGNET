"""
CAEG-Net Architecture Formulations
==================================
Contains:
1. CAEGNet: Complete ensemble with dynamic weighted fusion (121,579 params).
2. ConfidenceFallbackCAEGNet: Final locked champion model (F2 / A2-OOF, 121,724 params)
   incorporating dynamic confidence shrinkage toward equal-expert centroid.
3. StandardInputMoE: Conventional raw-input mixture-of-experts baseline (126,011 params).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import LSTMExpert, TCNExpert, CNNExpert
from .router import ContextFeatureEncoder, ContextGatingNetwork, HorizonContextGatingNetwork


class CAEGNet(nn.Module):
    """
    CAEG-Net: Context-Adaptive Expert Gating Network for Short-Term Load Forecasting.
    Dynamic Fusion: Y_hat = w_LSTM * Y_hat_LSTM + w_TCN * Y_hat_TCN + w_CNN * Y_hat_CNN
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
        y_lstm = self.lstm_expert(x)
        y_tcn = self.tcn_expert(x)
        y_cnn = self.cnn_expert(x)

        e_c = self.context_encoder(c)
        weights = self.gating_network(e_c)

        if self.horizon_dependent:
            y_pred = (
                weights[:, :, 0] * y_lstm
                + weights[:, :, 1] * y_tcn
                + weights[:, :, 2] * y_cnn
            )
        else:
            y_pred = (
                weights[:, 0:1] * y_lstm
                + weights[:, 1:2] * y_tcn
                + weights[:, 2:3] * y_cnn
            )

        if return_diagnostics:
            expert_dict = {
                "lstm": y_lstm,
                "tcn": y_tcn,
                "cnn": y_cnn,
            }
            return y_pred, weights, expert_dict
        return y_pred

    def get_gating_weights(self, c: torch.Tensor) -> torch.Tensor:
        e_c = self.context_encoder(c)
        return self.gating_network(e_c)


class ConfidenceFallbackCAEGNet(nn.Module):
    """
    Final Locked CAEG-Net Architecture (F2 / A2-OOF, 121,724 Parameters).
    Combines 3-expert backbone (120,504 params), context router (1,075 params),
    and confidence fallback shrinkage head (145 params):
        y_final = lambda * y_adaptive + (1 - lambda) * y_equal
        where lambda = Sigmoid(MLP(c)) in (0, 1), empirically lambda approx 0.51
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        context_dim: int = 7,
        latent_context_dim: int = 16,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        tcn_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.context_dim = context_dim

        self.caeg = CAEGNet(
            input_dim=input_dim,
            horizon=horizon,
            context_dim=context_dim,
            latent_context_dim=latent_context_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            tcn_channels=tcn_channels,
            dropout=dropout,
            horizon_dependent=False,
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        y_adapt, w, diag = self.caeg(x, c, return_diagnostics=True)
        y_l = diag["lstm"]
        y_t = diag["tcn"]
        y_c = diag["cnn"]
        y_equal = (y_l + y_t + y_c) / 3.0

        lam = self.confidence_head(c)
        y_final = lam * y_adapt + (1.0 - lam) * y_equal

        if return_diagnostics:
            diag["lambda"] = lam
            diag["y_adaptive"] = y_adapt
            diag["y_equal"] = y_equal
            return y_final, w, diag
        return y_final

    def get_gating_weights(self, c: torch.Tensor) -> torch.Tensor:
        return self.caeg.get_gating_weights(c)


class StandardInputMoE(nn.Module):
    """
    Standard Input-Based Mixture-of-Experts Baseline.
    Gating receives raw lookback X [B, 168, 1] instead of explicit domain context C.
    Total Parameters: 126,011
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

        self.gate_mlp = nn.Sequential(
            nn.Linear(lookback * input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        y_lstm = self.lstm_expert(x)
        y_tcn = self.tcn_expert(x)
        y_cnn = self.cnn_expert(x)

        x_flat = x.view(x.size(0), -1)
        logits = self.gate_mlp(x_flat)
        weights = F.softmax(logits, dim=-1)

        y_pred = (
            weights[:, 0:1] * y_lstm
            + weights[:, 1:2] * y_tcn
            + weights[:, 2:3] * y_cnn
        )

        if return_diagnostics:
            return y_pred, weights, {"lstm": y_lstm, "tcn": y_tcn, "cnn": y_cnn}
        return y_pred
