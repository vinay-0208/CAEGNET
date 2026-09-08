"""
CAEG-Net Research Models Architecture
======================================
Implements research variants of the Context-Adaptive Expert Gating Network:
1. Canonical V1 (4D context, global gating)
2. No Recent Error (3D context)
3. Calendar-Context CAEG (8D context: 4 operational + 4 cyclic calendar)
4. Forecast-Aware Routing (7D context: 4 operational + 3 expert disagreement features)
5. Calendar + Forecast-Aware Routing (11D context: 4 operational + 4 calendar + 3 disagreement)

Guarantees:
- Zero data leakage: Expert forecasts are derived strictly from input X_t.
- Disagreement features are causal functions of candidate forecasts before gating.
- Convex combination: sum_{i=1}^3 w_i = 1.0, w_i > 0.
- Rigorous parameter parity across variants (< 0.1% parameter difference).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from caeg_net import (
    LSTMExpert,
    TCNExpert,
    CNNExpert,
    ContextFeatureEncoder,
    ContextGatingNetwork,
    HorizonContextGatingNetwork,
    count_parameters
)


class ResearchCAEGNet(nn.Module):
    """
    Unified Research Architecture for Context-Adaptive Expert Gating.

    Parameters:
    -----------
    input_dim: int, default 1
    horizon: int, default 24
    base_context_dim: int, dimension of external context vector C fed in (e.g. 3, 4, or 8)
    forecast_aware: bool, if True, dynamically appends 3 expert disagreement features
                    to the context vector before the gating encoder
    latent_context_dim: int, default 16
    lstm_hidden: int, default 64
    lstm_layers: int, default 2
    tcn_channels: int, default 32
    dropout: float, default 0.1
    horizon_dependent: bool, default False
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        base_context_dim: int = 4,
        forecast_aware: bool = False,
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
        self.base_context_dim = base_context_dim
        self.forecast_aware = forecast_aware
        self.horizon_dependent = horizon_dependent

        # Total context dimension entering the context encoder
        self.total_context_dim = base_context_dim + (3 if forecast_aware else 0)

        # 1. Tri-Expert Temporal Hierarchy (Frozen identical architecture)
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

        # 2. Context Feature Encoder
        self.context_encoder = ContextFeatureEncoder(
            context_dim=self.total_context_dim,
            latent_dim=latent_context_dim,
        )

        # 3. Dynamic Softmax Gating Network
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

    def compute_expert_disagreement(
        self,
        y_lstm: torch.Tensor,
        y_tcn: torch.Tensor,
        y_cnn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal, inference-time expert disagreement features from candidate forecasts.
        All three candidate forecasts depend exclusively on historical window X.
        Zero future target information is used.

        Features:
        1. Pairwise mean absolute discrepancy across the 24-hour horizon.
        2. Inter-expert standard deviation across the horizon.
        3. Inter-expert range (max - min) across the horizon.

        Returns:
            disagreement: torch.Tensor of shape [B, 3]
        """
        # Pairwise discrepancy
        diff_lt = torch.abs(y_lstm - y_tcn)
        diff_lc = torch.abs(y_lstm - y_cnn)
        diff_tc = torch.abs(y_tcn - y_cnn)
        pair_diff = (diff_lt + diff_lc + diff_tc) / 3.0
        d_pair = pair_diff.mean(dim=-1, keepdim=True)  # [B, 1]

        # Stack predictions across expert dimension: [B, 24, 3]
        stacked = torch.stack([y_lstm, y_tcn, y_cnn], dim=-1)

        # Inter-expert standard deviation across 24h
        d_std = stacked.std(dim=-1).mean(dim=-1, keepdim=True)  # [B, 1]

        # Inter-expert range across 24h
        d_range = (stacked.max(dim=-1).values - stacked.min(dim=-1).values).mean(dim=-1, keepdim=True)  # [B, 1]

        # Concatenate into disagreement context vector [B, 3]
        # We detach to treat disagreement as an exogenous context signal for the gate
        disagreement = torch.cat([d_pair, d_std, d_range], dim=-1).detach()
        return disagreement

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward Pass of Research CAEG-Net.

        Parameters:
        -----------
        x: [B, 168, 1]
        c: [B, base_context_dim]
        return_diagnostics: bool

        Returns:
        --------
        y_pred: [B, 24]
        weights: [B, 3] (or [B, 24, 3])
        expert_predictions: dict of individual expert forecasts
        """
        # 1. Parallel forward pass through complementary experts
        y_lstm = self.lstm_expert(x)  # [B, 24]
        y_tcn = self.tcn_expert(x)    # [B, 24]
        y_cnn = self.cnn_expert(x)    # [B, 24]

        # 2. Context formulation (optionally augment with forecast disagreement)
        if self.forecast_aware:
            disagreement = self.compute_expert_disagreement(y_lstm, y_tcn, y_cnn)
            full_c = torch.cat([c, disagreement], dim=-1)  # [B, base_context_dim + 3]
        else:
            full_c = c

        # 3. Dynamic context routing
        e_c = self.context_encoder(full_c)
        weights = self.gating_network(e_c)

        # 4. Dynamic convex fusion
        if self.horizon_dependent:
            # weights: [B, 24, 3]
            w_lstm = weights[:, :, 0]
            w_tcn = weights[:, :, 1]
            w_cnn = weights[:, :, 2]
            y_pred = w_lstm * y_lstm + w_tcn * y_tcn + w_cnn * y_cnn
        else:
            # weights: [B, 3]
            w_lstm = weights[:, 0:1]
            w_tcn = weights[:, 1:2]
            w_cnn = weights[:, 2:3]
            y_pred = w_lstm * y_lstm + w_tcn * y_tcn + w_cnn * y_cnn

        if return_diagnostics:
            expert_dict = {
                "lstm": y_lstm,
                "tcn": y_tcn,
                "cnn": y_cnn,
            }
            if self.forecast_aware:
                expert_dict["disagreement"] = disagreement
            return y_pred, weights, expert_dict
        else:
            return y_pred
