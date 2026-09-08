"""
Original CAEG-Net Architecture & Phase 5 Research Extensions
============================================================
Preserves and enhances the canonical original CAEG-Net expert family:
1. LSTMExpert (2-layer recurrent network, 56,152 params)
2. TCNExpert (6-stage causal dilated residual conv, 36,952 params)
3. CNNExpert (3-stage 1D convolutional motif detector, 27,400 params)
4. ContextFeatureEncoder (4D -> 16D MLP, 384 params)
5. ContextGatingNetwork (16D -> 3D Softmax router, 643 params)
Total Canonical V1 Parameters: 121,531.

Phase 5 Controlled Ablation Extensions (Transferring Validated Hypotheses):
- Group A: Training stabilization (gradient clipping, plateau scheduler)
- Group B: Context features (weekly lag-168 profile corr, 48h range ratio)
- Group C: Inter-expert disagreement difficulty signal (detached D_t in R^3)
- Group D: Conservative routing (w = (1 - rho) * w0 + rho * q, w0 = [1/3, 1/3, 1/3])
- Group E: Routing regularization (entropy penalty or KL divergence to uniform)
- Group F: Auxiliary expert supervision (L_total = L_fused + lambda_aux * L_expert_aux)
- Group G: Horizon-aware routing (w in R^[24, 3])
"""

from typing import Dict, List, Optional, Tuple, Union
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
    CAEGNet as CAEGNetV1,
    StandardInputMoE,
    count_parameters,
)


class OriginalCAEGNetPhase5(nn.Module):
    """
    Phase 5 Canonical CAEG-Net with Research-Guided Extensions.
    Operates strictly on the original LSTM + TCN + CNN expert family.
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
        use_disagreement: bool = False,
        conservative_rho: Optional[float] = None,
        horizon_dependent: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.context_dim = context_dim
        self.use_disagreement = use_disagreement
        self.conservative_rho = conservative_rho
        self.horizon_dependent = horizon_dependent
        self.temperature = float(temperature)

        # 1. Canonical Experts (LSTM, TCN, CNN)
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
        # If disagreement is used, 3 detached disagreement features are appended to context
        total_context_dim = context_dim + (3 if use_disagreement else 0)
        self.context_encoder = nn.Sequential(
            nn.Linear(total_context_dim, latent_context_dim),
            nn.LayerNorm(latent_context_dim),
            nn.ReLU(),
            nn.Linear(latent_context_dim, latent_context_dim),
            nn.ReLU(),
        )

        # 3. Router Network
        if horizon_dependent:
            self.router = nn.Sequential(
                nn.Linear(latent_context_dim, 48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(48, horizon * 3),
            )
        else:
            self.router = nn.Sequential(
                nn.Linear(latent_context_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 3),
            )

        # Base equal weight prior: w0 = [1/3, 1/3, 1/3]
        w0 = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)
        self.register_buffer("w0", w0)

    def compute_expert_disagreement(
        self,
        y_lstm: torch.Tensor,
        y_tcn: torch.Tensor,
        y_cnn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute detached inter-expert disagreement vector in R^3:
        [mean_pairwise_abs_diff, std_dev, range]
        Strictly detached from autograd to prevent feedback corruption.
        """
        diff_12 = torch.abs(y_lstm - y_tcn)
        diff_13 = torch.abs(y_lstm - y_cnn)
        diff_23 = torch.abs(y_tcn - y_cnn)
        d_pair = ((diff_12 + diff_13 + diff_23) / 3.0).mean(dim=-1, keepdim=True)

        stacked = torch.stack([y_lstm, y_tcn, y_cnn], dim=-1)
        d_std = stacked.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)
        d_range = (stacked.max(dim=-1).values - stacked.min(dim=-1).values).mean(dim=-1, keepdim=True)

        d_vec = torch.cat([d_pair, d_std, d_range], dim=-1).detach()
        return d_vec

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho: Optional[float] = None,
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        tau = temperature if temperature is not None else self.temperature
        rho = conservative_rho if conservative_rho is not None else self.conservative_rho

        # 1. Expert Forward Passes
        y_lstm = self.lstm_expert(x)
        y_tcn = self.tcn_expert(x)
        y_cnn = self.cnn_expert(x)

        # 2. Context Construction (Optionally appending detached disagreement)
        if self.use_disagreement:
            disagreement = self.compute_expert_disagreement(y_lstm, y_tcn, y_cnn)
            c_input = torch.cat([c, disagreement], dim=-1)
        else:
            disagreement = None
            c_input = c

        # 3. Context Encoding & Routing
        e_c = self.context_encoder(c_input)
        B = x.shape[0]

        if self.horizon_dependent:
            raw_logits = self.router(e_c).view(B, self.horizon, 3)
            q = F.softmax(raw_logits / tau, dim=-1)  # [B, 24, 3]
            if rho is not None:
                w0 = self.w0.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
                weights = (1.0 - rho) * w0 + rho * q
            else:
                weights = q

            y_pred = (
                weights[:, :, 0] * y_lstm
                + weights[:, :, 1] * y_tcn
                + weights[:, :, 2] * y_cnn
            )
        else:
            raw_logits = self.router(e_c)  # [B, 3]
            q = F.softmax(raw_logits / tau, dim=-1)  # [B, 3]
            if rho is not None:
                w0 = self.w0.unsqueeze(0)  # [1, 3]
                weights = (1.0 - rho) * w0 + rho * q
            else:
                weights = q

            y_pred = (
                weights[:, 0:1] * y_lstm
                + weights[:, 1:2] * y_tcn
                + weights[:, 2:3] * y_cnn
            )

        if return_diagnostics:
            eps = 1e-8
            entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1)
            kl_div = torch.sum(weights * torch.log(3.0 * weights + eps), dim=-1)

            diag = {
                "expert_predictions": {"lstm": y_lstm, "tcn": y_tcn, "cnn": y_cnn},
                "weights": weights,
                "q_weights": q,
                "disagreement": disagreement,
                "entropy": entropy,
                "mean_entropy": entropy.mean(),
                "kl_div": kl_div,
                "mean_kl": kl_div.mean(),
            }
            return y_pred, weights, diag
        return y_pred


def compute_phase5_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expert_preds: Optional[Dict[str, torch.Tensor]] = None,
    weights: Optional[torch.Tensor] = None,
    lambda_aux: float = 0.0,
    beta_entropy: float = 0.0,
    lambda_kl: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standardized Phase 5 Tripartite Objective:
    L_total = L_fused + lambda_aux * L_expert_aux - beta_entropy * H(w) + lambda_kl * KL(w || w0)
    where L_expert_aux = (1/3) * (L_LSTM + L_TCN + L_CNN).
    """
    l_fused = F.mse_loss(y_pred, y_true)

    l_aux = torch.tensor(0.0, device=y_pred.device)
    if lambda_aux > 0.0 and expert_preds is not None:
        l_lstm = F.mse_loss(expert_preds["lstm"], y_true)
        l_tcn = F.mse_loss(expert_preds["tcn"], y_true)
        l_cnn = F.mse_loss(expert_preds["cnn"], y_true)
        l_aux = (l_lstm + l_tcn + l_cnn) / 3.0

    l_entropy = torch.tensor(0.0, device=y_pred.device)
    l_kl = torch.tensor(0.0, device=y_pred.device)
    if weights is not None:
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()
        l_entropy = entropy
        kl = torch.sum(weights * torch.log(3.0 * weights + eps), dim=-1).mean()
        l_kl = kl

    l_total = l_fused + lambda_aux * l_aux - beta_entropy * l_entropy + lambda_kl * l_kl

    telemetry = {
        "loss_total": float(l_total.item()),
        "loss_fused": float(l_fused.item()),
        "loss_aux": float(l_aux.item()) if isinstance(l_aux, torch.Tensor) else 0.0,
        "loss_entropy": float(l_entropy.item()) if isinstance(l_entropy, torch.Tensor) else 0.0,
        "loss_kl": float(l_kl.item()) if isinstance(l_kl, torch.Tensor) else 0.0,
    }
    return l_total, telemetry


class LearnedStaticEnsembleV1(nn.Module):
    """
    Learned Static Convex Ensemble for the original LSTM + TCN + CNN experts.
    w_i = Softmax(alpha_i / tau) (3 trainable scalar parameters).
    """
    def __init__(
        self,
        lstm_expert: nn.Module,
        tcn_expert: nn.Module,
        cnn_expert: nn.Module,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.lstm_expert = lstm_expert
        self.tcn_expert = tcn_expert
        self.cnn_expert = cnn_expert
        self.temperature = float(temperature)

        # Freeze expert backbones
        for expert in [self.lstm_expert, self.tcn_expert, self.cnn_expert]:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

        self.logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            y_lstm = self.lstm_expert(x)
            y_tcn = self.tcn_expert(x)
            y_cnn = self.cnn_expert(x)

        weights = F.softmax(self.logits / self.temperature, dim=0)
        y_pred = weights[0] * y_lstm + weights[1] * y_tcn + weights[2] * y_cnn
        return y_pred, weights
