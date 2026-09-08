"""
CAEG-Net V2 Research Architecture Implementation
=================================================
Formally specified in:
- research/results/PHASE_2_RESEARCH_MODEL_SPECIFICATION.md
- research/results/PHASE_2_ARCHITECTURE_SCHEMA.json

Key Structural Guarantees:
1. Tri-Expert Inductive Complementarity:
   - Expert 1: GatedRecurrentExpert (2-layer GRU, hidden=54, autoregressive chronological tracking)
   - Expert 2: MultiScaleCausalTCNExpert (5-block dilated causal conv, c=34, k=4, RF=187h > 168h)
   - Expert 3: PatchTemporalExpert (13 daily patches, patch_len=24, stride=12, embed=48, bottleneck mixer)
2. Forecast-Aware Disagreement Routing:
   - 3 candidate forecasts generated strictly from historical lookback X_t in R^[168, 1].
   - Detached consensus metrics D_t in R^3 (pairwise MAE, std, range) concatenated with 6D context C_t.
   - Total router input u_t = [C_t || D_t] in R^9.
3. Regularized Convex Routing:
   - Context Encoder: Linear(9, 32) -> LayerNorm(32) -> ReLU -> Linear(32, 32) -> ReLU.
   - Router: Linear(32, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 3) -> Softmax(z / tau).
   - Temperature tau = 0.5 (candidates: {0.2, 0.5, 0.8, 1.0}).
   - Convex fusion: y_hat = w_1 * y_1 + w_2 * y_2 + w_3 * y_3, sum(w_i) = 1.0, w_i >= 0.
4. Tripartite Objective:
   - L_total = L_fused + 0.15 * ((1/3) * sum_i L_expert_i) + 0.001 * H(w).
   - Entropy H(w) = -sum_i w_i * ln(w_i + 1e-8).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Expert 1: Gated Recurrent Unit Expert
# =====================================================================

class GatedRecurrentExpert(nn.Module):
    """
    Expert 1: 2-layer stacked GRU for continuous chronological state tracking.

    Specification:
    - Input: X_t in R^[B, 168, 1]
    - GRU Backbone: 2 layers, hidden_dim=54, batch_first=True, dropout=0.1
    - Sequence Pooling: Final hidden state h_L in R^[B, 54]
    - Projection Head: Linear(54, 54) -> ReLU -> Dropout(0.1) -> Linear(54, 24)
    - Output: y_hat_1 in R^[B, 24]
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 54,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        y_hat = self.head(h_last)
        return y_hat


# =====================================================================
# 2. Expert 2: Multi-Scale Dilated Causal TCN Expert
# =====================================================================

class CausalConv1d(nn.Module):
    """
    1D convolution with strictly causal left-padding.
    Guarantees no future temporal leakage: output at step t depends only on inputs <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class CausalTCNResidualBlock(nn.Module):
    """
    Residual block with two causal dilated convolutions, BatchNorm1d, ReLU, and Dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return out + res


class MultiScaleCausalTCNExpert(nn.Module):
    """
    Expert 2: Multi-Scale Dilated Causal TCN with exponential dilations.

    Specification:
    - Input: X_t in R^[B, 168, 1] -> transposed to R^[B, 1, 168]
    - 5 residual blocks, c=34, k=4, dilations d in {1, 2, 4, 8, 16}
    - Receptive Field: 1 + 2 * sum_{i=0}^4 (4 - 1) * 2^i = 1 + 6 * 31 = 187h (> 168h)
    - Sequence Pooling: Last causal time step z_L in R^[B, 34]
    - Projection Head: Linear(34, 34) -> ReLU -> Dropout(0.1) -> Linear(34, 24)
    - Output: y_hat_2 in R^[B, 24]
    """
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 34,
        kernel_size: int = 4,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.horizon = horizon

        blocks = []
        c_in = in_channels
        for d in dilations:
            blocks.append(
                CausalTCNResidualBlock(
                    in_channels=c_in,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            c_in = channels
        self.network = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, horizon),
        )

    @property
    def receptive_field(self) -> int:
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_t = x.transpose(1, 2)
        out = self.network(x_t)
        z_last = out[:, :, -1]
        y_hat = self.head(z_last)
        return y_hat


# =====================================================================
# 3. Expert 3: Patch-Temporal Linear Expert (Research Candidate)
# =====================================================================

class PatchTemporalExpert(nn.Module):
    """
    Expert 3: Sub-series patch tokenization model preserving local diurnal shape.

    Specification:
    - Input: X_t in R^[B, 168]
    - Patch length P_L = 24h, Stride S = 12h -> P = (168 - 24)//12 + 1 = 13 patches
    - Patch Embedding: Linear(24, 48) -> LayerNorm(48) -> GELU
    - Dense Mixer: Flattened R^[B, 13 * 48] = R^[B, 624] projected through bottleneck:
      Linear(624, 96) -> GELU -> Dropout(0.1) -> Linear(96, 24)
    - Output: y_hat_3 in R^[B, 24]
    """
    def __init__(
        self,
        seq_len: int = 168,
        patch_len: int = 24,
        stride: int = 12,
        embed_dim: int = 48,
        hidden_dim: int = 96,
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.num_patches = (seq_len - patch_len) // stride + 1
        self.flatten_dim = self.num_patches * embed_dim

        self.embed = nn.Sequential(
            nn.Linear(patch_len, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(-1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        e = self.embed(patches)
        e_flat = e.flatten(start_dim=1)
        y_hat = self.mixer(e_flat)
        return y_hat


# =====================================================================
# 4. Context Feature Encoder & Routing Network
# =====================================================================

class ContextFeatureEncoder(nn.Module):
    """
    Nonlinear MLP encoder projecting the 9D context vector u_t into a 32D latent representation.

    Specification:
    Linear(9, 32) -> LayerNorm(32) -> ReLU -> Linear(32, 32) -> ReLU
    """
    def __init__(self, in_dim: int = 9, latent_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class ContextAdaptiveRouter(nn.Module):
    """
    Routing head projecting latent context into convex expert weights w_t in R^3.

    Specification:
    Linear(32, 32) -> ReLU -> Dropout(0.1) -> Linear(32, 3) -> Softmax(z / tau)

    Temperature scaling:
    tau = 0.5 (candidate validation range: {0.2, 0.5, 0.8, 1.0}).
    """
    def __init__(
        self,
        latent_dim: int = 32,
        num_experts: int = 3,
        dropout: float = 0.1,
        temperature: float = 0.5,
        horizon_aware: bool = False,
        horizon: int = 24,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.horizon_aware = horizon_aware
        self.horizon = horizon

        out_dim = (horizon * num_experts) if horizon_aware else num_experts
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, e: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        tau = temperature if temperature is not None else self.temperature
        logits = self.head(e)

        if self.horizon_aware:
            b = e.size(0)
            logits = logits.view(b, self.horizon, self.num_experts)
            weights = F.softmax(logits / tau, dim=-1)
        else:
            weights = F.softmax(logits / tau, dim=-1)

        return weights


# =====================================================================
# 5. Full Proposed Research Architecture: CAEG-Net V2
# =====================================================================

class CAEGNetV2(nn.Module):
    """
    CAEG-Net V2: Context-Adaptive Mixture-of-Experts with Temperature Regularization
    and Forecast-Aware Disagreement Routing.
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        seq_len: int = 168,
        base_context_dim: int = 6,
        disagreement_dim: int = 3,
        gru_hidden: int = 54,
        gru_layers: int = 2,
        tcn_channels: int = 34,
        tcn_kernel_size: int = 4,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        patch_len: int = 24,
        patch_stride: int = 12,
        patch_embed: int = 48,
        patch_hidden: int = 96,
        latent_context_dim: int = 32,
        dropout: float = 0.1,
        temperature: float = 0.5,
        forecast_aware: bool = True,
        horizon_aware: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.seq_len = seq_len
        self.base_context_dim = base_context_dim
        self.disagreement_dim = disagreement_dim
        self.forecast_aware = forecast_aware
        self.horizon_aware = horizon_aware
        self.temperature = temperature

        self.total_router_dim = base_context_dim + (disagreement_dim if forecast_aware else 0)

        # 1. Three Complementary Experts
        self.gru_expert = GatedRecurrentExpert(
            input_dim=input_dim,
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
            horizon=horizon,
            dropout=dropout,
        )
        self.tcn_expert = MultiScaleCausalTCNExpert(
            in_channels=input_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            horizon=horizon,
            dropout=dropout,
        )
        self.patch_expert = PatchTemporalExpert(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            embed_dim=patch_embed,
            hidden_dim=patch_hidden,
            horizon=horizon,
            dropout=dropout,
        )

        # 2. Context Encoder & Router
        self.context_encoder = ContextFeatureEncoder(
            in_dim=self.total_router_dim,
            latent_dim=latent_context_dim,
        )
        self.router = ContextAdaptiveRouter(
            latent_dim=latent_context_dim,
            num_experts=3,
            dropout=dropout,
            temperature=temperature,
            horizon_aware=horizon_aware,
            horizon=horizon,
        )

    def compute_expert_disagreement(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal, inference-time consensus metrics from candidate expert forecasts.
        Explicit autograd detachment terminates gradient flow into expert backbones.
        """
        diff_12 = torch.abs(y1 - y2)
        diff_13 = torch.abs(y1 - y3)
        diff_23 = torch.abs(y2 - y3)
        d_pair = ((diff_12 + diff_13 + diff_23) / 3.0).mean(dim=-1, keepdim=True)

        stacked = torch.stack([y1, y2, y3], dim=-1)
        d_std = stacked.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)
        d_range = (stacked.max(dim=-1).values - stacked.min(dim=-1).values).mean(dim=-1, keepdim=True)

        d_vec = torch.cat([d_pair, d_std, d_range], dim=-1).detach()
        return d_vec

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        y1 = self.gru_expert(x)
        y2 = self.tcn_expert(x)
        y3 = self.patch_expert(x)

        if self.forecast_aware:
            disagreement = self.compute_expert_disagreement(y1, y2, y3)
            u = torch.cat([c, disagreement], dim=-1)
        else:
            disagreement = None
            u = c

        e = self.context_encoder(u)
        weights = self.router(e, temperature=temperature)

        if self.horizon_aware:
            w1 = weights[:, :, 0]
            w2 = weights[:, :, 1]
            w3 = weights[:, :, 2]
            y_fused = w1 * y1 + w2 * y2 + w3 * y3
        else:
            w1 = weights[:, 0:1]
            w2 = weights[:, 1:2]
            w3 = weights[:, 2:3]
            y_fused = w1 * y1 + w2 * y2 + w3 * y3

        if return_diagnostics:
            expert_dict = {
                "gru": y1,
                "tcn": y2,
                "patch": y3,
            }
            diag = {
                "expert_predictions": expert_dict,
                "weights": weights,
            }
            if disagreement is not None:
                diag["disagreement"] = disagreement
            return y_fused, weights, diag
        else:
            return y_fused


# =====================================================================
# 6. Tripartite Training Loss Function
# =====================================================================

def compute_caeg_v2_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expert_preds: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    lambda_aux: float = 0.15,
    beta_entropy: float = 0.001,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    L_total = L_fused + 0.15 * ((1/3) * sum_i L_expert_i) + 0.001 * H(w)
    """
    l_fused = F.mse_loss(y_pred, y_true)

    l_gru = F.mse_loss(expert_preds["gru"], y_true)
    l_tcn = F.mse_loss(expert_preds["tcn"], y_true)
    l_patch = F.mse_loss(expert_preds["patch"], y_true)
    l_expert_aux = (l_gru + l_tcn + l_patch) / 3.0

    entropy = -torch.sum(weights * torch.log(weights + eps), dim=-1).mean()
    l_total = l_fused + lambda_aux * l_expert_aux + beta_entropy * entropy

    telemetry = {
        "loss_total": float(l_total.item()),
        "loss_fused": float(l_fused.item()),
        "loss_aux": float(l_expert_aux.item()),
        "loss_entropy": float(entropy.item()),
        "loss_gru": float(l_gru.item()),
        "loss_tcn": float(l_tcn.item()),
        "loss_patch": float(l_patch.item()),
        "lambda_aux": lambda_aux,
        "beta_entropy": beta_entropy,
    }
    return l_total, telemetry


# =====================================================================
# 7. Helper: Parameter Counting Utility
# =====================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {
        "total": total,
        "trainable": trainable,
    }

    if hasattr(model, "gru_expert"):
        breakdown["gru_expert"] = sum(p.numel() for p in model.gru_expert.parameters())
    if hasattr(model, "tcn_expert"):
        breakdown["tcn_expert"] = sum(p.numel() for p in model.tcn_expert.parameters())
    if hasattr(model, "patch_expert"):
        breakdown["patch_expert"] = sum(p.numel() for p in model.patch_expert.parameters())
    if hasattr(model, "context_encoder"):
        breakdown["context_encoder"] = sum(p.numel() for p in model.context_encoder.parameters())
    if hasattr(model, "router"):
        breakdown["router"] = sum(p.numel() for p in model.router.parameters())

    return breakdown
