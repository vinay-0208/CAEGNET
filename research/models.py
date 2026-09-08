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
# 6. Standard Input-Conditioned MoE Baseline
# =====================================================================

class StandardInputMoE(nn.Module):
    """
    Standard Input-Conditioned Mixture-of-Experts Baseline.

    Shares identical expert capacity and backbones with CAEG-Net V2:
    - Expert 1: GatedRecurrentExpert (GRU-54)
    - Expert 2: MultiScaleCausalTCNExpert (TCN-34)
    - Expert 3: PatchTemporalExpert (PatchLinear)

    The router receives the raw lookback load vector X in R^168 directly
    via an MLP encoder rather than explicit domain context features C or
    inter-expert disagreement feedback.
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        seq_len: int = 168,
        gru_hidden: int = 54,
        gru_layers: int = 2,
        tcn_channels: int = 34,
        tcn_kernel_size: int = 4,
        tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        patch_len: int = 24,
        patch_stride: int = 12,
        patch_embed: int = 48,
        patch_hidden: int = 96,
        latent_dim: int = 32,
        dropout: float = 0.1,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.seq_len = seq_len
        self.temperature = temperature

        # Same 3 complementary experts as CAEG-Net V2
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

        # Raw input lookback encoder: Linear(168, 32) -> ReLU -> Dropout -> Linear(32, 32) -> ReLU
        self.input_encoder = nn.Sequential(
            nn.Linear(seq_len * input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.router = ContextAdaptiveRouter(
            latent_dim=latent_dim,
            num_experts=3,
            dropout=dropout,
            temperature=temperature,
        )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,  # c is ignored by design
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        y1 = self.gru_expert(x)
        y2 = self.tcn_expert(x)
        y3 = self.patch_expert(x)

        # Flatten raw historical load sequence [B, 168, 1] -> [B, 168]
        x_flat = x.view(x.size(0), -1)
        e = self.input_encoder(x_flat)
        weights = self.router(e, temperature=temperature)

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
            return y_fused, weights, diag
        else:
            return y_fused


# =====================================================================
# 7. Tripartite Training Loss Function
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
# 8. Helper: Parameter Counting Utility
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
    if hasattr(model, "input_encoder"):
        breakdown["input_encoder"] = sum(p.numel() for p in model.input_encoder.parameters())
    if hasattr(model, "router"):
        breakdown["router"] = sum(p.numel() for p in model.router.parameters())
    if hasattr(model, "router_head"):
        breakdown["router_head"] = sum(p.numel() for p in model.router_head.parameters())

    return breakdown


# =====================================================================
# 9. CAEG-Net V3: Horizon-Dependent Context-Adaptive MoE
# =====================================================================

class CAEGNetV3(nn.Module):
    """
    CAEG-Net V3: Horizon-Dependent Context-Adaptive Mixture-of-Experts
    with Multi-Horizon Disagreement Feedback and Anti-Starvation Balancing.

    Key Architectural Innovations:
    1. Horizon-Dependent Routing:
       Router emits W in R^[B, 24, 3] with sum_{i=1}^3 W[b, h, i] = 1.0 for each h in {1..24}.
       Resolves the horizon-invariance bottleneck identified in Phase 6.
    2. Multi-Horizon Disagreement & Shape Feedback:
       Detached candidate forecasts yield 6D feedback:
       - Near-horizon disagreement (h=1..6)
       - Mid-horizon disagreement (h=7..18)
       - Far-horizon disagreement (h=19..24)
       - Global pairwise standard deviation
       - Global pairwise range
       - Diurnal range difference (Patch ptp - TCN ptp)
       Total router input = 6D base context + 6D disagreement = 12 dimensions.
    3. Calm-Regime Prior Fallback:
       Encourages weights to smoothly default to [1/3, 1/3, 1/3] under low difficulty.
    """
    def __init__(
        self,
        input_dim: int = 1,
        horizon: int = 24,
        seq_len: int = 168,
        base_context_dim: int = 6,
        disagreement_dim: int = 6,
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
        router_hidden_dim: int = 48,
        dropout: float = 0.1,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.seq_len = seq_len
        self.base_context_dim = base_context_dim
        self.disagreement_dim = disagreement_dim
        self.temperature = temperature
        self.total_router_dim = base_context_dim + disagreement_dim

        # 1. Three Complementary Expert Backbones
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

        # 2. 12D Context Feature Encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(self.total_router_dim, latent_context_dim),
            nn.LayerNorm(latent_context_dim),
            nn.ReLU(),
            nn.Linear(latent_context_dim, latent_context_dim),
            nn.ReLU(),
        )

        # 3. Horizon-Dependent Routing Head: R^32 -> R^[24 * 3]
        self.router_head = nn.Sequential(
            nn.Linear(latent_context_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, horizon * 3),
        )

    def compute_multi_horizon_disagreement(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal, detached multi-horizon consensus & shape features from candidate forecasts.
        """
        diff_12 = torch.abs(y1 - y2)
        diff_13 = torch.abs(y1 - y3)
        diff_23 = torch.abs(y2 - y3)
        d_pair = (diff_12 + diff_13 + diff_23) / 3.0  # [B, 24]

        # Segmented horizon disagreements
        d_near = d_pair[:, :6].mean(dim=-1, keepdim=True)    # [B, 1]
        d_mid = d_pair[:, 6:18].mean(dim=-1, keepdim=True)   # [B, 1]
        d_far = d_pair[:, 18:].mean(dim=-1, keepdim=True)    # [B, 1]

        # Global consensus dispersion
        stacked = torch.stack([y1, y2, y3], dim=-1)           # [B, 24, 3]
        d_std = stacked.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True) # [B, 1]
        d_range = (stacked.max(dim=-1).values - stacked.min(dim=-1).values).mean(dim=-1, keepdim=True) # [B, 1]

        # Diurnal range difference: Patch ptp - TCN ptp
        ptp_patch = y3.max(dim=-1, keepdim=True).values - y3.min(dim=-1, keepdim=True).values # [B, 1]
        ptp_tcn = y2.max(dim=-1, keepdim=True).values - y2.min(dim=-1, keepdim=True).values   # [B, 1]
        d_shape = ptp_patch - ptp_tcn                         # [B, 1]

        d_vec = torch.cat([d_near, d_mid, d_far, d_std, d_range, d_shape], dim=-1).detach()
        return d_vec

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        tau = temperature if temperature is not None else self.temperature

        # 1. Candidate Expert Predictions
        y1 = self.gru_expert(x)
        y2 = self.tcn_expert(x)
        y3 = self.patch_expert(x)

        # 2. Multi-Horizon Detached Disagreement
        disagreement = self.compute_multi_horizon_disagreement(y1, y2, y3)
        u = torch.cat([c, disagreement], dim=-1)  # [B, 12]

        # 3. Context Encoding & Horizon-Dependent Routing
        e = self.context_encoder(u)
        logits = self.router_head(e)  # [B, horizon * 3]
        b = x.size(0)
        logits = logits.view(b, self.horizon, 3)
        weights = F.softmax(logits / tau, dim=-1)  # [B, 24, 3]

        # 4. Horizon-Wise Convex Combination
        w1 = weights[:, :, 0]  # [B, 24]
        w2 = weights[:, :, 1]  # [B, 24]
        w3 = weights[:, :, 2]  # [B, 24]
        y_fused = w1 * y1 + w2 * y2 + w3 * y3  # [B, 24]

        if return_diagnostics:
            expert_dict = {
                "gru": y1,
                "tcn": y2,
                "patch": y3,
            }
            diag = {
                "expert_predictions": expert_dict,
                "weights": weights,
                "disagreement": disagreement,
            }
            return y_fused, weights, diag
        else:
            return y_fused


# =====================================================================
# 10. CAEG-Net V3 Loss Function
# =====================================================================

def compute_caeg_v3_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expert_preds: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    lambda_aux: float = 0.40,
    beta_prior: float = 0.002,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    CAEG-Net V3 Tripartite Loss with Horizon-Dependent Uniform Prior:
    L_total = L_fused + lambda_aux * L_expert_aux + beta_prior * L_prior

    where:
    L_fused = MSE(y_pred, y_true)
    L_expert_aux = (1/3) * sum_{i=1}^3 MSE(y_expert_i, y_true)
    L_prior = (1/24) * sum_{h=1}^24 sum_{i=1}^3 w_{h,i} * ln(3 * w_{h,i} + eps)
    (KL divergence to uniform [1/3, 1/3, 1/3], penalizing unneeded deviations in calm regimes).
    """
    l_fused = F.mse_loss(y_pred, y_true)

    l_gru = F.mse_loss(expert_preds["gru"], y_true)
    l_tcn = F.mse_loss(expert_preds["tcn"], y_true)
    l_patch = F.mse_loss(expert_preds["patch"], y_true)
    l_expert_aux = (l_gru + l_tcn + l_patch) / 3.0

    # weights: [B, 24, 3]
    # KL(w || [1/3, 1/3, 1/3]) = sum_i w_i * (ln(w_i) - ln(1/3)) = sum_i w_i * ln(3 * w_i)
    kl_uniform = torch.sum(weights * torch.log(3.0 * weights + eps), dim=-1)  # [B, 24]
    l_prior = kl_uniform.mean()  # scalar

    l_total = l_fused + lambda_aux * l_expert_aux + beta_prior * l_prior

    telemetry = {
        "loss_total": float(l_total.item()),
        "loss_fused": float(l_fused.item()),
        "loss_aux": float(l_expert_aux.item()),
        "loss_prior": float(l_prior.item()),
        "loss_gru": float(l_gru.item()),
        "loss_tcn": float(l_tcn.item()),
        "loss_patch": float(l_patch.item()),
        "lambda_aux": lambda_aux,
        "beta_prior": beta_prior,
    }
    return l_total, telemetry


# =====================================================================
# 11. Decoupled CAEG-Net (D-CAEG) & Learned Static Ensemble
# =====================================================================

class LearnedStaticEnsemble(nn.Module):
    """
    Experiment B Baseline: 3 Learned Global Scalar Weights on Frozen Experts.
    w_i = Softmax(alpha_i / tau)
    No context features u_t used. Fuses: y = sum_i w_i * y_i.
    """
    def __init__(
        self,
        gru_expert: nn.Module,
        tcn_expert: nn.Module,
        patch_expert: nn.Module,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.gru_expert = gru_expert
        self.tcn_expert = tcn_expert
        self.patch_expert = patch_expert
        self.temperature = temperature

        # Freeze all expert parameters
        self.freeze_experts()

        # Learnable unconstrained scalar logits for the 3 experts
        self.logits = nn.Parameter(torch.zeros(3))

    def freeze_experts(self):
        for expert in [self.gru_expert, self.tcn_expert, self.patch_expert]:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        tau = temperature if temperature is not None else self.temperature

        with torch.no_grad():
            y1 = self.gru_expert(x)
            y2 = self.tcn_expert(x)
            y3 = self.patch_expert(x)

        weights = F.softmax(self.logits / tau, dim=-1)
        b = x.size(0)
        weights_expanded = weights.unsqueeze(0).expand(b, -1)

        w1 = weights_expanded[:, 0:1]
        w2 = weights_expanded[:, 1:2]
        w3 = weights_expanded[:, 2:3]
        y_fused = w1 * y1 + w2 * y2 + w3 * y3

        if return_diagnostics:
            diag = {
                "expert_predictions": {"gru": y1, "tcn": y2, "patch": y3},
                "weights": weights_expanded,
            }
            return y_fused, weights_expanded, diag
        return y_fused


class DecoupledCAEGNet(nn.Module):
    """
    CAEG-Net Decoupled (D-CAEG) Architecture:
    Stage 1: Frozen independently-trained complementary temporal experts.
    Stage 2: Context-adaptive scalar routing head trained on frozen predictions.

    Zero gradient flow into expert backbones.
    Router input u_t = [C_t (6D) || D_t (3D)] in R^9.
    """
    def __init__(
        self,
        gru_expert: nn.Module,
        tcn_expert: nn.Module,
        patch_expert: nn.Module,
        base_context_dim: int = 6,
        disagreement_dim: int = 3,
        latent_context_dim: int = 32,
        dropout: float = 0.1,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.gru_expert = gru_expert
        self.tcn_expert = tcn_expert
        self.patch_expert = patch_expert
        self.base_context_dim = base_context_dim
        self.disagreement_dim = disagreement_dim
        self.total_router_dim = base_context_dim + disagreement_dim
        self.temperature = temperature

        # Explicitly freeze all expert parameters
        self.freeze_experts()

        # Context Feature Encoder (Linear(9, 32) -> LayerNorm -> ReLU -> Linear(32, 32) -> ReLU)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.total_router_dim, latent_context_dim),
            nn.LayerNorm(latent_context_dim),
            nn.ReLU(),
            nn.Linear(latent_context_dim, latent_context_dim),
            nn.ReLU(),
        )

        # Context-Adaptive Scalar Routing Head
        self.router = nn.Sequential(
            nn.Linear(latent_context_dim, latent_context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_context_dim, 3),
        )

    def freeze_experts(self):
        for expert in [self.gru_expert, self.tcn_expert, self.patch_expert]:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

    def compute_expert_disagreement(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
    ) -> torch.Tensor:
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
        tau = temperature if temperature is not None else self.temperature

        # Generate candidate forecasts from frozen experts (zero autograd tracking)
        with torch.no_grad():
            y1 = self.gru_expert(x)
            y2 = self.tcn_expert(x)
            y3 = self.patch_expert(x)

        disagreement = self.compute_expert_disagreement(y1, y2, y3)
        u = torch.cat([c, disagreement], dim=-1)

        e = self.context_encoder(u)
        logits = self.router(e)
        weights = F.softmax(logits / tau, dim=-1)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]
        y_fused = w1 * y1 + w2 * y2 + w3 * y3

        if return_diagnostics:
            diag = {
                "expert_predictions": {"gru": y1, "tcn": y2, "patch": y3},
                "weights": weights,
                "disagreement": disagreement,
            }
            return y_fused, weights, diag
        return y_fused


# =====================================================================
# 12. Phase 5 Optimization: Shrinkage-Regularized CAEG-Net (CAEG-Net SR)
# =====================================================================

class ShrinkageRegularizedCAEG(nn.Module):
    """
    CAEG-Net SR (Shrinkage-Regularized Context-Adaptive Ensemble):
    w_t = softmax(log(w0) + (alpha / tau) * delta_t)
    where w0 = [1/3, 1/3, 1/3].

    Guarantees:
    - alpha = 0: w_t = [1/3, 1/3, 1/3] identically (exact equal ensemble).
    - alpha > 0: controls the maximum expressive deviation of the adaptive router.
    - Frozen backbones: expert parameters have requires_grad=False and run in eval() mode.
    - Computes analytical KL(w_t || w0) = sum_i w_i * log(3 * w_i) >= 0.
    """
    def __init__(
        self,
        gru_expert: nn.Module,
        tcn_expert: nn.Module,
        patch_expert: nn.Module,
        alpha: float = 1.0,
        temperature: float = 0.5,
        context_dim: int = 6,
        disagreement_dim: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru_expert = gru_expert
        self.tcn_expert = tcn_expert
        self.patch_expert = patch_expert
        self.alpha = float(alpha)
        self.temperature = float(temperature)

        self.freeze_experts()

        router_in_dim = context_dim + disagreement_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(router_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # Base equal log-prior: log(1/3)
        log_w0 = torch.tensor([-1.09861228867, -1.09861228867, -1.09861228867], dtype=torch.float32)
        self.register_buffer("log_w0", log_w0)

    def freeze_experts(self):
        for expert in [self.gru_expert, self.tcn_expert, self.patch_expert]:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

    def compute_expert_disagreement(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
    ) -> torch.Tensor:
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
        alpha: Optional[float] = None,
        temperature: Optional[float] = None,
        return_diagnostics: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        tau = temperature if temperature is not None else self.temperature
        a = alpha if alpha is not None else self.alpha

        with torch.no_grad():
            y1 = self.gru_expert(x)
            y2 = self.tcn_expert(x)
            y3 = self.patch_expert(x)

        disagreement = self.compute_expert_disagreement(y1, y2, y3)
        u = torch.cat([c, disagreement], dim=-1)

        e = self.context_encoder(u)
        delta = self.router(e)

        # w_t = softmax(log(w0) + (alpha / tau) * delta)
        logits = self.log_w0.unsqueeze(0) + (a / tau) * delta
        weights = F.softmax(logits, dim=-1)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]
        y_fused = w1 * y1 + w2 * y2 + w3 * y3

        # KL(w || w0) = sum_i w_i * log(3 * w_i)
        eps = 1e-8
        kl_div = torch.sum(weights * torch.log(3.0 * weights + eps), dim=-1)

        if return_diagnostics:
            diag = {
                "expert_predictions": {"gru": y1, "tcn": y2, "patch": y3},
                "weights": weights,
                "disagreement": disagreement,
                "delta": delta,
                "kl_div": kl_div,
                "mean_kl": kl_div.mean(),
                "alpha": a,
            }
            return y_fused, weights, diag
        return y_fused


CAEGNetSR = ShrinkageRegularizedCAEG


def compute_shrinkage_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weights: torch.Tensor,
    lambda_dev: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 5 Shrinkage Loss Function:
    L = L_forecast + lambda_dev * KL(w || w0)
    where w0 = [1/3, 1/3, 1/3].
    """
    l_forecast = F.mse_loss(y_pred, y_true)
    kl = torch.sum(weights * torch.log(3.0 * weights + eps), dim=-1).mean()
    l_total = l_forecast + lambda_dev * kl

    telemetry = {
        "loss_total": float(l_total.item()),
        "loss_forecast": float(l_forecast.item()),
        "loss_kl": float(kl.item()),
        "lambda_dev": float(lambda_dev),
    }
    return l_total, telemetry


