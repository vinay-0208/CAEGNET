"""
Phase 15B Controlled Model Architectures
========================================
Implements the exact experimental controls and hypothesis-driven candidates
grounded in Phase 15A diagnostic discoveries:
1. Control A: Current F2 / A2-OOF (ConfidenceFallbackCAEGNet, 121,724 params)
2. Control B: Fixed Shrinkage (lambda = 0.51 scalar, 121,579 params)
3. Control C: Horizon-Aware Routing Only (3 groups: 1-8, 9-16, 17-24, lambda = 1.0)
4. Control D: Dynamic Confidence Only (Global router + disagreement-gated confidence)
5. Candidate E1: Horizon-Grouped Routing + Fixed Shrinkage (HGR-FS, lambda = 0.51)
6. Candidate E2: Horizon-Grouped Routing + Disagreement-Gated Confidence (HGR-DGS)
7. Candidate E3: Horizon-Grouped Routing + Dynamic Fallback (HGR-CF)

All architectures preserve the exact canonical 120,504-parameter expert family:
LSTM (56,152), TCN (36,952), CNN (27,400).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.original_caeg import OriginalCAEGNetPhase5


class ControlA_CurrentF2(nn.Module):
    """
    Control A: Exact authoritative Phase 14 F2 / A2-OOF model.
    Global routing vector w in Delta^2 + 145-parameter dynamic confidence head.
    y_final = lambda * y_adaptive + (1 - lambda) * y_equal
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
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
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_adaptive, w, diag = self.caeg(x, c, conservative_rho=conservative_rho, temperature=temperature)
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        lam = self.confidence_head(c)
        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        diag["lambda"] = lam
        diag["y_adaptive"] = y_adaptive
        diag["y_equal"] = y_equal
        diag["expert_predictions"] = {"lstm": y_l, "tcn": y_t, "cnn": y_c}
        return y_final, w, diag


class ControlB_FixedShrinkage(nn.Module):
    """
    Control B: Global router with fixed scalar shrinkage lambda* = 0.51.
    Zero-parameter confidence head.
    y_final = 0.51 * y_adaptive + 0.49 * y_equal
    """
    def __init__(self, context_dim: int = 7, lambda_val: float = 0.51):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.lambda_scalar = float(lambda_val)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_adaptive, w, diag = self.caeg(x, c, conservative_rho=conservative_rho, temperature=temperature)
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        lam = torch.full((x.shape[0], 1), self.lambda_scalar, device=x.device, dtype=x.dtype)
        y_final = self.lambda_scalar * y_adaptive + (1.0 - self.lambda_scalar) * y_equal
        diag["lambda"] = lam
        diag["y_adaptive"] = y_adaptive
        diag["y_equal"] = y_equal
        diag["expert_predictions"] = {"lstm": y_l, "tcn": y_t, "cnn": y_c}
        return y_final, w, diag


class HorizonGroupedRouter(nn.Module):
    """
    Horizon-Grouped Router producing 3 sets of 3-simplex weights:
    Group 1 (Short):  h in {1..8}
    Group 2 (Medium): h in {9..16}
    Group 3 (Long):   h in {17..24}
    """
    def __init__(self, context_dim: int = 7, latent_dim: int = 16):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.router_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 9),  # 3 groups x 3 experts
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        e_c = self.context_encoder(c)
        logits = self.router_head(e_c).view(-1, 3, 3)  # [B, 3 groups, 3 experts]
        w_groups = F.softmax(logits, dim=-1)           # [B, 3, 3] on Delta^2
        return w_groups


class ControlC_HorizonRoutingOnly(nn.Module):
    """
    Control C: Horizon-Aware Routing Only without confidence fallback (lambda = 1.0).
    Applies group 1 weights to horizons 1-8, group 2 to 9-16, group 3 to 17-24.
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.hgr_router = HorizonGroupedRouter(context_dim=context_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_l = self.caeg.lstm_expert(x)  # [B, 24]
        y_t = self.caeg.tcn_expert(x)   # [B, 24]
        y_c = self.caeg.cnn_expert(x)   # [B, 24]
        y_equal = (y_l + y_t + y_c) / 3.0

        w_groups = self.hgr_router(c)   # [B, 3, 3]

        # Expand weights across 24 horizons:
        # Group 0: h=0..7, Group 1: h=8..15, Group 2: h=16..23
        w_h = torch.zeros(x.shape[0], 24, 3, device=x.device, dtype=x.dtype)
        w_h[:, 0:8, :] = w_groups[:, 0:1, :]
        w_h[:, 8:16, :] = w_groups[:, 1:2, :]
        w_h[:, 16:24, :] = w_groups[:, 2:3, :]

        # Fused prediction
        y_fused = (
            w_h[:, :, 0] * y_l +
            w_h[:, :, 1] * y_t +
            w_h[:, :, 2] * y_c
        )

        w_global = w_groups.mean(dim=1)  # [B, 3] average weight across horizon
        diag = {
            "weights_horizon": w_h,
            "weights_groups": w_groups,
            "lambda": torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
            "y_adaptive": y_fused,
            "y_equal": y_equal,
            "expert_predictions": {"lstm": y_l, "tcn": y_t, "cnn": y_c},
        }
        return y_fused, w_global, diag


class ControlD_DynamicConfidenceOnly(nn.Module):
    """
    Control D: Global router + Dynamic Confidence Head explicitly conditioned
    on context (7D) and detached forecast spread D_t = std(y_L, y_T, y_C) (1D) -> 8D total.
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_adaptive, w, diag = self.caeg(x, c, conservative_rho=conservative_rho, temperature=temperature)
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        # Detached forecast spread (unbiased=False)
        stacked = torch.stack([y_l, y_t, y_c], dim=-1)
        d_std = stacked.std(dim=-1, unbiased=False).mean(dim=-1, keepdim=True).detach()  # [B, 1]

        conf_in = torch.cat([c, d_std], dim=-1)  # [B, 8]
        lam = self.confidence_head(conf_in)      # [B, 1]

        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        diag["lambda"] = lam
        diag["disagreement_spread"] = d_std
        diag["y_adaptive"] = y_adaptive
        diag["y_equal"] = y_equal
        diag["expert_predictions"] = {"lstm": y_l, "tcn": y_t, "cnn": y_c}
        return y_final, w, diag


class CandidateE1_HGR_FixedShrinkage(nn.Module):
    """
    Candidate E1: Horizon-Grouped Routing (3 heads) + Fixed Shrinkage (lambda = 0.51).
    Combines lead-time specialization with empirical centroid regularization.
    """
    def __init__(self, context_dim: int = 7, lambda_val: float = 0.51):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.hgr_router = HorizonGroupedRouter(context_dim=context_dim)
        self.lambda_scalar = float(lambda_val)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        w_groups = self.hgr_router(c)   # [B, 3, 3]
        w_h = torch.zeros(x.shape[0], 24, 3, device=x.device, dtype=x.dtype)
        w_h[:, 0:8, :] = w_groups[:, 0:1, :]
        w_h[:, 8:16, :] = w_groups[:, 1:2, :]
        w_h[:, 16:24, :] = w_groups[:, 2:3, :]

        y_adaptive = (
            w_h[:, :, 0] * y_l +
            w_h[:, :, 1] * y_t +
            w_h[:, :, 2] * y_c
        )

        y_final = self.lambda_scalar * y_adaptive + (1.0 - self.lambda_scalar) * y_equal
        lam = torch.full((x.shape[0], 1), self.lambda_scalar, device=x.device, dtype=x.dtype)
        w_global = w_groups.mean(dim=1)

        diag = {
            "weights_horizon": w_h,
            "weights_groups": w_groups,
            "lambda": lam,
            "y_adaptive": y_adaptive,
            "y_equal": y_equal,
            "expert_predictions": {"lstm": y_l, "tcn": y_t, "cnn": y_c},
        }
        return y_final, w_global, diag


class CandidateE2_HGR_DisagreementConfidence(nn.Module):
    """
    Candidate E2: Horizon-Grouped Routing + Disagreement-Gated Group Confidence (HGR-DGS).
    Computes group-specific forecast spreads D_g in R^3.
    Confidence head outputs lambda_g in (0, 1)^3 for each horizon group.
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.hgr_router = HorizonGroupedRouter(context_dim=context_dim)
        # Context (7) + 3 group spreads (3) = 10 input features
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim + 3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        w_groups = self.hgr_router(c)   # [B, 3, 3]
        w_h = torch.zeros(x.shape[0], 24, 3, device=x.device, dtype=x.dtype)
        w_h[:, 0:8, :] = w_groups[:, 0:1, :]
        w_h[:, 8:16, :] = w_groups[:, 1:2, :]
        w_h[:, 16:24, :] = w_groups[:, 2:3, :]

        y_adaptive = (
            w_h[:, :, 0] * y_l +
            w_h[:, :, 1] * y_t +
            w_h[:, :, 2] * y_c
        )

        # Compute group-specific detached spreads
        stacked = torch.stack([y_l, y_t, y_c], dim=-1)  # [B, 24, 3]
        s_g0 = stacked[:, 0:8, :].std(dim=-1, unbiased=False).mean(dim=1, keepdim=True)
        s_g1 = stacked[:, 8:16, :].std(dim=-1, unbiased=False).mean(dim=1, keepdim=True)
        s_g2 = stacked[:, 16:24, :].std(dim=-1, unbiased=False).mean(dim=1, keepdim=True)
        spread_groups = torch.cat([s_g0, s_g1, s_g2], dim=-1).detach()  # [B, 3]

        conf_in = torch.cat([c, spread_groups], dim=-1)  # [B, 10]
        lam_groups = self.confidence_head(conf_in)       # [B, 3]

        lam_h = torch.zeros(x.shape[0], 24, device=x.device, dtype=x.dtype)
        lam_h[:, 0:8] = lam_groups[:, 0:1]
        lam_h[:, 8:16] = lam_groups[:, 1:2]
        lam_h[:, 16:24] = lam_groups[:, 2:3]

        y_final = lam_h * y_adaptive + (1.0 - lam_h) * y_equal
        w_global = w_groups.mean(dim=1)

        diag = {
            "weights_horizon": w_h,
            "weights_groups": w_groups,
            "lambda": lam_groups.mean(dim=-1, keepdim=True),
            "lambda_groups": lam_groups,
            "lambda_horizon": lam_h,
            "disagreement_groups": spread_groups,
            "y_adaptive": y_adaptive,
            "y_equal": y_equal,
            "expert_predictions": {"lstm": y_l, "tcn": y_t, "cnn": y_c},
        }
        return y_final, w_global, diag


class CandidateE3_HGR_GlobalConfidence(nn.Module):
    """
    Candidate E3: Horizon-Grouped Routing + Global Dynamic Confidence Head (HGR-CF).
    Combines 3-group horizon routing with canonical 145-parameter scalar confidence head.
    """
    def __init__(self, context_dim: int = 7):
        super().__init__()
        self.caeg = OriginalCAEGNetPhase5(context_dim=context_dim)
        self.hgr_router = HorizonGroupedRouter(context_dim=context_dim)
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
        conservative_rho=None,
        temperature=None,
        return_diagnostics: bool = True,
    ):
        y_l = self.caeg.lstm_expert(x)
        y_t = self.caeg.tcn_expert(x)
        y_c = self.caeg.cnn_expert(x)
        y_equal = (y_l + y_t + y_c) / 3.0

        w_groups = self.hgr_router(c)   # [B, 3, 3]
        w_h = torch.zeros(x.shape[0], 24, 3, device=x.device, dtype=x.dtype)
        w_h[:, 0:8, :] = w_groups[:, 0:1, :]
        w_h[:, 8:16, :] = w_groups[:, 1:2, :]
        w_h[:, 16:24, :] = w_groups[:, 2:3, :]

        y_adaptive = (
            w_h[:, :, 0] * y_l +
            w_h[:, :, 1] * y_t +
            w_h[:, :, 2] * y_c
        )

        lam = self.confidence_head(c)  # [B, 1]
        y_final = lam * y_adaptive + (1.0 - lam) * y_equal
        w_global = w_groups.mean(dim=1)

        diag = {
            "weights_horizon": w_h,
            "weights_groups": w_groups,
            "lambda": lam,
            "y_adaptive": y_adaptive,
            "y_equal": y_equal,
            "expert_predictions": {"lstm": y_l, "tcn": y_t, "cnn": y_c},
        }
        return y_final, w_global, diag


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns structured parameter counts directly from instantiated PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Expert parameters
    expert_params = 0
    if hasattr(model, "caeg"):
        expert_params = sum(
            p.numel() for p in model.caeg.lstm_expert.parameters()
        ) + sum(
            p.numel() for p in model.caeg.tcn_expert.parameters()
        ) + sum(
            p.numel() for p in model.caeg.cnn_expert.parameters()
        )
    
    # Confidence parameters
    conf_params = 0
    if hasattr(model, "confidence_head"):
        conf_params = sum(p.numel() for p in model.confidence_head.parameters())
        
    router_params = total_params - expert_params - conf_params
    return {
        "expert_params": expert_params,
        "router_params": router_params,
        "confidence_params": conf_params,
        "total_params": total_params,
    }
