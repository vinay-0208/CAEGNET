import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextFeatureEncoder(nn.Module):
    def __init__(self, context_dim: int = 7, latent_dim: int = 16):
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
        return self.encoder(c)

class ContextGatingNetwork(nn.Module):
    def __init__(self, latent_dim: int = 16, num_experts: int = 3, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, e_c: torch.Tensor) -> torch.Tensor:
        logits = self.routing_mlp(e_c)
        weights = F.softmax(logits, dim=-1)
        return weights

class HorizonContextGatingNetwork(nn.Module):
    def __init__(self, latent_dim: int = 16, horizon: int = 24, num_experts: int = 3, hidden_dim: int = 48, dropout: float = 0.1):
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
        B = e_c.shape[0]
        logits = self.routing_mlp(e_c).view(B, self.horizon, self.num_experts)
        weights = F.softmax(logits, dim=-1)
        return weights
