"""JEPA-style Latent Encoder - Trained via prediction consistency."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class JEPAEncoder(nn.Module):
    """
    Joint Embedding Predictive Architecture encoder.
    Maps observations to latent space, trained via prediction consistency loss.
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 128]
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Context encoder (online)
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU()
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Target encoder (EMA updated)
        self.target_encoder = self._build_target_encoder(hidden_dims)
        self._ema_update(tau=1.0)  # Initialize to same weights
        
        # Predictor (asymmetric - prevents collapse)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.ema_tau = 0.99
        
    def _build_target_encoder(self, hidden_dims: List[int]) -> nn.Module:
        layers = []
        in_dim = self.obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU()
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.latent_dim))
        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def _ema_update(self, tau: float = None):
        """Update target encoder via exponential moving average."""
        tau = tau if tau is not None else self.ema_tau
        for p_online, p_target in zip(self.encoder.parameters(), 
                                       self.target_encoder.parameters()):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1 - tau)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.encoder(obs.float())
    
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode with target (EMA) encoder - no gradient."""
        with torch.no_grad():
            return self.target_encoder(obs.float())
    
    def compute_jepa_loss(
        self, 
        obs_t: torch.Tensor, 
        obs_tp1: torch.Tensor,
        predicted_z_tp1: torch.Tensor
    ) -> torch.Tensor:
        """
        JEPA loss: prediction in latent space should match target encoding.
        
        Args:
            obs_t: Current observation
            obs_tp1: Next observation  
            predicted_z_tp1: SSM-predicted next latent
        """
        # Target encoding of actual next obs (no gradient)
        target_z = self.encode_target(obs_tp1)
        
        # Predictor output from SSM prediction
        pred_z = self.predictor(predicted_z_tp1)
        
        # Cosine similarity loss (prevents collapse better than MSE)
        loss = 1 - F.cosine_similarity(pred_z, target_z, dim=-1).mean()
        
        # Update target encoder
        self._ema_update()
        
        return loss
    
    def get_representation_stats(self, z: torch.Tensor) -> dict:
        """Get stats about learned representations."""
        return {
            "z_mean": z.mean().item(),
            "z_std": z.std().item(),
            "z_norm": z.norm(dim=-1).mean().item()
        }
