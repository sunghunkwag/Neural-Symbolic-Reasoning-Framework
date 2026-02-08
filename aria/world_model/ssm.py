"""State-Space Model - Learned dynamics for multi-step prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class StateSpaceModel(nn.Module):
    """
    Continuous State-Space Model with learned A, B, C, D matrices.
    Computes z_{t+1:t+H} predictions from current latent and actions.
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State transition: z_{t+1} = A @ z_t + B @ a_t
        # Learned as MLPs for nonlinearity
        self.state_transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
        # Residual connection weight (learned)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))
        
    def step(
        self, 
        z_t: torch.Tensor, 
        a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step prediction: z_t, a_t -> z_{t+1}, sigma_{t+1}
        """
        # Ensure float32
        z_t = z_t.float()
        a_t = a_t.float()
        
        # Concatenate state and action
        za = torch.cat([z_t, a_t], dim=-1)
        
        # Predict next state with residual connection
        delta_z = self.state_transition(za)
        z_next = z_t + torch.sigmoid(self.residual_gate) * delta_z
        
        # Estimate uncertainty
        sigma = self.uncertainty_net(z_next)
        
        return z_next, sigma
    
    def rollout(
        self,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        horizon: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step rollout: predict H steps into the future.
        
        Args:
            z_0: Initial latent state [batch, latent_dim]
            actions: Action sequence [batch, H, action_dim]
            horizon: Override action sequence length
            
        Returns:
            z_trajectory: [batch, H, latent_dim]
            sigmas: [batch, H, 1]
        """
        H = horizon if horizon else actions.shape[1]
        batch_size = z_0.shape[0]
        
        z_trajectory = []
        sigmas = []
        z_t = z_0
        
        for t in range(H):
            a_t = actions[:, t] if t < actions.shape[1] else torch.zeros(
                batch_size, self.action_dim, device=z_0.device
            )
            z_t, sigma_t = self.step(z_t, a_t)
            z_trajectory.append(z_t)
            sigmas.append(sigma_t)
        
        return torch.stack(z_trajectory, dim=1), torch.stack(sigmas, dim=1)
    
    def compute_prediction_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction loss with uncertainty weighting.
        Returns both the loss and raw prediction error.
        """
        # Prediction error
        pred_error = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1)
        
        # Negative log-likelihood with learned uncertainty
        # loss = 0.5 * (error / sigma^2 + log(sigma^2))
        nll = 0.5 * (pred_error / (sigma.squeeze(-1) ** 2 + 1e-6) + 
                     torch.log(sigma.squeeze(-1) ** 2 + 1e-6))
        
        return nll.mean(), pred_error.mean()
