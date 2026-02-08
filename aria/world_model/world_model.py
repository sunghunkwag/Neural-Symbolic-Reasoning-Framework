"""Combined World Model - JEPA Encoder + SSM Dynamics."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List, Optional
from collections import deque
import numpy as np

from .encoder import JEPAEncoder
from .ssm import StateSpaceModel
from ..config import WorldModelConfig

class WorldModel(nn.Module):
    """
    Complete World Model combining JEPA encoder and SSM dynamics.
    Provides predictions, uncertainty, and novelty scoring.
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = JEPAEncoder(
            obs_dim=config.obs_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.encoder_hidden
        )
        
        self.ssm = StateSpaceModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.ssm_hidden_dim
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
        
        # Novelty tracking
        self.prediction_error_ema = 0.0
        self.novelty_history = deque(maxlen=1000)
        
        # Learning statistics
        self.train_steps = 0
        self.total_loss = 0.0
        
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        return self.encoder(obs.float())
    
    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict future states given current obs and action sequence.
        
        Returns:
            z_trajectory: Predicted latent trajectory
            sigmas: Uncertainty estimates
            novelty: Novelty score based on prediction capability
        """
        z_0 = self.encode(obs)
        z_trajectory, sigmas = self.ssm.rollout(z_0, actions)
        
        # Novelty is based on uncertainty
        novelty = sigmas.mean(dim=(1, 2))
        
        return z_trajectory, sigmas, novelty
    
    def train_step(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_obs_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step on a batch of transitions.
        REAL LEARNING - weights actually update.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Ensure float32
        obs_batch = obs_batch.float()
        action_batch = action_batch.float()
        next_obs_batch = next_obs_batch.float()
        
        # Encode current and next observations
        z_t = self.encode(obs_batch)
        z_tp1_target = self.encoder.encode_target(next_obs_batch)
        
        # Predict next latent
        z_tp1_pred, sigma = self.ssm.step(z_t, action_batch)
        
        # SSM prediction loss
        ssm_loss, pred_error = self.ssm.compute_prediction_loss(
            z_tp1_pred, z_tp1_target, sigma
        )
        
        # JEPA consistency loss
        jepa_loss = self.encoder.compute_jepa_loss(
            obs_batch, next_obs_batch, z_tp1_pred
        )
        
        # Combined loss
        total_loss = ssm_loss + 0.5 * jepa_loss
        
        # Backward pass - REAL GRADIENT UPDATE
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # Update novelty tracking
        self._update_novelty(pred_error.item())
        
        self.train_steps += 1
        self.total_loss += total_loss.item()
        
        return {
            "ssm_loss": ssm_loss.item(),
            "jepa_loss": jepa_loss.item(),
            "total_loss": total_loss.item(),
            "pred_error": pred_error.item(),
            "novelty": self.get_novelty_score()
        }
    
    def _update_novelty(self, pred_error: float):
        """Update novelty score via EMA."""
        alpha = self.config.novelty_ema_alpha
        self.prediction_error_ema = (
            alpha * pred_error + (1 - alpha) * self.prediction_error_ema
        )
        self.novelty_history.append(pred_error)
    
    def get_novelty_score(self) -> float:
        """
        Compute novelty score based on prediction error.
        High novelty = world model struggling = interesting region.
        """
        if len(self.novelty_history) < 10:
            return 0.0
        
        recent = list(self.novelty_history)[-100:]
        baseline = np.percentile(recent, 50)
        
        # Novelty is how much current error exceeds baseline
        if baseline > 0:
            novelty = self.prediction_error_ema / (baseline + 1e-6)
            return min(novelty, 2.0)  # Cap at 2x
        return 0.0
    
    def is_compression_failure(self) -> bool:
        """Check if current novelty indicates compression failure."""
        return self.get_novelty_score() > self.config.novelty_threshold
    
    def get_state_dict_for_checkpoint(self) -> dict:
        """Get state for checkpointing (for self-modification rollback)."""
        return {
            "encoder": self.encoder.state_dict(),
            "ssm": self.ssm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "novelty_ema": self.prediction_error_ema
        }
    
    def load_from_checkpoint(self, checkpoint: dict):
        """Load state from checkpoint."""
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.ssm.load_state_dict(checkpoint["ssm"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
        self.prediction_error_ema = checkpoint["novelty_ema"]
