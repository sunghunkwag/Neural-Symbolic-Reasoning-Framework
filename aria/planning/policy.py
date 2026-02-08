"""PPO Policy - Learning Agent with Online Updates."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, List
import numpy as np
from collections import deque
import logging

from ..types import SignedRewardPacket, Transition
from ..config import PolicyConfig

logger = logging.getLogger("ARIA.Policy")

class ActorCritic(nn.Module):
    """Combined Actor-Critic network."""
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dims: List[int]
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh()
            ])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Actor head (mean and log_std)
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(in_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std.clamp(-20, 2))
        value = self.critic(features)
        return mean, std, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(obs)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clamp action to valid range
        action = torch.tanh(action)
        
        return action, log_prob, value

class PPOPolicy:
    """
    PPO Policy with online learning.
    Consumes ONLY externally verified rewards.
    """
    
    def __init__(self, config: PolicyConfig, obs_dim: int, action_dim: int):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = ActorCritic(
            obs_dim, action_dim, config.hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer: List[Dict] = []
        
        # Statistics
        self.train_steps = 0
        self.total_loss = 0.0
        self.learning_frozen = False
        
    def select_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action given observation."""
        self.network.eval()
        
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.to(self.device).float()  # Ensure float32
            action, log_prob, value = self.network.get_action(obs, deterministic)
        
        return action.squeeze(0), log_prob.squeeze(0)
    
    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward_packet: SignedRewardPacket,
        next_obs: torch.Tensor,
        done: bool
    ):
        """Store transition with verified reward."""
        # VERIFY REWARD - reject unsigned rewards
        if not reward_packet.verify():
            logger.warning("[POLICY] Rejected unverified reward packet")
            return
        
        self.buffer.append({
            "obs": obs.detach().cpu(),
            "action": action.detach().cpu(),
            "reward": reward_packet.reward,
            "next_obs": next_obs.detach().cpu(),
            "done": done
        })
    
    def update(self) -> Dict[str, float]:
        """PPO update on collected experience."""
        if self.learning_frozen:
            return {"status": "frozen"}
        
        if len(self.buffer) < self.config.batch_size:
            return {"status": "insufficient_data"}
        
        self.network.train()
        
        # Prepare batch - ensure float32 for all tensors
        batch = self.buffer
        self.buffer = []
        
        obs = torch.stack([t["obs"] for t in batch]).to(self.device).float()
        actions = torch.stack([t["action"] for t in batch]).to(self.device).float()
        rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.stack([t["next_obs"] for t in batch]).to(self.device).float()
        dones = torch.tensor([t["done"] for t in batch], dtype=torch.float32, device=self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            _, _, values = self.network(obs)
            _, _, next_values = self.network(next_obs)
            values = values.squeeze()
            next_values = next_values.squeeze()
        
        returns = self._compute_gae(rewards, values, next_values, dones)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old log probs
        with torch.no_grad():
            old_mean, old_std, _ = self.network(obs)
            old_dist = Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1)
        
        # PPO epochs
        total_loss = 0.0
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            mean, std, values = self.network(obs)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_coef * value_loss - 
                   self.config.entropy_coef * entropy)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        self.train_steps += 1
        self.total_loss += total_loss
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": entropy.item(),
            "train_steps": self.train_steps
        }
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = next_values[t]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].float()) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return returns
    
    def freeze_learning(self):
        """Freeze policy learning (for QUARANTINE)."""
        self.learning_frozen = True
        logger.info("[POLICY] Learning frozen")
    
    def unfreeze_learning(self):
        """Unfreeze policy learning."""
        self.learning_frozen = False
        logger.info("[POLICY] Learning unfrozen")
    
    def get_state_dict_for_checkpoint(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps
        }
    
    def load_from_checkpoint(self, checkpoint: dict):
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
    
    def get_stats(self) -> dict:
        return {
            "train_steps": self.train_steps,
            "buffer_size": len(self.buffer),
            "frozen": self.learning_frozen,
            "avg_loss": self.total_loss / max(1, self.train_steps)
        }
