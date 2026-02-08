"""Goal Generator - 3 Channel Autonomous Goal Invention."""
import torch
import numpy as np
from typing import Optional, List, Dict
from collections import deque
from dataclasses import dataclass
import logging

from ..types import Goal, GoalChannel, GoalStatus
from ..config import GoalConfig
from .channels import SelfModificationChannel

logger = logging.getLogger("ARIA.GoalGenerator")

@dataclass
class FailurePattern:
    """Detected pattern of repeated failures."""
    pattern_id: str
    description: str
    state_signature: torch.Tensor
    occurrences: int = 0
    last_seen: int = 0

class GoalGenerator:
    """
    Autonomous Goal Generator with 4 channels:
    1. Compression Failure - high novelty / prediction error
    2. Failure Pattern - repeated failure class detection
    3. Anomaly Discovery - latent space outliers
    4. Self-Modification (RSI) - failure-driven evolution
    
    Goals are NOT hard-coded or enumerable.
    """
    
    def __init__(self, config: GoalConfig, latent_dim: int):
        self.config = config
        self.latent_dim = latent_dim
        
        # Channel 1: Compression failure tracking
        self.novelty_history = deque(maxlen=500)
        self.compression_threshold_adaptive = config.compression_threshold
        
        # Channel 2: Failure pattern detection
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.recent_failures = deque(maxlen=100)
        
        # Channel 3: Anomaly detection - maintain latent manifold estimate
        self.latent_samples = deque(maxlen=1000)
        self.manifold_mean: Optional[torch.Tensor] = None
        self.manifold_cov: Optional[torch.Tensor] = None
        
        # Channel 4: Self-Modification (RSI)
        self.rsi_channel = SelfModificationChannel(config)
        
        # Statistics
        self.goals_generated = 0
        self.step_count = 0
        
    def step(
        self, 
        latent_state: torch.Tensor,
        novelty_score: float,
        episode_failed: bool,
        reward: float
    ) -> Optional[Goal]:
        """
        Process current state and potentially generate a goal.
        Returns Goal if generated, None otherwise.
        """
        self.step_count += 1
        
        # Ensure float32
        latent_state = latent_state.float()
        
        # Update internal state
        self._update_novelty(novelty_score)
        self._update_manifold(latent_state)
        if episode_failed:
            self._record_failure(latent_state)
        
        # Channel 4: RSI Check (Highest Priority - System Health)
        # We check this first or alongside others.
        # It uses prediction error (novelty_score) as proxy for system failure.
        rsi_goal = self.rsi_channel.check(novelty_score, self.step_count, latent_state)
        if rsi_goal:
            self.goals_generated += 1
            return rsi_goal

        # Try each channel
        goal = self._try_compression_failure(latent_state, novelty_score)
        if goal:
            return goal
            
        goal = self._try_failure_pattern()
        if goal:
            return goal
            
        goal = self._try_anomaly_discovery(latent_state)
        if goal:
            return goal
        
        return None
    
    def _try_compression_failure(
        self, 
        latent_state: torch.Tensor,
        novelty_score: float
    ) -> Optional[Goal]:
        """
        Channel 1: Generate goal when world model struggles to predict.
        High novelty = interesting region worth exploring.
        """
        if novelty_score > self.compression_threshold_adaptive:
            # Sustained high novelty check
            recent = list(self.novelty_history)[-20:]
            if len(recent) >= 5 and np.mean(recent) > self.config.compression_threshold * 0.8:
                goal = Goal(
                    channel=GoalChannel.COMPRESSION_FAILURE,
                    description=f"REDUCE_UNCERTAINTY(region_{self.step_count})",
                    target_state=latent_state.clone().detach(),
                    trigger_value=novelty_score,
                    priority=novelty_score
                )
                
                # Adapt threshold to prevent goal spam
                self.compression_threshold_adaptive = min(
                    self.compression_threshold_adaptive * 1.1,
                    self.config.compression_threshold * 2.0
                )
                
                self.goals_generated += 1
                logger.info(f"[GOAL-GEN] Compression failure goal: {goal.description}")
                return goal
                
        # Slowly decay adaptive threshold
        self.compression_threshold_adaptive = max(
            self.compression_threshold_adaptive * 0.999,
            self.config.compression_threshold * 0.5
        )
        return None
    
    def _try_failure_pattern(self) -> Optional[Goal]:
        """
        Channel 2: Generate goal when a failure class repeats >= 3 times.
        """
        for pattern_id, pattern in self.failure_patterns.items():
            if pattern.occurrences >= self.config.failure_pattern_min_count:
                # Only generate if not recently addressed
                # Only generate if not recently addressed
                if self.step_count - pattern.last_seen < 50:
                    # Escalation Logic: If pattern persists despite previous goals, try Logic Synthesis
                    if pattern.occurrences > 5:
                        channel = GoalChannel.LOGIC_FAILURE
                        description = f"SYNTHESIZE_LOGIC({pattern_id})"
                        priority = pattern.occurrences / 5.0 # Higher priority
                    else:
                        channel = GoalChannel.FAILURE_PATTERN
                        description = f"PREVENT_FAILURE_CLASS({pattern_id})"
                        priority = pattern.occurrences / 10.0

                    goal = Goal(
                        channel=channel,
                        description=description,
                        target_state=pattern.state_signature.clone(),
                        trigger_value=float(pattern.occurrences),
                        priority=priority
                    )
                    
                    # Reset pattern to prevent repeated goals
                    pattern.occurrences = 0
                    
                    self.goals_generated += 1
                    logger.info(f"[GOAL-GEN] Failure pattern goal: {goal.description}")
                    return goal
        return None
    
    def _try_anomaly_discovery(
        self, 
        latent_state: torch.Tensor
    ) -> Optional[Goal]:
        """
        Channel 3: Generate goal for latent space outliers.
        Uses Mahalanobis distance to manifold estimate.
        """
        if self.manifold_mean is None or len(self.latent_samples) < 100:
            return None
        
        # Compute Mahalanobis distance
        diff = latent_state.squeeze() - self.manifold_mean
        try:
            inv_cov = torch.linalg.pinv(self.manifold_cov + 1e-4 * torch.eye(self.latent_dim))
            mahal_dist = torch.sqrt(diff @ inv_cov @ diff).item()
        except:
            return None
        
        # Normalize to [0, 1] range approximately
        normalized_dist = min(mahal_dist / 10.0, 1.0)
        
        if normalized_dist > self.config.anomaly_distance_threshold:
            goal = Goal(
                channel=GoalChannel.ANOMALY_DISCOVERY,
                description=f"EXPLORE_ANOMALY({self.step_count})",
                target_state=latent_state.clone().detach(),
                trigger_value=normalized_dist,
                priority=normalized_dist
            )
            
            self.goals_generated += 1
            logger.info(f"[GOAL-GEN] Anomaly discovery goal: {goal.description}")
            return goal
        
        return None
    
    def _update_novelty(self, novelty_score: float):
        self.novelty_history.append(novelty_score)
    
    def _update_manifold(self, latent_state: torch.Tensor):
        """Update running estimate of latent manifold."""
        z = latent_state.detach().squeeze()
        self.latent_samples.append(z)
        
        # Update statistics periodically
        if self.step_count % 50 == 0 and len(self.latent_samples) >= 50:
            samples = torch.stack(list(self.latent_samples))
            self.manifold_mean = samples.mean(dim=0)
            self.manifold_cov = torch.cov(samples.T)
    
    def _record_failure(self, latent_state: torch.Tensor):
        """Record failure for pattern detection."""
        z = latent_state.detach().squeeze()
        self.recent_failures.append((self.step_count, z))
        
        # Cluster recent failures
        self._detect_failure_patterns()
    
    def _detect_failure_patterns(self):
        """Simple failure pattern detection via clustering."""
        if len(self.recent_failures) < 3:
            return
        
        # Get recent failure states
        failures = list(self.recent_failures)[-20:]
        states = torch.stack([f[1] for f in failures])
        
        # Simple k-means-like clustering
        for i, (step, state) in enumerate(failures):
            pattern_id = f"pattern_{hash(tuple(state[:4].tolist())) % 1000}"
            
            if pattern_id in self.failure_patterns:
                self.failure_patterns[pattern_id].occurrences += 1
                self.failure_patterns[pattern_id].last_seen = step
            else:
                self.failure_patterns[pattern_id] = FailurePattern(
                    pattern_id=pattern_id,
                    description=f"Failure near state signature",
                    state_signature=state.clone(),
                    occurrences=1,
                    last_seen=step
                )
    
    def get_stats(self) -> dict:
        return {
            "goals_generated": self.goals_generated,
            "active_patterns": len(self.failure_patterns),
            "manifold_samples": len(self.latent_samples),
            "adaptive_threshold": self.compression_threshold_adaptive
        }
