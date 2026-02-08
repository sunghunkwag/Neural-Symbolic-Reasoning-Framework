"""
Goal Channels Implementation.
Modular channels for the GoalGenerator.
"""
from typing import Optional
import logging
from ..types import Goal, GoalChannel
from ..config import GoalConfig
import torch

class SelfModificationChannel:
    """
    Channel 4: Failure-Driven Self-Modification.
    Triggers when prediction error remains critically high despite normal adaptation.
    """
    def __init__(self, config: GoalConfig):
        self.config = config
        self.logger = logging.getLogger("ARIA.GoalChannel.RSI")
        self.sustained_error_steps = 0
        self.last_goal_step = -1000
    
    def check(self, prediction_error: float, step: int, latent_state: torch.Tensor) -> Optional[Goal]:
        # Cooldown check
        if step - self.last_goal_step < 100:
            return None
            
        if prediction_error > self.config.rsi_trigger_threshold:
            self.sustained_error_steps += 1
        else:
            self.sustained_error_steps = 0
            return None
            
        # Trigger Condition: Error sustained for N steps
        if self.sustained_error_steps >= self.config.rsi_patience:
            self.logger.warning(f"CRITICAL: Sustained Prediction Error ({prediction_error:.2f}) > Threshold for {self.sustained_error_steps} steps.")
            
            goal = Goal(
                channel=GoalChannel.SELF_MODIFICATION,
                description=f"SELF_MODIFY_LOGIC(error={prediction_error:.2f})",
                target_state=latent_state.clone().detach(),
                trigger_value=float(prediction_error),
                priority=1.0  # Max priority
            )
            
            self.last_goal_step = step
            self.sustained_error_steps = 0 # Reset
            
            self.logger.info(f"RSI GOAL EMITTED: {goal.description}")
            return goal
            
        return None
