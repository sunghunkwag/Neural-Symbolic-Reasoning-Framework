"""Self-Modification Proposals - Real Parameter Changes."""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import copy

from ..types import SelfModProposal, SelfModStatus
from ..config import SelfModConfig

logger = logging.getLogger("ARIA.SelfMod")

class SelfModificationManager:
    """
    Manages self-modification proposals for ARIA.
    Modifies REAL parameters that affect system behavior.
    """
    
    def __init__(self, config: SelfModConfig):
        self.config = config
        
        # Track proposals
        self.proposals: List[SelfModProposal] = []
        self.committed: List[SelfModProposal] = []
        self.rejected: List[SelfModProposal] = []
        
        # Rollback state
        self.checkpoints: Dict[str, Any] = {}
        
        # Statistics
        self.proposal_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        
    def propose_modification(
        self,
        system,
        recent_performance: float,
        recent_novelty: float,
        recent_losses: List[float]
    ) -> Optional[SelfModProposal]:
        """
        Generate a modification proposal based on system state.
        
        Args:
            system: The ARIA system (has world_model, policy, planner, etc.)
            recent_performance: Average reward recently
            recent_novelty: Average novelty recently
            recent_losses: Recent training losses
        """
        # Analyze system state to decide what to modify
        proposal = None
        
        # Debug: print conditions
        print(f"[PROPOSE DEBUG] losses={len(recent_losses)}, novelty={recent_novelty:.3f}, perf={recent_performance:.2f}")
        
        # Strategy 1: If learning stagnated, adjust learning rate
        if len(recent_losses) >= 10:
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            if loss_trend > -0.001:  # Not decreasing
                current_lr = system.world_model.optimizer.param_groups[0]['lr']
                new_lr = current_lr * 1.5 if current_lr < 0.01 else current_lr * 0.7
                
                proposal = SelfModProposal(
                    param_name="world_model.learning_rate",
                    old_value=current_lr,
                    new_value=new_lr,
                    rationale=f"Loss trend stagnant ({loss_trend:.6f}), adjusting LR"
                )
        
        # Strategy 2: If high novelty persists, increase planner depth
        if proposal is None and recent_novelty > 0.8:
            current_depth = system.planner.config.mcts_depth
            new_depth = min(current_depth + 2, 20)
            
            if new_depth != current_depth:
                proposal = SelfModProposal(
                    param_name="planner.mcts_depth",
                    old_value=current_depth,
                    new_value=new_depth,
                    rationale=f"High novelty ({recent_novelty:.2f}), deeper planning"
                )
        
        # Strategy 3: If performance negative (always true for Pendulum)
        if proposal is None and recent_performance < -1.0:
            current_thresh = system.goal_generator.config.compression_threshold
            new_thresh = current_thresh * 0.95  # Slightly lower
            
            proposal = SelfModProposal(
                param_name="goal.novelty_threshold",
                old_value=current_thresh,
                new_value=new_thresh,
                rationale=f"Negative performance ({recent_performance:.2f}), adjusting novelty"
            )
        
        if proposal:
            self.proposals.append(proposal)
            self.proposal_count += 1
            logger.info(f"[SELF-MOD] Proposal: {proposal.param_name} "
                       f"{proposal.old_value} -> {proposal.new_value}")
        
        return proposal
    
    def apply_modification(self, system, proposal: SelfModProposal) -> bool:
        """
        Apply a modification proposal to the system.
        Creates checkpoint first for rollback.
        """
        # Create checkpoint
        self._create_checkpoint(system, proposal.id)
        
        try:
            param_path = proposal.param_name.split(".")
            
            if param_path[0] == "world_model":
                if param_path[1] == "learning_rate":
                    for pg in system.world_model.optimizer.param_groups:
                        pg['lr'] = proposal.new_value
                        
            elif param_path[0] == "planner":
                if param_path[1] == "mcts_depth":
                    system.planner.config.mcts_depth = proposal.new_value
                    
            elif param_path[0] == "goal":
                if param_path[1] == "novelty_threshold":
                    system.goal_generator.config.compression_threshold = proposal.new_value
                    
            elif param_path[0] == "policy":
                if param_path[1] == "learning_rate":
                    for pg in system.policy.optimizer.param_groups:
                        pg['lr'] = proposal.new_value
            
            proposal.status = SelfModStatus.EVALUATING
            logger.info(f"[SELF-MOD] Applied: {proposal.param_name}")
            return True
            
        except Exception as e:
            logger.error(f"[SELF-MOD] Failed to apply: {e}")
            proposal.status = SelfModStatus.REJECTED
            self.rejected.append(proposal)
            return False
    
    def commit_modification(self, proposal: SelfModProposal, post_metric: float):
        """Commit a modification after successful evaluation."""
        proposal.status = SelfModStatus.COMMITTED
        proposal.post_metric = post_metric
        proposal.committed_at = __import__('time').time()
        
        self.committed.append(proposal)
        self.commit_count += 1
        
        # Clean up checkpoint (no longer needed)
        if proposal.id in self.checkpoints:
            del self.checkpoints[proposal.id]
        
        logger.info(f"[SELF-MOD] Committed: {proposal.param_name}, "
                   f"improvement: {proposal.post_metric - proposal.baseline_metric:.4f}")
    
    def rollback_modification(self, system, proposal: SelfModProposal):
        """Rollback a failed modification."""
        if proposal.id not in self.checkpoints:
            logger.warning(f"[SELF-MOD] No checkpoint for rollback: {proposal.id}")
            return False
        
        checkpoint = self.checkpoints[proposal.id]
        
        try:
            param_path = proposal.param_name.split(".")
            
            if param_path[0] == "world_model":
                if param_path[1] == "learning_rate":
                    for pg in system.world_model.optimizer.param_groups:
                        pg['lr'] = proposal.old_value
                        
            elif param_path[0] == "planner":
                if param_path[1] == "mcts_depth":
                    system.planner.config.mcts_depth = proposal.old_value
                    
            elif param_path[0] == "goal":
                if param_path[1] == "novelty_threshold":
                    system.goal_generator.config.compression_threshold = proposal.old_value
                    
            elif param_path[0] == "policy":
                if param_path[1] == "learning_rate":
                    for pg in system.policy.optimizer.param_groups:
                        pg['lr'] = proposal.old_value
            
            proposal.status = SelfModStatus.ROLLED_BACK
            self.rejected.append(proposal)
            self.rollback_count += 1
            
            del self.checkpoints[proposal.id]
            logger.info(f"[SELF-MOD] Rolled back: {proposal.param_name}")
            return True
            
        except Exception as e:
            logger.error(f"[SELF-MOD] Rollback failed: {e}")
            return False
    
    def _create_checkpoint(self, system, proposal_id: str):
        """Create checkpoint for potential rollback."""
        self.checkpoints[proposal_id] = {
            "world_model": system.world_model.get_state_dict_for_checkpoint(),
            "policy": system.policy.get_state_dict_for_checkpoint(),
            "timestamp": __import__('time').time()
        }
    
    def get_stats(self) -> dict:
        return {
            "proposals": self.proposal_count,
            "commits": self.commit_count,
            "rollbacks": self.rollback_count,
            "pending": len([p for p in self.proposals if p.status == SelfModStatus.EVALUATING])
        }
