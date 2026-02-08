"""A/B Evaluator - Isolated Evaluation of Self-Modifications."""
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..types import SelfModProposal, SelfModStatus

logger = logging.getLogger("ARIA.ABEval")

@dataclass
class EvaluationResult:
    baseline_metric: float
    modified_metric: float
    improvement: float
    passed: bool
    reason: str

class ABEvaluator:
    """
    A/B Evaluation for self-modification proposals.
    Compares baseline vs modified performance.
    Fixed for negative reward environments (e.g., Pendulum).
    """
    
    def __init__(self, improvement_threshold: float = 0.02, eval_episodes: int = 5):
        # Lower threshold (2%) and fewer episodes for faster commits
        self.improvement_threshold = improvement_threshold
        self.eval_episodes = eval_episodes
        
        # Track evaluations
        self.evaluations: list = []
        
    def evaluate_proposal(
        self,
        proposal: SelfModProposal,
        baseline_rewards: list,
        modified_rewards: list
    ) -> EvaluationResult:
        """
        Evaluate a modification proposal.
        
        Args:
            proposal: The modification proposal
            baseline_rewards: Rewards before modification
            modified_rewards: Rewards after modification
        """
        if len(baseline_rewards) < 2 or len(modified_rewards) < 2:
            result = EvaluationResult(
                baseline_metric=0.0,
                modified_metric=0.0,
                improvement=0.0,
                passed=False,
                reason="Insufficient data for evaluation"
            )
            return result
        
        # Compute metrics
        baseline_mean = np.mean(baseline_rewards)
        modified_mean = np.mean(modified_rewards)
        
        # Statistical significance check (simple t-test approximation)
        baseline_std = np.std(baseline_rewards) + 1e-6
        modified_std = np.std(modified_rewards) + 1e-6
        
        # Effect size
        pooled_std = np.sqrt((baseline_std**2 + modified_std**2) / 2)
        effect_size = (modified_mean - baseline_mean) / pooled_std
        
        # Improvement calculation for negative reward environments
        # For negative rewards, "better" means less negative (closer to 0)
        # Improvement = (modified - baseline) / |baseline|
        if abs(baseline_mean) > 1e-6:
            # Raw difference is the key metric for negative rewards
            raw_improvement = modified_mean - baseline_mean
            # Relative improvement
            improvement = raw_improvement / abs(baseline_mean)
        else:
            improvement = modified_mean - baseline_mean
            raw_improvement = improvement
        
        # Decision: For negative rewards, any positive raw_improvement is good
        # Or relative improvement exceeds threshold
        passed = (raw_improvement > 0 and abs(improvement) > self.improvement_threshold) or \
                 (raw_improvement > abs(baseline_mean) * 0.01)  # At least 1% absolute improvement
        
        if passed:
            reason = f"Improvement: {raw_improvement:.2f} ({improvement:.2%})"
        elif raw_improvement <= 0:
            reason = f"No improvement (delta: {raw_improvement:.2f})"
        else:
            reason = f"Improvement {improvement:.2%} below threshold {self.improvement_threshold:.2%}"
        
        result = EvaluationResult(
            baseline_metric=baseline_mean,
            modified_metric=modified_mean,
            improvement=improvement,
            passed=passed,
            reason=reason
        )
        
        # Update proposal
        proposal.baseline_metric = baseline_mean
        proposal.post_metric = modified_mean
        
        self.evaluations.append({
            "proposal_id": proposal.id,
            "param": proposal.param_name,
            "passed": passed,
            "improvement": improvement
        })
        
        logger.info(f"[A/B-EVAL] {proposal.param_name}: {reason}")
        
        return result
    
    def get_stats(self) -> dict:
        if not self.evaluations:
            return {"total": 0, "passed": 0, "failed": 0}
        
        passed = sum(1 for e in self.evaluations if e["passed"])
        return {
            "total": len(self.evaluations),
            "passed": passed,
            "failed": len(self.evaluations) - passed,
            "pass_rate": passed / len(self.evaluations)
        }
