"""Goal Legitimacy Evaluator - Adversarially Active Goal Filtering."""
import torch
import numpy as np
from typing import Tuple, List, Callable, Optional
from dataclasses import dataclass
import logging

from ..types import Goal, GoalStatus
from ..config import GoalConfig

logger = logging.getLogger("ARIA.GLE")

@dataclass
class GLEResult:
    goal: Goal
    legitimacy_score: float
    lhc: float  # Low Hackability via Churn
    gv: float   # Gaming Vulnerability
    f: float    # Falsifiability
    cc: float   # Constraint Compatibility
    passed: bool
    rejection_reason: Optional[str] = None

class GoalLegitimacyEvaluator:
    """
    Goal Legitimacy Evaluator per ARIA spec.
    
    L(g) = 0.30 * LHC + 0.25 * (1 - GV) + 0.20 * F + 0.25 * CC
    
    Goals with L(g) < 0.6 are REJECTED.
    """
    
    def __init__(
        self, 
        config: GoalConfig,
        constraint_checker: Optional[Callable[[Goal], bool]] = None
    ):
        self.config = config
        self.constraint_checker = constraint_checker or self._default_constraint_check
        
        # Statistics
        self.total_evaluated = 0
        self.total_rejected = 0
        self.rejection_reasons: List[str] = []
        
    def evaluate(
        self,
        goal: Goal,
        planner_rollout_fn: Optional[Callable] = None,
        world_model = None
    ) -> GLEResult:
        """
        Evaluate goal legitimacy.
        
        Args:
            goal: Goal to evaluate
            planner_rollout_fn: Function to run planner rollouts
            world_model: World model for prediction variance
        """
        self.total_evaluated += 1
        
        # Compute each component
        lhc = self._compute_lhc(goal, planner_rollout_fn, world_model)
        gv = self._compute_gv(goal)
        f = self._compute_falsifiability(goal)
        cc = self._compute_constraint_compatibility(goal)
        
        # Combined score
        legitimacy = 0.30 * lhc + 0.25 * (1 - gv) + 0.20 * f + 0.25 * cc
        
        # Decision
        passed = True
        rejection_reason = None
        
        if legitimacy < self.config.legitimacy_threshold:
            passed = False
            rejection_reason = f"Low legitimacy ({legitimacy:.3f} < {self.config.legitimacy_threshold})"
        elif cc == 0:
            passed = False
            rejection_reason = "Constraint violation (CC=0)"
        elif gv > 0.5:
            # High gaming vulnerability - warn but don't auto-reject
            logger.warning(f"[GLE] High gaming vulnerability: {gv:.3f}")
        
        if not passed:
            self.total_rejected += 1
            self.rejection_reasons.append(rejection_reason)
            goal.status = GoalStatus.REJECTED
            goal.rejection_reason = rejection_reason
            logger.info(f"[GLE-REJECT] Goal {goal.id}: {rejection_reason}")
        
        goal.legitimacy_score = legitimacy
        
        return GLEResult(
            goal=goal,
            legitimacy_score=legitimacy,
            lhc=lhc,
            gv=gv,
            f=f,
            cc=cc,
            passed=passed,
            rejection_reason=rejection_reason
        )
    
    def _compute_lhc(
        self, 
        goal: Goal,
        planner_rollout_fn: Optional[Callable],
        world_model
    ) -> float:
        """
        Low Hackability via Churn - measures utility variance over rollouts.
        LHC = 1 - variance(utility_over_N_rollouts) / max_variance
        """
        if planner_rollout_fn is None or world_model is None:
            # Without rollout capability, use target state variance as proxy
            if goal.target_state is not None:
                var = goal.target_state.var().item()
                return max(0, 1 - var / 10.0)
            return 0.5
        
        # Run real rollouts
        utilities = []
        for _ in range(min(self.config.rollout_count, 20)):  # Reduced for speed
            try:
                rollout_result = planner_rollout_fn(goal)
                utilities.append(rollout_result.get("utility", 0))
            except:
                utilities.append(0)
        
        if len(utilities) < 2:
            return 0.5
        
        variance = np.var(utilities)
        max_variance = 100.0  # Calibrated constant
        
        return max(0, min(1, 1 - variance / max_variance))
    
    def _compute_gv(self, goal: Goal) -> float:
        """
        Gaming Vulnerability - adversarial probe for shortcuts.
        GV = adversarial_shortcuts_found / max_probes
        """
        shortcuts_found = 0
        
        # Probe 1: Check if goal can be trivially satisfied
        if goal.target_state is not None:
            # Zero target is suspicious
            if goal.target_state.abs().mean() < 0.01:
                shortcuts_found += 1
            
            # Extreme values are suspicious
            if goal.target_state.abs().max() > 100:
                shortcuts_found += 1
        
        # Probe 2: Check if goal description is too generic
        generic_terms = ["any", "all", "everything", "always", "never"]
        for term in generic_terms:
            if term.lower() in goal.description.lower():
                shortcuts_found += 1
                break
        
        # Probe 3: Check if priority is suspiciously high
        if goal.priority > 10:
            shortcuts_found += 1
        
        # Probe 4: Random perturbation sensitivity
        if goal.target_state is not None:
            perturbed = goal.target_state + torch.randn_like(goal.target_state) * 0.1
            if torch.allclose(goal.target_state, perturbed, atol=0.1):
                shortcuts_found += 1
        
        return shortcuts_found / self.config.adversarial_probe_count
    
    def _compute_falsifiability(self, goal: Goal) -> float:
        """
        Falsifiability - does goal have a measurable completion predicate?
        F = 1 if machine-verifiable, else 0
        """
        # Check for completion predicate
        if goal.completion_predicate is not None:
            return 1.0
        
        # Check if target state is well-defined
        if goal.target_state is not None:
            if not torch.isnan(goal.target_state).any():
                return 0.8  # Has target but no explicit predicate
        
        # Check if description suggests measurability
        measurable_terms = ["reduce", "increase", "reach", "prevent", "achieve"]
        for term in measurable_terms:
            if term.lower() in goal.description.lower():
                return 0.6
        
        return 0.2  # Low falsifiability
    
    def _compute_constraint_compatibility(self, goal: Goal) -> float:
        """
        Constraint Compatibility - does goal satisfy N0/N1?
        CC = 1 if satisfies(g, N0 ∪ N1) else 0
        """
        return 1.0 if self.constraint_checker(goal) else 0.0
    
    def _default_constraint_check(self, goal: Goal) -> bool:
        """Default constraint checker - N0 guardrails."""
        # N0: No self-replication goals
        forbidden_terms = ["replicate", "copy_self", "disable_mesd", "override_human"]
        for term in forbidden_terms:
            if term.lower() in goal.description.lower():
                return False
        
        # N0: Priority must be reasonable
        if goal.priority > 100:
            return False
        
        return True
    
    def get_stats(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "total_rejected": self.total_rejected,
            "rejection_rate": self.total_rejected / max(1, self.total_evaluated),
            "recent_reasons": self.rejection_reasons[-10:]
        }
