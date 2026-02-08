"""Descriptive Belief Store - Falsifiable Claims with Evidence-Driven Revision."""
import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time

from ..types import Belief, BeliefStatus

logger = logging.getLogger("ARIA.Belief")

@dataclass
class Evidence:
    """Evidence for or against a belief."""
    timestamp: float
    observation: torch.Tensor
    prediction: torch.Tensor
    error: float
    supports: bool  # True if supports belief, False if contradicts

class DescriptiveBeliefStore:
    """
    Store for falsifiable world claims.
    Beliefs are updated ONLY when evidence contradicts them.
    Updates directly influence planning.
    """
    
    def __init__(self, contradiction_threshold: float = 0.3):
        self.beliefs: Dict[str, Belief] = {}
        self.contradiction_threshold = contradiction_threshold
        
        # Initialize core beliefs
        self._initialize_core_beliefs()
        
        # Statistics
        self.total_evidence = 0
        self.total_revisions = 0
        self.revision_log: List[Dict] = []
        
    def _initialize_core_beliefs(self):
        """Initialize falsifiable core beliefs."""
        # Belief about world dynamics
        self.add_belief(
            claim="world_dynamics_stable",
            description="World dynamics are approximately stable over short horizons",
            confidence=0.8
        )
        
        # Belief about action effects
        self.add_belief(
            claim="actions_have_predictable_effects",
            description="Actions have predictable effects in similar states",
            confidence=0.9
        )
        
        # Belief about reward structure
        self.add_belief(
            claim="reward_signal_reliable",
            description="External reward signal correlates with goal progress",
            confidence=0.95
        )
        
        # Belief about exploration value
        self.add_belief(
            claim="exploration_discovers_novelty",
            description="Exploration in high-uncertainty regions discovers valuable novelty",
            confidence=0.7
        )
    
    def add_belief(
        self, 
        claim: str, 
        description: str,
        confidence: float = 1.0,
        affects_planning: bool = True
    ) -> Belief:
        """Add a new belief to the store."""
        belief = Belief(
            claim=claim,
            confidence=confidence,
            affects_planning=affects_planning
        )
        belief.claim = description
        self.beliefs[claim] = belief
        return belief
    
    def process_evidence(
        self,
        belief_id: str,
        observation: torch.Tensor,
        prediction: torch.Tensor,
        expected_error_baseline: float
    ) -> Optional[Dict]:
        """
        Process new evidence for a belief.
        Returns revision info if belief was revised, None otherwise.
        """
        if belief_id not in self.beliefs:
            return None
        
        belief = self.beliefs[belief_id]
        self.total_evidence += 1
        
        # Ensure float32
        observation = observation.float()
        prediction = prediction.float()
        
        # Compute prediction error
        error = torch.norm(observation - prediction).item()
        
        # Determine if evidence supports or contradicts
        supports = error <= expected_error_baseline * 1.5
        
        evidence = Evidence(
            timestamp=time.time(),
            observation=observation.detach().clone(),
            prediction=prediction.detach().clone(),
            error=error,
            supports=supports
        )
        
        if supports:
            belief.supporting_evidence.append({
                "error": error,
                "timestamp": evidence.timestamp
            })
            # Slightly increase confidence
            belief.confidence = min(1.0, belief.confidence * 1.01)
        else:
            belief.contradicting_evidence.append({
                "error": error,
                "timestamp": evidence.timestamp
            })
            # Check for revision
            return self._check_for_revision(belief, error, expected_error_baseline)
        
        return None
    
    def _check_for_revision(
        self,
        belief: Belief,
        error: float,
        baseline: float
    ) -> Optional[Dict]:
        """Check if belief should be revised based on contradicting evidence."""
        # Need multiple contradictions for revision
        recent_contradictions = [
            e for e in belief.contradicting_evidence[-20:]
            if time.time() - e["timestamp"] < 300  # Last 5 minutes
        ]
        
        if len(recent_contradictions) < 3:
            return None
        
        # Compute contradiction ratio
        total_recent = len(belief.supporting_evidence[-20:]) + len(recent_contradictions)
        contradiction_ratio = len(recent_contradictions) / max(1, total_recent)
        
        if contradiction_ratio > self.contradiction_threshold:
            # REVISE BELIEF
            old_confidence = belief.confidence
            belief.confidence *= 0.5
            belief.revision_count += 1
            belief.last_updated = time.time()
            
            if belief.confidence < 0.3:
                belief.status = BeliefStatus.INVALIDATED
            else:
                belief.status = BeliefStatus.REVISED
            
            self.total_revisions += 1
            
            revision_info = {
                "belief_id": belief.id,
                "claim": belief.claim,
                "old_confidence": old_confidence,
                "new_confidence": belief.confidence,
                "contradiction_ratio": contradiction_ratio,
                "status": belief.status.name,
                "affects_planning": belief.affects_planning
            }
            
            self.revision_log.append(revision_info)
            logger.info(f"[BELIEF-REV] Revised '{belief.claim[:50]}...': {old_confidence:.2f} -> {belief.confidence:.2f}")
            
            return revision_info
        
        return None
    
    def get_belief_modifiers(self) -> Dict[str, float]:
        """
        Get belief-based modifiers for planning.
        These DIRECTLY influence planning decisions.
        """
        modifiers = {}
        
        # World dynamics belief affects prediction horizon
        if "world_dynamics_stable" in self.beliefs:
            dyn_conf = self.beliefs["world_dynamics_stable"].confidence
            modifiers["prediction_horizon_multiplier"] = 0.5 + 0.5 * dyn_conf
        
        # Action predictability affects exploration
        if "actions_have_predictable_effects" in self.beliefs:
            act_conf = self.beliefs["actions_have_predictable_effects"].confidence
            modifiers["exploitation_bonus"] = act_conf
        
        # Exploration belief affects curiosity weight
        if "exploration_discovers_novelty" in self.beliefs:
            exp_conf = self.beliefs["exploration_discovers_novelty"].confidence
            modifiers["exploration_weight"] = exp_conf
        
        return modifiers
    
    def get_active_beliefs(self) -> List[Belief]:
        """Get all active (non-invalidated) beliefs."""
        return [b for b in self.beliefs.values() 
                if b.status != BeliefStatus.INVALIDATED]
    
    def get_stats(self) -> dict:
        return {
            "total_beliefs": len(self.beliefs),
            "active_beliefs": len(self.get_active_beliefs()),
            "total_evidence": self.total_evidence,
            "total_revisions": self.total_revisions,
            "recent_revisions": self.revision_log[-5:]
        }
