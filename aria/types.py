"""ARIA v2.1 Core Data Types."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable
import torch
import numpy as np
import time
import uuid

class GoalChannel(Enum):
    COMPRESSION_FAILURE = auto()
    FAILURE_PATTERN = auto()
    ANOMALY_DISCOVERY = auto()
    LOGIC_FAILURE = auto()
    SELF_MODIFICATION = auto()

class GoalStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    ACHIEVED = auto()
    FAILED = auto()
    REJECTED = auto()

class BeliefStatus(Enum):
    ACTIVE = auto()
    REVISED = auto()
    INVALIDATED = auto()

class SelfModStatus(Enum):
    PROPOSED = auto()
    EVALUATING = auto()
    COMMITTED = auto()
    REJECTED = auto()
    ROLLED_BACK = auto()

@dataclass
class Goal:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channel: GoalChannel = GoalChannel.COMPRESSION_FAILURE
    description: str = ""
    target_state: Optional[torch.Tensor] = None
    trigger_value: float = 0.0
    priority: float = 0.0
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    legitimacy_score: float = 0.0
    rejection_reason: Optional[str] = None
    completion_predicate: Optional[Callable[[torch.Tensor], bool]] = None

    def __hash__(self):
        return hash(self.id)

@dataclass
class Belief:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    claim: str = ""
    confidence: float = 1.0
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    status: BeliefStatus = BeliefStatus.ACTIVE
    last_updated: float = field(default_factory=time.time)
    affects_planning: bool = True
    revision_count: int = 0

@dataclass
class SelfModProposal:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    param_name: str = ""
    old_value: Any = None
    new_value: Any = None
    rationale: str = ""
    status: SelfModStatus = SelfModStatus.PROPOSED
    baseline_metric: float = 0.0
    post_metric: float = 0.0
    created_at: float = field(default_factory=time.time)
    committed_at: Optional[float] = None

@dataclass
class SignedRewardPacket:
    """External reward signal - treated as trusted (env is E0)."""
    t: int
    reward: float
    obs_hash: int
    source: str = "gymnasium_env"
    
    def verify(self) -> bool:
        # In production: cryptographic verification
        # Here: env reward is trusted by design
        return self.source == "gymnasium_env"

@dataclass 
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: float
    next_obs: torch.Tensor
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Grid:
    """Immutable Grid Wrapper."""
    data: np.ndarray
    
    def __init__(self, data: Any):
        if isinstance(data, Grid):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.int8)
        else:
            self.data = np.array(data, dtype=np.int8)
            
    @property
    def H(self) -> int: return self.data.shape[0]
    
    @property
    def W(self) -> int: return self.data.shape[1]
    
    def to_list(self) -> List[List[int]]:
        return self.data.tolist()
        
    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)
        
    def __repr__(self):
        return f"Grid({self.data.shape[0]}x{self.data.shape[1]})"

    def copy(self) -> 'Grid':
        return Grid(self.data.copy())

@dataclass
class Metrics:
    episode: int = 0
    step: int = 0
    goals_generated: int = 0
    goals_rejected: int = 0
    goals_completed: int = 0
    beliefs_revised: int = 0
    self_mod_proposals: int = 0
    self_mod_commits: int = 0
    episode_reward: float = 0.0
    baseline_performance: float = 0.0
    current_performance: float = 0.0
