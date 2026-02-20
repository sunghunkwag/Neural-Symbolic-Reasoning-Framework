"""Recursive Self-Improvement (RSI) Engine for ARIA v3.0."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import copy


@dataclass
class HyperparameterConfig:
    """Current hyperparameters of the MCTS solver."""
    max_depth: int = 5
    mcts_iterations: int = 1000
    exploration_constant: float = 1.41
    action_sample_rate: float = 1.0  # Sampling rate for branching
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'max_depth': float(self.max_depth),
            'mcts_iterations': float(self.mcts_iterations),
            'exploration_constant': self.exploration_constant,
            'action_sample_rate': self.action_sample_rate
        }


class RecursiveSelfImprovement:
    """
    Recursive Self-Improvement Engine.
    
    Automatically proposes and tests modifications to solver hyperparameters.
    Commits improvements that increase average task performance.
    """
    
    def __init__(self, initial_hyperparams: HyperparameterConfig):
        self.current_hyperparams = copy.deepcopy(initial_hyperparams)
        self.performance_history = []  # List of (hyperparams, score)
        self.modification_history = []  # Successful modifications
        self.proposal_counter = 0
    
    def evaluate_performance(
        self,
        recent_scores: list,
        window_size: int = 20
    ) -> Tuple[float, float]:
        """
        Evaluate solver performance from recent task results.
        
        Returns:
            (avg_score, trend) where trend is positive if improving
        """
        if not recent_scores:
            return 0.0, 0.0
        
        scores = recent_scores[-window_size:]
        avg = np.mean(scores)
        
        if len(scores) > 1:
            trend = scores[-1] - scores[0]
        else:
            trend = 0.0
        
        return avg, trend
    
    def propose_modification(
        self,
        avg_score: float,
        trend: float
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a candidate modification to hyperparameters.
        
        Strategy:
        - If trend negative: increase exploration (more iterations, higher C)
        - If avg_score low: increase search depth
        - If converged: reduce branching factor (sample fewer actions)
        """
        modification = {}
        
        if trend < -0.1:  # Performance declining
            modification['mcts_iterations'] = int(self.current_hyperparams.mcts_iterations * 1.3)
            modification['exploration_constant'] = self.current_hyperparams.exploration_constant * 1.1
            reason = 'performance_declining'
        
        elif avg_score < 0.4:  # Baseline poor
            modification['max_depth'] = min(8, self.current_hyperparams.max_depth + 1)
            modification['mcts_iterations'] = int(self.current_hyperparams.mcts_iterations * 1.2)
            reason = 'baseline_poor'
        
        elif trend > 0.05 and avg_score > 0.7:  # Good convergence
            modification['action_sample_rate'] = max(0.3, self.current_hyperparams.action_sample_rate * 0.8)
            reason = 'reduce_branching'
        
        elif avg_score > 0.8:  # Near optimal
            modification['mcts_iterations'] = max(500, int(self.current_hyperparams.mcts_iterations * 0.7))
            reason = 'efficiency_boost'
        
        else:
            return None
        
        modification['reason'] = reason
        modification['timestamp'] = time.time()
        return modification if len(modification) > 2 else None
    
    def evaluate_modification(
        self,
        modification: Dict[str, Any],
        test_scores: list
    ) -> Tuple[float, bool]:
        """
        Test proposed modification on held-out test tasks.
        
        Returns:
            (improvement_delta, should_commit) where delta = new_score - old_score
        """
        if not test_scores:
            return 0.0, False
        
        new_score = np.mean(test_scores)
        baseline_score = np.mean(self.performance_history[-10:] if self.performance_history else [0.5])
        
        improvement = new_score - baseline_score
        should_commit = improvement > 0.02  # At least 2% improvement
        
        return improvement, should_commit
    
    def commit_modification(self, modification: Dict[str, Any]) -> None:
        """
        Apply successful modification permanently to hyperparameters.
        """
        # Copy current and update
        new_params = copy.deepcopy(self.current_hyperparams)
        
        if 'max_depth' in modification:
            new_params.max_depth = modification['max_depth']
        if 'mcts_iterations' in modification:
            new_params.mcts_iterations = modification['mcts_iterations']
        if 'exploration_constant' in modification:
            new_params.exploration_constant = modification['exploration_constant']
        if 'action_sample_rate' in modification:
            new_params.action_sample_rate = modification['action_sample_rate']
        
        self.current_hyperparams = new_params
        self.modification_history.append(modification)
    
    def record_performance(self, task_score: float) -> None:
        """
        Record a single task result.
        """
        self.performance_history.append((copy.deepcopy(self.current_hyperparams), task_score))
    
    def should_propose_modification(self, num_tasks_since_last_proposal: int = 10) -> bool:
        """
        Check if it's time to propose a new modification.
        """
        return num_tasks_since_last_proposal >= 10
    
    def get_current_params(self) -> HyperparameterConfig:
        """Return current optimized hyperparameters."""
        return copy.deepcopy(self.current_hyperparams)
    
    def __repr__(self) -> str:
        return (
            f"RSI[iterations={self.current_hyperparams.mcts_iterations}, "
            f"depth={self.current_hyperparams.max_depth}, "
            f"C={self.current_hyperparams.exploration_constant:.2f}, "
            f"commits={len(self.modification_history)}]"
        )
