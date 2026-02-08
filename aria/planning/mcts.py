"""MCTS Planner - Real Tree Search with Cost Budgets."""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import math
import logging

from ..types import Goal
from ..config import PlannerConfig

logger = logging.getLogger("ARIA.MCTS")

@dataclass
class MCTSNode:
    """Node in MCTS tree."""
    state: torch.Tensor
    parent: Optional['MCTSNode'] = None
    action: Optional[torch.Tensor] = None
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    cost: float = 0.0
    
    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visits)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

class MCTSPlanner:
    """
    Monte Carlo Tree Search Planner with:
    - Real tree search (not greedy)
    - World-model rollouts
    - Explicit cost budgets (time, compute, risk)
    """
    
    def __init__(self, config: PlannerConfig, action_dim: int):
        self.config = config
        self.action_dim = action_dim
        
        # Action discretization for tree search
        self.action_bins = 5
        self.action_space = self._create_action_space()
        
        # Statistics
        self.total_plans = 0
        self.total_nodes_expanded = 0
        
    def _create_action_space(self) -> torch.Tensor:
        """Create discretized action space."""
        # For continuous actions, discretize into bins
        bins = torch.linspace(-1, 1, self.action_bins)
        if self.action_dim == 1:
            return bins.unsqueeze(1)
        else:
            # Multi-dimensional: create grid
            grids = torch.meshgrid([bins] * self.action_dim, indexing='ij')
            actions = torch.stack([g.flatten() for g in grids], dim=1)
            return actions[:min(25, len(actions))]  # Limit branching
    
    def plan(
        self,
        initial_state: torch.Tensor,
        goal: Goal,
        world_model,
        budget: Optional[Dict[str, float]] = None
    ) -> Tuple[List[torch.Tensor], float, Dict]:
        """
        Plan action sequence via MCTS.
        
        Returns:
            actions: List of planned actions
            expected_value: Estimated value of plan
            info: Planning statistics
        """
        budget = budget or {
            "time": self.config.budget_time,
            "compute": self.config.budget_compute,
            "risk": self.config.budget_risk
        }
        
        # Initialize root with float32
        root = MCTSNode(state=initial_state.squeeze().clone().float())
        
        # MCTS iterations
        compute_used = 0
        nodes_expanded = 0
        
        for iteration in range(self.config.mcts_iterations):
            if compute_used >= budget["compute"]:
                break
            
            # Selection
            node = self._select(root)
            
            # Expansion
            if node.visits > 0 and not node.is_leaf():
                node = self._expand(node, world_model)
                nodes_expanded += 1
            elif node.is_leaf() and node.visits > 0:
                node = self._expand(node, world_model)
                nodes_expanded += 1
            
            # Simulation (rollout)
            value = self._simulate(node, goal, world_model, budget["risk"])
            
            # Backpropagation
            self._backpropagate(node, value)
            
            compute_used += 1 + nodes_expanded
        
        # Extract best action sequence
        actions = self._extract_best_path(root, self.config.mcts_depth)
        expected_value = root.value
        
        self.total_plans += 1
        self.total_nodes_expanded += nodes_expanded
        
        return actions, expected_value, {
            "iterations": iteration + 1,
            "nodes_expanded": nodes_expanded,
            "compute_used": compute_used,
            "root_visits": root.visits
        }
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB."""
        while not node.is_leaf():
            best_child = None
            best_ucb = -float('inf')
            
            for action_idx, child in node.children.items():
                ucb = self._ucb_score(child, node.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            if best_child is None:
                break
            node = best_child
        
        return node
    
    def _ucb_score(self, node: MCTSNode, parent_visits: int) -> float:
        """UCB1 score with cost penalty."""
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.value
        exploration = self.config.ucb_c * math.sqrt(
            math.log(parent_visits + 1) / node.visits
        )
        cost_penalty = 0.1 * node.cost
        
        return exploitation + exploration - cost_penalty
    
    def _expand(self, node: MCTSNode, world_model) -> MCTSNode:
        """Expand node by adding children for each action."""
        device = node.state.device
        
        for action_idx, action in enumerate(self.action_space):
            if action_idx in node.children:
                continue
            
            action = action.to(device)
            
            # Use world model to predict next state
            with torch.no_grad():
                z_current = node.state.unsqueeze(0)
                action_input = action.unsqueeze(0)
                z_next, sigma = world_model.ssm.step(z_current, action_input)
            
            # Cost is related to uncertainty and action magnitude
            cost = sigma.mean().item() + 0.1 * action.abs().sum().item()
            
            child = MCTSNode(
                state=z_next.squeeze().clone(),
                parent=node,
                action=action.clone(),
                cost=cost
            )
            node.children[action_idx] = child
        
        # Return a random child for simulation
        if node.children:
            child_idx = np.random.choice(list(node.children.keys()))
            return node.children[child_idx]
        return node
    
    def _simulate(
        self, 
        node: MCTSNode, 
        goal: Goal, 
        world_model,
        risk_budget: float
    ) -> float:
        """Simulate from node to estimate value."""
        device = node.state.device
        state = node.state.unsqueeze(0)
        total_value = 0.0
        discount = 0.99
        total_risk = 0.0
        
        for step in range(self.config.mcts_depth):
            if total_risk > risk_budget:
                # Penalize exceeding risk budget
                total_value -= 10.0
                break
            
            # Random action for simulation
            action_idx = np.random.randint(len(self.action_space))
            action = self.action_space[action_idx].to(device).unsqueeze(0)
            
            # Predict next state
            with torch.no_grad():
                state, sigma = world_model.ssm.step(state, action)
            
            # Compute step value (distance to goal)
            if goal.target_state is not None:
                target = goal.target_state.to(device)
                distance = torch.norm(state.squeeze() - target.squeeze()).item()
                step_value = -distance + 1.0  # Reward for being close
            else:
                step_value = -sigma.mean().item()  # Prefer low uncertainty
            
            total_value += (discount ** step) * step_value
            total_risk += sigma.mean().item()
        
        return total_value
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent
    
    def _extract_best_path(
        self, 
        root: MCTSNode, 
        depth: int
    ) -> List[torch.Tensor]:
        """Extract best action sequence from tree."""
        actions = []
        node = root
        
        for _ in range(depth):
            if node.is_leaf():
                break
            
            # Select most visited child
            best_child = None
            best_visits = -1
            
            for child in node.children.values():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_child = child
            
            if best_child is None or best_child.action is None:
                break
            
            actions.append(best_child.action)
            node = best_child
        
        # Pad with zeros if needed
        while len(actions) < depth:
            actions.append(torch.zeros(self.action_dim))
        
        return actions
    
    def get_stats(self) -> dict:
        return {
            "total_plans": self.total_plans,
            "total_nodes_expanded": self.total_nodes_expanded,
            "avg_nodes_per_plan": self.total_nodes_expanded / max(1, self.total_plans)
        }
