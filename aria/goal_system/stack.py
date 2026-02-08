"""Goal Stack - Priority Queue for Active Goals."""
import heapq
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from ..types import Goal, GoalStatus

logger = logging.getLogger("ARIA.GoalStack")

@dataclass(order=True)
class PrioritizedGoal:
    """Wrapper for heap operations."""
    priority: float
    goal: Goal = field(compare=False)

class GoalStack:
    """
    Priority queue for managing active goals.
    Invariant: All goals have L(g) >= 0.6 and CC = 1.
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._heap: List[PrioritizedGoal] = []
        self._goal_map = {}  # id -> goal for fast lookup
        
        # Statistics
        self.total_pushed = 0
        self.total_completed = 0
        self.total_failed = 0
        
    def push(self, goal: Goal) -> bool:
        """
        Push goal onto stack.
        Returns True if accepted, False if rejected (already exists or full).
        """
        if goal.id in self._goal_map:
            return False
        
        if len(self._heap) >= self.max_size:
            # Remove lowest priority
            if self._heap and -self._heap[0].priority < goal.priority:
                self.pop()
            else:
                return False
        
        goal.status = GoalStatus.PENDING
        # Use negative priority for max-heap behavior
        entry = PrioritizedGoal(priority=-goal.priority, goal=goal)
        heapq.heappush(self._heap, entry)
        self._goal_map[goal.id] = goal
        
        self.total_pushed += 1
        logger.info(f"[GOAL-STACK] Pushed: {goal.description} (priority={goal.priority:.2f})")
        return True
    
    def pop(self) -> Optional[Goal]:
        """Pop highest priority goal."""
        while self._heap:
            entry = heapq.heappop(self._heap)
            goal = entry.goal
            
            if goal.id in self._goal_map:
                del self._goal_map[goal.id]
                goal.status = GoalStatus.ACTIVE
                return goal
        
        return None
    
    def peek(self) -> Optional[Goal]:
        """Peek at highest priority goal without removing."""
        for entry in sorted(self._heap):
            if entry.goal.id in self._goal_map:
                return entry.goal
        return None
    
    def terminate(self, goal_id: str, reason: str = "completed") -> bool:
        """
        Terminate a goal (achieved or failed).
        """
        if goal_id not in self._goal_map:
            return False
        
        goal = self._goal_map[goal_id]
        
        if reason == "completed" or reason == "achieved":
            goal.status = GoalStatus.ACHIEVED
            self.total_completed += 1
        else:
            goal.status = GoalStatus.FAILED
            self.total_failed += 1
        
        del self._goal_map[goal_id]
        logger.info(f"[GOAL-STACK] Terminated: {goal.description} ({reason})")
        return True
    
    def reject(self, goal_id: str) -> bool:
        """Remove goal without achieving."""
        if goal_id not in self._goal_map:
            return False
        
        goal = self._goal_map[goal_id]
        goal.status = GoalStatus.REJECTED
        del self._goal_map[goal_id]
        return True
    
    def get_active(self) -> List[Goal]:
        """Get all active goals."""
        return [g for g in self._goal_map.values() if g.status == GoalStatus.ACTIVE]
    
    def __len__(self) -> int:
        return len(self._goal_map)
    
    def __contains__(self, goal_id: str) -> bool:
        return goal_id in self._goal_map
    
    def get_stats(self) -> dict:
        return {
            "size": len(self),
            "total_pushed": self.total_pushed,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed
        }
