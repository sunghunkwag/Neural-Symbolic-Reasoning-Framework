"""Episodic Memory System for ARIA v3.0 - AGI Improvements."""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import json


@dataclass
class EpisodeRecord:
    """A single recorded episode of program synthesis."""
    task_id: str
    task_features: Dict[str, Any]  # Input dimensions, output dims, etc.
    program_representation: str    # Serialized program
    score: float                   # Final synthesis score
    num_steps: int                 # MCTS iterations used
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'task_features': self.task_features,
            'program_repr': self.program_representation,
            'score': self.score,
            'steps': self.num_steps,
            'timestamp': self.timestamp
        }


class EpisodicMemory:
    """
    Episodic Memory System for ARIA v3.0.
    
    Stores successful (and failed) program synthesis trajectories.
    Enables:
    1. Transfer learning via similar past solutions
    2. Pattern discovery of effective primitives
    3. Warm-start MCTS with biased initialization
    """
    
    def __init__(self, max_episodes: int = 10000, memory_file: Optional[str] = None):
        self.max_episodes = max_episodes
        self.episodes: List[EpisodeRecord] = []
        self.primitive_frequencies = defaultdict(int)
        self.task_similarity_cache = {}
        self.memory_file = memory_file
        
        # Load from disk if available
        if memory_file:
            self.load_from_disk(memory_file)
    
    def store_episode(
        self,
        task_id: str,
        task_features: Dict[str, Any],
        program_repr: str,
        score: float,
        num_steps: int
    ) -> None:
        """
        Store a successful synthesis episode.
        
        Args:
            task_id: Unique identifier for the task
            task_features: Dict with 'input_dims', 'output_dims', 'num_examples', etc.
            program_repr: String representation of the synthesized program
            score: Final task score (0.0 to 1.0)
            num_steps: Number of MCTS iterations used
        """
        record = EpisodeRecord(
            task_id=task_id,
            task_features=task_features,
            program_representation=program_repr,
            score=score,
            num_steps=num_steps
        )
        
        self.episodes.append(record)
        
        # Extract and count primitives
        self._extract_primitives(program_repr)
        
        # FIFO eviction if over limit
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        
        # Clear similarity cache (becomes stale)
        self.task_similarity_cache.clear()
    
    def _extract_primitives(self, program_repr: str) -> None:
        """Extract and count primitive functions from program representation."""
        # Simple heuristic: split on common delimiters
        tokens = program_repr.replace('(', ' ').replace(')', ' ').split()
        for token in tokens:
            if token and not token.isdigit() and token not in ['INPUT', 'OUTPUT']:
                self.primitive_frequencies[token] += 1
    
    def recall_similar(
        self,
        query_features: Dict[str, Any],
        k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[EpisodeRecord]:
        """
        Retrieve k most similar past episodes using cosine similarity on features.
        
        Args:
            query_features: Features of the new task
            k: Number of results to return
            similarity_threshold: Minimum similarity to include
        
        Returns:
            List of top-k similar episodes, sorted by similarity (descending)
        """
        if not self.episodes:
            return []
        
        # Compute similarity to all past episodes
        similarities = []
        for episode in self.episodes:
            sim = self._compute_feature_similarity(query_features, episode.task_features)
            if sim >= similarity_threshold:
                similarities.append((sim, episode))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in similarities[:k]]
    
    def _compute_feature_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """
        Compute cosine similarity between two feature dictionaries.
        
        Features must be numeric for this simple implementation.
        """
        # Extract numeric values
        keys = set(features1.keys()) & set(features2.keys())
        if not keys:
            return 0.0
        
        v1 = [float(features1[k]) for k in keys if isinstance(features1[k], (int, float))]
        v2 = [float(features2[k]) for k in keys if isinstance(features2[k], (int, float))]
        
        if not v1 or not v2:
            return 0.0
        
        # Cosine similarity
        v1, v2 = np.array(v1), np.array(v2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def get_frequent_primitives(self, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Return the most frequently used primitives in successful episodes.
        
        Can be cached/pre-compiled for faster DSL searches.
        """
        sorted_prims = sorted(
            self.primitive_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_prims[:top_k]
    
    def get_success_rate(self) -> float:
        """
        Compute overall success rate (fraction of episodes with score > 0.5).
        """
        if not self.episodes:
            return 0.0
        successful = sum(1 for ep in self.episodes if ep.score > 0.5)
        return successful / len(self.episodes)
    
    def get_avg_steps_for_score(self, min_score: float = 0.8) -> Optional[float]:
        """
        Get average MCTS steps needed to achieve given score threshold.
        
        Useful for setting iteration budgets.
        """
        successful = [ep.num_steps for ep in self.episodes if ep.score >= min_score]
        return np.mean(successful) if successful else None
    
    def save_to_disk(self, filepath: str) -> None:
        """
        Persist memory to disk as JSON.
        """
        data = {
            'episodes': [ep.to_dict() for ep in self.episodes],
            'primitive_frequencies': dict(self.primitive_frequencies),
            'timestamp': time.time()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_disk(self, filepath: str) -> None:
        """
        Load memory from disk.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for ep_dict in data.get('episodes', []):
                record = EpisodeRecord(
                    task_id=ep_dict['task_id'],
                    task_features=ep_dict['task_features'],
                    program_representation=ep_dict['program_repr'],
                    score=ep_dict['score'],
                    num_steps=ep_dict['steps'],
                )
                self.episodes.append(record)
            
            self.primitive_frequencies = defaultdict(int, data.get('primitive_frequencies', {}))
        except FileNotFoundError:
            pass  # No prior memory
    
    def clear(self) -> None:
        """Clear all stored episodes."""
        self.episodes.clear()
        self.primitive_frequencies.clear()
        self.task_similarity_cache.clear()
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self.episodes)}, success_rate={self.get_success_rate():.2%})"
