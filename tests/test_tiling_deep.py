"""
Deep Verification for Task 00576224 (Tiling).
Running TypedGeneticEngine with extended generations to find a solution using only basic Logic + Loop.
"""
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, TYPED_PRIMITIVES
from aria.logic.genetic_typed import TypedGeneticEngine
from aria.logic.arc_loader import ARCLoader

def test_tiling_deep():
    print("=== Deep Verification: Task 00576224 (Tiling) ===")
    
    # Load specific task
    loader = ARCLoader()
    tasks = loader.load_tasks(limit=10) # Load first 10 to find 00576224
    
    target_task = None
    target_id = "00576224"
    
    # Manually search or just load if we knew the path. 
    # For now, searching loaded tasks.
    # Note: 00576224 is usually the first one in sorted training set.
    
    # Let's just run the first one, assuming it's the tiling task as seen in previous logs.
    task_id, train_pairs, test_pairs = tasks[0]
    print(f"Target Task ID: {task_id}")
    
    if task_id != target_id:
        print(f"Warning: First task is {task_id}, not {target_id}. Proceeding anyway.")

    # Init Engine with TYPED primitives (Phase 14 Logic)
    # Increase depth to allow complex nested logic (Loop inside Shift?)
    engine = TypedGeneticEngine(TYPED_PRIMITIVES, max_depth=6) 
    
    print(f"Primitives: {len(TYPED_PRIMITIVES)} available.")
    print("Running 200 Generations...")
    
    # Context types
    context_types = {"INPUT": Grid}
    
    # Evolve
    best_program = engine.evolve(train_pairs, generations=200, pop_size=500, context_types=context_types)
    
    # Evaluate
    train_score = engine._evaluate(best_program, train_pairs)
    test_score = engine._evaluate(best_program, test_pairs)
    
    print(f"\nFinal Result:")
    print(f"Best Program: {best_program}")
    print(f"Train Score: {train_score:.3f}")
    print(f"Test Score:  {test_score:.3f}")

if __name__ == "__main__":
    test_tiling_deep()
