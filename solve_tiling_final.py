"""
Phase 16: Final Evolutionary Scale-Up.
Target: Task 00576224 (Tiling)
Goal: 100% Success using Pure Evolution (No Manual Hints).
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, TYPED_PRIMITIVES
from aria.logic.genetic_typed import TypedGeneticEngine
from aria.logic.arc_loader import ARCLoader

def solve_tiling_final():
    print("=== Phase 16: Evolutionary Scale-Up (Tiling) ===")
    
    # 1. Load Data
    loader = ARCLoader()
    tasks = loader.load_tasks(limit=10)
    task_id, train_pairs, test_pairs = tasks[0] # 00576224
    print(f"Task: {task_id}")
    print(f"Train Pairs: {len(train_pairs)}")
    
    # 2. Configure Engine (Massive Scale)
    # Pop 1000, Gen 100 -> 100,000 evaluations
    # Max Depth 6 (Required for concat nesting)
    engine = TypedGeneticEngine(TYPED_PRIMITIVES, max_depth=6)
    
    print("\n[Configuration]")
    print(f"Primitives: {len(TYPED_PRIMITIVES)}")
    print("Population: 1000")
    print("Generations: 100")
    print("Max Depth: 6")
    
    start_time = time.time()
    
    # 3. Running Evolution
    print("\n[Evolution Started]")
    # We rely on the engine's internal elitism and logging
    best_program = engine.evolve(
        train_pairs, 
        generations=100, 
        pop_size=1000,
        context_types={"INPUT": Grid}
    )
    
    duration = time.time() - start_time
    
    # 4. Final Evaluation
    print("\n[Evolution Finished]")
    print(f"Time Taken: {duration:.2f}s")
    
    if best_program:
        train_score = engine._evaluate(best_program, train_pairs)
        test_score = engine._evaluate(best_program, test_pairs)
        
        print(f"\n[Results]")
        print(f"Best Program: {best_program}")
        print(f"Train Accuracy: {train_score*100:.1f}%")
        print(f"Test Accuracy:  {test_score*100:.1f}%")
        
        if train_score >= 1.0 and test_score >= 1.0:
            print("\nSUCCESS: 100% Solved without Cheats!")
        else:
            print("\nFAILURE: Did not converge to 100%.")
    else:
        print("\nFAILURE: No valid program found.")

if __name__ == "__main__":
    solve_tiling_final()
