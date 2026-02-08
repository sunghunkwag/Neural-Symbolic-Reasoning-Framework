
import numpy as np
import sys
import os
import time

# Ensure aria_core is in path
sys.path.append(os.getcwd())

from aria.types import Grid
from aria.logic.mcts_solver import MCTSSolver
from aria.logic.dsl import TYPED_PRIMITIVES

def solve_atomic():
    print("--- MCTS Verification: Atomic Identity ---")
    
    # Task: 2x2 Identity
    data = np.zeros((2, 2), dtype=int)
    data[0, 0] = 1
    grid = Grid(data)
    examples = [(grid, grid)]
    
    print("Problem: Identity (return INPUT)")
    
    solver = MCTSSolver(TYPED_PRIMITIVES, max_depth=3)
    
    start_time = time.time()
    best_program = solver.solve(examples, iterations=500)
    end_time = time.time()
    
    print(f"Solved in {end_time - start_time:.4f}s")
    print(f"Program: {best_program}")
    
    if best_program:
        ctx = {"INPUT": grid}
        out = best_program.execute(ctx)
        score = 1.0 if out == grid else 0.0
        print(f"Verification Score: {score}")
        if score == 1.0:
            print("SUCCESS: Reasoning Engine Solved Atomic Task.")
        else:
            print("FAILURE: Program incorrect.")
    else:
        print("FAILURE: No program found.")

if __name__ == "__main__":
    solve_atomic()
