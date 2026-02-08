
import numpy as np
import sys
import os
import time

# Ensure aria_core is in path
sys.path.append(os.getcwd())

from aria.types import Grid
from aria.logic.mcts_solver import MCTSSolver
from aria.logic.dsl import TYPED_PRIMITIVES

def solve_complex():
    print("--- MCTS Verification: Shift and Recolor ---")
    
    # Task: Shift (1, 1) and Recolor 1->2
    # Input: 5x5 grid with Blue(1) at (1,1)
    # Output: 5x5 grid with Red(2) at (2,2)
    data = np.zeros((5, 5), dtype=int)
    data[1, 1] = 1 
    input_grid = Grid(data)
    
    out_data = np.zeros((5, 5), dtype=int)
    out_data[2, 2] = 2
    output_grid = Grid(out_data)
    
    examples = [(input_grid, output_grid)]
    
    print("Problem: Shift(1, 1) then ColorReplace(1->2)")
    
    solver = MCTSSolver(TYPED_PRIMITIVES, max_depth=3) # Depth 3 should be enough
    
    start_time = time.time()
    # Chain of Research (Sequential)
    best_program_chain = solver.solve_sequential(examples, iterations=50000, max_steps=5) 
    end_time = time.time()
    
    print(f"Solved in {end_time - start_time:.4f}s")
    print(f"Program Chain: {best_program_chain}")
    
    # Execute Chain
    ctx = {"INPUT": input_grid}
    curr = input_grid
    for prog in best_program_chain:
        try:
             curr = prog.execute({"INPUT": curr})
        except:
             pass
             
    score = 1.0 if curr == output_grid else 0.0
    print(f"Verification Score: {score}")
    if score == 1.0:
        print("SUCCESS: Reasoning Engine Solved Complex Task via Chain-of-Thought.")
    else:
        print("FAILURE: Program chain incorrect.")

if __name__ == "__main__":
    solve_complex()
