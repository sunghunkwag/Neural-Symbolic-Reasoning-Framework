"""
Manual Verification of DSL Expressiveness.
Target: Task 00576224 (Tiling 2x2 -> 6x6)
Logic: Construct the output using only shift() and bitwise_or().
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, shift, bitwise_or
from aria.logic.arc_loader import ARCLoader

def verify_tiling_logic():
    print("=== Manual DSL Verification: Tiling ===")
    loader = ARCLoader()
    tasks = loader.load_tasks(limit=1)
    task_id, train_pairs, test_pairs = tasks[0] # 00576224
    
    input_grid, target_grid = train_pairs[0]
    W, H = input_grid.W, input_grid.H # 2, 2
    
    print(f"Task: {task_id}")
    print(f"Input: {W}x{H}")
    print(f"Target: {target_grid.W}x{target_grid.H}")
    
    # Logic: Construct 6x6 from 2x2 using concat
    # Row: [Input, Input, Input]
    # hconcat only takes 2 args (binary)
    # r = hconcat(INPUT, hconcat(INPUT, INPUT))
    
    from aria.logic.dsl import hconcat, vconcat, flip_x
    
    # Analyze Pattern:
    # Row 1-2: Input (Normal) - Rep 3 times
    # Row 3-4: Input (Flipped X) - Rep 3 times
    # Row 5-6: Input (Normal) - Rep 3 times
    
    # 1. Build Row A (Normal)
    # 2 -> 4 -> 6
    row_a = hconcat(input_grid, hconcat(input_grid, input_grid))
    
    # 2. Build Row B (Flipped)
    flipped = flip_x(input_grid)
    row_b = hconcat(flipped, hconcat(flipped, flipped))
    
    # 3. Build Col (A -> B -> A)
    # 2 -> 4 -> 6
    col_step1 = vconcat(row_a, row_b)
    final_grid = vconcat(col_step1, row_a)
    
    print(f"Constructed Shape: {final_grid.W}x{final_grid.H}")
    
    # Verify
    if final_grid.data.shape == target_grid.data.shape and (final_grid.data == target_grid.data).all():
        print("SUCCESS: Manually constructed solution matches Target!")
        return True
    else:
        print("FAILURE: Constructed grid does not match target.")
        # Debug
        print("Target:\n", target_grid.data)
        print("Constructed:\n", final_grid.data)
        return False 

if __name__ == "__main__":
    verify_tiling_logic()
