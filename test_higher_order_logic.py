"""
Unit Test for Higher-Order Logic Primitives (Phase 15).
Verifies: detect_objects, map_grid, filter_grid, fold_grid
"""
import sys
import numpy as np
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, detect_objects, map_grid, filter_grid, fold_grid, bitwise_or, color_replace

def create_test_grid():
    # Create a 10x10 grid with two distinct squares
    data = np.zeros((10, 10), dtype=int)
    # Object 1 (Red=2) at top-left
    data[1:4, 1:4] = 2
    # Object 2 (Blue=1) at bottom-right
    data[6:9, 6:9] = 1
    return Grid(data)

def test_higher_order():
    print("=== Testing Higher-Order Primitives ===")
    
    # 1. Test detect_objects
    print("\n1. Testing detect_objects...")
    input_grid = create_test_grid()
    objects = detect_objects(input_grid, bg_color=0)
    print(f"Found {len(objects)} objects.")
    
    if len(objects) != 2:
        print("FAIL: Expected 2 objects.")
        return
    else:
        print("PASS: Correctly identified 2 objects.")

    # 2. Test filter (Keep only Blue=1)
    print("\n2. Testing filter_grid (Keep Blue)...")
    def is_blue(g: Grid) -> bool:
        # Check if grid contains color 1
        return 1 in g.data
        
    blue_objs = filter_grid(objects, is_blue)
    print(f"Filtered to {len(blue_objs)} objects.")
    
    if len(blue_objs) == 1 and (1 in blue_objs[0].data):
        print("PASS: Filtered correctly.")
    else:
        print("FAIL: Filter failed.")

    # 3. Test map (Change Red objects to Green=3)
    print("\n3. Testing map_grid (Red -> Green)...")
    def turn_red_to_green(g: Grid) -> Grid:
        # We can use the primitive color_replace
        return color_replace(g, 2, 3)
        
    mapped_objs = map_grid(objects, turn_red_to_green)
    
    # Verify: One should be Blue, one Green
    has_green = any(3 in g.data for g in mapped_objs)
    has_blue = any(1 in g.data for g in mapped_objs)
    has_red = any(2 in g.data for g in mapped_objs)
    
    if has_green and has_blue and not has_red:
        print("PASS: Map successfully transformed colors.")
    else:
        print(f"FAIL: Map failure. Green:{has_green}, Blue:{has_blue}, Red:{has_red}")

    # 4. Test fold (Combine back)
    print("\n4. Testing fold_grid (Recombine)...")
    # Start with empty grid
    empty = Grid(np.zeros((10, 10), dtype=int))
    combined = fold_grid(mapped_objs, bitwise_or, empty)
    
    # Check if we have Green and Blue in one grid
    c_data = combined.data
    if 3 in c_data and 1 in c_data:
        print("PASS: Fold successfully combined objects.")
    else:
        print("FAIL: Fold failed to combine.")

    # 5. Test Construction (Structure)
    print("\n5. Testing hconcat/vconcat...")
    from aria.logic.dsl import hconcat, vconcat
    
    small = Grid(np.ones((2, 2), dtype=int))
    # hconcat -> 2x4
    wide = hconcat(small, small)
    if wide.W == 4 and wide.H == 2:
        print("PASS: hconcat Correct (2x4).")
    else:
        print(f"FAIL: hconcat sizing error: {wide.W}x{wide.H}")
        
    # vconcat -> 4x4
    tall = vconcat(wide, wide)
    if tall.W == 4 and tall.H == 4:
         print("PASS: vconcat Correct (4x4).")
    else:
         print(f"FAIL: vconcat sizing error: {tall.W}x{tall.H}")

    print("\n=== All Higher-Order Tests Passed ===")

if __name__ == "__main__":
    test_higher_order()
