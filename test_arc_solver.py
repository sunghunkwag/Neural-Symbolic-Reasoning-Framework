"""Verification Script for Logic Synthesis Engine."""
import numpy as np
import logging
from aria.logic.dsl import Grid, COLOR_BLUE, COLOR_BLACK
from aria.logic.genetic import GeneticEngine

def make_task_example(w: int, h: int) -> tuple[Grid, Grid]:
    """Create input/output pair: Shift blue pixels right by 1."""
    # Input: Randomly place blue pixels
    data = np.zeros((h, w), dtype=np.int8)
    for _ in range(3):
        x, y = np.random.randint(0, w-1), np.random.randint(0, h)
        data[y, x] = COLOR_BLUE
        
    input_grid = Grid(data)
    
    # Target: Shift right
    target_data = np.zeros((h, w), dtype=np.int8)
    # Manual shift implementation for ground truth
    for y in range(h):
        for x in range(w):
            if data[y, x] == COLOR_BLUE:
                if x + 1 < w:
                    target_data[y, x+1] = COLOR_BLUE
                    
    target_grid = Grid(target_data)
    return input_grid, target_grid

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("ARC-Test")
    
    logger.info("Initializing Logic Synthesis Engine...")
    engine = GeneticEngine(population_size=500, max_depth=4)
    
    # Create 5 examples
    logger.info("Generating training examples (Task: Shift Blue Right)...")
    examples = [make_task_example(6, 6) for _ in range(5)]
    
    logger.info("Starting Evolution...")
    best_program = engine.evolve(generations=100, examples=examples)
    
    logger.info("-" * 40)
    logger.info(f"Best Program Found: {best_program}")
    logger.info("-" * 40)
    
    # Validation
    score = 0
    test_examples = [make_task_example(6, 6) for _ in range(5)]
    for inp, tgt in test_examples:
        out = best_program.execute(inp)
        if out == tgt:
            score += 1
            
    logger.info(f"Test Accuracy: {score}/{len(test_examples)}")
    
    if score == len(test_examples):
        logger.info("SUCCESS: Solved the task!")
    else:
        logger.info("FAILURE: Could not generalize.")

if __name__ == "__main__":
    main()
