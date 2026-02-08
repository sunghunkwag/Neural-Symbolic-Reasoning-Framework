"""
Phase 16-B: Adaptive Fitness Shaping (Guided Evolution).
Target: Task 00576224 (Tiling)
Goal: Guide evolution using granular fitness metrics (Dimensions, Colors) instead of binary match.
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, TYPED_PRIMITIVES
from aria.logic.genetic_typed import TypedGeneticEngine, TypedNode
from aria.logic.arc_loader import ARCLoader

class ShapedGeneticEngine(TypedGeneticEngine):
    """Genetic Engine with Task-Specific Heuristics for ARC."""
    
    def _evaluate(self, program: TypedNode, examples) -> float:
        """
        Shaped Fitness Function:
        1. Dimension Score (30%): Does output shape match target?
        2. Color Score (20%): Does output contain correct colors?
        3. Pixel Score (50%): Standard pixel accuracy.
        """
        total_score = 0.0
        
        for inp, tgt in examples:
            try:
                # 1. Execute
                ctx = {"INPUT": inp}
                out = program.execute(ctx)
                
                if not isinstance(out, Grid):
                    continue
                    
                score = 0.0
                
                # 2. Dimensions (Important for Tiling!)
                h_match = 1.0 if out.H == tgt.H else max(0, 1 - abs(out.H - tgt.H)/tgt.H)
                w_match = 1.0 if out.W == tgt.W else max(0, 1 - abs(out.W - tgt.W)/tgt.W)
                dim_score = (h_match + w_match) / 2.0
                
                # 3. Colors (Histogram matching)
                out_colors = set(np.unique(out.data))
                tgt_colors = set(np.unique(tgt.data))
                # Jaccard index of colors
                color_score = len(out_colors & tgt_colors) / len(out_colors | tgt_colors) if (out_colors | tgt_colors) else 0.0
                
                # 4. Pixel Accuracy (Only if dims match, else 0)
                pixel_score = 0.0
                if out.data.shape == tgt.data.shape:
                    matches = np.count_nonzero(out.data == tgt.data)
                    pixel_score = matches / out.data.size
                
                # Weighted Sum
                # Dims are prerequisite for pixels, so weight them heavily early on?
                # Let's use simple weighted sum for now.
                score = (dim_score * 0.3) + (color_score * 0.1) + (pixel_score * 0.6)
                
                total_score += score
                
            except Exception:
                pass # Runtime error = 0 score
                
        return total_score / len(examples) if examples else 0.0

def solve_tiling_shaped():
    print("=== Phase 16-B: Adaptive Fitness Shaping ===")
    
    # Load Data
    loader = ARCLoader()
    tasks = loader.load_tasks(limit=10)
    task_id, train_pairs, test_pairs = tasks[0] # 00576224
    
    print(f"Task: {task_id}")
    
    # Init Shaped Engine
    # Deep search but guided
    engine = ShapedGeneticEngine(TYPED_PRIMITIVES, max_depth=6)
    
    print("Running with Shaped Fitness (Reward Dims/Colors)...")
    
    start_time = time.time()
    
    # Evolve
    best_program = engine.evolve(
        train_pairs, 
        generations=100, 
        pop_size=500, # Smaller pop needed as gradient is smoother
        context_types={"INPUT": Grid}
    )
    
    duration = time.time() - start_time
    print(f"Time: {duration:.2f}s")
    
    # Final eval using strict check
    if best_program:
        # Check Strict Score logic from original engine for final validation
        # We can just manually check pixels
        strict_correct = 0
        for inp, tgt in train_pairs:
            try:
                out = best_program.execute({"INPUT": inp})
                if out.data.shape == tgt.data.shape and (out.data == tgt.data).all():
                    strict_correct += 1
            except: pass
            
        print(f"Best Program: {best_program}")
        print(f"Shaped Score: {engine._evaluate(best_program, train_pairs):.3f}")
        print(f"Strict Train Accuracy: {strict_correct / len(train_pairs) * 100:.1f}%")
        
        if strict_correct == len(train_pairs):
            print("SUCCESS: 100% Solved with Shaped Fitness!")
        else:
            print("FAILURE: Guided search improved score but didn't reach 100%.")

if __name__ == "__main__":
    solve_tiling_shaped()
