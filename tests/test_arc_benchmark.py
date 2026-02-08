"""
ARC-AGI 2 Benchmark Test Harness.
Tests the Logic Synthesis Engine on official ARC-AGI-2 tasks.
"""
import sys
import os
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ARC_BENCHMARK")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.arc_loader import (
    load_arc_task, load_training_examples, load_test_examples,
    list_tasks, evaluate_program, ARCLoader
)
from aria.logic.genetic_typed import TypedGeneticEngine
from aria.logic.dsl import TYPED_PRIMITIVES

def run_benchmark(data_dir: str, max_tasks: int = 10, generations: int = 50):
    """
    Run the genetic engine on ARC-AGI-2 tasks.
    
    Args:
    """
    print("=== ARC-AGI-2 Benchmark Test (High-Fidelity) ===")
    
    # 1. Load Tasks
    loader = ARCLoader()
    tasks = loader.load_tasks(limit=10)
    print(f"Tasks to test: {len(tasks)}")
    
    import gc
    
    # 2. Configure Engine (Optimized for 8GB RAM)
    # Pop 200 (Safe) vs 500 (Heavy)
    # Generations 100 (Deep search retained)
    POP_SIZE = 200
    GENERATIONS = 100
    print(f"Generations per task: {GENERATIONS}")
    print(f"Population size: {POP_SIZE} (Reduced for Memory Safety)")
    
    engine = TypedGeneticEngine(TYPED_PRIMITIVES, max_depth=6)
    
    logger.info("")
    
    results = []
    
    for i, task in enumerate(tasks):
        # Explicit GC to free memory between tasks
        gc.collect()
        
        task_id = task[0]
        # task structure from loader is (id, train, test)
        train_examples = task[1]
        test_examples = task[2]
        
        logger.info(f"[{i+1}/{len(tasks)}] Task: {task_id}")
        
        try:
            # Log task info
            if train_examples:
                inp_shape = train_examples[0][0].data.shape
                out_shape = train_examples[0][1].data.shape
                logger.info(f"  Train pairs: {len(train_examples)}, Test pairs: {len(test_examples)}")
                logger.info(f"  Input shape: {inp_shape}, Output shape: {out_shape}")
            
            # Run evolution
            start_time = time.time()
            # Correctly pass pop_size to evolve()
            best_program = engine.evolve(
                examples=train_examples,
                generations=GENERATIONS, 
                pop_size=POP_SIZE
            )
            elapsed = time.time() - start_time
            
            # Evaluate on train and test
            train_acc = evaluate_program(best_program, train_examples)
            test_acc = evaluate_program(best_program, test_examples)
            
            logger.info(f"  Best: {best_program}")
            logger.info(f"  Train Acc: {train_acc*100:.1f}%, Test Acc: {test_acc*100:.1f}%")
            logger.info(f"  Time: {elapsed:.1f}s")
            logger.info("")
            
            results.append({
                'task_id': task_id,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'program': str(best_program),
                'time': elapsed
            })
            
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results.append({
                'task_id': task_id,
                'train_acc': 0.0,
                'test_acc': 0.0,
                'program': None,
                'time': 0.0,
                'error': str(e)
            })
            logger.info("")
    
    # Summary
    logger.info("=== Summary ===")
    train_solved = sum(1 for r in results if r['train_acc'] == 1.0)
    test_solved = sum(1 for r in results if r['test_acc'] == 1.0)
    avg_train = sum(r['train_acc'] for r in results) / len(results) if results else 0
    avg_test = sum(r['test_acc'] for r in results) / len(results) if results else 0
    
    logger.info(f"Tasks tested: {len(results)}")
    logger.info(f"Train Solved (100%): {train_solved}/{len(results)}")
    logger.info(f"Test Solved (100%): {test_solved}/{len(results)}")
    logger.info(f"Avg Train Acc: {avg_train*100:.1f}%")
    logger.info(f"Avg Test Acc: {avg_test*100:.1f}%")
    
    return results

if __name__ == "__main__":
    # Path to ARC-AGI-2 data directory
    data_dir = Path(__file__).parent / "ARC-AGI-2" / "data"
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Run on first 10 tasks (Safety Mode)
    # Note: 400 tasks would take hours. 10 is a good representative set.
    results = run_benchmark(str(data_dir), max_tasks=10, generations=100)
