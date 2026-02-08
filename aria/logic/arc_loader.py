"""
ARC-AGI 2 Data Loader.
Loads tasks from the official ARC-AGI-2 dataset.
"""
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from .dsl import Grid

def load_arc_task(filepath: str) -> Dict[str, Any]:
    """Load a single ARC task from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def arc_pair_to_grid(pair: Dict[str, Any]) -> Tuple[Grid, Grid]:
    """Convert ARC pair to our Grid objects."""
    input_arr = np.array(pair['input'], dtype=np.int32)
    output_arr = np.array(pair['output'], dtype=np.int32)
    return Grid(input_arr), Grid(output_arr)

def load_training_examples(task_data: Dict[str, Any]) -> List[Tuple[Grid, Grid]]:
    """Extract training examples from task."""
    examples = []
    for pair in task_data['train']:
        examples.append(arc_pair_to_grid(pair))
    return examples

def load_test_examples(task_data: Dict[str, Any]) -> List[Tuple[Grid, Grid]]:
    """Extract test examples from task."""
    examples = []
    for pair in task_data['test']:
        examples.append(arc_pair_to_grid(pair))
    return examples

def list_tasks(data_dir: str, split: str = 'training') -> List[str]:
    """List all task files in a split directory."""
    split_dir = Path(data_dir) / split
    return sorted([str(f) for f in split_dir.glob('*.json')])

def evaluate_program(program, examples: List[Tuple[Grid, Grid]]) -> float:
    """Evaluate a program on examples, return accuracy (0-1)."""
    if program is None:
        return 0.0
    
    correct = 0
    for inp, target in examples:
        try:
            output = program.execute(inp)
            if output.data.shape == target.data.shape and (output.data == target.data).all():
                correct += 1
        except:
            pass
    return correct / len(examples) if examples else 0.0

class ARCLoader:
    """Class wrapper for ARC loading utilities."""
    def __init__(self, data_dir: str = "ARC-AGI-2/data"):
        self.data_dir = data_dir
        
    def load_tasks(self, limit: int = 10) -> List[Tuple[str, List[Tuple[Grid, Grid]], List[Tuple[Grid, Grid]]]]:
        """Load tasks and return list of (id, train_pairs, test_pairs)."""
        task_files = list_tasks(self.data_dir)
        tasks = []
        for fpath in task_files[:limit]:
            try:
                data = load_arc_task(fpath)
                train = load_training_examples(data)
                test = load_test_examples(data)
                task_id = Path(fpath).stem
                tasks.append((task_id, train, test))
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        return tasks
