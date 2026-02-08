"""Genetic Programming Engine for Logic Synthesis."""
import random
import copy
from typing import List, Callable, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import inspect

from .dsl import Grid, PRIMITIVES, shift, rotate_cw, rotate_ccw, flip_x, flip_y, color_replace, flood_fill, subgrid, overlay

@dataclass
class ProgramNode:
    """AST Node for Genetic Program."""
    func: Callable
    args: List[Any]  # Can be other ProgramNodes or terminal values
    depth: int = 0
    
    def execute(self, input_grid: Grid) -> Grid:
        evaluated_args = []
        for arg in self.args:
            if isinstance(arg, ProgramNode):
                evaluated_args.append(arg.execute(input_grid))
            elif isinstance(arg, tuple) and arg[0] == 'input':
                evaluated_args.append(input_grid)
            else:
                evaluated_args.append(arg)
        
        try:
            return self.func(*evaluated_args)
        except Exception:
            # Return input on failure to be safe
            return input_grid
            
    def __repr__(self):
        args_str = []
        for arg in self.args:
            if isinstance(arg, tuple) and arg[0] == 'input':
                args_str.append("INPUT")
            else:
                args_str.append(str(arg))
        return f"{self.func.__name__}({', '.join(args_str)})"

class GeneticEngine:
    """
    Evolves programs to solve grid logic tasks.
    Uses extensive diversity maintenance and elitism.
    """
    
    def __init__(self, population_size: int = 2000, max_depth: int = 4):
        self.pop_size = population_size
        self.max_depth = max_depth
        self.population: List[ProgramNode] = []
        self.fitness_scores: List[float] = []
        
    def initialize_population(self):
        """Create random initial population."""
        # Seed with very simple 1-depth programs for quick wins
        self.population = []
        
        # 20% Simple 1-depth primitives
        for _ in range(int(self.pop_size * 0.2)):
            self.population.append(self._random_program(depth=0, force_terminal=True))
            
        # 80% Random depth
        for _ in range(self.pop_size - len(self.population)):
            self.population.append(self._random_program(depth=0))
        
    def _random_program(self, depth: int, force_terminal: bool = False) -> ProgramNode:
        """Recursively generate a random program."""
        # Bias towards simple transforms at top level
        if depth == 0 or force_terminal:
            # Higher chance of simple transforms
            candidates = PRIMITIVES 
        else:
            candidates = PRIMITIVES

        func, arg_types = random.choice(candidates)
        
        # If max depth reached, force terminals only (no sub-programs)
        if depth >= self.max_depth or force_terminal:
             args = self._generate_args(func, arg_types, depth, force_terminal=True)
        else:
             args = self._generate_args(func, arg_types, depth, force_terminal=False)
                
        return ProgramNode(func, args, depth)

    def _generate_args(self, func, arg_types, depth, force_terminal):
        args = []
        for arg_type in arg_types:
            if arg_type == Grid:
                if not force_terminal and depth < self.max_depth and random.random() < 0.3:
                    args.append(self._random_program(depth + 1))
                else:
                    args.append(('input',))
            elif arg_type == int:
                # Standard random integer generation (unbiased)
                # Bias towards small numbers [-5, 5] and color codes [0-9]
                if random.random() < 0.5:
                    args.append(random.randint(0, 9)) # Colors
                else:
                    args.append(random.randint(-5, 5)) # Genetic mutation range
            else:
                args.append(0)
        return args

    def evaluate_fitness(self, examples: List[Tuple[Grid, Grid]]):
        """Evaluate population against examples."""
        scores = []
        for program in self.population:
            score = 0.0
            for input_grid, target_grid in examples:
                try:
                    output = program.execute(input_grid)
                    
                    # OBJECT-AWARE FITNESS (F1 Score on non-background)
                    # Background is 0 (Black)
                    
                    # Get indices of non-background pixels
                    out_mask = output.data != 0
                    tgt_mask = target_grid.data != 0
                    
                    # True Positives: Correct color at non-background locations
                    # (Strict match: position AND color)
                    tp = np.logical_and(out_mask, output.data == target_grid.data).sum()
                    
                    # False Positives: Non-background in output that shouldn't be there (or wrong color)
                    # fp_mask = np.logical_and(out_mask, output.data != target_grid.data) # Wrong color or wrong position
                    # Actually, simple count: Total predicted non-bg - TP
                    pred_count = out_mask.sum()
                    fp = pred_count - tp
                    
                    # False Negatives: Target non-bg that were missed
                    tgt_count = tgt_mask.sum()
                    fn = tgt_count - tp
                    
                    # Safe separate calculation
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                        
                    # Identity penalty still applies
                    if (output.data == input_grid.data).all() and not (input_grid.data == target_grid.data).all():
                        f1 *= 0.5
                        
                    score += f1
                except:
                    score += 0.0
            scores.append(score / len(examples))
        self.fitness_scores = scores
        
    def evolve(self, generations: int, examples: List[Tuple[Grid, Grid]]) -> ProgramNode:
        """Run evolution loop."""
        self.initialize_population()
        
        best_program = None
        best_score = -1.0
        
        for gen in range(generations):
            self.evaluate_fitness(examples)
            
            max_score = max(self.fitness_scores)
            if max_score > best_score:
                best_score = max_score
                best_idx = self.fitness_scores.index(max_score)
                best_program = self.population[best_idx]
                
            # Log progress periodically
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Score {best_score:.3f} | {best_program}")

            if best_score >= 1.0:
                break
                
            # Selection & Reproduction
            new_pop = []
            
            # Elitism (Keep top 10% to preserve good solutions)
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            elite_count = int(self.pop_size * 0.1)
            for i in range(elite_count):
                new_pop.append(copy.deepcopy(self.population[sorted_indices[i]]))
                
            # Fill rest
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                if random.random() < 0.7:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                    
                if random.random() < 0.7:  # Increased mutation rate
                    child = self._mutate(child)
                    
                new_pop.append(child)
                
            self.population = new_pop
            
            # Diversity injection: if best score stagnant, replace bottom 50%
            if gen > 10 and self.fitness_scores and max(self.fitness_scores) < 1.0:
                 for k in range(int(self.pop_size * 0.5)):
                     idx = random.randint(int(self.pop_size * 0.1), self.pop_size - 1)
                     self.population[idx] = self._random_program(depth=0)
            
        return best_program
        
    def _tournament_select(self) -> ProgramNode:
        k = 7 # Increased tournament size for stronger selection pressure
        indices = random.sample(range(self.pop_size), k)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]
        
    def _crossover(self, p1: ProgramNode, p2: ProgramNode) -> ProgramNode:
        """Safe Subtree exchange."""
        child = copy.deepcopy(p1)
        if not child.args or not p2.args:
            return child
            
        idx1 = random.randint(0, len(child.args)-1)
        
        # Find compatible arg in p2
        candidates_p2 = []
        for i, arg in enumerate(p2.args):
            # Check rough type compatibility
            type1 = type(child.args[idx1])
            type2 = type(arg)
            if type1 == type2:
                candidates_p2.append(i)
                
        if candidates_p2:
            idx2 = random.choice(candidates_p2)
            child.args[idx1] = copy.deepcopy(p2.args[idx2])
            
        return child
        
    def _mutate(self, p: ProgramNode) -> ProgramNode:
        """Safe Point mutation."""
        p_copy = copy.deepcopy(p)
        if random.random() < 0.4:
            # Change function (only if arg count matches)
            current_arg_count = len(p_copy.args)
            candidates = [prim for prim in PRIMITIVES if len(prim[1]) == current_arg_count]
            if candidates:
                new_func, _ = random.choice(candidates)
                p_copy.func = new_func
        else:
            # Change argument
            if len(p_copy.args) > 0:
                idx = random.randint(0, len(p_copy.args)-1)
                arg = p_copy.args[idx]
                if isinstance(arg, int):
                    # Standard integer mutation
                    if random.random() < 0.5:
                        p_copy.args[idx] += random.choice([-1, 1])
                    else:
                         p_copy.args[idx] = random.randint(-5, 5)
                elif isinstance(arg, ProgramNode):
                    # Replace subtree
                    p_copy.args[idx] = self._random_program(depth=p_copy.depth + 1)
        return p_copy
