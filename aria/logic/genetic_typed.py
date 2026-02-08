import random
import inspect
import copy
import numpy as np
from typing import List, Dict, Any, Type, Tuple, Callable
from dataclasses import dataclass
from aria.types import Grid
from .dsl import TYPED_PRIMITIVES
from .import genetic_strategy

# --- Type System ---
# We use Python's built-in types plus our DSL types.
# Core Types: Grid, int, bool, List[Grid], lambda?

@dataclass
class TypedNode:
    """Base class for Typed GP Nodes."""
    return_type: Type
    
    def execute(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

@dataclass
class PrimitiveNode(TypedNode):
    """A function call node."""
    func: Callable
    args: List[TypedNode]
    name: str

    def execute(self, context: Dict[str, Any]) -> Any:
        # Evaluate args
        evaluated_args = [arg.execute(context) for arg in self.args]
        return self.func(*evaluated_args)

    def __repr__(self):
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.name}({args_str})"

@dataclass
class VariableNode(TypedNode):
    """A variable access node (e.g., 'INPUT')."""
    name: str
    
    def execute(self, context: Dict[str, Any]) -> Any:
        return context[self.name]
    
    def __repr__(self):
        return self.name

@dataclass
class ConstantNode(TypedNode):
    """A literal value (e.g., 1, 0, Color)."""
    value: Any
    
    def execute(self, context: Dict[str, Any]) -> Any:
        return self.value
    
    def __repr__(self):
        return str(self.value)

@dataclass
class LambdaNode(TypedNode):
    """A lambda function definition."""
    arg_names: List[str]
    arg_types: List[Type]
    body: TypedNode
    
    def execute(self, context: Dict[str, Any]) -> Any:
        # Return a python callable that executes the body with extended context
        def lambda_func(*args):
            local_ctx = context.copy()
            for name, val in zip(self.arg_names, args):
                local_ctx[name] = val
            return self.body.execute(local_ctx)
        return lambda_func
        
    def __repr__(self):
        args_str = ", ".join(self.arg_names)
        return f"(lambda {args_str}: {self.body})"

class TypedGeneticEngine:
    """
    Genetic Programming Engine with Type Constraints.
    Can generate programs like: map(detect_objects(INPUT), lambda x: color(x, RED))
    """
    def __init__(self, primitives: List[Tuple], max_depth: int = 5):
        self.primitives = primitives
        self.max_depth = max_depth
        
        # Meta-Parameters (Delegated to genetic_strategy)
        self.terminal_prob = genetic_strategy.get_terminal_prob()
        self.subtree_mutation_max_depth = 2 # Keep static for now or parameterize
        
        # Index primitives by return type for fast lookup
        self.primitives_by_type = {}
        for func, arg_types, ret_type in primitives:
            if ret_type not in self.primitives_by_type:
                self.primitives_by_type[ret_type] = []
            self.primitives_by_type[ret_type].append((func, arg_types))

    def generate_program(self, target_type: Type, context_types: Dict[str, Type], depth: int = 0) -> TypedNode:
        """
        Generate a random program tree that returns `target_type`.
        """
        # Handle Callable types (Lambdas)
        # Robust check for typing.Callable
        is_callable = False
        try:
            origin = getattr(target_type, "__origin__", None)
            if origin is Callable:
                is_callable = True
            elif str(target_type).startswith("typing.Callable"):
                is_callable = True
            # Also check collections.abc.Callable for newer Python
            elif hasattr(origin, "__name__") and origin.__name__ == "Callable":
                is_callable = True
        except Exception:
             pass
            
        if is_callable:
            # Extract arg types and return type
            # Callable[[Arg1, Arg2], Ret]
            try:
                args = target_type.__args__
                # args[0] should be a list of types for input args
                # args[1] is return type
                param_types = args[0]
                ret_type = args[1]
                
                # Verify param_types is a list (or iterable)
                if not isinstance(param_types, (list, tuple)):
                     # If it's a single type, wrap it
                     param_types = [param_types]
                     
            except Exception as e:
                # Fallback for weird typing objects
                # print(f"DEBUG: Failed to parse Callable type: {target_type} - {e}")
                raise ValueError(f"Could not parse Callable type: {target_type}")
            
            # Create unique variable names for lambda args
            arg_names = [f"var_{depth}_{i}" for i in range(len(param_types))]
            
            # Extend context with lambda args
            new_context = context_types.copy()
            for name, type_ in zip(arg_names, param_types):
                new_context[name] = type_
                
            # Generate body
            if depth >= self.max_depth:
                # If we're at max depth and still need to generate a lambda body,
                # we MUST find a terminal or a very shallow function.
                terminals = self._get_valid_terminals(ret_type, new_context)
                if terminals:
                    body = random.choice(terminals)
                else:
                    # Last resort: try to find a function with 0 args or raise
                    functions = self._get_valid_functions(ret_type)
                    if functions:
                        # Try to pick a function with few arguments to limit further recursion
                        min_args_func = min(functions, key=lambda f: len(f[1]))
                        func, arg_types = min_args_func
                        if not arg_types:
                             body = PrimitiveNode(ret_type, func, [], func.__name__)
                        else:
                             # If we're here, we slightly overshoot max_depth but it's finite
                             body = self.generate_program(ret_type, new_context, depth + 1)
                    else:
                        raise ValueError(f"Max depth reached with no terminals for {ret_type}")
            else:
                body = self.generate_program(ret_type, new_context, depth + 1)
            
            return LambdaNode(target_type, arg_names, param_types, body)

        # Base case: max depth or coin flip -> prefer Terminals (Vars/Constants)
        # We allow terminals at any depth now to support identity or early pruning.
        if depth >= self.max_depth or random.random() < self.terminal_prob:
            terminals = self._get_valid_terminals(target_type, context_types)
            if terminals:
                return random.choice(terminals)
        
        # Recursive case: Function Call
        functions = self._get_valid_functions(target_type)
        if not functions:
            # Fallback to terminal
            terminals = self._get_valid_terminals(target_type, context_types)
            if terminals:
                return random.choice(terminals)
            raise ValueError(f"Cannot generate node of type {target_type} at depth {depth}")

        func, arg_types = random.choice(functions)
        
        # Generate arguments
        children = []
        for arg_type in arg_types:
            children.append(self.generate_program(arg_type, context_types, depth + 1))
            
        return PrimitiveNode(target_type, func, children, func.__name__)

    def _get_valid_terminals(self, target_type: Type, context_types: Dict[str, Type]) -> List[TypedNode]:
        terminals = []
        
        # 1. Variables in context
        for name, type_ in context_types.items():
            if type_ == target_type:
                terminals.append(VariableNode(target_type, name))
                
        # 2. Constants (Integers/Booleans)
        if target_type == int:
            # Generate random small integers
            terminals.append(ConstantNode(int, random.randint(-5, 5)))
        elif target_type == bool:
            terminals.append(ConstantNode(bool, True))
            terminals.append(ConstantNode(bool, False))
            
        return terminals

    def _get_valid_functions(self, target_type: Type) -> List[Tuple]:
        # Handle List[Grid] vs Grid specificity if needed
        # For now, exact match
        return self.primitives_by_type.get(target_type, [])

    def crossover(self, parent1: TypedNode, parent2: TypedNode) -> TypedNode:
        """Type-safe crossover."""
        # 1. Select a random node in parent1
        nodes1 = self._get_all_nodes(parent1)
        if not nodes1: return copy.deepcopy(parent1)
        target1 = random.choice(nodes1)
        
        # 2. Find compatible nodes in parent2
        nodes2 = self._get_all_nodes(parent2)
        compatible = [n for n in nodes2 if n.return_type == target1.return_type]
        
        if not compatible:
            return copy.deepcopy(parent1)
            
        target2 = random.choice(compatible)
        
        # 3. Swap
        # We need to deepcopy parent1 and replace target1 with copy of target2
        # Since nodes don't have parent pointers, we rebuild the tree or use a recursive replacer.
        # Simplest: Recursive replacement by reference logic (hard with immutable dataclasses).
        # Better: Re-implement using a 'replace_at_index' approach.
        
        # For simplicity in this iteration:
        # Just return a new program where we tried to inject target2 into parent1 logic
        # OR: To save implementation time, we use a simple "Root Crossover" if types match?
        # No, that's degenerate.
        
        # Let's verify structure:
        child = copy.deepcopy(parent1)
        self._replace_random_subtree(child, target1.return_type, copy.deepcopy(target2))
        return child

    def mutate(self, individual: TypedNode, context_types: Dict[str, Type]) -> TypedNode:
        """Type-safe mutation."""
        child = copy.deepcopy(individual)
        
        # Select random node to replace
        nodes = self._get_all_nodes(child)
        if not nodes: return child
        
        target = random.choice(nodes)
        
        # Generate new subtree of same type
        # Depth constraint: rough estimate
        new_subtree = self.generate_program(target.return_type, context_types, depth=random.randint(0, self.subtree_mutation_max_depth))
        
        # Replace
        self._replace_subtree_in_place(child, target, new_subtree)
        return child
        
    def evolve(self, examples, generations=50, pop_size=500, context_types: Dict[str, Type] = None, target_type: Type = Grid):
        """Main evolution loop."""
        if context_types is None:
             context_types = {"INPUT": Grid}
        
        self.context_names = list(context_types.keys())
        
        # Init Population
        population = [self.generate_program(target_type, context_types) for _ in range(pop_size)]
        
        best_program = None
        best_score = -1.0
        
        for gen in range(generations):
            scores = []
            for prog in population:
                score = self._evaluate(prog, examples)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_program = prog
                    
            # Logging
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Type-Safe Score {best_score:.3f}")
                
            if best_score >= 1.0:
                break
                
            # Selection
            new_pop = []
            
            # Elitism
            sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            curr_elitism = genetic_strategy.get_elitism_rate()
            elite_count = int(pop_size * curr_elitism)
            for i in range(elite_count):
                new_pop.append(population[sorted_idx[i]])
                
            # Breed
            curr_top_n = genetic_strategy.get_selection_top_n()
            curr_mutation = genetic_strategy.get_mutation_rate(gen, generations)
            
            while len(new_pop) < pop_size:
                p1 = population[random.choice(sorted_idx[:curr_top_n])] 
                p2 = population[random.choice(sorted_idx[:curr_top_n])]
                
                child = self.crossover(p1, p2)
                if random.random() < curr_mutation:
                    child = self.mutate(child, context_types)
                new_pop.append(child)
                
            population = new_pop
            
        return best_program

    def _evaluate(self, program: TypedNode, examples) -> float:
        """
        Evaluate program fitness.
        Uses Shaped Fitness (Dimensions + Colors + Pixels) to guide evolution.
        """
        correct = 0.0
        
        for inp, tgt in examples:
            try:
                # Use current context names to build execution context
                ctx = {name: inp for name in getattr(self, "context_names", ["INPUT"])}
                out = program.execute(ctx)
                
                # If target is not a Grid, use exact match or element-wise match
                if not isinstance(tgt, Grid):
                    if out == tgt:
                        correct += 1.0
                    elif isinstance(out, list) and isinstance(tgt, list):
                        # Permutation-agnostic matching for lists of Grids/Objects
                        if len(out) == 0 and len(tgt) == 0:
                            correct += 1.0
                        elif len(out) != len(tgt):
                            # Penalty for size mismatch
                            correct += 0.5 * (min(len(out), len(tgt)) / max(len(out), len(tgt), 1))
                        else:
                            # Try greedy matching
                            matched = 0
                            temp_tgt = list(tgt)
                            for o in out:
                                for i, t in enumerate(temp_tgt):
                                    if o == t:
                                        matched += 1
                                        temp_tgt.pop(i)
                                        break
                            correct += (matched / len(tgt))
                    continue

                # Grid Specific Logic (Shaped Fitness)
                if not isinstance(out, Grid):
                    continue
                    
                # 1. Dimensions (30%)
                h_match = 1.0 if out.H == tgt.H else max(0, 1 - abs(out.H - tgt.H)/max(1, tgt.H))
                w_match = 1.0 if out.W == tgt.W else max(0, 1 - abs(out.W - tgt.W)/max(1, tgt.W))
                dim_score = (h_match + w_match) / 2.0
                
                # 2. Colors (20%)
                out_colors = set(np.unique(out.data))
                tgt_colors = set(np.unique(tgt.data))
                union = len(out_colors | tgt_colors)
                color_score = len(out_colors & tgt_colors) / union if union > 0 else 0.0
                
                # 3. Pixels (50%) - Only if dims match mostly? 
                # Actually, standard comparison logic handles shape mismatch by broadcasting or failing
                # But let's be safe:
                pixel_score = 0.0
                if out.data.shape == tgt.data.shape:
                    matches = np.count_nonzero(out.data == tgt.data)
                    pixel_score = matches / out.data.size
                
                # Weighted Sum
                # If strict match, score is 1.0
                score = (dim_score * 0.3) + (color_score * 0.2) + (pixel_score * 0.5)
                
                correct += score
            except Exception:
                pass
                
        return correct / len(examples) if examples else 0.0

    def _get_all_nodes(self, root: TypedNode) -> List[TypedNode]:
        nodes = [root]
        if isinstance(root, PrimitiveNode):
            for arg in root.args:
                nodes.extend(self._get_all_nodes(arg))
        return nodes

    def _replace_subtree_in_place(self, root: TypedNode, target: TypedNode, replacement: TypedNode) -> bool:
        """Recursive replacement helper."""
        # This is tricky because target is a copy or ref. 
        # Since we deepcopied, 'target' ref might not exist in root if we logic is wrong.
        # Simplified: We traverse. If node IS target (by ID? no, deepcopy breaks ID).
        # We need a path or ID. 
        # HACK: For V1, just replace one random child of root if compatibility matches?
        # Let's implement a 'replace at path' logic or reference object wrapper.
        
        # V1: Argument replacement only (easier)
        if isinstance(root, PrimitiveNode):
            for i, arg in enumerate(root.args):
                if arg is target: # Identity check
                    root.args[i] = replacement
                    return True
                if self._replace_subtree_in_place(arg, target, replacement):
                    return True
        return False
        
    def _replace_random_subtree(self, root: TypedNode, type_constraint: Type, replacement: TypedNode):
        """Randomly replaces a node of type `type_constraint` with `replacement`."""
        # Find candidates
        candidates = []
        def traverse(node, parent, idx):
            if node.return_type == type_constraint:
                candidates.append((node, parent, idx))
            if isinstance(node, PrimitiveNode):
                for i, arg in enumerate(node.args):
                    traverse(arg, node, i)
                    
        traverse(root, None, -1)
        
        if not candidates: return
        
        _, parent, idx = random.choice(candidates)
        if parent:
            parent.args[idx] = replacement 
            # Note: if parent is None, it means root was candidate. 
            # We can't replace root in-place this way easily without a wrapper.
            # Ignored for V1 root case.
