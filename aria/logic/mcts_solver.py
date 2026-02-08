
import math
import random
import copy
import numpy as np
from typing import List, Dict, Any, Type, Tuple, Optional
from dataclasses import dataclass, field

from aria.types import Grid
from aria.logic.genetic_typed import TypedNode, PrimitiveNode, VariableNode, ConstantNode, LambdaNode
from aria.logic.dsl import TYPED_PRIMITIVES

# --- Logic State Representation ---

@dataclass
class HoleNode(TypedNode):
    """Represents a 'Hole' in the AST that needs to be filled."""
    def execute(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError("Cannot execute a program with Holes!")
    
    def __repr__(self):
        return f"?[{self.return_type.__name__}]"

@dataclass
class MCTSNode:
    """Node in the MCTS Search Tree (State = Partial Program)."""
    program: TypedNode
    holes: List[HoleNode] # References to HoleNodes within 'program'
    
    parent: Optional['MCTSNode'] = None
    action: Optional[Any] = None # The action that led here (primitive/terminal)
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    
    visits: int = 0
    value_sum: float = 0.0
    
    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visits)
    
    def is_fully_expanded(self, possible_actions: int) -> bool:
        return len(self.children) == possible_actions

class MCTSSolver:
    """
    Solves synthesis tasks using Monte Carlo Tree Search.
    This replaces random evolution with structured reasoning.
    """
    def __init__(self, primitives: List[Tuple], max_depth: int = 5):
        self.primitives = primitives
        self.max_depth = max_depth
        
        # Index primitives
        self.primitives_by_type = {}
        for func, arg_types, ret_type in primitives:
            if ret_type not in self.primitives_by_type:
                self.primitives_by_type[ret_type] = []
            self.primitives_by_type[ret_type].append((func, arg_types))

    def solve(self, examples, iterations=1000, context_types=None, target_type=Grid):
        """Main search loop."""
        if context_types is None:
            context_types = {"INPUT": Grid}
            
        # Dynamic Vocabulary extraction
        self.task_constants = {0, 1} # Always include basic ints
        for inp, tgt in examples:
             if isinstance(inp, Grid):
                 self.task_constants.update(np.unique(inp.data))
             if isinstance(tgt, Grid):
                 self.task_constants.update(np.unique(tgt.data))
        
        # Root is a single Hole
        root_hole = HoleNode(target_type)
        root = MCTSNode(program=root_hole, holes=[root_hole])
        
        best_program = None
        best_score = -1.0
        
        for i in range(iterations):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion
            if node.holes:
                # We can expand if there are holes
                node = self._expand(node, context_types)
                
            # 3. Simulation
            score, completed_program = self._simulate(node, examples, context_types)
            
            # Thought Stream Logger (Traceable Consciousness)
            if i < 5:
                print(f"[MCTS-TRACE] Iter {i}: Score={score:.4f} | Prog: {completed_program}")

            # Check best
            if score > best_score:
                best_score = score
                best_program = completed_program
                # Early exit if perfect
                if best_score >= 1.0:
                    break
            
            # 4. Backpropagation
            self._backpropagate(node, score)
            
        return best_program

    def solve_sequential(self, examples, iterations=1000, max_steps=5, context_types=None, target_type=Grid):
        """
        Solves the task using Chain-of-Thought (Sequential) Reasoning.
        Finds a program P1 that improves state, then P2 that improves P1(state), etc.
        """
        current_examples = examples
        program_chain = []
        
        for step in range(max_steps):
            print(f"--- Reasoning Step {step+1}/{max_steps} ---")
            
            # Run MCTS to find *incremental* improvement
            # We reduce iterations per step to keep total budget reasonable
            step_iters = iterations // max_steps
            best_prog = self.solve(current_examples, iterations=step_iters, context_types=context_types, target_type=target_type)
            
            if not best_prog:
                print("  No improvement found.")
                break
                
            # Check improvement
            # Note: self.solve returns best program found.
            # We need to compute score of this program on current_examples
            score = self._evaluate(best_prog, current_examples, context_types or {"INPUT": Grid})
            print(f"  Step Program: {best_prog}, Score: {score:.4f}")
            
            # If score is perfect, we are done with this step (and total)
            # But we need to check if *previous* steps + this step = perfect.
            # Actually, `best_prog` transforms `current_examples` to `target`.
            # So if score 1.0, we solved the *residual* problem.
            
            # Update examples for next step
            # New Input = best_prog(Current Input)
            # Target remains same.
            new_examples = []
            valid_transform = True
            for inp, tgt in current_examples:
                try:
                    ctx = {"INPUT": inp} # Simplified context for chain
                    # If best_prog has holes, execute fails. mcts solve ensures it doesn't return holes unless failure.
                    out = best_prog.execute(ctx)
                    new_examples.append((out, tgt))
                except:
                    valid_transform = False
                    break
            
            if not valid_transform:
                print("  Invalid transform execution.")
                break
                
            program_chain.append(best_prog)
            current_examples = new_examples
            
            # Check if solved
            total_correct = 0
            for inp, tgt in current_examples:
                if inp == tgt: total_correct += 1
            if total_correct == len(current_examples):
                print("  Residual task solved!")
                break
                
            # Optimization: If score didn't improve over Identity (Start state of this step)
            # We evaluate 'INPUT' prog on current_examples.
            # But MCTS `solve` starts with -1.0 best_score.
            # It returns the best it found.
            # If `best_prog` is just `INPUT` (Identity), then we made no progress.
            # Check if trivial identity?
            # VariableNode("INPUT")
            if isinstance(best_prog, VariableNode) and best_prog.name == "INPUT":
                 print("  Stagnation (Identity found). stopping.")
                 break
                 
        return program_chain
        """Main search loop."""
        if context_types is None:
            context_types = {"INPUT": Grid}
            
        # Dynamic Vocabulary extraction
        self.task_constants = {0, 1} # Always include basic ints
        for inp, tgt in examples:
             if isinstance(inp, Grid):
                 self.task_constants.update(np.unique(inp.data))
             if isinstance(tgt, Grid):
                 self.task_constants.update(np.unique(tgt.data))
        
        # Root is a single Hole
        root_hole = HoleNode(target_type)
        root = MCTSNode(program=root_hole, holes=[root_hole])
        
        best_program = None
        best_score = -1.0
        
        for i in range(iterations):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion
            if node.holes:
                # We can expand if there are holes
                node = self._expand(node, context_types)
                
            # 3. Simulation
            score, completed_program = self._simulate(node, examples, context_types)
            
            # Thought Stream Logger (Traceable Consciousness)
            if i < 5:
                print(f"[MCTS-TRACE] Iter {i}: Score={score:.4f} | Prog: {completed_program}")

            # Check best
            if score > best_score:
                best_score = score
                best_program = completed_program
                # Early exit if perfect
                if best_score >= 1.0:
                    break
            
            # 4. Backpropagation
            self._backpropagate(node, score)
            
        return best_program

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB Selection to find a leaf or expandable node."""
        while not node.holes: # If no holes, it's a terminal/leaf in search
             if not node.children:
                 return node
             # It shouldn't have children if it has no holes... wait.
             # A completed program is a terminal in Search Tree.
             return node
             
        # If node has holes, checks if it is fully expanded?
        # In this implementation, _expand adds ONE child.
        # So we select until we hit a node that has unexplored actions OR is terminal.
        
        # Actually, standard MCTS:
        # If not all children expanded, return self (to expand).
        # Else, pick best child and recurse.
        
        # How many possible actions?
        next_hole = node.holes[0]
        possible_actions = self._get_possible_actions(next_hole.return_type, context_types=None) # Hack: context needed
        # We need context_types in select?
        # Let's handle 'context_types' stored in MCTSNode if needed, or pass it down.
        # For variable selection, we need context.
        # Let's simplify: pass context to _get_possible_actions dynamically or store in solver.
        
        # Simpler Logic:
        # Just traverse down existing children using UCB.
        # If we hit a node that has potential children not yet created, return that node.
        
        while node.children:
            # Are there unexpanded actions?
            # Hard to know total count efficiently.
            # Let's use UCB to pick a child.
            # If UCB picks a child that doesn't exist? No, we iterate existing.
            
            # Wait, expansion creates a NEW child.
            # If we haven't tried all actions, we should stay here?
            # Standard: if len(children) < len(legal_moves): return node
            
            # Need to know legal moves count.
            # Passed context_types? Assume standard {"INPUT": Grid} for now or store in init.
            
            # Optimization: Just proceed if fully expanded.
            # If not fully expanded, we return node.
            # How to track 'fully expanded'? 
            # We'll just try to expand. If expand returns None (no more moves), then we treat as full.
            
            legal_count = self._count_legal_actions(node)
            if len(node.children) < legal_count:
                return node
                
            # UCB Select
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                ucb = child.value + 1.41 * math.sqrt(math.log(node.visits) / (child.visits + 1))
                if ucb > best_score:
                    best_score = ucb
                    best_child = child
            
            node = best_child
            
        return node

    def _expand(self, node: MCTSNode, context_types: Dict[str, Type]) -> MCTSNode:
        """Add a new child by filling the first hole with a valid action."""
        hole_to_fill = node.holes[0] # Deterministic: always fill first hole (DFS-like construction)
        rarity_depth = self._get_depth(node.program, hole_to_fill)
        
        # Get all valid actions
        actions = self._get_possible_actions(hole_to_fill.return_type, context_types, rarity_depth)
        
        # Shuffle to randomize expansion order
        random.shuffle(actions)
        
        for action in actions:
            action_key = str(action) # Simple string key
            if action_key not in node.children:
                # Create Child
                new_program, new_holes = self._apply_action(node.program, node.holes, action)
                child = MCTSNode(program=new_program, holes=new_holes, parent=node, action=action)
                node.children[action_key] = child
                return child
                
        # If all expanded, return None or self?
        # Should back out.
        return node # Should not happen if check logic is right

    def _simulate(self, node: MCTSNode, examples, context_types) -> Tuple[float, TypedNode]:
        """Random rollout to complete the partial program."""
        # 1. Copy program
        curr_prog = copy.deepcopy(node.program)
        
        # 2. Find all holes
        # (Need to rediscover holes in the copy, because node.holes points to original nodes)
        # Helper to find holes
        holes = self._find_holes(curr_prog)
        
        # 3. Randomly fill holes
        depth = 0
        while holes:
            h = holes.pop(0)
            # Fill h
            # Pick random action
            possibles = self._get_possible_actions(h.return_type, context_types, depth)
            if not possibles:
                # Fallback: Try to use a terminal if available regardless of type matching strictness?
                # Or just abort this rollout
                # self.logger.warning(f"Simulate: No actions for type {h.return_type} at depth {depth}")
                return 0.0, curr_prog # Return minimal score
            
            try:
                action = random.choice(possibles)
            except IndexError:
                 return 0.0, curr_prog

            # Apply
            # Replaces 'h' in 'curr_prog'.
            # Note: _apply_action logic needs to handle replacement in tree.
            # Since 'h' is an object in 'curr_prog', we can mutate 'h' if we change it to the new node?
            # No, TypedNode is dataclass.
            # We need a robust replacement.
            
            # Simple Hack: define HoleNode as mutable wrapper?
            # Better: _replace_node(root, target, replacement) works if target is unique object.
            
            new_node, new_sub_holes = self._instantiate_action(action)
            
            # Special case: Root replacement
            if h is curr_prog:
                curr_prog = new_node
            else:
                self._replace_node(curr_prog, h, new_node)
            
            # Add new holes
            # Insert at front for DFS? Or back for BFS?
            holes = new_sub_holes + holes # DFS
            
            depth += 1
            if depth > 20: break # Safety
            
        # 4. Evaluate
        try:
            score = self._evaluate(curr_prog, examples, context_types)
        except Exception as e:
            # Log execution failure
            # print(f"Execution Error: {e}")
            score = 0.0

        return score, curr_prog

    def _backpropagate(self, node: MCTSNode, score: float):
        while node:
            node.visits += 1
            node.value_sum += score
            node = node.parent

    # --- Helpers ---

    def _get_possible_actions(self, target_type: Type, context_types: Dict, current_depth: int = 0):
        actions = []
        
        # 1. Primitives
        if current_depth < self.max_depth:
            funcs = self.primitives_by_type.get(target_type, [])
            for f, args in funcs:
                actions.append(("FUNC", f, args))
                
        # 2. Terminals (Vars)
        if context_types:
            for name, typ in context_types.items():
                if typ == target_type:
                    actions.append(("VAR", name))
                    
        # 3. Constants (Simplified)
        if target_type == int:
            # ARC Colors 0-9 + maybe explicit simple ints
            for i in range(10):
                actions.append(("CONST", i))
        elif target_type == bool:
            actions.append(("CONST", True))
            actions.append(("CONST", False))
        elif target_type == float:
            actions.append(("CONST", 0.0))
            actions.append(("CONST", 1.0))
        elif hasattr(target_type, "__origin__") and target_type.__origin__ is list:
             # Generics like List[Grid]
             actions.append(("CONST", []))
        
        # 4. Functional References (for HOFs)
        # Check if target_type is Callable[[Arg...], Ret]
        # logic for typing.Callable inspection is messy across python versions, generic check:
        if str(target_type).startswith("typing.Callable"):
            # Extract args and ret
            # target_type.__args__ usually [Arg1, Arg2, ... Ret]
            try:
                args_and_ret = target_type.__args__
                param_types = args_and_ret[:-1]
                ret_type = args_and_ret[-1]
                
                # Search primitives
                for p_func, p_args, p_ret in self.primitives:
                    if p_ret == ret_type and list(p_args) == list(param_types):
                         actions.append(("FUNC_REF", p_func))
            except:
                pass

        return actions

    def _apply_action(self, root_prog, holes, action):
        """Returns new_root, new_holes_list"""
        # Deepcopy first
        new_root = copy.deepcopy(root_prog)
        # Find the hole corresponding to holes[0]
        # Since we deepcopied, we can't use object identity from 'holes' list directly.
        # We need a path approach (e.g. [0, 1, 0]).
        # OR: Just re-find holes in new_root. The first hole in 'root' corresponds to first hole in 'new_root' if deterministic.
        all_holes = self._find_holes(new_root)
        target_hole = all_holes[0]
        
        new_node, new_sub_holes = self._instantiate_action(action)
        
        # Replace
        self._replace_node(new_root, target_hole, new_node)
        
        # Update hole list
        # Remove first hole, add new holes at front
        remaining_holes = all_holes[1:]
        final_holes = new_sub_holes + remaining_holes
        
        return new_root, final_holes

    def _instantiate_action(self, action):
        kind, val, *rest = action
        if kind == "FUNC":
            func, arg_types = val, rest[0]
            args = []
            holes = []
            for t in arg_types:
                h = HoleNode(t)
                args.append(h)
                holes.append(h)
            return PrimitiveNode(Any, func, args, func.__name__), holes
        elif kind == "VAR":
            return VariableNode(Any, val), []
        elif kind == "CONST":
            return ConstantNode(int, val), []
        elif kind == "FUNC_REF":
             # Functional Reference is just a constant holding the function
             return ConstantNode(Any, val), []
        return None, []

    def _find_holes(self, root) -> List[HoleNode]:
        found = []
        if isinstance(root, HoleNode):
            found.append(root)
        elif isinstance(root, PrimitiveNode):
            for a in root.args:
                found.extend(self._find_holes(a))
        elif isinstance(root, LambdaNode):
            found.extend(self._find_holes(root.body))
        return found
        
    def _replace_node(self, root, target, replacement):
        # Recursive replace by identity
        # Limitation: works because HoleNodes are unique objects
        if isinstance(root, PrimitiveNode):
            for i, arg in enumerate(root.args):
                if arg is target:
                    root.args[i] = replacement
                    return True
                if self._replace_node(arg, target, replacement):
                    return True
        return False
        
    def _evaluate(self, program, examples, context_types):
        total_score = 0.0
        for inp, tgt in examples:
            try:
                ctx = {name: inp for name in context_types.keys()}
                out = program.execute(ctx)
                if out == tgt:
                    total_score += 1.0
                elif isinstance(out, Grid) and isinstance(tgt, Grid):
                    # Pixel-wise partial credit
                    if out.H == tgt.H and out.W == tgt.W:
                         matches = np.sum(out.data == tgt.data)
                         total = out.H * out.W
                         pixel_score = (matches / total)
                         
                         # Histogram Score (Feature Matching)
                         # Rewards getting the right colors even if in wrong place
                         out_counts = np.bincount(out.data.flatten(), minlength=10)
                         tgt_counts = np.bincount(tgt.data.flatten(), minlength=10)
                         # Simple intersect over union or similar
                         # We use 1 - L1 distance normalized
                         diff = np.abs(out_counts - tgt_counts).sum()
                         hist_score = max(0, 1.0 - (diff / total))
                         
                         raw_score = (pixel_score * 0.8) + (hist_score * 0.2)
                         
                         # Penalty for trivial empty output if target is not empty
                         if np.max(tgt.data) > 0 and np.max(out.data) == 0:
                             raw_score *= 0.5 
                             
                         total_score += raw_score
                    else:
                         # Dimension mismatch penalty, but maybe small score
                         total_score += 0.0
                else:
                    # Wrong type 
                    pass
            except: 
                pass
        return total_score / len(examples) if examples else 0.0

    def _count_legal_actions(self, node):
        if not node.holes: return 0
        h = node.holes[0]
        # Hacky: pass empty context or default if stored?
        # MCTSNode doesn't store context.
        # Assume standard context for counting check
        return len(self._get_possible_actions(h.return_type, {"INPUT": Grid}))
    
    def _get_depth(self, root, target, current_depth=0):
        if root is target:
            return current_depth
        if isinstance(root, PrimitiveNode):
            for arg in root.args:
                d = self._get_depth(arg, target, current_depth + 1)
                if d != -1: return d
        elif isinstance(root, LambdaNode):
            return self._get_depth(root.body, target, current_depth + 1)
        return -1
