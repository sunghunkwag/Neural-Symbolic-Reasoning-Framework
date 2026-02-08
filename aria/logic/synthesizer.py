"""
Logic Synthesizer — Phase 20 (Structural RSI)
Bridges ARIA Core Goal System with Genetic Programming Engine.
Supports both parameter optimization and algorithmic evolution.
"""
import logging
import time
import importlib.util
import sys
import random
import ast
import inspect
from typing import List, Tuple, Optional, Callable, Any, Dict, Type

from aria.types import Grid
from .dsl import TYPED_PRIMITIVES
from .primitives import filter_list, is_even, is_positive, logical_and
from .mcts_solver import MCTSSolver
from .genetic_typed import TypedNode, PrimitiveNode, VariableNode, TypedGeneticEngine
import copy

class LogicSynthesizer:
    """
    Bridge between ARIA Core and Neural-Symbolic Logic Synthesis (Phase 23).
    Prioritizes MCTS Reasoning (O1-style) over Genetic Evolution.
    """
    def __init__(self):
        self.logger = logging.getLogger("LogicSynthesizer")
        self.genetic_engine = TypedGeneticEngine(TYPED_PRIMITIVES)
        self.reasoning_engine = MCTSSolver(TYPED_PRIMITIVES, max_depth=3)
        self.logger.info("LogicSynthesizer initialized with MCTSSolver (Reasoning) + Genetic (Backup)")
        
    def solve(self, examples: List[Tuple[Any, Any]], target_type: Type = Grid, context_types: Dict[str, Type] = None, task_name: str = "Unknown", use_mcts: bool = True) -> Optional[TypedNode]:
        """Synthesize a program that satisfies the examples."""
        self.logger.info(f"SYNTHESIS: Started for task '{task_name}' (Mode: {'MCTS' if use_mcts else 'Genetic'})")
        try:
            if context_types is None:
                context_types = {"INPUT": Grid}
            
            best_program = None
            
            if use_mcts:
                # Phase 23: Reasoning Engine (Sequential Chain-of-Thought)
                # 20k iters is a good balance for general tasks
                chain = self.reasoning_engine.solve_sequential(
                    examples, 
                    iterations=20000, 
                    max_steps=5,
                    context_types=context_types, 
                    target_type=target_type
                )
                # Combine chain? solve_sequential returns list of programs.
                # Actually, solve_sequential returns [P1, P2, ...].
                # We need to compile them into ONE program if possible?
                # Or just return the *last* one if it handles residual?
                # No, sequential means Pn(Pn-1(...)).
                # We need to compose them.
                # But TypedNode doesn't support easy composition unless we wrap in Lambda?
                # Helper to compose:
                if chain:
                    # If chain has 1 item, return it.
                    if len(chain) == 1:
                        best_program = chain[0]
                    else:
                        # Composition Logic: P_final(x) = Pn(...P1(x))
                        # This requires nesting.
                        # But P2 expects 'INPUT' to be the result of P1.
                        # P2 uses VariableNode("INPUT").
                        # We can simply substitute "INPUT" in P2 with P1.
                        # Recursive substitution.
                        combined = chain[0]
                        for next_prog in chain[1:]:
                            # Replace 'INPUT' var in next_prog with 'combined' AST
                            # Need deep copy of next_prog
                            next_prog_copy = copy.deepcopy(next_prog)
                            # Find all VariableNode("INPUT")
                            # We need a helper for AST replacement.
                            # MCTS solver has _replace_node but specialized for Holes.
                            # Let's just use the LAST program if we can't easily compose?
                            # No, last program solves residual.
                            # If we can't compose, we fail.
                            
                            # Let's implement simple substitution here.
                            if self._substitute_input(next_prog_copy, combined):
                                combined = next_prog_copy
                            else:
                                self.logger.warning("Failed to compose program chain.")
                                # Fallback: just return last (might be wrong)
                                combined = next_prog_copy
                        best_program = combined
            
            if not best_program:
                 # Fallback to Genetic
                 self.logger.info("MCTS failed or disabled. Falling back to Genetic Engine.")
                 best_program = self.genetic_engine.evolve(
                    examples, 
                    generations=100, 
                    pop_size=500,
                    context_types=context_types,
                    target_type=target_type
                )
            
            if best_program:
                # Verify final score
                # Note: engine._evaluate works for TypedNode
                score = self.genetic_engine._evaluate(best_program, examples)
                self.logger.info(f"SYNTHESIS: Final Score: {score:.3f}")
                if score >= 1.0:
                    self.logger.info(f"SYNTHESIS: SUCCESS! Program: {best_program}")
                    return best_program
                return best_program
            return None
        except Exception as e:
            self.logger.error(f"SYNTHESIS FAILED: {e}", exc_info=True)
            return None

    def _substitute_input(self, root: TypedNode, replacement: TypedNode) -> bool:
        """Recursively replaces VariableNode('INPUT') with 'replacement'."""
        # This modifies root in-place.
        # TypedNode children are in 'args' list for PrimitiveNode.
        import copy # Ensure copy is avail
        
        replaced = False
        if isinstance(root, PrimitiveNode):
            for i, arg in enumerate(root.args):
                if isinstance(arg, VariableNode) and arg.name == "INPUT":
                    root.args[i] = copy.deepcopy(replacement)
                    replaced = True # Mark as we found something
                else:
                    if self._substitute_input(arg, replacement):
                        replaced = True
        return replaced

    def evolve_algorithm(self, file_path: str, function_name: str, examples: List[Tuple[Any, Any]], input_type: Type, output_type: Type):
        """
        Phase 20: Algorithmic Evolution.
        Finds a structural replacement for a broken function in a file.
        """
        # 1. Inspect function signature to get arg name
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
            
        arg_name = "INPUT" # Fallback
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                if node.args.args:
                    arg_name = node.args.args[0].arg
                break
        
        print(f"RSI: Detected argument name for {function_name}: {arg_name}")
        context_types = {arg_name: input_type}
        
        # 2. Evolve solution
        best_prog = self.solve(
            examples, 
            target_type=output_type, 
            context_types=context_types,
            task_name=f"Evolve_{function_name}"
        )
        
        if not best_prog or self.engine._evaluate(best_prog, examples) < 1.0:
            print("RSI: Failed to evolve a 100% correct algorithm.")
            return False

        # 2. Convert Evolved Node to Python Source
        evolved_code = repr(best_prog)
        print(f"RSI: Evolved Logic: {evolved_code}")

        # 3. Patch the file using AST
        with open(file_path, 'r') as f:
            source = f.read()
            tree = ast.parse(source)

        # Inject Imports if missing - comprehensive list
        # We use a wildcard import to ensure all primitives are available for the evolved code
        import_stmt = "from aria.logic.dsl import *"
        has_import = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'aria.logic.dsl':
                has_import = True
                break

        if not has_import:
             import_node = ast.parse(import_stmt).body[0]
             tree.body.insert(0, import_node)

        # Verify evolved code syntax
        try:
            evolved_ast = ast.parse(evolved_code)
        except SyntaxError as e:
            self.logger.error(f"RSI: Evolved code has syntax error: {e}")
            return False

        class CodeReplaced(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == function_name:
                    # Replace the entire body with: return <evolved_code>
                    # We wrap in a return statement
                    if isinstance(evolved_ast.body[0], ast.Expr):
                        ret_val = evolved_ast.body[0].value
                    else:
                        ret_val = evolved_ast.body[0] # Fallback

                    new_body = [ast.Return(value=ret_val)]
                    node.body = new_body
                    print(f"RSI: Function '{function_name}' body replaced in AST.")
                return node

        new_tree = CodeReplaced().visit(tree)
        ast.fix_missing_locations(new_tree)

        # 4. Write back
        # Using ast.unparse (Python 3.9+)
        try:
            patched_content = ast.unparse(new_tree)
            with open(file_path, 'w') as f:
                f.write(patched_content)
            print(f"RSI: File {file_path} patched with structural logic.")
            return True
        except Exception as e:
            print(f"RSI: Failed to write patched file: {e}")
            return False

    def optimize_file(self, file_path: str, target_function: str = None):
        """Phase 20: Parameter Optimization via AST."""
        print(f"RSI: Optimizing {file_path} constants...")

        with open(file_path, 'r') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self.logger.error(f"RSI: Syntax error in {file_path}: {e}")
            return False

        class SleepOptimizer(ast.NodeTransformer):
            def visit_Call(self, node):
                # Detect sleep(1.0) or time.sleep(1.0)
                is_sleep = False
                if isinstance(node.func, ast.Name) and node.func.id == 'sleep':
                    is_sleep = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr == 'sleep':
                    is_sleep = True

                if is_sleep and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and (arg.value == 1.0 or arg.value == 1):
                        # Replace with 0.001
                        print("RSI: Found sleep(1.0), optimizing to sleep(0.001)")
                        return ast.Call(
                            func=node.func,
                            args=[ast.Constant(value=0.001)],
                            keywords=node.keywords
                        )
                return node

        optimizer = SleepOptimizer()
        new_tree = optimizer.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            # Verify syntax
            new_source = ast.unparse(new_tree)
            ast.parse(new_source) # Double check

            with open(file_path, 'w') as f:
                f.write(new_source)
            print("RSI: Patch Applied (AST Safe).")
            return True
        except Exception as e:
            print(f"RSI: Failed to apply AST patch: {e}")
            return False
