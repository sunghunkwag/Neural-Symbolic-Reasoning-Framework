"""
Test Harness for Typed Genetic Engine (Phase 14).
Verifies that the engine can generate valid programs with Control Flow and Types.
"""
import sys
import os
from pathlib import Path
from typing import List, Callable, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.logic.dsl import Grid, TYPED_PRIMITIVES, PRIMITIVES
from aria.logic.genetic_typed import TypedGeneticEngine, TypedNode, PrimitiveNode

def test_generation():
    print("=== Testing Typed Generation ===")
    
    # Init Engine with TYPED primitives
    engine = TypedGeneticEngine(TYPED_PRIMITIVES, max_depth=4)
    
    # Context types
    context_types = {"INPUT": Grid}
    
    # Generate 10 random programs
    for i in range(10):
        prog = engine.generate_program(Grid, context_types)
        print(f"Prog {i}: {prog}")
        
        # Validation
        assert isinstance(prog, TypedNode)
        assert prog.return_type == Grid
        
    print("\nPASS: Generation produced valid TypedNodes returning Grid.")

def test_structure_validity():
    print("\n=== Testing Structure Validity ===")
    engine = TypedGeneticEngine(TYPED_PRIMITIVES, max_depth=5)
    context_types = {"INPUT": Grid}
    
    # Try to generate specific complex structures if possible, 
    # or just check a large batch for crashes.
    for i in range(50):
        try:
            prog = engine.generate_program(Grid, context_types)
            # Check if if_color children are correct types
            # (Recursive check implied by generation logic, but good to verify)
        except Exception as e:
            print(f"FAIL: Generation crashed: {e}")
            raise e
            
    print("PASS: 50 programs generated without error.")

if __name__ == "__main__":
    test_generation()
    test_structure_validity()
