
import os
import sys

# Ensure aria_core is in path
sys.path.append(os.getcwd())

from aria.logic.synthesizer import LogicSynthesizer
from aria.logic.dsl import Grid # Import if needed for types

def test_algorithmic_rsi():
    print("=== Phase 20: Algorithmic RSI Verification ===")
    
    synth = LogicSynthesizer()
    
    target_file = os.path.join("aria", "rsi", "algorithmic_target.py")
    
    # Define Examples for "Positive Even Numbers"
    # Note: Logic = filter_list(INPUT, lambda x: and(is_positive(x), is_even(x)))
    examples = [
        ([1, 2, 3, 4], [2, 4]),
        ([-2, -1, 0, 5, 6], [6]),
        ([10, 11, 12, 13], [10, 12]),
        ([0, -4, 7], []),
        ([8, 10, 12], [8, 10, 12]),
        ([-10, -8, -6], [])
    ]
    
    # Run Algorithmic Evolution
    # We specify input/output types as List[int]
    from typing import List
    success = synth.evolve_algorithm(
        file_path=target_file,
        function_name="process_data",
        examples=examples,
        input_type=List[int],
        output_type=List[int]
    )
    
    if success:
        print("\n=== VERIFICATION SUCCESS: File Patched structurally! ===")
        # 1. Inspect file content
        with open(target_file, 'r') as f:
            content = f.read()
            print("\nPatched File Preview:")
            print("-" * 20)
            print(content)
            print("-" * 20)
            
        # 2. Run the patched code
        import importlib.util
        spec = importlib.util.spec_from_file_location("patched_target", target_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        test_val = [2, -4, 6, 7, 8]
        result = mod.process_data(test_val)
        print(f"Validation Run: process_data({test_val}) -> {result}")
        if result == [2, 6, 8]:
            print("LOGIC CORRECTLY EVOLVED!")
        else:
            print(f"LOGIC FAILED VALIDATION. Expected [2, 6, 8], got {result}")
    else:
        print("\n=== VERIFICATION FAILED: Synthesis did not reach 100% ===")

if __name__ == "__main__":
    test_algorithmic_rsi()
