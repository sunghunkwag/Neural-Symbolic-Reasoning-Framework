print("Hello World")
try:
    import aria.logic.dsl
    print("Imported aria.logic.dsl")
except Exception as e:
    print(f"Failed to import dsl: {e}")

try:
    import aria.logic.genetic
    print("Imported aria.logic.genetic")
    engine = aria.logic.genetic.GeneticEngine()
    print(f"Engine instantiated. Pop size: {engine.pop_size}")
    
    count_shift = 0
    count_target = 0
    N = 10000
    print(f"Starting loop for N={N}")
    for i in range(N):
        prog = engine._random_program(depth=0, force_terminal=True)
        if prog.func.__name__ == 'shift':
             count_shift += 1
             if prog.func != aria.logic.dsl.shift:
                 print(f"IDENTITY MISMATCH: {prog.func} id={id(prog.func)} vs {aria.logic.dsl.shift} id={id(aria.logic.dsl.shift)}")
             else:
                 # It matches. Then why args are bad?
                 # Print args for first few
                 if count_shift < 5:
                     print(f"Shift args: {prog.args}")
             
             if len(prog.args) >= 3:
                 dx = prog.args[1]
                 dy = prog.args[2]
                 if dx == 1 and dy == 0:
                     count_target += 1
             
    print(f"Loop finished.")
    print(f"Total Programs: {N}")
    print(f"Shift primitives: {count_shift} ({count_shift/N*100:.2f}%)")
    print(f"Target 'shift(?, 1, 0)': {count_target} ({count_target/N*100:.2f}%)")
except Exception as e:
    print(f"Failed to instatiate: {e}")
    import traceback
    traceback.print_exc()
