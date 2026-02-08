def main():
    print("Starting main...")
    engine = GeneticEngine()
    print("Engine initialized")
    count_shift = 0
    count_target = 0
    
    N = 10
    print(f"Generating {N} random programs...")
    
    for _ in range(N):
        prog = engine._random_program(depth=0, force_terminal=True)
        # Check if function is shift
        if prog.func.__name__ == 'shift':
            count_shift += 1
            # Check args. Shift args are [dx, dy] ?
            # Wait, ProgramNode args for shift: [input, dx, dy] ?? 
            # Or just [dx, dy]?
            # Let's check prog.args
            # If args[0] is 'input' or ProgramNode, and then dx, dy...
            try:
                # args structure depends on _generate_args
                # It appends Grid arg first (which becomes 'input' or Node)
                # Then dx, dy.
                if len(prog.args) == 3:
                    dx = prog.args[1]
                    dy = prog.args[2]
                    if dx == 1 and dy == 0:
                        count_target += 1
            except:
                pass
                
    print(f"Total Programs: {N}")
    print(f"Shift primitives: {count_shift} ({count_shift/N*100:.2f}%)")
    print(f"Target 'shift(?, 1, 0)': {count_target} ({count_target/N*100:.2f}%)")
    
    if count_target > 0:
        print("PASS: Generator can produce target.")
    else:
        print("FAIL: Generator cannot produce target.")
