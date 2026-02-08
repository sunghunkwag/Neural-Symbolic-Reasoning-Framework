
def get_mutation_rate(gen, max_gen):
    # Evolved in Phase 22: High mutation (0.9) is superior for small populations
    # We use a slight annealing to stabilize toward the end.
    return 0.9 * (1.1 - (gen / (max_gen + 1)))

def get_elitism_rate():
    return 0.1

def get_selection_top_n():
    return 50

def get_terminal_prob():
    # Meta-RSI finding: terminals are crucial for short solutions
    return 0.5
