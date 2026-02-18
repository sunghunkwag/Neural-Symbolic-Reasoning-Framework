# Neural-Symbolic Reasoning Framework

> **Research Note**: This is a proof-of-concept exploring the integration of Monte Carlo Tree Search (MCTS) with domain-specific languages for program synthesis. It is not a general-purpose synthesizer.

## Abstract

This repository explores whether **Monte Carlo Tree Search (MCTS)** can outperform standard genetic algorithms in program synthesis tasks where the search space is highly structured but sparse.

The core hypothesis is that replacing stochastic mutation with lookahead search (MCTS) allows for solving tasks that require sequential logic (e.g., "Shift then Recolor"), which often trap greedy evolutionary approaches in local optima.

---

## Core Architecture

### 1. MCTS Logic Solver (`mcts_solver.py`)
- **Mechanism**: UCB1 exploration over a grammar of partial ASTs.
- **Goal**: Navigate the combinatorial explosion of program synthesis more efficiently than random walk.
- ** Limitation**: The state space grows exponentially with program length. MCTS is only effective for very short programs (< 5 DSL primitives).

### 2. Sequential Decomposition
- **Concept**: Attempting to break tasks into $P_{final}(x) = P_n(...P_1(x))$.
- **Status**: Manual decomposition works for toy examples (Shift+Recolor), but automated decomposition remains unsolved.

---

## Experimental Observations

We compared MCTS against a baseline Genetic Algorithm (GA) on a strict subset of the ARC-Mini benchmark.

| Task Type | Baseline (GA) | MCTS (Ours) | Observation |
| :--- | :--- | :--- | :--- |
| **Atomic (Identity)** | Solved | Solved | Trivial baseline. |
| **Composite (Shift+Recolor)** | Failed (0%) | Solved (100%)* | *Only with constrained DSL. |

**Key Failure Mode**: When the DSL is expanded to include more than ~10 primitives, MCTS performance degrades rapidly, often failing to converge within a reasonable time budget due to the branching factor.

---

## Usage (Experimental)

```python
from aria.logic.synthesizer import LogicSynthesizer
from aria.types import Grid

# Initialize Synthesizer
# Warning: High memory usage for deep search trees
synthesizer = LogicSynthesizer()

# Attempt to solve (may hang if search space is too large)
examples = [(input_grid, output_grid)]
program = synthesizer.solve(examples, use_mcts=True)
print(program)
```

## Repository Structure
- `aria/logic/mcts_solver.py`: The search implementation.
- `aria/logic/synthesizer.py`: Integration logic.
- `tests/`: Benchmarks on specific (cherry-picked) tasks.

## Disclaimer
This is research code. It is brittle, unoptimized, and intended only for reproducing specific experimental results on program synthesis search methods.
