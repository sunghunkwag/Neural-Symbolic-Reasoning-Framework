# ARIA v3.0: AGI-Oriented Improvements

## Executive Summary

ARIA v2.1 demonstrates MCTS can solve constrained program synthesis tasks, but suffers from fundamental scalability and learning limitations. ARIA v3.0 introduces mechanisms for:

1. **Episodic Memory** - Learn from past successes/failures
2. **Recursive Self-Improvement (RSI)** - Improve the solver itself
3. **Meta-Cognition** - Aware of performance gaps, autonomously adjust
4. **Dynamic DSL Expansion** - Discover new primitives from failures

---

## Part 1: Core Limitations Analysis

### ARIA v2.1 Bottlenecks

| Problem | Impact | AGI Implication |
|---------|--------|------------------|
| Exponential branching in MCTS state space | Scales poorly beyond ~5 primitives | Cannot generalize to arbitrary tasks |
| No memory mechanism | Learns nothing from past trajectories | Violates efficient learning principle |
| Stateless search | Every task starts from scratch | No meta-learning or transfer |
| No introspection | Cannot detect failure modes | Cannot autonomously improve |
| Fixed DSL | Cannot discover new operations | Bounded intelligence by initial choice |

### Why These Matter for AGI

**AGI requires:**
- Accumulating knowledge across tasks
- Detecting own failure modes autonomously
- Improving its own reasoning mechanisms
- Discovering new capabilities as needed

---

## Part 2: ARIA v3.0 Architecture

### 2.1 Episodic Memory System

**Goal**: Store successful program synthesis trajectories and reuse components.

```python
class EpisodicMemory:
    def __init__(self, max_episodes=10000):
        self.trajectories = []  # (task_features, solved_program, score)
        self.primitives_used = defaultdict(int)  # freq of each primitive
        
    def store(self, task_features, program, score):
        """Store a successful synthesis trajectory."""
        self.trajectories.append({
            'features': task_features,
            'program': program,
            'score': score,
            'timestamp': time.time()
        })
        
    def recall_similar(self, query_features, k=5):
        """Retrieve k most similar past solutions via cosine similarity."""
        # Compute similarity between query_features and stored features
        # Return top-k trajectories to initialize search
        pass
    
    def get_frequent_subprograms(self):
        """Extract common patterns from successful programs."""
        # Return frequently-used sub-expressions
        # These can be cached as "macro-primitives"
        pass
```

**Integration**: When solving a new task, query memory for similar past tasks. Use their solutions as search initialization (biasing).

### 2.2 Recursive Self-Improvement (RSI) Engine

**Goal**: Solver modifies its own parameters to improve performance.

```python
class RecursiveSelfImprovement:
    def __init__(self, mcts_solver):
        self.solver = mcts_solver
        self.hyperparams = {
            'max_depth': 5,
            'mcts_iterations': 1000,
            'exploration_constant': 1.41,
            'action_sample_rate': 0.5  # Proportion of actions sampled
        }
        self.performance_history = []  # (params, avg_score)
    
    def propose_modification(self):
        """
        Generate a candidate modification to solver hyperparameters.
        Candidates based on performance trend analysis.
        """
        recent_scores = self.performance_history[-20:]
        if not recent_scores:
            return None
        
        avg_score = np.mean([s for _, s in recent_scores])
        trend = recent_scores[-1][1] - recent_scores[0][1] if len(recent_scores) > 1 else 0
        
        modification = {}
        
        if trend < 0:  # Performance declining
            modification['mcts_iterations'] *= 1.5  # Increase search budget
            modification['exploration_constant'] *= 1.1  # Explore more
        elif avg_score < 0.5:  # Baseline poor
            modification['max_depth'] = min(7, self.hyperparams['max_depth'] + 1)
        
        return modification
    
    def evaluate_modification(self, modification, test_tasks, test_budget=100):
        """
        Test proposed modification on held-out tasks.
        Return improvement_score.
        """
        # Temporarily apply modification
        old_params = self.solver.hyperparams.copy()
        self.solver.hyperparams.update(modification)
        
        # Run on test tasks with limited budget
        scores = []
        for task in test_tasks[:test_budget]:
            score = self.solver.solve(task)
            scores.append(score)
        
        # Restore
        self.solver.hyperparams = old_params
        
        return np.mean(scores)
    
    def commit_modification(self, modification):
        """Apply successful modification permanently."""
        self.solver.hyperparams.update(modification)
        self.hyperparams = modification
```

**Integration**: Every N episodes, propose and test modifications. Commit if they improve average performance.

### 2.3 Meta-Cognition Layer

**Goal**: Monitor performance, detect failure modes, trigger recovery strategies.

```python
class MetaCognitionLayer:
    def __init__(self):
        self.recent_failures = []  # Task features of failures
        self.performance_baseline = None
        self.failure_patterns = {}  # Signature -> count
    
    def analyze_failure(self, task_features, failure_reason):
        """
        Categorize failure.
        Examples: "exponential_branching", "no_matching_primitives", etc.
        """
        pattern_sig = self._compute_signature(task_features, failure_reason)
        self.failure_patterns[pattern_sig] += 1
        
        # If pattern repeats > 3 times, trigger intervention
        if self.failure_patterns[pattern_sig] > 3:
            self._trigger_recovery(pattern_sig)
    
    def _trigger_recovery(self, pattern_sig):
        """
        Autonomously select recovery strategy.
        """
        if 'exponential_branching' in pattern_sig:
            # Reduce max_depth, sample fewer actions
            return {'action': 'reduce_branching', 'factor': 0.5}
        elif 'no_primitives' in pattern_sig:
            # Trigger DSL learning
            return {'action': 'learn_dsl'}
        elif 'timeout' in pattern_sig:
            # Increase time budget
            return {'action': 'increase_budget'}
        return None
```

### 2.4 Dynamic DSL Learning

**Goal**: Discover new primitives when existing DSL is insufficient.

```python
class DynamicDSLLearner:
    def __init__(self, base_dsl):
        self.dsl = base_dsl
        self.learned_primitives = []
        self.failed_task_buffer = []
    
    def observe_failure(self, task_examples, reason):
        """
        When solver fails, buffer the task.
        """
        if reason == 'no_matching_primitives':
            self.failed_task_buffer.append(task_examples)
    
    def synthesize_new_primitive(self, examples):
        """
        Given failed examples, synthesize a new primitive.
        E.g., if many tasks need 'rotate_90', learn that primitive.
        """
        # Analyze input/output pairs to infer missing operation
        # Could use meta-synthesis: synthesize a program that transforms inputs to outputs
        # This program becomes a new primitive
        
        inferred_program = self._infer_operation(examples)
        if inferred_program:
            self.learned_primitives.append(inferred_program)
            # Add to DSL
            self.dsl.add_primitive(inferred_program)
            return inferred_program
        return None
    
    def _infer_operation(self, examples):
        """Use subset of examples to infer hidden operation."""
        # This is itself a synthesis problem!
        # Could use heuristics: check for rotation, flip, permutation, etc.
        pass
```

---

## Part 3: Infinite Loop Dialog with Improvements

The improvements are designed as an **infinite refinement loop**:

```
Step 1: Attempt task with current MCTS + DSL
Step 2: Memory recalls similar past solutions (warm-start)
Step 3: Execute search
    ├─ Success: Store trajectory in episodic memory
    └─ Failure: Trigger meta-cognition
Step 4: Meta-cognition analyzes failure pattern
    └─ Identify recovery strategy
Step 5: Apply recovery strategy
    ├─ Reduce branching
    ├─ Trigger RSI (modify hyperparams)
    └─ Trigger DSL learning (discover new ops)
Step 6: Every N iterations, commit successful RSI modifications
Step 7: Return to Step 1 with improved state
```

**Key Property**: Each iteration improves the state slightly (memory grows, DSL expands, hyperparams refine).

---

## Part 4: Implementation Strategy

### Phase 1: Episodic Memory
```
aria/memory/  
├── episodic_memory.py       # Trajectory storage + retrieval
├── feature_extractor.py     # Task feature computation
└── memory_retrieval.py      # Cosine similarity-based recall
```

### Phase 2: Meta-Cognition
```
aria/meta_cognition/
├── failure_analyzer.py      # Pattern detection
├── recovery_strategies.py   # Recovery action dispatch
└── performance_monitor.py   # Track metrics over time
```

### Phase 3: RSI Engine
```
aria/rsi/
├── hyperparameter_learner.py  # Propose + test modifications
├── trend_analysis.py          # Detect performance plateaus
└── modification_history.py    # Track successful RSI steps
```

### Phase 4: DSL Learning
```
aria/dsl_learning/
├── primitive_synthesizer.py    # Infer missing operations
├── meta_synthesis_solver.py    # Synthesize operations via solving
└── dsl_expander.py            # Dynamically add to DSL
```

---

## Part 5: AGI Properties Achieved

| Property | ARIA v2.1 | ARIA v3.0 |
|----------|-----------|----------|
| Learns from past | ✗ | ✓ (episodic memory) |
| Improves its own code | ✗ | ✓ (RSI) |
| Detects own failures | ✗ | ✓ (meta-cognition) |
| Discovers new capabilities | ✗ | ✓ (DSL learning) |
| Infinite refinement loop | ✗ | ✓ |

---

## Part 6: Validation & Testing

### Test 1: Memory Acceleration
Solve 100 tasks from a domain. Compare:
- First 50 without memory
- Next 50 with memory

**Expected**: Memory recalls yield 30%+ faster solution times.

### Test 2: RSI Effectiveness
Run solver on task suite, apply RSI every 10 tasks.

**Expected**: Average score improves over iterations due to hyperparameter tuning.

### Test 3: Failure Recovery
Introduce tasks that violate DSL constraints (no matching primitives).

**Expected**: Meta-cognition detects pattern, DSL learning synthesizes missing primitive.

### Test 4: Infinite Loop Stability
Run loop for 1000 iterations on diverse task suite.

**Expected**: System remains stable, continuously improves, no catastrophic failures.

---

## Conclusion

ARIA v3.0 moves from passive static search toward **active self-improving systems**. The infinite loop ensures continuous refinement: learn from memory, detect failures, improve mechanisms, discover new capabilities. This is closer to AGI than v2.1's one-shot synthesis.
