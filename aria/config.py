"""ARIA v2.1 Configuration - All hyperparameters and settings."""
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class WorldModelConfig:
    obs_dim: int = 4
    action_dim: int = 1
    latent_dim: int = 64
    ssm_hidden_dim: int = 128
    prediction_horizon: int = 10
    encoder_hidden: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-3
    novelty_ema_alpha: float = 0.01
    novelty_threshold: float = 0.8

@dataclass
class GoalConfig:
    compression_threshold: float = 0.8
    failure_pattern_min_count: int = 3
    anomaly_distance_threshold: float = 0.9
    legitimacy_threshold: float = 0.6
    rollout_count: int = 100
    adversarial_probe_count: int = 10
    rsi_trigger_threshold: float = 1.5  # High error threshold for RSI trigger
    rsi_patience: int = 20  # Steps of sustained error before RSI

@dataclass
class PlannerConfig:
    mcts_iterations: int = 50
    mcts_depth: int = 10
    ucb_c: float = 1.414
    budget_time: float = 1.0
    budget_compute: int = 1000
    budget_risk: float = 0.5

@dataclass
class PolicyConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64

@dataclass
class SelfModConfig:
    proposal_interval: int = 50  # More frequent proposals
    eval_episodes: int = 2  # Reduced for faster evaluation (was 5)
    improvement_threshold: float = 0.02  # Lower threshold (2%)
    modifiable_params: List[str] = field(default_factory=lambda: [
        "world_model.learning_rate",
        "planner.mcts_depth",
        "goal.novelty_threshold",
        "policy.learning_rate"
    ])

@dataclass
class ARIAConfig:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    self_mod: SelfModConfig = field(default_factory=SelfModConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: str = "INFO"
    seed: int = 42
