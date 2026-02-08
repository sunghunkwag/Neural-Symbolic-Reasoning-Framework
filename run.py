"""ARIA v2.1 - Entry Point."""
import argparse
import logging
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from aria.config import ARIAConfig, WorldModelConfig
from aria.core import ARIACore

def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def create_environment(env_name: str = "Pendulum-v1"):
    """Create gymnasium environment."""
    try:
        import gymnasium as gym
        env = gym.make(env_name)
        return env
    except ImportError:
        logging.error("gymnasium not installed. Run: pip install gymnasium")
        sys.exit(1)

def run_aria(
    episodes: int = 500,
    env_name: str = "Pendulum-v1",
    log_level: str = "INFO",
    seed: int = 42
):
    """
    Run ARIA v2.1 AGI Core.
    
    This is a PRODUCTION-GRADE implementation that:
    - Learns from REAL signals
    - Accumulates world-model errors forcing belief revision
    - Autonomously invents, evaluates, and pursues goals
    - Self-modifies to measurably improve performance
    """
    setup_logging(log_level)
    logger = logging.getLogger("ARIA.Run")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    logger.info(f"Creating environment: {env_name}")
    env = create_environment(env_name)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create config
    config = ARIAConfig(
        world_model=WorldModelConfig(
            obs_dim=obs_dim,
            action_dim=action_dim
        ),
        seed=seed,
        log_level=log_level
    )
    
    # Create ARIA Core
    logger.info("Initializing ARIA Core...")
    aria = ARIACore(config)
    
    # Run episodes
    logger.info(f"Starting {episodes} episodes...")
    start_time = time.time()
    
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # ARIA step - REAL learning happening here
            action = aria.step(obs, 0 if step == 0 else reward, done, info)
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        # Final step with done=True
        aria.step(obs, reward, True, info)
        
        # Logging
        if (episode + 1) % 10 == 0:
            metrics = aria.get_metrics()
            logger.info(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Goals: {metrics['goals_generated']} gen, {metrics['goals_rejected']} rej | "
                f"Beliefs Rev: {metrics['beliefs_revised']} | "
                f"Self-Mod: {metrics['self_mod_proposals']} prop, {metrics['self_mod_commits']} commit"
            )
        
        # Early success check
        if (episode + 1) % 50 == 0:
            criteria = aria.validate_success_criteria()
            if criteria["all_passed"]:
                logger.info("SUCCESS: All criteria met!")
                break
    
    # Final results
    elapsed = time.time() - start_time
    metrics = aria.get_metrics()
    criteria = aria.validate_success_criteria()
    
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Episodes: {metrics['episode']}")
    logger.info(f"Total Steps: {metrics['step']}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info("-" * 60)
    logger.info(f"Goals Generated: {metrics['goals_generated']}")
    logger.info(f"Goals Rejected: {metrics['goals_rejected']}")
    logger.info(f"Beliefs Revised: {metrics['beliefs_revised']}")
    logger.info(f"Self-Mod Proposals: {metrics['self_mod_proposals']}")
    logger.info(f"Self-Mod Commits: {metrics['self_mod_commits']}")
    logger.info(f"Baseline Performance: {metrics['baseline_performance']:.3f}")
    logger.info(f"Current Performance: {metrics['current_performance']:.3f}")
    logger.info(f"Improvement: {metrics['improvement']:.3f}")
    logger.info("-" * 60)
    logger.info("SUCCESS CRITERIA:")
    for criterion, passed in criteria.items():
        if criterion != "all_passed":
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {criterion}")
    logger.info("-" * 60)
    
    if criteria["all_passed"]:
        logger.info("🎉 ALL CRITERIA PASSED - IMPLEMENTATION VALID")
    else:
        failed = [k for k, v in criteria.items() if not v and k != "all_passed"]
        logger.warning(f"❌ FAILED CRITERIA: {failed}")
    
    # Save results
    results = {
        "metrics": metrics,
        "criteria": criteria,
        "elapsed_seconds": elapsed
    }
    
    results_path = Path(__file__).parent / "aria_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    env.close()
    return criteria["all_passed"]

def main():
    parser = argparse.ArgumentParser(description="ARIA v2.1 AGI Core")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    success = run_aria(
        episodes=args.episodes,
        env_name=args.env,
        log_level=args.log_level,
        seed=args.seed
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
