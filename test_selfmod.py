"""Quick diagnostic test for self-modification - extended run."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import gymnasium as gym
from aria.config import ARIAConfig, WorldModelConfig
from aria.core import ARIACore
import numpy as np

env = gym.make('Pendulum-v1')
config = ARIAConfig(world_model=WorldModelConfig(obs_dim=3, action_dim=1))
aria = ARIACore(config)
obs, _ = env.reset(seed=42)
prev_done = False
prev_reward = 0.0
episode_count = 0

print('Starting extended loop (1500 steps)...')
for i in range(1500):
    # IMPORTANT: Pass done from PREVIOUS step to aria.step
    # This correctly signals when an episode ended
    a = aria.step(obs, prev_reward, prev_done)
    
    # Get new state from environment
    obs, reward, term, trunc, info = env.step(a)
    done = term or trunc
    
    # Track episode completion
    if done:
        episode_count += 1
        obs, _ = env.reset()
    
    # Store for next iteration (done will signal to aria that episode ended)
    prev_done = done
    prev_reward = reward
    
    if i % 100 == 0:
        m = aria.get_metrics()
        print(f"Step {i}: episodes={episode_count}, proposals={m['self_mod_proposals']}, commits={m['self_mod_commits']}, beliefs={m['beliefs_revised']}")

print('='*60)
print('FINAL METRICS:')
m = aria.get_metrics()
for k, v in m.items():
    print(f"  {k}: {v}")
print(f"  total_episodes: {episode_count}")

# Check success criteria
criteria = aria.validate_success_criteria()
print('='*60)
print('SUCCESS CRITERIA:')
for k, v in criteria.items():
    status = "PASS" if v else "FAIL"
    print(f"  {k}: {status}")
