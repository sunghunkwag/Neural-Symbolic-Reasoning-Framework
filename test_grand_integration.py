
import torch
import numpy as np
import logging
import time
from aria.core import ARIACore
from aria.config import ARIAConfig, GoalConfig
from aria.rsi import optimization_target

def test_failure_driven_evolution():
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IntegrationTest")
    logger.info("=== Phase 19: Grand Integration Test Starting ===")

    # 1. Configuration
    config = ARIAConfig()
    # Set low patience for fast verification
    config.goal.rsi_patience = 5
    config.goal.rsi_trigger_threshold = 0.5 
    
    # 2. Initialize ARIA Core
    core = ARIACore(config)
    
    # 3. Simulate Failure Loop
    # We will feed high novelty scores (prediction error) to trigger the RSI channel
    logger.info("Simulating High Prediction Error to trigger RSI Channel...")
    
    obs = np.zeros(config.world_model.obs_dim)
    
    for step in range(10):
        # Fake a high novelty score (prediction error)
        # Note: In a real run, this comes from the world model
        # For this test, we override/mock the novelty passed to goal_generator.step
        
        # Triggering the step with high novelty (error = 2.0)
        # We need to simulate the step logic but force high error
        
        # Normally: novelty = self.world_model.get_novelty_score()
        # Here we manually drive the generator inside the loop or mock the core behavior
        
        # Let's bypass the world model's internal score and inject directly into goal generator
        # to ensure the RSI trigger happens predictably.
        
        latent_state = torch.zeros(config.world_model.latent_dim)
        novelty = 2.0 # Critical error
        
        logger.info(f"Step {step}: Error = {novelty:.2f}")
        
        # This will be called inside core.step() normally.
        # Here we trigger the core's goal generation logic.
        
        # Simulate Core Step manually to control novelty
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        z_t = core.world_model.encode(obs_t)
        
        # Inject Goal Generation
        generated_goal = core.goal_generator.step(
            z_t, 
            novelty,
            episode_failed=True, # Signal failure
            reward=-1.0
        )
        
        if generated_goal:
            logger.info(f"Goal Generated: {generated_goal.channel} - {generated_goal.description}")
            if generated_goal.channel.name == "SELF_MODIFICATION":
                logger.info("RSI Goal Detected! Processing in Core...")
                core._handle_rsi_goal(generated_goal)
                break
        
        time.sleep(0.1)

    # 4. Final Verification
    logger.info("Checking if 'optimization_target.py' was modified...")
    with open(optimization_target.__file__, 'r') as f:
        content = f.read()
    
    if "sleep(0.000)" in content:
        logger.info("=== INTEGRATION SUCCESS: Failure-Driven Evolution Loop Verified! ===")
    else:
        logger.error("=== INTEGRATION FAILED: Code was not modified. ===")

if __name__ == "__main__":
    test_failure_driven_evolution()
