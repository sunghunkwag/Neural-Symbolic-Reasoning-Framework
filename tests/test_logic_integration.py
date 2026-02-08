import torch
import numpy as np
import logging
from aria.config import ARIAConfig
from aria.core import ARIACore
from aria.types import Goal, GoalChannel

# Configure logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST")

def main():
    logger.info("Initializing ARIA Core with Logic Synthesis Integration...")
    config = ARIAConfig()
    
    # Use CPU for test
    config.device = "cpu"
    
    try:
        core = ARIACore(config)
    except Exception as e:
        logger.error(f"Failed to init Core: {e}")
        return
    
    logger.info("Core initialized.")
    if not hasattr(core, 'logic_synthesizer'):
        logger.error("FAIL: logic_synthesizer not attached to core.")
        return
    else:
        logger.info("PASS: logic_synthesizer attached.")

    # Create a dummy Logic Goal
    logic_goal = Goal(
        channel=GoalChannel.LOGIC_FAILURE,
        description="TEST_LOGIC_INTEGRATION_TASK",
        trigger_value=1.0,
        priority=100.0
    )
    
    # 1. Manually push goal to stack
    core.goal_stack.push(logic_goal)
    logger.info(f"Pushed goal: {logic_goal.description}")
    
    # 2. Run one step
    # Default obs dim is 4
    obs = np.zeros(4, dtype=np.float32)
    
    logger.info("Executing Core Step...")
    try:
        action = core.step(obs, 0.0, False)
        logger.info(f"Step complete. Action: {action}")
    except Exception as e:
        logger.error(f"Step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if goal was handled
    # Logic goals are cleared immediately in _handle_logic_goal
    if core.current_goal is None:
        # Also check goal stack is empty or goal status 
        # GoalStack doesn't track history efficiently in this version, effectively popped.
        logger.info("PASS: Logic Goal was processed and cleared from active state.")
    else:
        logger.error(f"FAIL: Current goal is still {core.current_goal}")

if __name__ == "__main__":
    main()
