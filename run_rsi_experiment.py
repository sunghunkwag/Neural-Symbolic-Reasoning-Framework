
import logging
from aria.logic.synthesizer import LogicSynthesizer
from aria.rsi import optimization_target

def run_experiment():
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RSI_Experiment")
    
    logger.info("=== RSI Micro-Experiment: Starting ===")
    logger.info("Goal: Optimize 'optimization_target.py' to remove inefficiency.")
    
    # Initialize Engine
    synthesizer = LogicSynthesizer()
    
    # Path to target
    target_path = optimization_target.__file__
    logger.info(f"Target File: {target_path}")
    
    # Run Optimization
    # This will read the file, evolve the sleep parameter, and rewrite it.
    success = synthesizer.optimize_file(target_path, target_function="slow_function")
    
    if success:
        logger.info("=== RSI Experiment: SUCCESS ===")
        logger.info("The engine has successfully modified its own codebase.")
        logger.info("Check 'optimization_target.py' for changes.")
    else:
        logger.error("=== RSI Experiment: FAILED ===")

if __name__ == "__main__":
    run_experiment()
