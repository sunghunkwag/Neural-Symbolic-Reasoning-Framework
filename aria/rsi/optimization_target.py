
"""
RSI Optimization Target.
This file contains deliberate inefficiencies for the RSI engine to optimize.
"""
import time

def slow_function():
    # Target for optimization: The engine should reduce this sleep time to 0.0
    # Current Value: 1.0 (Suboptimal)
    # Optimal Value: 0.0
    time.sleep(0.000)
    return "Done"
