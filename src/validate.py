"""
Validation Script
This file explicitly initializes the virtual emulator namespace to systematically verify that the nes-py environment wrappers construct without throwing C++ backend segmentation allocation errors.
"""

import sys
from env_setup import create_mario_environment

def validate_environment_initialization():
    """
    Executes a formal instantiation of the wrapper matrix and forces exactly one operational step to verify rendering stability natively.
    """
    print("Initiating formal environment validation sequence specifically to check emulation backend...")
    
    try:
        # Construct the deeply wrapped environment precisely as the training script will
        environment = create_mario_environment()
        
        # Reset the environment explicitly to force ROM memory loading
        observation = environment.reset()
        
        # Construct a synthetic zero array action mapping effectively standing still
        action = [0]
        
        # Execute exactly one step formally to verify physics and memory manipulation algorithms
        observation, reward, done, info = environment.step(action)
        
        print("Validation explicitly successful. The environment instantiated and completed a mathematical step perfectly without crashing.")
        sys.exit(0)
    except Exception as general_exception:
        print(f"Validation formally failed. Explicit exception caught mapping error: {general_exception}")
        sys.exit(1)

if __name__ == "__main__":
    validate_environment_initialization()
