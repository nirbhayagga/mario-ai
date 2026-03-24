"""
Validation Script
This file explicitly initializes the virtual emulator namespace to systematically verify that the nes-py environment wrappers construct without throwing C++ backend segmentation allocation errors.
"""

import argparse
import sys
from env_setup import create_mario_environment

def validate_environment_initialization():
    """
    Executes a formal instantiation of the wrapper matrix and forces exactly one operational step to verify rendering stability natively.
    """
    parser = argparse.ArgumentParser(description="Mario Environment Validation")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0", help="The Gym environment ID.")
    args = parser.parse_args()
    
    # Detect if we are using multi-level training (comma-separated list)
    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    print(f"Initiating formal environment validation sequence for: {args.env_id}...")
    
    try:
        # Construct the deeply wrapped environment precisely as the training script will
        environment = create_mario_environment(env_id=target_envs)
        
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
