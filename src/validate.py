"""
Validation Script
Verifies that the environment wrappers initialize correctly and can complete
at least one step without crashing.
"""

import argparse
import sys
from env_setup import create_mario_environment


def validate_environment():
    parser = argparse.ArgumentParser(description="Mario Environment Validation")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    args = parser.parse_args()

    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    print(f"Validating environment: {args.env_id}...")

    try:
        env = create_mario_environment(env_id=target_envs)
        observation = env.reset()
        observation, reward, done, info = env.step([0])
        env.close()
        print("Validation passed. Environment initialized and stepped successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    validate_environment()
