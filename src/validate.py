"""
Validation Script
Verifies that the environment wrappers initialize correctly and can complete
at least one step without crashing. Uses the same configuration that training
will use (action space, custom reward) so it catches config-specific bugs.
"""

import argparse
import sys
from env_setup import create_mario_environment


def validate_environment():
    parser = argparse.ArgumentParser(description="Mario Environment Validation")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    parser.add_argument("--action_space", type=str, default="simple",
                        choices=["simple", "complex", "right_only"],
                        help="Action space to validate with.")
    parser.add_argument("--no_custom_reward", action="store_true",
                        help="Validate without custom reward wrappers.")
    args = parser.parse_args()

    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    use_custom_reward = not args.no_custom_reward
    print(f"Validating environment: {args.env_id}")
    print(f"  action_space={args.action_space}, custom_reward={use_custom_reward}")

    try:
        env = create_mario_environment(
            env_id=target_envs,
            action_space=args.action_space,
            use_custom_reward=use_custom_reward,
        )
        observation = env.reset()
        # One action per env: a short list makes SubprocVecEnv block forever.
        observation, reward, done, info = env.step([0] * env.num_envs)
        env.close()
        print("Validation passed. Environment initialized and stepped successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    validate_environment()
