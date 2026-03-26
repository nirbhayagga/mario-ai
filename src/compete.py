"""
AI vs Human Competition Module
Runs a human play session followed by an AI evaluation for comparison.
"""

import os
import time
import argparse


def execute_competition_mode():
    parser = argparse.ArgumentParser(description="Mario Human vs AI Competition")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model weights.")
    args = parser.parse_args()

    print(f"Competition mode: {args.env_id}")
    print("PLAYER 1 (HUMAN): Get ready!")
    time.sleep(2)
    os.system(f"python src/manual_play.py --env_id {args.env_id}")

    print("PLAYER 2 (AI): Running evaluation...")
    time.sleep(2)
    eval_cmd = f"python src/evaluate.py --env_id {args.env_id}"
    if args.model_path:
        eval_cmd += f" --model_path {args.model_path}"
    os.system(eval_cmd)

    print("Competition complete.")


if __name__ == "__main__":
    execute_competition_mode()
