"""
AI vs Human Competition Module
Runs a human play session followed by an AI evaluation for comparison.
"""

import subprocess
import sys
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
    subprocess.run(
        [sys.executable, "src/manual_play.py", "--env_id", args.env_id],
        check=True,
    )

    print("PLAYER 2 (AI): Running evaluation...")
    time.sleep(2)
    eval_cmd = [sys.executable, "src/evaluate.py", "--env_id", args.env_id]
    if args.model_path:
        eval_cmd += ["--model_path", args.model_path]
    subprocess.run(eval_cmd, check=True)

    print("Competition complete.")


if __name__ == "__main__":
    execute_competition_mode()
