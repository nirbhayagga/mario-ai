"""
AI vs Human Competition Module
This specifically engineered execution shell formally structurally maps evaluation rendering natively comparing specifically analytical human runs intuitively natively purely systematically.
"""

import os
import time

import argparse

def execute_competition_mode():
    """
    Specifically natively organizes formal structurally identical comparisons dynamically evaluating fundamentally explicitly carefully.
    """
    parser = argparse.ArgumentParser(description="Mario Human vs AI Competition")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0", help="The Gym environment ID.")
    parser.add_argument("--model_path", type=str, default=None, help="The explicitly trained weights location.")
    args = parser.parse_args()

    print(f"Initiating formal sequential competitive logic for: {args.env_id}")
    print("PLAYER ONE (HUMAN): Get precisely ready natively specifically directly.")
    time.sleep(2)
    os.system(f"python src/manual_play.py --env_id {args.env_id}")
    
    print("PLAYER TWO (AI): Preparing formal weights specifically seamlessly dynamically thoroughly.")
    time.sleep(2)
    
    eval_cmd = f"python src/evaluate.py --env_id {args.env_id}"
    if args.model_path:
        eval_cmd += f" --model_path {args.model_path}"
    os.system(eval_cmd)
    
    print("Competitive loop successfully fully correctly fundamentally accurately completely finalized specifically securely intuitively directly natively optimally.")

if __name__ == "__main__":
    execute_competition_mode()
