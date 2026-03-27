"""
Plotting Utility Module
Generates a static reward curve plot from TensorBoard event logs.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_reward_curve(log_dir, output_path):
    """
    Extracts scalar data from TensorBoard logs and plots it.
    """
    print(f"Searching for logs in {log_dir}...")
    
    # Find active event files
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(root, f))
    
    if not event_files:
        print("No TensorBoard events found.")
        return

    # Use the most recent event file
    latest_event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading {latest_event_file}...")

    event_acc = EventAccumulator(latest_event_file)
    event_acc.Reload()

    # Stable Baselines3 logs 'rollout/ep_rew_mean' or 'train/ent_loss' etc.
    tags = event_acc.Tags().get('scalars', [])
    
    # Target tag for SB3
    target_tag = 'rollout/ep_rew_mean'
    if target_tag not in tags:
        # Fallback to any scalar if target not found
        if tags:
            fallback = tags[0]
            print(f"'{target_tag}' not found. Falling back to '{fallback}'.")
            target_tag = fallback
        else:
            print("No scalar tags found in logs.")
            return

    scalar_events = event_acc.Scalars(target_tag)
    
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=target_tag)
    plt.title("Mario Training Progress")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario Metric Plotter")
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs", help="Directory containing TensorBoard logs.")
    parser.add_argument("--output_path", type=str, default="logs/reward_curve.png", help="Path to save the output plot.")
    
    args = parser.parse_args()
    plot_reward_curve(args.log_dir, args.output_path)
