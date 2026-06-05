"""
Plotting Utility Module
Generates a static reward curve plot from TensorBoard event logs.
Handles multiple event files (e.g. from resumed training runs) by merging
all scalars by step number.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_reward_curve(log_dir, output_path):
    """
    Extracts scalar data from all TensorBoard event files in the log directory
    and plots a combined reward curve.
    """
    print(f"Searching for logs in {log_dir}...")
    
    # Find all event files (SB3 creates new ones on resume / mid-training)
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(root, f))
    
    if not event_files:
        print("No TensorBoard events found.")
        return

    print(f"Found {len(event_files)} event file(s).")

    # Stable Baselines3 logs 'rollout/ep_rew_mean' as the primary metric
    target_tag = 'rollout/ep_rew_mean'

    # Merge scalars from all event files by step number
    all_steps = []
    all_values = []
    resolved_tag = None

    for ef in sorted(event_files, key=os.path.getmtime):
        event_acc = EventAccumulator(ef)
        event_acc.Reload()

        tags = event_acc.Tags().get('scalars', [])
        if not tags:
            continue

        # Determine which tag to plot
        if resolved_tag is None:
            if target_tag in tags:
                resolved_tag = target_tag
            else:
                resolved_tag = tags[0]
                print(f"'{target_tag}' not found. Falling back to '{resolved_tag}'.")

        if resolved_tag not in tags:
            continue

        scalar_events = event_acc.Scalars(resolved_tag)
        all_steps.extend([e.step for e in scalar_events])
        all_values.extend([e.value for e in scalar_events])

    if not all_steps:
        print("No scalar data found across event files.")
        return

    # Sort by step (events from multiple files may interleave)
    sorted_pairs = sorted(zip(all_steps, all_values))
    steps = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    print(f"Plotting {len(steps)} data points for '{resolved_tag}'.")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=resolved_tag)
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
