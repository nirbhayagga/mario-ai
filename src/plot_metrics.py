"""
Plotting Utility Module
Generates a training-progress figure from TensorBoard event logs: the shaped
training episode reward (SB3's rolling mean over the last 100 episodes) and
the unshaped evaluation reward recorded by EvalCallback. Multiple event files
(e.g. from resumed training runs) are merged by step number.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

TRAIN_REWARD_TAG = "rollout/ep_rew_mean"
FLAG_RATE_TAG = "rollout/flag_rate"
MAX_X_TAG = "rollout/max_x_mean"
EVAL_REWARD_TAG = "eval/mean_reward"

# (tag, title, y-label, y-limits or None); every tag present in the log gets a panel.
PANELS = [
    (FLAG_RATE_TAG, "Level clear rate during training (rolling mean of last 100 episodes)", "Clear rate", (0, 1)),
    (MAX_X_TAG, "Furthest x-position reached per episode (rolling mean of last 100 episodes)", "x position", None),
    (TRAIN_REWARD_TAG, "Training episode reward (shaped, rolling mean of last 100 episodes)", "Mean reward", None),
    (EVAL_REWARD_TAG, "Evaluation reward (raw game reward, deterministic policy)", "Mean reward", None),
]
REQUIRED_ANY = (TRAIN_REWARD_TAG, EVAL_REWARD_TAG)


def find_event_files(log_dir: str) -> list[str]:
    """SB3 creates a new event file per learn() call, so resumed runs have several."""
    event_files = []
    for root, _, files in os.walk(log_dir):
        event_files.extend(os.path.join(root, f) for f in files if "tfevents" in f)
    return sorted(event_files, key=os.path.getmtime)


def load_scalars(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Returns {tag: [(step, value), ...]} merged across all event files, sorted by step."""
    scalars = {}
    for event_file in find_event_files(log_dir):
        accumulator = EventAccumulator(event_file)
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            scalars.setdefault(tag, []).extend((e.step, e.value) for e in accumulator.Scalars(tag))
    return {tag: sorted(points) for tag, points in scalars.items()}


def plot_training_curves(log_dir: str, output_path: str) -> bool:
    """Plots one panel per available reward tag. Returns False if neither exists."""
    print(f"Searching for logs in {log_dir}...")
    scalars = load_scalars(log_dir)
    panels = [panel for panel in PANELS if panel[0] in scalars]
    if not any(tag in scalars for tag in REQUIRED_ANY):
        available = ", ".join(sorted(scalars)) or "none"
        print(f"No reward curves found. Expected {TRAIN_REWARD_TAG} or {EVAL_REWARD_TAG}; available tags: {available}")
        return False

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), sharex=True, squeeze=False)
    for ax, (tag, title, ylabel, ylim) in zip(axes[:, 0], panels):
        steps, values = zip(*scalars[tag])
        marker = "o" if len(steps) < 50 else None
        ax.plot(steps, values, marker=marker, markersize=3, label=tag)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="lower right")
        print(f"Plotting {len(steps)} data points for '{tag}'.")
    axes[-1, 0].set_xlabel("Timesteps")
    fig.suptitle("Mario PPO Training Progress")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario Metric Plotter")
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="Directory containing TensorBoard logs.")
    parser.add_argument("--output_path", type=str, default="logs/reward_curve.png",
                        help="Path to save the output plot.")
    args = parser.parse_args()
    sys.exit(0 if plot_training_curves(args.log_dir, args.output_path) else 1)
