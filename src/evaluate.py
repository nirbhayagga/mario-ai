"""
Evaluation and Video Export Module
Runs the trained policy on every level of the run. Per level, --episodes
episodes with seeded random no-op starts feed a clear-rate summary, then the
best of them (clears first, then furthest x) is replayed with the same seed
and recorded as .mp4 and .gif with a metadata overlay. Recording the best
episode instead of an arbitrary one matters because a deterministic policy
can have one degenerate trajectory (an early death) from a specific start
while playing well from every other. A single environment serves both
inference and rendering so the overlay is always in sync with what the
model saw.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

import argparse
import json
import os
import random
import re

import cv2
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv

from env_setup import create_mario_environment
from media import GifRecorder

# One frame is rendered per agent step, and each step is 4 NES frames at 60 Hz,
# so writing 15 frames per second plays the game back in real time.
REAL_TIME_FPS = 15.0
GIF_TARGET_FPS = 15.0


def video_fps(speed: float) -> float:
    """Frames per second that play the recording back at `speed` x real time."""
    return REAL_TIME_FPS * speed


def gif_stride(speed: float) -> int:
    """Keeps the GIF near GIF_TARGET_FPS by dropping frames as playback speed rises."""
    return max(1, int(round(video_fps(speed) / GIF_TARGET_FPS)))


def parse_evaluation_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO Mario Evaluation")
    parser.add_argument("--model_path", type=str, default="mario_ppo_final.zip",
                        help="Path to the serialized model weights.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Hardware device for inference.")
    parser.add_argument("--video_output", type=str, default="evaluation_run.mp4",
                        help="Output path for the evaluation video. A .gif is written alongside it.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode (stops earlier on level clear or death).")
    parser.add_argument("--video_speed", type=float, default=2.0,
                        help="Playback speed of the recording relative to real time (1.0 = real time).")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodes to run for clear-rate statistics, each from a seeded random "
                             "no-op start. Only the best one is replayed and recorded.")
    parser.add_argument("--action_space", type=str, default="simple", choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for the episode starts, making the whole evaluation reproducible. "
                             "A random base seed is drawn (and reported) if omitted.")
    return parser.parse_args()


def run_episode(vec_env: VecEnv, model: PPO, max_steps: int,
                on_step: Callable[[dict[str, Any], int, float], None] | None = None) -> dict[str, Any]:
    """
    Plays one episode with the deterministic policy and returns its outcome.
    Ends on flag capture, on death, or after max_steps. If given, on_step is
    called after every step with (info, action, total_reward).
    """
    obs = vec_env.reset()
    total_reward = 0.0
    max_x = 0
    cleared = False
    steps = 0

    while steps < max_steps:
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        max_x = max(max_x, int(info[0].get("x_pos", 0)))

        if on_step is not None:
            on_step(info[0], int(action[0]), total_reward)

        if info[0].get("flag_get", False):
            cleared = True
            break
        # Stop on first death - prevents misleading score accumulation across resets
        if done[0]:
            break

    return {"cleared": cleared, "reward": total_reward, "steps": steps, "max_x": max_x}


def summarize_episodes(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates run_episode() outcomes into clear-rate and progress statistics."""
    n = len(results)
    return {
        "episodes": n,
        "clear_rate": sum(r["cleared"] for r in results) / n,
        "mean_reward": sum(r["reward"] for r in results) / n,
        "mean_max_x": sum(r["max_x"] for r in results) / n,
        "best_max_x": max(r["max_x"] for r in results),
        "mean_steps": sum(r["steps"] for r in results) / n,
    }


def draw_overlay(bgr_frame: np.ndarray, x_pos: int, score: int, action_name: str, total_reward: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr_frame, f"X: {x_pos}", (5, 20), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Score: {score}", (5, 35), font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Action: {action_name}", (5, 50), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Reward: {total_reward:.0f}", (5, 65), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def level_tag(env_id: str) -> str:
    """SuperMarioBros-1-2-v0 -> 1-2 (falls back to the full ID)."""
    match = re.search(r"-(\d+-\d+)-v\d+$", env_id)
    return match.group(1) if match else env_id


def output_paths(video_output: str, env_id: str, multi_level: bool) -> tuple[str, str]:
    """Returns (mp4 path, gif path); multi-level runs get a per-level suffix."""
    stem, ext = os.path.splitext(video_output)
    if multi_level:
        stem = f"{stem}_{level_tag(env_id)}"
    return stem + (ext or ".mp4"), stem + ".gif"


def best_episode_index(results: list[dict[str, Any]]) -> int:
    """The episode worth recording: clears beat non-clears, then furthest x,
    then reward (which, between two clears, rewards the faster one)."""
    return max(range(len(results)),
               key=lambda i: (results[i]["cleared"], results[i]["max_x"], results[i]["reward"]))


def evaluate_level(model: PPO, env_id: str, args: argparse.Namespace, movement: list[list[str]],
                   video_path: str, gif_path: str) -> tuple[list[dict[str, Any]], int, int]:
    """Runs the clear-rate episodes headless with per-episode seeds, then
    replays the best one with its seed and records it. Returns
    (episode outcomes, base seed, index of the recorded episode)."""

    base_seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    print(f"\n[{env_id}] Running {args.episodes} episodes (base seed {base_seed})")

    sample_env = create_mario_environment(env_id=env_id, action_space=args.action_space, use_noop=True)
    results = []
    for i in range(args.episodes):
        sample_env.seed(base_seed + i)
        results.append(run_episode(sample_env, model, args.max_steps))
        print_episode(i + 1, results[-1])
    sample_env.close()

    best = best_episode_index(results)
    fps = video_fps(args.video_speed)
    print(f"  Recording episode {best + 1} (best of {args.episodes}) to {video_path} "
          f"({args.video_speed:g}x speed, {fps:g} fps)")

    # Same seed + deterministic policy = the identical trajectory, this time
    # recorded. n_envs=1 -> DummyVecEnv, whose .envs[0] exposes the wrapped gym
    # env; render() passes through the observation wrappers to the base NES
    # env for a clean 256x240 RGB frame.
    record_env = create_mario_environment(env_id=env_id, action_space=args.action_space, use_noop=True)
    record_env.seed(base_seed + best)
    inner_env = record_env.venv.envs[0]

    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (256, 240))
    gif = GifRecorder(gif_path, fps=fps, every=gif_stride(args.video_speed))

    def record(info: dict[str, Any], action_int: int, total_reward: float) -> None:
        bgr_frame = cv2.cvtColor(inner_env.render(mode="rgb_array"), cv2.COLOR_RGB2BGR)
        action_name = str(movement[action_int]) if action_int < len(movement) else "UNKNOWN"
        draw_overlay(bgr_frame, int(info.get("x_pos", 0)), int(info.get("score", 0)), action_name, total_reward)
        video_writer.write(bgr_frame)
        gif.add(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

    recorded = run_episode(record_env, model, args.max_steps, on_step=record)
    video_writer.release()
    gif.save()
    record_env.close()
    if recorded != results[best]:
        print(f"  Note: the recorded replay diverged from the sampled episode "
              f"(max X {recorded['max_x']} vs {results[best]['max_x']}).")

    return results, base_seed, best


def print_episode(index: int, result: dict[str, Any]) -> None:
    outcome = "CLEARED" if result["cleared"] else "DIED"
    print(f"  Episode {index} [{outcome}]: reward {result['reward']:.0f}, "
          f"{result['steps']} steps, max X {result['max_x']}")


def format_summary_table(per_level: dict[str, dict[str, Any]]) -> str:
    """Plain-text table of per-level statistics, one row per level."""
    lines = [f"{'Level':<24}{'Clear rate':>11}{'Mean reward':>13}{'Mean max X':>12}{'Best X':>8}"]
    for env_id, stats in per_level.items():
        lines.append(f"{level_tag(env_id):<24}{stats['clear_rate']:>11.0%}{stats['mean_reward']:>13.0f}"
                     f"{stats['mean_max_x']:>12.0f}{stats['best_max_x']:>8}")
    return "\n".join(lines)


def format_markdown_table(per_level: dict[str, dict[str, Any]]) -> str:
    """The same statistics as a GitHub-flavoured markdown table, ready for the README."""
    lines = ["| Level | Clear rate | Mean game reward | Mean max X | Best X |",
             "| :--- | :--- | :--- | :--- | :--- |"]
    for env_id, stats in per_level.items():
        n = stats["episodes"]
        clears = round(stats["clear_rate"] * n)
        lines.append(f"| {level_tag(env_id)} | {stats['clear_rate']:.0%} ({clears}/{n}) | {stats['mean_reward']:.0f} "
                     f"| {stats['mean_max_x']:.0f} | {stats['best_max_x']} |")
    return "\n".join(lines)


def export_evaluation_video() -> None:
    args = parse_evaluation_arguments()
    levels = [e.strip() for e in args.env_id.split(",")]

    if args.action_space == "complex":
        movement = COMPLEX_MOVEMENT
    elif args.action_space == "right_only":
        movement = RIGHT_ONLY
    else:
        movement = SIMPLE_MOVEMENT

    model = PPO.load(args.model_path, device=args.device)
    if args.seed is not None:
        set_random_seed(args.seed)

    per_level = {}
    for env_id in levels:
        video_path, gif_path = output_paths(args.video_output, env_id, multi_level=len(levels) > 1)
        results, base_seed, best = evaluate_level(model, env_id, args, movement, video_path, gif_path)
        per_level[env_id] = {
            **summarize_episodes(results),
            "base_seed": base_seed,
            "recorded_episode": best + 1,
            "video": video_path,
            "gif": gif_path,
            "episodes_detail": results,
        }

    summary_path = os.path.join(os.path.dirname(args.video_output) or ".", "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model_path": args.model_path, "episodes_per_level": args.episodes, "levels": per_level},
                  f, indent=2)

    print(f"\nEvaluation summary ({args.episodes} episodes per level):")
    print(format_summary_table(per_level))
    print("\nMarkdown:")
    print(format_markdown_table(per_level))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    export_evaluation_video()
