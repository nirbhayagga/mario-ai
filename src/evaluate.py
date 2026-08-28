"""
Evaluation and Video Export Module
Runs the trained policy on every level of the run. Per level, the first
episode is recorded as .mp4 and .gif with a metadata overlay and the rest
feed a clear-rate summary; the per-level table is printed and written to
evaluation_summary.json. A single environment serves both inference and
rendering so the overlay is always in sync with what the model saw.
"""

import argparse
import json
import os
import re

import cv2
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from env_setup import create_mario_environment
from media import GifRecorder

GIF_FRAME_STRIDE = 2  # every 2nd frame at 30 fps -> 15 fps GIF, ~half the size


def parse_evaluation_arguments():
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
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodes to run for clear-rate statistics. The first is recorded; "
                             "the rest use random no-op starts and are not rendered.")
    parser.add_argument("--action_space", type=str, default="simple", choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random no-op starts of the extra episodes. Unseeded if omitted.")
    return parser.parse_args()


def run_episode(vec_env, model, max_steps, on_step=None):
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


def summarize_episodes(results):
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


def draw_overlay(bgr_frame, x_pos, score, action_name, total_reward):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr_frame, f"X: {x_pos}", (5, 20), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Score: {score}", (5, 35), font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Action: {action_name}", (5, 50), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, f"Reward: {total_reward:.0f}", (5, 65), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def level_tag(env_id):
    """SuperMarioBros-1-2-v0 -> 1-2 (falls back to the full ID)."""
    match = re.search(r"-(\d+-\d+)-v\d+$", env_id)
    return match.group(1) if match else env_id


def output_paths(video_output, env_id, multi_level):
    """Returns (mp4 path, gif path); multi-level runs get a per-level suffix."""
    stem, ext = os.path.splitext(video_output)
    if multi_level:
        stem = f"{stem}_{level_tag(env_id)}"
    return stem + (ext or ".mp4"), stem + ".gif"


def evaluate_level(model, env_id, args, movement, video_path, gif_path):
    """Records episode 1 on a deterministic env, then samples the remaining
    episodes with random no-op starts. Returns the list of episode outcomes."""

    # n_envs=1 -> DummyVecEnv, whose .envs[0] exposes the wrapped gym env;
    # render() passes through the observation wrappers to the base NES env
    # for a clean 256x240 RGB frame.
    vec_env = create_mario_environment(env_id=env_id, action_space=args.action_space)
    inner_env = vec_env.venv.envs[0]

    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (256, 240))
    gif = GifRecorder(gif_path, fps=30.0, every=GIF_FRAME_STRIDE)

    def record(info, action_int, total_reward):
        bgr_frame = cv2.cvtColor(inner_env.render(mode="rgb_array"), cv2.COLOR_RGB2BGR)
        action_name = str(movement[action_int]) if action_int < len(movement) else "UNKNOWN"
        draw_overlay(bgr_frame, int(info.get("x_pos", 0)), int(info.get("score", 0)), action_name, total_reward)
        video_writer.write(bgr_frame)
        gif.add(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

    print(f"\n[{env_id}] Recording episode 1 to {video_path} (max {args.max_steps} steps)")
    results = [run_episode(vec_env, model, args.max_steps, on_step=record)]
    video_writer.release()
    gif.save()
    vec_env.close()
    print_episode(1, results[0])

    if args.episodes > 1:
        sample_env = create_mario_environment(env_id=env_id, action_space=args.action_space, use_noop=True)
        if args.seed is not None:
            sample_env.seed(args.seed)
        for i in range(2, args.episodes + 1):
            results.append(run_episode(sample_env, model, args.max_steps))
            print_episode(i, results[-1])
        sample_env.close()

    return results


def print_episode(index, result):
    outcome = "CLEARED" if result["cleared"] else "DIED"
    print(f"  Episode {index} [{outcome}]: reward {result['reward']:.0f}, "
          f"{result['steps']} steps, max X {result['max_x']}")


def format_summary_table(per_level):
    """Plain-text table of per-level statistics, one row per level."""
    lines = [f"{'Level':<24}{'Clear rate':>11}{'Mean reward':>13}{'Mean max X':>12}{'Best X':>8}"]
    for env_id, stats in per_level.items():
        lines.append(f"{level_tag(env_id):<24}{stats['clear_rate']:>11.0%}{stats['mean_reward']:>13.0f}"
                     f"{stats['mean_max_x']:>12.0f}{stats['best_max_x']:>8}")
    return "\n".join(lines)


def export_evaluation_video():
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
        results = evaluate_level(model, env_id, args, movement, video_path, gif_path)
        per_level[env_id] = {
            **summarize_episodes(results),
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
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    export_evaluation_video()
