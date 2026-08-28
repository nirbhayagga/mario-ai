"""
Progress Visualization Module
Renders multiple model checkpoints into a single side-by-side montage video,
plus a reduced GIF of it that is small enough to embed in a README.
"""

import argparse
import cv2
import os
import glob
import re
import numpy as np
from stable_baselines3 import PPO

from env_setup import create_mario_environment
from media import GifRecorder

# The full montage (512x480 at 30 fps for 1000 steps) is far too large as a
# GIF, so the GIF is written at half resolution and a third of the frames.
MONTAGE_GIF_SCALE = 0.5
MONTAGE_GIF_STRIDE = 3


def checkpoint_step(path):
    """Extracts the step count from a CheckpointCallback filename
    (e.g. mario_ppo_agent_250000_steps.zip -> 250000). Unparseable names sort first."""
    match = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(match.group(1)) if match else -1


def select_checkpoints(model_files, max_models=4):
    """Returns up to max_models checkpoints evenly spaced across training,
    always including the earliest and latest, sorted by step count.

    A plain lexicographic sort puts 1000000 before 200000 and slicing the
    first four then shows early checkpoints only, not training progress.
    """
    ordered = sorted(model_files, key=checkpoint_step)
    if len(ordered) <= max_models:
        return ordered
    last = len(ordered) - 1
    indices = [round(i * last / (max_models - 1)) for i in range(max_models)]
    return [ordered[i] for i in indices]


def gif_frame(bgr_frame, scale=MONTAGE_GIF_SCALE):
    """Downscales a BGR video frame and converts it to RGB for the GIF recorder."""
    height, width = bgr_frame.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.cvtColor(cv2.resize(bgr_frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def create_montage_video(checkpoint_dir, output_path,
                         env_id="SuperMarioBros-v0", num_steps=1000, device="auto"):
    """Loads up to 4 checkpoint models and renders them in a grid layout."""

    model_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not model_files:
        print(f"No models found in {checkpoint_dir}")
        return

    models_to_compare = select_checkpoints(model_files, max_models=4)
    num_models = len(models_to_compare)

    envs = []
    models = []
    names = []

    print(f"Loading {num_models} models for comparison on {env_id}...")

    for path in models_to_compare:
        target_env = [e.strip() for e in env_id.split(",")][0] if "," in env_id else env_id
        env = create_mario_environment(env_id=target_env, n_envs=1)
        model = PPO.load(path, device=device)
        envs.append(env)
        models.append(model)
        names.append(os.path.basename(path).replace(".zip", ""))

    grid_w = 2 if num_models > 1 else 1
    grid_h = (num_models + 1) // 2
    frame_w, frame_h = 256, 240
    out_w = frame_w * grid_w
    out_h = frame_h * grid_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (out_w, out_h))
    gif_path = os.path.splitext(output_path)[0] + ".gif"
    gif = GifRecorder(gif_path, fps=30.0, every=MONTAGE_GIF_STRIDE)

    observations = [env.reset() for env in envs]

    print(f"Generating montage: {output_path} ({num_steps} steps)...")

    for _ in range(num_steps):
        frames = []
        for j in range(num_models):
            action, _ = models[j].predict(observations[j], deterministic=True)
            observations[j], _, done_arr, _ = envs[j].step(action)
            if bool(done_arr[0]):
                observations[j] = envs[j].reset()

            rgb_frame = envs[j].venv.envs[0].render(mode='rgb_array')
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            cv2.putText(bgr_frame, names[j], (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            frames.append(bgr_frame)

        while len(frames) < grid_w * grid_h:
            frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

        rows = []
        for r in range(grid_h):
            row = np.hstack(frames[r * grid_w:(r + 1) * grid_w])
            rows.append(row)

        montage = np.vstack(rows)
        video_writer.write(montage)
        gif.add(gif_frame(montage))

    video_writer.release()
    gif.save()
    for env in envs:
        env.close()
    print(f"Montage generation complete: {output_path} and {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario Progress Visualization")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_models",
                        help="Directory containing model checkpoints.")
    parser.add_argument("--output_path", type=str, default="videos/progress_montage.mp4",
                        help="Path to the output montage video.")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to render per model.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device for model inference.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    create_montage_video(args.checkpoint_dir, args.output_path,
                         args.env_id, args.num_steps, args.device)
