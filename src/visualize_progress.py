"""
Progress Visualization Module
Renders multiple model checkpoints into a single side-by-side montage video.
"""

import argparse
import cv2
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from env_setup import create_mario_environment


def create_montage_video(checkpoint_dir, output_path,
                         env_id="SuperMarioBros-v0", num_steps=1000, device="cpu"):
    """Loads up to 4 checkpoint models and renders them in a grid layout."""

    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
    if not model_files:
        print(f"No models found in {checkpoint_dir}")
        return

    models_to_compare = model_files[:4]
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

    observations = [env.reset() for env in envs]

    print(f"Generating montage: {output_path} ({num_steps} steps)...")

    for i in range(num_steps):
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

    video_writer.release()
    for env in envs:
        env.close()
    print("Montage generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario Progress Visualization")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_models",
                        help="Directory containing model checkpoints.")
    parser.add_argument("--output_path", type=str, default="videos/progress_montage.mp4",
                        help="Path to the output montage video.")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to render per model.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model inference.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    create_montage_video(args.checkpoint_dir, args.output_path,
                         args.env_id, args.num_steps, args.device)
