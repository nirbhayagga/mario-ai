"""
Progress Visualization Module
This script renders multiple model checkpoints into a single side-by-side montage video.
"""

import argparse
import cv2
import os
import glob
from stable_baselines3 import PPO
from env_setup import create_mario_environment
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

def create_montage_video(checkpoint_dir, output_path, num_steps=1000, device="cpu"):
    """
    Loads all .zip models in checkpoint_dir and renders them in a grid.
    """
    # Find all model files
    model_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.zip")))
    if not model_files:
        print(f"No models found in {checkpoint_dir}")
        return

    # Limit to 4 models for a 2x2 grid, or 2 for a 1x2 grid
    models_to_compare = model_files[:4]
    num_models = len(models_to_compare)
    
    envs = []
    models = []
    names = []

    print(f"Loading {num_models} models for comparison...")

    for path in models_to_compare:
        env = create_mario_environment(n_envs=1)
        model = PPO.load(path, device=device)
        envs.append(env)
        models.append(model)
        names.append(os.path.basename(path).replace(".zip", ""))

    # Video Writer Setup
    # Each Mario frame is 256x240 (original). 
    # For a 2x2 grid of 4 models: 512x480
    grid_w = 2 if num_models > 1 else 1
    grid_h = (num_models + 1) // 2
    
    frame_w, frame_h = 256, 240
    out_w = frame_w * grid_w
    out_h = frame_h * grid_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (out_w, out_h))

    observations = [env.reset() for env in envs]
    dones = [False] * num_models

    print(f"Generating montage: {output_path} ({num_steps} steps)...")

    for i in range(num_steps):
        frames = []
        for j in range(num_models):
            if not dones[j]:
                action, _ = models[j].predict(observations[j], deterministic=True)
                observations[j], _, done, _ = envs[j].step(action)
                if done:
                    observations[j] = envs[j].reset()
            
            # Use env.envs[0] to access the underlying environment for rendering
            # Note: evaluate.py uses env.envs[0].render()
            rgb_frame = envs[j].envs[0].render(mode='rgb_array')
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Overlay model name
            cv2.putText(bgr_frame, names[j], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            frames.append(bgr_frame)

        # Build grid
        # Pad with black frames if needed
        while len(frames) < grid_w * grid_h:
            frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

        # Horizontal tiles
        rows = []
        for r in range(grid_h):
            row = np.hstack(frames[r*grid_w : (r+1)*grid_w])
            rows.append(row)
        
        # Vertical composite
        montage = np.vstack(rows)
        video_writer.write(montage)

    video_writer.release()
    print("Montage generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario Progress Visualization")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_models", help="Directory containing model checkpoints.")
    parser.add_argument("--output_path", type=str, default="videos/progress_montage.mp4", help="Path to the output montage video.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to render.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model inference.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    create_montage_video(args.checkpoint_dir, args.output_path, args.num_steps, args.device)
