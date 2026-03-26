"""
Evaluation and Video Export Module
Loads trained model weights and renders a gameplay video with metadata overlay.
"""

import argparse
import cv2
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from env_setup import create_mario_environment


def parse_evaluation_arguments():
    parser = argparse.ArgumentParser(description="PPO Mario Evaluation")
    parser.add_argument("--model_path", type=str, default="mario_ppo_final.zip",
                        help="Path to the serialized model weights.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Hardware device for inference.")
    parser.add_argument("--video_output", type=str, default="evaluation_run.mp4",
                        help="Output path for the evaluation video.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    return parser.parse_args()


def export_evaluation_video():
    """
    Loads the trained model, runs one full episode through the vectorized
    environment (for correct observation shape), and renders video frames
    from a separate unwrapped environment to avoid frame-skip artifacts.
    """
    args = parse_evaluation_arguments()

    # Use the first level if comma-separated list was passed
    if "," in args.env_id:
        target_env = [e.strip() for e in args.env_id.split(",")][0]
    else:
        target_env = args.env_id

    # Create the vectorized env for model inference (must match training shape)
    vec_env = create_mario_environment(env_id=target_env)

    # Create a separate raw env for clean video rendering (no frame skip, no grayscale)
    render_env = gym_super_mario_bros.make(target_env)
    render_env = JoypadSpace(render_env, SIMPLE_MOVEMENT)

    model = PPO.load(args.model_path, device=args.device)

    # Reset both environments
    obs = vec_env.reset()
    render_env.reset()

    # Video writer at native NES resolution and 30 FPS for smooth playback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.video_output, fourcc, 30.0, (256, 240))

    total_reward = 0
    episode_done = False

    print(f"Recording evaluation to {args.video_output}")

    while not episode_done:
        # Get action from the model using the vectorized observation
        action, _ = model.predict(obs, deterministic=True)

        # Step both environments with the same action
        obs, reward, done, info = vec_env.step(action)
        action_int = int(action[0])

        # Step the render env 4 times to match the SkipFrame(4) wrapper
        for _ in range(4):
            render_obs, _, render_done, render_info = render_env.step(action_int)
            if render_done:
                render_env.reset()
                break

        total_reward += float(reward[0])

        # Render the clean frame from the unwrapped environment
        rendered_frame = render_env.render(mode='rgb_array')
        bgr_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

        # Extract metadata for overlay
        x_pos = int(info[0].get('x_pos', 0))
        score = int(info[0].get('score', 0))
        action_name = str(SIMPLE_MOVEMENT[action_int]) if action_int < len(SIMPLE_MOVEMENT) else "UNKNOWN"

        # Draw overlay text
        cv2.putText(bgr_frame, f"X: {x_pos}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, f"Score: {score}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, f"Action: {action_name}", (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        video_writer.write(bgr_frame)

        # Check done from the vectorized env (returns array)
        episode_done = bool(done[0])

    print(f"Evaluation complete. Total reward: {total_reward:.0f}")

    video_writer.release()
    render_env.close()
    vec_env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    export_evaluation_video()
