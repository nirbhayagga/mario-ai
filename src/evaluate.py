"""
Evaluation and Video Export Module
Loads trained model weights and renders a gameplay video with metadata overlay.
Uses a single environment for both inference and rendering to avoid desync.
"""

import argparse
import cv2
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
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
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum evaluation steps (stops earlier on level clear).")
    parser.add_argument("--action_space", type=str, default="simple", choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only).")
    return parser.parse_args()


def export_evaluation_video():
    """
    Loads the trained model and records a gameplay video. Uses the same
    vectorized environment for both model inference and rendering - render()
    passes through all observation wrappers to the base NES env, giving us
    full-color 256x240 frames that are perfectly in sync with the model's
    observations.
    """
    args = parse_evaluation_arguments()

    # Use the first level if comma-separated list was passed
    if "," in args.env_id:
        target_env = [e.strip() for e in args.env_id.split(",")][0]
    else:
        target_env = args.env_id

    # Single env for both inference and rendering (n_envs=1 -> DummyVecEnv)
    vec_env = create_mario_environment(env_id=target_env, action_space=args.action_space)
    model = PPO.load(args.model_path, device=args.device)

    obs = vec_env.reset()

    # Access the inner gym env for full-color rendering.
    # VecFrameStack.venv -> DummyVecEnv, .envs[0] -> wrapped gym env.
    # render() passes through all wrappers (GrayScale, Resize, etc.) to the
    # base NES env, returning a clean 256x240 RGB frame.
    inner_env = vec_env.venv.envs[0]

    # Video writer at native NES resolution and 30 FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.video_output, fourcc, 30.0, (256, 240))

    total_reward = 0
    max_steps = args.max_steps
    step = 0
    cleared = False

    # Resolve action space for overlay
    if args.action_space == "complex":
        movement = COMPLEX_MOVEMENT
    elif args.action_space == "right_only":
        movement = RIGHT_ONLY
    else:
        movement = SIMPLE_MOVEMENT

    print(f"Recording evaluation to {args.video_output} (max {max_steps} steps)")

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        action_int = int(action[0])
        total_reward += float(reward[0])

        # Render full-color frame from the base NES env (same env the model sees)
        rendered_frame = inner_env.render(mode='rgb_array')
        bgr_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

        # Overlay metadata from the env's info dict (perfectly in sync)
        x_pos = int(info[0].get('x_pos', 0))
        score = int(info[0].get('score', 0))
        action_name = str(movement[action_int]) if action_int < len(movement) else "UNKNOWN"

        cv2.putText(bgr_frame, f"X: {x_pos}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, f"Score: {score}", (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, f"Action: {action_name}", (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr_frame, f"Reward: {total_reward:.0f}", (5, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        video_writer.write(bgr_frame)

        # Stop on level clear
        if info[0].get('flag_get', False):
            cleared = True
            print(f"Level cleared at step {step + 1}!")
            break

        # Stop on first death - prevents misleading score accumulation across resets
        if done[0]:
            print(f"Mario died at step {step + 1} (x_pos={x_pos}).")
            break

    result = "CLEARED" if cleared else "DIED"
    print(f"Evaluation complete [{result}]. Total reward: {total_reward:.0f} ({step + 1} steps, max X: {x_pos})")

    video_writer.release()
    vec_env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    export_evaluation_video()
