"""
Super Mario Bros RL Pipeline
Unified entry point for training, evaluation, validation, and interactive play.
"""

import argparse
import subprocess
import os
import sys


def run_command(command):
    """Executes a shell command, streaming output to the terminal."""
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


def map_level(level_str):
    """Converts shorthand level notation (e.g. '1-1') to the full Gym ID."""
    level_str = level_str.strip()
    if "-v" in level_str:
        return f"SuperMarioBros-{level_str}"
    return f"SuperMarioBros-{level_str}-v0"


def build_train_command(python_cmd, args, env_id, output_dir, unknown, n_envs_override=None):
    """Builds the train.py subprocess command with all flags properly forwarded."""
    cmd = python_cmd + [
        "src/train.py",
        "--total_timesteps", str(args.total_timesteps),
        "--device", args.device,
        "--n_envs", str(n_envs_override or args.n_envs),
        "--env_id", env_id,
        "--output_dir", output_dir,
        "--n_epochs", str(args.n_epochs),
    ]
    if args.save_freq is not None:
        cmd += ["--save_freq", str(args.save_freq)]
    if args.load_model:
        cmd += ["--load_model", args.load_model]
    if args.no_custom_reward:
        cmd += ["--no_custom_reward"]
    if args.no_lr_schedule:
        cmd += ["--no_lr_schedule"]
    cmd += ["--action_space", args.action_space]
    cmd += unknown
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Mario RL Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "validate", "train", "evaluate", "manual", "compete",
            "visualize", "plot", "dashboard", "train_general", "all",
        ],
        help="Pipeline mode (default: all).",
    )

    # Training / Evaluation arguments
    parser.add_argument("--level", type=str, default="1-1",
                        help="Mario level (e.g. 1-1, 1-2, 2-1). Comma-separated for multi-level.")
    parser.add_argument("--total_timesteps", type=int, default=2000000,
                        help="Total timesteps for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Hardware device for training and inference.")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--no_custom_reward", action="store_true",
                        help="DISABLE reward shaping (flagpole bonus, death penalty, stuck penalty).")
    parser.add_argument("--no_lr_schedule", action="store_true",
                        help="DISABLE linear learning rate annealing.")
    parser.add_argument("--action_space", type=str, default="simple", choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only).")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a .zip model to resume training from.")

    # Evaluation / Visualization arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a model file for evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing checkpoints for montage visualization.")
    parser.add_argument("--video_output", type=str, default=None,
                        help="Destination path for exported .mp4 video files.")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Checkpoint save frequency. Auto-computed for ~15 checkpoints if not set.")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO optimization epochs per rollout.")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum evaluation steps (stops earlier on level clear).")

    args, unknown = parser.parse_known_args()

    # Resolve environment IDs
    if "," in args.level:
        levels = [map_level(l) for l in args.level.split(",")]
        env_id = ",".join(levels)
    else:
        env_id = map_level(args.level)

    python_cmd = [sys.executable]

    # Generate dynamic run ID and output paths
    base_level = "general" if args.mode == "train_general" else args.level.replace(",", "_")
    run_id = f"{base_level}_{args.total_timesteps}steps"
    if not args.no_custom_reward:
        run_id += "_rew"
    if not args.no_lr_schedule:
        run_id += "_lr"
    if args.load_model:
        run_id += "_resume"
    if args.action_space != "simple":
        run_id += f"_{args.action_space}"

    output_dir = os.path.join("runs", run_id)

    # Resolve paths with fallbacks to the dynamic output directory
    # Prefer best_model.zip (saved by EvalCallback) over the final model
    best_model = os.path.join(output_dir, "best_model.zip")
    final_model = os.path.join(output_dir, "mario_ppo_final.zip")
    default_model = best_model if os.path.exists(best_model) else final_model
    model_path = args.model_path or default_model
    checkpoint_dir = args.checkpoint_dir or os.path.join(output_dir, "checkpoints")
    video_output = args.video_output or os.path.join(output_dir, "evaluation_run.mp4")

    # ---- Mode dispatch ----

    if args.mode == "validate":
        run_command(python_cmd + ["src/validate.py", "--env_id", env_id])

    elif args.mode == "train":
        run_command(build_train_command(python_cmd, args, env_id, output_dir, unknown))

    elif args.mode == "evaluate":
        model_to_use = args.load_model if args.load_model else model_path
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        run_command(python_cmd + [
            "src/evaluate.py",
            "--model_path", model_to_use,
            "--video_output", video_output,
            "--device", args.device,
            "--env_id", env_id,
            "--max_steps", str(args.max_steps),
            "--action_space", args.action_space,
        ] + unknown)

    elif args.mode == "visualize":
        montage_output = args.video_output or os.path.join(output_dir, "progress_montage.mp4")
        os.makedirs(os.path.dirname(montage_output), exist_ok=True)
        run_command(python_cmd + [
            "src/visualize_progress.py",
            "--checkpoint_dir", checkpoint_dir,
            "--output_path", montage_output,
            "--device", args.device,
            "--env_id", env_id,
        ] + unknown)

    elif args.mode == "plot":
        os.makedirs(output_dir, exist_ok=True)
        run_command(python_cmd + [
            "src/plot_metrics.py",
            "--log_dir", os.path.join(output_dir, "tensorboard"),
            "--output_path", os.path.join(output_dir, "reward_curve.png"),
        ] + unknown)

    elif args.mode == "dashboard":
        run_command(
            ["tensorboard", "--logdir", os.path.join(output_dir, "tensorboard")] + unknown
        )

    elif args.mode == "train_general":
        general_levels = [
            "SuperMarioBros-1-1-v0",
            "SuperMarioBros-1-2-v0",
            "SuperMarioBros-2-1-v0",
            "SuperMarioBros-3-1-v0",
            "SuperMarioBros-4-1-v0",
        ]
        general_env_id = ",".join(general_levels)
        # Ensure at least one env per level; user can request more for better throughput
        general_n_envs = max(args.n_envs, len(general_levels))
        run_command(build_train_command(
            python_cmd, args, general_env_id, output_dir, unknown,
            n_envs_override=general_n_envs,
        ))

    elif args.mode == "manual":
        run_command(python_cmd + ["src/manual_play.py", "--env_id", env_id])

    elif args.mode == "compete":
        run_command(python_cmd + [
            "src/compete.py", "--env_id", env_id, "--model_path", model_path,
        ])

    elif args.mode == "all":
        print("\n=== STAGE 1: VALIDATION ===")
        run_command(python_cmd + ["src/validate.py", "--env_id", env_id])

        print("\n=== STAGE 2: TRAINING ===")
        run_command(build_train_command(python_cmd, args, env_id, output_dir, unknown))

        print("\n=== STAGE 3: EVALUATION ===")
        # Prefer best_model (peak performance) over final model (may have collapsed)
        best_path = os.path.join(output_dir, "best_model.zip")
        final_path = os.path.join(output_dir, "mario_ppo_final.zip")
        model_to_use = best_path if os.path.exists(best_path) else final_path
        print(f"Using model: {model_to_use}")
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        run_command(python_cmd + [
            "src/evaluate.py",
            "--model_path", model_to_use,
            "--video_output", video_output,
            "--device", args.device,
            "--env_id", env_id,
            "--max_steps", str(args.max_steps),
            "--action_space", args.action_space,
        ] + unknown)

        print("\n=== STAGE 4: VISUALIZATION MONTAGE ===")
        run_command(python_cmd + [
            "src/visualize_progress.py",
            "--checkpoint_dir", checkpoint_dir,
            "--output_path", os.path.join(output_dir, "progress_montage.mp4"),
            "--device", args.device,
            "--env_id", env_id,
        ] + unknown)

        print("\n=== STAGE 5: PLOTTING METRICS ===")
        run_command(python_cmd + [
            "src/plot_metrics.py",
            "--log_dir", os.path.join(output_dir, "tensorboard"),
            "--output_path", os.path.join(output_dir, "reward_curve.png"),
        ] + unknown)

        print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
