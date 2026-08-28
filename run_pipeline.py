"""
Super Mario Bros RL Pipeline
Unified entry point for training, evaluation, validation, and interactive play.
"""

import argparse
import json
import os
import subprocess
import sys


def run_command(command):
    """Executes a shell command, streaming output to the terminal."""
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


def load_run_config(run_dir):
    """Reads the run_config.json written by train.py, or {} if there is none."""
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


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
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    cmd += ["--action_space", args.action_space]
    cmd += unknown
    return cmd


def build_validate_command(python_cmd, args, env_id):
    """Builds the validate.py subprocess command."""
    cmd = python_cmd + [
        "src/validate.py",
        "--env_id", env_id,
        "--action_space", args.action_space,
    ]
    if args.no_custom_reward:
        cmd += ["--no_custom_reward"]
    return cmd


def build_evaluate_command(python_cmd, args, env_id, model_path, video_output, extra):
    """Builds the evaluate.py subprocess command."""
    cmd = python_cmd + [
        "src/evaluate.py",
        "--model_path", model_path,
        "--video_output", video_output,
        "--device", args.device,
        "--env_id", env_id,
        "--max_steps", str(args.max_steps),
        "--episodes", str(args.eval_episodes),
        "--video_speed", str(args.video_speed),
        "--action_space", args.action_space,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd + extra


def build_visualize_command(python_cmd, args, env_id, checkpoint_dir, output_path, extra):
    """Builds the visualize_progress.py subprocess command."""
    return python_cmd + [
        "src/visualize_progress.py",
        "--checkpoint_dir", checkpoint_dir,
        "--output_path", output_path,
        "--device", args.device,
        "--env_id", env_id,
    ] + extra


def build_plot_command(python_cmd, output_dir, extra):
    """Builds the plot_metrics.py subprocess command."""
    return python_cmd + [
        "src/plot_metrics.py",
        "--log_dir", os.path.join(output_dir, "tensorboard"),
        "--output_path", os.path.join(output_dir, "reward_curve.png"),
    ] + extra


def main():
    parser = argparse.ArgumentParser(description="Mario RL Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "validate", "train", "evaluate", "manual",
            "visualize", "plot", "dashboard", "all",
        ],
        help="Pipeline mode (default: all).",
    )

    # Training / Evaluation arguments
    parser.add_argument("--level", type=str, default=None,
                        help="Mario level (e.g. 1-1, 1-2, 2-1). Comma-separated for multi-level. "
                             "Defaults to 1-1, or to the level recorded in --run_dir's run_config.json.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Existing run directory. Replaces the derived runs/<...> path and supplies "
                             "--level/--action_space from its run_config.json when they are not given.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for training and evaluation. Unseeded if omitted.")
    parser.add_argument("--total_timesteps", type=int, default=2000000,
                        help="Total timesteps for training.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Hardware device for training and inference (auto picks CUDA when available).")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--no_custom_reward", action="store_true",
                        help="DISABLE stuck penalty, early termination, and reward clipping.")
    parser.add_argument("--no_lr_schedule", action="store_true",
                        help="DISABLE linear learning rate annealing.")
    parser.add_argument("--action_space", type=str, default=None, choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only). "
                             "Defaults to simple, or to the value in --run_dir's run_config.json.")
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
                        help="Maximum evaluation steps per episode (stops earlier on level clear).")
    parser.add_argument("--video_speed", type=float, default=2.0,
                        help="Playback speed of evaluation recordings relative to real time (1.0 = real time).")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Episodes to run in evaluate mode for clear-rate statistics. "
                             "The first is recorded to video; the rest use random no-op starts.")

    args, unknown = parser.parse_known_args()

    # An existing run's config fills in whatever the command line left out,
    # so post-training modes do not need the original training flags.
    run_config = load_run_config(args.run_dir) if args.run_dir else {}
    if args.action_space is None:
        args.action_space = run_config.get("action_space", "simple")
    if args.level is None and "env_id" not in run_config:
        args.level = "1-1"

    if args.level is not None:
        env_id = ",".join(map_level(lvl) for lvl in args.level.split(","))
    else:
        env_id = run_config["env_id"]

    python_cmd = [sys.executable]

    if args.run_dir:
        output_dir = args.run_dir
    else:
        # Derive the run ID from the flags that shape the experiment
        base_level = args.level.replace(",", "_")
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
    # Unrecognized CLI args are forwarded only to the script the mode targets.
    # In "all" mode they are treated as training hyperparameters, because the
    # downstream stages use parse_args() and abort on an unknown flag.

    if args.mode == "validate":
        run_command(build_validate_command(python_cmd, args, env_id))

    elif args.mode == "train":
        run_command(build_train_command(python_cmd, args, env_id, output_dir, unknown))

    elif args.mode == "evaluate":
        model_to_use = args.load_model if args.load_model else model_path
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        run_command(build_evaluate_command(
            python_cmd, args, env_id, model_to_use, video_output, unknown))

    elif args.mode == "visualize":
        montage_output = args.video_output or os.path.join(output_dir, "progress_montage.mp4")
        os.makedirs(os.path.dirname(montage_output), exist_ok=True)
        run_command(build_visualize_command(
            python_cmd, args, env_id, checkpoint_dir, montage_output, unknown))

    elif args.mode == "plot":
        os.makedirs(output_dir, exist_ok=True)
        run_command(build_plot_command(python_cmd, output_dir, unknown))

    elif args.mode == "dashboard":
        run_command(
            ["tensorboard", "--logdir", os.path.join(output_dir, "tensorboard")] + unknown
        )

    elif args.mode == "manual":
        run_command(python_cmd + ["src/manual_play.py", "--env_id", env_id])

    elif args.mode == "all":
        print("\n=== STAGE 1: VALIDATION ===")
        run_command(build_validate_command(python_cmd, args, env_id))

        print("\n=== STAGE 2: TRAINING ===")
        run_command(build_train_command(python_cmd, args, env_id, output_dir, unknown))

        print("\n=== STAGE 3: EVALUATION ===")
        # Prefer best_model (peak performance) over final model (may have collapsed)
        best_path = os.path.join(output_dir, "best_model.zip")
        final_path = os.path.join(output_dir, "mario_ppo_final.zip")
        model_to_use = best_path if os.path.exists(best_path) else final_path
        print(f"Using model: {model_to_use}")
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        run_command(build_evaluate_command(
            python_cmd, args, env_id, model_to_use, video_output, []))

        print("\n=== STAGE 4: VISUALIZATION MONTAGE ===")
        run_command(build_visualize_command(
            python_cmd, args, env_id, checkpoint_dir,
            os.path.join(output_dir, "progress_montage.mp4"), []))

        print("\n=== STAGE 5: PLOTTING METRICS ===")
        run_command(build_plot_command(python_cmd, output_dir, []))

        print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
