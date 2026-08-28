"""
Super Mario Bros RL Pipeline
Unified entry point: python run_pipeline.py <command> [options]

Commands: validate, train, evaluate, visualize, plot, dashboard, manual, all.
Each command declares only the options it uses; `<command> --help` lists them.
The orchestrator never imports from src/: it builds argv lists and runs the
stage scripts as subprocesses.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any

RUNS_ROOT = "runs"

# PPO hyperparameters whose defaults live in src/train.py. They are declared
# here so `train --help` is complete, but only forwarded when explicitly set.
PPO_FLAGS = (
    ("learning_rate", float, "Optimizer learning rate (train.py default 2.5e-4)."),
    ("n_steps", int, "Steps per env per rollout (default 512)."),
    ("batch_size", int, "Minibatch size (default 256)."),
    ("n_epochs", int, "Optimization epochs per rollout (default 10)."),
    ("ent_coef", float, "Entropy coefficient (default 0.01)."),
    ("gamma", float, "Discount factor (default 0.99)."),
    ("vf_coef", float, "Value function loss coefficient (default 0.5)."),
    ("clip_range", float, "PPO clipping parameter (default 0.1)."),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_command(command: list[str]) -> None:
    """Executes a command, streaming output to the terminal; exits on failure."""
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


def map_level(level_str: str) -> str:
    """Converts shorthand level notation (e.g. '1-1') to the full Gym ID."""
    level_str = level_str.strip()
    if "-v" in level_str:
        return f"SuperMarioBros-{level_str}"
    return f"SuperMarioBros-{level_str}-v0"


def load_run_config(run_dir: str) -> dict[str, Any]:
    """Reads the run_config.json written by train.py, or {} if there is none."""
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def latest_run_dir() -> str | None:
    """The most recently trained run under runs/ (by run_config.json mtime), or None."""
    configs = glob.glob(os.path.join(RUNS_ROOT, "*", "run_config.json"))
    if not configs:
        return None
    return os.path.dirname(max(configs, key=os.path.getmtime))


def derive_run_dir(args: argparse.Namespace) -> str:
    """runs/<level>_<timesteps>steps[_rew][_lr][_resume][_<action_space>]"""
    run_id = f"{args.level.replace(',', '_')}_{args.total_timesteps}steps"
    if not args.no_custom_reward:
        run_id += "_rew"
    if not args.no_lr_schedule:
        run_id += "_lr"
    if args.load_model:
        run_id += "_resume"
    if args.action_space != "simple":
        run_id += f"_{args.action_space}"
    return os.path.join(RUNS_ROOT, run_id)


def default_model(run_dir: str) -> str:
    """Prefers best_model.zip (saved by EvalCallback) over the possibly collapsed final model."""
    best = os.path.join(run_dir, "best_model.zip")
    return best if os.path.exists(best) else os.path.join(run_dir, "mario_ppo_final.zip")


def resolve_run(args: argparse.Namespace, mode: str) -> tuple[str, str, str | None]:
    """
    Resolves (env_id, action_space, run_dir) for a command.

    mode "new": the run directory is --run_dir or derived from the training
    flags. mode "existing": it is --run_dir or the latest run under runs/.
    mode "none": the command has no run directory. In every case, an existing
    run_config.json fills in --level / --action_space when they were not given.
    """
    run_dir = getattr(args, "run_dir", None)
    if mode == "existing" and run_dir is None:
        run_dir = latest_run_dir()
        if run_dir is None:
            sys.exit(f"No runs found under {RUNS_ROOT}/. Train first or pass --run_dir.")
        print(f"Using most recent run: {run_dir}")

    config = load_run_config(run_dir) if run_dir else {}
    if args.action_space is None:
        args.action_space = config.get("action_space", "simple")
    if args.level is not None:
        env_id = ",".join(map_level(lvl) for lvl in args.level.split(","))
    elif "env_id" in config:
        env_id = config["env_id"]
    else:
        args.level = "1-1"
        env_id = map_level(args.level)

    if mode == "new" and run_dir is None:
        run_dir = derive_run_dir(args)
    return env_id, args.action_space, run_dir


# ---------------------------------------------------------------------------
# Subprocess command builders
# ---------------------------------------------------------------------------

def build_validate_command(args: argparse.Namespace, env_id: str) -> list[str]:
    cmd = [sys.executable, "src/validate.py", "--env_id", env_id, "--action_space", args.action_space]
    if args.no_custom_reward:
        cmd += ["--no_custom_reward"]
    return cmd


def build_train_command(args: argparse.Namespace, env_id: str, run_dir: str) -> list[str]:
    cmd = [
        sys.executable, "src/train.py",
        "--total_timesteps", str(args.total_timesteps),
        "--device", args.device,
        "--n_envs", str(args.n_envs),
        "--env_id", env_id,
        "--output_dir", run_dir,
        "--action_space", args.action_space,
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
    for name, _, _ in PPO_FLAGS:
        value = getattr(args, name)
        if value is not None:
            cmd += [f"--{name}", str(value)]
    return cmd


def build_evaluate_command(args: argparse.Namespace, env_id: str, model_path: str, video_output: str) -> list[str]:
    cmd = [
        sys.executable, "src/evaluate.py",
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
    return cmd


def build_visualize_command(args: argparse.Namespace, env_id: str, checkpoint_dir: str, output_path: str,
                            num_steps: int) -> list[str]:
    return [
        sys.executable, "src/visualize_progress.py",
        "--checkpoint_dir", checkpoint_dir,
        "--output_path", output_path,
        "--device", args.device,
        "--env_id", env_id,
        "--num_steps", str(num_steps),
    ]


def build_plot_command(run_dir: str) -> list[str]:
    return [
        sys.executable, "src/plot_metrics.py",
        "--log_dir", os.path.join(run_dir, "tensorboard"),
        "--output_path", os.path.join(run_dir, "reward_curve.png"),
    ]


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    env_id, _, _ = resolve_run(args, "none")
    run_command(build_validate_command(args, env_id))


def cmd_train(args: argparse.Namespace) -> None:
    env_id, _, run_dir = resolve_run(args, "new")
    run_command(build_train_command(args, env_id, run_dir))


def cmd_evaluate(args: argparse.Namespace) -> None:
    env_id, _, run_dir = resolve_run(args, "existing")
    model_path = args.model_path or default_model(run_dir)
    video_output = args.video_output or os.path.join(run_dir, "evaluation_run.mp4")
    os.makedirs(os.path.dirname(video_output) or ".", exist_ok=True)
    run_command(build_evaluate_command(args, env_id, model_path, video_output))


def cmd_visualize(args: argparse.Namespace) -> None:
    env_id, _, run_dir = resolve_run(args, "existing")
    checkpoint_dir = args.checkpoint_dir or os.path.join(run_dir, "checkpoints")
    output_path = args.video_output or os.path.join(run_dir, "progress_montage.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    run_command(build_visualize_command(args, env_id, checkpoint_dir, output_path, args.num_steps))


def cmd_plot(args: argparse.Namespace) -> None:
    _, _, run_dir = resolve_run(args, "existing")
    run_command(build_plot_command(run_dir))


def cmd_dashboard(args: argparse.Namespace) -> None:
    _, _, run_dir = resolve_run(args, "existing")
    run_command(["tensorboard", "--logdir", os.path.join(run_dir, "tensorboard"), "--port", str(args.port)])


def cmd_manual(args: argparse.Namespace) -> None:
    env_id, _, _ = resolve_run(args, "none")
    run_command([sys.executable, "src/manual_play.py", "--env_id", env_id])


def cmd_all(args: argparse.Namespace) -> None:
    env_id, _, run_dir = resolve_run(args, "new")

    print("\n=== STAGE 1: VALIDATION ===")
    run_command(build_validate_command(args, env_id))

    print("\n=== STAGE 2: TRAINING ===")
    run_command(build_train_command(args, env_id, run_dir))

    print("\n=== STAGE 3: EVALUATION ===")
    model_path = default_model(run_dir)
    print(f"Using model: {model_path}")
    run_command(build_evaluate_command(args, env_id, model_path, os.path.join(run_dir, "evaluation_run.mp4")))

    print("\n=== STAGE 4: VISUALIZATION MONTAGE ===")
    run_command(build_visualize_command(
        args, env_id, os.path.join(run_dir, "checkpoints"), os.path.join(run_dir, "progress_montage.mp4"),
        args.num_steps))

    print("\n=== STAGE 5: PLOTTING METRICS ===")
    run_command(build_plot_command(run_dir))

    print("\n=== PIPELINE COMPLETE ===")


# ---------------------------------------------------------------------------
# Argument groups shared between commands
# ---------------------------------------------------------------------------

def add_level_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level", type=str, default=None,
                        help="Mario level, e.g. 1-1 or 1-1-v3; comma-separated for multi-level. "
                             "Defaults to 1-1, or to the level recorded in the run's run_config.json.")
    parser.add_argument("--action_space", type=str, default=None, choices=["simple", "complex", "right_only"],
                        help="Movement set: simple (7 actions), right_only (5), complex (12). "
                             "Defaults to simple, or to the run's run_config.json.")


def add_reward_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no_custom_reward", action="store_true",
                        help="Disable stuck penalty, early termination, reward clipping, and random no-op starts.")


def add_existing_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory to use. Defaults to the most recently trained run under runs/.")
    add_level_args(parser)


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto picks CUDA when available.")


def add_seed_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for PPO, numpy/torch, and every environment. Unseeded if omitted.")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory to write the run to. Defaults to runs/<level>_<timesteps>steps[_rew][_lr]...")
    add_level_args(parser)
    add_reward_arg(parser)
    parser.add_argument("--total_timesteps", type=int, default=2000000, help="Total training timesteps.")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Parallel environments. Multi-level runs cycle levels across them.")
    parser.add_argument("--no_lr_schedule", action="store_true", help="Disable linear learning rate decay.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a .zip model to resume from. All hyperparameters given here apply on resume.")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Checkpoint every N env steps. Auto-computed for ~15 checkpoints if not set.")
    add_device_arg(parser)
    add_seed_arg(parser)
    ppo = parser.add_argument_group("PPO hyperparameters (defaults in src/train.py)")
    for name, kind, help_text in PPO_FLAGS:
        ppo.add_argument(f"--{name}", type=kind, default=None, help=help_text)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per evaluation episode (stops earlier on clear or death).")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Episodes per level for the clear-rate summary. The first is recorded; "
                             "the rest use random no-op starts.")
    parser.add_argument("--video_speed", type=float, default=2.0,
                        help="Playback speed of recordings relative to real time (1.0 = real time, 15 fps).")


def add_montage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num_steps", type=int, default=1000, help="Steps to render per checkpoint.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Super Mario Bros PPO pipeline. Run '<command> --help' for that command's options.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command", required=True)

    p = commands.add_parser("validate", help="Build the environment stack and step it once.")
    add_level_args(p)
    add_reward_arg(p)
    p.set_defaults(func=cmd_validate)

    p = commands.add_parser("train", help="Train PPO on one or more levels.")
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = commands.add_parser("evaluate", help="Record gameplay and compute per-level clear rate for a trained run.")
    add_existing_run_args(p)
    p.add_argument("--model_path", type=str, default=None,
                   help="Model to evaluate. Defaults to the run's best_model.zip, then mario_ppo_final.zip.")
    p.add_argument("--video_output", type=str, default=None,
                   help="Output .mp4 path (a .gif is written alongside). Defaults to <run_dir>/evaluation_run.mp4.")
    add_eval_args(p)
    add_device_arg(p)
    add_seed_arg(p)
    p.set_defaults(func=cmd_evaluate)

    p = commands.add_parser("visualize", help="Render a side-by-side montage of 4 checkpoints.")
    add_existing_run_args(p)
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory of checkpoints. Defaults to <run_dir>/checkpoints.")
    p.add_argument("--video_output", type=str, default=None,
                   help="Output .mp4 path (a .gif is written alongside). Defaults to <run_dir>/progress_montage.mp4.")
    add_montage_args(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_visualize)

    p = commands.add_parser("plot", help="Plot training and evaluation curves from TensorBoard logs.")
    add_existing_run_args(p)
    p.set_defaults(func=cmd_plot)

    p = commands.add_parser("dashboard", help="Launch TensorBoard for a run.")
    add_existing_run_args(p)
    p.add_argument("--port", type=int, default=6006, help="TensorBoard port.")
    p.set_defaults(func=cmd_dashboard)

    p = commands.add_parser("manual", help="Play a level yourself with the keyboard.")
    add_level_args(p)
    p.set_defaults(func=cmd_manual)

    p = commands.add_parser("all", help="validate, train, evaluate, visualize, and plot in sequence.")
    add_train_args(p)
    add_eval_args(p)
    add_montage_args(p)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
