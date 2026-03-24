"""
Super Mario Bros RL Pipeline Script
A unified entry point for managing training, evaluation, validation, and manual play.
"""

import argparse
import subprocess
import os
import sys

def run_command(command):
    """Executes a shell command and monitors for progress."""
    print(f"Executing: {' '.join(command)}")
    try:
        # Use subprocess.run to output directly to the current shell
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Mario RL Pipeline Orchestrator")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["validate", "train", "evaluate", "manual", "compete", "visualize", "plot", "dashboard", "train_general", "all"],
        help="The pipeline operation mode to execute."
    )
    
    # Training / Evaluation Arguments
    parser.add_argument("--level", type=str, default="1-1", help="The Mario level to run (e.g., 1-1, 1-2, 2-1).")
    parser.add_argument("--total_timesteps", type=int, default=2000000, help="Total timesteps for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Hardware device selection.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments to run.")
    parser.add_argument("--use_custom_reward", action="store_true", help="Enable advanced reward shaping.")
    parser.add_argument("--lr_schedule", action="store_true", help="Enable linear learning rate annealing.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pre-trained model to resume or evaluate.")
    
    # Evaluation / Visualization Arguments
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model file for evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory for progress visualization.")
    parser.add_argument("--video_output", type=str, default=None, help="Folder to export evaluation videos.")

    args, unknown = parser.parse_known_args()

    # Map level "1-1" to "SuperMarioBros-1-1-v0"
    def map_level(l):
        l = l.strip()
        if "-v" in l:
            return f"SuperMarioBros-{l}"
        return f"SuperMarioBros-{l}-v0"

    if "," in args.level:
        levels = [map_level(l) for l in args.level.split(",")]
        env_id = ",".join(levels)
    else:
        env_id = map_level(args.level)
    
    # Base python command within the venv (assumes venv/bin/python)
    python_cmd = [sys.executable] # Use current interpreter if running from venv

    # Generate Dynamic Run ID and Paths
    base_level = "general" if args.mode == "train_general" else args.level.replace(",", "_")
    run_id = f"{base_level}_{args.total_timesteps}steps"
    if args.use_custom_reward: run_id += "_rew"
    if args.lr_schedule: run_id += "_lr"
    if args.load_model: run_id += "_resume"
    
    output_dir = os.path.join("runs", run_id)
    
    # Resolve dynamic paths (fallback to dynamically constructed paths if not explicitly overridden)
    model_path = args.model_path or os.path.join(output_dir, "mario_ppo_final.zip")
    checkpoint_dir = args.checkpoint_dir or os.path.join(output_dir, "checkpoints")
    video_output = args.video_output or os.path.join(output_dir, "evaluation_run.mp4")

    if args.mode == "validate":
        run_command(python_cmd + ["src/validate.py", "--env_id", env_id])
    
    elif args.mode == "train":
        train_cmd = python_cmd + [
            "src/train.py", 
            "--total_timesteps", str(args.total_timesteps),
            "--device", args.device,
            "--n_envs", str(args.n_envs),
            "--env_id", env_id,
            "--output_dir", output_dir
        ]
        if args.load_model:
            train_cmd += ["--load_model", args.load_model]
        
        train_cmd += unknown
        run_command(train_cmd)
        
    elif args.mode == "evaluate":
        # Use load_model as fallback for model_path if provided
        model_to_use = args.load_model if args.load_model else model_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        eval_cmd = python_cmd + [
            "src/evaluate.py",
            "--model_path", model_to_use,
            "--video_output", video_output,
            "--device", args.device,
            "--env_id", env_id
        ] + unknown
        run_command(eval_cmd)
        
    elif args.mode == "visualize":
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        vis_cmd = python_cmd + [
            "src/visualize_progress.py",
            "--checkpoint_dir", checkpoint_dir,
            "--output_path", video_output,
            "--device", args.device,
            "--env_id", env_id
        ] + unknown
        run_command(vis_cmd)

    elif args.mode == "plot":
        os.makedirs(output_dir, exist_ok=True)
        plot_cmd = python_cmd + [
            "src/plot_metrics.py",
            "--log_dir", os.path.join(output_dir, "tensorboard"),
            "--output_path", os.path.join(output_dir, "reward_curve.png")
        ] + unknown
        run_command(plot_cmd)
        
    elif args.mode == "dashboard":
        db_cmd = ["tensorboard", "--logdir", os.path.join(output_dir, "tensorboard")] + unknown
        run_command(db_cmd)
        
    elif args.mode == "train_general":
        # Curated list of levels for "General" training
        general_levels = [
            "SuperMarioBros-1-1-v0",
            "SuperMarioBros-1-2-v0",
            "SuperMarioBros-2-1-v0",
            "SuperMarioBros-3-1-v0",
            "SuperMarioBros-4-1-v0"
        ]
        levels_str = ",".join(general_levels)
        
        train_cmd = python_cmd + [
            "src/train.py", 
            "--env_id", levels_str,
            "--total_timesteps", str(args.total_timesteps),
            "--device", args.device,
            "--n_envs", str(len(general_levels)), # Automatically use one env per level
            "--output_dir", output_dir
        ]
        if args.load_model:
            train_cmd += ["--load_model", args.load_model]
        
        train_cmd += unknown
        run_command(train_cmd)

    elif args.mode == "manual":
        run_command(python_cmd + ["src/manual_play.py", "--env_id", env_id])
        
    elif args.mode == "compete":
        compete_cmd = python_cmd + ["src/compete.py", "--env_id", env_id, "--model_path", model_path]
        run_command(compete_cmd)
        
    elif args.mode == "all":
        print("\n=== STAGE 1: VALIDATION ===")
        run_command(python_cmd + ["src/validate.py", "--env_id", env_id])
        
        print("\n=== STAGE 2: TRAINING ===")
        train_cmd = python_cmd + [
            "src/train.py", 
            "--total_timesteps", str(args.total_timesteps),
            "--device", args.device,
            "--n_envs", str(args.n_envs),
            "--env_id", env_id,
            "--output_dir", output_dir
        ]
        if args.load_model:
            train_cmd += ["--load_model", args.load_model]
        if args.use_custom_reward:
            train_cmd += ["--use_custom_reward"]
        if args.lr_schedule:
            train_cmd += ["--lr_schedule"]
        train_cmd += unknown
        run_command(train_cmd)
        
        print("\n=== STAGE 3: EVALUATION ===")
        model_to_use = os.path.join(output_dir, "mario_ppo_final.zip")
        os.makedirs(os.path.dirname(video_output), exist_ok=True)
        eval_cmd = python_cmd + [
            "src/evaluate.py",
            "--model_path", model_to_use,
            "--video_output", video_output,
            "--device", args.device,
            "--env_id", env_id
        ] + unknown
        run_command(eval_cmd)
        
        print("\n=== STAGE 4: VISUALIZATION MONTAGE ===")
        vis_cmd = python_cmd + [
            "src/visualize_progress.py",
            "--checkpoint_dir", checkpoint_dir,
            "--output_path", os.path.join(output_dir, "progress_montage.mp4"),
            "--device", args.device,
            "--env_id", env_id
        ] + unknown
        run_command(vis_cmd)
        
        print("\n=== STAGE 5: PLOTTING METRICS ===")
        plot_cmd = python_cmd + [
            "src/plot_metrics.py",
            "--log_dir", os.path.join(output_dir, "tensorboard"),
            "--output_path", os.path.join(output_dir, "reward_curve.png")
        ] + unknown
        run_command(plot_cmd)
        
        print("\n=== UNIFIED PIPELINE COMPLETE! ===")

if __name__ == "__main__":
    main()
