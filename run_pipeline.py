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
        choices=["validate", "train", "evaluate", "manual", "compete", "visualize", "plot", "dashboard"],
        help="The pipeline operation mode to execute."
    )
    
    # Training / Evaluation Arguments
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Hardware device selection.")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments to run.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pre-trained model to resume or evaluate.")
    
    # Evaluation / Visualization Arguments
    parser.add_argument("--model_path", type=str, default="models/mario_ppo_final.zip", help="Path to the model file for evaluation.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_models", help="Directory for progress visualization.")
    parser.add_argument("--video_output", type=str, default="videos/evaluation_run.mp4", help="Folder to export evaluation videos.")

    args, unknown = parser.parse_known_args()

    # Base python command within the venv (assumes venv/bin/python)
    python_cmd = [sys.executable] # Use current interpreter if running from venv

    if args.mode == "validate":
        run_command(python_cmd + ["src/validate.py"])
    
    elif args.mode == "train":
        train_cmd = python_cmd + [
            "src/train.py", 
            "--total_timesteps", str(args.total_timesteps),
            "--device", args.device,
            "--n_envs", str(args.n_envs)
        ]
        if args.load_model:
            train_cmd += ["--load_model", args.load_model]
        
        train_cmd += unknown
        run_command(train_cmd)
        
    elif args.mode == "evaluate":
        # Use load_model as fallback for model_path if provided
        model_to_use = args.load_model if args.load_model else args.model_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(args.video_output), exist_ok=True)
        eval_cmd = python_cmd + [
            "src/evaluate.py",
            "--model_path", model_to_use,
            "--video_output", args.video_output,
            "--device", args.device
        ] + unknown
        run_command(eval_cmd)
        
    elif args.mode == "visualize":
        os.makedirs(os.path.dirname(args.video_output), exist_ok=True)
        vis_cmd = python_cmd + [
            "src/visualize_progress.py",
            "--checkpoint_dir", args.checkpoint_dir,
            "--output_path", args.video_output,
            "--device", args.device
        ] + unknown
        run_command(vis_cmd)

    elif args.mode == "plot":
        plot_cmd = python_cmd + [
            "src/plot_metrics.py",
            "--log_dir", "tensorboard_logs",
            "--output_path", "logs/reward_curve.png"
        ] + unknown
        run_command(plot_cmd)
        
    elif args.mode == "dashboard":
        db_cmd = ["tensorboard", "--logdir", "tensorboard_logs"] + unknown
        run_command(db_cmd)
        
    elif args.mode == "manual":
        run_command(python_cmd + ["src/manual_play.py"])
        
    elif args.mode == "compete":
        run_command(python_cmd + ["src/compete.py"])

if __name__ == "__main__":
    main()
