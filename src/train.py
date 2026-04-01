"""
Training Module
Constructs the PPO model, parses CLI hyperparameters, and runs the training loop
with TensorBoard logging and periodic checkpointing.
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env_setup import create_mario_environment


def parse_training_arguments():
    parser = argparse.ArgumentParser(description="PPO Mario Training")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Hardware device for training.")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total environment steps to train for.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Optimizer learning rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size per gradient update.")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per environment to collect before each update.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient for exploration.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for future rewards.")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="Value function loss coefficient.")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO optimization epochs per rollout.")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Checkpoint save frequency. Auto-computed for ~15 checkpoints if not set.")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a pre-trained model to resume from.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0",
                        help="The Gym environment ID.")
    parser.add_argument("--no_custom_reward", action="store_true",
                        help="DISABLE reward shaping (flagpole bonus, death penalty, stuck penalty).")
    parser.add_argument("--no_lr_schedule", action="store_true",
                        help="DISABLE linear learning rate annealing.")
    parser.add_argument("--output_dir", type=str, default="./runs/default_run",
                        help="Base directory for models, checkpoints, and logs.")
    return parser.parse_args()


def linear_schedule(initial_value):
    """Returns a function that linearly decays from initial_value to 0."""
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def execute_training():
    args = parse_training_arguments()

    # Handle comma-separated multi-level IDs
    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    print(f"Training on device: {args.device}")
    print(f"Environment: {target_envs}")
    use_custom_reward = not args.no_custom_reward
    use_lr_schedule = not args.no_lr_schedule

    print(f"Custom reward: {use_custom_reward} | LR schedule: {use_lr_schedule}")

    env = create_mario_environment(
        env_id=target_envs,
        n_envs=args.n_envs,
        use_custom_reward=use_custom_reward,
        use_noop=use_custom_reward,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    tb_log = os.path.join(args.output_dir, "tensorboard")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    # Auto-compute save frequency to get ~15 checkpoints per run
    if args.save_freq is not None:
        save_freq = args.save_freq
    else:
        total_env_steps = args.total_timesteps // max(args.n_envs, 1)
        save_freq = max(10000, total_env_steps // 15)
    print(f"Checkpoint frequency: every {save_freq} env steps (~{args.total_timesteps // (max(args.n_envs, 1) * save_freq)} checkpoints)")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=ckpt_dir,
        name_prefix='mario_ppo_agent',
    )

    lr = args.learning_rate
    if use_lr_schedule:
        lr = linear_schedule(args.learning_rate)

    if args.load_model:
        print(f"Resuming from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            device=args.device,
            tensorboard_log=tb_log,
        )
        # Apply current LR settings to the resumed model
        model.learning_rate = lr
    else:
        print("Initializing new PPO model.")

        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            device=args.device,
            verbose=1,
            tensorboard_log=tb_log,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = os.path.join(args.output_dir, "mario_ppo_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    execute_training()
