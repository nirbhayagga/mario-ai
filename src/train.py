"""
Training Module
Constructs the PPO model, parses CLI hyperparameters, and runs the training loop
with TensorBoard logging and periodic checkpointing.
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_device
from callbacks import EpisodeStatsCallback
from env_setup import create_mario_environment


def parse_training_arguments():
    parser = argparse.ArgumentParser(description="PPO Mario Training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Hardware device for training (auto picks CUDA when available).")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total environment steps to train for.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Optimizer learning rate.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Minibatch size per gradient update.")
    parser.add_argument("--n_steps", type=int, default=512,
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
                        help="DISABLE stuck penalty, early termination, and reward clipping.")
    parser.add_argument("--no_lr_schedule", action="store_true",
                        help="DISABLE linear learning rate annealing.")
    parser.add_argument("--action_space", type=str, default="simple", choices=["simple", "complex", "right_only"],
                        help="Action space to use (simple, complex, or right_only).")
    parser.add_argument("--clip_range", type=float, default=0.1,
                        help="PPO clipping parameter (tighter = more stable).")
    parser.add_argument("--output_dir", type=str, default="./runs/default_run",
                        help="Base directory for models, checkpoints, and logs.")
    return parser.parse_args()


def linear_schedule(initial_value):
    """Returns a function that linearly decays from initial_value to near-zero.
    A small floor (1e-7) prevents the final training steps from being
    completely wasted on zero-gradient updates."""
    def func(progress_remaining):
        return max(progress_remaining * initial_value, 1e-7)
    return func


def execute_training():
    args = parse_training_arguments()

    # Handle comma-separated multi-level IDs
    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    print(f"Training on device: {get_device(args.device)}")
    print(f"Environment: {target_envs}")
    use_custom_reward = not args.no_custom_reward
    use_lr_schedule = not args.no_lr_schedule

    print(f"Custom reward: {use_custom_reward} | LR schedule: {use_lr_schedule}")

    env = create_mario_environment(
        env_id=target_envs,
        n_envs=args.n_envs,
        use_custom_reward=use_custom_reward,
        use_noop=use_custom_reward,
        action_space=args.action_space,
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

    actual_timesteps_per_save = save_freq * max(args.n_envs, 1)
    print(f"Checkpoint frequency: every {save_freq} env steps "
          f"({actual_timesteps_per_save} total timesteps) -> ~15 checkpoints")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=ckpt_dir,
        name_prefix='mario_ppo_agent',
    )

    # Eval callback: periodically evaluate and save the best model.
    # This protects against catastrophic forgetting where the final model
    # may be far worse than intermediate ones. Evaluation uses the raw game
    # reward (no shaping) but keeps random no-op starts: with a deterministic
    # policy on a deterministic env, the n_eval_episodes would otherwise be
    # identical replays of a single trajectory.
    eval_env = create_mario_environment(
        env_id=target_envs if isinstance(target_envs, str) else target_envs[0],
        n_envs=1,
        use_custom_reward=False,
        use_noop=True,
        action_space=args.action_space,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=os.path.join(args.output_dir, "eval_logs"),
        eval_freq=save_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    combined_callback = CallbackList([checkpoint_callback, eval_callback, EpisodeStatsCallback()])

    lr = args.learning_rate
    if use_lr_schedule:
        lr = linear_schedule(args.learning_rate)

    if args.load_model:
        print(f"Resuming from: {args.load_model}")
        # CLI flags take precedence over whatever was saved in the checkpoint.
        # They must be passed as load() kwargs: SB3 rebuilds the LR/clip_range
        # schedules and the rollout buffer inside load(), so assigning these
        # attributes afterwards is ignored (learning_rate) or crashes on the
        # first update (clip_range has already become a callable).
        model = PPO.load(
            args.load_model,
            env=env,
            device=args.device,
            tensorboard_log=tb_log,
            learning_rate=lr,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
        )
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
            clip_range=args.clip_range,
            device=args.device,
            verbose=1,
            tensorboard_log=tb_log,
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=combined_callback,
        progress_bar=True,
    )

    final_path = os.path.join(args.output_dir, "mario_ppo_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    execute_training()
