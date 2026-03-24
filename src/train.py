"""
Training Orchestrator Module
This file specifically constructs the stable baseline algorithms, parses explicit architectural command line arguments, and initializes the training phase.
It securely implements native explicit TensorBoard logging arrays completely seamlessly tracking algorithmic validation mathematically alongside Rich progress interfaces.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env_setup import create_mario_environment

def parse_training_arguments():
    """
    Initializes standard argument parsing specifically designed to avoid hardcoding hyperparameter configurations and hardware selection.
    """
    parser = argparse.ArgumentParser(description="PPO Mario Training Execution Argument Parser")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Explicit hardware tensor processing device selection.")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="The total maximum quantity of mathematical environment steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The optimizer sequential learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="The absolute batch size utilized per network update phase.")
    parser.add_argument("--n_steps", type=int, default=2048, help="The exact number of operational sequential steps to accumulate before triggering gradient updates.")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor calculating future reward persistence.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient balancing actor-critic loss.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments to run.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to a pre-trained model to resume training from.")
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-v0", help="The Gym environment ID.")
    parser.add_argument("--use_custom_reward", action="store_true", help="Enable advanced reward shaping.")
    parser.add_argument("--lr_schedule", action="store_true", help="Enable linear learning rate annealing.")
    parser.add_argument("--output_dir", type=str, default="./runs/default_run", help="Base directory for saving models, checkpoints, and logs.")
    return parser.parse_args()

def linear_schedule(initial_value):
    """
    Linear learning rate schedule logic.
    """
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def execute_training_routine():
    """
    Constructs the operational logic mapping TensorBoard cleanly and executes exactly the structural optimization natively rendering terminal progress.
    """
    
    # Retrieve explicitly requested parameters strictly from the formal terminal arguments
    args = parse_training_arguments()
    
    # Detect if we are using multi-level training (comma-separated list)
    if "," in args.env_id:
        target_envs = [e.strip() for e in args.env_id.split(",")]
    else:
        target_envs = args.env_id

    print(f"Initializing specifically targeted training sequence natively mapped directly to device axis: {args.device}")
    
    # Instantiate the mathematically wrapped core emulator arrays
    environment_instance = create_mario_environment(
        env_id=target_envs, 
        n_envs=args.n_envs, 
        use_custom_reward=args.use_custom_reward,
        use_noop=args.use_custom_reward # Use noop when custom rewards are enabled
    )
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    tb_log = os.path.join(args.output_dir, "tensorboard")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    # Define an atomic checkpoint callback system formally explicitly mapped to logically dump secure network weights periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=ckpt_dir, 
        name_prefix='mario_ppo_agent'
    )
    
    # Construct the primary operational Proximal Policy Optimization framework mapping the CnnPolicy matrix securely injecting strictly formal TensorBoard parameters
    if args.load_model:
        print(f"Resuming training from specifically targeted serialized model: {args.load_model}")
        neural_network_model = PPO.load(
            args.load_model, 
            env=environment_instance, 
            device=args.device,
            tensorboard_log=tb_log
        )
    else:
        print("Initializing specifically targeted new neural weight matrices natively.")
        
        # Determine the learning rate schedule mechanism
        lr = args.learning_rate
        if args.lr_schedule:
            lr = linear_schedule(args.learning_rate)
            
        neural_network_model = PPO(
            policy="CnnPolicy", 
            env=environment_instance, 
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            device=args.device,
            verbose=1,
            tensorboard_log=tb_log
        )
    
    # Execute exactly the secure learning mathematical loop formally invoking the specific Rich progress bar matrices comprehensively mathematically
    # Note: progress_bar=True natively securely maps tqdm or rich console visualization dynamically structurally
    neural_network_model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)
    
    # Natively explicitly force formal serialized saving immediately after exactly terminating the stepping array structurally securely
    final_save_path = os.path.join(args.output_dir, "mario_ppo_final")
    neural_network_model.save(final_save_path)
    print(f"Formal analytical training loop phase concluded perfectly securely serializing output explicitly specifically to {final_save_path}")

if __name__ == "__main__":
    execute_training_routine()
