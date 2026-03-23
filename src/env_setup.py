"""
Environment Setup Module
This file intricately constructs the specific execution environment necessary for the Proximal Policy Optimization agent.
We natively wrap the 'gym-super-mario-bros' environment with several sophisticated observation transformations.
"""

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

def create_mario_environment(env_id="SuperMarioBros-v0", n_envs=1):
    """
    Instantiates and wraps the Reinforcement Learning environment.
    Supports single or multiple parallel environments, which can be different levels.
    """
    
    def make_env(target_id):
        def _init():
            env = gym_super_mario_bros.make(target_id)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, shape=84)
            return env
        return _init

    if isinstance(env_id, list):
        # Multi-level training: Create each level in the list
        print(f"Creating specifically targeted multi-level environment matrix: {env_id}")
        env = SubprocVecEnv([make_env(eid) for eid in env_id])
    elif n_envs > 1:
        # Single-level parallel training
        env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    else:
        # Single-level sequential training
        env = DummyVecEnv([make_env(env_id)])
    
    # Finally, stack the four most recent operational frames
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    return env
