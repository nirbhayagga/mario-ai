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
    Supports single or multiple parallel environments.
    """
    
    def make_env():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        return env

    if n_envs > 1:
        # Use SubprocVecEnv for parallel execution across CPU cores
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        # Use DummyVecEnv for single environment execution
        env = DummyVecEnv([make_env])
    
    # Finally, stack the four most recent operational frames
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    return env
