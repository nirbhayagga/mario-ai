"""
Environment Setup Module
This file intricately constructs the specific execution environment necessary for the Proximal Policy Optimization agent.
We natively wrap the 'gym-super-mario-bros' environment with several sophisticated observation transformations.
"""

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv

class CustomRewardWrapper(gym.Wrapper):
    """
    Specifically engineered reward shaping layer to accelerate convergence.
    """
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        # 1. Reward for Flagpole (Massive success)
        if info['flag_get']:
            reward += 500
            
        # 2. Penalty for Death
        if done and info['flag_get'] == False:
            reward -= 50
            
        return state, reward, done, info

class NoopResetEnv(gym.Wrapper):
    """
    Introduces initial state entropy by performing a random number of NOOPs (0-30).
    """
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0) # 0 is NOOP in SIMPLE_MOVEMENT
            if done:
                obs = self.env.reset(**kwargs)
        return obs

def create_mario_environment(env_id="SuperMarioBros-v0", n_envs=1, use_custom_reward=False, use_noop=False):
    """
    Instantiates and wraps the Reinforcement Learning environment.
    Supports single or multiple parallel environments, which can be different levels.
    """
    
    def make_env(target_id):
        def _init():
            env = gym_super_mario_bros.make(target_id)
            if use_noop:
                env = NoopResetEnv(env)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            if use_custom_reward:
                env = CustomRewardWrapper(env)
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
