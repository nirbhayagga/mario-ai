"""
Environment Setup Module
Constructs the wrapped gym-super-mario-bros environment with observation
transformations required for PPO training with a CNN policy.
"""

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv


class CustomRewardWrapper(gym.Wrapper):
    """Reward shaping: bonus for reaching the flagpole, penalty for death."""

    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if info['flag_get']:
            reward += 500

        if done and not info['flag_get']:
            reward -= 50

        return state, reward, done, info


class NoopResetEnv(gym.Wrapper):
    """Randomizes the starting state by executing 1-30 NOOP actions on reset."""

    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class SkipFrame(gym.Wrapper):
    """
    Repeats the chosen action for `skip` consecutive frames and accumulates
    reward. Returns only the final frame. Reduces training time by ~75% and
    gives the frame stack a wider temporal window (~260ms at skip=4).
    """

    def __init__(self, env, skip=4):
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treats each life loss as a terminal state. Without this wrapper, Mario
    respawns after death without a done signal, breaking the MDP assumption
    that the agent's state should not teleport discontinuously.
    """

    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = info.get('life', 0)
        if self.lives > lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.lives = 3
        return obs


def create_mario_environment(env_id="SuperMarioBros-v0", n_envs=1,
                             use_custom_reward=False, use_noop=False):
    """
    Builds the full environment stack: JoypadSpace, EpisodicLife, SkipFrame,
    optional reward shaping, grayscale, resize to 84x84, then VecFrameStack(4).
    Supports single level, parallel single level, and multi-level training.
    """

    def make_env(target_id):
        def _init():
            env = gym_super_mario_bros.make(target_id)
            if use_noop:
                env = NoopResetEnv(env)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = EpisodicLifeEnv(env)
            env = SkipFrame(env, skip=4)
            if use_custom_reward:
                env = CustomRewardWrapper(env)
            env = GrayScaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, shape=84)
            return env
        return _init

    if isinstance(env_id, list):
        # Cycle levels across n_envs so each level gets balanced coverage
        actual_n = max(n_envs, len(env_id))
        print(f"Creating multi-level environment: {env_id} ({actual_n} envs)")
        env = SubprocVecEnv([make_env(env_id[i % len(env_id)]) for i in range(actual_n)])
    elif n_envs > 1:
        env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_id)])

    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
