"""
Environment Setup Module
Constructs the wrapped gym-super-mario-bros environment with observation
transformations required for PPO training with a CNN policy.
"""

import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv


class CustomRewardWrapper(gym.Wrapper):
    """
    Stuck penalty and early termination only.

    The base gym-super-mario-bros environment already provides good reward
    shaping (x-velocity + clock penalty + death penalty). This wrapper adds
    only what the base env lacks: detection and punishment of the agent
    getting physically stuck against obstacles like pipes.

    Applied BEFORE SkipFrame, so frame counts are real NES frames.
    """

    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.current_x = 0
        self.x_stuck_frames = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)

        if x_pos <= self.current_x:
            self.x_stuck_frames += 1
        else:
            self.x_stuck_frames = 0
            self.current_x = max(self.current_x, x_pos)

        # ~1.3 seconds stuck (80 real frames at 60fps): continuous small penalty
        if self.x_stuck_frames > 80:
            reward -= 1.0

        # ~10 seconds stuck (600 real frames): end episode to save training time
        if self.x_stuck_frames > 600:
            done = True
            reward -= 15

        return state, reward, done, info

    def reset(self, **kwargs):
        self.current_x = 0
        self.x_stuck_frames = 0
        return self.env.reset(**kwargs)


class RewardClipWrapper(gym.RewardWrapper):
    """
    Clips rewards to [-15, 15] to prevent gradient explosions in PPO.
    Applied AFTER SkipFrame so the accumulated 4-frame reward is clipped.
    """

    def __init__(self, env, low=-15.0, high=15.0):
        super(RewardClipWrapper, self).__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward):
        return float(np.clip(reward, self.low, self.high))


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
        self.lives = 0
        return obs


def create_mario_environment(env_id="SuperMarioBros-v0", n_envs=1,
                             use_custom_reward=False, use_noop=False, action_space="simple"):
    """
    Builds the full environment stack:
      JoypadSpace -> EpisodicLife -> [CustomReward] -> SkipFrame -> [RewardClip]
      -> GrayScale -> Resize -> VecFrameStack(4)

    CustomReward is placed BEFORE SkipFrame so it sees individual frames.
    RewardClip is placed AFTER SkipFrame so the accumulated reward is clipped.
    """

    def make_env(target_id):
        def _init():
            env = gym_super_mario_bros.make(target_id)
            if use_noop:
                env = NoopResetEnv(env)
            if action_space == "complex":
                movement = COMPLEX_MOVEMENT
            elif action_space == "right_only":
                movement = RIGHT_ONLY
            else:
                movement = SIMPLE_MOVEMENT
            env = JoypadSpace(env, movement)
            env = EpisodicLifeEnv(env)
            if use_custom_reward:
                env = CustomRewardWrapper(env)
            env = SkipFrame(env, skip=4)
            if use_custom_reward:
                env = RewardClipWrapper(env, low=-15.0, high=15.0)
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
